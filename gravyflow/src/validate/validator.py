from __future__ import annotations

from pathlib import Path
import logging
import sys
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union, TYPE_CHECKING

import h5py
import numpy as np

import gravyflow as gf
from .config import ValidationConfig
from .bank import ValidationBank

if TYPE_CHECKING:  # avoids importing keras at runtime
    import keras

logger = logging.getLogger(__name__)


class Validator:
    """Orchestrates validation runs + checkpointing + dashboard plotting."""

    # ----------------------------- construction/logging -----------------------------

    def __init__(self):
        self.logger = self._setup_logger()

        # runtime identity/config
        self.name: str = "model"
        self.config: ValidationConfig = ValidationConfig()
        self.dataset_args: dict = {}

        self.input_duration_seconds: float = gf.Defaults.onsource_duration_seconds
        self.offsource_duration_seconds: float = gf.Defaults.offsource_duration_seconds

        # cached results
        self.efficiency_data: Optional[Dict[str, Any]] = None
        self.far_scores: Optional[np.ndarray] = None
        self.noise_gps_times: Optional[np.ndarray] = None
        self.roc_data: Optional[Dict[str, Any]] = None
        self.worst_performers: Optional[Dict[str, Any]] = None
        self.real_events: Optional[List[Dict[str, Any]]] = None

        # runtime-only
        self.bank: Optional[ValidationBank] = None

    @staticmethod
    def _setup_logger(level: int = logging.INFO) -> logging.Logger:
        log = logging.getLogger("validator")
        if not log.handlers:
            log.addHandler(logging.StreamHandler(sys.stdout))
        log.setLevel(level)
        return log

    def _restore_runtime(self, *, name: str, config: ValidationConfig, dataset_args: dict, level: int) -> None:
        self.logger = self._setup_logger(level)
        self.name = str(name) if name else self.name
        self.config = config or ValidationConfig()
        self.dataset_args = dataset_args or {}

        self.input_duration_seconds = self.dataset_args.get(
            "onsource_duration_seconds", getattr(self, "input_duration_seconds", gf.Defaults.onsource_duration_seconds)
        )
        self.offsource_duration_seconds = self.dataset_args.get(
            "offsource_duration_seconds",
            getattr(self, "offsource_duration_seconds", gf.Defaults.offsource_duration_seconds),
        )

    @staticmethod
    def _has_scored_real_events(events: Optional[List[Dict[str, Any]]]) -> bool:
        return bool(events) and any(e.get("score") is not None for e in events)

    # --------------------------------- public API ---------------------------------

    @classmethod
    def validate(
        cls,
        model: "keras.Model",
        name: str,
        dataset_args: dict,
        config: ValidationConfig = None,
        checkpoint_file_path: Path = None,
        logging_level: int = logging.INFO,
        heart: gf.Heart = None,
        **_legacy_kwargs,  # absorb legacy kwargs
    ) -> "Validator":
        v = cls()
        v._restore_runtime(name=name, config=config or ValidationConfig(), dataset_args=dataset_args, level=logging_level)

        # Try checkpoint
        if checkpoint_file_path and checkpoint_file_path.exists():
            try:
                v = cls.load(checkpoint_file_path, logging_level=logging_level)
                v._restore_runtime(name=name, config=config or ValidationConfig(), dataset_args=dataset_args, level=logging_level)

                if v.efficiency_data is not None and v.far_scores is not None:
                    if cls._has_scored_real_events(v.real_events):
                        v.logger.info("Loaded complete validation data (including scored real events) from checkpoint.")
                        # still attach a live bank for downstream calls (ROC pools, etc.)
                        v.bank = ValidationBank(model=model, dataset_args=dataset_args, config=v.config, heart=heart)
                        v.bank.far_scores = v.far_scores
                        v.bank.snrs = np.asarray(v.efficiency_data.get("snrs")) if v.efficiency_data else None
                        v.bank.scores = np.asarray(v.efficiency_data.get("scores")) if v.efficiency_data else None
                        v.bank.injection_masks = np.ones_like(v.bank.scores) if v.bank.scores is not None else None
                        return v
                    v.logger.info("Loaded partial validation data. Real events missing scores - regenerating...")
            except Exception as e:
                v.logger.warning(f"Failed to load checkpoint: {e}. Regenerating.")

        # Live bank (does the heavy lifting)
        bank = ValidationBank(model=model, dataset_args=dataset_args, config=v.config, heart=heart)

        # Noise
        if v.far_scores is None:
            bank.generate_noise()
            v.far_scores, v.noise_gps_times = bank.far_scores, bank.noise_gps_times
            if checkpoint_file_path:
                v.save(checkpoint_file_path)
                v.logger.info("Checkpoint saved after noise generation")
        else:
            bank.far_scores = v.far_scores
            bank.noise_gps_times = v.noise_gps_times

        # Injections
        if v.efficiency_data is None:
            bank.generate()
            v.efficiency_data = bank.get_efficiency_data()
            v.worst_performers = bank.get_worst_performers()
            if checkpoint_file_path:
                v.save(checkpoint_file_path)
                v.logger.info("Checkpoint saved after injection scoring")
        else:
            bank.snrs = np.asarray(v.efficiency_data.get("snrs")) if v.efficiency_data else None
            bank.scores = np.asarray(v.efficiency_data.get("scores")) if v.efficiency_data else None
            bank.injection_masks = np.ones_like(bank.scores) if bank.scores is not None else None

        # ROC
        if v.roc_data is None:
            v.roc_data = bank.get_roc_curves()

        # Real events (optional; network/catalog I/O inside gravyflow)
        has_scored = cls._has_scored_real_events(v.real_events)
        v.logger.info(f"Real events check: {len(v.real_events) if v.real_events else 0} events loaded, has_scored={has_scored}")
        if not has_scored:
            try:
                bank.generate_real_events()
                v.real_events = bank.get_real_events()
                if checkpoint_file_path:
                    v.save(checkpoint_file_path)
                    v.logger.info("Checkpoint saved after real events scoring")
            except Exception as e:
                v.logger.warning(f"Real events generation failed: {e}")
                v.real_events = []

        if checkpoint_file_path:
            v.save(checkpoint_file_path)

        v.bank = bank
        return v

    # -------------------------------- checkpoint I/O --------------------------------

    @staticmethod
    def _h5_write_scalar(g: h5py.Group, k: str, v: Any) -> None:
        if v is None:
            return
        if isinstance(v, str):
            g.create_dataset(k, data=v.encode())
        else:
            g.create_dataset(k, data=v)

    @staticmethod
    def _h5_read_value(ds) -> Any:
        v = ds[()]
        if isinstance(v, bytes):
            return v.decode()
        if isinstance(v, np.ndarray):
            return v.item() if v.shape == () else v
        return v

    @classmethod
    def _save_samples(cls, parent: h5py.Group, samples: List[Dict[str, Any]]) -> None:
        for i, sample in enumerate(samples):
            sg = parent.create_group(f"sample_{i}")
            for k, v in sample.items():
                if v is None:
                    continue
                cls._h5_write_scalar(sg, k, v.encode() if isinstance(v, str) else v)

    def save(self, file_path: Path) -> None:
        file_path = Path(file_path)
        self.logger.info(f"Saving validation data to {file_path}")
        gf.ensure_directory_exists(file_path.parent)

        with gf.open_hdf5_file(file_path, self.logger, mode="w") as f:
            self._h5_write_scalar(f, "name", self.name)
            self._h5_write_scalar(f, "input_duration_seconds", float(self.input_duration_seconds))
            self._h5_write_scalar(f, "offsource_duration_seconds", float(getattr(self, "offsource_duration_seconds", 0.0)))

            if self.efficiency_data:
                g = f.create_group("efficiency_data")
                for k in ("snrs", "scores", "mass1", "mass2", "gps_times", "central_times", "hpeak", "hrss"):
                    arr = self.efficiency_data.get(k)
                    if arr is not None and (not isinstance(arr, np.ndarray) or arr.size):
                        g.create_dataset(k, data=arr)

            if self.far_scores is not None:
                g = f.create_group("far_scores")
                g.create_dataset("scores", data=self.far_scores)
                if self.noise_gps_times is not None and len(self.noise_gps_times) > 0:
                    g.create_dataset("gps_times", data=self.noise_gps_times)

            if self.roc_data:
                g = f.create_group("roc_data")
                keys = list(self.roc_data.keys())
                g.create_dataset("keys", data=np.array([k.encode() for k in keys], dtype=h5py.string_dtype()))
                for k, v in self.roc_data.items():
                    sub = g.create_group(k)
                    sub.create_dataset("fpr", data=v["fpr"])
                    sub.create_dataset("tpr", data=v["tpr"])
                    sub.create_dataset("roc_auc", data=v["roc_auc"])

            if self.worst_performers:
                g = f.create_group("worst_performers")

                fn = (self.worst_performers or {}).get("false_negatives", {})
                if fn:
                    fn_g = g.create_group("false_negatives")
                    for bin_key, samples in fn.items():
                        cls = fn_g.create_group(str(bin_key))
                        self._save_samples(cls, samples)

                fp = (self.worst_performers or {}).get("false_positives", [])
                if fp:
                    fp_g = g.create_group("false_positives")
                    self._save_samples(fp_g, fp)

            if self.real_events:
                re_g = f.create_group("real_events")
                for i, event in enumerate(self.real_events):
                    eg = re_g.create_group(f"event_{i}")
                    for k, v in event.items():
                        if v is None:
                            continue
                        self._h5_write_scalar(eg, k, v)

    @classmethod
    def load(cls, file_path: Path, logging_level: int = logging.INFO) -> "Validator":
        v = cls()
        v.logger = cls._setup_logger(logging_level)

        # ensure attributes always exist (prevents AttributeError on older checkpoints/call paths)
        v.efficiency_data = None
        v.far_scores = None
        v.noise_gps_times = None
        v.roc_data = None
        v.worst_performers = None
        v.real_events = None

        with h5py.File(file_path, "r") as f:
            v.name = f["name"][()].decode() if "name" in f else "Unknown"
            v.input_duration_seconds = float(f["input_duration_seconds"][()]) if "input_duration_seconds" in f else 1.0
            v.offsource_duration_seconds = float(f["offsource_duration_seconds"][()]) if "offsource_duration_seconds" in f else gf.Defaults.offsource_duration_seconds

            if "efficiency_data" in f:
                g = f["efficiency_data"]
                v.efficiency_data = {
                    "snrs": g["snrs"][:] if "snrs" in g else np.array([]),
                    "scores": g["scores"][:] if "scores" in g else np.array([]),
                    "mass1": g["mass1"][:] if "mass1" in g else np.array([]),
                    "mass2": g["mass2"][:] if "mass2" in g else np.array([]),
                    "gps_times": g["gps_times"][:] if "gps_times" in g else np.array([]),
                    "central_times": g["central_times"][:] if "central_times" in g else np.array([]),
                    "hpeak": g["hpeak"][:] if "hpeak" in g else np.array([]),
                    "hrss": g["hrss"][:] if "hrss" in g else np.array([]),
                }

            if "far_scores" in f:
                g = f["far_scores"]
                v.far_scores = g["scores"][:]
                v.noise_gps_times = g["gps_times"][:] if "gps_times" in g else None

            if "roc_data" in f:
                g = f["roc_data"]
                v.roc_data = {}
                keys = [k.decode() for k in g["keys"][:]] if "keys" in g else list(g.keys())
                for k in keys:
                    if k not in g or k == "keys":
                        continue
                    sub = g[k]
                    v.roc_data[k] = {
                        "fpr": sub["fpr"][:],
                        "tpr": sub["tpr"][:],
                        "roc_auc": float(sub["roc_auc"][()]),
                    }

            if "worst_performers" in f:
                g = f["worst_performers"]
                v.worst_performers = {"false_negatives": {}, "false_positives": []}

                if "false_negatives" in g:
                    fn_g = g["false_negatives"]
                    for bin_key in fn_g:
                        bg = fn_g[bin_key]
                        v.worst_performers["false_negatives"][bin_key] = []
                        for sample_key in sorted(bg.keys()):
                            sg = bg[sample_key]
                            sample = {k: cls._h5_read_value(sg[k]) for k in sg}
                            v.worst_performers["false_negatives"][bin_key].append(sample)

                if "false_positives" in g:
                    fp_g = g["false_positives"]
                    for sample_key in sorted(fp_g.keys()):
                        sg = fp_g[sample_key]
                        v.worst_performers["false_positives"].append({k: cls._h5_read_value(sg[k]) for k in sg})

            if "real_events" in f:
                v.real_events = []
                re_g = f["real_events"]
                for event_key in sorted(re_g.keys()):
                    eg = re_g[event_key]
                    v.real_events.append({k: cls._h5_read_value(eg[k]) for k in eg})
            else:
                v.real_events = None  # triggers regeneration

        return v

    # ------------------------------------ plotting ------------------------------------

    def _get_valid_segments(self) -> Optional[np.ndarray]:
        """Best-effort extraction (or re-fetch) of valid_segments for the GPS plot."""
        args = getattr(self, "dataset_args", None) or {}
        noise_obt = args.get("noise_obtainer")
        if not noise_obt or not hasattr(noise_obt, "ifo_data_obtainer"):
            self.logger.info("GPS plot: noise_obtainer has no ifo_data_obtainer")
            return None

        obt = noise_obt.ifo_data_obtainer
        if not obt:
            self.logger.info("GPS plot: ifo_data_obtainer is None")
            return None

        seg = getattr(obt, "valid_segments_adjusted", None)
        if seg is not None:
            self.logger.info(f"GPS plot: Using valid_segments_adjusted, shape={np.asarray(seg).shape}")
            return np.asarray(seg)

        seg = getattr(obt, "valid_segments", None)
        if seg is not None:
            self.logger.info(f"GPS plot: Using valid_segments, shape={np.asarray(seg).shape}")
            return np.asarray(seg)

        self.logger.info("GPS plot: Reloading valid segments for distribution plot...")
        try:
            ifos = getattr(noise_obt, "ifos", None) or getattr(getattr(self, "config", None), "ifos", None)
            if not ifos:
                self.logger.warning("GPS plot: noise_obtainer has no IFOs configured")
                return None

            seg = obt.get_valid_segments(ifos=ifos, seed=42, group_name=args.get("group", "test"))
            if seg is None:
                self.logger.warning("GPS plot: get_valid_segments returned None")
                return None

            seg = np.asarray(seg)
            obt.valid_segments = seg  # cache back onto object
            self.logger.info(f"GPS plot: Successfully fetched segments, shape={seg.shape}")
            return seg
        except Exception as e:
            self.logger.warning(f"GPS plot: Failed to fetch valid segments: {e}")
            return None

    def plot(self, output_path: Path = None, comparison_validators: Optional[List["Validator"]] = None):
        """Generate validation dashboard (Panel template)."""
        import panel as pn
        from .plotting import (
            generate_efficiency_plot,
            generate_far_curves,
            generate_gps_distribution_plot,
            generate_roc_curves,
            generate_parameter_space_plot,
            generate_waveform_plot,
            generate_real_events_table,
        )

        comparison_validators = comparison_validators or []
        all_validators = [*comparison_validators, self]

        eff_plot, eff_slider = generate_efficiency_plot(all_validators, fars=self.config.far_thresholds)
        far_plot = generate_far_curves(all_validators)

        valid_segments = self._get_valid_segments()
        gps_dist_plot = generate_gps_distribution_plot(
            all_validators,
            valid_segments=valid_segments,
            onsource_duration_seconds=getattr(self, "input_duration_seconds", 1.0),
            offsource_duration_seconds=getattr(self, "offsource_duration_seconds", 16.0),
        )

        roc_plot, roc_select = generate_roc_curves(all_validators)
        param_plot, param_controls = generate_parameter_space_plot(all_validators)

        tabs = pn.Tabs(
            ("Efficiency", pn.Column(eff_slider, eff_plot)),
            ("ROC Curves", pn.Column(roc_select, roc_plot)),
            ("FAR Curves", pn.Column(far_plot, gps_dist_plot)),
            ("Parameter Space", pn.Column(param_controls, param_plot) if param_controls else param_plot),
        )

        if self.real_events:
            events_table, _ = generate_real_events_table(
                self.real_events,
                far_scores=self.far_scores,
                input_duration_seconds=self.input_duration_seconds,
                far_thresholds=self.config.far_thresholds,
            )
            tabs.append(("Real Events", events_table))

        if self.worst_performers:
            fn = self.worst_performers.get("false_negatives", {}) or {}
            if fn:
                fn_tabs = []
                sorted_bins = sorted(
                    fn.keys(), key=lambda x: float(str(x).split("-")[0]) if "-" in str(x) else 0.0
                )
                for bin_key in sorted_bins:
                    samples = fn.get(bin_key) or []
                    plots = []
                    for sample in samples[:10]:
                        try:
                            plots.append(generate_waveform_plot(sample, self.input_duration_seconds))
                        except Exception as e:
                            self.logger.warning(f"Error plotting FN sample: {e}")
                    if plots:
                        fn_tabs.append((f"SNR {bin_key}", pn.Column(*plots)))
                if fn_tabs:
                    tabs.append(("Worst False Negatives", pn.Tabs(*fn_tabs)))

            fp = self.worst_performers.get("false_positives", []) or []
            if fp:
                plots = []
                for sample in fp[:25]:
                    try:
                        plots.append(generate_waveform_plot(sample, self.input_duration_seconds))
                    except Exception as e:
                        self.logger.warning(f"Error plotting FP sample: {e}")
                if plots:
                    tabs.append(("Worst False Positives", pn.Column(*plots)))

        template = pn.template.FastListTemplate(
            title=f"Validation Dashboard: {self.name}",
            theme="dark",
            theme_toggle=False,
            main=[tabs],
        )

        if output_path:
            output_path = Path(output_path)
            gf.ensure_directory_exists(output_path.parent)
            template.save(str(output_path), resources="inline")
            self.logger.info(f"Saved report to {output_path}")

        return template