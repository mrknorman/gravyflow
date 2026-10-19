from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import h5py
import numpy as np

import gravyflow as gf
from .bank import Heartbeat, ValidationBank, noise_background
from .config import ValidationConfig, check_held_out, run_record_of
from .metrics import Background, EfficiencyCurve, FarThreshold, efficiency_curve, roc_curve
from .provenance import StaleCheckpointError, model_fingerprint, require_match
from .volume import InjectionVolumes, SensitiveVolumeCurve, sensitive_volume_curve
from gravyflow.src.utils.fingerprint import config_fingerprint

if TYPE_CHECKING:  # avoids importing keras at runtime
    import keras

logger = logging.getLogger(__name__)


def _other_format(path, found: Optional[int], current: int) -> str:
    """The refusal of a record of another format, or of none."""
    what = ("has no format version (it was written before records had one)" if found is None
            else f"is in validation format {found}")
    return (f"{path} {what}; this version reads only format {current}, and converts none. Validate "
            "the model again (Validator.validate) to write a record of this format, at a new path or "
            "after deleting this one.")


class Validator:
    """
    A model's validation results: scores of a noise background, of injections
    and, optionally, of real events, with the metrics derived from them.

    Results can be checkpointed to an HDF5 file. The file records the sha256 of
    the model and of the validation settings, and is reused only when both
    match: a checkpoint written for another model or other settings is refused.
    """

    # 3: worst performers are batch-first examples; an injection is
    # (generators, detectors, samples).
    # 4: every stored array and score is of physical strain; format 3 could
    # hold internally scaled strain (a raw ONSOURCE input) and its scores.
    # 5: the sensitive-volume set (``volume_data``): injections at physical
    # distances drawn uniformly in volume out to its recorded reach (``reach_mpc``),
    # with their SNRs.
    FORMAT_VERSION = 5

    def __init__(self, name: str = "model", config: Optional[ValidationConfig] = None,
                 dataset_args: Optional[dict] = None):
        self.name = name
        self.config = config or ValidationConfig()
        self.dataset_args = dataset_args or {}
        self.input_duration_seconds = float(self.dataset_args.get(
            "onsource_duration_seconds", gf.Defaults.onsource_duration_seconds))
        self.offsource_duration_seconds = float(self.dataset_args.get(
            "offsource_duration_seconds", gf.Defaults.offsource_duration_seconds))

        self.far_scores: Optional[np.ndarray] = None
        self.noise_gps_times: Optional[np.ndarray] = None
        self.noise_streams: Optional[Dict[str, np.ndarray]] = None  # time slides (`noise_background`)
        self.efficiency_data: Optional[Dict[str, np.ndarray]] = None
        # The sensitive-volume set: its scores, distances, labels and reach (`generate_volume`).
        self.volume_data: Optional[Dict[str, np.ndarray]] = None
        self.worst_performers: Dict[str, Any] = {"false_negatives": {}, "false_positives": []}
        self.real_events: Optional[List[Dict[str, Any]]] = None

        self.model_sha256: Optional[str] = None
        self.config_sha256: Optional[str] = None
        self.bank: Optional[ValidationBank] = None

    # --------------------------------------------------------------- run

    @classmethod
    def validate(
        cls,
        model: "keras.Model",
        name: str,
        dataset_args: dict,
        config: Optional[ValidationConfig] = None,
        checkpoint_file_path: Optional[Path] = None,
        heart: Optional[Heartbeat] = None,
    ) -> "Validator":
        """
        Score ``model`` and return its results.

        The validator reads the training record ``model``'s weights carry
        (`run_record_of`; every model gravyflow trains or loads carries its own:
        ``TrainingRun.train``'s, ``RunDirectory.load_model``'s, a ``gf.Model``'s).
        It refuses a model whose data was split by another spec than the data it
        scores, whose selection split is the one it would score, or whose record
        names no split (`check_held_out`). A model with no record at all, one
        gravyflow never trained (an external one), is scored with a warning: the
        split it saw is unknown.

        With ``checkpoint_file_path``, results are saved after each stage, and a
        run resumes from the stages already saved there, provided the file was
        written for this model and these settings.

        A CBC population in noise of physical strain also gets a sensitive-volume
        set (`ValidationBank.generate_volume`); any other has none.

        Raises:
            StaleCheckpointError: the checkpoint is in another format, or belongs
                to another model or to other settings.
        """
        v = cls(name, config, dataset_args)
        bank = ValidationBank(model, dataset_args, v.config, heart)
        check_held_out(v.config.group, bank.split_spec, run_record_of(model))
        v.model_sha256 = model_fingerprint(model)
        v.config_sha256 = config_fingerprint(v.config, dataset_args)
        path = Path(checkpoint_file_path) if checkpoint_file_path else None
        if path is not None and path.exists():
            saved = cls.load(path)
            require_match("model", saved.model_sha256, v.model_sha256, path)
            require_match("validation configuration", saved.config_sha256, v.config_sha256, path)
            for attr in ("far_scores", "noise_gps_times", "noise_streams", "efficiency_data", "volume_data",
                         "worst_performers", "real_events"):
                setattr(v, attr, getattr(saved, attr))
            logger.info(f"Resuming validation from {path}.")

        def checkpoint():
            if path is not None:
                v.save(path)

        if v.far_scores is None:
            bank.generate_noise()
            v.far_scores, v.noise_gps_times, v.noise_streams = (
                bank.far_scores, bank.noise_gps_times, bank.noise_streams)
            v.worst_performers["false_positives"] = bank.get_worst_performers()["false_positives"]
            checkpoint()
        bank.far_scores, bank.noise_streams = v.far_scores, v.noise_streams

        if v.efficiency_data is None:
            bank.generate()
            v.efficiency_data = bank.get_efficiency_data()
            v.worst_performers["false_negatives"] = bank.get_worst_performers()["false_negatives"]
            checkpoint()

        if v.volume_data is None and bank.has_volume:
            bank.generate_volume()
            v.volume_data = bank.get_volume_data()
            checkpoint()

        if v.config.score_real_events and v.real_events is None:
            bank.generate_real_events()
            v.real_events = bank.get_real_events()
            checkpoint()

        v.bank = bank
        return v

    # ----------------------------------------------------------- metrics

    def background(self) -> Background:
        """
        The noise scores as a background: streams of time-slid windows of real
        noise, or independently drawn windows of synthetic noise (`noise_background`).
        """
        if self.far_scores is None:
            raise ValueError(f"Validator {self.name!r} has no noise scores.")
        return noise_background(self.far_scores, self.noise_streams, self.input_duration_seconds,
                                self.config.cluster_seconds)

    def efficiency_curves(self, fars_hz=None) -> Dict[float, Tuple[FarThreshold, EfficiencyCurve]]:
        """Efficiency against SNR at each target FAR (Hz), with binomial intervals."""
        cfg = self.config
        background = self.background()
        curves = {}
        for far in (cfg.far_thresholds if fars_hz is None else fars_hz):
            threshold = background.threshold_for_far(float(far), cfg.confidence)
            curves[float(far)] = (threshold, efficiency_curve(
                self.efficiency_data["snrs"], self.efficiency_data["scores"], threshold.threshold,
                cfg.snr_bin_edges, cfg.confidence, cfg.efficiency_interval))
        return curves

    def sensitive_volume_curve(self, fars_hz=None) -> Optional[SensitiveVolumeCurve]:
        """
        Sensitive volume (Gpc^3) and distance (Mpc) within the reach R the volume set was
        drawn to, at each target FAR (Hz), with their injection, background and joint
        intervals, noise floor, and the estimated volume beyond R with its interval
        (`volume`; Euclidean, detector-frame masses); None without a volume set (a burst
        population, or WHITE noise). The bootstrap draws from the validator's own seed,
        so a record gives the same curve whenever it is read.
        """
        data = self.volume_data
        if self.far_scores is None or data is None:
            return None
        cfg = self.config
        return sensitive_volume_curve(
            InjectionVolumes(data["scores"], float(data["reach_mpc"]), data["distances"], data["snrs"]),
            self.background(), cfg.far_thresholds if fars_hz is None else fars_hz,
            gf.SeedTree(cfg.seed).stream(cfg.group, "validator/volume"),
            cfg.volume_bootstrap_samples, cfg.confidence, cfg.efficiency_interval,
            cfg.volume_noise_floor_fraction, cfg.volume_beyond_reach_fraction)

    def roc_curves(self) -> Dict[str, Dict[str, Any]]:
        """Exact ROC curves of noise against injection populations, keyed by population."""
        if self.far_scores is None or self.efficiency_data is None:
            return {}
        snrs, scores = self.efficiency_data["snrs"], self.efficiency_data["scores"]
        pools = {f"SNR≥{self.config.default_roc_min_snr:g}": snrs >= self.config.default_roc_min_snr}
        for pool in self.config.extra_roc_pools:
            lo, hi = pool if isinstance(pool, (tuple, list)) else (float(pool) - 0.5, float(pool) + 0.5)
            pools[f"SNR {lo:g}-{hi:g}"] = (snrs >= lo) & (snrs < hi)
        results = {}
        for key, mask in pools.items():
            if not np.any(mask):
                logger.warning(f"No injections in ROC population {key}.")
                continue
            roc = roc_curve(self.far_scores, scores[mask])
            results[key] = {"fpr": roc.false_positive_rate, "tpr": roc.true_positive_rate,
                            "roc_auc": roc.auc}
        return results

    # -------------------------------------------------------- checkpoint

    @staticmethod
    def _write_value(group: h5py.Group, key: str, value: Any) -> None:
        if value is None:
            return
        group.create_dataset(key, data=value.encode() if isinstance(value, str) else value)

    @staticmethod
    def _read_value(dataset) -> Any:
        value = dataset[()]
        if isinstance(value, bytes):
            return value.decode()
        if isinstance(value, np.ndarray) and value.shape == ():
            return value.item()
        return value

    @classmethod
    def _write_samples(cls, parent: h5py.Group, samples: List[Dict[str, Any]]) -> None:
        for i, sample in enumerate(samples):
            group = parent.create_group(f"sample_{i:04d}")
            for k, v in sample.items():
                cls._write_value(group, k, v)

    @classmethod
    def _read_samples(cls, parent: h5py.Group) -> List[Dict[str, Any]]:
        return [{k: cls._read_value(parent[key][k]) for k in parent[key]} for key in sorted(parent)]

    def save(self, file_path: Path) -> None:
        file_path = Path(file_path)
        gf.ensure_directory_exists(file_path.parent)
        with h5py.File(file_path, "w") as f:
            f.attrs["format_version"] = self.FORMAT_VERSION
            f.attrs["name"] = self.name
            f.attrs["model_sha256"] = self.model_sha256 or ""
            f.attrs["config_sha256"] = self.config_sha256 or ""
            f.attrs["config"] = json.dumps(dataclasses.asdict(self.config))
            f.attrs["input_duration_seconds"] = self.input_duration_seconds
            f.attrs["offsource_duration_seconds"] = self.offsource_duration_seconds

            if self.far_scores is not None:
                g = f.create_group("far_scores")
                g.create_dataset("scores", data=self.far_scores)
                self._write_value(g, "gps_times", self.noise_gps_times)
                if self.noise_streams is not None:
                    streams = g.create_group("streams")
                    for k, v in self.noise_streams.items():
                        streams.create_dataset(k, data=v)
            for key in ("efficiency_data", "volume_data"):
                data = getattr(self, key)
                if data is not None:
                    g = f.create_group(key)
                    for k, v in data.items():
                        g.create_dataset(k, data=v)

            worst = f.create_group("worst_performers")
            fn = worst.create_group("false_negatives")
            for bin_key, samples in self.worst_performers.get("false_negatives", {}).items():
                self._write_samples(fn.create_group(bin_key), samples)
            self._write_samples(worst.create_group("false_positives"),
                                self.worst_performers.get("false_positives", []))

            if self.real_events is not None:
                self._write_samples(f.create_group("real_events"), self.real_events)

    @classmethod
    def load(cls, file_path: Path) -> "Validator":
        """
        Read saved results. Provenance is checked by `validate`, not here, so
        that results can be loaded for comparison plots.

        Raises:
            StaleCheckpointError: the record is of another format: nothing in it is read.
        """
        with h5py.File(file_path, "r") as f:
            found = f.attrs.get("format_version")
            if found != cls.FORMAT_VERSION:
                raise StaleCheckpointError(
                    _other_format(file_path, None if found is None else int(found), cls.FORMAT_VERSION))
            config = ValidationConfig(**json.loads(f.attrs["config"]))
            v = cls(str(f.attrs["name"]), config, {
                "onsource_duration_seconds": float(f.attrs["input_duration_seconds"]),
                "offsource_duration_seconds": float(f.attrs["offsource_duration_seconds"]),
            })
            v.model_sha256 = str(f.attrs["model_sha256"]) or None
            v.config_sha256 = str(f.attrs["config_sha256"]) or None
            if "far_scores" in f:
                v.far_scores = f["far_scores/scores"][:]
                v.noise_gps_times = f["far_scores/gps_times"][:] if "gps_times" in f["far_scores"] else None
                if "streams" in f["far_scores"]:
                    v.noise_streams = {k: f["far_scores/streams"][k][:] for k in f["far_scores/streams"]}
            for key in ("efficiency_data", "volume_data"):
                if key in f:
                    setattr(v, key, {k: f[key][k][()] for k in f[key]})
            worst = f["worst_performers"]
            v.worst_performers = {
                "false_negatives": {k: cls._read_samples(worst["false_negatives"][k])
                                    for k in worst["false_negatives"]},
                "false_positives": cls._read_samples(worst["false_positives"]),
            }
            if "real_events" in f:
                v.real_events = cls._read_samples(f["real_events"])
        return v

    # ---------------------------------------------------------- plotting

    def _cached_segments(self) -> Optional[np.ndarray]:
        """The real-noise segments the noise obtainer already holds, for the GPS plot."""
        noise = (self.dataset_args or {}).get("noise_obtainer")
        obtainer = getattr(noise, "ifo_data_obtainer", None)
        plan = getattr(obtainer, "sampling_plan", None)
        segments = plan.rows if plan is not None else getattr(obtainer, "valid_segments", None)
        return None if segments is None else np.asarray(segments)

    def plot(self, output_path: Path = None, comparison_validators: Optional[List["Validator"]] = None):
        """Generate the validation dashboard (a Panel template); save it if given a path."""
        import panel as pn
        from .plotting import (
            generate_efficiency_plot,
            generate_far_curves,
            generate_gps_distribution_plot,
            generate_roc_curves,
            generate_parameter_space_plot,
            generate_sensitive_volume_plot,
            generate_waveform_plot,
            generate_real_events_table,
        )

        validators = [*(comparison_validators or []), self]
        eff_plot, eff_controls = generate_efficiency_plot(validators)
        far_tab = [generate_far_curves(validators)]
        if any(v.noise_gps_times is not None for v in validators):
            far_tab.append(generate_gps_distribution_plot(
                validators,
                valid_segments=self._cached_segments(),
                onsource_duration_seconds=self.input_duration_seconds,
                offsource_duration_seconds=self.offsource_duration_seconds,
            ))
        roc_plot, roc_select = generate_roc_curves(validators)
        param_plot, param_controls = generate_parameter_space_plot(validators)

        volume_plot = generate_sensitive_volume_plot(validators)
        tabs = pn.Tabs(
            ("Efficiency", pn.Column(eff_controls, eff_plot)),
            *([("Sensitive Volume", volume_plot)] if any(p.renderers for p in volume_plot.children) else []),
            ("ROC Curves", pn.Column(roc_select, roc_plot)),
            ("FAR Curves", pn.Column(*far_tab)),
            ("Parameter Space", pn.Column(param_controls, param_plot) if param_controls else param_plot),
        )
        if self.real_events:
            events_table, _ = generate_real_events_table(self.real_events, self.input_duration_seconds)
            tabs.append(("Real Events", events_table))

        fn = self.worst_performers.get("false_negatives", {})
        fn_tabs = []
        for bin_key in sorted(fn, key=lambda k: float(k.split("-")[0])):
            plots = [generate_waveform_plot(s, self.input_duration_seconds) for s in fn[bin_key][:10]]
            if plots:
                fn_tabs.append((f"SNR {bin_key}", pn.Column(*plots)))
        if fn_tabs:
            tabs.append(("Worst False Negatives", pn.Tabs(*fn_tabs)))
        fp = self.worst_performers.get("false_positives", [])
        if fp:
            tabs.append(("Worst False Positives", pn.Column(
                *[generate_waveform_plot(s, self.input_duration_seconds) for s in fp[:25]])))

        template = pn.template.FastListTemplate(
            title=f"Validation Dashboard: {self.name}", theme="dark", theme_toggle=False, main=[tabs],
        )
        if output_path:
            output_path = Path(output_path)
            gf.ensure_directory_exists(output_path.parent)
            template.save(str(output_path), resources="inline")
            logger.info(f"Saved report to {output_path}")
        return template
