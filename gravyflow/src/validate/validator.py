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
from .efficiency import EfficiencyAgainstSnr, efficiency_against_snr
from .metrics import Background, roc_curve, sigma_level
from .provenance import StaleCheckpointError, model_fingerprint, require_match
from .stream import stream_background
from .volume import (
    BinnedVolume, InjectionVolumes, SensitiveVolumeCurve, SensitiveVt, VolumeRatio, binned_volume, chirp_weights,
    sensitive_volume_curve, sensitive_vt, volume_ratio,
)
from . import family
from gravyflow.src.dataset.features.injection import chirp_mass
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
    # 6: the volume set placed per mass: chirp distances (``chirp_distances``) drawn
    # uniformly in volume out to its recorded chirp reach (``reach_mpc``), with their
    # physical distances; intervals at 1 and 2 sigma (``interval_sigmas``).
    # 7: the mode (``config.mode``): stream mode's records (the default) hold every
    # stream window's score with its streams' extents and stride, and each
    # injection's found statistic (-inf: no trigger within the found window), its
    # trigger's time from the merger, its merger's time and its records; window
    # mode's are format 6's.
    FORMAT_VERSION = 7

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
        # Glitch robustness (`glitches`): each labelled glitch and its statistic; each background
        # trigger's nearest glitch. Timing (`timing`): the timing record and the training cost.
        self.glitch_data: Optional[Dict[str, np.ndarray]] = None
        self.trigger_glitches: Optional[Dict[str, np.ndarray]] = None
        self.timing: Optional[Dict[str, Any]] = None
        self.training_cost: Optional[Dict[str, Any]] = None

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

        In stream mode (``config.mode``, the default) the model is slid along the
        data and its triggers are the background, and each injection is scored
        locally in that data (`stream`); in window mode each example is one window.

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
                         "worst_performers", "real_events", *cls._EXTRA_RECORDS):
                setattr(v, attr, getattr(saved, attr))
            logger.info(f"Resuming validation from {path}.")

        def checkpoint():
            if path is not None:
                v.save(path)

        stream = v.config.stream
        if v.far_scores is None:
            bank.generate_stream_noise() if stream else bank.generate_noise()
            v.far_scores, v.noise_gps_times, v.noise_streams = (
                bank.far_scores, bank.noise_gps_times, bank.noise_streams)
            v.worst_performers["false_positives"] = bank.get_worst_performers()["false_positives"]
            checkpoint()
        bank.far_scores, bank.noise_streams = v.far_scores, v.noise_streams

        if v.efficiency_data is None:
            bank.generate_stream_injections() if stream else bank.generate()
            v.efficiency_data = bank.get_efficiency_data()
            v.worst_performers["false_negatives"] = bank.get_worst_performers()["false_negatives"]
            checkpoint()

        if v.volume_data is None and bank.has_volume:
            bank.generate_stream_volume() if stream else bank.generate_volume()
            v.volume_data = bank.get_volume_data()
            checkpoint()

        if v.config.score_real_events and v.real_events is None:
            bank.generate_real_events()
            v.real_events = bank.get_real_events()
            checkpoint()

        if v.config.score_glitches and v.glitch_data is None:
            from .glitches import glitch_data, trigger_glitch_record

            bank.generate_glitches()
            v.glitch_data = glitch_data(bank.glitch_stage.scored, bank.glitch_stage.scores)
            v.trigger_glitches = trigger_glitch_record(v, bank.glitch_stage.all, bank.glitch_ifos)
            checkpoint()

        v.bank = bank
        return v

    # ----------------------------------------------------------- metrics

    def background(self) -> Background:
        """
        The noise scores as a background. Stream mode: the triggers of the streams the
        model was slid along (`stream.StreamBackground`). Window mode: streams of
        time-slid windows of real noise, or independently drawn windows of synthetic
        noise (`noise_background`).
        """
        if self.far_scores is None:
            raise ValueError(f"Validator {self.name!r} has no noise scores.")
        if self.config.stream:
            return stream_background(self.far_scores, self.noise_streams, self.input_duration_seconds,
                                     self.config.stream_cluster_seconds, self.config.found_window_seconds)
        return noise_background(self.far_scores, self.noise_streams, self.input_duration_seconds,
                                self.config.cluster_seconds)

    def efficiency_curves(self, fars_hz=None) -> Dict[float, EfficiencyAgainstSnr]:
        """
        Efficiency against SNR at each target FAR (Hz) (`efficiency`): the binned record,
        the rolling curve (±0.5 SNR) and its sigmoid fit with the 50% and 90% SNR points,
        the chance of a noise-only pass subtracted, each with joint intervals at every
        level of ``interval_sigmas``. The bootstrap draws from the validator's own seed,
        so a record gives the same curves whenever it is read.
        """
        return self._efficiency(self.efficiency_data, fars_hz)

    def volume_set_efficiency_curves(self, fars_hz=None) -> Optional[Dict[float, EfficiencyAgainstSnr]]:
        """
        The volume set's own efficiency against SNR (its injections' SNRs are recorded): a
        cross-check on the SNR set's, which it should match where the model depends on SNR
        alone. None without a volume set.
        """
        return None if self.volume_data is None else self._efficiency(self.volume_data, fars_hz)

    def _efficiency(self, data, fars_hz) -> Dict[float, EfficiencyAgainstSnr]:
        cfg = self.config
        fars = cfg.far_thresholds if fars_hz is None else fars_hz
        results = efficiency_against_snr(
            data["snrs"], data["scores"], self.background(), fars,
            gf.SeedTree(cfg.seed).stream(cfg.group, "validator/efficiency"), bin_edges=cfg.snr_bin_edges,
            snr_range=cfg.snr_range, num_bootstrap=cfg.efficiency_bootstrap_samples, sigmas=cfg.interval_sigmas,
            method=cfg.efficiency_interval)
        return {float(far): result for far, result in zip(fars, results)}

    def sensitive_volume_curve(self, fars_hz=None) -> Optional[SensitiveVolumeCurve]:
        """
        Sensitive volume (Gpc^3) and distance (Mpc) within the per-mass reach the volume
        set was drawn to, at each target FAR (Hz), the chance of a noise-only pass
        subtracted, with their injection, background and joint intervals at every level
        of ``interval_sigmas`` (`volume`; Euclidean, detector-frame masses); each found
        injection weighted by its mass's reach volume (`chirp_weights`). None without a
        volume set (a burst population, or WHITE noise). The bootstrap draws from the
        validator's own seed, so a record gives the same curve whenever it is read.
        """
        data = self.volume_data
        if self.far_scores is None or data is None:
            return None
        cfg = self.config
        return sensitive_volume_curve(
            self.injection_volumes(), self.background(), cfg.far_thresholds if fars_hz is None else fars_hz,
            gf.SeedTree(cfg.seed).stream(cfg.group, "validator/volume"),
            num_bootstrap=cfg.volume_bootstrap_samples, sigmas=cfg.interval_sigmas, method=cfg.efficiency_interval,
            chance_fraction=cfg.volume_chance_fraction)

    def injection_volumes(self) -> Optional[InjectionVolumes]:
        """The volume set as the volume reads it: chirp distances within the chirp reach, chirp-mass weights."""
        data = self.volume_data
        if data is None:
            return None
        return InjectionVolumes(data["scores"], float(data["reach_mpc"]), data["chirp_distances"],
                                chirp_weights(data["mass1"], data["mass2"]))

    def roc_curves(self) -> Dict[str, Dict[str, Any]]:
        """
        Exact ROC curves of noise against injection populations, keyed by population:
        the noise is every window in window mode, and every background trigger in stream
        mode, as the injections' scores are triggers' there.
        """
        if self.far_scores is None or self.efficiency_data is None:
            return {}
        negatives = self.background().window_scores if self.config.stream else self.far_scores
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
            roc = roc_curve(negatives, scores[mask])
            results[key] = {"fpr": roc.false_positive_rate, "tpr": roc.true_positive_rate,
                            "roc_auc": roc.auc}
        return results

    # ------------------------------------------- the metric family (`family`)

    def _rng(self, name: str) -> np.random.Generator:
        """A metric's own random stream, from the validator's seed: a record gives the same result whenever read."""
        return gf.SeedTree(self.config.seed).stream(self.config.group, f"validator/{name}")

    def _fars(self, fars_hz) -> List[float]:
        return list(self.config.far_thresholds if fars_hz is None else fars_hz)

    @property
    def live_time_seconds(self) -> float:
        """The live time of the data searched (`family.live_time_seconds`): VT's."""
        return family.live_time_seconds(self.background())

    def detection_roc(self, fars_hz=None) -> Optional[family.DetectionRoc]:
        """Detection probability against FAR per SNR band and over every injection (`family.detection_roc`)."""
        if self.far_scores is None or self.efficiency_data is None:
            return None
        cfg = self.config
        lo, hi = cfg.snr_range
        bands = tuple((max(a, lo), min(b, hi)) for a, b in family.ROC_SNR_BANDS if a < hi and b > lo)
        return family.detection_roc(
            self.efficiency_data["snrs"], self.efficiency_data["scores"], self.background(), self._fars(fars_hz),
            self._rng("roc"), bands=bands, num_bootstrap=cfg.efficiency_bootstrap_samples, sigmas=cfg.interval_sigmas,
            method=cfg.efficiency_interval)

    def sensitive_vt(self, fars_hz=None) -> Optional[SensitiveVt]:
        """VT (Gpc^3 yr): the sensitive volume within the reach times the live time searched (`volume.sensitive_vt`)."""
        curve = self.sensitive_volume_curve(fars_hz)
        return None if curve is None else sensitive_vt(curve, self.live_time_seconds)

    def binned_volumes(self, fars_hz=None, num_bins: int = 5) -> Dict[str, BinnedVolume]:
        """
        The sensitive volume within the reach in ``num_bins`` equal bins of chirp mass, of mass
        ratio (``m2 / m1 <= 1``) and, where the volume set records spins (``spin1z``, ``spin2z``),
        of the effective spin (`volume.binned_volume`): the bins span the set's own range, the
        same for every model on the same injections. Empty without a volume set.
        """
        data = self.volume_data
        if self.far_scores is None or data is None:
            return {}
        cfg = self.config
        m1, m2 = np.maximum(data["mass1"], data["mass2"]), np.minimum(data["mass1"], data["mass2"])
        parameters = {"chirp mass (M☉)": chirp_mass(m1, m2), "mass ratio": m2 / m1}
        if "spin1z" in data and "spin2z" in data:
            parameters["effective spin"] = (m1 * data["spin1z"] + m2 * data["spin2z"]) / (m1 + m2)
        injections, background, fars = self.injection_volumes(), self.background(), self._fars(fars_hz)
        results = {}
        for name, values in parameters.items():
            lo, hi = float(np.min(values)), float(np.max(values))
            edges = np.linspace(lo, hi if hi > lo else lo + 1.0, num_bins + 1)
            results[name] = binned_volume(injections, background, fars, values, edges, self._rng("volume"),
                                          parameter=name, num_bootstrap=cfg.volume_bootstrap_samples,
                                          sigmas=cfg.interval_sigmas, method=cfg.efficiency_interval,
                                          chance_fraction=cfg.volume_chance_fraction)
        return results

    def recovered_counts(self, fars_hz=None) -> Dict[str, family.RecoveredCounts]:
        """The injections found at each target FAR, of the SNR set and of the volume set (`family.recovered_counts`)."""
        if self.far_scores is None:
            return {}
        cfg, background, fars = self.config, self.background(), self._fars(fars_hz)
        sets = {"SNR set": self.efficiency_data, "volume set": self.volume_data}
        return {name: family.recovered_counts(data["scores"], background, fars, sigmas=cfg.interval_sigmas,
                                              method=cfg.efficiency_interval)
                for name, data in sets.items() if data is not None}

    def timing_accuracy(self, fars_hz=None) -> Optional[family.TimingAccuracy]:
        """Trigger time against merger of the SNR set's found injections (`family.timing_accuracy`): stream mode's."""
        data = self.efficiency_data
        if self.far_scores is None or data is None or "trigger_offsets" not in data:
            return None
        return family.timing_accuracy(data["scores"], data["trigger_offsets"], self.background(), self._fars(fars_hz),
                                      sigmas=self.config.interval_sigmas)

    def far_calibration(self, fars_hz=None) -> Optional[family.FarCalibration]:
        """The FAR claimed on half the background checked on the other (`family.far_calibration`); None unsplittable."""
        if self.far_scores is None:
            return None
        halves = family.held_out_split(self.background())
        if halves is None:
            return None
        return family.far_calibration(*halves, self._fars(fars_hz), sigmas=self.config.interval_sigmas)

    def real_event_summary(self, fars_hz=None, training_periods=None) -> Optional[family.RealEventSummary]:
        """
        The real events' statistics, FARs and counts at each target (`family.real_event_summary`),
        those inside ``training_periods`` (GPS ``(start, end)``: an as-published model's) flagged.
        """
        if self.far_scores is None or not self.real_events:
            return None
        return family.real_event_summary(self.real_events, self.background(),
                                         family.FAR_LEVELS_HZ if fars_hz is None else fars_hz,
                                         training_periods=training_periods, sigmas=self.config.interval_sigmas)

    def matched_filter_ratio(self, matched_filter: "Validator", fars_hz=None) -> Optional[VolumeRatio]:
        """
        This model's sensitive volume over the matched filter's, on the same injections and
        background data (`volume.volume_ratio`): ``matched_filter`` is the matched filter's
        result through this validator. None where either has no volume set.
        """
        if (self.volume_data is None or matched_filter is None or matched_filter.volume_data is None
                or self.far_scores is None or matched_filter.far_scores is None):
            return None
        cfg = self.config
        return volume_ratio(self.injection_volumes(), self.background(), matched_filter.injection_volumes(),
                            matched_filter.background(), self._fars(fars_hz), self._rng("volume"),
                            num_bootstrap=cfg.volume_bootstrap_samples, sigmas=cfg.interval_sigmas)

    # -------------------------------------------------------- checkpoint

    #: Records kept beside the stages' results, written when present and read back when found.
    _EXTRA_RECORDS = ("glitch_data", "trigger_glitches", "timing", "training_cost")

    def _write_extra(self, f: h5py.File) -> None:
        for key in ("glitch_data", "trigger_glitches"):
            data = getattr(self, key)
            if data is not None:
                g = f.create_group(key)
                for k, v in data.items():
                    v = np.asarray(v)
                    g.create_dataset(k, data=v.astype("S") if v.dtype.kind == "U" else v)
        for key in ("timing", "training_cost"):
            if getattr(self, key) is not None:
                f.attrs[key] = json.dumps(getattr(self, key))

    def _read_extra(self, f: h5py.File) -> None:
        for key in ("glitch_data", "trigger_glitches"):
            if key in f:
                setattr(self, key, {k: (f[key][k][()].astype(str) if f[key][k].dtype.kind == "S" else f[key][k][()])
                                    for k in f[key]})
        for key in ("timing", "training_cost"):
            if key in f.attrs:
                setattr(self, key, json.loads(f.attrs[key]))

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
            self._write_extra(f)

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
                    v.noise_streams = {k: f["far_scores/streams"][k][()] for k in f["far_scores/streams"]}
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
            v._read_extra(f)
        return v

    # ---------------------------------------------------------- plotting

    def mode_summary(self) -> str:
        """How the result was measured, for its dashboard's header: the mode, and in stream mode its stride and rules."""
        cfg = self.config
        if not cfg.stream:
            return f"{self.name}: window mode (each example one window; kept for compatibility and reproduction)."
        stride = (float(np.asarray(self.noise_streams["stride_seconds"])) if self.noise_streams is not None
                  and "stride_seconds" in self.noise_streams else cfg.stream_stride_seconds(self.input_duration_seconds))
        return (f"{self.name}: stream mode, stride {stride:g} s over a {self.input_duration_seconds:g} s window; a "
                f"trigger is the loudest score within {cfg.stream_cluster_seconds:g} s; an injection is found by the "
                f"loudest trigger within ±{cfg.found_window_seconds:g} s of its merger.")

    def _cached_segments(self) -> Optional[np.ndarray]:
        """The real-noise segments the noise obtainer already holds, for the GPS plot."""
        noise = (self.dataset_args or {}).get("noise_obtainer")
        obtainer = getattr(noise, "ifo_data_obtainer", None)
        plan = getattr(obtainer, "sampling_plan", None)
        segments = plan.rows if plan is not None else getattr(obtainer, "valid_segments", None)
        return None if segments is None else np.asarray(segments)

    def plot(self, output_path: Path = None, comparison_validators: Optional[List["Validator"]] = None,
             matched_filter: Optional["Validator"] = None, training_periods=None,
             twins: Optional[Dict[str, "Validator"]] = None):
        """
        Generate the validation dashboard (a Panel template); save it if given a path.

        Its tabs: Summary (the headline numbers at each false-alarm-rate level), Efficiency,
        Volume, ROC and false alarms, Parameter Space, Real events, Glitches, Timing and
        the worst cases.
        ``matched_filter``: the matched filter's result on the same injections and data,
        for the ratio of volumes; ``training_periods``: GPS ``(start, end)`` pairs of an
        as-published model's training data, whose real events are flagged; ``twins`` maps
        a validator's name to the validator of its Gaussian twin, for the Glitches tab's
        real-against-twin comparison.
        """
        import panel as pn
        from .family_plots import (
            generate_binned_volume_plot,
            generate_detection_roc_plot,
            generate_far_calibration_plot,
            generate_real_event_levels,
            generate_summary_tab,
            generate_vt_plot,
        )
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
        from .tabs import generate_glitch_tab, generate_timing_tab

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
        has_volume = any(getattr(child, "renderers", None) for child in volume_plot.children)
        detection_plot, detection_select = generate_detection_roc_plot(validators)
        window_roc_note = pn.pane.Markdown(
            "**Window-level ROC** (true against false positives per scored window; in stream mode per background "
            "trigger), kept as a bridge to papers that report only that.", sizing_mode="stretch_width")
        tabs = pn.Tabs(
            ("Summary", generate_summary_tab(validators, matched_filter)),
            ("Efficiency", pn.Column(eff_controls, eff_plot)),
            *([("Volume", pn.Column(volume_plot, generate_vt_plot(validators),
                                    generate_binned_volume_plot(validators)))] if has_volume else []),
            ("ROC and false alarms", pn.Column(detection_select, detection_plot, *far_tab,
                                               generate_far_calibration_plot(validators), window_roc_note,
                                               roc_select, roc_plot)),
            ("Parameter Space", pn.Column(param_controls, param_plot) if param_controls else param_plot),
        )
        if self.real_events:
            events_table, _ = generate_real_events_table(self.real_events, self.input_duration_seconds)
            tabs.append(("Real events", pn.Column(generate_real_event_levels(self, training_periods), events_table)))
        tabs.append(("Glitches", generate_glitch_tab(validators, twins)))
        tabs.append(("Timing", generate_timing_tab(validators)))

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

        from .volume_3d import generate_volume_3d_plot
        volume_3d = generate_volume_3d_plot(self)
        if volume_3d is not None:
            tabs.append(("3D Volume", volume_3d))

        header = pn.pane.Markdown("  \n".join(v.mode_summary() for v in validators), sizing_mode="stretch_width")
        template = pn.template.FastListTemplate(
            title=f"Validation Dashboard: {self.name}", theme="dark", theme_toggle=False, main=[header, tabs],
        )
        if output_path:
            output_path = Path(output_path)
            gf.ensure_directory_exists(output_path.parent)
            template.save(str(output_path), resources="inline")
            logger.info(f"Saved report to {output_path}")
        return template
