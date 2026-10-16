"""
Scores a model on validation data: a noise background (for false-alarm rates),
injections (for efficiency) and, optionally, real GW events.

Scores are logits (see `metrics`). The validator draws its own data: each stage
builds its dataset from the seed tree rooted at ``ValidationConfig.seed``,
scoped to ``validator/{stage}`` (``far`` or ``inj``), in the held-out
``ValidationConfig.group``. Training never draws under ``validator/``, so the
validator's streams cannot coincide with training's, whatever the seeds.
"""

from __future__ import annotations

import heapq
import logging
import math
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

import gravyflow as gf
from .config import ValidationConfig
from .metrics import Background, logit_scores
from .utils import _extract_sample_data

logger = logging.getLogger(__name__)

# Per-example labels of the injection set, by attribute name.
_INJECTION_LABELS = {
    "snrs": gf.ScalingTypes.SNR,
    "hpeak": gf.ScalingTypes.HPEAK,
    "hrss": gf.ScalingTypes.HRSS,
    "mass1": gf.WaveformParameters.MASS_1_MSUN,
    "mass2": gf.WaveformParameters.MASS_2_MSUN,
    "central_times": gf.ReturnVariables.CENTRAL_TIME,
}
_CBC_LABELS = ("mass1", "mass2")


def require_logit_model(model) -> None:
    """Refuse a model whose output is a probability: saturated probabilities tie."""
    last = model.layers[-1]
    activation = getattr(getattr(last, "activation", None), "__name__", None)
    if activation in ("softmax", "sigmoid") or type(last).__name__ == "Softmax":
        raise ValueError(
            f"Model {model.name!r} outputs probabilities ({activation or 'Softmax'}). The "
            "validator ranks logits: build the output layer without an activation, and train "
            "with a loss that takes logits (from_logits=True)."
        )


def real_event_inputs(model, onsource, offsource, ifos, sample_rate_hertz, onsource_seconds):
    """
    A model's inputs for real events, prepared as the dataset prepares its examples:
    whitened by the offsource and cropped to the onsource, with the rolling Pearson
    correlation over the dataset's lag range. Returns the whitened onsource and the
    inputs by name.
    """
    from gravyflow.src.dataset.conditioning.whiten import whiten
    from gravyflow.src.dataset.dataset import get_max_arrival_time_difference

    whitened = np.asarray(gf.crop_samples(
        whiten(onsource, offsource, sample_rate_hertz), onsource_seconds, sample_rate_hertz))
    inputs = {
        gf.ReturnVariables.WHITENED_ONSOURCE.name: lambda: whitened,
        gf.ReturnVariables.ONSOURCE.name: lambda: np.asarray(
            gf.crop_samples(onsource, onsource_seconds, sample_rate_hertz)),
        gf.ReturnVariables.OFFSOURCE.name: lambda: offsource,
        gf.ReturnVariables.ROLLING_PEARSON_ONSOURCE.name: lambda: np.asarray(gf.rolling_pearson(
            whitened, get_max_arrival_time_difference(ifos), sample_rate_hertz)),
    }
    names = [t.name for t in model.inputs]
    missing = [name for name in names if name not in inputs]
    if missing:
        raise ValueError(f"Real events cannot provide the model's inputs {missing}.")
    return whitened, {name: np.asarray(inputs[name](), dtype=np.float32) for name in names}


def per_example(value, n: int, name: str, per_detector: bool = False) -> np.ndarray:
    """
    One float64 value per example, from a label array of any singleton layout.

    ``per_detector`` labels are batch-first, (B, I), or (B, G, I) from one
    generator, and give the first detector's value. Anything else that is not
    one value per example raises: two generators' values are never merged.
    """
    array = np.asarray(value, dtype=np.float64)
    if per_detector and array.ndim in (2, 3) and array.shape[0] == n and array.shape[1:-1] in ((), (1,)):
        return array[..., 0].reshape(n)
    squeezed = np.squeeze(array)
    if squeezed.shape == (n,) or (n == 1 and squeezed.ndim == 0):
        return squeezed.reshape(n)
    raise ValueError(f"Label {name}: expected one value per example ({n}), got shape {array.shape}.")


class ValidationBank:
    """Generates and stores a model's validation scores, one stage at a time."""

    def __init__(
        self,
        model,
        dataset_args: dict,
        config: Optional[ValidationConfig] = None,
        heart: Optional[gf.Heart] = None,
    ):
        require_logit_model(model)
        if not dataset_args.get("input_variables"):
            raise ValueError("dataset_args must name the model's input_variables.")
        self.model = model
        self.dataset_args = dataset_args
        self.config = config or ValidationConfig()
        self.heart = heart

        self.far_scores: Optional[np.ndarray] = None
        self.noise_gps_times: Optional[np.ndarray] = None
        self.efficiency: Optional[Dict[str, np.ndarray]] = None
        self.real_events: Optional[List[Dict[str, Any]]] = None
        self._worst_false_positives: List[Dict[str, Any]] = []
        self._worst_false_negatives: Dict[str, List[Dict[str, Any]]] = {}

    # ------------------------------------------------------------ helpers

    @property
    def window_seconds(self) -> float:
        return float(self.dataset_args.get("onsource_duration_seconds", gf.Defaults.onsource_duration_seconds))

    @property
    def _real_noise(self) -> bool:
        noise = self.dataset_args.get("noise_obtainer")
        return getattr(noise, "noise_type", None) is gf.NoiseType.REAL

    def _num_batches(self) -> int:
        return math.ceil(self.config.num_examples / self.config.batch_size)

    def _dataset(self, stage: str, waveform_generators: list, outputs: list) -> gf.Dataset:
        args = deepcopy(self.dataset_args)
        args.update(
            waveform_generators=waveform_generators,
            num_examples_per_batch=self.config.batch_size,
            steps_per_epoch=self._num_batches(),
            output_variables=outputs,
            group=self.config.group,
            seed=gf.SeedTree(self.config.seed).sub(f"validator/{stage}"),
        )
        if self._real_noise:
            args["output_variables"] = [*outputs, gf.ReturnVariables.START_GPS_TIME]
        return gf.Dataset(**args)

    def _score(self, x) -> np.ndarray:
        if self.heart:
            self.heart.beat()
        return logit_scores(self.model.predict_on_batch(x))

    def _gps(self, y, n: int) -> Optional[np.ndarray]:
        if not self._real_noise:
            return None
        return per_example(y[gf.ReturnVariables.START_GPS_TIME.name], n, "START_GPS_TIME", per_detector=True)

    # ------------------------------------------------------------- stages

    def generate_noise(self) -> None:
        """Score noise-only examples: the background for false-alarm rates."""
        dataset = self._dataset("far", [], [gf.ReturnVariables.WHITENED_ONSOURCE])
        keep = self.config.num_worst_per_bin * 5
        scores, gps, heap = [], [], []
        for b in tqdm(range(self._num_batches()), desc="Scoring noise", unit="batch"):
            x, y = dataset[b]
            y = y or {}
            batch_scores = self._score(x)
            scores.append(batch_scores)
            batch_gps = self._gps(y, batch_scores.size)
            if batch_gps is not None:
                gps.append(batch_gps)
            for i, s in enumerate(batch_scores):
                key = (float(s), -(b * self.config.batch_size + i))  # ties: earliest example
                if len(heap) < keep:
                    heapq.heappush(heap, (key, _extract_sample_data(x, y, i, float(s))))
                elif key > heap[0][0]:
                    heapq.heapreplace(heap, (key, _extract_sample_data(x, y, i, float(s))))
        self.far_scores = np.concatenate(scores)[: self.config.num_examples]
        self.noise_gps_times = np.concatenate(gps)[: self.config.num_examples] if gps else None
        self._worst_false_positives = [sample for _, sample in sorted(heap, key=lambda t: t[0], reverse=True)]

    def generate(self) -> None:
        """Score injections (every example injected, SNR uniform over the range)."""
        cfg = self.config
        generators = self.dataset_args.get("waveform_generators")
        generators = generators if isinstance(generators, list) else [generators]
        if len(generators) != 1 or generators[0] is None:
            raise ValueError("The injection set takes exactly one waveform generator.")
        generator = deepcopy(generators[0])
        generator.scaling_method.value = gf.Distribution(
            min_=cfg.snr_range[0], max_=cfg.snr_range[1], type_=gf.DistributionType.UNIFORM
        )
        generator.injection_chance = 1.0
        # Component masses exist only for CBC injections (a burst set has none).
        labels = {k: v for k, v in _INJECTION_LABELS.items()
                  if k not in _CBC_LABELS or isinstance(generator, gf.CBCGenerator)}
        outputs = [gf.ReturnVariables.WHITENED_ONSOURCE, gf.ReturnVariables.WHITENED_INJECTIONS,
                   *labels.values()]
        dataset = self._dataset("inj", [generator], outputs)

        edges = cfg.snr_bin_edges
        worst: List[list] = [[] for _ in range(edges.size - 1)]
        columns: Dict[str, list] = {k: [] for k in ("scores", "gps_times", *labels)}
        for b in tqdm(range(self._num_batches()), desc="Scoring injections", unit="batch"):
            x, y = dataset[b]
            batch_scores = self._score(x)
            n = batch_scores.size
            columns["scores"].append(batch_scores)
            for key, label in labels.items():
                # The central time is per detector, (B, G=1, I): keep the first detector's.
                columns[key].append(per_example(y[label.name], n, label.name,
                                                per_detector=key == "central_times"))
            batch_gps = self._gps(y, n)
            if batch_gps is not None:
                columns["gps_times"].append(batch_gps)
            bins = np.clip(np.searchsorted(edges, columns["snrs"][-1], side="right") - 1, 0, len(worst) - 1)
            for i, (bin_index, s) in enumerate(zip(bins, batch_scores)):
                key = (-float(s), -(b * cfg.batch_size + i))  # lowest scores are the worst
                heap = worst[bin_index]
                if len(heap) < cfg.num_worst_per_bin:
                    heapq.heappush(heap, (key, _extract_sample_data(x, y, i, float(s))))
                elif key > heap[0][0]:
                    heapq.heapreplace(heap, (key, _extract_sample_data(x, y, i, float(s))))

        n_total = cfg.num_examples
        self.efficiency = {k: np.concatenate(v)[:n_total] for k, v in columns.items() if v}
        self._worst_false_negatives = {
            f"{edges[i]:g}-{edges[i + 1]:g}": [s for _, s in sorted(heap, key=lambda t: t[0], reverse=True)]
            for i, heap in enumerate(worst)
        }

    def generate_real_events(self, observing_runs: List = None) -> None:
        """
        Score real GW events from the GWTC catalogues, for evaluation only.

        Needs network access. Each event is prepared as the dataset prepares
        its examples (whitened by its offsource, cropped to the onsource), so
        its score is comparable with the background's, and its FAR is read
        from that background.
        """
        from gravyflow.src.dataset.features.event import EventConfidence, get_events_with_params
        from gravyflow.src.dataset.dataset import noise_ifos

        if self.far_scores is None:
            raise ValueError("Score the noise background (generate_noise) before real events.")
        if observing_runs is None:
            observing_runs = [gf.ObservingRun.O1, gf.ObservingRun.O2, gf.ObservingRun.O3]
        ifos = noise_ifos(self.dataset_args.get("ifos"), self.dataset_args.get("noise_obtainer"))
        letters = {ifo.name[0] for ifo in ifos}

        events, seen = [], set()
        for e in sorted(get_events_with_params(
                observing_runs, event_types=[EventConfidence.CONFIDENT, EventConfidence.MARGINAL]),
                key=lambda e: e["gps"]):
            key = round(e["gps"], 1)
            network = e.get("network", "")
            if key in seen or (network and not letters <= set(network)):
                continue
            seen.add(key)
            e["event_type"] = "MARGINAL" if "marginal" in str(e.get("catalog", "")).lower() else "CONFIDENT"
            events.append(e)
        logger.info(f"Scoring {len(events)} real events.")

        sample_rate = self.dataset_args.get("sample_rate_hertz", gf.Defaults.sample_rate_hertz)
        onsource_seconds = self.window_seconds
        obtainer = gf.TransientDataObtainer(
            observing_runs=observing_runs,
            data_quality=gf.DataQuality.BEST,
            data_labels=[gf.DataLabel.EVENTS],
            event_names=[e["name"] for e in events],
        )
        chunks = obtainer.get_onsource_offsource_chunks(
            sample_rate_hertz=sample_rate,
            onsource_duration_seconds=onsource_seconds,
            offsource_duration_seconds=self.dataset_args.get("offsource_duration_seconds", 16.0),
            num_examples_per_batch=min(len(events), 256) or 1,
            ifos=ifos,
            scale_factor=gf.Defaults.scale_factor,
        )
        onsource, offsource, gps = [], [], []
        for batch in chunks:
            on = np.asarray(batch[gf.ReturnVariables.ONSOURCE])
            off = np.asarray(batch[gf.ReturnVariables.OFFSOURCE])
            times = np.asarray(batch.get(gf.ReturnVariables.TRANSIENT_GPS_TIME,
                                         batch.get(gf.ReturnVariables.START_GPS_TIME)))
            on, off = (on, off) if on.ndim == 3 else (on[None], off[None])
            onsource.append(on)
            offsource.append(off)
            gps.extend(per_example(times, on.shape[0], "event GPS", per_detector=True))
            if len(gps) >= len(events):
                break

        background = self.background()
        if onsource:
            whitened, x = real_event_inputs(
                self.model, np.concatenate(onsource), np.concatenate(offsource),
                ifos, sample_rate, onsource_seconds,
            )
            for i, (t, s) in enumerate(zip(gps, logit_scores(self.model.predict(x, verbose=0)))):
                event = next((e for e in events if abs(e["gps"] - t) < 20.0), None)
                if event is None:
                    continue
                far = background.far(float(s), self.config.confidence)
                event.update(score=float(s), status="scored", far_hz=far.rate_hz,
                             far_upper_hz=far.upper_hz, whitened_strain=whitened[i])
        self.real_events = events

    # ------------------------------------------------------------- results

    def background(self) -> Background:
        """The noise scores as a background of independently drawn windows."""
        if self.far_scores is None:
            raise ValueError("Call generate_noise() first.")
        return Background.independent_windows(self.far_scores, self.window_seconds)

    def get_efficiency_data(self) -> Dict[str, np.ndarray]:
        """The injection set's scores and labels, one entry per example."""
        if self.efficiency is None:
            raise ValueError("Call generate() first.")
        return dict(self.efficiency)

    def get_worst_performers(self) -> Dict[str, Any]:
        """Lowest-scoring injections per SNR bin, and highest-scoring noise."""
        return {"false_negatives": self._worst_false_negatives,
                "false_positives": self._worst_false_positives}

    def get_real_events(self) -> List[Dict[str, Any]]:
        """Real events, CONFIDENT first, then MARGINAL."""
        events = self.real_events or []
        return ([e for e in events if e.get("event_type") == "CONFIDENT"]
                + [e for e in events if e.get("event_type") == "MARGINAL"])
