"""
Scores a model on validation data: a noise background (for false-alarm rates),
injections (for efficiency) and, optionally, real GW events.

Scores are logits (see `metrics`). The validator draws its own data: each stage
builds its dataset from the seed tree rooted at ``ValidationConfig.seed``,
scoped to ``validator/{stage}`` (``far``, ``inj``, ``vol`` or ``horizon``), in the held-out
``ValidationConfig.group``. Training never draws under ``validator/``, so the
validator's streams cannot coincide with training's, whatever the seeds.

Every window the validator scores, whatever its origin (the dataset's own
noise, time-slid real noise, a real event), is conditioned by its stage
dataset's code (``Dataset.model_inputs``) in that dataset's window geometry, so
every score is comparable with every other.
"""

from __future__ import annotations

import heapq
import logging
import math
from copy import deepcopy
from functools import cached_property
from typing import Any, Dict, List, Optional, Protocol

import numpy as np
from tqdm import tqdm

import gravyflow as gf
from .config import ValidationConfig
from .metrics import Background, ScoreStream, logit_scores
from .slides import TimeSlideBackground, TimeSlides
from .volume import uniform_in_volume
from .utils import _extract_sample_data

logger = logging.getLogger(__name__)


class Heartbeat(Protocol):
    """What the validator beats as it scores, for a supervisor that watches the process: anything with ``beat()``."""

    def beat(self) -> None: ...

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
#: The noise windows whose median PSD the reach is computed in (`ValidationBank.reach_distance`).
REACH_NOISE_WINDOWS = 64
# Per-example labels of the sensitive-volume set (`generate_volume`).
_VOLUME_LABELS = {
    "distances": gf.WaveformParameters.DISTANCE_MPC,
    "snrs": gf.ScalingTypes.SNR,
    "mass1": gf.WaveformParameters.MASS_1_MSUN,
    "mass2": gf.WaveformParameters.MASS_2_MSUN,
}


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


def noise_background(scores, streams: Optional[Dict[str, np.ndarray]], window_seconds: float,
                     cluster_seconds: Optional[float]) -> Background:
    """
    The background of noise ``scores``. With ``streams`` (each stream's number of
    windows, slide and block, in the order of the scores), streams of windows tiled
    one window apart, whose above-threshold windows starting within
    ``cluster_seconds`` (default: one window) of the one before are one trigger;
    without, windows drawn independently.
    """
    if streams is None:
        return Background.independent_windows(scores, window_seconds)
    edges = np.cumsum(streams["lengths"])[:-1]
    return Background(
        [ScoreStream(part, window_seconds, window_seconds, int(block), int(slide))
         for part, block, slide in zip(np.split(np.asarray(scores), edges), streams["blocks"], streams["slides"])],
        window_seconds if cluster_seconds is None else cluster_seconds,
    )


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


def event_times(batch) -> np.ndarray:
    """
    Each example's real-event time, from a transient batch: its ``TRANSIENT_GPS_TIME``
    on the first detector labelled EVENTS. A detector the event was not seen in is
    read at a partner time, away from the event, and labelled NOISE.
    """
    times = np.asarray(batch[gf.ReturnVariables.TRANSIENT_GPS_TIME], dtype=np.float64)
    seen = np.asarray(batch[gf.ReturnVariables.DATA_LABEL]) == gf.DataLabel.EVENTS.value
    if not np.all(np.any(seen, axis=1)):
        raise ValueError("An example of a real event is labelled EVENTS on no detector.")
    return times[np.arange(len(times)), np.argmax(seen, axis=1)]


class ValidationBank:
    """Generates and stores a model's validation scores, one stage at a time."""

    def __init__(
        self,
        model,
        dataset_args: dict,
        config: Optional[ValidationConfig] = None,
        heart: Optional[Heartbeat] = None,
    ):
        require_logit_model(model)
        if not dataset_args.get("input_variables"):
            raise ValueError("dataset_args must name the model's input_variables.")
        self.model = model
        self.dataset_args = dataset_args
        self.config = config or ValidationConfig()
        self.heart = heart
        if self.has_volume and self.config.volume_reach_mpc is None:
            # A population with no loudest source has no reach: refused here, before any stage runs.
            from .reach import population_box
            population_box(self._generator())

        self.far_scores: Optional[np.ndarray] = None
        self.noise_gps_times: Optional[np.ndarray] = None
        # Time slides' streams: their windows, slides and blocks (`noise_background`).
        self.noise_streams: Optional[Dict[str, np.ndarray]] = None
        self.efficiency: Optional[Dict[str, np.ndarray]] = None
        self.volume: Optional[Dict[str, np.ndarray]] = None
        self.real_events: Optional[List[Dict[str, Any]]] = None
        self._worst_false_positives: List[Dict[str, Any]] = []
        self._worst_false_negatives: Dict[str, List[Dict[str, Any]]] = {}

    # ------------------------------------------------------------ helpers

    @property
    def window_seconds(self) -> float:
        """
        The length of every scored window, and the stride of a background's
        streams: the noise dataset's window (`Dataset.window_spec`), the one its
        time slides tile, so a stream's livetime is the span of the windows scored.
        """
        return float(self._noise_dataset.window_spec.onsource_duration_seconds)

    @property
    def _real_noise(self) -> bool:
        noise = self.dataset_args.get("noise_obtainer")
        return getattr(noise, "noise_type", None) is gf.NoiseType.REAL

    def _num_batches(self, num_examples: Optional[int] = None) -> int:
        return math.ceil((num_examples or self.config.num_examples) / self.config.batch_size)

    def stage_dataset(self, stage: str, waveform_generators=(),
                      outputs=(gf.ReturnVariables.WHITENED_ONSOURCE,),
                      num_examples: Optional[int] = None, batch_size: Optional[int] = None) -> gf.Dataset:
        """
        Stage ``stage``'s dataset (``far``, ``inj``, ``vol`` or ``horizon``) of
        ``num_examples`` (default ``num_examples``) in batches of ``batch_size``
        (default ``batch_size``): its own seeds, in the held-out split.
        """
        batch_size = batch_size or self.config.batch_size
        args = deepcopy(self.dataset_args)
        args.update(
            waveform_generators=list(waveform_generators),
            num_examples_per_batch=batch_size,
            steps_per_epoch=math.ceil((num_examples or self.config.num_examples) / batch_size),
            output_variables=list(outputs),
            group=self.config.group,
            seed=gf.SeedTree(self.config.seed).sub(f"validator/{stage}"),
        )
        if self._real_noise:
            args["output_variables"] = [*outputs, gf.ReturnVariables.START_GPS_TIME]
        return gf.Dataset(**args)

    @cached_property
    def _noise_dataset(self) -> gf.Dataset:
        """
        The noise background's dataset, built once per bank (after `split_spec` has
        been checked): `generate_noise` scores its windows, or its conditioning of
        time-slid ones, and real events are conditioned by it too (`model_inputs`).
        """
        return self.stage_dataset("far")

    @property
    def split_spec(self) -> Optional[gf.SplitSpec]:
        """
        How the real data the bank reads is split: the split every stage that
        reads real data reads under, and so the one a model is checked against
        (`check_held_out`). The noise stages read the noise obtainer, split by the
        one computation of a split (`split_spec_of`, which `Dataset.split_spec`
        makes; none named, the default noise, whose split is the default). The
        real-events stage, when it is on (`score_real_events`), reads real events
        under the noise's split; beside synthetic noise, which reads no real data
        and names none, they are the bank's only real data, and are split by the
        default, as real data that names no split is. None therefore means only
        that no stage reads real data (synthetic noise alone, split-neutral), never
        real data left unchecked. It reads specifications, not data, so a model is
        refused before any dataset is built or any strain fetched.
        """
        from gravyflow.src.dataset.dataset import split_spec_of

        spec = split_spec_of([self.dataset_args.get("noise_obtainer")])
        if spec is None and self.config.score_real_events:
            return gf.DEFAULT_SPLIT
        return spec

    def _score(self, x) -> np.ndarray:
        if self.heart:
            self.heart.beat()
        return logit_scores(self.model.predict_on_batch(x))

    def score_windows(self, dataset: gf.Dataset, onsource, offsource) -> np.ndarray:
        """
        Scores of raw windows in ``dataset``'s geometry (padded onsource, offsource),
        conditioned by ``dataset.model_inputs``: exactly as its own examples are scored.
        """
        return self._score(dataset.model_inputs(onsource, offsource))

    def _gps(self, y, n: int) -> Optional[np.ndarray]:
        if not self._real_noise:
            return None
        return per_example(y[gf.ReturnVariables.START_GPS_TIME.name], n, "START_GPS_TIME", per_detector=True)

    # ------------------------------------------------------------- stages

    def generate_noise(self) -> None:
        """
        Score noise-only windows: the background for false-alarm rates. Real noise
        is time-slid (`slides`); synthetic noise has no time to slide, so its
        windows are drawn independently by the dataset.
        """
        if self._real_noise:
            self._generate_time_slides()
            return
        dataset = self._noise_dataset
        batches = (dataset[b] for b in range(self._num_batches()))
        self._score_noise((self._score(x), x, y or {}) for x, y in batches)

    def _generate_time_slides(self) -> None:
        from gravyflow.src.dataset.dataset import noise_ifos

        cfg = self.config
        dataset = self._noise_dataset
        ifos = noise_ifos(self.dataset_args.get("ifos"), self.dataset_args.get("noise_obtainer"))
        background = TimeSlideBackground(
            dataset.noise_obtainer.ifo_data_obtainer, ifos, self.split_spec, cfg.group, dataset.window_spec,
            TimeSlides(cfg.num_time_slides, cfg.time_slide_step_seconds),
            gf.SeedTree(cfg.seed).sub("validator/far").stream(cfg.group, "noise/segment_order"),
        )
        streams = []

        def scored():
            for batch in background.batches(cfg.batch_size, cfg.num_examples):
                if not streams or streams[-1] is not batch.stream:
                    streams.append(batch.stream)
                x = dataset.model_inputs(batch.onsource, batch.offsource)
                yield self._score(x), x, {gf.ReturnVariables.START_GPS_TIME.name: batch.start_gps}

        self._score_noise(scored())
        self.noise_streams = {
            "lengths": np.array([stream.num_windows for stream in streams], dtype=np.int64),
            "slides": np.array([stream.slide for stream in streams], dtype=np.int64),
            "blocks": np.array([stream.block for stream in streams], dtype=np.int64),
        }
        logger.info(f"Time slides: {len(streams)} streams from {len(background.stretches)} stretches.")

    def _score_noise(self, batches) -> None:
        """Keep the scores, GPS times and highest-scoring examples of (scores, inputs, outputs) batches."""
        keep = self.config.num_worst_per_bin * 5
        scores, gps, heap = [], [], []
        served = 0
        for batch_scores, x, y in tqdm(batches, desc="Scoring noise", unit="batch"):
            scores.append(batch_scores)
            batch_gps = self._gps(y, batch_scores.size)
            if batch_gps is not None:
                gps.append(batch_gps)
            for i, s in enumerate(batch_scores):
                key = (float(s), -(served + i))  # ties: earliest example
                if len(heap) < keep:
                    heapq.heappush(heap, (key, _extract_sample_data(x, y, i, float(s))))
                elif key > heap[0][0]:
                    heapq.heapreplace(heap, (key, _extract_sample_data(x, y, i, float(s))))
            served += batch_scores.size
        if not scores:
            raise ValueError(f"The '{self.config.group}' split gave no noise windows to score.")
        self.far_scores = np.concatenate(scores)[: self.config.num_examples]
        self.noise_gps_times = np.concatenate(gps)[: self.config.num_examples] if gps else None
        self._worst_false_positives = [sample for _, sample in sorted(heap, key=lambda t: t[0], reverse=True)]

    def _generators(self) -> list:
        """The dataset's waveform generators, as a list."""
        generators = self.dataset_args.get("waveform_generators")
        return [g for g in (generators if isinstance(generators, list) else [generators]) if g is not None]

    def _generator(self):
        """A copy of the one waveform generator the injection sets draw from."""
        generators = self._generators()
        if len(generators) != 1:
            raise ValueError("The injection set takes exactly one waveform generator.")
        return deepcopy(generators[0])

    def generate(self) -> None:
        """Score injections (every example injected, SNR uniform over the range)."""
        cfg = self.config
        generator = self._generator()
        # At network SNR over the range, whatever the generator was trained to scale by.
        generator.scaling_method = gf.ScalingMethod(
            value=gf.Distribution(min_=cfg.snr_range[0], max_=cfg.snr_range[1],
                                  type_=gf.DistributionType.UNIFORM),
            type_=gf.ScalingTypes.SNR,
        )
        generator.injection_chance = 1.0
        # Component masses and distances exist only for CBC injections (a burst set has none).
        labels = {k: v for k, v in _INJECTION_LABELS.items()
                  if k not in _CBC_LABELS or isinstance(generator, gf.CBCGenerator)}
        outputs = [gf.ReturnVariables.WHITENED_ONSOURCE, gf.ReturnVariables.WHITENED_INJECTIONS,
                   *labels.values()]
        dataset = self.stage_dataset("inj", [generator], outputs)

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

    # ------------------------------------------------------- sensitive volume

    @property
    def has_volume(self) -> bool:
        """
        Whether the injections have a sensitive volume: one CBC population (whose
        amplitude goes as one over a physical distance) in noise of physical strain
        (not WHITE, which is dimensionless).
        """
        noise = self.dataset_args.get("noise_obtainer")
        generators = self._generators()
        return (len(generators) == 1 and isinstance(generators[0], gf.CBCGenerator)
                and getattr(noise, "noise_type", None) is not gf.NoiseType.WHITE)

    def reach_distance(self) -> float:
        """
        The population's reach (Mpc): the distance at which its loudest possible
        source has network SNR ``volume_reach_snr``, so that every source beyond
        it has less, whatever its sky position, orientation and parameters
        (`reach.reach_distance`: face-on, the best response every detector can
        have, maximised over the population's masses and aligned spins). The noise's
        PSD is the median, over `REACH_NOISE_WINDOWS` windows of the stage's own
        noise, of the data path's estimate (``Dataset.spectral_config``). It reads
        the population's bounds and the noise, never the injections drawn, nor the
        batch size; ``volume_reach_mpc`` overrides it.
        """
        from gravyflow.src.dataset.tools.psd import estimate_psd
        from . import reach

        dataset = self.stage_dataset("horizon", outputs=(gf.ReturnVariables.OFFSOURCE,),
                                     num_examples=REACH_NOISE_WINDOWS, batch_size=REACH_NOISE_WINDOWS)
        _, y = dataset[0]
        spectrum = estimate_psd(y[gf.ReturnVariables.OFFSOURCE.name], dataset.sample_rate_hertz,
                                dataset.spectral_config)
        psds = np.median(spectrum.value(), axis=0)
        return reach.reach_distance(self._generator(), psds, dataset.sample_rate_hertz,
                                    dataset.window_spec.onsource_duration_seconds, dataset.spectral_config,
                                    self.config.volume_reach_snr)

    def generate_volume(self) -> None:
        """
        Score the sensitive-volume set: ``num_volume_injections`` sources from the
        population, each at a physical distance drawn uniformly in volume out to the
        population's reach R (``volume_reach_mpc``, else `reach_distance`):
        the generated source placed at that distance, never scaled to an SNR.
        Distances are Euclidean and masses in the detector frame, as the generator
        draws them.
        """
        cfg = self.config
        if not self.has_volume:
            raise ValueError("Only one CBC population in noise of physical strain has a sensitive volume.")
        n = cfg.num_volume_injections
        reach = cfg.volume_reach_mpc or self.reach_distance()
        targets = uniform_in_volume(n, reach, gf.SeedTree(cfg.seed).stream(cfg.group, "validator/distance"))
        generator = self._generator()
        generator.scaling_method = gf.ScalingMethod(value=targets, type_=gf.WaveformParameters.DISTANCE_MPC)
        generator.injection_chance = 1.0
        outputs = [gf.ReturnVariables.WHITENED_ONSOURCE, *_VOLUME_LABELS.values()]
        dataset = self.stage_dataset("vol", [generator], outputs, num_examples=n)
        columns: Dict[str, list] = {k: [] for k in ("scores", *_VOLUME_LABELS)}
        for b in tqdm(range(self._num_batches(n)), desc="Scoring the volume set", unit="batch"):
            x, y = dataset[b]
            batch_scores = self._score(x)
            columns["scores"].append(batch_scores)
            for key, label in _VOLUME_LABELS.items():
                columns[key].append(per_example(y[label.name], batch_scores.size, label.name))
        volume = {k: np.concatenate(v)[:n] for k, v in columns.items()}
        # The dataset reads the targets in order, batch after batch. The label is the
        # target by construction, so this proves the order, not the placement: that the
        # injected strain is the generated source's at its drawn distance, scaled by
        # drawn over target, is the injection path's contract, and is tested there.
        if not np.allclose(volume["distances"], targets, rtol=1e-5, atol=0):
            raise RuntimeError("The volume set's injections were not given their targets in the order drawn.")
        volume["distances"] = targets
        volume["reach_mpc"] = np.float64(reach)
        self.volume = volume

    def generate_real_events(self, observing_runs: List = None) -> None:
        """
        Score real GW events from the GWTC catalogues, for evaluation only.

        Real events are test data: the confident events, each once (by name),
        are scored when the validator's split is ``test``, and none otherwise;
        marginal events are never examples. ``observing_runs`` defaults to the
        default runs (``DEFAULT_OBSERVING_RUNS``, O1 to O4). Needs network access. Each event is
        read in the background's window geometry and conditioned by the
        background's dataset (`score_windows`), so its score is comparable with
        the background's, and its FAR is read from that background.

        The events are read under the bank's split (`split_spec`), the one the
        model is checked against. A bank whose split counts no real events
        (``score_real_events`` off) beside synthetic noise names none, and refuses
        by name before it reads any.
        """
        from gravyflow.src.dataset.acquisition.base import DEFAULT_OBSERVING_RUNS
        from gravyflow.src.dataset.features.event import EventConfidence, get_events_with_params
        from gravyflow.src.dataset.dataset import noise_ifos

        if self.far_scores is None:
            raise ValueError("Score the noise background (generate_noise) before real events.")
        if observing_runs is None:
            observing_runs = list(DEFAULT_OBSERVING_RUNS)
        ifos = noise_ifos(self.dataset_args.get("ifos"), self.dataset_args.get("noise_obtainer"))

        if self.config.group != "test":
            logger.info(f"No real events are in the '{self.config.group}' split: they are all test data.")
            self.real_events = []
            return
        split_spec = self.split_spec
        if split_spec is None:
            raise ValueError(
                "Real events are real data, read under the split the model is checked against, and this "
                "bank's names none: its noise is synthetic and it scores no real events "
                "(ValidationConfig.score_real_events is False)."
            )
        events, seen = [], set()
        for e in sorted(get_events_with_params(observing_runs, event_types=[EventConfidence.CONFIDENT]),
                        key=lambda e: e["gps"]):
            if e["name"] in seen:
                continue
            seen.add(e["name"])
            e["event_type"] = "CONFIDENT"
            events.append(e)
        logger.info(f"Scoring {len(events)} real events.")

        noise = self.dataset_args.get("noise_obtainer")
        obtainer = gf.TransientDataObtainer(
            observing_runs=observing_runs,
            data_quality=gf.DataQuality.BEST,
            data_labels=[gf.DataLabel.EVENTS],
            event_names=[e["name"] for e in events],
            split_spec=split_spec,
            # The background's source (a NoiseObtainer's reader's, or a transient obtainer's own).
            source=getattr(getattr(noise, "ifo_data_obtainer", noise), "source", None),
        )
        dataset = self._noise_dataset
        window = dataset.window_spec
        chunks = obtainer.get_onsource_offsource_chunks(
            sample_rate_hertz=window.sample_rate_hertz,
            onsource_duration_seconds=window.onsource_duration_seconds,
            crop_duration_seconds=window.crop_duration_seconds,
            offsource_duration_seconds=window.offsource_duration_seconds,
            num_examples_per_batch=min(len(events), 256) or 1,
            ifos=ifos,
            group="test",
        )
        onsource, offsource, gps = [], [], []
        for batch in chunks:
            on = np.asarray(batch[gf.ReturnVariables.ONSOURCE])
            off = np.asarray(batch[gf.ReturnVariables.OFFSOURCE])
            on, off = (on, off) if on.ndim == 3 else (on[None], off[None])
            onsource.append(on)
            offsource.append(off)
            gps.extend(event_times(batch))
            if len(gps) >= len(events):
                break

        background = self.background()
        if onsource:
            x = dataset.model_inputs(np.concatenate(onsource), np.concatenate(offsource))
            whitened = x.get(gf.ReturnVariables.WHITENED_ONSOURCE.name)
            for i, (t, s) in enumerate(zip(gps, self._score(x))):
                event = next((e for e in events if abs(e["gps"] - t) < 20.0), None)
                if event is None:
                    continue
                far = background.far(float(s), self.config.confidence)
                event.update(score=float(s), status="scored", far_hz=far.rate_hz, far_upper_hz=far.upper_hz)
                if whitened is not None:
                    event["whitened_strain"] = np.asarray(whitened[i])
        self.real_events = events

    # ------------------------------------------------------------- results

    def background(self) -> Background:
        """The noise scores as a background: time-slid streams, or independent windows."""
        if self.far_scores is None:
            raise ValueError("Call generate_noise() first.")
        return noise_background(self.far_scores, self.noise_streams, self.window_seconds,
                                self.config.cluster_seconds)

    def get_efficiency_data(self) -> Dict[str, np.ndarray]:
        """The injection set's scores and labels, one entry per example."""
        if self.efficiency is None:
            raise ValueError("Call generate() first.")
        return dict(self.efficiency)

    def get_volume_data(self) -> Dict[str, np.ndarray]:
        """The volume set's scores, distances and labels, and its reach (``reach_mpc``)."""
        if self.volume is None:
            raise ValueError("Call generate_volume() first.")
        return dict(self.volume)

    def get_worst_performers(self) -> Dict[str, Any]:
        """Lowest-scoring injections per SNR bin, and highest-scoring noise."""
        return {"false_negatives": self._worst_false_negatives,
                "false_positives": self._worst_false_positives}

    def get_real_events(self) -> List[Dict[str, Any]]:
        """Real events, CONFIDENT first, then MARGINAL."""
        events = self.real_events or []
        return ([e for e in events if e.get("event_type") == "CONFIDENT"]
                + [e for e in events if e.get("event_type") == "MARGINAL"])
