"""
The per-detector path: a model that scores each detector separately is scored
once per detector, and every time slide then costs only the coincidence step.

- **Scoring.** Each detector's strain over each of the network's segments is cut
  into the model's windows at its stride (`stream.StreamGeometry`) and scored
  once, one detector at a time (`score_detectors`): `DetectorScores`.
- **The ring.** Segment s holds ``n_s`` whole windows and stands on the ring of
  wrapping slides (`slides.WrappedSlides`) as ``n_s`` strides from its first
  window's data: window i of the ring is at ``i * stride``. A lag is a whole number
  of strides, so in slide k detector j's ring window ``i + j m_k`` (wrapped) is
  paired with detector 0's window i: across a segment's end into the next segment,
  and from the last back to the first.
- **Coincidence.** A slide's network scores combine each ring window's paired
  per-detector scores (``combine``, by default their sum): one `metrics.ScoreStream`
  per slide, the whole ring long, each window one stride of livetime (a ring has no
  ends), so ``K`` slides give the slides' `background_seconds` (`slide_streams`). Its
  triggers and false-alarm rates are stream mode's (`stream.StreamBackground`).
- **The joint path** (`joint_slide_streams`) scores the very same pairings with a
  model that looks at every detector at once: a full run per slide. For a model
  whose network score is ``combine`` of per-detector scores, the two paths give the
  same streams, so the same background, and the same found injections
  (`found_statistic`).
- **In the validator.** A model declares itself per-detector by its wrapper,
  `PerDetectorModel`, with its combination rule; with wrapping slides
  (``ValidationConfig.wrapping_slides``) its record keeps each detector's scores and
  the slides (`ring_record`), from which the background is rebuilt
  (`ring_background_streams`).
- **Stored triggers.** The matched filter's per-detector triggers
  (`detection.DetectorTriggers`) take the same slides: detector j's trigger times are
  moved round the ring (`slides.WrappedSlides.slid_times`) and coincided with
  detector 0's (`detection.coincide`) in each slide (`slide_coincidences`).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from gravyflow.src.detection.matched_filter import COINCIDENCE_MARGIN_SECONDS
from gravyflow.src.detection.stream import Coincidences, DetectorTriggers, coincide
from .metrics import ScoreStream
from .slides import WrappedSlides
from .stream import StreamGeometry, cluster, loudest_within

#: A per-detector model: windows of one detector (onsource (B, 1, P), offsource (B, 1, O)) and its index -> (B,) scores.
DetectorModel = Callable[[np.ndarray, np.ndarray, int], np.ndarray]
#: A joint model: windows of every detector (onsource (B, I, P), offsource (B, I, O)) -> (B,) scores.
JointModel = Callable[[np.ndarray, np.ndarray], np.ndarray]


def _sum(scores: np.ndarray) -> np.ndarray:
    return np.sum(scores, axis=0)


#: The combination rules a per-detector model may declare (`PerDetectorModel`): (detectors, N) -> (N,).
COMBINE_RULES = {"sum": _sum, "max": lambda scores: np.max(scores, axis=0)}


def combine_rule(name: str) -> Callable[[np.ndarray], np.ndarray]:
    """The combination rule ``name`` (`COMBINE_RULES`)."""
    if isinstance(name, bytes):
        name = name.decode()
    if name not in COMBINE_RULES:
        raise ValueError(f"combine must be one of {sorted(COMBINE_RULES)}, got {name!r}.")
    return COMBINE_RULES[name]


class PerDetectorModel:
    """
    A model that scores each detector separately, with the rule (``combine``, a name
    of `COMBINE_RULES`) that makes its network score of the detectors' scores.

    ``model`` (a Keras logit model) takes one detector's model inputs (each input's
    detector axis, axis 1, of length 1). As a network model (`predict_on_batch`) it
    splits each input along its detector axis, scores each detector, and combines:
    so its injections, scored at zero lag, pass the very combination its
    background does. With wrapping slides (``ValidationConfig.wrapping_slides``) the
    validator scores each detector's data once and builds every slide's background
    from those scores (`slide_streams`). Anything else (its layers, weights, training
    record) is the wrapped model's.
    """

    def __init__(self, model, combine: str = "sum"):
        self.model = model
        self.combine = combine.decode() if isinstance(combine, bytes) else str(combine)
        self.combine_scores = combine_rule(self.combine)

    def __getattr__(self, name):
        if name == "model":  # not yet set: never recurse
            raise AttributeError(name)
        return getattr(self.model, name)

    def detector_scores(self, inputs) -> np.ndarray:
        """Each detector's scores of network ``inputs`` (each (B, I, ...)): (I, B)."""
        from .metrics import logit_scores

        arrays = {k: np.asarray(v) for k, v in inputs.items()}
        num = {a.shape[1] for a in arrays.values()}
        if len(num) != 1:
            raise ValueError(f"The inputs hold different numbers of detectors on axis 1: {sorted(num)}.")
        return np.stack([logit_scores(self.model.predict_on_batch({k: a[:, j:j + 1] for k, a in arrays.items()}))
                         for j in range(num.pop())])

    def predict_on_batch(self, inputs) -> np.ndarray:
        """The network scores of ``inputs``: each detector scored, then combined (B,)."""
        return np.asarray(self.combine_scores(self.detector_scores(inputs)), dtype=np.float64)


@dataclass(frozen=True)
class DetectorScores:
    """
    Each detector's window scores over the network's segments, in ring order:
    ``scores`` (detectors, N), ``segment_windows`` the windows of each segment and
    ``first_gps`` the GPS time of each segment's first sample.
    """
    scores: np.ndarray
    segment_windows: np.ndarray
    first_gps: np.ndarray
    geometry: StreamGeometry

    @property
    def num_detectors(self) -> int:
        return int(self.scores.shape[0])

    @property
    def segments(self) -> np.ndarray:
        """Each segment's place on the ring: its first window's data, then ``n_s`` strides."""
        first = np.asarray(self.first_gps, dtype=np.float64)
        return np.stack([first, first + self.segment_windows * self.geometry.stride_seconds], axis=1)

    def slides(self, minimum_seconds: float, step_seconds: Optional[float] = None,
               max_slides: Optional[int] = None) -> WrappedSlides:
        """
        Wrapping slides of these segments, ``k step`` apart: ``step`` a whole number of
        strides, by default the fewest that exceed ``minimum_seconds``.
        """
        stride = self.geometry.stride_seconds
        if step_seconds is None:
            step = (np.floor(minimum_seconds / stride + 1e-9) + 1) * stride
        else:
            step = float(step_seconds)
            if abs(step / stride - round(step / stride)) > 1e-6:
                raise ValueError(f"step_seconds {step_seconds} is not a whole number of {stride} s strides.")
        return WrappedSlides.spaced(self.segments, self.num_detectors, minimum_seconds, step, max_slides)


def _locate(segment_windows: np.ndarray, ring_index: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """The segment and window within it of each ring window (taken round the ring)."""
    starts = np.concatenate([[0], np.cumsum(segment_windows)[:-1]])
    ring_index = np.mod(ring_index, int(np.sum(segment_windows)))
    segment = np.searchsorted(starts, ring_index, side="right") - 1
    return segment, ring_index - starts[segment]


def _strides(slides: WrappedSlides, slide: int, stride_seconds: float) -> int:
    """Slide ``slide``'s lag in strides, refused unless whole."""
    lag = slides.lags[slide] / stride_seconds
    whole = int(round(lag))
    if abs(lag - whole) > 1e-6:
        raise ValueError(f"Slide {slide}'s lag {slides.lags[slide]} s is not a whole number of "
                         f"{stride_seconds} s strides.")
    return whole


def _check_ring(slides: WrappedSlides, segments: np.ndarray) -> None:
    if slides.segments.shape != segments.shape or not np.allclose(slides.segments, segments, rtol=0, atol=1e-6):
        raise ValueError("The slides were planned on other segments than the scores' ring.")


def score_detectors(model: DetectorModel, strains: Sequence[np.ndarray], first_gps: Sequence[float],
                    geometry: StreamGeometry, batch_size: int = 256) -> DetectorScores:
    """
    Score each detector's strain once: ``strains`` holds each segment's strain
    (detectors, samples), ``first_gps`` its first sample's GPS time; every whole
    window of every segment, at the stride, through ``model`` one detector at a time.
    """
    if len(strains) != len(first_gps):
        raise ValueError("Give one first-sample GPS time per segment.")
    if np.any(np.diff(np.asarray(first_gps, dtype=np.float64)) <= 0):
        raise ValueError("Give the segments in time order: the ring lays them end to end in that order.")
    counts = np.array([geometry.num_windows(np.shape(s)[1]) for s in strains], dtype=np.int64)
    if np.any(counts < 1):
        raise ValueError(f"Segment {int(np.flatnonzero(counts < 1)[0])} holds no whole window of the model's.")
    num_detectors = int(np.shape(strains[0])[0])
    rows = []
    for j in range(num_detectors):
        parts = []
        for strain, n in zip(strains, counts):
            detector = np.asarray(strain, dtype=np.float64)[j:j + 1]
            for first in range(0, int(n), batch_size):
                index = np.arange(first, min(first + batch_size, int(n)))
                onsource, offsource = (np.asarray(a, dtype=np.float64)
                                       for a in geometry.gather(detector, index * geometry.stride_samples))
                parts.append(np.asarray(model(onsource, offsource, j), dtype=np.float64).reshape(-1))
        rows.append(np.concatenate(parts))
    return DetectorScores(np.stack(rows), counts, np.asarray(first_gps, dtype=np.float64), geometry)


def slide_streams(scores: DetectorScores, slides: WrappedSlides,
                  combine: Callable[[np.ndarray], np.ndarray] = _sum) -> List[ScoreStream]:
    """
    The per-detector path's background: in each slide, each ring window's paired
    per-detector scores combined (``combine``: (detectors, N) -> (N,)), as one stream
    per slide (slide k numbered k + 1), the ring one block.
    """
    _check_ring(slides, scores.segments)
    stride = scores.geometry.stride_seconds
    streams = []
    for k in range(len(slides)):
        m = _strides(slides, k, stride)
        paired = np.stack([np.roll(scores.scores[j], -j * m) for j in range(scores.num_detectors)])
        streams.append(_ring_stream(np.asarray(combine(paired), dtype=np.float64), stride, k, m,
                                    scores.num_detectors))
    return streams


def _ring_stream(scores: np.ndarray, stride: float, slide: int, lag_strides: int, num_detectors: int) -> ScoreStream:
    """Slide ``slide``'s stream round the ring: detector j's data of ring window i are its ring window ``i + j m``."""
    return ScoreStream(scores, stride, stride, 0, slide + 1,
                       detector_offsets=tuple(j * lag_strides * stride for j in range(num_detectors)),
                       ring_seconds=scores.size * stride)


def paired_windows(geometry: StreamGeometry, segment_windows: np.ndarray, slides: WrappedSlides, slide: int,
                   ring_index: np.ndarray, num_detectors: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Each detector's (segment, window) paired with detector 0's ring windows ``ring_index`` in ``slide``."""
    m = _strides(slides, slide, geometry.stride_seconds)
    return [_locate(segment_windows, np.asarray(ring_index) + j * m) for j in range(num_detectors)]


def _gather_paired(strains, geometry, pairs) -> Tuple[np.ndarray, np.ndarray]:
    """The paired windows' data: onsource (B, I, P) and offsource (B, I, O)."""
    onsource, offsource = [], []
    for j, (segment, window) in enumerate(pairs):
        on = np.empty((segment.size, geometry.window.num_onsource_samples))
        off = np.empty((segment.size, geometry.window.num_offsource_samples))
        for s in np.unique(segment):
            rows = segment == s
            detector = np.asarray(strains[s], dtype=np.float64)[j:j + 1]
            o, f = geometry.gather(detector, window[rows] * geometry.stride_samples)
            on[rows], off[rows] = o[:, 0], f[:, 0]
        onsource.append(on)
        offsource.append(off)
    return np.stack(onsource, axis=1), np.stack(offsource, axis=1)


def joint_slide_streams(model: JointModel, strains: Sequence[np.ndarray], first_gps: Sequence[float],
                        geometry: StreamGeometry, slides: WrappedSlides, batch_size: int = 256) -> List[ScoreStream]:
    """
    The joint path's background: in each slide, every ring window of detector 0 with
    its paired windows (`paired_windows`) scored at once by ``model``; a full run per
    slide. Streams as `slide_streams` gives them.
    """
    counts = np.array([geometry.num_windows(np.shape(s)[1]) for s in strains], dtype=np.int64)
    first = np.asarray(first_gps, dtype=np.float64)
    _check_ring(slides, np.stack([first, first + counts * geometry.stride_seconds], axis=1))
    total, num_detectors = int(counts.sum()), int(np.shape(strains[0])[0])
    streams = []
    for k in range(len(slides)):
        parts = []
        for start in range(0, total, batch_size):
            ring = np.arange(start, min(start + batch_size, total))
            pairs = paired_windows(geometry, counts, slides, k, ring, num_detectors)
            parts.append(np.asarray(model(*_gather_paired(strains, geometry, pairs)), dtype=np.float64).reshape(-1))
        stride = geometry.stride_seconds
        streams.append(_ring_stream(np.concatenate(parts), stride, k, _strides(slides, k, stride), num_detectors))
    return streams


def ring_record(slides: WrappedSlides, geometry: StreamGeometry, segment_windows, combine: Optional[str] = None
                ) -> dict:
    """
    What a validation record keeps of wrapping slides beside its zero-lag segments, so
    that its background is rebuilt from its scores (`ring_background_streams`): each
    slide's lag, where each segment's first window's data start, the least shift the
    lags kept clear of, the detectors and, for a per-detector model, its combination
    rule (the record's scores then each detector's, not each slide's).
    """
    record = {"ring_lags_seconds": np.asarray(slides.lags, dtype=np.float64),
              "ring_first_seconds": np.asarray(slides.segments[:, 0], dtype=np.float64),
              "ring_segment_windows": np.asarray(segment_windows, dtype=np.int64),
              "ring_minimum_seconds": np.float64(slides.minimum_seconds),
              "ring_detectors": np.int64(slides.num_detectors)}
    if combine is not None:
        record["ring_combine"] = str(combine)
    return record


def ring_background_streams(scores, record: Mapping, stride_seconds: float) -> List[ScoreStream]:
    """
    The wrapping slides' streams of a record (`ring_record`): from each detector's
    scores (`slide_streams`) where the record names a combination rule, else the
    slides' own network scores, slide after slide.
    """
    windows = np.asarray(record["ring_segment_windows"], dtype=np.int64)
    first = np.asarray(record["ring_first_seconds"], dtype=np.float64)
    num_detectors = int(np.asarray(record["ring_detectors"]))
    slides = WrappedSlides(np.stack([first, first + windows * stride_seconds], axis=1),
                           record["ring_lags_seconds"], num_detectors,
                           float(np.asarray(record["ring_minimum_seconds"])))
    scores = np.asarray(scores, dtype=np.float64)
    total = int(windows.sum())
    if "ring_combine" in record:
        rule = np.asarray(record["ring_combine"]).item()
        detector = DetectorScores(scores.reshape(num_detectors, total), windows, first,
                                  _StrideOnly(stride_seconds))
        return slide_streams(detector, slides, combine_rule(rule))
    if scores.size != len(slides) * total:
        raise ValueError(f"The record holds {scores.size} scores, not {len(slides)} slides of {total} windows.")
    return [_ring_stream(part, stride_seconds, k, _strides(slides, k, stride_seconds), num_detectors)
            for k, part in enumerate(scores.reshape(len(slides), total))]


@dataclass(frozen=True)
class _StrideOnly:
    """A ring's geometry as a record keeps it: its stride."""
    stride_seconds: float


def found_statistic(window_scores: np.ndarray, ring_index: np.ndarray, merger_ring_seconds: float,
                    geometry: StreamGeometry, cluster_seconds: float, found_window_seconds: float) -> float:
    """
    An injection's found statistic from the network scores of the ring windows
    around it (``ring_index``, their scores with the signal added): the loudest
    trigger within ``found_window_seconds`` of its merger, at ring time
    ``merger_ring_seconds``; ``-inf`` for none. Either path's local scores give it.
    """
    times = np.asarray(ring_index, dtype=np.float64) * geometry.stride_seconds \
        + geometry.centre_samples / geometry.sample_rate_hertz
    scores = np.asarray(window_scores, dtype=np.float64)
    keep = cluster(times, scores, cluster_seconds)
    stat, _ = loudest_within(times[keep], scores[keep], [merger_ring_seconds], found_window_seconds)
    return float(stat[0])


def slide_coincidences(triggers: Mapping[str, DetectorTriggers], slides: WrappedSlides, cluster_seconds: float = 1.0,
                       margin_seconds: float = COINCIDENCE_MARGIN_SECONDS) -> List[Coincidences]:
    """
    The coincidences of stored per-detector triggers (the first detector the
    reference) in each wrapping slide: detector j's trigger times moved round the
    ring (`slides.WrappedSlides.slid_times`), then `detection.coincide`. Every trigger
    must lie inside the slides' segments; gates outside them are dropped. The
    background they span is the slides' `background_seconds`.
    """
    names = list(triggers)
    if len(names) != slides.num_detectors:
        raise ValueError(f"The slides were planned for {slides.num_detectors} detectors, not {len(names)}.")
    for name in names:
        slides.to_ring(triggers[name].time)  # refuses a trigger outside the segments
    result = []
    for k in range(len(slides)):
        moved = {names[0]: triggers[names[0]]}
        for j, name in enumerate(names[1:], start=1):
            original = triggers[name]
            gates = original.gates
            inside = np.zeros(gates.size, dtype=bool)
            if gates.size:
                index = np.searchsorted(slides.segments[:, 0], gates, side="right") - 1
                inside = (index >= 0) & (gates < slides.segments[np.clip(index, 0, None), 1])
            moved[name] = replace(original, time=slides.slid_times(original.time, k, j),
                                  gates=slides.slid_times(gates[inside], k, j))
        result.append(coincide(moved, cluster_seconds, margin_seconds))
    return result


__all__ = [
    "COMBINE_RULES",
    "DetectorScores",
    "PerDetectorModel",
    "combine_rule",
    "found_statistic",
    "joint_slide_streams",
    "paired_windows",
    "ring_background_streams",
    "ring_record",
    "score_detectors",
    "slide_coincidences",
    "slide_streams",
]
