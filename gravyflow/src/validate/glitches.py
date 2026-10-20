"""
Glitch robustness: how a model answers glitches, the noise that a search's tail is made of.

- **Per-class false positives** (`glitch_false_positive_rates`). The model is run over
  every labelled glitch (`score_glitches`): each glitch's stretch of strain is scored
  as an injection is, locally on the stream's grid (`stream.local_windows`), and the
  glitch's statistic is the loudest trigger within ``found_window_seconds`` of its peak
  (``-inf`` where none lies there). At each false-alarm-rate level, a class's rate is
  the fraction of its glitches whose statistic passes the background's threshold, with
  its binomial interval at every level of ``sigmas``. Labels are Gravity Spy's
  (`dataset.features.gravity_spy`, O1 to O3b, H1 and L1); O4 has no public labels.
- **Which glitches make the tail** (`trigger_glitch_record`, `tail_by_class`). Each
  background trigger's distance to the nearest labelled glitch, in any detector at that
  detector's own (slid) time, and that glitch's class; the classes of the triggers above
  a threshold.
- **Real against its Gaussian twin** (`twin_comparison`). The sensitive volume on real
  data over the volume on its Gaussian twin (coloured Gaussian noise of the same
  segments' spectra, with the same injections), at each false-alarm-rate level: a ratio
  below one is what glitches and non-stationarity cost.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np

from gravyflow.src.dataset.features import gravity_spy
from .metrics import DEFAULT_SIGMAS, binomial_interval, sigma_level
from .stream import StreamGeometry, cluster, found_in_stretch, local_windows


# ------------------------------------------------------------ the glitch list

@dataclass(frozen=True)
class Glitches:
    """
    Labelled glitches: each one's detector, peak GPS time and class, sorted by
    detector then time. ``source`` names where the labels came from.
    """
    ifos: np.ndarray
    gps: np.ndarray
    labels: np.ndarray
    source: str = ""

    def __post_init__(self):
        ifos, gps, labels = (np.asarray(self.ifos).astype(str), np.asarray(self.gps, dtype=np.float64),
                             np.asarray(self.labels).astype(str))
        if not (ifos.shape == gps.shape == labels.shape) or gps.ndim != 1:
            raise ValueError(f"A glitch list's detectors, times and labels must be one-dimensional and of the same "
                             f"length, got {ifos.shape}, {gps.shape} and {labels.shape}.")
        order = np.lexsort((gps, ifos))
        object.__setattr__(self, "ifos", ifos[order])
        object.__setattr__(self, "gps", gps[order])
        object.__setattr__(self, "labels", labels[order])

    def __len__(self) -> int:
        return int(self.gps.size)

    @property
    def classes(self) -> Tuple[str, ...]:
        """The classes present, sorted."""
        return tuple(str(label) for label in np.unique(self.labels))

    def of(self, ifo: str) -> Tuple[np.ndarray, np.ndarray]:
        """Detector ``ifo``'s glitches: their peak times, ascending, and their classes."""
        keep = self.ifos == str(ifo)
        return self.gps[keep], self.labels[keep]

    @classmethod
    def from_gravity_spy(cls, ifos: Sequence[str], start_gps_time: float, end_gps_time: float,
                         min_ml_confidence: float = gravity_spy.DEFAULT_MIN_ML_CONFIDENCE) -> "Glitches":
        """
        The Gravity Spy glitches of ``ifos`` with peaks in [start, end] and a classification
        confidence of at least ``min_ml_confidence`` (`gravity_spy.query`: the public
        release, md5-checked and cached on first use).
        """
        ifo_column, gps, labels = [], [], []
        for ifo in ifos:
            rows = gravity_spy.query(ifo, start_gps_time, end_gps_time, min_ml_confidence=min_ml_confidence)
            gps.append(np.asarray(rows["event_time"], dtype=np.float64))
            labels.append(np.asarray(rows["ml_label"]).astype(str))
            ifo_column.append(np.full(gps[-1].size, str(ifo)))
        return cls(np.concatenate(ifo_column), np.concatenate(gps), np.concatenate(labels),
                   source=f"Gravity Spy, {gravity_spy.RELEASE_CITATION}; ml_confidence >= {min_ml_confidence:g}")


# --------------------------------------------------------- nearest glitch

@dataclass(frozen=True)
class NearestGlitch:
    """
    Each trigger's nearest glitch over every detector: its distance (s), the trigger's
    time less the glitch's in that detector (s), its class, detector and peak time.
    NaN, ``""`` where no detector has a glitch.
    """
    distance_seconds: np.ndarray
    offset_seconds: np.ndarray
    label: np.ndarray
    ifo: np.ndarray
    glitch_gps: np.ndarray


def nearest_glitch(times, ifos: Sequence[str], glitches: Glitches) -> NearestGlitch:
    """
    The nearest glitch to each trigger. ``times`` is (triggers, detectors): each trigger's
    GPS time in each detector of ``ifos`` (in a time slide, each detector's own slid
    time). Ties go to the earlier glitch, then to the earlier detector.
    """
    times = np.asarray(times, dtype=np.float64)
    if times.ndim != 2 or times.shape[1] != len(ifos):
        raise ValueError(f"times must be (triggers, {len(ifos)} detectors), got {times.shape}.")
    n = times.shape[0]
    best = np.full(n, np.inf)
    offset, when = np.full(n, np.nan), np.full(n, np.nan)
    label, which = np.full(n, "", dtype=object), np.full(n, "", dtype=object)
    for j, ifo in enumerate(ifos):
        gps, labels = glitches.of(ifo)
        if gps.size == 0:
            continue
        t = times[:, j]
        right = np.clip(np.searchsorted(gps, t, side="left"), 0, gps.size - 1)
        left = np.clip(right - 1, 0, gps.size - 1)
        pick = np.where(np.abs(t - gps[left]) <= np.abs(t - gps[right]), left, right)
        distance = np.abs(t - gps[pick])
        better = distance < best
        best = np.where(better, distance, best)
        offset = np.where(better, t - gps[pick], offset)
        when = np.where(better, gps[pick], when)
        label[better] = labels[pick][better]
        which[better] = ifo
    return NearestGlitch(np.where(np.isfinite(best), best, np.nan), offset, label.astype(str), which.astype(str), when)


# ------------------------------------------------------- per-class rates

@dataclass(frozen=True)
class GlitchClassRates:
    """
    The false-positive rate of each glitch class at each false-alarm-rate level: of a
    class's ``counts`` glitches, the ``passed`` whose statistic is at or above the level's
    ``threshold``, their fraction ``rate`` (F, C) and its binomial ``lower`` and ``upper``
    bounds at each of ``sigmas`` (S, F, C). NaN at a threshold the background cannot
    resolve.
    """
    labels: Tuple[str, ...]
    counts: np.ndarray
    far_hz: np.ndarray
    threshold: np.ndarray
    passed: np.ndarray
    rate: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    sigmas: Tuple[float, ...]
    method: str

    def to_dict(self) -> Dict:
        return {"labels": list(self.labels), "counts": self.counts.tolist(), "far_hz": self.far_hz.tolist(),
                "threshold": self.threshold.tolist(), "passed": self.passed.tolist(), "rate": self.rate.tolist(),
                "lower": self.lower.tolist(), "upper": self.upper.tolist(), "sigmas": list(self.sigmas),
                "method": self.method}


def glitch_false_positive_rates(scores, labels, thresholds, far_hz=None, sigmas: Sequence[float] = DEFAULT_SIGMAS,
                                method: str = "clopper-pearson") -> GlitchClassRates:
    """
    Each class's false-positive rate at each threshold (`GlitchClassRates`): ``scores``
    are the glitches' statistics (`score_glitches`; ``-inf`` passes nothing), ``labels``
    their classes, ``thresholds`` the background's at each level (``far_hz``, recorded).
    """
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels).astype(str).reshape(-1)
    if scores.size != labels.size:
        raise ValueError(f"Give one label per score: {scores.size} scores, {labels.size} labels.")
    thresholds = np.atleast_1d(np.asarray(thresholds, dtype=np.float64))
    far = np.full(thresholds.shape, np.nan) if far_hz is None else np.atleast_1d(np.asarray(far_hz, dtype=np.float64))
    classes, code = np.unique(labels, return_inverse=True)
    counts = np.bincount(code, minlength=classes.size)
    with np.errstate(invalid="ignore"):
        passing = scores[None, :] >= thresholds[:, None]                       # (F, N); False at NaN
    passed = np.stack([np.bincount(code[row], minlength=classes.size) for row in passing]) if thresholds.size \
        else np.zeros((0, classes.size), dtype=np.int64)
    unresolved = np.isnan(thresholds)[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = np.where(unresolved, np.nan, passed / counts[None, :])
    bounds = [binomial_interval(passed, np.broadcast_to(counts, passed.shape), sigma_level(s), method) for s in sigmas]
    lower = np.stack([np.where(unresolved, np.nan, lo) for lo, _ in bounds])
    upper = np.stack([np.where(unresolved, np.nan, hi) for _, hi in bounds])
    return GlitchClassRates(tuple(str(c) for c in classes), counts, far, thresholds, passed, rate, lower, upper,
                            tuple(float(s) for s in sigmas), method)


def validator_glitch_rates(validator, fars_hz=None) -> Optional[GlitchClassRates]:
    """A validator's per-class rates at its levels, from its glitch record (`Validator.glitch_data`)."""
    data = getattr(validator, "glitch_data", None)
    if data is None or validator.far_scores is None:
        return None
    cfg = validator.config
    fars = list(cfg.far_thresholds if fars_hz is None else fars_hz)
    background = validator.background()
    thresholds = [background.threshold_for_far(f).threshold for f in fars]
    return glitch_false_positive_rates(data["scores"], data["labels"], thresholds, fars, cfg.interval_sigmas,
                                       cfg.efficiency_interval)


# ----------------------------------------------- the model over every glitch

def score_glitches(score: Callable[[np.ndarray, np.ndarray], np.ndarray],
                   strain_around: Callable[[str, float, int], np.ndarray], geometry: StreamGeometry,
                   glitches: Glitches, cluster_seconds: float, found_window_seconds: float,
                   batch_size: int = 64) -> np.ndarray:
    """
    Run the model over every glitch: each glitch's statistic, in the order of ``glitches``.

    ``strain_around(ifo, gps, half_samples)`` gives the network's raw strain (detectors,
    ``2 half_samples``) with the glitch's peak at sample ``half_samples``; ``score(onsource,
    offsource)`` scores windows ((B, I, P), (B, I, O)) as the stream does, conditioning
    included. The windows the found test reads are scored (every window within
    ``found_window_seconds + cluster_seconds`` of the peak, `stream.local_windows`), clustered
    into triggers, and the loudest trigger within ``found_window_seconds`` of the peak is the
    glitch's statistic: ``-inf`` where none lies there.
    """
    reach = float(found_window_seconds) + float(cluster_seconds)
    half = geometry.local_half_samples(reach)
    index, starts = local_windows(geometry, half, half, reach)
    statistics = np.full(len(glitches), -np.inf)
    for first in range(0, len(glitches), int(batch_size)):
        rows = range(first, min(first + int(batch_size), len(glitches)))
        onsource, offsource = [], []
        for k in rows:
            stretch = np.asarray(strain_around(str(glitches.ifos[k]), float(glitches.gps[k]), half), dtype=np.float64)
            if stretch.ndim != 2 or stretch.shape[1] != 2 * half:
                raise ValueError(f"strain_around gave {stretch.shape} for the glitch at {glitches.gps[k]}; a stretch "
                                 f"is (detectors, {2 * half} samples).")
            on, off = geometry.gather(stretch, starts)
            onsource.append(on)
            offsource.append(off)
        scores = np.asarray(score(np.concatenate(onsource), np.concatenate(offsource)), dtype=np.float64)
        scores = scores.reshape(len(rows), index.size)
        for row, k in enumerate(rows):
            statistics[k], _ = found_in_stretch(geometry, index, scores[row], half, cluster_seconds,
                                                found_window_seconds)
    return statistics


@dataclass(frozen=True)
class GlitchStage:
    """
    The glitch stage's result: every labelled glitch of the held-out time (``all``, for
    the background's nearest-glitch record), the glitches scored (``scored``: those with
    room for their stretch, at most ``per_class`` of each class) and their statistics.
    """
    all: Glitches
    scored: Glitches
    scores: np.ndarray


def _subset(glitches: Glitches, keep: np.ndarray) -> Glitches:
    return Glitches(glitches.ifos[keep], glitches.gps[keep], glitches.labels[keep], glitches.source)


def glitch_stage(obtainer, ifos: Sequence, split_spec, group: str, geometry: StreamGeometry,
                 score: Callable[[np.ndarray, np.ndarray], np.ndarray], cluster_seconds: float,
                 found_window_seconds: float, per_class: Optional[int] = None,
                 rng: Optional[np.random.Generator] = None, glitches: Optional[Glitches] = None) -> GlitchStage:
    """
    Run the model over every labelled glitch of split ``group`` of a real-noise obtainer.

    The held-out time is every detector's readable time in the split (`SplitSpec.admissible`),
    intersected, less the known events' vetoes; the glitch veto noise draws under does not
    apply, since the glitches are the point. ``glitches`` defaults to Gravity Spy's over that
    time (`Glitches.from_gravity_spy`, which downloads its catalogue on first use). A glitch
    is scored when its whole stretch lies inside the held-out time: every detector's strain
    at zero lag, read through the obtainer's own ``acquire``, the glitch's peak at the
    stretch's centre (`score_glitches`). ``per_class`` keeps at most that many of each class,
    drawn by ``rng``.
    """
    from gravyflow.src.dataset.acquisition.gwosc_data import intersect_segments, subtract_segments

    ifos = list(ifos)
    names = [getattr(ifo, "name", str(ifo)) for ifo in ifos]
    rate = geometry.sample_rate_hertz
    half = geometry.local_half_samples(float(found_window_seconds) + float(cluster_seconds))
    pad = half / rate + 1.0  # a second's margin each side, for the source's sample grid
    time = None
    for ifo in ifos:
        held_out = split_spec.admissible(obtainer.get_all_segment_times(ifo), group)
        time = held_out if time is None else intersect_segments(time, held_out)
    vetoes = np.reshape(np.asarray(obtainer.get_event_veto_segments(), dtype=np.float64), (-1, 2))
    if vetoes.size:
        time = subtract_segments(time, vetoes)
    time = np.reshape(np.asarray(time, dtype=np.float64), (-1, 2))
    if time.size == 0:
        raise ValueError(f"No held-out time of split {group!r} on {names}: no glitch to score.")
    if glitches is None:
        glitches = Glitches.from_gravity_spy(names, float(time[:, 0].min()), float(time[:, 1].max()))
    everything = _subset(glitches, np.isin(glitches.ifos, names))
    k = np.searchsorted(time[:, 0], everything.gps, side="right") - 1
    inside = (k >= 0) & (everything.gps - pad >= time[np.clip(k, 0, None), 0]) \
        & (everything.gps + pad <= time[np.clip(k, 0, None), 1])
    keep = np.flatnonzero(inside)
    if per_class is not None:
        rng = rng if rng is not None else np.random.default_rng(0)
        keep = np.concatenate([rng.permutation(keep[everything.labels[keep] == label])[:int(per_class)]
                               for label in np.unique(everything.labels[keep])] or [np.empty(0, dtype=np.int64)])
    scored = _subset(everything, np.isin(np.arange(len(everything)), keep))

    def strain_around(ifo: str, gps: float, half_samples: int) -> np.ndarray:
        rows = np.array([[[gps - pad, gps + pad]] * len(ifos)], dtype=np.float64)
        (row,) = list(obtainer.acquire(rate, rows, ifos))
        first = np.asarray(row.start_gps_time, dtype=np.float64).reshape(-1)
        out = []
        for j, data in enumerate(row.data):
            centre = int(round((gps - first[j]) * rate))
            out.append(np.asarray(data, dtype=np.float64)[centre - half_samples:centre + half_samples])
        return np.stack(out)

    scores = score_glitches(score, strain_around, geometry, scored, cluster_seconds, found_window_seconds)
    return GlitchStage(everything, scored, scores)


def glitch_data(glitches: Glitches, scores) -> Dict[str, np.ndarray]:
    """A validator's glitch record (`Validator.glitch_data`): each glitch and its statistic."""
    scores = np.asarray(scores, dtype=np.float64)
    if scores.size != len(glitches):
        raise ValueError(f"{scores.size} statistics for {len(glitches)} glitches.")
    return {"ifos": glitches.ifos, "gps": glitches.gps, "labels": glitches.labels, "scores": scores}


# ------------------------------------------------------- the background tail

def background_triggers(validator) -> Tuple[np.ndarray, np.ndarray]:
    """
    A validator's background triggers: their statistics and each one's GPS time in each
    detector (triggers, detectors), the centre of its window's onsource. Stream mode:
    each stream's clustered triggers (`stream.cluster`); window mode: every window. Real
    noise only: synthetic noise has no GPS time.
    """
    if validator.noise_gps_times is None:
        raise ValueError("The background has no GPS times (synthetic noise): no glitch lies in it.")
    scores = np.asarray(validator.far_scores, dtype=np.float64)
    starts = np.asarray(validator.noise_gps_times, dtype=np.float64)
    if starts.ndim == 1:
        starts = starts[:, None]
    centre = starts + float(validator.input_duration_seconds) / 2.0
    if not validator.config.stream:
        return scores, centre
    streams = validator.noise_streams
    stride = float(np.asarray(streams["stride_seconds"]))
    keep, offset = [], 0
    for length in np.asarray(streams["lengths"], dtype=np.int64):
        part = scores[offset:offset + length]
        keep.append(offset + cluster(np.arange(part.size) * stride, part, validator.config.stream_cluster_seconds))
        offset += int(length)
    keep = np.concatenate(keep) if keep else np.empty(0, dtype=np.int64)
    return scores[keep], centre[keep]


def trigger_glitch_record(validator, glitches: Glitches, ifos: Sequence[str]) -> Dict[str, np.ndarray]:
    """
    Each background trigger's nearest glitch (`nearest_glitch`), in any of ``ifos`` (the
    background's detectors, in order) at that detector's own time: the record
    `Validator.trigger_glitches` keeps.
    """
    scores, times = background_triggers(validator)
    near = nearest_glitch(times, list(ifos), glitches)
    return {"scores": scores, "gps": times, "distance_seconds": near.distance_seconds,
            "offset_seconds": near.offset_seconds, "label": near.label, "ifo": near.ifo,
            "glitch_gps": near.glitch_gps}


def tail_by_class(record: Dict[str, np.ndarray], threshold: float, near_seconds: float = 1.0) -> Dict[str, int]:
    """
    The classes of the background triggers at or above ``threshold``: each counted under
    its nearest glitch's class when that glitch lies within ``near_seconds``, else under
    ``"none"``.
    """
    scores = np.asarray(record["scores"], dtype=np.float64)
    with np.errstate(invalid="ignore"):
        tail = scores >= threshold
        near = np.asarray(record["distance_seconds"], dtype=np.float64) <= near_seconds
    names = np.where(near, np.asarray(record["label"]).astype(str), "none")[tail]
    classes, counts = np.unique(names, return_counts=True)
    return {str(c): int(n) for c, n in zip(classes, counts)}


# ------------------------------------------------- real against its twin

@dataclass(frozen=True)
class TwinComparison:
    """
    The volume on real data against its Gaussian twin at each level (``far_hz``): both
    volumes (Gpc^3), their ratio (real over twin) and its bounds at each of ``sigmas``
    (S, F). ``quotable`` where both volumes are.
    """
    far_hz: np.ndarray
    real_gpc3: np.ndarray
    twin_gpc3: np.ndarray
    ratio: np.ndarray
    ratio_lower: np.ndarray
    ratio_upper: np.ndarray
    sigmas: Tuple[float, ...]
    quotable: np.ndarray

    def to_dict(self) -> Dict:
        return {key: (value.tolist() if isinstance(value, np.ndarray) else list(value))
                for key, value in self.__dict__.items()}


def twin_comparison(real, twin) -> TwinComparison:
    """
    The real data's sensitive volume over its Gaussian twin's (`SensitiveVolumeCurve`s at
    the same levels). The ratio's bounds combine the two joint intervals' log-widths in
    quadrature, the lower from the real's lower and the twin's upper, the upper from the
    real's upper and the twin's lower: exact for independent log-normal errors, and wider
    than the truth when the two share injections, whose errors then partly cancel.
    """
    if not np.array_equal(np.asarray(real.far_hz), np.asarray(twin.far_hz)):
        raise ValueError("The real and twin volumes must be at the same false-alarm-rate levels.")
    if tuple(real.sigmas) != tuple(twin.sigmas):
        raise ValueError(f"The real and twin intervals are at different levels: {real.sigmas} and {twin.sigmas}.")
    vr, vt = (np.asarray(c.volume_within_reach_gpc3, dtype=np.float64) for c in (real, twin))
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = vr / vt
        log_width = lambda bound, v: np.abs(np.log(np.asarray(bound, dtype=np.float64) / v))  # noqa: E731
        lower = ratio * np.exp(-np.hypot(log_width(real.joint_lower_gpc3, vr), log_width(twin.joint_upper_gpc3, vt)))
        upper = ratio * np.exp(np.hypot(log_width(real.joint_upper_gpc3, vr), log_width(twin.joint_lower_gpc3, vt)))
    return TwinComparison(np.asarray(real.far_hz, dtype=np.float64), vr, vt, ratio, lower, upper,
                          tuple(float(s) for s in real.sigmas),
                          np.asarray(real.quotable, dtype=bool) & np.asarray(twin.quotable, dtype=bool))


__all__ = [
    "GlitchClassRates",
    "GlitchStage",
    "Glitches",
    "NearestGlitch",
    "TwinComparison",
    "background_triggers",
    "glitch_data",
    "glitch_false_positive_rates",
    "glitch_stage",
    "nearest_glitch",
    "score_glitches",
    "tail_by_class",
    "trigger_glitch_record",
    "twin_comparison",
    "validator_glitch_rates",
]
