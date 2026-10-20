"""
The metric family: every number the standard reports beside efficiency and volume, all
from the same injections and background, each with its interval at every level (1 and
2 sigma).

- **False-alarm-rate levels** (`FAR_LEVELS_HZ`): one per month, one per year, one per
  ten years and one per hundred years. A level is quoted only where the background has
  at least `metrics.MIN_BACKGROUND_EVENTS` independent events at or above its threshold;
  below that its floor is shown (`far_floor_hz`).
- **ROC** (`detection_roc`): detection probability against false-alarm rate, per SNR
  band, as efficiency is read at a target FAR (`efficiency`): the injections found at the
  background's threshold for the target, the chance of a noise-only pass subtracted,
  with the joint interval of the injections, the threshold and the chance. The
  population's own curve is the volume set's (`population_detection_probability`): the
  fraction of the population's volume within the reach that is found. The window-level
  ROC (`metrics.roc_curve`) is kept as a bridge to papers that report only that.
- **Recovered counts** (`recovered_counts`): injections found at each level, with the
  number noise alone would find, and a binomial interval on the count.
- **Timing accuracy** (`timing_accuracy`): each found injection's trigger time against
  its true merger, at each level: the median and 90th percentile of ``|dt|`` and the
  median of ``dt`` (its bias), each with a distribution-free interval from the order
  statistics. In stream mode the found test admits only triggers within the found
  window, so ``|dt|`` is at most ``found_window_seconds`` by construction.
- **False-alarm calibration** (`far_calibration`): the background split by block into
  halves that share no data (`held_out_split`); the threshold a target's FAR gives on
  one is checked on the other: the rate there, with its Poisson interval, against the
  rate claimed.
- **Real events** (`real_event_summary`): each known event's statistic and its FAR from
  the background, the events found at each level, and a flag for events inside a
  model's training period (for an as-published model, whose training data we did not
  choose).

Sensitive VT, the binned volume and the ratio to the matched filter are the volume's
(`volume.sensitive_vt`, `volume.binned_volume`, `volume.volume_ratio`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats

from .efficiency import _GroupedScores, _efficiency_range, _joint_bands, _signal_fraction
from .metrics import (
    DEFAULT_SIGMAS,
    MIN_BACKGROUND_EVENTS,
    Background,
    _injections,
    binomial_interval,
    chance_subtracted_interval,
    resample_groups,
    sigma_level,
    subtract_chance,
)
from .volume import SECONDS_PER_MONTH, SECONDS_PER_YEAR, SensitiveVolumeCurve

#: The standard's false-alarm-rate levels (Hz), loosest first, and their names.
FAR_LEVELS_HZ = (1.0 / SECONDS_PER_MONTH, 1.0 / SECONDS_PER_YEAR, 0.1 / SECONDS_PER_YEAR, 0.01 / SECONDS_PER_YEAR)
FAR_LEVEL_NAMES = ("1 per month", "1 per year", "1 per 10 years", "1 per 100 years")
#: The ROC's SNR bands, ``[lo, hi)``: clipped to the injected range.
ROC_SNR_BANDS = ((0.0, 5.0), (5.0, 10.0), (10.0, 15.0), (15.0, 20.0))


def quotable_at(background: Background, thresholds) -> np.ndarray:
    """
    Whether the background quotes a rate at each threshold: at least
    `metrics.MIN_BACKGROUND_EVENTS` independent events at or above it, as its FAR curve
    counts them (`metrics.FarCurve.quotable`). False at NaN.
    """
    t = np.atleast_1d(np.asarray(thresholds, dtype=np.float64))
    curve = background.far_curve()
    # The count at t is the curve's at its last breakpoint >= t (breakpoints descend).
    at = np.searchsorted(-curve.thresholds, -np.where(np.isnan(t), np.inf, t), side="right") - 1
    quotable = np.concatenate([[False], curve.quotable])
    return np.where(np.isnan(t), False, quotable[at + 1])


def far_floor_hz(background: Background) -> Tuple[float, float]:
    """
    How deep a background reaches: the least FAR it resolves (one trigger in its
    livetime) and the least it quotes (`metrics.MIN_BACKGROUND_EVENTS` independent events
    at or above the threshold, as its FAR curve counts them; NaN where none is quotable).
    """
    curve = background.far_curve()
    quoted = curve.rate_hz[curve.quotable]
    return 1.0 / background.livetime_seconds, float(quoted.min()) if quoted.size else float("nan")


def _targets(background: Background, fars_hz):
    fars = np.asarray(fars_hz, dtype=np.float64).reshape(-1)
    targets = [background.threshold_for_far(float(far)) for far in fars]
    return fars, targets, np.array([t.threshold for t in targets]), np.array([t.resolved for t in targets])


# ----------------------------------------------------------------------- ROC

@dataclass(frozen=True)
class DetectionRoc:
    """
    Detection probability against false-alarm rate for injection bands (``bands``, names
    ``labels``): at each target FAR (Hz), the background's threshold and the FAR it
    achieves, the injections of each band found there (``detected`` of ``total``), the
    probability of detection (the chance subtracted) and its joint interval at each of
    ``sigmas`` (``(levels, targets, bands)``). ``quotable`` per target: the background
    has enough independent events there.
    """
    far_hz: np.ndarray
    threshold: np.ndarray
    achieved_far_hz: np.ndarray
    quotable: np.ndarray
    labels: Tuple[str, ...]
    bands: Tuple[Tuple[float, float], ...]
    detected: np.ndarray
    total: np.ndarray
    probability: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    sigmas: Tuple[float, ...]
    num_bootstrap: int


def _band_label(lo: float, hi: float) -> str:
    return f"SNR {lo:g}-{hi:g}"


def detection_roc(
    snrs,
    scores,
    background: Background,
    fars_hz: Sequence[float],
    rng: np.random.Generator,
    *,
    bands: Sequence[Tuple[float, float]] = ROC_SNR_BANDS,
    num_bootstrap: int = 500,
    sigmas: Sequence[float] = DEFAULT_SIGMAS,
    method: str = "clopper-pearson",
) -> DetectionRoc:
    """
    Detection probability against FAR in each SNR band ``[lo, hi)`` (the last band's top
    closed) and over all the injections (``"all SNRs"``), with the joint interval
    efficiency carries (`efficiency`: the threshold redrawn from the background, the found
    counts as binomial, the chance the estimator's; at least as wide as the binomial
    interval and the range over the FAR's band of thresholds). A band with no injection
    is dropped.
    """
    if num_bootstrap < 1:
        raise ValueError(f"num_bootstrap must be >= 1, got {num_bootstrap}.")
    snrs, scores = _injections(snrs, scores)
    sigmas = tuple(float(s) for s in sigmas)
    fars, targets, thresholds, resolved = _targets(background, fars_hz)
    chance = background.found_by_chance(thresholds)
    members, labels, kept = [], [], []
    top = max(hi for _, hi in bands) if len(bands) else np.inf
    for lo, hi in bands:
        inside = (snrs >= lo) & ((snrs < hi) | ((hi == top) & (snrs == hi)))
        if inside.any():
            members.append(inside)
            labels.append(_band_label(lo, hi))
            kept.append((float(lo), float(hi)))
    members.append(np.ones(snrs.size, dtype=bool))
    labels.append("all SNRs")
    kept.append((float(snrs.min()), float(snrs.max())))

    shape = (len(sigmas), fars.size, len(members))
    detected = np.zeros((fars.size, len(members)), dtype=np.int64)
    probability = np.full((fars.size, len(members)), np.nan)
    lower, upper = np.full(shape, np.nan), np.full(shape, np.nan)
    totals = np.array([m.sum() for m in members], dtype=np.int64)
    replicate_thresholds = [background.bootstrap_thresholds(fars, rng) for _ in range(num_bootstrap)]
    bands_at = {i: [background.far_band_thresholds(float(fars[i]), sigma_level(s)) for s in sigmas]
                for i in np.flatnonzero(resolved)}
    for j, inside in enumerate(members):
        grouped = _GroupedScores(np.where(inside, 0, -1), scores, 1)
        found = grouped.found(np.where(resolved, thresholds, np.inf))[:, 0]
        detected[:, j] = np.where(resolved, found, 0)
        with np.errstate(invalid="ignore", divide="ignore"):
            point = np.where(resolved, subtract_chance(found / totals[j], chance.mean), np.nan)
        probability[:, j] = point
        binomial_lower, binomial_upper = chance_subtracted_interval(found, totals[j], chance, sigmas, method)
        binomial_lower[:, ~resolved], binomial_upper[:, ~resolved] = np.nan, np.nan
        replicates = np.full((num_bootstrap, fars.size), np.nan)
        for b, replicate in enumerate(replicate_thresholds):
            counts, drawn = resample_groups(rng, grouped.found(np.where(np.isnan(replicate), np.inf, replicate)),
                                            grouped.total)
            replicates[b] = np.where(np.isnan(replicate), np.nan,
                                     _signal_fraction(counts, drawn, chance.mean)[:, 0])
        at_bands = np.full((len(sigmas), 2, fars.size), np.nan)
        for i, band in bands_at.items():
            for level, (high, low) in enumerate(band):
                at_bands[level, :, i] = _efficiency_range(grouped, lambda f: (f, grouped.total), scores,
                                                          background, high, low)[:, 0]
        band_lower, band_upper = _joint_bands(binomial_lower, binomial_upper, replicates, at_bands, sigmas)
        lower[:, :, j], upper[:, :, j] = band_lower, band_upper
    return DetectionRoc(
        far_hz=fars, threshold=thresholds, achieved_far_hz=np.array([t.achieved.rate_hz for t in targets]),
        quotable=np.array([t.quotable for t in targets]), labels=tuple(labels), bands=tuple(kept),
        detected=detected, total=totals, probability=probability, lower=lower, upper=upper, sigmas=sigmas,
        num_bootstrap=int(num_bootstrap))


@dataclass(frozen=True)
class PopulationDetection:
    """
    The population's detection probability against FAR: the fraction of the population's
    volume within the reach that is found (`volume.SensitiveVolumeCurve`: V over
    ``population_volume_gpc3``), its joint interval at each of ``sigmas`` the volume's,
    likewise scaled (``(levels, targets)``).
    """
    far_hz: np.ndarray
    probability: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    quotable: np.ndarray
    sigmas: Tuple[float, ...]


def population_detection_probability(curve: SensitiveVolumeCurve) -> PopulationDetection:
    """The population's detection probability at each of a volume curve's targets (`PopulationDetection`)."""
    scale = curve.population_volume_gpc3
    return PopulationDetection(curve.far_hz, curve.volume_within_reach_gpc3 / scale, curve.joint_lower_gpc3 / scale,
                               curve.joint_upper_gpc3 / scale, curve.quotable, curve.sigmas)


# ---------------------------------------------------------- recovered counts

@dataclass(frozen=True)
class RecoveredCounts:
    """
    Injections found at each target FAR (Hz): ``found`` of ``total`` at the background's
    threshold, the number noise alone would find there (``by_chance``: the chance times
    the total), and a binomial interval on the found count at each of ``sigmas``
    (``(levels, targets)``). Unresolved targets have NaN thresholds, counts of 0 and NaN
    intervals; ``quotable`` as the background's.
    """
    far_hz: np.ndarray
    threshold: np.ndarray
    quotable: np.ndarray
    found: np.ndarray
    total: int
    by_chance: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    sigmas: Tuple[float, ...]


def recovered_counts(scores, background: Background, fars_hz: Sequence[float], *,
                     sigmas: Sequence[float] = DEFAULT_SIGMAS, method: str = "clopper-pearson") -> RecoveredCounts:
    """The injections found at each of ``fars_hz`` (`RecoveredCounts`)."""
    scores = np.sort(_injections(np.zeros(np.size(scores)), scores)[1])
    sigmas = tuple(float(s) for s in sigmas)
    fars, targets, thresholds, resolved = _targets(background, fars_hz)
    n = scores.size
    found = np.where(resolved, n - np.searchsorted(scores, np.where(resolved, thresholds, np.inf), side="left"), 0)
    chance = np.where(resolved, background.found_by_chance(thresholds).mean, np.nan)
    lower, upper = np.full((len(sigmas), fars.size), np.nan), np.full((len(sigmas), fars.size), np.nan)
    for level, sigma in enumerate(sigmas):
        low, high = binomial_interval(found, n, sigma_level(sigma), method)
        lower[level], upper[level] = np.where(resolved, low * n, np.nan), np.where(resolved, high * n, np.nan)
    return RecoveredCounts(fars, thresholds, np.array([t.quotable for t in targets]), found.astype(np.int64), int(n),
                           chance * n, lower, upper, sigmas)


# ----------------------------------------------------------- timing accuracy

def quantile_interval(sorted_values: np.ndarray, q: float, confidence: float) -> Tuple[float, float]:
    """
    The distribution-free central interval on the ``q``-quantile of the law ``sorted_values``
    (ascending) were drawn from: the order statistics ``x_(l)`` and ``x_(u)`` (1-based) with
    ``l`` and ``u - 1`` the binomial ``(n, q)``'s ``alpha / 2`` and ``1 - alpha / 2``
    quantiles, clipped to the sample; NaN for no values.
    """
    n = sorted_values.size
    if n == 0:
        return np.nan, np.nan
    alpha = 1.0 - confidence
    low = int(stats.binom.ppf(alpha / 2, n, q))
    high = int(stats.binom.ppf(1 - alpha / 2, n, q)) + 1
    return float(sorted_values[min(max(low, 1), n) - 1]), float(sorted_values[min(max(high, 1), n) - 1])


@dataclass(frozen=True)
class TimingAccuracy:
    """
    Trigger time against true merger, ``dt`` (s), of the injections found at each target
    FAR (Hz): ``num_found`` of them; the median of ``|dt|``, its 90th percentile and the
    median of ``dt`` (the bias), each with its distribution-free interval at each of
    ``sigmas`` (`quantile_interval`; ``(levels, targets)``). NaN where none is found.
    """
    far_hz: np.ndarray
    threshold: np.ndarray
    num_found: np.ndarray
    median_abs_seconds: np.ndarray
    median_abs_lower: np.ndarray
    median_abs_upper: np.ndarray
    p90_abs_seconds: np.ndarray
    p90_abs_lower: np.ndarray
    p90_abs_upper: np.ndarray
    median_seconds: np.ndarray
    median_lower: np.ndarray
    median_upper: np.ndarray
    sigmas: Tuple[float, ...]


def timing_accuracy(scores, trigger_offsets, background: Background, fars_hz: Sequence[float], *,
                    sigmas: Sequence[float] = DEFAULT_SIGMAS) -> TimingAccuracy:
    """
    Timing accuracy (`TimingAccuracy`) of the injections found at each of ``fars_hz``:
    ``trigger_offsets`` is each injection's trigger time less its merger's (stream mode's
    record), NaN where nothing found it.
    """
    scores = _injections(np.zeros(np.size(scores)), scores)[1]
    offsets = np.asarray(trigger_offsets, dtype=np.float64).reshape(-1)
    if offsets.shape != scores.shape:
        raise ValueError(f"{offsets.size} trigger offsets for {scores.size} injections.")
    sigmas = tuple(float(s) for s in sigmas)
    fars, _, thresholds, resolved = _targets(background, fars_hz)
    shape = (len(sigmas), fars.size)
    out = {name: np.full(fars.size, np.nan) for name in ("median_abs", "p90_abs", "median")}
    bounds = {f"{name}_{end}": np.full(shape, np.nan) for name in out for end in ("lower", "upper")}
    num_found = np.zeros(fars.size, dtype=np.int64)
    for i in np.flatnonzero(resolved):
        dt = offsets[(scores >= thresholds[i]) & np.isfinite(offsets)]
        num_found[i] = dt.size
        if dt.size == 0:
            continue
        for name, values, q in (("median_abs", np.sort(np.abs(dt)), 0.5), ("p90_abs", np.sort(np.abs(dt)), 0.9),
                                ("median", np.sort(dt), 0.5)):
            out[name][i] = np.quantile(values, q)
            for level, sigma in enumerate(sigmas):
                bounds[f"{name}_lower"][level, i], bounds[f"{name}_upper"][level, i] = quantile_interval(
                    values, q, sigma_level(sigma))
    return TimingAccuracy(fars, thresholds, num_found, out["median_abs"], bounds["median_abs_lower"],
                          bounds["median_abs_upper"], out["p90_abs"], bounds["p90_abs_lower"],
                          bounds["p90_abs_upper"], out["median"], bounds["median_lower"], bounds["median_upper"],
                          sigmas)


# ------------------------------------------------------ false-alarm calibration

def held_out_split(background: Background) -> Optional[Tuple[Background, Background]]:
    """
    The background in two halves that share no data: its streams by block, alternate
    blocks to each (every slide of a block's data stays with it), as the same kind of
    background. None where it has fewer than two blocks (or no streams: independent
    windows hold no order to split by).
    """
    blocks = background._blocks() if background.streams else {}
    if len(blocks) < 2:
        return None
    halves = ([], [])
    for k, block in enumerate(sorted(blocks, key=lambda b: (str(type(b)), b))):
        halves[k % 2].extend(blocks[block])
    if hasattr(background, "found_window_seconds"):
        make = lambda streams: type(background)(streams, background.cluster_seconds, background.found_window_seconds)
    else:
        make = lambda streams: Background(streams, background.cluster_seconds)
    return make(halves[0]), make(halves[1])


@dataclass(frozen=True)
class FarCalibration:
    """
    A FAR claimed on one background checked on another that shares none of its data
    (`held_out_split`): at each target (Hz), the threshold the calibration background gives
    it and the FAR it claims there (``claimed_hz``); the held-out background's count at or
    above that threshold, its rate (``measured_hz``) with its interval at each of ``sigmas``
    (``(levels, targets)``), and their ratio. ``consistent`` (levels, targets): the claimed
    rate lies within the held-out rate's interval. ``quotable``: both backgrounds have
    enough independent events there.
    """
    far_hz: np.ndarray
    threshold: np.ndarray
    claimed_hz: np.ndarray
    held_out_count: np.ndarray
    held_out_livetime_seconds: float
    measured_hz: np.ndarray
    lower_hz: np.ndarray
    upper_hz: np.ndarray
    ratio: np.ndarray
    consistent: np.ndarray
    quotable: np.ndarray
    sigmas: Tuple[float, ...]


def far_calibration(calibration: Background, held_out: Background, fars_hz: Sequence[float], *,
                    sigmas: Sequence[float] = DEFAULT_SIGMAS) -> FarCalibration:
    """The FAR claimed on ``calibration`` checked on ``held_out`` at each of ``fars_hz`` (`FarCalibration`)."""
    sigmas = tuple(float(s) for s in sigmas)
    fars, targets, thresholds, resolved = _targets(calibration, fars_hz)
    claimed = np.array([t.achieved.rate_hz for t in targets])
    shape = (len(sigmas), fars.size)
    lower, upper = np.full(shape, np.nan), np.full(shape, np.nan)
    count = np.zeros(fars.size, dtype=np.int64)
    for i in np.flatnonzero(resolved):
        count[i] = int(held_out.far_count(thresholds[i]))
        for level, sigma in enumerate(sigmas):
            estimate = held_out.far(thresholds[i], sigma_level(sigma))
            lower[level, i], upper[level, i] = estimate.lower_hz, estimate.upper_hz
    T = held_out.livetime_seconds
    measured = np.where(resolved, count / T, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(resolved & (claimed > 0), measured / claimed, np.nan)
    enough = quotable_at(held_out, thresholds)
    return FarCalibration(fars, thresholds, claimed, count, float(T), measured, lower, upper, ratio,
                          resolved[None, :] & (lower <= claimed) & (claimed <= upper),
                          np.array([t.quotable for t in targets]) & enough, sigmas)


# ---------------------------------------------------------------- real events

@dataclass(frozen=True)
class RealEventSummary:
    """
    The known events in the test data: one row per scored event (``rows``: its name, GPS
    time, statistic, FAR from the background with its interval at each of ``sigmas``,
    whether the background quotes its FAR (``quotable``), and ``in_training``: whether it lies in
    a training period given), and at each target FAR (Hz) the events found
    (``found``: statistic at or above the background's threshold) and those found outside
    every training period (``found_outside_training``). Unresolved targets count 0.
    """
    rows: Tuple[Dict[str, Any], ...]
    far_hz: np.ndarray
    threshold: np.ndarray
    resolved: np.ndarray
    found: np.ndarray
    found_outside_training: np.ndarray
    sigmas: Tuple[float, ...]


def in_periods(gps: float, periods: Optional[Sequence[Tuple[float, float]]]) -> bool:
    """Whether GPS time ``gps`` lies in any of ``periods`` (``(start, end)``, both ends in)."""
    return any(float(start) <= float(gps) <= float(end) for start, end in (periods or ()))


def real_event_summary(events: Sequence[Dict[str, Any]], background: Background, fars_hz: Sequence[float], *,
                       training_periods: Optional[Sequence[Tuple[float, float]]] = None,
                       sigmas: Sequence[float] = DEFAULT_SIGMAS) -> RealEventSummary:
    """
    The real events' summary (`RealEventSummary`): each scored event's FAR read from
    ``background`` at its statistic, and the events found at each of ``fars_hz``.
    ``training_periods`` (GPS ``(start, end)`` pairs) flags the events inside them.
    """
    sigmas = tuple(float(s) for s in sigmas)
    rows: List[Dict[str, Any]] = []
    for event in events:
        score = event.get("score")
        if score is None or np.isnan(float(score)):  # unscored; -inf: scored, with no trigger near it
            continue
        estimates = [background.far(float(score), sigma_level(s)) for s in sigmas]
        rows.append(dict(name=event.get("name", "?"), gps=float(event.get("gps", np.nan)), score=float(score),
                         far_hz=estimates[0].rate_hz,
                         far_interval_hz=np.array([[e.lower_hz, e.upper_hz] for e in estimates]),
                         quotable=bool(quotable_at(background, float(score))[0]),
                         in_training=in_periods(event.get("gps", np.nan), training_periods)))
    fars, _, thresholds, resolved = _targets(background, fars_hz)
    scores = np.array([r["score"] for r in rows], dtype=np.float64)
    inside = np.array([r["in_training"] for r in rows], dtype=bool)
    found = np.zeros(fars.size, dtype=np.int64)
    outside = np.zeros(fars.size, dtype=np.int64)
    for i in np.flatnonzero(resolved):
        passed = scores >= thresholds[i]
        found[i], outside[i] = int(passed.sum()), int((passed & ~inside).sum())
    return RealEventSummary(tuple(rows), fars, thresholds, resolved, found, outside, sigmas)


__all__ = [
    "FAR_LEVELS_HZ",
    "FAR_LEVEL_NAMES",
    "ROC_SNR_BANDS",
    "DetectionRoc",
    "FarCalibration",
    "PopulationDetection",
    "RealEventSummary",
    "RecoveredCounts",
    "TimingAccuracy",
    "detection_roc",
    "far_calibration",
    "far_floor_hz",
    "held_out_split",
    "in_periods",
    "quotable_at",
    "population_detection_probability",
    "quantile_interval",
    "real_event_summary",
    "recovered_counts",
    "timing_accuracy",
]
