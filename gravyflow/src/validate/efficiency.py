"""
Efficiency against SNR at target false-alarm rates, with every error that moves it.

At a target FAR the threshold is the one the noise background gives it
(`Background.threshold_for_far`), and the efficiency is read there, as the fixed-threshold
curves of `metrics` read it, the chance of a noise-only pass subtracted
(`metrics.subtract_chance`):

- the **binned** curve (`metrics.efficiency_curve`), the record's estimator;
- the owner's **rolling** curve (`metrics.rolling_efficiency_curve`: a window of +-0.5 in
  SNR at 50 points), the one the dashboard plots;
- the **sigmoid fit** (`metrics.fit_efficiency_sigmoid`) and its 50% and 90% SNR points.

Their intervals, at each of the levels (1 and 2 sigma), carry three errors, as the
sensitive volume's joint interval does (`volume`): the injections' (finite Monte
Carlo), the background's (the threshold read from a finite livetime: where the
background allows few triggers it is the larger), and the chance's. A bootstrap
redraws all three together: the threshold from the background
(`Background.bootstrap_thresholds`, counting a time-slid background's independent
information), the chance from its law at that threshold (`Background.found_by_chance`),
and each SNR group's found count as binomial (the injections' SNRs held as designed).
Each band's ends are the central range of its replicates, at least as far out as the
binomial interval at the threshold and as the efficiency at the two thresholds the
FAR's interval allows (`Background.far_band_thresholds`). The bands are pointwise.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import List, Sequence, Tuple

import numpy as np

from .metrics import (
    DEFAULT_SIGMAS,
    ROLLING_HALF_WIDTH,
    Background,
    ChanceLaw,
    EfficiencyCurve,
    FarThreshold,
    RollingEfficiency,
    SigmoidFit,
    _injections,
    bin_index,
    efficiency_curve,
    fit_bins,
    fit_sigmoid,
    nested,
    resample_groups,
    rolling_efficiency_curve,
    rolling_windows,
    sigma_level,
    sigmoid_fit_from_replicates,
    subtract_chance,
)


@dataclass(frozen=True)
class EfficiencyAgainstSnr:
    """
    Efficiency against SNR at one target FAR (Hz): the background's threshold for it,
    the binned record, the rolling curve and the sigmoid fit, each with its joint
    intervals (module docstring) from ``num_bootstrap`` replicates.
    """
    far_hz: float
    threshold: FarThreshold
    binned: EfficiencyCurve
    rolling: RollingEfficiency
    fit: SigmoidFit
    num_bootstrap: int


class _GroupedScores:
    """
    Injection scores in groups (``index``, -1 for none), counted above many
    thresholds at once: within each group sorted, keyed by group and rank, so that a
    threshold's count in every group is one search.
    """

    def __init__(self, index: np.ndarray, scores: np.ndarray, num_groups: int):
        keep = index >= 0
        self.num_groups = num_groups
        self.total = np.bincount(index[keep], minlength=num_groups)
        self.ends = np.cumsum(self.total)
        self._sorted = np.sort(scores)
        self._stride = scores.size + 1
        # A score's rank: the number of scores strictly below it; below a threshold exactly when the score is.
        rank = np.searchsorted(self._sorted, scores[keep], side="left")
        self._keys = np.sort(index[keep].astype(np.int64) * self._stride + rank)

    def found(self, thresholds: np.ndarray) -> np.ndarray:
        """The number of each group's scores ``>= threshold``: (thresholds, groups)."""
        below = np.searchsorted(self._sorted, thresholds, side="left")
        queries = np.arange(self.num_groups, dtype=np.int64)[None, :] * self._stride + below[:, None]
        return self.ends[None, :] - np.searchsorted(self._keys, queries.ravel(), side="left").reshape(queries.shape)


def _signal_fraction(found: np.ndarray, total: np.ndarray, chance: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return subtract_chance(np.where(total > 0, found / total, np.nan), chance[:, None])


def _joint_bands(point_lower, point_upper, replicates, at_bands, sigmas) -> Tuple[np.ndarray, np.ndarray]:
    """
    Each level's band: the replicates' central range (``replicates`` (B, P)), at least as
    far out as the binomial ends (``point_lower``/``point_upper``, (levels, P)) and the
    efficiencies at the FAR band's thresholds (``at_bands`` (levels, 2, P)); within [0, 1].
    """
    lowers, uppers = [], []
    for level, sigma in enumerate(sigmas):
        alpha = 1.0 - sigma_level(sigma)
        with np.errstate(invalid="ignore"), np.testing.suppress_warnings() as sup:
            sup.filter(RuntimeWarning)
            low, high = np.nanquantile(replicates, [alpha / 2, 1 - alpha / 2], axis=0)
        lowers.append(np.clip(np.fmin.reduce([low, point_lower[level], *at_bands[level]]), 0.0, 1.0))
        uppers.append(np.clip(np.fmax.reduce([high, point_upper[level], *at_bands[level]]), 0.0, 1.0))
    lower, upper = nested(np.array(lowers), np.array(uppers))
    unmeasured = np.isnan(point_lower)
    return np.where(unmeasured, np.nan, lower), np.where(unmeasured, np.nan, upper)


def efficiency_against_snr(
    snrs,
    scores,
    background: Background,
    fars_hz: Sequence[float],
    rng: np.random.Generator,
    bin_edges: Sequence[float],
    snr_range: Tuple[float, float],
    num_bootstrap: int = 1000,
    sigmas: Sequence[float] = DEFAULT_SIGMAS,
    method: str = "clopper-pearson",
) -> List[EfficiencyAgainstSnr]:
    """
    Efficiency against SNR at each of ``fars_hz`` (`EfficiencyAgainstSnr`): the binned
    curve in ``bin_edges``, the rolling curve over ``snr_range``, and the sigmoid fit,
    with joint intervals at each of ``sigmas`` (module docstring). A target the
    background cannot resolve has NaN curves.
    """
    if num_bootstrap < 1:
        raise ValueError(f"num_bootstrap must be >= 1, got {num_bootstrap}.")
    snrs, scores = _injections(snrs, scores)
    sigmas = tuple(float(s) for s in sigmas)
    fars = np.asarray(fars_hz, dtype=np.float64).reshape(-1)
    targets = [background.threshold_for_far(float(far)) for far in fars]
    thresholds = np.array([t.threshold for t in targets])
    resolved = np.array([t.resolved for t in targets])
    chance = background.found_by_chance(thresholds)
    laws = [ChanceLaw(chance.a[i], chance.b[i]) for i in range(fars.size)]

    # The point curves, at each target's threshold.
    edges = np.asarray(bin_edges, dtype=np.float64)
    binned = [efficiency_curve(snrs, scores, t, edges, law, sigmas, method) for t, law in zip(thresholds, laws)]
    rolling = [rolling_efficiency_curve(snrs, scores, t, snr_range, law, sigmas, method)
               for t, law in zip(thresholds, laws)]

    # The groups each curve counts: the bins; the rolling windows' elementary pieces; the fit's bins.
    centres = rolling_windows(snr_range, snrs)
    window_edges = np.unique(np.concatenate([centres - ROLLING_HALF_WIDTH, centres + ROLLING_HALF_WIDTH]))
    pieces = np.searchsorted(window_edges, snrs, side="right") - 1
    pieces = np.where((pieces >= 0) & (pieces < window_edges.size - 1), pieces, -1)
    first = np.searchsorted(window_edges, centres - ROLLING_HALF_WIDTH)
    last = np.searchsorted(window_edges, centres + ROLLING_HALF_WIDTH)
    fit_edges, fit_index = fit_bins(snrs, snr_range)
    groups = {
        "binned": _GroupedScores(bin_index(snrs, edges), scores, edges.size - 1),
        "rolling": _GroupedScores(pieces, scores, window_edges.size - 1),
        "fit": _GroupedScores(fit_index, scores, fit_edges.size - 1),
    }
    fit_total = groups["fit"].total
    with np.errstate(invalid="ignore", divide="ignore"):
        fit_snr = np.bincount(fit_index[fit_index >= 0], weights=snrs[fit_index >= 0],
                              minlength=fit_edges.size - 1) / fit_total

    def window_counts(piece_counts: np.ndarray) -> np.ndarray:
        cumulative = np.concatenate([np.zeros((piece_counts.shape[0], 1)), np.cumsum(piece_counts, axis=1)], axis=1)
        return cumulative[:, last] - cumulative[:, first]

    point_fit = fit_sigmoid(fit_snr, groups["fit"].found(thresholds), np.broadcast_to(fit_total, (fars.size, fit_total.size)),
                            chance.mean, snr_range)

    # The joint bootstrap: threshold, chance and found counts redrawn together.
    binned_replicates = np.full((num_bootstrap, fars.size, edges.size - 1), np.nan)
    rolling_replicates = np.full((num_bootstrap, fars.size, centres.size), np.nan)
    fit_replicates = np.full((2, num_bootstrap, fars.size), np.nan)
    for b in range(num_bootstrap):
        replicate = background.bootstrap_thresholds(fars, rng)
        chances = background.found_by_chance(replicate).sample(rng, 1)[0]
        binned_replicates[b] = _signal_fraction(*resample_groups(rng, groups["binned"].found(replicate),
                                                                 groups["binned"].total), chances)
        found, total = resample_groups(rng, groups["rolling"].found(replicate), groups["rolling"].total)
        rolling_replicates[b] = _signal_fraction(window_counts(found), window_counts(total), chances)
        found, total = resample_groups(rng, groups["fit"].found(replicate), fit_total)
        fit_replicates[:, b] = fit_sigmoid(fit_snr, found, total, chances, snr_range,
                                           start=(point_fit[0], point_fit[2]))[:2]

    results = []
    for i, (far, target) in enumerate(zip(fars, targets)):
        if not resolved[i]:
            fit = sigmoid_fit_from_replicates(np.nan, np.nan, tuple(np.full(num_bootstrap + 1, np.nan)
                                                                    for _ in range(3)), sigmas)
            results.append(EfficiencyAgainstSnr(float(far), target, binned[i], rolling[i], fit, int(num_bootstrap)))
            continue
        bands = [background.far_band_thresholds(float(far), sigma_level(s)) for s in sigmas]

        def at_bands(curve_at):
            return np.array([[curve_at(t) for t in band] for band in bands])

        band_laws = lambda t: background.found_by_chance(t)
        binned_at = at_bands(lambda t: efficiency_curve(snrs, scores, t, edges, band_laws(t), sigmas, method).efficiency)
        rolling_at = at_bands(lambda t: rolling_efficiency_curve(snrs, scores, t, snr_range, band_laws(t), sigmas,
                                                                 method).efficiency)
        lower, upper = _joint_bands(binned[i].lower, binned[i].upper, binned_replicates[:, i], binned_at, sigmas)
        binned_i = replace(binned[i], lower=lower, upper=upper)
        lower, upper = _joint_bands(rolling[i].lower, rolling[i].upper, rolling_replicates[:, i], rolling_at, sigmas)
        rolling_i = replace(rolling[i], lower=lower, upper=upper)
        points = tuple(np.concatenate([[p[i]], r]) for p, r in
                       zip(point_fit, (fit_replicates[0, :, i], fit_replicates[1, :, i], np.full(num_bootstrap, np.nan))))
        fit = sigmoid_fit_from_replicates(thresholds[i], chance.mean[i], points, sigmas)
        results.append(EfficiencyAgainstSnr(float(far), target, binned_i, rolling_i, fit, int(num_bootstrap)))
    return results
