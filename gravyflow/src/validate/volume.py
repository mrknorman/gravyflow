"""
Sensitive volume and sensitive distance against false-alarm rate, within a reach.

**The estimand is the volume within the reach R**, exactly what MLGWSC-1's
``evaluate.py`` computes: the injections are drawn as an injection campaign
draws them, each source from the population, placed at a physical distance
drawn uniformly in volume out to the reach ``reach_mpc`` (the validator's is
the distance beyond which no source of the population can have a stated SNR:
`ValidationBank.reach_distance`), and a found injection (``score >= threshold``,
as everywhere in `metrics`) stands for the reference volume
``V_ref = 4/3 pi R^3`` times its population weight ``w`` (1 when the draw is the
population's own; a power of the chirp mass for a draw uniform in chirp
distance), so

    V(R, threshold) = V_ref sum_i w_i [score_i >= threshold] / N.

Every result carries its R: a volume within one reach is not comparable with
one within another. The volume is Euclidean (no cosmology) and the
population's masses are in the detector frame, as in ``evaluate.py``.
Unweighted, V is ``V_ref k / N`` for ``k`` found of ``N``, and its injection
interval is the exact binomial one. The sensitive distance is the radius of a
sphere of that volume, ``(3 V / 4 pi)^(1/3)``. At a target FAR, the threshold is
the one the noise background gives that FAR (`Background.threshold_for_far`).
Three intervals are reported, each two-sided and central at ``confidence``:

- **injections**: the finite Monte Carlo, a binomial interval on V at the
  threshold (`InjectionVolumes.interval`);
- **background**: the finite livetime, the V at the two thresholds between
  which the FAR's Poisson interval puts the true threshold
  (`Background.far_band_thresholds`);
- **joint**: a bootstrap over injections and the background's thresholds
  (`Background.bootstrap_thresholds`, which counts a time-slid background's
  independent information, not its slid windows), each end at least as far
  out as the other two intervals' (`sensitive_volume_curve`).

**The noise floor.** An injection is found when its window passes the
threshold, whatever passes it: one beyond reach is found by its noise alone,
with the chance ``p`` that a noise window passes (`Background.window_exceedance`).
Those add about ``p V_ref`` to V, which is part of what the threshold finds,
not an error; it is recorded at each target, and a point where it exceeds a
stated fraction of V is marked noise-dominated, and not quoted.

**The volume beyond the reach, estimated.** A detector that finds sources
beyond R is measured within R all the same: that is the estimand. Beside V(R),
each target reports an estimate, with its own interval, of the volume found by
signal (above the noise floor) between R and `BEYOND_REACH_FACTOR` times R:
what V would gain if the reach were that much longer. It is bounded because the
whole volume beyond R is not finite for a detector that finds any source above
noise: a score that grows with the SNR passes a threshold at the noise's rate
plus a term of order SNR (or SNR^2), which falls slower than the volume grows.
The estimate rests on one model, stated: **a source's chance of being found
depends on it only through its SNR** (and the noise), so a source moved from
``d`` to ``k d`` is found as often as the population's sources of a ``k``
times smaller SNR. Since SNR goes as one over distance, the injections at
``d >= R / k``, moved ``k`` times as far, are a draw uniform in the volume
between R and ``k R``, each at its SNR over ``k``; the found chance there is
read from the injections themselves, binned in SNR (equal-count bins of the
injections whose SNR is in the moved ones' range), less the noise's chance
``p``:

    V_beyond = V_ref k^3 / N sum_{d_i >= R / k} w_i (found(snr_i / k) - p).

Its interval is the joint bootstrap's: the injections redrawn (both as the
moved sources and as the found chance's estimate) and the thresholds redrawn,
with ``p`` the noise's chance at the estimated threshold in every replicate (the
estimator's own, so a threshold's error moves the found chance against it, as
it moves the truth); each end at least as far out as at the background
interval's thresholds and as the injections' binomial interval on its found sum
(`InjectionVolumes.beyond_reach_interval`). It is a warning, never a refusal: a target where it
exceeds a stated fraction of V(R) is marked reach-limited, and `quote` reports it
and warns. Where the found chance depends on more than the SNR (a detector that
finds one part of the population's masses better), the estimate is biased by
the difference between the moved sources' parameters and those of the
injections of the same SNR.

Units: distances in Mpc, volumes in Gpc^3 (Euclidean: no cosmology), FAR in Hz;
`SECONDS_PER_YEAR` and `SECONDS_PER_MONTH` convert it for display.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from .metrics import DEFAULT_CONFIDENCE, Background, binomial_interval

logger = logging.getLogger(__name__)

#: The volume beyond the reach is estimated out to this many times the reach.
BEYOND_REACH_FACTOR = 2.0
#: The found chance beyond the reach is binned in SNR: at least this many injections a bin, at most this many bins.
BEYOND_REACH_BIN_SIZE = 50
BEYOND_REACH_MAX_BINS = 40

GPC3_PER_MPC3 = 1e-9
#: A Julian year, and a twelfth of it.
SECONDS_PER_YEAR = 365.25 * 86_400.0
SECONDS_PER_MONTH = SECONDS_PER_YEAR / 12.0


def sphere_volume_gpc3(radius_mpc) -> np.ndarray:
    """The volume (Gpc^3) of a sphere of ``radius_mpc``."""
    return 4.0 / 3.0 * np.pi * np.asarray(radius_mpc, dtype=np.float64) ** 3 * GPC3_PER_MPC3


def sensitive_distance(volume_gpc3) -> np.ndarray:
    """The sensitive distance (Mpc) of a sensitive volume (Gpc^3): ``(3 V / 4 pi)^(1/3)``."""
    volume_mpc3 = np.asarray(volume_gpc3, dtype=np.float64) / GPC3_PER_MPC3
    return np.cbrt(3.0 * volume_mpc3 / (4.0 * np.pi))


def uniform_in_volume(num: int, reach_mpc: float, rng: np.random.Generator) -> np.ndarray:
    """``num`` distances (Mpc) drawn uniformly in the volume of a sphere of radius ``reach_mpc``: all > 0."""
    if not reach_mpc > 0:
        raise ValueError(f"reach_mpc must be > 0, got {reach_mpc}.")
    return reach_mpc * np.cbrt(1.0 - rng.random(num))


@dataclass(frozen=True)
class InjectionVolumes:
    """
    Injections drawn uniformly in volume out to the reach ``reach_mpc``: injection
    ``i``, at ``distances[i]`` (Mpc; a chirp distance for a draw uniform in chirp
    distance) with optimal network SNR ``snrs[i]``, scored ``scores[i]``, stands,
    when found, for the reference volume times ``weights[i]`` (1 by default; a draw
    uniform in chirp distance gives each its chirp-mass weight). The distances and
    SNRs are the volume beyond the reach's (module docstring).
    """
    scores: np.ndarray
    reach_mpc: float
    distances: np.ndarray
    snrs: np.ndarray
    weights: Optional[np.ndarray] = None

    def __post_init__(self):
        scores = np.asarray(self.scores, dtype=np.float64).reshape(-1)
        distances = np.asarray(self.distances, dtype=np.float64).reshape(-1)
        snrs = np.asarray(self.snrs, dtype=np.float64).reshape(-1)
        weights = np.ones_like(scores) if self.weights is None else \
            np.asarray(self.weights, dtype=np.float64).reshape(-1)
        if scores.size == 0 or any(a.shape != scores.shape for a in (weights, distances, snrs)):
            raise ValueError(f"{weights.size} weights, {distances.size} distances and {snrs.size} SNRs for "
                             f"{scores.size} injection scores: need one of each per injection.")
        if not (np.all(np.isfinite(scores)) and np.all(np.isfinite(weights)) and np.all(weights >= 0)):
            raise ValueError("Injection scores must be finite, and their weights finite and >= 0.")
        if not (np.all(np.isfinite(snrs)) and np.all(snrs >= 0)):
            raise ValueError("Injection SNRs must be finite and >= 0.")
        if not self.reach_mpc > 0:
            raise ValueError(f"reach_mpc must be > 0, got {self.reach_mpc}.")
        if not np.all((distances > 0) & (distances <= self.reach_mpc)):
            raise ValueError(f"Injection distances must be in (0, reach_mpc = {self.reach_mpc}].")
        order = np.argsort(scores, kind="stable")
        for name, value in (("scores", scores), ("distances", distances), ("snrs", snrs), ("weights", weights),
                            ("reach_mpc", float(self.reach_mpc)),
                            # Scores ascending, for the injections found at a threshold: a suffix.
                            ("_sorted_scores", scores[order]), ("_order", order)):
            object.__setattr__(self, name, value)
        self._bin_beyond_reach()

    def _bin_beyond_reach(self) -> None:
        """
        The design of the volume beyond the reach (module docstring): the moved
        injections (``distances >= reach / k``, each at its SNR over ``k``), and
        the SNR bins in which the injections give the found chance at their SNRs.
        """
        k = BEYOND_REACH_FACTOR
        moved = self.distances >= self.reach_mpc / k
        moved_snrs = self.snrs[moved] / k
        # The injections whose SNR is in the moved ones' range (at least a bin's worth) read the found chance.
        lowest = np.sort(self.snrs)[min(BEYOND_REACH_BIN_SIZE, self.snrs.size) - 1]
        reading = self.snrs <= max(moved_snrs.max(initial=0.0), lowest)
        num_bins = int(np.clip(np.count_nonzero(reading) // BEYOND_REACH_BIN_SIZE, 1, BEYOND_REACH_MAX_BINS))
        edges = np.quantile(self.snrs[reading], np.linspace(0.0, 1.0, num_bins + 1)[1:-1])
        for name, value in (("_moved", moved), ("_reading", reading), ("_num_bins", num_bins),
                            ("_reading_bin", np.searchsorted(edges, self.snrs[reading], side="right")),
                            ("_moved_bin", np.searchsorted(edges, moved_snrs, side="right"))):
            object.__setattr__(self, name, value)

    @property
    def num_injections(self) -> int:
        return self.scores.size

    @property
    def reference_volume_gpc3(self) -> float:
        """The population's volume within the reach, ``4/3 pi R^3``."""
        return float(sphere_volume_gpc3(self.reach_mpc))

    def _found_sum(self, thresholds, sorted_values) -> np.ndarray:
        """The sum of ``sorted_values`` (in score order) over the injections found at each threshold; NaN at NaN."""
        t = np.asarray(thresholds, dtype=np.float64)
        unresolved = np.isnan(t)
        first = np.searchsorted(self._sorted_scores, np.where(unresolved, np.inf, t), side="left")
        suffix = np.concatenate([np.cumsum(sorted_values[::-1])[::-1], [0.0]])
        return np.where(unresolved, np.nan, suffix[first])

    def volume(self, thresholds, multiplicity=None) -> np.ndarray:
        """
        V within the reach (Gpc^3) at each threshold; NaN at a NaN threshold (an
        unresolved FAR). ``multiplicity`` counts each injection that many times (a
        bootstrap draw).
        """
        w = self.weights if multiplicity is None else self.weights * np.asarray(multiplicity, dtype=np.float64)
        return self.reference_volume_gpc3 * self._found_sum(thresholds, w[self._order]) / self.num_injections

    def beyond_reach(self, thresholds, noise_chance, multiplicity=None) -> np.ndarray:
        """
        The estimated volume (Gpc^3) found by signal between the reach and
        `BEYOND_REACH_FACTOR` times it, at each threshold (module docstring), given
        ``noise_chance``, the chance that a noise window passes it; NaN at a NaN
        threshold. ``multiplicity`` counts each injection that many times, both as a
        moved source and as a reading of the found chance (a bootstrap draw).
        """
        each, moved = self._beyond_reach_shares(multiplicity)
        return self._beyond_reach_scale * (self._found_sum(thresholds, each[self._order])
                                           - np.asarray(noise_chance, dtype=np.float64) * moved)

    def beyond_reach_interval(self, thresholds, noise_chance, confidence: float = DEFAULT_CONFIDENCE,
                              method: str = "clopper-pearson") -> Tuple[np.ndarray, np.ndarray]:
        """
        The injections' interval on the volume beyond the reach at each threshold: the
        binomial interval on its found sum (`_found_interval`), less the noise's part.
        Where few injections are found (a low FAR, few injections) it is the exact
        term a bootstrap of the found count understates.
        """
        each, moved = self._beyond_reach_shares()
        noise = np.asarray(noise_chance, dtype=np.float64) * moved
        lower, upper = self._found_interval(thresholds, each, confidence, method)
        return self._beyond_reach_scale * (lower - noise), self._beyond_reach_scale * (upper - noise)

    @property
    def _beyond_reach_scale(self) -> float:
        return self.reference_volume_gpc3 * BEYOND_REACH_FACTOR**3 / self.num_injections

    def _beyond_reach_shares(self, multiplicity=None) -> Tuple[np.ndarray, float]:
        """
        What each injection stands for, found, in the volume beyond the reach: a
        reading injection, its bin's moved sources' weight over its bin's readings
        (each counted ``multiplicity`` times); and the moved sources' total weight.
        """
        m = np.ones(self.num_injections) if multiplicity is None else np.asarray(multiplicity, dtype=np.float64)
        bins = self._num_bins
        readings = np.bincount(self._reading_bin, weights=m[self._reading], minlength=bins)
        moved = np.bincount(self._moved_bin, weights=(self.weights * m)[self._moved], minlength=bins)
        each = np.zeros(self.num_injections)
        share = np.divide(moved, readings, out=np.zeros(bins), where=readings > 0)
        each[self._reading] = m[self._reading] * share[self._reading_bin]
        return each, float(moved.sum())

    def standard_error(self, thresholds) -> np.ndarray:
        """
        The Monte Carlo standard error of V: ``V_ref sqrt(sum w^2 f - (sum w f)^2 / N) / N``;
        unweighted, ``V_ref sqrt(N p (1 - p)) / N``.
        """
        w = self.weights[self._order]
        s1, s2 = self._found_sum(thresholds, w), self._found_sum(thresholds, w**2)
        n = self.num_injections
        return self.reference_volume_gpc3 * np.sqrt(np.maximum(s2 - s1**2 / n, 0.0)) / n

    def interval(self, thresholds, confidence: float = DEFAULT_CONFIDENCE,
                 method: str = "clopper-pearson") -> Tuple[np.ndarray, np.ndarray]:
        """A binomial interval on V at each threshold (`_found_interval` of the weights)."""
        lower, upper = self._found_interval(thresholds, self.weights, confidence, method)
        return self.reference_volume_gpc3 * lower / self.num_injections, \
            self.reference_volume_gpc3 * upper / self.num_injections

    def _found_interval(self, thresholds, values, confidence: float,
                        method: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        A binomial interval (`binomial_interval`, by ``method``) on the sum of
        ``values`` (one per injection) over the injections found at each threshold:
        exact on the found count when the values are equal. Otherwise the sum is
        taken as ``k = (sum v f)^2 / sum v^2 f`` effective injections of N, each of
        value ``sum v^2 f / sum v f``, which matches the sum and its variance. With
        nothing found, a found injection's value is the largest. NaN at NaN.
        """
        v = np.asarray(values, dtype=np.float64)[self._order]
        s1, s2 = self._found_sum(thresholds, v), self._found_sum(thresholds, v**2)
        found = np.nan_to_num(s1) > 0
        with np.errstate(divide="ignore", invalid="ignore"):
            each = np.where(found, s2 / s1, v.max())
            effective = np.where(found, s1**2 / s2, 0.0)
        lower, upper = binomial_interval(effective, self.num_injections, confidence, method)
        scale = self.num_injections * each
        unresolved = np.isnan(s1)
        return np.where(unresolved, np.nan, scale * lower), np.where(unresolved, np.nan, scale * upper)


# ------------------------------------------------------------- the curve

@dataclass(frozen=True)
class VolumeQuote:
    """
    The number a comparison quotes at one target FAR: the sensitive volume within
    the reach ``reach_mpc`` with its joint interval, and beside it the estimated
    volume beyond the reach (out to ``beyond_reach_factor`` times it) with its
    interval, ``reach_limited`` when that estimate exceeds the curve's stated
    fraction of V (module docstring). Volumes in Gpc^3 (Euclidean).
    """
    far_hz: float
    reach_mpc: float
    volume_within_reach_gpc3: float
    lower_gpc3: float
    upper_gpc3: float
    beyond_reach_gpc3: float
    beyond_reach_lower_gpc3: float
    beyond_reach_upper_gpc3: float
    beyond_reach_factor: float
    reach_limited: bool


@dataclass(frozen=True)
class SensitiveVolumeCurve:
    """
    Sensitive volume within the reach (Gpc^3) against target FAR (Hz), with its
    three intervals, and the estimated volume beyond the reach with its interval.

    At each target: the background's threshold for it (NaN, and every volume
    NaN, when the background cannot resolve it), the FAR that threshold
    achieves with its Poisson interval, V within the reach ``reach_mpc``, and the
    bounds of each interval; ``injection_error_gpc3`` is the Monte Carlo standard
    error. ``noise_floor_gpc3`` is the volume noise alone finds (module
    docstring); ``noise_dominated`` marks the targets where it exceeds
    ``noise_floor_fraction`` of V. ``beyond_reach_gpc3`` is the estimated volume
    found by signal between the reach and ``beyond_reach_factor`` times it, with
    its interval; ``reach_limited`` marks the targets where it exceeds
    ``beyond_reach_fraction`` of V, a warning. ``quotable`` is the one rule of what
    `quote` gives.
    """
    far_hz: np.ndarray
    threshold: np.ndarray
    resolved: np.ndarray
    achieved_far_hz: np.ndarray
    far_lower_hz: np.ndarray
    far_upper_hz: np.ndarray
    volume_within_reach_gpc3: np.ndarray
    injection_error_gpc3: np.ndarray
    injection_lower_gpc3: np.ndarray
    injection_upper_gpc3: np.ndarray
    background_lower_gpc3: np.ndarray
    background_upper_gpc3: np.ndarray
    joint_lower_gpc3: np.ndarray
    joint_upper_gpc3: np.ndarray
    noise_floor_gpc3: np.ndarray
    noise_floor_fraction: float
    beyond_reach_gpc3: np.ndarray
    beyond_reach_lower_gpc3: np.ndarray
    beyond_reach_upper_gpc3: np.ndarray
    beyond_reach_factor: float
    beyond_reach_fraction: float
    reach_mpc: float
    data_reuse: int
    num_injections: int
    num_bootstrap: int
    confidence: float

    @property
    def distance_within_reach_mpc(self) -> np.ndarray:
        """The sensitive distance (Mpc) of the volume within the reach, at each target."""
        return sensitive_distance(self.volume_within_reach_gpc3)

    @property
    def reference_volume_gpc3(self) -> float:
        """The volume of the reach's sphere."""
        return float(sphere_volume_gpc3(self.reach_mpc))

    @property
    def noise_dominated(self) -> np.ndarray:
        """The resolved targets whose noise floor exceeds ``noise_floor_fraction`` of V."""
        with np.errstate(invalid="ignore"):
            return self.resolved & (self.noise_floor_gpc3 > self.noise_floor_fraction * self.volume_within_reach_gpc3)

    @property
    def reach_limited(self) -> np.ndarray:
        """The resolved targets whose estimated volume beyond the reach exceeds ``beyond_reach_fraction`` of V."""
        with np.errstate(invalid="ignore"):
            return self.resolved & (self.beyond_reach_gpc3 > self.beyond_reach_fraction * self.volume_within_reach_gpc3)

    @property
    def quotable(self) -> np.ndarray:
        """The targets `quote` gives: resolved, and not noise-dominated. A reach-limited one is quoted with a warning."""
        return self.resolved & ~self.noise_dominated

    def quote(self, far_hz: float) -> VolumeQuote:
        """
        V within the reach at the target ``far_hz``, with its joint interval and the
        estimated volume beyond the reach: the number a comparison quotes. Refuses,
        by name, a target the background cannot resolve and one noise-dominated (V
        mostly what noise alone finds); warns at a reach-limited one.
        """
        matches = np.flatnonzero(np.isclose(self.far_hz, far_hz, rtol=1e-9, atol=0))
        if matches.size == 0:
            raise ValueError(f"{far_hz:g} Hz is not one of the curve's target FARs.")
        i = int(matches[0])
        if not self.quotable[i]:
            why = ("the background cannot resolve it" if not self.resolved[i] else
                   f"it is noise-dominated: noise alone finds more than {self.noise_floor_fraction:.0%} of V")
            raise ValueError(f"No sensitive volume to quote at {far_hz:g} Hz: {why}.")
        quote = VolumeQuote(
            far_hz=float(self.far_hz[i]), reach_mpc=self.reach_mpc,
            volume_within_reach_gpc3=float(self.volume_within_reach_gpc3[i]),
            lower_gpc3=float(self.joint_lower_gpc3[i]), upper_gpc3=float(self.joint_upper_gpc3[i]),
            beyond_reach_gpc3=float(self.beyond_reach_gpc3[i]),
            beyond_reach_lower_gpc3=float(self.beyond_reach_lower_gpc3[i]),
            beyond_reach_upper_gpc3=float(self.beyond_reach_upper_gpc3[i]),
            beyond_reach_factor=self.beyond_reach_factor, reach_limited=bool(self.reach_limited[i]))
        if quote.reach_limited:
            logger.warning(
                f"At {far_hz:g} Hz the volume within the reach ({self.reach_mpc:.0f} Mpc), "
                f"{quote.volume_within_reach_gpc3:.3g} Gpc^3, misses an estimated {quote.beyond_reach_gpc3:.3g} "
                f"Gpc^3 ({quote.beyond_reach_lower_gpc3:.3g}-{quote.beyond_reach_upper_gpc3:.3g}) found between it "
                f"and {self.beyond_reach_factor:g} times it: more than {self.beyond_reach_fraction:.0%} of V.")
        return quote


def sensitive_volume_curve(
    injections: InjectionVolumes,
    background: Background,
    fars_hz: Sequence[float],
    rng: np.random.Generator,
    num_bootstrap: int = 1000,
    confidence: float = DEFAULT_CONFIDENCE,
    method: str = "clopper-pearson",
    noise_floor_fraction: float = 0.1,
    beyond_reach_fraction: float = 0.1,
) -> SensitiveVolumeCurve:
    """
    V within the reach against FAR at each of ``fars_hz``, with its injection,
    background and joint intervals, and the estimated volume beyond the reach with
    its interval (module docstring).

    The bootstrap draws ``num_bootstrap`` replicates from ``rng``, each the
    background's thresholds (`Background.bootstrap_thresholds`) and the injections
    drawn with replacement, and takes the central ``confidence`` range of the
    replicates' volumes at each target. Where the target allows few triggers, no
    replicate can put the threshold above the loudest background trigger, while
    the true threshold may lie there: the background interval's high threshold
    can (+inf when even no trigger leaves the FAR's upper bound above the
    target), so each interval's ends are at least as far out as the volumes at
    the background interval's thresholds. Where few injections are found, the
    replicates' found counts understate their error: each end is also at least as
    far out as the injections' binomial interval.
    """
    if num_bootstrap < 1:
        raise ValueError(f"num_bootstrap must be >= 1, got {num_bootstrap}.")
    fars = np.asarray(fars_hz, dtype=np.float64).reshape(-1)
    targets = [background.threshold_for_far(float(far), confidence) for far in fars]
    threshold = np.array([t.threshold for t in targets])
    resolved = np.array([t.resolved for t in targets])
    noise_chance = np.where(resolved, background.window_exceedance(threshold), np.nan)

    def volumes(thresholds, multiplicity=None) -> np.ndarray:
        """V within the reach and the estimate beyond it, at each target's threshold: (2, targets)."""
        return np.stack([injections.volume(thresholds, multiplicity),
                         injections.beyond_reach(thresholds, noise_chance, multiplicity)])

    bands = np.array([background.far_band_thresholds(float(far), confidence) if ok else (np.nan, np.nan)
                      for far, ok in zip(fars, resolved)]).reshape(-1, 2)
    at_bands = np.stack([volumes(bands[:, 0]), volumes(bands[:, 1])])  # (high, low) thresholds

    n = injections.num_injections
    replicates = np.empty((num_bootstrap, 2, fars.size))
    for b in range(num_bootstrap):
        thresholds = background.bootstrap_thresholds(fars, rng)
        replicates[b] = volumes(thresholds, np.bincount(rng.integers(0, n, n), minlength=n))
    alpha = 1.0 - confidence
    ends = np.full((2, 2, fars.size), np.nan)
    if np.any(resolved):
        ends[:, :, resolved] = np.quantile(replicates[:, :, resolved], [alpha / 2, 1 - alpha / 2], axis=0)
    volume, beyond = volumes(threshold)
    injection_lower, injection_upper = injections.interval(threshold, confidence, method)
    # Each term's interval: the injections' (exact where few are found, which a bootstrap
    # of the found count understates) and the background's.
    terms = np.stack([np.stack([injection_lower, injection_upper]),
                      np.stack(injections.beyond_reach_interval(threshold, noise_chance, confidence, method))],
                     axis=1)  # (end, quantity, target)
    lower = np.minimum.reduce([ends[0], at_bands.min(axis=0), terms[0]])
    upper = np.maximum.reduce([ends[1], at_bands.max(axis=0), terms[1]])
    return SensitiveVolumeCurve(
        far_hz=fars,
        threshold=threshold,
        resolved=resolved,
        achieved_far_hz=np.array([t.achieved.rate_hz for t in targets]),
        far_lower_hz=np.array([t.achieved.lower_hz for t in targets]),
        far_upper_hz=np.array([t.achieved.upper_hz for t in targets]),
        volume_within_reach_gpc3=volume,
        injection_error_gpc3=injections.standard_error(threshold),
        injection_lower_gpc3=injection_lower,
        injection_upper_gpc3=injection_upper,
        background_lower_gpc3=at_bands[0, 0],
        background_upper_gpc3=at_bands[1, 0],
        joint_lower_gpc3=lower[0],
        joint_upper_gpc3=upper[0],
        noise_floor_gpc3=noise_chance * injections.reference_volume_gpc3,
        noise_floor_fraction=float(noise_floor_fraction),
        beyond_reach_gpc3=beyond,
        beyond_reach_lower_gpc3=lower[1],
        beyond_reach_upper_gpc3=upper[1],
        beyond_reach_factor=BEYOND_REACH_FACTOR,
        beyond_reach_fraction=float(beyond_reach_fraction),
        reach_mpc=injections.reach_mpc,
        data_reuse=background.data_reuse,
        num_injections=n,
        num_bootstrap=int(num_bootstrap),
        confidence=float(confidence),
    )
