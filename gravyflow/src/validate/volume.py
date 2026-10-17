"""
Sensitive volume and sensitive distance against false-alarm rate.

The injections are drawn as an injection campaign draws them: each source from
the population, placed at a physical distance drawn uniformly in volume out to
a fixed reach ``max_distance_mpc``, which the population defines and never the
sample (the validator's is the horizon of the population's loudest source:
`ValidationBank.horizon_distance`). A found injection (``score >= threshold``,
as everywhere in `metrics`) stands for the reference volume
``V_ref = 4/3 pi max_distance^3`` times its population weight ``w`` (1 when the
draw is the population's own; a power of the chirp mass for a draw uniform in
chirp distance), so

    V(threshold) = V_ref sum_i w_i [score_i >= threshold] / N.

Unweighted, V is ``V_ref k / N`` for ``k`` found of ``N``, and its injection
interval is the exact binomial one.

The sensitive distance is the radius of a sphere of that volume,
``R = (3 V / 4 pi)^(1/3)``. At a target FAR, the threshold is the one the noise
background gives that FAR (`Background.threshold_for_far`). Three intervals are
reported, each two-sided and central at ``confidence``:

- **injections**: the finite Monte Carlo, a binomial interval on V at the
  threshold (`InjectionVolumes.interval`);
- **background**: the finite livetime, the V at the two thresholds between
  which the FAR's Poisson interval puts the true threshold
  (`Background.far_band_thresholds`);
- **joint**: a bootstrap over injections and background blocks
  (`Background.bootstrap`), the threshold found again in each replicate, with
  its lower end at most the background interval's (`sensitive_volume_curve`).

**The noise floor.** An injection is found when its window passes the
threshold, whatever passes it: one beyond reach is found by its noise alone,
with the chance ``p`` that a noise window passes (`Background.window_exceedance`).
Those add about ``p V_ref`` to V, which is part of what the threshold finds,
not an error; it is recorded at each target, and a point where it exceeds a
stated fraction of V is marked noise-dominated.

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


def uniform_in_volume(num: int, max_distance_mpc: float, rng: np.random.Generator) -> np.ndarray:
    """``num`` distances (Mpc) drawn uniformly in the volume of a sphere of ``max_distance_mpc``: all > 0."""
    if not max_distance_mpc > 0:
        raise ValueError(f"max_distance_mpc must be > 0, got {max_distance_mpc}.")
    return max_distance_mpc * np.cbrt(1.0 - rng.random(num))


@dataclass(frozen=True)
class InjectionVolumes:
    """
    Injections drawn uniformly in volume out to ``max_distance_mpc``: injection
    ``i`` scored ``scores[i]`` and stands, when found, for the reference volume
    times ``weights[i]`` (1 by default; a draw uniform in chirp distance gives
    each its chirp-mass weight).
    """
    scores: np.ndarray
    max_distance_mpc: float
    weights: Optional[np.ndarray] = None

    def __post_init__(self):
        scores = np.asarray(self.scores, dtype=np.float64).reshape(-1)
        weights = np.ones_like(scores) if self.weights is None else \
            np.asarray(self.weights, dtype=np.float64).reshape(-1)
        if scores.size == 0 or scores.shape != weights.shape:
            raise ValueError(f"{weights.size} weights for {scores.size} injection scores: need one per injection.")
        if not (np.all(np.isfinite(scores)) and np.all(np.isfinite(weights)) and np.all(weights >= 0)):
            raise ValueError("Injection scores must be finite, and their weights finite and >= 0.")
        if not self.max_distance_mpc > 0:
            raise ValueError(f"max_distance_mpc must be > 0, got {self.max_distance_mpc}.")
        order = np.argsort(scores, kind="stable")
        object.__setattr__(self, "scores", scores)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "max_distance_mpc", float(self.max_distance_mpc))
        # Scores ascending, for the injections found at a threshold: a suffix.
        object.__setattr__(self, "_sorted_scores", scores[order])
        object.__setattr__(self, "_order", order)

    @property
    def num_injections(self) -> int:
        return self.scores.size

    @property
    def reference_volume_gpc3(self) -> float:
        """The population's volume, ``4/3 pi max_distance^3``."""
        return float(sphere_volume_gpc3(self.max_distance_mpc))

    def _found_sum(self, thresholds, sorted_values) -> np.ndarray:
        """The sum of ``sorted_values`` (in score order) over the injections found at each threshold; NaN at NaN."""
        t = np.asarray(thresholds, dtype=np.float64)
        unresolved = np.isnan(t)
        first = np.searchsorted(self._sorted_scores, np.where(unresolved, np.inf, t), side="left")
        suffix = np.concatenate([np.cumsum(sorted_values[::-1])[::-1], [0.0]])
        return np.where(unresolved, np.nan, suffix[first])

    def volume(self, thresholds, multiplicity=None) -> np.ndarray:
        """
        V (Gpc^3) at each threshold; NaN at a NaN threshold (an unresolved FAR).
        ``multiplicity`` counts each injection that many times (a bootstrap draw).
        """
        w = self.weights[self._order]
        if multiplicity is not None:
            w = w * np.asarray(multiplicity, dtype=np.float64)[self._order]
        return self.reference_volume_gpc3 * self._found_sum(thresholds, w) / self.num_injections

    def _moments(self, thresholds) -> Tuple[np.ndarray, np.ndarray]:
        w = self.weights[self._order]
        return self._found_sum(thresholds, w), self._found_sum(thresholds, w**2)

    def standard_error(self, thresholds) -> np.ndarray:
        """
        The Monte Carlo standard error of V: ``V_ref sqrt(sum w^2 f - (sum w f)^2 / N) / N``;
        unweighted, ``V_ref sqrt(N p (1 - p)) / N``.
        """
        s1, s2 = self._moments(thresholds)
        n = self.num_injections
        return self.reference_volume_gpc3 * np.sqrt(np.maximum(s2 - s1**2 / n, 0.0)) / n

    def interval(self, thresholds, confidence: float = DEFAULT_CONFIDENCE,
                 method: str = "clopper-pearson") -> Tuple[np.ndarray, np.ndarray]:
        """
        A binomial interval on V at each threshold (`binomial_interval`, by
        ``method``): exact on the found count when the weights are equal. Weighted,
        V is taken as ``k = (sum w f)^2 / sum w^2 f`` effective injections of N, each
        of weight ``sum w^2 f / sum w f``, which matches V and its variance. With
        nothing found, a found injection's weight is the largest.
        """
        s1, s2 = self._moments(thresholds)
        found = np.nan_to_num(s1) > 0
        with np.errstate(divide="ignore", invalid="ignore"):
            each = np.where(found, s2 / s1, self.weights.max())
            effective = np.where(found, s1**2 / s2, 0.0)
        lower, upper = binomial_interval(effective, self.num_injections, confidence, method)
        scale = self.reference_volume_gpc3 * each
        unresolved = np.isnan(s1)
        return np.where(unresolved, np.nan, scale * lower), np.where(unresolved, np.nan, scale * upper)


# ------------------------------------------------------------- the curve

@dataclass(frozen=True)
class SensitiveVolumeCurve:
    """
    Sensitive volume (Gpc^3) against target FAR (Hz), with its three intervals.

    At each target: the background's threshold for it (NaN, and every volume
    NaN, when the background cannot resolve it), the FAR that threshold
    achieves with its Poisson interval, V, and the bounds of each interval;
    ``injection_error_gpc3`` is the Monte Carlo standard error.
    ``joint_unresolved`` counts the bootstrap replicates whose background could
    not resolve the target (their livetime too short); where any could not, the
    joint interval is NaN. ``noise_floor_gpc3`` is the volume noise alone finds
    (module docstring); ``noise_dominated`` marks the targets where it exceeds
    ``noise_floor_fraction`` of V.
    """
    far_hz: np.ndarray
    threshold: np.ndarray
    resolved: np.ndarray
    achieved_far_hz: np.ndarray
    far_lower_hz: np.ndarray
    far_upper_hz: np.ndarray
    volume_gpc3: np.ndarray
    injection_error_gpc3: np.ndarray
    injection_lower_gpc3: np.ndarray
    injection_upper_gpc3: np.ndarray
    background_lower_gpc3: np.ndarray
    background_upper_gpc3: np.ndarray
    joint_lower_gpc3: np.ndarray
    joint_upper_gpc3: np.ndarray
    joint_unresolved: np.ndarray
    noise_floor_gpc3: np.ndarray
    noise_floor_fraction: float
    max_distance_mpc: float
    num_injections: int
    num_bootstrap: int
    confidence: float

    @property
    def distance_mpc(self) -> np.ndarray:
        """The sensitive distance (Mpc) at each target."""
        return sensitive_distance(self.volume_gpc3)

    @property
    def reference_volume_gpc3(self) -> float:
        return float(sphere_volume_gpc3(self.max_distance_mpc))

    @property
    def noise_dominated(self) -> np.ndarray:
        """The resolved targets whose noise floor exceeds ``noise_floor_fraction`` of V."""
        with np.errstate(invalid="ignore"):
            return self.resolved & (self.noise_floor_gpc3 > self.noise_floor_fraction * self.volume_gpc3)


def sensitive_volume_curve(
    injections: InjectionVolumes,
    background: Background,
    fars_hz: Sequence[float],
    rng: np.random.Generator,
    num_bootstrap: int = 1000,
    confidence: float = DEFAULT_CONFIDENCE,
    method: str = "clopper-pearson",
    noise_floor_fraction: float = 0.1,
) -> SensitiveVolumeCurve:
    """
    V against FAR at each of ``fars_hz``, with its injection, background and
    joint intervals (module docstring).

    The joint interval's bootstrap draws ``num_bootstrap`` replicates from
    ``rng``, each a background bootstrap (`Background.bootstrap`) and the
    injections drawn with replacement, and takes the central ``confidence``
    range of the replicates' V at the threshold each one's own background gives
    the target. Where the target allows few triggers, no replicate can put the
    threshold above the loudest background trigger, while the true threshold may
    lie there: the background interval's high threshold can (+inf when even no
    trigger leaves the FAR's upper bound above the target), so the joint lower
    end is at most the background interval's.
    """
    if num_bootstrap < 1:
        raise ValueError(f"num_bootstrap must be >= 1, got {num_bootstrap}.")
    fars = np.asarray(fars_hz, dtype=np.float64).reshape(-1)
    targets = [background.threshold_for_far(float(far), confidence) for far in fars]
    threshold = np.array([t.threshold for t in targets])
    resolved = np.array([t.resolved for t in targets])
    bands = np.array([background.far_band_thresholds(float(far), confidence) if ok else (np.nan, np.nan)
                      for far, ok in zip(fars, resolved)]).reshape(-1, 2)
    # The higher threshold finds less.
    background_lower, background_upper = injections.volume(bands[:, 0]), injections.volume(bands[:, 1])

    n = injections.num_injections
    replicates = np.empty((num_bootstrap, fars.size))
    for b in range(num_bootstrap):
        replica = background.bootstrap(rng)
        multiplicity = np.bincount(rng.integers(0, n, n), minlength=n)
        thresholds = [replica.threshold_for_far(float(far), confidence).threshold if ok else np.nan
                      for far, ok in zip(fars, resolved)]
        replicates[b] = injections.volume(thresholds, multiplicity)
    unresolved = np.where(resolved, np.isnan(replicates).sum(axis=0), 0)
    if np.any(unresolved):
        logger.warning(
            "The background bootstrap could not resolve the target FAR in some replicates (blocks of unequal "
            "livetime); the joint interval is NaN at: " + ", ".join(
                f"{far:.3g} Hz ({count} of {num_bootstrap})" for far, count in zip(fars, unresolved) if count))
    alpha = 1.0 - confidence
    joint = np.full((2, fars.size), np.nan)
    complete = resolved & (unresolved == 0)
    if np.any(complete):
        joint[:, complete] = np.quantile(replicates[:, complete], [alpha / 2, 1 - alpha / 2], axis=0)
    injection_lower, injection_upper = injections.interval(threshold, confidence, method)
    return SensitiveVolumeCurve(
        far_hz=fars,
        threshold=threshold,
        resolved=resolved,
        achieved_far_hz=np.array([t.achieved.rate_hz for t in targets]),
        far_lower_hz=np.array([t.achieved.lower_hz for t in targets]),
        far_upper_hz=np.array([t.achieved.upper_hz for t in targets]),
        volume_gpc3=injections.volume(threshold),
        injection_error_gpc3=injections.standard_error(threshold),
        injection_lower_gpc3=injection_lower,
        injection_upper_gpc3=injection_upper,
        background_lower_gpc3=background_lower,
        background_upper_gpc3=background_upper,
        joint_lower_gpc3=np.minimum(joint[0], background_lower),
        joint_upper_gpc3=joint[1],
        joint_unresolved=unresolved.astype(np.int64),
        noise_floor_gpc3=np.where(resolved, background.window_exceedance(threshold), np.nan)
        * injections.reference_volume_gpc3,
        noise_floor_fraction=float(noise_floor_fraction),
        max_distance_mpc=injections.max_distance_mpc,
        num_injections=n,
        num_bootstrap=int(num_bootstrap),
        confidence=float(confidence),
    )
