"""
Sensitive volume and sensitive distance against false-alarm rate, within the reach.

**The reach is per mass.** Nothing beyond the distance at which even the loudest
source of a mass (its best sky, orientation and spins) has a network SNR of about
2 can be found, so the volume set places each source out to its own mass's reach
and no further: drawn uniformly in *chirp distance*, ``d_c = d (M_ref / M)^(5/6)``
(``M`` the source's detector-frame chirp mass, ``M_ref`` `REFERENCE_CHIRP_MASS_MSUN`,
two neutron stars of 1.4), out to the chirp reach ``R`` (`reach.chirp_reach`), so a
source of chirp mass ``M`` lies out to ``R (M / M_ref)^(5/6)``, its mass's reach,
and the reach of every mass is at least where its best case has the reach SNR. A
found injection (``score >= threshold``, as everywhere in `metrics`) stands for its
mass's reach volume, ``V_R (M / M_ref)^(5/2)`` with ``V_R = 4/3 pi R^3``: its
weight (`chirp_weights`) keeps the volume exact. A draw uniform in physical
distance out to one reach for every mass is the same estimator with weights 1.

**The chance of a noise-only pass is subtracted.** An injection is found when
its window passes the threshold, whatever passes it: noise alone passes with the
chance ``c`` (`metrics.Background.found_by_chance`), everywhere in the reach, so
of the population's volume ``V_pop = V_R mean(w)`` noise alone finds ``c V_pop``,
and the volume found by signal is

    V = (V_found - c V_pop) / (1 - c),    V_found = V_R sum_i w_i [score_i >= t] / N.

V is the estimand: without the subtraction the found volume would grow with the
reach's cube however far the detector sees. At tight thresholds ``c`` is
negligible; where what noise alone finds exceeds a stated fraction of what is
found, the result rests on the subtraction and is not quoted (`quotable`).

Every result states its reach: ``R`` (chirp distance), the reach SNR it was drawn
for, and the range of physical reaches its masses span. The volume is Euclidean
(no cosmology) and the masses are in the detector frame, as in MLGWSC-1's
``evaluate.py``. The sensitive distance is the radius of a sphere of that volume,
``(3 V / 4 pi)^(1/3)``. At a target FAR the threshold is the one the noise
background gives it (`Background.threshold_for_far`). Three intervals are
reported, two-sided and central at each level (1 and 2 sigma):

- **injections**: the finite Monte Carlo, a binomial interval on the found volume
  at the threshold (`InjectionVolumes.interval`), the chance subtracted;
- **background**: the finite livetime, V at the two thresholds between which the
  FAR's interval puts the true threshold (`Background.far_band_thresholds`);
- **joint**: a bootstrap over the injections, the background's thresholds
  (`Background.bootstrap_thresholds`, which counts a time-slid background's
  independent information) and the chance (its law at each replicate's
  threshold), each end at least as far out as the other two intervals'.

The found fraction against distance (`found_fraction_by_distance`), drawn with the
chance level, is the visual check that the reach is long enough: at its far end it
falls to the chance. No other check of the reach is made.

Units: distances in Mpc, volumes in Gpc^3 (Euclidean: no cosmology), FAR in Hz;
`SECONDS_PER_YEAR` and `SECONDS_PER_MONTH` convert it for display.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from gravyflow.src.dataset.features.injection import REFERENCE_CHIRP_MASS_MSUN, chirp_mass
from .metrics import DEFAULT_SIGMAS, Background, binomial_interval, nested, sigma_level

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


def chirp_weights(mass_1, mass_2) -> np.ndarray:
    """
    Each source's weight in a draw uniform in chirp distance: its mass's reach volume
    over the chirp reach's, ``(M / M_ref)^(5/2)`` (module docstring).
    """
    masses = (np.asarray(m, dtype=np.float64) for m in (mass_1, mass_2))
    return (chirp_mass(*masses) / REFERENCE_CHIRP_MASS_MSUN) ** 2.5


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
    distance), scored ``scores[i]``, stands, when found, for the reference volume
    times ``weights[i]`` (1 by default; a draw uniform in chirp distance gives each
    its `chirp_weights`).
    """
    scores: np.ndarray
    reach_mpc: float
    distances: np.ndarray
    weights: Optional[np.ndarray] = None

    def __post_init__(self):
        scores = np.asarray(self.scores, dtype=np.float64).reshape(-1)
        distances = np.asarray(self.distances, dtype=np.float64).reshape(-1)
        weights = np.ones_like(scores) if self.weights is None else \
            np.asarray(self.weights, dtype=np.float64).reshape(-1)
        if scores.size == 0 or any(a.shape != scores.shape for a in (weights, distances)):
            raise ValueError(f"{weights.size} weights and {distances.size} distances for {scores.size} "
                             "injection scores: need one of each per injection.")
        if not (np.all(np.isfinite(scores)) and np.all(np.isfinite(weights)) and np.all(weights >= 0)):
            raise ValueError("Injection scores must be finite, and their weights finite and >= 0.")
        if not self.reach_mpc > 0:
            raise ValueError(f"reach_mpc must be > 0, got {self.reach_mpc}.")
        if not np.all((distances > 0) & (distances <= self.reach_mpc)):
            raise ValueError(f"Injection distances must be in (0, reach_mpc = {self.reach_mpc}].")
        order = np.argsort(scores, kind="stable")
        for name, value in (("scores", scores), ("distances", distances), ("weights", weights),
                            ("reach_mpc", float(self.reach_mpc)),
                            # Scores ascending, for the injections found at a threshold: a suffix.
                            ("_sorted_scores", scores[order]), ("_order", order)):
            object.__setattr__(self, name, value)

    @property
    def num_injections(self) -> int:
        return self.scores.size

    @property
    def reference_volume_gpc3(self) -> float:
        """The volume within the reach ``R``, ``4/3 pi R^3``."""
        return float(sphere_volume_gpc3(self.reach_mpc))

    @property
    def population_volume_gpc3(self) -> float:
        """The volume the population's reaches enclose, ``V_R mean(w)``: the most V can be."""
        return self.reference_volume_gpc3 * float(self.weights.mean())

    @property
    def reach_range_mpc(self) -> Tuple[float, float]:
        """The least and the greatest of the sources' own reaches, ``R w^(1/3)``."""
        reaches = self.reach_mpc * np.cbrt(self.weights)
        return float(reaches.min()), float(reaches.max())

    def _found_sum(self, thresholds, sorted_values) -> np.ndarray:
        """The sum of ``sorted_values`` (in score order) over the injections found at each threshold; NaN at NaN."""
        t = np.asarray(thresholds, dtype=np.float64)
        unresolved = np.isnan(t)
        first = np.searchsorted(self._sorted_scores, np.where(unresolved, np.inf, t), side="left")
        suffix = np.concatenate([np.cumsum(sorted_values[::-1])[::-1], [0.0]])
        return np.where(unresolved, np.nan, suffix[first])

    def found_volume(self, thresholds, multiplicity=None) -> np.ndarray:
        """
        The volume found at each threshold (Gpc^3), noise-only passes included: NaN at a
        NaN threshold (an unresolved FAR). ``multiplicity`` counts each injection that
        many times (a bootstrap draw).
        """
        w = self.weights if multiplicity is None else self.weights * np.asarray(multiplicity, dtype=np.float64)
        return self.reference_volume_gpc3 * self._found_sum(thresholds, w[self._order]) / self.num_injections

    def _population(self, multiplicity=None) -> float:
        if multiplicity is None:
            return self.population_volume_gpc3
        return self.reference_volume_gpc3 * float(np.sum(self.weights * multiplicity)) / self.num_injections

    def volume(self, thresholds, chance=0.0, multiplicity=None) -> np.ndarray:
        """
        V within the reach (Gpc^3) at each threshold: what is found less what noise alone
        finds with ``chance`` (module docstring). NaN at a NaN threshold.
        """
        c = np.asarray(chance, dtype=np.float64)
        return (self.found_volume(thresholds, multiplicity) - c * self._population(multiplicity)) / (1.0 - c)

    def standard_error(self, thresholds) -> np.ndarray:
        """
        The Monte Carlo standard error of the found volume:
        ``V_R sqrt(sum w^2 f - (sum w f)^2 / N) / N``; unweighted, ``V_R sqrt(N p (1 - p)) / N``.
        """
        w = self.weights[self._order]
        s1, s2 = self._found_sum(thresholds, w), self._found_sum(thresholds, w**2)
        n = self.num_injections
        return self.reference_volume_gpc3 * np.sqrt(np.maximum(s2 - s1**2 / n, 0.0)) / n

    def interval(self, thresholds, confidence: float = sigma_level(1.0), method: str = "clopper-pearson",
                 chance=0.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        A binomial interval (`binomial_interval`, by ``method``) on the found volume at
        each threshold, its ends mapped as V is (the ``chance`` subtracted). Exact on the
        found count when the weights are equal. Otherwise the found weight is taken as
        ``k = (sum w f)^2 / sum w^2 f`` effective injections of N, each of weight
        ``sum w^2 f / sum w f``, which matches V and its variance. With nothing found,
        a found injection's weight is the largest. NaN at NaN.
        """
        w = self.weights[self._order]
        s1, s2 = self._found_sum(thresholds, w), self._found_sum(thresholds, w**2)
        found = np.nan_to_num(s1) > 0
        with np.errstate(divide="ignore", invalid="ignore"):
            each = np.where(found, s2 / s1, w.max())
            effective = np.where(found, s1**2 / s2, 0.0)
        lower, upper = binomial_interval(effective, self.num_injections, confidence, method)
        scale = self.reference_volume_gpc3 * each
        c = np.asarray(chance, dtype=np.float64)
        subtract = lambda found_volume: (found_volume - c * self.population_volume_gpc3) / (1.0 - c)
        unresolved = np.isnan(s1)
        return (np.where(unresolved, np.nan, subtract(scale * lower)),
                np.where(unresolved, np.nan, subtract(scale * upper)))


def found_fraction_by_distance(injections: InjectionVolumes, threshold: float,
                               num_bins: int = 20) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    The injections found at ``threshold`` in each of ``num_bins`` equal bins of
    distance (chirp distance for a chirp-distance draw) out to the reach: ``(edges,
    found, total)``. Beside the chance level, it shows whether the reach is long
    enough: the found fraction should fall to the chance at the far end. NaN found
    at a NaN threshold.
    """
    edges = np.linspace(0.0, injections.reach_mpc, num_bins + 1)
    total, _ = np.histogram(injections.distances, edges)
    if np.isnan(threshold):
        return edges, np.full(num_bins, np.nan), total
    found, _ = np.histogram(injections.distances[injections.scores >= threshold], edges)
    return edges, found, total


# ------------------------------------------------------------- the curve

@dataclass(frozen=True)
class VolumeQuote:
    """
    The number a comparison quotes at one target FAR: the sensitive volume within the
    reach (Gpc^3, Euclidean) with its joint interval at each of ``sigmas``, and the
    reach it is within: ``reach_mpc`` (chirp distance), spanning ``reach_range_mpc``
    of physical distance over the population's masses.
    """
    far_hz: float
    reach_mpc: float
    reach_range_mpc: Tuple[float, float]
    volume_within_reach_gpc3: float
    lower_gpc3: Tuple[float, ...]
    upper_gpc3: Tuple[float, ...]
    sigmas: Tuple[float, ...]


@dataclass(frozen=True)
class SensitiveVolumeCurve:
    """
    Sensitive volume within the reach (Gpc^3) against target FAR (Hz), with its
    three intervals at each of ``sigmas`` (each ``(levels, targets)``).

    At each target: the background's threshold for it (NaN, and every volume NaN,
    when the background cannot resolve it), the FAR that threshold achieves with its
    interval, the volume found there (``found_gpc3``), the chance of a noise-only
    pass (``chance``) and the volume noise alone finds (``chance_gpc3``, that chance
    of ``population_volume_gpc3``), V within the reach (the chance subtracted), and
    each interval's bounds; ``injection_error_gpc3`` is the found volume's Monte Carlo
    standard error. ``chance_dominated`` marks the targets where noise alone finds
    more than ``chance_fraction`` of what is found; ``quotable`` is the one rule of
    what `quote` gives.
    """
    far_hz: np.ndarray
    threshold: np.ndarray
    resolved: np.ndarray
    achieved_far_hz: np.ndarray
    far_lower_hz: np.ndarray
    far_upper_hz: np.ndarray
    volume_within_reach_gpc3: np.ndarray
    found_gpc3: np.ndarray
    chance: np.ndarray
    chance_gpc3: np.ndarray
    injection_error_gpc3: np.ndarray
    injection_lower_gpc3: np.ndarray
    injection_upper_gpc3: np.ndarray
    background_lower_gpc3: np.ndarray
    background_upper_gpc3: np.ndarray
    joint_lower_gpc3: np.ndarray
    joint_upper_gpc3: np.ndarray
    chance_fraction: float
    reach_mpc: float
    reach_range_mpc: Tuple[float, float]
    population_volume_gpc3: float
    data_reuse: int
    num_injections: int
    num_bootstrap: int
    sigmas: Tuple[float, ...]

    @property
    def distance_within_reach_mpc(self) -> np.ndarray:
        """The sensitive distance (Mpc) of the volume within the reach, at each target."""
        return sensitive_distance(self.volume_within_reach_gpc3)

    @property
    def reference_volume_gpc3(self) -> float:
        """The volume of the chirp reach's sphere."""
        return float(sphere_volume_gpc3(self.reach_mpc))

    @property
    def chance_dominated(self) -> np.ndarray:
        """The resolved targets where noise alone finds more than ``chance_fraction`` of what is found."""
        with np.errstate(invalid="ignore"):
            return self.resolved & (self.chance_gpc3 > self.chance_fraction * self.found_gpc3)

    @property
    def quotable(self) -> np.ndarray:
        """The targets `quote` gives: resolved, and not chance-dominated."""
        return self.resolved & ~self.chance_dominated

    def quote(self, far_hz: float) -> VolumeQuote:
        """
        V within the reach at the target ``far_hz``, with its joint interval at each
        level and the reach: the number a comparison quotes. Refuses, by name, a target
        the background cannot resolve and one chance-dominated (V rests mostly on the
        subtraction of what noise alone finds).
        """
        matches = np.flatnonzero(np.isclose(self.far_hz, far_hz, rtol=1e-9, atol=0))
        if matches.size == 0:
            raise ValueError(f"{far_hz:g} Hz is not one of the curve's target FARs.")
        i = int(matches[0])
        if not self.quotable[i]:
            why = ("the background cannot resolve it" if not self.resolved[i] else
                   f"noise alone finds more than {self.chance_fraction:.0%} of what is found there")
            raise ValueError(f"No sensitive volume to quote at {far_hz:g} Hz: {why}.")
        return VolumeQuote(
            far_hz=float(self.far_hz[i]), reach_mpc=self.reach_mpc, reach_range_mpc=self.reach_range_mpc,
            volume_within_reach_gpc3=float(self.volume_within_reach_gpc3[i]),
            lower_gpc3=tuple(float(v) for v in self.joint_lower_gpc3[:, i]),
            upper_gpc3=tuple(float(v) for v in self.joint_upper_gpc3[:, i]), sigmas=self.sigmas)


def sensitive_volume_curve(
    injections: InjectionVolumes,
    background: Background,
    fars_hz: Sequence[float],
    rng: np.random.Generator,
    num_bootstrap: int = 1000,
    sigmas: Sequence[float] = DEFAULT_SIGMAS,
    method: str = "clopper-pearson",
    chance_fraction: float = 0.1,
) -> SensitiveVolumeCurve:
    """
    V within the reach against FAR at each of ``fars_hz``, the chance of a noise-only
    pass subtracted, with its injection, background and joint intervals at each of
    ``sigmas`` (module docstring).

    The bootstrap draws ``num_bootstrap`` replicates from ``rng``, each the
    background's thresholds (`Background.bootstrap_thresholds`), the chance at them
    from its law, and the injections drawn with replacement, and takes the central
    range of the replicates' volumes at each level. Where the target allows few
    triggers, no replicate can put the threshold above the loudest background
    trigger, while the true threshold may lie there: the background interval's high
    threshold can (+inf when even no trigger leaves the FAR's upper bound above the
    target), so each end is at least as far out as the volumes at the background
    interval's thresholds. Where few injections are found, the replicates' found
    counts understate their error: each end is also at least as far out as the
    injections' binomial interval.
    """
    if num_bootstrap < 1:
        raise ValueError(f"num_bootstrap must be >= 1, got {num_bootstrap}.")
    sigmas = tuple(float(s) for s in sigmas)
    fars = np.asarray(fars_hz, dtype=np.float64).reshape(-1)
    targets = [background.threshold_for_far(float(far)) for far in fars]
    threshold = np.array([t.threshold for t in targets])
    resolved = np.array([t.resolved for t in targets])
    chance = background.found_by_chance(threshold).mean

    n = injections.num_injections
    replicates = np.empty((num_bootstrap, fars.size))
    for b in range(num_bootstrap):
        replicate = background.bootstrap_thresholds(fars, rng)
        replicates[b] = injections.volume(replicate, background.found_by_chance(replicate).sample(rng, 1)[0],
                                          np.bincount(rng.integers(0, n, n), minlength=n))

    shape = (len(sigmas), fars.size)
    intervals = {name: np.full(shape, np.nan) for name in
                 ("far_lower", "far_upper", "injection_lower", "injection_upper", "background_lower",
                  "background_upper", "joint_lower", "joint_upper")}
    for level, sigma in enumerate(sigmas):
        confidence = sigma_level(sigma)
        for i, far in enumerate(fars):
            if resolved[i]:
                achieved = background.far(threshold[i], confidence)
                intervals["far_lower"][level, i], intervals["far_upper"][level, i] = achieved.lower_hz, achieved.upper_hz
        bands = np.array([background.far_band_thresholds(float(far), confidence) if ok else (np.nan, np.nan)
                          for far, ok in zip(fars, resolved)]).reshape(-1, 2)
        at_bands = np.stack([injections.volume(bands[:, j], background.found_by_chance(bands[:, j]).mean)
                             for j in (0, 1)])  # (high, low) thresholds
        injection_lower, injection_upper = injections.interval(threshold, confidence, method, chance)
        alpha = 1.0 - confidence
        ends = np.full((2, fars.size), np.nan)
        if np.any(resolved):
            ends[:, resolved] = np.quantile(replicates[:, resolved], [alpha / 2, 1 - alpha / 2], axis=0)
        intervals["injection_lower"][level], intervals["injection_upper"][level] = injection_lower, injection_upper
        intervals["background_lower"][level], intervals["background_upper"][level] = at_bands.min(axis=0), \
            at_bands.max(axis=0)
        intervals["joint_lower"][level] = np.minimum.reduce([ends[0], at_bands.min(axis=0), injection_lower])
        intervals["joint_upper"][level] = np.maximum.reduce([ends[1], at_bands.max(axis=0), injection_upper])
    for term in ("background", "joint"):
        intervals[f"{term}_lower"], intervals[f"{term}_upper"] = nested(intervals[f"{term}_lower"],
                                                                         intervals[f"{term}_upper"])
    return SensitiveVolumeCurve(
        far_hz=fars,
        threshold=threshold,
        resolved=resolved,
        achieved_far_hz=np.array([t.achieved.rate_hz for t in targets]),
        far_lower_hz=intervals["far_lower"],
        far_upper_hz=intervals["far_upper"],
        volume_within_reach_gpc3=injections.volume(threshold, chance),
        found_gpc3=injections.found_volume(threshold),
        chance=chance,
        chance_gpc3=chance * injections.population_volume_gpc3,
        injection_error_gpc3=injections.standard_error(threshold),
        injection_lower_gpc3=intervals["injection_lower"],
        injection_upper_gpc3=intervals["injection_upper"],
        background_lower_gpc3=intervals["background_lower"],
        background_upper_gpc3=intervals["background_upper"],
        joint_lower_gpc3=intervals["joint_lower"],
        joint_upper_gpc3=intervals["joint_upper"],
        chance_fraction=float(chance_fraction),
        reach_mpc=injections.reach_mpc,
        reach_range_mpc=injections.reach_range_mpc,
        population_volume_gpc3=injections.population_volume_gpc3,
        data_reuse=background.data_reuse,
        num_injections=n,
        num_bootstrap=int(num_bootstrap),
        sigmas=sigmas,
    )
