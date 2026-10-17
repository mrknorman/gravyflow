"""
Sensitive volume and sensitive distance against false-alarm rate.

The sensitive volume at a threshold is the population-averaged volume in which
a source is found (``score >= threshold``, as everywhere in `metrics`). It is
estimated by importance sampling over injections: each injection ``i`` carries
the volume ``q_i`` it stands for when it is found, and

    V(threshold) = sum_i q_i [score_i >= threshold] / N.

For injections drawn uniformly in volume out to ``d_max``, ``q_i`` is the
reference volume ``4/3 pi d_max^3`` times the injection's population weight
(1, or a chirp-distance weight), and V is the familiar found fraction of the
reference volume (`InjectionVolumes.uniform_in_volume`). The validator's own
injections are scaled to an SNR drawn uniformly over a range instead; each
then has an equivalent distance, and ``q_i`` is the ratio of the target
density, uniform in volume, to the density the SNR draw gave that distance
(`InjectionVolumes.from_snr_scaled`).

The sensitive distance is the radius of a sphere of that volume,
``R = (3 V / 4 pi)^(1/3)``. At a target FAR, the threshold is the one the noise
background gives that FAR (`Background.threshold_for_far`). Three intervals are
reported, each two-sided and central at ``confidence``:

- **injections**: the finite Monte Carlo, a weighted-binomial interval on V at
  the threshold (`InjectionVolumes.interval`);
- **background**: the finite livetime, the V at the two thresholds between
  which the FAR's Poisson interval puts the true threshold
  (`Background.far_band_thresholds`);
- **joint**: a bootstrap over injections and background blocks
  (`Background.resample`), the threshold found again in each replicate.

Units: distances in Mpc, volumes in Gpc^3 (Euclidean: no cosmology), FAR in Hz;
`SECONDS_PER_YEAR` and `SECONDS_PER_MONTH` convert it for display.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np

from .metrics import DEFAULT_CONFIDENCE, Background, binomial_interval

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


@dataclass(frozen=True)
class InjectionVolumes:
    """
    Injections as an importance-sampled volume integral: injection ``i`` scored
    ``scores[i]`` and stands for ``volumes_gpc3[i]`` when it is found, so that
    ``V(t) = sum_i volumes_gpc3[i] [scores[i] >= t] / N``.

    ``max_distance_mpc`` is the reach of the target population, uniform in
    volume out to it; nothing beyond it counts.
    """
    scores: np.ndarray
    volumes_gpc3: np.ndarray
    max_distance_mpc: float

    def __post_init__(self):
        scores = np.asarray(self.scores, dtype=np.float64).reshape(-1)
        volumes = np.asarray(self.volumes_gpc3, dtype=np.float64).reshape(-1)
        if scores.size == 0 or scores.shape != volumes.shape:
            raise ValueError(f"{volumes.size} volumes for {scores.size} injection scores: need one per injection.")
        if not (np.all(np.isfinite(scores)) and np.all(np.isfinite(volumes)) and np.all(volumes >= 0)):
            raise ValueError("Injection scores must be finite, and their volumes finite and >= 0.")
        if not self.max_distance_mpc > 0:
            raise ValueError(f"max_distance_mpc must be > 0, got {self.max_distance_mpc}.")
        order = np.argsort(scores, kind="stable")
        object.__setattr__(self, "scores", scores)
        object.__setattr__(self, "volumes_gpc3", volumes)
        # Scores ascending, for the injections found at a threshold: a suffix.
        object.__setattr__(self, "_sorted_scores", scores[order])
        object.__setattr__(self, "_order", order)

    # ------------------------------------------------------------ builders

    @classmethod
    def uniform_in_volume(cls, scores, max_distance_mpc: float, weights=None) -> "InjectionVolumes":
        """
        Injections drawn uniformly in volume out to ``max_distance_mpc``, each
        standing for the reference volume times its population weight (1 by
        default; a power of the chirp mass for a draw uniform in chirp distance).
        """
        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        weights = np.ones_like(scores) if weights is None else np.asarray(weights, dtype=np.float64)
        return cls(scores, sphere_volume_gpc3(max_distance_mpc) * weights.reshape(-1), float(max_distance_mpc))

    @classmethod
    def from_snr_scaled(
        cls,
        scores,
        snrs,
        distances_mpc,
        snr_range: Tuple[float, float],
        top_snr_width: float,
        horizon_snr: float,
    ) -> "InjectionVolumes":
        """
        Injections scaled to an optimal network SNR drawn uniformly over
        ``snr_range = (a, b)``, each labelled with the distance its scaled
        amplitude implies (``distances_mpc``).

        Each injection's source has ``D = snr x distance``: its SNR at 1 Mpc,
        times 1 Mpc, fixed by its masses, sky, orientation and noise, not by the
        draw. Given the source, the draw puts it at ``d = D / snr`` with density
        ``D / ((b - a) d^2)``; the target is uniform in volume, ``4 pi d^2``
        per unit reference volume, out to ``d_max``. So a found injection stands
        for ``4 pi (b - a) d^4 / D`` (zero beyond ``d_max``).

        Sources nearer than the draw reaches, at SNR above ``b``, are drawn
        nowhere. They are carried by the injections of the top SNR bin,
        ``[b - top_snr_width, b]``: each also stands for its own source at every
        SNR above ``b``, the sphere ``4/3 pi min(D / b, d_max)^3``, over the
        probability ``top_snr_width / (b - a)`` of landing in the bin. That
        assumes a source found (missed) at the top of the range is found
        (missed) when nearer, which holds when the efficiency there is
        saturated. Sources at SNR below ``a`` count as missed.

        ``d_max`` is the distance at which the loudest source drawn (the largest
        ``D``) has SNR ``horizon_snr``: beyond it every source is quieter.
        Injections with no SNR or distance (0, NaN) are sources at no finite
        distance, and count for nothing.
        """
        a, b = (float(x) for x in snr_range)
        if not b > a >= 0:
            raise ValueError(f"snr_range must be increasing and non-negative, got {snr_range}.")
        if not 0 < top_snr_width <= b - a:
            raise ValueError(f"top_snr_width must be in (0, {b - a}], got {top_snr_width}.")
        if not horizon_snr > 0:
            raise ValueError(f"horizon_snr must be > 0, got {horizon_snr}.")
        snrs = np.asarray(snrs, dtype=np.float64).reshape(-1)
        distances = np.asarray(distances_mpc, dtype=np.float64).reshape(-1)
        if snrs.shape != distances.shape:
            raise ValueError(f"{distances.size} distances for {snrs.size} SNRs.")
        valid = np.isfinite(snrs) & np.isfinite(distances) & (snrs > 0) & (distances > 0)
        if not np.any(valid):
            raise ValueError("No injection has a positive SNR and distance: there is no volume to integrate.")
        d = np.where(valid, distances, 0.0)
        D = np.where(valid, snrs, 0.0) * d
        max_distance = float(D.max() / horizon_snr)
        with np.errstate(divide="ignore", invalid="ignore"):
            importance = np.where(valid & (d <= max_distance), 4.0 * np.pi * (b - a) * d**4 / D, 0.0)
        top = valid & (snrs >= b - top_snr_width)
        nearer = 4.0 / 3.0 * np.pi * np.minimum(D / b, max_distance) ** 3 * (b - a) / top_snr_width
        volumes = (importance + np.where(top, nearer, 0.0)) * GPC3_PER_MPC3
        return cls(scores, volumes, max_distance)

    # ------------------------------------------------------------- volumes

    @property
    def num_injections(self) -> int:
        return self.scores.size

    @property
    def reference_volume_gpc3(self) -> float:
        """The target population's volume, ``4/3 pi max_distance^3``."""
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
        q = self.volumes_gpc3[self._order]
        if multiplicity is not None:
            q = q * np.asarray(multiplicity, dtype=np.float64)[self._order]
        return self._found_sum(thresholds, q) / self.num_injections

    def _moments(self, thresholds) -> Tuple[np.ndarray, np.ndarray]:
        q = self.volumes_gpc3[self._order]
        return self._found_sum(thresholds, q), self._found_sum(thresholds, q**2)

    def standard_error(self, thresholds) -> np.ndarray:
        """
        The Monte Carlo standard error of V: ``sqrt(sum q^2 f - (sum q f)^2 / N) / N``.
        Unweighted (``q`` the reference volume), it is ``V_ref sqrt(N p (1 - p)) / N``.
        """
        s1, s2 = self._moments(thresholds)
        n = self.num_injections
        return np.sqrt(np.maximum(s2 - s1**2 / n, 0.0)) / n

    def interval(self, thresholds, confidence: float = DEFAULT_CONFIDENCE,
                 method: str = "clopper-pearson") -> Tuple[np.ndarray, np.ndarray]:
        """
        A weighted-binomial interval on V at each threshold: V as a binomial count
        of ``k = (sum q f)^2 / sum q^2 f`` effective injections out of N, each of
        volume ``sum q^2 f / sum q f``, which matches V and its variance and is the
        exact binomial interval (`binomial_interval`, by ``method``) when every
        volume is equal. With nothing found, the volume of a found injection is
        the largest any injection stands for.
        """
        s1, s2 = self._moments(thresholds)
        found = np.nan_to_num(s1) > 0
        with np.errstate(divide="ignore", invalid="ignore"):
            each = np.where(found, s2 / s1, self.volumes_gpc3.max())
            effective = np.where(found, s1**2 / s2, 0.0)
        lower, upper = binomial_interval(effective, self.num_injections, confidence, method)
        unresolved = np.isnan(s1)
        return np.where(unresolved, np.nan, each * lower), np.where(unresolved, np.nan, each * upper)


# ------------------------------------------------------------- the curve

@dataclass(frozen=True)
class SensitiveVolumeCurve:
    """
    Sensitive volume (Gpc^3) against target FAR (Hz), with its three intervals.

    At each target: the background's threshold for it (NaN, and every volume
    NaN, when the background cannot resolve it), the FAR that threshold
    achieves with its Poisson interval, V, and the bounds of each interval;
    ``injection_error_gpc3`` is the Monte Carlo standard error.
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


def sensitive_volume_curve(
    injections: InjectionVolumes,
    background: Background,
    fars_hz: Sequence[float],
    rng: np.random.Generator,
    num_bootstrap: int = 1000,
    confidence: float = DEFAULT_CONFIDENCE,
    method: str = "clopper-pearson",
) -> SensitiveVolumeCurve:
    """
    V against FAR at each of ``fars_hz``, with its injection, background and
    joint intervals (module docstring). The joint interval's bootstrap draws
    ``num_bootstrap`` replicates from ``rng``, each a background resampled by
    block and the injections resampled with replacement, and takes the central
    ``confidence`` range of the replicates' V at the threshold each one's own
    background gives the target.
    """
    if num_bootstrap < 1:
        raise ValueError(f"num_bootstrap must be >= 1, got {num_bootstrap}.")
    fars = np.asarray(fars_hz, dtype=np.float64).reshape(-1)
    targets = [background.threshold_for_far(float(far), confidence) for far in fars]
    threshold = np.array([t.threshold for t in targets])
    resolved = np.array([t.resolved for t in targets])
    bands = np.array([background.far_band_thresholds(float(far), confidence) if ok else (np.nan, np.nan)
                      for far, ok in zip(fars, resolved)]).reshape(-1, 2)

    n = injections.num_injections
    replicates = np.empty((num_bootstrap, fars.size))
    for b in range(num_bootstrap):
        replica = background.resample(rng)
        multiplicity = np.bincount(rng.integers(0, n, n), minlength=n)
        thresholds = [replica.threshold_for_far(float(far), confidence).threshold if ok else np.nan
                      for far, ok in zip(fars, resolved)]
        replicates[b] = injections.volume(thresholds, multiplicity)
    alpha = 1.0 - confidence
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # an all-NaN column: an unresolved target
        joint = np.nanquantile(replicates, [alpha / 2, 1 - alpha / 2], axis=0)
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
        # The higher threshold finds less.
        background_lower_gpc3=injections.volume(bands[:, 0]),
        background_upper_gpc3=injections.volume(bands[:, 1]),
        joint_lower_gpc3=joint[0],
        joint_upper_gpc3=joint[1],
        max_distance_mpc=injections.max_distance_mpc,
        num_injections=n,
        num_bootstrap=int(num_bootstrap),
        confidence=float(confidence),
    )
