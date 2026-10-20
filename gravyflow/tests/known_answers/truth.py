"""
Synthetic detectors and populations whose detection results are known exactly.

Each detector draws scores (``noise``, ``signal``) and states, on paper, what
any correct estimator must find from them: the chance that a noise window
passes a threshold (so the false-alarm rate of independent windows), the
efficiency at an SNR, the 50% and 90% SNR points, the ROC and its area. The
populations state the sensitive volume within a reach, and the rolling
efficiency curve's expectation. Nothing here calls the validator: it is the
paper the validator is checked against, and `test_truth` checks the paper
against brute force.

The detectors:

- `GaussianBank`: score ``max_j (snr m_j + n_j)`` over templates of match
  ``m_j``, with ``n_j`` independent unit Gaussians. One template of match 1 is
  the single-trial detector (score = SNR + N(0, 1), efficiency
  ``Phi(snr - t)``); one matched template among ``N`` is the max of ``N``
  trials; a spread of matches is a bank.
- `LogisticDetector`: score ``snr + s L`` with ``L`` standard logistic noise. Its
  efficiency is exactly a logistic sigmoid in SNR, the shape the validator fits,
  so the fitted 50% and 90% points have exact answers: ``t`` and ``t + s ln 9``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy import integrate, optimize, special, stats

#: The validator's two interval levels: 1 sigma and 2 sigma, two-sided.
ONE_SIGMA = float(special.erf(1.0 / np.sqrt(2.0)))   # 0.682689...
TWO_SIGMA = float(special.erf(2.0 / np.sqrt(2.0)))   # 0.954499...
LEVELS = (ONE_SIGMA, TWO_SIGMA)
LEVEL_IDS = ("1sigma", "2sigma")

GPC3_PER_MPC3 = 1e-9


def sphere_volume_gpc3(radius_mpc) -> np.ndarray:
    return 4.0 / 3.0 * np.pi * np.asarray(radius_mpc, dtype=np.float64) ** 3 * GPC3_PER_MPC3


# ---------------------------------------------------------------- detectors

@dataclass(frozen=True)
class GaussianBank:
    """Score ``max_j (snr m_j + n_j)``, ``n_j ~ N(0, 1)`` independent (module docstring)."""
    matches: Tuple[float, ...] = (1.0,)

    @classmethod
    def single(cls) -> "GaussianBank":
        return cls((1.0,))

    @classmethod
    def max_of(cls, trials: int) -> "GaussianBank":
        """The signal in one of ``trials`` independent trials; the score is their maximum."""
        return cls((1.0,) + (0.0,) * (trials - 1))

    @property
    def _m(self) -> np.ndarray:
        return np.asarray(self.matches, dtype=np.float64)

    # Exact distribution of the score.
    def signal_cdf(self, x, snr) -> np.ndarray:
        x, snr = np.broadcast_arrays(np.asarray(x, np.float64), np.asarray(snr, np.float64))
        return np.prod(stats.norm.cdf(x[..., None] - snr[..., None] * self._m), axis=-1)

    def signal_pdf(self, x, snr) -> np.ndarray:
        x, snr = np.broadcast_arrays(np.asarray(x, np.float64), np.asarray(snr, np.float64))
        z = x[..., None] - snr[..., None] * self._m
        cdf, pdf = stats.norm.cdf(z), stats.norm.pdf(z)
        total = np.zeros(x.shape)
        for j in range(self._m.size):
            total += pdf[..., j] * np.prod(np.delete(cdf, j, axis=-1), axis=-1)
        return total

    def signal_sf(self, threshold, snr) -> np.ndarray:
        """The efficiency: P(score >= threshold) at ``snr``. Computed as 1 - prod Phi, stable in the tail."""
        t, snr = np.broadcast_arrays(np.asarray(threshold, np.float64), np.asarray(snr, np.float64))
        log_cdf = np.sum(stats.norm.logcdf(t[..., None] - snr[..., None] * self._m), axis=-1)
        return -np.expm1(log_cdf)

    def noise_sf(self, threshold) -> np.ndarray:
        """The chance that one noise window passes ``threshold``."""
        return self.signal_sf(threshold, 0.0)

    def threshold_for_noise_sf(self, p: float) -> float:
        """The threshold a noise window passes with chance ``p``: exact inverse of `noise_sf`."""
        n = self._m.size
        # 1 - Phi(t)^n = p  <=>  Phi(t) = (1 - p)^(1/n).
        return float(stats.norm.ppf(np.exp(np.log1p(-p) / n)))

    def noise(self, rng: np.random.Generator, size: int) -> np.ndarray:
        return self.signal(rng, np.zeros(size))

    def signal(self, rng: np.random.Generator, snr) -> np.ndarray:
        snr = np.asarray(snr, dtype=np.float64)
        return np.max(snr[..., None] * self._m + rng.normal(size=snr.shape + (self._m.size,)), axis=-1)


@dataclass(frozen=True)
class LogisticDetector:
    """Score ``snr + scale L``, ``L`` standard logistic: efficiency ``expit((snr - t) / scale)``."""
    scale: float = 1.0

    def signal_cdf(self, x, snr) -> np.ndarray:
        return special.expit((np.asarray(x, np.float64) - snr) / self.scale)

    def signal_pdf(self, x, snr) -> np.ndarray:
        e = special.expit((np.asarray(x, np.float64) - snr) / self.scale)
        return e * (1.0 - e) / self.scale

    def signal_sf(self, threshold, snr) -> np.ndarray:
        return special.expit((np.asarray(snr, np.float64) - threshold) / self.scale)

    def noise_sf(self, threshold) -> np.ndarray:
        return self.signal_sf(threshold, 0.0)

    def threshold_for_noise_sf(self, p: float) -> float:
        return float(-self.scale * special.logit(p))

    def noise(self, rng: np.random.Generator, size: int) -> np.ndarray:
        return self.signal(rng, np.zeros(size))

    def signal(self, rng: np.random.Generator, snr) -> np.ndarray:
        snr = np.asarray(snr, dtype=np.float64)
        return snr + self.scale * rng.logistic(size=snr.shape)

    def snr_at_efficiency(self, threshold: float, efficiency: float) -> float:
        """Closed form: ``t + scale logit(efficiency)``; 50% at ``t``, 90% at ``t + scale ln 9``."""
        return float(threshold + self.scale * special.logit(efficiency))


# --------------------------------------------------------- derived truths

def far_hz(detector, threshold: float, window_seconds: float) -> float:
    """FAR of independent windows of ``window_seconds``: one trigger per passing window."""
    return float(detector.noise_sf(threshold)) / window_seconds


def threshold_for_far(detector, far: float, window_seconds: float) -> float:
    """The exact threshold whose FAR (independent windows) is ``far``."""
    return detector.threshold_for_noise_sf(far * window_seconds)


def clustered_trigger_rate_hz(detector, threshold: float, num_windows: int, cluster_windows: int,
                              stride_seconds: float, window_seconds: float) -> float:
    """
    Expected triggers per second of one stream of ``num_windows`` independent
    windows tiled ``stride_seconds`` apart, where a passing window within
    ``cluster_windows`` windows of an earlier passing one joins its trigger:
    window ``i`` opens a trigger when it passes and its ``min(i, k)``
    predecessors do not, so the count's mean is ``sum_i p (1 - p)^min(i, k)``.
    """
    p = float(detector.noise_sf(threshold))
    i = np.arange(num_windows)
    expected = np.sum(p * (1.0 - p) ** np.minimum(i, cluster_windows))
    livetime = (num_windows - 1) * stride_seconds + window_seconds
    return float(expected / livetime)


def snr_at_efficiency(detector, threshold: float, efficiency: float) -> float:
    """The SNR at which ``detector`` finds a fraction ``efficiency`` of signals."""
    if hasattr(detector, "snr_at_efficiency"):
        return detector.snr_at_efficiency(threshold, efficiency)
    return float(optimize.brentq(lambda s: detector.signal_sf(threshold, s) - efficiency, -50.0, 200.0,
                                 xtol=1e-12))


def auc(detector, snr: float) -> float:
    """The ROC area: P(signal score > noise score) = int F_noise(x) f_signal(x) dx."""
    value, _ = integrate.quad(lambda x: detector.signal_cdf(x, 0.0) * detector.signal_pdf(x, snr),
                              -np.inf, np.inf, epsabs=1e-12, limit=200)
    return float(value)


def roc_tpr(detector, fpr, snr: float) -> np.ndarray:
    """The true-positive rate at each false-positive rate ``fpr`` (the exact ROC)."""
    thresholds = np.array([detector.threshold_for_noise_sf(float(f)) for f in np.atleast_1d(fpr)])
    return detector.signal_sf(thresholds, snr)


def binned_efficiency(detector, threshold: float, low: float, high: float) -> float:
    """Expected efficiency of a bin [low, high) of injections uniform in SNR."""
    value, _ = integrate.quad(lambda s: detector.signal_sf(threshold, s), low, high, epsabs=1e-12)
    return float(value / (high - low))


def rolling_efficiency(detector, threshold: float, snr, half_width: float,
                       snr_range: Tuple[float, float]) -> np.ndarray:
    """
    The expectation of a rolling efficiency at ``snr`` (window ``±half_width``)
    of injections uniform in SNR over ``snr_range``: the efficiency averaged over
    the window, cut to the range.
    """
    out = []
    for x in np.atleast_1d(snr):
        lo, hi = max(x - half_width, snr_range[0]), min(x + half_width, snr_range[1])
        out.append(binned_efficiency(detector, threshold, lo, hi))
    return np.asarray(out)


# --------------------------------------------------------------- volumes

@dataclass(frozen=True)
class Population:
    """
    Sources of masses ``masses`` (labels) drawn with chances ``mass_fractions``;
    a source of mass ``m`` at distance ``d`` (Mpc) with orientation factor ``g``
    (uniform on [0, 1]; 1 is optimal sky and orientation) has optimal SNR
    ``amplitudes[m] g / d``. Its reach is where the best-case source (``g = 1``)
    has SNR ``reach_snr``: ``R_m = amplitudes[m] / reach_snr`` (the validator's
    per-mass reach; SNR about 2).
    """
    amplitudes: Tuple[float, ...]
    mass_fractions: Tuple[float, ...]
    reach_snr: float = 2.0

    @property
    def reaches_mpc(self) -> np.ndarray:
        return np.asarray(self.amplitudes, np.float64) / self.reach_snr

    def draw(self, rng: np.random.Generator, num: int, reach_mpc=None):
        """
        ``num`` sources: mass index, distance uniform in volume out to each one's
        reach (or one common ``reach_mpc``), orientation factor and optimal SNR.
        """
        mass = rng.choice(len(self.amplitudes), size=num, p=np.asarray(self.mass_fractions))
        reach = self.reaches_mpc[mass] if reach_mpc is None else np.full(num, float(reach_mpc))
        distance = reach * np.cbrt(1.0 - rng.random(num))
        g = rng.random(num)
        snr = np.asarray(self.amplitudes, np.float64)[mass] * g / distance
        return mass, distance, g, snr


def _found_fraction_within(detector, threshold: float, snr_at_reach: float) -> float:
    """
    The found fraction of sources uniform in volume within a sphere, with ``g``
    uniform on [0, 1] and SNR ``snr_at_reach g / u`` at fractional radius ``u``:
    ``int_0^1 int_0^1 3 u^2 P(found | snr_at_reach g / u) du dg``.
    """
    def inner(g):
        # The efficiency turns fastest where the SNR crosses the threshold: a breakpoint for the quadrature.
        crossing = snr_at_reach * g / threshold if threshold > 0 else np.inf
        points = [crossing] if 0.0 < crossing < 1.0 else None
        value, _ = integrate.quad(lambda u: 3 * u * u * detector.signal_sf(threshold, snr_at_reach * g / u),
                                  0.0, 1.0, epsabs=1e-12, limit=200, points=points)
        return value
    value, _ = integrate.quad(inner, 0.0, 1.0, epsabs=1e-11, limit=200)
    return float(value)


def volume_within_reach_gpc3(detector, threshold: float, population: Population, reach_mpc=None) -> float:
    """
    The exact sensitive volume (Gpc^3) within the reach: per mass, the sphere of
    its reach (or the common ``reach_mpc``) times the fraction found in it,
    averaged over the masses: ``sum_m pi_m 4/3 pi R_m^3 f_m``.
    """
    total = 0.0
    for amplitude, fraction, reach in zip(population.amplitudes, population.mass_fractions,
                                          population.reaches_mpc):
        r = reach if reach_mpc is None else float(reach_mpc)
        total += fraction * sphere_volume_gpc3(r) * _found_fraction_within(detector, threshold, amplitude / r)
    return float(total)


def step_volume_gpc3(population: Population, snr_threshold: float) -> float:
    """
    Closed form, for a noiseless detector (found iff SNR >= ``snr_threshold``)
    with ``snr_threshold`` at or above the reach SNR: a source is found out to
    ``A g / rho``, inside every reach, so ``V = sum_m pi_m 4/3 pi (A_m / rho)^3 E[g^3]``
    with ``E[g^3] = 1/4``.
    """
    a = np.asarray(population.amplitudes, np.float64)
    return float(np.sum(np.asarray(population.mass_fractions) * sphere_volume_gpc3(a / snr_threshold)) / 4.0)


@dataclass(frozen=True)
class StepDetector:
    """Noiseless: score = SNR exactly. Noise scores are 0 (for the volume closed form only)."""

    def signal_sf(self, threshold, snr) -> np.ndarray:
        return (np.asarray(snr, np.float64) >= threshold).astype(np.float64)


def coverage_tolerance(nominal: float, replicas: int, z: float = 4.0) -> float:
    """``z`` Monte Carlo standard errors of a coverage estimate from ``replicas`` replicas."""
    return z * float(np.sqrt(nominal * (1.0 - nominal) / replicas))

