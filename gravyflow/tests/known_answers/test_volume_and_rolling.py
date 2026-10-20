"""
Known answers for the sensitive volume within the per-mass reach and the rolling
efficiency curve with its sigmoid fit (injections drawn out to each mass's own reach,
a rolling window in SNR with pointwise bands, a fitted logistic's 50% and 90% points,
intervals at 1 and 2 sigma), written before that API landed.

The API is reached only through `api` and the three adapters below it: they are
the only lines changed to enable the module (the truths and the tolerances are
untouched). The landed API:

- ``volume.InjectionVolumes(scores, reach_mpc, distances, weights)`` and
  ``volume.sensitive_volume_curve(injections, background, fars_hz, rng, *,
  num_bootstrap, sigmas)``, its intervals at each level in ``sigmas`` (``(levels,
  targets)``). The per-mass reach is carried as weights: each injection is drawn
  out to its own mass's reach ``R_m`` and weighted ``(R_m / R_ref)^3`` against the
  largest reach ``R_ref``, which keeps the volume exact.
- ``metrics.rolling_efficiency_curve(snrs, scores, threshold, *, half_width,
  sigmas)`` with ``snr``, ``efficiency``, ``lower``, ``upper`` (pointwise, ``(levels,
  points)``), and ``metrics.fit_efficiency_sigmoid(snrs, scores, threshold, *,
  chance, snr_range, rng, num_bootstrap, sigmas)`` with ``snr_50``, ``snr_90`` and
  their bootstrap bounds ``lower``, ``upper`` (``(levels, 2)``).

The adapters ask for the one level a test names, as sigma.

The sigmoid fit's point test uses 300,000 injections (50,000 when written): its 90%
point's error has an sd of 0.027 at 50,000 (40 seeds, no
bias), so the tolerance of 0.05 was 1.8 sd; at 300,000 it is at least 4 sd.
"""

import dataclasses
import types

import numpy as np
import pytest

from gravyflow.src.validate import metrics
from gravyflow.tests.known_answers import truth

Z = 4.0


def api():
    """The modules the API lives in, imported only when a test runs."""
    from gravyflow.src.validate import metrics as metrics_module, volume
    return volume, metrics_module


def sigma_of(confidence):
    """The level in sigma of a normal law whose central probability is ``confidence``."""
    from scipy import special
    return float(np.sqrt(2.0) * special.erfinv(confidence))


def volume_curve(scores, distances, snrs, reaches, background, fars, rng, num_bootstrap, confidence):
    volume, _ = api()
    reference = float(np.max(reaches))
    injections = volume.InjectionVolumes(scores, reference, distances, (reaches / reference) ** 3)
    curve = volume.sensitive_volume_curve(injections, background, fars, rng, num_bootstrap=num_bootstrap,
                                          sigmas=(sigma_of(confidence),))
    # The one level asked for: each interval's bounds at each target.
    return dataclasses.replace(curve, joint_lower_gpc3=curve.joint_lower_gpc3[0],
                               joint_upper_gpc3=curve.joint_upper_gpc3[0])


def rolling_curve(snrs, scores, threshold, half_width, confidence):
    _, m = api()
    curve = m.rolling_efficiency_curve(snrs, scores, threshold, half_width=half_width,
                                       sigmas=(sigma_of(confidence),))
    return dataclasses.replace(curve, lower=curve.lower[0], upper=curve.upper[0])


def sigmoid_fit(snrs, scores, threshold, rng, num_bootstrap, confidence):
    _, m = api()
    snrs = np.asarray(snrs)
    fit = m.fit_efficiency_sigmoid(snrs, scores, threshold, chance=None,
                                   snr_range=(float(np.floor(snrs.min())), float(np.ceil(snrs.max()))), rng=rng,
                                   num_bootstrap=num_bootstrap, sigmas=(sigma_of(confidence),))
    return types.SimpleNamespace(snr50=fit.snr_50, snr90=fit.snr_90,
                                 snr50_lower=fit.lower[0, 0], snr50_upper=fit.upper[0, 0],
                                 snr90_lower=fit.lower[0, 1], snr90_upper=fit.upper[0, 1])


# --------------------------------------------------------------- the volume

DETECTOR = truth.GaussianBank.single()
POPULATION = truth.Population(amplitudes=(400.0, 1000.0, 2500.0), mass_fractions=(0.5, 0.3, 0.2))


def campaign(rng, num):
    mass, distance, g, snr = POPULATION.draw(rng, num)
    return DETECTOR.signal(rng, snr), distance, snr, POPULATION.reaches_mpc[mass]


def test_the_volume_within_the_per_mass_reach_is_the_paper_s():
    # A long background puts the threshold close to the true one; the truth is read
    # at the estimator's own threshold, so the comparison is the injections' alone.
    rng = np.random.default_rng(41)
    n_bg, far = 2_000_000, 1e-3
    background = metrics.Background.independent_windows(DETECTOR.noise(rng, n_bg), 1.0)
    scores, distances, snrs, reaches = campaign(rng, 200_000)
    curve = volume_curve(scores, distances, snrs, reaches, background, [far], rng, 1, truth.ONE_SIGMA)
    t_hat = float(curve.threshold[0])
    expected = truth.volume_within_reach_gpc3(DETECTOR, t_hat, POPULATION)
    assert abs(curve.volume_within_reach_gpc3[0] - expected) <= Z * curve.injection_error_gpc3[0]
    # And the estimator's threshold is the paper's for that FAR, to the background's precision.
    t_true = truth.threshold_for_far(DETECTOR, far, 1.0)
    assert abs(DETECTOR.noise_sf(t_hat) - DETECTOR.noise_sf(t_true)) <= Z * np.sqrt(far / n_bg)


@pytest.mark.parametrize("confidence", truth.LEVELS, ids=truth.LEVEL_IDS)
def test_the_volume_s_joint_interval_covers_the_true_volume(confidence):
    # The estimand at a target FAR: the paper's volume at the paper's threshold for it.
    rng = np.random.default_rng(42)
    replicas, n_bg, n_inj, far = 100, 20_000, 4_000, 1e-3
    expected = truth.volume_within_reach_gpc3(DETECTOR, truth.threshold_for_far(DETECTOR, far, 1.0), POPULATION)
    covered = []
    for _ in range(replicas):
        background = metrics.Background.independent_windows(DETECTOR.noise(rng, n_bg), 1.0)
        scores, distances, snrs, reaches = campaign(rng, n_inj)
        c = volume_curve(scores, distances, snrs, reaches, background, [far], rng, 200, confidence)
        covered.append(c.joint_lower_gpc3[0] <= expected <= c.joint_upper_gpc3[0])
    assert np.mean(covered) >= confidence - truth.coverage_tolerance(confidence, replicas)


# ----------------------------------------------------- the rolling efficiency

LOGISTIC = truth.LogisticDetector(0.8)
RANGE = (2.0, 20.0)
HALF_WIDTH = 0.5
THRESHOLD = 8.0


def snr_set(rng, num):
    """The SNR set: injections even over the range."""
    snr = rng.uniform(*RANGE, num)
    return snr, LOGISTIC.signal(rng, snr)


def test_the_rolling_efficiency_is_the_window_average_of_the_paper_s():
    rng = np.random.default_rng(43)
    snr, scores = snr_set(rng, 100_000)
    curve = rolling_curve(snr, scores, THRESHOLD, HALF_WIDTH, truth.ONE_SIGMA)
    expected = truth.rolling_efficiency(LOGISTIC, THRESHOLD, curve.snr, HALF_WIDTH, RANGE)
    # Each point holds the injections within its window: about num * width / range.
    count = 100_000 * (2 * HALF_WIDTH) / (RANGE[1] - RANGE[0])
    se = np.sqrt(np.maximum(expected * (1 - expected), 1e-6) / count)
    np.testing.assert_array_less(np.abs(np.asarray(curve.efficiency) - expected), Z * se + 1e-9)


@pytest.mark.parametrize("confidence", truth.LEVELS, ids=truth.LEVEL_IDS)
def test_the_rolling_bands_cover_pointwise_at_their_level(confidence):
    # Pointwise: at each SNR, across replicas, the band covers the window-averaged truth.
    rng = np.random.default_rng(44)
    replicas = 200
    grid = np.linspace(THRESHOLD - 3, THRESHOLD + 3, 13)
    expected = truth.rolling_efficiency(LOGISTIC, THRESHOLD, grid, HALF_WIDTH, RANGE)
    covered = []
    for _ in range(replicas):
        snr, scores = snr_set(rng, 5_000)
        curve = rolling_curve(snr, scores, THRESHOLD, HALF_WIDTH, confidence)
        x = np.asarray(curve.snr)
        lower, upper = np.interp(grid, x, curve.lower), np.interp(grid, x, curve.upper)
        covered.append((lower <= expected) & (expected <= upper))
    per_point = np.array(covered).mean(axis=0)
    assert per_point.min() >= confidence - truth.coverage_tolerance(confidence, replicas)


def test_the_sigmoid_fit_finds_the_paper_s_50_and_90_percent_points():
    # The logistic detector's efficiency is exactly a sigmoid: 50% at t, 90% at t + s ln 9.
    rng = np.random.default_rng(45)
    snr, scores = snr_set(rng, 300_000)
    fit = sigmoid_fit(snr, scores, THRESHOLD, rng, 200, truth.ONE_SIGMA)
    for point, q in ((fit.snr50, 0.5), (fit.snr90, 0.9)):
        expected = truth.snr_at_efficiency(LOGISTIC, THRESHOLD, q)
        assert abs(point - expected) <= 0.05, (q, point, expected)


@pytest.mark.parametrize("confidence", truth.LEVELS, ids=truth.LEVEL_IDS)
def test_the_sigmoid_fit_s_bootstrap_intervals_cover_the_paper_s_points(confidence):
    rng = np.random.default_rng(46)
    replicas = 100
    hits50, hits90 = [], []
    for _ in range(replicas):
        snr, scores = snr_set(rng, 3_000)
        fit = sigmoid_fit(snr, scores, THRESHOLD, rng, 200, confidence)
        hits50.append(fit.snr50_lower <= truth.snr_at_efficiency(LOGISTIC, THRESHOLD, 0.5) <= fit.snr50_upper)
        hits90.append(fit.snr90_lower <= truth.snr_at_efficiency(LOGISTIC, THRESHOLD, 0.9) <= fit.snr90_upper)
    tolerance = truth.coverage_tolerance(confidence, replicas)
    assert np.mean(hits50) >= confidence - tolerance
    assert np.mean(hits90) >= confidence - tolerance
