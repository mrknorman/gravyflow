"""
Sensitive volume against independent references: the closed-form volume of a
threshold detector averaged over sky and orientation, the coverage of each
interval over many replicas, and the binomial error of an unweighted draw.
"""

import numpy as np
import pytest
from scipy import integrate, stats

from gravyflow.src.validate.metrics import Background, poisson_interval
from gravyflow.src.validate.volume import (
    GPC3_PER_MPC3,
    InjectionVolumes,
    sensitive_distance,
    sensitive_volume_curve,
    sphere_volume_gpc3,
)

SNR_RANGE = (0.0, 15.0)
TOP_WIDTH = 1.0
HORIZON_SNR = 4.0
# The horizon (Mpc): an overhead, face-on source's distance at SNR 1.
HORIZON_MPC = 4000.0


# ------------------------------------------------------------------ helpers

def antenna_response(cos_theta, phi, psi, cos_iota):
    """
    One interferometer's response to a source, relative to an overhead face-on one
    (Finn and Chernoff 1993): its SNR at a given distance is this times the
    optimal source's.
    """
    common = 0.5 * (1 + cos_theta**2)
    f_plus = common * np.cos(2 * phi) * np.cos(2 * psi) - cos_theta * np.sin(2 * phi) * np.sin(2 * psi)
    f_cross = common * np.cos(2 * phi) * np.sin(2 * psi) + cos_theta * np.sin(2 * phi) * np.cos(2 * psi)
    return np.sqrt(f_plus**2 * (1 + cos_iota**2) ** 2 / 4 + f_cross**2 * cos_iota**2)


def mean_cubed_response(n=32) -> float:
    """The sky- and orientation-average of the response cubed, by quadrature (no sampling)."""
    x, w = np.polynomial.legendre.leggauss(n)
    angles = np.pi * np.arange(n) / n  # the response has period pi in phi and psi
    ct, phi, psi, ci = np.meshgrid(x, angles, angles, x, indexing="ij")
    weights = w[:, None, None, None] * w[None, None, None, :] / (4.0 * n * n)
    return float(np.sum(weights * antenna_response(ct, phi, psi, ci) ** 3))


def isotropic_sources(rng, n):
    """Each source's SNR at 1 Mpc (times 1 Mpc): the horizon times its response."""
    return HORIZON_MPC * antenna_response(rng.uniform(-1, 1, n), rng.uniform(0, np.pi, n),
                                          rng.uniform(0, np.pi, n), rng.uniform(-1, 1, n))


def snr_scaled(rng, D, scores_of_snr):
    """The validator's draw: each source scaled to an SNR uniform over the range."""
    snrs = rng.uniform(*SNR_RANGE, D.size)
    return InjectionVolumes.from_snr_scaled(scores_of_snr(snrs), snrs, D / snrs, SNR_RANGE, TOP_WIDTH,
                                            HORIZON_SNR)


def threshold_detector_volume(threshold_snr: float) -> float:
    """
    The closed form: a source is found exactly when its SNR is at least ``threshold_snr``,
    so it is found within ``D / threshold_snr``, and V = 4/3 pi E[D^3] / threshold^3.
    """
    return float(sphere_volume_gpc3(HORIZON_MPC / threshold_snr)) * mean_cubed_response()


def coverage_tolerance(replicas: int, level: float) -> float:
    """Three binomial standard deviations of a coverage measured over ``replicas``."""
    return 3.0 * np.sqrt(level * (1 - level) / replicas)


# ------------------------------------------------------------------ the oracle

def test_the_quadrature_reproduces_the_sky_and_orientation_average():
    # Finn and Chernoff (1993): the horizon is 2.26 times the radius of the averaged volume.
    assert mean_cubed_response() ** (-1 / 3) == pytest.approx(2.26, abs=0.01)
    assert mean_cubed_response(24) == pytest.approx(mean_cubed_response(40), rel=1e-6)


def test_a_threshold_detector_has_the_closed_form_volume():
    rng = np.random.default_rng(1)
    injections = snr_scaled(rng, isotropic_sources(rng, 40_000), lambda snrs: snrs)
    truth = threshold_detector_volume(8.0)
    volume = injections.volume(8.0)
    lower, upper = injections.interval(8.0, confidence=0.9)
    assert lower < truth < upper, (lower, truth, upper)
    assert volume == pytest.approx(truth, rel=0.03)
    # The reach is the loudest source's distance at the horizon SNR.
    assert injections.max_distance_mpc == pytest.approx(HORIZON_MPC / HORIZON_SNR, rel=0.02)
    assert injections.max_distance_mpc <= HORIZON_MPC / HORIZON_SNR
    np.testing.assert_allclose(sensitive_distance(sphere_volume_gpc3(123.0)), 123.0, rtol=1e-12)


def test_a_threshold_above_the_snr_range_finds_only_the_nearer_sources_it_assumes():
    # Every injection scores 0 and is missed: only a score of 0 or below finds anything,
    # and then every source within the reach is found, nearer than the draw or not.
    rng = np.random.default_rng(2)
    D = np.full(20_000, HORIZON_MPC)
    injections = snr_scaled(rng, D, np.zeros_like)
    assert injections.volume(0.5) == 0.0
    assert injections.volume(-np.inf) == pytest.approx(injections.reference_volume_gpc3, rel=0.03)
    assert np.isnan(injections.volume(np.nan))


@pytest.mark.parametrize("level", [0.9, 0.5])
def test_the_injection_interval_covers_the_truth_at_its_level(level):
    replicas, n = 400, 2000
    rng = np.random.default_rng(3)
    truth = threshold_detector_volume(8.0)
    covered = 0
    for _ in range(replicas):
        injections = snr_scaled(rng, isotropic_sources(rng, n), lambda snrs: snrs)
        lower, upper = injections.interval(8.0, confidence=level)
        covered += lower <= truth <= upper
    assert abs(covered / replicas - level) <= coverage_tolerance(replicas, level), covered / replicas


def test_an_unweighted_draw_has_the_binomial_interval_and_error():
    # Uniform in volume: V is the found fraction of the reference volume, its interval
    # Clopper-Pearson's, and its error V_ref sqrt(N p (1 - p)) / N.
    scores = np.array([3.0, 1.0, 2.0, 0.0, 5.0, 4.0, 1.5, 2.5])
    injections = InjectionVolumes.uniform_in_volume(scores, 100.0)
    v_ref = 4.0 / 3.0 * np.pi * 100.0**3 * GPC3_PER_MPC3
    k, n = 3, scores.size  # found at a threshold of 3
    assert injections.volume(3.0) == pytest.approx(v_ref * k / n, rel=1e-12)
    p = k / n
    assert injections.standard_error(3.0) == pytest.approx(v_ref * np.sqrt(n * p * (1 - p)) / n, rel=1e-12)
    lower, upper = injections.interval(3.0, confidence=0.9)
    assert lower == pytest.approx(v_ref * stats.beta.ppf(0.05, k, n - k + 1), rel=1e-12)
    assert upper == pytest.approx(v_ref * stats.beta.ppf(0.95, k + 1, n - k), rel=1e-12)
    # Nothing found: the upper bound of no successes in N, of the largest volume.
    lower, upper = injections.interval(10.0, confidence=0.9)
    assert lower == 0.0 and upper == pytest.approx(v_ref * (1 - 0.05 ** (1 / n)), rel=1e-12)
    # A multiplicity counts an injection that many times.
    assert injections.volume(3.0, multiplicity=[2, 0, 0, 0, 1, 0, 0, 0]) == pytest.approx(v_ref * 3 / n)


def test_injection_volumes_refuse_what_they_cannot_integrate():
    with pytest.raises(ValueError, match="one per injection"):
        InjectionVolumes(np.zeros(3), np.zeros(2), 1.0)
    with pytest.raises(ValueError, match=">= 0"):
        InjectionVolumes(np.zeros(2), np.array([1.0, -1.0]), 1.0)
    with pytest.raises(ValueError, match="no volume"):
        InjectionVolumes.from_snr_scaled(np.zeros(2), np.zeros(2), np.zeros(2), SNR_RANGE, 1.0, 4.0)
    with pytest.raises(ValueError, match="top_snr_width"):
        InjectionVolumes.from_snr_scaled(np.zeros(2), np.ones(2), np.ones(2), SNR_RANGE, 20.0, 4.0)


# -------------------------------------------------------- the FAR intervals

def test_far_band_thresholds_bracket_the_threshold_by_the_poisson_interval():
    rng = np.random.default_rng(4)
    background = Background.independent_windows(rng.normal(size=5000), 1.0)
    far = 0.01
    high, low = background.far_band_thresholds(far, 0.9)
    assert high > background.threshold_for_far(far).threshold > low
    T = background.livetime_seconds
    for threshold, end in ((high, 1), (low, 0)):
        # The end of the interval is within far at the threshold, and beyond it just below.
        count = background.far_count(threshold)
        assert poisson_interval(count, 0.9)[end] / T <= far
        below = background.far_count(np.nextafter(threshold, -np.inf))
        assert poisson_interval(below, 0.9)[end] / T > far
    # A background too short for any interval to reach below the target: nothing passes high.
    short = Background.independent_windows(rng.normal(size=10), 1.0)
    assert short.far_band_thresholds(0.1, 0.9)[0] == np.inf


def test_resampling_draws_blocks_or_independent_windows():
    from gravyflow.src.validate.metrics import ScoreStream

    rng = np.random.default_rng(5)
    independent = Background.independent_windows(np.arange(6.0), 2.0)
    replica = independent.resample(rng)
    assert replica.livetime_seconds == 12.0 and replica.num_windows == 6
    assert set(replica._opens) <= set(range(6))
    streams = [ScoreStream(np.array([3.0, 3.0]), 1.0, 1.0, block=0),
               ScoreStream(np.array([0.0, 3.0, 0.0]), 1.0, 1.0, block=1),
               ScoreStream(np.array([3.0]), 1.0, 1.0, block=1)]
    slid = Background(streams, cluster_seconds=1.0)
    for _ in range(20):
        replica = slid.resample(rng)
        blocks = [stream.block for stream in replica.streams]
        # Two blocks drawn, each whole: block 1's two streams together.
        assert len(blocks) in (2, 3, 4) and blocks.count(1) % 2 == 0
        assert replica.cluster_seconds == 1.0


#: The synthetic detector's score: SNR less this, plus N(0, 1), so that it finds sources
#: around SNR 8, well inside the range and the population, at the FARs tested.
SCORE_OFFSET = 6.0


def synthetic_detection(rng, num_noise, num_injections, D):
    """Noise scores N(0, 1) in 1 s windows; an injection scores its SNR less the offset plus N(0, 1)."""
    background = Background.independent_windows(rng.normal(size=num_noise), 1.0)
    injections = snr_scaled(rng, D, lambda snrs: snrs - SCORE_OFFSET + rng.normal(size=snrs.size))
    return background, injections


def synthetic_truth(far_hz: float) -> float:
    """
    V at a true FAR for the synthetic detector of sources all at one ``D``: the threshold is
    the normal quantile of the FAR per window, and V = 4 pi D^3 int found(rho) rho^-4 d rho
    from the reach's SNR up.
    """
    threshold = stats.norm.isf(far_hz * 1.0)
    found = lambda rho: stats.norm.sf(threshold + SCORE_OFFSET - rho) * rho**-4
    # Beyond SNR 15 every source is found to within 1e-11.
    below, _ = integrate.quad(found, HORIZON_SNR, 15.0, epsabs=0, epsrel=1e-10)
    return 4.0 * np.pi * HORIZON_MPC**3 * (below + 15.0**-3 / 3.0) * GPC3_PER_MPC3


def test_the_curve_reads_volume_at_the_backgrounds_threshold():
    rng = np.random.default_rng(6)
    background, injections = synthetic_detection(rng, 20_000, 20_000, np.full(20_000, HORIZON_MPC))
    curve = sensitive_volume_curve(injections, background, [1e-2, 1e-3, 1e-5], rng, num_bootstrap=200)
    assert curve.resolved.tolist() == [True, True, False]
    assert np.all(np.isnan([curve.volume_gpc3[2], curve.joint_lower_gpc3[2], curve.background_lower_gpc3[2],
                            curve.injection_upper_gpc3[2]]))
    for i in (0, 1):
        assert curve.volume_gpc3[i] == injections.volume(background.threshold_for_far(curve.far_hz[i]).threshold)
        for lower, upper in ((curve.injection_lower_gpc3, curve.injection_upper_gpc3),
                             (curve.background_lower_gpc3, curve.background_upper_gpc3),
                             (curve.joint_lower_gpc3, curve.joint_upper_gpc3)):
            assert lower[i] < curve.volume_gpc3[i] < upper[i]
        assert curve.volume_gpc3[i] == pytest.approx(synthetic_truth(curve.far_hz[i]), rel=0.1)
    assert curve.volume_gpc3[0] > curve.volume_gpc3[1]
    np.testing.assert_array_equal(curve.distance_mpc, sensitive_distance(curve.volume_gpc3))
    assert curve.num_injections == 20_000 and curve.num_bootstrap == 200


def test_the_joint_interval_covers_the_truth_at_its_level():
    # The joint bootstrap resamples the background as well as the injections, so it
    # covers the volume at the true FAR, whose threshold the background only estimates.
    # Here the background is short (about 10 triggers) and the injections many, so the
    # injections' interval alone, blind to the threshold's error, falls well short.
    replicas, far, num_noise, num_injections = 300, 0.02, 500, 20_000
    rng = np.random.default_rng(7)
    truth = synthetic_truth(far)
    joint = injections_alone = 0
    for _ in range(replicas):
        background, injections = synthetic_detection(rng, num_noise, num_injections,
                                                     np.full(num_injections, HORIZON_MPC))
        curve = sensitive_volume_curve(injections, background, [far], rng, num_bootstrap=200)
        joint += curve.joint_lower_gpc3[0] <= truth <= curve.joint_upper_gpc3[0]
        injections_alone += curve.injection_lower_gpc3[0] <= truth <= curve.injection_upper_gpc3[0]
    tolerance = coverage_tolerance(replicas, 0.9)
    assert abs(joint / replicas - 0.9) <= tolerance, joint / replicas
    assert injections_alone / replicas < 0.9 - tolerance, injections_alone / replicas
