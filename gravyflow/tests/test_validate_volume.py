"""
Sensitive volume against independent references: the closed-form volume of a
threshold detector averaged over sky and orientation; a realistic detector, whose
efficiency is smooth and whose noise can pass the threshold, against its volume
computed by quadrature; and the coverage of each interval over many replicas.
"""

import numpy as np
import pytest
from scipy import stats

from gravyflow.src.validate.metrics import Background, ScoreStream, poisson_interval
from gravyflow.src.validate.volume import (
    GPC3_PER_MPC3,
    InjectionVolumes,
    sensitive_distance,
    sensitive_volume_curve,
    sphere_volume_gpc3,
    uniform_in_volume,
)

# The horizon (Mpc): an overhead, face-on source's distance at SNR 1.
HORIZON_MPC = 4000.0
# The population's reach: its loudest source has SNR 4 there.
MAX_DISTANCE_MPC = HORIZON_MPC / 4.0


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


def response_quadrature(n=32):
    """The response at the nodes of a sky and orientation quadrature, and their weights (summing to 1)."""
    x, w = np.polynomial.legendre.leggauss(n)
    angles = np.pi * np.arange(n) / n  # the response has period pi in phi and psi
    ct, phi, psi, ci = np.meshgrid(x, angles, angles, x, indexing="ij")
    weights = w[:, None, None, None] * w[None, None, None, :] / (4.0 * n * n) * np.ones_like(ct)
    return antenna_response(ct, phi, psi, ci).ravel(), weights.ravel()


def mean_cubed_response(n=32) -> float:
    """The sky- and orientation-average of the response cubed, by quadrature (no sampling)."""
    response, weights = response_quadrature(n)
    return float(np.sum(weights * response**3))


def population(rng, n):
    """
    Sources drawn as an injection campaign draws them: isotropic in sky and
    orientation, uniform in volume out to the reach. Returns each one's optimal SNR.
    """
    D = HORIZON_MPC * antenna_response(rng.uniform(-1, 1, n), rng.uniform(0, np.pi, n),
                                       rng.uniform(0, np.pi, n), rng.uniform(-1, 1, n))
    return D / uniform_in_volume(n, MAX_DISTANCE_MPC, rng)


def threshold_detector_volume(threshold_snr: float) -> float:
    """
    The closed form: a source is found exactly when its SNR is at least ``threshold_snr``,
    so it is found within ``D / threshold_snr`` (inside the reach), and
    V = 4/3 pi E[D^3] / threshold^3.
    """
    return float(sphere_volume_gpc3(HORIZON_MPC / threshold_snr)) * mean_cubed_response()


def realistic_scores(rng, snrs):
    """A realistic detector: a window scores its signal's SNR plus N(0, 1) noise; noise alone, N(0, 1)."""
    return snrs + rng.normal(size=np.shape(snrs))


def realistic_truth(far_hz: float) -> float:
    """
    V at a true FAR for the realistic detector, by quadrature: the threshold is the normal
    quantile of the FAR per 1 s window, and V = V_ref E[P(SNR + n >= t)] over the
    population, the SNR of a source of response g at x d_max being H g / (x d_max), x^3
    uniform: V_ref sum_g w_g int_0^1 3 x^2 Phi(H g / (x d_max) - t) dx. Sources found by
    their noise alone, however far, are in it.
    """
    threshold = stats.norm.isf(far_hz * 1.0)
    response, weights = response_quadrature(16)
    x, w = np.polynomial.legendre.leggauss(256)
    x, w = 0.5 * (x + 1), 0.5 * w
    snr = HORIZON_MPC * response[:, None] / (x[None, :] * MAX_DISTANCE_MPC)
    found = np.sum(weights[:, None] * w[None, :] * 3 * x[None, :] ** 2 * stats.norm.sf(threshold - snr))
    return float(sphere_volume_gpc3(MAX_DISTANCE_MPC)) * found


def coverage_tolerance(replicas: int, level: float) -> float:
    """Three binomial standard deviations of a coverage measured over ``replicas``."""
    return 3.0 * np.sqrt(level * (1 - level) / replicas)


# ------------------------------------------------------------------ the oracles

def test_the_quadrature_reproduces_the_sky_and_orientation_average():
    # Finn and Chernoff (1993): the horizon is 2.26 times the radius of the averaged volume.
    assert mean_cubed_response() ** (-1 / 3) == pytest.approx(2.26, abs=0.01)
    assert mean_cubed_response(24) == pytest.approx(mean_cubed_response(40), rel=1e-6)
    # The realistic detector's truth, at a FAR low enough that its noise finds nothing,
    # is the threshold detector's, smoothed: between the closed forms either side.
    t = stats.norm.isf(1e-9)
    assert threshold_detector_volume(t + 1) < realistic_truth(1e-9) < threshold_detector_volume(t - 1)


def test_a_threshold_detector_has_the_closed_form_volume():
    rng = np.random.default_rng(1)
    snrs = population(rng, 40_000)
    injections = InjectionVolumes(snrs, MAX_DISTANCE_MPC)
    truth = threshold_detector_volume(8.0)
    lower, upper = injections.interval(8.0, confidence=0.9)
    assert lower < truth < upper, (lower, truth, upper)
    assert injections.volume(8.0) == pytest.approx(truth, rel=0.05)
    assert injections.volume(-np.inf) == pytest.approx(injections.reference_volume_gpc3, rel=1e-12)
    assert np.isnan(injections.volume(np.nan))
    np.testing.assert_allclose(sensitive_distance(sphere_volume_gpc3(123.0)), 123.0, rtol=1e-12)


def test_distances_uniform_in_volume_fill_the_sphere_evenly():
    d = uniform_in_volume(200_000, 300.0, np.random.default_rng(2))
    assert np.all((d > 0) & (d <= 300.0))
    # (d / d_max)^3 is uniform on (0, 1].
    assert stats.kstest((d / 300.0) ** 3, "uniform").pvalue > 1e-3


def test_an_unweighted_draw_has_the_binomial_interval_and_error():
    # Uniform in volume: V is the found fraction of the reference volume, its interval
    # Clopper-Pearson's, and its error V_ref sqrt(N p (1 - p)) / N.
    scores = np.array([3.0, 1.0, 2.0, 0.0, 5.0, 4.0, 1.5, 2.5])
    injections = InjectionVolumes(scores, 100.0)
    v_ref = 4.0 / 3.0 * np.pi * 100.0**3 * GPC3_PER_MPC3
    k, n = 3, scores.size  # found at a threshold of 3
    assert injections.volume(3.0) == pytest.approx(v_ref * k / n, rel=1e-12)
    p = k / n
    assert injections.standard_error(3.0) == pytest.approx(v_ref * np.sqrt(n * p * (1 - p)) / n, rel=1e-12)
    lower, upper = injections.interval(3.0, confidence=0.9)
    assert lower == pytest.approx(v_ref * stats.beta.ppf(0.05, k, n - k + 1), rel=1e-12)
    assert upper == pytest.approx(v_ref * stats.beta.ppf(0.95, k + 1, n - k), rel=1e-12)
    # Nothing found: the upper bound of no successes in N.
    lower, upper = injections.interval(10.0, confidence=0.9)
    assert lower == 0.0 and upper == pytest.approx(v_ref * (1 - 0.05 ** (1 / n)), rel=1e-12)
    # A multiplicity counts an injection that many times.
    assert injections.volume(3.0, multiplicity=[2, 0, 0, 0, 1, 0, 0, 0]) == pytest.approx(v_ref * 3 / n)
    # Population weights (a chirp-distance draw): each found injection stands for its weight.
    weighted = InjectionVolumes(scores, 100.0, weights=np.arange(1.0, 9.0))
    assert weighted.volume(3.0) == pytest.approx(v_ref * (1 + 5 + 6) / n, rel=1e-12)


def test_injection_volumes_refuse_what_they_cannot_integrate():
    with pytest.raises(ValueError, match="one per injection"):
        InjectionVolumes(np.zeros(3), 1.0, weights=np.ones(2))
    with pytest.raises(ValueError, match=">= 0"):
        InjectionVolumes(np.zeros(2), 1.0, weights=np.array([1.0, -1.0]))
    with pytest.raises(ValueError, match="max_distance_mpc"):
        InjectionVolumes(np.zeros(2), 0.0)
    with pytest.raises(ValueError, match="max_distance_mpc"):
        uniform_in_volume(2, -1.0, np.random.default_rng(0))


# ------------------------------------------------------------ a realistic detector

def test_a_realistic_detector_is_unbiased_and_its_injection_interval_covers():
    # At the true threshold, the injection interval alone is the whole error: over
    # replicas, V is unbiased and the interval covers the truth at its level, the
    # sources its noise finds included.
    replicas, n, far = 300, 4000, 1e-3
    rng = np.random.default_rng(3)
    truth = realistic_truth(far)
    threshold = stats.norm.isf(far)
    volumes, covered = np.empty(replicas), 0
    for r in range(replicas):
        injections = InjectionVolumes(realistic_scores(rng, population(rng, n)), MAX_DISTANCE_MPC)
        volumes[r] = injections.volume(threshold)
        lower, upper = injections.interval(threshold, confidence=0.9)
        covered += lower <= truth <= upper
    assert abs(volumes.mean() - truth) <= 3 * volumes.std() / np.sqrt(replicas), (volumes.mean(), truth)
    assert abs(covered / replicas - 0.9) <= coverage_tolerance(replicas, 0.9), covered / replicas


def realistic_curve(rng, allowed: int, far: float, num_injections: int, num_bootstrap: int):
    """The realistic detector's curve at ``far``, from a background of 1 s windows allowing ``allowed`` triggers."""
    background = Background.independent_windows(rng.normal(size=int(round(allowed / far))), 1.0)
    injections = InjectionVolumes(realistic_scores(rng, population(rng, num_injections)), MAX_DISTANCE_MPC)
    return sensitive_volume_curve(injections, background, [far], rng, num_bootstrap=num_bootstrap)


@pytest.mark.parametrize("allowed", [1, 2, 3, 10])
def test_the_joint_interval_covers_the_truth_where_the_background_allows_few_triggers(allowed):
    # The background estimates the threshold from `allowed` triggers; the truth is V at the
    # true FAR's threshold. Where it allows one or two, the loudest trigger bounds no
    # bootstrap replicate's threshold from above, and the joint interval's lower end is
    # the background interval's.
    replicas, far = 300, 1e-2
    rng = np.random.default_rng(10 + allowed)
    truth = realistic_truth(far)
    below = above = 0
    for _ in range(replicas):
        curve = realistic_curve(rng, allowed, far, num_injections=2000, num_bootstrap=100)
        below += truth < curve.joint_lower_gpc3[0]
        above += truth > curve.joint_upper_gpc3[0]
    # A central 90% interval misses at most 5% on each side.
    tolerance = coverage_tolerance(replicas, 0.05)
    assert below / replicas <= 0.05 + tolerance and above / replicas <= 0.05 + tolerance, (below, above)


def test_the_joint_interval_is_not_the_injection_interval():
    # A short background (10 triggers) and many injections: the injection interval alone,
    # blind to the threshold's error, falls well short; the joint covers.
    replicas, far = 200, 1e-2
    rng = np.random.default_rng(20)
    truth = realistic_truth(far)
    joint = injections_alone = 0
    for _ in range(replicas):
        curve = realistic_curve(rng, 10, far, num_injections=20_000, num_bootstrap=100)
        joint += curve.joint_lower_gpc3[0] <= truth <= curve.joint_upper_gpc3[0]
        injections_alone += curve.injection_lower_gpc3[0] <= truth <= curve.injection_upper_gpc3[0]
    tolerance = coverage_tolerance(replicas, 0.9)
    assert abs(joint / replicas - 0.9) <= tolerance, joint / replicas
    assert injections_alone / replicas < 0.9 - tolerance, injections_alone / replicas


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


def test_bootstrap_draws_blocks_or_independent_windows():
    rng = np.random.default_rng(5)
    independent = Background.independent_windows(np.arange(6.0), 2.0)
    replica = independent.bootstrap(rng)
    assert replica.livetime_seconds == 12.0 and replica.num_windows == 6
    assert set(replica._opens) <= set(range(6))
    streams = [ScoreStream(np.array([3.0, 3.0]), 1.0, 1.0, block=0),
               ScoreStream(np.array([0.0, 3.0, 0.0]), 1.0, 1.0, block=1),
               ScoreStream(np.array([3.0]), 1.0, 1.0, block=1)]
    slid = Background(streams, cluster_seconds=1.0)
    for _ in range(20):
        replica = slid.bootstrap(rng)
        blocks = [stream.block for stream in replica.streams]
        # Two blocks drawn, each whole: block 1's two streams together.
        assert len(blocks) in (2, 3, 4) and blocks.count(1) % 2 == 0
        assert replica.cluster_seconds == 1.0


def test_the_curve_reads_volume_at_the_backgrounds_threshold_with_its_noise_floor():
    rng = np.random.default_rng(6)
    background = Background.independent_windows(rng.normal(size=20_000), 1.0)
    injections = InjectionVolumes(realistic_scores(rng, population(rng, 20_000)), MAX_DISTANCE_MPC)
    curve = sensitive_volume_curve(injections, background, [1e-1, 1e-2, 1e-3, 1e-5], rng, num_bootstrap=200)
    assert curve.resolved.tolist() == [True, True, True, False]
    assert np.all(np.isnan([curve.volume_gpc3[3], curve.joint_lower_gpc3[3], curve.background_lower_gpc3[3],
                            curve.injection_upper_gpc3[3], curve.noise_floor_gpc3[3]]))
    for i in (0, 1, 2):
        threshold = background.threshold_for_far(curve.far_hz[i]).threshold
        assert curve.volume_gpc3[i] == injections.volume(threshold)
        for lower, upper in ((curve.injection_lower_gpc3, curve.injection_upper_gpc3),
                             (curve.background_lower_gpc3, curve.background_upper_gpc3),
                             (curve.joint_lower_gpc3, curve.joint_upper_gpc3)):
            assert lower[i] < curve.volume_gpc3[i] < upper[i]
        assert curve.volume_gpc3[i] == pytest.approx(realistic_truth(curve.far_hz[i]), rel=0.1)
        # Noise alone finds the fraction of windows the threshold passes, of the whole reach.
        assert curve.noise_floor_gpc3[i] == pytest.approx(
            np.mean(background._opens >= threshold) * injections.reference_volume_gpc3, rel=1e-12)
    # At 0.1 Hz a tenth of the noise windows pass: most of what is found, noise found.
    assert curve.noise_dominated.tolist() == [True, False, False, False]
    assert curve.noise_floor_fraction == 0.1 and curve.joint_unresolved.tolist() == [0, 0, 0, 0]
    assert curve.volume_gpc3[0] > curve.volume_gpc3[1] > curve.volume_gpc3[2]
    np.testing.assert_array_equal(curve.distance_mpc, sensitive_distance(curve.volume_gpc3))
    assert curve.num_injections == 20_000 and curve.num_bootstrap == 200


def test_replicates_that_cannot_resolve_the_target_are_counted_never_dropped(caplog):
    # Blocks of unequal livetime: a replicate of mostly short blocks allows no trigger at
    # a target the background resolves. The joint interval is not taken over the rest.
    rng = np.random.default_rng(8)
    streams = [ScoreStream(rng.normal(size=size), 1.0, 1.0, block=b) for b, size in enumerate([1, 1, 1, 197])]
    background = Background(streams)
    injections = InjectionVolumes(realistic_scores(rng, population(rng, 1000)), MAX_DISTANCE_MPC)
    with caplog.at_level("WARNING"):
        curve = sensitive_volume_curve(injections, background, [1e-2, 1e-1], rng, num_bootstrap=200)
    assert curve.resolved.tolist() == [True, True]
    assert curve.joint_unresolved[0] > 0 and curve.joint_unresolved[1] > 0
    assert np.all(np.isnan(curve.joint_lower_gpc3)) and np.all(np.isnan(curve.joint_upper_gpc3))
    assert "0.01 Hz" in caplog.text and "of 200" in caplog.text
    assert np.all(np.isfinite(curve.injection_lower_gpc3)) and np.all(np.isfinite(curve.background_upper_gpc3))
