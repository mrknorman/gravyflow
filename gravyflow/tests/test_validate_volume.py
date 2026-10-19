"""
Sensitive volume against independent references: the closed-form volume of a
threshold detector averaged over sky and orientation; a realistic detector, whose
efficiency is smooth and whose noise can pass the threshold, against its volume
computed by quadrature; the coverage of each interval over many replicas; and the
volume beyond the reach, estimated beside the volume within it, against its
quadrature and over replicas, on independent windows and on time slides.
"""

import numpy as np
import pytest
from scipy import stats

from gravyflow.src.validate.metrics import Background, ScoreStream, poisson_interval
from gravyflow.src.validate.volume import (
    BEYOND_REACH_FACTOR,
    GPC3_PER_MPC3,
    InjectionVolumes,
    VolumeQuote,
    sensitive_distance,
    sensitive_volume_curve,
    sphere_volume_gpc3,
    uniform_in_volume,
)

# The horizon (Mpc): an overhead, face-on source's distance at SNR 1.
HORIZON_MPC = 4000.0
# The population's reach: its loudest source has SNR 4 there.
REACH_MPC = HORIZON_MPC / 4.0


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


def population(rng, n, reach=REACH_MPC):
    """
    Sources drawn as an injection campaign draws them: isotropic in sky and
    orientation, uniform in volume out to the reach. Returns each one's optimal SNR
    and distance.
    """
    D = HORIZON_MPC * antenna_response(rng.uniform(-1, 1, n), rng.uniform(0, np.pi, n),
                                       rng.uniform(0, np.pi, n), rng.uniform(-1, 1, n))
    distances = uniform_in_volume(n, reach, rng)
    return D / distances, distances


def detected(rng, n, detector, reach=REACH_MPC):
    """``n`` sources of the population out to ``reach``, scored by ``detector`` (of their SNRs)."""
    snrs, distances = population(rng, n, reach)
    return InjectionVolumes(detector(rng, snrs), reach, distances, snrs)


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


def realistic_truth(far_hz: float, beyond: bool = False) -> float:
    """
    V at a true FAR for the realistic detector, by quadrature: the threshold is the normal
    quantile of the FAR per 1 s window, and V = V_ref E[P(SNR + n >= t)] over the
    population, the SNR of a source of response g at x R being H g / (x R), x^3
    uniform: V_ref sum_g w_g int_0^1 3 x^2 Phi(H g / (x R) - t) dx. Sources found by
    their noise alone, however far, are in it. ``beyond``: the volume found by signal
    between the reach and `BEYOND_REACH_FACTOR` times it, the same integral from x = 1
    to x = k, less the noise's chance ``far_hz``.
    """
    threshold = stats.norm.isf(far_hz * 1.0)
    response, weights = response_quadrature(16)
    x, w = np.polynomial.legendre.leggauss(256)
    low, high = (1.0, BEYOND_REACH_FACTOR) if beyond else (0.0, 1.0)
    x, w = low + 0.5 * (high - low) * (x + 1), 0.5 * (high - low) * w
    snr = HORIZON_MPC * response[:, None] / (x[None, :] * REACH_MPC)
    found = stats.norm.sf(threshold - snr) - (far_hz if beyond else 0.0)
    return float(sphere_volume_gpc3(REACH_MPC)) * np.sum(weights[:, None] * w[None, :] * 3 * x[None, :] ** 2
                                                                * found)


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
    injections = detected(rng, 40_000, lambda rng, snrs: snrs)
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
    distances = np.linspace(10.0, 100.0, scores.size)
    injections = InjectionVolumes(scores, 100.0, distances, np.ones(scores.size))
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
    weighted = InjectionVolumes(scores, 100.0, distances, np.ones(scores.size), weights=np.arange(1.0, 9.0))
    assert weighted.volume(3.0) == pytest.approx(v_ref * (1 + 5 + 6) / n, rel=1e-12)


def test_the_volume_beyond_the_reach_moves_the_outer_injections_out_at_their_snr_over_k():
    # Eight injections, all reading the found chance in one SNR bin (fewer than a bin's worth):
    # the five at d >= R / k, moved k times as far, stand for the volume between R and k R, each
    # found at the bin's found fraction, less the noise's chance.
    scores = np.array([3.0, 1.0, 2.0, 0.0, 5.0, 4.0, 1.5, 2.5])
    distances = np.array([10.0, 20.0, 40.0, 49.0, 50.0, 60.0, 80.0, 100.0])
    injections = InjectionVolumes(scores, 100.0, distances, np.linspace(1.0, 2.0, 8))
    k, v_ref, n = BEYOND_REACH_FACTOR, sphere_volume_gpc3(100.0), scores.size
    moved, found = 4, 3 / 8  # d >= 50; three of eight found at a threshold of 3
    for p in (0.0, 0.1):
        assert injections.beyond_reach(3.0, p) == pytest.approx(v_ref * k**3 / n * moved * (found - p), rel=1e-12)
    assert np.isnan(injections.beyond_reach(np.nan, 0.0))
    # With a weight, each moved injection stands for its weight; a multiplicity counts it (as a moved
    # source and as a reading) that many times.
    weighted = InjectionVolumes(scores, 100.0, distances, np.ones(n), weights=np.arange(1.0, 9.0))
    assert weighted.beyond_reach(3.0, 0.0) == pytest.approx(v_ref * k**3 / n * (5 + 6 + 7 + 8) * found, rel=1e-12)
    twice = np.array([0, 0, 0, 0, 2, 0, 0, 0])  # the loudest moved injection, twice: found 2 of 2
    assert injections.beyond_reach(3.0, 0.0, twice) == pytest.approx(v_ref * k**3 / n * 2, rel=1e-12)


def test_injection_volumes_refuse_what_they_cannot_integrate():
    ones = np.ones(2)
    with pytest.raises(ValueError, match="one of each per injection"):
        InjectionVolumes(np.zeros(3), 1.0, np.ones(3), np.ones(3), weights=np.ones(2))
    with pytest.raises(ValueError, match="one of each per injection"):
        InjectionVolumes(np.zeros(3), 1.0, ones, np.ones(3))
    with pytest.raises(ValueError, match="one of each per injection"):
        InjectionVolumes(np.zeros(2), 1.0, ones, np.ones(3))
    with pytest.raises(ValueError, match=">= 0"):
        InjectionVolumes(np.zeros(2), 1.0, ones, ones, weights=np.array([1.0, -1.0]))
    with pytest.raises(ValueError, match="SNRs must be finite and >= 0"):
        InjectionVolumes(np.zeros(2), 1.0, ones, np.array([1.0, np.nan]))
    with pytest.raises(ValueError, match="reach_mpc"):
        InjectionVolumes(np.zeros(2), 0.0, ones, ones)
    # Every injection is inside the reach it was drawn to.
    with pytest.raises(ValueError, match=r"in \(0, reach_mpc"):
        InjectionVolumes(np.zeros(2), 1.0, np.array([0.5, 1.5]), ones)
    with pytest.raises(ValueError, match="reach_mpc"):
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
        injections = detected(rng, n, realistic_scores)
        volumes[r] = injections.volume(threshold)
        lower, upper = injections.interval(threshold, confidence=0.9)
        covered += lower <= truth <= upper
    assert abs(volumes.mean() - truth) <= 3 * volumes.std() / np.sqrt(replicas), (volumes.mean(), truth)
    assert abs(covered / replicas - 0.9) <= coverage_tolerance(replicas, 0.9), covered / replicas


def realistic_curve(rng, allowed: int, far: float, num_injections: int, num_bootstrap: int, slides: int = 0,
                    detector=realistic_scores):
    """
    The realistic detector's curve at ``far``, from a background of 1 s windows allowing
    ``allowed`` triggers: independent windows, or ``slides`` time slides that each read the
    same windows again (a slide of k, rolled by k), as a score that one detector's
    excursions dominate reads them: every window's score appears once in each slide.
    """
    if slides == 0:
        background = Background.independent_windows(rng.normal(size=int(round(allowed / far))), 1.0)
    else:
        noise = rng.normal(size=int(round(allowed / far / slides)))
        background = Background([ScoreStream(np.roll(noise, k), 1.0, 1.0, 0, k) for k in range(1, slides + 1)])
    injections = detected(rng, num_injections, detector)
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


@pytest.mark.parametrize("slides", [0, 16])
def test_both_intervals_cover_the_truth_on_time_slides(slides):
    # Sixteen slides of the same windows hold the information of one: a background
    # that counted every slid window as independent would put the threshold 4 times too
    # precisely. Counting its data's reuse, the joint interval and the interval beyond
    # the reach cover as they do on independent windows (the same 10 events allowed).
    replicas, far = 300, 1e-2
    rng = np.random.default_rng(30 + slides)
    truth, beyond = realistic_truth(far), realistic_truth(far, beyond=True)
    misses = np.zeros((2, 2))
    for _ in range(replicas):
        curve = realistic_curve(rng, 10 * max(slides, 1), far, num_injections=20_000, num_bootstrap=100,
                                slides=slides)
        assert curve.data_reuse == max(slides, 1)
        for q, (value, lower, upper) in enumerate(((truth, curve.joint_lower_gpc3, curve.joint_upper_gpc3),
                                                   (beyond, curve.beyond_reach_lower_gpc3,
                                                    curve.beyond_reach_upper_gpc3))):
            misses[q] += value < lower[0], value > upper[0]
    tolerance = coverage_tolerance(replicas, 0.05)
    assert np.all(misses / replicas <= 0.05 + tolerance), misses / replicas


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


def test_bootstrap_thresholds_draw_the_count_as_poisson_in_independent_events():
    # Unslid, a replicate's threshold is at or below a threshold of count c exactly when a
    # Poisson count of mean c is within the target's allowance a: the bootstrap of independent
    # windows. Slid sixteen times, the count is in events of sixteen windows each.
    rng = np.random.default_rng(5)
    noise = rng.normal(size=400)
    replicas, far = 4000, 0.05  # a = 20 triggers allowed
    for slides in (0, 16):
        background = Background.independent_windows(noise, 1.0) if slides == 0 else Background(
            [ScoreStream(np.roll(noise[:25], k), 1.0, 1.0, 0, k) for k in range(1, slides + 1)])
        reuse = max(slides, 1)
        assert background.data_reuse == reuse and background.livetime_seconds == 400.0
        thresholds = np.array([background.bootstrap_thresholds([far, 1e-3, 0.1], rng) for _ in range(replicas)])
        assert np.all(np.isnan(thresholds[:, 1]))  # 0.4 triggers allowed: unresolved
        assert np.all(thresholds[:, 2] <= thresholds[:, 0])  # one process: a higher target, a lower threshold
        for count in (12, 20, 30):
            u = background._threshold_allowing(count)
            expected = stats.poisson.cdf(20 // reuse, background.far_count(u) / reuse)
            assert np.mean(thresholds[:, 0] <= u) == pytest.approx(expected, abs=4 * np.sqrt(0.25 / replicas))


def test_the_curve_reads_volume_at_the_backgrounds_threshold_with_its_noise_floor():
    rng = np.random.default_rng(6)
    background = Background.independent_windows(rng.normal(size=20_000), 1.0)
    injections = detected(rng, 20_000, realistic_scores)
    curve = sensitive_volume_curve(injections, background, [1e-1, 1e-2, 1e-3, 1e-5], rng, num_bootstrap=200)
    assert curve.resolved.tolist() == [True, True, True, False]
    assert np.all(np.isnan([curve.volume_within_reach_gpc3[3], curve.joint_lower_gpc3[3],
                            curve.background_lower_gpc3[3], curve.injection_upper_gpc3[3], curve.noise_floor_gpc3[3],
                            curve.beyond_reach_gpc3[3], curve.beyond_reach_upper_gpc3[3]]))
    for i in (0, 1, 2):
        threshold = background.threshold_for_far(curve.far_hz[i]).threshold
        noise_chance = np.mean(background._opens >= threshold)
        assert curve.volume_within_reach_gpc3[i] == injections.volume(threshold)
        assert curve.beyond_reach_gpc3[i] == injections.beyond_reach(threshold, noise_chance)
        for value, lower, upper in (
                (curve.volume_within_reach_gpc3, curve.injection_lower_gpc3, curve.injection_upper_gpc3),
                (curve.volume_within_reach_gpc3, curve.background_lower_gpc3, curve.background_upper_gpc3),
                (curve.volume_within_reach_gpc3, curve.joint_lower_gpc3, curve.joint_upper_gpc3),
                (curve.beyond_reach_gpc3, curve.beyond_reach_lower_gpc3, curve.beyond_reach_upper_gpc3)):
            assert lower[i] < value[i] < upper[i]
        assert curve.volume_within_reach_gpc3[i] == pytest.approx(realistic_truth(curve.far_hz[i]), rel=0.1)
        # Noise alone finds the fraction of windows the threshold passes, of the whole reach.
        assert curve.noise_floor_gpc3[i] == pytest.approx(noise_chance * injections.reference_volume_gpc3, rel=1e-12)
    # At 0.1 Hz a tenth of the noise windows pass: most of what is found, noise found.
    assert curve.noise_dominated.tolist() == [True, False, False, False]
    assert curve.quotable.tolist() == [False, True, True, False]
    assert curve.noise_floor_fraction == 0.1 and curve.beyond_reach_fraction == 0.1
    assert curve.volume_within_reach_gpc3[0] > curve.volume_within_reach_gpc3[1] > curve.volume_within_reach_gpc3[2]
    np.testing.assert_array_equal(curve.distance_within_reach_mpc, sensitive_distance(curve.volume_within_reach_gpc3))
    assert curve.reach_mpc == REACH_MPC and curve.beyond_reach_factor == BEYOND_REACH_FACTOR
    assert curve.num_injections == 20_000 and curve.num_bootstrap == 200 and curve.data_reuse == 1


# --------------------------------------------------------- beyond the reach

def offset_scores(offset):
    """A detector that needs ``offset`` more SNR than the realistic one: none of the reach's outer shell is loud enough."""
    return lambda rng, snrs: snrs - offset + rng.normal(size=snrs.shape)


def test_the_volume_beyond_the_reach_is_estimated_and_quoted_with_a_warning(caplog):
    # The realistic detector finds sources of SNR 2-4 (a threshold near 3): with the reach
    # where the loudest has SNR 4, it finds sources beyond it. V within the reach is the
    # estimand, quoted; the estimate beyond it agrees with its quadrature and with V
    # measured to twice the reach, and, more than a tenth of V, marks the point
    # reach-limited and warns.
    rng = np.random.default_rng(40)
    background = Background.independent_windows(rng.normal(size=20_000), 1.0)
    fars = [1e-2, 1e-3]
    curve = sensitive_volume_curve(detected(rng, 20_000, realistic_scores), background, fars, rng, num_bootstrap=100)
    longer = sensitive_volume_curve(detected(rng, 80_000, realistic_scores, 2 * REACH_MPC), background,
                                    fars, rng, num_bootstrap=50)
    for i, far in enumerate(fars):
        truth = realistic_truth(far, beyond=True)
        assert curve.beyond_reach_lower_gpc3[i] < truth < curve.beyond_reach_upper_gpc3[i]
        # At the same threshold (the same background), V to twice the reach, less its noise floor,
        # is V within it, less its own, plus what lies between.
        measured = (longer.volume_within_reach_gpc3[i] - longer.noise_floor_gpc3[i]
                    - (curve.volume_within_reach_gpc3[i] - curve.noise_floor_gpc3[i]))
        # Within three standard errors of the injections (the estimate's own is about the longer set's).
        tolerance = 3 * np.sqrt(2 * longer.injection_error_gpc3[i] ** 2 + curve.injection_error_gpc3[i] ** 2)
        assert abs(curve.beyond_reach_gpc3[i] - measured) < tolerance, (curve.beyond_reach_gpc3[i], measured)
    assert curve.reach_limited.tolist() == [True, True] and curve.quotable.tolist() == [True, True]
    with caplog.at_level("WARNING"):
        quote = curve.quote(1e-3)
    assert quote == VolumeQuote(
        1e-3, REACH_MPC, curve.volume_within_reach_gpc3[1], curve.joint_lower_gpc3[1],
        curve.joint_upper_gpc3[1], curve.beyond_reach_gpc3[1], curve.beyond_reach_lower_gpc3[1],
        curve.beyond_reach_upper_gpc3[1], BEYOND_REACH_FACTOR, True)
    assert "misses an estimated" in caplog.text and f"({REACH_MPC:.0f} Mpc)" in caplog.text
    # A detector that needs SNR 3 more finds nothing beyond the reach to speak of: quoted with no warning.
    caplog.clear()
    quiet = sensitive_volume_curve(detected(rng, 20_000, offset_scores(3.0)), background, fars, rng,
                                   num_bootstrap=50)
    assert not quiet.reach_limited.any()
    with caplog.at_level("WARNING"):
        assert not quiet.quote(1e-3).reach_limited
    assert "misses" not in caplog.text


def test_the_interval_beyond_the_reach_covers_at_its_level():
    # Over replicas of the realistic detector (fresh injections and background each), the
    # interval on the volume beyond the reach covers its quadrature as a central 90%
    # interval should, at a target of 100 triggers and one of 10.
    replicas, fars = 200, [1e-2, 1e-3]
    rng = np.random.default_rng(43)
    truths = [realistic_truth(far, beyond=True) for far in fars]
    misses = np.zeros((2, 2))
    for _ in range(replicas):
        background = Background.independent_windows(rng.normal(size=10_000), 1.0)
        curve = sensitive_volume_curve(detected(rng, 4000, realistic_scores), background, fars, rng, num_bootstrap=100)
        misses[:, 0] += np.array(truths) < curve.beyond_reach_lower_gpc3
        misses[:, 1] += np.array(truths) > curve.beyond_reach_upper_gpc3
    assert np.all(misses / replicas <= 0.05 + coverage_tolerance(replicas, 0.05)), misses / replicas


def test_quote_refuses_every_number_not_to_quote_by_name():
    rng = np.random.default_rng(42)
    background = Background.independent_windows(rng.normal(size=20_000), 1.0)
    curve = sensitive_volume_curve(detected(rng, 20_000, offset_scores(-1.0)), background,
                                   [1e-1, 1e-3, 1e-5], rng, num_bootstrap=50)
    assert curve.noise_dominated[0] and not curve.resolved[2]
    assert curve.quotable.tolist() == [False, True, False]
    with pytest.raises(ValueError, match="noise-dominated"):
        curve.quote(1e-1)
    with pytest.raises(ValueError, match="cannot resolve"):
        curve.quote(1e-5)
    with pytest.raises(ValueError, match="not one of the curve's target FARs"):
        curve.quote(2e-3)
