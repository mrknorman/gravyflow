"""
Sensitive volume against independent references: the closed-form volume of a
threshold detector averaged over sky and orientation; a realistic detector, whose
efficiency is smooth and whose noise can pass the threshold, against its volume
computed by quadrature, with the chance of a noise-only pass subtracted; a
population drawn uniformly in chirp distance, each source out to its own mass's
reach and weighted to keep the volume exact; and the coverage of each interval,
at 1 and 2 sigma, over many replicas, on independent windows and on time slides.
"""

import numpy as np
import pytest
from scipy import stats

from gravyflow.src.validate.metrics import Background, ChanceLaw, ScoreStream, poisson_interval, sigma_level
from gravyflow.src.validate.volume import (
    GPC3_PER_MPC3,
    REFERENCE_CHIRP_MASS_MSUN,
    InjectionVolumes,
    VolumeQuote,
    chirp_weights,
    found_fraction_by_distance,
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
    return InjectionVolumes(detector(rng, snrs), reach, distances)


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




def realistic_found(far_hz: float, reach=REACH_MPC) -> float:
    """
    The volume the realistic detector finds at a true FAR, by quadrature: the threshold is
    the normal quantile of the FAR per 1 s window, and it is V_ref E[P(SNR + n >= t)] over
    the population, the SNR of a source of response g at x R being H g / (x R), x^3
    uniform: V_ref sum_g w_g int_0^1 3 x^2 Phi(H g / (x R) - t) dx. Sources found by
    their noise alone, however far, are in it.
    """
    threshold = stats.norm.isf(far_hz * 1.0)
    response, weights = response_quadrature(16)
    x, w = np.polynomial.legendre.leggauss(256)
    x, w = 0.5 * (x + 1), 0.5 * w
    snr = HORIZON_MPC * response[:, None] / (x[None, :] * reach)
    found = stats.norm.sf(threshold - snr)
    return float(sphere_volume_gpc3(reach)) * np.sum(weights[:, None] * w[None, :] * 3 * x[None, :] ** 2 * found)


def realistic_truth(far_hz: float, reach=REACH_MPC) -> float:
    """
    V within the reach at a true FAR, the estimand: what the realistic detector finds less
    what noise alone finds, the chance ``p = FAR x 1 s`` of a noise window passing, over the
    fraction it leaves, ``(found - p V_ref) / (1 - p)``.
    """
    p = far_hz * 1.0
    return (realistic_found(far_hz, reach) - p * float(sphere_volume_gpc3(reach))) / (1 - p)


def coverage_tolerance(replicas: int, level: float) -> float:
    """Three binomial standard deviations of a coverage measured over ``replicas``."""
    return 3.0 * np.sqrt(level * (1 - level) / replicas)


#: The chance of each side's miss of a central interval at 1 and 2 sigma.
TAILS = tuple((1 - sigma_level(s)) / 2 for s in (1.0, 2.0))


def assert_misses_within_tails(misses, replicas):
    """``misses`` (levels, 2 sides) counted over ``replicas``: each side at most its tail, to 3 sigma."""
    for level, tail in enumerate(TAILS):
        assert np.all(misses[level] / replicas <= tail + coverage_tolerance(replicas, tail)), (level, misses / replicas)


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
    lower, upper = injections.interval(8.0, confidence=sigma_level(2.0))
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
    injections = InjectionVolumes(scores, 100.0, distances)
    v_ref = 4.0 / 3.0 * np.pi * 100.0**3 * GPC3_PER_MPC3
    k, n = 3, scores.size  # found at a threshold of 3
    assert injections.volume(3.0) == pytest.approx(v_ref * k / n, rel=1e-12)
    assert injections.population_volume_gpc3 == pytest.approx(v_ref, rel=1e-12)
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
    # Population weights (a chirp-distance draw): each found injection stands for its weight, and
    # the population fills its weights' mean of the reference volume.
    weighted = InjectionVolumes(scores, 100.0, distances, weights=np.arange(1.0, 9.0))
    assert weighted.volume(3.0) == pytest.approx(v_ref * (1 + 5 + 6) / n, rel=1e-12)
    assert weighted.population_volume_gpc3 == pytest.approx(v_ref * 4.5, rel=1e-12)


def test_the_chance_of_a_noise_only_pass_is_subtracted_from_the_volume():
    # An injection is found when its window passes, whatever passes it: noise alone passes with
    # chance c, everywhere in the reach, so V is what is found less c of the population's volume,
    # over 1 - c. Its injection interval maps the found volume's binomial ends the same way.
    scores = np.array([3.0, 1.0, 2.0, 0.0, 5.0, 4.0, 1.5, 2.5])
    injections = InjectionVolumes(scores, 100.0, np.linspace(10.0, 100.0, 8), weights=np.arange(1.0, 9.0))
    found, population, c = injections.found_volume(3.0), injections.population_volume_gpc3, 0.25
    assert injections.volume(3.0, chance=c) == pytest.approx((found - c * population) / (1 - c), rel=1e-12)
    assert injections.volume(3.0) == found
    lower, upper = injections.interval(3.0, confidence=0.9)
    chance_lower, chance_upper = injections.interval(3.0, confidence=0.9, chance=c)
    assert chance_lower == pytest.approx((lower - c * population) / (1 - c), rel=1e-12)
    assert chance_upper == pytest.approx((upper - c * population) / (1 - c), rel=1e-12)


def test_injection_volumes_refuse_what_they_cannot_integrate():
    ones = np.ones(2)
    with pytest.raises(ValueError, match="one of each per injection"):
        InjectionVolumes(np.zeros(3), 1.0, np.ones(3), weights=np.ones(2))
    with pytest.raises(ValueError, match="one of each per injection"):
        InjectionVolumes(np.zeros(3), 1.0, ones)
    with pytest.raises(ValueError, match=">= 0"):
        InjectionVolumes(np.zeros(2), 1.0, ones, weights=np.array([1.0, -1.0]))
    with pytest.raises(ValueError, match="finite"):
        InjectionVolumes(np.array([0.0, np.nan]), 1.0, ones)
    with pytest.raises(ValueError, match="reach_mpc"):
        InjectionVolumes(np.zeros(2), 0.0, ones)
    # Every injection is inside the reach it was drawn to.
    with pytest.raises(ValueError, match=r"in \(0, reach_mpc"):
        InjectionVolumes(np.zeros(2), 1.0, np.array([0.5, 1.5]))
    with pytest.raises(ValueError, match="reach_mpc"):
        uniform_in_volume(2, -1.0, np.random.default_rng(0))


# ------------------------------------------------------------ a realistic detector

def test_a_realistic_detector_is_unbiased_and_its_injection_interval_covers():
    # At the true threshold, where the chance of a noise-only pass is known, the injection
    # interval alone is the whole error: over replicas, V (the chance subtracted) is unbiased and
    # its interval covers the truth at each level.
    replicas, n, far = 300, 4000, 1e-2
    rng = np.random.default_rng(3)
    truth = realistic_truth(far)
    threshold = stats.norm.isf(far)
    volumes, covered = np.empty(replicas), np.zeros(2)
    for r in range(replicas):
        injections = detected(rng, n, realistic_scores)
        volumes[r] = injections.volume(threshold, chance=far)
        for j, sigma in enumerate((1.0, 2.0)):
            lower, upper = injections.interval(threshold, confidence=sigma_level(sigma), chance=far)
            covered[j] += lower <= truth <= upper
    assert abs(volumes.mean() - truth) <= 3 * volumes.std() / np.sqrt(replicas), (volumes.mean(), truth)
    for j, sigma in enumerate((1.0, 2.0)):
        level = sigma_level(sigma)
        assert abs(covered[j] / replicas - level) <= coverage_tolerance(replicas, level), covered / replicas


def test_without_the_subtraction_the_volume_grows_with_the_reach():
    # A detector that finds every source of SNR 8 or more, and otherwise passes a window as noise
    # does (chance 1e-2): the found volume counts what noise alone finds everywhere in the reach,
    # so it grows with the reach's cube though the detector sees nothing past D / 8; V, the chance
    # subtracted, is the same within any reach past that, the closed form.
    rng = np.random.default_rng(52)
    p = 1e-2
    sharp = lambda rng, snrs: np.where(snrs >= 8.0, 1e3, rng.normal(size=snrs.shape))
    threshold = stats.norm.isf(p)
    found, volume = {}, {}
    for reach in (REACH_MPC / 2, REACH_MPC):  # both past the farthest source of SNR 8, D / 8
        injections = detected(rng, 400_000, sharp, reach)
        found[reach] = injections.found_volume(threshold)
        volume[reach] = injections.volume(threshold, chance=p)
        lower, upper = injections.interval(threshold, sigma_level(2.0), chance=p)
        assert lower <= threshold_detector_volume(8.0) <= upper, (reach, lower, upper)
    assert found[REACH_MPC] / found[REACH_MPC / 2] > 1.5
    assert volume[REACH_MPC] / volume[REACH_MPC / 2] == pytest.approx(1.0, abs=0.1)


def noise_background(rng, num_windows: int, slides: int = 0, label_slides: bool = True) -> Background:
    """
    ``num_windows`` 1 s windows of N(0, 1) noise: independent, or ``slides`` time slides that
    each read the same ``num_windows / slides`` windows again (a slide of k, rolled by k), as a
    score that one detector's excursions dominate reads them: every window's score appears once
    in each slide. ``label_slides`` False labels every slide 0, as a background blind to its
    data's reuse would.
    """
    if slides == 0:
        return Background.independent_windows(rng.normal(size=num_windows), 1.0)
    noise = rng.normal(size=num_windows // slides)
    return Background([ScoreStream(np.roll(noise, k), 1.0, 1.0, 0, k if label_slides else 0)
                       for k in range(1, slides + 1)])


def realistic_curve(rng, allowed: int, far: float, num_injections: int, num_bootstrap: int, slides: int = 0,
                    detector=realistic_scores):
    """The realistic detector's curve at ``far``, from a background (`noise_background`) allowing ``allowed`` triggers."""
    background = noise_background(rng, int(round(allowed / far)), slides)
    injections = detected(rng, num_injections, detector)
    return sensitive_volume_curve(injections, background, [far], rng, num_bootstrap=num_bootstrap)


def joint_misses(curve, truth) -> np.ndarray:
    """(levels, 2): whether the truth is below and above the joint interval at each level."""
    return np.stack([truth < curve.joint_lower_gpc3[:, 0], truth > curve.joint_upper_gpc3[:, 0]], axis=1)


@pytest.mark.parametrize("allowed", [1, 2, 3, 10])
def test_the_joint_interval_covers_the_truth_where_the_background_allows_few_triggers(allowed):
    # The background estimates the threshold from `allowed` triggers; the truth is V at the
    # true FAR's threshold. Where it allows one or two, the loudest trigger bounds no
    # bootstrap replicate's threshold from above, and the joint interval's lower end is
    # the background interval's.
    replicas, far = 300, 1e-2
    rng = np.random.default_rng(10 + allowed)
    truth = realistic_truth(far)
    misses = np.zeros((2, 2))
    for _ in range(replicas):
        curve = realistic_curve(rng, allowed, far, num_injections=2000, num_bootstrap=100)
        assert curve.sigmas == (1.0, 2.0)
        misses += joint_misses(curve, truth)
    assert_misses_within_tails(misses, replicas)


@pytest.mark.parametrize("slides", [0, 16])
def test_the_joint_interval_covers_the_truth_on_time_slides(slides):
    # Sixteen slides of the same windows hold the information of one: a background
    # that counted every slid window as independent would put the threshold 4 times too
    # precisely. Counting its data's reuse, the joint interval covers as it does on
    # independent windows (the same 10 events allowed).
    replicas, far = 300, 1e-2
    rng = np.random.default_rng(30 + slides)
    truth = realistic_truth(far)
    misses = np.zeros((2, 2))
    for _ in range(replicas):
        curve = realistic_curve(rng, 10 * max(slides, 1), far, num_injections=20_000, num_bootstrap=100,
                                slides=slides)
        assert curve.data_reuse == max(slides, 1)
        misses += joint_misses(curve, truth)
    assert_misses_within_tails(misses, replicas)


def test_the_joint_interval_covers_where_few_injections_are_found():
    # A silent detector (injections scored as noise is) at 1e-3 Hz: about four of 4000 injections
    # are found, by their noise, and a bootstrap of so few understates their error; the binomial
    # interval on the found count keeps the joint interval covering V, which, the chance of a
    # noise-only pass subtracted, is 0.
    replicas, far = 200, 1e-3
    rng = np.random.default_rng(44)
    misses = np.zeros((2, 2))
    for _ in range(replicas):
        background = Background.independent_windows(rng.normal(size=10_000), 1.0)
        curve = sensitive_volume_curve(detected(rng, 4000, lambda rng, snrs: rng.normal(size=snrs.shape)),
                                       background, [far], rng, num_bootstrap=100)
        misses += joint_misses(curve, 0.0)
    assert_misses_within_tails(misses, replicas)


def test_the_joint_interval_is_not_the_injection_interval():
    # A short background (10 triggers) and many injections: the injection interval alone,
    # blind to the threshold's error, falls well short; the joint covers.
    replicas, far = 200, 1e-2
    rng = np.random.default_rng(20)
    truth = realistic_truth(far)
    joint = injections_alone = 0
    level = sigma_level(2.0)
    for _ in range(replicas):
        curve = realistic_curve(rng, 10, far, num_injections=20_000, num_bootstrap=100)
        joint += curve.joint_lower_gpc3[1, 0] <= truth <= curve.joint_upper_gpc3[1, 0]
        injections_alone += curve.injection_lower_gpc3[1, 0] <= truth <= curve.injection_upper_gpc3[1, 0]
    tolerance = coverage_tolerance(replicas, level)
    assert joint / replicas >= level - tolerance, joint / replicas
    assert injections_alone / replicas < level - tolerance, injections_alone / replicas


# ---------------------------------------------------- the per-mass reach

#: Chirp masses (detector frame) of the test population, uniform; a source of chirp mass M at chirp
#: distance d_c is at d_c (M / M_ref)^(5/6), and its horizon is HORIZON_MPC (M / M_ref)^(5/6) loudness(M).
CHIRP_MASS_RANGE = (2.0, 40.0)
#: The chirp-distance reach: the loudest source (loudness 1) has SNR 4 at its mass's reach.
CHIRP_REACH_MPC = HORIZON_MPC / 4.0


def loudness(chirp_mass):
    """A source's loudness for its chirp mass, beyond the inspiral's M^(5/6): at most 1, falling with mass."""
    return np.exp(-(np.asarray(chirp_mass) - CHIRP_MASS_RANGE[0]) / 30.0)


def chirp_population(rng, n):
    """
    Sources drawn as the validator draws its volume set: chirp mass from the population,
    chirp distance uniform in volume out to the chirp reach, so each is placed out to its own
    mass's reach R (M / M_ref)^(5/6). Returns each one's SNR, chirp distance, physical distance
    and chirp mass.
    """
    chirp_mass = rng.uniform(*CHIRP_MASS_RANGE, n)
    g = antenna_response(rng.uniform(-1, 1, n), rng.uniform(0, np.pi, n), rng.uniform(0, np.pi, n),
                         rng.uniform(-1, 1, n))
    chirp_distance = uniform_in_volume(n, CHIRP_REACH_MPC, rng)
    scale = (chirp_mass / REFERENCE_CHIRP_MASS_MSUN) ** (5 / 6)
    snrs = HORIZON_MPC * scale * loudness(chirp_mass) * g / (chirp_distance * scale)
    return snrs, chirp_distance, chirp_distance * scale, chirp_mass


def chirp_truth(far_hz: float) -> float:
    """V within each mass's reach at a true FAR, the chance subtracted, by quadrature over mass, sky, orientation and distance."""
    p = far_hz
    threshold = stats.norm.isf(p)
    m, wm = np.polynomial.legendre.leggauss(12)
    m = CHIRP_MASS_RANGE[0] + 0.5 * (m + 1) * (CHIRP_MASS_RANGE[1] - CHIRP_MASS_RANGE[0])
    wm = 0.5 * wm
    response, wg = response_quadrature(10)
    x, wx = np.polynomial.legendre.leggauss(128)
    x, wx = 0.5 * (x + 1), 0.5 * wx
    total = 0.0
    for mass, weight in zip(m, wm):
        snr = HORIZON_MPC * loudness(mass) * response[:, None] / (x[None, :] * CHIRP_REACH_MPC)
        signal = (stats.norm.sf(threshold - snr) - p) / (1 - p)
        within = np.sum(wg[:, None] * wx[None, :] * 3 * x[None, :] ** 2 * signal)
        reach = CHIRP_REACH_MPC * (mass / REFERENCE_CHIRP_MASS_MSUN) ** (5 / 6)
        total += weight * float(sphere_volume_gpc3(reach)) * within
    return total


def test_the_reference_chirp_mass_is_two_neutron_stars():
    assert REFERENCE_CHIRP_MASS_MSUN == pytest.approx(1.4 * 2 ** (-1 / 5), rel=1e-12)
    np.testing.assert_allclose(chirp_weights([30.0, 1.4], [30.0, 1.4]),
                               [(30 * 2 ** (-1 / 5) / REFERENCE_CHIRP_MASS_MSUN) ** 2.5, 1.0], rtol=1e-12)


def test_a_chirp_distance_draw_weighted_by_chirp_mass_is_unbiased_and_covers():
    # Each source out to its own mass's reach, chirp distance uniform in volume: a found injection
    # stands for its mass's reach volume, the chirp reach's times (M / M_ref)^(5/2). Weighted, V
    # is unbiased against the quadrature and its interval covers at each level; unweighted, it is
    # the wrong number.
    replicas, n, far = 200, 4000, 1e-2
    rng = np.random.default_rng(50)
    truth = chirp_truth(far)
    threshold = stats.norm.isf(far)
    volumes, unweighted, covered = np.empty(replicas), np.empty(replicas), np.zeros(2)
    for r in range(replicas):
        snrs, chirp_distance, distance, chirp_mass = chirp_population(rng, n)
        weights = (chirp_mass / REFERENCE_CHIRP_MASS_MSUN) ** 2.5
        assert np.all(distance <= CHIRP_REACH_MPC * weights ** (1 / 3) * (1 + 1e-12))
        scores = realistic_scores(rng, snrs)
        injections = InjectionVolumes(scores, CHIRP_REACH_MPC, chirp_distance, weights)
        volumes[r] = injections.volume(threshold, chance=far)
        unweighted[r] = InjectionVolumes(scores, CHIRP_REACH_MPC, chirp_distance).volume(threshold, chance=far)
        for j, sigma in enumerate((1.0, 2.0)):
            lower, upper = injections.interval(threshold, confidence=sigma_level(sigma), chance=far)
            covered[j] += lower <= truth <= upper
    assert abs(volumes.mean() - truth) <= 3 * volumes.std() / np.sqrt(replicas), (volumes.mean(), truth)
    assert abs(unweighted.mean() - truth) > 10 * volumes.std() / np.sqrt(replicas)
    for j, sigma in enumerate((1.0, 2.0)):
        level = sigma_level(sigma)
        assert abs(covered[j] / replicas - level) <= coverage_tolerance(replicas, level), covered / replicas


def test_the_found_fraction_against_distance_is_counted_exactly():
    # The visual check of the reach: the injections found in each distance bin, of those there;
    # far out, at a reach long enough, the fraction falls to the chance level.
    rng = np.random.default_rng(51)
    injections = detected(rng, 20_000, realistic_scores)
    edges, found, total = found_fraction_by_distance(injections, 2.0, num_bins=10)
    np.testing.assert_allclose(edges, np.linspace(0, REACH_MPC, 11), rtol=1e-12)
    expected_total, _ = np.histogram(injections.distances, edges)
    expected_found, _ = np.histogram(injections.distances[injections.scores >= 2.0], edges)
    np.testing.assert_array_equal(total, expected_total)
    np.testing.assert_array_equal(found, expected_found)
    assert np.all(np.isnan(found_fraction_by_distance(injections, np.nan, num_bins=10)[1]))


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


def test_the_chance_of_passing_a_chosen_threshold_is_a_beta_law():
    # A threshold just above the c-th highest of n exchangeable windows is passed by a fresh
    # window with a chance distributed as Beta(c, n - c + 1), whatever their distribution: the
    # window below it counts. Sixteen slides of the same windows hold one reading's information.
    independent = Background.independent_windows(np.arange(10.0), 1.0)
    threshold = independent.threshold_for_far(0.3).threshold  # just above 6, the fourth highest
    assert independent.window_exceedance(threshold) == 0.3
    np.testing.assert_array_equal(independent.chosen_window_chance(threshold), (4.0, 7.0))
    np.testing.assert_array_equal(independent.chosen_window_chance(0.0), (10.0, 1.0))  # no window below
    slid = Background([ScoreStream(np.roll(np.arange(10.0), k), 1.0, 1.0, 0, k) for k in range(1, 5)])
    threshold = slid.threshold_for_far(0.3).threshold  # 12 allowed: just above 6, twelve windows pass
    assert slid.window_exceedance(threshold) == 0.3
    np.testing.assert_array_equal(slid.chosen_window_chance(threshold), (4.0, 7.0))
    # Over draws of 100 uniform windows, the chance of passing the threshold for three triggers,
    # just above the fourth highest, is exactly Beta(4, 97).
    rng = np.random.default_rng(9)
    chances = np.empty(4000)
    for r in range(chances.size):
        background = Background.independent_windows(rng.random(100), 1.0)
        threshold = background.threshold_for_far(0.03).threshold
        np.testing.assert_array_equal(background.chosen_window_chance(threshold), (4.0, 97.0))
        chances[r] = 1 - threshold
    assert stats.kstest(chances, stats.beta(4, 97).cdf).pvalue > 1e-3


def test_the_curve_reads_volume_at_the_backgrounds_threshold_with_the_chance_subtracted():
    rng = np.random.default_rng(6)
    background = Background.independent_windows(rng.normal(size=20_000), 1.0)
    injections = detected(rng, 20_000, realistic_scores)
    curve = sensitive_volume_curve(injections, background, [1e-1, 1e-2, 1e-3, 1e-5], rng, num_bootstrap=200)
    assert curve.resolved.tolist() == [True, True, True, False]
    assert curve.sigmas == (1.0, 2.0)
    for field in ("joint", "injection", "background"):
        assert getattr(curve, f"{field}_lower_gpc3").shape == (2, 4)
    assert curve.far_lower_hz.shape == curve.far_upper_hz.shape == (2, 4)
    assert np.all(np.isnan([curve.volume_within_reach_gpc3[3], curve.found_gpc3[3], curve.chance[3],
                            curve.chance_gpc3[3]]))
    assert np.all(np.isnan(np.concatenate([curve.joint_lower_gpc3[:, 3], curve.background_lower_gpc3[:, 3],
                                           curve.injection_upper_gpc3[:, 3]])))
    for i in (0, 1, 2):
        threshold = background.threshold_for_far(curve.far_hz[i]).threshold
        chance = float(background.found_by_chance(threshold).mean[0])
        assert curve.chance[i] == pytest.approx(chance, rel=1e-12)
        assert curve.found_gpc3[i] == injections.found_volume(threshold)
        assert curve.volume_within_reach_gpc3[i] == pytest.approx(injections.volume(threshold, chance), rel=1e-12)
        # Noise alone finds the chance of a pass, of the population's whole volume.
        assert curve.chance_gpc3[i] == pytest.approx(chance * injections.population_volume_gpc3, rel=1e-12)
        value = curve.volume_within_reach_gpc3[i]
        for term in ("injection", "background", "joint"):
            lower, upper = getattr(curve, f"{term}_lower_gpc3")[:, i], getattr(curve, f"{term}_upper_gpc3")[:, i]
            assert lower[1] <= lower[0] < value < upper[0] <= upper[1], (term, lower, value, upper)
        assert (curve.far_lower_hz[1, i] <= curve.far_lower_hz[0, i] <= curve.far_upper_hz[0, i]
                <= curve.far_upper_hz[1, i])
        assert value == pytest.approx(realistic_truth(curve.far_hz[i]), rel=0.1)
    # At 0.1 Hz a tenth of the noise windows pass: most of what is found, noise found.
    assert curve.chance_dominated.tolist() == [True, False, False, False]
    assert curve.quotable.tolist() == [False, True, True, False]
    assert curve.chance_fraction == 0.1
    assert curve.volume_within_reach_gpc3[1] > curve.volume_within_reach_gpc3[2]
    np.testing.assert_array_equal(curve.distance_within_reach_mpc, sensitive_distance(curve.volume_within_reach_gpc3))
    assert curve.reach_mpc == REACH_MPC and curve.reach_range_mpc == (REACH_MPC, REACH_MPC)
    assert curve.population_volume_gpc3 == injections.population_volume_gpc3
    assert curve.num_injections == 20_000 and curve.num_bootstrap == 200 and curve.data_reuse == 1


def offset_scores(offset):
    """A detector that needs ``offset`` more SNR than the realistic one."""
    return lambda rng, snrs: snrs - offset + rng.normal(size=snrs.shape)


def test_quote_gives_the_volume_within_the_reach_at_both_levels_and_refuses_what_not_to_quote():
    rng = np.random.default_rng(42)
    background = Background.independent_windows(rng.normal(size=20_000), 1.0)
    curve = sensitive_volume_curve(detected(rng, 20_000, offset_scores(-1.0)), background,
                                   [1e-1, 1e-3, 1e-5], rng, num_bootstrap=50)
    assert curve.chance_dominated[0] and not curve.resolved[2]
    assert curve.quotable.tolist() == [False, True, False]
    quote = curve.quote(1e-3)
    assert quote == VolumeQuote(1e-3, REACH_MPC, (REACH_MPC, REACH_MPC), curve.volume_within_reach_gpc3[1],
                                tuple(curve.joint_lower_gpc3[:, 1]), tuple(curve.joint_upper_gpc3[:, 1]), (1.0, 2.0))
    with pytest.raises(ValueError, match="noise alone finds more than 10%"):
        curve.quote(1e-1)
    with pytest.raises(ValueError, match="cannot resolve"):
        curve.quote(1e-5)
    with pytest.raises(ValueError, match="not one of the curve's target FARs"):
        curve.quote(2e-3)
