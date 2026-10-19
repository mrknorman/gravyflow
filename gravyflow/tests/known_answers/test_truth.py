"""
The paper checked against brute force: every truth `truth` states is
reproduced by drawing from its detector or population, or by an independent
closed form. A wrong truth would certify wrong numbers, so it is tested first.
"""

import numpy as np
import pytest
from scipy import special, stats

from gravyflow.tests.known_answers import truth

DETECTORS = {
    "single": truth.GaussianBank.single(),
    "max-of-8": truth.GaussianBank.max_of(8),
    "bank": truth.GaussianBank((1.0, 0.9, 0.7, 0.5, 0.2)),
    "logistic": truth.LogisticDetector(0.8),
}


def within(estimate: float, expected: float, standard_error: float, z: float = 4.0) -> None:
    assert abs(estimate - expected) <= z * standard_error, (
        f"{estimate} is {abs(estimate - expected) / standard_error:.1f} standard errors from {expected}")


def test_levels_are_one_and_two_sigma():
    assert truth.ONE_SIGMA == pytest.approx(0.6826894921, abs=1e-10)
    assert truth.TWO_SIGMA == pytest.approx(0.9544997361, abs=1e-10)


def test_the_single_trial_detector_is_phi_of_snr_less_threshold():
    d = truth.GaussianBank.single()
    snr, t = np.linspace(0, 10, 11), 4.0
    np.testing.assert_allclose(d.signal_sf(t, snr), stats.norm.cdf(snr - t), rtol=1e-12)
    np.testing.assert_allclose(d.noise_sf(t), stats.norm.sf(t), rtol=1e-12)
    # The ROC area of the single trial: Phi(snr / sqrt 2).
    assert truth.auc(d, 1.5) == pytest.approx(stats.norm.cdf(1.5 / np.sqrt(2)), abs=1e-9)


def test_the_max_of_n_trials_closed_forms():
    n, t, snr = 8, 3.0, 4.0
    d = truth.GaussianBank.max_of(n)
    assert d.noise_sf(t) == pytest.approx(1 - stats.norm.cdf(t) ** n, rel=1e-12)
    assert d.signal_sf(t, snr) == pytest.approx(1 - stats.norm.cdf(t - snr) * stats.norm.cdf(t) ** (n - 1),
                                                rel=1e-12)


def test_the_logistic_detector_closed_forms():
    d = truth.LogisticDetector(0.8)
    assert truth.snr_at_efficiency(d, 5.0, 0.5) == pytest.approx(5.0)
    assert truth.snr_at_efficiency(d, 5.0, 0.9) == pytest.approx(5.0 + 0.8 * np.log(9.0))
    assert d.signal_sf(5.0, 5.0 + 0.8 * np.log(9.0)) == pytest.approx(0.9)


@pytest.mark.parametrize("name", DETECTORS)
def test_the_threshold_inverts_the_noise_chance(name):
    d = DETECTORS[name]
    for p in (0.3, 1e-2, 1e-5, 1e-9):
        assert d.noise_sf(d.threshold_for_noise_sf(p)) == pytest.approx(p, rel=1e-9)


@pytest.mark.parametrize("name", DETECTORS)
def test_the_snr_points_invert_the_efficiency(name):
    d = DETECTORS[name]
    for q in (0.5, 0.9):
        assert d.signal_sf(3.0, truth.snr_at_efficiency(d, 3.0, q)) == pytest.approx(q, abs=1e-9)


@pytest.mark.parametrize("name", DETECTORS)
def test_the_pdf_is_the_cdf_s_derivative_and_integrates_to_one(name):
    d = DETECTORS[name]
    x, h = np.linspace(-3, 8, 23), 1e-5
    np.testing.assert_allclose(d.signal_pdf(x, 2.5), (d.signal_cdf(x + h, 2.5) - d.signal_cdf(x - h, 2.5)) / (2 * h),
                               rtol=1e-5, atol=1e-9)
    np.testing.assert_allclose(d.signal_cdf(x, 2.5), 1 - d.signal_sf(x, 2.5), atol=1e-12)


@pytest.mark.parametrize("name", DETECTORS)
def test_draws_match_the_stated_distribution(name):
    d, rng, n = DETECTORS[name], np.random.default_rng(11), 200_000
    t = d.threshold_for_noise_sf(0.05)
    p = d.noise_sf(t)
    within(np.mean(d.noise(rng, n) >= t), p, np.sqrt(p * (1 - p) / n))
    for snr in (1.0, 3.0, 6.0):
        q = float(d.signal_sf(t, snr))
        within(np.mean(d.signal(rng, np.full(n, snr)) >= t), q, np.sqrt(max(q * (1 - q), 1e-12) / n))
    # The AUC: the chance a signal outscores noise.
    a = truth.auc(d, 1.5)
    within(np.mean(d.signal(rng, np.full(n, 1.5)) > d.noise(rng, n)), a, np.sqrt(a * (1 - a) / n))


def test_the_clustered_trigger_rate_matches_brute_force():
    d, rng = truth.GaussianBank.single(), np.random.default_rng(12)
    t, n, k, reps = 1.0, 50, 3, 20_000
    s = d.noise(rng, reps * n).reshape(reps, n) >= t
    opens = s.copy()
    for lag in range(1, k + 1):
        opens[:, lag:] &= ~s[:, :-lag]
    counts = opens.sum(axis=1)
    rate = truth.clustered_trigger_rate_hz(d, t, n, k, 1.0, 1.0)
    within(counts.mean() / n, rate, counts.std() / np.sqrt(reps) / n)


def test_binned_and_rolling_efficiency_are_window_averages():
    d = truth.GaussianBank.single()
    # Over a bin the efficiency is the mean of Phi(snr - t): its integral is closed form.
    t, lo, hi = 5.0, 3.0, 7.0
    antiderivative = lambda x: x * stats.norm.cdf(x) + stats.norm.pdf(x)  # noqa: E731
    exact = (antiderivative(hi - t) - antiderivative(lo - t)) / (hi - lo)
    assert truth.binned_efficiency(d, t, lo, hi) == pytest.approx(exact, abs=1e-10)
    # A rolling window cut by the range's end averages only what is inside.
    assert truth.rolling_efficiency(d, t, [2.2], 0.5, (2.0, 20.0))[0] == pytest.approx(
        truth.binned_efficiency(d, t, 2.0, 2.7))


POPULATION = truth.Population(amplitudes=(400.0, 1000.0, 2500.0), mass_fractions=(0.5, 0.3, 0.2))


def test_the_step_detector_s_volume_is_closed_form():
    # Quadrature against the closed form: a noiseless detector at SNR 8.
    for rho in (2.0, 8.0):
        assert truth.volume_within_reach_gpc3(truth.StepDetector(), rho, POPULATION) == pytest.approx(
            truth.step_volume_gpc3(POPULATION, rho), rel=1e-6)


@pytest.mark.parametrize("common_reach", [None, 600.0])
def test_the_volume_within_reach_matches_an_injection_campaign(common_reach):
    # Brute force: draw the campaign as the standard draws it (uniform in volume out to
    # each mass's reach, weighted by its sphere), and count what is found.
    d, rng, n = truth.GaussianBank.single(), np.random.default_rng(13), 400_000
    t = 5.0
    mass, distance, g, snr = POPULATION.draw(rng, n, common_reach)
    reach = POPULATION.reaches_mpc[mass] if common_reach is None else np.full(n, common_reach)
    found = d.signal(rng, snr) >= t
    per = truth.sphere_volume_gpc3(reach) * found
    expected = truth.volume_within_reach_gpc3(d, t, POPULATION, common_reach)
    within(per.mean(), expected, per.std() / np.sqrt(n))


def test_the_per_mass_found_fraction_does_not_depend_on_mass():
    # With R_m = A_m / reach_snr every mass sees the same SNR field per unit radius: V_m scales as R_m^3.
    d, t = truth.GaussianBank.single(), 4.0
    one = [truth.volume_within_reach_gpc3(d, t, truth.Population((a,), (1.0,))) for a in (300.0, 900.0)]
    assert one[1] / one[0] == pytest.approx(27.0, rel=1e-8)


def test_the_logistic_efficiency_is_a_sigmoid():
    d = truth.LogisticDetector(1.3)
    snr = np.linspace(0, 12, 13)
    np.testing.assert_allclose(d.signal_sf(6.0, snr), special.expit((snr - 6.0) / 1.3))
