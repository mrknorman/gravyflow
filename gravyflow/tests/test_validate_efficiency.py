"""
Efficiency against SNR at a target false-alarm rate, against a detector whose
efficiency is known in closed form (score = SNR + N(0, 1), noise N(0, 1)): the
binned record, the owner's rolling curve and the sigmoid fit, each with the
chance of a noise-only pass subtracted, and their bands at 1 and 2 sigma
carrying the threshold's own uncertainty where the background is thin.
"""

import numpy as np
import pytest
from scipy import stats

from gravyflow.src.validate.efficiency import efficiency_against_snr
from gravyflow.src.validate.metrics import (
    DEFAULT_SIGMAS,
    Background,
    efficiency_curve,
    fit_efficiency_sigmoid,
    rolling_efficiency_curve,
    sigma_level,
)

SNR_RANGE = (0.0, 20.0)
EDGES = np.arange(0.0, 21.0, 1.0)


def signal_efficiency(snr, far_hz):
    """The truth at a true FAR: the chance an injection of this SNR passes, less a noise-only pass, over 1 - p."""
    p = far_hz
    threshold = stats.norm.isf(p)
    return (stats.norm.sf(threshold - np.asarray(snr)) - p) / (1 - p)


def window_truth(centres, half_width, far_hz):
    """The signal efficiency averaged over each window of uniform SNR (the rolling curve's estimand)."""
    x, w = np.polynomial.legendre.leggauss(64)
    snr = np.asarray(centres)[:, None] + half_width * x[None, :]
    return np.sum(0.5 * w[None, :] * signal_efficiency(snr, far_hz), axis=1)


def draw(rng, num_injections, num_windows):
    snrs = rng.uniform(*SNR_RANGE, num_injections)
    return snrs, snrs + rng.normal(size=num_injections), Background.independent_windows(rng.normal(size=num_windows), 1.0)


def test_the_curves_at_a_target_are_the_fixed_threshold_curves_at_its_threshold():
    # At each target the point estimates are the fixed-threshold curves at the background's
    # threshold, with the chance of a noise-only pass at that threshold subtracted.
    rng = np.random.default_rng(1)
    snrs, scores, background = draw(rng, 20_000, 20_000)
    results = efficiency_against_snr(snrs, scores, background, [1e-2, 1e-3, 1e-6], rng, EDGES, SNR_RANGE,
                                     num_bootstrap=100)
    assert [r.far_hz for r in results] == [1e-2, 1e-3, 1e-6]
    for result in results[:2]:
        t = result.threshold.threshold
        chance = background.found_by_chance(t)
        binned = efficiency_curve(snrs, scores, t, EDGES, chance)
        rolling = rolling_efficiency_curve(snrs, scores, t, SNR_RANGE, chance=chance)
        np.testing.assert_array_equal(result.binned.efficiency, binned.efficiency)
        np.testing.assert_array_equal(result.rolling.efficiency, rolling.efficiency)
        np.testing.assert_array_equal(result.rolling.snr, rolling.snr)
        fit = fit_efficiency_sigmoid(snrs, scores, t, chance, SNR_RANGE, rng, 10)
        assert result.fit.snr_50 == pytest.approx(fit.snr_50, rel=1e-9)
        assert result.fit.snr_90 == pytest.approx(fit.snr_90, rel=1e-9)
        # The bands: both levels, nested, around the point, at least as wide as the binomial ones.
        for curve, fixed in ((result.binned, binned), (result.rolling, rolling)):
            assert curve.sigmas == DEFAULT_SIGMAS and curve.lower.shape[0] == 2
            ok = np.isfinite(curve.efficiency)
            assert np.all(curve.lower[1, ok] <= curve.lower[0, ok]) and np.all(curve.upper[0, ok] <= curve.upper[1, ok])
            assert np.all(curve.lower[:, ok] <= fixed.lower[:, ok]) and np.all(curve.upper[:, ok] >= fixed.upper[:, ok])
        assert np.all(result.fit.lower[0] <= [result.fit.snr_50, result.fit.snr_90])
        assert np.all([result.fit.snr_50, result.fit.snr_90] <= result.fit.upper[0])
        assert result.fit.snr_50 == pytest.approx(stats.norm.isf(result.far_hz), abs=0.3)
    unresolved = results[2]
    assert not unresolved.threshold.resolved
    assert np.all(np.isnan(unresolved.rolling.efficiency)) and np.isnan(unresolved.fit.snr_50)


@pytest.mark.parametrize("allowed", [3, 30])
def test_the_bands_cover_where_the_background_is_thin(allowed):
    # A background allowing 3 triggers puts the threshold, and so the whole curve, where its
    # few loudest windows fall: the binomial band alone misses the transition. With the
    # threshold's uncertainty joined in (a bootstrap over the background and the injections),
    # the pointwise bands cover the truth at the true FAR's threshold at each level, through the
    # transition, binned and rolling.
    replicas, far = 200, 1e-2
    rng = np.random.default_rng(10 + allowed)
    centre = stats.norm.isf(far)
    points = centre + np.array([-1.5, -0.5, 0.0, 0.5, 1.5])
    covered = {"binned": np.zeros((2, 2)), "rolling": np.zeros((2, points.size))}
    edges = np.array([centre - 2.0, centre - 1.0, centre, centre + 1.0, centre + 2.0])
    bins_truth = np.array([window_truth([0.5 * (a + b)], 0.5 * (b - a), far)[0] for a, b in zip(edges[1:-2],
                                                                                                edges[2:-1])])
    for _ in range(replicas):
        snrs, scores, background = draw(rng, 5000, int(round(allowed / far)))
        (result,) = efficiency_against_snr(snrs, scores, background, [far], rng, edges, SNR_RANGE,
                                           num_bootstrap=200)
        rolling = result.rolling
        index = [int(np.argmin(np.abs(rolling.snr - p))) for p in points]
        truth = window_truth(rolling.snr[index], rolling.half_width, far)
        covered["rolling"] += (rolling.lower[:, index] <= truth) & (truth <= rolling.upper[:, index])
        binned = result.binned
        covered["binned"] += (binned.lower[:, 1:3] <= bins_truth) & (bins_truth <= binned.upper[:, 1:3])
    for j, sigma in enumerate(DEFAULT_SIGMAS):
        level = sigma_level(sigma)
        tolerance = 3 * np.sqrt(level * (1 - level) / replicas)
        for name, hits in covered.items():
            assert np.all(hits[j] / replicas >= level - tolerance), (name, sigma, hits / replicas)
