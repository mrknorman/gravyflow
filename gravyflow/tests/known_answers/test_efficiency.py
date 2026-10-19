"""
Efficiency against SNR and the ROC of synthetic detectors against their paper
values (`truth`): the binned efficiency the record keeps, its binomial
intervals' coverage at both levels, and the ROC's points and area.
"""

import numpy as np
import pytest
from scipy import special

from gravyflow.src.validate import metrics
from gravyflow.tests.known_answers import truth

DETECTORS = {
    "single": truth.GaussianBank.single(),
    "max-of-8": truth.GaussianBank.max_of(8),
    "bank": truth.GaussianBank((1.0, 0.9, 0.7, 0.5, 0.2)),
    "logistic": truth.LogisticDetector(0.8),
}
EDGES = np.arange(2.0, 21.0, 1.0)
Z = 4.0


def sigma_of(confidence: float) -> float:
    """The level in sigma of a normal law whose central probability is ``confidence``."""
    return float(np.sqrt(2.0) * special.erfinv(confidence))


def bin_truths(d, threshold):
    return np.array([truth.binned_efficiency(d, threshold, a, b) for a, b in zip(EDGES[:-1], EDGES[1:])])


@pytest.mark.parametrize("name", DETECTORS)
def test_the_binned_efficiency_is_the_paper_s(name):
    d, rng = DETECTORS[name], np.random.default_rng(31)
    t = d.threshold_for_noise_sf(1e-4)
    snr = rng.uniform(EDGES[0], EDGES[-1], 60_000)
    curve = metrics.efficiency_curve(snr, d.signal(rng, snr), t, EDGES)
    expected = bin_truths(d, t)
    se = np.sqrt(np.maximum(expected * (1 - expected), 1e-6) / curve.total)
    np.testing.assert_array_less(np.abs(curve.efficiency - expected), Z * se)
    # Each bin's SNR is the mean of its injections': the middle of a bin of uniform SNRs.
    np.testing.assert_allclose(curve.snr_mean, (EDGES[:-1] + EDGES[1:]) / 2, atol=0.02)


@pytest.mark.parametrize("confidence", truth.LEVELS, ids=truth.LEVEL_IDS)
@pytest.mark.parametrize("name", ["single", "logistic"])
def test_the_binned_efficiency_intervals_cover_at_their_level(confidence, name):
    # Pooled over the bins in the curve's transition, where an exact interval is
    # tested (bins at 0 or 1 are covered almost surely, and would hide a failure),
    # and bin by bin.
    d, rng = DETECTORS[name], np.random.default_rng(32)
    replicas, n = 400, 1500
    t = d.threshold_for_noise_sf(1e-4)
    expected = bin_truths(d, t)
    transition = (expected >= 0.05) & (expected <= 0.95)
    covered = []
    for _ in range(replicas):
        snr = rng.uniform(EDGES[0], EDGES[-1], n)
        # One level, given in sigma: every argument after the edges is keyword-only.
        curve = metrics.efficiency_curve(snr, d.signal(rng, snr), t, EDGES, sigmas=(sigma_of(confidence),))
        covered.append((curve.lower[0] <= expected) & (expected <= curve.upper[0]))
    covered = np.array(covered)[:, transition]
    tolerance = truth.coverage_tolerance(confidence, replicas)
    assert covered.mean() >= confidence - tolerance / np.sqrt(transition.sum())
    assert covered.mean(axis=0).min() >= confidence - tolerance


@pytest.mark.parametrize("name", DETECTORS)
@pytest.mark.parametrize("snr", [1.0, 2.5])
def test_the_roc_area_is_the_paper_s(name, snr):
    d, rng = DETECTORS[name], np.random.default_rng(33)
    n = 20_000
    a = truth.auc(d, snr)
    roc = metrics.roc_curve(d.noise(rng, n), d.signal(rng, np.full(n, snr)))
    # Hanley and McNeil's standard error of the Mann-Whitney area.
    q1, q2 = a / (2 - a), 2 * a * a / (1 + a)
    se = np.sqrt((a * (1 - a) + (n - 1) * (q1 - a * a) + (n - 1) * (q2 - a * a)) / (n * n))
    assert abs(roc.auc - a) <= Z * se, (roc.auc, a)


@pytest.mark.parametrize("name", DETECTORS)
def test_every_roc_point_is_the_paper_s_at_its_threshold(name):
    # At each of the curve's thresholds, the false- and true-positive rates are the
    # noise's and the signal's pass chances there.
    d, rng = DETECTORS[name], np.random.default_rng(34)
    n, snr = 20_000, 2.0
    roc = metrics.roc_curve(d.noise(rng, n), d.signal(rng, np.full(n, snr)))
    for target in (0.3, 0.1, 0.01):
        i = int(np.argmin(np.abs(roc.false_positive_rate - target)))
        p_noise = float(d.noise_sf(roc.thresholds[i]))
        p_signal = float(d.signal_sf(roc.thresholds[i], snr))
        assert abs(roc.false_positive_rate[i] - p_noise) <= Z * np.sqrt(p_noise * (1 - p_noise) / n)
        assert abs(roc.true_positive_rate[i] - p_signal) <= Z * np.sqrt(p_signal * (1 - p_signal) / n)
    np.testing.assert_allclose(truth.roc_tpr(d, [0.1], snr), d.signal_sf(d.threshold_for_noise_sf(0.1), snr))
