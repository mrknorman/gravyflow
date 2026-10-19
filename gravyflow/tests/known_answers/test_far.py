"""
False-alarm rates of synthetic detectors against their paper values
(`truth`): independent windows, the threshold for a target rate, clustered
streams, and time slides as the validator builds them (`noise_background`).
Seeds are fixed; tolerances are four Monte Carlo standard errors, set from the
paper, never tuned to a seed.
"""

import numpy as np
import pytest
from scipy import special, stats

from gravyflow.src.validate import metrics
from gravyflow.src.validate.bank import noise_background
from gravyflow.tests.known_answers import truth

DETECTORS = {
    "single": truth.GaussianBank.single(),
    "max-of-8": truth.GaussianBank.max_of(8),
    "bank": truth.GaussianBank((1.0, 0.9, 0.7, 0.5, 0.2)),
    "logistic": truth.LogisticDetector(0.8),
}
Z = 4.0


@pytest.mark.parametrize("name", DETECTORS)
def test_the_far_of_independent_windows_is_the_window_pass_chance_over_its_length(name):
    d, rng = DETECTORS[name], np.random.default_rng(21)
    n, window = 400_000, 0.5
    background = metrics.Background.independent_windows(d.noise(rng, n), window)
    assert background.livetime_seconds == n * window
    for p in (1e-2, 1e-4):
        t = d.threshold_for_noise_sf(p)
        estimate = background.far(t)
        expected = truth.far_hz(d, t, window)
        assert abs(estimate.rate_hz - expected) <= Z * np.sqrt(n * p) / background.livetime_seconds, (
            estimate.rate_hz, expected)


@pytest.mark.parametrize("confidence", truth.LEVELS, ids=truth.LEVEL_IDS)
@pytest.mark.parametrize("mean_count", [4.0, 40.0])
def test_the_far_interval_covers_the_true_rate_at_its_level(confidence, mean_count):
    d, rng = truth.GaussianBank.single(), np.random.default_rng(22)
    replicas, n = 600, 20_000
    t = d.threshold_for_noise_sf(mean_count / n)
    true = truth.far_hz(d, t, 1.0)
    covered = []
    for _ in range(replicas):
        f = metrics.Background.independent_windows(d.noise(rng, n), 1.0).far(t, confidence)
        covered.append(f.lower_hz <= true <= f.upper_hz)
    # Garwood's interval is conservative on a binomial count: coverage at least nominal.
    assert np.mean(covered) >= confidence - truth.coverage_tolerance(confidence, replicas)


@pytest.mark.parametrize("confidence", truth.LEVELS, ids=truth.LEVEL_IDS)
@pytest.mark.parametrize("allowed", [1, 5])
def test_the_threshold_for_a_target_far_is_the_paper_s_order_statistic(confidence, allowed):
    # Paper: the threshold for a target allowing k triggers is the k-th loudest of n
    # windows (the order statistic whose true FAR is the target's on average), so n times
    # its true pass chance is Gamma(k) (to O(1/n)): its mean is k, and the achieved
    # interval (k counted) covers the true FAR with chance
    # P(G_k^-1(a/2) <= Gamma(k) <= G_(k+1)^-1(1 - a/2)).
    d, rng = truth.GaussianBank.single(), np.random.default_rng(23 + allowed)
    replicas, n = 600, 20_000
    scaled, covered = [], []
    for _ in range(replicas):
        background = metrics.Background.independent_windows(d.noise(rng, n), 1.0)
        target = background.threshold_for_far(allowed / n, confidence)
        assert target.resolved and target.achieved.count == allowed
        true = truth.far_hz(d, target.threshold, 1.0)
        scaled.append(n * float(d.noise_sf(target.threshold)))
        covered.append(target.achieved.lower_hz <= true <= target.achieved.upper_hz)
    k, a = allowed, 1 - confidence
    assert abs(np.mean(scaled) - k) <= Z * np.sqrt(k) / np.sqrt(replicas)
    paper = (stats.gamma.cdf(special.gammaincinv(k + 1, 1 - a / 2), k)
             - stats.gamma.cdf(special.gammaincinv(k, a / 2), k))
    assert abs(np.mean(covered) - paper) <= truth.coverage_tolerance(paper, replicas)


@pytest.mark.parametrize("cluster_windows", [1, 3])
@pytest.mark.parametrize("pass_chance", [1e-3, 5e-2])
def test_a_clustered_stream_counts_the_paper_s_triggers(cluster_windows, pass_chance):
    # A passing window within the cluster time of an earlier passing one joins its
    # trigger: the mean count is sum_i p (1 - p)^min(i, k), and the FAR's envelope
    # (the most triggers at any higher threshold) adds nothing measurable.
    d, rng = truth.GaussianBank.single(), np.random.default_rng(25)
    replicas, n, stride = 400, 5_000, 0.25
    t = d.threshold_for_noise_sf(pass_chance)
    expected = truth.clustered_trigger_rate_hz(d, t, n, cluster_windows, stride, stride)
    counts, fars = [], []
    for _ in range(replicas):
        background = metrics.Background([metrics.ScoreStream(d.noise(rng, n), stride, stride)],
                                        cluster_seconds=cluster_windows * stride)
        counts.append(int(background.trigger_count(t)))
        fars.append(background.far(t).rate_hz)
    livetime = (n - 1) * stride + stride
    rates = np.array(counts) / livetime
    se = max(rates.std(), np.sqrt(expected / livetime)) / np.sqrt(replicas)
    assert abs(rates.mean() - expected) <= Z * se, (rates.mean(), expected)
    assert abs(np.mean(fars) - expected) <= Z * se, (np.mean(fars), expected)


def slid_background(rng, num_windows: int, num_slides: int, combine):
    """
    Two detectors' per-window statistics over one stretch, slid as the validator
    slides them (slide k pairs h_i with l_(i+k): num_windows - k whole windows, no
    wrap), scored by ``combine`` and read back as the validator reads them.
    """
    h, l = rng.normal(size=num_windows), rng.normal(size=num_windows)
    lengths = np.array([num_windows - k for k in range(1, num_slides + 1)])
    scores = np.concatenate([combine(h[:num_windows - k], l[k:]) for k in range(1, num_slides + 1)])
    streams = {"lengths": lengths, "slides": np.arange(1, num_slides + 1), "blocks": np.zeros(num_slides, int)}
    return noise_background(scores, streams, 1.0, None), lengths


def test_time_slides_count_each_slide_s_own_livetime():
    # Each slide is scored on its own whole windows: slide k of a stretch of N windows
    # holds N - k, and the background's livetime is their sum, exactly.
    rng = np.random.default_rng(26)
    background, lengths = slid_background(rng, 1000, 16, lambda h, l: (h + l) / np.sqrt(2))
    assert background.livetime_seconds == float(np.sum(1000 - np.arange(1, 17)))
    assert background.num_windows == lengths.sum()
    assert sum(b.livetime_seconds for b in background.by_block().values()) == background.livetime_seconds


@pytest.mark.parametrize("combine", ["sum", "max"])
def test_time_slides_estimate_the_true_far_without_bias(combine):
    # Reused data is counted once per slide, and that is right for the rate: every slid
    # window's score has the noise's distribution, so the mean count is the paper's,
    # whatever the correlation between slides. (Its spread is not Poisson's when one
    # detector dominates, so its interval's coverage is pending: `pending_time_slide_coverage`.)
    rng = np.random.default_rng(27)
    replicas, num_windows, num_slides = 300, 2000, 8
    if combine == "sum":
        fn, d = (lambda h, l: (h + l) / np.sqrt(2)), truth.GaussianBank.single()
    else:
        fn, d = np.maximum, truth.GaussianBank((0.0, 0.0))
    lengths = num_windows - np.arange(1, num_slides + 1)
    t = d.threshold_for_noise_sf(30.0 / lengths.sum())
    expected = sum(truth.clustered_trigger_rate_hz(d, t, int(m), 1, 1.0, 1.0) * m for m in lengths) / lengths.sum()
    rates = np.array([slid_background(rng, num_windows, num_slides, fn)[0].far(t).rate_hz
                      for _ in range(replicas)])
    assert abs(rates.mean() - expected) <= Z * rates.std() / np.sqrt(replicas), (rates.mean(), expected)
