"""
The detection metrics against independent references: brute-force counts,
scipy's Mann-Whitney U, binomial and Poisson distributions, and synthetic
streams with known trigger rates.
"""

import numpy as np
import pytest
from scipy import stats

from gravyflow.src.validate.metrics import (
    Background,
    ScoreStream,
    binomial_interval,
    efficiency_curve,
    logit_scores,
    poisson_interval,
    roc_curve,
)


# ------------------------------------------------------------------ helpers

def naive_trigger_count(streams, cluster_seconds, threshold):
    """Cluster above-threshold windows by the gap rule, one stream at a time."""
    total = 0
    for scores, stride in streams:
        last = None
        for i, s in enumerate(scores):
            if s >= threshold:
                if last is None or (i - last) * stride > cluster_seconds + 1e-12:
                    total += 1
                last = i
    return total


def random_streams(rng, num_streams=4):
    streams = []
    for _ in range(num_streams):
        n = int(rng.integers(1, 60))
        stride = float(rng.choice([0.25, 0.5, 1.0]))
        # Small integers make ties common.
        streams.append((rng.integers(0, 6, n).astype(float), stride))
    return streams


# ------------------------------------------------------------------- scores

def test_logit_scores_read_two_class_and_single_logits():
    two = np.array([[0.0, 2.0], [1.0, -1.0]])
    np.testing.assert_allclose(logit_scores(two), [2.0, -2.0])
    np.testing.assert_allclose(logit_scores(np.array([[3.0], [-1.0]])), [3.0, -1.0])
    np.testing.assert_allclose(logit_scores(np.array([0.5, 1.5])), [0.5, 1.5])
    with pytest.raises(ValueError):
        logit_scores(np.zeros((2, 3)))
    with pytest.raises(ValueError):
        logit_scores(np.array([np.nan, 1.0]))


def test_logit_scores_read_one_named_output():
    np.testing.assert_allclose(logit_scores({"INJECTION_PRESENT": np.array([[3.0], [-1.0]])}), [3.0, -1.0])
    with pytest.raises(ValueError, match="2 named outputs"):
        logit_scores({"a": np.zeros((2, 1)), "b": np.zeros((2, 1))})


def test_logits_do_not_saturate_where_softmax_ties():
    logits = np.array([[0.0, 40.0], [0.0, 41.0]], dtype=np.float32)
    softmax = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    assert softmax[0, 1] == softmax[1, 1] == np.float32(1.0)  # tied in probability
    scores = logit_scores(logits)
    assert scores[1] > scores[0]                               # ranked on logits


# --------------------------------------------------------------------- ROC

@pytest.mark.parametrize("seed", [0, 1, 2])
def test_roc_auc_matches_mann_whitney_with_ties(seed):
    rng = np.random.default_rng(seed)
    neg = rng.integers(0, 8, 300).astype(float)
    pos = rng.integers(2, 10, 200).astype(float)
    roc = roc_curve(neg, pos)
    u = stats.mannwhitneyu(pos, neg).statistic / (pos.size * neg.size)
    assert roc.auc == pytest.approx(u, abs=1e-12)


def test_roc_points_are_exact_counts():
    rng = np.random.default_rng(3)
    neg, pos = rng.normal(size=50), rng.normal(1.0, size=40)
    roc = roc_curve(neg, pos)
    assert (roc.false_positive_rate[0], roc.true_positive_rate[0]) == (0.0, 0.0)
    assert (roc.false_positive_rate[-1], roc.true_positive_rate[-1]) == (1.0, 1.0)
    for t, fpr, tpr in zip(roc.thresholds, roc.false_positive_rate, roc.true_positive_rate):
        assert fpr == np.mean(neg >= t) and tpr == np.mean(pos >= t)


def test_roc_auc_is_one_for_separable_saturated_scores():
    # The case a fixed threshold grid reports as AUC 0.000 (GF-B71).
    rng = np.random.default_rng(1)
    noise = rng.uniform(0.998, 0.9999, 4000).astype(np.float32)
    signal = np.ones(4000, dtype=np.float32)
    assert roc_curve(noise, signal).auc == 1.0
    assert roc_curve(signal, noise).auc == 0.0
    assert roc_curve(np.ones(10), np.ones(10)).auc == 0.5


# --------------------------------------------------------------- intervals

@pytest.mark.parametrize("confidence", [0.68, 0.9, 0.95])
@pytest.mark.parametrize("count", [0, 1, 3, 10, 57])
def test_poisson_interval_is_garwood(count, confidence):
    lower, upper = poisson_interval(count, confidence)
    alpha = 1 - confidence
    # Upper bound: P(X <= k | mu = upper) = alpha / 2.
    assert stats.poisson.cdf(count, upper) == pytest.approx(alpha / 2, rel=1e-9)
    if count == 0:
        assert lower == 0
    else:
        # Lower bound: P(X >= k | mu = lower) = alpha / 2.
        assert stats.poisson.sf(count - 1, lower) == pytest.approx(alpha / 2, rel=1e-9)


@pytest.mark.parametrize("method,scipy_method", [("clopper-pearson", "exact"), ("wilson", "wilson")])
@pytest.mark.parametrize("k,n", [(0, 10), (3, 10), (10, 10), (17, 40), (499, 1000)])
def test_binomial_interval_matches_scipy(method, scipy_method, k, n):
    lower, upper = binomial_interval(k, n, 0.9, method)
    reference = stats.binomtest(k, n).proportion_ci(confidence_level=0.9, method=scipy_method)
    assert lower == pytest.approx(reference.low, abs=1e-12)
    assert upper == pytest.approx(reference.high, abs=1e-12)


def test_binomial_interval_is_nan_for_an_empty_bin():
    lower, upper = binomial_interval(0, 0)
    assert np.isnan(lower) and np.isnan(upper)


# ------------------------------------------------------------- FAR, counts

def test_independent_windows_count_every_passing_window():
    rng = np.random.default_rng(4)
    scores = rng.integers(0, 20, 1000).astype(float)
    bg = Background.independent_windows(scores, window_seconds=1.0)
    assert bg.livetime_seconds == 1000.0
    for t in np.unique(scores):
        assert bg.trigger_count(t) == np.count_nonzero(scores >= t)
        assert bg.far_count(t) == np.count_nonzero(scores >= t)
        assert bg.far(t).rate_hz == np.count_nonzero(scores >= t) / 1000.0


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("cluster_seconds", [0.0, 0.5, 1.0, 2.5])
def test_clustered_count_matches_a_naive_reference(seed, cluster_seconds):
    rng = np.random.default_rng(seed)
    raw = random_streams(rng)
    bg = Background([ScoreStream(s, stride, 1.0) for s, stride in raw], cluster_seconds)
    thresholds = np.unique(np.concatenate([s for s, _ in raw] + [[-1.0, 99.0]]))
    for t in thresholds:
        assert bg.trigger_count(t) == naive_trigger_count(raw, cluster_seconds, t)
    # The FAR count is the running maximum of the literal count from above.
    naive = [naive_trigger_count(raw, cluster_seconds, t) for t in thresholds]
    for i, t in enumerate(thresholds):
        assert bg.far_count(t) == max(naive[i:])


def test_a_long_excursion_is_one_trigger():
    scores = np.r_[np.zeros(10), np.full(30, 5.0), np.zeros(10)]
    bg = Background([ScoreStream(scores, 0.25, 1.0)], cluster_seconds=0.5)
    assert bg.trigger_count(5.0) == 1
    assert bg.livetime_seconds == 49 * 0.25 + 1.0


def test_clustering_merges_when_the_threshold_falls():
    # Two peaks bridged by a medium plateau: 2 triggers at 9, 1 at 5.
    scores = np.array([0, 9, 5, 5, 5, 9, 0], dtype=float)
    bg = Background([ScoreStream(scores, 1.0, 1.0)], cluster_seconds=1.0)
    assert bg.trigger_count(9.0) == 2 and bg.trigger_count(5.0) == 1
    assert bg.far_count(5.0) == 2   # the envelope never falls as the threshold falls


@pytest.mark.parametrize("seed", range(5))
def test_threshold_for_far_inverts_the_far(seed):
    rng = np.random.default_rng(seed)
    raw = random_streams(rng, num_streams=6)
    bg = Background([ScoreStream(s, stride, 1.0) for s, stride in raw], cluster_seconds=1.0)
    for target in np.geomspace(1.0 / bg.livetime_seconds, 2.0, 25):
        result = bg.threshold_for_far(target)
        assert result.resolved
        allowed = np.floor(target * bg.livetime_seconds * (1 + 1e-12))
        assert result.achieved.count == bg.far_count(result.threshold) <= allowed
        assert result.achieved.rate_hz <= target * (1 + 1e-12)
        if np.isfinite(result.threshold):
            # Anything below the threshold exceeds the target.
            assert bg.far_count(np.nextafter(result.threshold, -np.inf)) > allowed


def test_unreachable_far_is_unresolved_never_a_rate():
    # GF-B70: the old code returned a rate as the score threshold.
    scores = np.random.default_rng(0).normal(size=1000)
    bg = Background.independent_windows(scores, 1.0)
    low = bg.threshold_for_far(1e-4)
    assert not low.resolved and np.isnan(low.threshold)
    high = bg.threshold_for_far(10.0)
    assert high.resolved and high.threshold == -np.inf and high.achieved.count == 1000


def test_ties_above_the_allowance_move_the_threshold_above_them():
    # GF-B69: 50 noise windows tie at the top; at 1e-3 Hz over 10^4 s only 10 may pass.
    noise = np.zeros(10_000)
    noise[:50] = 1.0
    bg = Background.independent_windows(noise, 1.0)
    result = bg.threshold_for_far(1e-3)
    assert result.threshold > 1.0 and result.achieved.count == 0
    # 1e-2 Hz allows 100: the tie at 1.0 fits, the next value down (0) does not.
    result = bg.threshold_for_far(1e-2)
    assert 0.0 < result.threshold <= 1.0 and result.achieved.count == 50
    # Efficiency and FAR use one comparison (>=): a candidate on the threshold passes.
    signal = np.full(7, result.threshold)
    curve = efficiency_curve(np.full(7, 10.0), signal, result.threshold, [5, 15])
    assert curve.efficiency[0] == 1.0


def test_far_curve_matches_point_queries():
    rng = np.random.default_rng(9)
    bg = Background([ScoreStream(rng.normal(size=500), 0.5, 1.0)], cluster_seconds=1.0)
    curve = bg.far_curve()
    assert np.all(np.diff(curve.thresholds) < 0) and np.all(np.diff(curve.counts) >= 0)
    for t, c in zip(curve.thresholds[::37], curve.counts[::37]):
        assert bg.far(t).count == c


@pytest.mark.parametrize("rate_hz", [0.02, 0.1])
def test_known_rate_streams_recover_the_rate_and_cover_it(rate_hz):
    """Excursions arriving as a Poisson process are counted one each, and the
    90% interval covers the true mean in at least 90% of runs (Garwood is
    conservative)."""
    rng = np.random.default_rng(11)
    stride, cluster, duration = 0.25, 1.0, 2000.0
    n = int(duration / stride)
    slot = 12                       # 3 s slots: longer than an excursion plus the cluster window
    mean = rate_hz * duration       # the true expected number of triggers
    runs, covered, counts = 300, 0, []
    for _ in range(runs):
        scores = rng.uniform(-5.0, 0.0, n)                      # quiet background below 0
        k = rng.poisson(mean)
        for s in rng.choice(n // slot - 1, size=k, replace=False) * slot:
            scores[s:s + int(rng.integers(1, 7))] = rng.uniform(5.0, 9.0)   # 1-6 windows above 5
        far = Background([ScoreStream(scores, stride, 1.0)], cluster).far(5.0)
        assert far.count == k
        counts.append(k)
        covered += far.lower_hz * far.livetime_seconds <= mean <= far.upper_hz * far.livetime_seconds
    assert np.mean(counts) / duration == pytest.approx(rate_hz, rel=0.1)
    assert 0.88 <= covered / runs <= 0.99


def test_by_block_partitions_the_livetime():
    streams = [ScoreStream(np.arange(5.0), 1.0, 1.0, block="a"),
               ScoreStream(np.arange(3.0), 1.0, 1.0, block="b"),
               ScoreStream(np.arange(4.0), 1.0, 1.0, block="a")]
    blocks = Background(streams).by_block()
    assert set(blocks) == {"a", "b"}
    assert blocks["a"].livetime_seconds + blocks["b"].livetime_seconds == Background(streams).livetime_seconds


# --------------------------------------------------------------- efficiency

def test_efficiency_curve_counts_bins_exactly():
    rng = np.random.default_rng(5)
    snrs = rng.uniform(0, 20, 5000)
    scores = snrs + rng.normal(0, 3, 5000)
    edges = np.arange(0, 21, 2.0)
    curve = efficiency_curve(snrs, scores, 8.0, edges)
    for i in range(edges.size - 1):
        inside = (snrs >= edges[i]) & ((snrs < edges[i + 1]) | (i == edges.size - 2))
        assert curve.total[i] == inside.sum()
        assert curve.detected[i] == (inside & (scores >= 8.0)).sum()
        assert curve.snr_mean[i] == pytest.approx(snrs[inside].mean())
    assert np.all((curve.lower <= curve.efficiency) & (curve.efficiency <= curve.upper))


def test_efficiency_at_an_unresolved_far_is_nan():
    curve = efficiency_curve([10.0, 11.0], [1.0, 2.0], float("nan"), [5, 15])
    assert np.all(np.isnan(curve.efficiency)) and curve.total[0] == 2
