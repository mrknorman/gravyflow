"""
The detection metrics against independent references: brute-force counts,
scipy's Mann-Whitney U, binomial and Poisson distributions, and synthetic
streams with known trigger rates.
"""

import numpy as np
import pytest
from scipy import stats

from gravyflow.src.validate.metrics import (
    DEFAULT_SIGMAS,
    ROLLING_HALF_WIDTH,
    ROLLING_MIN_INJECTIONS,
    ROLLING_POINTS,
    Background,
    ChanceLaw,
    ScoreStream,
    binomial_interval,
    efficiency_curve,
    fit_efficiency_sigmoid,
    logit_scores,
    poisson_interval,
    rolling_efficiency_curve,
    roc_curve,
    sigma_level,
    subtract_chance,
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
    # Exact: a difference or a copy of small dyadic values has no rounding.
    np.testing.assert_array_equal(logit_scores(two), [2.0, -2.0])
    np.testing.assert_array_equal(logit_scores(np.array([[3.0], [-1.0]])), [3.0, -1.0])
    np.testing.assert_array_equal(logit_scores(np.array([0.5, 1.5])), [0.5, 1.5])
    with pytest.raises(ValueError, match=r"shape \(2, 3\)"):
        logit_scores(np.zeros((2, 3)))
    with pytest.raises(ValueError, match="1 non-finite"):
        logit_scores(np.array([np.nan, 1.0]))


def test_logit_scores_read_one_named_output():
    np.testing.assert_array_equal(logit_scores({"INJECTION_PRESENT": np.array([[3.0], [-1.0]])}), [3.0, -1.0])
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
    # Representation: both are ratios of integer counts of at most 6e4 pairs, so 1e-12 is
    # a few hundred float64 ulp of a value near 1.
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
    # The case a fixed threshold grid reports as AUC 0.000.
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
    # rel=1e-9 is the reference kind: the interval is a root of this cdf, solved to ~1e-10.
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
    # Reference: scipy's own root solves are good to ~1e-12 on probabilities.
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
            # The threshold is the allowed-th loudest trigger: just above it, fewer pass. Where a tie
            # jumps the count past the allowance there, it is just above the tie: below it, too many pass.
            if result.achieved.count == allowed:
                assert bg.far_count(np.nextafter(result.threshold, np.inf)) < allowed
            else:
                assert bg.far_count(np.nextafter(result.threshold, -np.inf)) > allowed


def test_unreachable_far_is_unresolved_never_a_rate():
    # The old code returned a rate as the score threshold.
    scores = np.random.default_rng(0).normal(size=1000)
    bg = Background.independent_windows(scores, 1.0)
    low = bg.threshold_for_far(1e-4)
    assert not low.resolved and np.isnan(low.threshold)
    high = bg.threshold_for_far(10.0)
    assert high.resolved and high.threshold == -np.inf and high.achieved.count == 1000


def test_ties_above_the_allowance_move_the_threshold_above_them():
    # 50 noise windows tie at the top; at 1e-3 Hz over 10^4 s only 10 may pass.
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
    """Excursions arriving as a Poisson process are counted one each (exactly),
    and the 90% interval covers the true mean in about 90% of runs or more
    (Garwood is conservative)."""
    rng = np.random.default_rng(11)
    stride, cluster, duration = 0.25, 1.0, 2000.0
    n = int(duration / stride)
    slot = 12                       # 3 s slots: longer than an excursion plus the cluster window
    mean = rate_hz * duration       # the true expected number of triggers
    runs, covered = 300, 0
    for _ in range(runs):
        scores = rng.uniform(-5.0, 0.0, n)                      # quiet background below 0
        k = rng.poisson(mean)
        for s in rng.choice(n // slot - 1, size=k, replace=False) * slot:
            scores[s:s + int(rng.integers(1, 7))] = rng.uniform(5.0, 9.0)   # 1-6 windows above 5
        far = Background([ScoreStream(scores, stride, 1.0)], cluster).far(5.0, 0.9)
        assert far.count == k
        covered += far.lower_hz * far.livetime_seconds <= mean <= far.upper_hz * far.livetime_seconds
    # Statistical: the nominal coverage is 0.9 and Garwood is conservative, so the
    # observed fraction falls below 0.9 - 4 sigma (sigma = sqrt(0.9 * 0.1 / runs)) only
    # for a broken interval. The upper bound only rules out an interval so wide it
    # always covers (measured: 0.877-0.947 over seeds 0-9 at both rates).
    sigma = np.sqrt(0.9 * 0.1 / runs)
    assert 0.9 - 4 * sigma <= covered / runs <= 0.99


@pytest.mark.parametrize("combine", ["sum", "max"])
def test_the_far_interval_of_time_slides_covers_the_true_rate(combine):
    # Two detectors' noise slid against each other (slide k pairs h_i with l_(i+k)): a loud
    # excursion in one detector passes in many slides, so the slid count's variance is several
    # times Poisson's (for the maximum, near the slides' number). The interval counts the
    # background's independent information, its count over its data's reuse, and covers the true
    # rate at 1 and 2 sigma; Garwood on the slid count as if independent does not, for the maximum.
    rng = np.random.default_rng(12)
    replicas, num_windows, num_slides = 400, 2000, 8
    lengths = num_windows - np.arange(1, num_slides + 1)
    fn = (lambda h, l: (h + l) / np.sqrt(2)) if combine == "sum" else np.maximum
    pass_chance = 20.0 / lengths.sum()
    # The threshold whose window pass chance is pass_chance, for the combined statistic.
    t = stats.norm.isf(pass_chance) if combine == "sum" else stats.norm.isf(1 - np.sqrt(1 - pass_chance))
    true_rate = pass_chance  # unclustered 1 s windows: a trigger per passing window
    covered, naive = np.zeros(2), np.zeros(2)
    for _ in range(replicas):
        h, l = rng.normal(size=num_windows), rng.normal(size=num_windows)
        background = Background([ScoreStream(fn(h[:num_windows - k], l[k:]), 1.0, 1.0, 0, k)
                                 for k in range(1, num_slides + 1)])
        assert background.data_reuse == num_slides
        for j, sigma in enumerate(DEFAULT_SIGMAS):
            level = sigma_level(sigma)
            far = background.far(t, level)
            covered[j] += far.lower_hz <= true_rate <= far.upper_hz
            lower, upper = poisson_interval(far.count, level)
            naive[j] += lower / far.livetime_seconds <= true_rate <= upper / far.livetime_seconds
    for j, sigma in enumerate(DEFAULT_SIGMAS):
        level = sigma_level(sigma)
        tolerance = 3 * np.sqrt(level * (1 - level) / replicas)
        assert covered[j] / replicas >= level - tolerance, (sigma, covered / replicas)
        if combine == "max":
            assert naive[j] / replicas < level - tolerance, (sigma, naive / replicas)


def test_by_block_partitions_the_livetime():
    streams = [ScoreStream(np.arange(5.0), 1.0, 1.0, block="a"),
               ScoreStream(np.arange(3.0), 1.0, 1.0, block="b"),
               ScoreStream(np.arange(4.0), 1.0, 1.0, block="a")]
    blocks = Background(streams).by_block()
    assert set(blocks) == {"a", "b"}
    assert blocks["a"].livetime_seconds + blocks["b"].livetime_seconds == Background(streams).livetime_seconds


# ------------------------------------------------------- levels and chance

def test_intervals_are_reported_at_one_and_two_sigma():
    # Every interval the validator reports is two-sided and central, at 1 sigma (68.27%) and
    # 2 sigma (95.45%) of a normal law.
    assert DEFAULT_SIGMAS == (1.0, 2.0)
    assert sigma_level(1.0) == pytest.approx(0.682689492, abs=1e-9)
    assert sigma_level(2.0) == pytest.approx(0.954499736, abs=1e-9)


def test_the_chance_of_a_noise_only_pass_is_its_beta_law():
    # A threshold chosen from ten windows at the third highest (the target allows three): a fresh
    # noise window passes it with a chance distributed as Beta(3, 8). One between windows, just
    # above the fourth highest, is passed as Beta(4, 7). Above every window nothing passes.
    background = Background.independent_windows(np.arange(10.0), 1.0)
    at_third = background.threshold_for_far(0.3).threshold
    assert at_third == 7.0
    np.testing.assert_array_equal(background.found_by_chance([at_third]).a, [3.0])
    assert background.found_by_chance([at_third]).mean[0] == pytest.approx(3 / 11)
    threshold = 6.5
    chance = background.found_by_chance([threshold, np.inf, np.nan])
    np.testing.assert_array_equal(chance.a[:2], [4.0, 0.0])
    assert chance.mean[0] == pytest.approx(4 / 11) and chance.mean[1] == 0.0 and np.isnan(chance.mean[2])
    level = sigma_level(1.0)
    low, high = chance.interval(level)
    assert low[0] == pytest.approx(stats.beta.ppf((1 - level) / 2, 4, 7))
    assert high[0] == pytest.approx(stats.beta.ppf((1 + level) / 2, 4, 7))
    assert low[1] == high[1] == 0.0
    draws = chance.sample(np.random.default_rng(0), 4000)
    assert draws.shape == (4000, 3) and np.all(draws[:, 1] == 0.0)
    assert stats.kstest(draws[:, 0], stats.beta(4, 7).cdf).pvalue > 1e-3
    # Subtracting a chance c from a found fraction f: the fraction found by signal, (f - c) / (1 - c).
    assert subtract_chance(0.6, 0.2) == pytest.approx(0.5)
    assert subtract_chance(0.2, 0.2) == 0.0 and subtract_chance(1.0, 0.3) == 1.0


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
    # Both levels, the 2 sigma interval around the 1 sigma one.
    assert curve.sigmas == (1.0, 2.0) and curve.lower.shape == (2, edges.size - 1)
    assert np.all((curve.lower[1] <= curve.lower[0]) & (curve.lower[0] <= curve.efficiency))
    assert np.all((curve.efficiency <= curve.upper[0]) & (curve.upper[0] <= curve.upper[1]))
    for j, sigma in enumerate(curve.sigmas):
        lower, upper = binomial_interval(curve.detected, curve.total, sigma_level(sigma))
        np.testing.assert_allclose(curve.lower[j], lower, rtol=1e-12)
        np.testing.assert_allclose(curve.upper[j], upper, rtol=1e-12)


def test_the_chance_of_a_noise_only_pass_is_subtracted_from_the_efficiency():
    # A window is found when its score passes, whatever passes it: noise alone passes with chance
    # c, so the fraction found by signal is (k / n - c) / (1 - c). Each end of the interval maps
    # the binomial end through the chance's own end on the far side, and is clipped to [0, 1].
    snrs = np.repeat([1.0, 5.0, 9.0], 100)
    scores = np.concatenate([np.r_[np.ones(10), np.zeros(90)], np.r_[np.ones(55), np.zeros(45)], np.ones(100)])
    chance = ChanceLaw(np.array([10.0]), np.array([91.0]))  # mean 0.1
    curve = efficiency_curve(snrs, scores, 0.5, [0.0, 3.0, 7.0, 10.0], chance=chance)
    c = 10 / 101
    np.testing.assert_allclose(curve.efficiency, [(0.1 - c) / (1 - c), (0.55 - c) / (1 - c), 1.0], rtol=1e-12)
    assert curve.chance == pytest.approx(c)
    for j, sigma in enumerate(curve.sigmas):
        level = sigma_level(sigma)
        lower, upper = binomial_interval(curve.detected, curve.total, level)
        c_low, c_high = chance.interval(level)
        np.testing.assert_allclose(curve.lower[j], np.clip((lower - c_high) / (1 - c_high), 0, 1), rtol=1e-12)
        np.testing.assert_allclose(curve.upper[j], np.clip((upper - c_low) / (1 - c_low), 0, 1), rtol=1e-12)


def test_efficiency_at_an_unresolved_far_is_nan():
    curve = efficiency_curve([10.0, 11.0], [1.0, 2.0], float("nan"), [5, 15])
    assert np.all(np.isnan(curve.efficiency)) and np.all(np.isnan(curve.lower)) and curve.total[0] == 2


def test_the_rolling_efficiency_is_a_window_of_half_an_snr_each_side():
    # The rolling curve: a window of +-0.5 in SNR, evaluated at 50 points from the range's low
    # end plus 0.5 to its high end less 0.5; a point with fewer than 20 injections in its window
    # has none. Its bands are pointwise binomial intervals at 1 and 2 sigma on each window's count.
    assert (ROLLING_HALF_WIDTH, ROLLING_POINTS, ROLLING_MIN_INJECTIONS) == (0.5, 50, 20)
    rng = np.random.default_rng(6)
    snrs = rng.uniform(0, 20, 6000)
    snrs = snrs[(snrs < 15.0) | (snrs > 18.0)]  # a gap: too few injections near 16.5
    scores = snrs + rng.normal(0, 1.0, snrs.size)
    curve = rolling_efficiency_curve(snrs, scores, 8.0, snr_range=(0.0, 20.0))
    np.testing.assert_allclose(curve.snr, np.linspace(0.5, 19.5, 50), rtol=1e-12)
    assert curve.half_width == 0.5 and curve.pointwise and curve.sigmas == (1.0, 2.0)
    for i, centre in enumerate(curve.snr):
        inside = (snrs >= centre - 0.5) & (snrs < centre + 0.5)
        assert curve.total[i] == inside.sum() and curve.detected[i] == (inside & (scores >= 8.0)).sum()
        if inside.sum() < 20:
            assert np.isnan(curve.efficiency[i]) and np.all(np.isnan(curve.lower[:, i]))
            continue
        assert curve.efficiency[i] == pytest.approx(curve.detected[i] / curve.total[i])
        for j, sigma in enumerate(curve.sigmas):
            lower, upper = binomial_interval(curve.detected[i], curve.total[i], sigma_level(sigma))
            assert (curve.lower[j, i], curve.upper[j, i]) == pytest.approx((lower, upper), rel=1e-12)
    assert np.isnan(curve.efficiency).sum() >= 1
    assert np.all(np.isnan(rolling_efficiency_curve(snrs, scores, np.nan, snr_range=(0.0, 20.0)).efficiency))


def logistic_found(rng, snrs, centre, width, chance):
    """Found with chance c + (1 - c) / (1 + exp(-(snr - centre) / width)): a logistic efficiency over a chance floor."""
    p = chance + (1 - chance) / (1 + np.exp(-(snrs - centre) / width))
    return (rng.random(snrs.size) < p).astype(float)


def test_the_sigmoid_fit_recovers_its_50_and_90_percent_points_and_its_bootstrap_covers():
    # Found indicators drawn from a logistic efficiency over a chance floor of 0.05: the fit,
    # with the floor subtracted, recovers the 50% and 90% SNR points, and over replicas each
    # point's bootstrap interval covers the truth at its level (pointwise, 3 sigma tolerance).
    centre, width, c = 8.0, 0.7, 0.05
    truth = np.array([centre, centre + width * np.log(9.0)])
    chance = ChanceLaw(np.array([50.0]), np.array([950.0]))
    rng = np.random.default_rng(7)
    replicas, n = 120, 10_000
    covered = np.zeros((2, 2))
    estimates = np.empty((replicas, 2))
    for r in range(replicas):
        snrs = rng.uniform(0, 20, n)
        fit = fit_efficiency_sigmoid(snrs, logistic_found(rng, snrs, centre, width, c), 0.5, chance=chance,
                                     snr_range=(0.0, 20.0), rng=rng, num_bootstrap=200)
        estimates[r] = fit.snr_50, fit.snr_90
        covered += (fit.lower <= truth) & (truth <= fit.upper)
    assert fit.sigmas == (1.0, 2.0) and fit.lower.shape == (2, 2) and fit.num_bootstrap == 200
    np.testing.assert_allclose(estimates.mean(axis=0), truth, atol=3 * estimates.std(axis=0).max() / np.sqrt(replicas))
    for j, sigma in enumerate(fit.sigmas):
        level = sigma_level(sigma)
        tolerance = 3 * np.sqrt(level * (1 - level) / replicas)
        assert np.all(np.abs(covered[j] / replicas - level) <= tolerance), (sigma, covered[j] / replicas)
    # The fitted curve, and its parameters: the logistic's centre and width.
    assert fit.efficiency(np.array([fit.snr_50, fit.snr_90])) == pytest.approx([0.5, 0.9], abs=1e-9)


def test_the_sigmoid_fit_has_no_point_it_did_not_measure():
    # A point outside the injected range, or a fit with no transition (nothing found), is NaN.
    rng = np.random.default_rng(8)
    snrs = rng.uniform(0, 10, 5000)
    beyond = fit_efficiency_sigmoid(snrs, logistic_found(rng, snrs, 9.0, 1.0, 0.0), 0.5, chance=None,
                                    snr_range=(0.0, 10.0), rng=rng, num_bootstrap=50)
    assert np.isfinite(beyond.snr_50) and np.isnan(beyond.snr_90) and np.all(np.isnan(beyond.lower[:, 1]))
    silent = fit_efficiency_sigmoid(snrs, np.zeros(snrs.size), 0.5, chance=None, snr_range=(0.0, 10.0), rng=rng,
                                    num_bootstrap=50)
    assert np.isnan(silent.snr_50) and np.isnan(silent.snr_90)
    unresolved = fit_efficiency_sigmoid(snrs, snrs, np.nan, chance=None, snr_range=(0.0, 10.0), rng=rng, num_bootstrap=50)
    assert np.isnan(unresolved.snr_50)
