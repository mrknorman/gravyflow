"""
The metric family (`validate.family`, and VT, binned volume and the ratio to the
matched filter in `validate.volume`): each metric's known answer on synthetic records
where it is exact, then the validator's own record and its dashboard tabs.
"""

import numpy as np
import pytest
from scipy import stats

from gravyflow.src.validate.family import (
    FAR_LEVELS_HZ,
    detection_roc,
    far_calibration,
    far_floor_hz,
    held_out_split,
    live_time_seconds,
    population_detection_probability,
    real_event_summary,
    recovered_counts,
    timing_accuracy,
)
from gravyflow.src.validate.metrics import Background, ScoreStream, binomial_interval, poisson_interval, sigma_level
from gravyflow.src.validate.stream import StreamBackground
from gravyflow.src.validate.volume import (
    SECONDS_PER_MONTH,
    SECONDS_PER_YEAR,
    InjectionVolumes,
    binned_volume,
    sensitive_volume_curve,
    sensitive_vt,
    volume_ratio,
)

#: A background of 100 one-second windows scoring 0 .. 99: at a target allowing k triggers the threshold
#: is 100 - k, and the chance of a noise-only pass is Beta(k, 101 - k), on average k / 101.
NOISE = np.arange(100.0)


def background():
    return Background.independent_windows(NOISE, 1.0)


def chance(k):
    return k / 101.0


def signal(fraction, k):
    return (fraction - chance(k)) / (1.0 - chance(k))


def rng():
    return np.random.default_rng(7)


# ----------------------------------------------------------------------- ROC

def test_the_roc_is_the_probability_of_detection_at_each_far():
    # Band 0-5: 4 injections, all found. Band 5-10: 10, half found. All: 9 of 14.
    snrs = np.array([2.0] * 4 + [7.0] * 10)
    scores = np.array([1e3] * 4 + [1e3] * 5 + [-1e3] * 5)
    roc = detection_roc(snrs, scores, background(), [0.1, 0.5, 1e-3], rng(), bands=((0, 5), (5, 10), (10, 15)),
                        num_bootstrap=200)
    assert roc.labels == ("SNR 0-5", "SNR 5-10", "all SNRs")  # the empty band is dropped
    assert roc.total.tolist() == [4, 10, 14]
    assert roc.threshold[:2].tolist() == [90.0, 50.0] and np.isnan(roc.threshold[2])
    assert roc.detected.tolist() == [[4, 5, 9], [4, 5, 9], [0, 0, 0]]
    np.testing.assert_allclose(roc.probability[0], [signal(1.0, 10), signal(0.5, 10), signal(9 / 14, 10)], rtol=1e-12)
    np.testing.assert_allclose(roc.probability[1], [signal(1.0, 50), signal(0.5, 50), signal(9 / 14, 50)], rtol=1e-12)
    assert np.all(np.isnan(roc.probability[2])) and np.all(np.isnan(roc.lower[:, 2]))
    np.testing.assert_allclose(roc.achieved_far_hz[:2], [0.1, 0.5])
    # Every interval holds its point, and the 2 sigma interval the 1 sigma one.
    ok = np.isfinite(roc.probability)
    assert np.all(roc.lower[0][ok] <= roc.probability[ok] + 1e-12)
    assert np.all(roc.probability[ok] <= roc.upper[0][ok] + 1e-12)
    assert np.all(roc.lower[1][ok] <= roc.lower[0][ok]) and np.all(roc.upper[0][ok] <= roc.upper[1][ok])
    assert roc.quotable.tolist() == [True, True, False]


def test_the_populations_detection_probability_is_its_found_share_of_the_volume():
    # Ten equal-weight injections, six found: at the target allowing 10 noise triggers the
    # population's detection probability is the found fraction with the chance subtracted.
    scores = np.array([1e3] * 6 + [-1e3] * 4)
    injections = InjectionVolumes(scores, 100.0, np.linspace(10.0, 100.0, 10))
    curve = sensitive_volume_curve(injections, background(), [0.1], rng(), num_bootstrap=100)
    population = population_detection_probability(curve)
    np.testing.assert_allclose(population.probability, [signal(0.6, 10)], rtol=1e-12)
    np.testing.assert_allclose(population.lower, curve.joint_lower_gpc3 / injections.population_volume_gpc3)


# ------------------------------------------------------------- VT and bins

def test_vt_is_the_volume_times_the_live_time():
    scores = np.array([1e3] * 6 + [-1e3] * 4)
    injections = InjectionVolumes(scores, 100.0, np.linspace(10.0, 100.0, 10))
    curve = sensitive_volume_curve(injections, background(), [0.1, 1e-3], rng(), num_bootstrap=100)
    vt = sensitive_vt(curve, 2.0 * SECONDS_PER_YEAR)
    expected = injections.reference_volume_gpc3 * signal(0.6, 10) * 2.0
    np.testing.assert_allclose(vt.vt_gpc3_yr[0], expected, rtol=1e-12)
    assert np.isnan(vt.vt_gpc3_yr[1]) and vt.live_time_years == 2.0
    np.testing.assert_allclose(vt.lower_gpc3_yr, curve.joint_lower_gpc3 * 2.0)
    np.testing.assert_allclose(vt.upper_gpc3_yr, curve.joint_upper_gpc3 * 2.0)
    with pytest.raises(ValueError):
        sensitive_vt(curve, 0.0)


def test_the_binned_volume_is_each_bins_own_volume():
    # Chirp masses 15 (four, all found), 25 (four, one found), none at 35.
    values = np.array([15.0] * 4 + [25.0] * 4)
    scores = np.array([1e3] * 4 + [1e3] + [-1e3] * 3)
    injections = InjectionVolumes(scores, 100.0, np.linspace(10.0, 100.0, 8))
    binned = binned_volume(injections, background(), [0.1], values, [10, 20, 30, 40], rng(),
                           parameter="chirp mass", num_bootstrap=100)
    assert binned.total.tolist() == [4, 4, 0] and binned.curves[2] is None
    reference = injections.reference_volume_gpc3
    np.testing.assert_allclose(binned.volume_gpc3[0, :2], [reference * signal(1.0, 10), reference * signal(0.25, 10)],
                               rtol=1e-12)
    assert np.isnan(binned.volume_gpc3[0, 2]) and binned.lower_gpc3.shape == (2, 1, 3)
    with pytest.raises(ValueError):
        binned_volume(injections, background(), [0.1], values[:3], [10, 20], rng())


def test_the_ratio_to_the_matched_filter_is_on_the_same_injections():
    distances = np.linspace(10.0, 100.0, 10)
    model = InjectionVolumes(np.array([1e3] * 3 + [-1e3] * 7), 100.0, distances)
    reference = InjectionVolumes(np.array([1e3] * 6 + [-1e3] * 4), 100.0, distances)
    ratio = volume_ratio(model, background(), reference, background(), [0.1, 1e-3], rng(), num_bootstrap=200)
    np.testing.assert_allclose(ratio.ratio[0], signal(0.3, 10) / signal(0.6, 10), rtol=1e-12)
    assert np.isnan(ratio.ratio[1]) and ratio.quotable.tolist() == [True, False]
    assert ratio.lower[1, 0] <= ratio.lower[0, 0] <= ratio.ratio[0] <= ratio.upper[0, 0] <= ratio.upper[1, 0]
    same = volume_ratio(model, background(), model, background(), [0.1], rng(), num_bootstrap=50)
    assert same.ratio[0] == 1.0 and same.lower[:, 0].tolist() == [1.0, 1.0] and same.upper[:, 0].tolist() == [1.0, 1.0]
    other = InjectionVolumes(model.scores, 100.0, distances[::-1])
    with pytest.raises(ValueError, match="same injections"):
        volume_ratio(model, background(), other, background(), [0.1], rng())


# ------------------------------------------------------- counts and timing

def test_recovered_counts_are_the_injections_found_at_each_level():
    scores = np.concatenate([np.full(7, 95.0), np.full(13, 10.0)])
    counts = recovered_counts(scores, background(), [0.1, 0.95, 1e-3])
    assert counts.found.tolist() == [7, 20, 0] and counts.total == 20
    np.testing.assert_allclose(counts.by_chance[:2], [20 * chance(10), 20 * chance(95)], rtol=1e-12)
    for level, sigma in enumerate((1.0, 2.0)):
        low, high = binomial_interval(7, 20, sigma_level(sigma))
        assert counts.lower[level, 0] == pytest.approx(20 * low) and counts.upper[level, 0] == pytest.approx(20 * high)
    assert np.isnan(counts.by_chance[2]) and np.isnan(counts.lower[:, 2]).all()


def test_timing_accuracy_is_the_median_and_90th_percentile_of_the_found():
    dt = 0.01 * np.arange(1, 11) * (-1.0) ** np.arange(1, 11)  # -0.01, 0.02, ..., 0.10
    scores = np.concatenate([np.full(10, 95.0), [10.0, -np.inf]])  # one below the threshold, one found by nothing
    offsets = np.concatenate([dt, [0.5, np.nan]])
    timing = timing_accuracy(scores, offsets, background(), [0.1, 1e-3])
    assert timing.num_found.tolist() == [10, 0]
    assert timing.median_abs_seconds[0] == pytest.approx(0.055)
    assert timing.p90_abs_seconds[0] == pytest.approx(0.091)
    assert timing.median_seconds[0] == pytest.approx(0.005)
    # The 1 sigma interval on a median of ten: the 3rd and 8th order statistics (binomial(10, 1/2)).
    assert (timing.median_abs_lower[0, 0], timing.median_abs_upper[0, 0]) == pytest.approx((0.03, 0.08))
    assert np.isnan(timing.median_abs_seconds[1])


# ------------------------------------------------------- false-alarm calibration

def test_a_claimed_far_is_checked_on_held_out_background():
    # Claimed: 10 of the calibration's 100 s at threshold 90. Held out: 200 s, 27 at or above it.
    held_out = Background.independent_windows(np.concatenate([np.full(27, 95.0), np.full(173, 0.0)]), 1.0)
    calibration = far_calibration(background(), held_out, [0.1, 1e-3])
    assert calibration.claimed_hz[0] == pytest.approx(0.1) and calibration.held_out_count[0] == 27
    assert calibration.measured_hz[0] == pytest.approx(0.135) and calibration.ratio[0] == pytest.approx(1.35)
    for level, sigma in enumerate((1.0, 2.0)):
        alpha = 1.0 - sigma_level(sigma)
        lower, upper = stats.chi2.ppf(alpha / 2, 54) / 2, stats.chi2.ppf(1 - alpha / 2, 56) / 2  # Garwood on 27
        assert calibration.lower_hz[level, 0] == pytest.approx(lower / 200, rel=1e-9)
        assert calibration.upper_hz[level, 0] == pytest.approx(upper / 200, rel=1e-9)
        assert calibration.consistent[level, 0] == (lower / 200 <= 0.1 <= upper / 200)
    assert calibration.consistent[:, 0].tolist() == [False, True]
    assert np.isnan(calibration.measured_hz[1]) and not calibration.consistent[:, 1].any()


def test_the_held_out_halves_share_no_data():
    streams = [ScoreStream(np.arange(8.0) + b, 0.25, 1.0, block=b, slide=s) for b in range(4) for s in (1, 2)]
    full = StreamBackground(streams, 1.0, 0.1)
    first, second = held_out_split(full)
    assert {s.block for s in first.streams} == {0, 2} and {s.block for s in second.streams} == {1, 3}
    assert isinstance(first, StreamBackground) and first.data_reuse == 2
    assert first.livetime_seconds + second.livetime_seconds == pytest.approx(full.livetime_seconds)
    assert held_out_split(background()) is None
    assert live_time_seconds(full) == pytest.approx(full.livetime_seconds / 2)
    assert far_floor_hz(full) == pytest.approx((1 / full.livetime_seconds, 20 / full.livetime_seconds))


# ---------------------------------------------------------------- real events

def test_real_events_have_their_far_and_counts_at_each_level():
    events = [dict(name="A", gps=10.0, score=5.0), dict(name="B", gps=20.0, score=50.0),
              dict(name="C", gps=30.0, score=95.0), dict(name="D", gps=40.0)]
    summary = real_event_summary(events, background(), [0.1, 0.5, 1e-3], training_periods=[(15.0, 25.0)])
    assert [r["name"] for r in summary.rows] == ["A", "B", "C"]  # D was not scored
    assert [r["in_training"] for r in summary.rows] == [False, True, False]
    assert [r["far_hz"] for r in summary.rows] == pytest.approx([0.95, 0.50, 0.05])
    lower, upper = poisson_interval(5, sigma_level(1.0))
    assert summary.rows[2]["far_interval_hz"][0] == pytest.approx([lower / 100, upper / 100])
    assert summary.rows[2]["background_events"] == 5
    assert summary.found.tolist() == [1, 2, 0] and summary.found_outside_training.tolist() == [1, 1, 0]
    assert summary.resolved.tolist() == [True, True, False]


def test_the_levels_are_one_per_month_to_one_per_hundred_years():
    np.testing.assert_allclose(np.array(FAR_LEVELS_HZ) * SECONDS_PER_YEAR, [12.0, 1.0, 0.1, 0.01])
    assert FAR_LEVELS_HZ[0] == 1.0 / SECONDS_PER_MONTH
