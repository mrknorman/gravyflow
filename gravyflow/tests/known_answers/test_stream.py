"""
Known answers of stream mode, on a synthetic detector whose truth is exact.

The detector scores a window every ``STRIDE`` seconds of a continuous stream: its
noise scores are independent unit Gaussians, and a signal adds ``A`` to the window
nearest its merger (shifted by a timing error ``delta``). On paper:

- **Triggers.** A window is a trigger when it is the loudest within ``c`` (1 s) of
  itself, ``m = c / STRIDE`` windows either side: of ``2m + 1`` independent
  scores each is the loudest with chance ``1 / (2m + 1)``, so triggers scoring at
  least ``t`` come at the rate ``(1 - Phi(t)^(2m + 1)) / ((2m + 1) STRIDE)``.
- **Chance.** Triggers are more than ``c`` apart, so at most one lies within
  ``+-w`` (0.1 s) of a time: noise alone finds an injection at ``t`` with chance
  exactly the trigger rate times ``2w``, whatever the stride.
- **The found rule.** A loud signal's window is the trigger near its merger, and
  it finds the injection exactly when its time lies within ``+-w`` of the merger:
  the nearest window of a merger uniform in time is uniform within ``STRIDE / 2``
  of it, so with no timing error the chance is ``2w / STRIDE`` (0.8 at a quarter
  of a one-second window), and with a timing error ``delta`` it is the overlap
  of ``[delta - STRIDE / 2, delta + STRIDE / 2]`` with ``[-w, w]`` over ``STRIDE``.
- **Slides.** A two-detector network scored as the larger of its detectors'
  scores has noise ``Phi(t)^2``: its triggers' rate is the one-detector rate with
  ``Phi`` squared, on every slide, and the background counts each slide's
  livetime and reuses each detector's data once per slide.
"""

import numpy as np
import pytest
from scipy import stats

from gravyflow.src.detection.triggers import Triggers
from gravyflow.src.validate.efficiency import efficiency_against_snr
from gravyflow.src.validate.metrics import ScoreStream
from gravyflow.src.validate.stream import StreamBackground, cluster, loudest_within, stream_triggers

STRIDE, CLUSTER, FOUND = 0.25, 1.0, 0.1
NEIGHBOURS = int(round(CLUSTER / STRIDE))  # either side: the cluster's edges are inclusive


def trigger_rate_hz(threshold, noise_cdf=stats.norm.cdf):
    """The paper's rate of triggers scoring at least ``threshold`` (module docstring)."""
    size = 2 * NEIGHBOURS + 1
    return (1.0 - noise_cdf(threshold) ** size) / (size * STRIDE)


def noise_background(rng, num_streams=40, num_windows=20_000):
    streams = [ScoreStream(rng.normal(size=num_windows), STRIDE, 1.0, block=k) for k in range(num_streams)]
    return StreamBackground(streams, CLUSTER, FOUND), streams


def test_a_trigger_is_the_loudest_score_within_the_cluster_of_itself():
    # Exact, by hand: 1 s either side at a 0.25 s stride is four windows either side.
    times = np.arange(16) * STRIDE
    scores = np.array([1, 5, 2, 0, 0, 0, 3, 0, 0, 0, 0, 0, 4, 4, 0, 9.0])
    # 5 at 0.25 s is the loudest out to 1.25 s; 3 at 1.5 s has nothing louder within 1 s (5 is 1.25 s
    # away); 9 at 3.75 s is within 1 s of both 4s (3.0 s and 3.25 s), so neither of them is a trigger.
    np.testing.assert_array_equal(cluster(times, scores, CLUSTER), [1, 6, 15])
    triggers = stream_triggers(times, scores, CLUSTER, FOUND)
    assert isinstance(triggers, Triggers)
    np.testing.assert_array_equal(triggers.time, times[[1, 6, 15]])
    np.testing.assert_array_equal(triggers.stat, [5.0, 3.0, 9.0])
    np.testing.assert_array_equal(triggers.var, [FOUND] * 3)


def test_the_loudest_trigger_within_the_found_window_finds_an_injection():
    times, stats_ = np.array([1.0, 2.0, 2.05, 4.0]), np.array([3.0, 1.0, 2.0, 7.0])
    stat, time = loudest_within(times, stats_, np.array([2.02, 4.1, 3.0, 0.9]), FOUND)
    np.testing.assert_array_equal(stat, [2.0, 7.0, -np.inf, 3.0])
    np.testing.assert_array_equal(time, [2.05, 4.0, np.nan, 1.0])


@pytest.mark.parametrize("far_hz", [0.1, 0.01, 0.002])
def test_the_background_counts_the_papers_triggers(far_hz):
    rng = np.random.default_rng(71)
    background, streams = noise_background(rng)
    t = float(stats.norm.ppf((1.0 - far_hz * (2 * NEIGHBOURS + 1) * STRIDE) ** (1.0 / (2 * NEIGHBOURS + 1))))
    assert trigger_rate_hz(t) == pytest.approx(far_hz, rel=1e-9)
    # Each stream's livetime less a cluster at each end, where no trigger is taken.
    assert background.livetime_seconds == pytest.approx(sum(s.livetime_seconds - 2 * CLUSTER for s in streams))
    expected = trigger_rate_hz(t) * background.livetime_seconds
    count = background.far_count(t)
    # Poisson, 5 sd: no edge allowance, the edges' triggers are not taken.
    assert abs(count - expected) <= 5.0 * np.sqrt(expected)


@pytest.mark.parametrize("far_hz", [0.2, 0.02])
def test_noise_alone_finds_an_injection_with_the_rate_times_the_window(far_hz):
    rng = np.random.default_rng(72)
    background, streams = noise_background(rng)
    t = background.threshold_for_far(far_hz).threshold
    truth = trigger_rate_hz(t) * 2 * FOUND
    # The background's law of the chance, at the threshold it chose for the target.
    assert background.found_by_chance(t).mean[0] == pytest.approx(truth, rel=0.06)
    # Injections of nothing, at mergers uniform in time: found by noise triggers alone.
    found = []
    for stream in streams:
        times = np.arange(stream.scores.size) * STRIDE
        keep = cluster(times, stream.scores, CLUSTER)
        mergers = rng.uniform(10.0, times[-1] - 10.0, 2_000)
        found.append(loudest_within(times[keep], stream.scores[keep], mergers, FOUND)[0] >= t)
    found = np.concatenate(found)
    assert abs(found.mean() - truth) <= 5.0 * np.sqrt(truth * (1 - truth) / found.size)


def loud_detector(rng, mergers, delta, amplitude=50.0, length=4_000):
    """One stream per merger: noise, and the window nearest ``merger + delta`` raised by ``amplitude``."""
    times = np.arange(length) * STRIDE
    found = []
    for merger in mergers:
        scores = rng.normal(size=length)
        scores[np.argmin(np.abs(times - (merger + delta)))] += amplitude
        keep = cluster(times, scores, CLUSTER)
        found.append(loudest_within(times[keep], scores[keep], np.array([merger]), FOUND)[0][0])
    return np.array(found)


@pytest.mark.parametrize("delta", [0.0, 0.05, -0.12])
def test_a_loud_signal_is_found_when_its_window_lies_within_the_found_window(delta):
    rng = np.random.default_rng(73)
    mergers = rng.uniform(100.0, 900.0, 4_000)
    found = loud_detector(rng, mergers, delta) >= 10.0
    low, high = max(delta - STRIDE / 2, -FOUND), min(delta + STRIDE / 2, FOUND)
    truth = max(high - low, 0.0) / STRIDE
    assert abs(found.mean() - truth) <= 5.0 * np.sqrt(max(truth * (1 - truth), 1e-4) / found.size)


def test_chance_is_subtracted_at_loose_levels():
    # Injections of nothing are found by noise alone, at the chance; subtracted, the efficiency
    # is nothing, within its band, at a loose FAR where the chance is a few per cent.
    rng = np.random.default_rng(74)
    background, streams = noise_background(rng, num_streams=20)
    scores = []
    for stream in streams:
        times = np.arange(stream.scores.size) * STRIDE
        keep = cluster(times, stream.scores, CLUSTER)
        scores.append(loudest_within(times[keep], stream.scores[keep], rng.uniform(10.0, times[-1] - 10.0, 1_000),
                                     FOUND)[0])
    scores = np.concatenate(scores)
    snrs = rng.uniform(0.0, 20.0, scores.size)
    (result,) = efficiency_against_snr(snrs, scores, background, [0.2], np.random.default_rng(1),
                                       bin_edges=[0.0, 20.0], snr_range=(0.0, 20.0), num_bootstrap=200)
    assert result.binned.chance == pytest.approx(0.2 * 2 * FOUND, rel=0.1)
    assert result.binned.lower[1, 0] <= 0.0 <= result.binned.upper[1, 0]
    assert abs(result.binned.efficiency[0]) < 0.01


def test_slides_count_each_slide_and_reuse_each_detector_once_per_slide():
    rng = np.random.default_rng(75)
    num_windows, num_slides, step = 40_000, 6, 37
    h, l = rng.normal(size=num_windows), rng.normal(size=num_windows)
    streams = [ScoreStream(np.maximum(h[:num_windows - k * step], l[k * step:]), STRIDE, 1.0, block=0, slide=k)
               for k in range(1, num_slides + 1)]
    background = StreamBackground(streams, CLUSTER, FOUND)
    assert background.data_reuse == num_slides
    assert background.livetime_seconds == pytest.approx(sum(s.livetime_seconds - 2 * CLUSTER for s in streams))
    t = 2.5
    expected = trigger_rate_hz(t, lambda x: stats.norm.cdf(x) ** 2) * background.livetime_seconds
    # The slides share each detector's data, so the count's spread is up to num_slides times Poisson's.
    assert abs(background.far_count(t) - expected) <= 5.0 * np.sqrt(num_slides * expected)
    lower, upper = background.count_interval(background.far_count(t), 0.9545)
    assert lower <= expected <= upper


def test_no_trigger_is_taken_within_a_cluster_of_a_streams_ends():
    # The loudest scores sit at the stream's two ends: each is the loudest within 1 s of itself
    # only because its neighbourhood runs off the stream, so neither is a trigger, and the
    # livetime is less 1 s at each end. The middle's 5 is a trigger.
    scores = np.zeros(17)
    scores[[0, 16, 8]] = [9.0, 8.0, 5.0]
    stream = ScoreStream(scores, STRIDE, 1.0)
    background = StreamBackground([stream], CLUSTER, FOUND)
    assert background.trigger_indices(stream).tolist() == [8]
    assert background.num_triggers == 1 and background.far_count(4.0) == 1
    assert background.livetime_seconds == pytest.approx(16 * STRIDE + 1.0 - 2 * CLUSTER)
    short = ScoreStream(scores[:5], STRIDE, 1.0)
    assert StreamBackground([short], CLUSTER, FOUND).livetime_seconds == 0.0
