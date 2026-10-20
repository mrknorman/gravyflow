"""
Independent background events counted by the data they share
(`metrics.disjoint_events`, `Background.independent_count`).

Known answers: one detector's loud excursion, met again in every slide, counts
once; slides of independent noise count close to every trigger; the quotable
depth of wrapping slides grows with their number; a background, slid or not,
whose triggers lie further apart than a window's padded onsource (the data that
set a score) counts every trigger, as
before; and a ring's join adds no trigger.
"""

import numpy as np
import pytest

from gravyflow.src.dataset.config import WindowSpec
from gravyflow.src.validate.metrics import MIN_BACKGROUND_EVENTS, ScoreStream, disjoint_events, poisson_interval
from gravyflow.src.validate.per_detector import DetectorScores, slide_streams
from gravyflow.src.validate.slides import WrappedSlides
from gravyflow.src.validate.stream import StreamBackground, StreamGeometry, stream_background

RATE = 32.0
WINDOW = WindowSpec(sample_rate_hertz=RATE, onsource_duration_seconds=1.0, offsource_duration_seconds=0.5,
                    crop_duration_seconds=0.0)
GEOMETRY = StreamGeometry.of(WINDOW, 0.25)
FOOTPRINT = 1.0  # the data that set a score: a window's padded onsource
MINIMUM = 1.5  # the least shift: the light-travel time and the input


def ring(scores, num_slides, step_strides=7):
    """Detector scores (2, N) on a ring of one segment, and ``num_slides`` slides ``step_strides`` apart."""
    detectors = DetectorScores(np.asarray(scores, dtype=np.float64), np.array([scores.shape[1]]), np.array([0.0]),
                               GEOMETRY)
    slides = WrappedSlides(detectors.segments, 0.25 * step_strides * np.arange(1, num_slides + 1), 2, MINIMUM)
    return detectors, slides


# ------------------------------------------------------------- the greedy count

def test_triggers_further_apart_than_a_window_s_data_are_each_independent():
    scores = np.array([5.0, 4.0, 3.0, 2.0])
    positions = np.array([[0.0, 0.0], [1.5, 10.0], [3.0, 20.0], [100.0, 30.0]])
    np.testing.assert_array_equal(disjoint_events(scores, positions, FOOTPRINT), np.sort(scores))


def test_a_trigger_sharing_one_detector_s_data_with_a_louder_one_is_not_counted():
    scores = np.array([5.0, 4.0, 3.0])
    # The second shares detector 1's data with the first; the third shares none.
    positions = np.array([[0.0, 50.0], [20.0, 50.5], [40.0, 80.0]])
    np.testing.assert_array_equal(disjoint_events(scores, positions, FOOTPRINT), [3.0, 5.0])
    # Loudest first: the quieter of two overlapping triggers is the one dropped.
    np.testing.assert_array_equal(disjoint_events(scores[::-1], positions[::-1], FOOTPRINT), [3.0, 5.0])


def test_data_overlap_is_taken_round_the_ring():
    scores = np.array([5.0, 4.0])
    positions = np.array([[0.2, 0.0], [99.5, 40.0]])  # 0.7 s apart round a 100 s ring
    assert disjoint_events(scores, positions, FOOTPRINT).size == 2
    assert disjoint_events(scores, positions, FOOTPRINT, ring_seconds=100.0).size == 1


# ---------------------------------------------------------------- known answers

def test_one_detector_s_loud_excursion_repeated_in_every_slide_counts_once():
    rng = np.random.default_rng(0)
    num_slides, n = 12, 2000
    scores = rng.normal(0.0, 1.0, (2, n))
    scores[1, 700] = 50.0  # one detector's excursion: every slide pairs it with other data of detector 0
    detectors, slides = ring(scores, num_slides)
    background = StreamBackground(slide_streams(detectors, slides, lambda s: np.max(s, axis=0)), 1.0, 0.1, FOOTPRINT)
    assert background.far_count(40.0) == num_slides  # one loud trigger in every slide
    assert background.independent_count(background.far_count(40.0)) == 1
    # Its interval is one event's, counted in every slide.
    lower, upper = background.count_interval(num_slides)
    expected = poisson_interval(1.0)
    assert lower == pytest.approx(num_slides * expected[0]) and upper == pytest.approx(num_slides * expected[1])


def test_one_excursion_in_stepped_slides_counts_once():
    """Stream mode's stepped slides of real noise: detector j read j k step later, the excursion in detector 1."""
    rng = np.random.default_rng(1)
    step, stride, n = 4.0, 0.25, 1600
    data = rng.normal(0.0, 1.0, (2, n + 16 * 40))
    data[1, 900] = 50.0
    streams = []
    for k in range(1, 9):
        shift = int(k * step / stride)
        streams.append(ScoreStream(np.maximum(data[0, :n], data[1, shift:shift + n]), stride, stride, 0, k,
                                   detector_offsets=(1000.0, 1000.0 + k * step)))
    background = StreamBackground(streams, 1.0, 0.1, FOOTPRINT)
    assert background.far_count(40.0) == 8 and background.independent_count(8) == 1


def test_slides_of_independent_noise_count_close_to_every_trigger():
    rng = np.random.default_rng(2)
    num_slides, n = 10, 200_000
    detectors, slides = ring(rng.normal(0.0, 1.0, (2, n)), num_slides, step_strides=400)
    # A coincidence statistic: both detectors must be loud.
    background = StreamBackground(slide_streams(detectors, slides, lambda s: np.min(s, axis=0)), 0.5, 0.1,
                                  footprint_seconds=0.25)
    assert background.data_reuse == num_slides
    counts = np.array([20, 50, 100])
    independent = background.independent_count(counts)
    assert np.all(independent >= 0.9 * counts), independent
    assert np.all(independent > counts // num_slides)  # the old rule, count / slides


def test_the_quotable_depth_of_wrapping_slides_grows_with_their_number():
    rng = np.random.default_rng(3)
    n = 8000
    scores = rng.normal(0.0, 1.0, (2, n))
    depth = []
    for num_slides in (4, 16, 64):
        detectors, slides = ring(scores, num_slides, step_strides=17)
        background = StreamBackground(slide_streams(detectors, slides, lambda s: np.min(s, axis=0)), 0.5, 0.1,
                                      footprint_seconds=0.25)
        curve = background.far_curve()
        assert np.array_equal(curve.quotable, curve.independent_events >= MIN_BACKGROUND_EVENTS)
        depth.append(curve.rate_hz[curve.quotable].min())  # the lowest FAR quotable
    assert depth[0] > 2.5 * depth[1] > 2.5 * 2.5 * depth[2], depth


def test_an_unslid_background_whose_triggers_lie_further_apart_than_the_padded_onsource_counts_every_trigger():
    rng = np.random.default_rng(4)
    streams = [ScoreStream(rng.normal(0.0, 1.0, 4000), 0.25, 0.25, b, 0, detector_offsets=(1e4 * b, 1e4 * b))
               for b in range(3)]
    plain = StreamBackground(streams, 1.0, 0.1)
    # Triggers lie more than a cluster (1 s) apart, and a score's data are its 1 s padded onsource.
    placed = StreamBackground(streams, 1.0, 0.1, FOOTPRINT)
    counts = np.arange(0, 400, 7)
    np.testing.assert_array_equal(placed.independent_count(counts), counts)
    np.testing.assert_array_equal(placed.independent_count(counts), plain.independent_count(counts))
    np.testing.assert_allclose(placed.count_interval(counts), plain.count_interval(counts))
    # Clustered tighter than the data that set a score, neighbouring triggers share data: fewer count.
    tight = StreamBackground(streams, 0.5, 0.1, FOOTPRINT)
    count = int(tight.far_count(-np.inf))
    assert 0.5 * count < tight.independent_count(count) < count


def test_slid_triggers_further_apart_than_a_window_s_data_count_as_before_with_one_slide_s_reuse():
    # Two slides whose triggers lie far apart in every detector: every trigger is independent.
    streams = [ScoreStream(np.array([0.0, 9.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 1.0, 1.0, 0, 1,
                           detector_offsets=(0.0, 100.0)),
               ScoreStream(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 8.0, 0.0, 0.0]), 1.0, 1.0, 0, 2,
                           detector_offsets=(500.0, 700.0))]
    background = StreamBackground(streams, 1.0, 0.1, FOOTPRINT)
    assert background.far_count(8.0) == 2 and background.independent_count(2) == 2
    assert background.reuse_at(2) == 1.0


def test_a_ring_s_join_adds_no_trigger():
    stride, n = 0.25, 400
    scores = -np.linspace(1.0, 2.0, n)  # falling, no ties: the only trigger away from the ends is at 200
    scores[0], scores[-1] = 5.0, 6.0  # neighbours across the join: the louder alone is a trigger on the ring
    scores[200] = 3.0
    stream = ScoreStream(scores, stride, stride, 0, 1, detector_offsets=(0.0, 0.0), ring_seconds=n * stride)
    background = StreamBackground([stream], 1.0, 0.1)
    np.testing.assert_array_equal(background.window_scores, [3.0])  # neither end, once, and not twice
    assert background.livetime_seconds == pytest.approx(n * stride - 2.0)


def test_a_record_of_placed_streams_counts_by_shared_data():
    rng = np.random.default_rng(5)
    data = rng.normal(0.0, 1.0, (2, 2000))
    data[1, 900] = 50.0
    scores, lengths, offsets = [], [], []
    for k in range(1, 6):
        scores.append(np.maximum(data[0, :1000], data[1, 16 * k:16 * k + 1000]))
        lengths.append(1000)
        offsets.append((0.0, 4.0 * k))
    record = {"lengths": np.array(lengths), "slides": np.arange(1, 6), "blocks": np.zeros(5, dtype=int),
              "stride_seconds": np.float64(0.25), "detector_offsets": np.array(offsets),
              "footprint_seconds": np.float64(FOOTPRINT)}
    background = stream_background(np.concatenate(scores), record, 1.0, 1.0, 0.1)
    assert background.far_count(40.0) == 5 and background.independent_count(5) == 1
    del record["detector_offsets"]
    old = stream_background(np.concatenate(scores), record, 1.0, 1.0, 0.1)
    assert old.independent_count(5) == 1 and old.independent_count(4) == 0  # the old rule, count / slides
