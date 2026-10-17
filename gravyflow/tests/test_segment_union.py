"""
Segment arithmetic over random interval sets, checked against the time each list covers.

Endpoints are whole seconds, so the time a list covers is fixed by which half-second
midpoints it covers: a result is right when it covers exactly the midpoints it should, is
sorted, and is disjoint with a gap between neighbours (touching intervals are merged).
"""

import numpy as np
import pytest

import gravyflow as gf
from gravyflow.src.dataset.acquisition.gwosc_data import (
    coalesce_segments, intersect_segments, subtract_segments, within_segments,
)
from gravyflow.src.dataset.features.glitch import overlaps_segments

SPAN = 60
MIDPOINTS = np.arange(SPAN) + 0.5
TRIALS = 300


def obtainer():
    return gf.IFODataObtainer(
        data_quality=gf.DataQuality.BEST,
        data_labels=gf.DataLabel.NOISE,
        observing_runs=gf.ObservingRun.O3,
    )


def random_segments(rng, most=8):
    starts = rng.integers(0, SPAN, size=rng.integers(0, most + 1))
    lengths = rng.integers(0, 15, size=len(starts))  # zero-length pieces included
    return np.stack([starts, np.minimum(starts + lengths, SPAN)], axis=1).astype(float).reshape(-1, 2)


def covered(segments):
    """Which midpoints lie inside any of ``segments``."""
    segments = np.reshape(segments, (-1, 2))
    return ((segments[:, :1] < MIDPOINTS) & (MIDPOINTS < segments[:, 1:])).any(axis=0)


def assert_canonical(result, expected_coverage):
    """Sorted, disjoint with a gap between neighbours, no empty piece, covering what it should."""
    assert result.ndim == 2 and result.shape[1] == 2
    assert np.all(result[:, 1] > result[:, 0])
    assert np.all(result[1:, 0] > result[:-1, 1])
    np.testing.assert_array_equal(covered(result), expected_coverage)


def trials(seed):
    rng = np.random.default_rng(seed)
    return (rng for _ in range(TRIALS))


@pytest.mark.parametrize("union", ["coalesce_segments", "compress_segments"])
def test_the_union_is_preserved_sorted_and_disjoint(union):
    union = coalesce_segments if union == "coalesce_segments" else obtainer().compress_segments
    for rng in trials(1):
        segments = random_segments(rng)
        rng.shuffle(segments)
        assert_canonical(union(segments), covered(segments))


@pytest.mark.parametrize("segments, expected", [
    ([[0, 10], [5, 15], [10, 20]], [[0, 20]]),
    ([[0, 50], [40, 120], [110, 200], [250, 300], [290, 400], [500, 520], [600, 800]],
     [[0, 200], [250, 400], [500, 520], [600, 800]]),
    ([[0, 5], [5, 10], [10, 12]], [[0, 12]]),
    ([[20, 30], [0, 100], [40, 50]], [[0, 100]]),
], ids=["chain_of_three", "chains", "touching", "contained"])
def test_a_chain_merges_into_one_segment(segments, expected):
    """Each segment meets the next: their union is one segment, none of it lost."""
    np.testing.assert_array_equal(obtainer().compress_segments(np.array(segments, float)), expected)
    np.testing.assert_array_equal(coalesce_segments(np.array(segments, float)), expected)


def test_intersection_covers_the_time_in_both():
    for rng in trials(2):
        first, second = random_segments(rng), random_segments(rng)
        assert_canonical(intersect_segments(first, second), covered(first) & covered(second))


def test_subtraction_covers_the_time_in_no_veto():
    for rng in trials(3):
        segments, vetoes = random_segments(rng), random_segments(rng)
        assert_canonical(subtract_segments(segments, vetoes), covered(segments) & ~covered(vetoes))


def test_an_interval_is_within_the_union_when_it_has_no_gap():
    for rng in trials(4):
        segments, intervals = random_segments(rng), random_segments(rng, most=12)
        intervals = intervals[intervals[:, 1] > intervals[:, 0]]
        coverage = covered(segments)
        expected = [coverage[int(start):int(end)].all() for start, end in intervals]
        np.testing.assert_array_equal(within_segments(intervals, segments), expected)


def test_an_interval_overlaps_the_union_when_they_share_time():
    for rng in trials(5):
        segments, intervals = random_segments(rng), random_segments(rng, most=12)
        intervals = intervals[intervals[:, 1] > intervals[:, 0]]
        coverage = covered(segments)
        expected = [coverage[int(start):int(end)].any() for start, end in intervals]
        np.testing.assert_array_equal(overlaps_segments(intervals, segments), expected)
