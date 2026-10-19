"""
A standard dataset is test data: every second of its pinned segments lies in the test
split, at least half the guard from any time of another split, and no published event
of its list lies in them.

The split is re-derived here from its definition, never read back from the library's
split code: block k = [8192 k, 8192 (k + 1)) is test when the first 8 bytes of
sha256("gravyflow-split-v1:k"), as a fraction of 2^64, are below 0.10, and 64 s (half
the 128 s guard) at each edge with a block of another split belongs to no split. The
test then checks that the presets' partition is that one, so a change to it fails here
rather than passing against the change.
"""

import hashlib

import numpy as np
import pytest

import gravyflow as gf

VERSION, TEST_FRACTION, BLOCK, HALF_GUARD = "gravyflow-split-v1", 0.10, 8192, 64


def is_test(block: int) -> bool:
    digest = hashlib.sha256(f"{VERSION}:{block}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2.0**64 < TEST_FRACTION


@pytest.mark.parametrize("name", sorted(gf.STANDARD_DATASETS))
def test_a_presets_partition_is_the_one_derived_here(name):
    split = gf.STANDARD_DATASETS[name].split
    assert (split.version, split.fraction("test"), split.block_seconds, split.guard_seconds) == (
        VERSION, TEST_FRACTION, BLOCK, 2 * HALF_GUARD)
    assert gf.STANDARD_DATASETS[name].split_name == "test"


@pytest.mark.parametrize("name", sorted(gf.STANDARD_DATASETS))
def test_a_presets_segments_lie_wholly_in_the_test_partition(name):
    segments = gf.standard_dataset(name).segments
    for start, end in segments.tolist():
        first, last = start // BLOCK, (end - 1) // BLOCK
        blocks = range(first, last + 1)
        assert all(is_test(k) for k in blocks), f"{name}: [{start}, {end}) meets a block of another split"
        if not is_test(first - 1):
            assert start >= first * BLOCK + HALF_GUARD, f"{name}: [{start}, {end}) is inside the guard"
        if not is_test(last + 1):
            assert end <= (last + 1) * BLOCK - HALF_GUARD, f"{name}: [{start}, {end}) is inside the guard"


@pytest.mark.parametrize("name", sorted(gf.STANDARD_DATASETS))
def test_a_presets_events_lie_outside_its_segments(name):
    preset = gf.standard_dataset(name)
    segments = preset.segments
    for event in preset.events:
        k = np.searchsorted(segments[:, 0], event["gps"], side="right") - 1
        assert k < 0 or event["gps"] >= segments[k, 1], f"{name}: {event['name']} is inside a segment"
        assert is_test(int(event["gps"] // BLOCK)), f"{name}: {event['name']} is not in a test block"


def test_the_twin_reads_the_same_test_time():
    for name in gf.STANDARD_DATASETS:
        np.testing.assert_array_equal(gf.standard_dataset(name, gaussian_twin=True).segments,
                                      gf.standard_dataset(name).segments)
