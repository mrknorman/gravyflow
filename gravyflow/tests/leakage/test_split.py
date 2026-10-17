"""
Split assignment in absolute time, and the guard between splits.
"""

import numpy as np
import pytest
from gwpy.segments import Segment, SegmentList

import gravyflow as gf

from . import audit
from . import fixtures as fx

IFOS = ["H1", "L1"]
SPLITS = ("train", "validate", "test")
GUARD_SECONDS = 128.0


def _time_sets(fixture, runs):
    """{split: SegmentList} of every detector's served time, inside O3, for one configuration."""
    o3 = gf.ObservingRun.O3.value
    bound = SegmentList([Segment(o3.start_gps_time, o3.end_gps_time)])
    out = {}
    for split in SPLITS:
        obtainer = gf.IFODataObtainer(
            data_quality=gf.DataQuality.BEST, data_labels=[gf.DataLabel.NOISE],
            observing_runs=[gf.ObservingRun[r] for r in runs], cache_segments=False,
            augmentations=[], prefetch_segments=0,
        )
        valid = np.asarray(obtainer.get_valid_segments([gf.IFO[name] for name in IFOS], 7, split))
        segments = SegmentList(Segment(float(a), float(b)) for a, b in valid.reshape(-1, 2))
        out[split] = (segments.coalesce() & bound).coalesce()
    return out


def test_split_is_absolute(world):
    """
    A GPS time's split depends only on the time: the same O3 time sets result whether
    the runs are [O3], [O1, O2, O3] or [O3, O1].
    """
    o3 = gf.ObservingRun.O3.value.start_gps_time
    o1 = gf.ObservingRun.O1.value.start_gps_time
    o2 = gf.ObservingRun.O2.value.start_gps_time
    intervals = [(o1 + 1e5, o1 + 2e5), (o2 + 1e5, o2 + 2e5), (o3 + 1e4, o3 + 1e4 + 400 * 8192.0)]
    fixture = fx._fixture(intervals, 256.0, "O3", ("O3",))
    configurations = {
        "[O3]": ("O3",),
        "[O1, O2, O3]": ("O1", "O2", "O3"),
        "[O3, O1]": ("O3", "O1"),
    }
    with world(fixture):
        sets = {name: _time_sets(fixture, runs) for name, runs in configurations.items()}
    reference = sets["[O3]"]
    for split in SPLITS:
        assert abs(reference[split]) > 0, split
    for a, b in [(s, t) for i, s in enumerate(SPLITS) for t in SPLITS[i + 1:]]:
        assert abs(reference[a] & reference[b]) == 0, (a, b)
    differences = {
        (name, split): float(abs((sets[name][split] - reference[split]) | (reference[split] - sets[name][split])))
        for name in configurations for split in SPLITS
    }
    moved = {key: seconds for key, seconds in differences.items() if seconds > 0}
    assert not moved, f"seconds of O3 time whose split changed with the configuration: {moved}"


PAIRS = [("train", "test"), ("validate", "train"), ("test", "validate")]


@pytest.mark.parametrize("mode", [gf.SamplingMode.RANDOM, gf.SamplingMode.GRID], ids=lambda m: m.name)
@pytest.mark.parametrize("pair", PAIRS, ids=lambda p: "|".join(p))
def test_guard_between_splits(world, tmp_path, pair, mode):
    """
    Across an edge between two splits' blocks, every footprint of one split lies at
    least 128 s from every footprint of the other, in the served time and in decoded
    windows, cold and then warm from one cache root.
    """
    fixture = fx.straddle(*pair, half_width=512.0)
    with world(fixture, cache_dir=tmp_path / "cache") as harness:
        served = {split: fx.served_segments(fixture, IFOS, split) for split in pair}
        records = {}
        for warm in (False, True):
            sets = {
                f"{split} {'warm' if warm else 'cold'}": (split, fx.noise_batches(
                    fixture, tmp_path / "segments" / split, split, mode=mode, cache_segments=True))
                for split in pair
            }
            records.update(audit.collect(harness, sets, 2, IFOS, fx.window(), False, warm=warm))
    for split in pair:
        audit.assert_warm_equals_cold(records[f"{split} cold"], records[f"{split} warm"])
    lists = [
        SegmentList(Segment(float(a), float(b)) for ifo in IFOS for a, b in served[split][ifo]).coalesce()
        for split in pair
    ]
    assert all(len(segments) for segments in lists)
    gap, closest = audit._distance(*lists)
    assert gap >= GUARD_SECONDS, f"served time of {pair} is {gap} s apart at {closest}"
    audit.assert_split_gap(list(records.values()), GUARD_SECONDS)
