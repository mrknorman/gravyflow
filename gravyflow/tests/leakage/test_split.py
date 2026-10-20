"""
Split assignment in absolute time, the guard between splits, the fractions, and
split names: checked and required.
"""

from collections import Counter
from dataclasses import dataclass

import numpy as np
import pytest
from gwpy.segments import Segment, SegmentList

import gravyflow as gf
from gravyflow.src.dataset.acquisition.transient_segment import TransientSegment
from gravyflow.src.dataset.features.glitch import GlitchType
from gravyflow.src.dataset.features.transient_index import TransientIndex
from gravyflow.src.dataset.splits import SPLIT_NAMES, _block_uniform
from gravyflow.src.model.examples.base import validation_dataset_args
from gravyflow.src.rng.tree import DIRECT_SPLIT
from gravyflow.src.utils.gps import gps_to_key

from . import audit
from . import fixtures as fx

IFOS = ["H1", "L1"]
SPLITS = ("train", "validate", "test")
GUARD_SECONDS = 128.0


def _time_sets(fixture, runs, inside="O3"):
    """{split: SegmentList} of every detector's served time, inside run ``inside``, for one configuration."""
    run = gf.ObservingRun[inside].value
    bound = SegmentList([Segment(run.start_gps_time, run.end_gps_time)])
    out = {}
    for split in SPLITS:
        obtainer = gf.IFODataObtainer(
            data_quality=gf.DataQuality.BEST, data_labels=[gf.DataLabel.NOISE],
            observing_runs=[gf.ObservingRun[r] for r in runs],
            augmentations=[],
        )
        valid = np.asarray(obtainer.get_valid_segments([gf.IFO[name] for name in IFOS], 7, split))
        segments = SegmentList(Segment(float(a), float(b)) for a, b in valid.reshape(-1, 2))
        out[split] = (segments.coalesce() & bound).coalesce()
    return out


@pytest.mark.parametrize("inside, configurations", [
    ("O3", {"[O3]": ("O3",), "[O1, O2, O3]": ("O1", "O2", "O3"), "[O3, O1]": ("O3", "O1"),
            "[O3, O4]": ("O3", "O4"), "[O1, O2, O3, O4]": ("O1", "O2", "O3", "O4")}),
    ("O4", {"[O4]": ("O4",), "[O1, O2, O3, O4]": ("O1", "O2", "O3", "O4"), "[O4, O3]": ("O4", "O3")}),
], ids=["O3 time", "O4 time"])
def test_split_is_absolute(world, inside, configurations):
    """
    A GPS time's split depends only on the time: the same O3 time sets result whether
    the runs are [O3], [O1, O2, O3], [O3, O1] or include O4; and O4's time is split
    into train, validate and test as every run's is, whatever runs are configured with it.
    """
    o1 = gf.ObservingRun.O1.value.start_gps_time
    o2 = gf.ObservingRun.O2.value.start_gps_time
    o3 = gf.ObservingRun.O3.value.start_gps_time
    o4 = gf.ObservingRun.O4.value.start_gps_time
    intervals = [(o1 + 1e5, o1 + 2e5), (o2 + 1e5, o2 + 2e5), (o3 + 1e4, o3 + 1e4 + 400 * 8192.0),
                 (o4 + 1e4, o4 + 1e4 + 400 * 8192.0)]
    fixture = fx._fixture(intervals, 256.0, "O3", ("O3",))
    with world(fixture):
        sets = {name: _time_sets(fixture, runs, inside) for name, runs in configurations.items()}
    reference = sets[next(iter(configurations))]
    for split in SPLITS:
        assert abs(reference[split]) > 0, split
    for a, b in [(s, t) for i, s in enumerate(SPLITS) for t in SPLITS[i + 1:]]:
        assert abs(reference[a] & reference[b]) == 0, (a, b)
    differences = {
        (name, split): float(abs((sets[name][split] - reference[split]) | (reference[split] - sets[name][split])))
        for name in configurations for split in SPLITS
    }
    moved = {key: seconds for key, seconds in differences.items() if seconds > 0}
    assert not moved, f"seconds of {inside} time whose split changed with the configuration: {moved}"


def test_o4_blocks_take_the_default_fractions():
    """
    O4's blocks (O4a's start to O4b's end) fall in test, validate and train at the
    default fractions, as every run's do. A run never enters the split's fingerprint.
    """
    spec = gf.SplitSpec()
    o4 = gf.ObservingRun.O4.value
    blocks = range(spec.block_of(o4.start_gps_time), spec.block_of(o4.end_gps_time) + 1)
    counts = Counter(spec.block_split(block) for block in blocks)
    for name, fraction in spec.fractions:
        assert abs(counts[name] / len(blocks) - fraction) < 0.02, (name, counts, len(blocks))


PAIRS = [("train", "test"), ("validate", "train"), ("test", "validate")]


@pytest.mark.parametrize("mode", [gf.SamplingMode.RANDOM, gf.SamplingMode.GRID], ids=lambda m: m.name)
@pytest.mark.parametrize("pair", PAIRS, ids=lambda p: "|".join(p))
def test_guard_between_splits(world, tmp_path, pair, mode):
    """
    Across an edge between two splits' blocks, every footprint of one split lies at
    least 128 s from every footprint of the other, in the served time and in decoded
    windows, cold and then warm from one cache root.

    The cache root is the test's own temporary directory, and the obtainers read on
    demand (no read ahead), so every fetch is made by the batch being drawn. With read
    ahead, a cold build's background reads of the batches after the ones drawn could
    still be running, or not yet started, when the warm build began, and on a loaded
    host the warm build's fetch count would include them.
    """
    fixture = fx.straddle(*pair, half_width=512.0)
    with world(fixture, cache_dir=tmp_path / "cache") as harness:
        served = {split: fx.served_segments(fixture, IFOS, split) for split in pair}
        records = {}
        for warm in (False, True):
            sets = {
                f"{split} {'warm' if warm else 'cold'}": (split, fx.noise_batches(
                    fixture, split, mode=mode, prefetch_depth=0))
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


# =============================================================================
# FRACTIONS
# =============================================================================

#: About 10k absolute blocks, around O3.
BLOCKS = range(150000, 160000)


def split_moves(before, after, blocks=BLOCKS):
    """{(split before, split after): blocks} for every block whose split changed."""
    moves = Counter()
    for block in blocks:
        a, b = before.block_split(block), after.block_split(block)
        if a != b:
            moves[(a, b)] += 1
    return dict(moves)


def assert_monotone(before, after, raised, blocks=BLOCKS):
    """Raising ``raised``'s fraction moved blocks, and only from train into ``raised``."""
    moves = split_moves(before, after, blocks)
    assert moves, f"raising {raised} moved no block"
    assert set(moves) == {("train", raised)}, f"raising {raised} moved {moves}"


def spec(test=0.1, validate=0.1, version="gravyflow-split-v1"):
    return gf.SplitSpec(fractions={"test": test, "validate": validate, "train": 1.0 - test - validate},
                        version=version)


@dataclass(frozen=True)
class CumulativeSplit(gf.SplitSpec):
    """A planted leak: test, validate and train stacked in order on the hash."""

    def block_split(self, block):
        u = _block_uniform(self.version, int(block))
        edge = 0.0
        for name, fraction in self.fractions:
            edge += fraction
            if u < edge:
                return name
        return self.fractions[-1][0]


def test_fractions_are_monotone():
    """
    Raising one split's fraction takes time from train alone: validate 0.10 against
    0.20 leaves test unchanged, and test 0.10 against 0.20 leaves validate unchanged.
    So do the records: a transient changes split only from train to the raised one.
    """
    assert_monotone(spec(), spec(validate=0.2), "validate")
    assert_monotone(spec(), spec(test=0.2), "test")
    assert_monotone(spec(version="fold-1"), spec(test=0.3, version="fold-1"), "test")

    times = 1.23e9 + np.sort(np.random.default_rng(3).uniform(0.0, 3.0e7, 4000))
    index = {}
    for name, fractions in (("before", spec()), ("after", spec(validate=0.2))):
        records = TransientIndex([
            TransientSegment(gps_key=gps_to_key(float(t)), transient_gps_time=float(t),
                             start_gps_time=float(t) - 48.2, end_gps_time=float(t) + 48.2,
                             label=gf.DataLabel.GLITCHES, kind=GlitchType.BLIP,
                             observing_run=gf.ObservingRun.O3, seen_in=[gf.IFO.L1])
            for t in times
        ])
        records.assign_splits(fractions)
        index[name] = records._groups
    moved = Counter((index["before"].get(k), index["after"].get(k))
                    for k in set(index["before"]) | set(index["after"])
                    if index["before"].get(k) != index["after"].get(k))
    assert moved and set(moved) <= {("train", "validate"), ("train", None), (None, "validate")}, moved


def test_fractions_check_sees_a_cumulative_layout():
    """The planted leak: stacking validate on test moves validate when test grows."""
    before = CumulativeSplit(fractions=spec().fractions)
    after = CumulativeSplit(fractions=spec(test=0.2).fractions)
    assert ("validate", "test") in split_moves(before, after)
    with pytest.raises(AssertionError, match="moved"):
        assert_monotone(before, after, "test")


# =============================================================================
# NAMES, AND A SPLIT NAMED EVERYWHERE
# =============================================================================

WINDOW = dict(sample_rate_hertz=256.0, onsource_duration_seconds=1.0, offsource_duration_seconds=4.0,
              crop_duration_seconds=0.5, num_examples_per_batch=2, steps_per_epoch=1)


def white():
    return gf.NoiseObtainer(noise_type=gf.NoiseType.WHITE, ifos=[gf.IFO.L1])


def indexed():
    index = TransientIndex([])
    index.assign_splits(gf.SplitSpec())
    return index


#: Every entry point that takes a split, called with ``group`` (omitted when None).
ENTRY_POINTS = {
    "Dataset": lambda **g: gf.Dataset(noise_obtainer=white(), **WINDOW, **g),
    "ComposedDataset": lambda **g: gf.ComposedDataset(
        pools=[gf.FeaturePool(name="noise", label=0, probability=1.0, noise_obtainer=white())],
        **WINDOW, **g),
    "NoiseObtainer": lambda **g: next(white()(**g)),
    "noise get_valid_segments": lambda **g: gf.IFODataObtainer(
        data_quality=gf.DataQuality.BEST, data_labels=[gf.DataLabel.NOISE],
        observing_runs=gf.ObservingRun.O3).get_valid_segments([gf.IFO.L1], 1, g.get("group")),
    "transient obtainer": lambda **g: next(gf.IFODataObtainer(
        data_quality=gf.DataQuality.BEST, data_labels=[gf.DataLabel.GLITCHES],
        observing_runs=gf.ObservingRun.O3)(ifos=[gf.IFO.L1], **g)),
    "TransientIndex": lambda **g: list(indexed().iter(group=g.get("group", "unnamed"))),
    "ValidationConfig": lambda **g: gf.ValidationConfig(**g),
    "validation_dataset_args": lambda **g: validation_dataset_args(gf.NoiseType.WHITE, **g),
}


def refusal(call, **group):
    """The message ``call`` refuses with (KeyError or ValueError), or None when it accepts."""
    try:
        call(**group)
    except (KeyError, ValueError) as error:
        return str(error)
    return None


def expected_refusals(name):
    """
    The texts the split check refuses ``name`` with: not a split; and for the reserved
    name, where a dataset's seed tree checks it first, reserved.
    """
    texts = [f"Unknown split {name!r}: a split is one of {list(SPLIT_NAMES)}."]
    if name == DIRECT_SPLIT:
        texts.append(f"The split '{DIRECT_SPLIT}' is reserved for draws made outside a dataset.")
    return texts


def assert_refuses(call, name):
    message = refusal(call, group=name)
    assert message is not None, f"split {name!r} was accepted"
    assert any(text in message for text in expected_refusals(name)), \
        f"split {name!r} refused by another check: {message}"


@pytest.mark.parametrize("entry", list(ENTRY_POINTS))
@pytest.mark.parametrize("name", ["val", "Train", "direct", "all"])
def test_split_names_checked(entry, name):
    """Every entry point refuses a name that is not a split, by name."""
    assert_refuses(ENTRY_POINTS[entry], name)


def test_names_check_sees_an_accepted_name(monkeypatch):
    """The planted leak: a check that lets any name through is caught."""
    monkeypatch.setattr(gf.SplitSpec, "check_name", staticmethod(lambda name: name))
    with pytest.raises(AssertionError, match="was accepted"):
        assert_refuses(ENTRY_POINTS["ValidationConfig"], "val")


#: The entry points that build or serve data without a named split.
UNNAMED = ["Dataset", "ComposedDataset", "NoiseObtainer", "transient obtainer"]


def assert_needs_a_name(call, entry):
    message = refusal(call)
    assert message is not None and "Name the split" in message, \
        f"{entry} served data without a named split ({message})"


@pytest.mark.parametrize("entry", UNNAMED)
def test_split_is_named(entry):
    """No dataset or obtainer serves data without a named split."""
    assert_needs_a_name(ENTRY_POINTS[entry], entry)


def test_evaluation_arguments_build_only_a_held_out_split():
    """validation_dataset_args builds test by default, validate on request, never train."""
    assert gf.Dataset(**validation_dataset_args(gf.NoiseType.WHITE, steps_per_epoch=1)).group == "test"
    assert gf.Dataset(**validation_dataset_args(
        gf.NoiseType.WHITE, group="validate", steps_per_epoch=1)).group == "validate"
    assert "never 'train'" in refusal(ENTRY_POINTS["validation_dataset_args"], group="train")


def test_named_check_sees_a_default_split():
    """The planted leak: a dataset that defaults its split is caught."""
    def defaulting(**group):
        return gf.Dataset(noise_obtainer=white(), **WINDOW, **{"group": "train", **group})

    with pytest.raises(AssertionError, match="without a named split"):
        assert_needs_a_name(defaulting, "a defaulting Dataset")
