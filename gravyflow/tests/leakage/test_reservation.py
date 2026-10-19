"""
Benchmark time is reserved: MLGWSC-1's O3a segments, the strain its test month was
cut from, are in no train or validate draw of real data by default. H1 is held out
over each segment, and L1 over each segment and 240 s either side, since the
challenge shifts L1 by up to 240 s within a segment. Test keeps the time: the
reservation holds data out of training and model selection, never out of
evaluation. Only the explicit challenge-faithful mode (``gf.SplitSpec(reserved=())``)
serves it, and that mode is in the split's fingerprint.

The segments here are literal rows of the challenge's ``segments.csv`` (rows 0 and
11), never read back from the library, so the check does not restate the rule it
judges.
"""

import numpy as np
import pytest

import gravyflow as gf
from gravyflow.src.dataset.acquisition.transient_segment import TransientSegment
from gravyflow.src.dataset.features.glitch import GlitchType
from gravyflow.src.dataset.features.transient_index import TransientIndex
from gravyflow.src.utils.gps import gps_to_key

from . import audit
from . import fixtures as fx

IFOS = ["H1", "L1"]
MODES = [gf.SamplingMode.RANDOM, gf.SamplingMode.GRID]
L1_GUARD_SECONDS = 240.0

#: MLGWSC-1 segments.csv row 0: it runs from a train block (151148) into a validate
#: block (151149) of the golden table.
SEGMENT_0 = (1238205077.0, 1238215138.0)
#: Row 11: it holds the whole test block 151225, between two train blocks.
SEGMENT_11 = (1238834634.0, 1238853959.0)


def reserved(segment):
    """The time ``segment`` holds out, per detector: H1 the segment, L1 with its guard."""
    start, end = segment
    return {"H1": [(start, end)], "L1": [(start - L1_GUARD_SECONDS, end + L1_GUARD_SECONDS)]}


def around_segment_0():
    """Train time before row 0, the segment itself, and validate time after it."""
    return fx._fixture([(1238203000.0, 1238218000.0)], 256.0, "O3", ("O3",))


def around_segment_11():
    """Train time before row 11 and its test block, inside the segment."""
    return fx._fixture([(1238834000.0, 1238845000.0)], 256.0, "O3", ("O3",))


def faithful():
    """The overrides of an obtainer in challenge-faithful mode."""
    return dict(split_spec=gf.SplitSpec(reserved=()))


def _check_reserved(world, fixture, segment, splits, **overrides):
    """
    Both layers, as the veto tests check them: every split's served time avoids the
    reserved time (exhaustive), and decoded RANDOM and GRID windows lie in that served
    time and avoid it too.
    """
    with world(fixture) as harness:
        served = {split: fx.served_segments(fixture, IFOS, split, **overrides) for split in splits}
        sets = {
            f"{split} {mode.name}": (split, fx.noise_batches(fixture, split, mode=mode, **overrides))
            for split in splits for mode in MODES
        }
        records = audit.collect(harness, sets, 2, IFOS, fx.window(), onsource_cropped=False)
    for split in splits:
        audit.assert_avoids(served[split], reserved(segment))
    for record in records.values():
        audit.assert_inside([record], served[record.split])
        audit.assert_avoids([record], reserved(segment))


def test_benchmark_time_is_reserved_from_training(world):
    """
    No train or validate noise window, in RANDOM or GRID, reads H1 inside an MLGWSC-1
    segment or L1 within 240 s of one; both splits still serve the time around it.
    """
    _check_reserved(world, around_segment_0(), SEGMENT_0, ("train", "validate"))


def test_reservation_check_sees_challenge_faithful_time(world):
    """
    The opt-out, and the check's planted leak: in challenge-faithful mode train and
    validate read the segment, and the check fails on it.
    """
    fixture = around_segment_0()
    with world(fixture):
        for split in ("train", "validate"):
            with pytest.raises(AssertionError, match="forbidden time served"):
                audit.assert_avoids(fx.served_segments(fixture, IFOS, split, **faithful()), reserved(SEGMENT_0))


def test_benchmark_time_stays_in_test(world):
    """
    The reservation holds data out of training, never out of evaluation: test serves
    row 11's time on both detectors, while train beside it keeps clear.
    """
    fixture = around_segment_11()
    _check_reserved(world, fixture, SEGMENT_11, ("train",))
    with world(fixture):
        served = fx.served_segments(fixture, IFOS, "test")
    with pytest.raises(AssertionError, match="forbidden time served"):
        audit.assert_avoids(served, {"H1": [SEGMENT_11]}, ifos=["H1"])
    with pytest.raises(AssertionError, match="forbidden time served"):
        audit.assert_avoids(served, {"L1": [SEGMENT_11]}, ifos=["L1"])


def test_partners_avoid_benchmark_time(world):
    """A glitch's partner in train or validate never reads reserved time on its detector."""
    fixture = around_segment_0()
    with world(fixture):
        obtainer = fx.ifo_data_obtainer(fixture, data_labels=[gf.DataLabel.GLITCHES])
        for split in ("train", "validate"):
            partners = {ifo: obtainer._partner_segments(gf.IFO[ifo], split) for ifo in IFOS}
            audit.assert_avoids(partners, reserved(SEGMENT_0))


def blip(t, ifo="H1"):
    return TransientSegment(gps_key=gps_to_key(float(t)), transient_gps_time=float(t),
                            start_gps_time=float(t) - 48.2, end_gps_time=float(t) + 48.2,
                            label=gf.DataLabel.GLITCHES, kind=GlitchType.BLIP,
                            observing_run=gf.ObservingRun.O3, seen_in=[gf.IFO[ifo]])


#: Glitches in row 0's train block: inside the segment, inside only L1's guard (an
#: H1 glitch: the guard holds every detector's records out, whichever saw them), and
#: clear of both.
INSIDE, L1_GUARD_ONLY, CLEAR = 1238208000.0, 1238204900.0, 1238204600.0


def _splits(spec):
    index = TransientIndex([blip(INSIDE), blip(L1_GUARD_ONLY), blip(CLEAR)])
    excluded = index.assign_splits(spec)
    return {t: index._groups.get(gps_to_key(t)) for t in (INSIDE, L1_GUARD_ONLY, CLEAR)}, excluded


def test_glitches_in_benchmark_time_are_in_no_split():
    """
    A glitch record whose footprint meets reserved time of any detector is in no
    split, and counted as reserved; challenge-faithful mode keeps it in train.
    """
    groups, excluded = _splits(gf.SplitSpec())
    assert groups == {INSIDE: None, L1_GUARD_ONLY: None, CLEAR: "train"}, groups
    assert excluded == {"lies in reserved time (mlgwsc1-o3a)": 2}, excluded
    groups, excluded = _splits(gf.SplitSpec(reserved=()))
    assert groups == {INSIDE: "train", L1_GUARD_ONLY: "train", CLEAR: "train"}, groups
    assert excluded == {}, excluded


def test_reservation_names_the_mode(world):
    """
    The mode is part of the split's identity: the default and challenge-faithful
    specs fingerprint differently and say which they are, and a dataset's record of
    its data (``seed_metadata``, kept beside a model) names the mode.
    """
    strict, faithful = gf.SplitSpec(), gf.SplitSpec(reserved=())
    assert strict == gf.DEFAULT_SPLIT
    assert strict.fingerprint() != faithful.fingerprint()
    assert strict.mode == "reserved: mlgwsc1-o3a"
    assert faithful.mode == "challenge-faithful"
    assert gf.SplitSpec(reserved=(gf.MLGWSC1_O3A,)).fingerprint() == strict.fingerprint()
    assert np.array_equal(strict.reserved_time(gf.IFO.L1)[0],
                          [SEGMENT_0[0] - L1_GUARD_SECONDS, SEGMENT_0[1] + L1_GUARD_SECONDS])
    fixture = around_segment_0()
    with world(fixture):
        for spec in (strict, faithful):
            record = fx.dataset(fixture, "train", split_spec=spec).seed_metadata
            assert (record["split_fingerprint"], record["split_mode"]) == (spec.fingerprint(), spec.mode)


def test_a_reservation_is_checked_as_it_is_made():
    """Reservations are named apart and are gf.Reservation objects; a repr stays short."""
    other = gf.Reservation.from_segments("other", [[0.0, 10.0]], {"V1": 5.0})
    assert np.array_equal(other.of("V1"), [[-5.0, 15.0]]) and len(other.of("H1")) == 0
    spec = gf.SplitSpec(reserved=(gf.MLGWSC1_O3A, other))
    assert spec.mode == "reserved: mlgwsc1-o3a, other"
    assert spec.reserving(-1.0, 0.0, "V1") is other and spec.reserving(-6.0, -5.0, "V1") is None
    with pytest.raises(ValueError, match="distinct names"):
        gf.SplitSpec(reserved=(other, other))
    with pytest.raises(TypeError, match="gf.Reservation"):
        gf.SplitSpec(reserved=([(0.0, 1.0)],))
    with pytest.raises(ValueError, match="non-negative"):
        gf.Reservation.from_segments("bad", [[0.0, 1.0]], {"H1": -1.0})
    assert len(repr(gf.DEFAULT_SPLIT)) < 400
