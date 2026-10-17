"""
Detectors are never read at coincident times by default; a glitch's other detectors
get partner windows.

"""

import itertools

import numpy as np
import pytest

import gravyflow as gf

from . import audit
from . import fixtures as fx
from .fakes import EventRecord, GlitchRecord

RV = gf.ReturnVariables
IFOS = ["H1", "L1"]
FOOTPRINT_SECONDS = fx.OFFSOURCE_SECONDS + fx.ONSOURCE_SECONDS + 2 * fx.CROP_SECONDS


def offset_pieces(rate=256.0):
    """H1 [B+1024, B+2048] against L1 [B, B+2048], and equal 40 s and 60 s pieces."""
    edge = fx.boundary("O3", "train", "train", skip=5)
    h1 = [(edge + 1024.0, edge + 2048.0), (edge + 3000.0, edge + 3040.0), (edge + 4000.0, edge + 4060.0)]
    l1 = [(edge, edge + 2048.0), (edge + 3000.0, edge + 3040.0), (edge + 4000.0, edge + 4060.0)]
    base = fx._fixture(l1, rate, "O3", ("O3",))
    return fx.Fixture(base.encoding, {"H1": np.array(h1), "L1": np.array(l1)}, "O3", ("O3",))


def _min_separation(prints, rate):
    starts = prints[..., 0] / rate
    return float(np.min(np.abs(starts[:, 0] - starts[:, 1])))


LAYOUTS = {
    "identical pieces": fx.odd_pieces,
    "offset pieces": offset_pieces,
}


@pytest.mark.parametrize("mode", [gf.SamplingMode.RANDOM, gf.SamplingMode.GRID], ids=lambda m: m.name)
@pytest.mark.parametrize("layout", list(LAYOUTS))
def test_detectors_never_coincide(world, tmp_path, layout, mode):
    """In every example, the detectors' windows start at least two footprints apart."""
    fixture = LAYOUTS[layout]()
    with world(fixture) as harness:
        batches = fx.noise_batches(fixture, tmp_path, "train", mode=mode)
        record = audit.collect(harness, {"train": ("train", batches)}, 8, IFOS, fx.window(), False)["train"]
    separation = _min_separation(record.footprints, 256.0)
    overlapping = int(np.sum(np.abs(np.diff(record.footprints[..., 0], axis=1)) / 256.0 < 2 * FOOTPRINT_SECONDS))
    assert separation >= 2 * FOOTPRINT_SECONDS, (
        f"{overlapping} of {record.num_examples} examples read H1 and L1 within "
        f"{2 * FOOTPRINT_SECONDS} s; closest {separation} s"
    )


#: More batches than one GRID pass over any layout here serves: a pass that has not
#: ended by then never ends.
GRID_PASS_BOUND = 2000


def _grid_pass_length(world, fixture, directory):
    """The number of batches one GRID pass serves, refusing a pass that does not end."""
    with world(fixture):
        batches = fx.noise_batches(fixture, directory, "train", mode=gf.SamplingMode.GRID)
        length = sum(1 for _ in itertools.islice(batches, GRID_PASS_BOUND + 1))
    assert length <= GRID_PASS_BOUND, f"a GRID pass served more than {GRID_PASS_BOUND} batches: it never ends"
    return length


@pytest.mark.parametrize("layout", list(LAYOUTS))
def test_each_window_once(world, tmp_path, layout):
    """
    One GRID pass ends, serves each detector's window at most once, and keeps every
    example's detectors two footprints apart.
    """
    fixture = LAYOUTS[layout]()
    length = _grid_pass_length(world, fixture, tmp_path / "count")
    assert length > 0, "a GRID pass served nothing"
    with world(fixture) as harness:
        batches = fx.noise_batches(fixture, tmp_path / "pass", "train", mode=gf.SamplingMode.GRID)
        record = audit.collect(harness, {"grid": ("train", batches)}, length, IFOS, fx.window(), False)["grid"]
    starts = record.footprints[..., 0]
    for d, ifo in enumerate(IFOS):
        repeats = record.num_examples - len(np.unique(starts[:, d]))
        assert repeats == 0, f"{ifo}: {repeats} of {record.num_examples} windows served twice in one pass"
    separation = _min_separation(record.footprints, 256.0)
    assert separation >= 2 * FOOTPRINT_SECONDS, f"H1 and L1 read {separation} s apart in GRID"


#: Rows for the constructive draw, as (bounds, origins in samples, separation in samples):
#: two detectors on one piece whose starts can be separated only near its ends (rejection
#: alone rarely finds a pair), and three detectors on offset pieces.
TIGHT_ROWS = {
    "one piece": ([(0, 100), (0, 100)], [0.0, 0.0], 95.0),
    "offset pieces": ([(0, 1000), (0, 1000), (0, 1000)], [0.0, 250.0, -130.0], 300.0),
}


@pytest.mark.parametrize("draws", [0, None], ids=["constructed only", "after rejection draws"])
@pytest.mark.parametrize("row", list(TIGHT_ROWS))
def test_constructed_starts_keep_the_separation(monkeypatch, row, draws):
    """
    RANDOM places a detector's start constructively, from the starts far enough from
    the others', once its rejection draws keep conflicting; those starts keep the
    whole separation. Driven directly: with no rejection draws (every conflicting
    example constructed), and after the library's draws on tight rows.
    """
    from gravyflow.src.dataset.acquisition import base

    constructed = []
    allowed_draw = base._allowed_draw

    def counting(*args):
        constructed.append(allowed_draw(*args))
        return constructed[-1]

    monkeypatch.setattr(base, "_allowed_draw", counting)
    if draws is not None:
        monkeypatch.setattr(base, "SEPARATION_DRAWS", draws)
    bounds, origins, separation = TIGHT_ROWS[row]
    starts = base.separated_starts(np.random.default_rng(1), bounds, origins, separation, 2000)
    assert len(starts) > 0 and sum(start is not None for start in constructed) >= 50
    clock = starts + np.asarray(origins)
    gaps = [np.min(np.abs(clock[:, i] - clock[:, j])) for i, j in itertools.combinations(range(len(bounds)), 2)]
    assert min(gaps) >= separation, f"constructed starts {min(gaps)} samples apart, under {separation}"


#: The span a transient's record reads, each side of its centre.
TRANSIENT_HALF_SPAN_SECONDS = 48.2
#: How far a partner's centre lies from the transient's at least: two spans.
PARTNER_SEPARATION_SECONDS = 4 * TRANSIENT_HALF_SPAN_SECONDS


def _glitch_world(glitch_ifo="L1"):
    """
    A ``glitch_ifo``-only glitch at ``peak``, both detectors in data over
    [peak - 400, peak + 400] of one train block; the other detector has a 60 s gap
    and a glitch of its own where a partner could otherwise go.
    """
    other = "H1" if glitch_ifo == "L1" else "L1"
    edge = fx.boundary("O3", "train", "train", skip=3)
    peak = edge + 1000.0
    fixture = fx._fixture([(peak - 400.0, peak + 400.0)], 4096.0, "O3", ("O3",))
    segments = dict(fixture.segments)
    segments[other] = np.array([(peak - 400.0, peak + 200.0), (peak + 260.0, peak + 400.0)])
    fixture = fx.Fixture(fixture.encoding, segments, "O3", ("O3",))
    rows = [
        GlitchRecord(glitch_ifo, peak, peak - 0.5, 1.0, "Blip", 0.95),
        GlitchRecord(other, peak - 260.0, peak - 260.5, 1.0, "Blip", 0.95),
    ]
    return fixture.with_catalogues(glitches=rows), peak


def _glitch_batch(world, tmp_path, fixture):
    with world(fixture) as harness:
        obtainer = gf.IFODataObtainer(
            data_quality=gf.DataQuality.BEST, data_labels=[gf.DataLabel.GLITCHES],
            observing_runs=[gf.ObservingRun.O3], augmentations=[], prefetch_segments=0,
        )
        noise = gf.NoiseObtainer(ifo_data_obtainer=obtainer, ifos=[gf.IFO.H1, gf.IFO.L1],
                                 noise_type=gf.NoiseType.REAL, 
                                 data_directory_path=tmp_path)
        ds = gf.Dataset(
            noise_obtainer=noise, sample_rate_hertz=256.0,
            onsource_duration_seconds=fx.ONSOURCE_SECONDS, crop_duration_seconds=fx.CROP_SECONDS,
            offsource_duration_seconds=fx.OFFSOURCE_SECONDS, num_examples_per_batch=1,
            group="train", seed=7, ifos=[gf.IFO.H1, gf.IFO.L1],
            input_variables=[RV.ONSOURCE, RV.OFFSOURCE],
            output_variables=[RV.START_GPS_TIME, RV.DATA_LABEL],
        )
        drawn = []

        def draw():
            drawn.append(ds[0])
            yield drawn[-1]

        record = audit.collect(harness, {"glitch": ("train", draw())}, 1, IFOS, fx.window(256.0),
                               onsource_cropped=True, stride=16, adjacent=False)["glitch"]
        fetches = harness.source.snapshot()[0]
    outputs = drawn[0][1]
    return record, np.asarray(outputs[RV.DATA_LABEL.name]), fetches


def test_glitch_detectors_never_coincide(world, tmp_path):
    """An L1-only glitch over [H1, L1]: H1 is read at least two footprints away."""
    fixture, peak = _glitch_world()
    record, _, _ = _glitch_batch(world, tmp_path, fixture)
    separation = float(np.abs(np.diff(record.footprints[0, :, 0]))[0] / 4096.0)
    assert separation >= 2 * FOOTPRINT_SECONDS, f"H1 and L1 read {separation} s apart"


def _inside(span, segments):
    return any(a <= span[0] and span[1] <= b for a, b in segments)


def _meets(span, extents):
    return any(span[0] < b and a < span[1] for a, b in extents)


@pytest.mark.parametrize("glitch_ifo", ["L1", "H1"])
def test_partner_windows(world, tmp_path, glitch_ifo):
    """
    A one-detector glitch over [H1, L1]: the other detector gets a partner window,
    read whole in its own data and the record's split, away from the glitch and from
    the other detector's own glitches, labelled NOISE beside the GLITCH; each
    detector's START_GPS_TIME is its own decoded window; and two builds (separate
    caches) choose the same partner.
    """
    fixture, peak = _glitch_world(glitch_ifo)
    other = "H1" if glitch_ifo == "L1" else "L1"
    g, p = IFOS.index(glitch_ifo), IFOS.index(other)
    first, labels, fetches = _glitch_batch(world, tmp_path / "first", fixture)
    second, _, _ = _glitch_batch(world, tmp_path / "second", fixture)

    seconds = first.seconds()[0]
    assert abs(seconds[p, 0] - seconds[g, 0]) >= 2 * FOOTPRINT_SECONDS, f"{other} is read at the glitch's time"
    expected = [0.0, 0.0]
    expected[g] = 1.0
    assert labels.tolist() == [expected], f"labels {labels.tolist()}, expected {expected} (NOISE, GLITCH)"
    decoded_start = seconds[:, 0] + fx.OFFSOURCE_SECONDS + fx.CROP_SECONDS
    np.testing.assert_allclose(first.start_labels[0], decoded_start, atol=1 / 256.0,
                               err_msg="START_GPS_TIME is not each detector's own window")
    assert np.array_equal(first.footprints, second.footprints), "two builds chose different partners"

    glitch_span = (peak - TRANSIENT_HALF_SPAN_SECONDS, peak + TRANSIENT_HALF_SPAN_SECONDS)
    reads = [(f.start, f.end) for f in fetches if f.ifo == other]
    assert reads, f"no {other} fetch"
    own_glitches = [row.extent for row in fixture.glitches if row.ifo == other]
    for span in reads:
        assert not _meets(span, [glitch_span]), f"an {other} fetch {span} meets the glitch's span"
        assert _inside(span, fixture.segments[other]), f"the {other} fetch {span} leaves its data"
        assert not _meets(span, own_glitches), f"the {other} fetch {span} meets an {other} glitch"
        assert gf.DEFAULT_SPLIT.assign(*span) == "train", f"the {other} fetch {span} leaves the split"
        centre = (span[0] + span[1]) / 2.0
        assert abs(centre - peak) >= PARTNER_SEPARATION_SECONDS, f"{other} read at {centre - peak:+.1f} s"


# =============================================================================
# PARTNERS: THEIR SPLIT, BLOCK AND EVENTS, AND RECORDS WITHOUT ONE
# =============================================================================

#: Half the footprint of the partner world's event: 2 s in band, so 48.2 s each side.
EVENT_HALF_FOOTPRINT_SECONDS = 48.2


def _golden_block(*splits):
    """The first block k of the golden O3 table where blocks k, k + 1, ... are ``splits``."""
    table = fx.golden_blocks("O3")
    return next(k for k in sorted(table) if all(table.get(k + i) == s for i, s in enumerate(splits)))


def _partner_world():
    """
    L1-only glitches over [H1, L1], each in a block of its own, where H1 has data only
    near the glitch, so its partner has one narrow place to go:

    - ``block``: 250 s into a train block after a train block, H1 over 300 s each side
      of the edge: a partner fits only in the record's block's first 9 s, the rest of
      H1 being in the block before;
    - ``split``: 250 s into a train block after a test block, H1 from 600 s before the
      edge to 300 s after: a partner fits only in the guard (or the test block), so
      there is none;
    - ``event``: mid train block, an event 300 s after the glitch, and H1 from 100 s
      before the glitch to the end of the event's footprint and guard: a partner fits
      only there, so there is none;
    - ``test``, ``validate``: mid block, H1 over 600 s each side: partners served.

    Returns the fixture, the glitches' times by case, and the event's footprint.
    """
    B = fx.BLOCK_SECONDS
    trains = _golden_block("train", "train", "train", "train")
    after_train, middle = (trains + 2) * B, (trains + 3) * B + B / 2
    after_test = (_golden_block("test", "train") + 1) * B
    held_out = {split: (_golden_block("train", split, "train") + 1) * B + B / 2 for split in ("test", "validate")}
    event = middle + 300.0
    event_end = event + EVENT_HALF_FOOTPRINT_SECONDS + fx.GUARD_SECONDS + TRANSIENT_HALF_SPAN_SECONDS
    times = {"block": after_train + 250.0, "split": after_test + 250.0, "event": middle, **held_out}
    h1 = [(after_train - 300.0, after_train + 300.0), (after_test - 600.0, after_test + 300.0),
          (middle - 100.0, event_end)] + [(t - 600.0, t + 600.0) for t in held_out.values()]
    l1 = [(t - 400.0, t + 400.0) for t in times.values()]
    base = fx._fixture(sorted(l1), 256.0, "O3", ("O3",))
    fixture = fx.Fixture(base.encoding, {"H1": np.array(sorted(h1)), "L1": np.array(sorted(l1))}, "O3", ("O3",))
    rows = [GlitchRecord("L1", t, t - 0.5, 1.0, "Blip", 0.95) for t in times.values()]
    footprint = (event - EVENT_HALF_FOOTPRINT_SECONDS, event + EVENT_HALF_FOOTPRINT_SECONDS)
    return fixture.with_catalogues(glitches=rows, events=[EventRecord("GWPARTNER", event, 2.0)]), times, footprint


def _served_partners(world):
    """
    {split: (records' own L1 spans, their H1 partners' spans, the split's rejections)}
    in the partner world, from the library's served time (``get_valid_segments``).
    """
    fixture, _, _ = _partner_world()
    served = {}
    with world(fixture):
        for split in ("train", "validate", "test"):
            obtainer = fx.ifo_data_obtainer(fixture, data_labels=[gf.DataLabel.GLITCHES])
            spans = obtainer.get_valid_segments([gf.IFO.H1, gf.IFO.L1], 7, split)
            served[split] = (spans[:, 1], spans[:, 0], obtainer.get_acquisition_stats()["rejections"])
    return served


def _other_split_distance(span, split):
    """
    Seconds from ``span`` to the nearest time of a split other than ``split`` in the
    blocks about it, by the golden table (infinite when they are all ``split``).
    """
    table = fx.golden_blocks("O3")
    return min((max(fx.block_start("O3", k) - span[1], span[0] - fx.block_start("O3", k + 1), 0.0)
                for k in range(fx.block_of("O3", span[0]) - 1, fx.block_of("O3", span[1]) + 2)
                if table[k] != split), default=np.inf)


def test_partners_keep_to_their_split_and_block(world):
    """
    A partner reads only its record's split, half the guard from every other split's
    time (judged by the golden table), and its record's block, though the detector has
    data in other splits and in the next block of the record's split.
    """
    for split, (own, partners, _) in _served_partners(world).items():
        assert len(partners), f"no {split} record served a partner"
        for span, partner in zip(own, partners):
            assert fx.golden_split("O3", *span) == split
            gap = _other_split_distance(partner, split)
            assert gap >= fx.GUARD_SECONDS / 2, f"a {split} partner {partner} lies {gap:.1f} s from another split"
            blocks = {fx.block_of("O3", t) for t in (*span, *partner)}
            assert len(blocks) == 1, f"a {split} partner {partner} leaves its record's block {span}"


def test_partners_avoid_events(world):
    """
    No partner reads an event's footprint or its guard, though the record's detector
    has data there and nowhere else a partner fits.
    """
    _, _, (start, end) = _partner_world()
    guarded = [(start - fx.GUARD_SECONDS, end + fx.GUARD_SECONDS)]
    for split, (_, partners, _) in _served_partners(world).items():
        for partner in partners:
            assert not _meets(partner, guarded), f"a {split} partner {partner} meets an event's guard {guarded}"


def test_records_without_a_partner_are_counted(world):
    """
    A record with no partner time is not served, and its split counts it under "no
    partner": the two train records whose partner could read only a guard; none in
    test or validate.
    """
    _, times, _ = _partner_world()
    served = _served_partners(world)
    own, _, rejections = served["train"]
    centres = sorted(round((a + b) / 2.0, 3) for a, b in own)
    assert centres == [round(times["block"], 3)], f"train served the records at {centres}"
    assert rejections.get("no partner") == 2, f"train rejections {rejections}"
    for split in ("validate", "test"):
        assert "no partner" not in served[split][2], f"{split} rejections {served[split][2]}"
