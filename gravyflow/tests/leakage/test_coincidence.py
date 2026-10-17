"""
Detectors are never read at coincident times by default; a glitch's other detectors
get partner windows.

"""

import itertools

import numpy as np
import pytest

import gravyflow as gf
from gravyflow.src.dataset.config import TransientDefaults

from . import audit
from . import fixtures as fx
from .fakes import EventRecord, GlitchRecord
from .provenance import batch_arrays

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
#: Transient windows are read from the transient cache's stored 4096 Hz: at that rate
#: the resampler is the identity, so the fixture's broadband code survives the read.
TRANSIENT_RATE = TransientDefaults.CACHE_SAMPLE_RATE_HERTZ
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
            noise_obtainer=noise, sample_rate_hertz=TRANSIENT_RATE,
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

        record = audit.collect(harness, {"glitch": ("train", draw())}, 1, IFOS, fx.window(TRANSIENT_RATE),
                               onsource_cropped=True, adjacent=False)["glitch"]
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
    np.testing.assert_allclose(first.start_labels[0], decoded_start, atol=1 / TRANSIENT_RATE,
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


def _golden_block(*splits, after=-np.inf):
    """The first block k > ``after`` of the golden O3 table where blocks k, k + 1, ... are ``splits``."""
    table = fx.golden_blocks("O3")
    return next(k for k in sorted(table)
                if k > after and all(table.get(k + i) == s for i, s in enumerate(splits)))


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


# =============================================================================
# THE COINCIDENT OPT-IN AND ITS FLAG
# =============================================================================

def _flagged(batches, flags):
    """The batches, each passed on after its COINCIDENT flag is appended to ``flags``."""
    for batch in batches:
        flags.append(batch_arrays(batch, [RV.COINCIDENT])[RV.COINCIDENT])
        yield batch


def _intersection(fixture):
    """The time every detector of ``fixture`` holds data, as (N, 2) GPS intervals."""
    from gravyflow.src.dataset.acquisition.gwosc_data import intersect_segments

    return intersect_segments(*(np.asarray(fixture.segments[ifo]) for ifo in IFOS))


def _noise_record(world, directory, fixture, mode, coincidence):
    """Eight batches of train noise under ``coincidence``: their Records and flags."""
    flags = []
    with world(fixture) as harness:
        batches = fx.noise_batches(fixture, directory, "train", mode=mode, coincidence=coincidence)
        record = audit.collect(harness, {"noise": ("train", _flagged(batches, flags))}, 8, IFOS,
                               fx.window(), False)["noise"]
    return record, np.concatenate(flags)


@pytest.mark.parametrize("mode", [gf.SamplingMode.RANDOM, gf.SamplingMode.GRID], ids=lambda m: m.name)
def test_coincident_noise_carries_the_flag(world, tmp_path, mode):
    """
    Under the opt-in, every example's detectors start at one GPS time, inside the time
    both detectors hold, and carry COINCIDENT = 1; by default every example carries 0
    and its detectors start two footprints apart.
    """
    fixture = offset_pieces()
    record, flags = _noise_record(world, tmp_path / "on", fixture, mode, gf.Coincidence.COINCIDENT)
    assert flags.shape == (record.num_examples,) and np.all(flags == 1), f"flags {np.unique(flags)}"
    starts = record.footprints[..., 0]
    assert np.array_equal(starts[:, 0], starts[:, 1]), "a coincident example's detectors start apart"
    common = _intersection(fixture)
    audit.assert_inside([record], {ifo: common for ifo in IFOS})

    record, flags = _noise_record(world, tmp_path / "off", fixture, mode, gf.Coincidence.SEPARATED)
    assert flags.shape == (record.num_examples,) and np.all(flags == 0), f"flags {np.unique(flags)}"
    assert _min_separation(record.footprints, 256.0) >= 2 * FOOTPRINT_SECONDS


@pytest.mark.parametrize("coincidence", [gf.Coincidence.COINCIDENT, gf.Coincidence.SEPARATED],
                         ids=lambda c: c.name)
def test_a_dataset_batch_carries_the_flag(world, tmp_path, coincidence):
    """
    A gf.Dataset's batch carries its noise's COINCIDENT flag, (B,) float32: 1 for
    every example under the opt-in, 0 for every one by default.
    """
    fixture = offset_pieces()
    with world(fixture):
        ds = fx.dataset(fixture, tmp_path / "data", "train", output_variables=[RV.COINCIDENT],
                        coincidence=coincidence)
        _, output = ds[0]
    flags = np.asarray(output[RV.COINCIDENT.name])
    expected = 1.0 if coincidence == gf.Coincidence.COINCIDENT else 0.0
    assert flags.shape == (8,) and str(flags.dtype) == "float32", (flags.shape, flags.dtype)
    np.testing.assert_array_equal(flags, expected)


def _coincident_world():
    """
    L1-only glitches in one train block, both detectors in data about each: at
    ``clean``; at ``glitched``, with an H1 glitch 20 s after it; at ``gapped``, where
    H1 has a 20 s gap. H1 has room for partners far from all three.
    """
    edge = fx.boundary("O3", "train", "train", skip=3)
    times = {"clean": edge + 1000.0, "glitched": edge + 2000.0, "gapped": edge + 3000.0}
    fixture = fx._fixture([(edge + 400.0, edge + 3600.0)], 4096.0, "O3", ("O3",))
    gapped = times["gapped"]
    segments = dict(fixture.segments)
    segments["H1"] = np.array([(edge + 400.0, gapped - 10.0), (gapped + 10.0, edge + 3600.0)])
    fixture = fx.Fixture(fixture.encoding, segments, "O3", ("O3",))
    rows = [GlitchRecord("L1", t, t - 0.5, 1.0, "Blip", 0.95) for t in times.values()]
    rows.append(GlitchRecord("H1", times["glitched"] + 20.0, times["glitched"] + 19.5, 1.0, "Blip", 0.95))
    return fixture.with_catalogues(glitches=rows), times


def _transient_batches(obtainer, split, batch=1, rate=TRANSIENT_RATE):
    return obtainer.get_onsource_offsource_chunks(
        sample_rate_hertz=rate, onsource_duration_seconds=fx.ONSOURCE_SECONDS,
        crop_duration_seconds=fx.CROP_SECONDS, offsource_duration_seconds=fx.OFFSOURCE_SECONDS,
        num_examples_per_batch=batch, ifos=[gf.IFO.H1, gf.IFO.L1], group=split, seed=7,
    )


def test_coincident_glitches_read_every_detector(world):
    """
    Under the opt-in a glitch is read on every detector at its time: the detector it
    was not seen in is labelled NOISE, and the record is served only where that
    detector holds the span in its data and no glitch of its own; the others are
    counted (the H1 glitch itself among them: L1's glitch lies in its window). Its
    batch carries COINCIDENT = 1, the same record's default batch 0.
    """
    fixture, times = _coincident_world()
    with world(fixture):
        obtainer = fx.ifo_data_obtainer(fixture, data_labels=[gf.DataLabel.GLITCHES],
                                        coincidence=gf.Coincidence.COINCIDENT)
        spans = obtainer.get_valid_segments([gf.IFO.H1, gf.IFO.L1], 7, "train")
        rejections = obtainer.get_acquisition_stats()["rejections"]
        assert spans.shape == (1, 2, 2), f"served {spans.tolist()}"
        assert np.array_equal(spans[0, 0], spans[0, 1]), "H1 is not read at the glitch's time"
        assert abs(np.mean(spans[0, 1]) - times["clean"]) < 1.0
        assert rejections == {"data quality": 1, "transient in another detector": 2}, rejections

        batch = next(iter(_transient_batches(obtainer, "train")))
        assert batch_arrays(batch, [RV.COINCIDENT])[RV.COINCIDENT].tolist() == [1]
        assert np.asarray(batch[RV.DATA_LABEL]).tolist() == [[0, 1]]
        np.testing.assert_allclose(np.asarray(batch[RV.TRANSIENT_GPS_TIME])[0], [times["clean"]] * 2)

        separated = fx.ifo_data_obtainer(fixture, data_labels=[gf.DataLabel.GLITCHES])
        flags = [batch_arrays(b, [RV.COINCIDENT])[RV.COINCIDENT].tolist()
                 for b in itertools.islice(_transient_batches(separated, "train"), 3)]
        assert flags == [[0]] * 3, f"default glitch flags {flags}"


def _assert_unflagged_apart(batch):
    """Every example flagged 0 reads its detectors two transient footprints apart."""
    flags = batch_arrays(batch, [RV.COINCIDENT])[RV.COINCIDENT]
    centres = np.asarray(batch[RV.TRANSIENT_GPS_TIME])
    for flag, row in zip(flags, centres):
        if flag == 0:
            gap = float(np.min(np.abs(np.diff(np.sort(row)))))
            assert gap >= PARTNER_SEPARATION_SECONDS, f"an unflagged example reads detectors {gap} s apart"


def test_the_flag_check_sees_a_glitch_read_at_its_time(world, monkeypatch):
    """
    The planted leak: a default obtainer whose partner is placed at the glitch's own
    time, so its other detector is read coincident in an unflagged batch, is caught:
    an unflagged example's detectors must lie two footprints apart.
    """
    from gravyflow.src.dataset.acquisition import transient

    fixture, _ = _coincident_world()
    with world(fixture):
        obtainer = fx.ifo_data_obtainer(fixture, data_labels=[gf.DataLabel.GLITCHES])
        _assert_unflagged_apart(next(iter(_transient_batches(obtainer, "train"))))
        monkeypatch.setattr(transient, "partner_time", lambda allowed, gps, **kwargs: gps)
        planted = fx.ifo_data_obtainer(fixture, data_labels=[gf.DataLabel.GLITCHES])
        batch = next(iter(_transient_batches(planted, "train")))
    with pytest.raises(AssertionError, match="unflagged"):
        _assert_unflagged_apart(batch)


# =============================================================================
# EVENTS: THEIR DETECTORS FROM DATA QUALITY
# =============================================================================

def _event_partner_world():
    """
    Two 2 s events mid test block: ``both``, in both detectors' data; ``h1_only``,
    where L1 has a 10 s gap at the merger. L1 has room for a partner far from both.
    """
    block = _golden_block("train", "test", "train") + 1
    middle = fx.block_start("O3", block) + fx.BLOCK_SECONDS / 2
    times = {"both": middle - 2000.0, "h1_only": middle + 2000.0}
    fixture = fx._fixture([(middle - 3000.0, middle + 3000.0)], 4096.0, "O3", ("O3",))
    gap = times["h1_only"]
    segments = dict(fixture.segments)
    segments["L1"] = np.array([(middle - 3000.0, gap - 5.0), (gap + 5.0, middle + 3000.0)])
    fixture = fx.Fixture(fixture.encoding, segments, "O3", ("O3",))
    events = [EventRecord(f"GW{name.upper()}", t, 2.0) for name, t in times.items()]
    return fixture.with_catalogues(events=events), times


def test_event_detectors_from_data_quality(world):
    """
    An event is read at its time on each detector whose data holds its span, and on
    each other at a partner time, labelled NOISE: an event in both detectors' data is
    labelled EVENTS on both and flagged COINCIDENT; one with an L1 gap at its merger
    is labelled [EVENTS, NOISE], its L1 read two footprints away, and not flagged.
    """
    fixture, times = _event_partner_world()
    with world(fixture):
        obtainer = fx.ifo_data_obtainer(fixture, data_labels=[gf.DataLabel.EVENTS])
        obtainer.get_valid_segments([gf.IFO.H1, gf.IFO.L1], 7, "test")
        served = {record.name: [ifo.name for ifo in record.seen_in] for record in obtainer.transient_segments}
        assert served == {"GWBOTH": IFOS, "GWH1_ONLY": ["H1"]}, served
        batch = next(iter(_transient_batches(obtainer, "test", batch=2)))
    centres = np.asarray(batch[RV.TRANSIENT_GPS_TIME])
    order = np.argsort(centres[:, 0])
    labels = np.asarray(batch[RV.DATA_LABEL])[order].tolist()
    flags = batch_arrays(batch, [RV.COINCIDENT])[RV.COINCIDENT][order].tolist()
    assert labels == [[2, 2], [2, 0]], labels
    assert flags == [1, 0], flags
    np.testing.assert_allclose(centres[order][0], [times["both"]] * 2)
    assert abs(centres[order][1][1] - times["h1_only"]) >= PARTNER_SEPARATION_SECONDS
    _assert_unflagged_apart(batch)


def _test_time_partner_world():
    """
    Events seen in H1 only (L1 has a 10 s gap at each merger), one in a block of each
    split, each with a test block next to it: ``train`` and ``validate``, the block
    after is test; ``test``, its own block. ``skip``: a train block between a test
    block before it, where L1 has no data, and a test block three after, where it has.
    Both detectors have data over each case's blocks. Returns the fixture, the
    events' times by case and the block (index) each case's partner must lie in.
    """
    B = fx.BLOCK_SECONDS
    patterns = {
        "train": (("train", "test"), 0, 0, 1),
        "validate": (("validate", "test"), 0, 0, 1),
        "test": (("train", "test", "train"), 1, 0, 0),
        "skip": (("test", "train", "train", "train", "test"), 1, 0, 3),
    }
    times, partner_blocks, h1, l1, after = {}, {}, [], [], -np.inf
    for name, (splits, offset, first, last) in patterns.items():
        block = _golden_block(*splits, after=after) + offset
        after = block + last + 1  # The cases' data never meet.
        times[name] = t = fx.block_start("O3", block) + B / 2
        partner_blocks[name] = block + last
        start, end = fx.block_start("O3", block + first), fx.block_start("O3", block + last + 1)
        h1.append((start, end))
        l1 += [(start, t - 5.0), (t + 5.0, end)]
    base = fx._fixture(sorted(h1), 4096.0, "O3", ("O3",))
    fixture = fx.Fixture(base.encoding, {"H1": np.array(sorted(h1)), "L1": np.array(sorted(l1))}, "O3", ("O3",))
    events = [EventRecord(f"GW{name.upper()}", t, 2.0) for name, t in times.items()]
    return fixture.with_catalogues(events=events), times, partner_blocks


def test_an_events_partner_is_test_time(world):
    """
    An event seen in only some requested detectors reads each other at a partner in
    the nearest test block of its run that holds one, whatever its own block's split:
    every event is served, and every partner span lies in test time, half a guard
    from train and validate (the golden table), in the block expected.
    """
    fixture, times, partner_blocks = _test_time_partner_world()
    with world(fixture):
        obtainer = fx.ifo_data_obtainer(fixture, data_labels=[gf.DataLabel.EVENTS])
        spans = obtainer.get_valid_segments([gf.IFO.H1, gf.IFO.L1], 7, "test")
        rejections = obtainer.get_acquisition_stats()["rejections"]
    assert "no partner" not in rejections, rejections
    by_time = {round(float(np.mean(span[0])), 3): span[1] for span in spans}
    assert sorted(by_time) == sorted(round(t, 3) for t in times.values()), sorted(by_time)
    for name, t in times.items():
        partner = by_time[round(t, 3)]
        assert fx.golden_split("O3", *partner) == "test", f"the {name} event's partner {partner}"
        gap = _other_split_distance(partner, "test")
        assert gap >= fx.GUARD_SECONDS / 2, f"the {name} event's partner {partner} lies {gap:.1f} s from train or validate"
        assert {fx.block_of("O3", x) for x in partner} == {partner_blocks[name]}, (name, partner)


def test_the_validator_reads_an_events_time_on_its_own_detectors():
    """
    A real event's time is its time on the detectors it was seen in: an event whose
    first detector is a partner is still found at its merger.
    """
    from gravyflow.src.validate.bank import event_times

    batch = {
        RV.TRANSIENT_GPS_TIME: np.array([[100.0, 100.0], [5000.0, 200.0]]),
        RV.DATA_LABEL: np.array([[2, 2], [0, 2]]),
    }
    assert event_times(batch).tolist() == [100.0, 200.0]


# =============================================================================
# TRAINING ON REAL EVENTS: ONLY UNDER THE OPT-IN
# =============================================================================

def test_training_on_real_events_needs_the_opt_in(world):
    """
    ``train_on_real_events=True`` is refused unless coincident data is opted in; under
    the opt-in an event in a train block is served in train, flagged COINCIDENT.
    """
    edge = fx.boundary("O3", "train", "train", skip=3)
    event = EventRecord("GWTRAIN", edge + 2000.0, 2.0)
    fixture = fx._fixture([(edge + 1000.0, edge + 3000.0)], 4096.0, "O3", ("O3",)).with_catalogues(
        events=[event])
    with world(fixture):
        with pytest.raises(ValueError, match="COINCIDENT"):
            fx.ifo_data_obtainer(fixture, data_labels=[gf.DataLabel.EVENTS], train_on_real_events=True)
        obtainer = fx.ifo_data_obtainer(fixture, data_labels=[gf.DataLabel.EVENTS], train_on_real_events=True,
                                        coincidence=gf.Coincidence.COINCIDENT)
        obtainer.get_valid_segments([gf.IFO.H1, gf.IFO.L1], 7, "train")
        assert [record.name for record in obtainer.transient_segments] == ["GWTRAIN"]
        batch = next(iter(_transient_batches(obtainer, "train")))
    assert batch_arrays(batch, [RV.COINCIDENT])[RV.COINCIDENT].tolist() == [1]


# =============================================================================
# AN EVALUATION PASS SERVES EACH TRANSIENT ONCE
# =============================================================================

#: More batches than any evaluation pass here serves: a pass not ended by then never ends.
EVALUATION_PASS_BOUND = 50


@pytest.mark.parametrize("balanced", [False, True], ids=["equal weights", "balanced"])
@pytest.mark.parametrize("split", ["validate", "test"])
def test_an_evaluation_pass_serves_each_transient_once(world, split, balanced):
    """
    A validate or test pass over transients serves each record once and ends, though
    its batches do not divide the records and balancing weights them unequally.
    """
    block = _golden_block("train", split, "train") + 1
    middle = fx.block_start("O3", block) + fx.BLOCK_SECONDS / 2
    peaks = [middle + 400.0 * k for k in range(-2, 3)]
    fixture = fx._fixture([(middle - 2000.0, middle + 2000.0)], 4096.0, "O3", ("O3",))
    kinds = ["Blip", "Blip", "Blip", "Koi_Fish", "Koi_Fish"]
    fixture = fixture.with_catalogues(glitches=[
        GlitchRecord("L1", t, t - 0.5, 1.0, kind, 0.95) for t, kind in zip(peaks, kinds)])
    with world(fixture):
        obtainer = fx.ifo_data_obtainer(fixture, data_labels=[gf.DataLabel.GLITCHES],
                                        balanced_glitch_types=balanced)
        batches = list(itertools.islice(_transient_batches(obtainer, split, batch=2), EVALUATION_PASS_BOUND))
    assert len(batches) < EVALUATION_PASS_BOUND, "an evaluation pass never ends"
    centres = np.concatenate([np.asarray(b[RV.TRANSIENT_GPS_TIME])[:, 1] for b in batches])
    assert sorted(centres.tolist()) == sorted(peaks), f"served {sorted(centres.tolist())}"


@pytest.mark.parametrize("split", ["validate", "test"])
def test_an_evaluation_pass_that_serves_nothing_is_refused(world, monkeypatch, split):
    """
    An evaluation pass none of whose records can be fetched is refused by name, as a
    training split is, never ended after zero batches.
    """
    block = _golden_block("train", split, "train") + 1
    middle = fx.block_start("O3", block) + fx.BLOCK_SECONDS / 2
    peaks = [middle + 400.0 * k for k in range(-2, 3)]
    fixture = fx._fixture([(middle - 2000.0, middle + 2000.0)], 4096.0, "O3", ("O3",)).with_catalogues(
        glitches=[GlitchRecord("L1", t, t - 0.5, 1.0, "Blip", 0.95) for t in peaks])
    monkeypatch.setattr(gf.TransientDataObtainer, "_download", lambda self, spans, rate: {})
    with world(fixture):
        obtainer = fx.ifo_data_obtainer(fixture, data_labels=[gf.DataLabel.GLITCHES])
        with pytest.raises(ValueError, match=f"None of the 5 transients of split '{split}' can be served"):
            list(itertools.islice(_transient_batches(obtainer, split, batch=2), EVALUATION_PASS_BOUND))
