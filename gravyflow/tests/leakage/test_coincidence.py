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
from .fakes import GlitchRecord

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
