"""
Detectors are never read at coincident times by default; a glitch's other detectors
get partner windows.

"""

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


def _glitch_world(tmp_path, glitch_ifo="L1"):
    edge = fx.boundary("O3", "train", "train", skip=3)
    peak = edge + 1000.0
    fixture = fx._fixture([(peak - 400.0, peak + 400.0)], 4096.0, "O3", ("O3",))
    rows = [GlitchRecord(glitch_ifo, peak, peak - 0.5, 1.0, "Blip", 0.95)]
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
    fixture, peak = _glitch_world(tmp_path)
    record, _, _ = _glitch_batch(world, tmp_path, fixture)
    separation = float(np.abs(np.diff(record.footprints[0, :, 0])) / 4096.0)
    assert separation >= 2 * FOOTPRINT_SECONDS, f"H1 and L1 read {separation} s apart"


def test_partner_windows(world, tmp_path):
    """
    An L1-only glitch over [H1, L1]: H1 gets a partner window away from the glitch,
    labelled NOISE, with L1 labelled GLITCH; no H1 fetch overlaps the L1 window; and
    two builds choose the same partner.
    """
    fixture, peak = _glitch_world(tmp_path)
    first, labels, fetches = _glitch_batch(world, tmp_path, fixture)
    second, _, _ = _glitch_batch(world, tmp_path, fixture)
    h1_start = first.footprints[0, 0, 0] / 4096.0
    l1 = first.footprints[0, 1] / 4096.0
    assert abs(h1_start - l1[0]) >= 2 * FOOTPRINT_SECONDS, "H1 is read at the glitch's time"
    assert labels.tolist() == [[0.0, 1.0]], f"labels {labels.tolist()}, expected [NOISE, GLITCH]"
    assert np.array_equal(first.footprints, second.footprints)
    h1_fetches = [(f.first / 4096.0, f.stop / 4096.0) for f in fetches if f.ifo == "H1"]
    assert all(b <= l1[0] or a >= l1[1] for a, b in h1_fetches), "an H1 fetch covers the L1 window"
