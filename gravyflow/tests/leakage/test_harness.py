"""
Self-tests of the leakage harness: the instrument is proven before it judges.

Every decoder refusal and every assertion helper is shown to refuse a planted
defect, so a helper that would pass vacuously fails here.
"""

import os
import socket
from pathlib import Path

import numpy as np
import pytest

import gravyflow as gf

from . import audit
from . import fixtures as fx
from .fakes import EventRecord, GlitchRecord, StrainSource
from .provenance import (
    CODE_CEILING, DecodeRefused, Encoding, OutsideFixture, decode_row, decode_rows,
)

RV = gf.ReturnVariables
IFOS = ["H1", "L1"]


# =============================================================================
# ROUND TRIPS: every sample the library returns decodes exactly
# =============================================================================

@pytest.mark.parametrize("rate", [256.0, 2048.0])
@pytest.mark.parametrize("mode", [gf.SamplingMode.RANDOM, gf.SamplingMode.GRID])
def test_noise_round_trip_decodes_every_sample(world, tmp_path, rate, mode):
    fixture = fx.straddle("train", "test", half_width=128.0 if rate > 256 else 512.0, rate=rate)
    window = fx.window(rate)
    with world(fixture) as harness:
        obtained = fx.noise_batches(fixture, tmp_path, "train", rate=rate, mode=mode)
        ds = fx.dataset(fixture, tmp_path, "train", rate=rate, mode=mode)
        records = audit.collect(harness, {"obtainer": ("train", obtained)}, 2, IFOS,
                                window, onsource_cropped=False)
        records.update(audit.collect(harness, {"dataset": ("train", fx.dataset_batches(ds))}, 2,
                                     IFOS, window, onsource_cropped=True))
    for record in records.values():
        assert record.num_examples == 16
        # collect decoded every row; each spans its window exactly.
        lengths = record.footprints[..., 1] - record.footprints[..., 0]
        assert np.all(lengths == window.offsource + window.onsource + 2 * window.crop)
    audit.assert_inside_one_fetch(records.values(), harness)


def _train_record_time(start):
    """
    The first time from ``start``, in 10 s steps, whose transient record the golden
    table puts in train: its footprint, 48.2 s each side, lies in train blocks.
    """
    for step in range(100):
        time = start + 10.0 * step
        if fx.golden_split("O3", time - 48.2, time + 48.2) == "train":
            return time
    raise LookupError(f"no train record time within 1000 s of {start}")


def test_transient_round_trip_decodes_every_sample(world, tmp_path):
    """
    Glitch windows, fetched at 4096 Hz and thinned by stride to 256 Hz, decode
    exactly; the padded onsource of the detector a glitch was seen in is centred on
    it, and the other detector's on a partner time away from it.
    """
    edge = fx.boundary("O3", "train", "train", skip=3)
    peaks = np.array([_train_record_time(edge + 1000.0), _train_record_time(edge + 3000.0)])
    fixture = fx._fixture([(t - 400.0, t + 400.0) for t in peaks], 4096.0, "O3", ("O3",))
    fixture = fixture.with_catalogues(glitches=[
        GlitchRecord("H1", peaks[0], peaks[0] - 0.5, 1.0, "Blip", 0.95),
        GlitchRecord("L1", peaks[1], peaks[1] - 0.5, 1.0, "Blip", 0.95),
    ])
    window = fx.window(256.0)
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
            offsource_duration_seconds=fx.OFFSOURCE_SECONDS, num_examples_per_batch=2,
            group="train", seed=7, ifos=[gf.IFO.H1, gf.IFO.L1],
            input_variables=[RV.ONSOURCE, RV.OFFSOURCE], output_variables=[RV.START_GPS_TIME],
        )
        # The transient path takes its offsource from earlier in the cached window,
        # not adjacent to the onsource, so the seam is not required here.
        records = audit.collect(harness, {"glitches": ("train", fx.dataset_batches(ds))}, 1,
                                IFOS, window, onsource_cropped=True, stride=16, adjacent=False)
    record = records["glitches"]
    padded_end = record.footprints[..., 1] / 4096.0
    centres = padded_end - (fx.ONSOURCE_SECONDS / 2 + fx.CROP_SECONDS)
    assert record.num_examples == 2
    # The H1 glitch on H1 (column 0), the L1 glitch on L1 (column 1).
    seen = [int(np.argmin(np.abs(centres[:, 0] - peaks[0]))), int(np.argmin(np.abs(centres[:, 1] - peaks[1])))]
    assert sorted([centres[seen[0], 0], centres[seen[1], 1]]) == sorted(peaks)
    # The other detector of each example is read two record spans away.
    assert np.all(np.abs(centres[:, 0] - centres[:, 1]) >= 4 * 48.2)


# =============================================================================
# THE DECODER REFUSES, BY NAME
# =============================================================================

@pytest.fixture
def code():
    return Encoding.from_intervals([(1000.0, 1010.0), (2000.0, 2010.0)], 256.0)


def _row(code, ifo="H1", first=1000 * 256 + 5, n=64):
    return code.encode(ifo, first, n).astype(np.float32)


def test_decoder_reads_detector_island_and_sample(code):
    span = decode_row(_row(code, "L1", 2000 * 256 + 7, 32), "L1", code)
    assert (span.ifo, span.first, span.num_samples) == ("L1", 2000 * 256 + 7, 32)
    with pytest.raises(OutsideFixture):
        code.encode("H1", 1009 * 256, 512)  # runs past the island's end


@pytest.mark.parametrize("defect, reason", [
    (lambda r: np.zeros_like(r), "zero samples"),
    (lambda r: r * 1e21, "outside the code"),
    (lambda r: r * 1e-21, "outside the code"),
    (lambda r: r + 0.5, "not integer codes"),
    (lambda r: r * 1.5, "not integer codes"),
    (lambda r: np.delete(r, 20), "dropped sample"),
    (lambda r: np.insert(r, 20, r[20]), "duplicated sample"),
    (lambda r: r[::-1].copy(), "time-reversed"),
    (lambda r: -r, "sign-flipped"),
    (lambda r: np.concatenate([r[:10], r[30:]]), "dropped sample"),
    (lambda r: np.where(np.arange(r.size) == 7, np.nan, r), "non-finite"),
])
def test_decoder_refuses_each_defect(code, defect, reason):
    with pytest.raises(DecodeRefused, match=reason):
        decode_row(defect(_row(code)), "H1", code)


def test_decoder_refuses_swapped_detectors(code):
    with pytest.raises(DecodeRefused, match="detectors swapped"):
        decode_row(_row(code, "L1"), "H1", code)
    rows = np.stack([_row(code, "L1"), _row(code, "H1")])[None]
    with pytest.raises(DecodeRefused, match=r"\[0, 0\] \(H1\)"):
        decode_rows(rows, IFOS, code)


def test_decoder_refuses_values_outside_the_fixture(code):
    with pytest.raises(DecodeRefused, match="outside"):
        decode_row(np.arange(CODE_CEILING, CODE_CEILING + 8, dtype=np.float64), "H1", code)
    # A row of the same code past the last island (the code extends; the fixture does not).
    wider = Encoding.from_intervals([(1000.0, 1010.0), (2000.0, 2020.0)], 256.0)
    beyond = wider.encode("H1", 2015 * 256, 300)
    with pytest.raises(DecodeRefused, match="outside the fixture"):
        decode_row(beyond, "H1", code)
    # A row too short to carry its block's digits refuses rather than guesses.
    with pytest.raises(DecodeRefused, match="too short"):
        decode_row(_row(code, n=4), "H1", code)


def test_decoder_reads_an_exact_stride_only(code):
    row = code.encode("H1", 1000 * 256, 640)[::16]
    assert decode_row(row, "H1", code, stride=16).end == 1000 * 256 + 640
    with pytest.raises(DecodeRefused, match="expected 8"):
        decode_row(row, "H1", code, stride=8)


def test_the_fake_refuses_requests_outside_the_fixture(code):
    source = StrainSource(code)
    with pytest.raises(OutsideFixture):
        source.fetch(1005.0, 1020.0, gf.IFO.H1, sample_rate_hertz=256.0)
    with pytest.raises(OutsideFixture):
        source.fetch(1001.0, 1002.0, gf.IFO.H1, sample_rate_hertz=4096.0)
    fetches, errors = source.snapshot()
    assert fetches == () and len(errors) == 2
    served = source.fetch(1001.0, 1002.0, gf.IFO.L1, sample_rate_hertz=256.0)
    assert decode_row(np.asarray(served.value), "L1", code).first == 1001 * 256


def test_encoding_refuses_a_layout_beyond_the_code():
    with pytest.raises(ValueError, match="the code holds"):
        Encoding.from_intervals([(0.0, 10.0), (3e8, 3e8 + 10.0)], 256.0)
    with pytest.raises(ValueError, match="power of two"):
        Encoding.from_intervals([(0.0, 10.0)], 1000.0)
    with pytest.raises(ValueError, match="overlap"):
        Encoding.from_intervals([(0.0, 10.0), (5.0, 20.0)], 256.0)


# =============================================================================
# COLLECTION: cold builds must fetch, warm builds must not
# =============================================================================

def _synthetic_batch(code, firsts, window):
    """A batch cut straight from the code, without the fake fetch."""
    on, off = [], []
    for first in firsts:
        off.append([code.encode(ifo, first, window.offsource) for ifo in IFOS])
        on.append([code.encode(ifo, first + window.offsource + window.crop, window.onsource)
                   for ifo in IFOS])
    return {RV.ONSOURCE: np.array(on), RV.OFFSOURCE: np.array(off)}


def _fetching_then(harness, start, batch):
    harness.source.fetch(start, start + 1.0, gf.IFO.H1, sample_rate_hertz=256.0)
    yield batch


def test_collect_refuses_cold_without_fetches_warm_with_them_and_errors(world):
    fixture = fx.straddle("train", "test", half_width=64.0)
    window = fx.window()
    start = float(fixture.segments["H1"][0][0])
    with world(fixture) as harness:
        first = fixture.encoding.islands[0].first
        batch = _synthetic_batch(fixture.encoding, [first + 100 * k for k in range(8)], window)
        with pytest.raises(AssertionError, match="fetched nothing"):
            audit.collect(harness, {"a": ("train", iter([batch]))}, 1, IFOS, window, True)
        records = audit.collect(harness, {"a": ("train", iter([batch]))}, 1, IFOS, window, True,
                                warm=True)
        assert records["a"].num_examples == 8
        with pytest.raises(AssertionError, match="should read its cache"):
            audit.collect(harness, {"b": ("train", _fetching_then(harness, start, batch))}, 1,
                          IFOS, window, True, warm=True)
        with pytest.raises(OutsideFixture):
            harness.source.fetch(1.0, 2.0, gf.IFO.H1, sample_rate_hertz=256.0)
        with pytest.raises(AssertionError, match="refused 1 fetch"):
            audit.collect(harness, {"c": ("train", _fetching_then(harness, start, batch))}, 1,
                          IFOS, window, True)


def test_warm_build_reads_the_cache_and_equals_the_cold_build(world, tmp_path):
    fixture = fx.straddle("train", "test", half_width=256.0)
    window = fx.window()
    with world(fixture) as harness:
        builds = {}
        for warm in (False, True):
            batches = fx.noise_batches(fixture, tmp_path / "segments", "train", cache_segments=True)
            builds[warm] = audit.collect(harness, {"train": ("train", batches)}, 2, IFOS, window,
                                         False, warm=warm)["train"]
    audit.assert_warm_equals_cold(builds[False], builds[True])
    planted = audit.Records(**{**builds[True].__dict__, "footprints": builds[True].footprints + 1})
    with pytest.raises(AssertionError, match="differ"):
        audit.assert_warm_equals_cold(builds[False], planted)


# =============================================================================
# EVERY ASSERTION HELPER REFUSES A PLANTED LEAK
# =============================================================================

def _records(name, split, seconds, rate=256.0):
    """Records whose every footprint is given in seconds: (N, I, 2)."""
    prints = np.rint(np.asarray(seconds, dtype=np.float64) * rate).astype(np.int64)
    return audit.Records(name, split, tuple(IFOS), prints, rate)


def _block(split, offset=100.0, n=8, origin="O3"):
    """n pairs of 6 s footprints inside the first table block of ``split``."""
    edge = fx.boundary(origin, "validate" if split == "train" else "train", split)
    starts = edge + offset + 10.0 * np.arange(n)
    one = np.stack([starts, starts + 6.0], axis=-1)
    return np.stack([one, one + 1000.0], axis=1)


def test_assert_in_split_refuses_a_footprint_in_another_split():
    audit.assert_in_split([_records("t", "test", _block("test"))], "O3")
    planted = _block("test")
    planted[3, 1] = _block("train")[0, 0]
    with pytest.raises(AssertionError, match="in a train block"):
        audit.assert_in_split([_records("t", "test", planted)], "O3")
    edge = fx.boundary("O3", "train", "test")
    crossing = _block("test")
    crossing[0, 0] = [edge - 3.0, edge + 3.0]
    with pytest.raises(AssertionError, match="crosses the edge"):
        audit.assert_in_split([_records("t", "test", crossing)], "O3")
    outside = _block("test")
    outside[0, 0] = [fx.origin_gps("O3") + 1e7, fx.origin_gps("O3") + 1e7 + 6.0]
    with pytest.raises(AssertionError, match="outside the golden table"):
        audit.assert_in_split([_records("t", "test", outside)], "O3")


def test_assert_served_in_split_refuses_served_time_past_a_block_edge():
    """A served segment that crosses into, or lies in, another split's block fails."""
    edge = fx.boundary("O3", "train", "test")
    served = {"H1": [(edge, edge + 600.0)], "L1": [(edge + 10.0, edge + 2048.0)]}
    audit.assert_served_in_split(served, "test", "O3")
    with pytest.raises(AssertionError, match="crosses the edge"):
        audit.assert_served_in_split({**served, "L1": [(edge - 0.5, edge + 600.0)]}, "test", "O3")
    with pytest.raises(AssertionError, match="in a train block"):
        audit.assert_served_in_split({**served, "H1": [(edge - 600.0, edge)]}, "test", "O3")
    with pytest.raises(AssertionError, match="is empty"):
        audit.assert_served_in_split({"H1": [], "L1": []}, "test", "O3")


def test_assert_split_gap_refuses_close_splits():
    edge = fx.boundary("O3", "train", "test")
    train = np.stack([np.stack([[edge - 300.0 - 7 * k, edge - 294.0 - 7 * k]] * 2) for k in range(8)])
    test = np.stack([np.stack([[edge + 200.0 + 7 * k, edge + 206.0 + 7 * k]] * 2) for k in range(8)])
    audit.assert_split_gap([_records("a", "train", train), _records("b", "test", test)], 128.0)
    planted = test.copy()
    planted[5, 1] = [edge - 250.0, edge - 244.0]  # 44 s after train's last footprint
    with pytest.raises(AssertionError, match="closer than 128"):
        audit.assert_split_gap([_records("a", "train", train), _records("b", "test", planted)], 128.0)
    with pytest.raises(AssertionError, match="two splits"):
        audit.assert_split_gap([_records("a", "train", train)], 128.0)


def test_assert_avoids_refuses_forbidden_time():
    prints = _block("train")
    record = _records("a", "train", prints)
    clear = [(prints[0, 0, 0] - 100.0, prints[0, 0, 0] - 50.0)]
    audit.assert_avoids([record], clear)
    hit = [(prints[4, 1, 0] + 1.0, prints[4, 1, 0] + 2.0)]
    with pytest.raises(AssertionError, match="forbidden time served"):
        audit.assert_avoids([record], hit)
    with pytest.raises(AssertionError, match=r"\('L1'"):
        audit.assert_avoids([record], {"L1": hit, "H1": clear})
    with pytest.raises(AssertionError, match="vacuous"):
        audit.assert_avoids([record], [])
    served = {"H1": [(0.0, 100.0)], "L1": [(0.0, 100.0)]}
    with pytest.raises(AssertionError, match="forbidden time served"):
        audit.assert_avoids(served, [(50.0, 51.0)])
    with pytest.raises(AssertionError, match="served-time set is empty"):
        audit.assert_avoids({"H1": []}, [(50.0, 51.0)])


def test_assert_inside_refuses_a_footprint_outside_the_intervals():
    prints = _block("train")
    record = _records("a", "train", prints)
    span = (prints[..., 0].min() - 1.0, prints[..., 1].max() + 1.0)
    audit.assert_inside([record], {"H1": [span], "L1": [span]})
    with pytest.raises(AssertionError, match="outside the allowed"):
        audit.assert_inside([record], {"H1": [span], "L1": [(span[0], span[1] - 10.0)]})
    with pytest.raises(AssertionError, match="outside the allowed"):
        audit.assert_inside([record], {"H1": [span]})


def test_assert_inside_one_fetch_refuses_a_footprint_across_two_fetches(world):
    fixture = fx.straddle("train", "test", half_width=64.0)
    with world(fixture) as harness:
        start = float(fixture.segments["H1"][0][0])
        for ifo in (gf.IFO.H1, gf.IFO.L1):
            harness.source.fetch(start, start + 50.0, ifo, sample_rate_hertz=256.0)
            harness.source.fetch(start + 50.0, start + 100.0, ifo, sample_rate_hertz=256.0)
        inside = np.stack([np.stack([[start + k, start + k + 6.0]] * 2) for k in range(8)])
        audit.assert_inside_one_fetch([_records("a", "train", inside)], harness)
        across = inside.copy()
        across[2, 0] = [start + 47.0, start + 53.0]  # inside the union, not inside one fetch
        with pytest.raises(AssertionError, match="outside every fetch"):
            audit.assert_inside_one_fetch([_records("a", "train", across)], harness)


def test_helpers_refuse_empty_and_small_inputs():
    with pytest.raises(AssertionError, match="no records"):
        audit.assert_in_split([], "O3")
    small = _records("a", "train", _block("train", n=3))
    with pytest.raises(AssertionError, match="at least 8"):
        audit.assert_in_split([small], "O3")
    with pytest.raises(AssertionError, match="at least 8"):
        audit.assert_avoids([small], [(0.0, 1.0)])


# =============================================================================
# THE GOLDEN TABLE, THE CATALOGUES AND THE GUARD
# =============================================================================

def test_golden_table_agrees_with_the_library():
    """
    Every literal of the table is the split the library admits a second of the block's
    middle to, and no other split admits it.
    """
    spec = gf.SplitSpec()
    checked = 0
    for origin in fx.GOLDEN_BLOCKS:
        for block, split in fx.golden_blocks(origin).items():
            t = fx.block_start(origin, block) + fx.BLOCK_SECONDS / 2.0
            for name in ("train", "validate", "test"):
                admitted = len(spec.admissible([[t, t + 1.0]], name)) == 1
                assert admitted == (name == split), (origin, block, split, name)
            checked += 1
    assert checked == 1200
    assert set(fx.golden_blocks("O3").values()) == {"train", "validate", "test"}


def test_catalogues_reach_every_consumer(world):
    """One sentinel event and one glitch reach the producers and the noise veto."""
    from gravyflow.src.dataset.features import event as event_module
    from gravyflow.src.dataset.features import segment_builders

    # Wide enough that train keeps time beyond the event's footprint and guard.
    fixture = fx.straddle("train", "test", half_width=512.0)
    edge = float(np.mean(fixture.segments["H1"][0]))
    sentinel = EventRecord("GWSENTINEL", edge - 100.4, 2.0)
    glitch = GlitchRecord("L1", edge - 50.0, edge - 50.5, 1.0, "Koi_Fish", 0.97)
    fixture = fixture.with_catalogues(glitches=[glitch], events=[sentinel])
    with world(fixture):
        assert [e["name"] for e in event_module.get_events_with_params()] == ["GWSENTINEL"]
        assert [e["name"] for e in segment_builders.get_events_with_params()] == ["GWSENTINEL"]
        assert event_module.get_all_event_times().tolist() == [sentinel.gps]
        assert event_module.get_event_catalogue()["name"].tolist() == ["GWSENTINEL"]
        rows = gf.get_glitch_catalogue(gf.IFO.L1, observing_run=gf.ObservingRun.O3)
        assert rows["event_time"].tolist() == [glitch.peak]
        assert len(gf.get_glitch_catalogue(gf.IFO.H1, observing_run=gf.ObservingRun.O3)["event_time"]) == 0
        served = fx.served_segments(fixture, IFOS, "train")
    # The noise vetoes removed the sentinel event and the glitch from the train side.
    for ifo in IFOS:
        audit.assert_avoids({ifo: served[ifo]}, [sentinel.extent])
    audit.assert_avoids({"L1": served["L1"]}, [glitch.extent])


def test_the_guard_refuses_and_records_every_socket(refusals):
    """Remote and loopback connections, and lookups of another host, are refused by name."""
    with pytest.raises(ConnectionRefusedError, match="name lookup of 'gwosc.org'"):
        socket.getaddrinfo("gwosc.org", 443)
    for address in (("203.0.113.9", 80), ("127.0.0.1", 9)):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            with pytest.raises(ConnectionRefusedError, match="refused connection to"):
                sock.connect(address)
            with pytest.raises(ConnectionRefusedError, match="refused connection to"):
                sock.connect_ex(address)
        finally:
            sock.close()
    socket.getaddrinfo("localhost", 80)  # this machine's own name still resolves
    assert len(refusals.seen) == 5, refusals.seen


def test_without_the_fake_real_noise_is_refused_and_zero(world, tmp_path, refusals):
    """With the real fetch in place, the guard refuses it, and every row is zero-filled."""
    # Each side keeps its half width less half the 128 s guard.
    fixture = fx.straddle("train", "test", half_width=128.0)
    with world(fixture, strain=False):
        batch = next(fx.noise_batches(fixture, tmp_path, "train"))
    assert refusals.seen, "the real fetch reached no socket"
    onsource = np.asarray(batch[RV.ONSOURCE])
    assert onsource.size and np.all(onsource == 0)
    with pytest.raises(DecodeRefused, match="zero samples"):
        decode_rows(onsource, IFOS, fixture.encoding)


def test_every_cache_lives_under_the_one_cache_root():
    from gravyflow.src.dataset.acquisition import gwosc_data
    from gravyflow.src.dataset.features import gravity_spy
    from gravyflow.src.dataset.features.glitch_cache import generate_transient_cache_path

    root = Path(os.environ["GRAVYFLOW_CACHE_DIR"])
    obtainer = gf.IFODataObtainer(
        observing_runs=gf.ObservingRun.O3, data_quality=gf.DataQuality.BEST,
        data_labels=[gf.DataLabel.NOISE],
    )
    for path in (
        generate_transient_cache_path(prefix="glitch"),
        generate_transient_cache_path(prefix="event"),
        gravity_spy.cache_dir(),
        gwosc_data._timeline_cache_path("O1", "H1_DATA", 0, 1),
        obtainer.generate_file_path(256.0, "train", gf.NoiseObtainer().data_directory_path),
    ):
        assert root in Path(path).parents, path


def test_strain_downloads_are_cached_under_the_one_cache_root(monkeypatch):
    """gwpy's GWOSC strain download goes to astropy's cache, which is moved under the root."""
    from astropy.config import get_cache_dir_path
    from gwpy.timeseries import TimeSeries

    from gravyflow.src.dataset.acquisition import base

    def no_local_frames(**kwargs):
        raise ValueError("no local frames")

    seen = []

    def fetch_open_data(*args, **kwargs):
        seen.append((get_cache_dir_path(), kwargs.get("cache")))
        return TimeSeries(np.ones(16), t0=args[1], sample_rate=16)

    monkeypatch.setattr(base, "find_urls", no_local_frames)
    monkeypatch.setattr(TimeSeries, "fetch_open_data", fetch_open_data)
    base.BaseDataObtainer.get_segment_data_static(1238166018.0, 1238166019.0, gf.IFO.L1, "HOFT_C01",
                                                   "DCS-CALIB_STRAIN_CLEAN_C01", 4096.0)
    (directory, cache), = seen
    assert cache is True
    assert Path(os.environ["GRAVYFLOW_CACHE_DIR"]) in directory.parents, directory
    assert Path(os.environ["GRAVYFLOW_CACHE_DIR"]) not in get_cache_dir_path().parents
