"""
Caches serve only their own parameters: a warm catalogue cache answers a query
exactly as a cold one would, whatever was asked of it before, and the strain store
serves every configuration and split of one source the same tiles.
"""

import os
from pathlib import Path

import numpy as np
import pytest
from gwpy.segments import Segment, SegmentList

import gravyflow as gf
from gravyflow.src.dataset.acquisition import gwosc_data
from gravyflow.src.dataset.features import gravity_spy
from gravyflow.tests.test_public_sources import _fixture_rows

from . import fixtures as fx

ERAS = ("O1", "O2", "O3a", "O3b")
RUNS = {"O1": gf.ObservingRun.O1, "O2": gf.ObservingRun.O2, "O3": gf.ObservingRun.O3}


def _expected_times(ifo, era, floor):
    bounds = gwosc_data.GWOSC_RUNS[era]
    times = [float(r["event_time"]) for r in _fixture_rows(ifo, era)
             if float(r["ml_confidence"]) >= floor
             and bounds["gps_start"] <= float(r["event_time"]) <= bounds["gps_end"]]
    return np.sort(np.asarray(times))


def _segments(array):
    return SegmentList(Segment(float(a), float(b)) for a, b in np.asarray(array)).coalesce()


def _livetime(segments):
    return float(sum(abs(s) for s in segments))


def test_catalogue_caches(offline_public_sources, tmp_path):
    """
    A Gravity Spy query at a confidence floor returns exactly the catalogue's
    rows at or above it, before and after a query at another floor. The GWOSC
    timeline cache, asked for each detector and flag set in turn, gives each
    the answer a cold cache gives it alone, and a superset of flags gives a
    subset of time.
    """
    served = {}
    for ifo in ("H1", "L1"):
        for era in ERAS:
            bounds = gwosc_data.GWOSC_RUNS[era]
            for floor in (0.9, 0.0, 0.9):
                rows = gravity_spy.query(ifo, bounds["gps_start"], bounds["gps_end"], min_ml_confidence=floor)
                assert np.all(rows["ml_confidence"] >= floor)
                np.testing.assert_array_equal(np.sort(rows["event_time"]), _expected_times(ifo, era, floor),
                                              err_msg=f"{ifo} {era} at floor {floor}")
                served[(ifo, era, floor)] = rows["event_time"].size
    # The floor must matter somewhere, or the check is empty.
    assert any(served[(ifo, era, 0.0)] > served[(ifo, era, 0.9)] for ifo in ("H1", "L1") for era in ERAS)

    def segments(run, ifo, flags):
        obtainer = gf.IFODataObtainer(
            data_quality=gf.DataQuality.BEST, data_labels=[gf.DataLabel.NOISE],
            observing_runs=RUNS[run], gwosc_flags=flags,
        )
        return _segments(obtainer.get_all_segment_times(getattr(gf.IFO, ifo)))

    def timeline_requests():
        return sum("/timeline/" in url for url in offline_public_sources.requests)

    def cold(run, ifo, flags):
        """The answer of a cache that has seen nothing else: a fresh root, filled by this query alone."""
        with pytest.MonkeyPatch.context() as patch:
            patch.setenv("GRAVYFLOW_CACHE_DIR", str(tmp_path / "cold" / f"{run}-{ifo}-{len(flags)}"))
            before = timeline_requests()
            answer = segments(run, ifo, flags)
            assert timeline_requests() > before, f"{run} {ifo} {flags}: the cold query fetched nothing"
            return answer

    # One warm cache answers every query in turn: each detector and flag set,
    # in both orders, then each again. Every answer must be the cold one.
    stricter, detectors_differ = [], []
    for run in RUNS:
        cold_answers = {}
        for ifo, order in (("H1", (gf.GWOSC_BURST_FLAGS, gf.GWOSC_DATA_FLAGS)),
                           ("L1", (gf.GWOSC_DATA_FLAGS, gf.GWOSC_BURST_FLAGS))):
            got = {flags: segments(run, ifo, flags) for flags in order}
            for flags in order:
                cold_answers[(ifo, flags)] = cold(run, ifo, flags)
                assert got[flags] == cold_answers[(ifo, flags)], \
                    f"{run} {ifo} {flags}: the warm cache answers otherwise than a cold one"
            data, burst = got[gf.GWOSC_DATA_FLAGS], got[gf.GWOSC_BURST_FLAGS]
            assert _livetime(burst) > 0, f"{run} {ifo}: no time with the burst flags"
            outside = burst - data
            assert _livetime(outside) == 0, f"{run} {ifo}: more flags added {outside}"
            stricter.append(_livetime(burst) < _livetime(data))
            # Asked again, each answer comes from the warm cache, unchanged.
            before = timeline_requests()
            for flags in order:
                assert segments(run, ifo, flags) == got[flags], f"{run} {ifo} {flags} changed when warm"
            assert timeline_requests() == before, f"{run} {ifo}: the repeat was not served from the cache"
        detectors_differ.append(any(cold_answers[("H1", f)] != cold_answers[("L1", f)] for f in order))
    # The flags and the detector must matter somewhere, or the checks are empty.
    assert any(stricter), "no run where the extra flags remove time; the property is untested"
    assert any(detectors_differ), "no run where the detectors' time differs; the cache key is untested"


# =============================================================================
# THE SPLIT FINGERPRINT
# =============================================================================

SPEC_A = gf.SplitSpec()
SPEC_B = gf.SplitSpec(version="fold-1")


def _index(spec):
    from gravyflow.src.dataset.acquisition.transient_segment import TransientSegment
    from gravyflow.src.dataset.features.glitch import GlitchType
    from gravyflow.src.dataset.features.transient_index import TransientIndex
    from gravyflow.src.utils.gps import gps_to_key

    times = 1.24e9 + np.arange(20) * 5000.0
    index = TransientIndex([
        TransientSegment(gps_key=gps_to_key(t), transient_gps_time=t, start_gps_time=t - 48.2,
                         end_gps_time=t + 48.2, label=gf.DataLabel.GLITCHES, kind=GlitchType.BLIP,
                         observing_run=gf.ObservingRun.O3, seen_in=[gf.IFO.L1])
        for t in times
    ])
    index.assign_splits(spec)
    return index


def _refuses_another_split(call):
    """``call`` refuses by the fingerprint check (its message), or returns None."""
    try:
        call()
    except ValueError as error:
        return str(error) if "SplitSpec" in str(error) or "split fingerprint" in str(error) else None
    return None


def test_split_fingerprint_recorded(world, tmp_path):
    """
    The transient index file and a dataset's metadata carry the spec's fingerprint, as a
    run's record does (its own test is in test_evaluation); loading or evaluating under
    another spec, or with no split recorded, is refused; an index holding an object
    array is never unpickled.
    """
    from gravyflow.src.dataset.features.transient_index import TransientIndex
    from gravyflow.src.validate.config import check_held_out

    path = tmp_path / "index.npz"
    _index(SPEC_A).save(path)
    assert TransientIndex.load(path, SPEC_A).split_fingerprint == SPEC_A.fingerprint()
    assert _refuses_another_split(lambda: TransientIndex.load(path, SPEC_B))

    unsplit = dict(np.load(path, allow_pickle=False))
    unsplit.pop("split_fingerprint")
    np.savez(tmp_path / "unsplit.npz", **unsplit)
    assert _refuses_another_split(lambda: TransientIndex.load(tmp_path / "unsplit.npz", SPEC_A))

    pickled = {**dict(np.load(path, allow_pickle=False)), "kinds": np.array([object()], dtype=object)}
    np.savez(tmp_path / "pickled.npz", **pickled)
    with pytest.raises(ValueError, match="allow_pickle"):
        TransientIndex.load(tmp_path / "pickled.npz", SPEC_A)

    # The golden table's blocks, with another guard: a spec of its own.
    narrow = gf.SplitSpec(guard_seconds=64.0)
    fixture = fx.straddle("train", "test", half_width=512.0)
    with world(fixture):
        ds = fx.dataset(fixture, "train", split_spec=narrow)
        assert ds.seed_metadata["split_fingerprint"] == narrow.fingerprint() != SPEC_A.fingerprint()

    record ={"fingerprint": SPEC_A.fingerprint(), "selection_split": "validate"}
    check_held_out("test", SPEC_A, {"split": record})
    assert _refuses_another_split(lambda: check_held_out("test", SPEC_B, {"split": record}))
    assert _refuses_another_split(lambda: check_held_out("test", SPEC_A, {"split": {}}))
    assert _refuses_another_split(lambda: check_held_out("test", SPEC_A, {"config_sha256": "0"}))
    with pytest.raises(ValueError, match="no selection split"):
        check_held_out("test", SPEC_A, {"split": {"fingerprint": SPEC_A.fingerprint()}})


def test_fingerprint_check_sees_a_spec_mismatch_let_through(tmp_path, monkeypatch):
    """The planted leak: a loader that ignores the fingerprint is caught."""
    from gravyflow.src.dataset.features import transient_index

    monkeypatch.setattr(transient_index, "require_split_fingerprint", lambda *args, **kwargs: None)
    path = tmp_path / "index.npz"
    _index(SPEC_A).save(path)
    assert _refuses_another_split(lambda: transient_index.TransientIndex.load(path, SPEC_B)) is None


# =============================================================================
# THE STORE: a tile's content depends only on what it holds, and splits share no reading state
# =============================================================================

def _draw(batches, count=2):
    """``count`` batches of an iterator, every entry as a NumPy array."""
    return [{key: np.asarray(value) for key, value in next(batches).items()} for _ in range(count)]


def _same(a, b):
    return len(a) == len(b) and all(
        x.keys() == y.keys() and all(np.array_equal(x[key], y[key]) for key in x) for x, y in zip(a, b))


def test_warm_reads_equal_cold_and_every_rate_reads_the_same_tiles(world):
    """
    Noise: a tile's content depends only on its source, detector, rate and time,
    so a warm read equals a cold one, another output rate reads the tiles already
    fetched, and a tile of another source at the same place is never served.
    """
    from gravyflow.src.dataset.store import Store, TileKey, read

    from .audit import collect
    from .fakes import FakeSource

    fixture = fx.straddle("train", "test", half_width=512.0, rate=2048.0)
    with world(fixture) as harness:
        root = Path(os.environ["GRAVYFLOW_CACHE_DIR"])
        assert root in read.process_store().root.parents
        # A canary: another source's tile at the very place the reads use.
        start = float(fixture.segments["H1"][0][0])
        tile = int(FakeSource.TILE_SECONDS)
        canary = TileKey("fake-canary", "H1", 2048, tile, int(start // tile) + 2)
        Store(root / "store").commit(canary, np.full(canary.length, 7.0))

        cold = _draw(fx.noise_batches(fixture, "train", rate=2048.0))
        fetched = len(harness.source.snapshot()[0])
        assert fetched > 0
        _draw(fx.noise_batches(fixture, "train", rate=1024.0))  # resampled from the same tiles
        warm = _draw(fx.noise_batches(fixture, "train", rate=2048.0))
        assert len(harness.source.snapshot()[0]) == fetched, "a warm read, or another rate, fetched"
        assert _same(cold, warm)
        # The warm build decodes to the fake's own samples: no canary byte was served.
        records = collect(harness, {"warm": ("train", fx.noise_batches(fixture, "train", rate=2048.0))},
                          2, ["H1", "L1"], fx.window(2048.0), onsource_cropped=False, warm=True)
        assert records["warm"].num_examples == 16


def test_alternating_splits_on_one_obtainer_change_neither_splits_bytes(world):
    """
    The store: train and validate drawn in turn from one obtainer serve
    exactly what each serves alone, and neither stream ends early. Tiles carry no
    split and no process keeps their contents, so no reading state passes between them.
    """
    fixture = fx.straddle("validate", "train", half_width=512.0)
    splits = ("train", "validate")
    with world(fixture):
        alone = {split: _draw(fx.noise_batches(fixture, split), 3) for split in splits}
        noise = fx.noise_obtainer(fixture)
        streams = {split: noise(
            sample_rate_hertz=256.0, onsource_duration_seconds=fx.ONSOURCE_SECONDS,
            crop_duration_seconds=fx.CROP_SECONDS, offsource_duration_seconds=fx.OFFSOURCE_SECONDS,
            num_examples_per_batch=8, group=split, seed=7, sampling_mode=gf.SamplingMode.RANDOM,
        ) for split in splits}
        turns = {split: [] for split in splits}
        for _ in range(3):
            for split in splits:
                turns[split] += _draw(streams[split], 1)
    for split in splits:
        assert _same(turns[split], alone[split]), split


# =============================================================================
# TRANSIENTS: every example holds its own time's strain, from the tiles as they are
# =============================================================================

def _forget_every_tile():
    """Forget every tile the process's store holds, as another holder may: the count."""
    from gravyflow.src.dataset.store import read
    from gravyflow.src.dataset.store.tile import key_from_path

    store = read.process_store()
    keys = [key_from_path(store.root, path) for path in store.root.rglob("*.json")]
    store.forget(keys)
    return len(keys)


def _assert_own_strain(batch, offset=0.0, rate=2048.0):
    """
    Every detector of every example, served at ``rate``, is centred on the sample nearest
    its own centre, in the strain of the generation valued its GPS time minus ``REF`` plus
    ``offset``.
    """
    from gravyflow.tests import test_transient_windows as tw

    centres = tw.transient_times_from(batch, tw.REF - offset)
    error = centres - tw.nearest_sample(batch[gf.ReturnVariables.TRANSIENT_GPS_TIME], rate)
    assert np.max(np.abs(error)) <= 0.25 / rate, f"an example holds another time's strain: {error} s"
    tw.assert_never_filled(batch)


def test_transient_rows_match_key():
    """
    An example served for a record holds the strain of that record's time from the tiles
    as they are when it is read. Holder A reads; holder B forgets every tile and refills
    them after the source republishes other bytes; holder B forgets every tile again in
    the middle of one of A's reads. Every read of A serves its own times, from the
    generation the store holds: tiles have no row index or chunk to go stale.
    """
    from gravyflow.src.dataset.store import read
    from gravyflow.tests import test_transient_windows as tw

    class ForgettingFetch(tw.GridFetch):
        """The fake fetch; at its ``forget_at``-th call, holder B forgets every tile."""

        forget_at = None

        def __call__(self, ifo, sample_rate_hertz, interval):
            strain = super().__call__(ifo, sample_rate_hertz, interval)
            if len(self.calls) == self.forget_at:
                assert _forget_every_tile() > 0
            return strain

    both = [gf.IFO.H1, gf.IFO.L1]
    fetch = ForgettingFetch()
    with tw.offline(fetch):
        holder_a = tw.obtainer()(ifos=both, group=tw.SPLIT, seed=1,
                                 **{**tw.WINDOW, "num_examples_per_batch": 1})
        _assert_own_strain(next(holder_a))
        # B forgets every tile; the source republishes; B refills at another rate.
        assert _forget_every_tile() > 0
        fetch.offset = 1000.0
        _assert_own_strain(tw.first_batch(both, sample_rate_hertz=1024.0), fetch.offset, 1024.0)
        _assert_own_strain(next(holder_a), fetch.offset)
        # B forgets every tile, and again while A is inside a read (at A's second
        # fetch, with its first tile committed and open), then A reads on.
        assert _forget_every_tile() > 0
        fetch.forget_at = len(fetch.calls) + 2
        for _ in range(4):
            _assert_own_strain(next(holder_a), fetch.offset)
        assert len(fetch.calls) > fetch.forget_at, "no read followed the forget"
    assert read.process_store().verify() == [], "the store was left inconsistent"


def test_transient_warm_equals_cold():
    """
    Transients: two configurations differing in exactly one key (the rate) on one root
    read the same tiles, so a warm read equals a cold one; a tile of another source at
    the very place the reads use is never served; and everything written is under
    ``GRAVYFLOW_CACHE_DIR``, in the store: no transient cache or download cache.
    """
    from gravyflow.src.dataset.store import Store, TileKey
    from gravyflow.tests import test_transient_windows as tw

    root = Path(os.environ["GRAVYFLOW_CACHE_DIR"])
    before = {path for path in root.rglob("*") if path.is_file()}
    both = [gf.IFO.H1, gf.IFO.L1]
    canary = TileKey("fake-canary", "L1", 4096, tw.TILE_SECONDS, tw.tile_of(tw.TIMES[0]))
    Store(root / "store").commit(canary, np.full(canary.length, 7.0))
    with tw.offline() as fetch:
        cold = tw.first_batch(both)
        fetched = len(fetch.calls)
        other = tw.first_batch(both, sample_rate_hertz=1024.0)
        warm = tw.first_batch(both)
    assert fetched > 0 and len(fetch.calls) == fetched, "a warm read, or another rate, fetched"
    for batch, rate in ((cold, 2048.0), (other, 1024.0), (warm, 2048.0)):
        _assert_own_strain(batch, rate=rate)
    for key in cold:
        np.testing.assert_array_equal(np.asarray(cold[key]), np.asarray(warm[key]), err_msg=str(key))
    written = {path for path in root.rglob("*") if path.is_file()} - before
    assert written and all((root / "store") in path.parents for path in written), sorted(written)
