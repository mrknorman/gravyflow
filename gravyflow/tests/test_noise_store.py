"""
Real noise read through the store: a read is a tile view, the one resampler
and the margin; the timeline is the authority, so a source that does not deliver what it
says it holds is refused by name, never zero-filled; nothing is truncated or overwritten
on a read error; old segment caches are never read; and the memory cap holds.

Every test reads a ``ToneSource``: band-limited strain computed from GPS time, offline,
on small tiles, with planted outages and gaps.
"""

import threading
import tracemalloc
import warnings
from math import gcd
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
import pytest

import gravyflow as gf
from gravyflow.src.dataset.acquisition.sources import SegmentStrain
from gravyflow.src.utils.resample import resample, resample_margin

#: A tile boundary of the tone source's 64 s grid (and of GWOSC's 4096 s grid), in O3a.
T0 = 1_238_171_648.0
#: A piece that crosses one tile boundary, well inside two tiles.
PIECE = (T0 + 10.0, T0 + 110.0)
RATE = 2048.0


class ToneSource(gf.DataSource):
    """
    Strain that is a sum of tones below 900 Hz, a pure function of GPS time; NaN in
    ``gaps``, and every fetch touching ``outages`` refused. Counts fetches per thread.
    """

    TONES = ((37.0, 0.3), (211.0, 1.1), (877.0, 2.0))

    def __init__(self, native: float = 4096.0, tile: float = 64.0,
                 gaps: Sequence[Tuple[float, float]] = (), outages: Sequence[Tuple[float, float]] = (),
                 data: Tuple[float, float] = (T0 - 4096.0, T0 + 4096.0)):
        self.native, self.tile = float(native), float(tile)
        self.gaps, self.outages, self.data = tuple(gaps), tuple(outages), tuple(data)
        self.fetches = 0
        self._lock = threading.Lock()

    def segments(self, detector, flags, interval):
        start, end = max(interval[0], self.data[0]), min(interval[1], self.data[1])
        return np.array([[start, end]]) if end > start else np.empty((0, 2))

    def rates(self):
        return (self.native,)

    def tile_seconds(self, sample_rate_hertz):
        return self.tile

    def source_id(self, gps, sample_rate_hertz):
        return f"tone-{self.native:g}"

    def fetch(self, detector, sample_rate_hertz, interval):
        start, end = map(float, interval)
        with self._lock:
            self.fetches += 1
        if any(a < end and start < b for a, b in self.outages):
            raise gf.StrainUnavailable(self, detector, interval, "planted outage")
        first, stop = int(np.ceil(start * self.native)), int(np.ceil(end * self.native))
        t = np.arange(first, stop, dtype=np.float64)
        t -= T0 * self.native
        t /= self.native
        values, tone = np.zeros(len(t)), np.empty(len(t))
        for frequency, phase in self.TONES:  # in place: a lean decode, as a source's is
            np.multiply(t, 2 * np.pi * frequency, out=tone)
            tone += phase
            values += np.sin(tone, out=tone)
        for a, b in self.gaps:
            values[(t + T0 >= a) & (t + T0 < b)] = np.nan
        return SegmentStrain(values, first / self.native, self.native, self.source_id(start, self.native))

    def specification(self):
        return {"native": self.native, "tile": self.tile, "gaps": self.gaps,
                "outages": self.outages, "data": self.data}


@pytest.fixture
def cache(tmp_path, monkeypatch) -> Path:
    """A cache root of the test's own, so its store starts empty."""
    root = tmp_path / "cache"
    monkeypatch.setenv("GRAVYFLOW_CACHE_DIR", str(root))
    return root


@pytest.fixture
def quick_retries(monkeypatch):
    """No backoff between a failed fetch's retries."""
    from gravyflow.src.dataset.store import read

    monkeypatch.setattr(read, "FETCH_BACKOFF_SECONDS", 0.0)


def obtainer(source, **kwargs):
    return gf.NoiseDataObtainer(
        data_quality=gf.DataQuality.BEST, data_labels=[gf.DataLabel.NOISE],
        observing_runs=[gf.ObservingRun.O3], augmentations=[], source=source, **kwargs,
    )


def acquire(source, piece=PIECE, rate=RATE, ifo=gf.IFO.H1, **kwargs):
    return list(obtainer(source, **kwargs).acquire(rate, np.array([[piece]]), [ifo]))


def interior(native: float, rate: float, piece) -> Tuple[int, int]:
    """The test's own oracle for the first sample and length a read of ``piece`` serves."""
    divisor = gcd(int(native), int(rate))
    up, down = int(rate) // divisor, int(native) // divisor
    margin = -(-resample_margin(native, rate) // down) * down
    lo, hi = int(np.ceil(piece[0] * native)), int(np.ceil(piece[1] * native))
    first = up * -(-(lo + margin) // down)
    count = ((hi - margin) * up // down - first) // int(rate) * int(rate)
    return first, count


def store_tiles(source, ifo="H1"):
    """Every committed tile of ``source``'s detector, opened, in order."""
    from gravyflow.src.dataset.store import Store
    from gravyflow.src.dataset.store.tile import key_from_path

    store = Store(gf.src.dataset.acquisition.gwosc_data.user_cache_dir("store"))
    records = sorted(store.root.glob(f"v*/tone-{source.native:g}/{ifo}/*/*/*.json"),
                     key=lambda path: int(path.stem))
    return [store.open(key_from_path(store.root, path)) for path in records]


# ---------------------------------------------------------------------------
# A read is a tile view, resampled, its margin cut away
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("native", [2048.0, 4096.0, 16384.0])
def test_a_noise_read_is_a_tile_view_resampled(cache, native):
    source = ToneSource(native=native)
    [row] = acquire(source)
    first, count = interior(native, RATE, PIECE)
    served = np.asarray(row.data[0])
    assert row.start_gps_time == [first / RATE] and len(served) == count > 0
    assert PIECE[0] <= first / RATE and first / RATE + count / RATE <= PIECE[1]

    # The tiles the read committed, whole, resampled together: the read is their slice.
    tiles = store_tiles(source)
    assert [tile.key.index for tile in tiles] == [int(T0 // 64), int(T0 // 64) + 1]
    raw = np.concatenate([tile.samples for tile in tiles])
    whole, t0 = resample(raw, tiles[0].key.t0, native, RATE)
    offset = first - int(round(t0 * RATE))
    assert np.array_equal(served, whole[offset:offset + count])
    assert source.fetches == 2  # each tile fetched once, whole

    # Warm equals cold: a fresh obtainer reads the same bytes and fetches nothing.
    [again] = acquire(source)
    assert source.fetches == 2
    assert np.array_equal(np.asarray(again.data[0]), served)


def test_every_read_is_exact_at_its_edges(cache):
    """No served sample is an edge approximation: a shorter read is a slice of a longer one."""
    source = ToneSource(native=16384.0)
    [long] = acquire(source)
    [short] = acquire(source, piece=(PIECE[0] + 20.0, PIECE[1] - 30.0))
    shift = int(round((short.start_gps_time[0] - long.start_gps_time[0]) * RATE))
    short, long = np.asarray(short.data[0]), np.asarray(long.data[0])
    assert np.array_equal(short, long[shift:shift + len(short)])


# ---------------------------------------------------------------------------
# Contradictions refuse by name; exclusions plan around them
# ---------------------------------------------------------------------------

def test_an_outage_refuses_by_name_and_is_never_zero_filled(cache, quick_retries):
    outage = (T0 + 64.0, T0 + 128.0)
    source = ToneSource(outages=[outage])
    with pytest.raises(gf.StrainUnavailable) as refused:
        acquire(source)
    message = str(refused.value)
    assert "H1" in message and "planted outage" in message and "exclusions=" in message
    assert refused.value.exclusion[:2] == ("H1", outage)
    assert source.fetches == 1 + 3  # the first tile once; the second three times
    # A failed fetch writes nothing: only the delivered tile is in the store.
    assert [tile.key.index for tile in store_tiles(source)] == [int(T0 // 64)]


class Misdelivering(ToneSource):
    """A tone source whose deliveries are altered by ``alter(strain)``."""

    def __init__(self, alter, **kwargs):
        super().__init__(**kwargs)
        self.alter = alter

    def fetch(self, detector, sample_rate_hertz, interval):
        return self.alter(super().fetch(detector, sample_rate_hertz, interval))


MISDELIVERIES = {
    "a start that is not finite": lambda s: s._replace(t0=float("nan")),
    "an infinite start": lambda s: s._replace(t0=float("inf")),
    "integer samples": lambda s: s._replace(data=np.asarray(s.data).astype(np.int32)),
    "complex samples": lambda s: s._replace(data=np.asarray(s.data).astype(np.complex128)),
    "samples off the grid": lambda s: s._replace(t0=s.t0 + 0.4 / s.sample_rate_hertz),
    "another rate": lambda s: s._replace(sample_rate_hertz=2 * s.sample_rate_hertz),
}


@pytest.mark.parametrize("delivery", MISDELIVERIES)
def test_a_delivery_that_is_not_the_tile_is_refused_by_name_and_stores_nothing(cache, quick_retries, delivery):
    source = Misdelivering(MISDELIVERIES[delivery])
    with pytest.raises(gf.StrainUnavailable, match="exclusions=") as refused:
        acquire(source)
    assert "not tile" in str(refused.value), str(refused.value)
    assert store_tiles(source) == []


def test_ensure_commits_what_a_read_needs_and_refuses_what_it_would(cache, quick_retries):
    from gravyflow.src.dataset.store import read

    source = ToneSource()
    first, count = int(PIECE[0] * RATE), int(10 * RATE)
    assert read.ensure(source, gf.IFO.H1, RATE, first, count) is None
    fetched = source.fetches
    assert fetched == len(store_tiles(source)) > 0
    strain = read.read(source, gf.IFO.H1, RATE, first, count)
    assert source.fetches == fetched and np.all(np.isfinite(strain.data))  # offline after ensure
    gap = (PIECE[0] + 5.0, PIECE[0] + 5.5)
    with pytest.raises(gf.StrainUnavailable, match="NON_FINITE"):
        read.ensure(ToneSource(gaps=[gap], tile=32.0), gf.IFO.H1, RATE, first, count)


def test_a_failed_fetch_names_its_tries(cache, quick_retries):
    with pytest.raises(gf.StrainUnavailable, match=r"3 tries failed.*planted outage"):
        acquire(ToneSource(outages=[PIECE]))


def test_a_fetch_needs_gwpy_only_when_its_source_does(cache, monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "gwpy", None)  # an import of gwpy now fails
    [row] = acquire(ToneSource())
    assert np.all(np.isfinite(np.asarray(row.data[0])))


def test_non_finite_samples_inside_the_timeline_refuse_by_name(cache):
    gap = (T0 + 50.0, T0 + 50.5)
    with pytest.raises(gf.StrainUnavailable, match="NON_FINITE") as refused:
        acquire(ToneSource(gaps=[gap]))
    assert refused.value.exclusion == ("H1", gap, "NON_FINITE")


def test_the_named_exclusion_plans_around_the_contradiction(cache):
    gap = (T0 + 50.0, T0 + 50.5)
    source = ToneSource(gaps=[gap])
    reader = obtainer(source, exclusions=[("H1", gap, "a NaN burst")])
    readable = reader.get_all_segment_times(gf.IFO.H1)
    assert readable.tolist() == [[source.data[0], gap[0]], [gap[1], source.data[1]]]
    assert reader.get_all_segment_times(gf.IFO.L1).tolist() == [list(source.data)]
    # Each readable piece around the gap reads cleanly.
    for piece in ([PIECE[0], gap[0]], [gap[1], PIECE[1]]):
        [row] = list(reader.acquire(RATE, np.array([[piece]]), [gf.IFO.H1]))
        assert np.all(np.isfinite(np.asarray(row.data[0])))
    # The exclusions are part of the obtainer's specification, so of every fingerprint.
    assert reader.specification()["exclusions"] == (("H1", gap, "a NaN burst"),)


# ---------------------------------------------------------------------------
# Nothing truncated on a read error; caches of an older format are never read
# ---------------------------------------------------------------------------

def test_a_read_error_refuses_by_name_and_changes_no_byte(cache):
    source = ToneSource()
    acquire(source)
    [tile, _] = store_tiles(source)
    samples = tile.key.samples_path(cache / "store", tile.record.sha256)
    samples.chmod(0o644)
    with open(samples, "r+b") as handle:  # an outside edit, the size kept
        handle.seek(-8, 2)
        handle.write(b"\x00" * 8)
    samples.chmod(0o444)
    altered = samples.read_bytes()
    with pytest.raises(gf.src.dataset.store.TileRefused, match="sha256 differs"):
        acquire(source)
    assert samples.read_bytes() == altered  # refused, never truncated or rewritten


def test_old_segment_caches_are_named_once_and_never_read_or_written(cache, monkeypatch):
    from gravyflow.src.dataset.store import read

    monkeypatch.setattr(read, "_legacy_warned", False)
    monkeypatch.chdir(cache.parent)
    segments, older = cache / "segments", cache.parent / "generator_data"
    for directory in (segments, older):
        directory.mkdir(parents=True)
    (segments / "segment_data_0123456789abcdef.hdf5").write_bytes(b"not an HDF5 file" * 100)
    (older / "segment_data_fedcba9876543210.hdf5").write_bytes(b"older")
    before = {path: path.read_bytes() for directory in (segments, older) for path in directory.iterdir()}
    with pytest.warns(UserWarning, match=r"never read.*segment_data_0123456789abcdef\.hdf5 \(1\.6 KiB\)"
                                         r".*segment_data_fedcba9876543210\.hdf5 \(5 B\)"):
        acquire(ToneSource())
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        acquire(ToneSource())  # once per process
    assert {path: path.read_bytes() for directory in (segments, older)
            for path in directory.iterdir()} == before


@pytest.mark.parametrize("trouble", ["a file gone before it is named", "a directory that cannot be listed"])
def test_naming_old_caches_never_fails_a_read(cache, monkeypatch, trouble):
    from gravyflow.src.dataset.store import read

    monkeypatch.setattr(read, "_legacy_warned", False)
    segments = cache / "segments"
    segments.mkdir(parents=True)
    old = segments / "segment_data_0123456789abcdef.hdf5"
    old.write_bytes(b"old")
    real_stat, real_glob = Path.stat, Path.glob

    def stat(path, *args, **kwargs):
        if path.name == old.name and trouble.startswith("a file"):
            raise FileNotFoundError(path)
        return real_stat(path, *args, **kwargs)

    def glob(path, pattern, *args, **kwargs):
        if path == segments and trouble.startswith("a directory"):
            raise PermissionError(path)
        return real_glob(path, pattern, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", stat)
    monkeypatch.setattr(Path, "glob", glob)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        [row] = acquire(ToneSource())
    assert np.all(np.isfinite(np.asarray(row.data[0])))


# ---------------------------------------------------------------------------
# The memory cap
# ---------------------------------------------------------------------------

def test_the_memory_cap_holds_below_the_working_set(cache, monkeypatch):
    from gravyflow.src.dataset.store import read

    native, tile = 16384.0, 64.0
    fetch_bytes = int(native * tile) * read.FETCH_BYTES_PER_SAMPLE
    cap = fetch_bytes * 3 // 2  # two fetches never fit at once
    monkeypatch.setattr(gf.Defaults, "memory_bytes", cap)
    source = ToneSource(native=native, tile=tile)
    pieces = [(T0 + 64.0 * k + 2.0, T0 + 64.0 * k + 40.0) for k in range(4)]
    served, errors = [], []

    def work(piece):
        try:
            [row] = acquire(source, piece=piece)
            served.append(np.asarray(row.data[0]).nbytes)
        except Exception as error:  # reported below
            errors.append(error)

    tracemalloc.start()
    try:
        threads = [threading.Thread(target=work, args=(piece,)) for piece in pieces]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(120)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert errors == []
    budget = read.process_store().budget
    assert budget.capacity == cap and budget.peak <= cap and budget.in_flight == 0
    assert budget.waits > 0  # the cap was below the working set, and admission waited
    assert peak <= cap + sum(served), (peak, cap, served)


def test_a_fetch_larger_than_the_cap_is_refused_before_any_fetch(cache, monkeypatch):
    monkeypatch.setattr(gf.Defaults, "memory_bytes", 1024)
    source = ToneSource()
    with pytest.raises(gf.src.dataset.store.BudgetRefused, match="exceeds the whole memory cap"):
        acquire(source)
    assert source.fetches == 0



def test_a_read_larger_than_the_cap_is_refused_before_any_fetch(cache, monkeypatch):
    from gravyflow.src.dataset.store import read

    source = ToneSource()
    fetch_bytes = int(source.native * source.tile) * read.FETCH_BYTES_PER_SAMPLE
    monkeypatch.setattr(gf.Defaults, "memory_bytes", 2 * fetch_bytes)  # every tile's fetch fits
    piece = (T0 - 200.0, T0 + 200.0)  # a span of about 400 s: 4.7 times a tile's fetch
    with pytest.raises(gf.src.dataset.store.BudgetRefused, match="exceeds the whole memory cap"):
        acquire(source, piece=piece)
    first, count = interior(source.native, RATE, piece)
    with pytest.raises(gf.src.dataset.store.BudgetRefused, match="exceeds the whole memory cap"):
        read.read(source, "H1", RATE, first, count)
    assert source.fetches == 0

# ---------------------------------------------------------------------------
# Held strain is finite by construction (C1)
# ---------------------------------------------------------------------------

def test_a_row_of_non_finite_strain_is_refused_never_zeroed():
    with pytest.raises(ValueError, match="non-finite"):
        gf.IFOData([np.array([0.0, np.nan, 1.0], dtype=np.float32)], 2048.0, [T0])
