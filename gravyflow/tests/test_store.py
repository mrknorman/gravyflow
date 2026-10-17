"""
The store, standalone: the tile format and its identity checks, the write-once commit
shared by processes, its refusals by name, the byte accountant, and raw window views.

The cross-process tests run this file again as a child (``python test_store.py
ROLE ...``): its children meet at a barrier of files so their commits overlap.
"""

import errno
import json
import os
import signal
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

import gravyflow as gf
from gravyflow.src.dataset.store import (
    STORE_FORMAT, BudgetRefused, BudgetTimeout, ByteBudget, Fetched, Store, StoreConflict,
    StoreError, StoreWriteError, TileIncomplete, TileKey, TileMissing, TileRefused,
    TileWaitTimeout, Unheld, UnheldSamples, covering, read_window, reap, seal,
)
from gravyflow.src.dataset.store import commit as commit_module

#: A small grid for the tests: 4 s tiles at 256 Hz, 1024 samples each.
RATE, SECONDS = 256, 4


def key(index=7, detector="H1", source="FAKE-1"):
    return TileKey(source, detector, RATE, SECONDS, index)


def strain(index, offset=0.0):
    """A tile's fetched samples, float64 and distinct per index."""
    return np.random.default_rng(index).standard_normal(RATE * SECONDS) * 1e-21 + offset


def sha_of(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def files_in(directory):
    return sorted(path.name for path in Path(directory).iterdir())


# --- The format ---------------------------------------------------------------------


def test_a_key_names_its_place_on_the_absolute_grid(tmp_path):
    gwosc = TileKey("GWOSC-O3a_4KHZ_R1", "H1", 4096, 4096, 302286)
    assert gwosc.t0 == 1238163456.0 and gwosc.t0 == 302286 * 4096
    assert gwosc.length == 4096 * 4096 and gwosc.start == 302286 * 4096 * 4096
    assert gwosc.samples_path(tmp_path) == (
        tmp_path / f"v{STORE_FORMAT}" / "GWOSC-O3a_4KHZ_R1" / "H1" / "4096" / "4096" / "302286.npy")
    assert gwosc.record_path(tmp_path).name == "302286.json"
    assert covering("S", "L1", RATE, SECONDS, 9.5, 16.0) == tuple(
        TileKey("S", "L1", RATE, SECONDS, k) for k in (2, 3))
    for bad in ("", "../up", ".hidden", "a/b", 3):
        with pytest.raises(ValueError, match="one plain path component"):
            TileKey(bad, "H1", RATE, SECONDS, 0)
    for name, args in {"rate": (4096.5, 4096, 0), "tile_seconds": (4096, True, 0),
                       "index": (4096, 4096, 1.0)}.items():
        with pytest.raises(TypeError, match=f"tile's {name} must be a whole number"):
            TileKey("S", "H1", *args)
    with pytest.raises(ValueError, match="index must be at least 0"):
        TileKey("S", "H1", 4096, 4096, -1)


def test_sealing_marks_every_unheld_sample_and_poisons_it():
    samples = strain(1)
    samples[[10, 11, 500]] = [np.nan, np.inf, -np.inf]
    samples[100:130] = 7.0  # whatever a source put where its file had no data
    sealed, unheld = seal(key(), samples, not_in_file=[(100, 130), (1000, 1024)])
    assert sealed.dtype == np.float32
    assert unheld == ((10, 12, Unheld.NON_FINITE), (100, 130, Unheld.NOT_IN_FILE),
                      (500, 501, Unheld.NON_FINITE), (1000, 1024, Unheld.NOT_IN_FILE))
    held = np.ones(len(samples), dtype=bool)
    for start, stop, _ in unheld:
        held[start:stop] = False
    assert np.array_equal(sealed[held], samples[held].astype(np.float32))
    assert np.isnan(sealed[~held]).all()
    # Unheld bytes are canonical, so equal held samples seal to equal bytes.
    other = samples.copy()
    other[100:130] = -3.0
    other[10] = np.float64(np.frombuffer(b"\x01\x00\x00\x00\x00\x00\xf8\x7f", np.float64)[0])
    assert seal(key(), other, [(100, 130), (1000, 1024)])[0].tobytes() == sealed.tobytes()


def test_sealing_refuses_what_is_not_a_whole_tile():
    with pytest.raises(ValueError, match="holds 1024 samples"):
        seal(key(), np.zeros(1023))
    with pytest.raises(ValueError, match=r"\[1000, 1025\) lies outside"):
        seal(key(), np.zeros(1024), not_in_file=[(1000, 1025)])
    with pytest.raises(TypeError, match="real numbers"):
        seal(key(), np.zeros(1024, dtype=complex))
    with pytest.raises(ValueError, match="overflow float32"):
        seal(key(), np.full(1024, 1e300))


def test_a_committed_tile_reads_back_exactly(tmp_path):
    store = Store(tmp_path)
    samples = strain(7)
    samples[3] = np.nan
    commit = store.commit(key(), samples, not_in_file=[(1000, 1024)], tools={"gwpy": "3.0"})
    assert (commit.linked_samples, commit.linked_record, commit.contended) == (True, True, False)
    tile = store.open(key())
    assert np.array_equal(tile.samples[4:1000], samples[4:1000].astype(np.float32))
    assert tile.record.held == ((0, 3), (4, 1000))
    assert tile.record.fetch["tools"] == {"gwpy": "3.0"}
    # Plain .npy and JSON: any reader can open them, and they are read-only.
    samples_path, record_path = key().samples_path(tmp_path), key().record_path(tmp_path)
    assert np.array_equal(np.load(samples_path), tile.samples, equal_nan=True)
    record = json.loads(record_path.read_text())
    assert record["t0"] == key().t0 and record["sha256"] == sha_of(samples_path)
    assert record["unheld"] == [[3, 4, "NON_FINITE"], [1000, 1024, "NOT_IN_FILE"]]
    # The record names the committed file as it stays, so an open skips the hash.
    final = samples_path.stat()
    assert record["samples_identity"] == [final.st_dev, final.st_ino, final.st_ctime_ns]
    for path in (samples_path, record_path):
        assert stat.S_IMODE(path.stat().st_mode) == 0o444
    assert files_in(samples_path.parent) == ["7.json", "7.npy"]
    assert store.verify() == []


def test_keys_on_two_grids_are_two_tiles(tmp_path):
    # T is part of the key, so it is part of the path: a key on another grid is
    # another tile, never a collision reported as a changed source.
    store = Store(tmp_path)
    coarse, fine = TileKey("S", "H1", RATE, 16, 2), TileKey("S", "H1", RATE, 8, 2)
    assert coarse.samples_path(tmp_path) != fine.samples_path(tmp_path)
    for grid in (fine, coarse):
        store.commit(grid, np.arange(grid.length, dtype=np.float32))
    for grid in (fine, coarse):
        tile = store.open(grid, verify=True)
        assert tile.key == grid and np.array_equal(tile.samples, np.arange(grid.length))
    assert store.verify() == []


def test_window_views_are_exact_and_never_cross_unheld_samples(tmp_path):
    store = Store(tmp_path)
    first, second = strain(7), strain(8)
    store.commit(key(7), first)
    store.commit(key(8), second, not_in_file=[(512, 1024)])
    tiles = [store.open(key(7)), store.open(key(8))]
    start = key(7).start
    view = tiles[0].view(start + 100, start + 300)
    assert np.shares_memory(view, tiles[0].samples) and not view.flags.writeable
    assert np.array_equal(view, first[100:300].astype(np.float32))
    # Sample n of a tile is at t0 + n / rate: the window's place is its index.
    assert (start + 100) / RATE == key(7).t0 + 100 / RATE
    across = read_window(tiles, start + 1000, start + 1024 + 500)
    assert np.array_equal(across, np.concatenate([first[1000:], second[:500]]).astype(np.float32))
    with pytest.raises(UnheldSamples, match=r"NOT_IN_FILE") as refusal:
        read_window(tiles, start + 1000, start + 1024 + 513)
    assert f"GPS [{key(8).t0 + 2.0}, {key(8).t0 + 4.0})" in str(refusal.value)
    with pytest.raises(ValueError, match="not all inside"):
        tiles[0].view(start - 1, start + 10)
    with pytest.raises(ValueError, match=r"needs tiles \[7\]"):
        read_window(tiles, start, start + 10)


# --- Identity -----------------------------------------------------------------------


def test_a_renamed_or_moved_tile_is_refused_by_name(tmp_path):
    store = Store(tmp_path)
    store.commit(key(5), strain(5))
    target = key(6)
    target.directory(tmp_path).mkdir(parents=True, exist_ok=True)
    for suffix in ("npy", "json"):
        os.rename(key(5).directory(tmp_path) / f"5.{suffix}", target.directory(tmp_path) / f"6.{suffix}")
    with pytest.raises(TileRefused, match=r"states index 5 \(the path says 6\)"):
        store.open(target)
    moved = key(6, detector="L1")
    moved.directory(tmp_path).mkdir(parents=True)
    for suffix in ("npy", "json"):
        os.rename(target.directory(tmp_path) / f"6.{suffix}", moved.directory(tmp_path) / f"6.{suffix}")
    with pytest.raises(TileRefused, match=r"detector 'H1' \(the path says 'L1'\), index 5"):
        store.open(moved)
    assert len(store.verify()) == 1


def test_a_swapped_or_altered_samples_file_is_refused_by_name(tmp_path):
    store = Store(tmp_path)
    store.commit(key(5), strain(5))
    store.commit(key(6), strain(6))
    directory = key(5).directory(tmp_path)
    os.rename(directory / "5.npy", directory / "swap")
    os.rename(directory / "6.npy", directory / "5.npy")
    os.rename(directory / "swap", directory / "6.npy")
    for index in (5, 6):
        with pytest.raises(TileRefused, match="sha256 differs from its record's"):
            store.open(key(index))
    assert [line.split(":")[0] for line in store.verify()] == [
        str(key(5).record_path(tmp_path)), str(key(6).record_path(tmp_path))]
    # Altered in place, the very file committed: caught by a full check.
    store.commit(key(9), strain(9))
    path = key(9).samples_path(tmp_path)
    path.chmod(0o644)
    with open(path, "r+b") as handle:
        handle.seek(-4, os.SEEK_END)
        handle.write(np.float32(1.0).tobytes())
    with pytest.raises(TileRefused, match="sha256 differs"):
        store.open(key(9), verify=True)
    assert len(store.verify()) == 3


def test_a_file_that_reuses_the_committed_inode_is_hashed_never_trusted(tmp_path):
    store = Store(tmp_path)
    store.commit(key(), strain(7))
    samples_path, record_path = key().samples_path(tmp_path), key().record_path(tmp_path)
    samples_path.unlink()
    np.save(samples_path, seal(key(), strain(7, offset=1e-22))[0])  # another file in its place
    # A filesystem that reuses the freed inode gives the new file the committed
    # inode number, but never its status-change time: forge that reuse.
    fields = json.loads(record_path.read_text())
    replaced = samples_path.stat()
    record_path.chmod(0o644)
    for forged in ([replaced.st_dev, replaced.st_ino, fields["samples_identity"][2]],
                   [replaced.st_dev + 1, replaced.st_ino, replaced.st_ctime_ns]):
        record_path.write_text(json.dumps({**fields, "samples_identity": forged}))
        with pytest.raises(TileRefused, match="sha256 differs from its record's"):
            store.open(key())
    # The whole identity is what skips the hash on open; a full check still refuses.
    identity = [replaced.st_dev, replaced.st_ino, replaced.st_ctime_ns]
    record_path.write_text(json.dumps({**fields, "samples_identity": identity}))
    assert store.open(key()).key == key()
    with pytest.raises(TileRefused, match="sha256 differs"):
        store.open(key(), verify=True)


def test_a_malformed_record_or_samples_file_is_refused(tmp_path):
    store = Store(tmp_path)
    store.commit(key(), strain(7))
    record_path = key().record_path(tmp_path)
    good = json.loads(record_path.read_text())
    record_path.chmod(0o644)
    for fields, cause in [({**good, "store_format": STORE_FORMAT + 1}, "store_format"),
                          ({**good, "t0": good["t0"] + 1}, "t0"),
                          ({k: v for k, v in good.items() if k != "sha256"}, "malformed"),
                          ({**good, "unheld": [[5, 9, "NON_FINITE"], [8, 12, "NON_FINITE"]]}, "unsorted")]:
        record_path.write_text(json.dumps(fields))
        with pytest.raises(TileRefused, match=cause):
            store.open(key())
    record_path.write_text(json.dumps(good))
    samples_path = key().samples_path(tmp_path)
    samples_path.chmod(0o644)
    np.save(samples_path, np.zeros(1024, dtype=np.float64))
    with pytest.raises(TileRefused, match="float64"):
        store.open(key())


# --- The commit ---------------------------------------------------------------------


def test_differing_content_is_a_conflict_and_nothing_is_overwritten(tmp_path):
    store = Store(tmp_path)
    store.commit(key(), strain(7))
    samples_path = key().samples_path(tmp_path)
    before = (sha_of(samples_path), samples_path.stat().st_mtime_ns)
    with pytest.raises(StoreConflict, match="the source changed under the store") as conflict:
        store.commit(key(), strain(7, offset=1e-22))
    assert "Store.forget the tile" in str(conflict.value) and callable(Store.forget)
    assert (sha_of(samples_path), samples_path.stat().st_mtime_ns) == before
    # The record's content excludes the commit metadata: another fetch time and tool
    # agree; another hash does not.
    record_path = key().record_path(tmp_path)
    record = json.loads(record_path.read_text())
    record_path.unlink()
    record_path.write_text(json.dumps({**record, "fetch": {"tools": {"x": "1"}}, "samples_identity": [1, 1, 1]}))
    assert store.commit(key(), strain(7)).contended
    record_path.unlink()
    record_path.write_text(json.dumps({**record, "unheld": [[0, 1, "NON_FINITE"]]}))
    with pytest.raises(StoreConflict, match="committed record .* differs"):
        store.commit(key(), strain(7))
    assert files_in(samples_path.parent) == ["7.json", "7.npy"]


def test_a_record_without_samples_is_refused_and_repaired_by_the_next_writer(tmp_path):
    store = Store(tmp_path)
    store.commit(key(), strain(7))
    key().samples_path(tmp_path).unlink()  # a forget that raced a commit
    with pytest.raises(TileIncomplete, match="record but no samples"):
        store.open(key())
    commit = store.commit(key(), strain(7))
    assert (commit.linked_samples, commit.linked_record) == (True, False)
    # The record names the samples it was written with; the new file hashes equal.
    assert np.array_equal(store.open(key()).samples, strain(7).astype(np.float32))


def test_a_differing_writer_links_nothing_under_a_committed_record(tmp_path):
    store = Store(tmp_path)
    store.commit(key(), strain(7))
    samples_path = key().samples_path(tmp_path)
    samples_path.unlink()  # a forget that raced a commit, or an outside rm
    with pytest.raises(StoreConflict, match="the source changed under the store"):
        store.commit(key(), strain(7, offset=1e-22))
    assert files_in(samples_path.parent) == ["7.json"]  # nothing of its own left behind
    with pytest.raises(TileIncomplete):
        store.open(key())
    store.commit(key(), strain(7))  # the committed record's tile is still recoverable
    assert np.array_equal(store.open(key(), verify=True).samples, strain(7).astype(np.float32))


def test_forget_removes_the_record_before_the_samples(tmp_path, monkeypatch):
    store = Store(tmp_path)
    store.commit(key(), strain(7))
    held = store.open(key())
    removed = []
    unlink = Path.unlink
    monkeypatch.setattr(Path, "unlink", lambda path, **kw: (removed.append(path.name), unlink(path, **kw)))
    assert store.forget([key(), key(8)]) == (key(),)
    assert removed == ["7.json", "7.npy", "8.json", "8.npy"]
    with pytest.raises(TileMissing):
        store.open(key())
    assert np.array_equal(held.view(key().start, key().start + 4), strain(7)[:4].astype(np.float32))


def test_the_reaper_removes_only_old_temporaries(tmp_path):
    store = Store(tmp_path)
    directory = key().directory(tmp_path)
    directory.mkdir(parents=True)
    old, new, other = (directory / name for name in (
        f"{commit_module.TMP_PREFIX}1-old", f"{commit_module.TMP_PREFIX}2-new", "unrelated"))
    for path in (old, new, other):
        path.write_bytes(b"x")
    stale = time.time() - commit_module.REAP_AGE_SECONDS - 60
    os.utime(old, (stale, stale))
    os.utime(other, (stale, stale))
    store.commit(key(), strain(7))
    assert files_in(directory) == sorted(["7.json", "7.npy", new.name, "unrelated"])


#: The errno of every failure a write can meet: a full disk, a quota, a permission and
#: the lock errors network filesystems raise.
WRITE_ERRORS = ("ENOSPC", "EDQUOT", "EACCES", "EPERM", "ENOLCK", "EAGAIN")


def _error(code):
    return OSError(getattr(errno, code), os.strerror(getattr(errno, code)))


def _fail(code):
    def failing(*args, **kwargs):
        raise _error(code)
    return failing


def _half_then_fail(code):
    def write_array(handle, array, **kwargs):
        handle.write(array.tobytes()[: array.nbytes // 2])
        raise _error(code)
    return write_array


@pytest.mark.parametrize("code", WRITE_ERRORS)
@pytest.mark.parametrize("step", ["create", "write", "fsync", "link-samples", "link-record"])
def test_write_errors_raise_by_name_and_never_truncate(tmp_path, monkeypatch, code, step):
    store = Store(tmp_path)
    store.commit(key(6), strain(6))  # a committed neighbour that must stay untouched
    neighbour = key(6).samples_path(tmp_path)
    before = (sha_of(neighbour), neighbour.stat().st_mtime_ns)
    links = []

    def link_second(source, target):
        links.append(target)
        if len(links) == 2:
            raise _error(code)
        return real_link(source, target)

    real_link = os.link
    patches = {
        "create": (os, "open", _fail(code)),
        "write": (np.lib.format, "write_array", _half_then_fail(code)),
        "fsync": (os, "fsync", _fail(code)),
        "link-samples": (os, "link", _fail(code)),
        "link-record": (os, "link", link_second),
    }
    target, name, replacement = patches[step]
    with monkeypatch.context() as patch:
        patch.setattr(target, name, replacement)
        with pytest.raises(StoreWriteError, match=f"{code} ") as failure:
            store.commit(key(7), strain(7))
    assert failure.value.errno == getattr(errno, code)
    assert str(key(7).directory(tmp_path)) in str(failure.value)
    with pytest.raises(TileMissing):
        store.open(key(7))
    assert (sha_of(neighbour), neighbour.stat().st_mtime_ns) == before
    leftover = [n for n in files_in(key(7).directory(tmp_path)) if n.startswith(commit_module.TMP_PREFIX)]
    assert leftover == []
    store.commit(key(7), strain(7))  # the next writer commits it whole
    assert np.array_equal(store.open(key(7)).samples, strain(7).astype(np.float32))


def test_a_read_only_directory_is_a_write_error_by_name(tmp_path):
    store = Store(tmp_path)
    store.commit(key(6), strain(6))
    directory = key(6).directory(tmp_path)
    directory.chmod(0o555)
    try:
        with pytest.raises(StoreWriteError, match="EACCES") as failure:
            store.commit(key(7), strain(7))
        with pytest.raises(StoreWriteError, match="EACCES"):
            store.forget([key(6)])
    finally:
        directory.chmod(0o755)
    assert failure.value.errno == errno.EACCES
    assert files_in(directory) == ["6.json", "6.npy"]
    assert store.verify() == []


# --- The accountant -----------------------------------------------------------------


def test_the_byte_budget_admits_exactly_to_its_cap():
    budget = ByteBudget(100)
    with budget.admit(60, timeout=1), budget.admit(40, timeout=1):
        assert budget.in_flight == 100
        with pytest.raises(BudgetTimeout, match="1 bytes waited 0.05 s while 100 of 100"):
            with budget.admit(1, timeout=0.05):
                pass
        with budget.admit(0, timeout=0):
            pass
    assert (budget.in_flight, budget.peak, budget.waits) == (0, 100, 1)
    with budget.admit(100, timeout=0):
        pass
    with pytest.raises(BudgetRefused, match="101 bytes exceeds the whole memory cap of 100"):
        budget.check(101)
    with pytest.raises(BudgetRefused):
        with budget.admit(101, timeout=10):
            pass
    for bad in (0, -1, 1.5, True):
        with pytest.raises(ValueError):
            ByteBudget(bad)
    for bad in (-1, 1.5, True, "1"):
        with pytest.raises(ValueError, match="whole number"):
            budget.check(bad)
    # Byte counts computed with numpy are whole numbers too, as they are for a TileKey.
    counted = ByteBudget(np.int64(100))
    with counted.admit(np.prod(np.array([6, 10], dtype=np.int64)), timeout=0):
        assert counted.in_flight == 60 and type(counted.in_flight) is int
    assert counted.capacity == 100 and type(counted.capacity) is int


def test_the_byte_budget_never_exceeds_its_cap_under_threads():
    budget, lock = ByteBudget(1000), threading.Lock()
    held, highest = [0], [0]

    def worker(seed):
        rng = np.random.default_rng(seed)
        for _ in range(50):
            nbytes = int(rng.integers(1, 1001))
            with budget.admit(nbytes, timeout=30):
                with lock:
                    held[0] += nbytes
                    highest[0] = max(highest[0], held[0])
                time.sleep(0.0005)
                with lock:
                    held[0] -= nbytes

    threads = [threading.Thread(target=worker, args=(seed,)) for seed in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(60)
    assert highest[0] <= 1000 and budget.peak <= 1000
    assert budget.waits > 0 and budget.in_flight == 0


# --- One fetch per (stream, tile) in a process --------------------------------------


def test_ensure_fetches_a_tile_once_per_stream(tmp_path):
    store, calls, started = Store(tmp_path, budget=ByteBudget(1 << 20)), [], threading.Event()

    def fetch():
        calls.append(threading.current_thread().name)
        started.set()
        time.sleep(0.2)
        assert store.budget.in_flight == 4096
        return Fetched(strain(7))

    results = []
    threads = [threading.Thread(target=lambda: results.append(store.ensure(key(), fetch, nbytes=4096)))
               for _ in range(8)]
    threads[0].start()
    started.wait(10)
    for thread in threads[1:]:
        thread.start()
    for thread in threads:
        thread.join(30)
    assert len(calls) == 1 and len(results) == 8
    assert all(np.array_equal(tile.samples, results[0].samples) for tile in results)
    assert store.budget.in_flight == 0 and store._flights == {}

    # Two streams never share an in-flight fetch: each fetches, one commits.
    gate, calls = threading.Barrier(2, timeout=10), []

    def both():
        calls.append(1)
        gate.wait()
        return Fetched(strain(8))

    threads = [threading.Thread(target=store.ensure, args=(key(8), both), kwargs={"stream": s})
               for s in ("train", "validate")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(30)
    assert len(calls) == 2 and store.open(key(8))


def test_a_failed_fetch_leaves_the_table_and_reaches_every_waiter(tmp_path):
    store, release, errors = Store(tmp_path), threading.Event(), []

    def failing():
        release.wait(10)
        raise ConnectionError("source down")

    def wait_for_it():
        try:
            store.ensure(key(), failing)
        except ConnectionError as error:
            errors.append(error)

    leader = threading.Thread(target=wait_for_it)
    leader.start()
    deadline = time.monotonic() + 10
    while not store._flights:
        assert time.monotonic() < deadline, "the leader never started its fetch"
        time.sleep(0.001)
    waiter = threading.Thread(target=wait_for_it)
    waiter.start()
    with pytest.raises(TileWaitTimeout, match="Waited 0.05 s"):
        store.ensure(key(), failing, timeout=0.05)
    release.set()
    leader.join(10)
    waiter.join(10)
    assert len(errors) == 2 and store._flights == {}
    with pytest.raises(TileMissing):
        store.open(key())
    assert store.ensure(key(), lambda: Fetched(strain(7))).key == key()
    with pytest.raises(BudgetRefused):
        Store(tmp_path / "small", budget=ByteBudget(10)).ensure(
            key(), lambda: Fetched(strain(7)), nbytes=11, timeout=5)


# --- Processes ------------------------------------------------------------------------

#: The tiles each cross-process run races on.
RACE_TILES = 20


def _child(role, *args, timeout=120):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(Path(gf.__file__).parents[1]), env.get("PYTHONPATH", "")])
    return subprocess.Popen([sys.executable, __file__, role, *map(str, args)], env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def _report(process, timeout=120):
    out, err = process.communicate(timeout=timeout)
    assert process.returncode == 0, err
    return json.loads(out.strip().splitlines()[-1])


def test_two_processes_commit_each_tile_once_and_a_racing_reader_never_errors(tmp_path):
    reader = _child("reader", tmp_path)
    writers = [_child("writer", tmp_path, name, 0.0, "reader") for name in ("a", "b")]
    reports = [_report(writer) for writer in writers]
    read = _report(reader)
    for tile in range(RACE_TILES):
        linked = [report["linked_samples"][tile] for report in reports]
        assert sorted(linked) == [False, True], (tile, reports)
        assert [report["conflicts"][tile] for report in reports] == [None, None]
    assert files_in(key().directory(tmp_path / "store")) == sorted(
        f"{k}.{suffix}" for k in range(RACE_TILES) for suffix in ("json", "npy"))
    contended = sum(report["contended"] for report in reports)
    assert contended >= RACE_TILES  # the loser of every tile saw the winner's link
    print(f"two writers over {RACE_TILES} tiles: contended {contended}; reader {read}")
    assert read["errors"] == [] and read["torn"] == 0 and read["served"] == RACE_TILES
    assert read["missing"] > 0  # it read while the tiles were being committed
    assert Store(tmp_path / "store").verify() == []


def test_two_processes_with_different_content_conflict_and_keep_the_first(tmp_path):
    writers = [_child("writer", tmp_path, name, offset) for name, offset in (("a", 0.0), ("b", 1e-22))]
    reports = [_report(writer) for writer in writers]
    store = Store(tmp_path / "store")
    for tile in range(RACE_TILES):
        winner = [r for r in reports if r["linked_samples"][tile]]
        loser = [r for r in reports if not r["linked_samples"][tile]]
        assert len(winner) == 1 and loser[0]["conflicts"][tile] == "StoreConflict"
        expected = strain(tile, offset=winner[0]["offset"]).astype(np.float32)
        assert np.array_equal(store.open(key(tile)).samples, expected)


def test_a_writer_killed_between_its_links_leaves_a_recoverable_state(tmp_path):
    killed = _child("killed", tmp_path)
    killed.communicate(timeout=120)
    assert killed.returncode == -signal.SIGKILL
    store = Store(tmp_path / "store")
    directory = key().directory(store.root)
    names = files_in(directory)
    assert "7.npy" in names and "7.json" not in names
    with pytest.raises(TileMissing):
        store.open(key())
    commit = store.commit(key(), strain(7))
    assert (commit.linked_samples, commit.linked_record) == (False, True)
    assert np.array_equal(store.open(key()).samples, strain(7).astype(np.float32))
    # The dead writer's temporary is reaped by name once old enough.
    assert len([n for n in files_in(directory) if n.startswith(commit_module.TMP_PREFIX)]) == 1
    assert reap(directory, max_age=0) == 1 and files_in(directory) == ["7.json", "7.npy"]
    assert store.verify() == []


def _barrier(root, name, tile, peers, timeout=60):
    """Wait, bounded, until every peer has reached ``tile``."""
    (root / f"ready-{tile}-{name}").touch()
    deadline = time.monotonic() + timeout
    while not all((root / f"ready-{tile}-{peer}").exists() for peer in peers):
        if time.monotonic() > deadline:
            raise TimeoutError(f"Gave up after {timeout} s waiting for {peers} at tile {tile}.")
        time.sleep(0.0002)


def _writer(root, name, offset, *watchers):
    """Commit every race tile, meeting the other writer before each (and ``watchers`` first)."""
    root, offset = Path(root), float(offset)
    _barrier(root, name, "start", ("a", "b", *watchers))
    store = Store(root / "store")
    report = {"offset": offset, "linked_samples": [], "contended": 0, "conflicts": []}
    for tile in range(RACE_TILES):
        samples = strain(tile, offset)
        _barrier(root, name, tile, ("a", "b"))
        try:
            commit = store.commit(key(tile), samples)
        except StoreConflict:
            report["linked_samples"].append(False)
            report["conflicts"].append("StoreConflict")
            continue
        report["linked_samples"].append(commit.linked_samples)
        report["contended"] += commit.contended
        report["conflicts"].append(None)
    return report


def _reader(root, timeout=60):
    store = Store(Path(root) / "store")
    (Path(root) / "ready-start-reader").touch()
    report = {"reads": 0, "missing": 0, "torn": 0, "served": 0, "errors": []}
    pending, deadline = set(range(RACE_TILES)), time.monotonic() + timeout
    expected = {tile: strain(tile).astype(np.float32) for tile in pending}
    while pending and time.monotonic() < deadline:
        for tile in sorted(pending):
            report["reads"] += 1
            try:
                window = store.open(key(tile)).view(key(tile).start, key(tile).start + key(tile).length)
            except TileMissing:
                report["missing"] += 1
                continue
            except Exception as error:  # every other outcome is the defect this counts
                report["errors"].append(repr(error))
                continue
            report["torn"] += not np.array_equal(window, expected[tile])
            report["served"] += 1
            pending.discard(tile)
    return report


def _killed(root):
    """Commit one tile and die by SIGKILL right after linking its samples."""
    real_link = os.link

    def link_then_die(source, target):
        real_link(source, target)
        if str(target).endswith(".npy"):
            os.kill(os.getpid(), signal.SIGKILL)

    os.link = link_then_die
    Store(Path(root) / "store").commit(key(), strain(7))


if __name__ == "__main__":
    role, arguments = sys.argv[1], sys.argv[2:]
    result = {"writer": _writer, "reader": _reader, "killed": _killed}[role](*arguments)
    print(json.dumps(result))
