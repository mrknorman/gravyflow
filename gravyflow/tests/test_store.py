"""
The store, standalone: the tile format and its identity checks, the write-once commit
shared by processes, its refusals by name, the byte accountant, and raw window views.

The cross-process tests run this file again as a child (``python test_store.py
ROLE ...``): its children meet at a barrier of files so their commits overlap. One
child instead interleaves two writers and a forget one filesystem operation at a time,
in every order (Interleavings, below).
"""

import errno
import json
import os
import re
import shutil
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
    STORE_FORMAT, BudgetRefused, BudgetTimeout, ByteBudget, Fetched, ForgetRaced, Store,
    StoreConflict, StoreError, StoreWriteError, TileIncomplete, TileKey, TileMissing,
    TileRefused, TileWaitTimeout, Unheld, UnheldSamples, covering, read_window, reap, seal,
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


def samples_file(root, tile_key):
    """The samples file a committed tile's record names."""
    return tile_key.samples_path(root, json.loads(tile_key.record_path(root).read_text())["sha256"])


def tile_files(root, tile_key):
    """The names of a whole tile's two files."""
    return sorted([tile_key.record_path(root).name, samples_file(root, tile_key).name])


# --- The format ---------------------------------------------------------------------


def test_a_key_names_its_place_on_the_absolute_grid(tmp_path):
    gwosc = TileKey("GWOSC-O3a_4KHZ_R1", "H1", 4096, 4096, 302286)
    assert gwosc.t0 == 1238163456.0 and gwosc.t0 == 302286 * 4096
    assert gwosc.length == 4096 * 4096 and gwosc.start == 302286 * 4096 * 4096
    digest = "0123456789abcdef" * 4
    assert gwosc.samples_path(tmp_path, digest) == (
        tmp_path / f"v{STORE_FORMAT}" / "GWOSC-O3a_4KHZ_R1" / "H1" / "4096" / "4096" / f"302286.{digest}.npy")
    assert gwosc.record_path(tmp_path).name == "302286.json"
    # Samples are named by a sha256 alone, so no name read from a record leaves the tile.
    for bad in ("../" + digest[3:], digest.upper(), digest[:-1], None):
        with pytest.raises(ValueError, match="named by a sha256"):
            gwosc.samples_path(tmp_path, bad)
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
    record_path = key().record_path(tmp_path)
    record = json.loads(record_path.read_text())
    samples_path = key().samples_path(tmp_path, record["sha256"])
    assert np.array_equal(np.load(samples_path), tile.samples, equal_nan=True)
    # The samples are named by their own sha256, which the record states.
    assert record["t0"] == key().t0 and record["sha256"] == sha_of(samples_path)
    assert record["unheld"] == [[3, 4, "NON_FINITE"], [1000, 1024, "NOT_IN_FILE"]]
    # The record names the committed file as it stays, so an open skips the hash.
    final = samples_path.stat()
    assert record["samples_identity"] == [final.st_dev, final.st_ino, final.st_ctime_ns]
    for path in (samples_path, record_path):
        assert stat.S_IMODE(path.stat().st_mode) == 0o444
    assert files_in(samples_path.parent) == sorted(["7.json", f"7.{record['sha256']}.npy"])
    assert store.verify() == []


def test_keys_on_two_grids_are_two_tiles(tmp_path):
    # T is part of the key, so it is part of the path: a key on another grid is
    # another tile, never a collision reported as a changed source.
    store = Store(tmp_path)
    coarse, fine = TileKey("S", "H1", RATE, 16, 2), TileKey("S", "H1", RATE, 8, 2)
    assert coarse.record_path(tmp_path) != fine.record_path(tmp_path)
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


def test_a_window_never_joins_tiles_of_two_grids(tmp_path):
    # Tile 1 of the 4 s grid and tile 2 of the 3 s grid have the indices a window
    # across the first one's end needs, but they are not consecutive tiles.
    store = Store(tmp_path)
    fine, coarse = TileKey("S", "H1", RATE, SECONDS, 1), TileKey("S", "H1", RATE, 3, 2)
    for grid in (fine, coarse):
        store.commit(grid, np.arange(grid.length, dtype=np.float32))
    tiles = [store.open(fine), store.open(coarse)]
    with pytest.raises(ValueError, match=r"needs tiles \[1, 2\] of one detector and rate"):
        read_window(tiles, fine.start + fine.length - 24, fine.start + fine.length + 76)


# --- Identity -----------------------------------------------------------------------


def _move_tile(root, source, target):
    """Rename a committed tile's two files to ``target``'s names, as an outside tool might."""
    target.directory(root).mkdir(parents=True, exist_ok=True)
    digest = json.loads(source.record_path(root).read_text())["sha256"]
    os.rename(source.samples_path(root, digest), target.samples_path(root, digest))
    os.rename(source.record_path(root), target.record_path(root))


def test_a_renamed_or_moved_tile_is_refused_by_name(tmp_path):
    store = Store(tmp_path)
    store.commit(key(5), strain(5))
    target = key(6)
    _move_tile(tmp_path, key(5), target)
    with pytest.raises(TileRefused, match=r"states index 5 \(the path says 6\)"):
        store.open(target)
    moved = key(6, detector="L1")
    _move_tile(tmp_path, target, moved)
    with pytest.raises(TileRefused, match=r"detector 'H1' \(the path says 'L1'\), index 5"):
        store.open(moved)
    assert len(store.verify()) == 1


def test_a_swapped_or_altered_samples_file_is_refused_by_name(tmp_path):
    store = Store(tmp_path)
    store.commit(key(5), strain(5))
    store.commit(key(6), strain(6))
    five, six = samples_file(tmp_path, key(5)), samples_file(tmp_path, key(6))
    swap = five.with_name("swap")
    os.rename(five, swap)
    os.rename(six, five)
    os.rename(swap, six)
    for index in (5, 6):
        with pytest.raises(TileRefused, match="sha256 differs from its record's"):
            store.open(key(index))
    assert [line.split(":")[0] for line in store.verify()] == [
        str(key(5).record_path(tmp_path)), str(key(6).record_path(tmp_path))]
    # Altered in place, the very file committed: caught by a full check.
    store.commit(key(9), strain(9))
    path = samples_file(tmp_path, key(9))
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
    samples_path, record_path = samples_file(tmp_path, key()), key().record_path(tmp_path)
    samples_path.unlink()
    with open(samples_path, "wb") as handle:  # another file in its place, from outside
        np.save(handle, seal(key(), strain(7, offset=1e-22))[0])
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
                          ({**good, "sha256": "../../" + good["sha256"][6:]}, "malformed"),
                          ({**good, "unheld": [[5, 9, "NON_FINITE"], [8, 12, "NON_FINITE"]]}, "unsorted")]:
        record_path.write_text(json.dumps(fields))
        with pytest.raises(TileRefused, match=cause):
            store.open(key())
    record_path.write_text(json.dumps(good))
    samples_path = samples_file(tmp_path, key())
    samples_path.chmod(0o644)
    with open(samples_path, "wb") as handle:
        np.save(handle, np.zeros(1024, dtype=np.float64))
    with pytest.raises(TileRefused, match="float64"):
        store.open(key())


# --- The commit ---------------------------------------------------------------------


def test_differing_content_is_a_conflict_and_nothing_is_overwritten(tmp_path):
    store = Store(tmp_path)
    store.commit(key(), strain(7))
    samples_path = samples_file(tmp_path, key())
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
    assert files_in(samples_path.parent) == sorted(["7.json", samples_path.name])
    # Samples altered under their name are found by the next writer of those bytes.
    record_path.unlink()
    samples_path.chmod(0o644)
    with open(samples_path, "r+b") as handle:
        handle.seek(-4, os.SEEK_END)
        handle.write(np.float32(1.0).tobytes())
    altered = sha_of(samples_path)
    with pytest.raises(StoreConflict, match=f"holds bytes with sha256 {altered}, not the sha256 its name"):
        store.commit(key(), strain(7))
    assert files_in(samples_path.parent) == [samples_path.name] and sha_of(samples_path) == altered


def test_a_record_without_samples_is_refused_and_repaired_by_the_next_writer(tmp_path):
    store = Store(tmp_path)
    store.commit(key(), strain(7))
    samples_file(tmp_path, key()).unlink()  # a forget that raced a commit
    with pytest.raises(TileIncomplete, match="record but no samples"):
        store.open(key())
    commit = store.commit(key(), strain(7))
    assert (commit.linked_samples, commit.linked_record) == (True, False)
    # The record names the samples it was written with; the new file hashes equal.
    assert np.array_equal(store.open(key()).samples, strain(7).astype(np.float32))


def test_a_differing_writer_links_nothing_under_a_committed_record(tmp_path):
    store = Store(tmp_path)
    store.commit(key(), strain(7))
    samples_path = samples_file(tmp_path, key())
    samples_path.unlink()  # a forget that raced a commit, or an outside rm
    with pytest.raises(StoreConflict, match="the source changed under the store"):
        store.commit(key(), strain(7, offset=1e-22))
    assert files_in(samples_path.parent) == ["7.json"]  # nothing of its own left behind
    with pytest.raises(TileIncomplete):
        store.open(key())
    store.commit(key(), strain(7))  # the committed record's tile is still recoverable
    assert np.array_equal(store.open(key(), verify=True).samples, strain(7).astype(np.float32))


def test_forget_removes_the_record_then_every_samples_file_of_its_tile(tmp_path, monkeypatch):
    store = Store(tmp_path)
    store.commit(key(), strain(7))
    lost = samples_file(tmp_path, key())
    key().record_path(tmp_path).unlink()  # a forget's first step, then a writer of other bytes
    store.commit(key(), strain(7, offset=1e-22))
    store.commit(key(77), strain(77))  # a neighbour whose name starts like the tile's
    held = store.open(key())
    removed = []
    unlink = Path.unlink
    monkeypatch.setattr(Path, "unlink", lambda path, **kw: (removed.append(path.name), unlink(path, **kw)))
    committed = key().samples_path(tmp_path, held.record.sha256)
    assert store.forget([key(), key(8)]) == (key(),)
    assert removed[0] == "7.json" and sorted(removed[1:3]) == sorted([lost.name, committed.name])
    assert removed[3:] == ["8.json"]  # an absent tile has nothing to remove
    assert files_in(lost.parent) == tile_files(tmp_path, key(77))
    with pytest.raises(TileMissing):
        store.open(key())
    expected = strain(7, offset=1e-22)[:4].astype(np.float32)
    assert np.array_equal(held.view(key().start, key().start + 4), expected)
    assert store.open(key(77), verify=True).key == key(77)


@pytest.mark.parametrize("checked", [".npy", ".json"])
def test_a_commit_that_loses_a_race_with_forget_says_so_by_name(tmp_path, monkeypatch, checked):
    # A forget lands between a link that found the file taken and the check of that file.
    store = Store(tmp_path)
    store.commit(key(), strain(7))
    real_link = commit_module._link

    def forget_after_taken(source, target):
        linked = real_link(source, target)
        if not linked and target.name.endswith(checked):
            store.forget([key()])
        return linked

    monkeypatch.setattr(commit_module, "_link", forget_after_taken)
    with pytest.raises(ForgetRaced, match="lost a race with Store.forget"):
        store.commit(key(), strain(7))
    assert files_in(key().directory(tmp_path)) == []  # absent, and no temporary left
    monkeypatch.undo()
    store.commit(key(), strain(7))
    assert store.verify() == []


def test_ensure_fetches_again_when_a_forget_wins_and_gives_up_by_name(tmp_path, monkeypatch):
    store, fetches = Store(tmp_path), []
    real_commit = store.commit
    races = iter(["checked file forgotten", "tile forgotten after the commit"])

    def racing_commit(*args, **kwargs):
        race = next(races, None)
        if race == "checked file forgotten":
            raise ForgetRaced(race)
        done = real_commit(*args, **kwargs)
        if race is not None:
            store.forget([key()])
        return done

    def fetch():
        fetches.append(1)
        return Fetched(strain(7))

    monkeypatch.setattr(store, "commit", racing_commit)
    tile = store.ensure(key(), fetch)
    assert len(fetches) == commit_module.RACE_ATTEMPTS == 3  # each lost race fetches again
    assert np.array_equal(tile.samples, strain(7).astype(np.float32))
    # A forget that wins every race: ensure gives up, by name, after its bound.
    store.forget([key()])
    fetches.clear()
    races = iter(["tile forgotten after the commit"] * 10)
    with pytest.raises(ForgetRaced, match="forgotten while this caller committed it, 3 times"):
        store.ensure(key(), fetch)
    assert len(fetches) == 3 and store._flights == {}


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
    assert files_in(directory) == sorted([*tile_files(tmp_path, key()), new.name, "unrelated"])


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
    neighbour = samples_file(tmp_path, key(6))
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
    assert files_in(directory) == tile_files(tmp_path, key(6))
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
        name for k in range(RACE_TILES) for name in tile_files(tmp_path / "store", key(k)))
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
    assert [re.fullmatch(r"7\.[0-9a-f]{64}\.npy", n) is not None for n in names if n[0] != "."] == [True]
    with pytest.raises(TileMissing):
        store.open(key())
    commit = store.commit(key(), strain(7))
    assert (commit.linked_samples, commit.linked_record) == (False, True)
    assert np.array_equal(store.open(key()).samples, strain(7).astype(np.float32))
    # The dead writer's temporary is reaped by name once old enough.
    assert len([n for n in files_in(directory) if n.startswith(commit_module.TMP_PREFIX)]) == 1
    assert reap(directory, max_age=0) == 1 and files_in(directory) == tile_files(store.root, key())
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


# --- Interleavings ----------------------------------------------------------------------
#
# A child process runs two writers of different content and one forget of the same tile as
# threads, one shared filesystem operation at a time, in an order the parent chooses. An
# audit hook parks each actor before every operation on the tile's directory or files
# (open, link, unlink, listing), other than its own temporaries: everything else an actor
# does (sealing, writing and hashing its temporaries) is private to it. Every end state is
# then judged through the public API alone.

#: The audited operations through which an actor reads or changes a tile's files, and the
#: index of the path among each one's arguments.
SHARED_EVENTS = {"open": 0, "os.link": 1, "os.remove": 0, "os.scandir": 0}

#: The raced tile and a neighbour in its directory that a forget of the tile must not touch.
RACED, NEIGHBOUR = TileKey("RACE", "H1", 4, 4, 3), TileKey("RACE", "H1", 4, 4, 33)

#: The two writers' contents in each race: different bytes for one tile, and the same.
RACES = {"differ": {"A": np.arange(16.0), "B": np.arange(16.0) + 1.0},
         "equal": {"A": np.arange(16.0), "B": np.arange(16.0)}}


class Interleaver:
    """Runs actors as threads that take turns at each shared operation on ``directory``."""

    #: The interleaver whose actors the process's audit hook parks.
    active = None

    def __init__(self, directory):
        self.directory = os.fspath(directory)
        self.turns = threading.Condition()
        self.names, self.parked, self.finished, self.trace = {}, {}, {}, []
        self.granted = None

    @classmethod
    def hook(cls, event, args):
        interleaver = cls.active
        if interleaver is not None and event in SHARED_EVENTS:
            interleaver._park(event, args[SHARED_EVENTS[event]])

    def _park(self, event, path):
        name = self.names.get(threading.get_ident())
        if name is None or isinstance(path, int):
            return
        path = os.fsdecode(path)
        parent, base = os.path.split(path)
        if path != self.directory and (parent != self.directory or base.startswith(commit_module.TMP_PREFIX)):
            return
        operation = f"{event} {base}"
        with self.turns:
            self.parked[name] = operation
            self.turns.notify_all()
            if not self.turns.wait_for(lambda: self.granted == name, timeout=30):
                raise TimeoutError(f"Gave up after 30 s: {name} was never let past {operation}.")
            self.granted = None
            del self.parked[name]
            self.trace.append(f"{name} {operation}")

    def run(self, actors, choose, timeout=30):
        """Run every actor to its end; ``choose(parked)`` picks who performs the next operation."""
        def body(name, action):
            self.names[threading.get_ident()] = name
            try:
                outcome = action()
            except Exception as error:  # an actor's outcome, judged by the caller
                outcome = error
            with self.turns:
                self.finished[name] = outcome
                self.turns.notify_all()

        threads = [threading.Thread(target=body, args=item, daemon=True) for item in actors.items()]
        settled = lambda: self.granted is None and len(self.parked) + len(self.finished) == len(actors)
        with self.turns:
            # One actor at a time runs to its first shared operation, so no two threads
            # ever contend for the interpreter.
            for name, thread in zip(actors, threads):
                thread.start()
                if not self.turns.wait_for(lambda: name in self.parked or name in self.finished, timeout):
                    raise TimeoutError(f"Gave up after {timeout} s: {name} never reached a shared operation.")
            while True:
                if not self.turns.wait_for(settled, timeout):
                    raise TimeoutError(f"Gave up after {timeout} s: the actors never settled ({self.trace}).")
                if not self.parked:
                    break
                self.granted = choose(dict(self.parked))
                self.turns.notify_all()
        for thread in threads:
            thread.join(timeout)
        return dict(self.finished)


def _phases(*phases):
    """
    A schedule of phases (actor, until): the actor alone performs operations until the
    next one it would perform matches the regex ``until`` (None: until it ends).
    """
    pending = list(phases)

    def choose(parked):
        while pending:
            name, until = pending[0]
            if name in parked and not (until and re.search(until, parked[name])):
                return name
            pending.pop(0)
        return sorted(parked)[0]
    return choose


def _every_schedule(run):
    """Call ``run(choose)`` once for every order of its choices, depth first."""
    prefix = []
    while True:
        choices = []

        def choose(parked):
            names = sorted(parked)
            chosen = prefix[len(choices)] if len(choices) < len(prefix) else names[0]
            assert chosen in names, f"replaying {prefix} met {names} at choice {len(choices)}"
            choices.append((names, chosen))
            return chosen

        yield run(choose)
        while choices:
            names, chosen = choices.pop()
            later = names[names.index(chosen) + 1:]
            if later:
                prefix = [chosen for _, chosen in choices] + [later[0]]
                break
        else:
            return


def _judge(store, outcomes, contents):
    """
    The end state of a race over ``RACED`` and every way it breaks the store's promises:
    each writer commits, conflicts or loses to the forget by name, the tile is whole,
    absent or incomplete (never a record over other bytes), the next writer repairs it,
    and the neighbour is untouched.
    """
    problems = [f"{name} raised {outcome!r}" for name, outcome in outcomes.items()
                if isinstance(outcome, Exception) and not isinstance(outcome, (StoreConflict, ForgetRaced))]
    sealed = [seal(RACED, samples)[0] for samples in contents.values()]
    try:
        served = store.open(RACED, verify=True).samples
        state = "whole"
        if not any(np.array_equal(served, content) for content in sealed):
            problems.append("the whole tile serves bytes no writer wrote")
    except TileMissing:
        state = "absent"
    except TileIncomplete:
        state = "incomplete"
    except TileRefused as refusal:
        return "refused", problems + [f"a record names other bytes than its samples: {refusal}"]
    if state != "whole":
        for samples, content in zip(contents.values(), sealed):
            try:
                store.commit(RACED, samples)
            except StoreConflict:
                continue
            except StoreError as error:
                problems.append(f"the next writer of the {state} tile raised {error!r}")
                break
            if not np.array_equal(store.open(RACED, verify=True).samples, content):
                problems.append(f"the {state} tile, repaired, serves other bytes")
            break
        else:
            problems.append(f"the {state} tile is stuck: every next writer conflicts")
    try:
        store.open(NEIGHBOUR, verify=True)
    except StoreError as error:
        problems.append(f"the neighbour was harmed: {error!r}")
    return state, problems


def _race(root, choose, contents, judged=None):
    """
    Writers A and B of ``contents`` and a forget of ``RACED``, as ``choose`` orders them.
    ``judged`` caches the judgement of each distinct end state: the actors' outcomes,
    and each file's name with the samples it holds or, for a record, names.
    """
    root = Path(root)
    store = Store(root)
    store.commit(NEIGHBOUR, np.zeros(NEIGHBOUR.length))  # also reaps the directory, once
    interleaver = Interleaver(RACED.directory(root))
    actors = {name: (lambda samples=samples: store.commit(RACED, samples))
              for name, samples in contents.items()}
    actors["F"] = lambda: store.forget([RACED])
    Interleaver.active = interleaver
    try:
        outcomes = interleaver.run(actors, choose)
    finally:
        Interleaver.active = None
    named = {name: type(outcome).__name__ for name, outcome in outcomes.items()}
    end = (tuple(sorted(named.items())), tuple(
        (path.name, json.loads(path.read_bytes())["sha256"] if path.suffix == ".json" else sha_of(path))
        for path in sorted(RACED.directory(root).iterdir())))
    if judged is None or end not in judged:
        judgement = _judge(store, outcomes, contents)
        if judged is None:
            judged = {}
        judged[end] = judgement
    state, problems = judged[end]
    return {"state": state, "problems": problems, "trace": interleaver.trace, "outcomes": named}


def _races(scratch):
    """
    The child's reports. ``once``, one known race of different contents: A commits its
    samples and stops before its record; a forget; B commits its samples and stops
    before its record; A finishes, then B. ``every``, for each of ``RACES``, every
    interleaving: how many, the end states and outcomes, and each problem.
    """
    sys.addaudithook(Interleaver.hook)
    record = r"os\.link .*\.json$"
    once = _race(Path(scratch) / "once", _phases(
        ("A", record), ("F", None), ("B", record), ("A", None), ("B", None)), RACES["differ"])
    # Durability is not what is raced here, and every run would pay for it.
    commit_module.os.fsync = lambda descriptor: None
    return {"once": once, "every": {race: _race_all(Path(scratch) / race, contents)
                                    for race, contents in RACES.items()}}


def _race_all(scratch, contents):
    report = {"runs": 0, "states": {}, "outcomes": {}, "problems": []}
    runs, judged = iter(range(1 << 30)), {}

    def run(choose):
        root = Path(scratch) / f"run-{next(runs)}"
        try:
            return _race(root, choose, contents, judged)
        finally:
            shutil.rmtree(root)

    for result in _every_schedule(run):
        report["runs"] += 1
        report["states"][result["state"]] = report["states"].get(result["state"], 0) + 1
        for name, outcome in result["outcomes"].items():
            counted = report["outcomes"].setdefault(name, {})
            counted[outcome] = counted.get(outcome, 0) + 1
        if result["problems"]:
            report["problems"].append({"problems": result["problems"], "trace": result["trace"]})
    return report


@pytest.fixture(scope="module")
def races(tmp_path_factory):
    """The child that runs the known race and every interleaving of it, once per module."""
    return _report(_child("races", tmp_path_factory.mktemp("races")), timeout=500)


def test_a_forget_between_two_writers_never_leaves_a_record_over_other_samples(races):
    result = races["once"]
    print(result)
    assert result["problems"] == [], result
    assert result["outcomes"] == {"A": "Commit", "B": "StoreConflict", "F": "tuple"}
    # A's record lacks the samples the forget removed: refused by name, then repaired.
    assert result["state"] == "incomplete"


def test_every_interleaving_of_two_writers_and_a_forget_is_whole_absent_or_repairable(races):
    for race, report in races["every"].items():
        print(race, {name: report[name] for name in ("runs", "states", "outcomes")})
        assert report["problems"] == [], (race, len(report["problems"]), report["problems"][:3])
        # Every kind of end state and outcome was reached, so the races were explored;
        # writers of the same bytes never conflict.
        assert set(report["states"]) == {"whole", "absent", "incomplete"}, race
        outcomes = {"Commit", "ForgetRaced"} | ({"StoreConflict"} if race == "differ" else set())
        for writer in ("A", "B"):
            assert set(report["outcomes"][writer]) == outcomes, (race, report["outcomes"])


if __name__ == "__main__":
    role, arguments = sys.argv[1], sys.argv[2:]
    result = {"writer": _writer, "reader": _reader, "killed": _killed, "races": _races}[role](*arguments)
    print(json.dumps(result))


def test_verify_lists_orphan_samples_and_forget_alone_removes_them(tmp_path):
    # A writer that lost the race for a tile's record leaves samples no record names:
    # here, the record of the first content is gone when other content commits.
    store = Store(tmp_path)
    store.commit(key(), strain(7))
    orphan = samples_file(tmp_path, key())
    key().record_path(tmp_path).unlink()
    store.commit(key(), strain(7, offset=1e-22))
    assert orphan.exists() and samples_file(tmp_path, key()) != orphan
    [line] = store.verify()
    assert line.startswith(f"{orphan}: orphan samples, named by no record")
    assert np.array_equal(store.open(key()).samples, (strain(7, offset=1e-22)).astype(np.float32))
    store.forget([key()])
    assert store.verify() == [] and not orphan.exists()

