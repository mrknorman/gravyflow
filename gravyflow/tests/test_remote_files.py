"""
Remote files: pinned by size and sha256, fetched into the store only when asked,
and refused, keeping nothing, when the bytes are not the pinned ones. MLGWSC-1's
segment list is one: shipped, and pinned to the authors' copy.

Every download here is from a ``file://`` URL in a temporary directory; no socket
opens.
"""

import hashlib
import os
from collections import namedtuple

import numpy as np
import pytest

import gravyflow as gf
from gravyflow.src import resources
from gravyflow.src.dataset.acquisition import mlgwsc1
from gravyflow.src.dataset.store import files
from gravyflow.src.dataset.store.commit import TMP_PREFIX, StoreWriteError
from gravyflow.src.dataset.store.files import (
    RemoteFetchFailed, RemoteFile, RemoteFileMismatch, RemoteFileMissing,
)

PAYLOAD = b"pinned bytes\n"
SHA256 = hashlib.sha256(PAYLOAD).hexdigest()


def published(tmp_path, data=PAYLOAD, name="data/file.bin"):
    """A file served at a file:// URL, and its RemoteFile pinned to ``PAYLOAD``."""
    source = tmp_path / "published" / "file.bin"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(data)
    return source, RemoteFile(name, source.as_uri(), SHA256, len(PAYLOAD),
                              licence="Apache-2.0", redistributable=True)


def leftovers(remote, root):
    """The files in ``remote``'s store directory, temporaries included."""
    directory = remote.path(root).parent
    return sorted(p.name for p in directory.iterdir()) if directory.exists() else []


def test_a_fetch_is_explicit_and_kept_once(tmp_path):
    root = tmp_path / "store"
    source, remote = published(tmp_path)
    with pytest.raises(RemoteFileMissing, match=r"gravyflow\.fetch\('data/file\.bin'\)"):
        files.cached(remote, root)
    path = files.fetch(remote, root)
    assert path == root / "files" / SHA256 / "file.bin" and path.read_bytes() == PAYLOAD
    assert not os.access(path, os.W_OK), "a committed file is read-only"
    assert files.cached(remote, root) == path
    source.unlink()  # a second fetch reads the store, not the source
    assert files.fetch(remote, root) == path
    assert leftovers(remote, root) == ["file.bin"]


def test_the_default_root_is_the_store_in_the_user_cache(tmp_path):
    _, remote = published(tmp_path)
    path = gf.fetch(remote)
    assert path.is_relative_to(os.environ["GRAVYFLOW_CACHE_DIR"]) and path.parts[-3] == "files"


@pytest.mark.parametrize("data, message", [
    (b"other bytes!\n", "has sha256"),
    (PAYLOAD + b"more", "more bytes than pinned|announces"),
    (PAYLOAD[:-1], "sent 12 bytes|announces"),
], ids=["sha256", "longer", "shorter"])
def test_other_bytes_are_refused_and_nothing_kept(tmp_path, data, message):
    root = tmp_path / "store"
    _, remote = published(tmp_path, data)
    with pytest.raises(RemoteFileMismatch, match=message):
        files.fetch(remote, root)
    assert leftovers(remote, root) == []


def test_a_failed_download_keeps_nothing(tmp_path):
    root = tmp_path / "store"
    _, remote = published(tmp_path)
    missing = RemoteFile(remote.name, (tmp_path / "absent.bin").as_uri(), SHA256, len(PAYLOAD),
                         licence=None, redistributable=False)
    with pytest.raises(RemoteFetchFailed, match="absent.bin"):
        files.fetch(missing, root)
    assert leftovers(missing, root) == []


def test_too_little_disk_is_refused_before_downloading(tmp_path, monkeypatch):
    root = tmp_path / "store"
    _, remote = published(tmp_path)
    usage = namedtuple("usage", "total used free")
    monkeypatch.setattr(files.shutil, "disk_usage", lambda path: usage(100, 100, 3))
    with pytest.raises(StoreWriteError, match="ENOSPC"):
        files.fetch(remote, root)
    assert leftovers(remote, root) == []


def test_an_altered_file_is_refused(tmp_path):
    root = tmp_path / "store"
    _, remote = published(tmp_path)
    path = files.fetch(remote, root)
    path.chmod(0o644)
    path.write_bytes(PAYLOAD + b"x")
    with pytest.raises(RemoteFileMismatch, match="altered"):
        files.cached(remote, root)


def test_an_explicit_fetch_rehashes_the_cached_file(tmp_path):
    """
    A file altered in place at its pinned size passes ``cached`` (size only, by
    design) but not an explicit fetch, which refuses it, removes it and names why.
    """
    root = tmp_path / "store"
    _, remote = published(tmp_path)
    path = files.fetch(remote, root)
    path.chmod(0o644)
    path.write_bytes(b"x" * len(PAYLOAD))
    assert files.cached(remote, root) == path
    with pytest.raises(RemoteFileMismatch, match=f"sha256 {hashlib.sha256(b'x' * len(PAYLOAD)).hexdigest()}"):
        files.fetch(remote, root)
    assert leftovers(remote, root) == [], "nothing corrupt is kept"
    assert files.fetch(remote, root).read_bytes() == PAYLOAD


def test_an_explicit_fetch_removes_a_cached_file_of_another_size(tmp_path):
    root = tmp_path / "store"
    _, remote = published(tmp_path)
    path = files.fetch(remote, root)
    path.chmod(0o644)
    path.write_bytes(PAYLOAD + b"x")
    with pytest.raises(RemoteFileMismatch, match="altered"):
        files.fetch(remote, root)
    assert leftovers(remote, root) == []


def test_a_lost_race_verifies_the_committed_file(tmp_path, monkeypatch):
    """A writer that finds the name taken checks those bytes against the pin."""
    root = tmp_path / "store"
    _, remote = published(tmp_path)
    path = remote.path(root)
    real_cached = files.cached

    def racing(remote_file, root_dir=None):  # another writer commits altered bytes first
        try:
            return real_cached(remote_file, root_dir)
        except RemoteFileMissing:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"x" * len(PAYLOAD))
            raise

    monkeypatch.setattr(files, "cached", racing)
    with pytest.raises(files.StoreConflict, match="altered"):
        files.fetch(remote, root)
    assert path.read_bytes() == b"x" * len(PAYLOAD), "nothing is overwritten"
    assert not any(name.startswith(TMP_PREFIX) for name in leftovers(remote, root))


@pytest.mark.parametrize("change, message", [
    (dict(name="/abs/file"), "plain relative path"),
    (dict(name="a/../b"), "plain relative path"),
    (dict(url="ftp://host/file"), "url must be one of"),
    (dict(sha256="ABC"), "64 lowercase hex"),
    (dict(size=-1), "whole number of bytes"),
    (dict(licence=None, redistributable=True), "no licence"),
    (dict(redistributable=False, url=files.GRAVYFLOW_RELEASES + "download/v1/x"), "authors' source"),
    (dict(redistributable=False, url="http://github.com/mrknorman/gravyflow/releases/download/v1/x"),
     "authors' source"),
    (dict(redistributable=False, url="https://GitHub.com:443/MrKNorman/GravyFlow/releases/download/v1/x"),
     "authors' source"),
    (dict(redistributable=False, url="https://www.github.com/mrknorman/gravyflow//releases/download/v1/x"),
     "authors' source"),
], ids=["absolute", "parent", "scheme", "sha256", "size", "unlicensed", "rehosted", "rehosted-http",
        "rehosted-case", "rehosted-www"])
def test_an_entry_is_checked_as_it_is_made(change, message):
    entry = dict(name="model/x.keras", url="https://example.org/x.keras", sha256=SHA256, size=1,
                 licence="MIT", redistributable=True)
    with pytest.raises(ValueError, match=message):
        RemoteFile(**{**entry, **change})


def test_the_manifest_names_its_files_where_they_live():
    assert all(entry.name == name for name, entry in resources.MANIFEST.items())
    with pytest.raises(ValueError, match="not in gravyflow's manifest"):
        resources.fetch("absent/file")


def test_an_entry_from_another_source_is_not_a_gravyflow_release():
    for url in ("https://github.com/mrknorman/gravyflow-data/releases/x",
                "https://example.org/mrknorman/gravyflow/releases/x"):
        assert not RemoteFile("m/x", url, SHA256, 1, licence=None, redistributable=False).redistributable


def test_fetching_a_shipped_file_returns_the_shipped_copy(monkeypatch):
    """A file installed with the package is never downloaded, by name or by its entry."""
    def no_network(*args, **kwargs):
        raise AssertionError("a shipped file was downloaded")

    monkeypatch.setattr(files.urllib.request, "urlopen", no_network)
    shipped = resources.RESOURCES / "mlgwsc1" / "segments.csv"
    assert resources.fetch(mlgwsc1.SEGMENTS) == shipped
    assert gf.fetch(resources.MANIFEST[mlgwsc1.SEGMENTS]) == shipped


# =============================================================================
# MLGWSC-1
# =============================================================================

def test_the_shipped_segment_list_is_the_pinned_one():
    """The shipped copy is byte-for-byte the authors' file the manifest pins."""
    entry = resources.MANIFEST[mlgwsc1.SEGMENTS]
    shipped = resources.path(mlgwsc1.SEGMENTS)
    assert shipped == resources.RESOURCES / "mlgwsc1" / "segments.csv"
    data = shipped.read_bytes()
    assert (len(data), hashlib.sha256(data).hexdigest()) == (entry.size, entry.sha256)
    assert (entry.licence, entry.redistributable) == ("Apache-2.0", True)


def test_the_segments_are_the_challenges():
    """362 O3a segments, GPS 1238205077 to 1253966055, 7,111,579 s, sorted and disjoint."""
    segments = mlgwsc1.segments()
    assert segments.shape == (362, 2)
    assert (segments[0, 0], segments[-1, 1]) == (1238205077.0, 1253966055.0)
    assert np.sum(segments[:, 1] - segments[:, 0]) == 7111579.0
    assert np.all(segments[1:, 0] > segments[:-1, 1])


def test_the_reservation_is_the_segments_exactly():
    """H1 and L1 each reserve exactly the 362 segments; no other detector reserves any time."""
    segments = mlgwsc1.segments()
    reservation = gf.MLGWSC1_O3A
    assert np.array_equal(reservation.of("H1"), segments)
    assert np.array_equal(reservation.of(gf.IFO.L1), segments)
    assert np.array_equal(reservation.of(None), segments)
    assert len(reservation.of("V1")) == 0
    assert gf.DEFAULT_SPLIT.reserved == (reservation,)
