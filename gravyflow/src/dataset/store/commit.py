"""
Committing tiles: each written once and whole, by ``os.link``, and never replaced,
extended or merged.

The commit, for one tile:

1. Seal the whole fetched tile (``tile.seal``) and write it to a temporary file in the
   tile's directory, fsync'd and read-only; its sha256 names it.
2. When the tile has a record, its sha256 must be this writer's, or the commit is
   ``StoreConflict`` (the source changed under the store) before anything is linked.
3. ``os.link`` the temporary to ``k.<sha256>.npy``. The link fails if that file
   exists; then it is hashed against its name: equal is a no-op, different is
   ``StoreConflict`` (the file was altered). Nothing is overwritten. The temporary is
   removed, and the committed file's identity (``tile.identity``) read from an open
   handle on it.
4. Write the record the same way and link it to ``k.json``, with the same check over
   the record's content (every field but the commit metadata).

A record can only name the samples file of the bytes it hashed, so no race leaves a
record over other samples. A tile exists only when its record does, and a record
implies complete samples, unless a ``forget`` raced the commit: then the record may
lack its samples file, the one incomplete state, which open refuses by name and the
next writer of the record's content repairs. A writer stopped between the two links,
or one that lost the race for the record, leaves a samples file no record names: the
tile's next writer of the same bytes adopts it, and ``forget`` removes it. A commit
whose check finds the file it is checking removed by a ``forget`` raises
``ForgetRaced``. Every error writing, linking or removing raises ``StoreWriteError``
naming the path and the errno; none is swallowed, and no committed file is ever opened
to write.

Temporary files are reaped by name and age, once per directory, when a process first
writes there. ``forget`` is the store's only deletion.
"""

from __future__ import annotations

import contextlib
import datetime
import errno as errno_codes
import json
import os
import secrets
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Hashable, Iterable, Iterator, List, Mapping, Optional, Tuple

import numpy as np

from gravyflow.src.dataset.acquisition.gwosc_data import user_cache_dir

from .budget import ByteBudget
from .tile import (
    Identity, StoreError, TileIncomplete, TileKey, TileMissing, TileRecord, content, file_sha256,
    identity, read_record, seal,
)
from .view import Tile, open_tile, verify

#: Every temporary file the store writes starts with this, so the reaper finds them.
TMP_PREFIX = ".gfstore-tmp-"

#: A temporary file older than this is a dead writer's and is reaped.
REAP_AGE_SECONDS = 86400.0

#: How long a caller waits for another caller's fetch of the same tile, by default.
DEFAULT_WAIT_SECONDS = 600.0

#: How many times ``ensure`` fetches and commits a tile that a ``forget`` keeps removing.
RACE_ATTEMPTS = 3


class StoreWriteError(StoreError):
    """Writing, linking or removing a store file failed. Nothing committed was changed."""

    def __init__(self, action: str, path: Path, error: OSError):
        self.path = Path(path)
        self.errno = error.errno
        name = errno_codes.errorcode.get(error.errno, str(error.errno))
        super().__init__(
            f"The store could not {action} {path}: {name} ({error.strerror}). No committed "
            "file was changed."
        )


class StoreConflict(StoreError):
    """
    A tile's committed content differs from what this writer fetched for it: the
    source changed under the store, or a committed file was altered. Nothing is
    overwritten.
    """


class ForgetRaced(StoreError):
    """
    A ``forget`` removed a tile's file while a commit was checking it: a lost race, not
    an I/O failure. The tile is left absent or incomplete, never mixed; commit it again.
    """


class TileWaitTimeout(StoreError, TimeoutError):
    """Another caller's fetch of a tile did not finish within this caller's bound."""


@dataclass(frozen=True)
class Fetched:
    """One whole tile as a source delivered it."""

    samples: np.ndarray
    not_in_file: Tuple[Tuple[int, int], ...] = ()
    tools: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Commit:
    """
    What one writer's commit of a tile did. A ``forget`` may remove either file after
    the commit linked or found it, so a commit never promises the tile is still whole;
    ``Store.ensure`` opens it and commits again when a forget won.
    """

    key: TileKey
    linked_samples: bool  # this writer's link made k.<sha256>.npy
    linked_record: bool  # this writer's link made k.json

    @property
    def contended(self) -> bool:
        """Another writer had linked one of the tile's files first."""
        return not (self.linked_samples and self.linked_record)


@contextlib.contextmanager
def _named(action: str, path: Path) -> Iterator[None]:
    """Raise any OSError in the block as a StoreWriteError naming ``action`` and ``path``."""
    try:
        yield
    except OSError as error:
        raise StoreWriteError(action, path, error) from error


def _write_temporary(directory: Path, data: Callable) -> Path:
    """A new read-only temporary file in ``directory`` holding what ``data`` writes, fsync'd."""
    path = directory / f"{TMP_PREFIX}{os.getpid()}-{secrets.token_hex(8)}"
    with _named("write the temporary file", path):
        handle = open(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444), "wb")
        try:
            with handle:
                data(handle)
                handle.flush()
                os.fsync(handle.fileno())
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(path)
            raise
    return path


def _link(source: Path, target: Path) -> bool:
    """Link ``source`` to ``target``: True if this call made it, False if it existed."""
    with _named("link", target):
        try:
            os.link(source, target)
        except FileExistsError:
            return False
    return True


def _remove_temporary(path: Path) -> None:
    with _named("remove the temporary file", path), contextlib.suppress(FileNotFoundError):
        os.unlink(path)


def _changed(key: TileKey, path: Path, committed: str, fetched: str) -> str:
    return (
        f"{key.describe()} is committed at {path} with sha256 {committed}, but this writer "
        f"fetched sha256 {fetched}: the source changed under the store. Nothing was "
        "overwritten; Store.forget the tile to fetch it again."
    )


def _forgotten(key: TileKey, path: Path) -> str:
    return (
        f"{key.describe()} lost a race with Store.forget: {path} was removed while this "
        "writer checked it. The tile is absent or incomplete, never mixed; commit it again."
    )


def reap(directory: Path, max_age: float = REAP_AGE_SECONDS) -> int:
    """Remove the store's temporary files older than ``max_age`` s in ``directory``."""
    cutoff, reaped = time.time() - max_age, 0
    with _named("reap temporary files in", directory), os.scandir(directory) as entries:
        for entry in entries:
            if not entry.name.startswith(TMP_PREFIX):
                continue
            try:
                if entry.stat(follow_symlinks=False).st_mtime < cutoff:
                    os.unlink(entry.path)
                    reaped += 1
            except FileNotFoundError:  # another process reaped it first
                continue
    return reaped


class Store:
    """
    The tile store under ``root`` (default: ``store/`` in the user cache, which
    honours ``GRAVYFLOW_CACHE_DIR``). Many processes may share one root. In one
    process, ``ensure`` fetches a tile once per (stream, tile) at a time, with the
    fetch admitted by ``budget`` when one is given.
    """

    def __init__(self, root: Optional[Path] = None, *, budget: Optional[ByteBudget] = None):
        self.root = Path(root) if root is not None else user_cache_dir("store")
        self.budget = budget
        self._lock = threading.Lock()
        self._flights: Dict[Tuple[Hashable, TileKey], Future] = {}
        self._reaped: set = set()

    def open(self, key: TileKey, *, verify: bool = False) -> Tile:
        """The committed tile ``key``, checked (``view.open_tile``)."""
        return open_tile(self.root, key, verify=verify)

    def verify(self) -> List[str]:
        """Every committed tile checked in full; one line per problem (``view.verify``)."""
        return verify(self.root)

    def commit(self, key: TileKey, samples, not_in_file: Iterable[Tuple[int, int]] = (),
               tools: Optional[Mapping[str, str]] = None) -> Commit:
        """Commit the whole fetched tile ``key``, once (see the module's protocol)."""
        sealed, unheld = seal(key, samples, not_in_file)
        directory = key.directory(self.root)
        with _named("make the tile directory", directory):
            directory.mkdir(parents=True, exist_ok=True)
        self._reap_once(directory)
        record_path = key.record_path(self.root)
        temporaries = []
        try:
            temporaries.append(_write_temporary(
                directory, lambda handle: np.lib.format.write_array(handle, sealed, allow_pickle=False)))
            with _named("hash the temporary file", temporaries[0]), open(temporaries[0], "rb") as handle:
                sha256 = file_sha256(handle)
            self._check_committed(key, sha256)
            linked_samples, samples_identity = self._link_samples(key, temporaries[0], sha256)
            record = TileRecord(key, sha256, unheld, samples_identity, {
                "tools": dict(tools or {}),
                "fetched_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            })
            encoded = record.encode()
            temporaries.append(_write_temporary(directory, lambda handle: handle.write(encoded)))
            linked_record = _link(temporaries[1], record_path)
            if not linked_record:
                self._check_record(key, record_path, encoded)
        except BaseException:
            for path in temporaries:  # best effort: the error in flight is the one to report
                with contextlib.suppress(OSError):
                    os.unlink(path)
            raise
        _remove_temporary(temporaries[1])
        return Commit(key, linked_samples, linked_record)

    def _check_committed(self, key: TileKey, sha256: str) -> None:
        """
        Refuse, before anything is linked, a tile whose record names other samples, so
        a writer that comes after a conflicting commit leaves no samples file behind.
        """
        record_path = key.record_path(self.root)
        try:
            with _named("read the committed record", record_path):
                committed = read_record(self.root, key)
        except TileMissing:
            return
        if committed.sha256 != sha256:
            raise StoreConflict(_changed(key, record_path, committed.sha256, sha256))

    def _link_samples(self, key: TileKey, temporary: Path, sha256: str) -> Tuple[bool, Identity]:
        """
        Link ``temporary`` as the samples file its sha256 names, or check the one another
        writer linked: whether this call made the link, and the committed file's
        identity. The identity is read from a handle on that very file once the
        temporary is removed, so the file's links, and with them its ctime, are final.
        """
        samples_path = key.samples_path(self.root, sha256)
        linked = _link(temporary, samples_path)
        committed = temporary if linked else samples_path
        with _named("verify the committed samples", committed):
            try:
                handle = open(committed, "rb")
            except FileNotFoundError:
                if linked:  # this writer's own temporary: no race removes it
                    raise
                raise ForgetRaced(_forgotten(key, samples_path)) from None
            with handle:
                if not linked and (found := file_sha256(handle)) != sha256:
                    raise StoreConflict(
                        f"{samples_path} holds bytes with sha256 {found}, not the sha256 its name "
                        f"states: {key.describe()}'s committed samples were altered. Nothing was "
                        "overwritten; Store.forget the tile to fetch it again."
                    )
                _remove_temporary(temporary)
                return linked, identity(os.fstat(handle.fileno()))

    @staticmethod
    def _check_record(key: TileKey, record_path: Path, encoded: bytes) -> None:
        with _named("read the committed record", record_path):
            try:
                committed = record_path.read_bytes()
            except FileNotFoundError:
                raise ForgetRaced(_forgotten(key, record_path)) from None
        try:
            same = content(json.loads(committed)) == content(json.loads(encoded))
        except ValueError:
            same = False
        if not same:
            raise StoreConflict(
                f"{key.describe()} has a committed record at {record_path} whose content differs "
                "from this writer's. Nothing was overwritten; Store.forget the tile to fetch it again."
            )

    def _reap_once(self, directory: Path) -> None:
        with self._lock:
            if directory in self._reaped:
                return
            self._reaped.add(directory)
        reap(directory)

    def ensure(self, key: TileKey, fetch: Callable[[], Fetched], *, stream: Hashable = None,
               nbytes: int = 0, timeout: float = DEFAULT_WAIT_SECONDS) -> Tile:
        """
        The committed tile ``key``, fetched and committed first when it is not.

        In this process one (stream, tile) is fetched at most once at a time: other
        callers wait for that fetch, at most ``timeout`` s. Keying by stream keeps
        every in-flight fetch to one stream, so streams of different splits share
        nothing, whatever the block length. A failed fetch leaves the table and is
        raised to every caller waiting on it; each retries under its own budget.
        Across processes a duplicate fetch costs bandwidth, never correctness. The
        fetch and its commit hold ``nbytes`` of the byte budget.

        A ``forget`` that removes the tile while it is committed wins the race, and the
        bytes fetched may predate it, so the tile is fetched and committed again, at
        most ``RACE_ATTEMPTS`` times in all before ``ForgetRaced`` is raised.
        """
        try:
            return self.open(key)
        except (TileMissing, TileIncomplete):
            pass
        flight = (stream, key)
        with self._lock:
            future = self._flights.get(flight)
            leading = future is None
            if leading:
                future = self._flights[flight] = Future()
        if not leading:
            try:
                return future.result(timeout)
            except TimeoutError:
                raise TileWaitTimeout(
                    f"Waited {timeout} s for another caller's fetch of {key.describe()}."
                ) from None
        try:
            admission = self.budget.admit(nbytes, timeout) if self.budget else contextlib.nullcontext()
            with admission:
                tile = self._fetch_and_commit(key, fetch)
        except BaseException as error:
            future.set_exception(error)
            raise
        finally:
            with self._lock:
                del self._flights[flight]
        future.set_result(tile)
        return tile

    def _fetch_and_commit(self, key: TileKey, fetch: Callable[[], Fetched]) -> Tile:
        """The tile fetched, committed and opened, again when a ``forget`` races it."""
        for _ in range(RACE_ATTEMPTS):
            fetched = fetch()
            try:
                self.commit(key, fetched.samples, fetched.not_in_file, fetched.tools)
                return self.open(key)
            except (ForgetRaced, TileMissing, TileIncomplete) as error:  # a forget won
                lost = error
        raise ForgetRaced(
            f"{key.describe()} was forgotten while this caller committed it, {RACE_ATTEMPTS} "
            "times running; the last race is chained below."
        ) from lost

    def forget(self, keys: Iterable[TileKey]) -> Tuple[TileKey, ...]:
        """
        Delete the tiles ``keys``: the store's only deletion, always explicit. Each
        record goes first, then every samples file of its tile, so a commit racing this
        leaves at worst a record without its samples, which is refused by name and
        repaired by its next writer. Readers keep their maps until they reopen.
        Returns the keys found.
        """
        removed = []
        for key in keys:
            found = False
            record_path = key.record_path(self.root)
            with _named("remove", record_path), contextlib.suppress(FileNotFoundError):
                record_path.unlink()
                found = True
            with _named("list the samples files in", key.directory(self.root)):
                samples_files = key.samples_files(self.root)
            for path in samples_files:
                with _named("remove", path), contextlib.suppress(FileNotFoundError):
                    path.unlink()
                    found = True
            if found:
                removed.append(key)
        return tuple(removed)
