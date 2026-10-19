"""
Remote files in the store: whole files pinned by sha256, fetched only when asked.

A ``RemoteFile`` names one file somewhere else (trained weights, a benchmark's
segment list, its noise): where it is published, its size, its sha256, its licence
and whether gravyflow may redistribute it. The manifest of every file gravyflow
knows is ``resources.MANIFEST``.

- ``cached(remote)`` is the committed file, or ``RemoteFileMissing`` naming the
  fetch that would make it: nothing is ever downloaded implicitly.
- ``fetch(remote)`` downloads it, explicitly, and commits it the way the store
  commits a tile: streamed into a read-only temporary file in its directory,
  fsync'd and hashed on the way, then ``os.link``'d to its name, which fails if the
  name exists, so nothing is overwritten. A download whose size or sha256 differs
  from the pin is refused and nothing is kept.

A file lives at ``<store root>/files/<sha256>/<basename>``: named by its content,
so a pin can never serve other bytes, and keeping its basename for readers that
read meaning into an extension (Keras's ``.keras``). The store root honours
``GRAVYFLOW_CACHE_DIR``.
"""

from __future__ import annotations

import contextlib
import errno
import hashlib
import http.client
import re
import shutil
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterator, Optional

from gravyflow.src.dataset.acquisition.gwosc_data import user_cache_dir

from .commit import StoreConflict, StoreWriteError, _link, _named, _remove_temporary, _write_temporary, reap
from .tile import StoreError, file_sha256

#: The store's directory of remote files, under its root.
FILES_DIRECTORY = "files"

#: Where gravyflow's own releases publish files: a file gravyflow may not
#: redistribute is never fetched from here, only from its authors' source.
GRAVYFLOW_RELEASES = "https://github.com/mrknorman/gravyflow/releases/"

#: The schemes a remote file may be published under (``file`` serves tests and
#: local mirrors).
URL_SCHEMES = ("https", "http", "file")

#: Bytes read from the network per step, and the default socket timeout, in s.
_CHUNK = 1 << 20
DEFAULT_TIMEOUT_SECONDS = 60.0

_SHA256 = re.compile(r"[0-9a-f]{64}")


class RemoteFileMissing(StoreError, FileNotFoundError):
    """A remote file has not been fetched into the store; nothing fetches it implicitly."""


class RemoteFileMismatch(StoreError):
    """Bytes offered as a remote file are not the pinned ones: another size or sha256."""


class RemoteFetchFailed(StoreError):
    """A remote file could not be downloaded; nothing was kept."""


@dataclass(frozen=True)
class RemoteFile:
    """
    One file published elsewhere, pinned by its content.

    Args:
        name: Its name in the manifest: a relative POSIX path (``model/x.keras``),
            whose basename the cached file keeps.
        url: Where its authors, or gravyflow's release, publish it; pinned to one
            commit or release where the host allows it.
        sha256: The sha256 of its bytes, lowercase hex.
        size: Its size in bytes.
        licence: Its licence as an SPDX identifier, or None when it has none.
        redistributable: Whether gravyflow may publish it itself. A file with no
            licence is not, and a file that is not is fetched only from its
            authors' source (never from ``GRAVYFLOW_RELEASES``).
    """

    name: str
    url: str
    sha256: str
    size: int
    licence: Optional[str]
    redistributable: bool

    def __post_init__(self):
        path = PurePosixPath(self.name) if isinstance(self.name, str) else None
        if path is None or not self.name or path.is_absolute() or ".." in path.parts or str(path) != self.name:
            raise ValueError(f"A remote file's name is a plain relative path, got {self.name!r}.")
        if urllib.parse.urlsplit(self.url).scheme not in URL_SCHEMES:
            raise ValueError(f"{self.name}: its url must be one of {URL_SCHEMES}, got {self.url!r}.")
        if not isinstance(self.sha256, str) or not _SHA256.fullmatch(self.sha256):
            raise ValueError(f"{self.name}: its sha256 must be 64 lowercase hex digits, got {self.sha256!r}.")
        if isinstance(self.size, bool) or not isinstance(self.size, int) or self.size < 0:
            raise ValueError(f"{self.name}: its size must be a whole number of bytes, got {self.size!r}.")
        if self.licence is not None and (not isinstance(self.licence, str) or not self.licence):
            raise ValueError(f"{self.name}: its licence is an SPDX identifier or None, got {self.licence!r}.")
        if not isinstance(self.redistributable, bool):
            raise ValueError(f"{self.name}: redistributable must be True or False.")
        if self.redistributable and self.licence is None:
            raise ValueError(f"{self.name} has no licence, so gravyflow may not redistribute it.")
        if not self.redistributable and self.url.startswith(GRAVYFLOW_RELEASES):
            raise ValueError(
                f"{self.name} may not be redistributed, so it is fetched from its authors' source, "
                f"never from gravyflow's releases ({self.url})."
            )

    def path(self, root: Optional[Path] = None) -> Path:
        """Where the store keeps it under ``root`` (default: the store's root)."""
        root = Path(root) if root is not None else user_cache_dir("store")
        return root / FILES_DIRECTORY / self.sha256 / PurePosixPath(self.name).name


def cached(remote: RemoteFile, root: Optional[Path] = None) -> Path:
    """
    The committed copy of ``remote`` under the store's ``root``; ``RemoteFileMissing``
    when it has not been fetched, naming the explicit fetch. A file of another size
    at its path was altered and is refused by name. Its sha256 was checked when it
    was committed and is not read again here: the file is read-only and named by
    that sha256, so only a change from outside the store could alter it.
    """
    path = remote.path(root)
    try:
        size = path.stat().st_size
    except FileNotFoundError:
        raise RemoteFileMissing(
            f"{remote.name} ({remote.size:,} bytes, from {remote.url}) is not in the store at {path}. "
            f"Nothing is downloaded implicitly: fetch it once with gravyflow.fetch({remote.name!r})."
        ) from None
    if size != remote.size:
        raise RemoteFileMismatch(
            f"{path} holds {size:,} bytes, not the {remote.size:,} pinned for {remote.name}: it was "
            "altered. Remove it and fetch it again."
        )
    return path


def fetch(remote: RemoteFile, root: Optional[Path] = None,
          timeout: float = DEFAULT_TIMEOUT_SECONDS) -> Path:
    """
    Download ``remote`` into the store under ``root``, once, and return its path; the
    committed copy when it is already there. Explicit by design: nothing else
    downloads. Refuses, keeping nothing: too little free disk for it
    (``StoreWriteError``, ENOSPC), a failed download (``RemoteFetchFailed``), and
    bytes of another size or sha256 (``RemoteFileMismatch``).
    """
    path = remote.path(root)
    try:
        return cached(remote, root)
    except RemoteFileMissing:
        pass
    directory = path.parent
    with _named("make the directory", directory):
        directory.mkdir(parents=True, exist_ok=True)
    reap(directory)
    free = shutil.disk_usage(directory).free
    if free < remote.size:
        raise StoreWriteError("download", path, OSError(
            errno.ENOSPC, f"{remote.name} needs {remote.size:,} bytes and {free:,} are free"))
    digest = hashlib.sha256()

    def download(handle) -> None:
        # Only the network's errors are a failed download; the temporary file's own
        # are StoreWriteError, named by _write_temporary.
        received = 0
        with _network(remote):
            response = urllib.request.urlopen(remote.url, timeout=timeout)
        with response:
            announced = response.headers.get("Content-Length")
            if announced is not None and int(announced) != remote.size:
                raise RemoteFileMismatch(_mismatch(remote, f"announces {int(announced):,} bytes"))
            while True:
                with _network(remote):
                    block = response.read(_CHUNK)
                if not block:
                    break
                received += len(block)
                if received > remote.size:
                    raise RemoteFileMismatch(_mismatch(remote, "sent more bytes than pinned"))
                digest.update(block)
                handle.write(block)
        if received != remote.size:
            raise RemoteFileMismatch(_mismatch(remote, f"sent {received:,} bytes"))
        if digest.hexdigest() != remote.sha256:
            raise RemoteFileMismatch(_mismatch(remote, f"has sha256 {digest.hexdigest()}"))

    temporary = _write_temporary(directory, download)
    try:
        if not _link(temporary, path):
            with _named("verify the committed file", path), open(path, "rb") as handle:
                if (found := file_sha256(handle)) != remote.sha256:
                    raise StoreConflict(
                        f"{path} holds bytes with sha256 {found}, not the {remote.sha256} its directory "
                        "names: it was altered. Nothing was overwritten; remove it and fetch it again."
                    )
    finally:
        _remove_temporary(temporary)
    return path


@contextlib.contextmanager
def _network(remote: RemoteFile) -> Iterator[None]:
    """Raise a network error in the block as ``RemoteFetchFailed`` naming ``remote``."""
    try:
        yield
    except (urllib.error.URLError, http.client.HTTPException, OSError) as error:
        raise RemoteFetchFailed(
            f"Downloading {remote.name} from {remote.url} failed: {error}. Nothing was kept."
        ) from error


def _mismatch(remote: RemoteFile, what: str) -> str:
    return (
        f"The download of {remote.name} from {remote.url} {what}, but its pin is {remote.size:,} bytes "
        f"with sha256 {remote.sha256}. Nothing was kept."
    )


__all__ = [
    "DEFAULT_TIMEOUT_SECONDS", "FILES_DIRECTORY", "GRAVYFLOW_RELEASES", "RemoteFetchFailed",
    "RemoteFile", "RemoteFileMismatch", "RemoteFileMissing", "cached", "fetch",
]
