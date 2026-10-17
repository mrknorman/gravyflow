"""
Where a materialised set's files live, and the one way they are written:
create-if-absent, never replace.

A writer commits each file whole, under its final name, only if no file has
that name. A file that already exists is compared: the same bytes are a no-op
(another writer, or an earlier run, committed the same content), other bytes
are a `MaterialiseConflict`. So writers racing on one set, or a run resumed
after an interruption, can only add what agrees with what is there.
"""

import os
import secrets
from pathlib import Path
from typing import Optional, Union

from gravyflow.src.dataset.materialise.format import MaterialiseConflict, sha256


class LocalBackend:
    """A set's files under a local directory. A commit is a hard link of a complete temporary file."""

    def __init__(self, root: Union[str, Path]):
        self.root = Path(root)

    def path(self, name: str) -> Path:
        return self.root / name

    def read(self, name: str) -> Optional[bytes]:
        """The bytes of ``name``, or None when it does not exist."""
        try:
            return self.path(name).read_bytes()
        except FileNotFoundError:
            return None

    def exists(self, name: str) -> bool:
        return self.path(name).exists()

    def create(self, name: str, data: bytes) -> None:
        """
        Commit ``data`` as ``name`` unless it exists. The bytes are written to a
        temporary of this writer's own (named by its process and a random token)
        and synced to the disk; then ``os.link`` creates the name atomically, for a
        file already complete, so readers never see part of one. Temporaries of
        ``name`` that a dead process left are removed first.

        Raises:
            MaterialiseConflict: ``name`` exists with other bytes.
        """
        target = self.path(name)
        target.parent.mkdir(parents=True, exist_ok=True)
        _remove_stale_temporaries(target)
        temporary = target.with_name(f".{target.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp")
        with open(temporary, "xb") as file:
            file.write(data)
            file.flush()
            os.fsync(file.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError:
            existing = target.read_bytes()
            if existing != data:
                raise MaterialiseConflict(
                    f"{target} holds other bytes (sha256 {sha256(existing)}) than this writer's "
                    f"(sha256 {sha256(data)}): two writers computed different bytes for it. Files are "
                    "never replaced; check that every writer runs one spec on one platform, with "
                    "bit-reproducible kernels."
                ) from None
        finally:
            temporary.unlink()


def _remove_stale_temporaries(target: Path) -> None:
    """Remove the temporaries of ``target`` whose writer's process is gone (an interrupted commit)."""
    for temporary in target.parent.glob(f".{target.name}.*.tmp"):
        pid = temporary.name[len(target.name) + 2:].split(".", 1)[0]
        if pid.isdigit() and not _alive(int(pid)):
            temporary.unlink(missing_ok=True)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # another user's process
        return True
    return True


def backend_for(root: Union[str, Path]) -> LocalBackend:
    """The backend of a set root: a local directory."""
    if "://" in str(root):
        raise ValueError(f"A materialised set is written to a local directory; got {root}.")
    return LocalBackend(Path(root).expanduser())
