"""
The bytes of a materialised set: its format version, its refusals, and the
canonical encodings of its manifest, shards and batches.

Every file of a set is named by, or checked against, the sha256 of its bytes, so
each encoding here is deterministic: the same content always gives the same
bytes, on every machine and in every run.
"""

import hashlib
import io
import json
import zipfile
from typing import Dict, Mapping

import numpy as np

#: The version of the layout and encodings below. A reader refuses any other.
FORMAT = 1

#: Batches per shard, by default: a shard is committed once all its batches are.
DEFAULT_SHARD_SIZE = 16

# A zip entry's timestamp is part of its bytes: one fixed time keeps a batch's bytes its content's.
_ZIP_TIME = (1980, 1, 1, 0, 0, 0)


class MaterialiseError(RuntimeError):
    """A materialised set cannot serve, or be written, as asked."""


class MaterialiseConflict(MaterialiseError):
    """A file of the set exists with other bytes than a writer's: files are never replaced."""


class IntegrityError(MaterialiseError):
    """A file of the set does not hold what its name or its listing says."""


class MissingShard(MaterialiseError):
    """A shard of the set was never completed: its batches are missing."""


class UnknownFormat(MaterialiseError):
    """The set was written in a format this gravyflow does not read."""


class FingerprintMismatch(MaterialiseError):
    """The set holds another dataset specification's batches."""


class ContractMismatch(MaterialiseError):
    """The set's batches have other names, shapes or dtypes than the reading dataset's."""


class SplitMismatch(MaterialiseError):
    """The set holds another split's batches."""


class ForeignPlatform(MaterialiseError):
    """Part of the set was written on another platform than the rest."""


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_bytes(value) -> bytes:
    """The canonical JSON of ``value``: sorted keys, no spaces, one trailing newline."""
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def encode_batch(arrays: Mapping[str, np.ndarray]) -> bytes:
    """
    An uncompressed ``.npz`` of ``arrays`` whose bytes depend only on their names
    and contents (``np.savez`` stamps each entry with the time it was written).
    """
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
        for name in sorted(arrays):
            info = zipfile.ZipInfo(f"{name}.npy", date_time=_ZIP_TIME)
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o644 << 16
            with archive.open(info, "w", force_zip64=True) as entry:
                np.lib.format.write_array(entry, np.asarray(arrays[name]), allow_pickle=False)
    return buffer.getvalue()


def decode_batch(data: bytes) -> Dict[str, np.ndarray]:
    """The arrays of an encoded batch, read without unpickling anything."""
    with np.load(io.BytesIO(data), allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}
