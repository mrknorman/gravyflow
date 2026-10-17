"""
A materialised set: one dataset's batches for one split, epoch and range of
batch indices, written once and read back verified.

Layout under a root directory, for a set whose id is the sha256 of its
manifest's bytes:

- ``<set_id>/manifest.json``, written before any batch: the format, the
  dataset's fingerprint (with its inputs: the canonical spec, the seed root, the
  split spec's fingerprint, the seed tree's and data's versions, the library
  versions), the platform record (`gravyflow.src.rng.platform.platform_record`),
  the stage, the batch contract (names, shapes, dtypes), the shard size S, the
  split, the effective epoch and the k range. Its id names everything that
  decides its batches, so two specs, platforms, epochs or ranges never share one.
- ``<set_id>/batches/<sha256>.npz``: one batch, uncompressed, named by the
  sha256 of its bytes. Besides its arrays (``inputs/<name>``,
  ``outputs/<name>``) it holds its own address (``address/set_id``,
  ``address/k``), so a listing that names the wrong file is caught.
- ``<set_id>/shards/<s>.json``, written once all its batches are (k in
  [sS, (s+1)S), within the range): each k's sha256, the set's id and the
  platform record of the writer.

Every file is committed create-if-absent (`backend.LocalBackend.create`), and
every read is verified: a batch's sha256 must equal its name and its shard's
entry, and hold its address; a shard must belong to the set and its platform;
the manifest must hash to the set's id and match the reading dataset.
"""

import json
from pathlib import Path
from typing import Callable, Dict, List, Mapping, Tuple, Union

import numpy as np

from gravyflow.src.dataset.materialise.backend import LocalBackend, backend_for
from gravyflow.src.dataset.materialise.format import (
    FORMAT, ContractMismatch, FingerprintMismatch, ForeignPlatform, IntegrityError, MissingShard, SplitMismatch,
    UnknownFormat, decode_batch, encode_batch, json_bytes, sha256,
)

Batch = Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]

_SIDES = ("inputs", "outputs")


class MaterialisedSet:
    """
    A set's directory, opened: its manifest read, its format known and its id
    verified. Reading a batch verifies it (see the module docstring).

    Raises:
        FileNotFoundError: no manifest at ``path``.
        UnknownFormat: a manifest of another format.
        IntegrityError: a manifest whose sha256 is not the set's id.
    """

    def __init__(self, path: Union[str, Path]):
        path = Path(path).expanduser()
        self.path, self.set_id = path, path.name
        self._backend = LocalBackend(path.parent)
        data = self._backend.read(self._name("manifest.json"))
        if data is None:
            raise FileNotFoundError(f"No materialised set at {path}: it has no manifest.json.")
        manifest = _json(data, path / "manifest.json")
        if manifest.get("format") != FORMAT:
            raise UnknownFormat(
                f"{path} is a materialised set of format {manifest.get('format')}; this gravyflow reads "
                f"format {FORMAT}.")
        if sha256(data) != self.set_id:
            raise IntegrityError(
                f"{path}/manifest.json has sha256 {sha256(data)}, not its set's id {self.set_id}: the "
                "manifest was edited or the set renamed.")
        self.manifest = manifest
        self.k_range = range(*manifest["k_range"])
        self.epoch = manifest["epoch"]
        self.shard_size = manifest["shard_size"]

    def __repr__(self):
        return f"MaterialisedSet({str(self.path)!r})"

    def _name(self, *parts: str) -> str:
        return "/".join((self.set_id, *parts))

    # ------------------------------------------------------------ shards

    def shards(self) -> List[Tuple[int, range]]:
        """Each shard's index and the k it holds, in order."""
        first, last = self.k_range.start // self.shard_size, (self.k_range.stop - 1) // self.shard_size
        return [(s, self._shard_ks(s)) for s in range(first, last + 1)]

    def _shard_ks(self, s: int) -> range:
        size = self.shard_size
        return range(max(s * size, self.k_range.start), min((s + 1) * size, self.k_range.stop))

    def missing(self) -> List[range]:
        """The k of each shard not yet committed."""
        return [ks for s, ks in self.shards() if not self._backend.exists(self._name("shards", f"{s}.json"))]

    def shard(self, s: int) -> Dict[int, str]:
        """
        Shard ``s``'s sha256 for each of its k, verified to belong to this set
        and its platform.

        Raises:
            MissingShard, ForeignPlatform, IntegrityError.
        """
        ks = self._shard_ks(s)
        name = self._name("shards", f"{s}.json")
        data = self._backend.read(name)
        if data is None:
            raise MissingShard(
                f"{self.path} has no shard {s}: k in [{ks.start}, {ks.stop}) were never completed. "
                "Run the dataset's materialise again with the same arguments: it resumes the missing shards.")
        shard = _json(data, self._backend.path(name))
        if shard.get("platform") != self.manifest["platform"]:
            raise ForeignPlatform(
                f"{self._backend.path(name)} was written on {shard.get('platform')}, the set on "
                f"{self.manifest['platform']}: a set's batches come from one platform.")
        entries = shard.get("batches", [])
        if shard.get("set_id") != self.set_id or [k for k, _ in entries] != list(ks):
            raise IntegrityError(
                f"{self._backend.path(name)} is not shard {s} of this set: it names set "
                f"{shard.get('set_id')} and k {[k for k, _ in entries]}, not {list(ks)}.")
        return dict(entries)

    def require_complete(self) -> None:
        """Verify every shard. Raises: MissingShard (the first missing), ForeignPlatform, IntegrityError."""
        for s, _ in self.shards():
            self.shard(s)

    # ------------------------------------------------------------ batches

    def serves(self, epoch: int, k: int) -> bool:
        """Whether batch ``k`` of ``epoch`` is in this set."""
        return epoch == self.epoch and k in self.k_range

    def read(self, k: int) -> Batch:
        """
        Batch ``k``: its inputs and outputs, verified against its name, its
        shard's entry, its address and the contract's names.

        Raises:
            MissingShard, ForeignPlatform, IntegrityError, ContractMismatch.
        """
        if k not in self.k_range:
            raise IndexError(f"Batch {k} is outside {self.path}'s k in [{self.k_range.start}, {self.k_range.stop}).")
        s = k // self.shard_size
        digest = self.shard(s)[k]
        name = self._name("batches", f"{digest}.npz")
        data = self._backend.read(name)
        if data is None:
            raise IntegrityError(f"Shard {s} of {self.path} lists {name} for batch {k}, which does not exist.")
        if sha256(data) != digest:
            raise IntegrityError(
                f"{self._backend.path(name)}, batch {k}, has sha256 {sha256(data)}, not its name: it was "
                "truncated or replaced.")
        arrays = decode_batch(data)
        address = (str(arrays.pop("address/set_id", None)), int(arrays.pop("address/k", -1)))
        if address != (self.set_id, k):
            raise IntegrityError(
                f"Shard {s} of {self.path} lists {name} for batch {k}, but it holds batch {address[1]} of "
                f"set {address[0]}.")
        expected = {f"{side}/{entry[0]}" for side in _SIDES for entry in self.manifest["contract"][side]}
        if set(arrays) != expected:
            raise ContractMismatch(
                f"Batch {k} of {self.path} holds {sorted(arrays)}; its contract names {sorted(expected)}.")
        batch = tuple({key.split("/", 1)[1]: value for key, value in arrays.items() if key.startswith(f"{side}/")}
                      for side in _SIDES)
        return batch

    # ------------------------------------------------------------ readers

    def check_reader(self, identity: Mapping) -> None:
        """
        Refuse a reading dataset whose ``identity`` (the manifest's split, fingerprint
        and contract, as the dataset would write them) is not this set's.

        Raises:
            SplitMismatch, FingerprintMismatch, ContractMismatch.
        """
        mine, theirs = self.manifest, identity
        split_fingerprint = (mine["fingerprint"]["split_fingerprint"], theirs["fingerprint"]["split_fingerprint"])
        if mine["split"] != theirs["split"] or split_fingerprint[0] != split_fingerprint[1]:
            raise SplitMismatch(
                f"{self.path} holds the {mine['split']!r} split (SplitSpec {split_fingerprint[0]}); the dataset "
                f"reading it serves {theirs['split']!r} (SplitSpec {split_fingerprint[1]}).")
        if mine["fingerprint"]["sha256"] != theirs["fingerprint"]["sha256"]:
            raise FingerprintMismatch(
                f"{self.path} holds the batches of dataset fingerprint {mine['fingerprint']['sha256']}; the "
                f"dataset reading it has fingerprint {theirs['fingerprint']['sha256']}: another spec, seed root "
                "or library version.")
        if mine["contract"] != theirs["contract"]:
            raise ContractMismatch(
                f"{self.path}'s batch contract is {mine['contract']}; the dataset reading it has "
                f"{theirs['contract']}.")


def write_set(root: Union[str, Path], manifest: Mapping, compute: Callable[[int], Tuple[Mapping, Mapping]]
              ) -> MaterialisedSet:
    """
    Write the set ``manifest`` describes under ``root``, computing batch k with
    ``compute(k)``, and return it complete. Shards already committed are
    verified and kept, so an interrupted run resumes where it stopped; a
    complete set is returned without computing anything.

    Raises:
        MaterialiseConflict: a file exists with other bytes than this writer's.
        MissingShard, ForeignPlatform, IntegrityError: a committed file is not what it should be.
    """
    backend = backend_for(root)
    data = json_bytes(manifest)
    set_id = sha256(data)
    backend.create(f"{set_id}/manifest.json", data)
    the_set = MaterialisedSet(backend.path(set_id))
    for s, ks in the_set.shards():
        shard_name = f"{set_id}/shards/{s}.json"
        if backend.exists(shard_name):
            the_set.shard(s)
            continue
        entries = []
        for k in ks:
            inputs, outputs = compute(k)
            arrays = {f"{side}/{name}": np.asarray(value)
                      for side, values in zip(_SIDES, (inputs, outputs)) for name, value in values.items()}
            batch = encode_batch({**arrays, "address/set_id": np.array(set_id), "address/k": np.array(k)})
            digest = sha256(batch)
            backend.create(f"{set_id}/batches/{digest}.npz", batch)
            entries.append([k, digest])
        backend.create(shard_name, json_bytes({
            "format": FORMAT, "set_id": set_id, "shard": s, "platform": manifest["platform"], "batches": entries}))
    the_set.require_complete()
    return the_set


def _json(data: bytes, where) -> dict:
    try:
        return json.loads(data)
    except ValueError as error:
        raise IntegrityError(f"{where} is not JSON: {error}.") from None
