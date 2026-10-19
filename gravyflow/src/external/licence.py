"""
The licence gate: what a release may publish.

Every remote file gravyflow knows carries its licence and whether gravyflow may
redistribute it (``RemoteFile``). Running a model locally needs neither: its weights
are fetched from their authors' source into the user's own store. Publishing a file
again (attaching weights to a release, or mirroring them) needs both, and every list
of files a release publishes passes through ``release_manifest``, which refuses any
file without a licence, or one its licence does not let gravyflow redistribute.
"""

from __future__ import annotations

from typing import Iterable, Tuple

from gravyflow.src.dataset.store.files import RemoteFile


class LicenceRefused(ValueError):
    """A file a release would publish has no licence, or may not be redistributed."""


def release_manifest(files: Iterable[RemoteFile]) -> Tuple[RemoteFile, ...]:
    """
    ``files`` as the manifest of a release that publishes them, once each, after the
    licence gate: ``LicenceRefused`` naming every file with no licence (a licence of
    None is never redistributable) or not marked redistributable. An external card's
    weights are ``card.files``.
    """
    files = tuple(dict.fromkeys(files))
    if not all(isinstance(f, RemoteFile) for f in files):
        raise TypeError("A release manifest lists RemoteFile entries.")
    # A RemoteFile with no licence cannot be marked redistributable, so the flag decides.
    refused = [f"{f.name} ({_reason(f)})" for f in files if not f.redistributable]
    if refused:
        raise LicenceRefused(
            "A release may publish only files whose licence lets gravyflow redistribute them; refused: "
            + "; ".join(refused) + ". Users fetch these from their authors' source instead.")
    return files


def _reason(file: RemoteFile) -> str:
    if file.licence is None:
        return "it has no licence"
    return f"its licence, {file.licence}, is not marked redistributable"


__all__ = ["LicenceRefused", "release_manifest"]
