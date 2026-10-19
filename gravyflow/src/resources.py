"""
The data files gravyflow ships, and the manifest of the files it does not.

``RESOURCES`` is the package's ``res`` directory: the noise profiles, the template
banks and the smaller trained models are installed with the package. ``MANIFEST``
holds the pinned files gravyflow knows that live elsewhere, each a ``RemoteFile``
pinned by size and sha256, with its licence and whether gravyflow may redistribute
it: weights above the wheel's size budget, attached to a GitHub release. GWOSC
strain and Gravity Spy's catalogues are not pinned files: their acquisition modules
fetch them, by their own paths.

``path(name)`` finds a file by its name under ``res``: the package's own copy, else
the store's copy of the manifest entry. A pinned file is never downloaded
implicitly; ``fetch`` is its one explicit download, into the store
(``dataset.store.files``). ``weights(name)`` is ``path`` for a trained model.
"""

from pathlib import Path
from typing import Dict, Union

from gravyflow.src.dataset.store.files import GRAVYFLOW_RELEASES, RemoteFile, cached
from gravyflow.src.dataset.store import files as _files

PACKAGE = Path(__file__).resolve().parents[1]
RESOURCES = PACKAGE / "res"

#: gravyflow's release that publishes the weights too large to ship.
_RELEASE = GRAVYFLOW_RELEASES + "download/v1.0.0/"

#: The pinned files gravyflow knows that are not (or not always) installed with it,
#: by their names under ``res``.
MANIFEST: Dict[str, RemoteFile] = {entry.name: entry for entry in (
    RemoteFile(
        "model/georgehuerta2017deep_classification.keras",
        _RELEASE + "georgehuerta2017deep_classification.keras",
        "d0e55edfdb8e96c74010ac174a9ba46fefe10973853be7115e329865eed46370", 69415157,
        licence="Apache-2.0", redistributable=True),
    RemoteFile(
        "model/georgehuerta2017deep_regression.keras",
        _RELEASE + "georgehuerta2017deep_regression.keras",
        "429bf415bcf41126f71b39928de2529b989c0345cc34e4d70c2c2429b15ba3f2", 69415130,
        licence="Apache-2.0", redistributable=True),
)}


def path(name: str) -> Path:
    """
    The file ``name`` (a path under ``res``): the package's own copy, else the
    store's copy of its manifest entry, which ``fetch`` makes; ``RemoteFileMissing``
    (a FileNotFoundError) naming that fetch otherwise. FileNotFoundError for a name
    gravyflow does not know.
    """
    bundled = RESOURCES / name
    if bundled.is_file():
        return bundled
    if name in MANIFEST:
        return cached(MANIFEST[name])
    raise FileNotFoundError(f"gravyflow has no file {name!r}: not in {RESOURCES}, nor in its manifest.")


def fetch(file: Union[str, RemoteFile]) -> Path:
    """
    Download ``file`` (a manifest name, or a ``RemoteFile``) into the store, once,
    explicitly, and return its path: the one download of a pinned file. It is checked
    against its pinned size and sha256 and kept only if both match. A manifest file
    the package ships is not downloaded: its shipped copy is returned.
    """
    if isinstance(file, str):
        if file not in MANIFEST:
            raise ValueError(f"{file!r} is not in gravyflow's manifest; known: {sorted(MANIFEST)}.")
        file = MANIFEST[file]
    shipped = RESOURCES / file.name
    if MANIFEST.get(file.name) == file and shipped.is_file():
        return shipped
    return _files.fetch(file)


def weights(name: str) -> Path:
    """
    The trained weights file ``name``: the package's own copy, else a fetched
    release download (``path(f"model/{name}")``).
    """
    try:
        return path(f"model/{name}")
    except _files.RemoteFileMissing:
        raise
    except FileNotFoundError:
        raise FileNotFoundError(f"gravyflow has no trained weights named {name!r}.") from None
