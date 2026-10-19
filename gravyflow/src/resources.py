"""
The data files gravyflow ships, and the manifest of the files it does not.

``RESOURCES`` is the package's ``res`` directory: the noise profiles, the template
banks, the smaller trained models and MLGWSC-1's segment list are installed with
the package. ``MANIFEST`` is every file gravyflow knows that lives elsewhere, each a
``RemoteFile`` pinned by size and sha256, with its licence and whether gravyflow may
redistribute it: weights above the wheel's size budget, attached to a GitHub
release, and benchmark data published by its authors.

``path(name)`` finds a file by its name under ``res``: the package's own copy, else
the store's copy of the manifest entry. Nothing is downloaded implicitly; ``fetch``
is the one explicit download, into the store (``dataset.store.files``).
``weights(name)`` is ``path`` for a trained model.
"""

from pathlib import Path
from typing import Dict, Union

from gravyflow.src.dataset.store.files import GRAVYFLOW_RELEASES, RemoteFile, cached
from gravyflow.src.dataset.store import files as _files

PACKAGE = Path(__file__).resolve().parents[1]
RESOURCES = PACKAGE / "res"

#: gravyflow's release that publishes the weights too large to ship.
_RELEASE = GRAVYFLOW_RELEASES + "download/v1.0.0/"

#: Every file gravyflow knows that is not (or not always) installed with it, by its
#: name under ``res``.
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
    # MLGWSC-1's segment list: published only on the authors' server (the file the
    # challenge's generate_data.py downloads), so pinned by content, not by commit.
    # Shipped too (res/mlgwsc1, with its NOTICE), so the reservation holds offline.
    RemoteFile(
        "mlgwsc1/segments.csv",
        "https://www.atlas.aei.uni-hannover.de/work/marlin.schaefer/MDC/segments.csv",
        "f8da2e50914fa6c84ca79f1e8449211bc5596fd610e80084d1daf8a5988272c8", 9313,
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
    explicitly, and return its path: the one download gravyflow makes. It is checked
    against its pinned size and sha256 and kept only if both match.
    """
    if isinstance(file, str):
        if file not in MANIFEST:
            raise ValueError(f"{file!r} is not in gravyflow's manifest; known: {sorted(MANIFEST)}.")
        file = MANIFEST[file]
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
