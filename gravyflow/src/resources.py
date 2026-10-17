"""
The data files gravyflow ships, and the trained weights too large to ship.

``RESOURCES`` is the package's ``res`` directory: the noise profiles, the
template banks and the smaller trained models are installed with the package.
Weights above the wheel's size budget are attached to the GitHub release of
each version instead; ``weights(name)`` finds a model in either place, and
``download_weights(name)`` fetches a release file into the user cache and
checks it against the SHA-256 recorded here.
"""

import hashlib
import shutil
import tempfile
import urllib.request
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]
RESOURCES = PACKAGE / "res"

#: Where a version's release weights are published.
RELEASE_URL = "https://github.com/mrknorman/gravyflow/releases/download/v{version}/{name}"

#: The trained weights each release publishes rather than ships, by file name: their SHA-256.
RELEASE_WEIGHTS = {
    "georgehuerta2017deep_classification.keras": "d0e55edfdb8e96c74010ac174a9ba46fefe10973853be7115e329865eed46370",
    "georgehuerta2017deep_regression.keras": "429bf415bcf41126f71b39928de2529b989c0345cc34e4d70c2c2429b15ba3f2",
}


def _cached(name: str) -> Path:
    from gravyflow.src.dataset.acquisition.gwosc_data import user_cache_dir
    return user_cache_dir("weights") / name


def weights(name: str) -> Path:
    """
    The trained weights file ``name``: the package's own copy, else a release
    download already in the user cache. Raises FileNotFoundError otherwise.
    """
    bundled = RESOURCES / "model" / name
    if bundled.is_file():
        return bundled
    if name in RELEASE_WEIGHTS:
        cached = _cached(name)
        if cached.is_file():
            return cached
        raise FileNotFoundError(
            f"The weights {name} are a release download, not part of the installed package. "
            f"Fetch them once with gravyflow.src.resources.download_weights({name!r}), which saves "
            f"them to {cached}."
        )
    raise FileNotFoundError(f"gravyflow has no trained weights named {name!r}.")


def download_weights(name: str) -> Path:
    """
    Fetch the release weights ``name`` for this version into the user cache and
    return their path. Refuses, keeping nothing, a file whose SHA-256 differs
    from the one recorded for it.
    """
    from gravyflow import __version__
    if name not in RELEASE_WEIGHTS:
        raise ValueError(f"{name!r} is not a release download; known: {sorted(RELEASE_WEIGHTS)}.")
    target = _cached(name)
    url = RELEASE_URL.format(version=__version__, name=name)
    digest = hashlib.sha256()
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as part, urllib.request.urlopen(url) as response:
        for block in iter(lambda: response.read(1 << 20), b""):
            digest.update(block)
            part.write(block)
    if digest.hexdigest() != RELEASE_WEIGHTS[name]:
        Path(part.name).unlink()
        raise ValueError(
            f"The download of {name} from {url} has SHA-256 {digest.hexdigest()}, "
            f"not the recorded {RELEASE_WEIGHTS[name]}; nothing was kept."
        )
    shutil.move(part.name, target)
    return target
