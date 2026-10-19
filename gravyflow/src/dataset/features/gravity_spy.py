"""
The public Gravity Spy glitch catalogue.

Gravyflow's glitch lists come from the Gravity Spy machine-learning classifications of
LIGO glitches in O1, O2, O3a and O3b, released on Zenodo:

    J. Glanzer et al., "Data quality up to the third observing run of Advanced LIGO:
    Gravity Spy glitch classifications", Class. Quantum Grav. 40 065004 (2023),
    doi:10.1088/1361-6382/acb633.

    Gravity Spy Machine Learning Classifications of LIGO Glitches from Observing Runs
    O1, O2, O3a, and O3b, v1.0.0, Zenodo record 5649212, doi:10.5281/zenodo.5649212.
    Licence: CC-BY-4.0.

Please cite both when you use glitches from gravyflow.

The release has one CSV per detector and run (16 to 216 MB). Each is downloaded on first
use, checked against the md5 the record publishes, and reduced to a compact ``.npz`` of
the columns gravyflow reads, in the user cache (``~/.gravyflow/cache/gravityspy``). The
raw CSV is then deleted. Nothing is written inside the installed package.

Columns kept: ``event_time`` (peak GPS time), ``start_time`` (Omicron start, GPS),
``duration`` (s), ``peak_frequency`` (Hz), ``snr``, ``ml_confidence`` and ``ml_label``.

Label sets differ by era: O1 and O2 use 22 classes; O3 adds ``Blip_Low_Frequency`` and
``Fast_Scattering``. Labels are kept as published; ``GlitchType`` maps each one.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from gravyflow.src.dataset.acquisition import gwosc_data

logger = logging.getLogger(__name__)


class NoGlitchVetoWarning(UserWarning):
    """
    Noise is served with no glitch veto: no public Gravity Spy labels cover its detector
    and time (the release covers H1 and L1 in O1, O2, O3a and O3b; not O4, not Virgo).
    Its glitches stay in. Filter it by name to silence it, for example
    ``warnings.simplefilter("ignore", gf.NoGlitchVetoWarning)``.
    """


class PartialGlitchCoverageWarning(UserWarning):
    """
    Glitches were requested over several runs and some have no public Gravity Spy labels
    for a detector: the glitches of the covered runs are served, none from the others.
    Filter it by name to silence it, for example
    ``warnings.simplefilter("ignore", gf.PartialGlitchCoverageWarning)``.
    """


ZENODO_RECORD = 5649212
ZENODO_DOI = "10.5281/zenodo.5649212"
RELEASE_VERSION = "v1.0.0"
RELEASE_LICENCE = "CC-BY-4.0"
RELEASE_CITATION = (
    "Glanzer et al. 2023, Class. Quantum Grav. 40 065004; Gravity Spy ML classifications "
    "O1-O3b, Zenodo doi:10.5281/zenodo.5649212 (v1.0.0, CC-BY-4.0)"
)

#: (IFO, era) -> (file name, md5, size in bytes), from the record's file API.
RELEASE_FILES: Dict[Tuple[str, str], Tuple[str, str, int]] = {
    ("H1", "O1"): ("H1_O1.csv", "91963313b1574e083bc58915e0aa8ca1", 15963563),
    ("L1", "O1"): ("L1_O1.csv", "34c7d08070f302d3b366767610748f40", 27212877),
    ("H1", "O2"): ("H1_O2.csv", "1c3eb9432107c6fee671cc76ad632e69", 73098153),
    ("L1", "O2"): ("L1_O2.csv", "df0a831cb79509e2aae38a4651c796fc", 65785621),
    ("H1", "O3a"): ("H1_O3a.csv", "29aea278b622cd97496971f7c07f7d6a", 90238691),
    ("L1", "O3a"): ("L1_O3a.csv", "3d3d0409817499c84fb2e0a2b211b5a5", 142258875),
    ("H1", "O3b"): ("H1_O3b.csv", "590290fe3c7ddf8cd9fa85c3a09cd5e4", 111575139),
    ("L1", "O3b"): ("L1_O3b.csv", "90d0bc16448bca72818ed9abf267979c", 216062292),
}

#: The release's eras, by GWOSC run name.
ERAS: Tuple[str, ...] = ("O1", "O2", "O3a", "O3b")

#: Default classification-confidence floor.
DEFAULT_MIN_ML_CONFIDENCE = 0.9

#: Version of the compact cache format; bump it to rebuild every cached catalogue.
_COMPACT_FORMAT = 1

_COLUMNS = ("event_time", "start_time", "start_time_ns", "duration", "peak_frequency",
            "snr", "ml_confidence", "ml_label")

_memory_cache: Dict[Path, Dict[str, np.ndarray]] = {}


def clear_memory_cache() -> None:
    """Forget catalogues loaded in this process (the disk cache is kept)."""
    _memory_cache.clear()


def cache_dir() -> Path:
    """Where compact catalogues are kept: ``~/.gravyflow/cache/gravityspy``."""
    return gwosc_data.user_cache_dir("gravityspy")


def covers(ifo_name: str, start_gps_time: float, end_gps_time: float) -> bool:
    """Whether the release has ``ifo_name``'s glitches in any era overlapping [start, end]."""
    return any((ifo_name, era) in RELEASE_FILES for era in eras_overlapping(start_gps_time, end_gps_time))


def eras_overlapping(start_gps_time: float, end_gps_time: float) -> List[str]:
    """Release eras whose GWOSC run bounds overlap [start, end]."""
    eras = []
    for era in ERAS:
        run = gwosc_data.GWOSC_RUNS[era]
        if run["gps_start"] <= end_gps_time and run["gps_end"] >= start_gps_time:
            eras.append(era)
    return eras


def _download(ifo_name: str, era: str, destination: Path) -> None:
    """Download one release file and verify its size and md5, or raise IOError."""
    import requests

    file_name, md5, size = RELEASE_FILES[(ifo_name, era)]
    url = f"https://zenodo.org/api/records/{ZENODO_RECORD}/files/{file_name}/content"
    logger.info(f"Downloading Gravity Spy {file_name} ({size / 1e6:.0f} MB) from Zenodo")
    digest = hashlib.md5()
    written = 0
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with open(destination, "wb") as handle:
            for chunk in response.iter_content(chunk_size=1 << 20):
                handle.write(chunk)
                digest.update(chunk)
                written += len(chunk)
    if written != size or digest.hexdigest() != md5:
        destination.unlink(missing_ok=True)
        raise IOError(
            f"Gravity Spy {file_name} failed verification: got {written} bytes, "
            f"md5 {digest.hexdigest()}; the release lists {size} bytes, md5 {md5}."
        )


def build_compact_catalogue(csv_path: Path, out_path: Path, source: Dict) -> Dict[str, np.ndarray]:
    """
    Reduce one release CSV to the compact form and write it atomically to ``out_path``.

    Returns the compact arrays.
    """
    import pandas as pd

    frame = pd.read_csv(csv_path, usecols=list(_COLUMNS))
    labels, codes = np.unique(frame["ml_label"].astype(str).to_numpy(), return_inverse=True)
    arrays = {
        "event_time": frame["event_time"].to_numpy(np.float64),
        "start_time": frame["start_time"].to_numpy(np.float64)
        + frame["start_time_ns"].to_numpy(np.float64) * 1e-9,
        "duration": frame["duration"].to_numpy(np.float64),
        "peak_frequency": frame["peak_frequency"].to_numpy(np.float64),
        "snr": frame["snr"].to_numpy(np.float64),
        "ml_confidence": frame["ml_confidence"].to_numpy(np.float64),
        "label_code": codes.astype(np.int16),
        "labels": labels.astype(str),
    }
    meta = dict(source, format=_COMPACT_FORMAT, rows=int(len(frame)))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=out_path.parent, prefix=f".{out_path.stem}.", suffix=".npz")
    os.close(fd)
    try:
        np.savez_compressed(tmp, meta=np.array(json.dumps(meta)), **arrays)
        os.chmod(tmp, 0o644)
        os.replace(tmp, out_path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    return arrays


def _expected_source(ifo_name: str, era: str) -> Dict:
    file_name, md5, size = RELEASE_FILES[(ifo_name, era)]
    return {"doi": ZENODO_DOI, "version": RELEASE_VERSION, "file": file_name, "md5": md5}


def _read_compact(path: Path, source: Dict) -> Optional[Dict[str, np.ndarray]]:
    try:
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["meta"]))
            if meta.get("format") != _COMPACT_FORMAT or any(
                meta.get(key) != value for key, value in source.items()
            ):
                return None
            return {key: data[key] for key in data.files if key != "meta"}
    except (OSError, ValueError, KeyError) as error:
        logger.warning(f"Ignoring unreadable Gravity Spy cache {path}: {error}")
        return None


def load_catalogue(ifo_name: str, era: str) -> Dict[str, np.ndarray]:
    """
    The compact catalogue for one detector and era, downloading it on first use.

    Returns arrays ``event_time``, ``start_time``, ``duration``, ``peak_frequency``,
    ``snr``, ``ml_confidence``, ``label_code`` (index into ``labels``) and ``labels``.
    """
    if (ifo_name, era) not in RELEASE_FILES:
        raise ValueError(
            f"No public Gravity Spy release for {ifo_name} {era}: the release "
            f"({ZENODO_DOI}) covers H1 and L1 in {', '.join(ERAS)}."
        )
    directory = cache_dir()
    path = directory / f"{ifo_name}_{era}.npz"
    if path in _memory_cache:
        return _memory_cache[path]

    source = _expected_source(ifo_name, era)
    arrays = _read_compact(path, source) if path.exists() else None
    if arrays is None:
        fd, raw_name = tempfile.mkstemp(dir=directory, prefix=f".{ifo_name}_{era}.", suffix=".csv")
        os.close(fd)
        raw = Path(raw_name)
        try:
            _download(ifo_name, era, raw)
            arrays = build_compact_catalogue(raw, path, source)
        finally:
            raw.unlink(missing_ok=True)
    _memory_cache[path] = arrays
    return arrays


def query(
    ifo_name: str,
    start_gps_time: float,
    end_gps_time: float,
    labels: Optional[Iterable[str]] = None,
    min_ml_confidence: float = DEFAULT_MIN_ML_CONFIDENCE,
    exclude_labels: Sequence[str] = (),
) -> Dict[str, np.ndarray]:
    """
    Glitches of one detector with ``start <= event_time <= end``.

    Args:
        ifo_name: "H1" or "L1".
        labels: Gravity Spy labels to keep; None keeps every label not excluded.
        min_ml_confidence: Keep rows with ``ml_confidence >= min_ml_confidence``.
        exclude_labels: Labels to drop when ``labels`` is None.

    Returns:
        Arrays ``event_time``, ``start_time``, ``duration``, ``peak_frequency``, ``snr``,
        ``ml_confidence`` and ``ml_label`` (strings), sorted by event time.
    """
    eras = eras_overlapping(start_gps_time, end_gps_time)
    if not eras:
        raise ValueError(
            f"No public Gravity Spy release covers GPS {start_gps_time}-{end_gps_time}: "
            f"the release ({ZENODO_DOI}) covers {', '.join(ERAS)}."
        )
    wanted = None if labels is None else set(labels)
    pieces = []
    for era in eras:
        catalogue = load_catalogue(ifo_name, era)
        names = catalogue["labels"][catalogue["label_code"]]
        keep = (
            (catalogue["event_time"] >= start_gps_time)
            & (catalogue["event_time"] <= end_gps_time)
            & (catalogue["ml_confidence"] >= min_ml_confidence)
        )
        if wanted is not None:
            keep &= np.isin(names, list(wanted))
        elif exclude_labels:
            keep &= ~np.isin(names, list(exclude_labels))
        piece = {key: catalogue[key][keep] for key in
                 ("event_time", "start_time", "duration", "peak_frequency", "snr", "ml_confidence")}
        piece["ml_label"] = names[keep]
        pieces.append(piece)

    result = {key: np.concatenate([piece[key] for piece in pieces]) for key in pieces[0]}
    order = np.argsort(result["event_time"], kind="stable")
    return {key: value[order] for key, value in result.items()}
