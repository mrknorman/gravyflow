"""
Public GWOSC sources for real-noise acquisition.

This module supplies the three public facts that real-noise acquisition needs:

- **Observing-run bounds** (``GWOSC_RUNS``): the GPS start and end of each run, as
  published by the GWOSC run API (https://gwosc.org/api/v2/runs). The values are a
  snapshot of that API, so the library imports offline; ``fetch_gwosc_runs`` queries
  the live API, and the network tests compare the two.
- **Data-quality segments** (``get_gwosc_segments``): the GWOSC timeline segments for a
  detector, intersected over a set of flags. The GWOSC documentation defines every
  data-quality flag as the times that *pass* it ("When one of these data bits is set to
  1 at some time, it means the data are 'good' at that time: that it passes a test, or
  there are no hardware injections", https://gwosc.org/archive/dataset/O3a_4KHZ_R1/).
  Applying a category therefore means intersecting with its flag.
- **Strain sample rates** (``gwosc_sample_rate_hertz``): GWOSC publishes strain at
  4096 Hz and 16384 Hz only.

Timeline segments of a released run never change, so each list is cached in the
user cache (``~/.gravyflow/cache/gwosc_timeline``) after its first download.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

GWOSC_HOST = "https://gwosc.org"

#: GWOSC run bounds and the timeline dataset that holds each run's flags.
#: Source: https://gwosc.org/api/v2/runs and https://gwosc.org/api/v2/runs/{run}/datasets.
GWOSC_RUNS: Dict[str, Dict] = {
    "O1": {"gps_start": 1126051217, "gps_end": 1137254417, "dataset": "O1"},
    "O2": {"gps_start": 1164556817, "gps_end": 1187733618, "dataset": "O2_4KHZ_R1"},
    "O3a": {"gps_start": 1238166018, "gps_end": 1253977218, "dataset": "O3a_4KHZ_R1"},
    "O3b": {"gps_start": 1256655618, "gps_end": 1269363618, "dataset": "O3b_4KHZ_R1"},
    "O4a": {"gps_start": 1368195220, "gps_end": 1389456018, "dataset": "O4a_4KHZ_R1"},
    "O4b": {"gps_start": 1396417050, "gps_end": 1422118818, "dataset": "O4b_4KHZ_R1"},
}

#: Data present only.
GWOSC_DATA_FLAGS: Tuple[str, ...] = ("DATA",)
#: Data present, passing the burst category 1 and 2 vetoes.
GWOSC_BURST_FLAGS: Tuple[str, ...] = ("DATA", "BURST_CAT1", "BURST_CAT2")
#: Data present, passing the CBC category 1 and 2 vetoes.
GWOSC_CBC_FLAGS: Tuple[str, ...] = ("DATA", "CBC_CAT1", "CBC_CAT2")
#: Times with no CBC, burst, stochastic or detector-characterisation hardware injection.
#: GWOSC names these flags for the times that pass ("H1: no cbc injection", "H1: no
#: stoch injections"), so they are intersected like the others. The stochastic
#: injections are coherent between H1 and L1. ``NO_CW_HW_INJ`` is left out: the
#: continuous-wave injections run through nearly all of every run, so excluding them
#: would leave almost no data.
GWOSC_NO_HW_INJECTION_FLAGS: Tuple[str, ...] = (
    "NO_CBC_HW_INJ", "NO_BURST_HW_INJ", "NO_STOCH_HW_INJ", "NO_DETCHAR_HW_INJ",
)
#: The default real-noise selection: the burst vetoes, without hardware injections.
DEFAULT_GWOSC_FLAGS: Tuple[str, ...] = GWOSC_BURST_FLAGS + GWOSC_NO_HW_INJECTION_FLAGS

#: Strain sample rates GWOSC publishes, in Hz.
GWOSC_SAMPLE_RATES_HERTZ: Tuple[float, ...] = (4096.0, 16384.0)

_TIMEOUT_SECONDS = 120


class UpsamplingRefused(ValueError):
    """A request needs real data at a higher sample rate than the source has."""


def user_cache_dir(*parts: str) -> Path:
    """
    The user-level cache directory, ``~/.gravyflow/cache``, or a subdirectory of it.

    It is the same root the transient cache uses. The environment variable
    ``GRAVYFLOW_CACHE_DIR`` overrides it. Nothing is ever cached inside the installed
    package.
    """
    override = os.environ.get("GRAVYFLOW_CACHE_DIR")
    path = Path(override) if override else Path.home() / ".gravyflow" / "cache"
    for part in parts:
        path = path / part
    path.mkdir(parents=True, exist_ok=True)
    return path


def gwosc_sample_rate_hertz(sample_rate_hertz: float) -> float:
    """
    The GWOSC strain sample rate to fetch for a requested sample rate.

    Returns 4096 Hz when the request is at most 4096 Hz, otherwise 16384 Hz. The data
    is then downsampled to the request. A request above 16384 Hz raises
    UpsamplingRefused (a ValueError): upsampling adds no information.
    """
    sample_rate_hertz = float(sample_rate_hertz)
    if sample_rate_hertz <= 0:
        raise ValueError(f"Sample rate must be positive, got {sample_rate_hertz} Hz.")
    for rate in GWOSC_SAMPLE_RATES_HERTZ:
        if sample_rate_hertz <= rate:
            return rate
    raise UpsamplingRefused(
        f"Requested sample rate {sample_rate_hertz} Hz exceeds the highest rate GWOSC "
        f"publishes ({GWOSC_SAMPLE_RATES_HERTZ[-1]:.0f} Hz). Gravyflow does not upsample "
        "real data."
    )


def _get_json(url: str) -> Dict:
    import requests

    response = requests.get(
        url,
        headers={"User-Agent": "gravyflow"},
        timeout=_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


def fetch_gwosc_runs(host: str = GWOSC_HOST) -> Dict[str, Tuple[int, int]]:
    """Query the live GWOSC run API: {run name: (gps_start, gps_end)}."""
    runs = {}
    url = f"{host}/api/v2/runs?format=json"
    while url:
        page = _get_json(url)
        for run in page["results"]:
            runs[run["name"]] = (int(run["gps_start"]), int(run["gps_end"]))
        url = page.get("next")
    return runs


def _timeline_cache_path(dataset: str, flag_name: str, start: int, end: int) -> Path:
    return user_cache_dir("gwosc_timeline") / f"{dataset}_{flag_name}_{start}_{end}.json"


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(text)
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def fetch_timeline_segments(
    dataset: str,
    flag_name: str,
    start: int,
    end: int,
    host: str = GWOSC_HOST,
    use_cache: bool = True,
) -> np.ndarray:
    """
    Fetch one GWOSC timeline flag, e.g. ``H1_BURST_CAT2``, as an (N, 2) array of
    [start, end) GPS segments.

    It uses the timeline JSON endpoint, which returns a whole run in one request. The
    paginated v2 API returns the same segments (checked for O3a H1_DATA) but needs one
    request per 20 segments; it is the fallback if the timeline endpoint fails.
    """
    start, end = int(np.floor(start)), int(np.ceil(end))
    cache_path = _timeline_cache_path(dataset, flag_name, start, end)
    if use_cache and cache_path.exists():
        try:
            return _as_segment_array(json.loads(cache_path.read_text())["segments"])
        except (ValueError, KeyError) as error:
            logger.warning(f"Ignoring unreadable timeline cache {cache_path}: {error}")

    url = f"{host}/timeline/segments/json/{dataset}/{flag_name}/{start}/{end - start}/"
    try:
        segments = _get_json(url)["segments"]
    except Exception as error:
        logger.warning(
            f"GWOSC timeline endpoint failed for {flag_name} ({error}); "
            "falling back to the paginated v2 API, which is slow for veto flags."
        )
        from gwosc.timeline import get_segments

        segments = [list(segment) for segment in get_segments(flag_name, start, end, host=host)]

    if use_cache:
        _atomic_write_text(
            cache_path,
            json.dumps({"dataset": dataset, "flag": flag_name, "start": start,
                        "end": end, "segments": segments}),
        )
    return _as_segment_array(segments)


def _as_segment_array(segments: Iterable) -> np.ndarray:
    array = np.asarray(list(segments), dtype=np.float64)
    return array.reshape(-1, 2)


def coalesce_segments(segments: np.ndarray) -> np.ndarray:
    """Sort segments, drop any of zero length and merge any that overlap or touch."""
    segments = _as_segment_array(segments)
    segments = segments[segments[:, 1] > segments[:, 0]]
    if len(segments) == 0:
        return segments
    segments = segments[np.argsort(segments[:, 0], kind="stable")]
    merged = [segments[0].copy()]
    for start, end in segments[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append(np.array([start, end]))
    return np.asarray(merged, dtype=np.float64)


def intersect_segments(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Intersection of two segment lists, coalesced."""
    first, second = coalesce_segments(first), coalesce_segments(second)
    i = j = 0
    out = []
    while i < len(first) and j < len(second):
        start = max(first[i, 0], second[j, 0])
        end = min(first[i, 1], second[j, 1])
        if start < end:
            out.append((start, end))
        if first[i, 1] < second[j, 1]:
            i += 1
        else:
            j += 1
    return _as_segment_array(out)


def subtract_segments(segments: np.ndarray, vetoes: np.ndarray) -> np.ndarray:
    """
    The times in ``segments`` that lie in no veto, as a sorted, coalesced (N, 2) array.

    Both lists are sorted and coalesced first, so they may arrive in any order and may
    overlap. Pieces of zero length are dropped.
    """
    segments, vetoes = coalesce_segments(segments), coalesce_segments(vetoes)
    if len(segments) == 0 or len(vetoes) == 0:
        return segments
    out = []
    j = 0
    for start, end in segments:
        # Skip vetoes that end before this segment starts; they end before every
        # later segment too, because both lists are sorted and coalesced.
        while j < len(vetoes) and vetoes[j, 1] <= start:
            j += 1
        current = start
        k = j
        while k < len(vetoes) and vetoes[k, 0] < end:
            if vetoes[k, 0] > current:
                out.append((current, vetoes[k, 0]))
            current = max(current, vetoes[k, 1])
            if current >= end:
                break
            k += 1
        if current < end:
            out.append((current, end))
    return _as_segment_array(out)


def get_gwosc_segments(
    ifo_name: str,
    gwosc_runs: Sequence[str],
    flags: Sequence[str] = DEFAULT_GWOSC_FLAGS,
    start: Optional[float] = None,
    end: Optional[float] = None,
    host: str = GWOSC_HOST,
) -> np.ndarray:
    """
    Times when a detector passes every flag in ``flags``, over the given GWOSC runs.

    Args:
        ifo_name: Detector name, e.g. "H1".
        gwosc_runs: GWOSC run names, e.g. ("O3a", "O3b").
        flags: GWOSC flag names without the detector prefix. Each flag lists the times
            that pass it, so the result is their intersection.
        start, end: Optional GPS bounds to clip the result to.

    Returns:
        A sorted, coalesced (N, 2) float64 array of [start, end) GPS segments.
    """
    flags = tuple(flags)
    if not flags:
        raise ValueError("At least one GWOSC flag is required (e.g. 'DATA').")

    per_run = []
    for run_name in gwosc_runs:
        if run_name not in GWOSC_RUNS:
            raise ValueError(
                f"Unknown GWOSC run {run_name!r}; known runs: {sorted(GWOSC_RUNS)}."
            )
        run = GWOSC_RUNS[run_name]
        segments = None
        for flag in flags:
            flag_segments = fetch_timeline_segments(
                run["dataset"], f"{ifo_name}_{flag}", run["gps_start"], run["gps_end"], host=host
            )
            segments = (
                coalesce_segments(flag_segments)
                if segments is None
                else intersect_segments(segments, flag_segments)
            )
        per_run.append(segments)

    segments = coalesce_segments(np.concatenate(per_run)) if per_run else _as_segment_array([])
    if start is not None or end is not None:
        bound = np.array([[
            -np.inf if start is None else float(start),
            np.inf if end is None else float(end),
        ]])
        segments = intersect_segments(segments, bound)
    return segments
