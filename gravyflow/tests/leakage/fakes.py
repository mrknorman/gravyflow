"""
Fakes at the library's own seams: strain, GWOSC timelines and catalogues.

- **Strain.** ``BaseDataObtainer.get_segment_data_static`` is the one strain fetch
  that noise and transients both call. ``installed`` replaces it with
  ``StrainSource.fetch``, which serves the fixture's provenance code (see
  ``provenance``) at the requested rate and logs every request.
- **Segments.** Fake GWOSC timeline lists are written where ``gwosc_data`` caches
  them, under a temporary ``GRAVYFLOW_CACHE_DIR``, so the real intersect and clip
  code runs on them.
- **Catalogues.** Gravity Spy rows are written as release CSVs and reduced to the
  compact cache by the library's own builder; event tables replace the event
  producers, in their own modules and wherever a module bound them by name.

Everything is undone when ``installed`` exits. Batches must be drawn inside it,
eagerly: prefetch executors are shut down before the patches are removed, so no
background fetch can reach the real network afterwards.
"""

from __future__ import annotations

import contextlib
import csv
import gc
import io
import json
import os
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pytest

from .provenance import CODE_DETECTORS, Encoding, OutsideFixture

#: Every GWOSC timeline flag the library may ask for.
GWOSC_FLAGS: Tuple[str, ...] = (
    "DATA", "BURST_CAT1", "BURST_CAT2", "CBC_CAT1", "CBC_CAT2",
    "NO_CBC_HW_INJ", "NO_BURST_HW_INJ", "NO_STOCH_HW_INJ", "NO_DETCHAR_HW_INJ", "NO_CW_HW_INJ",
)
#: The flags whose gaps are hardware injections, excluding continuous-wave ones.
HARDWARE_INJECTION_FLAGS: Tuple[str, ...] = (
    "NO_CBC_HW_INJ", "NO_BURST_HW_INJ", "NO_STOCH_HW_INJ", "NO_DETCHAR_HW_INJ",
)


# =============================================================================
# STRAIN
# =============================================================================

@dataclass(frozen=True)
class Fetch:
    """One request to the strain seam and what it served (sample indices)."""

    ifo: str
    start: float
    end: float
    first: int
    num_samples: int
    thread: str

    @property
    def stop(self) -> int:
        return self.first + self.num_samples


class StrainSource:
    """Serves the provenance code at the seam, and logs every request and error."""

    def __init__(self, encoding: Encoding):
        self.encoding = encoding
        self.fetches: List[Fetch] = []
        self.errors: List[str] = []
        self._lock = threading.Lock()

    def fetch(
        self,
        segment_start_gps_time,
        segment_end_gps_time,
        ifo,
        frame_type=None,
        channel=None,
        sample_rate_hertz=None,
    ):
        from gwpy.timeseries import TimeSeries

        name = getattr(ifo, "name", str(ifo))
        rate = self.encoding.sample_rate_hertz
        try:
            if sample_rate_hertz is None or float(sample_rate_hertz) != rate:
                raise OutsideFixture(
                    f"{name} requested at {sample_rate_hertz} Hz; the fixture serves {rate:g} Hz."
                )
            first = int(np.ceil(np.float64(segment_start_gps_time) * rate))
            stop = int(np.ceil(np.float64(segment_end_gps_time) * rate))
            values = self.encoding.encode(name, first, stop - first)
        except Exception as error:
            with self._lock:
                self.errors.append(f"{name} [{segment_start_gps_time}, {segment_end_gps_time}): {error}")
            raise
        with self._lock:
            self.fetches.append(Fetch(
                name, float(segment_start_gps_time), float(segment_end_gps_time),
                first, stop - first, threading.current_thread().name,
            ))
        return TimeSeries(values, t0=first / rate, sample_rate=rate)

    def snapshot(self) -> Tuple[Tuple[Fetch, ...], Tuple[str, ...]]:
        with self._lock:
            return tuple(self.fetches), tuple(self.errors)

    def clear(self) -> None:
        with self._lock:
            self.fetches.clear()
            self.errors.clear()


def close_obtainers() -> int:
    """Shut down, waiting, every live obtainer's prefetch executor. Returns the count."""
    from gravyflow.src.dataset.acquisition.base import BaseDataObtainer

    closed = 0
    for obj in gc.get_objects():
        if isinstance(obj, BaseDataObtainer) and getattr(obj, "_prefetch_executor", None) is not None:
            obj.close()
            closed += 1
    return closed


# =============================================================================
# TIMELINES
# =============================================================================

def write_timelines(
    data_segments: Dict[str, np.ndarray],
    flag_gaps: Optional[Dict[Tuple[str, str], np.ndarray]] = None,
) -> None:
    """
    Write the GWOSC timeline of every run, detector and flag into the current
    ``GRAVYFLOW_CACHE_DIR``: DATA is ``data_segments[ifo]``; every other flag is DATA
    minus ``flag_gaps[(ifo, flag)]``.
    """
    from gwpy.segments import Segment, SegmentList
    from gravyflow.src.dataset.acquisition import gwosc_data

    flag_gaps = flag_gaps or {}
    for ifo in CODE_DETECTORS:
        data = SegmentList(Segment(*map(float, s)) for s in data_segments.get(ifo, ())).coalesce()
        for run in gwosc_data.GWOSC_RUNS.values():
            bound = SegmentList([Segment(float(run["gps_start"]), float(run["gps_end"]))])
            in_run = (data & bound).coalesce()
            for flag in GWOSC_FLAGS:
                gaps = SegmentList(Segment(*map(float, s)) for s in flag_gaps.get((ifo, flag), ()))
                passing = (in_run - gaps.coalesce()).coalesce() if flag != "DATA" else in_run
                path = gwosc_data._timeline_cache_path(
                    run["dataset"], f"{ifo}_{flag}", int(run["gps_start"]), int(run["gps_end"])
                )
                path.write_text(json.dumps({
                    "dataset": run["dataset"], "flag": f"{ifo}_{flag}",
                    "start": int(run["gps_start"]), "end": int(run["gps_end"]),
                    "segments": [[float(a), float(b)] for a, b in passing],
                }))


# =============================================================================
# CATALOGUES
# =============================================================================

@dataclass(frozen=True)
class GlitchRecord:
    """One Gravity Spy row: peak time, Omicron extent, class and confidence."""

    ifo: str
    peak: float
    start: float
    duration: float
    label: str = "Blip"
    confidence: float = 0.95
    snr: float = 10.0
    peak_frequency: float = 100.0

    @property
    def extent(self) -> Tuple[float, float]:
        """The whole extent: the Omicron extent widened to contain the peak."""
        return (min(self.start, self.peak), max(self.start + self.duration, self.peak))


@dataclass(frozen=True)
class EventRecord:
    """One catalogued event, with the time its signal spends above 20 Hz."""

    name: str
    gps: float
    in_band_seconds: float
    confident: bool = True

    @property
    def masses(self) -> Tuple[float, float]:
        """Equal component masses whose Newtonian time above 20 Hz is ``in_band_seconds``."""
        solar_mass_seconds = 4.925490947e-6
        chirp_time = ((5.0 / 256.0) * (np.pi * 20.0) ** (-8.0 / 3.0) / self.in_band_seconds) ** 0.6
        component = chirp_time / solar_mass_seconds * 2.0 ** 0.2
        return (component, component)

    @property
    def extent(self) -> Tuple[float, float]:
        """The in-band signal: from 20 Hz to the merger."""
        return (self.gps - self.in_band_seconds, self.gps)

    def as_catalogue_row(self, observing_run: str) -> Dict:
        mass1, mass2 = self.masses
        return {
            "name": self.name, "gps": float(self.gps), "mass1": mass1, "mass2": mass2,
            "distance": 400.0, "p_bbh": 1.0, "p_bns": 0.0, "p_nsbh": 0.0,
            "catalog": "GWTC-fake-confident" if self.confident else "GWTC-fake-marginal",
            "observing_run": observing_run,
            "event_type": 1 if self.confident else 0,
        }


_GRAVITY_SPY_HEADER = (
    "event_time", "ifo", "peak_time", "peak_time_ns", "start_time", "start_time_ns", "duration",
    "peak_frequency", "central_freq", "bandwidth", "snr", "gravityspy_id", "ml_label",
    "ml_confidence",
)


def _gravity_spy_csv(rows: Sequence[GlitchRecord]) -> str:
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(_GRAVITY_SPY_HEADER)
    for index, row in enumerate(rows):
        start_s = int(np.floor(row.start))
        start_ns = int(round((row.start - start_s) * 1e9))
        peak_s = int(np.floor(row.peak))
        writer.writerow([
            repr(float(row.peak)), row.ifo, peak_s, int(round((row.peak - peak_s) * 1e9)),
            start_s, start_ns, repr(float(row.duration)), row.peak_frequency, row.peak_frequency,
            10.0, row.snr, f"fake{index:05d}", row.label, row.confidence,
        ])
    return out.getvalue()


def write_gravity_spy(glitches: Sequence[GlitchRecord], scratch: Path) -> None:
    """
    Write the compact Gravity Spy cache of every detector and era from ``glitches``,
    through the library's own CSV reducer, into the current ``GRAVYFLOW_CACHE_DIR``.
    """
    from gravyflow.src.dataset.acquisition import gwosc_data
    from gravyflow.src.dataset.features import gravity_spy

    scratch.mkdir(parents=True, exist_ok=True)
    for (ifo, era) in gravity_spy.RELEASE_FILES:
        run = gwosc_data.GWOSC_RUNS[era]
        rows = [
            g for g in glitches
            if g.ifo == ifo and run["gps_start"] <= g.peak <= run["gps_end"]
        ]
        csv_path = scratch / f"{ifo}_{era}.csv"
        csv_path.write_text(_gravity_spy_csv(rows))
        gravity_spy.build_compact_catalogue(
            csv_path, gravity_spy.cache_dir() / f"{ifo}_{era}.npz",
            gravity_spy._expected_source(ifo, era),
        )
    gravity_spy.clear_memory_cache()


def _event_run(gps: float) -> str:
    import gravyflow as gf

    for run in gf.ObservingRun:
        if run.value.start_gps_time <= gps <= run.value.end_gps_time:
            return run.name
    return "Unknown"


#: Modules that bind an event producer by name.
EVENT_CONSUMERS: Tuple[str, ...] = (
    "gravyflow.src.dataset.features.segment_builders",
    "gravyflow.src.validate.bank",
)


def leftover_fakes() -> List[str]:
    """Module attributes still bound to a fake after its patch was undone."""
    found = []
    for module in list(sys.modules.values()):
        if module is None or not getattr(module, "__name__", "").startswith("gravyflow"):
            continue
        for attribute, value in list(vars(module).items()):
            if getattr(value, "_leakage_fake", False):
                found.append(f"{module.__name__}.{attribute}")
    return found


def install_events(monkeypatch: pytest.MonkeyPatch, events: Sequence[EventRecord]) -> None:
    """
    Replace every event producer with the fixture's table: in ``event``'s namespace,
    on ``BaseDataObtainer``, and at every module-level binding of the originals.
    """
    from gravyflow.src.dataset.acquisition.base import BaseDataObtainer
    from gravyflow.src.dataset.features import event as event_module

    events = list(events)
    confident = np.array([e.gps for e in events if e.confident], dtype=np.float64)
    marginal = np.array([e.gps for e in events if not e.confident], dtype=np.float64)

    def confident_times(use_cache: bool = True):
        return confident.copy()

    def marginal_times(use_cache: bool = True):
        return marginal.copy()

    def all_times(use_cache: bool = True):
        return np.concatenate([confident, marginal])

    def times_by_type(event_types, use_cache: bool = True):
        wanted = {int(t) for t in event_types}
        parts = [confident] if int(event_module.EventConfidence.CONFIDENT) in wanted else []
        if int(event_module.EventConfidence.MARGINAL) in wanted:
            parts.append(marginal)
        return np.concatenate(parts) if parts else np.array([])

    def events_with_params(observing_runs=None, event_types=None, event_names=None):
        if event_types is None:
            event_types = [event_module.EventConfidence.CONFIDENT]
        wanted = {int(t) for t in event_types}
        runs = None
        if observing_runs:
            runs = {getattr(r, "name", str(r)) for r in observing_runs}
        rows = []
        for e in sorted(events, key=lambda e: e.gps):
            if int(e.confident) not in wanted:
                continue
            run = _event_run(e.gps)
            if runs is not None and run not in runs:
                continue
            if event_names and e.name not in event_names:
                continue
            rows.append(e.as_catalogue_row(run))
        return rows

    for fake in (confident_times, marginal_times, all_times, times_by_type, events_with_params):
        fake._leakage_fake = True
    # Import every consumer first: a module first imported while the patch is on
    # would bind the fake by name and keep it after the patch is undone.
    import importlib

    for name in EVENT_CONSUMERS:
        importlib.import_module(name)

    replacements = {
        event_module.get_confident_event_times: confident_times,
        event_module.get_marginal_event_times: marginal_times,
        event_module.get_all_event_times: all_times,
        event_module.get_event_times_by_type: times_by_type,
        event_module.get_events_with_params: events_with_params,
    }
    for module in list(sys.modules.values()):
        if module is None or not getattr(module, "__name__", "").startswith("gravyflow"):
            continue
        for attribute, value in list(vars(module).items()):
            for original, fake in replacements.items():
                if value is original:
                    monkeypatch.setattr(module, attribute, fake)
    monkeypatch.setattr(BaseDataObtainer, "get_all_event_times", lambda self: all_times())


# =============================================================================
# THE INSTALLED HARNESS
# =============================================================================

@dataclass
class Harness:
    """What ``installed`` exposes to a test."""

    source: StrainSource
    cache_dir: Path
    monkeypatch: pytest.MonkeyPatch

    @property
    def encoding(self) -> Encoding:
        return self.source.encoding


@contextlib.contextmanager
def installed(fixture, root: Path, cache_dir: Optional[Path] = None, strain: bool = True):
    """
    Install ``fixture`` (a ``fixtures.Fixture``): strain, timelines and catalogues,
    with ``GRAVYFLOW_CACHE_DIR`` at ``cache_dir`` (default: a new directory under
    ``root``). Draw every batch inside the block. ``strain=False`` leaves the real
    strain fetch in place (for the guard's own test).
    """
    from gravyflow.src.dataset.acquisition.base import BaseDataObtainer
    from gravyflow.src.dataset.features import gravity_spy

    root = Path(root)
    cache_dir = Path(cache_dir) if cache_dir is not None else root / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    source = StrainSource(fixture.encoding)
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("GRAVYFLOW_CACHE_DIR", str(cache_dir))
        gravity_spy.clear_memory_cache()
        write_timelines(fixture.segments, fixture.flag_gaps)
        write_gravity_spy(fixture.glitches, root / "gravityspy_csv")
        install_events(mp, fixture.events)
        if strain:
            mp.setattr(BaseDataObtainer, "get_segment_data_static", staticmethod(source.fetch))
        try:
            yield Harness(source, cache_dir, mp)
        finally:
            close_obtainers()
            gravity_spy.clear_memory_cache()
    stale = leftover_fakes()
    if stale:
        raise AssertionError(f"Fakes outlived their patch (bound at import): {stale}")


__all__ = [
    "GWOSC_FLAGS", "HARDWARE_INJECTION_FLAGS", "Fetch", "StrainSource", "close_obtainers",
    "write_timelines", "write_gravity_spy", "GlitchRecord", "EventRecord", "install_events", "Harness", "installed",
]
