"""
Where real strain comes from: one data source per obtainer, named explicitly.

A data source (``DataSource``) answers everything a read of real strain asks of where
the strain lives:

- ``segments``: when a detector has data passing a set of GWOSC flags;
- ``rates``: the sample rates it publishes strain at, its native rates;
- ``tile_seconds``: the length of one stored tile of its strain at a rate;
- ``source_id``: the name of the strain it serves at a time and rate, which no other
  source's strain shares;
- ``fetch``: the strain of one detector over an interval, at a native rate.

The source is a parameter of every data obtainer (``source=``; ``gf.Defaults.source``,
which is ``GWOSC()``, when it is not given). It is part of the obtainer's
``specification()``, so every fingerprint, and every cache key, carries it.

- ``GWOSC()``: the public GWOSC timeline, and the strain of each run's bulk release
  (``gwosc_data.GWOSC_STRAIN_DATASETS``), never an event release.
- ``LocalFrames(frametypes, channels, sample_rate_hertz)``: frames found through
  gwdatafind, by the frame type and channel it names for each GWOSC run, and the GWOSC
  timeline wherever those frames are found.

A source never falls back to another, and never reads another frame type or rate than
it names. A fetch that fails raises ``StrainUnavailable``, which names the source, the
detector, the interval and the cause.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Any, Dict, Mapping, NamedTuple, Optional, Sequence, Tuple

import numpy as np

from gravyflow.src.utils.resample import whole_hertz
from .gwosc_data import (
    GWOSC_RUNS, GWOSC_SAMPLE_RATES_HERTZ, GWOSC_STRAIN_DATASETS, UpsamplingRefused,
    coalesce_segments, get_gwosc_segments, gwosc_runs_overlapping, intersect_segments,
)

#: The length of GWOSC's bulk strain files, which start on multiples of it in GPS time,
#: and of LVK's calibrated frame files: one stored tile of either.
FILE_SECONDS = 4096.0
#: How long one GWOSC download waits for the server, in seconds.
DOWNLOAD_TIMEOUT_SECONDS = 30


class SegmentStrain(NamedTuple):
    """
    One detector's strain over one interval: its samples, the GPS time of the first
    sample as the source returned it, the rate and the name of the strain it is
    (``DataSource.source_id``).
    """
    data: Any
    t0: float
    sample_rate_hertz: float
    source: str


def _detector_name(detector) -> str:
    return getattr(detector, "name", str(detector))


class StrainUnavailable(RuntimeError):
    """A source could not deliver the strain it was asked for: names the source,
    the detector, the interval and the cause."""

    def __init__(self, source: "DataSource", detector, interval: Sequence[float], cause: Any):
        self.source = source
        self.detector = _detector_name(detector)
        self.interval = (float(interval[0]), float(interval[1]))
        self.cause = cause
        reason = f"{type(cause).__name__}: {cause}" if isinstance(cause, BaseException) else str(cause)
        super().__init__(
            f"{source!r} could not deliver {self.detector} strain over GPS "
            f"[{self.interval[0]}, {self.interval[1]}): {reason}"
        )


class DataSource(ABC):
    """
    Where an obtainer's real strain and data-quality segments come from (see the
    module). A source is a value: its ``specification()`` is all that tells it from
    another, two sources of one specification are equal, and a copy of one is itself.
    """

    @abstractmethod
    def segments(self, detector, flags: Sequence[str], interval: Sequence[float]) -> np.ndarray:
        """Times in ``interval`` when ``detector`` has data passing every flag: (N, 2) GPS."""

    @abstractmethod
    def rates(self) -> Tuple[float, ...]:
        """The sample rates the source publishes strain at, ascending, in Hz."""

    @abstractmethod
    def tile_seconds(self, sample_rate_hertz: float) -> float:
        """The length of one stored tile of the source's strain at a rate, in seconds."""

    @abstractmethod
    def source_id(self, gps: float, sample_rate_hertz: float) -> str:
        """The name of the strain served in the tile starting at ``gps``, at a rate."""

    @abstractmethod
    def fetch(self, detector, sample_rate_hertz: float, interval: Sequence[float]) -> SegmentStrain:
        """
        ``detector``'s strain over ``interval``, at the native ``sample_rate_hertz``.

        Raises:
            StrainUnavailable: the strain could not be had, naming why.
        """

    @abstractmethod
    def specification(self) -> Dict[str, Any]:
        """Everything that determines the strain the source serves (fingerprints read it)."""

    def __eq__(self, other) -> bool:
        return type(self) is type(other) and self.specification() == other.specification()

    def __hash__(self) -> int:
        return hash(type(self).__qualname__)

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self


def fetch_rate(source: DataSource, sample_rate_hertz: float) -> float:
    """
    The native rate a request is fetched at: the lowest of ``source.rates()`` at or above
    ``sample_rate_hertz``, from which the one resampler reaches the request.

    Raises:
        ValueError: a rate that is not a positive whole number of hertz (``whole_hertz``).
        UpsamplingRefused: a rate above every rate the source publishes; real data is
            never upsampled.
    """
    requested = float(whole_hertz(sample_rate_hertz))
    for rate in sorted(source.rates()):
        if requested <= rate:
            return float(rate)
    raise UpsamplingRefused(
        f"Requested sample rate {requested:g} Hz exceeds the highest rate {source!r} "
        f"publishes ({max(source.rates()):g} Hz). Gravyflow does not upsample real data."
    )


def timeline_segments(detector, flags: Sequence[str], interval: Sequence[float]) -> np.ndarray:
    """
    Times in ``interval`` when ``detector`` passes every GWOSC flag in ``flags``, from the
    public GWOSC timeline over the GWOSC runs that overlap it (``gwosc_data``): a sorted,
    coalesced (N, 2) GPS array. Refuses, by name, a flag that looks like a DQSegDB flag.
    """
    flags = (flags,) if isinstance(flags, str) else tuple(flags)
    for flag in flags:
        if ":" in flag or "-" in flag:
            raise ValueError(
                f"Flag {flag!r} looks like a DQSegDB flag. Real-noise segments come "
                "from the public GWOSC timeline: use GWOSC flags such as 'DATA', "
                "'BURST_CAT1', 'BURST_CAT2', 'CBC_CAT1' or 'CBC_CAT2'."
            )
    start, end = float(interval[0]), float(interval[1])
    return get_gwosc_segments(
        _detector_name(detector), gwosc_runs_overlapping(start, end), flags, start=start, end=end
    )


def _one_run(start: float, end: float) -> str:
    """The GWOSC run that [start, end) overlaps, refusing none and several by name."""
    runs = gwosc_runs_overlapping(start, end)
    if len(runs) != 1:
        raise ValueError(
            f"GPS [{start}, {end}) overlaps {'no GWOSC run' if not runs else f'the GWOSC runs {runs}'}; "
            f"a fetch lies in exactly one of {sorted(GWOSC_RUNS)}."
        )
    return runs[0]


def _served(source: DataSource, detector, interval, series, sample_rate_hertz: float,
            source_id: str) -> SegmentStrain:
    """``series`` (a gwpy TimeSeries) as a SegmentStrain, refusing another rate than asked."""
    served_rate = float(series.sample_rate.value)
    if served_rate != float(sample_rate_hertz):
        raise StrainUnavailable(
            source, detector, interval,
            f"its strain was served at {served_rate:g} Hz, not its native {float(sample_rate_hertz):g} Hz",
        )
    return SegmentStrain(np.asarray(series.value), float(series.t0.value), served_rate, source_id)


class GWOSC(DataSource):
    """
    Public strain and data quality from GWOSC (https://gwosc.org), the default source.

    Segments come from the GWOSC timeline. Strain is each run's bulk release at 4096 Hz
    or 16384 Hz (``GWOSC_STRAIN_DATASETS``), downloaded by gwpy and kept nowhere but the
    store: gwpy's download cache is off, so a file is never kept twice. The release is
    named on every download, so an event release that overlaps the interval is never
    served in its place.
    """

    def segments(self, detector, flags, interval) -> np.ndarray:
        return timeline_segments(detector, flags, interval)

    def rates(self) -> Tuple[float, ...]:
        return GWOSC_SAMPLE_RATES_HERTZ

    def tile_seconds(self, sample_rate_hertz) -> float:
        return FILE_SECONDS

    def source_id(self, gps, sample_rate_hertz) -> str:
        return f"gwosc-{self._dataset(gps, gps + self.tile_seconds(sample_rate_hertz), sample_rate_hertz)}"

    def _dataset(self, start: float, end: float, sample_rate_hertz: float) -> str:
        """The bulk release holding [start, end) at a rate, refusing a rate not published."""
        rate = float(sample_rate_hertz)
        if rate not in self.rates():
            raise ValueError(
                f"GWOSC publishes {' and '.join(f'{r:g}' for r in self.rates())} Hz strain, "
                f"not {rate:g} Hz: fetch at a native rate (fetch_rate)."
            )
        run = _one_run(start, end)
        release = GWOSC_STRAIN_DATASETS.get(run, {}).get(rate)
        if release is None:
            raise ValueError(f"No GWOSC strain release is named for run {run} at {rate:g} Hz.")
        return release

    def fetch(self, detector, sample_rate_hertz, interval) -> SegmentStrain:
        start, end = float(interval[0]), float(interval[1])
        dataset = self._dataset(start, end, sample_rate_hertz)
        # Real data is the realdata extra's: gwpy loads only when strain is fetched.
        from gwpy.timeseries import TimeSeries

        try:
            series = TimeSeries.fetch_open_data(
                _detector_name(detector), start, end,
                sample_rate=int(sample_rate_hertz),
                dataset=f"^{re.escape(dataset)}$",
                cache=False,  # the store keeps the tile; a download is read once and deleted
                timeout=DOWNLOAD_TIMEOUT_SECONDS,
            )
        except Exception as error:
            raise StrainUnavailable(self, detector, interval, error) from error
        return _served(self, detector, interval, series, sample_rate_hertz, f"gwosc-{dataset}")

    def specification(self) -> Dict[str, Any]:
        return {"strain_datasets": {run: {f"{rate:g}": name for rate, name in releases.items()}
                                    for run, releases in GWOSC_STRAIN_DATASETS.items()}}

    def __repr__(self) -> str:
        return "GWOSC()"


#: The frames ``LocalFrames()`` reads when given no table: LVK's calibrated, cleaned
#: strain of O3 and O4, at 16384 Hz, on clusters that hold it. GWOSC's own O1 and O2
#: frames (``LOSC_4_V1``, ``LOSC-STRAIN``; ``GWOSC_O2_4KHZ_R1``, ``GWOSC-4KHZ_R1_STRAIN``)
#: are 4096 Hz, so they are named in a table of their own.
LOCAL_FRAMETYPES: Dict[str, str] = {run: "HOFT_C01" for run in ("O3a", "O3b", "O4a", "O4b")}
LOCAL_CHANNELS: Dict[str, str] = {run: "DCS-CALIB_STRAIN_CLEAN_C01" for run in LOCAL_FRAMETYPES}


class LocalFrames(DataSource):
    """
    Strain from local frame files, found through gwdatafind.

    Args:
        frametypes: {GWOSC run name: frame type}, e.g. ``{"O3a": "HOFT_C01"}``, read as
            ``"{detector}_{frame type}"``. Default ``LOCAL_FRAMETYPES``.
        channels: {GWOSC run name: channel}, e.g. ``{"O3a": "DCS-CALIB_STRAIN_CLEAN_C01"}``,
            read as ``"{detector}:{channel}"``, for the same runs. Default ``LOCAL_CHANNELS``.
        sample_rate_hertz: The rate of every frame named: the source's one native rate.

    Its segments are the GWOSC timeline where its frames are found (``find_times``), so
    it plans only time it can read, and only time GWOSC's timeline has published. A run
    it names no frame type for is refused by name, never read from another source or
    frame type. Reading frames needs the credentials of the cluster that holds them.
    """

    def __init__(
        self,
        frametypes: Optional[Mapping[str, str]] = None,
        channels: Optional[Mapping[str, str]] = None,
        sample_rate_hertz: float = 16384.0,
    ):
        frametypes = dict(LOCAL_FRAMETYPES if frametypes is None else frametypes)
        channels = dict(LOCAL_CHANNELS if channels is None else channels)
        if set(frametypes) != set(channels):
            raise ValueError(
                f"frametypes and channels name different runs: {sorted(frametypes)} and {sorted(channels)}."
            )
        unknown = sorted(set(frametypes) - set(GWOSC_RUNS))
        if unknown:
            raise ValueError(f"Unknown GWOSC runs {unknown}; known runs: {sorted(GWOSC_RUNS)}.")
        self.frametypes = {run: str(frametypes[run]) for run in sorted(frametypes)}
        self.channels = {run: str(channels[run]) for run in sorted(channels)}
        self.sample_rate_hertz = float(whole_hertz(sample_rate_hertz))

    def _frames(self, start: float, end: float) -> Tuple[str, str]:
        """(frame type, channel) of the one run [start, end) lies in, refusing a run not named."""
        run = _one_run(start, end)
        if run not in self.frametypes:
            raise ValueError(
                f"{self!r} names no frame type for GWOSC run {run} (GPS [{start}, {end})): name "
                "its frame type and channel, or read it with GWOSC()."
            )
        return self.frametypes[run], self.channels[run]

    def segments(self, detector, flags, interval) -> np.ndarray:
        name = _detector_name(detector)
        start, end = float(interval[0]), float(interval[1])
        from gwdatafind import find_times

        found = []
        for run in gwosc_runs_overlapping(start, end):
            bounds = GWOSC_RUNS[run]
            lo, hi = max(start, float(bounds["gps_start"])), min(end, float(bounds["gps_end"]))
            frametype, _ = self._frames(lo, hi)
            found += [(float(a), float(b)) for a, b in find_times(name[0], f"{name}_{frametype}", lo, hi)]
        return intersect_segments(timeline_segments(detector, flags, interval), coalesce_segments(found))

    def rates(self) -> Tuple[float, ...]:
        return (self.sample_rate_hertz,)

    def tile_seconds(self, sample_rate_hertz) -> float:
        return FILE_SECONDS

    def source_id(self, gps, sample_rate_hertz) -> str:
        frametype, channel = self._frames(gps, gps + self.tile_seconds(sample_rate_hertz))
        return f"frames-{frametype}-{channel}"

    def fetch(self, detector, sample_rate_hertz, interval) -> SegmentStrain:
        if float(sample_rate_hertz) != self.sample_rate_hertz:
            raise ValueError(f"{self!r} serves {self.sample_rate_hertz:g} Hz, not {float(sample_rate_hertz):g} Hz.")
        name = _detector_name(detector)
        start, end = float(interval[0]), float(interval[1])
        frametype, channel = self._frames(start, end)
        from gwdatafind import find_urls
        from gwpy.timeseries import TimeSeries

        try:
            files = find_urls(
                site=name[0], frametype=f"{name}_{frametype}", gpsstart=start, gpsend=end,
                urltype="file", on_gaps="raise",
            )
            if not files:
                raise FileNotFoundError(f"gwdatafind found no {name}_{frametype} frames")
            # Single-threaded: a forked reader would fork JAX's threads with it.
            series = TimeSeries.read(files, channel=f"{name}:{channel}", start=start, end=end, nproc=1)
        except Exception as error:
            raise StrainUnavailable(self, detector, interval, error) from error
        return _served(self, detector, interval, series, sample_rate_hertz,
                       self.source_id(start, sample_rate_hertz))

    def specification(self) -> Dict[str, Any]:
        return {"frametypes": dict(self.frametypes), "channels": dict(self.channels),
                "sample_rate_hertz": self.sample_rate_hertz}

    def __repr__(self) -> str:
        return (f"LocalFrames(frametypes={self.frametypes}, channels={self.channels}, "
                f"sample_rate_hertz={self.sample_rate_hertz:g})")


__all__ = [
    "DataSource", "GWOSC", "LocalFrames", "SegmentStrain", "StrainUnavailable",
    "LOCAL_FRAMETYPES", "LOCAL_CHANNELS", "FILE_SECONDS", "fetch_rate", "timeline_segments",
]
