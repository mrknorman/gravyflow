"""
Time slides: the background of a detector network's real noise.

A coincident search's false alarms come from noise that looks coherent across
detectors by chance. Its background is measured by sliding the detectors' data
against each other, far enough that no real signal, and no window's data, is
shared between them, and scoring the slid data exactly as the search scores
its own (`ValidationBank.score_windows`).

- **Stretches.** Time in which every detector has usable data of the scored
  split: each detector's admissible, vetoed segments (``available_segments``),
  intersected, and cut on the real-noise bin grid. A stretch lies inside one
  split by construction, and every slid window lies inside its stretch, so the
  slides honour the split and its guard: checked, never assumed, against the
  split the scored data was certified by (the validator's ``split_spec``, the
  one a model's record is checked against), not one read again here.
- **Slides.** In slide k, detector j's data are read ``j k step`` later than
  detector 0's: no two detectors are ever slid by the same amount, and the
  step defaults to twice a window's footprint, the separation of detectors the
  noise path keeps. A single detector has nothing to slide against: its
  background is its own data, unslid (slide 0).
- **Windows.** Each slide of a stretch is a stream of windows tiled one onsource
  apart (window mode), or one stride apart (stream mode, ``stride_seconds``),
  every one whole: no window wraps round the stretch. A slide of k steps
  therefore holds fewer windows than one of k - 1, and its livetime is its own
  windows' span (`metrics.ScoreStream`), counted exactly per slide.
- **Stream mode** reads a stream's slid strain back whole (`stream_strain`): the
  data its windows were cut from, detector j's read ``j k step`` later, so that an
  injection is scored on the very stream the background was (`stream`).
- **Wrapping slides** (`WrappedSlides`) reach far deeper. The network's segments,
  laid end to end in time order, form one ring of length ``L`` (their summed
  length); in slide k detector j is read ``j lag_k`` further round the ring,
  wrapping round each segment's end into the next segment and from the last
  segment back to the first. No data is lost to a slide, so each slide adds the
  whole ``L`` of background: ``K`` slides give ``K L``. Every lag between two
  detectors, of every slide, lies further than `minimum_shift_seconds` (the
  light-travel time plus the model's input length) round the ring from zero lag
  and from every other slide's lag between them, so no slide reuses zero lag or
  another slide's pairing. Two times a ring distance apart are at least that far
  apart in GPS time, so the rule holds in GPS time too.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional, Sequence, Tuple

import numpy as np
from keras import ops

import gravyflow as gf
from gravyflow.src.dataset.acquisition.base import (
    SEGMENT_GRID_ORIGIN_GPS, gather_windows, window_offsets_seconds,
)
from gravyflow.src.dataset.acquisition.gwosc_data import intersect_segments
from gravyflow.src.dataset.config import WindowSpec

#: How far apart two detectors' first samples may be and still be one stretch's.
_ALIGNMENT_SECONDS = 1e-6


def footprint_seconds(window: WindowSpec) -> float:
    """The span of one window's data: its offsource and its padded onsource."""
    return (window.num_onsource_samples + window.num_offsource_samples) / float(window.sample_rate_hertz)


@dataclass(frozen=True)
class TimeSlides:
    """
    Slides 1 .. ``num_slides`` of a network, detector j slid by ``j k step`` in
    slide k; ``step_seconds`` defaults to twice a window's footprint.
    """
    num_slides: int = 16
    step_seconds: Optional[float] = None

    def __post_init__(self):
        if int(self.num_slides) != self.num_slides or self.num_slides < 1:
            raise ValueError(f"num_slides must be a whole number >= 1, got {self.num_slides}.")
        if self.step_seconds is not None and not self.step_seconds > 0:
            raise ValueError(f"step_seconds must be > 0, got {self.step_seconds}.")

    def step(self, window: WindowSpec) -> float:
        return float(self.step_seconds) if self.step_seconds is not None else 2.0 * footprint_seconds(window)

    def slides(self, num_detectors: int) -> range:
        """The slides of a network: 1 .. num_slides, or only 0 for a single detector."""
        return range(1, self.num_slides + 1) if num_detectors > 1 else range(0, 1)


def minimum_shift_seconds(ifos: Sequence, window: Optional[WindowSpec] = None, input_seconds: float = 0.0) -> float:
    """
    The least shift between two detectors that a slide may use: the network's
    largest light-travel time plus the model's input length (``window``'s
    footprint, its offsource and padded onsource, or ``input_seconds``).
    """
    names = [ifo.name if hasattr(ifo, "name") else str(ifo) for ifo in ifos]
    light = gf.Network([gf.IFO[name] for name in names]).max_arrival_time_difference_seconds if len(names) > 1 else 0.0
    return float(light) + (footprint_seconds(window) if window is not None else float(input_seconds))


class WrappedSlides:
    """
    Slides that wrap round the network's segments (module docstring).

    Args:
        segments: (start, end) pairs, GPS seconds: the time every detector has
            data in, disjoint; they are laid end to end in time order as one ring.
        lags: each slide's lag (s): in slide k detector j is read ``j lags[k]``
            further round the ring than detector 0.
        num_detectors: the detectors slid.
        minimum_seconds: the least ring distance any lag between two detectors
            may lie from zero lag and from another slide's lag between them
            (`minimum_shift_seconds`); a lag too near is refused by slide.
    """

    def __init__(self, segments, lags, num_detectors: int, minimum_seconds: float):
        segments = np.asarray(segments, dtype=np.float64).reshape(-1, 2)
        segments = segments[np.argsort(segments[:, 0], kind="stable")]
        if not len(segments) or not np.all(segments[:, 1] > segments[:, 0]):
            raise ValueError("WrappedSlides need at least one segment, each with end > start.")
        if np.any(segments[1:, 0] < segments[:-1, 1]):
            raise ValueError("WrappedSlides' segments must not overlap.")
        if int(num_detectors) != num_detectors or num_detectors < 2:
            raise ValueError(f"A slide needs at least two detectors, got {num_detectors}.")
        if not minimum_seconds >= 0:
            raise ValueError(f"minimum_seconds must be >= 0, got {minimum_seconds}.")
        self.segments = segments
        self.lags = np.asarray(lags, dtype=np.float64).reshape(-1)
        self.num_detectors = int(num_detectors)
        self.minimum_seconds = float(minimum_seconds)
        lengths = segments[:, 1] - segments[:, 0]
        #: Where each segment starts round the ring.
        self.ring_starts = np.concatenate([[0.0], np.cumsum(lengths)[:-1]])
        self.ring_seconds = float(lengths.sum())
        self._check_lags()

    @classmethod
    def spaced(cls, segments, num_detectors: int, minimum_seconds: float, step_seconds: Optional[float] = None,
               max_slides: Optional[int] = None) -> "WrappedSlides":
        """
        Lags ``k step`` for k = 1 .. K, as many as the ring holds (at most
        ``max_slides``): ``step`` (by default just over ``minimum_seconds``) must exceed
        ``minimum_seconds``, and the largest lag between two detectors,
        ``(num_detectors - 1) K step``, stays a step short of the ring's length.
        """
        step = float(step_seconds) if step_seconds is not None else float(minimum_seconds) * (1 + 1e-9) + 1e-9
        if not step > minimum_seconds:
            raise ValueError(f"step_seconds {step} must exceed minimum_seconds {minimum_seconds}.")
        segments = np.asarray(segments, dtype=np.float64).reshape(-1, 2)
        ring = float((segments[:, 1] - segments[:, 0]).sum())
        count = int(np.floor((ring - step) / ((num_detectors - 1) * step) * (1 + 1e-12)))
        while count > 0 and ring - (num_detectors - 1) * count * step <= minimum_seconds:
            count -= 1
        if max_slides is not None:
            count = min(count, int(max_slides))
        if count < 1:
            raise ValueError(f"A ring of {ring} s holds no slide {step} s apart from zero lag.")
        return cls(segments, step * np.arange(1, count + 1), num_detectors, minimum_seconds)

    def __len__(self) -> int:
        return self.lags.size

    def _circular(self, values) -> np.ndarray:
        """Ring distance from zero of each of ``values``."""
        wrapped = np.mod(values, self.ring_seconds)
        return np.minimum(wrapped, self.ring_seconds - wrapped)

    def _check_lags(self) -> None:
        if not np.all(np.isfinite(self.lags)):
            raise ValueError("WrappedSlides' lags must be finite.")
        for d in range(1, self.num_detectors):
            pair = d * self.lags
            near_zero = self._circular(pair) <= self.minimum_seconds
            if near_zero.any():
                k = int(np.flatnonzero(near_zero)[0])
                raise ValueError(
                    f"Slide {k} (lag {self.lags[k]} s) puts detectors {d} apart at {pair[k]} s, within "
                    f"{self.minimum_seconds} s of zero lag round a ring of {self.ring_seconds} s.")
            order = np.argsort(np.mod(pair, self.ring_seconds), kind="stable")
            sorted_pair = np.mod(pair, self.ring_seconds)[order]
            gaps = np.diff(sorted_pair)
            close = gaps <= self.minimum_seconds
            if close.any():
                i = int(np.flatnonzero(close)[0])
                a, b = sorted((int(order[i]), int(order[i + 1])))
                raise ValueError(
                    f"Slides {a} and {b} put detectors {d} apart at lags within {self.minimum_seconds} s of each "
                    "other round the ring: they would reuse one pairing of the data.")

    @property
    def background_seconds(self) -> float:
        """The background the slides give: every slide the whole ring."""
        return self.lags.size * self.ring_seconds

    def to_ring(self, gps) -> np.ndarray:
        """Each GPS time's position round the ring; a time outside every segment is refused."""
        gps = np.asarray(gps, dtype=np.float64)
        index = np.searchsorted(self.segments[:, 0], gps, side="right") - 1
        inside = (index >= 0) & (gps < self.segments[np.clip(index, 0, None), 1])
        if not np.all(inside):
            bad = gps[~inside].reshape(-1)[0]
            raise ValueError(f"The time {bad} lies outside every segment of the slides.")
        return self.ring_starts[index] + (gps - self.segments[index, 0])

    def from_ring(self, position) -> np.ndarray:
        """The GPS time of each ring position (taken round the ring)."""
        position = np.mod(np.asarray(position, dtype=np.float64), self.ring_seconds)
        index = np.searchsorted(self.ring_starts, position, side="right") - 1
        return self.segments[index, 0] + (position - self.ring_starts[index])

    def lag(self, slide: int, detector: int) -> float:
        """How far round the ring detector ``detector`` is read in slide ``slide``."""
        return float(detector * self.lags[slide])

    def slid_times(self, gps, slide: int, detector: int) -> np.ndarray:
        """
        Detector ``detector``'s times ``gps`` as they stand against detector 0 in slide
        ``slide``: its data read ``lag`` further round the ring stand ``lag`` earlier,
        wrapped round segments.
        """
        return self.from_ring(self.to_ring(gps) - self.lag(slide, detector))


@dataclass(frozen=True)
class SlideStream:
    """
    One stretch's windows at one slide, tiled one onsource apart: ``num_windows``
    of them, served in order. ``block`` is the split block the stretch starts in,
    which the stretch's slides share: their data are the same, so correlated.
    """
    stretch: Tuple[float, float]
    slide: int
    num_windows: int
    block: int
    #: The samples of the stream's slid strain (`TimeSlideBackground.stream_strain`): every
    #: detector's, less its slide; 0 where not counted.
    num_samples: int = 0


@dataclass(frozen=True)
class SlideBatch:
    """
    Consecutive windows of one stream: each detector's padded onsource (B, I, P)
    and the offsource just before it (B, I, O), and the GPS time of each
    detector's first cropped onsource sample (B, I), as ``START_GPS_TIME``.
    """
    stream: SlideStream
    onsource: object
    offsource: object
    start_gps: np.ndarray


class TimeSlideBackground:
    """
    The time-slid windows of split ``group`` of a real-noise obtainer's
    detectors (``ifos``), for windows of geometry ``window``. ``split_spec`` is
    how the obtainer's data is split (`split_spec_of`): every stretch must lie
    in its split ``group``, and its blocks are the streams'.

    ``stretches`` are visited in an order drawn from ``rng`` (a NumPy Generator),
    and each stretch's slides in turn, until a budget of windows is spent. Windows
    are ``stride_seconds`` apart: by default one onsource (window mode).
    """

    def __init__(self, obtainer, ifos: Sequence, split_spec: "gf.SplitSpec", group: str, window: WindowSpec,
                 slides: TimeSlides, rng: np.random.Generator, stride_seconds: Optional[float] = None):
        if not hasattr(obtainer, "available_segments"):
            raise TypeError(
                f"Time slides read a real-noise obtainer's segments; {type(obtainer).__name__} has none."
            )
        self.obtainer = obtainer
        self.ifos = list(ifos)
        self.split_spec = split_spec
        self.group = gf.SplitSpec.check_name(group)
        self.window = window
        self.slides = slides
        rate = float(window.sample_rate_hertz)
        self.step_samples = int(round(slides.step(window) * rate))
        stride = window.onsource_duration_seconds if stride_seconds is None else stride_seconds
        self.stride_samples = int(round(stride * rate))
        if self.stride_samples < 1:
            raise ValueError(f"A stride of {stride} s is less than one sample at {rate} Hz.")
        self.stretches = self._stretches(rng)
        self._loaded = None  # the last stretch read back by `stream_strain`, with its data

    # ---------------------------------------------------------------- stretches

    def _stretches(self, rng: np.random.Generator) -> np.ndarray:
        """Every detector's usable time of the split, intersected and cut on the bin grid."""
        segments = None
        for ifo in self.ifos:
            available = self.obtainer.available_segments(ifo, self.group)
            segments = available if segments is None else intersect_segments(segments, available)
        stretches = self.obtainer.cut_segments(
            segments, self.obtainer.max_segment_duration_seconds, SEGMENT_GRID_ORIGIN_GPS)
        for start, end in stretches:
            if self.split_spec.assign(start, end) != self.group:
                raise ValueError(
                    f"The stretch [{start}, {end}) is not inside split {self.group!r}: a slide of it "
                    "could pair data of two splits."
                )
        first = min(self.slides.slides(len(self.ifos)))
        rate = float(self.window.sample_rate_hertz)
        stretches = np.array([s for s in stretches
                              if self._num_windows(int(np.floor((s[1] - s[0]) * rate + 1e-6)), first) > 0])
        return stretches.reshape(-1, 2)[rng.permutation(len(stretches))]

    def _num_windows(self, num_samples: int, slide: int) -> int:
        """Whole windows a stretch of ``num_samples`` holds at ``slide``: its last detector's must fit."""
        footprint = self.window.num_onsource_samples + self.window.num_offsource_samples
        offset = (len(self.ifos) - 1) * slide * self.step_samples
        return max((num_samples - footprint - offset) // self.stride_samples + 1, 0)

    def _load(self, stretch) -> Tuple[list, float]:
        """
        Each detector's strain over ``stretch`` and its first sample's GPS. A detector's
        data the source does not deliver is refused by name (``TimelineContradiction``),
        never skipped: its ``exclusions`` entry plans around it.
        """
        rows = np.repeat(np.asarray(stretch, dtype=np.float64)[None, None, :], len(self.ifos), axis=1)
        (row,) = list(self.obtainer.acquire(self.window.sample_rate_hertz, rows, self.ifos))
        first_gps = np.asarray(row.start_gps_time, dtype=np.float64)
        if np.ptp(first_gps) > _ALIGNMENT_SECONDS:
            raise ValueError(f"The detectors' data over [{stretch[0]}, {stretch[1]}) start at different "
                             f"times {first_gps.tolist()}: slides would be off by the difference.")
        return row.data, float(first_gps[0])

    # ------------------------------------------------------------------ windows

    def batches(self, batch_size: int, max_windows: int) -> Iterator[SlideBatch]:
        """
        Batches of up to ``batch_size`` windows, ``max_windows`` in all: each
        stretch's slides in turn, the last stream ended short at the budget.
        """
        remaining = int(max_windows)
        for stretch in self.stretches:
            if remaining <= 0:
                return
            data, first_gps = self._load(stretch)
            num_samples = min(int(ops.shape(d)[0]) for d in data)
            for slide in self.slides.slides(len(self.ifos)):
                count = min(self._num_windows(num_samples, slide), remaining)
                if count <= 0:
                    break
                stream = SlideStream((float(stretch[0]), float(stretch[1])), slide, count,
                                     self.split_spec.block_of(stretch[0]),
                                     num_samples - (len(self.ifos) - 1) * slide * self.step_samples)
                for first in range(0, count, batch_size):
                    yield self._batch(stream, data, first_gps, np.arange(first, min(first + batch_size, count)))
                remaining -= count

    def stream_strain(self, stream: SlideStream) -> Tuple[np.ndarray, float]:
        """
        ``stream``'s slid strain, (detectors, ``stream.num_samples``), float64: detector
        j's data read ``j k step`` later than detector 0's, so that the stream's window
        ``i`` is cut from samples ``i * stride`` on, as `batches` cuts it; and the GPS time
        of detector 0's first sample. The stretch is read again (the last one read is kept).
        """
        key = tuple(stream.stretch)
        if self._loaded is None or self._loaded[0] != key:
            data, first_gps = self._load(np.asarray(stream.stretch, dtype=np.float64))
            self._loaded = (key, [np.asarray(d, dtype=np.float64) for d in data], first_gps)
        _, data, first_gps = self._loaded
        n = stream.num_samples
        offsets = [j * stream.slide * self.step_samples for j in range(len(data))]
        if n < 1 or any(offset + n > d.shape[0] for offset, d in zip(offsets, data)):
            raise ValueError(f"The stretch [{stream.stretch[0]}, {stream.stretch[1]}) read back holds fewer "
                             f"samples than its stream at slide {stream.slide} was scored on.")
        return np.stack([d[offset:offset + n] for offset, d in zip(offsets, data)]), first_gps

    def detector_offsets(self, stream: SlideStream, num_detectors: int) -> np.ndarray:
        """Where the stream's data start on each detector's timeline (GPS s): detector j's ``j k step`` later."""
        step = self.step_samples / float(self.window.sample_rate_hertz)
        return stream.stretch[0] + np.arange(num_detectors) * stream.slide * step

    def ring_segments(self, num_windows: int = 0) -> Iterator[Tuple[np.ndarray, float, SlideStream]]:
        """
        The segments a ring of wrapping slides lays end to end: every stretch, in time
        order, each its strain (detectors, samples), float64, its first sample's GPS time
        and its zero-lag stream. ``num_windows`` is not a bound: the ring is all the data.
        """
        for stretch in self.stretches[np.argsort(self.stretches[:, 0], kind="stable")]:
            data, first_gps = self._load(stretch)
            num_samples = min(int(ops.shape(d)[0]) for d in data)
            count = self._num_windows(num_samples, 0)
            strain = np.stack([np.asarray(d, dtype=np.float64)[:num_samples] for d in data])
            yield strain, first_gps, SlideStream((float(stretch[0]), float(stretch[1])), 0, count,
                                                 self.split_spec.block_of(stretch[0]), num_samples)

    def _batch(self, stream: SlideStream, data: list, first_gps: float, index: np.ndarray) -> SlideBatch:
        """Windows ``index`` of ``stream``: detector j's padded onsource starts ``j k step`` later."""
        window = self.window
        starts = window.num_offsource_samples + index * self.stride_samples
        onsource, offsource, padded_gps = [], [], []
        for j, strain in enumerate(data):
            shifted = starts + j * stream.slide * self.step_samples
            on, off = gather_windows(strain, shifted, window.num_onsource_samples, window.num_offsource_samples)
            onsource.append(on)
            offsource.append(off)
            padded_gps.append(first_gps + window_offsets_seconds(shifted, 1.0 / window.sample_rate_hertz))
        return SlideBatch(stream, ops.stack(onsource, axis=1), ops.stack(offsource, axis=1),
                          window.onsource_start_gps(np.stack(padded_gps, axis=1)))
