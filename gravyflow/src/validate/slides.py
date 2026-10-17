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
  apart, every one whole: no window wraps round the stretch. A slide of k steps
  therefore holds fewer windows than one of k - 1, and its livetime is its own
  windows' span (`metrics.ScoreStream`), counted exactly per slide.
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
    and each stretch's slides in turn, until a budget of windows is spent.
    """

    def __init__(self, obtainer, ifos: Sequence, split_spec: "gf.SplitSpec", group: str, window: WindowSpec,
                 slides: TimeSlides, rng: np.random.Generator):
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
        self.stride_samples = int(round(window.onsource_duration_seconds * rate))
        self.stretches = self._stretches(rng)

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
                                     self.split_spec.block_of(stretch[0]))
                for first in range(0, count, batch_size):
                    yield self._batch(stream, data, first_gps, np.arange(first, min(first + batch_size, count)))
                remaining -= count

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
