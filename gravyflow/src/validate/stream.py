"""
Stream mode: every test runs the model over continuous strain, as a search does.

The validator's default (`ValidationConfig.mode`): continuous streams, as a
search runs. Window mode, which scores each example as one window, is kept for
backward compatibility and to reproduce earlier results.

- **Streams.** The model is slid along continuous raw strain at its own stride
  (``stride_seconds``; a quarter of its window if unstated, recorded on the
  result). Each window is the model's own geometry, its offsource the data just
  before its onsource, and is conditioned by the model's own dataset code
  (``Dataset.model_inputs``: whitening included), so the validator supplies raw
  strain and the model does its own conditioning. A window's score stands at its
  onsource's centre (`StreamGeometry.window_time`).
- **Triggers.** A score is a trigger when it is the loudest within
  ``cluster_seconds`` (1 s) of itself: the matched filter's clustering
  (`detection.stream`), reused, so every model's triggers are made alike. They are
  `gf.Triggers`, each with the found window as its ``var``.
- **Background.** Time slides of the test data (real noise, `slides`), or
  independent segments of continuous synthetic noise (`SyntheticSegments`),
  scored whole at the stride: their triggers are the background
  (`StreamBackground`), and its FAR the triggers at or above a threshold per
  second of livetime, counted in the background's independent information as
  everywhere (`metrics.Background`).
- **What counts as found.** An injection is found at a threshold when the loudest
  trigger within the found window of its merger passes it: ``found_window_seconds``
  (0.1 s), but at least half the stride (`found_window`; recorded), so that a
  perfectly timed signal is never missed for the grid's sake; one
  trigger per injection, the loudest. An injection's score is therefore that
  trigger's statistic, ``-inf`` where no trigger lies in the window, and found is
  ``score >= threshold`` as in window mode, so the estimators (`efficiency`,
  `volume`) read it unchanged.
- **Chance.** Triggers lie more than ``cluster_seconds`` apart, at least twice
  the found window (refused otherwise), so at most one lies in a found window: noise alone finds an injection with the chance that a
  noise trigger passing the threshold lies there, the trigger rate times the
  window (``2 w``). As a law (`StreamBackground.found_by_chance`), the background's
  livetime is ``T / 2w`` found windows, of which the triggers at or above a
  threshold chosen from them fill ``c``: ``Beta(c / r, (n - c) / r + 1)``, window
  mode's law with found windows for windows. It is subtracted at every threshold.
- **Local injection scoring.** The clean data is scored once (the background).
  Each injection is then scored on a short stretch of that same data around its
  merger, the signal added: exactly the windows whose scores the found test reads
  (every window within ``found_window_seconds + cluster_seconds`` of the merger,
  on the stream's own grid) and their data. A window's score depends on its own
  data alone, so this is what scoring the whole stream with the injection added
  gives there (`test_validate_stream`). The stretch is read by the injection
  stage's dataset (`StreamReader`), which injects, scales and records as it does
  for windows. The stretch is ``injection_stretch_seconds`` (40 s) whatever the
  model, and the spectrum the clean stretch's own, so an injection's SNR, hrss and
  hpeak are the same for every model it is scored on.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

import gravyflow as gf
from gravyflow.src.dataset.acquisition.base import gather_windows
from gravyflow.src.dataset.config import WindowSpec
from gravyflow.src.detection.stream import _cluster_in_time
from gravyflow.src.detection.triggers import Triggers
from .metrics import Background, ChanceLaw, ScoreStream, chosen_count
from .slides import SlideBatch, SlideStream


# --------------------------------------------------------------- geometry

@dataclass(frozen=True)
class StreamGeometry:
    """
    Where a window model's windows lie on a stream of continuous strain: window ``i``'s
    data (its offsource, then its padded onsource) starts ``i * stride_samples`` into
    the stream, and its score stands at its onsource's centre.
    """
    window: WindowSpec
    stride_samples: int

    def __post_init__(self):
        window_samples = float(self.window.onsource_duration_seconds) * float(self.window.sample_rate_hertz)
        if not 1 <= self.stride_samples <= window_samples + 1e-6:
            raise ValueError(f"A stride of {self.stride_samples} samples must be at least one sample and at most "
                             f"the model's window ({self.window.onsource_duration_seconds} s): a longer one "
                             "leaves data no window reads.")

    @classmethod
    def of(cls, window: WindowSpec, stride_seconds: Optional[float]) -> "StreamGeometry":
        """
        The geometry at ``stride_seconds``, which must be a whole number of samples; by
        default (None) a quarter of the window, rounded to the nearest whole sample.
        """
        rate = float(window.sample_rate_hertz)
        if stride_seconds is None:
            return cls(window, max(int(round(float(window.onsource_duration_seconds) * rate / 4.0)), 1))
        stride = int(round(stride_seconds * rate))
        if stride < 1 or abs(stride - stride_seconds * rate) > 1e-6:
            raise ValueError(f"A stride of {stride_seconds} s is not a whole number of samples (>= 1) at "
                             f"{window.sample_rate_hertz} Hz.")
        return cls(window, stride)

    @property
    def sample_rate_hertz(self) -> float:
        return float(self.window.sample_rate_hertz)

    @property
    def stride_seconds(self) -> float:
        return self.stride_samples / self.sample_rate_hertz

    @property
    def footprint_samples(self) -> int:
        """One window's data: its offsource and its padded onsource."""
        return self.window.num_offsource_samples + self.window.num_onsource_samples

    @property
    def centre_samples(self) -> float:
        """A window's time from the start of its data: its onsource's centre."""
        return self.window.num_offsource_samples + self.window.num_onsource_samples / 2.0

    def num_windows(self, num_samples: int) -> int:
        """The whole windows a stream of ``num_samples`` holds."""
        return max((int(num_samples) - self.footprint_samples) // self.stride_samples + 1, 0)

    def window_time(self, index) -> np.ndarray:
        """Window ``index``'s time (s) from the stream's first sample."""
        return (np.asarray(index, dtype=np.float64) * self.stride_samples + self.centre_samples) / self.sample_rate_hertz

    def windows_near(self, sample: int, reach_seconds: float) -> Tuple[int, int]:
        """The first and last windows whose times lie within ``reach_seconds`` of stream sample ``sample``."""
        reach = reach_seconds * self.sample_rate_hertz
        first = math.ceil((sample - self.centre_samples - reach) / self.stride_samples - 1e-9)
        last = math.floor((sample - self.centre_samples + reach) / self.stride_samples + 1e-9)
        return first, last

    def local_half_samples(self, reach_seconds: float) -> int:
        """
        Half a local stretch around a merger: enough that every window within
        ``reach_seconds`` of the merger, data and all, lies inside it (`windows_near`).
        """
        return int(self.window.num_offsource_samples + math.ceil(self.window.num_onsource_samples / 2.0)
                   + math.ceil(reach_seconds * self.sample_rate_hertz) + 1)

    def gather(self, strain: np.ndarray, first_sample: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        The windows whose data start at ``first_sample`` of ``strain`` (detectors,
        samples): each one's padded onsource (B, I, P) and offsource (B, I, O).
        """
        starts = np.asarray(first_sample, dtype=np.int64) + self.window.num_offsource_samples
        onsource, offsource = [], []
        for detector in strain:
            on, off = gather_windows(detector, starts, self.window.num_onsource_samples,
                                     self.window.num_offsource_samples)
            onsource.append(np.asarray(on))
            offsource.append(np.asarray(off))
        return np.stack(onsource, axis=1), np.stack(offsource, axis=1)


# --------------------------------------------------------------- triggers

def found_window(found_window_seconds: float, stride_seconds: float) -> float:
    """
    The found window used at a stride: ``found_window_seconds``, but at least half the
    stride, so that a perfectly timed loud signal, whose nearest window lies within half
    a stride of its merger, is always found (at ±0.1 s and a 0.25 s stride it was found
    only 80% of the time, ``2 w / stride``).
    """
    return max(float(found_window_seconds), float(stride_seconds) / 2.0)


def cluster(times, scores, cluster_seconds: float) -> np.ndarray:
    """
    The scores that are triggers: each the loudest within ``cluster_seconds`` of
    itself (ties: the middle of a run of equal scores, `detection.stream`), in time
    order, as the matched filter clusters. Indices into ``times``. A run of equal
    scores is read whole only where the scores given hold it whole: local scoring
    reads every window within ``found_window_seconds + cluster_seconds`` of a merger.
    """
    return _cluster_in_time(np.asarray(times, dtype=np.float64), np.asarray(scores, dtype=np.float64),
                            float(cluster_seconds))


def stream_triggers(times, scores, cluster_seconds: float, found_window_seconds: float) -> Triggers:
    """A stream's triggers (`cluster`), as `gf.Triggers` whose ``var`` is the found window."""
    keep = cluster(times, scores, cluster_seconds)
    times, scores = np.asarray(times, dtype=np.float64), np.asarray(scores, dtype=np.float64)
    return Triggers(times[keep], scores[keep], np.full(keep.size, float(found_window_seconds)))


def loudest_within(trigger_times, trigger_stats, centres, half_width: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    For each of ``centres``, the loudest trigger within ``half_width`` of it (both
    ends in; of equal statistics, the middle one, the earlier of two): its statistic
    and time; ``-inf`` and NaN where none lies there.
    """
    times = np.asarray(trigger_times, dtype=np.float64)
    stats = np.asarray(trigger_stats, dtype=np.float64)
    centres = np.asarray(centres, dtype=np.float64).reshape(-1)
    order = np.argsort(times, kind="stable")
    times, stats = times[order], stats[order]
    lo = np.searchsorted(times, centres - half_width, side="left")
    hi = np.searchsorted(times, centres + half_width, side="right")
    best, when = np.full(centres.size, -np.inf), np.full(centres.size, np.nan)
    for k in np.flatnonzero(hi > lo):
        tied = np.flatnonzero(stats[lo[k]:hi[k]] == stats[lo[k]:hi[k]].max())
        j = lo[k] + int(tied[(tied.size - 1) // 2])
        best[k], when[k] = stats[j], times[j]
    return best, when


# ------------------------------------------------------------- background

class StreamBackground(Background):
    """
    The triggers of noise streams (`ScoreStream`, each a stream's window scores at
    its stride, in time order) as false-alarm rates (`metrics.Background`): a
    stream's triggers are its scores that are the loudest within
    ``cluster_seconds`` of themselves (`cluster`), counted at or above each
    threshold per second of the streams' livetime. No trigger is taken within
    ``cluster_seconds`` of a stream's ends, and the livetime is less those stretches
    (`trigger_indices`). The count is monotone in the threshold. ``window_scores``
    are the triggers' scores.

    Its chance (`found_by_chance`) is stream mode's: a noise trigger passing the
    threshold within ``found_window_seconds`` of a merger (module docstring).
    """

    def __init__(self, streams: Sequence[ScoreStream], cluster_seconds: float, found_window_seconds: float,
                 footprint_seconds: Optional[float] = None):
        if not cluster_seconds >= 0:
            raise ValueError(f"cluster_seconds must be >= 0, got {cluster_seconds}.")
        if not found_window_seconds > 0:
            raise ValueError(f"found_window_seconds must be > 0, got {found_window_seconds}.")
        if not cluster_seconds >= 2.0 * found_window_seconds:
            raise ValueError(f"Triggers must lie more than a found window (2 x {found_window_seconds} s) apart: "
                             f"cluster_seconds {cluster_seconds} is too short.")
        self.streams = tuple(streams)
        self.cluster_seconds = float(cluster_seconds)
        self.found_window_seconds = float(found_window_seconds)
        self.footprint_seconds = footprint_seconds
        self._independent_block = None
        self._disjoint = None
        triggers, positions = [], []
        for stream in self.streams:
            keep = self.trigger_indices(stream)
            triggers.append(stream.scores[keep])
            if stream.detector_offsets is not None:
                positions.append(stream.data_positions(keep))
        self._init_counts(np.concatenate(triggers) if triggers else np.empty(0), np.empty(0),
                          sum(self.stream_livetime_seconds(stream) for stream in self.streams))
        rings = {stream.ring_seconds for stream in self.streams}
        if (footprint_seconds is not None and positions and len(positions) == len(self.streams)
                and len(rings) == 1 and len({p.shape[1] for p in positions}) == 1):
            # Count independent events by the data their triggers share.
            self._count_disjoint(np.concatenate(triggers), np.concatenate(positions), footprint_seconds,
                                 rings.pop())

    def trigger_indices(self, stream: ScoreStream) -> np.ndarray:
        """
        ``stream``'s triggers (`cluster`), but none within ``cluster_seconds`` of either
        end of its windows' times, where a score's neighbourhood runs off the stream and
        it would be the loudest of fewer scores than any other. A ring of wrapping slides
        is a stream cut at its join, so nothing is counted across the join either.
        """
        times = np.arange(stream.scores.size) * stream.stride_seconds
        keep = cluster(times, stream.scores, self.cluster_seconds)
        span = times[-1] if times.size else 0.0
        return keep[(times[keep] >= self.cluster_seconds) & (times[keep] <= span - self.cluster_seconds)]

    def stream_livetime_seconds(self, stream: ScoreStream) -> float:
        """``stream``'s livetime less the ``cluster_seconds`` at each end its triggers are dropped from."""
        return max(stream.livetime_seconds - 2.0 * self.cluster_seconds, 0.0)

    @property
    def num_triggers(self) -> int:
        return self.num_windows

    @property
    def found_windows(self) -> float:
        """The livetime counted in found windows (``2 w`` each), at least the triggers' number."""
        return max(self.livetime_seconds / (2.0 * self.found_window_seconds), float(self.num_triggers))

    def found_by_chance(self, threshold) -> ChanceLaw:
        """
        The chance that noise alone finds an injection at each threshold: that a noise
        trigger at or above it lies within ``found_window_seconds`` of the merger. Of the
        livetime's ``n`` found windows, a threshold chosen from the triggers stands on
        ``c`` of them (`metrics.chosen_count`), so the chance is ``Beta(c / r, (n - c) / r
        + 1)``, ``r`` the count's overdispersion (`reuse_at`): on average the trigger rate
        times ``2 w``. Nothing past every trigger (an infinite threshold); certainty at
        ``-inf``, which every injection passes; NaN at NaN.
        """
        t = np.atleast_1d(np.asarray(threshold, dtype=np.float64))
        if self.num_triggers == 0:
            a, b = np.zeros(t.shape), np.ones(t.shape)
        else:
            n = self.found_windows
            count = chosen_count(self._opens, np.where(np.isfinite(t), t, 0.0))
            reuse = self.reuse_at(count)
            a, b = count / reuse, (n - count) / reuse + 1.0
        a = np.where(t == np.inf, 0.0, np.where(t == -np.inf, 1.0, a))
        b = np.where(t == -np.inf, 0.0, b)
        return ChanceLaw(np.where(np.isnan(t), np.nan, a), np.where(np.isnan(t), np.nan, b))

    def by_block(self) -> Dict:
        return {block: StreamBackground(streams, self.cluster_seconds, self.found_window_seconds,
                                        self.footprint_seconds)
                for block, streams in self._blocks().items()}


def stream_background(scores, streams: Dict[str, np.ndarray], window_seconds: float, cluster_seconds: float,
                      found_window_seconds: float) -> StreamBackground:
    """
    The stream background of a record's noise ``scores``: ``streams`` gives each
    stream's number of windows, slide and block, in the order of the scores, the
    stride they were scored at (``stride_seconds``) and the found window used
    (``found_window_seconds``; if not recorded, `found_window` of the one given).
    Where it also gives where each stream's data start on each detector's timeline
    (``detector_offsets``) and the data that set a score (``footprint_seconds``, the
    padded onsource), the background counts its independent events by the data they
    share (`metrics.disjoint_events`). A record of wrapping slides
    (``ring_lags_seconds``) is rebuilt slide by slide (`per_detector.ring_background_streams`).
    """
    stride = float(np.asarray(streams["stride_seconds"]))
    found_window_seconds = (float(np.asarray(streams["found_window_seconds"])) if "found_window_seconds" in streams
                            else found_window(found_window_seconds, stride))
    footprint = float(np.asarray(streams["footprint_seconds"])) if "footprint_seconds" in streams else None
    if "ring_lags_seconds" in streams:  # wrapping slides (`per_detector.ring_record`)
        from .per_detector import ring_background_streams

        return StreamBackground(ring_background_streams(scores, streams, stride), cluster_seconds,
                                found_window_seconds, footprint)
    edges = np.cumsum(streams["lengths"])[:-1]
    offsets = streams.get("detector_offsets")
    offsets = [None] * len(streams["lengths"]) if offsets is None else [tuple(o) for o in np.asarray(offsets)]
    return StreamBackground(
        [ScoreStream(part, stride, window_seconds, int(block), int(slide), detector_offsets=offset)
         for part, block, slide, offset in zip(np.split(np.asarray(scores), edges), streams["blocks"],
                                                streams["slides"], offsets)],
        cluster_seconds, found_window_seconds, footprint)


# ------------------------------------------------------------ the data

class SyntheticSegments:
    """
    Stream mode's data in synthetic noise: independent segments of continuous
    noise, each ``segment`` of a dataset whose onsource is the segment
    (``ONSOURCE``, one per batch, from the noise stage's seeds), served as streams
    of windows as time slides serve real noise (`slides.TimeSlideBackground`): one
    stream per segment, at slide 0, each its own block. Segment ``b`` stands on a
    timeline of its own from ``b * segment_seconds``: synthetic noise has no GPS time.
    """

    def __init__(self, dataset: "gf.Dataset", geometry: StreamGeometry, segment_seconds: float):
        self.dataset = dataset
        self.geometry = geometry
        self.segment_seconds = float(segment_seconds)
        self.num_samples = int(round(segment_seconds * geometry.sample_rate_hertz))
        if geometry.num_windows(self.num_samples) < 1:
            raise ValueError(f"A segment of {segment_seconds} s holds no window of the model's.")

    def segment(self, index: int) -> np.ndarray:
        """Segment ``index``'s strain, (detectors, samples), float64."""
        _, y = self.dataset[int(index)]
        return np.asarray(y[gf.ReturnVariables.ONSOURCE.name], dtype=np.float64)[0, :, :self.num_samples]

    def batches(self, batch_size: int, max_windows: int) -> Iterator[SlideBatch]:
        """Batches of up to ``batch_size`` windows, ``max_windows`` in all, segment after segment."""
        remaining, index, per_segment = int(max_windows), 0, self.geometry.num_windows(self.num_samples)
        while remaining > 0:
            strain = self.segment(index)
            count = min(per_segment, remaining)
            start = index * self.segment_seconds
            stream = SlideStream((start, start + self.segment_seconds), 0, count, index, self.num_samples)
            for first in range(0, count, batch_size):
                windows = np.arange(first, min(first + batch_size, count))
                onsource, offsource = self.geometry.gather(strain, windows * self.geometry.stride_samples)
                yield SlideBatch(stream, onsource, offsource, np.full((windows.size, strain.shape[0]), np.nan))
            remaining -= count
            index += 1

    def stream_strain(self, stream: SlideStream) -> Tuple[np.ndarray, float]:
        """The stream's strain (its segment) and its first sample's time on the segment's timeline."""
        return self.segment(int(round(stream.stretch[0] / self.segment_seconds))), float(stream.stretch[0])

    def detector_offsets(self, stream: SlideStream, num_detectors: int) -> np.ndarray:
        """
        Where the stream's data start on each detector's timeline (s): segment ``b``'s at
        ``b`` segments and footprints on, so that no two segments' windows seem to share data.
        """
        index = int(round(stream.stretch[0] / self.segment_seconds))
        footprint = self.geometry.footprint_samples / self.geometry.sample_rate_hertz
        return np.full(num_detectors, index * (self.segment_seconds + footprint))

    def ring_segments(self, num_windows: int) -> Iterator[Tuple[np.ndarray, float, SlideStream]]:
        """
        The segments a ring of wrapping slides lays end to end: enough for ``num_windows``
        windows, each its strain, its first sample's time and its zero-lag stream.
        """
        per_segment = self.geometry.num_windows(self.num_samples)
        for index in range(max(math.ceil(num_windows / per_segment), 1)):
            start = index * self.segment_seconds
            yield (self.segment(index), start,
                   SlideStream((start, start + self.segment_seconds), 0, per_segment, index, self.num_samples))


# ------------------------------------------------------- local injections

@dataclass(frozen=True)
class Placement:
    """
    Where each injection of a stage goes: into stream ``stream[k]`` (an index of the
    background's streams) with its merger at that stream's sample ``sample[k]``.
    """
    stream: np.ndarray
    sample: np.ndarray

    def __len__(self) -> int:
        return self.stream.size


def place_injections(num: int, streams: Sequence[SlideStream], half_samples: int,
                     rng: np.random.Generator, geometry: StreamGeometry, reach_seconds: float) -> Placement:
    """
    ``num`` mergers placed in the background's streams, uniformly over the samples
    whose local stretch (``half_samples`` either side) lies inside the stream and
    whose found test (every window within ``reach_seconds`` of the merger) reads only
    windows the background scored, its triggers none it drops at a stream's ends
    (`StreamBackground.trigger_indices`); sorted by stream and time so that each
    stream is read once.
    """
    reach = reach_seconds * geometry.sample_rate_hertz
    first = np.array([max(half_samples, math.ceil(geometry.centre_samples + reach)) for s in streams],
                     dtype=np.int64)
    last = np.array([min(s.num_samples - half_samples,
                         math.floor((s.num_windows - 1) * geometry.stride_samples + geometry.centre_samples - reach))
                     for s in streams], dtype=np.int64)
    room = np.maximum(last - first + 1, 0).astype(np.float64)
    if room.sum() <= 0:
        raise ValueError(f"No stream of the background is long enough for a local stretch of "
                         f"{2 * half_samples} samples within its scored windows: lengthen the streams "
                         "(stream_segment_seconds, or real noise's segments), or score more windows.")
    stream = rng.choice(len(streams), size=num, p=room / room.sum())
    sample = first[stream] + np.floor(rng.random(num) * room[stream]).astype(np.int64)
    order = np.lexsort((sample, stream))
    return Placement(stream[order].astype(np.int64), sample[order])


class StreamReader:
    """
    The noise of a local-injection stage's dataset: a reader of the caller's own
    (`noise.sequential_batcher`) serving, in order, each placed injection's stretch of
    the clean stream (``ONSOURCE``, ``2 half_samples`` centred on the merger: the local
    stretch padded by the window's crop either side, which the dataset crops back) and the
    same clean stretch, uncropped (``OFFSOURCE``: the spectrum the injection is scaled,
    whitened and recorded against, measured on the very noise it is added to and the
    same whatever the model's window),
    with the GPS time of each stretch's first returned sample per detector (NaN in
    synthetic noise). The dataset places each signal's merger at its onsource's centre. The
    last batch is filled by repeating its last stretch; the stage keeps only the
    placed ones.
    """

    def __init__(self, source, streams: Sequence[SlideStream], placement: Placement, geometry: StreamGeometry,
                 half_samples: int, split_spec: Optional["gf.SplitSpec"], step_samples: int = 0,
                 ifos: Sequence = (), crop_samples: int = 0):
        self.source, self.streams, self.placement = source, list(streams), placement
        self.crop_samples = int(crop_samples)  # the dataset crops this much off each end of the stretch
        # The detectors of the data it reads, in order: the dataset's noise detectors.
        self.ifos = list(ifos) or None
        self.geometry, self.half_samples, self.step_samples = geometry, int(half_samples), int(step_samples)
        # The split of the data it reads, for the dataset's split check (a reader of the
        # caller's own is taken to read real data).
        self.split_spec = split_spec

    def __deepcopy__(self, memo):
        return self  # its source holds data, read in order: one reader, never copied

    def stretches(self) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Each placed injection's (stretch (I, 2h), offsource (I, O), first-sample GPS (I,)), in order."""
        h, crop = self.half_samples, self.crop_samples
        loaded = None
        for k in range(len(self.placement)):
            index, m = int(self.placement.stream[k]), int(self.placement.sample[k])
            if loaded is None or loaded[0] != index:
                loaded = (index, *self.source.stream_strain(self.streams[index]))
            _, strain, first_gps = loaded
            stretch = strain[:, m - h:m + h]
            offsource = strain[:, m - h + crop:m + h - crop]  # the returned stretch, clean
            slide = self.streams[index].slide
            # The GPS time of each detector's first returned (cropped) sample, as an obtainer gives it.
            gps = first_gps + (m - h + self.crop_samples + np.arange(strain.shape[0]) * slide * self.step_samples) \
                / self.geometry.sample_rate_hertz
            yield stretch, offsource, gps

    def __call__(self, num_examples_per_batch: int, **_) -> Iterator[Dict]:
        RV = gf.ReturnVariables
        pending = []
        for item in self.stretches():
            pending.append(item)
            if len(pending) == num_examples_per_batch:
                yield self._batch(pending)
                pending = []
        if pending:
            yield self._batch(pending + [pending[-1]] * (num_examples_per_batch - len(pending)))

    @staticmethod
    def _batch(items) -> Dict:
        RV = gf.ReturnVariables
        stretch, offsource, gps = (np.stack(values) for values in zip(*items))
        # float64, as the stream was read: the dataset narrows it where it computes.
        return {RV.ONSOURCE: stretch.astype(np.float64), RV.OFFSOURCE: offsource.astype(np.float64),
                RV.START_GPS_TIME: gps}


def local_windows(geometry: StreamGeometry, half_samples: int, sample: int, reach_seconds: float
                  ) -> Tuple[np.ndarray, np.ndarray]:
    """
    The windows the found test of a merger at stream sample ``sample`` reads (every
    window within ``reach_seconds`` of it, `StreamGeometry.windows_near`): their
    indices on the stream's grid, and where their data start in the local stretch
    (``half_samples`` either side of the merger).
    """
    first, last = geometry.windows_near(sample, reach_seconds)
    index = np.arange(first, last + 1)
    return index, index * geometry.stride_samples - (sample - half_samples)


def found_in_stretch(geometry: StreamGeometry, index: np.ndarray, scores: np.ndarray, sample: int,
                     cluster_seconds: float, found_window_seconds: float) -> Tuple[float, float]:
    """
    One injection's found statistic from its local windows' scores (``index`` on the
    stream's grid): the loudest trigger within the found window of its merger at
    ``sample``, and that trigger's time from the merger (s); ``-inf`` and NaN for none.
    """
    times = geometry.window_time(index)
    keep = cluster(times, scores, cluster_seconds)
    merger = sample / geometry.sample_rate_hertz
    stat, when = loudest_within(times[keep], np.asarray(scores)[keep], [merger], found_window_seconds)
    return float(stat[0]), float(when[0] - merger)


__all__ = [
    "Placement",
    "StreamBackground",
    "StreamGeometry",
    "StreamReader",
    "SyntheticSegments",
    "cluster",
    "found_in_stretch",
    "found_window",
    "local_windows",
    "loudest_within",
    "place_injections",
    "stream_background",
    "stream_triggers",
]
