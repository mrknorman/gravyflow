"""
Noise mode data acquisition.

This module provides NoiseDataObtainer for acquiring random or grid-sampled
noise data from long IFO segments (NOISE mode).
"""

import logging
from itertools import count
from dataclasses import dataclass, field
from typing import Any, List, Dict, Optional, Sequence, Tuple, Union

import numpy as np
from keras import ops

import gravyflow as gf
from gravyflow.src.dataset.config import WindowSpec
from .base import (
    BaseDataObtainer, DataQuality, DataLabel, SegmentOrder, Coincidence,
    AcquisitionMode, SamplingMode, ObservingRun, IFOData, SEGMENT_GRID_ORIGIN_GPS,
    WINDOW_EDGE_BUFFER_SAMPLES, label_kinds, separable,
)
from .gwosc_data import intersect_segments
from gravyflow.src.dataset.store import read as store_read
from gravyflow.src.dataset.store.read import interiors
from gravyflow.src.dataset.store.prefetch import Prefetcher
from gravyflow.src.dataset.store.tile import TileKey
from gravyflow.src.dataset.store.view import UnheldSamples
from gravyflow.src.utils.resample import whole_hertz
from gravyflow.src.rng import DIRECT_SPLIT, BatchStreams, SeedTree
from .segment import SegmentCollection, NoiseSegment
from gravyflow.src.utils.numerics import ensure_list
from gravyflow.src.dataset.features.injection import ReturnVariables as RV


def couple_pieces(measures: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rows of one piece per detector, in which each detector's pieces take part in
    proportion to ``measures``.

    Each detector's pieces, in the order given, tile [0, 1) in proportion to their
    measures. The edges of every detector's tiles cut [0, 1) into intervals; each
    interval is a row, holding for each detector the piece whose tile contains it, so
    first pieces go with first pieces (the comonotone coupling). Every piece is in a
    row, and a piece's rows add up to its tile: at most ``sum(n_i) - k + 1`` rows for
    k detectors with n_i pieces.

    Returns, for each row: each detector's piece index (R, k), and the row's weight,
    its interval's width (R,).
    """
    tiles = []
    for measure in measures:
        measure = np.asarray(measure, dtype=np.float64)
        tile = np.concatenate([[0.0], np.cumsum(measure) / np.sum(measure)])
        tile[-1] = 1.0
        tiles.append(tile)
    edges = np.unique(np.concatenate(tiles))
    lower, upper = edges[:-1], edges[1:]
    # Edges equal but for rounding leave slivers; every tile is far wider.
    keep = upper - lower > 1e-12
    lower, upper = lower[keep], upper[keep]
    middle = (lower + upper) / 2.0

    index = np.stack([
        np.clip(np.searchsorted(tile, middle, side="right") - 1, 0, len(tile) - 2)
        for tile in tiles
    ], axis=1)
    return index, upper - lower


@dataclass(frozen=True)
class SamplingPlan:
    """
    How real-noise windows are drawn: the rows to load, and each row's budget.

    Within each bin, each detector's window starts are drawn uniformly over the starts
    of all its usable pieces, so no piece is sampled more densely than another; the
    bin's number of examples is set by the detector with the fewest starts, which is
    sampled at the nominal density and the others at or below it.

    - ``rows``: (M, detectors, 2) GPS pieces to load. A piece's rows together hold
      its share of the bin, and each draws over the whole piece.
- ``start_seconds``: (M,) the row's budget: the fewest seconds of window starts
      any detector owns in it. A row's expected number of examples is this times the
      nominal density.
    """
    rows: np.ndarray
    start_seconds: np.ndarray
    #: Rows left out because their detectors' windows cannot start far enough apart.
    inseparable: int = 0


#: How many consecutive row visits a RANDOM batch draws its examples' rows from.
MIXING_VISITS = 8
#: Joint draws of an example's row and window starts before the example is refused.
SEPARATION_ROUNDS = 64


class SeparationRefused(ValueError):
    """No example of a batch's rows has separated windows in ``SEPARATION_ROUNDS`` draws."""


@dataclass(frozen=True, eq=False)
class RandomPlan:
    """
    The RANDOM batches of one split, addressed by index: which rows batch g draws from.

    Row j is due ``expected[j]`` batches a pass, its start-seconds at the nominal
    density, and serves floor((p + 1) e_j + u_j) - floor(p e_j + u_j) of them in pass p,
    u_j drawn once (``noise/accrual``), so over any number of passes it serves its
    density to within one batch. Pass p visits its due rows in its own order
    (``noise/segment_order`` at epoch p, a new one every pass, unless the obtainer's
    ``segment_order`` fixes it), each for its batches; the passes run on without end.
    Batch g lies in the pass p whose earlier batches, sum_j floor(p e_j + u_j), number
    at most g. Every count is arithmetic on piece lengths: nothing is read to address a
    batch, and the epoch of the order streams counts passes, not training epochs.

    A batch draws each example's row from the ``MIXING_VISITS`` visits from its own on
    (``visits``), so no row feeds a run of batches alone.
    """
    tree: SeedTree
    split: str
    rows: np.ndarray  # (M, I, 2) GPS pieces
    firsts: np.ndarray  # (M, I) absolute index of the first sample read of each piece
    counts: np.ndarray  # (M, I) samples read of each piece
    start_seconds: np.ndarray  # (M,) each row's budget, which weighs its draws
    expected: np.ndarray  # (M,) batches due a pass
    accrual: np.ndarray  # (M,) u_j
    reorder: bool  # a new order every pass
    _orders: Dict[int, np.ndarray] = field(default_factory=dict, repr=False)

    @classmethod
    def build(cls, tree: SeedTree, split: str, plan: SamplingPlan, firsts, counts,
              batches_per_start_second: float, reorder: bool) -> "RandomPlan":
        expected = plan.start_seconds * batches_per_start_second
        if not np.sum(expected) > 0:
            raise ValueError(f"The {split!r} split's RANDOM plan has no row with window starts to draw.")
        accrual = tree.stream(split, "noise/accrual", epoch=0).random(len(expected))
        return cls(tree, split, plan.rows, firsts, counts, plan.start_seconds, expected, accrual, reorder)

    def batches_before(self, p: int) -> int:
        """The batches of the passes before pass ``p``."""
        return int(np.sum(np.floor(p * self.expected + self.accrual)))

    def due(self, p: int) -> np.ndarray:
        """Each row's batches in pass ``p``."""
        return (np.floor((p + 1) * self.expected + self.accrual)
                - np.floor(p * self.expected + self.accrual)).astype(np.int64)

    def locate(self, g: int) -> Tuple[int, int]:
        """The pass batch ``g`` lies in, and its position there."""
        lo, hi = 0, int(np.ceil((g + len(self.expected) + 1) / np.sum(self.expected))) + 1
        while hi - lo > 1:  # batches_before(lo) <= g < batches_before(hi)
            middle = (lo + hi) // 2
            lo, hi = (middle, hi) if self.batches_before(middle) <= g else (lo, middle)
        return lo, g - self.batches_before(lo)

    def order(self, p: int) -> np.ndarray:
        """The order pass ``p`` visits the rows in."""
        if p not in self._orders:
            self._orders[p] = (self.tree.stream(self.split, "noise/segment_order", epoch=p).permutation(len(self.rows))
                               if self.reorder else np.arange(len(self.rows)))
        return self._orders[p]

    def visits(self, g: int, width: int = MIXING_VISITS) -> np.ndarray:
        """The rows of the ``width`` visits from batch ``g``'s on: its own row first."""
        p, position = self.locate(int(g))
        order = self.order(p)
        due = self.due(p)[order]
        rows = list(order[due > 0][np.searchsorted(np.cumsum(due[due > 0]), position, side="right"):][:width])
        while len(rows) < width:
            p += 1
            order = self.order(p)
            rows.extend(order[self.due(p)[order] > 0][:width - len(rows)])
        return np.asarray(rows, dtype=np.int64)


def _start_on_grid(rng: np.random.Generator, lo: int, hi: int, quantum: int, phase: int) -> Optional[int]:
    """A start uniform over the integers x of [lo, hi) with x = phase mod ``quantum``; None if none."""
    first, last = -(-(lo - phase) // quantum), (hi - 1 - phase) // quantum
    return None if last < first else phase + quantum * int(rng.integers(first, last + 1))


def draw_examples(
        plan: RandomPlan,
        streams: BatchStreams,
        g: int,
        ifos: Sequence["gf.IFO"],
        num_examples: int,
        num_onsource_samples: int,
        num_offsource_samples: int,
        separation_samples: float,
        coincident: bool,
        quantum: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    The windows of batch ``g`` (at ``streams``): exactly ``num_examples`` examples, each
    drawn jointly, its row from the batch's visits (``RandomPlan.visits``) weighted by
    start-seconds (``noise/row``), then each detector's onsource start uniform over its
    piece (``noise/start/{IFO}``), redrawn whole until no two detectors start closer than
    ``separation_samples`` in absolute time, or, ``coincident``, every detector starting
    at one time. A window's first sample lies on the grid of ``quantum`` samples the
    read needs (``store.read``).

    Returns, per example: its row (B,), each detector's onsource start as an absolute
    sample index (B, I), and its rounds of draws (B,).

    Raises:
        SeparationRefused: an example found no window in ``SEPARATION_ROUNDS`` rounds,
            naming the rows it drew from.
    """
    visits = plan.visits(g)
    weights = plan.start_seconds[visits] / np.sum(plan.start_seconds[visits])
    row_stream = streams.stream("noise/row")
    start_streams = [streams.stream(f"noise/start/{ifo.name}") for ifo in ifos]
    lo, phase = num_offsource_samples, num_offsource_samples % quantum
    hi = plan.counts - num_onsource_samples - WINDOW_EDGE_BUFFER_SAMPLES
    rows = np.empty(num_examples, dtype=np.int64)
    starts = np.empty((num_examples, len(ifos)), dtype=np.int64)
    rounds = np.zeros(num_examples, dtype=np.int64)
    for example in range(num_examples):
        for attempt in range(1, SEPARATION_ROUNDS + 1):
            j = int(visits[row_stream.choice(len(visits), p=weights)])
            firsts = plan.firsts[j]
            if coincident:
                at = _start_on_grid(start_streams[0], int(np.max(firsts + lo)), int(np.min(firsts + hi[j])),
                                    quantum, phase)
                drawn = None if at is None else np.full(len(ifos), at, dtype=np.int64)
            else:
                offsets = [_start_on_grid(stream, lo, int(hi[j, i]), quantum, phase)
                           for i, stream in enumerate(start_streams)]
                drawn = None if None in offsets else firsts + np.array(offsets, dtype=np.int64)
                if drawn is not None and np.any(
                        np.abs(drawn[:, None] - drawn[None, :])[np.triu_indices(len(ifos), 1)] < separation_samples):
                    drawn = None
            if drawn is not None:
                rows[example], starts[example], rounds[example] = j, drawn, attempt
                break
        else:
            raise SeparationRefused(
                f"Batch {g} of the {plan.split!r} split found no example in {SEPARATION_ROUNDS} draws "
                f"among rows {sorted(set(visits.tolist()))} (GPS pieces {plan.rows[visits].tolist()}): "
                + ("their detectors share no start." if coincident
                   else f"their detectors cannot start {separation_samples} samples apart.")
            )
    return rows, starts, rounds


class NoiseDataObtainer(BaseDataObtainer):
    """
    Data obtainer for NOISE mode acquisition.
    
    Acquires random or grid-sampled chunks from long IFO data segments,
    suitable for training noise-only models or generating background samples.
    """
    
    def __init__(
            self,
            data_quality: DataQuality,
            data_labels: Union[DataLabel, List[DataLabel]],
            observing_runs: Union[ObservingRun, List[ObservingRun]] = None,
            segment_order: SegmentOrder = SegmentOrder.RANDOM,
            max_segment_duration_seconds: float = 2048.0,
            saturation: float = 8.0,
            overrides: dict = None,
            logging_level: int = logging.WARNING,
            augmentations: List = None,  # Training only; None is none
            gwosc_flags: Optional[Tuple[str, ...]] = None,
            split_spec=None,
            source=None,
            coincidence: Coincidence = Coincidence.SEPARATED,
            exclusions=(),
        ):
        if DataLabel.EVENTS in label_kinds(data_labels):
            raise ValueError(
                "Real events are not noise: noise vetoes every event, in every split. Serve "
                "events with a transient obtainer (data_labels=[DataLabel.EVENTS]), as test data."
            )

        super().__init__(
            data_quality=data_quality,
            data_labels=data_labels,
            observing_runs=observing_runs,
            segment_order=segment_order,
            max_segment_duration_seconds=max_segment_duration_seconds,
            saturation=saturation,
            overrides=overrides,
            logging_level=logging_level,
            augmentations=augmentations,
            gwosc_flags=gwosc_flags,
            split_spec=split_spec,
            source=source,
            coincidence=coincidence,
            exclusions=exclusions,
        )

        self.acquisition_mode = AcquisitionMode.NOISE

    def _served_state(self) -> Dict[str, Any]:
        return {
            **super()._served_state(),
            # The split's RANDOM plans, by their window and batch (``random_plan``).
            "_random_plans": {},
        }

    # =========================================================================
    # NOISE-SPECIFIC SEGMENT PROCESSING
    # =========================================================================

    def remove_unwanted_segments(
            self,
            ifo: gf.IFO,
            valid_segments: np.ndarray,
        ) -> np.ndarray:
        """
        Remove every published GW event and Gravity Spy glitch from noise segments.

        Every event's footprint and guard is vetoed (``get_event_veto_segments``):
        events are test data, never noise. Unless the obtainer's labels include
        GLITCHES, each glitch's whole extent is vetoed too
        (``get_glitch_veto_segments``). Both cover every configured run. Returns the
        vetoed segments, sorted and coalesced.
        """
        veto_segments = [self.get_event_veto_segments()]
        if DataLabel.GLITCHES not in label_kinds(self.data_labels):
            veto_segments.append(self.get_glitch_veto_segments(ifo))
        return self.veto_time_segments(
            valid_segments,
            np.concatenate([np.reshape(veto, (-1, 2)) for veto in veto_segments]),
        )

    def plan_sampling(
            self,
            segments: np.ndarray,
            minimum_duration_seconds: float,
            footprint_seconds: float,
            separation_seconds: float = 0.0,
            *,
            sample_rate_hertz: float,
        ) -> SamplingPlan:
        """
        The plan for drawing windows from ``segments`` (rows from ``get_valid_segments``).

        Within each ``max_segment_duration_seconds`` bin, each detector's pieces of at
        least ``minimum_duration_seconds`` are usable; a bin where some detector has
        none is skipped. A piece is measured as it is read at ``sample_rate_hertz``:
        its interior (``store.read.interiors``), the whole seconds whose read lies
        inside it, and its window starts span that length less ``footprint_seconds``. The usable pieces are re-paired with ``couple_pieces``
        in proportion to those starts, longest first, so every usable piece is drawn
        from and no short piece is drawn more densely than a long one. Bins keep the
        order in which ``segments`` first reach them.

        A row whose detectors' window starts cannot be ``separation_seconds`` apart
        (``separable``) is left out and counted: it could never be drawn from without
        coincident detectors.
        """
        segments = np.reshape(np.asarray(segments, dtype=np.float64), (len(segments), -1, 2))
        num_ifos = segments.shape[1]
        bins = np.floor(segments[:, 0, 0] / self.max_segment_duration_seconds)
        _, first = np.unique(bins, return_index=True)

        rows, start_seconds, inseparable = [], [], 0
        for bin_index in bins[np.sort(first)]:
            in_bin = segments[bins == bin_index]
            usable, origins, starts = [], [], []
            for j in range(num_ifos):
                pieces = np.unique(in_bin[:, j, :], axis=0)
                firsts, counts = interiors(self.source, sample_rate_hertz, pieces)
                lengths = counts / sample_rate_hertz
                keep = lengths >= minimum_duration_seconds
                pieces, lengths, firsts = pieces[keep], lengths[keep], firsts[keep]
                order = np.argsort(-lengths, kind="stable")
                usable.append(pieces[order])
                origins.append(firsts[order] / sample_rate_hertz)
                starts.append(lengths[order] - footprint_seconds)
            if any(len(pieces) == 0 for pieces in usable):
                continue
            index, weights = couple_pieces(starts)
            bin_rows = np.stack([usable[j][index[:, j]] for j in range(num_ifos)], axis=1)
            keep = np.array([
                separable([(origins[j][index[k, j]], origins[j][index[k, j]] + starts[j][index[k, j]])
                           for j in range(num_ifos)], separation_seconds)
                for k in range(len(bin_rows))
            ], dtype=bool)
            inseparable += int(np.sum(~keep))
            rows.append(bin_rows[keep])
            start_seconds.append((weights * min(np.sum(s) for s in starts))[keep])

        if not sum(len(r) for r in rows):
            return SamplingPlan(np.empty((0, num_ifos, 2)), np.empty(0), inseparable)
        return SamplingPlan(np.concatenate(rows), np.concatenate(start_seconds), inseparable)

    def calculate_bin_indices(self, segments, interval, start_period):
        """Calculate bin indices for segments based on time interval."""
        relative_starts = (segments[:, 0] - start_period) / interval
        relative_ends = (segments[:, 1] - start_period) / interval

        start_bins = np.floor(relative_starts).astype(int)
        end_bins = np.ceil(relative_ends).astype(int)

        return start_bins, end_bins

    def merge_bins(self, new_segments_list, interval):
        """
        Group each IFO's segments into absolute bins of ``interval`` seconds
        ([interval k, interval (k + 1)) in GPS), clipping them to the bin.

        Returns one entry per bin in which every IFO has data: a list holding, for each
        IFO, its (M, 2) pieces in that bin. The segments may be in any order.
        """
        non_empty = [np.reshape(s, (-1, 2)) for s in new_segments_list if np.size(s) > 0]
        if len(non_empty) < len(new_segments_list) or not non_empty:
            return []
        min_time = min(float(np.min(segment[:, 0])) for segment in non_empty)
        max_time = max(float(np.max(segment[:, 1])) for segment in non_empty)

        bins = np.arange(
            np.floor(min_time / interval) * interval, 
            np.ceil(max_time / interval) * interval, interval
        )

        result = [[] for _ in bins]

        for i, bin_start in enumerate(bins):
            for segments in non_empty:
                in_bin = segments[
                    (segments[:, 0] < bin_start + interval) 
                    & (segments[:, 1] > bin_start)
                ]
                if in_bin.size > 0:
                    in_bin[:, 0] = np.maximum(in_bin[:, 0], bin_start)
                    in_bin[:, 1] = np.minimum(
                        in_bin[:, 1], bin_start + interval
                    )
                    result[i].append(in_bin)
                else:
                    result[i].append(np.empty((0, 2)))
                    
        filtered_result = [
            bin for bin in result if all(len(arr) > 0 for arr in bin)
        ]

        return filtered_result
    
    def all_segments_per_bin(self, filtered_bins_result):
        """
        Every piece of every bin, as rows of one piece per IFO.

        Each IFO's pieces in a bin are sorted longest first and paired by
        ``couple_pieces`` in proportion to their lengths, so every piece of every IFO
        is in a row and the longest pieces go together. ``plan_sampling`` re-pairs
        them once the window, and so which pieces are usable, is known.

        Returns an array of shape (num_ifos, num_rows, 2).
        """
        if not filtered_bins_result:
            return np.empty((0, 0, 2))

        num_ifos = len(filtered_bins_result[0])
        rows_per_ifo = [[] for _ in range(num_ifos)]

        for bin in filtered_bins_result:
            ordered = []
            for segments in bin:
                segments = np.reshape(segments, (-1, 2))
                durations = segments[:, 1] - segments[:, 0]
                ordered.append(segments[np.argsort(-durations, kind="stable")])
            index, _ = couple_pieces([segments[:, 1] - segments[:, 0] for segments in ordered])
            for j, segments in enumerate(ordered):
                rows_per_ifo[j].extend(segments[index[:, j]])

        return np.array([np.array(rows) for rows in rows_per_ifo])

    # =========================================================================
    # ABSTRACT METHOD IMPLEMENTATIONS
    # =========================================================================

    def available_segments(self, ifo: gf.IFO, group_name: str) -> np.ndarray:
        """
        ``ifo``'s noise of split ``group_name``: its segments inside the split's
        admissible time (``SplitSpec.admissible``: its absolute blocks, less the
        guard at every edge with another split), less every vetoed event and
        glitch (``remove_unwanted_segments``). Sorted (N, 2) GPS intervals.
        """
        return self.split_spec.admissible(self.noise_admissible_segments(ifo), group_name)

    def coincident_segments(self, ifos: List[gf.IFO], group_name: str) -> np.ndarray:
        """The time of split ``group_name`` every IFO's noise holds (``available_segments``)."""
        common = self.available_segments(ifos[0], group_name)
        for ifo in ifos[1:]:
            common = intersect_segments(common, self.available_segments(ifo, group_name))
        return common

    def get_valid_segments(
            self,
            ifos: List[gf.IFO],
            seed: int,
            group_name: str,
            segment_order: SegmentOrder = None,
        ) -> List:
        """
        The segments of split ``group_name``: each IFO's segments inside the
        split's admissible time (``SplitSpec.admissible``: its absolute blocks,
        less the guard at every edge with another split), vetoed, cut on the
        absolute bin grid, and paired across IFOs bin by bin. Under the
        ``COINCIDENT`` opt-in every IFO's segments are the time they all hold
        (``coincident_segments``), so each row is one piece for every IFO.
        """
        ifos = ensure_list(ifos)

        if not segment_order:
            segment_order = self.segment_order

        # The served segments are one split's, in one order: another rebuilds them.
        if self._serve_split(self._split(ifos, group_name), order=(seed, segment_order)):
            self.valid_segments = []
            common = None
            if self.coincidence == Coincidence.COINCIDENT:
                common = self.coincident_segments(ifos, group_name)

            for ifo in ifos:
                valid_segments = self.available_segments(ifo, group_name) if common is None else common

                # Cut on the absolute grid [d k, d (k + 1)) in GPS (d = 2048 s by
                # default), the grid merge_bins uses, so whole blocks survive.
                valid_segments = self.cut_segments(
                    valid_segments,
                    self.max_segment_duration_seconds,
                    SEGMENT_GRID_ORIGIN_GPS
                )

                if len(valid_segments) == 0:
                    raise ValueError(f"IFO {ifo} has found no valid segments (NOISE mode)!")

                self.valid_segments.append(valid_segments)

            # Merge and process segments
            self.valid_segments = self.merge_bins(
                self.valid_segments,
                self.max_segment_duration_seconds
            )

            if not self.valid_segments:
                raise ValueError(
                    f"IFOs {[ifo.name for ifo in ifos]} share no "
                    f"{self.max_segment_duration_seconds} s bin with data (NOISE mode)!"
                )

            # Keep every piece of every bin, not just the longest.
            self.valid_segments = self.all_segments_per_bin(
                self.valid_segments
            )
                    
            self.valid_segments = np.swapaxes(self.valid_segments, 1, 0)
            
            self.valid_segments = self.order_segments(
                self.valid_segments, 
                segment_order,
                seed
            )

        return self.valid_segments

    def get_segments_as_collection(
        self, 
        ifo_index: int = 0,
        ifo: gf.IFO = None
    ) -> SegmentCollection:
        """
        Get valid_segments as a SegmentCollection for unified access.
        
        Converts the internal np.ndarray representation to SegmentCollection,
        providing object-oriented access to segment properties.
        
        Args:
            ifo_index: Index of IFO to extract (if valid_segments is 3D)
            ifo: Optional IFO to associate with segments
            
        Returns:
            SegmentCollection with NoiseSegment objects
        """
        if self.valid_segments is None or len(self.valid_segments) == 0:
            return SegmentCollection()
        
        # valid_segments can be (N, IFO, 2) or (N, 2)
        if self.valid_segments.ndim == 3:
            segments_2d = self.valid_segments[:, ifo_index, :]
        else:
            segments_2d = self.valid_segments
        
        return SegmentCollection.from_array(segments_2d, ifo=ifo)

    def _yield_onsource_offsource_chunks(
            self,
            sample_rate_hertz: float = None,
            onsource_duration_seconds: float = None,
            crop_duration_seconds: float = None,
            offsource_duration_seconds: float = None,
            num_examples_per_batch: int = None,
            ifos: List[gf.IFO] = None,
            seed: int = None,
            sampling_mode: SamplingMode = SamplingMode.RANDOM,
            whiten: bool = False,
            crop: bool = False,
            group: str = None
        ):
        """
        Generate onsource/offsource chunks for NOISE mode.
        
        Uses random or grid sampling from long data segments.
        
        Args:
            sample_rate_hertz: Sample rate in Hz
            onsource_duration_seconds: Final onsource window duration
            crop_duration_seconds: Cropping duration on each side (for edge effects)
            offsource_duration_seconds: Background window duration
            num_examples_per_batch: Batch size
            ifos: List of interferometers
            seed: Random seed
            sampling_mode: RANDOM or GRID sampling
            group: The split served (see _serve_group); None serves the
                segments already held, built or set by hand
        """
        # Create WindowSpec for consistent parameter handling
        window_spec = WindowSpec.from_params(
            sample_rate_hertz=sample_rate_hertz,
            onsource_duration_seconds=onsource_duration_seconds,
            offsource_duration_seconds=offsource_duration_seconds,
            crop_duration_seconds=crop_duration_seconds,
        )
        
        # Apply remaining defaults (not in WindowSpec)
        ifos = ensure_list(ifos) or [gf.IFO.L1]
        num_examples_per_batch = num_examples_per_batch or gf.Defaults.num_examples_per_batch
        seed = gf.src.rng.check_root(seed if seed is not None else gf.Defaults.seed)

        # With no group, the segments already held: built, or set by hand.
        if group is not None:
            self._serve_group(ifos, seed, group)
        elif self.valid_segments is None:
            raise ValueError("Name the split to serve: group='train', 'validate' or 'test'.")

        # Early validation
        if self.valid_segments is None or len(self.valid_segments) == 0:
            logging.warning("No valid segments available in NOISE mode. Generator will be empty.")
            return

        if not hasattr(self.valid_segments, 'ndim') or self.valid_segments.ndim != 3:
            logging.warning(f"valid_segments has unexpected shape. Expected 3D array. Generator will be empty.")
            return

        tree = SeedTree(seed)
        if sampling_mode == SamplingMode.GRID:
            plan = self.build_sampling_plan(window_spec)
            if len(plan.rows) == 0:
                logging.warning("No valid segments available in NOISE mode. Generator will be empty.")
                return
            emit = lambda onsource, offsource, gps: self._emit(window_spec, onsource, offsource, gps, whiten, crop)
            yield from self._grid_pass(window_spec, num_examples_per_batch, self._separation_seconds(window_spec),
                                       ifos, emit, self.coincidence == Coincidence.COINCIDENT)
            return

        batch = self.random_batcher(tree, window_spec, num_examples_per_batch, ifos, whiten, crop)
        for k in count():
            yield batch(tree.batch(self._served_group, 0, k) if self._served_group else tree.direct(k))

    def _separation_seconds(self, window_spec: WindowSpec) -> float:
        """
        How far apart an example's detectors start. A window's footprint runs from its
        offsource to its padded onsource's end. Detectors are never coincident: an
        example's windows start at least two footprints apart, so no two of them share
        a moment. Under the opt-in they start at one time.
        """
        if self.coincidence == Coincidence.COINCIDENT:
            return 0.0
        return 2.0 * (window_spec.num_onsource_samples + window_spec.num_offsource_samples) \
            / window_spec.sample_rate_hertz

    def build_sampling_plan(self, window_spec: WindowSpec) -> SamplingPlan:
        """The sampling plan of the served segments for windows of ``window_spec`` (``plan_sampling``)."""
        window_seconds = window_spec.total_onsource_duration_seconds + window_spec.offsource_duration_seconds
        footprint_seconds = (window_spec.num_onsource_samples + window_spec.num_offsource_samples) \
            / window_spec.sample_rate_hertz
        self.sampling_plan = self.plan_sampling(
            self.valid_segments,
            minimum_duration_seconds=2.0 * window_seconds,
            footprint_seconds=footprint_seconds + WINDOW_EDGE_BUFFER_SAMPLES / window_spec.sample_rate_hertz,
            separation_seconds=self._separation_seconds(window_spec),
            sample_rate_hertz=window_spec.sample_rate_hertz,
        )
        if self.sampling_plan.inseparable:
            logging.info(f"{self.sampling_plan.inseparable} rows left out: their detectors' "
                         "windows cannot start two footprints apart.")
        return self.sampling_plan

    def random_plan(self, tree: SeedTree, window_spec: WindowSpec, num_examples_per_batch: int) -> RandomPlan:
        """
        The served split's RANDOM plan (``RandomPlan``) for windows of ``window_spec``
        and batches of ``num_examples_per_batch``: its rows in time order, each piece
        measured as it is read, each row due its start-seconds at the nominal density
        of ``saturation`` examples per onsource duration of window starts.
        """
        split = self._served_group or DIRECT_SPLIT
        key = (tree, split, window_spec, num_examples_per_batch)
        if key not in self._random_plans:
            sampling = self.build_sampling_plan(window_spec)
            if len(sampling.rows) == 0:
                raise ValueError(f"The {split!r} split has no row long enough for a window "
                                 f"of {window_spec.total_onsource_duration_seconds + window_spec.offsource_duration_seconds} s.")
            order = np.lexsort(np.reshape(sampling.rows, (len(sampling.rows), -1)).T[::-1])
            sampling = SamplingPlan(sampling.rows[order], sampling.start_seconds[order], sampling.inseparable)
            firsts, counts = interiors(self.source, window_spec.sample_rate_hertz, np.reshape(sampling.rows, (-1, 2)))
            shape = sampling.rows.shape[:2]
            self._random_plans[key] = RandomPlan.build(
                tree, split, sampling, np.reshape(firsts, shape), np.reshape(counts, shape),
                self.saturation / (num_examples_per_batch * window_spec.onsource_duration_seconds),
                reorder=self.segment_order == SegmentOrder.RANDOM,
            )
        return self._random_plans[key]

    def random_batcher(
            self,
            tree: SeedTree,
            window_spec: WindowSpec,
            num_examples_per_batch: int,
            ifos: List[gf.IFO],
            whiten: bool = False,
            crop: bool = False,
            steps_per_epoch: Optional[int] = None,
        ):
        """
        The served split's RANDOM batches, addressed by index: ``batch(streams)`` is a
        pure function of the plan (``random_plan``) and its streams' address, (epoch,
        k), which is batch ``g = epoch * steps_per_epoch + k`` of the split's passes.
        Its windows (``draw_examples``) are read through the store, each its own read,
        and the batches after it are read ahead (``store.prefetch``,
        ``prefetch_depth`` batches on ``prefetch_workers`` threads), which never changes
        a batch.
        """
        plan = self.random_plan(tree, window_spec, num_examples_per_batch)
        rate = window_spec.sample_rate_hertz
        on, off = window_spec.num_onsource_samples, window_spec.num_offsource_samples
        separation = self._separation_seconds(window_spec) * rate
        quantum = store_read.geometry(self.source, rate).up
        coincident = self.coincidence == Coincidence.COINCIDENT
        stream = self._served_split

        def windows(address):
            streams = BatchStreams(*address)
            epoch = streams.epoch or 0
            if epoch and steps_per_epoch is None:
                raise ValueError("Batches of epochs after the first need the epoch's steps (steps_per_epoch).")
            g = epoch * (steps_per_epoch or 0) + streams.k
            _, starts, _ = draw_examples(plan, streams, g, ifos, num_examples_per_batch, on, off, separation,
                                         coincident, quantum)
            onsource, offsource = self.read_windows(ifos, rate, starts, on, off)
            return onsource, offsource, starts / rate

        prefetcher = Prefetcher(windows, name=plan.split, depth=self.prefetch_depth, workers=self.prefetch_workers)

        def batch(streams: BatchStreams) -> Dict:
            address = (streams.tree, streams.split, streams.epoch, streams.k, streams.prefix)
            ahead = [address[:3] + (k,) + address[4:] for k in range(streams.k + 1, streams.k + 1 + prefetcher.depth)
                     if steps_per_epoch is None or k < steps_per_epoch]
            onsource, offsource, gps = prefetcher.get(address, ahead)
            return self._emit(window_spec, ops.convert_to_tensor(onsource), ops.convert_to_tensor(offsource), gps,
                              whiten, crop)

        def held(ifo, start: int) -> None:
            """Commit every tile one window's read needs; refuse, as its read would, if it is not held."""
            g = store_read.geometry(self.source, rate)
            span_start, span_stop = store_read.read_span(g, start - off, off + on)
            seconds = whole_hertz(self.source.tile_seconds(g.native))
            length = g.native * seconds
            store = store_read.process_store()
            for index in range(span_start // length, (span_stop - 1) // length + 1):
                key = TileKey(self.source.source_id(index * seconds, g.native), ifo.name, g.native, seconds, index)
                tile = store_read._tile(store, self.source, ifo, key, stream)
                try:
                    tile.view(max(span_start, key.start), min(span_stop, key.start + key.length))
                except UnheldSamples:
                    # The read itself names the contradiction and the exclusion around it.
                    store_read.read(self.source, ifo, rate, start - off, off + on, stream=stream)

        def prepare(addresses: Sequence[BatchStreams]) -> List[store_read.TimelineContradiction]:
            """Every contradiction the windows of ``addresses`` meet, each tile committed once."""
            met = {}
            for streams in addresses:
                g = (streams.epoch or 0) * (steps_per_epoch or 0) + streams.k
                _, starts, _ = draw_examples(plan, streams, g, ifos, num_examples_per_batch, on, off, separation,
                                             coincident, quantum)
                for example in starts:
                    for ifo, start in zip(ifos, example):
                        try:
                            held(ifo, int(start))
                        except store_read.TimelineContradiction as error:
                            met.setdefault(error.exclusion, error)
            return list(met.values())

        batch.prefetcher = prefetcher
        batch.plan = plan
        batch.prepare = prepare
        return batch

    def read_windows(self, ifos: List[gf.IFO], sample_rate_hertz: float, starts: np.ndarray,
                     num_onsource_samples: int, num_offsource_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Each example's windows, read through the store (``store.read``), one read per
        window: (B, I, on) onsource from each start (absolute sample indices, (B, I)),
        and the (B, I, off) offsource just before it. The one place RANDOM reads strain.
        """
        stream = self._served_split
        data = np.array([[store_read.read(self.source, ifo, sample_rate_hertz, int(start) - num_offsource_samples,
                                          num_offsource_samples + num_onsource_samples, stream=stream).data
                          for ifo, start in zip(ifos, example)] for example in starts], dtype=np.float32)
        return data[..., num_offsource_samples:], data[..., :num_offsource_samples]

    def _emit(self, window_spec: WindowSpec, onsource, offsource, start_gps_times, whiten: bool, crop: bool) -> Dict:
        """
        A batch's dict, post-processed (``_post_process_batch``). Shapes: ONSOURCE and
        OFFSOURCE (B, I, samples); START_GPS_TIME, the first returned onsource sample's
        time, and DATA_LABEL (B, I), all NOISE; COINCIDENT (B,).
        """
        batch_size, num_ifos = onsource.shape[:2]
        coincident = self.coincidence == Coincidence.COINCIDENT
        return self._post_process_batch({
            RV.ONSOURCE: onsource,
            RV.OFFSOURCE: offsource,
            RV.START_GPS_TIME: window_spec.onsource_start_gps(start_gps_times),
            RV.DATA_LABEL: ops.full((batch_size, num_ifos), 0, dtype="int32"),  # NOISE = 0
            RV.COINCIDENT: ops.full((batch_size,), int(coincident and num_ifos > 1), dtype="int32"),
        }, window_spec.sample_rate_hertz, window_spec.onsource_duration_seconds, whiten, crop)

    def _grid_pass(self, window_spec, num_examples_per_batch: int, separation_seconds: float,
                   ifos: List[gf.IFO], emit, coincident: bool = False):
        """
        One GRID pass over the sampling plan, then the end: each row, in order, serves
        every window its pieces still hold whose detectors start ``separation_seconds``
        apart, or at one time when ``coincident`` (``IFOData.grid_windows``). A window
        is served at most once a pass, though
        its piece is in several rows, so the pass is finite and repeats nothing. The
        examples are batched across rows; the last, partial batch is not served.
        """
        used, held = {}, []
        for row in self.sampling_plan.rows:
            for segment in self.acquire(window_spec.sample_rate_hertz, row[None], ifos):
                keys = [(i, float(start), float(end)) for i, (start, end) in enumerate(row)]
                onsource, offsource, gps, marks = segment.grid_windows(
                    window_spec.num_onsource_samples,
                    window_spec.num_offsource_samples,
                    separation_seconds,
                    [used.get(key) for key in keys],
                    coincident,
                )
                used.update(zip(keys, marks))
                held.append((ops.convert_to_numpy(onsource), ops.convert_to_numpy(offsource), gps))
                onsource, offsource, gps = (np.concatenate(parts) for parts in zip(*held))
                served = len(gps) - len(gps) % num_examples_per_batch
                for first in range(0, served, num_examples_per_batch):
                    batch = slice(first, first + num_examples_per_batch)
                    yield emit(ops.convert_to_tensor(onsource[batch]), ops.convert_to_tensor(offsource[batch]),
                               gps[batch])
                held = [(onsource[served:], offsource[served:], gps[served:])]
        if held and len(held[0][2]):
            logging.info(f"The GRID pass ends with {len(held[0][2])} windows, short of a batch of "
                         f"{num_examples_per_batch}.")
