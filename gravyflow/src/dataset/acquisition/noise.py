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
from gravyflow.src.dataset.config import WindowSpec, batch_size
from .base import (
    BaseDataObtainer, DataQuality, DataLabel, SegmentOrder, Coincidence,
    AcquisitionMode, SamplingMode, ObservingRun, IFOData, SEGMENT_GRID_ORIGIN_GPS,
    WINDOW_EDGE_BUFFER_SAMPLES, label_kinds, separable, grid_slots, grid_starts, time_order, window_start_bounds,
)
from .gwosc_data import intersect_segments
from gravyflow.src.dataset.store import read as store_read
from gravyflow.src.dataset.store.read import interiors
from gravyflow.src.dataset.store.prefetch import Prefetcher
from gravyflow.src.rng import DIRECT_SPLIT, BatchStreams, SeedTree
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


#: How many batches of its pass a RANDOM batch draws its examples' rows from (``RandomPlan.mixed``).
MIXING_WIDTH = 8
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
    (``noise/segment_order`` at epoch p, a new one every pass), or in ``fixed_order``
    when the obtainer's ``segment_order`` fixes one, each row for its batches; the
    passes run on without end.
    Batch g lies in the pass p whose earlier batches, sum_j floor(p e_j + u_j), number
    at most g. Every count is arithmetic on piece lengths: nothing is read to address a
    batch, and the epoch of the order streams counts passes, not training epochs.

    A batch draws each example's row uniformly from the rows of ``MIXING_WIDTH`` batches
    of its pass, its own and those a stride of a ``MIXING_WIDTH``-th of the pass on from
    it, cyclically (``mixed``): so no row feeds a run of batches alone, and every batch
    of a pass is drawn from by exactly ``MIXING_WIDTH`` batches, so each row serves, in
    expectation, exactly its batches' examples in every pass.
    """
    tree: SeedTree
    split: str
    rows: np.ndarray  # (M, I, 2) GPS pieces
    firsts: np.ndarray  # (M, I) absolute index of the first sample read of each piece
    counts: np.ndarray  # (M, I) samples read of each piece
    expected: np.ndarray  # (M,) batches due a pass
    accrual: np.ndarray  # (M,) u_j
    fixed_order: Optional[np.ndarray]  # (M,) every pass's order; None draws a new one every pass
    _orders: Dict[int, np.ndarray] = field(default_factory=dict, repr=False)

    @classmethod
    def build(cls, tree: SeedTree, split: str, plan: SamplingPlan, firsts, counts,
              batches_per_start_second: float, fixed_order: Optional[np.ndarray] = None) -> "RandomPlan":
        expected = plan.start_seconds * batches_per_start_second
        if not np.sum(expected) > 0:
            raise ValueError(f"The {split!r} split's RANDOM plan has no row with window starts to draw.")
        accrual = tree.stream(split, "noise/accrual", epoch=0).random(len(expected))
        return cls(tree, split, plan.rows, firsts, counts, expected, accrual, fixed_order)

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
        if self.fixed_order is not None:
            return self.fixed_order
        if p not in self._orders:
            self._orders[p] = self.tree.stream(self.split, "noise/segment_order", epoch=p).permutation(len(self.rows))
        return self._orders[p]

    def mixed(self, g: int, width: int = MIXING_WIDTH) -> np.ndarray:
        """
        The rows batch ``g`` draws its examples from: those of the ``width`` batches of
        its pass at positions x + m K, modulo the pass's length L, for m below ``width``,
        where x is its own position and K = max(1, L // width). Every position of the
        pass is among exactly ``width`` batches' (counted with multiplicity), so a
        uniform draw over them serves each row exactly its batches' share, in
        expectation, in every pass; and the next batch draws from almost the same rows.
        """
        p, position = self.locate(int(g))
        order = self.order(p)
        due = self.due(p)[order]
        ends = np.cumsum(due)
        stride = max(1, int(ends[-1]) // width)
        positions = (position + stride * np.arange(width)) % int(ends[-1])
        return order[np.searchsorted(ends, positions, side="right")]


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
    drawn jointly, its row uniform over the batch's mixed rows (``RandomPlan.mixed``,
    ``noise/row``), then each detector's onsource start uniform over its
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
    mixed = plan.mixed(g)
    row_stream = streams.stream("noise/row")
    start_streams = [streams.stream(f"noise/start/{ifo.name}") for ifo in ifos]
    lo, phase = num_offsource_samples, num_offsource_samples % quantum
    hi = plan.counts - num_onsource_samples - WINDOW_EDGE_BUFFER_SAMPLES
    rows = np.empty(num_examples, dtype=np.int64)
    starts = np.empty((num_examples, len(ifos)), dtype=np.int64)
    rounds = np.zeros(num_examples, dtype=np.int64)
    for example in range(num_examples):
        for attempt in range(1, SEPARATION_ROUNDS + 1):
            j = int(mixed[row_stream.integers(len(mixed))])
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
                f"among rows {sorted(set(mixed.tolist()))} (GPS pieces {plan.rows[sorted(set(mixed.tolist()))].tolist()}): "
                + ("their detectors share no start." if coincident
                   else f"their detectors cannot start {separation_samples} samples apart.")
            )
    return rows, starts, rounds


@dataclass(frozen=True, eq=False)
class GridPlan:
    """
    The GRID set of one split, addressed by index: every example its rows hold
    (``grid_examples``), each window once, in the rows' time order.

    Batch k is examples [k B, (k + 1) B), so the set is floor(N / B) batches; the
    ``remainder``, fewer than B examples, is never served. GRID draws nothing, so
    every epoch serves the same set.
    """
    split: str
    starts: np.ndarray  # (N, I) each example's detectors' onsource starts, absolute sample indices
    num_examples_per_batch: int

    @property
    def batches(self) -> int:
        """The set's whole batches: floor(N / B)."""
        return len(self.starts) // self.num_examples_per_batch

    @property
    def remainder(self) -> int:
        """The examples after the last whole batch, never served."""
        return len(self.starts) % self.num_examples_per_batch

    def examples(self, k: int) -> np.ndarray:
        """Batch ``k``'s onsource starts, (B, I)."""
        k = int(k)
        if not 0 <= k < self.batches:
            raise IndexError(f"GRID batch {k} is outside the {self.batches} batches of the {self.split!r} set.")
        return self.starts[k * self.num_examples_per_batch:(k + 1) * self.num_examples_per_batch]


def grid_examples(
        firsts: np.ndarray,
        counts: np.ndarray,
        rows: np.ndarray,
        num_onsource_samples: int,
        num_offsource_samples: int,
        separation_samples: float,
        coincident: bool,
        quantum: int = 1,
    ) -> np.ndarray:
    """
    Every GRID example of ``rows`` ((M, I, 2) GPS pieces), in order, as each
    detector's onsource start, an absolute sample index: (N, I).

    Each piece's window starts (``window_start_bounds`` of its ``counts`` samples from
    absolute sample ``firsts``, both (M, I), as ``store.read.interiors`` measures them)
    are cut into slots one onsource apart, and each row pairs its pieces' free slots
    (``grid_starts``): detectors at least ``separation_samples`` apart, or, when
    ``coincident``, at one time. A piece in several rows serves each of its slots
    once, whichever row takes it. Nothing is read or drawn: the set is arithmetic on
    the pieces' lengths. A read starts on the grid of ``quantum`` samples
    (``store.read``), which every slot of a piece shares only if the onsource does.
    """
    if num_onsource_samples % quantum:
        raise ValueError(
            f"GRID slots are one onsource ({num_onsource_samples} samples) apart, but a read starts "
            f"only every {quantum} samples at this rate: choose an onsource of a multiple of {quantum} samples."
        )
    num_ifos = np.shape(firsts)[1] if np.ndim(firsts) == 2 else 0
    used, examples = {}, []
    for row, first, count in zip(rows, np.asarray(firsts, np.int64), np.asarray(counts, np.int64)):
        bounds = [window_start_bounds(int(n), num_onsource_samples, num_offsource_samples) for n in count]
        marks = [used.setdefault((i, float(start), float(end)), np.zeros(grid_slots(b, num_onsource_samples), bool))
                 for i, ((start, end), b) in enumerate(zip(row, bounds))]
        starts = grid_starts(bounds, first, num_onsource_samples, separation_samples, marks, coincident)
        examples.append(first[None, :] + starts)
    return np.concatenate(examples) if examples else np.empty((0, num_ifos), np.int64)


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
            prefetch_depth: int = 2,
            prefetch_workers: int = 2,
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
            prefetch_depth=prefetch_depth,
            prefetch_workers=prefetch_workers,
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
        guard at every edge with another split, and in train and validate less
        ``ifo``'s reserved benchmark time), less every vetoed event and glitch
        (``remove_unwanted_segments``). Sorted (N, 2) GPS intervals.
        """
        return self.split_spec.admissible(self.noise_admissible_segments(ifo), group_name, ifo)

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
        num_examples_per_batch = batch_size(num_examples_per_batch)
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
            batch = self.grid_batcher(window_spec, num_examples_per_batch, ifos, whiten, crop)
            for k in range(batch.batches):
                yield batch(tree.batch(self._served_group, 0, k) if self._served_group else tree.direct(k))
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

    def build_sampling_plan(self, window_spec: WindowSpec, minimum_samples: int = 0) -> SamplingPlan:
        """
        The sampling plan of the served segments for windows of ``window_spec``
        (``plan_sampling``): pieces of at least twice the window, and at least
        ``minimum_samples`` as read.
        """
        window_seconds = window_spec.total_onsource_duration_seconds + window_spec.offsource_duration_seconds
        footprint_seconds = (window_spec.num_onsource_samples + window_spec.num_offsource_samples) \
            / window_spec.sample_rate_hertz
        self.sampling_plan = self.plan_sampling(
            self.valid_segments,
            minimum_duration_seconds=max(2.0 * window_seconds, minimum_samples / window_spec.sample_rate_hertz),
            footprint_seconds=footprint_seconds + WINDOW_EDGE_BUFFER_SAMPLES / window_spec.sample_rate_hertz,
            separation_seconds=self._separation_seconds(window_spec),
            sample_rate_hertz=window_spec.sample_rate_hertz,
        )
        if self.sampling_plan.inseparable:
            logging.info(f"{self.sampling_plan.inseparable} rows left out: their detectors' "
                         "windows cannot start two footprints apart.")
        return self.sampling_plan

    def random_plan(self, tree: SeedTree, window_spec: WindowSpec, num_examples_per_batch: int,
                    minimum_samples: int = 0) -> RandomPlan:
        """
        The served split's RANDOM plan (``RandomPlan``) for windows of ``window_spec``
        and batches of ``num_examples_per_batch``: its rows in time order, each piece
        measured as it is read, each row due its start-seconds at the nominal density
        of ``saturation`` examples per onsource duration of window starts, visited in
        a new order every pass (``SegmentOrder.RANDOM``), or in time order
        (``CHRONOLOGICAL``) or shortest first (``SHORTEST_FIRST``, by the first
        detector's piece) every pass. Pieces shorter than ``minimum_samples`` as read
        are left out (``build_sampling_plan``).
        """
        split = self._served_group or DIRECT_SPLIT
        key = (tree, split, window_spec, num_examples_per_batch, minimum_samples)
        if key not in self._random_plans:
            sampling = self.build_sampling_plan(window_spec, minimum_samples)
            if len(sampling.rows) == 0:
                raise ValueError(f"The {split!r} split has no row long enough for a window "
                                 f"of {window_spec.total_onsource_duration_seconds + window_spec.offsource_duration_seconds} s.")
            order = time_order(sampling.rows)
            sampling = SamplingPlan(sampling.rows[order], sampling.start_seconds[order], sampling.inseparable)
            firsts, counts = interiors(self.source, window_spec.sample_rate_hertz, np.reshape(sampling.rows, (-1, 2)))
            shape = sampling.rows.shape[:2]
            fixed_order = {
                SegmentOrder.RANDOM: None,
                SegmentOrder.CHRONOLOGICAL: np.arange(len(sampling.rows)),
                SegmentOrder.SHORTEST_FIRST: np.argsort(sampling.rows[:, 0, 1] - sampling.rows[:, 0, 0], kind="stable"),
            }[self.segment_order]
            self._random_plans[key] = RandomPlan.build(
                tree, split, sampling, np.reshape(firsts, shape), np.reshape(counts, shape),
                self.saturation / (num_examples_per_batch * window_spec.onsource_duration_seconds),
                fixed_order,
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

        def starts_of(streams: BatchStreams) -> np.ndarray:
            g = (streams.epoch or 0) * (steps_per_epoch or 0) + streams.k
            return draw_examples(plan, streams, g, ifos, num_examples_per_batch, on, off, separation,
                                 coincident, quantum)[1]

        def prepare(addresses: Sequence[BatchStreams]) -> List[store_read.TimelineContradiction]:
            """Every contradiction the windows of ``addresses`` meet (``prepare_windows``)."""
            return self.prepare_windows(ifos, rate, map(starts_of, addresses), on, off)

        batch.prefetcher = prefetcher
        batch.plan = plan
        batch.prepare = prepare
        return batch

    def grid_plan(self, window_spec: WindowSpec, num_examples_per_batch: int) -> GridPlan:
        """
        The served split's GRID set (``GridPlan``) for windows of ``window_spec`` and
        batches of ``num_examples_per_batch``: every example of its sampling plan's rows
        (``grid_examples``), rows in time order, each piece measured as it is read.
        Logs the set's size and the remainder it leaves unserved.
        """
        split = self._served_group or DIRECT_SPLIT
        rows = self.build_sampling_plan(window_spec).rows
        rows = rows[time_order(rows)]
        rate = window_spec.sample_rate_hertz
        firsts, counts = interiors(self.source, rate, np.reshape(rows, (-1, 2)))
        plan = GridPlan(split, grid_examples(
            np.reshape(firsts, rows.shape[:2]), np.reshape(counts, rows.shape[:2]), rows,
            window_spec.num_onsource_samples, window_spec.num_offsource_samples,
            self._separation_seconds(window_spec) * rate, self.coincidence == Coincidence.COINCIDENT,
            store_read.geometry(self.source, rate).up,
        ), num_examples_per_batch)
        message = (f"The {split!r} GRID set has {len(plan.starts)} examples: {plan.batches} batches of "
                   f"{num_examples_per_batch}, and a remainder of {plan.remainder} examples not served.")
        if plan.batches:
            logging.info(message)
        else:
            logging.warning(message)
        return plan

    def grid_batcher(
            self,
            window_spec: WindowSpec,
            num_examples_per_batch: int,
            ifos: List[gf.IFO],
            whiten: bool = False,
            crop: bool = False,
        ):
        """
        The served split's GRID batches, addressed by index: ``batch(streams)`` is batch
        ``streams.k`` of the split's GRID set (``grid_plan``), whatever the epoch, since
        the set is the same every epoch; past its last batch it raises IndexError.
        ``batch.batches`` is the set's length, floor(N / B), and ``batch.plan`` the set.
        Its windows are read through the store (``read_windows``), and the batches after
        it are read ahead (``store.prefetch``), which never changes a batch.
        """
        plan = self.grid_plan(window_spec, num_examples_per_batch)
        rate = window_spec.sample_rate_hertz
        on, off = window_spec.num_onsource_samples, window_spec.num_offsource_samples

        def windows(k: int):
            starts = plan.examples(k)
            onsource, offsource = self.read_windows(ifos, rate, starts, on, off)
            return onsource, offsource, starts / rate

        prefetcher = Prefetcher(windows, name=plan.split, depth=self.prefetch_depth, workers=self.prefetch_workers)

        def batch(streams: BatchStreams) -> Dict:
            k = int(streams.k)
            onsource, offsource, gps = prefetcher.get(k, range(k + 1, min(k + 1 + prefetcher.depth, plan.batches)))
            return self._emit(window_spec, ops.convert_to_tensor(onsource), ops.convert_to_tensor(offsource), gps,
                              whiten, crop)

        def prepare(addresses: Sequence[BatchStreams]) -> List[store_read.TimelineContradiction]:
            """Every contradiction the windows of ``addresses`` meet (``prepare_windows``)."""
            return self.prepare_windows(ifos, rate, (plan.examples(streams.k) for streams in addresses), on, off)

        batch.prefetcher = prefetcher
        batch.plan = plan
        batch.batches = plan.batches
        batch.prepare = prepare
        return batch

    def prepare_windows(self, ifos: List[gf.IFO], sample_rate_hertz: float, batches,
                        num_onsource_samples: int, num_offsource_samples: int
                        ) -> List[store_read.TimelineContradiction]:
        """
        Commit every tile the windows of ``batches`` (each (B, I) onsource starts, absolute
        sample indices) read, without reading them (``store.read.ensure``), and return every
        contradiction met, one per exclusion: the one place a batcher's ``prepare`` meets
        the store.
        """
        met = {}
        for starts in batches:
            for example in starts:
                for ifo, start in zip(ifos, example):
                    try:
                        store_read.ensure(self.source, ifo, sample_rate_hertz, int(start) - num_offsource_samples,
                                          num_offsource_samples + num_onsource_samples, stream=self._served_split)
                    except store_read.TimelineContradiction as error:
                        met.setdefault(error.exclusion, error)
        return list(met.values())

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
