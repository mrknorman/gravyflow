"""
Noise mode data acquisition.

This module provides NoiseDataObtainer for acquiring random or grid-sampled
noise data from long IFO segments (NOISE mode).
"""

import logging
from dataclasses import dataclass
from typing import Any, List, Dict, Optional, Tuple, Union

import numpy as np
from numpy.random import default_rng
from keras import ops

import gravyflow as gf
from gravyflow.src.dataset.config import WindowSpec
from .base import (
    BaseDataObtainer, DataQuality, DataLabel, SegmentOrder,
    AcquisitionMode, SamplingMode, ObservingRun, IFOData, SEGMENT_GRID_ORIGIN_GPS,
    WINDOW_EDGE_BUFFER_SAMPLES, kept_seconds, label_kinds, separable,
)
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
            force_acquisition: bool = False,
            cache_segments: bool = True,
            overrides: dict = None,
            logging_level: int = logging.WARNING,
            augmentations: List = None,  # Training only; None is none
            prefetch_segments: int = 16,
            gwosc_flags: Optional[Tuple[str, ...]] = None,
            split_spec=None,
        ):

        super().__init__(
            data_quality=data_quality,
            data_labels=data_labels,
            observing_runs=observing_runs,
            segment_order=segment_order,
            max_segment_duration_seconds=max_segment_duration_seconds,
            saturation=saturation,
            force_acquisition=force_acquisition,
            cache_segments=cache_segments,
            overrides=overrides,
            logging_level=logging_level,
            augmentations=augmentations,
            prefetch_segments=prefetch_segments,
            gwosc_flags=gwosc_flags,
            split_spec=split_spec,
        )

        self.acquisition_mode = AcquisitionMode.NOISE

    def _served_state(self) -> Dict[str, Any]:
        return {
            **super()._served_state(),
            "_segment_generator": None,  # Reusable generator over the sampling plan
            "_current_row": None,  # The GPS pieces of the row being drawn from
            "current_segment": None,
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

        Unless the obtainer's labels ask for real events, each event's in-band signal
        is vetoed (``get_event_veto_segments``); unless they ask for glitches, each
        glitch's whole extent is vetoed (``get_glitch_veto_segments``). What they ask
        for is read by kind (``label_kinds``). Both cover every configured run.
        Returns the vetoed segments, sorted and coalesced.
        """
        asked = label_kinds(self.data_labels)
        veto_segments = []
        if DataLabel.EVENTS not in asked:
            veto_segments.append(self.get_event_veto_segments())
        if DataLabel.GLITCHES not in asked:
            veto_segments.append(self.get_glitch_veto_segments(ifo))
        if not veto_segments:
            return valid_segments
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
        ) -> SamplingPlan:
        """
        The plan for drawing windows from ``segments`` (rows from ``get_valid_segments``).

        Within each ``max_segment_duration_seconds`` bin, each detector's pieces of at
        least ``minimum_duration_seconds`` are usable; a bin where some detector has
        none is skipped. A piece is measured as it is loaded, in whole seconds
        (``kept_seconds``), and its window starts span that length less
        ``footprint_seconds``. The usable pieces are re-paired with ``couple_pieces``
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
            usable, starts = [], []
            for j in range(num_ifos):
                pieces = np.unique(in_bin[:, j, :], axis=0)
                lengths = kept_seconds(pieces[:, 1] - pieces[:, 0])
                keep = lengths >= minimum_duration_seconds
                pieces, lengths = pieces[keep], lengths[keep]
                order = np.argsort(-lengths, kind="stable")
                usable.append(pieces[order])
                starts.append(lengths[order] - footprint_seconds)
            if any(len(pieces) == 0 for pieces in usable):
                continue
            index, weights = couple_pieces(starts)
            bin_rows = np.stack([usable[j][index[:, j]] for j in range(num_ifos)], axis=1)
            keep = np.array([
                separable([(piece[0], piece[0] + starts[j][index[k, j]]) for j, piece in enumerate(row)],
                          separation_seconds)
                for k, row in enumerate(bin_rows)
            ], dtype=bool)
            inseparable += int(np.sum(~keep))
            rows.append(bin_rows[keep])
            start_seconds.append((weights * min(np.sum(s) for s in starts))[keep])

        if not sum(len(r) for r in rows):
            return SamplingPlan(np.empty((0, num_ifos, 2)), np.empty(0), inseparable)
        return SamplingPlan(np.concatenate(rows), np.concatenate(start_seconds), inseparable)

    def _planned_segments(
            self,
            plan: SamplingPlan,
            batches_per_start_second: float,
            sample_rate_hertz: float,
            ifos: List[gf.IFO],
        ):
        """
        Passes over ``plan``, without end: each due row's pieces, its loaded segment
        and its number of batches.

        Each pass, a row accrues its expected number of batches, its ``start_seconds``
        times ``batches_per_start_second``, and is due the whole batches it has
        accrued; the fraction left is carried to the next pass. Accruals start at a
        random fraction, so rows that expect less than a batch a pass come due in
        random order. Every row is thus drawn at its expected rate, to within one
        batch over any number of passes. A pass in which no row is due is not the
        end: the passes end only when one with rows due loads none of them.
        """
        expected = plan.start_seconds * batches_per_start_second
        if not np.any(expected > 0):
            return
        accrued = self.rng.random(len(expected))
        while True:
            accrued += expected
            due = np.floor(accrued).astype(int)
            accrued -= due
            loaded = not np.any(due)
            for row, num_batches in zip(plan.rows, due):
                if num_batches == 0:
                    continue
                for segment in self.acquire(sample_rate_hertz, row[None], ifos):
                    loaded = True
                    yield row, segment, num_batches
            if not loaded:
                return

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
        return self.remove_unwanted_segments(
            ifo, self.split_spec.admissible(self.get_all_segment_times(ifo), group_name)
        )

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
        absolute bin grid, and paired across IFOs bin by bin.
        """
        ifos = ensure_list(ifos)

        if not segment_order:
            segment_order = self.segment_order

        # The served segments are one split's, in one order: another rebuilds them.
        if self._serve_split(self._split(ifos, group_name), order=(seed, segment_order)):
            self.valid_segments = []

            for ifo in ifos:
                valid_segments = self.available_segments(ifo, group_name)

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

        # Use shared post-processing from base class
        def _post_process(batch):
            return self._post_process_batch(
                batch, 
                window_spec.sample_rate_hertz, 
                window_spec.onsource_duration_seconds,
                whiten, 
                crop
            )

        # With no group, the segments already held: built, or set by hand.
        if group is not None:
            self._serve_group(ifos, seed, group)
        elif self.valid_segments is None:
            raise ValueError("Name the split to serve: group='train', 'validate' or 'test'.")
        if self.rng is None:
            self.rng = default_rng(seed)
        
        # Use WindowSpec for sample counts (centralized calculation)
        num_onsource_samples = window_spec.num_onsource_samples
        num_offsource_samples = window_spec.num_offsource_samples

        # Early validation
        if self.valid_segments is None or len(self.valid_segments) == 0:
            logging.warning("No valid segments available in NOISE mode. Generator will be empty.")
            return
        
        if not hasattr(self.valid_segments, 'ndim') or self.valid_segments.ndim != 3:
            logging.warning(f"valid_segments has unexpected shape. Expected 3D array. Generator will be empty.")
            return
        
        # A window's footprint, offsource to padded onsource end. Detectors are never
        # coincident: an example's windows start at least two footprints apart, so no
        # two of them share a moment.
        footprint_seconds = (num_onsource_samples + num_offsource_samples) / window_spec.sample_rate_hertz
        separation_seconds = 2.0 * footprint_seconds

        if not self._current_batch_index and not self._current_segment_index:
            window_seconds = \
                window_spec.total_onsource_duration_seconds + window_spec.offsource_duration_seconds
            # A new plan starts new passes.
            self._segment_generator = None
            self.sampling_plan = self.plan_sampling(
                self.valid_segments,
                minimum_duration_seconds=2.0 * window_seconds,
                footprint_seconds=footprint_seconds
                + WINDOW_EDGE_BUFFER_SAMPLES / window_spec.sample_rate_hertz,
                separation_seconds=separation_seconds,
            )
            if self.sampling_plan.inseparable:
                logging.info(f"{self.sampling_plan.inseparable} rows left out: their detectors' "
                             "windows cannot start two footprints apart.")
        
        if self.sampling_plan is None or len(self.sampling_plan.rows) == 0:
            logging.warning("No valid segments available in NOISE mode. Generator will be empty.")
            return

        def emit(subarrays, background_chunks, start_gps_times):
            # Shapes: ONSOURCE/OFFSOURCE = (Batch, IFO, Samples) = BIS
            #         START_GPS_TIME = (Batch, IFO) = BI
            #         DATA_LABEL = (Batch, IFO) = BI (all NOISE = 0)
            batch_size, num_ifos = subarrays.shape[:2]
            return _post_process({
                RV.ONSOURCE: subarrays,
                RV.OFFSOURCE: background_chunks,
                # The windows' first samples, as the first returned onsource sample.
                RV.START_GPS_TIME: window_spec.onsource_start_gps(start_gps_times),
                RV.DATA_LABEL: ops.full((batch_size, num_ifos), 0, dtype="int32"),  # NOISE = 0
            })

        if sampling_mode == SamplingMode.GRID:
            yield from self._grid_pass(
                window_spec, num_examples_per_batch, separation_seconds, ifos, emit)
            return

        # Nominal density: saturation examples per onsource duration of window starts.
        batches_per_start_second = self.saturation / (
            num_examples_per_batch * window_spec.onsource_duration_seconds
        )

        # The rows that failed to hold a batch since the last one served.
        failed_rows = set()
        while True:
            while self._segment_exhausted:
                # Create generator ONCE and reuse it (preserves prefetch across batches)
                if self._segment_generator is None:
                    self._segment_generator = self._planned_segments(
                        self.sampling_plan,
                        batches_per_start_second,
                        window_spec.sample_rate_hertz,
                        ifos
                    )
                
                try:
                    (self._current_row, self.current_segment,
                     self._num_batches_in_current_segment) = next(self._segment_generator)
                except StopIteration:
                    logging.warning("No planned segment could be loaded. Generator exhausted.")
                    self._segment_generator = None
                    self.current_segment = None  # Clear to free memory
                    return

                min_num_samples = min([ops.shape(tensor)[0] for tensor in self.current_segment.data])

                if min_num_samples < (num_onsource_samples + num_offsource_samples):
                    logging.warning("Segment too short!")
                    self._segment_exhausted = True
                else: 
                    self._segment_exhausted = False
            
            while self._current_batch_index < self._num_batches_in_current_segment and not self._segment_exhausted:
                subarrays, background_chunks, start_gps_times = self.current_segment.random_subsection(
                    num_onsource_samples,
                    num_offsource_samples,
                    num_examples_per_batch,
                    self.rng.integers(1E10),
                    min_separation_seconds=separation_seconds,
                )

                if subarrays is None:
                    # The row cannot hold a batch of separated windows. Once every row
                    # has failed since the last batch, none will serve one.
                    self._segment_exhausted = True
                    failed_rows.add(np.asarray(self._current_row).tobytes())
                    if len(failed_rows) == len(self.sampling_plan.rows):
                        raise ValueError(
                            f"No row of the {self._served_group!r} split holds a batch of "
                            f"{num_examples_per_batch} {sampling_mode.name} windows whose detectors "
                            f"start {separation_seconds} s apart."
                        )
                    continue

                failed_rows.clear()
                self._current_batch_index += 1
                if not self._current_batch_index < self._num_batches_in_current_segment:
                    self._segment_exhausted = True
                    self._current_batch_index = 0

                yield emit(subarrays, background_chunks, start_gps_times)

    def _grid_pass(self, window_spec, num_examples_per_batch: int, separation_seconds: float,
                   ifos: List[gf.IFO], emit):
        """
        One GRID pass over the sampling plan, then the end: each row, in order, serves
        every window its pieces still hold whose detectors start ``separation_seconds``
        apart (``IFOData.grid_windows``). A window is served at most once a pass, though
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
