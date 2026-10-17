"""
Transient mode data acquisition.

This module provides TransientDataObtainer for acquiring data around
specific transient events (GW events, glitches) in TRANSIENT mode.
"""

import itertools
import logging
from collections import Counter
from typing import Any, List, Dict, Optional, Union, Tuple
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

logger = logging.getLogger(__name__)

import numpy as np
from numpy.random import default_rng
from keras import ops

import gravyflow as gf
from gravyflow.src.dataset.features.event import EventConfidence
from gravyflow.src.dataset.features.glitch_cache import TransientCache, generate_transient_cache_path
from .cache import DiskCache
from gravyflow.src.dataset.features.injection import ReturnVariables as RV
from .base import (
    BaseDataObtainer, DataQuality, DataLabel, SegmentOrder, 
    AcquisitionMode, SamplingMode, ObservingRun
)
from gravyflow.src.utils.numerics import ensure_list
from gravyflow.src.utils.gps import as_gps_times
from gravyflow.src.dataset.features.transient_index import TransientIndex
from .transient_segment import TransientSegment
from .gwosc_data import within_segments
from gravyflow.src.dataset.config import TransientDefaults, WindowSpec
from gravyflow.src.dataset.splits import ALL, split_fractions


# The transient caches an obtainer may open, one per prefix (`_initialize_transient_cache`).
TRANSIENT_CACHE_PREFIXES = ("transient", "glitch", "event")


class TransientDataObtainer(BaseDataObtainer):
    """
    Data obtainer for TRANSIENT mode acquisition.
    
    Acquires data windows centered around specific events (GW mergers, glitches),
    suitable for event classification or detection training.
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
            event_types: List[EventConfidence] = None,
            logging_level: int = logging.WARNING,
            augmentations: List = None,  # List of augmentation instances
            prefetch_segments: int = 64,  # Higher for smaller TRANSIENT segments
            # Class balancing
            balanced_glitch_types: bool = False,
            # Specific Names
            event_names: List[str] = None
        ):
        
        # Default observing runs matching BaseDataObtainer behavior
        if observing_runs is None:
            observing_runs = [ObservingRun.O1, ObservingRun.O2, ObservingRun.O3]
        if not isinstance(observing_runs, list):
            observing_runs = [observing_runs]
            
        self.observing_runs = observing_runs
        self.event_names = event_names

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
        )
        
        self.acquisition_mode = AcquisitionMode.TRANSIENT
        
        # Event type filtering
        if event_types is None:
            event_types = [EventConfidence.CONFIDENT]
        self.event_types = event_types
        
        # Class balancing
        self.balanced_glitch_types = balanced_glitch_types

    _segment_cache_maxsize = 128  # TRANSIENT uses larger cache for many small segments

    def _served_state(self) -> Dict[str, Any]:
        caches = {}
        for prefix in TRANSIENT_CACHE_PREFIXES:
            caches[f"_{prefix}_cache"] = None
            caches[f"_{prefix}_cache_logged"] = False
        return {
            **super()._served_state(),
            # TransientIndex - canonical source of truth for transient examples
            "_feature_index": None,
            # TransientSegment objects (replaces raw numpy segments)
            "transient_segments": [],
            # Records not served, by reason: a census of the split (_passing_data_quality,
            # _reject_for_no_data)
            "_rejections": Counter(),
            # The exact GPS times of records found without a requested detector's
            # strain: not served, nor fetched again, while this state stands
            "_unavailable": set(),
            "_cache_hit_count": 0,
            "_cache_miss_count": 0,
            "_last_log_count": 0,
            **caches,
        }

    #: Why a record is not served: its window leaves the usable segments of a
    #: requested detector, or a requested detector has no finite strain for it.
    REJECTED_BY_DATA_QUALITY = "data quality"
    REJECTED_FOR_NO_DATA = "no data"

    def _reject_for_no_data(self, segment: TransientSegment) -> None:
        """
        A requested detector has no strain for ``segment``: it is never zero-filled.
        It is counted once, as data quality is, and neither served nor fetched again
        while this split's served state stands (``_unavailable``); a later run fetches
        it again, so a failed fetch never persists as absence.
        """
        if segment.transient_gps_time in self._unavailable:
            return
        self._unavailable.add(segment.transient_gps_time)
        self._rejections[self.REJECTED_FOR_NO_DATA] = len(self._unavailable)
        logger.info(
            f"Transient at GPS {segment.transient_gps_time:.3f} not served (no data); "
            f"rejected: {dict(self._rejections)}"
        )

    def get_acquisition_stats(self) -> Dict[str, Any]:
        """
        As for noise, with ``rejections``: the records of the split not served, by
        reason (``REJECTED_BY_DATA_QUALITY``, ``REJECTED_FOR_NO_DATA``), each record
        counted once however many passes meet it; ``dropped_segments`` is their total.
        """
        return {
            **super().get_acquisition_stats(),
            "dropped_segments": sum(self._rejections.values()),
            "rejections": dict(self._rejections),
        }

    def reset_acquisition_stats(self) -> None:
        super().reset_acquisition_stats()
        self._rejections = Counter()

    def _passing_data_quality(
        self, segments: List[TransientSegment], ifos: List[gf.IFO]
    ) -> List[TransientSegment]:
        """
        The records whose acquired span lies, on every requested detector, inside one
        usable segment (``get_all_segment_times``: the obtainer's GWOSC flags, so by
        default no hardware injection either). The others are rejected, and their
        number is the split's ``REJECTED_BY_DATA_QUALITY`` count.
        """
        keep = np.ones(len(segments), dtype=bool)
        if segments:
            spans = np.array([[s.start_gps_time, s.end_gps_time] for s in segments], dtype=np.float64)
            for ifo in ifos:
                keep &= within_segments(spans, self.get_all_segment_times(ifo))
        rejected = int(np.sum(~keep))
        self._rejections.pop(self.REJECTED_BY_DATA_QUALITY, None)
        if rejected:
            self._rejections[self.REJECTED_BY_DATA_QUALITY] = rejected
            logger.info(
                f"{rejected} of {len(segments)} transients rejected: their windows leave "
                f"the usable data of {[ifo.name for ifo in ifos]}."
            )
        return list(itertools.compress(segments, keep))

    def _apply_augmentation(self, data):
        """
        Apply augmentations including TRANSIENT-specific ones (RandomShift, AddNoise).
        """
        from gravyflow.src.dataset.acquisition.augmentations import (
            SignReversal, TimeReversal, RandomShift, AddNoise
        )
        
        if self.rng is None or not self.augmentations:
            return data
        
        for aug in self.augmentations:
            if self.rng.random() >= aug.probability:
                continue
            
            if isinstance(aug, SignReversal):
                data = -data
            elif isinstance(aug, TimeReversal):
                data = ops.flip(data, axis=-1)
            elif isinstance(aug, RandomShift):
                # Shift event off-center
                num_samples = int(ops.shape(data)[-1])
                max_shift = int(aug.shift_fraction * num_samples)
                shift = self.rng.integers(-max_shift, max_shift + 1)
                data = ops.roll(data, shift, axis=-1)
            elif isinstance(aug, AddNoise):
                # Add noise perturbation
                data_np = ops.convert_to_numpy(data)
                noise = self.rng.normal(0, aug.amplitude, data_np.shape)
                std = np.std(data_np)
                noise = noise * std if std > 0 else noise
                data = ops.convert_to_tensor(data_np + noise, dtype=data.dtype)
        
        return data

    def build_feature_index(
        self,
        ifos: List[gf.IFO],
        groups: Dict[str, float] = None,
    ) -> "TransientIndex":
        """
        Build TransientIndex from segment builders.
        
        Returns TransientSegments directly.
        """
        from gravyflow.src.dataset.features.segment_builders import (
            build_glitch_segments, build_event_segments
        )
        from gravyflow.src.dataset.features.transient_index import TransientIndex
        from gravyflow.src.dataset.features.glitch import GlitchType
        from gravyflow.src.dataset.features.event import SourceType
        
        all_segments = []
        
        # Check what types of data are requested
        data_labels_list = self.data_labels if isinstance(self.data_labels, list) else [self.data_labels]
        
        # Check for specific glitch types or general GLITCHES label
        has_specific_glitches = any(isinstance(label, GlitchType) for label in data_labels_list)
        has_glitches_label = DataLabel.GLITCHES in data_labels_list
        
        if has_glitches_label or has_specific_glitches:
            # If specific types requested, pass them to build_glitch_segments
            glitch_types_to_fetch = None
            if has_specific_glitches:
                glitch_types_to_fetch = [label for label in data_labels_list if isinstance(label, GlitchType)]
            
            glitch_segments = build_glitch_segments(
                ifos=ifos,
                observing_runs=self.observing_runs,
                glitch_types=glitch_types_to_fetch  # Pass specific types if requested
            )
            all_segments.extend(glitch_segments)
        
        
        # Fetch event segments if requested (reuse data_labels_list from above)
        has_events = any(isinstance(et, EventConfidence) for et in data_labels_list)
        # Check for EVENTS label - handle both enum and integer value comparisons
        has_events_label = any(
            l == DataLabel.EVENTS or (hasattr(l, 'value') and l.value == DataLabel.EVENTS.value) or l == DataLabel.EVENTS.value
            for l in data_labels_list
        )
        if has_events_label or has_events:
            # Determine observing runs to query: if specific event names are requested, 
            # we should broaden the search (e.g., check all runs) to ensure they are found.
            # Otherwise, respect the strict observing_runs filter.
            effective_runs = self.observing_runs
            
            # Note: Previously we forced effective_runs=None here to search all runs.
            # Now we respect the user's observing_runs filter. If finding an event from a different
            # run is desired, set observing_runs=None or include the target run.
            
            logger.info(f"Building event segments (runs={effective_runs})")
            event_segments = build_event_segments(
                observing_runs=effective_runs,
                confidences=self.event_types if self.event_types else [EventConfidence.CONFIDENT],
                event_names=self.event_names
            )
            all_segments.extend(event_segments)
            logger.info(f"Built {len(event_segments)} event segments")
        
        # Create index
        index = TransientIndex(all_segments)
        index.dedupe()
        index.sort()
        
        # Filter by specific event names if requested
        if self.event_names:
            logger.info(f"Filtering to specific events: {self.event_names}")
            index = index.filter_by_names(self.event_names)
            logger.info(f"After filtering: {len(index)} segments")
        
        # Assign groups
        index.assign_groups(groups)
        
        # Apply balancing
        if self.balanced_glitch_types:
            index.apply_balancing(by="kind")
        
        self._feature_index = index
        return index


    def _get_sample_from_cache(
        self,
        cache,
        gps_time: float,
        sample_rate_hertz: float,
        onsource_duration: float,
        offsource_duration: float,
        scale_factor: float = 1.0,
        target_ifos: Optional[List[str]] = None,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
        """
        Attempt to retrieve a sample from cache (chunked -> disk), as windows of the
        request (``TransientCache.read_windows``).

        Args:
            gps_time: The transient's exact GPS time, its row's identity
            target_ifos: List of IFO names to retrieve (must be present in cache)

        Returns:
            (onsource, offsource, source) where source is 'chunk', 'disk', or 'miss'
        """
        idx = cache.index_of(gps_time)
        if idx is None:
            return None, None, 'miss'

        # Try chunked memory cache first (new fast path)
        # Chunks hold the request's windows already (_load_chunk_around_index).
        if cache._chunk_in_memory:
            result = cache.get_from_chunk(idx, target_ifos=target_ifos)
            if result is not None:
                onsource, offsource, _, _ = result
                return onsource * scale_factor, offsource * scale_factor, 'chunk'

        try:
            ons, off, _, _ = cache.get_batch(
                np.array([idx]),
                sample_rate_hertz=sample_rate_hertz,
                onsource_duration=onsource_duration,
                offsource_duration=offsource_duration,
                target_ifos=target_ifos
            )
            return ons[0] * scale_factor, off[0] * scale_factor, 'disk'
        except (ValueError, KeyError):
            # Request parameters incompatible or IFO missing
            return None, None, 'miss'

    def _prepare_batch(
        self,
        batch_subarrays: list,
        batch_backgrounds: list,
        batch_segments: list,  # List of TransientSegments
        window_spec: WindowSpec,
    ) -> dict:
        """
        Prepare batch tensors from accumulated lists.
        
        Returns:
            Dict with ReturnVariables keys:
            - ONSOURCE: (Batch, IFO, Samples) = BIS
            - OFFSOURCE: (Batch, IFO, Samples) = BIS  
            - START_GPS_TIME: (Batch, IFO) = BI, float64: the GPS time of the first
              onsource sample returned after cropping, as for noise
              (``WindowSpec.onsource_start_gps``), the window centred on the
              transient's sample (``WindowSpec.transient_window_start_gps``).
            - TRANSIENT_GPS_TIME: (Batch, IFO) = BI
              The GPS time of the event or glitch, the window's centre.
            - DATA_LABEL: (Batch, IFO) = BI (1 for GLITCHES, 2 for EVENTS)
            - SUB_TYPE: (Batch, IFO) = BI (GlitchType.value or SourceType.value)
            - GLITCH_TYPE: (Batch, IFO) = BI (GlitchType.value if glitch, -1 otherwise)
            - SOURCE_TYPE: (Batch, IFO) = BI (SourceType.value if event, -1 otherwise)
        """
        # Shape validation - identify mismatched arrays before stacking
        if batch_subarrays:
            shapes = [np.asarray(arr).shape for arr in batch_subarrays]
            unique_shapes = set(shapes)
            if len(unique_shapes) > 1:
                # Log which segments have which shapes for debugging
                logger.error(f"Shape mismatch in batch_subarrays! Unique shapes: {unique_shapes}")
                for i, (arr, seg) in enumerate(zip(batch_subarrays, batch_segments)):
                    logger.error(f"  Index {i}: shape={np.asarray(arr).shape}, GPS={seg.transient_gps_time:.3f}")
                # Filter to most common shape to avoid crash
                from collections import Counter
                shape_counts = Counter(shapes)
                most_common_shape = shape_counts.most_common(1)[0][0]
                valid_indices = [i for i, s in enumerate(shapes) if s == most_common_shape]
                batch_subarrays = [batch_subarrays[i] for i in valid_indices]
                batch_backgrounds = [batch_backgrounds[i] for i in valid_indices]
                batch_segments = [batch_segments[i] for i in valid_indices]
                logger.warning(f"Filtered batch to {len(valid_indices)} samples with shape {most_common_shape}")
        
        final_subarrays = ops.cast(ops.stack(batch_subarrays), "float32")
        final_background = ops.cast(ops.stack(batch_backgrounds), "float32")
        
        # Get dimensions from subarrays shape: (Batch, IFO, Samples)
        batch_size = int(ops.shape(final_subarrays)[0])
        num_ifos = int(ops.shape(final_subarrays)[1])
        
        # Extract GPS times from segments
        batch_gps_times = [seg.transient_gps_time for seg in batch_segments]
        gps_1d = as_gps_times(batch_gps_times)
        transient_gps = np.tile(np.expand_dims(gps_1d, axis=-1), (1, num_ifos))
        
        # Compute DATA_LABEL per-segment based on segment type
        # DataLabel.GLITCHES = 1, DataLabel.EVENTS = 2
        batch_data_labels_1d = np.array([
            DataLabel.GLITCHES.value if seg.is_glitch else DataLabel.EVENTS.value
            for seg in batch_segments
        ], dtype=np.int32)
        data_label = ops.tile(ops.expand_dims(batch_data_labels_1d, axis=-1), (1, num_ifos))
        
        # SUB_TYPE: Generic type value (GlitchType or SourceType)
        batch_sub_type_1d = np.array([
            seg.kind.value if (seg.is_glitch or seg.is_event) else -1
            for seg in batch_segments
        ], dtype=np.int32)
        sub_type = ops.tile(ops.expand_dims(batch_sub_type_1d, axis=-1), (1, num_ifos))
        
        # GLITCH_TYPE: Only filled for glitches, -1 for events
        batch_glitch_type_1d = np.array([
            seg.kind.value if seg.is_glitch else -1
            for seg in batch_segments
        ], dtype=np.int32)
        glitch_type = ops.tile(ops.expand_dims(batch_glitch_type_1d, axis=-1), (1, num_ifos))
        
        # SOURCE_TYPE: Only filled for events, -1 for glitches
        batch_source_type_1d = np.array([
            seg.kind.value if seg.is_event else -1
            for seg in batch_segments
        ], dtype=np.int32)
        source_type = ops.tile(ops.expand_dims(batch_source_type_1d, axis=-1), (1, num_ifos))
        
        return {
            RV.ONSOURCE: self._apply_augmentation(final_subarrays),
            RV.OFFSOURCE: self._apply_augmentation(final_background),
            RV.START_GPS_TIME: window_spec.onsource_start_gps(
                window_spec.transient_window_start_gps(transient_gps)
            ),
            RV.TRANSIENT_GPS_TIME: transient_gps,
            RV.DATA_LABEL: data_label,     # Per-segment: GLITCHES=1 or EVENTS=2
            RV.SUB_TYPE: sub_type,         # Per-segment: GlitchType.value or SourceType.value
            RV.GLITCH_TYPE: glitch_type,   # Per-segment: GlitchType.value or -1
            RV.SOURCE_TYPE: source_type,   # Per-segment: SourceType.value or -1
        }

    @staticmethod
    def _extract_feature_event(
        channels: List[Optional[Tuple[np.ndarray, float]]],
        sample_rate_hertz: float,
        num_onsource_samples: int,
        num_offsource_samples: int,
        transient_gps_time: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        The stored windows of one transient, one row per channel: the onsource
        centred on the transient's sample (``centre_index``), the sample nearest
        ``transient_gps_time`` counted from the channel's first returned sample, and
        the offsource of the samples just before it (the layout
        ``TransientCache.read_windows`` reads).

        ``channels`` holds, per channel, its samples and the GPS time of the first, or
        None. A channel that is None, does not hold both windows, or has any
        non-finite sample in them is a row of NaN: missing, never zero-filled.
        """
        from gravyflow.src.utils.tensor import centre_index

        onsource = np.full((len(channels), num_onsource_samples), np.nan, dtype=np.float32)
        offsource = np.full((len(channels), num_offsource_samples), np.nan, dtype=np.float32)
        for row, channel in enumerate(channels):
            if channel is None:
                continue
            samples, first_gps = channel
            # The transient's sample, from the true first-sample time, in float64.
            transient = int(np.round((float(transient_gps_time) - float(first_gps)) * sample_rate_hertz))
            start = transient - centre_index(num_onsource_samples)
            off_start = start - num_offsource_samples
            if off_start < 0 or start + num_onsource_samples > len(samples):
                continue
            on, off = samples[start:start + num_onsource_samples], samples[off_start:start]
            if np.all(np.isfinite(on)) and np.all(np.isfinite(off)):
                onsource[row], offsource[row] = on, off
        return onsource, offsource


    def _cluster_transients(
        self, 
        segments: np.ndarray, 
        request_overhead_seconds: float = None,
        data_download_rate: float = None,
        max_segment_seconds: float = None
    ) -> np.ndarray:
        """
        Cluster nearby transients using a greedy cost-optimized algorithm.
        Limits max segment size to prevent OOM errors during download.
        """
        from gravyflow.src.dataset.config import TransientDefaults
        if request_overhead_seconds is None:
            request_overhead_seconds = TransientDefaults.REQUEST_OVERHEAD_SECONDS
        if data_download_rate is None:
            data_download_rate = TransientDefaults.DATA_DOWNLOAD_RATE
        if max_segment_seconds is None:
            max_segment_seconds = TransientDefaults.MAX_SEGMENT_SECONDS
        if len(segments) == 0:
            return segments
        
        if len(segments) == 1:
            return segments
        
        gap_threshold = request_overhead_seconds / data_download_rate
            
        sorted_idx = np.argsort(segments[:, 0])
        sorted_segs = segments[sorted_idx]
        
        merged = []
        current_start, current_end = sorted_segs[0]
        
        for start, end in sorted_segs[1:]:
            gap = start - current_end
            
            segment_duration = current_end - current_start
            if (gap <= 0 or gap <= gap_threshold) and segment_duration < max_segment_seconds:
                current_end = max(current_end, end)
            else:
                merged.append([current_start, current_end])
                current_start, current_end = start, end
        
        merged.append([current_start, current_end])
        
        merged_array = np.array(merged)
        
        original_count = len(segments)
        merged_count = len(merged_array)
        if merged_count < original_count:
            logger.info(
                f"Greedy clustering: {original_count} transients -> "
                f"{merged_count} download segments (gap threshold: {gap_threshold:.0f}s)"
            )
        
        return merged_array

    def _get_segment_boundaries(self, indices: List[int]) -> np.ndarray:
        """
        Get GPS boundaries for specified segment indices.
        
        Directly accesses transient_segments instead of valid_segments array,
        providing cleaner access to segment boundaries.
        
        Args:
            indices: List of segment indices
            
        Returns:
            Array of shape (N, 2) with [start, end] GPS times
        """
        if not indices:
            return np.empty((0, 2), dtype=np.float64)
        return np.array([
            [self.transient_segments[i].start_gps_time, 
             self.transient_segments[i].end_gps_time]
            for i in indices
        ], dtype=np.float64)

    # =========================================================================
    # ABSTRACT METHOD IMPLEMENTATIONS
    # =========================================================================


    def get_valid_segments(
        self,
        ifos: List[gf.IFO],
        seed: int,
        groups: Dict[str, float] = None,
        group_name: str = "train",
        segment_order: SegmentOrder = None,
    ) -> np.ndarray:
        """
        Get valid segments for TRANSIENT mode.

        Segments come from TransientIndex which contains full TransientSegments.
        The split's members are fixed by the index (`TransientIndex.assign_groups`);
        ``seed`` only orders them (SegmentOrder.RANDOM).
        """
        if not isinstance(ifos, list):
            ifos = [ifos]
        
        if not segment_order:
            segment_order = self.segment_order
        
        groups = split_fractions(groups, group_name)

        # The served state is one split's, on its detectors: another rebuilds
        # it, index included, since the index carries the split's fractions
        # and the detectors' records.
        self._serve_split(self._split(ifos, group_name, groups))
        if self._feature_index is None:
            self.build_feature_index(ifos=ifos, groups=groups)
        
        # Get segments for this group, filtered by requested IFOs
        # Segments with seen_in overlapping requested IFOs will be included
        self.transient_segments = self._passing_data_quality(
            list(self._feature_index.iter(group=group_name, ifos_filter=ifos)), ifos
        )
        
        if len(self.transient_segments) == 0:
            logger.warning(f"No segments found for group '{group_name}'")
            return np.empty((0, len(ifos), 2), dtype=np.float64)
        
        # Shuffle if needed
        if segment_order == SegmentOrder.RANDOM:
            rng = default_rng(seed)
            rng.shuffle(self.transient_segments)
        elif segment_order == SegmentOrder.SHORTEST_FIRST:
            self.transient_segments.sort(key=lambda s: s.duration)
        
        logger.info(
            f"TRANSIENT MODE: {len(self.transient_segments)} segments for '{group_name}'"
        )
        
        # Return segments array for interface compatibility
        # Shape: (num_segments, num_ifos, 2) where [:, :, 0]=start, [:, :, 1]=end
        num_ifos = len(ifos)
        segments_2d = np.array([
            [seg.start_gps_time, seg.end_gps_time]
            for seg in self.transient_segments
        ], dtype=np.float64)
        
        expanded = np.expand_dims(segments_2d, axis=1)
        return np.repeat(expanded, num_ifos, axis=1)


    def _process_cache_misses(
        self,
        miss_indices: List[int],
        ifos: List,
        window_spec: WindowSpec,
        scale_factor: float,
        cache,
        batch_subarrays: List,
        batch_backgrounds: List,
        batch_segments: List,
        num_examples_per_batch: int
    ):
        """
        Download, extract, cache and prepare batches for cache misses.
        
        Uses parallel downloads with ThreadPoolExecutor for improved performance.
        Consolidates duplicated logic for processing segments that weren't in cache.
        Yields batches whenever batch_size is reached.
        
        Args:
            miss_indices: List of segment indices that need downloading
            ifos: List of IFOs to download
            window_spec: The windows requested
            scale_factor: Amplitude scale factor
            cache: DiskCache instance for storing/retrieving
            batch_subarrays: Accumulated onsource arrays (mutated)
            batch_backgrounds: Accumulated offsource arrays (mutated)
            batch_segments: Accumulated TransientSegments (mutated)
            num_examples_per_batch: Batch size threshold
            max_workers: Number of parallel download threads (default 8)
            
        Yields:
            Batch dicts when batch_size is reached
        """
        from gravyflow.src.dataset.features.glitch_cache import (
            CACHE_SAMPLE_RATE_HERTZ, CACHE_ONSOURCE_DURATION, CACHE_OFFSOURCE_DURATION
        )
        
        if len(miss_indices) == 0:
            return

        sample_rate_hertz = window_spec.sample_rate_hertz
        # Download at higher of requested or cache sample rate (can downsample but not upsample)
        download_sample_rate = max(sample_rate_hertz, CACHE_SAMPLE_RATE_HERTZ)
        # Refuse rates no public source holds, before any download.
        from .gwosc_data import gwosc_sample_rate_hertz
        gwosc_sample_rate_hertz(download_sample_rate)

        # Define universal IFO names for cache indexing
        universal_ifo_names = list(TransientDefaults.UNIVERSAL_IFOS)
        download_ifos_universal = [gf.IFO[name] for name in universal_ifo_names]
        
        # Pre-calculate extraction parameters
        num_cache_onsource = int(CACHE_ONSOURCE_DURATION * download_sample_rate)
        num_cache_offsource = int(CACHE_OFFSOURCE_DURATION * download_sample_rate)
        
        # Calculate output indices once
        try:
            output_indices = [universal_ifo_names.index(ifo.name) for ifo in ifos]
        except ValueError as e:
            logger.error(f"Requested IFO not found in universal set! Request: {[i.name for i in ifos]}, Universal: {universal_ifo_names}")
            raise e
        
        # ==============================================================
        # PARALLEL DOWNLOAD: Thread-safe stateless download function
        # ==============================================================
        # Uses stateless function to avoid race conditions with self.acquire()
        # which mutates shared state (_current_segment_index, etc.)
        # ==============================================================
        
        def download_segment_stateless(miss_idx: int) -> Tuple[int, List[Optional[Tuple[np.ndarray, float]]]]:
            """
            Thread-safe download for a single segment.
            
            Uses only static methods and captured local variables.
            Does NOT call self.acquire() to avoid race conditions.
            
            Args:
                miss_idx: Index into self.transient_segments
                
            Returns:
                (miss_idx, channels): per universal IFO, its samples at the download
                rate and the GPS time of the first as the source returned it, or None
                when it could not be had (never zeros).
            """
            segment = self.transient_segments[miss_idx]
            channels = []
            
            for ifo in download_ifos_universal:
                frame_type, channel = BaseDataObtainer.get_frame_channel_for_gps_static(
                    segment.start_gps_time,
                    self.frame_types,
                    self.channels,
                    self.start_gps_times,
                    self.end_gps_times
                )
                
                try:
                    segment_data = BaseDataObtainer.get_segment_data_static(
                        segment.start_gps_time,
                        segment.end_gps_time,
                        ifo,
                        frame_type,
                        channel,
                        sample_rate_hertz=download_sample_rate,
                    )
                    if segment_data.sample_rate.value < download_sample_rate:
                        raise ValueError("source rate below the download rate")
                    if segment_data.sample_rate.value != download_sample_rate:
                        segment_data = segment_data.resample(download_sample_rate)
                    channels.append((np.asarray(segment_data.value), float(segment_data.t0.value)))
                except Exception as error:
                    logger.debug(f"No {ifo.name} strain for GPS {segment.transient_gps_time:.3f}: {error}")
                    channels.append(None)
            
            return miss_idx, channels
        
        # Execute parallel downloads
        max_workers = min(32, len(miss_indices))
        downloaded_segments = {}
        failed_count = 0
        
        # Detect if running in interactive terminal
        import sys
        is_interactive = sys.stdout.isatty()
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(download_segment_stateless, idx): idx 
                for idx in miss_indices
            }
            
            pending = set(futures.keys())
            
            if is_interactive and len(futures) > 1:
                # Multi-progress bar mode for interactive terminals
                # Main progress bar at position 0
                main_pbar = tqdm(
                    total=len(futures),
                    desc=f"Downloads ({max_workers}w)",
                    unit="seg",
                    position=0,
                    leave=True,
                    dynamic_ncols=True
                )
                # Worker status bars (show up to 8 active workers)
                worker_pbars = {}
                next_worker_slot = 0
                max_worker_bars = min(8, max_workers)
                
                for future in as_completed(futures):
                    pending.discard(future)
                    idx = futures[future]
                    
                    # Clean up worker bar if it exists
                    if idx in worker_pbars:
                        slot = worker_pbars.pop(idx)
                        # Don't close - just mark as done in main bar
                    
                    try:
                        result_idx, event_segment = future.result(timeout=30)
                        if event_segment is not None:
                            downloaded_segments[result_idx] = event_segment
                        else:
                            failed_count += 1
                    except Exception as e:
                        failed_count += 1
                        logger.debug(f"Download failed for segment {idx}: {e}")
                    
                    main_pbar.update(1)
                    total_done = len(downloaded_segments) + failed_count
                    success_rate = len(downloaded_segments) / total_done * 100 if total_done > 0 else 0
                    main_pbar.set_postfix({
                        'ok': len(downloaded_segments),
                        'fail': failed_count,
                        'rate': f'{success_rate:.0f}%',
                        'active': len(pending)
                    }, refresh=True)
                
                main_pbar.close()
            else:
                # Simple single progress bar (for nohup/logs)
                pbar = tqdm(
                    total=len(futures),
                    desc=f"Downloading ({max_workers}w)",
                    unit="seg",
                    leave=False,
                    dynamic_ncols=True
                )
                
                for future in as_completed(futures):
                    pending.discard(future)
                    try:
                        idx, event_segment = future.result(timeout=30)
                        if event_segment is not None:
                            downloaded_segments[idx] = event_segment
                        else:
                            failed_count += 1
                    except Exception as e:
                        idx = futures[future]
                        failed_count += 1
                        logger.debug(f"Download failed for segment {idx}: {e}")
                    
                    pbar.update(1)
                    total_done = len(downloaded_segments) + failed_count
                    success_rate = len(downloaded_segments) / total_done * 100 if total_done > 0 else 0
                    pbar.set_postfix({
                        'ok': len(downloaded_segments),
                        'fail': failed_count,
                        'rate': f'{success_rate:.0f}%',
                        'active': len(pending)
                    }, refresh=True)
                
                pbar.close()
        
        logger.info(
            f"Downloaded {len(downloaded_segments)}/{len(miss_indices)} segments "
            f"({max_workers} workers, {failed_count} failed)"
        )
        
        # ==============================================================
        # SEQUENTIAL PROCESSING: Extract, cache, and batch results
        # ==============================================================
        
        for miss_idx in miss_indices:
            segment = self.transient_segments[miss_idx]
            channels = downloaded_segments.get(miss_idx)
            if channels is None:
                self._reject_for_no_data(segment)
                continue

            # The stored windows, a channel without finite strain stored as NaN: not
            # held, so a request naming its detector fetches it again (_fill_row).
            onsource_full, offsource_full = self._extract_feature_event(
                channels,
                download_sample_rate,
                num_cache_onsource,
                num_cache_offsource,
                segment.transient_gps_time,
            )
            # Get label and append to cache (buffered)
            label = segment.kind.value if segment.is_glitch else -1
            
            # Build metadata dict from segment
            from .transient_segment import ifos_to_bitmask
            from ..acquisition.base import DataLabel
            metadata = {
                'data_label': DataLabel.GLITCHES.value if segment.is_glitch else DataLabel.EVENTS.value,
                'seen_in': ifos_to_bitmask(segment.seen_in) if segment.seen_in else 0,
                'snr': segment.snr,
                'peak_freq': segment.peak_freq,
                'duration': segment.duration,
                'ml_confidence': segment.ml_confidence,
                'mass1': segment.mass1,
                'mass2': segment.mass2,
                'distance': segment.distance,
                'p_astro': segment.p_astro,
            }
            
            try:
                cache.append_single(
                    np.array(onsource_full),
                    np.array(offsource_full),
                    segment.transient_gps_time,
                    label,
                    metadata=metadata
                )
            except Exception as e:
                logger.warning(f"Failed to append sample to cache (GPS={segment.transient_gps_time:.3f}): {e}")

            if not (np.all(np.isfinite(onsource_full[output_indices]))
                    and np.all(np.isfinite(offsource_full[output_indices]))):
                self._reject_for_no_data(segment)
                continue
            
            # This request's windows, as every cache read takes them
            onsource_cropped, offsource_cropped = TransientCache.read_windows(
                np.asarray(onsource_full),
                np.asarray(offsource_full),
                download_sample_rate,
                sample_rate_hertz,
                window_spec.total_onsource_duration_seconds,
                window_spec.offsource_duration_seconds,
            )
            
            # Slice to requested IFOs
            onsource_output = onsource_cropped[output_indices, :]
            offsource_output = offsource_cropped[output_indices, :]

            batch_subarrays.append(onsource_output * scale_factor)
            batch_backgrounds.append(offsource_output * scale_factor)
            batch_segments.append(segment)
            
            if len(batch_subarrays) >= num_examples_per_batch:
                # A batch served is cached first: the caller may never resume.
                cache.flush_write_buffer()
                yield self._prepare_batch(batch_subarrays, batch_backgrounds, batch_segments, window_spec)
                batch_subarrays.clear()
                batch_backgrounds.clear()
                batch_segments.clear()

        cache.flush_write_buffer()

    def _initialize_transient_cache(
        self,
        ifos: List,
        sample_rate_hertz: float,
        onsource_duration_seconds: float,
        offsource_duration_seconds: float,
        enable_chunked_mode: bool = True,  # NEW: Enable chunked cache by default
        chunk_size: int = 5000,  # Number of samples per memory chunk
        prefix: str = "transient"
    ):
        """
        Initialize or retrieve the transient cache for event/glitch data.
        
        Handles cache path generation, file creation, validation, and
        optional chunked memory caching for improved performance.
        
        Args:
            ifos: List of IFOs to cache
            sample_rate_hertz: Requested sample rate
            onsource_duration_seconds: Requested onsource duration
            offsource_duration_seconds: Requested offsource duration
            enable_chunked_mode: If True, load cache chunks into memory for faster access
            chunk_size: Number of samples per memory chunk (default 5000, ~1-2GB)
            prefix: Cache type prefix ("transient", "glitch", "event")
            
        Returns:
            Tuple of (cache, cache_sample_rate) - TransientCache instance and its sample rate
        """
        from gravyflow.src.dataset.features.glitch_cache import (
            CACHE_SAMPLE_RATE_HERTZ, CACHE_ONSOURCE_DURATION, CACHE_OFFSOURCE_DURATION
        )
        
        # Determine data directory (None = the user cache, gwosc_data.user_cache_dir)
        data_dir = getattr(self, 'data_directory', None)
        
        # Separate cache files for glitches and events (both use same format)
        cache_path = generate_transient_cache_path(data_directory=data_dir, prefix=prefix)
        
        # Manage multiple caches (glitch, event)
        cache_attr = f"_{prefix}_cache"
        
        # Reuse existing cache if path matches (preserves GPS index)
        if getattr(self, cache_attr) is None or getattr(self, cache_attr).path != cache_path:
            setattr(self, cache_attr, DiskCache(cache_path, mode='a'))
        
        cache = getattr(self, cache_attr)
        
        # Validate cache compatibility (sample rate and duration only)
        # IFO validation is not needed - All caches use H1+L1+V1 format
        if cache.exists:
            try:
                cache.validate_request(sample_rate_hertz, onsource_duration_seconds, offsource_duration_seconds)
                
                # Use a unique logging attribute per cache
                logged_attr = f"_{prefix}_cache_logged"
                if not getattr(self, logged_attr):
                    meta = cache.get_metadata()
                    logger.info(f"{prefix.capitalize()} Cache: {meta['num_glitches']} items at {CACHE_SAMPLE_RATE_HERTZ}Hz")
                    setattr(self, logged_attr, True)
                    
                # Enable chunked mode for faster cache hits (if not already enabled)
                if enable_chunked_mode and not cache._chunk_in_memory:
                    meta = cache.get_metadata()
                    if meta['num_glitches'] > 0:
                        cache.enable_chunked_mode(
                            chunk_size=chunk_size,
                            sample_rate_hertz=sample_rate_hertz,
                            onsource_duration=onsource_duration_seconds,
                            offsource_duration=offsource_duration_seconds
                        )
            except ValueError as e:
                logger.warning(f"{prefix.capitalize()} cache incompatible: {e}. Resetting.")
                cache.reset()
        
        # Initialize cache file if needed (or if reset)
        # All caches use H1+L1+V1 format regardless of requested IFOs
        if not cache.exists:
            num_cache_onsource = int(CACHE_ONSOURCE_DURATION * CACHE_SAMPLE_RATE_HERTZ)
            num_cache_offsource = int(CACHE_OFFSOURCE_DURATION * CACHE_SAMPLE_RATE_HERTZ)
            universal_ifos = list(TransientDefaults.UNIVERSAL_IFOS)
            
            cache.initialize_file(
                sample_rate_hertz=CACHE_SAMPLE_RATE_HERTZ,
                onsource_duration=CACHE_ONSOURCE_DURATION,
                offsource_duration=CACHE_OFFSOURCE_DURATION,
                ifo_names=universal_ifos,
                num_ifos=len(universal_ifos),
                onsource_samples=num_cache_onsource,
                offsource_samples=num_cache_offsource
            )
            logger.info(f"Created {prefix.capitalize()} Cache: {cache_path}")
        
        return cache, CACHE_SAMPLE_RATE_HERTZ

    def _yield_onsource_offsource_chunks(
            self,
            sample_rate_hertz: float = None,
            onsource_duration_seconds: float = None,
            crop_duration_seconds: float = None,
            offsource_duration_seconds: float = None,
            num_examples_per_batch: int = None,
            ifos: List[gf.IFO] = None,
            scale_factor: float = None,
            seed: int = None,
            sampling_mode: SamplingMode = SamplingMode.RANDOM,
            whiten: bool = False,
            crop: bool = False,
            group: str = None
        ):
        """
        Generate onsource/offsource chunks for TRANSIENT mode.
        
        Extracts centered windows around each event/glitch.
        
        Args:
            sample_rate_hertz: Sample rate in Hz
            onsource_duration_seconds: Final onsource window duration
            crop_duration_seconds: Cropping duration on each side (for edge effects)
            offsource_duration_seconds: Background window duration
            num_examples_per_batch: Batch size
            ifos: List of interferometers
            scale_factor: Amplitude scaling factor
            seed: Random seed
            sampling_mode: RANDOM or GRID sampling
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
        scale_factor = scale_factor or gf.Defaults.scale_factor
        seed = gf.src.rng.check_root(seed if seed is not None else gf.Defaults.seed)

        # Use shared post-processing from base class
        def _post_process(batch):
            return self._post_process_batch(
                batch, 
                window_spec.sample_rate_hertz, 
                window_spec.onsource_duration_seconds,
                scale_factor,
                whiten, 
                crop
            )
        
        # Use WindowSpec for sample counts (centralized calculation)
        total_onsource_duration_seconds = window_spec.total_onsource_duration_seconds
        sample_rate_hertz = window_spec.sample_rate_hertz
        onsource_duration_seconds = window_spec.onsource_duration_seconds
        offsource_duration_seconds = window_spec.offsource_duration_seconds
        num_onsource_samples = window_spec.num_onsource_samples
        num_offsource_samples = window_spec.num_offsource_samples

        # With no group, the split served, else train (all for specific event names).
        self._serve_group(ifos, seed, group if group is not None else (
            self._served_group or (ALL if self.event_names else "train")
        ))
        if self.rng is None:
            self.rng = default_rng(seed)
        
        # Initialize or retrieve transient caches based on what segments we have
        # Optimization: if sparse request (few segments), disable chunked mode
        is_sparse_request = (len(self.transient_segments) < 100) or (self.event_names is not None)
        enable_chunking = (not is_sparse_request)
        
        # Helper to init a specific cache type
        def ensure_cache(prefix):
             return self._initialize_transient_cache(
                ifos, 
                sample_rate_hertz, 
                total_onsource_duration_seconds, 
                offsource_duration_seconds,
                enable_chunked_mode=enable_chunking,
                prefix=prefix
            )

        # We might need glitch cache, event cache, or both
        has_glitches = any(s.is_glitch for s in self.transient_segments)
        has_events = any(s.is_event for s in self.transient_segments)

        glitch_cache, _ = ensure_cache("glitch") if has_glitches else (None, None)
        event_cache, _ = ensure_cache("event") if has_events else (None, None)
        
        # Build batches: cache hits served instantly, misses downloaded in parallel
        rng = default_rng(seed)
        n_segments = len(self.transient_segments)
        
        # Weighted sampling: if segments have non-uniform weights (from apply_balancing),
        # use weighted choice to oversample rare classes. Otherwise, shuffle uniformly.
        weights = np.array([seg.weight for seg in self.transient_segments], dtype=np.float32)
        
        if np.allclose(weights, 1.0):
            # All weights equal - use simple shuffle (more efficient)
            segment_indices = np.arange(n_segments)
            rng.shuffle(segment_indices)
        else:
            # Weighted sampling - rare classes will appear more often
            # Normalize weights to probabilities
            probs = weights / weights.sum()
            # Sample with replacement to allow oversampling of rare classes
            segment_indices = rng.choice(n_segments, size=n_segments, replace=True, p=probs)
            logger.info(f"Using weighted sampling: {len(np.unique(segment_indices))} unique segments from {n_segments} total")
        
        # Guard: If no segments, raise error rather than spin forever
        if len(segment_indices) == 0:
            raise ValueError(
                f"No transient segments available for iteration. "
                f"len(transient_segments)={len(self.transient_segments)}"
            )
        
        while True:
            # DON'T shuffle here - keep deterministic order for cache hits
            batch_subarrays = []
            batch_backgrounds = []
            batch_segments = []
            
            # Track which indices need downloading
            miss_indices = []
            
            # Progress bar for batch assembly
            pbar = tqdm(
                segment_indices, 
                desc="Assembling batches",
                unit="seg",
                leave=False,
                dynamic_ncols=True
            )
            
            for seg_idx in pbar:
                # Update progress bar description with cache stats
                total = self._cache_hit_count + self._cache_miss_count
                if total > 0:
                    hit_rate = self._cache_hit_count / total * 100
                    pbar.set_postfix({
                        'hits': self._cache_hit_count,
                        'misses': self._cache_miss_count,
                        'hit%': f'{hit_rate:.0f}'
                    })
                # Get TransientSegment (has GPS key and times)
                segment = self.transient_segments[seg_idx]
                if segment.transient_gps_time in self._unavailable:
                    continue
                
                # Select appropriate cache
                active_cache = event_cache if segment.is_event else glitch_cache
                
                # Cache lookup (memory -> disk -> miss) using appropriate cache
                # Skip if force_acquisition is True or cache is not initialized
                onsource, offsource, source = None, None, 'miss'
                
                if active_cache is not None and not self.force_acquisition:
                    # Use GPS time from segment (no recalculation!)
                    onsource, offsource, source = self._get_sample_from_cache(
                        active_cache,
                        segment.transient_gps_time,
                        sample_rate_hertz,
                        total_onsource_duration_seconds,
                        offsource_duration_seconds,
                        scale_factor,
                        target_ifos=[i.name for i in ifos]  # Select specific IFOs from cache
                    )

                # A row without a requested detector's strain does not hold it (a
                # fetch failed, or it was never requested): a miss, fetched again.
                if (source != 'miss' and onsource is not None
                        and np.all(np.isfinite(onsource)) and np.all(np.isfinite(offsource))):
                    # Cache hit (chunk or disk) with valid data
                    batch_subarrays.append(onsource)
                    batch_backgrounds.append(offsource)
                    batch_segments.append(segment)
                    self._cache_hit_count += 1
                    
                    # Log periodically (every 1000 samples)
                    total_samples = self._cache_hit_count + self._cache_miss_count
                    if total_samples - self._last_log_count >= 1000:
                        hit_rate = self._cache_hit_count / total_samples * 100
                        logger.info(f"Cache stats: {self._cache_hit_count} hits, {self._cache_miss_count} misses ({hit_rate:.1f}% hit rate)")
                        self._last_log_count = total_samples
                    
                    if len(batch_subarrays) >= num_examples_per_batch:
                        yield _post_process(self._prepare_batch(batch_subarrays, batch_backgrounds, batch_segments, window_spec))
                        batch_subarrays = []
                        batch_backgrounds = []
                        batch_segments = []
                    continue
                
                # Cache miss, or a row not holding a requested detector - download
                miss_indices.append(seg_idx)
                self._cache_miss_count += 1
                
                # Download in batches for efficiency
                if len(miss_indices) >= num_examples_per_batch:
                    # Determine cache to use for this batch (assuming mixed types handled by helpers)
                    # NOTE: The _process_cache_misses helper takes a single 'cache' arg.
                    # Since we might have mixed types in miss_indices, we need to handle that.
                    # Simplification: Splitting miss_indices by type is complex inside the loop.
                    # HOWEVER, _process_cache_misses writes to 'cache'.
                    # Solution: Call process twice? Or make process handle it?
                    # For now, let's group miss_indices by type.
                    
                    # Group indices by type (event vs glitch)
                    event_misses = [idx for idx in miss_indices if self.transient_segments[idx].is_event]
                    glitch_misses = [idx for idx in miss_indices if self.transient_segments[idx].is_glitch]
                    
                    if event_misses and event_cache:
                         for batch in self._process_cache_misses(
                            event_misses, ifos, window_spec, scale_factor, event_cache, 
                            batch_subarrays, batch_backgrounds, batch_segments, num_examples_per_batch
                        ):
                            yield _post_process(batch)
                         
                    if glitch_misses and glitch_cache:
                         for batch in self._process_cache_misses(
                            glitch_misses, ifos, window_spec, scale_factor, glitch_cache, 
                            batch_subarrays, batch_backgrounds, batch_segments, num_examples_per_batch
                        ):
                            yield _post_process(batch)
                    
                    miss_indices = []

            # Handle remaining miss_indices after for loop ends
            if len(miss_indices) > 0:
                    event_misses = [idx for idx in miss_indices if self.transient_segments[idx].is_event]
                    glitch_misses = [idx for idx in miss_indices if self.transient_segments[idx].is_glitch]
                    
                    if event_misses and event_cache:
                         for batch in self._process_cache_misses(
                            event_misses, ifos, window_spec, scale_factor, event_cache, 
                            batch_subarrays, batch_backgrounds, batch_segments, num_examples_per_batch
                        ):
                            yield _post_process(batch)

                    if glitch_misses and glitch_cache:
                         for batch in self._process_cache_misses(
                            glitch_misses, ifos, window_spec, scale_factor, glitch_cache, 
                            batch_subarrays, batch_backgrounds, batch_segments, num_examples_per_batch
                        ):
                            yield _post_process(batch)

            # Yield any remaining partial batch at end of epoch
            if len(batch_subarrays) > 0:
                yield _post_process(self._prepare_batch(batch_subarrays, batch_backgrounds, batch_segments, window_spec))
