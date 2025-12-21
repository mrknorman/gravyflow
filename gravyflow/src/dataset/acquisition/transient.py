"""
Transient mode data acquisition.

This module provides TransientDataObtainer for acquiring data around
specific transient events (GW events, glitches) in TRANSIENT mode.
"""

import logging
import hashlib
from typing import List, Dict, Optional, Union, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

import numpy as np
from numpy.random import default_rng
from keras import ops

import gravyflow as gf
from gravyflow.src.dataset.features.event import EventConfidence
from gravyflow.src.dataset.features.glitch_cache import GlitchCache, generate_glitch_cache_path
from .cache import DiskCache
from gravyflow.src.dataset.features.injection import ReturnVariables as RV
from .base import (
    BaseDataObtainer, DataQuality, DataLabel, SegmentOrder, 
    AcquisitionMode, SamplingMode, ObservingRun, IFOData, ensure_even
)
from gravyflow.src.utils.numerics import ensure_list
from gravyflow.src.utils.shapes import ShapeEnforcer
from gravyflow.src.utils.gps import gps_to_key, gps_array_to_keys
from gravyflow.src.dataset.features.transient_index import TransientIndex
from .transient_segment import TransientSegment
from gravyflow.src.dataset.config import TransientDefaults, WindowSpec


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
            random_sign_reversal: bool = True,
            random_time_reversal: bool = True,
            augmentation_probability: float = 0.5,
            prefetch_segments: int = 64,  # Higher for smaller TRANSIENT segments
            # TRANSIENT mode augmentations
            random_shift: bool = False,
            shift_fraction: float = 0.25,
            add_noise: bool = False,
            noise_amplitude: float = 0.1,
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
            random_sign_reversal=random_sign_reversal,
            random_time_reversal=random_time_reversal,
            augmentation_probability=augmentation_probability,
            prefetch_segments=prefetch_segments,
        )
        
        self.acquisition_mode = AcquisitionMode.TRANSIENT
        self._segment_cache_maxsize = 128  # TRANSIENT uses smaller, more segments
        
        # Event type filtering
        if event_types is None:
            event_types = [EventConfidence.CONFIDENT]
        self.event_types = event_types
        
        # TRANSIENT-specific augmentations
        self.random_shift = random_shift
        self.shift_fraction = shift_fraction
        self.add_noise = add_noise
        self.noise_amplitude = noise_amplitude
        
        # Class balancing
        self.balanced_glitch_types = balanced_glitch_types
        
        # TransientIndex - canonical source of truth for transient examples
        self._feature_index: Optional[TransientIndex] = None
        
        # TransientSegment objects (replaces raw numpy segments)
        self.transient_segments: List[TransientSegment] = []

    def _apply_augmentation(self, data, is_transient: bool = True):
        """
        Apply augmentations including TRANSIENT-specific ones.
        """
        data = super()._apply_augmentation(data, is_transient=False)
        
        if self.rng is None:
            return data
        
        # TRANSIENT-specific augmentations
        if is_transient:
            # Random shift (move event off-center)
            if self.random_shift and self.rng.random() < self.augmentation_probability:
                num_samples = int(ops.shape(data)[-1])
                max_shift = int(self.shift_fraction * num_samples)
                shift = self.rng.integers(-max_shift, max_shift + 1)
                data = ops.roll(data, shift, axis=-1)
            
            # Add noise perturbations
            if self.add_noise and self.rng.random() < self.augmentation_probability:
                data_np = ops.convert_to_numpy(data)
                noise = self.rng.normal(0, self.noise_amplitude, data_np.shape)
                std = np.std(data_np)
                noise = noise * std if std > 0 else noise
                data = ops.convert_to_tensor(data_np + noise, dtype=data.dtype)
            
        return data

    def build_feature_index(
        self,
        ifos: List[gf.IFO],
        groups: Dict[str, float] = None,
        seed: int = 42
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
        
        
        # Fetch event segments if requested  
        data_labels_list = self.data_labels if isinstance(self.data_labels, list) else [self.data_labels]
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
            if self.event_names:
                effective_runs = None  # None = query all runs
            
            logger.info(f"Building event segments (runs={effective_runs})")
            event_segments = build_event_segments(
                observing_runs=effective_runs,
                confidences=self.event_types if self.event_types else [EventConfidence.CONFIDENT]
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
        if groups:
            index.assign_groups(groups, seed=seed)
        
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
        gps_key: Optional[int] = None,  # NEW: Pass key directly to avoid conversion
        target_ifos: Optional[List[str]] = None  # NEW: Requested IFOs for subset selection
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
        """
        Attempt to retrieve a sample from cache (memory first, then disk).
        
        Args:
            gps_key: Optional GPS key (preferred - no conversion needed)
            gps_time: GPS time (fallback if key not provided)
            target_ifos: List of IFO names to retrieve (must be present in cache)
            
        Returns:
            (onsource, offsource, source) where source is 'memory', 'disk', or 'miss'
        """
        # Compute key if not provided (backward compatibility)
        if gps_key is None:
            gps_key = gps_to_key(gps_time)
        
        # Try memory cache first (fastest)
        if cache.in_memory and cache.has_key(gps_key):
            idx = cache._gps_index[gps_key]
            onsource = cache._mem_onsource[idx]
            offsource = cache._mem_offsource[idx]
            
            # Filter channels if requested
            if target_ifos:
                meta = cache.get_metadata()
                stored_ifos = meta.get('ifo_names', [])
                if stored_ifos:
                    try:
                        channel_indices = [stored_ifos.index(ifo) for ifo in target_ifos]
                        onsource = onsource[channel_indices]
                        offsource = offsource[channel_indices]
                    except ValueError:
                        return None, None, 'miss'
                elif len(target_ifos) != onsource.shape[0]:
                    return None, None, 'miss'

            return onsource * scale_factor, offsource * scale_factor, 'memory'
        
        # Try disk cache (use key-based lookup)
        if cache.has_key(gps_key):
            # Resolve index directly
            idx = cache._gps_index[gps_key]
            
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
        
        # Cache miss
        return None, None, 'miss'

    def _prepare_batch(
        self,
        batch_subarrays: list,
        batch_backgrounds: list,
        batch_segments: list  # List of TransientSegments
    ) -> dict:
        """
        Prepare batch tensors from accumulated lists.
        
        Returns:
            Dict with ReturnVariables keys:
            - ONSOURCE: (Batch, IFO, Samples) = BIS
            - OFFSOURCE: (Batch, IFO, Samples) = BIS  
            - TRANSIENT_GPS_TIME: (Batch, IFO) = BI
              NOTE: This is the GPS time of the EVENT/GLITCH CENTER, not the start
              of the data window. For window start time, use the segment boundaries.
            - DATA_LABEL: (Batch, IFO) = BI (1 for GLITCHES, 2 for EVENTS)
            - SUB_TYPE: (Batch, IFO) = BI (GlitchType.value or SourceType.value)
            - GLITCH_TYPE: (Batch, IFO) = BI (GlitchType.value if glitch, -1 otherwise)
            - SOURCE_TYPE: (Batch, IFO) = BI (SourceType.value if event, -1 otherwise)
        """
        final_subarrays = ops.cast(ops.stack(batch_subarrays), "float32")
        final_background = ops.cast(ops.stack(batch_backgrounds), "float32")
        
        # Get dimensions from subarrays shape: (Batch, IFO, Samples)
        batch_size = int(ops.shape(final_subarrays)[0])
        num_ifos = int(ops.shape(final_subarrays)[1])
        
        # Extract GPS times from segments
        batch_gps_times = [seg.transient_gps_time for seg in batch_segments]
        gps_1d = ops.convert_to_tensor(batch_gps_times, dtype="float64")
        transient_gps = ops.tile(ops.expand_dims(gps_1d, axis=-1), (1, num_ifos))
        
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
            RV.ONSOURCE: self._apply_augmentation(final_subarrays, is_transient=True),
            RV.OFFSOURCE: self._apply_augmentation(final_background, is_transient=True),
            RV.TRANSIENT_GPS_TIME: transient_gps,
            RV.DATA_LABEL: data_label,     # Per-segment: GLITCHES=1 or EVENTS=2
            RV.SUB_TYPE: sub_type,         # Per-segment: GlitchType.value or SourceType.value
            RV.GLITCH_TYPE: glitch_type,   # Per-segment: GlitchType.value or -1
            RV.SOURCE_TYPE: source_type,   # Per-segment: SourceType.value or -1
        }

    def _extract_feature_event(
        self,
        event_segment: IFOData,
        num_onsource_samples: int,
        num_offsource_samples: int,
        sample_rate_hertz: float,
        known_gps_time: float = None  # Use this as ground truth if provided
    ) -> tuple:
        """
        Extract onsource/offsource windows from a feature segment.
        Centers extraction on the middle of the segment (where the event is).
        
        Args:
            event_segment: IFOData containing the downloaded segment
            num_onsource_samples: Number of samples for onsource window
            num_offsource_samples: Number of samples for offsource window
            sample_rate_hertz: Sample rate
            known_gps_time: If provided, use this as the event GPS time instead of computing
                           from segment position. Recommended to pass TransientSegment.transient_gps_time.
        
        Returns:
            Tuple of (onsource, offsource, event_gps_time) or (None, None, None) if extraction fails.
        """
        data_len = ops.shape(event_segment.data[0])[0]
        
        if data_len < num_onsource_samples:
            logger.warning(f"Feature segment too short ({data_len} < {num_onsource_samples})")
            return None, None, None
        
        center_idx = data_len // 2
        half_onsource = num_onsource_samples // 2
        start_onsource = center_idx - half_onsource
        end_onsource = start_onsource + num_onsource_samples
        
        # Use known GPS time (ground truth) if provided, otherwise compute from segment
        if known_gps_time is not None:
            event_gps_time = known_gps_time
        else:
            feature_gps_start = event_segment.start_gps_time[0]
            event_gps_time = feature_gps_start + (center_idx / sample_rate_hertz)
        
        temp_onsource = []
        temp_offsource = []
        
        for channel_data in event_segment.data:
            chunk = channel_data[start_onsource:end_onsource]
            if ops.shape(chunk)[0] != num_onsource_samples:
                return None, None, None
            temp_onsource.append(chunk)
            
            off_chunk = None
            
            # Try to get offsource BEFORE onsource
            available_before = start_onsource
            if available_before >= num_offsource_samples:
                off_start = start_onsource - num_offsource_samples
                off_chunk = channel_data[off_start:start_onsource]
            elif available_before > 0:
                off_chunk = channel_data[0:start_onsource]
                pad_needed = num_offsource_samples - available_before
                off_chunk = ops.pad(off_chunk, [(pad_needed, 0)], mode='edge')
            
            # If no data before, try AFTER onsource
            if off_chunk is None:
                available_after = data_len - end_onsource
                if available_after >= num_offsource_samples:
                    off_chunk = channel_data[end_onsource:end_onsource + num_offsource_samples]
                elif available_after > 0:
                    off_chunk = channel_data[end_onsource:data_len]
                    pad_needed = num_offsource_samples - available_after
                    off_chunk = ops.pad(off_chunk, [(0, pad_needed)], mode='edge')
            
            if off_chunk is None:
                logger.warning(
                    f"No offsource data available for event at GPS {event_gps_time:.1f}. Skipping."
                )
                return None, None, None
            
            temp_offsource.append(off_chunk)
        
        onsource_stacked = ops.stack(temp_onsource)
        offsource_stacked = ops.stack(temp_offsource)
        
        return onsource_stacked, offsource_stacked, event_gps_time


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
        Direct TransientSegment usage.
        """
        if not isinstance(ifos, list):
            ifos = [ifos]
        
        if not segment_order:
            segment_order = self.segment_order
        
        if not groups:
            groups = {"train": 0.98, "validate": 0.01, "test": 0.01}
        
        if group_name == "all":
            groups = {"all": 1.0}
        
        # Build index if needed
        if self._feature_index is None:
            self.build_feature_index(ifos=ifos, groups=groups, seed=seed)
        
        # Get segments for this group, filtered by requested IFOs
        # Segments with seen_in overlapping requested IFOs will be included
        self.transient_segments = list(self._feature_index.iter(
            group=group_name,
            ifos_filter=ifos
        ))
        
        if len(self.transient_segments) == 0:
            logger.warning(f"No segments found for group '{group_name}'")
            self.valid_segments = np.empty((0, len(ifos), 2), dtype=np.float64)
            return self.valid_segments
        
        # Shuffle if needed
        if segment_order == SegmentOrder.RANDOM:
            rng = default_rng(seed)
            rng.shuffle(self.transient_segments)
        elif segment_order == SegmentOrder.SHORTEST_FIRST:
            self.transient_segments.sort(key=lambda s: s.duration)
        
        # Create valid_segments array for compatibility
        num_ifos = len(ifos)
        segments_2d = np.array([
            [seg.start_gps_time, seg.end_gps_time]
            for seg in self.transient_segments
        ], dtype=np.float64)
        
        expanded = np.expand_dims(segments_2d, axis=1)
        self.valid_segments = np.repeat(expanded, num_ifos, axis=1)
        
        logger.info(
            f"TRANSIENT MODE: {len(self.transient_segments)} segments for '{group_name}'"
        )
        
        return self.valid_segments


    def _process_cache_misses(
        self,
        miss_indices: List[int],
        ifos: List,
        sample_rate_hertz: float,
        total_onsource_duration_seconds: float,
        offsource_duration_seconds: float,
        scale_factor: float,
        cache,
        batch_subarrays: List,
        batch_backgrounds: List,
        batch_segments: List,
        num_examples_per_batch: int
    ):
        """
        Download, extract, cache and prepare batches for cache misses.
        
        Consolidates duplicated logic for processing segments that weren't in cache.
        Yields batches whenever batch_size is reached.
        
        Args:
            miss_indices: List of segment indices that need downloading
            ifos: List of IFOs to download
            sample_rate_hertz: Target sample rate for output
            total_onsource_duration_seconds: Onsource duration including padding
            offsource_duration_seconds: Offsource duration
            scale_factor: Amplitude scale factor
            cache: DiskCache instance for storing/retrieving
            batch_subarrays: Accumulated onsource arrays (mutated)
            batch_backgrounds: Accumulated offsource arrays (mutated)
            batch_segments: Accumulated TransientSegments (mutated)
            num_examples_per_batch: Batch size threshold
            
        Yields:
            Batch dicts when batch_size is reached
        """
        from gravyflow.src.dataset.features.glitch import GlitchType
        from gravyflow.src.dataset.features.glitch_cache import (
            CACHE_SAMPLE_RATE_HERTZ, CACHE_ONSOURCE_DURATION, CACHE_OFFSOURCE_DURATION
        )
        
        if len(miss_indices) == 0:
            return
            
        miss_segments = self.valid_segments[miss_indices]
        
        # Download at higher of requested or cache sample rate (can downsample but not upsample)
        download_sample_rate = max(sample_rate_hertz, CACHE_SAMPLE_RATE_HERTZ)
        
        # =====================================================================
        # UNIVERSAL CACHE STRATEGY
        # =====================================================================
        # Always download ALL detectors (H1, L1, V1) regardless of what the user
        # requested. This ensures the cache is maximally reusable:
        #   - A cache built with [H1, L1, V1] can serve any subset
        #   - No need to rebuild cache when switching from L1-only to H1+L1
        #   - Downloading all upfront is one network request vs N separate ones
        # 
        # Requested IFOs are filtered at OUTPUT time, not download time.
        # =====================================================================
        
        # Convert UNIVERSAL_IFOS to IFO enum objects
        all_detector_ifos = [gf.IFO[name] for name in TransientDefaults.UNIVERSAL_IFOS]
        universal_ifo_names = list(TransientDefaults.UNIVERSAL_IFOS)
        
        # Build segments array for ALL detectors (repeating the same GPS boundaries)
        num_universal_ifos = len(all_detector_ifos)
        miss_segment_boundaries = miss_segments[:, 0, :]  # Shape (N, 2) - take first IFO's times
        universal_miss_segments = np.repeat(
            miss_segment_boundaries[:, np.newaxis, :], 
            num_universal_ifos, 
            axis=1
        )  # Shape (N, num_universal_ifos, 2)
        
        # Single acquire() call for ALL detectors
        segment_generator = self.acquire(
            download_sample_rate,
            universal_miss_segments,
            all_detector_ifos,
            1.0  # Download raw data for cache (scale=1.0)
        )
        
        for miss_idx_position, event_segment in enumerate(segment_generator):
            true_seg_idx = miss_indices[miss_idx_position]
            segment = self.transient_segments[true_seg_idx]
            
            # Extract at cache parameters
            num_cache_onsource = int(CACHE_ONSOURCE_DURATION * download_sample_rate)
            num_cache_offsource = int(CACHE_OFFSOURCE_DURATION * download_sample_rate)
            
            onsource_full, offsource_full, _ = self._extract_feature_event(
                event_segment,
                num_cache_onsource,
                num_cache_offsource,
                download_sample_rate
            )
            
            if onsource_full is None:
                # Track dropped segment (counter initialized in BaseDataObtainer.__init__)
                self._dropped_segments_count += 1
                logger.warning(f"Dropped segment (extraction failed): GPS={segment.transient_gps_time:.3f}, total dropped={self._dropped_segments_count}")
                continue

            # Data quality checks
            if np.isnan(onsource_full).any():
                logger.error(f"NAN DETECTED: In onsource_full after extract! GPS: {segment.transient_gps_time}")
            if np.all(onsource_full == 0):
                logger.error(f"ZERO SIGNAL DETECTED: onsource_full is all zeros! GPS: {segment.transient_gps_time}")

            # Get label from segment
            label = segment.kind.value if segment.is_glitch else -1
            
            # Append to cache
            try:
                cache.append_single(
                    np.array(onsource_full), 
                    np.array(offsource_full), 
                    segment.transient_gps_time,
                    label,
                    gps_key=segment.gps_key
                )
            except Exception as e:
                logger.warning(f"Failed to append sample to cache (GPS={segment.transient_gps_time:.3f}): {e}")
            
            # Crop and resample for this request
            onsource_cropped = GlitchCache.crop_and_resample(
                np.asarray(onsource_full),
                download_sample_rate,
                sample_rate_hertz,
                total_onsource_duration_seconds
            )
            offsource_cropped = GlitchCache.crop_and_resample(
                np.asarray(offsource_full),
                download_sample_rate,
                sample_rate_hertz,
                offsource_duration_seconds
            )
            
            # Filter to requested IFOs for output
            # universal_ifo_names is the cache order (H1, L1, V1)
            # ifos is the user's requested subset
            try:
                output_indices = [universal_ifo_names.index(ifo.name) for ifo in ifos]
            except ValueError as e:
                logger.error(f"Requested IFO not found in universal set! Request: {[i.name for i in ifos]}, Universal: {universal_ifo_names}")
                raise e

            # Slice the cropped data
            onsource_output = onsource_cropped[output_indices, :]
            offsource_output = offsource_cropped[output_indices, :]

            if np.isnan(onsource_output).any():
                logger.error(f"NAN DETECTED: In onsource_output! GPS: {segment.transient_gps_time}")
            
            batch_subarrays.append(onsource_output * scale_factor)
            batch_backgrounds.append(offsource_output * scale_factor)
            batch_segments.append(segment)
            
            if len(batch_subarrays) >= num_examples_per_batch:
                yield self._prepare_batch(batch_subarrays, batch_backgrounds, batch_segments)
                batch_subarrays.clear()
                batch_backgrounds.clear()
                batch_segments.clear()

    def _initialize_glitch_cache(
        self,
        ifos: List,
        sample_rate_hertz: float,
        onsource_duration_seconds: float,
        offsource_duration_seconds: float
    ):
        """
        Initialize or retrieve the glitch cache for transient data.
        
        Handles cache path generation, file creation, and validation.
        
        Args:
            ifos: List of IFOs to cache
            sample_rate_hertz: Requested sample rate
            onsource_duration_seconds: Requested onsource duration
            offsource_duration_seconds: Requested offsource duration
            
        Returns:
            Tuple of (cache, cache_sample_rate) - GlitchCache instance and its sample rate
        """
        from gravyflow.src.dataset.features.glitch_cache import (
            CACHE_SAMPLE_RATE_HERTZ, CACHE_ONSOURCE_DURATION, CACHE_OFFSOURCE_DURATION
        )
        
        # Determine data directory
        data_dir = Path("./generator_data")
        if hasattr(self, 'data_directory') and self.data_directory:
            data_dir = self.data_directory
        
        # Get observing run for cache path
        observing_run_name = "unknown"
        if self.observing_runs:
            if isinstance(self.observing_runs, list):
                observing_run_name = self.observing_runs[0].name
            else:
                observing_run_name = self.observing_runs.name
        
        # Universal Cache: path is independent of requested IFOs since cache
        # always contains all detectors (H1, L1, V1)
        cache_path = generate_glitch_cache_path(
            observing_run=observing_run_name,
            ifo="universal",  # Universal Cache - not specific to requested IFOs
            data_directory=data_dir
        )
        
        # Reuse existing cache if path matches (preserves GPS index)
        if not hasattr(self, '_glitch_cache') or self._glitch_cache is None or self._glitch_cache.path != cache_path:
            self._glitch_cache = DiskCache(cache_path, mode='a')
        
        cache = self._glitch_cache
        
        # Validate cache compatibility (sample rate and duration only)
        # IFO validation is not needed - Universal Cache always has H1+L1+V1
        if cache.exists:
            try:
                cache.validate_request(sample_rate_hertz, onsource_duration_seconds, offsource_duration_seconds)
                
                if not hasattr(self, '_cache_logged'):
                    meta = cache.get_metadata()
                    logger.info(f"Cache: {meta['num_glitches']} glitches at {CACHE_SAMPLE_RATE_HERTZ}Hz")
                    self._cache_logged = True
            except ValueError as e:
                logger.warning(f"Cache incompatible: {e}. Resetting.")
                cache.reset()
        
        # Initialize cache file if needed (or if reset)
        # Universal Cache: Always use H1+L1+V1 regardless of requested IFOs
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
            logger.info(f"Created Universal Cache: {cache_path}")
        
        return cache, CACHE_SAMPLE_RATE_HERTZ

    def get_onsource_offsource_chunks(
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
            crop: bool = False
        ):
        """
        Wrapper to enforce shape contracts on generator.
        
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
        gen = self._yield_onsource_offsource_chunks(
            sample_rate_hertz,
            onsource_duration_seconds,
            crop_duration_seconds,
            offsource_duration_seconds,
            num_examples_per_batch,
            ifos,
            scale_factor,
            seed,
            sampling_mode,
            whiten,
            crop
        )

        # Normalize ifos to list
        ifos = ensure_list(ifos) or [gf.IFO.L1]

        return ShapeEnforcer.wrap_generator(gen, num_ifos=len(ifos))

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
            crop: bool = False
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
        seed = seed or gf.Defaults.seed
        if self.rng is None:
            self.rng = default_rng(seed)

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

        # Auto-initialize segments if not already done
        # For specific event names, use group="all" to get all matching events
        if not self.transient_segments:
            group_name = "all" if self.event_names else "train"
            self.get_valid_segments(
                ifos=ifos, 
                seed=seed,
                group_name=group_name
        )
        
        # Initialize or retrieve glitch cache (extracted to helper for clarity)
        cache, _ = self._initialize_glitch_cache(
            ifos, sample_rate_hertz, onsource_duration_seconds, offsource_duration_seconds
        )
        
        # Build batches: cache hits served instantly, misses downloaded in parallel
        rng = default_rng(seed)
        segment_indices = np.arange(len(self.transient_segments))
        
        # Shuffle once with fixed seed so order is deterministic across runs
        # This ensures cache hits for previously downloaded glitches
        rng.shuffle(segment_indices)
        
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
            
            # Debug counters for cache tracking
            if not hasattr(self, '_cache_hit_count'):
                self._cache_hit_count = 0
                self._cache_miss_count = 0
                self._last_log_count = 0
            
            for seg_idx in segment_indices:
                # Get TransientSegment (has GPS key and times)
                segment = self.transient_segments[seg_idx]
                
                # Unified cache lookup (memory -> disk -> miss)
                # Use GPS time from segment (no recalculation!)
                onsource, offsource, source = self._get_sample_from_cache(
                    cache,
                    segment.transient_gps_time,
                    sample_rate_hertz,
                    total_onsource_duration_seconds,
                    offsource_duration_seconds,
                    scale_factor,
                    gps_key=segment.gps_key,  # Pass key - eliminates float→key conversion!
                    target_ifos=[i.name for i in ifos]  # Select specific IFOs from cache
                )
                
                if source != 'miss':
                    # Cache hit (memory or disk)
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
                        yield _post_process(self._prepare_batch(batch_subarrays, batch_backgrounds, batch_segments))
                        batch_subarrays = []
                        batch_backgrounds = []
                        batch_segments = []
                    continue
                
                # Cache miss - need to download this one
                miss_indices.append(seg_idx)
                self._cache_miss_count += 1
                
                # Download in batches for efficiency
                if len(miss_indices) >= num_examples_per_batch:
                    # Use consolidated helper for download/cache/batch
                    for batch in self._process_cache_misses(
                        miss_indices,
                        ifos,
                        sample_rate_hertz,
                        total_onsource_duration_seconds,
                        offsource_duration_seconds,
                        scale_factor,
                        cache,
                        batch_subarrays,
                        batch_backgrounds,
                        batch_segments,
                        num_examples_per_batch
                    ):
                        yield _post_process(batch)
                    miss_indices = []

            # Handle remaining miss_indices after for loop ends
            if len(miss_indices) > 0:
                # Use consolidated helper for remaining cache misses
                for batch in self._process_cache_misses(
                    miss_indices,
                    ifos,
                    sample_rate_hertz,
                    total_onsource_duration_seconds,
                    offsource_duration_seconds,
                    scale_factor,
                    cache,
                    batch_subarrays,
                    batch_backgrounds,
                    batch_segments,
                    num_examples_per_batch
                ):
                    yield _post_process(batch)

            # Yield any remaining partial batch at end of epoch
            if len(batch_subarrays) > 0:
                yield _post_process(self._prepare_batch(batch_subarrays, batch_backgrounds, batch_segments))
