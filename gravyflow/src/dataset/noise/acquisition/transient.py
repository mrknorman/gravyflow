"""
Transient mode data acquisition.

This module provides TransientDataObtainer for acquiring data around
specific transient events (GW events, glitches) in TRANSIENT mode.
"""

import logging
import hashlib
from typing import List, Dict, Optional, Union, Tuple
from pathlib import Path

import numpy as np
from numpy.random import default_rng
from keras import ops

import gravyflow as gf
from gravyflow.src.dataset.features.event import EventType
from gravyflow.src.dataset.features.glitch_cache import GlitchCache, generate_glitch_cache_path
from .base import (
    BaseDataObtainer, DataQuality, DataLabel, SegmentOrder, 
    AcquisitionMode, SamplingMode, ObservingRun, IFOData, ensure_even
)


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
            event_types: List[EventType] = None,
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
            event_names : List[str] = None
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
            event_types = [EventType.CONFIDENT]
        self.event_types = event_types
        
        # TRANSIENT-specific augmentations
        self.random_shift = random_shift
        self.shift_fraction = shift_fraction
        self.add_noise = add_noise
        self.noise_amplitude = noise_amplitude
        
        # Class balancing
        self.balanced_glitch_types = balanced_glitch_types
        
        # Feature storage
        self.feature_segments = None
        self._feature_labels = {}
        self._sorted_label_indices = None  # Cache for vectorized label lookup

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

    def _lookup_labels(self, gps_times):
        """
        Look up glitch type labels for given GPS times.
        
        Uses vectorized binary search for O(M log N) complexity where M is the
        number of requested GPS times and N is the number of known labels.
        """
        gps_times = np.asarray(gps_times)
        
        if not self._feature_labels:
            return np.full(len(gps_times), -1, dtype=np.int32)
        
        if gf.DataLabel.GLITCHES not in self._feature_labels:
            return np.full(len(gps_times), -1, dtype=np.int32)
        
        times_arr, labels_arr = self._feature_labels[gf.DataLabel.GLITCHES]
        
        if len(times_arr) == 0:
            return np.full(len(gps_times), -1, dtype=np.int32)
        
        # Cache sorted indices for repeated calls (lazy initialization)
        if not hasattr(self, '_sorted_label_indices') or self._sorted_label_indices is None:
            self._sorted_label_indices = np.argsort(times_arr)
            self._sorted_label_times = times_arr[self._sorted_label_indices]
            self._sorted_label_values = labels_arr[self._sorted_label_indices]
        
        sorted_times = self._sorted_label_times
        sorted_labels = self._sorted_label_values
        
        # Vectorized binary search - O(M log N)
        insert_indices = np.searchsorted(sorted_times, gps_times)
        
        # Clip to valid range
        insert_indices = np.clip(insert_indices, 0, len(sorted_times) - 1)
        
        # Calculate distance to nearest cached time
        diffs = np.abs(sorted_times[insert_indices] - gps_times)
        
        # Also check the previous index (searchsorted gives insertion point, not nearest)
        prev_indices = np.clip(insert_indices - 1, 0, len(sorted_times) - 1)
        prev_diffs = np.abs(sorted_times[prev_indices] - gps_times)
        
        # Use whichever is closer
        use_prev = prev_diffs < diffs
        final_indices = np.where(use_prev, prev_indices, insert_indices)
        final_diffs = np.where(use_prev, prev_diffs, diffs)
        
        # Apply tolerance threshold (1.0 second)
        labels = np.where(final_diffs < 1.0, sorted_labels[final_indices], -1)
        
        return labels.astype(np.int32)

    def _get_sample_from_cache(
        self,
        cache,
        gps_time: float,
        sample_rate_hertz: float,
        onsource_duration: float,
        offsource_duration: float,
        scale_factor: float = 1.0
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
        """
        Attempt to retrieve a sample from cache (memory first, then disk).
        
        Returns:
            (onsource, offsource, source) where source is 'memory', 'disk', or 'miss'
        """
        # Try memory cache first (fastest)
        if cache.in_memory and cache.has_gps(gps_time):
            closest_gps = cache.get_closest_gps(gps_time)
            if closest_gps is not None:
                idx = cache._gps_index.get(closest_gps)
                if idx is not None:
                    onsource = cache._mem_onsource[idx] * scale_factor
                    offsource = cache._mem_offsource[idx] * scale_factor
                    return onsource, offsource, 'memory'
        
        # Try disk cache
        if cache.has_gps(gps_time):
            result = cache.get_by_gps(
                gps_time,
                sample_rate_hertz=sample_rate_hertz,
                onsource_duration=onsource_duration,
                offsource_duration=offsource_duration
            )
            if result is not None:
                onsource, offsource, _, _ = result
                return onsource * scale_factor, offsource * scale_factor, 'disk'
        
        # Cache miss
        return None, None, 'miss'

    def _prepare_batch(
        self,
        batch_subarrays: list,
        batch_backgrounds: list,
        batch_gps_times: list
    ) -> Tuple:
        """
        Prepare batch tensors from accumulated lists.
        
        Returns:
            (subarrays, backgrounds, gps_tensor, labels)
        """
        final_subarrays = ops.cast(ops.stack(batch_subarrays), "float32")
        final_background = ops.cast(ops.stack(batch_backgrounds), "float32")
        final_gps = ops.expand_dims(
            ops.convert_to_tensor(batch_gps_times, dtype="float64"), 
            axis=-1
        )
        batch_labels = self._lookup_labels(batch_gps_times)
        
        return (
            self._apply_augmentation(final_subarrays, is_transient=True),
            self._apply_augmentation(final_background, is_transient=True),
            final_gps,
            batch_labels
        )

    def _extract_feature_event(
        self,
        event_segment: IFOData,
        num_onsource_samples: int,
        num_offsource_samples: int,
        sample_rate_hertz: float
    ) -> tuple:
        """
        Extract onsource/offsource windows from a feature segment.
        Centers extraction on the middle of the segment (where the event is).
        """
        data_len = ops.shape(event_segment.data[0])[0]
        
        if data_len < num_onsource_samples:
            logging.warning(f"Feature segment too short ({data_len} < {num_onsource_samples})")
            return None, None, None
        
        center_idx = data_len // 2
        half_onsource = num_onsource_samples // 2
        start_onsource = center_idx - half_onsource
        end_onsource = start_onsource + num_onsource_samples
        
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
                logging.warning(
                    f"No offsource data available for event at GPS {event_gps_time:.1f}. Skipping."
                )
                return None, None, None
            
            temp_offsource.append(off_chunk)
        
        onsource_stacked = ops.stack(temp_onsource)
        offsource_stacked = ops.stack(temp_offsource)
        
        return onsource_stacked, offsource_stacked, event_gps_time

    def return_wanted_segments(
            self,
            ifo: gf.IFO,
            valid_segments: np.ndarray,        
            start_padding_seconds: float = 32.0,
            end_padding_seconds: float = 32.0,
        ):
        """
        Get segments containing desired features (events/glitches).
        """
        from gravyflow.src.dataset.features.event import EventType
        from gravyflow.src.dataset.features.glitch import GlitchType
        
        wanted_segments = []
        feature_times = {}
        
        # Event handling
        has_events_supersede = DataLabel.EVENTS in self.data_labels
        event_types_in_labels = [
            label for label in self.data_labels 
            if isinstance(label, EventType)
        ]
        
        if has_events_supersede or event_types_in_labels:
            if has_events_supersede:
                event_times = gf.get_all_event_times()
            else:
                event_times = gf.get_event_times_by_type(event_types_in_labels)
            
            if len(event_times) > 0:
                wanted_segments.append(
                    self.pad_gps_times_with_veto_window(
                        event_times,
                        start_padding_seconds=start_padding_seconds,
                        end_padding_seconds=end_padding_seconds
                    )
                )
                feature_times[gf.DataLabel.EVENTS] = event_times
        
        # Glitch handling
        has_glitches_supersede = DataLabel.GLITCHES in self.data_labels
        glitch_types_in_labels = [
            label for label in self.data_labels 
            if isinstance(label, GlitchType)
        ]
        
        if has_glitches_supersede or glitch_types_in_labels:
            if has_glitches_supersede:
                glitch_types_to_fetch = None
            else:
                glitch_types_to_fetch = glitch_types_in_labels
            
            glitch_segments = gf.get_glitch_segments(
                ifo,
                start_gps_time=self.start_gps_times[0],
                end_gps_time=self.end_gps_times[0],
                glitch_types=glitch_types_to_fetch
            )
            
            if len(glitch_segments) > 0:
                padded_glitches = glitch_segments.copy()
                padded_glitches[:, 0] -= start_padding_seconds
                padded_glitches[:, 1] += end_padding_seconds
                wanted_segments.append(padded_glitches)
                
                glitch_times, glitch_labels = gf.get_glitch_times_with_labels(
                    ifo,
                    start_gps_time=self.start_gps_times[0],
                    end_gps_time=self.end_gps_times[0],
                    glitch_types=glitch_types_to_fetch,
                    balanced=self.balanced_glitch_types
                )
                feature_times[gf.DataLabel.GLITCHES] = glitch_times
                self._feature_labels = {
                    gf.DataLabel.GLITCHES: (glitch_times, glitch_labels)
                }
                self._sorted_label_indices = None  # Invalidate cache
            
        if wanted_segments:
            wanted_segments = np.concatenate(wanted_segments)
            
            valid_segments = self._find_segment_intersections(
                valid_segments,
                wanted_segments
            )
        else:
            raise ValueError("Cannot find any features which suit requirement!")

        if not any(np.any(segment) for segment in valid_segments):
            raise ValueError(
                "Cannot find any features which overlap required times!"
            )
        
        return valid_segments, feature_times

    def _find_segment_intersections(self, arr1, arr2):
        """Find intersections between two sets of segments."""
        latest_starts = np.maximum(arr1[:, None, 0], arr2[None, :, 0])
        earliest_ends = np.minimum(arr1[:, None, 1], arr2[None, :, 1])

        overlap_durations = np.clip(earliest_ends - latest_starts, 0, None)
        overlap_mask = overlap_durations > 0
        best_overlap_indices = np.argmax(overlap_durations, axis=-1)

        starts = latest_starts[np.arange(latest_starts.shape[0]), best_overlap_indices]
        ends = earliest_ends[np.arange(earliest_ends.shape[0]), best_overlap_indices]

        valid_mask = overlap_mask[np.arange(overlap_mask.shape[0]), best_overlap_indices]
        starts = starts[valid_mask]
        ends = ends[valid_mask]

        return np.vstack((starts, ends)).T

    def _cluster_transients(
        self, 
        segments: np.ndarray, 
        request_overhead_seconds: float = 15.0,
        data_download_rate: float = 0.01,
        max_segment_seconds: float = 512.0  # Limit to prevent OOM
    ) -> np.ndarray:
        """
        Cluster nearby transients using a greedy cost-optimized algorithm.
        Limits max segment size to prevent OOM errors during download.
        """
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
            logging.info(
                f"Greedy clustering: {original_count} transients -> "
                f"{merged_count} download segments (gap threshold: {gap_threshold:.0f}s)"
            )
        
        return merged_array

    def precache_transients(
        self,
        ifos: List,
        sample_rate_hertz: float,
        onsource_duration_seconds: float,
        offsource_duration_seconds: float,
        cache_path: Path = None,
        data_directory: Path = None,
        seed: int = None,
        group_name: str = "train",
        force_rebuild: bool = False,
        cap: int = None,
    ) -> Path:
        """
        Pre-download and cache all transient data using clustered downloads.
        
        This method:
        1. Clusters nearby transients into larger download segments
        2. Downloads each clustered segment once
        3. Extracts individual glitch windows from each segment
        4. Saves all extracted glitches to a GlitchCache HDF5 file
        """
        from gravyflow.src.dataset.features.glitch_cache import GlitchCache

        if not isinstance(ifos, list):
            ifos = [ifos]
        if seed is None:
            seed = gf.Defaults.seed
            
        if cache_path is None:
            if data_directory is None:
                data_directory = Path("./generator_data")
                
            ifo_str = "_".join([i.name for i in ifos])
            
            run_identifier = str(int(self.start_gps_times[0])) if self.start_gps_times else "unknown"
            if self.start_gps_times:
                for run in ObservingRun:
                    if run.value.start_gps_time <= self.start_gps_times[0] <= run.value.end_gps_time:
                        run_identifier = run.name
                        break
            
            filename = f"glitch_cache_{run_identifier}_{ifo_str}.h5"
            cache_path = data_directory / filename
        
        
        self.get_valid_segments(ifos=ifos, seed=seed, group_name=group_name)
        
        if self.acquisition_mode != AcquisitionMode.TRANSIENT:
            raise ValueError("precache_transients requires TRANSIENT mode")
        
        # feature_segments has the original 64s windows around each transient
        # Extract the GPS times from these (center of each window)
        original_segments = self.feature_segments  # Shape (N, 2)
        original_gps_times = (original_segments[:, 0] + original_segments[:, 1]) / 2  # Center time
        
        # Apply cap if requested (limit number of glitches)
        if cap is not None and cap > 0:
            logging.info(f"Applying cap to glitches: {len(original_segments)} -> {cap}")
            original_segments = original_segments[:cap]
            original_gps_times = original_gps_times[:cap]
        
        # Cluster for efficient downloading
        clustered_segments = self._cluster_transients(
            original_segments,
            max_segment_seconds=512.0  # Pass the limit explicitly
        )
        
        # Build mapping efficiently using sorted arrays and binary search
        # This prevents O(N*M) complexity which hangs for large N (e.g. 250k glitches)
        sorted_indices = np.argsort(original_gps_times)
        sorted_gps = original_gps_times[sorted_indices]
        
        transient_to_cluster = {}  # cluster_idx -> list of (gps_time, original_segment)
        
        # Clustered segments are already sorted by time (from _cluster_transients logic)
        for i, (cl_start, cl_end) in enumerate(clustered_segments):
            # Find range of transients falling within this cluster
            # valid transients: cl_start <= gps_time <= cl_end
            idx_start = np.searchsorted(sorted_gps, cl_start, side='left')
            idx_end = np.searchsorted(sorted_gps, cl_end, side='right')
            
            if idx_start < idx_end:
                # Get indices in original unsorted arrays
                orig_indices = sorted_indices[idx_start:idx_end]
                
                # Extract transients for this cluster
                cluster_transients = []
                for orig_idx in orig_indices:
                    cluster_transients.append((original_gps_times[orig_idx], original_segments[orig_idx]))
                    
                transient_to_cluster[i] = cluster_transients
        
        # Calculate extraction parameters using MAX cache settings
        # (User-provided durations/rates are only applied at load time)
        from gravyflow.src.dataset.features.glitch_cache import (
            CACHE_SAMPLE_RATE_HERTZ, CACHE_ONSOURCE_DURATION, CACHE_OFFSOURCE_DURATION
        )
        
        num_onsource_samples = int(CACHE_ONSOURCE_DURATION * CACHE_SAMPLE_RATE_HERTZ)
        num_offsource_samples = int(CACHE_OFFSOURCE_DURATION * CACHE_SAMPLE_RATE_HERTZ)
        
        # Storage for extracted glitches
        all_onsource = []
        all_offsource = []
        all_gps_times = []
        all_labels = []
        
        logging.info(f"Precaching {len(original_gps_times)} glitches from {len(clustered_segments)} download segments...")
        logging.info(f"Storing at {CACHE_SAMPLE_RATE_HERTZ}Hz, {CACHE_ONSOURCE_DURATION}s onsource, {CACHE_OFFSOURCE_DURATION}s offsource")
        
        # Incremental Saving & Resume Logic
        start_cluster_idx = 0
        last_processed_gps = -1.0
        resume_mode = False
        
        cache = GlitchCache(cache_path, mode='a' if cache_path.exists() else 'w')
        
        if cache.exists and not force_rebuild:
            # Check if we can resume (cache is always at max settings)
            try:
                cache.validate_request(CACHE_SAMPLE_RATE_HERTZ, CACHE_ONSOURCE_DURATION, CACHE_OFFSOURCE_DURATION)
                
                # Get progress
                start_cluster_idx = cache.get_attr('processed_clusters', 0)
                last_processed_gps = cache.get_last_gps()
                
                if start_cluster_idx > 0 or last_processed_gps > 0:
                    logging.info(f"Resuming precache from cluster index {start_cluster_idx}, last GPS {last_processed_gps}")
                    resume_mode = True
            except (ValueError, KeyError, OSError) as e:
                logging.warning(f"Existing partial cache incompatible or corrupt: {e}. Rebuilding...")
                force_rebuild = True
        
        # Check if already complete
        if resume_mode and start_cluster_idx >= len(clustered_segments):
             logging.info(f"Cache already complete ({start_cluster_idx} segments).")
             return cache_path

        if not resume_mode or force_rebuild:
            # Initialize fresh file
            cache = GlitchCache(cache_path, mode='w') # Re-open in write mode to wipe
            cache.initialize_file(
                sample_rate_hertz=CACHE_SAMPLE_RATE_HERTZ,
                onsource_duration=CACHE_ONSOURCE_DURATION,
                offsource_duration=CACHE_OFFSOURCE_DURATION,
                ifo_names=[i.name for i in ifos],
                num_ifos=len(ifos),
                onsource_samples=num_onsource_samples,
                offsource_samples=num_offsource_samples
            )
            start_cluster_idx = 0
            last_processed_gps = -1.0

        # Buffers for incremental saving
        batch_onsource = []
        batch_offsource = []
        batch_gps = []
        batch_labels = []
        batch_size_threshold = 1000  # Save every 1000 glitches
        total_extracted = 0

        # Download each clustered segment and extract all glitches within
        try:
            from tqdm import tqdm
            segment_iterator = tqdm(
                enumerate(clustered_segments), 
                total=len(clustered_segments),
                desc="Downloading segments",
                initial=start_cluster_idx,
                unit="seg"
            )
        except ImportError:
            segment_iterator = enumerate(clustered_segments)
        
        for cluster_idx, (seg_start, seg_end) in segment_iterator:
            # RESUME SKIPPING: Skip already processed clusters
            if cluster_idx < start_cluster_idx:
                continue

            transients_in_segment = transient_to_cluster.get(cluster_idx, [])
            if not transients_in_segment:
                continue
            
            # Download this segment
            try:
                segment_key = f"segments/segment_{ifos[0].name}_{seg_start}_{seg_end}"
                segment_data = self.get_segment(
                    seg_start,
                    seg_end,
                    CACHE_SAMPLE_RATE_HERTZ,
                    ifos[0],  # Primary IFO
                    segment_key
                )
                if segment_data is None:
                    logging.warning(f"Failed to download segment {cluster_idx}")
                    continue
            except Exception as e:
                logging.warning(f"Error downloading segment {cluster_idx}: {e}")
                continue
            
            # Extract each transient from this segment
            for gps_time, orig_seg in transients_in_segment:
                # RESUME SKIPPING: Prevent Duplicates
                if gps_time <= last_processed_gps:
                    continue

                try:
                    # Calculate sample indices relative to segment start
                    time_offset = gps_time - seg_start
                    center_sample = int(time_offset * CACHE_SAMPLE_RATE_HERTZ)
                    
                    # Onsource extraction
                    half_onsource = num_onsource_samples // 2
                    on_start = center_sample - half_onsource
                    on_end = on_start + num_onsource_samples
                    
                    # Offsource extraction (before onsource)
                    off_end = on_start
                    off_start = off_end - num_offsource_samples
                    
                    # Validate bounds
                    data_len = len(segment_data)
                    if on_start < 0 or on_end > data_len or off_start < 0:
                        continue
                    
                    # Extract data
                    onsource = segment_data[on_start:on_end]
                    offsource = segment_data[off_start:off_end]
                    
                    # Reshape for multi-IFO: (IFOs, samples)
                    onsource = np.array(onsource).reshape(1, -1)  # (1, samples)
                    offsource = np.array(offsource).reshape(1, -1)
                    
                    batch_onsource.append(onsource)
                    batch_offsource.append(offsource)
                    batch_gps.append(gps_time)
                    
                    # Get label using vectorized lookup
                    label = self._lookup_labels([gps_time])[0] if hasattr(self, '_feature_labels') else 0
                    batch_labels.append(label)
                    
                except Exception as e:
                    logging.debug(f"Failed to extract transient at {gps_time}: {e}")
                    continue
            
            # Flush batch if threshold reached
            if len(batch_gps) >= batch_size_threshold:
                 # Stack and append
                cache.append(
                    onsource=np.stack(batch_onsource, axis=0),
                    offsource=np.stack(batch_offsource, axis=0),
                    gps_times=np.array(batch_gps),
                    labels=np.array(batch_labels)
                )
                # Checkpoint: Update processed clusters to current index (conservative)
                cache.set_attr('processed_clusters', cluster_idx)
                
                total_extracted += len(batch_gps)
                # Clear buffers
                batch_onsource = []
                batch_offsource = []
                batch_gps = []
                batch_labels = []
                
            if (cluster_idx + 1) % 50 == 0:
                logging.info(f"Processed {cluster_idx + 1}/{len(clustered_segments)} segments, extracted {total_extracted + len(batch_gps)} glitches")
        
        # Flush remaining items
        if len(batch_gps) > 0:
            cache.append(
                onsource=np.stack(batch_onsource, axis=0),
                offsource=np.stack(batch_offsource, axis=0),
                gps_times=np.array(batch_gps),
                labels=np.array(batch_labels)
            )
            total_extracted += len(batch_gps)
        
        # Final checkpoint: all done
        if total_extracted > 0 or resume_mode:
            cache.set_attr('processed_clusters', len(clustered_segments))
        
        # If purely verifying return, total_extracted might be 0 if all skipped.
        # But we return cache_path regardless.
        
        logging.info(f"Precached {total_extracted} new glitches to {cache_path}")
        return cache_path

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
        ) -> List:
        """
        Get valid segments for TRANSIENT mode acquisition.
        """
        if not isinstance(ifos, list) and not isinstance(ifos, tuple):
            ifos = [ifos]

        if not segment_order:
            segment_order = self.segment_order

        if not groups:
            groups = {
                "train": 0.98,
                "validate": 0.01,
                "test": 0.01
            }

        if group_name == "all":
            groups = {"all": 1.0}
        elif group_name not in groups:
            raise KeyError(
                f"Group {group_name} not present in groups dictionary."
            )

        if self.valid_segments is None or len(self.valid_segments) != len(ifos):
            all_feature_segments_list = []
            
            # Use pre-populated segments if available
            if self.feature_segments is not None and len(self.feature_segments) > 0:
                all_feature_segments_list.append(self.feature_segments)
            else: 
                # Add epsilon buffer (0.2s) to account for epsilon trimming in get_segment (0.1s each end)
                # This ensures segments remain large enough for proper standardization
                padding = 32.0 + 0.2
                
                global_start_gps = min(self.start_gps_times)
                global_end_gps = max(self.end_gps_times)

                # EVENTS (Global)
                if DataLabel.EVENTS in self.data_labels:
                    event_times = gf.get_event_times_by_type(self.event_types)
                    event_times = [t for t in event_times if global_start_gps <= t <= global_end_gps]
                    
                    if len(event_times) > 0:
                        evt_segs = self.pad_gps_times_with_veto_window(
                            np.array(event_times), 
                            start_padding_seconds=padding, 
                            end_padding_seconds=padding
                        )
                        all_feature_segments_list.append(evt_segs)

                # GLITCHES (Per IFO)
                # Check for DataLabel.GLITCHES OR specific GlitchType enums in data_labels
                from gravyflow.src.dataset.features.glitch import GlitchType
                glitch_types_in_labels = [
                    label for label in self.data_labels 
                    if isinstance(label, GlitchType)
                ]
                has_glitches = DataLabel.GLITCHES in self.data_labels or len(glitch_types_in_labels) > 0
                
                if has_glitches:
                    # Determine which types to fetch
                    if DataLabel.GLITCHES in self.data_labels:
                        glitch_types_to_fetch = None  # All types
                    else:
                        glitch_types_to_fetch = glitch_types_in_labels
                    
                    for ifo in ifos:
                        try:
                            glitch_times, glitch_labels = gf.get_glitch_times_with_labels(
                                ifo,
                                start_gps_time=global_start_gps,
                                end_gps_time=global_end_gps,
                                glitch_types=glitch_types_to_fetch,
                                balanced=self.balanced_glitch_types
                            )
                            
                            if not hasattr(self, '_feature_labels'):
                                self._feature_labels = {}
                                
                            # Store arrays directly for vectorized lookup
                            # This matches expectation in _lookup_labels
                            current_times, current_labels = self._feature_labels.get(gf.DataLabel.GLITCHES, (np.array([]), np.array([])))
                            
                            if len(current_times) > 0:
                                new_times = np.concatenate([current_times, glitch_times])
                                new_labels = np.concatenate([current_labels, glitch_labels])
                            else:
                                new_times = glitch_times
                                new_labels = glitch_labels
                                
                            self._feature_labels[gf.DataLabel.GLITCHES] = (new_times, new_labels)
                            self._sorted_label_indices = None  # Invalidate cache

                            if len(glitch_times) > 0:
                                gl_segs = self.pad_gps_times_with_veto_window(
                                    np.array(glitch_times),
                                    start_padding_seconds=padding,
                                    end_padding_seconds=padding
                                )
                                all_feature_segments_list.append(gl_segs)
                        except Exception as e:
                            logging.warning(f"Failed to fetch glitches for {ifo}: {e}")
                            continue

            if not all_feature_segments_list:
                raise ValueError("No features (Events/Glitches) found in requested range.")
            
            # Union and Unique
            combined = np.concatenate(all_feature_segments_list)
            
            # If balanced_glitch_types is True, we explicitly WANT duplicates (oversampling).
            # So only apply unique if NOT balancing glitches (or if glitches aren't the primary data).
            if self.balanced_glitch_types and DataLabel.GLITCHES in self.data_labels:
                 unique_segments = combined
            else:
                 unique_segments = np.unique(combined, axis=0)
            
            unique_segments = self.order_segments(unique_segments, segment_order, seed)
            
            self.feature_segments = unique_segments
            
            # Apply Grouping
            if group_name == 'all':
                target_segments = unique_segments
            else:
                n_segments = len(unique_segments)
                total_weight = sum(groups.values())
                acc_weight = 0.0
                start_idx = 0
                end_idx = 0
                found = False
                
                for g_name, weight in sorted(groups.items()):
                    w = weight / total_weight
                    if g_name == group_name:
                        start_idx = int(acc_weight * n_segments)
                        end_idx = int((acc_weight + w) * n_segments)
                        found = True
                        break
                    acc_weight += w
                    
                if found:
                    target_segments = unique_segments[start_idx:end_idx]
                else:
                    target_segments = np.empty((0, 2))
            
            if len(target_segments) == 0:
                logging.warning(f"No feature segments found for group {group_name}")
            
            # Expand for Multi-IFO: (N, IFOs, 2)
            num_ifos = len(ifos)
            expanded = np.expand_dims(target_segments, axis=1)
            self.valid_segments = np.repeat(expanded, num_ifos, axis=1)

            if self.feature_segments is not None:
                num_features = len(self.feature_segments)
                if num_features > 0:
                    logging.info(f"TRANSIENT MODE: {num_features} feature segments ready")
                
            # Randomize/Order segments
            self.order_segments(self.valid_segments, segment_order, seed)

            self._cache_valid_segments(self.valid_segments, group_name)

        return self.valid_segments

    def _yield_events_direct(
        self,
        sample_rate_hertz: float,
        onsource_duration_seconds: float,
        padding_duration_seconds: float,
        offsource_duration_seconds: float,
        num_examples_per_batch: int,
        ifos: List[gf.IFO],
        scale_factor: float,
        seed: int
    ):
        """
        Yield events directly via download (no caching).
        Used when specific event_names are requested.
        """
        rng = default_rng(seed)
        
        total_padding_duration_seconds = padding_duration_seconds * 2.0
        total_onsource_duration_seconds = onsource_duration_seconds + total_padding_duration_seconds
        
        num_onsource_samples = ensure_even(int(total_onsource_duration_seconds * sample_rate_hertz))
        num_offsource_samples = ensure_even(int(offsource_duration_seconds * sample_rate_hertz))
        
        # Use acquire to download segments with parallel prefetching
        segment_generator = self.acquire(
            sample_rate_hertz,
            self.valid_segments,
            ifos,
            scale_factor
        )
        
        batch_subarrays = []
        batch_backgrounds = []
        batch_gps_times = []
        
        for event_segment in segment_generator:
            # Extract onsource/offsource
            onsource, offsource, gps_time = self._extract_feature_event(
                event_segment,
                num_onsource_samples,
                num_offsource_samples,
                sample_rate_hertz
            )
            
            if onsource is None:
                continue
            
            batch_subarrays.append(onsource)
            batch_backgrounds.append(offsource)
            batch_gps_times.append(gps_time)
            
            if len(batch_subarrays) >= num_examples_per_batch:
                final_subarrays = ops.cast(ops.stack(batch_subarrays), "float32")
                final_background = ops.cast(ops.stack(batch_backgrounds), "float32")
                final_gps = ops.expand_dims(
                    ops.convert_to_tensor(batch_gps_times, dtype="float64"), 
                    axis=-1
                )
                batch_labels = self._lookup_labels(batch_gps_times)
                yield self._apply_augmentation(final_subarrays, is_transient=True), self._apply_augmentation(final_background, is_transient=True), final_gps, batch_labels
                
                batch_subarrays = []
                batch_backgrounds = []
                batch_gps_times = []

        if len(batch_subarrays) > 0:
            final_subarrays = ops.cast(ops.stack(batch_subarrays), "float32")
            final_background = ops.cast(ops.stack(batch_backgrounds), "float32")
            final_gps = ops.expand_dims(
                ops.convert_to_tensor(batch_gps_times, dtype="float64"), 
                axis=-1
            )
            batch_labels = self._lookup_labels(batch_gps_times)
            yield self._apply_augmentation(final_subarrays, is_transient=True), self._apply_augmentation(final_background, is_transient=True), final_gps, batch_labels

    def get_onsource_offsource_chunks(
            self,
            sample_rate_hertz: float,
            onsource_duration_seconds: float,
            padding_duration_seconds: float,
            offsource_duration_seconds: float,
            num_examples_per_batch: int = None,
            ifos: List[gf.IFO] = None,
            scale_factor: float = None,
            seed: int = None,
            sampling_mode: SamplingMode = SamplingMode.RANDOM
        ):
        """
        Generate onsource/offsource chunks for TRANSIENT mode.
        
        Extracts centered windows around each event/glitch.
        """
        if ifos is None:
            ifos = [gf.IFO.L1]
        if num_examples_per_batch is None:
            num_examples_per_batch = gf.Defaults.num_examples_per_batch
        if scale_factor is None:
            scale_factor = gf.Defaults.scale_factor
        if seed is None:
            seed = gf.Defaults.seed
        if self.rng is None:
            self.rng = default_rng(seed)
        if not isinstance(ifos, list) and not isinstance(ifos, tuple):
            ifos = [ifos]
        
        total_padding_duration_seconds = padding_duration_seconds * 2.0
        total_onsource_duration_seconds = \
            onsource_duration_seconds + total_padding_duration_seconds 
        
        num_onsource_samples = ensure_even(int(total_onsource_duration_seconds * sample_rate_hertz))
        num_offsource_samples = ensure_even(int(offsource_duration_seconds * sample_rate_hertz))

        self.valid_segments_adjusted = self.valid_segments
        has_specific_events = hasattr(self, 'event_names') and self.event_names is not None and len(self.event_names) > 0
        
        if has_specific_events:
            yield from self._yield_events_direct(
                sample_rate_hertz=sample_rate_hertz,
                onsource_duration_seconds=onsource_duration_seconds,
                padding_duration_seconds=padding_duration_seconds,
                offsource_duration_seconds=offsource_duration_seconds,
                num_examples_per_batch=num_examples_per_batch,
                ifos=ifos,
                scale_factor=scale_factor,
                seed=seed
            )
            return
        
        # Default: Use cache for all transient data (glitches, unnamed events, etc.)
        # Construct cache path
        data_dir = Path("./generator_data") # Default used in precache
        if hasattr(self, 'data_directory') and self.data_directory:
             data_dir = self.data_directory
             
        run_id = "unknown"
        if self.observing_runs:
             # Just take the first one or derive roughly
             # If multiple runs, precache might be split? 
             # Current precache logic uses specific run logic.
             # We try to match it.
             pass 
             
        # Actually, simpler: IFODataObtainer doesn't easily know the exact filename without logic duplication.
        # But we can try the standard generate_glitch_cache_path if we exported it?
        # Yes, I imported it.
        
        # Simplified for production: Check the path that precache WOULD have created.
        observing_run_name = "unknown"
        if self.observing_runs:
             if isinstance(self.observing_runs, list):
                 observing_run_name = self.observing_runs[0].name
             else:
                 observing_run_name = self.observing_runs.name
        
        cache_path = generate_glitch_cache_path(
             observing_run=observing_run_name,
             ifo="_".join([i.name for i in ifos]),
             data_directory=data_dir
        )
        
        # Reuse existing cache if path matches (preserves GPS index)
        if not hasattr(self, '_glitch_cache') or self._glitch_cache is None or self._glitch_cache.path != cache_path:
            self._glitch_cache = GlitchCache(cache_path, mode='a')  # 'a' mode allows appending new glitches
        
        cache = self._glitch_cache
        
        # Log cache status (one time only)
        if cache.exists and not hasattr(self, '_cache_logged'):
            try:
                cache.validate_request(sample_rate_hertz, onsource_duration_seconds, offsource_duration_seconds)
                meta = cache.get_metadata()
                logging.info(f"Unified Data Path: cache has {meta['num_glitches']} glitches at {cache_path}")
                self._cache_logged = True
            except ValueError as e:
                logging.warning(f"Cache found but incompatible: {e}")
        
        # UNIFIED PATH: Cache-first with lazy append
        # All glitches go through the cache - either from existing cache or downloaded and appended
        from gravyflow.src.dataset.features.glitch_cache import (
            CACHE_SAMPLE_RATE_HERTZ, CACHE_ONSOURCE_DURATION, CACHE_OFFSOURCE_DURATION
        )
        
        # Initialize cache file if it doesn't exist
        if not cache.exists:
            num_cache_onsource = int(CACHE_ONSOURCE_DURATION * CACHE_SAMPLE_RATE_HERTZ)
            num_cache_offsource = int(CACHE_OFFSOURCE_DURATION * CACHE_SAMPLE_RATE_HERTZ)
            cache.initialize_file(
                sample_rate_hertz=CACHE_SAMPLE_RATE_HERTZ,
                onsource_duration=CACHE_ONSOURCE_DURATION,
                offsource_duration=CACHE_OFFSOURCE_DURATION,
                ifo_names=[i.name for i in ifos],
                num_ifos=len(ifos),
                onsource_samples=num_cache_onsource,
                offsource_samples=num_cache_offsource
            )
            logging.info(f"Created new cache file: {cache_path}")
        else:
            # Cache exists - use disk-based access (HDF5 chunked storage is efficient)
            # The hybrid path below uses has_gps() and get_by_gps() which read from disk
            pass
        # Build batches: cache hits served instantly, misses downloaded in parallel
        rng = default_rng(seed)
        segment_indices = np.arange(len(self.valid_segments_adjusted))
        
        # Precompute GPS times for all segments
        all_gps_times = []
        for seg in self.valid_segments_adjusted:
            seg_start = seg[0][0] if len(seg.shape) > 1 else seg[0]
            seg_end = seg[0][1] if len(seg.shape) > 1 else seg[1]
            all_gps_times.append((seg_start + seg_end) / 2)
        all_gps_times = np.array(all_gps_times)
        
        # Shuffle once with fixed seed so order is deterministic across runs
        # This ensures cache hits for previously downloaded glitches
        rng.shuffle(segment_indices)
        
        while True:
            # DON'T shuffle here - keep deterministic order for cache hits
            batch_subarrays = []
            batch_backgrounds = []
            batch_gps_times = []
            
            # Track which indices need downloading
            miss_indices = []
            
            # Debug counters for cache tracking
            if not hasattr(self, '_cache_hit_count'):
                self._cache_hit_count = 0
                self._cache_miss_count = 0
                self._last_log_count = 0
            
            for seg_idx in segment_indices:
                event_gps_time = all_gps_times[seg_idx]
                
                # Unified cache lookup (memory -> disk -> miss)
                onsource, offsource, source = self._get_sample_from_cache(
                    cache,
                    event_gps_time,
                    sample_rate_hertz,
                    total_onsource_duration_seconds,
                    offsource_duration_seconds,
                    scale_factor
                )
                
                if source != 'miss':
                    # Cache hit (memory or disk)
                    batch_subarrays.append(onsource)
                    batch_backgrounds.append(offsource)
                    batch_gps_times.append(event_gps_time)
                    self._cache_hit_count += 1
                    
                    # Log periodically (every 1000 samples)
                    total_samples = self._cache_hit_count + self._cache_miss_count
                    if total_samples - self._last_log_count >= 1000:
                        hit_rate = self._cache_hit_count / total_samples * 100
                        logging.info(f"Cache stats: {self._cache_hit_count} hits, {self._cache_miss_count} misses ({hit_rate:.1f}% hit rate)")
                        self._last_log_count = total_samples
                    
                    if len(batch_subarrays) >= num_examples_per_batch:
                        yield self._prepare_batch(batch_subarrays, batch_backgrounds, batch_gps_times)
                        batch_subarrays = []
                        batch_backgrounds = []
                        batch_gps_times = []
                    continue
                
                # Cache miss - need to download this one
                miss_indices.append(seg_idx)
                self._cache_miss_count += 1
                
                # Download in batches for efficiency
                if len(miss_indices) >= num_examples_per_batch:
                    # Get segments to download
                    miss_segments = self.valid_segments_adjusted[miss_indices]
                    
                    # Download at the higher of requested or cache sample rate
                    # (we can downsample but not upsample)
                    download_sample_rate = max(sample_rate_hertz, CACHE_SAMPLE_RATE_HERTZ)
                    
                    # Use parallel acquire for bulk download
                    segment_generator = self.acquire(
                        download_sample_rate,
                        miss_segments,
                        ifos,
                        1.0 # Download RAW data for cache (scale=1.0)
                    )
                    
                    for event_segment in segment_generator:
                        feature_gps_start = event_segment.start_gps_time[0]
                        data_len = len(event_segment.data[0])
                        center_idx = data_len // 2
                        gps_time = feature_gps_start + (center_idx / download_sample_rate)
                        
                        # Extract and cache
                        num_cache_onsource = int(CACHE_ONSOURCE_DURATION * download_sample_rate)
                        num_cache_offsource = int(CACHE_OFFSOURCE_DURATION * download_sample_rate)
                        
                        onsource_full, offsource_full, extracted_gps = self._extract_feature_event(
                            event_segment,
                            num_cache_onsource,
                            num_cache_offsource,
                            download_sample_rate
                        )
                        
                        if onsource_full is None:
                            continue

                        # NAN CHECK: Post-Extraction
                        if np.isnan(onsource_full).any():
                             logging.error(f"NAN DETECTED: In onsource_full after extract! GPS: {extracted_gps}")
                        
                        # ZERO CHECK
                        if np.all(onsource_full == 0):
                             logging.error(f"ZERO SIGNAL DETECTED: onsource_full is all zeros! GPS: {extracted_gps}")

                        # Save to cache using segment-window GPS (matches lookup calculation)
                        label = self._feature_labels.get(gps_time, 0) if hasattr(self, '_feature_labels') and isinstance(self._feature_labels, dict) else 0
                        try:
                            cache.append_single(np.array(onsource_full), np.array(offsource_full), gps_time, label)
                        except Exception:
                            pass
                        
                        # Crop for this request
                        onsource_cropped, offsource_cropped = self._crop_resample(
                            onsource_full, offsource_full,
                            download_sample_rate, sample_rate_hertz,
                            CACHE_ONSOURCE_DURATION, total_onsource_duration_seconds,
                            CACHE_OFFSOURCE_DURATION, offsource_duration_seconds
                        )

                        # NAN CHECK: Post-Crop
                        if np.isnan(onsource_cropped).any():
                             logging.error(f"NAN DETECTED: In onsource_cropped! GPS: {extracted_gps}")

                        batch_subarrays.append(onsource_cropped * scale_factor)
                        batch_backgrounds.append(offsource_cropped * scale_factor)
                        batch_gps_times.append(extracted_gps)
                        
                        if len(batch_subarrays) >= num_examples_per_batch:
                            yield self._prepare_batch(batch_subarrays, batch_backgrounds, batch_gps_times)
                            batch_subarrays = []
                            batch_backgrounds = []
                            batch_gps_times = []
                    
                    miss_indices = []

    def _crop_resample(
        self,
        onsource_full: np.ndarray,
        offsource_full: np.ndarray,
        source_rate: float,
        target_rate: float,
        source_ons_dur: float,
        target_ons_dur: float,
        source_off_dur: float,
        target_off_dur: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Crop and resample data from source settings to target settings."""
        onsource_full = np.asarray(onsource_full)
        offsource_full = np.asarray(offsource_full)
        
        # Resample if needed (downsampling only - upsampling not supported)
        if target_rate != source_rate:
            if target_rate > source_rate:
                raise ValueError(
                    f"Cannot upsample from {source_rate}Hz to {target_rate}Hz. "
                    "Upsampling is not supported."
                )
            ratio = int(source_rate / target_rate)
            if ratio > 1:
                onsource_full = onsource_full[..., ::ratio]
                offsource_full = offsource_full[..., ::ratio]
        
        # Crop onsource (center crop)
        if target_ons_dur < source_ons_dur:
            target_samples = int(target_ons_dur * target_rate)
            current_samples = onsource_full.shape[-1]
            start = (current_samples - target_samples) // 2
            onsource_full = onsource_full[..., start:start + target_samples]
        
        # Crop offsource (center crop)
        if target_off_dur < source_off_dur:
            target_samples = int(target_off_dur * target_rate)
            current_samples = offsource_full.shape[-1]
            start = (current_samples - target_samples) // 2
            offsource_full = offsource_full[..., start:start + target_samples]
        
        return onsource_full, offsource_full
