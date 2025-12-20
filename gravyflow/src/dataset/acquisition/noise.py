"""
Noise mode data acquisition.

This module provides NoiseDataObtainer for acquiring random or grid-sampled
noise data from long IFO segments (NOISE mode).
"""

import logging
import hashlib
from typing import List, Dict, Optional, Union

import numpy as np
from numpy.random import default_rng
from keras import ops

import gravyflow as gf
from .base import (
    BaseDataObtainer, DataQuality, DataLabel, SegmentOrder, 
    AcquisitionMode, SamplingMode, ObservingRun, IFOData, ensure_even
)
from gravyflow.src.dataset.features.injection import ReturnVariables as RV
from gravyflow.src.utils.shapes import ShapeEnforcer


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
            random_sign_reversal: bool = True,
            random_time_reversal: bool = True,
            augmentation_probability: float = 0.5,
            prefetch_segments: int = 16,
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
            random_sign_reversal=random_sign_reversal,
            random_time_reversal=random_time_reversal,
            augmentation_probability=augmentation_probability,
            prefetch_segments=prefetch_segments,
        )
        
        self.acquisition_mode = AcquisitionMode.NOISE
        self._segment_cache_maxsize = 8  # NOISE mode uses larger segments
        self._segment_generator = None  # Reusable generator (preserves prefetch)

    # =========================================================================
    # NOISE-SPECIFIC SEGMENT PROCESSING
    # =========================================================================

    def remove_unwanted_segments(
            self,
            ifo: gf.IFO,
            valid_segments: np.ndarray,
            get_times: List = None
        ):
        """
        Remove segments containing unwanted events/glitches from noise data.
        """
        if get_times is None:
            get_times = []

        veto_segments = []

        if DataLabel.EVENTS in get_times or DataLabel.EVENTS not in self.data_labels:
            event_times = self.get_all_event_times()
        else:
            event_times = []

        if DataLabel.GLITCHES in get_times or DataLabel.GLITCHES not in self.data_labels:
            glitch_times = gf.get_glitch_times(
                ifo,
                start_gps_time=self.start_gps_times[0],
                end_gps_time=self.end_gps_times[0]
            )
        else:
            glitch_times = []

        if DataLabel.EVENTS not in self.data_labels:
            veto_segments.append(
                self.pad_gps_times_with_veto_window(event_times)
            )

        if DataLabel.GLITCHES not in self.data_labels:
            veto_segments.append(
                gf.get_glitch_segments(
                    ifo,
                    start_gps_time=self.start_gps_times[0],
                    end_gps_time=self.end_gps_times[0]
                )
            )

        if veto_segments:
            veto_segments = np.concatenate(veto_segments)
            valid_segments = \
                self.veto_time_segments(valid_segments, veto_segments)

        feature_times = {
            gf.DataLabel.EVENTS: event_times,
            gf.DataLabel.GLITCHES: glitch_times    
        }

        return valid_segments, feature_times

    def find_segment_intersections(self, arr1, arr2):
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

    def remove_short_segments(
            self,
            segments: np.ndarray, 
            minimum_duration_seconds: float
        ) -> np.ndarray:
        """
        Removes segments where duration is less than minimum.
        
        Parameters:
            segments: Input array of shape [N, X, 2].
            minimum_duration_seconds: Minimum allowed duration.
            
        Returns:
            Filtered array with short segments removed.
        """
        durations = segments[:, :, 1] - segments[:, :, 0] 
        valid_columns = np.all(durations >= minimum_duration_seconds, axis=1) 
        filtered_segments = segments[valid_columns, :, :] 
        return filtered_segments

    def calculate_bin_indices(self, segments, interval, start_period):
        """Calculate bin indices for segments based on time interval."""
        relative_starts = (segments[:, 0] - start_period) / interval
        relative_ends = (segments[:, 1] - start_period) / interval

        start_bins = np.floor(relative_starts).astype(int)
        end_bins = np.ceil(relative_ends).astype(int)

        return start_bins, end_bins

    def merge_bins(self, new_segments_list, interval):
        """
        Merges split segments into bins of shape [G, N].
        """
        min_time = min(segment[0, 0] for segment in new_segments_list)
        max_time = max(segment[-1, 1] for segment in new_segments_list)

        bins = np.arange(
            np.floor(min_time / interval) * interval, 
            np.ceil(max_time / interval) * interval, interval
        )

        result = [[] for _ in bins]

        for i, bin_start in enumerate(bins):
            for segments in new_segments_list:
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
    
    def largest_segments_per_bin(self, filtered_bins_result):
        """
        Extracts the largest segment in each bin from each list.
        """
        largest_segments_list = [
            [] for _ in range(len(filtered_bins_result[0]))
        ]

        for bin in filtered_bins_result:
            for j, segments in enumerate(bin):
                durations = segments[:, 1] - segments[:, 0]  
                largest_segment_index = np.argmax(durations)  
                largest_segments_list[j].append(segments[largest_segment_index])  
        
        result_arrays = [
            np.array(segments) for segments in largest_segments_list
        ]

        return np.array(result_arrays)

    def get_segments_for_group(
        self,
        segments: np.ndarray,
        chunk_size: float,
        group_name: str,
        groups: Dict[str, float],
        start_time: float
    ) -> np.ndarray:
        """
        Deterministically assign segments to groups based on hash.
        """
        selected_segments = []
        
        group_names = list(groups.keys())
        probs = list(groups.values())
        total = sum(probs)
        probs = [p/total for p in probs]
        
        for start, end in segments:
            idx = int(np.floor((start - start_time) / chunk_size))
            
            h = hashlib.md5(str(idx).encode()).hexdigest()
            val = int(h, 16) / (2**128)
            
            cumulative = 0.0
            assigned_group = group_names[-1]
            for name, prob in zip(group_names, probs):
                cumulative += prob
                if val <= cumulative:
                    assigned_group = name
                    break
            
            if assigned_group == group_name:
                selected_segments.append([start, end])
                
        return np.array(selected_segments)

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
        Get valid segments for NOISE mode acquisition.
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
            self.valid_segments = []

            for ifo in ifos:
                valid_segments = self.get_all_segment_times(ifo)

                group_split_seconds: float = 8196.0

                valid_segments = self.cut_segments(
                    valid_segments, 
                    group_split_seconds,
                    self.start_gps_times[0]
                )

                valid_segments = self.get_segments_for_group(
                    valid_segments, 
                    group_split_seconds, 
                    group_name, 
                    groups,
                    self.start_gps_times[0]
                )

                valid_segments, feature_times = self.remove_unwanted_segments(
                    ifo,
                    valid_segments
                )

                valid_segments = self.cut_segments(
                    valid_segments, 
                    self.max_segment_duration_seconds,
                    self.start_gps_times[0]
                )

                if len(valid_segments) == 0:
                    raise ValueError(f"IFO {ifo} has found no valid segments (NOISE mode)!")

                self.valid_segments.append(valid_segments)

            # Merge and process segments
            self.valid_segments = self.merge_bins(
                self.valid_segments, 
                self.max_segment_duration_seconds
            )
                    
            self.valid_segments = self.largest_segments_per_bin(
                self.valid_segments
            )
                    
            self.valid_segments = np.swapaxes(self.valid_segments, 1, 0)
            
            self.valid_segments = self.order_segments(
                self.valid_segments, 
                segment_order,
                seed
            )
            
            self._cache_valid_segments(self.valid_segments, group_name)

        return self.valid_segments

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
        Wrapper to enforce shape contracts on generator.
        """
        gen = self._yield_onsource_offsource_chunks(
            sample_rate_hertz,
            onsource_duration_seconds,
            padding_duration_seconds,
            offsource_duration_seconds,
            num_examples_per_batch,
            ifos,
            scale_factor,
            seed,
            sampling_mode
        )

        # Ensure ifos is a list to count num_ifos
        if ifos is None:
            ifos = [gf.IFO.L1]
        elif not isinstance(ifos, (list, tuple)):
            ifos = [ifos]

        return ShapeEnforcer.wrap_generator(gen, num_ifos=len(ifos))

    def _yield_onsource_offsource_chunks(
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
        Generate onsource/offsource chunks for NOISE mode.
        
        Uses random or grid sampling from long data segments.
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

        # Early validation
        if self.valid_segments is None or len(self.valid_segments) == 0:
            logging.warning("No valid segments available in NOISE mode. Generator will be empty.")
            return
        
        if not hasattr(self.valid_segments, 'ndim') or self.valid_segments.ndim != 3:
            logging.warning(f"valid_segments has unexpected shape. Expected 3D array. Generator will be empty.")
            return
        
        if not self._current_batch_index and not self._current_segment_index:
            min_segment_duration_seconds = \
                total_onsource_duration_seconds + offsource_duration_seconds
            min_segment_duration_seconds *= 2.0
            
            self.valid_segments_adjusted = self.remove_short_segments(
                self.valid_segments, 
                min_segment_duration_seconds
            )
        
        if self.valid_segments_adjusted is None or len(self.valid_segments_adjusted) == 0:
            logging.warning("No valid segments available in NOISE mode. Generator will be empty.")
            return

        _consecutive_stop_iterations = 0
        
        while True:
            while self._segment_exhausted:
                # Create generator ONCE and reuse it (preserves prefetch across batches)
                if self._segment_generator is None:
                    self._segment_generator = self.acquire(
                        sample_rate_hertz,
                        self.valid_segments_adjusted,
                        ifos,
                        scale_factor
                    )
                
                try:
                    self.current_segment = next(self._segment_generator)
                    _consecutive_stop_iterations = 0
                except StopIteration:
                    _consecutive_stop_iterations += 1
                    if _consecutive_stop_iterations > 1:
                        logging.warning("No valid segments available. Generator exhausted.")
                        return
                    # Reset generator for next epoch
                    self._segment_generator = None
                    self._current_segment_index = 0
                    self._segment_exhausted = True
                    continue
            
                min_num_samples = min([ops.shape(tensor)[0] for tensor in self.current_segment.data])

                if min_num_samples < (num_onsource_samples + num_offsource_samples):
                    logging.warning("Segment too short!")
                    self._segment_exhausted = True
                else: 
                    self._segment_exhausted = False

                min_num_samples = ops.cast(min_num_samples, "float32")

                segment_duration = min_num_samples / sample_rate_hertz

                self._num_batches_in_current_segment = max(1, int(
                    segment_duration * self._get_effective_saturation()
                    / (num_examples_per_batch * onsource_duration_seconds)
                ))
            
            while self._current_batch_index < self._num_batches_in_current_segment and not self._segment_exhausted:

                if sampling_mode == SamplingMode.RANDOM:
                    subarrays, background_chunks, start_gps_times = self.current_segment.random_subsection(
                        num_onsource_samples, 
                        num_offsource_samples, 
                        num_examples_per_batch,
                        self.rng.integers(1E10)
                    )
                else:
                    # GRID mode
                    result = self.current_segment.grid_subsection(
                        num_onsource_samples, 
                        num_offsource_samples, 
                        num_examples_per_batch,
                        self._grid_position
                    )
                    subarrays, background_chunks, start_gps_times, new_grid_pos = result
                    if subarrays is not None:
                        self._grid_position = new_grid_pos

                if subarrays is None or background_chunks is None or start_gps_times is None:
                    if sampling_mode == SamplingMode.GRID:
                        self._segment_exhausted = True
                        self._grid_position = 0
                    continue
                
                self._current_batch_index += 1
                if not self._current_batch_index < self._num_batches_in_current_segment:
                    self._segment_exhausted = True
                    self._current_batch_index = 0
                    if sampling_mode == SamplingMode.GRID:
                        self._grid_position = 0

                # Yield dict for real noise
                # Shapes: ONSOURCE/OFFSOURCE = (Batch, IFO, Samples) = BIS
                #         START_GPS_TIME = (Batch, IFO) = BI
                #         DATA_LABEL = (Batch, IFO) = BI (all NOISE = 0)
                num_ifos = subarrays.shape[1]
                batch_size = subarrays.shape[0]
                yield {
                    RV.ONSOURCE: self._apply_augmentation(subarrays),
                    RV.OFFSOURCE: self._apply_augmentation(background_chunks),
                    RV.START_GPS_TIME: start_gps_times,  # Already (B, I)
                    RV.DATA_LABEL: ops.full((batch_size, num_ifos), 0, dtype="int32"),  # NOISE = 0
                }
