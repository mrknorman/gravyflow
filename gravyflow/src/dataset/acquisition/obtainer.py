"""
Factory function for backwards-compatible IFODataObtainer instantiation.

This module provides the IFODataObtainer factory function that automatically
selects the appropriate obtainer class (NoiseDataObtainer or TransientDataObtainer)
based on the data_labels parameter.
"""

from typing import Union, List
import logging

from .base import DataLabel, DataQuality, SegmentOrder, ObservingRun, SamplingMode
from .noise import NoiseDataObtainer
from .transient import TransientDataObtainer
from gravyflow.src.dataset.features.event import EventConfidence


def IFODataObtainer(
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
        prefetch_segments: int = 16,
        # TRANSIENT mode augmentations
        random_shift: bool = False,
        shift_fraction: float = 0.25,
        add_noise: bool = False,
        noise_amplitude: float = 0.1,
        # Class balancing
        balanced_glitch_types: bool = False,
    ):
    """
    Factory function that returns the appropriate IFO data obtainer.
    
    Automatically selects between NoiseDataObtainer and TransientDataObtainer
    based on the data_labels parameter:
    
    - If DataLabel.NOISE is in data_labels: Returns NoiseDataObtainer
    - Otherwise: Returns TransientDataObtainer (for EVENTS/GLITCHES)
    
    This function maintains full backwards compatibility with existing code
    that uses gf.IFODataObtainer(...).
    
    Args:
        data_quality: Quality level of data to acquire (RAW or BEST)
        data_labels: What type of data to acquire (NOISE, EVENTS, GLITCHES)
        observing_runs: Which observing runs to use (O1, O2, O3, O4)
        segment_order: How to order segments (RANDOM, CHRONOLOGICAL, SHORTEST_FIRST)
        max_segment_duration_seconds: Maximum segment length
        saturation: Oversampling factor for random sampling
        force_acquisition: Force fresh data acquisition even if cached
        cache_segments: Whether to cache segments to disk
        overrides: Dictionary of attribute overrides
        event_types: Filter events by type (for TRANSIENT mode)
        logging_level: Logging verbosity
        random_sign_reversal: Enable sign reversal augmentation
        random_time_reversal: Enable time reversal augmentation
        augmentation_probability: Probability of applying augmentations
        prefetch_segments: Number of segments to prefetch
        random_shift: Enable random shift augmentation (TRANSIENT only)
        shift_fraction: Maximum shift as fraction of onsource (TRANSIENT only)
        add_noise: Enable noise perturbation augmentation (TRANSIENT only)
        noise_amplitude: Noise amplitude relative to std (TRANSIENT only)
        balanced_glitch_types: Enable balanced sampling of glitch types
        
    Returns:
        NoiseDataObtainer or TransientDataObtainer instance
    """
    # Normalize data_labels to list
    if not isinstance(data_labels, list):
        data_labels = [data_labels]
    
    # Check if NOISE mode is requested
    if DataLabel.NOISE in data_labels:
        return NoiseDataObtainer(
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
    else:
        # TRANSIENT mode for events/glitches
        return TransientDataObtainer(
            data_quality=data_quality,
            data_labels=data_labels,
            observing_runs=observing_runs,
            segment_order=segment_order,
            max_segment_duration_seconds=max_segment_duration_seconds,
            saturation=saturation,
            force_acquisition=force_acquisition,
            cache_segments=cache_segments,
            overrides=overrides,
            event_types=event_types,
            logging_level=logging_level,
            random_sign_reversal=random_sign_reversal,
            random_time_reversal=random_time_reversal,
            augmentation_probability=augmentation_probability,
            prefetch_segments=prefetch_segments,
            random_shift=random_shift,
            shift_fraction=shift_fraction,
            add_noise=add_noise,
            noise_amplitude=noise_amplitude,
            balanced_glitch_types=balanced_glitch_types,
        )
