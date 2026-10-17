"""
Factory function for backwards-compatible IFODataObtainer instantiation.

This module provides the IFODataObtainer factory function that automatically
selects the appropriate obtainer class (NoiseDataObtainer or TransientDataObtainer)
based on the data_labels parameter.
"""

from typing import Optional, Tuple, Union, List
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
        augmentations: List = None,  # Training only; None is none
        prefetch_segments: int = 16,
        # Class balancing
        balanced_glitch_types: bool = False,
        # Specific Names
        event_names: List[str] = None,
        # Real-noise segment selection
        gwosc_flags: Optional[Tuple[str, ...]] = None,
        # The split
        split_spec: Optional["gf.SplitSpec"] = None,
        train_on_real_events: bool = False,
        # Where strain comes from
        source: Optional["gf.DataSource"] = None,
    ):
    """
    Factory function that returns the appropriate IFO data obtainer.
    
    Automatically selects between NoiseDataObtainer and TransientDataObtainer
    based on the data_labels parameter.
    
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
        augmentations: Augmentations a dataset applies to its training split only, per
            example (SignReversal, TimeReversal, RandomShift, AddNoise); none by default
        prefetch_segments: Number of segments to prefetch
        balanced_glitch_types: Enable balanced sampling of glitch types
        event_names: Specific event names to acquire (TRANSIENT only)
        gwosc_flags: GWOSC timeline flags whose intersection defines usable noise
            times (NOISE only). None uses DATA with BURST_CAT1 and BURST_CAT2
            applied; gf.GWOSC_DATA_FLAGS and gf.GWOSC_CBC_FLAGS are alternatives.
        split_spec: How GPS time is split into train, validate and test
            (``gf.SplitSpec``); None is the default split.
        train_on_real_events: Real events are evaluation data, all in ``test``;
            True is refused for now (TRANSIENT only; see TransientDataObtainer).
        source: Where real strain and its segments come from (``gf.DataSource``):
            ``gf.GWOSC()``, or ``gf.LocalFrames(...)`` by name. None takes
            ``gf.Defaults.source`` (``gf.GWOSC()``). Nothing falls back from it to
            another source.

    Returns:
        NoiseDataObtainer or TransientDataObtainer instance
    """
    # Normalize data_labels to list
    if not isinstance(data_labels, list):
        data_labels = [data_labels]
    
    # Check if NOISE mode is requested
    is_noise = any(
        isinstance(label, DataLabel) and label == DataLabel.NOISE
        for label in data_labels
    )
                 
    if is_noise:
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
            augmentations=augmentations,
            prefetch_segments=prefetch_segments,
            gwosc_flags=gwosc_flags,
            split_spec=split_spec,
            source=source,
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
            augmentations=augmentations,
            prefetch_segments=prefetch_segments,
            balanced_glitch_types=balanced_glitch_types,
            event_names=event_names,
            split_spec=split_spec,
            train_on_real_events=train_on_real_events,
            source=source,
        )
