"""
Builder functions for creating TransientSegments directly from API data.

These replace build_glitch_records/build_event_records and return
TransientSegments with acquisition boundaries already set.
"""
from typing import List
from gravyflow.src.dataset.acquisition.transient_segment import TransientSegment
from gravyflow.src.dataset.acquisition.base import DataLabel, ObservingRun
from gravyflow.src.dataset.features.glitch import GlitchType, get_glitch_times_with_labels, get_glitch_type_from_index
from gravyflow.src.dataset.features.event import EventConfidence, SourceType, get_events_with_params
from gravyflow.src.utils.gps import gps_to_key
import gravyflow as gf


# Default padding: must be >= onsource_half + offsource_max + epsilon
# For 32s onsource (±16s) + 32s offsource before = 48s needed before event
# Adding epsilon for numerical safety
DEFAULT_PADDING = 48.0 + 0.2


def build_glitch_segments(
    ifos: List["gf.IFO"],
    observing_runs: List[ObservingRun] = None,
    glitch_types: List[GlitchType] = None,
    padding: float = DEFAULT_PADDING
) -> List[TransientSegment]:
    """
    Build TransientSegments for glitches with acquisition boundaries.
    
    Args:
        ifos: List of IFOs to fetch glitches for.
        observing_runs: Filter to specific runs. None = [O3].
        glitch_types: Filter to specific types. None = all types.
        padding: Padding on each side of glitch GPS time (seconds).
    
    Returns:
        List of TransientSegments for glitches.
    """
    if observing_runs is None:
        observing_runs = [ObservingRun.O3]
    
    segments = []
    
    for run in observing_runs:
        for ifo in ifos:
            # Fetch glitch times and labels for this IFO/run
            gps_times, labels = get_glitch_times_with_labels(
                ifo=ifo,
                observing_run=run,
                glitch_types=glitch_types
            )
            
            for gps_time, label_idx in zip(gps_times, labels):
                kind = get_glitch_type_from_index(label_idx)
                if kind is None:
                    continue  # Skip invalid labels
                
                segment = TransientSegment(
                    gps_key=gps_to_key(gps_time),
                    transient_gps_time=gps_time,
                    start_gps_time=gps_time - padding,
                    end_gps_time=gps_time + padding,
                    label=DataLabel.GLITCHES,
                    kind=kind,
                    observing_run=run,
                    seen_in=[ifo],  # Glitches are single-detector
                    confidence=None,
                    name=None,
                    weight=1.0
                )
                segments.append(segment)
    
    return segments


def build_event_segments(
    observing_runs: List[ObservingRun] = None,
    confidences: List[EventConfidence] = None,
    padding: float = DEFAULT_PADDING,
    event_names: List[str] = None
) -> List[TransientSegment]:
    """
    Build TransientSegments for GW events with acquisition boundaries.
    
    Args:
        observing_runs: Filter to specific runs. None = all runs.
        confidences: Confidence levels. None = [CONFIDENT].
        padding: Padding on each side of event GPS time (seconds).
        event_names: Filter to specific event names.
    
    Returns:
        List of TransientSegments for events.
    """
    if confidences is None:
        confidences = [EventConfidence.CONFIDENT]
    
    segments = []
    
    # Fetch events
    events = get_events_with_params(
        observing_runs=observing_runs,
        event_types=confidences,
        event_names=event_names
    )
    
    # Warn if requested events were not found
    if event_names:
        found_names = {event['name'] for event in events}
        missing = set(event_names) - found_names
        if missing:
            import logging
            logger = logging.getLogger(__name__)
            runs_str = str([r.name for r in observing_runs]) if observing_runs else "ALL"
            logger.warning(
                f"Requested events not found in {runs_str}: {missing}. "
                "Check observing_runs filter."
            )
    
    def determine_active_ifos(event: dict) -> list:
        """Determine which IFOs were active for this event from metadata."""
        import gravyflow as gf
        
        # Try to parse from 'network' field (e.g., "H1L1" or "H1L1V1")
        network = event.get('network', '')
        if network:
            ifos = []
            if 'H1' in network:
                ifos.append(gf.IFO.H1)
            if 'L1' in network:
                ifos.append(gf.IFO.L1)
            if 'V1' in network:
                ifos.append(gf.IFO.V1)
            if ifos:
                return ifos
        
        # Fallback: check FAR columns for detector presence
        # (Events with FAR values for a detector were observed there)
        ifos = []
        if event.get('far_gstlal') is not None or event.get('far_pycbc') is not None:
            # Primary LIGO detectors were active
            ifos = [gf.IFO.H1, gf.IFO.L1]
        
        return ifos if ifos else [gf.IFO.H1, gf.IFO.L1]  # Conservative fallback
    
    for event in events:
        # Determine source type from probabilities
        p_bbh = event.get('p_bbh') or 0
        p_bns = event.get('p_bns') or 0
        p_nsbh = event.get('p_nsbh') or 0
        
        if p_bbh >= max(p_bns, p_nsbh):
            source_type = SourceType.BBH
        elif p_bns >= p_nsbh:
            source_type = SourceType.BNS
        else:
            source_type = SourceType.NSBH
        
        # Map run string to enum
        run_str = event.get('observing_run', 'O3')
        run_map = {'O1': ObservingRun.O1, 'O2': ObservingRun.O2, 'O3': ObservingRun.O3, 'O4': ObservingRun.O4}
        obs_run = run_map.get(run_str, ObservingRun.O3)
        
        # Determine confidence from catalog
        catalog = event.get('catalog', '')
        if 'marginal' in catalog.lower():
            confidence = EventConfidence.MARGINAL
        else:
            confidence = EventConfidence.CONFIDENT
        
        gps_time = event['gps']
        segment = TransientSegment(
            gps_key=gps_to_key(gps_time),
            transient_gps_time=gps_time,
            start_gps_time=gps_time - padding,
            end_gps_time=gps_time + padding,
            label=DataLabel.EVENTS,
            kind=source_type,
            observing_run=obs_run,
            seen_in=determine_active_ifos(event),  # Multi-detector events
            confidence=confidence,
            name=event.get('name'),
            weight=1.0
        )
        segments.append(segment)
    
    return segments
