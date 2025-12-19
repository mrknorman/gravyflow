"""
Compressed segment storage using numpy structured arrays.

Reduces memory footprint from 120 bytes/segment to ~40 bytes/segment
by storing minimal data and inflating TransientSegments on-demand.
"""
import numpy as np
from typing import Optional
from dataclasses import dataclass

from gravyflow.src.dataset.acquisition.transient_segment import TransientSegment
from gravyflow.src.dataset.acquisition.base import DataLabel, ObservingRun
from gravyflow.src.dataset.features.glitch import GlitchType
from gravyflow.src.dataset.features.event import EventConfidence, SourceType
from gravyflow.src.utils.gps import gps_to_key, key_to_gps
import gravyflow as gf


# Structured dtype for compressed storage (40 bytes vs 120 bytes for full TransientSegment)
COMPRESSED_SEGMENT_DTYPE = np.dtype([
    ('gps_key', np.int64),           # 8 bytes - Primary key
    ('label', np.uint8),             # 1 byte - DataLabel.value
    ('kind', np.uint16),             # 2 bytes - GlitchType or SourceType value
    ('run', np.uint8),               # 1 byte - ObservingRun index
    ('ifo_mask', np.uint8),          # 1 byte - IFO bitmask (bit 0=H1, bit 1=L1, bit 2=V1)
    ('confidence', np.int8),         # 1 byte - EventConfidence.value (-1 if None)
    ('weight', np.float16),          # 2 bytes - Reduced precision
    ('padding', np.float16),         # 2 bytes - Padding duration
])


def segment_to_compressed(segment: TransientSegment) -> np.ndarray:
    """
    Convert a TransientSegment to compressed numpy record.
    
    Args:
        segment: Full TransientSegment object
        
    Returns:
        Single-element structured array
    """
    # Calculate padding from boundaries
    padding = (segment.end_gps_time - segment.transient_gps_time)
    
    # Encode observing run as index
    run_idx = segment.observing_run.index
    
    # Encode seen_in as bitmask
    from gravyflow.src.dataset.acquisition.transient_segment import ifos_to_bitmask
    ifo_mask = ifos_to_bitmask(segment.seen_in)
    
    # Encode confidence (-1 if None)
    if segment.confidence is not None:
        conf_idx = segment.confidence.value  # IntEnum!
    else:
        conf_idx = -1
    
    record = np.array([(
        segment.gps_key,
        segment.label.value,  # IntEnum!
        segment.kind.value,   # IntEnum!
        run_idx,
        ifo_mask,
        conf_idx,
        segment.weight,
        padding
    )], dtype=COMPRESSED_SEGMENT_DTYPE)
    
    return record[0]


def compressed_to_segment(record: np.ndarray, name: Optional[str] = None) -> TransientSegment:
    """
    Inflate a compressed record to full TransientSegment.
    
    Args:
        record: Single structured array record
        name: Optional name (not stored in compressed format)
        
    Returns:
        Full TransientSegment object
    """
    # Decode GPS time from key
    gps_time = key_to_gps(int(record['gps_key']))
    
    # Decode label
    label = DataLabel(record['label'])
    
    # Decode kind (GlitchType or SourceType depending on label)
    if label == DataLabel.GLITCHES:
        kind = GlitchType(record['kind'])
    elif label == DataLabel.EVENTS:
        kind = SourceType(record['kind'])
    else:
        kind = None
    
    # Decode observing run
    observing_run = list(ObservingRun)[record['run']]
    
    # Decode seen_in from bitmask
    from gravyflow.src.dataset.acquisition.transient_segment import bitmask_to_ifos
    seen_in = bitmask_to_ifos(int(record['ifo_mask']))
    
    # Decode confidence (-1 means None)
    if record['confidence'] >= 0:
        confidence = EventConfidence(record['confidence'])
    else:
        confidence = None
    
    # Reconstruct boundaries from padding
    padding = float(record['padding'])
    start_gps = gps_time - padding
    end_gps = gps_time + padding
    
    return TransientSegment(
        gps_key=int(record['gps_key']),
        transient_gps_time=gps_time,
        start_gps_time=start_gps,
        end_gps_time=end_gps,
        label=label,
        kind=kind,
        observing_run=observing_run,
        seen_in=seen_in,
        confidence=confidence,
        name=name,
        weight=float(record['weight'])
    )


def segments_to_compressed_array(segments: list) -> np.ndarray:
    """
    Convert list of TransientSegments to compressed numpy array.
    
    Args:
        segments: List of TransientSegment objects
        
    Returns:
        Structured numpy array of shape (N,)
    """
    n = len(segments)
    compressed = np.zeros(n, dtype=COMPRESSED_SEGMENT_DTYPE)
    
    for i, seg in enumerate(segments):
        compressed[i] = segment_to_compressed(seg)
    
    return compressed


def compressed_array_to_segments(compressed: np.ndarray, names: Optional[dict] = None) -> list:
    """
    Inflate compressed array to list of TransientSegments.
    
    Args:
        compressed: Structured numpy array
        names: Optional dict mapping gps_key -> name
        
    Returns:
        List of TransientSegment objects
    """
    if names is None:
        names = {}
    
    segments = []
    for record in compressed:
        name = names.get(int(record['gps_key']))
        segment = compressed_to_segment(record, name=name)
        segments.append(segment)
    
    return segments
