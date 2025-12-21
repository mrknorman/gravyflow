"""
Unified segment classes for data acquisition.

This module provides a base Segment class and collection utilities used by
both NOISE and TRANSIENT acquisition modes.

Classes:
    Segment: Abstract base class for all segment types
    NoiseSegment: Segment for NOISE mode (time range for one IFO)
    SegmentCollection: Collection with array compatibility layer
"""
from dataclasses import dataclass, field
from abc import ABC
from typing import List, Tuple, Optional, TYPE_CHECKING, Iterator
import numpy as np

from gravyflow.src.utils.gps import gps_to_key

if TYPE_CHECKING:
    import gravyflow as gf
    from gravyflow.src.dataset.acquisition.base import ObservingRun


@dataclass
class Segment(ABC):
    """
    Abstract base class for all segment types.
    
    Provides common GPS time boundary handling used by both NOISE and
    TRANSIENT acquisition modes.
    
    Attributes:
        start_gps_time: Segment start time in GPS seconds
        end_gps_time: Segment end time in GPS seconds
    """
    start_gps_time: float
    end_gps_time: float
    
    @property
    def duration(self) -> float:
        """Segment duration in seconds."""
        return self.end_gps_time - self.start_gps_time
    
    @property
    def gps_key(self) -> int:
        """Integer GPS key for cache lookups (10ms precision)."""
        return gps_to_key(self.start_gps_time)
    
    @property
    def boundaries(self) -> Tuple[float, float]:
        """Segment boundaries as (start, end) tuple."""
        return (self.start_gps_time, self.end_gps_time)
    
    def to_array(self) -> np.ndarray:
        """Convert to [start, end] array."""
        return np.array([self.start_gps_time, self.end_gps_time], dtype=np.float64)


@dataclass
class NoiseSegment(Segment):
    """
    Segment for NOISE mode acquisition.
    
    Represents a contiguous time range of valid detector data for a single IFO.
    Used for sampling random or grid-based noise windows.
    
    Attributes:
        start_gps_time: Segment start time in GPS seconds
        end_gps_time: Segment end time in GPS seconds
        ifo: Interferometer this segment is for
        observing_run: Observing run (O1, O2, O3, O4)
    """
    ifo: "gf.IFO" = None
    observing_run: "ObservingRun" = None


class SegmentCollection:
    """
    Collection of Segment objects with array compatibility layer.
    
    Provides both object-oriented segment access and backward-compatible
    np.ndarray conversion for legacy code.
    
    Example:
        >>> collection = SegmentCollection([seg1, seg2, seg3])
        >>> len(collection)  # 3
        >>> collection[0]  # Returns Segment object
        >>> collection.to_array()  # Returns (N, 2) array
    """
    
    def __init__(self, segments: List[Segment] = None):
        self._segments: List[Segment] = segments or []
    
    def __len__(self) -> int:
        return len(self._segments)
    
    def __getitem__(self, idx: int) -> Segment:
        return self._segments[idx]
    
    def __iter__(self) -> Iterator[Segment]:
        return iter(self._segments)
    
    def __bool__(self) -> bool:
        return len(self._segments) > 0
    
    def append(self, segment: Segment) -> None:
        """Add a segment to the collection."""
        self._segments.append(segment)
    
    def extend(self, segments: List[Segment]) -> None:
        """Add multiple segments to the collection."""
        self._segments.extend(segments)
    
    def to_array(self) -> np.ndarray:
        """
        Convert to legacy (N, 2) array format.
        
        Returns:
            Array of shape (N, 2) with [start, end] for each segment.
        """
        if not self._segments:
            return np.empty((0, 2), dtype=np.float64)
        return np.array([s.boundaries for s in self._segments], dtype=np.float64)
    
    @classmethod
    def from_array(
        cls, 
        arr: np.ndarray, 
        ifo: "gf.IFO" = None,
        observing_run: "ObservingRun" = None
    ) -> "SegmentCollection":
        """
        Create collection from legacy (N, 2) array format.
        
        Args:
            arr: Array of shape (N, 2) with [start, end] times
            ifo: Optional IFO to assign to all segments
            observing_run: Optional observing run to assign
            
        Returns:
            SegmentCollection with NoiseSegment objects
        """
        segments = [
            NoiseSegment(
                start_gps_time=float(start),
                end_gps_time=float(end),
                ifo=ifo,
                observing_run=observing_run
            )
            for start, end in arr
        ]
        return cls(segments)
    
    def filter(self, predicate) -> "SegmentCollection":
        """Return new collection with only segments matching predicate."""
        return SegmentCollection([s for s in self._segments if predicate(s)])
    
    def sort_by_duration(self, ascending: bool = True) -> "SegmentCollection":
        """Return new collection sorted by segment duration."""
        sorted_segs = sorted(self._segments, key=lambda s: s.duration, reverse=not ascending)
        return SegmentCollection(sorted_segs)
    
    def shuffle(self, rng: np.random.Generator) -> None:
        """Shuffle segments in-place."""
        rng.shuffle(self._segments)
