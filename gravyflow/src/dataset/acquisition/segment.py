"""Segment, the abstract base of TransientSegment: a GPS time range."""
from dataclasses import dataclass
from abc import ABC
from typing import Tuple
import numpy as np

from gravyflow.src.utils.gps import gps_to_key


@dataclass
class Segment(ABC):
    """
    Abstract base class for all segment types.
    
    Provides the GPS time boundary handling of TransientSegment.
    
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
