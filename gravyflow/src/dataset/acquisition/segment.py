"""Segment, the abstract base of TransientSegment: a GPS time range."""
from dataclasses import dataclass
from abc import ABC


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
