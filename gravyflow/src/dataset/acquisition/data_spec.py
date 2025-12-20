"""
Unified data specification types for the acquisition pipeline.

This module provides normalized request specifications that replace
scattered type-checking logic throughout the pipeline.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import gravyflow as gf


@dataclass
class DataLabelSpec:
    """
    Normalized specification of what data types to acquire.
    
    Replaces scattered isinstance checks like:
        has_specific_glitches = any(isinstance(label, GlitchType) for label in data_labels_list)
        has_glitches_label = DataLabel.GLITCHES in data_labels_list
    
    With a single normalized object that's computed once during __init__.
    """
    include_glitches: bool = False
    include_events: bool = False
    glitch_types: Optional[List["gf.GlitchType"]] = None  # None = all types
    event_confidences: Optional[List["gf.EventConfidence"]] = None  # None = all
    
    @classmethod
    def from_labels(cls, labels) -> "DataLabelSpec":
        """
        Parse mixed list of DataLabel, GlitchType, EventConfidence.
        
        Args:
            labels: Single label or list of labels (DataLabel, GlitchType, or EventConfidence)
            
        Returns:
            Normalized DataLabelSpec
        """
        if not isinstance(labels, list):
            labels = [labels]
        
        spec = cls()
        glitch_types = []
        event_confidences = []
        
        for label in labels:
            if label == gf.DataLabel.GLITCHES:
                spec.include_glitches = True
            elif label == gf.DataLabel.EVENTS:
                spec.include_events = True
            elif isinstance(label, gf.GlitchType):
                spec.include_glitches = True
                glitch_types.append(label)
            elif isinstance(label, gf.EventConfidence):
                spec.include_events = True
                event_confidences.append(label)
        
        # Only set lists if specific types were requested
        if glitch_types:
            spec.glitch_types = glitch_types
        if event_confidences:
            spec.event_confidences = event_confidences
            
        return spec
    
    @property
    def wants_any_transients(self) -> bool:
        """True if either glitches or events are requested."""
        return self.include_glitches or self.include_events


@dataclass
class ValidationResult:
    """Result of data validation check."""
    is_valid: bool
    has_nan: bool = False
    has_all_zeros: bool = False
    gps_time: float = 0.0
    message: str = ""
    
    @classmethod
    def valid(cls, gps_time: float = 0.0) -> "ValidationResult":
        """Create a valid result."""
        return cls(is_valid=True, gps_time=gps_time)
    
    @classmethod
    def invalid(cls, gps_time: float, reason: str, has_nan: bool = False, has_all_zeros: bool = False) -> "ValidationResult":
        """Create an invalid result with reason."""
        return cls(
            is_valid=False,
            has_nan=has_nan,
            has_all_zeros=has_all_zeros,
            gps_time=gps_time,
            message=reason
        )
