"""
Acquisition module - IFO data acquisition for gravitational wave analysis.

This package provides data obtainers for acquiring strain data from 
LIGO/Virgo interferometers in two modes:

- NOISE mode: Random or grid sampling from long data segments
- TRANSIENT mode: Centered windows around specific events (GW mergers, glitches)

The main entry point is IFODataObtainer, which automatically selects
the appropriate obtainer class based on the data_labels parameter.
"""

# Re-export all public symbols from submodules
from .base import (
    # Enums
    DataQuality,
    DataLabel,
    SegmentOrder,
    AcquisitionMode,
    SamplingMode,
    ObservingRun,
    ObservingRunData,
    # Data structures
    IFOData,
    # Constants
    GPS_PRECISION_DECIMALS,
    GPS_TOLERANCE_SECONDS,
    SEGMENT_EPSILON_SECONDS,
    # Utility functions
    ensure_even,
    ifo_canonical_key,
    random_subsection,
    concatenate_batches,
    # Base class (for type hints)
    BaseDataObtainer,
)

from .noise import NoiseDataObtainer
from .transient import TransientDataObtainer
from .transient_segment import TransientSegment
from .segment import Segment, NoiseSegment, SegmentCollection
from .augmentations import (
    SignReversal,
    TimeReversal,
    RandomShift,
    AddNoise,
    Augmentation,
    default_augmentations,
)

# Import the factory function
from .obtainer import IFODataObtainer

__all__ = [
    # Enums
    'DataQuality',
    'DataLabel', 
    'SegmentOrder',
    'AcquisitionMode',
    'SamplingMode',
    'ObservingRun',
    'ObservingRunData',
    # Data structures
    'IFOData',
    # Constants
    'GPS_PRECISION_DECIMALS',
    'GPS_TOLERANCE_SECONDS',
    'SEGMENT_EPSILON_SECONDS',
    # Utility functions
    'ensure_even',
    'ifo_canonical_key',
    'random_subsection',
    'concatenate_batches',
    # Classes
    'BaseDataObtainer',
    'NoiseDataObtainer',
    'TransientDataObtainer',
    'IFODataObtainer',
    # Segment classes
    'Segment',
    'NoiseSegment',
    'SegmentCollection',
    'TransientSegment',
    # Augmentations
    'SignReversal',
    'TimeReversal',
    'RandomShift',
    'AddNoise',
    'Augmentation',
    'default_augmentations',
]
