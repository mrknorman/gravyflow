"""
Acquisition module - IFO data acquisition for gravitational wave analysis.

This package provides data obtainers for acquiring strain data from
LIGO/Virgo interferometers in two modes:

- NOISE mode: Random or grid sampling from long data segments
- TRANSIENT mode: Centered windows around specific events (GW mergers, glitches)

The main entry point is IFODataObtainer, which automatically selects
the appropriate obtainer class based on the data_labels parameter.

Each public name imports its submodule the first time it is used, as
``gravyflow``'s own names do: importing one submodule (``base`` for
``ObservingRun``, say) never imports the obtainers, so the catalogue modules
the obtainers import can import ``base`` without a cycle.
"""

import importlib

#: Each public name and the submodule that defines it.
_MODULES = {
    # Enums
    "DataQuality": "base",
    "DataLabel": "base",
    "SegmentOrder": "base",
    "AcquisitionMode": "base",
    "SamplingMode": "base",
    "Coincidence": "base",
    "ObservingRun": "base",
    "ObservingRunData": "base",
    # Data structures
    "IFOData": "base",
    # Constants
    "GPS_PRECISION_DECIMALS": "base",
    "GPS_TOLERANCE_SECONDS": "base",
    # Public GWOSC sources
    "GWOSC_RUNS": "gwosc_data",
    "GWOSC_DATA_FLAGS": "gwosc_data",
    "GWOSC_BURST_FLAGS": "gwosc_data",
    "GWOSC_CBC_FLAGS": "gwosc_data",
    "GWOSC_NO_HW_INJECTION_FLAGS": "gwosc_data",
    "DEFAULT_GWOSC_FLAGS": "gwosc_data",
    "UpsamplingRefused": "gwosc_data",
    # Data sources
    "DataSource": "sources",
    "GWOSC": "sources",
    "LocalFrames": "sources",
    "StrainUnavailable": "sources",
    # Utility functions
    "ensure_even": "base",
    # Classes
    "BaseDataObtainer": "base",
    "NoiseDataObtainer": "noise",
    "TransientDataObtainer": "transient",
    "IFODataObtainer": "obtainer",
    # Segment classes
    "Segment": "segment",
    "TransientSegment": "transient_segment",
    # Augmentations
    "SignReversal": "augmentations",
    "TimeReversal": "augmentations",
    "RandomShift": "augmentations",
    "AddNoise": "augmentations",
    "Augmentation": "augmentations",
}

__all__ = list(_MODULES)


def __getattr__(name):
    if name not in _MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(f"{__name__}.{_MODULES[name]}"), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
