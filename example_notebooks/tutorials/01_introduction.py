#!/usr/bin/env python3
"""
=============================================================================
TUTORIAL 01: Introduction to GravyFlow Acquisition Pipeline
=============================================================================

This tutorial provides an overview of the gravyflow data acquisition system,
which is designed for acquiring strain data from LIGO/Virgo gravitational
wave detectors.

LEARNING OBJECTIVES:
1. Understand the two acquisition modes: NOISE and TRANSIENT
2. Learn the key configuration enums
3. Understand the IFODataObtainer factory pattern
4. See the basic module structure

RUNTIME: < 30 seconds (no data download)
"""

# =============================================================================
# IMPORTS
# =============================================================================

import gravyflow as gf

# =============================================================================
# SECTION 1: THE TWO ACQUISITION MODES
# =============================================================================

print("=" * 70)
print("SECTION 1: Understanding Acquisition Modes")
print("=" * 70)

print("""
GravyFlow supports two fundamentally different ways to acquire detector data:

╔════════════════════════════════════════════════════════════════════════╗
║  MODE         │  DESCRIPTION                                           ║
╠════════════════════════════════════════════════════════════════════════╣
║  NOISE        │  Random or grid sampling from long data segments.      ║
║               │  Use for: noise-only models, background estimation.    ║
╠════════════════════════════════════════════════════════════════════════╣
║  TRANSIENT    │  Windows centered on specific events (GW events or     ║
║               │  glitches). Use for: event detection, classification.  ║
╚════════════════════════════════════════════════════════════════════════╝

The mode is selected automatically based on the `data_labels` parameter:
  - DataLabel.NOISE       → NOISE mode
  - DataLabel.GLITCHES    → TRANSIENT mode  
  - DataLabel.EVENTS      → TRANSIENT mode
""")

# =============================================================================
# SECTION 2: KEY CONFIGURATION ENUMS
# =============================================================================

print("=" * 70)
print("SECTION 2: Key Configuration Enums")
print("=" * 70)

# --- 2.1 ObservingRun ---
print("\n2.1 ObservingRun - Which observing run to use")
print("-" * 50)
print("Available runs:")
for run in gf.ObservingRun:
    data = run.value
    print(f"  {run.name}: {data.name}")
    print(f"    Start GPS: {data.start_gps_time:.0f}")
    print(f"    End GPS:   {data.end_gps_time:.0f}")

# --- 2.2 DataQuality ---
print("\n2.2 DataQuality - Data quality requirements")
print("-" * 50)
for quality in gf.DataQuality:
    print(f"  {quality.name}: ", end="")
    if quality == gf.DataQuality.BEST:
        print("Science-quality calibrated data (recommended)")
    else:
        print("Raw uncalibrated data")

# --- 2.3 DataLabel ---
print("\n2.3 DataLabel - What type of data to acquire")
print("-" * 50)
for label in gf.DataLabel:
    desc = {
        gf.DataLabel.NOISE: "Random noise samples (NOISE mode)",
        gf.DataLabel.GLITCHES: "Glitch-centered windows (TRANSIENT mode)",
        gf.DataLabel.EVENTS: "GW event-centered windows (TRANSIENT mode)",
    }
    print(f"  {label.name} (value={label.value}): {desc.get(label, 'Unknown')}")

# --- 2.4 SegmentOrder ---
print("\n2.4 SegmentOrder - Order of segment iteration")
print("-" * 50)
for order in gf.SegmentOrder:
    desc = {
        gf.SegmentOrder.RANDOM: "Shuffle segments randomly (default)",
        gf.SegmentOrder.CHRONOLOGICAL: "Process in GPS time order",
        gf.SegmentOrder.SHORTEST_FIRST: "Start with shortest segments",
    }
    print(f"  {order.name}: {desc.get(order, 'Unknown')}")

# --- 2.5 IFO (Interferometer) ---
print("\n2.5 IFO - Detector selection")
print("-" * 50)
for ifo in gf.IFO:
    data = ifo.value
    print(f"  {ifo.name}: {data.name}")
    print(f"    Location: ({data.latitude_radians:.4f} rad, {data.longitude_radians:.4f} rad)")
    print(f"    Arm length: {data.x_length_meters:.0f}m")

# =============================================================================
# SECTION 3: THE IFODataObtainer FACTORY
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 3: The IFODataObtainer Factory Pattern")
print("=" * 70)

print("""
IFODataObtainer is a FACTORY function that creates the appropriate 
obtainer class based on your data_labels:

    ┌─────────────────────────────────────────────────────────────────┐
    │  gf.IFODataObtainer(data_labels=[...], ...)                     │
    └───────────────────────────┬─────────────────────────────────────┘
                                │
                ┌───────────────┴───────────────┐
                ▼                               ▼
    ┌───────────────────────┐       ┌───────────────────────┐
    │   NoiseDataObtainer   │       │ TransientDataObtainer │
    │   (DataLabel.NOISE)   │       │ (GLITCHES or EVENTS)  │
    └───────────────────────┘       └───────────────────────┘

This means you use the same interface regardless of mode - the factory
handles selecting the right implementation.
""")

# Demonstrate the factory pattern
print("Example: Creating a NOISE mode obtainer")
print("-" * 50)
print("""
noise_obtainer = gf.IFODataObtainer(
    observing_runs=[gf.ObservingRun.O3],
    data_quality=gf.DataQuality.BEST,
    data_labels=[gf.DataLabel.NOISE],  # ← This selects NOISE mode
)
""")

print("Example: Creating a TRANSIENT mode obtainer")
print("-" * 50)
print("""
glitch_obtainer = gf.IFODataObtainer(
    observing_runs=[gf.ObservingRun.O3],
    data_quality=gf.DataQuality.BEST,
    data_labels=[gf.DataLabel.GLITCHES],  # ← This selects TRANSIENT mode
)
""")

# =============================================================================
# SECTION 4: DATA AUGMENTATIONS (New API)
# =============================================================================

print("=" * 70)
print("SECTION 4: Data Augmentations (Dataclass API)")
print("=" * 70)

print("""
GravyFlow uses a flexible dataclass-based augmentation system.
Each augmentation type has its own probability and parameters:
""")

print("Available augmentation types:")
print("-" * 50)

augmentation_types = [
    ("gf.SignReversal", "Flip sign (x → -x)", "All modes", "probability"),
    ("gf.TimeReversal", "Reverse time axis", "All modes", "probability"),
    ("gf.RandomShift", "Shift event off-center", "TRANSIENT only", "probability, shift_fraction"),
    ("gf.AddNoise", "Add noise perturbation", "TRANSIENT only", "probability, amplitude"),
]

for name, desc, mode, params in augmentation_types:
    print(f"  {name}")
    print(f"    Description: {desc}")
    print(f"    Applicable:  {mode}")
    print(f"    Parameters:  {params}")
    print()

# =============================================================================
# SECTION 5: MODULE STRUCTURE
# =============================================================================

print("=" * 70)
print("SECTION 5: Module Structure")
print("=" * 70)

print("""
The acquisition module is organized as follows:

gravyflow/
└── src/dataset/acquisition/
    ├── __init__.py          # Public API exports
    ├── base.py              # Shared enums, IFOData class, utilities
    ├── obtainer.py          # IFODataObtainer factory function
    ├── noise.py             # NoiseDataObtainer (NOISE mode)
    ├── transient.py         # TransientDataObtainer (TRANSIENT mode)
    ├── augmentations.py     # SignReversal, TimeReversal, etc.
    ├── segment.py           # Segment, NoiseSegment, SegmentCollection
    ├── transient_segment.py # TransientSegment (event/glitch metadata)
    └── cache.py             # Caching utilities

Key imports available directly from gravyflow:
""")

key_symbols = [
    "IFODataObtainer", "NoiseDataObtainer", "TransientDataObtainer",
    "DataQuality", "DataLabel", "SegmentOrder", "ObservingRun", "IFO",
    "SignReversal", "TimeReversal", "RandomShift", "AddNoise",
]
print("  " + ", ".join(key_symbols))

# =============================================================================
# NEXT STEPS
# =============================================================================

print("\n" + "=" * 70)
print("NEXT TUTORIALS")
print("=" * 70)

print("""
Continue with the following tutorials:

  02_noise_mode_basics.py     - Basic NOISE mode acquisition
  03_noise_mode_advanced.py   - Advanced NOISE features (grid, groups, multi-IFO)
  04_transient_mode_basics.py - Basic TRANSIENT mode (glitches)
  05_transient_mode_advanced.py - Advanced TRANSIENT (events, caching)
  06_augmentations.py         - Data augmentation in detail
  07_caching_performance.py   - Caching and performance optimization
""")

print("\n✓ Tutorial 01 complete!")
