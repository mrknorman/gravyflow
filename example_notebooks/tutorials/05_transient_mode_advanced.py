#!/usr/bin/env python3
"""
=============================================================================
TUTORIAL 05: TRANSIENT Mode - Advanced Features
=============================================================================

This tutorial covers advanced TRANSIENT mode features including GW events,
class balancing, specific event names, and the TransientCache system.

LEARNING OBJECTIVES:
1. Use DataLabel.EVENTS for gravitational wave events
2. Fetch specific events by name
3. Enable class balancing for glitch types
4. Understand ReturnVariables for metadata access
5. Learn about the TransientCache system

RUNTIME: ~3-4 minutes
"""

# =============================================================================
# IMPORTS
# =============================================================================

import gravyflow as gf
import numpy as np

# =============================================================================
# SECTION 1: GRAVITATIONAL WAVE EVENTS
# =============================================================================

print("=" * 70)
print("SECTION 1: Gravitational Wave Events")
print("=" * 70)

print("""
Besides glitches, TRANSIENT mode can acquire windows around confirmed
gravitational wave events from the GWTC catalogs.

Use DataLabel.EVENTS to enable GW event acquisition:

    gf.IFODataObtainer(
        data_labels=[gf.DataLabel.EVENTS],  # ← GW events
        ...
    )

Event data:
  - Events are from GWOSC event catalogs (GWTC-1, GWTC-2, GWTC-3)
  - Each event has a unique name (e.g., "GW150914", "GW190521")
  - Windows are centered on the merger time
""")

# Create an event obtainer
event_obtainer = gf.IFODataObtainer(
    observing_runs=[gf.ObservingRun.O3],
    data_quality=gf.DataQuality.BEST,
    data_labels=[gf.DataLabel.EVENTS],  # GW events
)

print(f"Created event obtainer: {type(event_obtainer).__name__}")

# =============================================================================
# SECTION 2: FETCHING SPECIFIC EVENTS BY NAME
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 2: Fetching Specific Events by Name")
print("=" * 70)

print("""
You can request specific events using the 'event_names' parameter.
This is useful for:
  - Reproducible analysis of specific detections
  - Studying particular events in detail
  - Testing on known signals

Example:
    obtainer = gf.IFODataObtainer(
        event_names=["GW190521", "GW190814"],  # ← Specific events
        ...
    )
""")

# Note: event_names filtering happens at the obtainer level
specific_obtainer = gf.IFODataObtainer(
    observing_runs=[gf.ObservingRun.O3],
    data_quality=gf.DataQuality.BEST,
    data_labels=[gf.DataLabel.EVENTS],
    # event_names=["GW190521", "GW190814"],  # Would filter to these
)

print("Note: When event_names is specified, only those events are returned")
print("regardless of observing_run filters.")

# =============================================================================
# SECTION 3: CLASS BALANCING FOR GLITCHES
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 3: Class Balancing for Glitch Types")
print("=" * 70)

print("""
Glitch types have VERY imbalanced distributions. Some types (like BLIP)
have thousands of examples, while others have only hundreds.

For classification training, use 'balanced_glitch_types=True':

    obtainer = gf.IFODataObtainer(
        data_labels=[gf.DataLabel.GLITCHES],
        balanced_glitch_types=True,  # ← Enable balancing
        ...
    )

How balancing works:
  1. Count examples of each glitch type
  2. Oversample minority classes to match majority
  3. Result: Equal probability of seeing any type

This prevents the model from just predicting the most common type.
""")

# Create a balanced glitch obtainer
balanced_obtainer = gf.IFODataObtainer(
    observing_runs=[gf.ObservingRun.O3],
    data_quality=gf.DataQuality.BEST,
    data_labels=[gf.DataLabel.GLITCHES],
    balanced_glitch_types=True,  # Enable class balancing
)

print("Created balanced glitch obtainer")
print("Each glitch type will be equally represented in batches")

# =============================================================================
# SECTION 4: RETURN VARIABLES
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 4: ReturnVariables - Accessing Metadata")
print("=" * 70)

print("""
ReturnVariables is an enum that defines all possible outputs from
the acquisition pipeline. Use these as keys to access batch data.

╔═══════════════════════════════════════════════════════════════════════╗
║  VARIABLE              │  DESCRIPTION                                 ║
╠═══════════════════════════════════════════════════════════════════════╣
║  Core Data:                                                           ║
║  ───────────────────────────────────────────────────────────────────  ║
║  ONSOURCE              │  Main data window (Batch, IFO, Samples)     ║
║  OFFSOURCE             │  Background window for PSD estimation       ║
║  WHITENED_ONSOURCE     │  Pre-whitened onsource (if requested)       ║
╠═══════════════════════════════════════════════════════════════════════╣
║  GPS Times:                                                           ║
║  ───────────────────────────────────────────────────────────────────  ║
║  START_GPS_TIME        │  Start of onsource window                    ║
║  TRANSIENT_GPS_TIME    │  Event/glitch center time                    ║
╠═══════════════════════════════════════════════════════════════════════╣
║  Labels:                                                              ║
║  ───────────────────────────────────────────────────────────────────  ║
║  DATA_LABEL            │  0=NOISE, 1=GLITCHES, 2=EVENTS              ║
║  GLITCH_TYPE           │  GlitchType.value (0-19) or -1              ║
║  SOURCE_TYPE           │  For injected signals                       ║
╚═══════════════════════════════════════════════════════════════════════╝
""")

# Show all ReturnVariables
print("All ReturnVariables:")
for rv in sorted(gf.ReturnVariables, key=lambda x: x.value.index):
    print(f"  {rv.name}: index={rv.value.index}")

# =============================================================================
# SECTION 5: TRANSIENT CACHE SYSTEM
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 5: TransientCache System")
print("=" * 70)

print("""
TRANSIENT mode uses a caching system to avoid re-downloading data:

┌─────────────────────────────────────────────────────────────────────┐
│                      TransientCache Flow                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Request GPS time → Check memory cache → Check disk cache → GWOSC │
│                            ↓                     ↓           ↓     │
│                          HIT?                  HIT?      Download   │
│                            ↓                     ↓           ↓     │
│                         Return              Load → Return → Cache   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

Cache benefits:
  1. MEMORY CACHE: Fastest, holds recently accessed segments
  2. DISK CACHE: HDF5 files with pre-extracted windows
  3. Persistence: Cached data survives restarts

Cache files are stored in:
  ~/.gravyflow/cache/transient/

To pre-populate the cache (for large training runs):
  
    gf.precache_transients(
        ifos=[gf.IFO.L1, gf.IFO.H1],
        sample_rate_hertz=2048.0,
        onsource_duration_seconds=1.0,
        offsource_duration_seconds=16.0,
    )
""")

# =============================================================================
# SECTION 6: COMBINING GLITCHES AND EVENTS
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 6: Combining Multiple Data Labels")
print("=" * 70)

print("""
You can combine multiple data labels for mixed training:

    obtainer = gf.IFODataObtainer(
        data_labels=[gf.DataLabel.GLITCHES, gf.DataLabel.EVENTS],
        ...
    )

This is useful for:
  - Training detectors that distinguish signals from glitches
  - Multi-class classification (noise vs glitch vs event)
  - Building robust models

The DATA_LABEL variable tells you which type each example is.
""")

# =============================================================================
# SECTION 7: DEMONSTRATION
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 7: Quick Demonstration")
print("=" * 70)

# Create a simple glitch obtainer and fetch data
print("Creating balanced glitch obtainer and fetching batch...")

obtainer = gf.IFODataObtainer(
    observing_runs=[gf.ObservingRun.O3],
    data_quality=gf.DataQuality.BEST,
    data_labels=[gf.DataLabel.GLITCHES],
    balanced_glitch_types=True,
)

transient = gf.NoiseObtainer(
    ifo_data_obtainer=obtainer,
    ifos=[gf.IFO.L1],
)

data_gen = transient(
    sample_rate_hertz=2048.0,
    onsource_duration_seconds=1.0,
    offsource_duration_seconds=16.0,
    num_examples_per_batch=16,
)

batch = next(data_gen)

# Count glitch types in batch
print("\nGlitch types in batch (with balancing):")
if gf.ReturnVariables.GLITCH_TYPE in batch:
    types = batch[gf.ReturnVariables.GLITCH_TYPE]
    unique, counts = np.unique(types, return_counts=True)
    for gt_val, count in zip(unique, counts):
        name = gf.GlitchType(gt_val).name if gt_val >= 0 else "UNKNOWN"
        print(f"  {name}: {count}")
    print(f"\nTotal unique types in batch: {len(unique)}")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 70)
print("SUMMARY: Advanced TRANSIENT Features")
print("=" * 70)

print("""
1. GW EVENTS:
   - Use data_labels=[gf.DataLabel.EVENTS]
   - Windows centered on merger time

2. SPECIFIC EVENTS:
   - Use event_names=["GW150914", ...] parameter
   - Overrides observing_run filters

3. CLASS BALANCING:
   - Use balanced_glitch_types=True
   - Equalizes glitch type distribution
   - Critical for classification training

4. RETURN VARIABLES:
   - ONSOURCE, OFFSOURCE: Core data
   - START_GPS_TIME, TRANSIENT_GPS_TIME: Timing
   - GLITCH_TYPE, DATA_LABEL: Labels

5. CACHING:
   - Memory + disk caching
   - ~/.gravyflow/cache/transient/
   - Use precache_transients() for large runs
""")

print("\n✓ Tutorial 05 complete!")
print("Next: 06_augmentations.py - Data augmentation system")
