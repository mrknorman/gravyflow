#!/usr/bin/env python3
"""
=============================================================================
TUTORIAL 04: TRANSIENT Mode - Basic Usage
=============================================================================

This tutorial demonstrates TRANSIENT mode - acquiring data windows centered
on specific events like glitches (instrumental artifacts).

LEARNING OBJECTIVES:
1. Understand TRANSIENT mode and how it differs from NOISE mode
2. Create a glitch-centered data obtainer
3. Understand GlitchType enum and filtering
4. Work with TransientSegment metadata

RUNTIME: ~3-4 minutes (downloads glitch data from GravitySpy + GWOSC)
"""

# =============================================================================
# IMPORTS
# =============================================================================

import gravyflow as gf
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# SECTION 1: UNDERSTANDING TRANSIENT MODE
# =============================================================================

print("=" * 70)
print("SECTION 1: Understanding TRANSIENT Mode")
print("=" * 70)

print("""
TRANSIENT mode acquires data CENTERED on specific events:

╔════════════════════════════════════════════════════════════════════════╗
║                           TRANSIENT MODE                               ║
╠════════════════════════════════════════════════════════════════════════╣
║                                                                        ║
║   Long data segment:                                                   ║
║   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ║
║              ▲              ▲          ▲                              ║
║              │              │          │                              ║
║         Glitch 1       Glitch 2    Glitch 3                          ║
║                                                                        ║
║   Windows extracted:                                                   ║
║      ┌───────┐          ┌───────┐    ┌───────┐                        ║
║      │   ●   │          │   ●   │    │   ●   │                        ║
║      └───────┘          └───────┘    └───────┘                        ║
║                                                                        ║
║   Key difference from NOISE mode:                                      ║
║   - NOISE: Random positions, no specific event                         ║
║   - TRANSIENT: Centered on known events (glitches, GW signals)        ║
║                                                                        ║
╚════════════════════════════════════════════════════════════════════════╝

Use cases:
  - Glitch classification (what type of artifact?)
  - Event detection (is there a signal?)
  - Signal characterization
""")

# =============================================================================
# SECTION 2: GLITCH TYPES
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 2: GlitchType Enum")
print("=" * 70)

print("""
GravitySpy has classified glitches into ~20 morphological types.
Each type corresponds to a different instrumental or environmental cause.
""")

print("Available GlitchType values:")
print("-" * 50)
for i, gt in enumerate(gf.GlitchType):
    print(f"  {gt.value:2d}: {gt.name}")

print(f"\nTotal: {len(gf.GlitchType)} glitch types")

# =============================================================================
# SECTION 3: CREATING A GLITCH OBTAINER
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 3: Creating a Glitch-Centered Obtainer")
print("=" * 70)

# The key parameter that enables TRANSIENT mode is data_labels=[gf.DataLabel.GLITCHES]
#
# Important parameters:
#   - data_labels: GLITCHES triggers TRANSIENT mode
#   - observing_runs: Which science runs to query for glitches
#   - Data is fetched from GravitySpy database

print("Creating TRANSIENT mode obtainer for glitches...")

glitch_obtainer = gf.IFODataObtainer(
    observing_runs=[gf.ObservingRun.O3],  # O3 has most labeled glitches
    data_quality=gf.DataQuality.BEST,
    data_labels=[gf.DataLabel.GLITCHES],  # ← TRANSIENT mode (glitches)
    segment_order=gf.SegmentOrder.RANDOM,
)

print(f"Created obtainer: {type(glitch_obtainer).__name__}")
print("Mode: TRANSIENT (glitch-centered windows)")

# =============================================================================
# SECTION 4: ACQUIRING GLITCH DATA
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 4: Acquiring Glitch Data")
print("=" * 70)

SAMPLE_RATE_HERTZ = 2048.0
ONSOURCE_DURATION_SECONDS = 1.0
OFFSOURCE_DURATION_SECONDS = 16.0
NUM_EXAMPLES = 8

# Create the NoiseObtainer wrapper (works for both NOISE and TRANSIENT modes)
transient_obtainer = gf.NoiseObtainer(
    ifo_data_obtainer=glitch_obtainer,
    ifos=[gf.IFO.L1],
)

print("Fetching glitch-centered windows...")
print("(This queries GravitySpy and downloads from GWOSC)")

data_gen = transient_obtainer(
    sample_rate_hertz=SAMPLE_RATE_HERTZ,
    onsource_duration_seconds=ONSOURCE_DURATION_SECONDS,
    offsource_duration_seconds=OFFSOURCE_DURATION_SECONDS,
    num_examples_per_batch=NUM_EXAMPLES,
)

batch = next(data_gen)

print(f"\nBatch keys: {list(batch.keys())}")

# =============================================================================
# SECTION 5: UNDERSTANDING TRANSIENT OUTPUT
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 5: TRANSIENT Mode Output")  
print("=" * 70)

onsource = batch[gf.ReturnVariables.ONSOURCE]
offsource = batch[gf.ReturnVariables.OFFSOURCE]

print(f"""
Data shapes (same BIS format as NOISE mode):
  Onsource:  {onsource.shape}
  Offsource: {offsource.shape}
""")

# TRANSIENT mode provides additional metadata
print("Additional metadata in TRANSIENT mode:")

# GPS times
if gf.ReturnVariables.START_GPS_TIME in batch:
    gps = batch[gf.ReturnVariables.START_GPS_TIME]
    print(f"\n  START_GPS_TIME (window start times):")
    for i, t in enumerate(gps[:4]):
        print(f"    Example {i}: GPS {t:.2f}")

# Glitch types (integer labels)
if gf.ReturnVariables.GLITCH_TYPE in batch:
    glitch_types = batch[gf.ReturnVariables.GLITCH_TYPE]
    print(f"\n  GLITCH_TYPE (integer labels):")
    for i, gt in enumerate(glitch_types[:4]):
        name = gf.GlitchType(gt).name if gt >= 0 else "UNKNOWN"
        print(f"    Example {i}: {gt} ({name})")

# Data labels
if gf.ReturnVariables.DATA_LABEL in batch:
    labels = batch[gf.ReturnVariables.DATA_LABEL]
    print(f"\n  DATA_LABEL: {labels[:4]} (all should be 1=GLITCHES)")

# =============================================================================
# SECTION 6: VISUALIZING GLITCHES
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 6: Visualizing Glitches")
print("=" * 70)

# Get glitch type names for labels
glitch_type_names = []
if gf.ReturnVariables.GLITCH_TYPE in batch:
    for gt in batch[gf.ReturnVariables.GLITCH_TYPE]:
        if gt >= 0:
            glitch_type_names.append(gf.GlitchType(gt).name)
        else:
            glitch_type_names.append("UNKNOWN")
else:
    glitch_type_names = ["N/A"] * onsource.shape[0]

# Create visualization
fig, axes = plt.subplots(2, 4, figsize=(16, 6))
fig.suptitle("TRANSIENT Mode: Glitch-Centered Windows from GravitySpy", fontsize=14)

time = np.arange(onsource.shape[2]) / SAMPLE_RATE_HERTZ - ONSOURCE_DURATION_SECONDS/2

for i in range(min(8, onsource.shape[0])):
    ax = axes[i // 4, i % 4]
    ax.plot(time, onsource[i, 0, :], 'b-', lw=0.5)
    
    # Add vertical line at center where glitch should be
    ax.axvline(x=0, color='r', linestyle='--', alpha=0.5, label='Glitch center')
    
    ax.set_title(f"{glitch_type_names[i]}", fontsize=10)
    ax.set_xlabel("Time from center (s)")
    ax.set_ylabel("Strain")
    ax.grid(True, alpha=0.3)
    
    # Zoom to show glitch structure
    ax.set_xlim(-0.5, 0.5)

plt.tight_layout()
plt.savefig("/home/michael.norman/gravyflow/example_notebooks/tutorials/04_glitches.png", dpi=100)
print("Saved plot to: tutorials/04_glitches.png")
plt.close()

# =============================================================================
# SECTION 7: TRANSIENT SEGMENT METADATA
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 7: TransientSegment Metadata")
print("=" * 70)

print("""
Each glitch/event is represented internally as a TransientSegment,
which contains rich metadata:

TransientSegment attributes:
  - transient_gps_time: Exact GPS time of the event center
  - start_gps_time: Start of the data window
  - end_gps_time: End of the data window
  - ifo: Which detector
  - data_label: GLITCHES or EVENTS
  - glitch_type: GlitchType enum value (if glitch)
  - event_name: Event catalog name (if GW event)

This metadata is tracked internally and can be accessed for
detailed analysis, cross-referencing with catalogs, etc.
""")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 70)
print("SUMMARY: TRANSIENT Mode Basics")
print("=" * 70)

print("""
1. TRANSIENT mode is selected with:
   - data_labels=[gf.DataLabel.GLITCHES]  (for glitches)
   - data_labels=[gf.DataLabel.EVENTS]    (for GW events)

2. Data is CENTERED on known events (not random like NOISE mode)

3. GlitchType enum has 20 morphological types from GravitySpy

4. Output includes additional metadata:
   - GLITCH_TYPE: Integer label (0-19)
   - START_GPS_TIME: Window start time
   - TRANSIENT_GPS_TIME: Event center time
   - DATA_LABEL: What type (NOISE=0, GLITCHES=1, EVENTS=2)

5. Glitches are at the CENTER of the onsource window
   (t=0 relative to window midpoint)
""")

print("\n✓ Tutorial 04 complete!")
print("Next: 05_transient_mode_advanced.py - Events, balancing, and caching")
