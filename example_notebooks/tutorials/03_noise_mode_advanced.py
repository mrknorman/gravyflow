#!/usr/bin/env python3
"""
=============================================================================
TUTORIAL 03: NOISE Mode - Advanced Features
=============================================================================

This tutorial covers advanced NOISE mode features including grid sampling,
train/validation splitting, and multi-detector acquisition.

LEARNING OBJECTIVES:
1. Understand SamplingMode: RANDOM vs GRID
2. Use groups for train/validation splitting
3. Acquire data from multiple detectors
4. Work with segment collections

RUNTIME: ~2-3 minutes
"""

# =============================================================================
# IMPORTS
# =============================================================================

import gravyflow as gf
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# CONFIGURATION
# =============================================================================

SAMPLE_RATE_HERTZ = 2048.0
ONSOURCE_DURATION_SECONDS = 1.0
OFFSOURCE_DURATION_SECONDS = 16.0
NUM_EXAMPLES = 4

# =============================================================================
# SECTION 1: SAMPLING MODES - RANDOM vs GRID
# =============================================================================

print("=" * 70)
print("SECTION 1: Sampling Modes")
print("=" * 70)

print("""
GravyFlow supports two sampling strategies for NOISE mode:

╔══════════════════════════════════════════════════════════════════════╗
║  MODE              │  DESCRIPTION                                    ║
╠══════════════════════════════════════════════════════════════════════╣
║  SamplingMode.RANDOM │  Pick random positions within each segment.   ║
║                      │  Good for: training (max diversity)           ║
╠══════════════════════════════════════════════════════════════════════╣
║  SamplingMode.GRID   │  Systematically step through segments.        ║
║                      │  Good for: inference (full coverage)          ║
╚══════════════════════════════════════════════════════════════════════╝

RANDOM mode (default):
  ┌──────────────────────────────────────────────────────────┐
  │  Segment A  │  Segment B           │  Segment C          │
  │   ●    ●    │  ●      ●       ●    │    ●   ●            │  ← Random
  └──────────────────────────────────────────────────────────┘

GRID mode:
  ┌──────────────────────────────────────────────────────────┐
  │  Segment A  │  Segment B           │  Segment C          │
  │  ●●●●●●●●●  │  ●●●●●●●●●●●●●●●●●●  │  ●●●●●●●●●●●●●●●●   │  ← Grid
  └──────────────────────────────────────────────────────────┘
""")

# Demonstrate SamplingMode enum
print("SamplingMode values:")
for mode in gf.SamplingMode:
    print(f"  gf.SamplingMode.{mode.name}")

# =============================================================================
# SECTION 2: TRAIN/VALIDATION SPLITTING WITH GROUPS
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 2: Train/Validation Splitting")
print("=" * 70)

print("""
The 'groups' parameter enables deterministic train/validation splitting.
Segments are assigned to groups based on a hash of their GPS times,
ensuring consistent splits across runs.

How it works:
  1. Each segment gets hashed based on GPS time and chunk position
  2. Hash determines which group the segment belongs to
  3. Same segments always go to same group (reproducible splits)

Example configuration:
  groups = {"train": 0.8, "validate": 0.2}
  
  This assigns 80% of segments to "train" and 20% to "validate"
""")

# Create obtainers for train and validation
print("Creating train obtainer (80% of segments)...")
train_obtainer = gf.IFODataObtainer(
    observing_runs=[gf.ObservingRun.O3],
    data_quality=gf.DataQuality.BEST,
    data_labels=[gf.DataLabel.NOISE],
    segment_order=gf.SegmentOrder.RANDOM,
)

print("Creating validation obtainer (20% of segments)...")
val_obtainer = gf.IFODataObtainer(
    observing_runs=[gf.ObservingRun.O3],
    data_quality=gf.DataQuality.BEST,
    data_labels=[gf.DataLabel.NOISE],
    segment_order=gf.SegmentOrder.RANDOM,
)

# The group assignment happens when creating the NoiseObtainer
# via the 'group' and 'groups' parameters

train_noise = gf.NoiseObtainer(
    ifo_data_obtainer=train_obtainer,
    ifos=[gf.IFO.L1],
    groups={"train": 0.8, "validate": 0.2},  # Split ratios
)

val_noise = gf.NoiseObtainer(
    ifo_data_obtainer=val_obtainer,
    ifos=[gf.IFO.L1],
    groups={"train": 0.8, "validate": 0.2},  # Same ratios
)

print(f"Train obtainer uses group: 'train' (80%)")
print(f"Validation obtainer uses group: 'validate' (20%)")
print("\nNote: Groups are passed to NoiseObtainer and selected via 'group' param when calling")

# =============================================================================
# SECTION 3: MULTI-DETECTOR ACQUISITION
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 3: Multi-Detector Acquisition")
print("=" * 70)

print("""
Acquire data from multiple detectors simultaneously.
This is essential for:
  - Coincidence detection (signal in multiple detectors)
  - Sky localization (triangulation)
  - Coherent analysis

Available detectors:
  - gf.IFO.L1: Livingston (Louisiana, USA)
  - gf.IFO.H1: Hanford (Washington, USA)  
  - gf.IFO.V1: Virgo (Cascina, Italy)

When using multiple IFOs, data is downloaded for each detector
and the output shape becomes (Batch, NumIFOs, Samples).
""")

# Create a multi-detector obtainer
multi_ifo_obtainer = gf.IFODataObtainer(
    observing_runs=[gf.ObservingRun.O3],
    data_quality=gf.DataQuality.BEST,
    data_labels=[gf.DataLabel.NOISE],
)

multi_noise = gf.NoiseObtainer(
    ifo_data_obtainer=multi_ifo_obtainer,
    ifos=[gf.IFO.L1, gf.IFO.H1],  # Two detectors
)

print("Created multi-detector obtainer with: L1 (Livingston), H1 (Hanford)")
print(f"Output shape will be: ({NUM_EXAMPLES}, 2, {int(SAMPLE_RATE_HERTZ * ONSOURCE_DURATION_SECONDS)})")

# Get data from multi-detector setup
print("\nFetching multi-detector data (this may take a moment)...")

data_gen = multi_noise(
    sample_rate_hertz=SAMPLE_RATE_HERTZ,
    onsource_duration_seconds=ONSOURCE_DURATION_SECONDS,
    offsource_duration_seconds=OFFSOURCE_DURATION_SECONDS,
    num_examples_per_batch=NUM_EXAMPLES,
)

batch = next(data_gen)
onsource = batch[gf.ReturnVariables.ONSOURCE]

print(f"\nMulti-IFO onsource shape: {onsource.shape}")
print(f"  - Batch:   {onsource.shape[0]} examples")
print(f"  - IFOs:    {onsource.shape[1]} detectors (L1, H1)")
print(f"  - Samples: {onsource.shape[2]} samples")

# Visualize multi-detector data
fig, axes = plt.subplots(2, 2, figsize=(12, 6))
time = np.arange(onsource.shape[2]) / SAMPLE_RATE_HERTZ
ifo_names = ["L1 (Livingston)", "H1 (Hanford)"]

for i in range(min(2, onsource.shape[0])):
    for j, ifo_name in enumerate(ifo_names):
        ax = axes[i, j]
        ax.plot(time, onsource[i, j, :], lw=0.5)
        ax.set_title(f"Example {i+1} - {ifo_name}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Strain")
        ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("/home/michael.norman/gravyflow/example_notebooks/tutorials/03_multi_ifo.png", dpi=100)
print("\nSaved plot to: tutorials/03_multi_ifo.png")
plt.close()

# =============================================================================
# SECTION 4: SEGMENT ORDERING STRATEGIES
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 4: Segment Ordering Strategies")
print("=" * 70)

print("""
The segment_order parameter controls how segments are processed:

╔═══════════════════════════════════════════════════════════════════════╗
║  ORDER            │  DESCRIPTION                                      ║
╠═══════════════════════════════════════════════════════════════════════╣
║  RANDOM           │  Shuffle segments randomly each epoch.            ║
║                   │  Best for: training (prevents order bias)         ║
╠═══════════════════════════════════════════════════════════════════════╣
║  CHRONOLOGICAL    │  Process segments in GPS time order.              ║
║                   │  Best for: time-series analysis, debug            ║
╠═══════════════════════════════════════════════════════════════════════╣
║  SHORTEST_FIRST   │  Start with shortest segments.                    ║
║                   │  Best for: quick testing, warm-up                 ║
╚═══════════════════════════════════════════════════════════════════════╝
""")

print("Example usage:")
print("""
    # Chronological for reproducible analysis
    obtainer = gf.IFODataObtainer(
        ...,
        segment_order=gf.SegmentOrder.CHRONOLOGICAL
    )
    
    # Shortest first for quick iteration during development
    obtainer = gf.IFODataObtainer(
        ...,
        segment_order=gf.SegmentOrder.SHORTEST_FIRST
    )
""")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 70)
print("SUMMARY: Advanced NOISE Mode Features")
print("=" * 70)

print("""
1. SAMPLING MODES:
   - SamplingMode.RANDOM: Best for training (default)
   - SamplingMode.GRID: Best for inference/coverage

2. TRAIN/VALIDATION SPLITTING:
   - Use 'groups' dict to define split ratios: {"train": 0.8, "validate": 0.2}
   - Use 'group' parameter to select which group to use
   - Splits are deterministic (hash-based on GPS times)

3. MULTI-DETECTOR:
   - Pass list of IFOs: ifos=[gf.IFO.L1, gf.IFO.H1, gf.IFO.V1]
   - Output shape: (Batch, NumIFOs, Samples)
   - Data is downloaded for each detector

4. SEGMENT ORDERING:
   - RANDOM: Shuffled (training)
   - CHRONOLOGICAL: Time-ordered (analysis)
   - SHORTEST_FIRST: Quick testing
""")

print("\n✓ Tutorial 03 complete!")
print("Next: 04_transient_mode_basics.py - Acquiring glitch-centered windows")
