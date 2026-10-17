#!/usr/bin/env python3
"""
=============================================================================
TUTORIAL 03: NOISE Mode - Advanced Features
=============================================================================

This tutorial covers advanced NOISE mode features including grid sampling,
train/validation splitting, and multi-detector acquisition.

LEARNING OBJECTIVES:
1. Understand SamplingMode: RANDOM vs GRID
2. Select a train, validate or test split
3. Acquire data from multiple detectors
4. Work with segment collections

RUNTIME: ~2-3 minutes
"""

# =============================================================================
# IMPORTS
# =============================================================================

import os
os.environ.setdefault("KERAS_BACKEND", "jax")  # gravyflow runs on Keras's JAX backend
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
║  SamplingMode.GRID   │  Step through each segment once, then end.    ║
║                      │  Good for: inference (each window once)       ║
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
One split, gf.SplitSpec, decides which of train, validate and test every
stretch of GPS time belongs to, for every route: 80% train, 10% validate and
10% test by default. The splits are disjoint and the same on every run,
whatever the seed or the observing runs.

How it works:
  1. GPS time is cut into absolute 8192 s blocks, counted from GPS 0
  2. A hash of each block's index decides its split
  3. 128 s at every edge between splits belongs to no split, so data of two
     splits is always at least 128 s apart

Name the split with the 'group' parameter on every call (it is required):
  noise(group="train"), noise(group="validate"), noise(group="test")
""")

split = gf.SplitSpec()  # the default: the obtainer takes it as split_spec
print(f"Split fractions: {dict(split.fractions)}, guard {split.guard_seconds} s")
print(f"Split fingerprint (each model trained on this split records it): {split.fingerprint()}")

# One obtainer serves every split: each call reads its split from a fresh copy.
obtainer = gf.IFODataObtainer(
    observing_runs=[gf.ObservingRun.O3],
    data_quality=gf.DataQuality.BEST,
    data_labels=[gf.DataLabel.NOISE],
    segment_order=gf.SegmentOrder.RANDOM,
    split_spec=split,
)

noise = gf.NoiseObtainer(
    ifo_data_obtainer=obtainer,
    ifos=[gf.IFO.L1],
)

print("Train batches:      noise(group='train')    (80% of segments)")
print("Validation batches: noise(group='validate') (10% of segments)")
print("Test batches:       noise(group='test')     (10% of segments)")

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
and the output shape becomes (Batch, NumIFOs, Samples). Each detector's
noise window is drawn at least two window footprints away from every other
detector's, so the noise in an example is never coincident across detectors.
Coincident noise is an opt-in: coincidence=gf.Coincidence.COINCIDENT draws
every detector at one time, and each batch says so (ReturnVariables.COINCIDENT).
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
   - One gf.SplitSpec: 80% train, 10% validate, 10% test, 128 s guard
   - Name the split with the 'group' parameter on every call
   - Splits are deterministic (absolute 8192 s GPS blocks, hashed)

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
