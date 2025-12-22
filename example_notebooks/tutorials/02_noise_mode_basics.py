#!/usr/bin/env python3
"""
=============================================================================
TUTORIAL 02: NOISE Mode - Basic Usage
=============================================================================

This tutorial demonstrates basic NOISE mode acquisition - sampling random
windows from long stretches of detector data.

LEARNING OBJECTIVES:
1. Create a NOISE mode data obtainer
2. Understand key parameters (sample rate, durations)
3. Acquire data and understand output shapes
4. Visualize noise samples

RUNTIME: ~2-3 minutes (downloads ~1 segment)

NOTE: This requires network access to GWOSC (Gravitational Wave Open Science).
"""

# =============================================================================
# IMPORTS
# =============================================================================

import gravyflow as gf
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# SECTION 1: CREATING A NOISE MODE OBTAINER
# =============================================================================

print("=" * 70)
print("SECTION 1: Creating a NOISE Mode Obtainer")
print("=" * 70)

# The key parameter that selects NOISE mode is data_labels=[gf.DataLabel.NOISE]
#
# Key parameters explained:
#   - observing_runs: Which LIGO/Virgo science runs to use (O1, O2, O3, O4)
#   - data_quality: BEST = calibrated science data (almost always use this)
#   - data_labels: What to acquire - NOISE means random sampling from segments
#   - segment_order: How to iterate through available segments

obtainer = gf.IFODataObtainer(
    observing_runs=[gf.ObservingRun.O3],  # Use O3 data (2019-2020)
    data_quality=gf.DataQuality.BEST,      # Calibrated science-quality
    data_labels=[gf.DataLabel.NOISE],      # ← NOISE mode
    segment_order=gf.SegmentOrder.RANDOM,  # Shuffle segments
)

print(f"Created obtainer: {type(obtainer).__name__}")
print(f"Mode: NOISE (random sampling from long segments)")

# =============================================================================
# SECTION 2: KEY ACQUISITION PARAMETERS
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 2: Understanding Key Parameters")
print("=" * 70)

# When acquiring data, you need to specify several key parameters:

# Sample rate: How many samples per second (Hz)
# Common values: 2048 Hz (analysis), 4096 Hz (full resolution), 16384 Hz (raw)
SAMPLE_RATE_HERTZ = 2048.0

# Onsource duration: The "main" window of interest
# This is the data you'll analyze or feed to your model
ONSOURCE_DURATION_SECONDS = 1.0

# Offsource duration: Background window for noise estimation / whitening
# Must be at least 8x the onsource for stable PSD estimation
OFFSOURCE_DURATION_SECONDS = 16.0

# Batch size: How many examples to acquire at once
NUM_EXAMPLES = 4

print(f"Sample rate:        {SAMPLE_RATE_HERTZ:.0f} Hz")
print(f"Onsource duration:  {ONSOURCE_DURATION_SECONDS:.1f}s → {int(SAMPLE_RATE_HERTZ * ONSOURCE_DURATION_SECONDS)} samples")
print(f"Offsource duration: {OFFSOURCE_DURATION_SECONDS:.1f}s → {int(SAMPLE_RATE_HERTZ * OFFSOURCE_DURATION_SECONDS)} samples")
print(f"Batch size:         {NUM_EXAMPLES} examples")

# =============================================================================
# SECTION 3: ACQUIRING DATA
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 3: Acquiring Data")
print("=" * 70)

# The get_onsource_offsource_chunks method returns a generator that yields
# batches of (onsource, offsource) data along with metadata.
#
# Key concept: We use gf.NoiseObtainer as a wrapper that handles the 
# iteration and returns structured results from an IFO data obtainer.

print("Setting up data acquisition...")
print("(This will download data from GWOSC - may take 1-2 minutes)")

# Create a NoiseObtainer that wraps our IFO data obtainer
# This provides a standardized interface regardless of mode
noise_obtainer = gf.NoiseObtainer(
    ifo_data_obtainer=obtainer,
    ifos=[gf.IFO.L1],  # Use Livingston detector
)

# Get the data generator by calling the NoiseObtainer
data_gen = noise_obtainer(
    sample_rate_hertz=SAMPLE_RATE_HERTZ,
    onsource_duration_seconds=ONSOURCE_DURATION_SECONDS,
    offsource_duration_seconds=OFFSOURCE_DURATION_SECONDS,
    num_examples_per_batch=NUM_EXAMPLES,
)

# Fetch one batch
print("\nFetching first batch...")
batch = next(data_gen)

# The batch is a dictionary with ReturnVariables keys
print(f"\nBatch keys: {list(batch.keys())}")

# =============================================================================
# SECTION 4: UNDERSTANDING OUTPUT SHAPES
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 4: Understanding Output Shapes (BIS Format)")
print("=" * 70)

# GravyFlow uses BIS format: (Batch, IFO, Samples)
# This is consistent across all acquisition modes

onsource = batch[gf.ReturnVariables.ONSOURCE]
offsource = batch[gf.ReturnVariables.OFFSOURCE]

print(f"""
Output tensors are in BIS format: (Batch, IFO, Samples)

Onsource shape:  {onsource.shape}
  - Batch:   {onsource.shape[0]} examples
  - IFO:     {onsource.shape[1]} detector(s)
  - Samples: {onsource.shape[2]} samples ({onsource.shape[2]/SAMPLE_RATE_HERTZ:.1f}s)

Offsource shape: {offsource.shape}
  - Batch:   {offsource.shape[0]} examples
  - IFO:     {offsource.shape[1]} detector(s)
  - Samples: {offsource.shape[2]} samples ({offsource.shape[2]/SAMPLE_RATE_HERTZ:.1f}s)
""")

# GPS times for each sample
if gf.ReturnVariables.START_GPS_TIME in batch:
    gps_times = batch[gf.ReturnVariables.START_GPS_TIME]
    print(f"GPS times (start of each window):")
    for i, t in enumerate(gps_times):
        print(f"  Example {i}: GPS {t:.2f}")

# =============================================================================
# SECTION 5: VISUALIZING NOISE SAMPLES
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 5: Visualizing Noise Samples")
print("=" * 70)

# Create time arrays
onsource_time = np.arange(onsource.shape[2]) / SAMPLE_RATE_HERTZ
offsource_time = np.arange(offsource.shape[2]) / SAMPLE_RATE_HERTZ

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("NOISE Mode: Random Samples from O3 Data (L1)", fontsize=14)

for i in range(min(4, onsource.shape[0])):
    ax = axes[i // 2, i % 2]
    
    # Plot onsource
    ax.plot(onsource_time, onsource[i, 0, :], 'b-', alpha=0.7, lw=0.5)
    
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Strain")
    ax.set_title(f"Sample {i+1}")
    ax.grid(True, alpha=0.3)
    
    # Add RMS annotation
    rms = np.sqrt(np.mean(onsource[i, 0, :]**2))
    ax.text(0.02, 0.98, f"RMS: {rms:.2e}", transform=ax.transAxes, 
            va='top', fontsize=9, family='monospace')

plt.tight_layout()
plt.savefig("/home/michael.norman/gravyflow/example_notebooks/tutorials/02_noise_samples.png", dpi=100)
print("Saved plot to: tutorials/02_noise_samples.png")
plt.close()

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 70)
print("SUMMARY: Key Takeaways")
print("=" * 70)

print("""
1. NOISE mode is selected with data_labels=[gf.DataLabel.NOISE]

2. Key parameters:
   - sample_rate_hertz: Typically 2048 or 4096 Hz
   - onsource_duration_seconds: Your analysis window
   - offsource_duration_seconds: Background for PSD (≥8x onsource)

3. Output format is BIS: (Batch, IFO, Samples)

4. Each batch contains:
   - ONSOURCE: Main data window
   - OFFSOURCE: Background for whitening/PSD
   - START_GPS_TIME: Start time of each window

5. Data is downloaded from GWOSC automatically
""")

print("\n✓ Tutorial 02 complete!")
print("Next: 03_noise_mode_advanced.py - Grid sampling, groups, multi-IFO")
