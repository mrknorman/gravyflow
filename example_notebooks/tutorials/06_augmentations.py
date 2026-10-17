#!/usr/bin/env python3
"""
=============================================================================
TUTORIAL 06: Data Augmentations
=============================================================================

This tutorial explains the dataclass-based augmentation system: each
augmentation is an object with its own probability and parameters, applied
per example to training data only.

LEARNING OBJECTIVES:
1. Understand the dataclass-based augmentation API
2. Configure each augmentation type independently
3. Know where augmentation applies, and where it never does
4. Read which augmentations each example had (AUGMENTATIONS)
5. See augmentation effects visually

RUNTIME: ~2-3 minutes (Section 4 needs network access to GWOSC)
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
# SECTION 1: THE AUGMENTATION API
# =============================================================================

print("=" * 70)
print("SECTION 1: Dataclass-Based Augmentation API")
print("=" * 70)

print("""
GravyFlow uses a flexible dataclass-based augmentation system.
Each augmentation is a separate object with its own parameters, given to the
IFO data obtainer:

    obtainer = gf.IFODataObtainer(
        augmentations=[
            gf.SignReversal(probability=0.5),
            gf.TimeReversal(probability=0.5),
            gf.RandomShift(probability=0.5, shift_fraction=0.2),
        ],
        # ... the other settings
    )

Where augmentation applies:
  - Nowhere by default: augmentations=None (or []) is none.
  - Training data only. A dataset applies its obtainer's augmentations to
    its group="train" batches; the validate and test splits, the
    validator and a GRID pass are never augmented, so one obtainer can
    serve every split. An obtainer called directly never augments.
  - Per example: each example draws each augmentation once, and the draw
    applies alike to its onsource and offsource, on every detector.
  - Before injection and whitening: an injected signal is never transformed.
  - The draws have their own random streams: adding or removing an
    augmentation never changes which windows are drawn.
""")

# =============================================================================
# SECTION 2: AVAILABLE AUGMENTATION TYPES
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 2: Available Augmentation Types")
print("=" * 70)

# --- SignReversal ---
print("\n2.1 SignReversal")
print("-" * 40)
print("""
Flips the sign of all data: x → -x

Physics justification:
  - A wave with both polarizations negated is the same source with its
    polarization angle turned by 90 degrees
  - Noise is symmetric under sign flip

Usage:
    gf.SignReversal(probability=0.5)  # 50% chance

Applies to: noise, glitches and real events
""")

# --- TimeReversal ---
print("\n2.2 TimeReversal")
print("-" * 40)
print("""
Reverses the time axis: x[t] → x[T-t]

Physics justification:
  - Stationary noise is time-reversible
  - Helps prevent overfitting to time-ordering

Usage:
    gf.TimeReversal(probability=0.5)  # 50% chance

Applies to: noise and glitches. Refused with real events: a chirp reversed
is no longer a chirp.
""")

# --- RandomShift ---
print("\n2.3 RandomShift (glitches only)")
print("-" * 40)
print("""
Rolls both windows by a random number of samples, so the glitch is
off-center. Samples rolled out of one end come back at the other.

Parameters:
  - probability: Chance of applying shift
  - shift_fraction: Max shift as fraction of the window, in [0, 0.5]
    (0.25 = ±25%)

Usage:
    gf.RandomShift(probability=0.5, shift_fraction=0.25)

Applies to: glitches. Refused with real events, and with noise, whose
windows are drawn uniformly in time already.
""")

# --- AddNoise ---
print("\n2.4 AddNoise")
print("-" * 40)
print("""
Adds Gaussian noise to the data.

Physics justification:
  - Helps prevent overfitting to specific noise realizations
  - Cached transients always have same noise background

Parameters:
  - probability: Chance of adding noise
  - amplitude: Noise level as a fraction of the example's own offsource
    std, per detector (0.1 = 10%); the same level on both windows

Usage:
    gf.AddNoise(probability=0.5, amplitude=0.1)

Applies to: noise, glitches and real events
""")

# =============================================================================
# SECTION 3: CONFIGURING AUGMENTATIONS
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 3: Configuring Augmentations")
print("=" * 70)

# Example configurations
print("Example configurations:")
print()

# Training with all augmentations
print("1. Glitches, strong augmentation:")
glitch_augs = [
    gf.SignReversal(probability=0.5),
    gf.TimeReversal(probability=0.5),
    gf.RandomShift(probability=0.5, shift_fraction=0.25),
    gf.AddNoise(probability=0.5, amplitude=0.1),
]
print(f"   augmentations={glitch_augs}")
print()

# Conservative augmentation
print("2. Noise, conservative augmentation (low probability):")
noise_augs = [
    gf.SignReversal(probability=0.2),
    gf.TimeReversal(probability=0.2),
]
print(f"   augmentations={noise_augs}")
print()

# Refused combinations
print("3. Refused at configuration:")
for labels, augmentation in (([gf.DataLabel.EVENTS], gf.TimeReversal()),
                             ([gf.DataLabel.NOISE], gf.RandomShift())):
    try:
        gf.IFODataObtainer(
            data_quality=gf.DataQuality.BEST,
            data_labels=labels,
            augmentations=[augmentation],
        )
    except ValueError as error:
        print(f"   {labels[0].name} with {type(augmentation).__name__}: {error}")
print()

print("4. No augmentation (the default):")
print(f"   augmentations={gf.IFODataObtainer(data_quality=gf.DataQuality.BEST, data_labels=[gf.DataLabel.NOISE]).augmentations}")

# =============================================================================
# SECTION 4: DEMONSTRATION WITH LIVE DATA
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 4: Demonstration with Live Data")
print("=" * 70)

print("Creating one obtainer for both splits...")

obtainer = gf.IFODataObtainer(
    observing_runs=[gf.ObservingRun.O3],
    data_quality=gf.DataQuality.BEST,
    data_labels=[gf.DataLabel.GLITCHES],
    augmentations=[
        gf.SignReversal(probability=0.5),
        gf.TimeReversal(probability=0.5),
    ],
)

dataset_settings = dict(
    noise_obtainer=gf.NoiseObtainer(ifo_data_obtainer=obtainer, ifos=[gf.IFO.L1]),
    sample_rate_hertz=2048.0,
    onsource_duration_seconds=1.0,
    offsource_duration_seconds=16.0,
    num_examples_per_batch=8,
    seed=42,
    input_variables=[gf.ReturnVariables.ONSOURCE],
    # Which augmentations each example had: SignReversal adds 1, TimeReversal 2.
    output_variables=[gf.ReturnVariables.AUGMENTATIONS],
)

print("Fetching a training batch and a validation batch...")
train_x, train_y = gf.Dataset(group="train", **dataset_settings)[0]
validate_x, validate_y = gf.Dataset(group="validate", **dataset_settings)[0]

print(f"  train AUGMENTATIONS:    {np.asarray(train_y['AUGMENTATIONS']).astype(int)}")
print(f"  validate AUGMENTATIONS: {np.asarray(validate_y['AUGMENTATIONS']).astype(int)}  (never augmented)")

# =============================================================================
# SECTION 5: VISUALIZING AUGMENTATION EFFECTS
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 5: Visualizing Augmentation Effects")
print("=" * 70)

# A validation example is the strain as recorded.
onsource = np.asarray(validate_x["ONSOURCE"])

# Create demonstration of transformations on sample data
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
fig.suptitle("Augmentation Effects Demonstration", fontsize=14)

# Use first sample
sample = onsource[0, 0, :]
time = np.arange(len(sample)) / 2048.0

# Original
axes[0, 0].plot(time, sample, 'b-', lw=0.5)
axes[0, 0].set_title("Original Data")
axes[0, 0].set_xlabel("Time (s)")
axes[0, 0].set_ylabel("Strain")
axes[0, 0].grid(True, alpha=0.3)

# Sign reversal
axes[0, 1].plot(time, -sample, 'r-', lw=0.5)
axes[0, 1].set_title("SignReversal: x → -x")
axes[0, 1].set_xlabel("Time (s)")
axes[0, 1].set_ylabel("Strain")
axes[0, 1].grid(True, alpha=0.3)

# Time reversal
axes[0, 2].plot(time, sample[::-1], 'g-', lw=0.5)
axes[0, 2].set_title("TimeReversal: x[t] → x[T-t]")
axes[0, 2].set_xlabel("Time (s)")
axes[0, 2].set_ylabel("Strain")
axes[0, 2].grid(True, alpha=0.3)

# Random shift (simulate)
shift_samples = int(0.2 * len(sample))  # 20% shift
shifted = np.roll(sample, shift_samples)
axes[1, 0].plot(time, shifted, 'm-', lw=0.5)
axes[1, 0].axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='Original center')
axes[1, 0].axvline(x=0.5 + shift_samples/2048.0, color='m', linestyle='--', alpha=0.5, label='Shifted center')
axes[1, 0].set_title("RandomShift: Shift off-center")
axes[1, 0].set_xlabel("Time (s)")
axes[1, 0].set_ylabel("Strain")
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].legend(fontsize=8)

# Add noise (simulate)
noise = np.random.normal(0, 0.1 * np.std(sample, dtype=np.float64), len(sample))  # float64: strain squared underflows float32
axes[1, 1].plot(time, sample + noise, 'c-', lw=0.5)
axes[1, 1].set_title("AddNoise: Small perturbation")
axes[1, 1].set_xlabel("Time (s)")
axes[1, 1].set_ylabel("Strain")
axes[1, 1].grid(True, alpha=0.3)

# Combined (Sign + Time reversal)
combined = -sample[::-1]
axes[1, 2].plot(time, combined, 'orange', lw=0.5)
axes[1, 2].set_title("Combined: Sign + Time Reversal")
axes[1, 2].set_xlabel("Time (s)")
axes[1, 2].set_ylabel("Strain")
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plot_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "06_augmentations.png")
plt.savefig(plot_path, dpi=100)
print(f"Saved plot to: {plot_path}")
plt.close()

# =============================================================================
# SECTION 6: BEST PRACTICES
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 6: Best Practices")
print("=" * 70)

print("""
1. TRAINING vs VALIDATION:
   - Give the augmentations to the obtainer; the training dataset applies
     them, and validation, test and the validator never do
   - One obtainer can serve every split

2. PROBABILITY CHOICES:
   - 0.5 is a good default (symmetric exploration)
   - Lower (0.2-0.3) for more conservative augmentation
   - 1.0 only for debugging/visualization

3. WHAT EACH DATA TYPE ACCEPTS:
   - Noise: SignReversal, TimeReversal, AddNoise
   - Glitches: all four
   - Real events: SignReversal, AddNoise

4. READING A BATCH:
   - AUGMENTATIONS says which augmentations each example had
     (SignReversal 1, TimeReversal 2, RandomShift 4, AddNoise 8, summed)
   - START_GPS_TIME is still the start of the window read

5. COMBINING WITH CACHING:
   - Augmentations are applied AFTER cache retrieval
   - Same cached data produces different augmented versions
""")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 70)
print("SUMMARY: Data Augmentations")
print("=" * 70)

print("""
1. DATACLASS-BASED API:
   - Each augmentation is a separate object
   - Per-augmentation probability and parameters

2. FOUR AUGMENTATION TYPES:
   - SignReversal(probability): x → -x
   - TimeReversal(probability): x[t] → x[T-t]; never real events
   - RandomShift(probability, shift_fraction): glitches only
   - AddNoise(probability, amplitude): relative to each example's offsource

3. WHERE THEY APPLY:
   - None by default
   - Training data only, per example, before injection and whitening
   - AUGMENTATIONS flags what each example had

4. BEST PRACTICES:
   - Use 0.5 probability as default
   - Different augments for different data
   - One obtainer for every split
""")

print("\n✓ Tutorial 06 complete!")
print("Next: 07_caching_performance.py - Caching and performance optimization")
