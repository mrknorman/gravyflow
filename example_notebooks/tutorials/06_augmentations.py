#!/usr/bin/env python3
"""
=============================================================================
TUTORIAL 06: Data Augmentations
=============================================================================

This tutorial explains the dataclass-based augmentation system, which allows
flexible per-augmentation configuration.

LEARNING OBJECTIVES:
1. Understand the new dataclass-based augmentation API
2. Configure each augmentation type independently
3. See augmentation effects visually
4. Learn mode-specific augmentations

RUNTIME: ~2-3 minutes
"""

# =============================================================================
# IMPORTS
# =============================================================================

import gravyflow as gf
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# SECTION 1: THE NEW AUGMENTATION API
# =============================================================================

print("=" * 70)
print("SECTION 1: Dataclass-Based Augmentation API")
print("=" * 70)

print("""
GravyFlow uses a flexible dataclass-based augmentation system.
Each augmentation is a separate object with its own parameters.

OLD API (deprecated):
    obtainer = gf.IFODataObtainer(
        random_sign_reversal=True,    # Boolean flags
        random_time_reversal=True,
        random_shift=True,
        shift_fraction=0.2,
        ...
    )

NEW API:
    obtainer = gf.IFODataObtainer(
        augmentations=[
            gf.SignReversal(probability=0.5),
            gf.TimeReversal(probability=0.5),
            gf.RandomShift(probability=0.5, shift_fraction=0.2),
        ],
        ...
    )

Benefits of new API:
  ✓ Per-augmentation probability control
  ✓ Clearer parameter organization
  ✓ Easier to add custom augmentations
  ✓ No augmentations = empty list []
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
  - GW signals have + and × polarizations
  - Both are physically valid, just different orientation
  - Noise is symmetric under sign flip

Usage:
    gf.SignReversal(probability=0.5)  # 50% chance

Applicable modes: NOISE, TRANSIENT
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

Applicable modes: NOISE, TRANSIENT
""")

# --- RandomShift ---
print("\n2.3 RandomShift (TRANSIENT mode only)")
print("-" * 40)
print("""
Shifts the event off-center by a random amount.

Physics justification:
  - Events aren't always perfectly centered in real detections
  - Prevents model from assuming event is always at t=0

Parameters:
  - probability: Chance of applying shift
  - shift_fraction: Max shift as fraction of window (0.25 = ±25%)

Usage:
    gf.RandomShift(probability=0.5, shift_fraction=0.25)

Applicable modes: TRANSIENT only
""")

# --- AddNoise ---
print("\n2.4 AddNoise (TRANSIENT mode only)")
print("-" * 40)
print("""
Adds small random noise perturbation to the data.

Physics justification:
  - Helps prevent overfitting to specific noise realizations
  - Cached transients always have same noise background

Parameters:
  - probability: Chance of adding noise
  - amplitude: Noise level as fraction of data std (0.1 = 10%)

Usage:
    gf.AddNoise(probability=0.5, amplitude=0.1)

Applicable modes: TRANSIENT only
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
print("1. Training with strong augmentation:")
train_augs = [
    gf.SignReversal(probability=0.5),
    gf.TimeReversal(probability=0.5),
    gf.RandomShift(probability=0.5, shift_fraction=0.25),
    gf.AddNoise(probability=0.5, amplitude=0.1),
]
print(f"   augmentations={train_augs}")
print()

# Validation with no augmentation
print("2. Validation with no augmentation:")
val_augs = []  # Empty list = no augmentations
print(f"   augmentations={val_augs}")
print()

# Conservative augmentation
print("3. Conservative augmentation (low probability):")
conservative_augs = [
    gf.SignReversal(probability=0.2),
    gf.TimeReversal(probability=0.2),
]
print(f"   augmentations={conservative_augs}")
print()

# Default augmentations
print("4. Default augmentations (if None or not specified):")
default_augs = gf.default_augmentations()
print(f"   default_augmentations() = {default_augs}")

# =============================================================================
# SECTION 4: DEMONSTRATION WITH LIVE DATA
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 4: Demonstration with Live Data")
print("=" * 70)

print("Creating obtainers with different augmentation settings...")

# No augmentation obtainer
no_aug_obtainer = gf.IFODataObtainer(
    observing_runs=[gf.ObservingRun.O3],
    data_quality=gf.DataQuality.BEST,
    data_labels=[gf.DataLabel.GLITCHES],
    augmentations=[],  # No augmentations
)

# Full augmentation obtainer
full_aug_obtainer = gf.IFODataObtainer(
    observing_runs=[gf.ObservingRun.O3],
    data_quality=gf.DataQuality.BEST,
    data_labels=[gf.DataLabel.GLITCHES],
    augmentations=[
        gf.SignReversal(probability=1.0),  # Always apply for demonstration
        gf.TimeReversal(probability=1.0),
    ],
)

print("Fetching data with and without augmentations...")

# Get data without augmentations
no_aug_transient = gf.NoiseObtainer(
    ifo_data_obtainer=no_aug_obtainer,
    ifos=[gf.IFO.L1],
)

data_gen_no_aug = no_aug_transient(
    sample_rate_hertz=2048.0,
    onsource_duration_seconds=1.0,
    offsource_duration_seconds=16.0,
    num_examples_per_batch=4,
    seed=42,
)
batch_no_aug = next(data_gen_no_aug)

# Get data with augmentations
full_aug_transient = gf.NoiseObtainer(
    ifo_data_obtainer=full_aug_obtainer,
    ifos=[gf.IFO.L1],
)

data_gen_full_aug = full_aug_transient(
    sample_rate_hertz=2048.0,
    onsource_duration_seconds=1.0,
    offsource_duration_seconds=16.0,
    num_examples_per_batch=4,
    seed=42,
)
batch_full_aug = next(data_gen_full_aug)

# =============================================================================
# SECTION 5: VISUALIZING AUGMENTATION EFFECTS
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 5: Visualizing Augmentation Effects")
print("=" * 70)

onsource_no_aug = batch_no_aug[gf.ReturnVariables.ONSOURCE]
onsource_full_aug = batch_full_aug[gf.ReturnVariables.ONSOURCE]

# Create demonstration of transformations on sample data
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
fig.suptitle("Augmentation Effects Demonstration", fontsize=14)

# Use first sample
sample = onsource_no_aug[0, 0, :]
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
noise = np.random.normal(0, 0.1 * np.std(sample), len(sample))
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
plt.savefig("/home/michael.norman/gravyflow/example_notebooks/tutorials/06_augmentations.png", dpi=100)
print("Saved plot to: tutorials/06_augmentations.png")
plt.close()

# =============================================================================
# SECTION 6: BEST PRACTICES
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 6: Best Practices")
print("=" * 70)

print("""
1. TRAINING vs VALIDATION:
   - Training: Use augmentations to increase diversity
   - Validation: Use augmentations=[] for consistent evaluation

2. PROBABILITY CHOICES:
   - 0.5 is a good default (symmetric exploration)
   - Lower (0.2-0.3) for more conservative augmentation
   - 1.0 only for debugging/visualization

3. MODE-SPECIFIC AUGMENTATIONS:
   - NOISE mode: SignReversal, TimeReversal work well
   - TRANSIENT mode: Add RandomShift, AddNoise for robustness

4. WHEN NOT TO AUGMENT:
   - Final inference on real data
   - When analyzing specific events
   - When comparing to published results

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
1. NEW DATACLASS-BASED API:
   - Each augmentation is a separate object
   - Per-augmentation probability and parameters

2. FOUR AUGMENTATION TYPES:
   - SignReversal(probability): x → -x
   - TimeReversal(probability): x[t] → x[T-t]
   - RandomShift(probability, shift_fraction): TRANSIENT only
   - AddNoise(probability, amplitude): TRANSIENT only

3. CONFIGURATION:
   - Training: augmentations=[gf.SignReversal(0.5), ...]
   - Validation: augmentations=[]
   - Default: gf.default_augmentations()

4. BEST PRACTICES:
   - Use 0.5 probability as default
   - Different augments for different modes
   - Always disable for final inference
""")

print("\n✓ Tutorial 06 complete!")
print("Next: 07_caching_performance.py - Caching and performance optimization")
