#!/usr/bin/env python3
"""
=============================================================================
TUTORIAL 07: Caching and Performance
=============================================================================

This tutorial covers caching mechanisms and performance optimization 
strategies for the gravyflow acquisition pipeline.

LEARNING OBJECTIVES:
1. Understand the strain store, where real noise is kept
2. Learn about TransientCache for glitches/events
3. Bound memory with the store's cap
4. Best practices for large-scale training

RUNTIME: ~1 minute (mostly informational)
"""

# =============================================================================
# IMPORTS
# =============================================================================

import os
os.environ.setdefault("KERAS_BACKEND", "jax")  # gravyflow runs on Keras's JAX backend
import gravyflow as gf
from pathlib import Path
import os

# =============================================================================
# SECTION 1: CACHING OVERVIEW
# =============================================================================

print("=" * 70)
print("SECTION 1: Caching Overview")
print("=" * 70)

print("""
GravyFlow keeps what it downloads, so each piece of strain is fetched once:

┌─────────────────────────────────────────────────────────────────────────┐
│                          WHAT IS KEPT, AND WHERE                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   THE STRAIN STORE (real noise)                                         │
│   ├── The source's own files (GWOSC: 4096 s each), raw, at the rate     │
│   │   the source publishes, each written once and never changed         │
│   ├── Read as memory-mapped views: no process holds a whole file        │
│   └── Shared by every rate, split and configuration of one source       │
│                                                                         │
│   TRANSIENT CACHE (glitches and events)                                 │
│   ├── TransientCache HDF5 files                                         │
│   └── Survives restarts                                                 │
│                                                                         │
│   THE SOURCE (network, slowest)                                         │
│   └── GWOSC by default; fetched only what is not yet kept               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

The operating system's page cache is the only memory cache of the store.
""")

# =============================================================================
# SECTION 2: THE STRAIN STORE
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 2: The Strain Store (real noise)")
print("=" * 70)

print("""
Every read of real noise goes through the store:

  1. the source's files that the read needs are fetched whole, once, and kept
     under ~/.gravyflow/cache/store/ (GRAVYFLOW_CACHE_DIR moves it);
  2. the read is a view of them, resampled by gravyflow's one resampler
     to the sample rate asked for;
  3. the resampler's margin at each end is cut away, so every sample served
     is exact.

There is nothing to switch on or off. A second pass, another sample rate,
another split, or another dataset over the same source reads only the disk.

The source's timeline is the authority on which time holds data. When it
says a time holds data and the source cannot deliver it (an outage, a missing
file, non-finite samples), the read is refused by name, never zero-filled:

    gf.StrainUnavailable: ... to plan around it, add
    exclusions=(('H1', (1238170000.0, 1238174096.0), '...'),) to the obtainer.

Pass that entry as the obtainer's exclusions= to plan around the time:

    obtainer = gf.IFODataObtainer(
        observing_runs=gf.ObservingRun.O3,
        data_quality=gf.DataQuality.BEST,
        data_labels=[gf.DataLabel.NOISE],
        exclusions=[("H1", (1238170000.0, 1238174096.0), "GWOSC outage")],
    )

Exclusions are part of the obtainer's fingerprint, so a run made with them is
told apart from one made without.

Segment caches of earlier gravyflow versions (~/.gravyflow/cache/segments/)
are never read; gravyflow names them once, with their sizes. Delete them to
reclaim the space.
""")

# =============================================================================
# SECTION 3: TRANSIENT CACHE
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 3: TransientCache (TRANSIENT mode only)")
print("=" * 70)

print("""
TransientCache stores pre-extracted glitch/event windows to avoid
re-downloading the same events repeatedly.

Cache structure:
    ~/.gravyflow/cache/transient/
    ├── L1_2048Hz_1.0s_16.0s_O3.h5
    ├── H1_2048Hz_1.0s_16.0s_O3.h5
    └── ...

File naming: {IFO}_{sample_rate}Hz_{onsource}s_{offsource}s_{runs}.h5

Each HDF5 file contains:
  - Onsource windows indexed by GPS time
  - Offsource windows
  - Metadata (glitch type, event name, etc.)

Cache benefits:
  ✓ 10-100x faster than re-downloading
  ✓ Consistent data across runs
  ✓ Enables offline training
""")

# =============================================================================
# SECTION 4: MEMORY
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 4: Memory and the Store's Cap")
print("=" * 70)

print("""
Stored files are memory-mapped and never held whole, so what a process holds
is what is in flight: the downloads being decoded and the windows being
resampled. One accountant per process bounds them:

    gf.Defaults.memory_bytes = 2 * 1024**3   # the default: 2 GiB per process

A download or read waits until its bytes fit; one larger than the whole cap
is refused before anything is fetched. Each worker process has a cap of its
own, so N workers use up to N times it.
""")

# =============================================================================
# SECTION 5: PRE-CACHING FOR LARGE RUNS
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 5: Pre-Caching for Large Training Runs")
print("=" * 70)

print("""
For large training runs, pre-populate the cache before training. The
cache keeps every window a dataset draws, so draw the dataset you will
train on, with the same settings:

Example script:

    # Pre-cache the glitch windows of one pass over the training data
    train_data = gf.Dataset(
        noise_obtainer=gf.NoiseObtainer(
            ifo_data_obtainer=gf.IFODataObtainer(
                observing_runs=[gf.ObservingRun.O3],
                data_quality=gf.DataQuality.BEST,
                data_labels=[gf.DataLabel.GLITCHES],
            ),
            ifos=[gf.IFO.L1, gf.IFO.H1],
        ),
        group="train",
        sample_rate_hertz=2048.0,
        onsource_duration_seconds=1.0,
        offsource_duration_seconds=16.0,
        input_variables=[gf.ReturnVariables.ONSOURCE],
        steps_per_epoch=1000,
        num_examples_per_batch=32,
    )
    for k in range(len(train_data)):
        train_data[k]

One pass draws len(train_data) = steps_per_epoch batches of
num_examples_per_batch windows, 32,000 here (the defaults), and caches
those: a split with more glitches than one pass draws is cached only in
part. A larger steps_per_epoch draws, and caches, more of them.

Run this ONCE before training:
  1. Downloads the windows it draws
  2. Stores them in TransientCache
  3. Training reads those windows from the cache (fast!)

Recommended workflow:
  ┌───────────────┐    ┌──────────────┐    ┌────────────────┐
  │  Pre-cache    │ →  │   Train      │ →  │   Evaluate     │
  │  (1-2 hours)  │    │   (hours)    │    │                │
  └───────────────┘    └──────────────┘    └────────────────┘
""")

# =============================================================================
# SECTION 6: CACHE MANAGEMENT
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 6: Cache Management")
print("=" * 70)

# Show cache locations (if they exist)
gravyflow_cache = Path.home() / ".gravyflow" / "cache"

print("Cache locations:")
print(f"  Base: {gravyflow_cache}")
print()

if gravyflow_cache.exists():
    # List cache directories
    for item in sorted(gravyflow_cache.iterdir()):
        if item.is_dir():
            # Count files and total size
            files = list(item.glob("**/*"))
            file_count = len([f for f in files if f.is_file()])
            total_size = sum(f.stat().st_size for f in files if f.is_file())
            print(f"  {item.name}/")
            print(f"    Files: {file_count}")
            print(f"    Size:  {total_size / 1e6:.1f} MB")
else:
    print("  (Cache directory not yet created)")
    print("  Cache is created on first data acquisition")

print("""
Clearing cache:
  # Clear all cache (careful!), when no gravyflow process is running
  rm -rf ~/.gravyflow/cache/

  # Clear only transient cache
  rm -rf ~/.gravyflow/cache/transient/

  # Old segment caches, which this gravyflow never reads
  rm -rf ~/.gravyflow/cache/segments/

  # Stored strain: forget the files of one source, detector and interval
  # (gf.src.dataset.store.Store.forget), or delete store/ when nothing reads it
""")

# =============================================================================
# SECTION 7: PERFORMANCE TIPS
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 7: Performance Tips")
print("=" * 70)

print("""
╔═══════════════════════════════════════════════════════════════════════╗
║  TIP                           │  DESCRIPTION                         ║
╠═══════════════════════════════════════════════════════════════════════╣
║  1. Pre-cache before training  │  Avoid network during training       ║
╠═══════════════════════════════════════════════════════════════════════╣
║  2. Size the memory cap        │  gf.Defaults.memory_bytes (2 GiB)   ║
╠═══════════════════════════════════════════════════════════════════════╣
║  3. Match cache parameters     │  Same sample_rate, durations         ║
╠═══════════════════════════════════════════════════════════════════════╣
║  4. Use SSD for cache          │  HDF5 reads benefit from fast I/O   ║
╠═══════════════════════════════════════════════════════════════════════╣
║  5. Multi-IFO single request   │  Download all IFOs together         ║
╠═══════════════════════════════════════════════════════════════════════╣
║  6. Batch size ≥ 16            │  Better GPU utilization              ║
╠═══════════════════════════════════════════════════════════════════════╣
║  7. Use group= for splits      │  Deterministic train/val without    ║
║                                │  separate cache files                ║
╚═══════════════════════════════════════════════════════════════════════╝
""")

# =============================================================================
# SECTION 8: EXAMPLE CONFIGURATION
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 8: Example Production Configuration")
print("=" * 70)

print("""
# Production training configuration with optimal caching

# One obtainer for every split
obtainer = gf.IFODataObtainer(
    observing_runs=[gf.ObservingRun.O3],
    data_quality=gf.DataQuality.BEST,
    data_labels=[gf.DataLabel.GLITCHES],
    segment_order=gf.SegmentOrder.RANDOM,
    balanced_glitch_types=True, # ← Class balancing
    augmentations=[             # ← Applied to the training split only
        gf.SignReversal(probability=0.5),
        gf.TimeReversal(probability=0.5),
        gf.RandomShift(probability=0.5, shift_fraction=0.2),
        gf.AddNoise(probability=0.5, amplitude=0.1),
    ],
)

# One dataset per split, the obtainer wrapped in a NoiseObtainer; every route
# reads the data obtainer's split, gf.SplitSpec (the default here)
dataset_settings = dict(
    sample_rate_hertz=2048.0,
    onsource_duration_seconds=1.0,
    offsource_duration_seconds=16.0,
    input_variables=[gf.ReturnVariables.ONSOURCE],
)
train_data = gf.Dataset(
    noise_obtainer=gf.NoiseObtainer(ifo_data_obtainer=obtainer, ifos=[gf.IFO.L1, gf.IFO.H1]),
    group="train",
    **dataset_settings,
)
val_data = gf.Dataset(
    noise_obtainer=gf.NoiseObtainer(ifo_data_obtainer=obtainer, ifos=[gf.IFO.L1, gf.IFO.H1]),
    group="validate",
    **dataset_settings,
)
""")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 70)
print("SUMMARY: Caching and Performance")
print("=" * 70)

print("""
1. WHAT IS KEPT:
   The strain store (real noise), the transient cache, then the source

2. THE STRAIN STORE:
   - The source's own files, raw, fetched once, read as views
   - Refusals name the exclusions entry that plans around them

3. TRANSIENT CACHE:
   - Stores pre-extracted windows
   - HDF5 files in ~/.gravyflow/cache/transient/

4. MEMORY:
   - gf.Defaults.memory_bytes: 2 GiB per process by default

5. PRE-CACHING:
   - Draw the training dataset before large runs: each pass caches
     the steps_per_epoch x num_examples_per_batch windows it draws
   - One-time cost, major training speedup

6. CACHE MANAGEMENT:
   - ~/.gravyflow/cache/
   - Clear with rm -rf if needed
""")

print("\n" + "=" * 70)
print("TUTORIAL SERIES COMPLETE!")
print("=" * 70)

print("""
You have completed all 7 tutorials in the acquisition pipeline series:

  01_introduction.py           ✓ Core concepts and enums
  02_noise_mode_basics.py      ✓ Basic NOISE acquisition  
  03_noise_mode_advanced.py    ✓ Grid sampling, splits, multi-IFO
  04_transient_mode_basics.py  ✓ Glitch-centered acquisition
  05_transient_mode_advanced.py ✓ Events, balancing, caching
  06_augmentations.py          ✓ Dataclass-based augmentations
  07_caching_performance.py    ✓ Caching and optimization

Next steps:
  - Try glitch_classification_example.py for a full training example
  - Explore the injection system for signal generation
  - Build your own models using gf.Dataset

Happy gravitational wave hunting! 🌌
""")

print("\n✓ Tutorial 07 complete!")
