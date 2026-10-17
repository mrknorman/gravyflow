#!/usr/bin/env python3
"""
=============================================================================
TUTORIAL 07: Caching and Performance
=============================================================================

This tutorial covers caching mechanisms and performance optimization 
strategies for the gravyflow acquisition pipeline.

LEARNING OBJECTIVES:
1. Understand the strain store, where all real strain is kept
2. See how glitches and events read the same store
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
│   THE STRAIN STORE (real noise, glitches, events and partners)          │
│   ├── The source's own files (GWOSC: 4096 s each), raw, at the rate     │
│   │   the source publishes, each written once and never changed         │
│   ├── Read as memory-mapped views: no process holds a whole file        │
│   └── Shared by every rate, split and configuration of one source       │
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

Strain caches of earlier gravyflow versions (segments/, the glitch, event and
partner caches *_cache_v*.h5, and the download cache astropy/) are never read;
gravyflow names them once, with their sizes. Delete them to reclaim the space.
""")

# =============================================================================
# SECTION 3: TRANSIENTS READ THE STORE
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 3: Transients Read the Store (TRANSIENT mode)")
print("=" * 70)

print("""
Glitches, events and their partners are read exactly as real noise is:
each detector's offsource and padded onsource are one view of the stored
files around the transient, resampled to the request's rate.

    ~/.gravyflow/cache/store/v<format>/<source>/<detector>/<rate>/<seconds>/
    ├── <k>.<sha256>.npy    the source's k-th file, raw
    └── <k>.json            its record: what it holds, and what it does not

Store benefits for transients:
  ✓ A file is fetched once, whatever rate or window asks for it
  ✓ Every rate and configuration reads the same bytes: warm equals cold
  ✓ A transient the source cannot deliver is refused by name, never
    skipped or zero-filled, and nothing is stored for it
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
# SECTION 5: FETCHING AHEAD FOR LARGE RUNS
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 5: Fetching Ahead for Large Training Runs")
print("=" * 70)

print("""
For large training runs, fetch the files a dataset reads before training:
draw the dataset you will train on once, with the same settings. Every
file it reads is kept, and any later rate or window reads it again:

Example script:

    # Fetch the files one pass over the training glitches reads
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
num_examples_per_batch examples, 32,000 here, and fetches the files
they read: each a whole source file (4096 s from GWOSC), so glitches
close in time share one.

Recommended workflow:
  ┌───────────────┐    ┌──────────────┐    ┌────────────────┐
  │  Fetch ahead  │ →  │   Train      │ →  │   Evaluate     │
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

  # Old segment, transient and download caches, which this gravyflow never reads
  rm -rf ~/.gravyflow/cache/segments/ ~/.gravyflow/cache/*_cache_v*.h5 ~/.gravyflow/cache/astropy/

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
║  1. Fetch ahead of training    │  Avoid network during training       ║
╠═══════════════════════════════════════════════════════════════════════╣
║  2. Size the memory cap        │  gf.Defaults.memory_bytes (2 GiB)   ║
╠═══════════════════════════════════════════════════════════════════════╣
║  3. Any rate reads the store   │  No refetch for a new sample_rate    ║
╠═══════════════════════════════════════════════════════════════════════╣
║  4. Use SSD for the store      │  Memory-mapped reads need fast I/O  ║
╠═══════════════════════════════════════════════════════════════════════╣
║  5. Multi-IFO single request   │  Download all IFOs together         ║
╠═══════════════════════════════════════════════════════════════════════╣
║  6. Batch size ≥ 16            │  Better GPU utilization              ║
╠═══════════════════════════════════════════════════════════════════════╣
║  7. Use group= for splits      │  Deterministic train/val from one   ║
║                                │  store                               ║
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
   The strain store (all real strain), then the source

2. THE STRAIN STORE:
   - The source's own files, raw, fetched once, read as views
   - Refusals name the exclusions entry that plans around them

3. TRANSIENTS:
   - Read from the store as noise is; refused by name when not delivered

4. MEMORY:
   - gf.Defaults.memory_bytes: 2 GiB per process by default

5. FETCHING AHEAD:
   - Draw the training dataset once before large runs: it fetches
     every file its steps_per_epoch x num_examples_per_batch examples read
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
