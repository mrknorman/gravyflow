#!/usr/bin/env python3
"""
=============================================================================
TUTORIAL 07: Caching and Performance
=============================================================================

This tutorial covers caching mechanisms and performance optimization 
strategies for the gravyflow acquisition pipeline.

LEARNING OBJECTIVES:
1. Understand segment caching
2. Learn about TransientCache for glitches/events
3. Configure prefetching for better performance
4. Best practices for large-scale training

RUNTIME: ~1 minute (mostly informational)
"""

# =============================================================================
# IMPORTS
# =============================================================================

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
GravyFlow uses multiple levels of caching to minimize data download time:

┌─────────────────────────────────────────────────────────────────────────┐
│                         CACHING HIERARCHY                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Level 1: MEMORY CACHE (fastest)                                       │
│   ├── Recently accessed segments                                        │
│   ├── Configurable size via prefetch settings                          │
│   └── Lost on program exit                                              │
│                                                                         │
│   Level 2: DISK CACHE (persistent)                                      │
│   ├── Segment metadata files                                            │
│   ├── TransientCache HDF5 files (for glitches/events)                  │
│   └── Survives restarts                                                 │
│                                                                         │
│   Level 3: GWOSC (network, slowest)                                     │
│   ├── Original data source                                              │
│   └── Download on cache miss                                            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

Cache hit order: Memory → Disk → GWOSC (network)
""")

# =============================================================================
# SECTION 2: SEGMENT CACHING
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 2: Segment Caching (cache_segments)")
print("=" * 70)

print("""
Segment caching stores metadata about available data segments.
This avoids querying GWOSC segment databases repeatedly.

Enable with:
    obtainer = gf.IFODataObtainer(
        cache_segments=True,  # ← Enable segment caching (default)
        ...
    )

What gets cached:
  - Valid segment GPS times
  - Segment durations
  - Data quality flags

Cache location:
  ~/.gravyflow/cache/segments/

First run: Queries segment database (slow)
Subsequent runs: Uses cached metadata (fast)
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
# SECTION 4: PREFETCHING
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 4: Prefetching for Performance")
print("=" * 70)

print("""
Prefetching downloads upcoming segments in advance while processing
current data, hiding network latency.

Configure with:
    obtainer = gf.IFODataObtainer(
        prefetch_segments=16,  # Number of segments to prefetch
        ...
    )

Recommended values:
  - NOISE mode: 8-16 segments (larger segments)
  - TRANSIENT mode: 32-64 segments (smaller, more numerous)

Trade-offs:
  Higher prefetch → More memory usage, less waiting
  Lower prefetch  → Less memory, possible stalls

Visual:
    Current segment       Prefetched            Not yet fetched
    ════════════════     ════════════════════  ═══════════════════
    [Processing...]      [Ready] [Ready] ...   [Not downloaded...]
""")

# =============================================================================
# SECTION 5: PRE-CACHING FOR LARGE RUNS
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 5: Pre-Caching for Large Training Runs")
print("=" * 70)

print("""
For large training runs, pre-populate the cache before training:

Example script:
    
    # Pre-cache all glitches
    gf.precache_transients(
        ifos=[gf.IFO.L1, gf.IFO.H1],
        sample_rate_hertz=2048.0,
        onsource_duration_seconds=1.0,
        offsource_duration_seconds=16.0,
        observing_runs=[gf.ObservingRun.O3],
        data_labels=[gf.DataLabel.GLITCHES],
    )

Run this ONCE before training:
  1. Downloads all glitch windows
  2. Stores in TransientCache
  3. Training reads from cache (fast!)

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
  # Clear all cache (careful!)
  rm -rf ~/.gravyflow/cache/
  
  # Clear only transient cache
  rm -rf ~/.gravyflow/cache/transient/
  
  # Clear only segment metadata
  rm -rf ~/.gravyflow/cache/segments/
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
║  2. Use appropriate prefetch   │  NOISE: 16, TRANSIENT: 64           ║
╠═══════════════════════════════════════════════════════════════════════╣
║  3. Match cache parameters     │  Same sample_rate, durations         ║
╠═══════════════════════════════════════════════════════════════════════╣
║  4. Use SSD for cache          │  HDF5 reads benefit from fast I/O   ║
╠═══════════════════════════════════════════════════════════════════════╣
║  5. Multi-IFO single request   │  Download all IFOs together         ║
╠═══════════════════════════════════════════════════════════════════════╣
║  6. Batch size ≥ 16            │  Better GPU utilization              ║
╠═══════════════════════════════════════════════════════════════════════╣
║  7. Use groups for splits      │  Deterministic train/val without    ║
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

# Training obtainer
train_obtainer = gf.IFODataObtainer(
    observing_runs=[gf.ObservingRun.O3],
    data_quality=gf.DataQuality.BEST,
    data_labels=[gf.DataLabel.GLITCHES],
    segment_order=gf.SegmentOrder.RANDOM,
    cache_segments=True,        # ← Enable segment caching
    prefetch_segments=64,       # ← High prefetch for TRANSIENT
    balanced_glitch_types=True, # ← Class balancing
    augmentations=[
        gf.SignReversal(probability=0.5),
        gf.TimeReversal(probability=0.5),
        gf.RandomShift(probability=0.5, shift_fraction=0.2),
        gf.AddNoise(probability=0.5, amplitude=0.1),
    ],
)

# Validation obtainer (no augmentation)
val_obtainer = gf.IFODataObtainer(
    observing_runs=[gf.ObservingRun.O3],
    data_quality=gf.DataQuality.BEST,
    data_labels=[gf.DataLabel.GLITCHES],
    segment_order=gf.SegmentOrder.RANDOM,
    cache_segments=True,
    prefetch_segments=64,
    balanced_glitch_types=True,
    augmentations=[],  # ← No augmentation for validation
)

# Use with TransientObtainer and groups
train_transient = gf.TransientObtainer(
    ifo_data_obtainer=train_obtainer,
    ifos=[gf.IFO.L1, gf.IFO.H1],
    group="train",
    groups={"train": 0.8, "validate": 0.2},
)

val_transient = gf.TransientObtainer(
    ifo_data_obtainer=val_obtainer,
    ifos=[gf.IFO.L1, gf.IFO.H1],
    group="validate",
    groups={"train": 0.8, "validate": 0.2},
)
""")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 70)
print("SUMMARY: Caching and Performance")
print("=" * 70)

print("""
1. CACHING HIERARCHY:
   Memory → Disk → Network (GWOSC)

2. SEGMENT CACHING:
   - cache_segments=True (default)
   - Stores segment metadata

3. TRANSIENT CACHE:
   - Stores pre-extracted windows
   - HDF5 files in ~/.gravyflow/cache/transient/

4. PREFETCHING:
   - prefetch_segments=16-64
   - Higher for TRANSIENT, lower for NOISE

5. PRE-CACHING:
   - Use gf.precache_transients() before large runs
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
  03_noise_mode_advanced.py    ✓ Grid sampling, groups, multi-IFO
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
