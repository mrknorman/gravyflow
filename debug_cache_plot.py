#!/usr/bin/env python
"""Debug script to plot raw cache data and check if glitches are visible."""

import numpy as np
import matplotlib.pyplot as plt
import h5py
from pathlib import Path

# Find the cache file
cache_dir = Path("./generator_data")
cache_files = list(cache_dir.glob("glitch_cache_*.h5"))

if not cache_files:
    print("No cache files found in ./generator_data")
    exit(1)

cache_path = cache_files[0]
print(f"Using cache: {cache_path}")

with h5py.File(cache_path, 'r') as f:
    meta = dict(f['metadata'].attrs)
    print(f"Cache metadata: {meta}")
    
    grp = f['glitches']
    onsource = grp['onsource'][:]  # Full array
    gps_times = grp['gps_times'][:]
    
    print(f"Onsource shape: {onsource.shape}")
    print(f"Num glitches: {len(gps_times)}")
    print(f"First 5 GPS times: {gps_times[:5]}")
    
    # Plot first 5 examples
    n_examples = min(5, len(gps_times))
    fig, axes = plt.subplots(n_examples, 1, figsize=(14, 3*n_examples))
    if n_examples == 1:
        axes = [axes]
    
    sample_rate = meta['sample_rate_hertz']
    duration = meta['onsource_duration']
    
    for i, ax in enumerate(axes):
        data = onsource[i, 0, :]  # First IFO
        time = np.arange(len(data)) / sample_rate
        
        ax.plot(time, data, 'b-', linewidth=0.5)
        ax.axvline(duration/2, color='r', linestyle='--', label='Center (expected glitch)')
        ax.set_title(f"Glitch {i}: GPS {gps_times[i]:.3f}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Strain")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = "./debug_cache_plots.png"
    plt.savefig(output_path, dpi=150)
    print(f"Saved plot to: {output_path}")
    plt.close()
