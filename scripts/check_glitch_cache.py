#!/usr/bin/env python3
"""
Check glitch cache integrity - NaNs, data sizes, and basic statistics.
"""
import sys
import numpy as np
import h5py
from pathlib import Path

def check_cache(cache_path: str):
    """Check a glitch cache for integrity issues."""
    path = Path(cache_path)
    
    if not path.exists():
        print(f"❌ Cache file not found: {path}")
        return False
    
    print(f"📁 Checking: {path}")
    print(f"   File size: {path.stat().st_size / 1024 / 1024:.2f} MB")
    print()
    
    with h5py.File(path, 'r') as f:
        if 'glitches' not in f:
            print("❌ Invalid cache format: 'glitches' group not found")
            return False
        
        grp = f['glitches']
        
        # Metadata
        print("📊 Metadata:")
        print(f"   Sample rate: {grp.attrs.get('sample_rate_hertz', 'N/A')} Hz")
        print(f"   Onsource duration: {grp.attrs.get('onsource_duration', 'N/A')} s")
        print(f"   Offsource duration: {grp.attrs.get('offsource_duration', 'N/A')} s")
        print(f"   IFOs: {list(grp.attrs.get('ifo_names', []))}")
        print()
        
        # Dataset shapes
        print("📐 Dataset shapes:")
        for name in ['onsource', 'offsource', 'gps_times', 'labels']:
            if name in grp:
                dset = grp[name]
                print(f"   {name}: {dset.shape} (dtype: {dset.dtype})")
        print()
        
        # Load and check for issues
        n_glitches = grp['gps_times'].shape[0]
        print(f"🔢 Total glitches: {n_glitches}")
        
        if n_glitches == 0:
            print("⚠️  Cache is empty!")
            return True
        
        # Check in chunks to avoid memory issues
        chunk_size = 1000
        total_nan_onsource = 0
        total_nan_offsource = 0
        total_zero_onsource = 0
        total_inf_onsource = 0
        
        print("\n🔍 Checking for data issues...")
        
        for start in range(0, n_glitches, chunk_size):
            end = min(start + chunk_size, n_glitches)
            
            onsource = grp['onsource'][start:end]
            offsource = grp['offsource'][start:end]
            
            # NaN checks
            nan_ons = np.isnan(onsource).sum()
            nan_offs = np.isnan(offsource).sum()
            total_nan_onsource += nan_ons
            total_nan_offsource += nan_offs
            
            # Inf checks
            total_inf_onsource += np.isinf(onsource).sum()
            
            # All-zero checks (per sample)
            for i in range(len(onsource)):
                if np.all(onsource[i] == 0):
                    total_zero_onsource += 1
            
            if (start + chunk_size) % 10000 == 0:
                print(f"   Checked {end}/{n_glitches}...")
        
        print()
        print("📋 Results:")
        print(f"   NaN values in onsource: {total_nan_onsource}")
        print(f"   NaN values in offsource: {total_nan_offsource}")
        print(f"   Inf values in onsource: {total_inf_onsource}")
        print(f"   All-zero onsource samples: {total_zero_onsource}")
        
        # Sample statistics
        print("\n📈 Sample statistics (first 100 samples):")
        sample_ons = grp['onsource'][:min(100, n_glitches)]
        print(f"   Onsource min: {sample_ons.min():.6e}")
        print(f"   Onsource max: {sample_ons.max():.6e}")
        print(f"   Onsource mean: {sample_ons.mean():.6e}")
        print(f"   Onsource std: {sample_ons.std():.6e}")
        
        # GPS time range
        gps_times = grp['gps_times'][:]
        print(f"\n🕐 GPS time range:")
        print(f"   Min: {gps_times.min():.1f}")
        print(f"   Max: {gps_times.max():.1f}")
        print(f"   Unique times: {len(np.unique(np.round(gps_times, 1)))}")
        
        # Label distribution
        labels = grp['labels'][:]
        unique_labels, counts = np.unique(labels, return_counts=True)
        print(f"\n🏷️  Label distribution:")
        for label, count in zip(unique_labels, counts):
            print(f"   Label {label}: {count} ({count/n_glitches*100:.1f}%)")
        
        # Summary
        has_issues = (total_nan_onsource > 0 or total_nan_offsource > 0 or 
                      total_inf_onsource > 0 or total_zero_onsource > n_glitches * 0.01)
        
        print()
        if has_issues:
            print("⚠️  ISSUES DETECTED - cache may have data quality problems")
        else:
            print("✅ Cache appears healthy!")
        
        return not has_issues

if __name__ == "__main__":
    default_path = "./gravyflow_data/glitch_cache_O3_L1.h5"
    cache_path = sys.argv[1] if len(sys.argv) > 1 else default_path
    check_cache(cache_path)
