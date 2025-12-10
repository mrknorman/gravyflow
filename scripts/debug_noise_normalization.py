#!/usr/bin/env python
"""
Debug script to compare noise normalizations between COLORED and REAL noise types.

This script generates samples from both noise types and compares their
statistical properties (mean, std, PSD shape) to identify normalization issues.
"""

import os
os.environ['KERAS_BACKEND'] = 'jax'

import numpy as np
from keras import ops
import gravyflow as gf

def analyze_noise(noise_type_name, noise_data, offsource_data, sample_rate_hertz):
    """Analyze noise statistics."""
    print(f"\n{'='*60}")
    print(f"Noise Type: {noise_type_name}")
    print(f"{'='*60}")
    
    # Convert to numpy for analysis
    onsource = np.array(noise_data)
    offsource = np.array(offsource_data)
    
    print(f"Onsource shape: {onsource.shape}")
    print(f"Offsource shape: {offsource.shape}")
    
    # Basic statistics
    print(f"\nOnsource statistics:")
    print(f"  Mean: {np.mean(onsource):.6e}")
    print(f"  Std:  {np.std(onsource):.6e}")
    print(f"  Min:  {np.min(onsource):.6e}")
    print(f"  Max:  {np.max(onsource):.6e}")
    
    print(f"\nOffsource statistics:")
    print(f"  Mean: {np.mean(offsource):.6e}")
    print(f"  Std:  {np.std(offsource):.6e}")
    print(f"  Min:  {np.min(offsource):.6e}")
    print(f"  Max:  {np.max(offsource):.6e}")
    
    # Compute PSD
    freqs, psd_val = gf.psd(
        ops.convert_to_tensor(offsource, dtype="float32"),
        nperseg=1024,
        sample_rate_hertz=sample_rate_hertz
    )
    
    freqs = np.array(freqs)
    psd_val = np.array(psd_val)
    
    # Find typical PSD values at key frequencies
    for f_target in [50, 100, 200]:
        idx = np.argmin(np.abs(freqs - f_target))
        print(f"  PSD at {freqs[idx]:.0f}Hz: {psd_val[0, idx]:.6e}")
    
    return onsource, offsource, freqs, psd_val

def main():
    print("="*60)
    print("Noise Normalization Comparison: COLORED vs REAL")
    print("="*60)
    
    # Parameters matching validation setup
    sample_rate_hertz = 2048.0
    onsource_duration_seconds = 1.0
    offsource_duration_seconds = 16.0
    crop_duration_seconds = 0.5
    num_examples_per_batch = 4
    seed = 42
    
    # The scale factor that should be applied
    scale_factor = gf.Defaults.scale_factor
    print(f"\nDefault scale_factor: {scale_factor}")
    
    # --- Test COLORED noise ---
    print("\n" + "="*60)
    print("Testing COLORED noise")
    print("="*60)
    
    colored_noise_obtainer = gf.NoiseObtainer(
        noise_type=gf.NoiseType.COLORED,
        ifos=gf.IFO.L1
    )
    
    colored_gen = colored_noise_obtainer(
        sample_rate_hertz=sample_rate_hertz,
        onsource_duration_seconds=onsource_duration_seconds,
        crop_duration_seconds=crop_duration_seconds,
        offsource_duration_seconds=offsource_duration_seconds,
        num_examples_per_batch=num_examples_per_batch,
        scale_factor=scale_factor,  # Explicitly pass scale factor
        seed=seed
    )
    
    colored_onsource, colored_offsource, _ = next(colored_gen)
    
    c_on, c_off, c_freqs, c_psd = analyze_noise(
        "COLORED", colored_onsource, colored_offsource, sample_rate_hertz
    )
    
    # --- Test REAL noise ---
    print("\n" + "="*60)
    print("Testing REAL noise")
    print("="*60)
    
    ifo_data_obtainer = gf.IFODataObtainer(
        observing_runs=[gf.ObservingRun.O3],
        data_quality=gf.DataQuality.BEST,
        data_labels=[gf.DataLabel.NOISE],
        force_acquisition=True,
        cache_segments=False
    )
    
    real_noise_obtainer = gf.NoiseObtainer(
        noise_type=gf.NoiseType.REAL,
        ifos=gf.IFO.L1,
        ifo_data_obtainer=ifo_data_obtainer
    )
    
    real_gen = real_noise_obtainer(
        sample_rate_hertz=sample_rate_hertz,
        onsource_duration_seconds=onsource_duration_seconds,
        crop_duration_seconds=crop_duration_seconds,
        offsource_duration_seconds=offsource_duration_seconds,
        num_examples_per_batch=num_examples_per_batch,
        scale_factor=scale_factor,  # Explicitly pass scale factor
        group="validate",
        seed=seed
    )
    
    real_onsource, real_offsource, gps_times = next(real_gen)
    print(f"GPS times: {gps_times}")
    
    r_on, r_off, r_freqs, r_psd = analyze_noise(
        "REAL", real_onsource, real_offsource, sample_rate_hertz
    )
    
    # --- Compare ---
    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60)
    
    print(f"\nStd ratio (REAL / COLORED): {np.std(r_on) / np.std(c_on):.4f}")
    print(f"If this is not ~1.0, there may be a normalization issue.\n")
    
    # Compare PSDs at key frequencies
    for f_target in [50, 100, 200]:
        c_idx = np.argmin(np.abs(c_freqs - f_target))
        r_idx = np.argmin(np.abs(r_freqs - f_target))
        
        c_psd_val = c_psd[0, c_idx]
        r_psd_val = r_psd[0, r_idx]
        ratio = r_psd_val / c_psd_val
        
        print(f"PSD ratio at {f_target}Hz: {ratio:.4f}")
    
    print("\n" + "="*60)
    print("Test complete")
    print("="*60)

if __name__ == "__main__":
    main()
