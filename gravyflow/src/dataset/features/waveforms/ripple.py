import sys
import os

# Add local ripple submodule to path
# Assuming gravyflow root is 3 levels up from this file
# .../gravyflow/src/dataset/features/waveforms/ripple.py
# We want .../gravyflow/ripple/src
current_dir = os.path.dirname(os.path.abspath(__file__))
gravyflow_root = os.path.abspath(os.path.join(current_dir, "../../../../../"))
ripple_src_path = os.path.join(gravyflow_root, "ripple", "src")

if ripple_src_path not in sys.path:
    sys.path.insert(0, ripple_src_path)

import ripplegw
# Alias ripplegw to ripple so existing imports work
sys.modules["ripple"] = ripplegw

import jax
import jax.numpy as jnp
import numpy as np
from ripple.waveforms import (
    IMRPhenomD, 
    IMRPhenomXAS, 
    IMRPhenomPv2, 
    TaylorF2, 
    IMRPhenomD_NRTidalv2
)
from typing import Union, List, Tuple
from ripple import ms_to_Mc_eta
import keras
from keras import ops

jax.config.update("jax_enable_x64", True)

# Constants
G_SI = 6.67430e-11
C_SI = 299792458.0
M_SOLAR_SI = 1.98855e30

def calc_minimum_frequency(
    mass_1_msun,
    mass_2_msun,
    duration_seconds
):
    """
    Calculates minimum frequency based on inputted masses and duration.
    """
    # Ensure inputs are tensors
    mass_1_msun = jnp.array(mass_1_msun)
    mass_2_msun = jnp.array(mass_2_msun)
    duration_seconds = jnp.array(duration_seconds)

    m1_kg = mass_1_msun * M_SOLAR_SI
    m2_kg = mass_2_msun * M_SOLAR_SI
    
    mc_kg = ( (m1_kg * m2_kg)**3 / (m1_kg + m2_kg) )**(1.0/5.0)
    
    term1 = (duration_seconds / 5.0)**(-3.0/8.0)
    term2 = 1.0 / (8.0 * np.pi)
    term3 = (G_SI * mc_kg / (C_SI**3))**(-5.0/8.0)
    
    # Ensure term2 is compatible dtype
    dtype = mass_1_msun.dtype
    if dtype == jnp.float32:
        term2 = jnp.array(term2, dtype=jnp.float32)
    
    f_min = term1 * term2 * term3
    
    # Clamp to be at least 1.0 Hz
    f_min = jnp.maximum(f_min, 1.0)
        
    return f_min

def calc_duration_from_f_min(
    mass_1_msun,
    mass_2_msun,
    f_min
):
    """
    Calculates duration based on inputted masses and minimum frequency.
    """
    mass_1_msun = ops.convert_to_tensor(mass_1_msun)
    mass_2_msun = ops.convert_to_tensor(mass_2_msun)
    f_min = ops.convert_to_tensor(f_min)

    m1_kg = mass_1_msun * M_SOLAR_SI
    m2_kg = mass_2_msun * M_SOLAR_SI
    
    mc_kg = ( (m1_kg * m2_kg)**3 / (m1_kg + m2_kg) )**(1.0/5.0)
    
    term2 = 1.0 / (8.0 * np.pi)
    term3 = (G_SI * mc_kg / (C_SI**3))**(-5.0/8.0)
    
    # Ensure term2 and term3 are compatible dtype
    dtype = mass_1_msun.dtype
    if dtype == "float32":
        term2 = ops.cast(term2, "float32")
        term3 = ops.cast(term3, "float32")
    
    # Also ensure f_min matches
    if f_min.dtype != dtype:
        f_min = ops.cast(f_min, dtype)

    duration = 5.0 * (f_min / (term2 * term3))**(-8.0/3.0)
    
    return duration

APPROXIMANTS = {
    "IMRPhenomD": IMRPhenomD,
    "IMRPhenomXAS": IMRPhenomXAS,
    "IMRPhenomPv2": IMRPhenomPv2,
    "TaylorF2": TaylorF2,
    "IMRPhenomD_NRTidalv2": IMRPhenomD_NRTidalv2
}

@jax.jit(static_argnames=["num_waveforms", "sample_rate_hertz", "duration_seconds", "approximant"])
def _generate_ripple_waveform_jit(
    num_waveforms: int = 1,
    sample_rate_hertz: float = 4096.0,
    duration_seconds: float = 2.0,
    mass_1_msun = 30.0,
    mass_2_msun = 30.0,
    inclination_radians = 0.0,
    distance_mpc = 1000.0,
    reference_orbital_phase_in = 0.0,
    ascending_node_longitude = 0.0,
    eccentricity = 0.0,
    mean_periastron_anomaly = 0.0,
    spin_1_in = [0.0, 0.0, 0.0],
    spin_2_in = [0.0, 0.0, 0.0],
    lambda_1 = 0.0,
    lambda_2 = 0.0,
    approximant: str = "IMRPhenomD",
    f_ref: float = 20.0,
    coalescence_time: float = None
):
    """
    JIT-compiled core of generate_ripple_waveform.
    """
    # ... (rest of the function body is unchanged, just indented? No, replace_file_content replaces the block)
    # Wait, I need to provide the body if I replace the definition.
    # But the body is long.
    # I can just rename the decorator and function name, then add the wrapper below.
    # But the wrapper needs to call the JIT function.
    
    # Let's try to rename the function in place, and add the wrapper.
    # But I can't easily insert the wrapper *after* the function without reading the whole file or guessing line numbers.
    # Actually, I can rename the function definition line.
    # And then add the wrapper at the end of the file?
    # Or rename it, and add the wrapper *before* it? No, wrapper calls it.
    
    # I'll rename the function to `_generate_ripple_waveform_jit`.
    # And then I'll append the wrapper `generate_ripple_waveform` at the end of the file.
    
    pass

# Actually, I'll use multi_replace to do it cleanly.
# 1. Rename the function definition.
# 2. Add the wrapper at the end of the file.

# Wait, `replace_file_content` replaces a block.
# I will replace lines 102-122 with the renamed function definition.
# And I will append the wrapper at the end of the file.

    """
    Generates time-domain waveforms using gwripple.
    Returns array of shape (num_waveforms, 2, num_samples).
    """
    
    if approximant not in APPROXIMANTS:
        raise ValueError(f"Approximant {approximant} not supported. Available: {list(APPROXIMANTS.keys())}")
        
    waveform_module = APPROXIMANTS[approximant]
    gen_func = getattr(waveform_module, f"gen_{approximant}_hphc")

    # Helper to ensure JAX array
    def ensure_jax(val, length):
        # Convert to JAX array/tensor first
        val = jnp.array(val)
             
        # Now val is array-like or tensor
        # If scalar (ndim=0), broadcast to length
        if len(val.shape) == 0:
             val = jnp.broadcast_to(val, (length,))
             
        return val

    # Handle spins
    # spin_1_in can be (3,) or (num_waveforms, 3) or flattened
    # If it's a list/tuple, convert to array
    
    spin_1_in = jnp.array(spin_1_in)
    spin_2_in = jnp.array(spin_2_in)
    
    # If single spin vector provided, tile it
    if spin_1_in.size == 3 and spin_1_in.ndim == 1:
        spin_1_in = jnp.tile(spin_1_in, (num_waveforms, 1))
    elif spin_1_in.size == 3 * num_waveforms:
        spin_1_in = spin_1_in.reshape(num_waveforms, 3)
        
    if spin_2_in.size == 3 and spin_2_in.ndim == 1:
        spin_2_in = jnp.tile(spin_2_in, (num_waveforms, 1))
    elif spin_2_in.size == 3 * num_waveforms:
        spin_2_in = spin_2_in.reshape(num_waveforms, 3)
    
    m1_jax = ensure_jax(mass_1_msun, num_waveforms)
    m2_jax = ensure_jax(mass_2_msun, num_waveforms)
    dist_jax = ensure_jax(distance_mpc, num_waveforms)
    inc_jax = ensure_jax(inclination_radians, num_waveforms)
    phi_jax = ensure_jax(reference_orbital_phase_in, num_waveforms)
    lambda1_jax = ensure_jax(lambda_1, num_waveforms)
    lambda2_jax = ensure_jax(lambda_2, num_waveforms)
    
    if coalescence_time is None:
        # Default to placing merger at 90% of duration to maximize inspiral visibility
        # while leaving room for ringdown.
        coalescence_time = duration_seconds * 0.9

    tc_jax = ensure_jax(coalescence_time, num_waveforms)
    
    # Calculate f_min for each waveform
    # Vectorized calculation
    # Use tc_jax (time to merger) to calculate the starting frequency at t=0
    # This prevents the waveform from starting before the buffer and wrapping around.
    f_mins = calc_minimum_frequency(m1_jax, m2_jax, tc_jax)
    
    # We need f_l for frequency grid.
    # ripple gen_func takes `fs`.
    # We define fs based on duration and sample rate.
    
    f_u = sample_rate_hertz / 2.0
    num_samples = int(sample_rate_hertz * duration_seconds)
    num_freqs = num_samples // 2 + 1
    # Use float64 for frequency grid to ensure phase precision
    fs = jnp.linspace(0, f_u, num_freqs, dtype=jnp.float64)
    
    # Mc, eta calculation
    # ms_to_Mc_eta takes (m1, m2) in solar masses
    # Vectorize it
    # ms_to_Mc_eta is JAX-jit compatible.
    
    Mc, eta = jax.vmap(ms_to_Mc_eta)(jnp.stack([m1_jax, m2_jax], axis=1))
    
    # Construct theta
    if approximant in ["IMRPhenomD", "IMRPhenomXAS"]:
        chi1 = spin_1_in[:, 2]
        chi2 = spin_2_in[:, 2]
        theta = jnp.stack([Mc, eta, chi1, chi2, dist_jax, tc_jax, phi_jax, inc_jax], axis=1)
        theta = theta.astype(jnp.float64)
        
    elif approximant == "IMRPhenomPv2":
        chi1x = spin_1_in[:, 0]
        chi1y = spin_1_in[:, 1]
        chi1z = spin_1_in[:, 2]
        chi2x = spin_2_in[:, 0]
        chi2y = spin_2_in[:, 1]
        chi2z = spin_2_in[:, 2]
        theta = jnp.stack([
            Mc, eta, 
            chi1x, chi1y, chi1z, 
            chi2x, chi2y, chi2z, 
            dist_jax, tc_jax, phi_jax, inc_jax
        ], axis=1)
        
    elif approximant in ["TaylorF2", "IMRPhenomD_NRTidalv2"]:
        chi1 = spin_1_in[:, 2]
        chi2 = spin_2_in[:, 2]
        theta = jnp.stack([
            Mc, eta, 
            chi1, chi2, 
            lambda1_jax, lambda2_jax, 
            dist_jax, tc_jax, phi_jax, inc_jax
        ], axis=1)
    else:
        raise ValueError(f"Parameter packing for {approximant} not implemented.")

    # Generate batch
    # We use jax.vmap directly.
    # gen_func is JIT-able.
    
    def gen_one(theta_i, f_min_i):
        hp, hc = gen_func(fs, theta_i, f_ref)
        
        # Apply taper to avoid ringing from hard frequency cut
        taper_width = 8.0 # Hz
        f_taper_end = f_min_i + taper_width
        
        mask_zero = fs < f_min_i
        mask_one = fs >= f_taper_end
        mask_taper = (fs >= f_min_i) & (fs < f_taper_end)
        
        # Sine-squared taper
        taper_factor = jnp.sin( (jnp.pi / 2.0) * (fs - f_min_i) / (f_taper_end - f_min_i) )**2
        
        final_mask = jnp.where(mask_zero, 0.0, jnp.where(mask_one, 1.0, taper_factor))

        # Use jnp.where to avoid NaN propagation from masked regions
        hp = jnp.where(mask_zero, 0.0, hp * final_mask)
        hc = jnp.where(mask_zero, 0.0, hc * final_mask)
        
        return hp, hc

    hp_freq, hc_freq = jax.vmap(gen_one)(theta, f_mins)
    
    # IFFT
    # jnp.fft.irfft
    
    hp_time = jax.vmap(lambda x: jnp.fft.irfft(x, n=num_samples))(hp_freq)
    hc_time = jax.vmap(lambda x: jnp.fft.irfft(x, n=num_samples))(hc_freq)
    
    hp_time = hp_time * sample_rate_hertz
    hc_time = hc_time * sample_rate_hertz
    
    # Zero out signal after merger + ringdown buffer to prevent wrapping artifacts
    # tc_jax is time to merger.
    # We add a generous buffer for ringdown.
    ringdown_buffer = 0.2 # seconds
    
    # Create time array
    # shape (num_samples,)
    times = jnp.linspace(0, duration_seconds, num_samples, endpoint=False)
    # Expand to (num_waveforms, num_samples)
    times = jnp.expand_dims(times, 0)
    
    # tc_jax is (num_waveforms,)
    # Expand to (num_waveforms, 1)
    tc_expanded = jnp.expand_dims(tc_jax, 1)
    
    # Mask: Keep signal where t < tc + buffer
    # This assumes the signal is physically zero after ringdown.
    # Any non-zero value there is likely noise or wrapping from generation.
    mask_post_merger = times < (tc_expanded + ringdown_buffer)
    mask_post_merger = mask_post_merger.astype(jnp.float32)
    
    hp_time = hp_time * mask_post_merger
    hc_time = hc_time * mask_post_merger
    
    waveforms = jnp.stack([hp_time, hc_time], axis=1)
    
    # Return JAX array (which Keras 3 can handle as Tensor)
    return waveforms

def generate_ripple_waveform(
    num_waveforms: int = 1,
    sample_rate_hertz: float = 4096.0,
    duration_seconds: float = 2.0,
    mass_1_msun = 30.0,
    mass_2_msun = 30.0,
    inclination_radians = 0.0,
    distance_mpc = 1000.0,
    reference_orbital_phase_in = 0.0,
    ascending_node_longitude = 0.0,
    eccentricity = 0.0,
    mean_periastron_anomaly = 0.0,
    spin_1_in = [0.0, 0.0, 0.0],
    spin_2_in = [0.0, 0.0, 0.0],
    lambda_1 = 0.0,
    lambda_2 = 0.0,
    approximant: str = "IMRPhenomD",
    f_ref: float = 20.0,
    coalescence_time: float = None
):
    """
    Wrapper for _generate_ripple_waveform_jit that ensures inputs are JAX-compatible.
    """
    # Convert inputs to numpy/jax arrays if they are TF tensors or other types
    def convert(val):
        if hasattr(val, "numpy"):
            return val.numpy()
        return val

    mass_1_msun = convert(mass_1_msun)
    mass_2_msun = convert(mass_2_msun)
    inclination_radians = convert(inclination_radians)
    distance_mpc = convert(distance_mpc)
    reference_orbital_phase_in = convert(reference_orbital_phase_in)
    ascending_node_longitude = convert(ascending_node_longitude)
    eccentricity = convert(eccentricity)
    mean_periastron_anomaly = convert(mean_periastron_anomaly)
    spin_1_in = convert(spin_1_in)
    spin_2_in = convert(spin_2_in)
    lambda_1 = convert(lambda_1)
    lambda_2 = convert(lambda_2)
    coalescence_time = convert(coalescence_time)

    return _generate_ripple_waveform_jit(
        num_waveforms=num_waveforms,
        sample_rate_hertz=sample_rate_hertz,
        duration_seconds=duration_seconds,
        mass_1_msun=mass_1_msun,
        mass_2_msun=mass_2_msun,
        inclination_radians=inclination_radians,
        distance_mpc=distance_mpc,
        reference_orbital_phase_in=reference_orbital_phase_in,
        ascending_node_longitude=ascending_node_longitude,
        eccentricity=eccentricity,
        mean_periastron_anomaly=mean_periastron_anomaly,
        spin_1_in=spin_1_in,
        spin_2_in=spin_2_in,
        lambda_1=lambda_1,
        lambda_2=lambda_2,
        approximant=approximant,
        f_ref=f_ref,
        coalescence_time=coalescence_time
    )
