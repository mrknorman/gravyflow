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

print(f"DEBUG: ripple_src_path: {ripple_src_path}")
print(f"DEBUG: sys.path[0]: {sys.path[0]}")
print(f"DEBUG: 'ripplegw' in sys.modules: {'ripplegw' in sys.modules}")

import ripplegw
# Alias ripplegw to ripple so existing imports work
sys.modules["ripple"] = ripplegw
print(f"DEBUG: ripple aliased to: {ripplegw.__file__}")

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
    # Ensure inputs are floats or arrays
    # If they are tensors, we might need to cast or use ops
    
    # We can use ops for compatibility
    m1_kg = mass_1_msun * M_SOLAR_SI
    m2_kg = mass_2_msun * M_SOLAR_SI
    
    mc_kg = ( (m1_kg * m2_kg)**3 / (m1_kg + m2_kg) )**(1.0/5.0)
    
    term1 = (duration_seconds / 5.0)**(-3.0/8.0)
    term2 = 1.0 / (8.0 * np.pi)
    term3 = (G_SI * mc_kg / (C_SI**3))**(-5.0/8.0)
    
    f_min = term1 * term2 * term3
    
    # Clamp to be at least 1.0 Hz
    # Use ops.maximum for tensor compatibility
    f_min = ops.maximum(f_min, 1.0)
        
    return f_min

def calc_duration_from_f_min(
    mass_1_msun,
    mass_2_msun,
    f_min
):
    """
    Calculates duration based on inputted masses and minimum frequency.
    """
    m1_kg = mass_1_msun * M_SOLAR_SI
    m2_kg = mass_2_msun * M_SOLAR_SI
    
    mc_kg = ( (m1_kg * m2_kg)**3 / (m1_kg + m2_kg) )**(1.0/5.0)
    
    term2 = 1.0 / (8.0 * np.pi)
    term3 = (G_SI * mc_kg / (C_SI**3))**(-5.0/8.0)
    
    # f_min = (duration / 5.0)^(-3/8) * term2 * term3
    # (f_min / (term2 * term3)) = (duration / 5.0)^(-3/8)
    # (f_min / (term2 * term3))^(-8/3) = duration / 5.0
    
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
        if not hasattr(val, 'shape') and not hasattr(val, 'dtype'): # Python scalar
             val = jnp.array(val)
        elif not isinstance(val, (jax.Array, np.ndarray)) and not ops.is_tensor(val):
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
    
    # DEBUG: Check f_mins
    jax.debug.print("DEBUG: f_mins: {}", f_mins)
    
    # We need f_l for frequency grid, but ripple generates on custom grid or we pass grid?
    # ripple gen_func takes `fs`.
    # We usually define fs based on duration and sample rate.
    
    f_u = sample_rate_hertz / 2.0
    num_samples = int(sample_rate_hertz * duration_seconds)
    num_freqs = num_samples // 2 + 1
    # Use float64 for frequency grid to ensure phase precision
    fs = jnp.linspace(0, f_u, num_freqs, dtype=jnp.float64)
    
    # Mc, eta calculation
    # ms_to_Mc_eta takes (m1, m2) in solar masses
    # Vectorize it
    # ms_to_Mc_eta is JAX-jit compatible? Yes, it's in ripple.
    
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
    
    waveforms = jnp.stack([hp_time, hc_time], axis=1)
    
    # Return JAX array (which Keras 3 can handle as Tensor)
    return waveforms
