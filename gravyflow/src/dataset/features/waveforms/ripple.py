import jax
import jax.numpy as jnp
import numpy as np
from ripple.waveforms import IMRPhenomD
from ripple import ms_to_Mc_eta
from typing import Tuple, List, Union
import logging

# Constants
G_SI = 6.67430e-11
C_SI = 299792458.0
M_SOLAR_SI = 1.98855e30

def calc_minimum_frequency(
    mass_1_msun: float,
    mass_2_msun: float,
    duration_seconds: float
) -> float:
    """
    Calculates minimum frequency based on inputted masses and duration.
    Matches the heuristic used in Cuphenom.
    """
    m1_kg = mass_1_msun * M_SOLAR_SI
    m2_kg = mass_2_msun * M_SOLAR_SI
    
    mc_kg = ( (m1_kg * m2_kg)**3 / (m1_kg + m2_kg) )**(1.0/5.0)
    
    # Formula from cuphenom.c:
    # min_frequency_hertz = pow(((double)duration.seconds/5.0),(-3.0/8.0))*(1.0/(8.0*M_PI))
    #        *(pow((G_SI*MC/(C_SI*C_SI*C_SI)),(-5.0/8.0)));
    
    term1 = (duration_seconds / 5.0)**(-3.0/8.0)
    term2 = 1.0 / (8.0 * np.pi)
    term3 = (G_SI * mc_kg / (C_SI**3))**(-5.0/8.0)
    
    f_min = term1 * term2 * term3
    
    # Clamp to be at least 1.0 Hz if calculated value is lower, 
    # matching cuphenom logic: (1.0 > min_frequency_hertz) + ...
    if f_min < 1.0:
        f_min = 1.0
        
    return f_min

@jax.jit
def _generate_ripple_waveform_jax(
    fs: jnp.ndarray,
    theta: jnp.ndarray,
    f_ref: float
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    return IMRPhenomD.gen_IMRPhenomD_hphc(fs, theta, f_ref)

def generate_ripple_waveform(
    num_waveforms: int = 1,
    sample_rate_hertz: float = 4096.0,
    duration_seconds: float = 2.0,
    mass_1_msun: Union[float, List[float], np.ndarray] = 30.0,
    mass_2_msun: Union[float, List[float], np.ndarray] = 30.0,
    inclination_radians: Union[float, List[float], np.ndarray] = 0.0,
    distance_mpc: Union[float, List[float], np.ndarray] = 1000.0,
    reference_orbital_phase_in: Union[float, List[float], np.ndarray] = 0.0,
    ascending_node_longitude: Union[float, List[float], np.ndarray] = 0.0,
    eccentricity: Union[float, List[float], np.ndarray] = 0.0,
    mean_periastron_anomaly: Union[float, List[float], np.ndarray] = 0.0,
    spin_1_in: Union[List[float], np.ndarray] = [0.0, 0.0, 0.0],
    spin_2_in: Union[List[float], np.ndarray] = [0.0, 0.0, 0.0]
) -> np.ndarray:
    """
    Generates time-domain waveforms using gwripple (IMRPhenomD).
    Returns numpy array of shape (num_waveforms, 2, num_samples).
    """
    
    # Ensure inputs are arrays of correct length
    def ensure_array(val, length, name):
        if np.isscalar(val):
            return np.full(length, val)
        val = np.array(val)
        if len(val) != length:
             # Handle case where single list/tuple was passed for spins but intended for all waveforms?
             # But cuphenom expects spin_1_in to be flattened array of 3*num_waveforms 
             # OR list of 3 values if num_waveforms=1?
             # Let's look at cuphenom.py:
             # spin_1_in = [0.0, 0.0, 0.0] default.
             # assert len(spin_1_in) == len(spin_2_in) == 3*num_waveforms
             pass 
        return val

    # Handle spins specifically as they are 3-vectors
    # Cuphenom expects flattened arrays of size 3*num_waveforms
    spin_1_in = np.array(spin_1_in)
    spin_2_in = np.array(spin_2_in)
    
    if spin_1_in.size != 3 * num_waveforms:
         # If user passed a single [sx, sy, sz], repeat it
         if spin_1_in.size == 3:
             spin_1_in = np.tile(spin_1_in, num_waveforms)
         else:
             raise ValueError(f"spin_1_in size {spin_1_in.size} does not match 3*{num_waveforms}")

    if spin_2_in.size != 3 * num_waveforms:
         if spin_2_in.size == 3:
             spin_2_in = np.tile(spin_2_in, num_waveforms)
         else:
             raise ValueError(f"spin_2_in size {spin_2_in.size} does not match 3*{num_waveforms}")
             
    spin_1_in = spin_1_in.reshape(num_waveforms, 3)
    spin_2_in = spin_2_in.reshape(num_waveforms, 3)
    
    # Only aligned spins for IMRPhenomD (z-component)
    chi1 = spin_1_in[:, 2]
    chi2 = spin_2_in[:, 2]

    mass_1_msun = ensure_array(mass_1_msun, num_waveforms, "mass_1_msun")
    mass_2_msun = ensure_array(mass_2_msun, num_waveforms, "mass_2_msun")
    inclination_radians = ensure_array(inclination_radians, num_waveforms, "inclination_radians")
    distance_mpc = ensure_array(distance_mpc, num_waveforms, "distance_mpc")
    reference_orbital_phase_in = ensure_array(reference_orbital_phase_in, num_waveforms, "reference_orbital_phase_in")
    
    # Calculate frequency grid
    # We need to calculate f_min for each waveform? 
    # Or just use the minimum of all f_mins to be safe?
    # Cuphenom calculates f_min per waveform in C, but here we need a common frequency grid for batching if possible.
    # However, ripple takes `fs` as input.
    # If we want to batch, we need a common `fs`.
    # Let's calculate the global minimum f_min required.
    
    f_mins = []
    for i in range(num_waveforms):
        f_mins.append(calc_minimum_frequency(mass_1_msun[i], mass_2_msun[i], duration_seconds))
    
    f_l = min(f_mins)
    # Ensure f_l is compatible with duration (1/duration spacing?)
    # Cuphenom: min_frequency_hertz = ...
    # It seems cuphenom uses this f_min as the starting frequency for generation?
    # But for FFT we need a grid from 0 or f_min to f_nyquist with df = 1/duration.
    
    df = 1.0 / duration_seconds
    f_u = sample_rate_hertz / 2.0
    
    # We should start from df, but ripple might need f_l > 0.
    # If f_l calculated is < df, we start from df.
    # If f_l > df, we can zero pad below f_l?
    # Ripple IMRPhenomD generation:
    # fs = jnp.arange(f_l, f_u, del_f)
    
    # To perform inverse FFT correctly to get `duration` length signal:
    # We need frequencies [0, df, 2df, ..., f_nyquist]
    # num_samples = sample_rate * duration
    # num_freqs = num_samples // 2 + 1
    
    num_samples = int(sample_rate_hertz * duration_seconds)
    num_freqs = num_samples // 2 + 1
    fs = np.linspace(0, f_u, num_freqs)
    
    # Prepare parameters for Ripple
    # Mc, eta = ms_to_Mc_eta(jnp.array([m1_msun, m2_msun]))
    # theta = [Mc, eta, chi1, chi2, dist_mpc, tc, phic, inclination]
    
    # Batch processing
    waveforms_list = []
    
    # JIT compilation handles batching if we map or vmap, but let's just loop or vmap.
    # Let's use vmap for efficiency.
    
    # Convert to JAX arrays
    m1_jax = jnp.array(mass_1_msun)
    m2_jax = jnp.array(mass_2_msun)
    chi1_jax = jnp.array(chi1)
    chi2_jax = jnp.array(chi2)
    dist_jax = jnp.array(distance_mpc)
    inc_jax = jnp.array(inclination_radians)
    phi_jax = jnp.array(reference_orbital_phase_in)
    tc_jax = jnp.zeros(num_waveforms) # Time of coalescence, usually 0 or centered? 
    # Cuphenom doesn't seem to take tc as input, defaults to something?
    # In cuphenom.c: const float duration_seconds = 2.0; ...
    # It generates waveform.
    
    # We need to handle the "start time" or "coalescence time" to match cuphenom.
    # Cuphenom usually puts the merger at the end or with some padding?
    # "front_padding_duration_seconds" in injection.py suggests we pad.
    # But the raw waveform from cuphenom:
    # It seems to generate the whole duration.
    # Let's assume tc = 0 is fine for now, or we might need to shift.
    # Ripple: tc = 0.0
    
    # Vectorized parameter conversion
    Mc, eta = jax.vmap(ms_to_Mc_eta)(jnp.stack([m1_jax, m2_jax], axis=1))
    
    theta = jnp.stack([
        Mc, 
        eta, 
        chi1_jax, 
        chi2_jax, 
        dist_jax, 
        tc_jax, 
        phi_jax, 
        inc_jax
    ], axis=1)
    
    # Generate waveforms
    # We need to mask frequencies below f_min for each waveform?
    # Ripple might handle f_ref.
    # Let's use f_ref = f_l (the global min).
    
    # We pass the full frequency grid `fs` to ripple.
    # But ripple might be unstable at very low freq?
    # IMRPhenomD valid from some f_min.
    # We should probably zero out frequencies below individual f_min.
    
    # Let's define a wrapper that generates and masks.
    
    @jax.jit
    def generate_batch(fs, theta, f_mins):
        
        def gen_one(theta_i, f_min_i):
            # Generate raw
            hp, hc = IMRPhenomD.gen_IMRPhenomD_hphc(fs, theta_i, f_min_i)
            # Mask below f_min_i
            mask = fs >= f_min_i
            hp = hp * mask
            hc = hc * mask
            return hp, hc

        return jax.vmap(gen_one)(theta, f_mins)

    f_mins_jax = jnp.array(f_mins)
    hp_freq, hc_freq = generate_batch(fs, theta, f_mins_jax)
    
    # Inverse FFT
    # irfft expects (N//2 + 1) complex points and returns N real points.
    # We need to ensure the output length is exactly num_samples.
    
    hp_time = jax.vmap(lambda x: jnp.fft.irfft(x, n=num_samples))(hp_freq)
    hc_time = jax.vmap(lambda x: jnp.fft.irfft(x, n=num_samples))(hc_freq)
    
    # Scale?
    # FFT normalization: numpy/jax irfft is scaled by 1/N? No, standard definition.
    # But frequency domain waveform from ripple is physical strain / Hz?
    # We need to check units.
    # Usually, to get time domain strain from FD strain:
    # h(t) = IFFT(h(f)) * fs (sampling rate)?
    # Or h(f) has units of 1/Hz?
    # If h(f) is the Fourier transform of h(t), then h(t) = int h(f) e^{2pi i f t} df
    # Discrete: h[n] = sum h[k] e^{...} * df
    # irfft computes sum h[k] e^{...} (without df factor usually, or 1/N factor).
    # numpy.fft.irfft: "The inverse of rfft... computed by ... / n"
    # We need to multiply by sample_rate to get correct amplitude if h(f) is spectral density?
    # Wait, h(f) from LAL/Ripple is usually "Fourier transform of h(t)".
    # So h(f) ~ h(t) * T (units: strain * sec).
    # Inverse FFT (discrete) sums up bins.
    # We need to multiply by df (frequency bin width) to approximate the integral?
    # df = 1/T.
    # So h(t) ~ sum h(f) * df.
    # irfft sum is just sum.
    # So we need to multiply by df = sample_rate_hertz / num_samples? No, df = 1/duration.
    # Let's check this scaling.
    # If we use `jnp.fft.irfft`, it divides by N (num_samples).
    # The integral approximation requires multiplying by `df`.
    # But `irfft` is `1/N * sum(...)`.
    # The integral is `sum(...) * df`.
    # So `irfft` result is `sum(...) / N`.
    # We want `sum(...) * df`.
    # So `h(t) = irfft_result * N * df`.
    # `N * df = num_samples * (sample_rate / num_samples) = sample_rate`.
    # So we multiply by `sample_rate_hertz`.
    
    hp_time = hp_time * sample_rate_hertz
    hc_time = hc_time * sample_rate_hertz
    
    # Stack: (num_waveforms, 2, num_samples)
    waveforms = jnp.stack([hp_time, hc_time], axis=1)
    
    return np.array(waveforms)
