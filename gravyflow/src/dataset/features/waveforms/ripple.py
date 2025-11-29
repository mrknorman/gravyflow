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
    Calculates minimum frequency based on inputted masses and duration.
    Matches the heuristic used in the legacy implementation.
    """
    m1_kg = mass_1_msun * M_SOLAR_SI
    m2_kg = mass_2_msun * M_SOLAR_SI
    
    mc_kg = ( (m1_kg * m2_kg)**3 / (m1_kg + m2_kg) )**(1.0/5.0)
    
    # Formula from legacy implementation:
    # min_frequency_hertz = pow(((double)duration.seconds/5.0),(-3.0/8.0))*(1.0/(8.0*M_PI))
    #        *(pow((G_SI*MC/(C_SI*C_SI*C_SI)),(-5.0/8.0)));
    
    term1 = (duration_seconds / 5.0)**(-3.0/8.0)
    term2 = 1.0 / (8.0 * np.pi)
    term3 = (G_SI * mc_kg / (C_SI**3))**(-5.0/8.0)
    
    f_min = term1 * term2 * term3
    
    # Clamp to be at least 1.0 Hz if calculated value is lower, 
    # matching legacy logic: (1.0 > min_frequency_hertz) + ...
    if f_min < 1.0:
        f_min = 1.0
        
    return f_min

APPROXIMANTS = {
    "IMRPhenomD": IMRPhenomD,
    "IMRPhenomXAS": IMRPhenomXAS,
    "IMRPhenomPv2": IMRPhenomPv2,
    "TaylorF2": TaylorF2,
    "IMRPhenomD_NRTidalv2": IMRPhenomD_NRTidalv2
}

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
    spin_2_in: Union[List[float], np.ndarray] = [0.0, 0.0, 0.0],
    lambda_1: Union[float, List[float], np.ndarray] = 0.0,
    lambda_2: Union[float, List[float], np.ndarray] = 0.0,
    approximant: str = "IMRPhenomD",
    f_ref: float = 20.0
) -> np.ndarray:
    """
    Generates time-domain waveforms using gwripple.
    Returns numpy array of shape (num_waveforms, 2, num_samples).
    """
    
    if approximant not in APPROXIMANTS:
        raise ValueError(f"Approximant {approximant} not supported. Available: {list(APPROXIMANTS.keys())}")
        
    waveform_module = APPROXIMANTS[approximant]
    gen_func = getattr(waveform_module, f"gen_{approximant}_hphc")

    # Ensure inputs are arrays of correct length
    def ensure_array(val, length, name):
        if np.isscalar(val):
            return np.full(length, val)
        val = np.array(val)
        if len(val) != length:
             pass 
        return val

    # Handle spins specifically as they are 3-vectors
    spin_1_in = np.array(spin_1_in)
    spin_2_in = np.array(spin_2_in)
    
    if spin_1_in.size != 3 * num_waveforms:
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
    
    mass_1_msun = ensure_array(mass_1_msun, num_waveforms, "mass_1_msun")
    mass_2_msun = ensure_array(mass_2_msun, num_waveforms, "mass_2_msun")
    inclination_radians = ensure_array(inclination_radians, num_waveforms, "inclination_radians")
    distance_mpc = ensure_array(distance_mpc, num_waveforms, "distance_mpc")
    reference_orbital_phase_in = ensure_array(reference_orbital_phase_in, num_waveforms, "reference_orbital_phase_in")
    lambda_1 = ensure_array(lambda_1, num_waveforms, "lambda_1")
    lambda_2 = ensure_array(lambda_2, num_waveforms, "lambda_2")
    
    f_mins = []
    for i in range(num_waveforms):
        f_mins.append(calc_minimum_frequency(mass_1_msun[i], mass_2_msun[i], duration_seconds))
    
    f_l = min(f_mins)
    df = 1.0 / duration_seconds
    f_u = sample_rate_hertz / 2.0
    
    num_samples = int(sample_rate_hertz * duration_seconds)
    num_freqs = num_samples // 2 + 1
    fs = np.linspace(0, f_u, num_freqs)
    
    # Convert to JAX arrays
    m1_jax = jnp.array(mass_1_msun)
    m2_jax = jnp.array(mass_2_msun)
    dist_jax = jnp.array(distance_mpc)
    inc_jax = jnp.array(inclination_radians)
    phi_jax = jnp.array(reference_orbital_phase_in)
    tc_jax = jnp.zeros(num_waveforms)
    lambda1_jax = jnp.array(lambda_1)
    lambda2_jax = jnp.array(lambda_2)
    
    Mc, eta = jax.vmap(ms_to_Mc_eta)(jnp.stack([m1_jax, m2_jax], axis=1))
    
    # Construct theta based on approximant
    if approximant in ["IMRPhenomD", "IMRPhenomXAS"]:
        chi1 = jnp.array(spin_1_in[:, 2])
        chi2 = jnp.array(spin_2_in[:, 2])
        theta = jnp.stack([Mc, eta, chi1, chi2, dist_jax, tc_jax, phi_jax, inc_jax], axis=1)
        
    elif approximant == "IMRPhenomPv2":
        chi1x = jnp.array(spin_1_in[:, 0])
        chi1y = jnp.array(spin_1_in[:, 1])
        chi1z = jnp.array(spin_1_in[:, 2])
        chi2x = jnp.array(spin_2_in[:, 0])
        chi2y = jnp.array(spin_2_in[:, 1])
        chi2z = jnp.array(spin_2_in[:, 2])
        theta = jnp.stack([
            Mc, eta, 
            chi1x, chi1y, chi1z, 
            chi2x, chi2y, chi2z, 
            dist_jax, tc_jax, phi_jax, inc_jax
        ], axis=1)
        
    elif approximant in ["TaylorF2", "IMRPhenomD_NRTidalv2"]:
        chi1 = jnp.array(spin_1_in[:, 2])
        chi2 = jnp.array(spin_2_in[:, 2])
        theta = jnp.stack([
            Mc, eta, 
            chi1, chi2, 
            lambda1_jax, lambda2_jax, 
            dist_jax, tc_jax, phi_jax, inc_jax
        ], axis=1)
    else:
        raise ValueError(f"Parameter packing for {approximant} not implemented.")

    @jax.jit
    def generate_batch(fs, theta, f_mins):
        
        def gen_one(theta_i, f_min_i):
            hp, hc = gen_func(fs, theta_i, f_ref)
            mask = fs >= f_min_i
            hp = hp * mask
            hc = hc * mask
            return hp, hc

        return jax.vmap(gen_one)(theta, f_mins)

    f_mins_jax = jnp.array(f_mins)
    hp_freq, hc_freq = generate_batch(fs, theta, f_mins_jax)
    
    hp_time = jax.vmap(lambda x: jnp.fft.irfft(x, n=num_samples))(hp_freq)
    hc_time = jax.vmap(lambda x: jnp.fft.irfft(x, n=num_samples))(hc_freq)
    
    hp_time = hp_time * sample_rate_hertz
    hc_time = hc_time * sample_rate_hertz
    
    waveforms = jnp.stack([hp_time, hc_time], axis=1)
    
    return np.array(waveforms)
