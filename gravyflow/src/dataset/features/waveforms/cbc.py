from enum import Enum

import jax
import jax.numpy as jnp
import numpy as np
import ripplegw
import keras
from keras import ops

jax.config.update("jax_enable_x64", True)

# Constants
G_SI = 6.67430e-11
C_SI = 299792458.0
M_SOLAR_SI = 1.98855e30
# The solar mass in seconds, G * M_sun / c^3 (about 4.93e-6 s).
M_SOLAR_SECONDS = G_SI * M_SOLAR_SI / C_SI**3


def _chirp_mass_seconds(mass_1_msun, mass_2_msun):
    """
    The chirp mass G * M_c / c^3, in seconds, of component masses in solar masses.

    Working in solar masses keeps every intermediate value small: in SI units the
    product of the two masses is about 1e63, beyond float32's range.
    """
    mass_1_msun = jnp.asarray(mass_1_msun)
    mass_2_msun = jnp.asarray(mass_2_msun)
    chirp_mass_msun = (
        (mass_1_msun * mass_2_msun) ** (3.0 / 5.0)
        / (mass_1_msun + mass_2_msun) ** (1.0 / 5.0)
    )
    return chirp_mass_msun * M_SOLAR_SECONDS


# Both functions below use the leading-order (Newtonian) chirp relation between a
# binary's gravitational-wave frequency f and its time to merger tau:
#
#     tau = 5 * m * (8 * pi * m * f) ** (-8 / 3),    m = G * M_c / c^3,
#
# solved for f in one and for tau in the other.

def calc_minimum_frequency(
    mass_1_msun,
    mass_2_msun,
    duration_seconds
):
    """
    The frequency a binary has ``duration_seconds`` before merger, at least 1 Hz.

    Masses are in solar masses. The result has the floating dtype of the inputs and
    is finite for float32 inputs.
    """
    chirp_mass = _chirp_mass_seconds(mass_1_msun, mass_2_msun)
    duration_seconds = jnp.asarray(duration_seconds)

    f_min = (5.0 * chirp_mass / duration_seconds) ** (3.0 / 8.0) / (8.0 * np.pi * chirp_mass)

    # Clamp to be at least 1.0 Hz
    return jnp.maximum(f_min, 1.0)


def calc_duration_from_f_min(
    mass_1_msun,
    mass_2_msun,
    f_min
):
    """
    The time to merger, in seconds, of a binary at frequency ``f_min``.

    Masses are in solar masses. The result has the floating dtype of the inputs and
    is finite for float32 inputs.
    """
    chirp_mass = _chirp_mass_seconds(mass_1_msun, mass_2_msun)
    f_min = jnp.asarray(f_min)

    return 5.0 * chirp_mass * (8.0 * np.pi * chirp_mass * f_min) ** (-8.0 / 3.0)

class Approximant(Enum):
    """Supported CBC approximants. Each value is the name ripple registers the model under."""
    IMRPhenomD = "IMRPhenomD"
    IMRPhenomXAS = "IMRPhenomXAS"
    IMRPhenomPv2 = "IMRPhenomPv2"
    TaylorF2 = "TaylorF2"
    IMRPhenomD_NRTidalv2 = "IMRPhenomD_NRTidalv2"


def _as_approximant(approximant) -> Approximant:
    """Return ``approximant`` as an ``Approximant``, accepting its string value."""
    if isinstance(approximant, Approximant):
        return approximant
    try:
        return Approximant(approximant)
    except ValueError:
        raise ValueError(
            f"Approximant '{approximant}' not supported. "
            f"Available: {[a.value for a in Approximant]}"
        ) from None


def _source_parameters(
    mass_1_msun,
    mass_2_msun,
    spin_1,
    spin_2,
    lambda_1,
    lambda_2,
    distance_mpc,
    reference_orbital_phase,
    inclination_radians,
) -> dict:
    """
    Every CBC source parameter gravyflow supplies, keyed by ripple's parameter names.

    Spins have their Cartesian components on the last axis. Each model takes the
    subset its ``parameter_names`` lists.
    """
    def f64(value):
        return jnp.asarray(value, dtype=jnp.float64)

    mass_1_msun = f64(mass_1_msun)
    mass_2_msun = f64(mass_2_msun)
    spin_1 = f64(spin_1)
    spin_2 = f64(spin_2)
    total_mass = mass_1_msun + mass_2_msun
    product = mass_1_msun * mass_2_msun

    return {
        "M_c": product ** (3.0 / 5.0) / total_mass ** (1.0 / 5.0),
        "eta": product / total_mass ** 2,
        "s1_x": spin_1[..., 0],
        "s1_y": spin_1[..., 1],
        "s1_z": spin_1[..., 2],
        "s2_x": spin_2[..., 0],
        "s2_y": spin_2[..., 1],
        "s2_z": spin_2[..., 2],
        "lambda_1": f64(lambda_1),
        "lambda_2": f64(lambda_2),
        "d_L": f64(distance_mpc),
        "phase_c": f64(reference_orbital_phase),
        "iota": f64(inclination_radians),
    }


def _frequency_domain_polarisations(
    approximant: Approximant,
    frequencies,
    parameters: dict,
    f_ref,
    coalescence_time,
):
    """
    Plus and cross polarisations of one source from ripple, on ``frequencies``.

    ``frequencies`` is a uniform grid starting at 0 Hz. The model is evaluated
    above 0 Hz only, and the 0 Hz bin is zero. Ripple's models place the merger
    at t = 0; a linear phase moves it to ``coalescence_time`` seconds.
    """
    model = ripplegw.waveform(approximant.value, f_ref=f_ref)
    missing = [name for name in model.parameter_names if name not in parameters]
    if missing:
        raise ValueError(  # pragma: no cover
            f"Approximant {approximant.value} needs parameters {missing}, "
            "which gravyflow does not supply."
        )

    # Some models read the grid spacing from their first two frequencies, so the
    # model sees a uniform grid moved up one bin, which never reaches 0 Hz, and
    # its output is rolled back down: bin k holds the model at bin k's
    # frequency, and bin 0 is zeroed. Under jit and vmap this runs at the
    # model's own speed; concatenating a zero bin onto the output of a model
    # evaluated above 0 Hz nearly doubled the cost.
    spacing = frequencies[1] - frequencies[0]
    strain = model(
        frequencies + spacing,
        {name: parameters[name] for name in model.parameter_names},
    )
    time_shift = jnp.exp(-2j * jnp.pi * frequencies * coalescence_time)
    dc = jnp.arange(frequencies.shape[0]) == 0

    hp = jnp.where(dc, 0.0, jnp.roll(strain["p"], 1) * time_shift)
    hc = jnp.where(dc, 0.0, jnp.roll(strain["c"], 1) * time_shift)
    return hp, hc


@jax.jit(static_argnames=["num_waveforms", "sample_rate_hertz", "duration_seconds", "approximant", "f_ref"])
def _generate_cbc_waveform_jit(
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
    approximant: Approximant = Approximant.IMRPhenomD,
    f_ref: float = 20.0,
    coalescence_time: float = None,
    min_frequency_hertz = None
):
    """
    Generates time-domain waveforms using ripple.
    Returns array of shape (num_waveforms, 2, num_samples).

    Each waveform starts at the frequency it has at the start of the buffer,
    ``coalescence_time`` before merger, or at ``min_frequency_hertz`` when that is
    given and higher, with an 8 Hz sine-squared taper above its start.
    """
    approximant = _as_approximant(approximant)

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
    if min_frequency_hertz is not None:
        # No content below the requested minimum frequency (the f_lower of LAL).
        f_mins = jnp.maximum(f_mins, ensure_jax(min_frequency_hertz, num_waveforms))
    
    f_u = sample_rate_hertz / 2.0
    num_samples = int(sample_rate_hertz * duration_seconds)
    num_freqs = num_samples // 2 + 1
    # Use float64 for frequency grid to ensure phase precision
    fs = jnp.linspace(0, f_u, num_freqs, dtype=jnp.float64)

    parameters = _source_parameters(
        m1_jax, m2_jax,
        spin_1_in, spin_2_in,
        lambda1_jax, lambda2_jax,
        dist_jax, phi_jax, inc_jax
    )

    def gen_one(parameters_i, f_min_i, tc_i):
        hp, hc = _frequency_domain_polarisations(approximant, fs, parameters_i, f_ref, tc_i)

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

    hp_freq, hc_freq = jax.vmap(gen_one)(parameters, f_mins, tc_jax)
    
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
    
    # Cast to float32 for downstream compatibility
    # (project_wave JIT is compiled for float32, and some GPU configurations
    # fail graph capture with float64)
    waveforms = waveforms.astype(jnp.float32)
    
    # Return JAX array (which Keras 3 can handle as Tensor)
    return waveforms

def _cbc_buffer_seconds(window_seconds: float) -> float:
    """
    The buffer a CBC window of ``window_seconds`` is generated on, in seconds.

    It is the next power of two at or above twice the window, and depends on
    nothing else, so one window length compiles one generator shape whatever
    the masses. With the merger half a window before the buffer's end and the
    last window kept, at least one window of buffer precedes the kept window.
    That is enough for each waveform's start taper, which spans a fixed
    fraction of its time to merger, to end before the kept window begins.
    """
    return float(2.0 ** np.ceil(np.log2(2.0 * float(window_seconds))))


def _generate_cbc_window(
    num_waveforms: int,
    sample_rate_hertz: float,
    window_seconds: float,
    min_frequency_hertz,
    approximant: Approximant = Approximant.IMRPhenomD,
    f_ref: float = 20.0,
    **source
):
    """
    CBC waveforms of ``window_seconds``, each with its merger at the window's centre.

    The waveforms are generated on the fixed buffer of ``_cbc_buffer_seconds``
    and the last ``window_seconds`` are kept. Each starts at the higher of
    ``min_frequency_hertz`` and its frequency at the buffer's start, so the kept
    window does not depend on the buffer or on the other sources in the batch.
    Returns an array of shape (num_waveforms, 2, int(window_seconds * sample_rate_hertz)).
    """
    buffer_seconds = _cbc_buffer_seconds(window_seconds)
    waveforms = _generate_cbc_waveform_jit(
        num_waveforms=num_waveforms,
        sample_rate_hertz=sample_rate_hertz,
        duration_seconds=buffer_seconds,
        approximant=approximant,
        f_ref=float(f_ref),
        coalescence_time=buffer_seconds - window_seconds / 2.0,
        min_frequency_hertz=min_frequency_hertz,
        **source
    )
    num_samples_keep = int(window_seconds * sample_rate_hertz)
    return waveforms[..., -num_samples_keep:]


def generate_cbc_waveform(
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
    approximant: Approximant = Approximant.IMRPhenomD,
    f_ref: float = 20.0,
    coalescence_time: float = None
):
    """
    Generate CBC waveforms using ripplegw.
    
    This wrapper quantizes duration to standard tiers to minimize JAX recompilations.
    The waveform is generated at the quantized duration, then sliced to the requested size.
    """
    from gravyflow.src.dataset.config import Defaults
    
    # Quantize duration to reduce JAX graph variants
    quantized_duration = Defaults.quantize_duration(duration_seconds)
    
    # Generate at quantized duration
    waveforms = _generate_cbc_waveform_jit(
        num_waveforms=num_waveforms,
        sample_rate_hertz=sample_rate_hertz,
        duration_seconds=quantized_duration,
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
        # The reference frequency configures the model, so it is static.
        f_ref=float(f_ref),
        coalescence_time=coalescence_time
    )
    
    # If we generated at a larger duration, slice to requested size
    if quantized_duration > duration_seconds:
        requested_samples = int(sample_rate_hertz * duration_seconds)
        # Take the last N samples (merger is near the end of the waveform)
        waveforms = waveforms[:, :, -requested_samples:]
    
    return waveforms

