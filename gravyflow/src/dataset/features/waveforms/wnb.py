import keras
from keras import ops
import jax.numpy as jnp
import jax
import numpy as np

def generate_envelopes(
    num_samples_array, 
    max_num_samples
    ):
    """
    Generate envelopes using Hann windows.
    """
    
    # num_samples_array: (N,)
    
    def create_envelope(num_samples):
        # Implement a functional Hann window:
        # w(n) = 0.5 * (1 - cos(2*pi*n / (N-1)))
        
        n = ops.arange(max_num_samples, dtype="float32")
        
        # Mask for valid samples: n < num_samples
        mask = n < num_samples
        
        # Calculate Hann window values
        # Avoid division by zero if num_samples=1 (though unlikely for WNB)
        N_minus_1 = ops.maximum(num_samples - 1.0, 1.0)
        
        # Only compute for valid n.
        # For n >= num_samples, we want 0.
        # For valid n, we want Hann formula.
        
        val = 0.5 * (1.0 - ops.cos(2.0 * np.pi * n / N_minus_1))
        
        return val * ops.cast(mask, "float32")

    # Vectorize over num_samples_array
    envelopes = jax.vmap(create_envelope)(num_samples_array)
    
    return envelopes

def adjust_envelopes_shape(filtered_noise, envelopes):
    # Determine the condition: whether envelopes have one extra sample compared to filtered_noise
    # filtered_noise: (Batch, 2, Time)
    # envelopes: (Batch, Time) -> expand to (Batch, 1, Time)
    
    env_shape = ops.shape(envelopes)
    noise_shape = ops.shape(filtered_noise)
    
    # Check last dim
    cond = env_shape[-1] == noise_shape[-1] + 1
    
    if cond:
        return envelopes[..., :-1]
    return envelopes

def wnb(
    num_waveforms: int,
    sample_rate_hertz: float,
    max_duration_seconds: float,
    duration_seconds,
    min_frequency_hertz,
    max_frequency_hertz,
    seed : int
):
    """
    Generates white noise bursts with user-defined frequency range and duration.
    """
    
    # Casting:
    min_frequency_hertz = ops.cast(min_frequency_hertz, "float32")
    max_frequency_hertz = ops.cast(max_frequency_hertz, "float32")

    # Convert duration to number of samples:
    num_samples_array = ops.cast(
        ops.floor(sample_rate_hertz * duration_seconds), "int32"
    )
    max_num_samples = int(ops.floor(max_duration_seconds * sample_rate_hertz))

    # Generate Gaussian noise:
    # (num_waveforms, 2, max_num_samples)
    key = jax.random.PRNGKey(seed)
    gaussian_noise = jax.random.normal(
        key,
        shape=[num_waveforms, 2, max_num_samples], 
        dtype="float32"
    )
    
    # Create time mask for valid duration:
    # tf.sequence_mask(lengths, maxlen)
    # JAX equivalent:
    indices = ops.arange(max_num_samples)
    # indices: (max_num_samples,)
    # num_samples_array: (num_waveforms,)
    # mask: (num_waveforms, max_num_samples)
    mask = indices[None, :] < num_samples_array[:, None]
    
    # Reverse the mask to align with the end of the buffer.
    # This matches the envelope generation logic which produces end-aligned windows.
    
    mask = ops.flip(mask, axis=-1)
    mask = ops.cast(mask, "float32")
    mask = ops.expand_dims(mask, axis=1) # (Batch, 1, Time)
    
    # Mask the noise:
    white_noise_burst = gaussian_noise * mask

    # Window function:
    # Apply a global Hann window over the entire max_duration.
    
    window = jnp.hanning(max_num_samples)
    window = ops.convert_to_tensor(window, dtype="float32")
    # Expand for broadcasting (1, 1, Time) or just (Time,)
    windowed_noise = white_noise_burst * window

    # Fourier transform:
    # rfft over last axis
    noise_freq_domain = jnp.fft.rfft(windowed_noise)
    
    # Frequency index limits:
    max_num_samples_f = float(max_num_samples)
    num_bins = max_num_samples // 2 + 1
    nyquist_freq = sample_rate_hertz / 2.0
    
    # Calculate the frequency sample indicies of the min and max frequencies: 
    min_freq_idx = ops.cast(
        ops.round(min_frequency_hertz * num_bins / nyquist_freq), "int32")
    max_freq_idx = ops.cast(
        ops.round(max_frequency_hertz * num_bins / nyquist_freq), "int32")

    # Create frequency masks using vectorized operations:
    total_freq_bins = num_bins
    freq_indices = ops.arange(total_freq_bins, dtype="int32")
    freq_indices = ops.expand_dims(freq_indices, 0)
    min_freq_idx = ops.expand_dims(min_freq_idx, -1)
    max_freq_idx = ops.expand_dims(max_freq_idx, -1)
    
    lower_mask = freq_indices >= min_freq_idx
    upper_mask = freq_indices <= max_freq_idx
    combined_mask = lower_mask & upper_mask
    
    # Cast to complex for multiplication
    # combined_mask is boolean.
    combined_mask = ops.cast(combined_mask, "float32")
    combined_mask = ops.cast(combined_mask, "complex64")
    combined_mask = ops.expand_dims(combined_mask, axis=1)

    # Filter out undesired frequencies:
    filtered_noise_freq = noise_freq_domain * combined_mask
    
    # Inverse Fourier transform:
    filtered_noise = jnp.fft.irfft(filtered_noise_freq, n=max_num_samples)
    
    # Generate envelopes:
    # Envelopes are per-waveform based on duration.
    # And they are padded to match the "end-aligned" convention (pad left).
    envelopes = generate_envelopes(num_samples_array, max_num_samples)
    
    # envelopes shape: (Batch, Time)
    envelopes = ops.expand_dims(envelopes, axis=1) # (Batch, 1, Time)

    envelopes = adjust_envelopes_shape(filtered_noise, envelopes)

    # Apply envelope:
    filtered_noise = filtered_noise * envelopes

    return filtered_noise

# Redefine generate_envelopes to be end-aligned
def generate_envelopes(
    num_samples_array, 
    max_num_samples
    ):
    
    def create_envelope(num_samples):
        n = ops.arange(max_num_samples, dtype="float32")
        
        # End aligned: valid when n >= max - num
        start_idx = max_num_samples - num_samples
        mask = n >= start_idx
        
        # We want the Hann window to be computed over 0..num-1, then mapped to start..max-1
        # effective_n = n - start_idx
        effective_n = n - start_idx
        
        N_minus_1 = ops.maximum(num_samples - 1.0, 1.0)
        
        val = 0.5 * (1.0 - ops.cos(2.0 * np.pi * effective_n / N_minus_1))
        
        return val * ops.cast(mask, "float32")

    envelopes = jax.vmap(create_envelope)(num_samples_array)
    return envelopes