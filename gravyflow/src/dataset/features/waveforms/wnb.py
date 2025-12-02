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
        # num_samples is a scalar tensor
        # jnp.hanning requires static integer size if jit-compiled?
        # Or we can use dynamic slice/pad.
        
        # Since max_num_samples is fixed for the batch, we can generate a large window and slice?
        # Or generate window of size num_samples?
        # JAX dynamic shapes are tricky.
        
        # Alternative: Generate full window of max_num_samples, but that's not right.
        # We need a window of length `num_samples`, padded to `max_num_samples`.
        
        # If we use jnp.hanning(N), N must be static for JIT.
        # But num_samples varies per waveform.
        
        # We can implement a functional Hann window:
        # w(n) = 0.5 * (1 - cos(2*pi*n / (N-1)))
        
        n = ops.arange(max_num_samples, dtype="float32")
        
        # Mask for valid samples: n < num_samples
        mask = n < num_samples
        
        # Calculate Hann window values
        # Avoid division by zero if num_samples=1 (though unlikely for WNB)
        N_minus_1 = ops.maximum(num_samples - 1.0, 1.0)
        
        # Only compute for valid n, but we can compute for all and mask.
        # However, for n >= num_samples, we want 0.
        # For valid n, we want Hann formula.
        
        val = 0.5 * (1.0 - ops.cos(2.0 * np.pi * n / N_minus_1))
        
        return val * ops.cast(mask, "float32")

    # Vectorize over num_samples_array
    envelopes = jax.vmap(create_envelope)(num_samples_array)
    
    return envelopes

def adjust_envelopes_shape(filtered_noise, envelopes):
    # Determine the condition: whether envelopes have one extra sample compared to filtered_noise
    # filtered_noise: (Batch, 2, Time) ? Or (Batch, Time)?
    # wnb returns (Batch, 2, Time).
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
    
    # tf.reverse(mask, axis=[-1]) ? 
    # Original code: mask = tf.reverse(mask, axis=[-1])
    # Why reverse? Maybe to align with some padding convention?
    # Usually sequence mask is [1, 1, ..., 0, 0].
    # If reversed: [0, 0, ..., 1, 1].
    # This implies the burst is at the END of the buffer?
    # Let's check injection.py usage.
    # WNBGenerator has front/back padding.
    # But wnb function itself...
    # "The function first generates Gaussian noise... A frequency mask... An envelope function..."
    
    # If I reverse the mask, I keep the END of the noise?
    # Gaussian noise is random, so start vs end doesn't matter much distribution-wise.
    # But envelope application matters.
    # generate_envelopes uses Hann window padded.
    # tf.pad(hann_win, [[max - num, 0]]) -> Pads at the BEGINNING (left).
    # So the window is at the END.
    # So the mask should also be at the END.
    # Yes, tf.reverse makes [0, 0, 1, 1].
    
    mask = ops.flip(mask, axis=-1)
    mask = ops.cast(mask, "float32")
    mask = ops.expand_dims(mask, axis=1) # (Batch, 1, Time)
    
    # Mask the noise:
    white_noise_burst = gaussian_noise * mask

    # Window function:
    # tf.signal.hann_window(max_num_samples)
    # This is a single window for the whole buffer?
    # Original code: window = tf.signal.hann_window(max_num_samples)
    # windowed_noise = white_noise_burst * window
    # This applies a global Hann window over the entire max_duration.
    
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
    
    # generate_envelopes logic I wrote above:
    # n < num_samples -> [1, 1, 0, 0] (start aligned)
    # But original code: tf.pad(hann, [[max-num, 0]]) -> [0, 0, 1, 1] (end aligned).
    
    # My generate_envelopes implementation:
    # n = arange(max)
    # mask = n < num_samples -> [1, 1, 0, 0]
    # val = ...
    # So my implementation produces START aligned envelopes.
    # I need to reverse/flip them to match END alignment?
    # Or change the logic to produce end-aligned.
    
    # Let's fix generate_envelopes logic to be end-aligned.
    # We want valid values for indices [max-num, max).
    # i.e. n >= max_num_samples - num_samples.
    
    # Let's rewrite generate_envelopes logic inside wnb or update the function.
    # I will update the function in the file content below.
    
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