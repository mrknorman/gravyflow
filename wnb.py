import tensorflow as tf

@tf.function
def generate_envelopes(num_samples_array: tf.Tensor, max_num_samples: tf.Tensor):
    """
    Generate envelopes using Hann windows.

    Parameters:
    - num_samples_array: tf.Tensor, tensor containing the lengths of each 
        sequence
    - max_num_samples: tf.Tensor, maximum length among all sequences

    Returns:
    - envelopes: tf.Tensor, tensor containing padded Hann windows for each 
        sequence
    """

    # Create a tensor array to store each envelope
    envelopes = tf.TensorArray(
        dtype=tf.float32, size=0, dynamic_size=True, clear_after_read=False
    )

    # Function to generate and store each envelope
    def loop_body(envelopes, num_samples):
        hann_win = tf.signal.hann_window(num_samples)
        padded_hann_win = tf.pad(hann_win, [[max_num_samples - num_samples, 0]])
        return envelopes.write(envelopes.size(), padded_hann_win)

    # Iterate through num_samples_array to generate and store each envelope
    envelopes = tf.foldl(loop_body, num_samples_array, initializer=envelopes)

    # Stack the tensor array into a single tensor
    envelopes = envelopes.stack()

    return envelopes

@tf.function
def generate_white_noise_burst(
    num_waveforms: int,
    sample_rate_hertz: float,
    max_duration_seconds: float,
    duration_seconds: tf.Tensor,
    min_frequency_hertz: tf.Tensor,
    max_frequency_hertz: tf.Tensor
) -> tf.Tensor:
    """Generates white noise bursts with desired frequency range and duration."""
    
    # Casting
    min_frequency_hertz = tf.cast(min_frequency_hertz, tf.float32)
    max_frequency_hertz = tf.cast(max_frequency_hertz, tf.float32)

    # Convert duration to number of samples
    num_samples_array = tf.cast(sample_rate_hertz * duration_seconds, tf.int32)
    max_num_samples = tf.cast(max_duration_seconds * sample_rate_hertz, tf.int32)

    # Generate Gaussian noise
    gaussian_noise = tf.random.normal([num_waveforms, 2, max_num_samples])

    # Create time mask for valid duration
    mask = tf.sequence_mask(num_samples_array, max_num_samples, dtype=tf.float32)
    mask = tf.reverse(mask, axis=[-1])
    mask = tf.expand_dims(mask, axis=1)
    
    # Mask the noise
    white_noise_burst = gaussian_noise * mask

    # Window function
    window = tf.signal.hann_window(max_num_samples)
    windowed_noise = white_noise_burst * window

    # Fourier transform
    noise_freq_domain = tf.signal.rfft(windowed_noise)

    # Frequency index limits
    max_num_samples_f = tf.cast(max_num_samples, tf.float32)
    num_bins = max_num_samples_f // 2 + 1
    nyquist_freq = sample_rate_hertz / 2.0

    min_freq_idx = tf.cast(
        tf.round(min_frequency_hertz * num_bins / nyquist_freq), tf.int32)
    max_freq_idx = tf.cast(
        tf.round(max_frequency_hertz * num_bins / nyquist_freq), tf.int32)

    # Create frequency masks using vectorized operations
    total_freq_bins = max_num_samples // 2 + 1
    freq_indices = tf.range(total_freq_bins, dtype=tf.int32)
    freq_indices = tf.expand_dims(freq_indices, 0)
    min_freq_idx = tf.expand_dims(min_freq_idx, -1)
    max_freq_idx = tf.expand_dims(max_freq_idx, -1)
    lower_mask = freq_indices >= min_freq_idx
    upper_mask = freq_indices <= max_freq_idx
    combined_mask = tf.cast(lower_mask & upper_mask, dtype=tf.complex64)
    combined_mask = tf.expand_dims(combined_mask, axis=1)

    # Filter out undesired frequencies
    filtered_noise_freq = noise_freq_domain * combined_mask

    # Inverse Fourier transform
    filtered_noise = tf.signal.irfft(filtered_noise_freq)
    
    envelopes = generate_envelopes(num_samples_array, max_num_samples)
    envelopes = tf.expand_dims(envelopes, axis=1)

    filtered_noise = filtered_noise * envelopes

    return filtered_noise
    

        