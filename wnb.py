import tensorflow as tf
import numpy as np

def generate_white_noise_burst(
    num_waveforms : int,
    sample_rate_hertz: float, 
    max_duration_seconds : float,
    duration_seconds: tf.Tensor, 
    min_frequency_hertz: tf.Tensor, 
    max_frequency_hertz: tf.Tensor, 
    enveloped: bool = True
    ):
    
    min_frequency_hertz = tf.cast(min_frequency_hertz, tf.float32)
    max_frequency_hertz = tf.cast(max_frequency_hertz, tf.float32)

    num_samples_array : tf.Tensor = sample_rate_hertz*duration_seconds
    num_samples_array = tf.cast(num_samples_array, tf.int32)
        
    max_num_samples = \
        tf.cast(max_duration_seconds * sample_rate_hertz, tf.int32)
    
    gaussian_noise = tf.random.normal([num_waveforms, max_num_samples])
    
    # Create a mask to zero out elements beyond the desired duration for each waveform
    mask = tf.sequence_mask(num_samples_array, max_num_samples, dtype=tf.float32)
    mask = tf.reverse(mask, axis=[-1])

    # Multiply the noise by the mask to zero out undesired elements:
    white_noise_burst = gaussian_noise * mask
    
    # Create a window function for the maximum duration
    window = tf.signal.hann_window(max_num_samples)

    # Apply the window to the white noise
    windowed_noise = white_noise_burst * window

    # Transform to the frequency domain using RFFT
    noise_freq_domain = tf.signal.rfft(windowed_noise)
    
    max_num_samples_f = \
        tf.cast(max_duration_seconds * sample_rate_hertz, tf.float32)

    num_bins = max_num_samples_f // 2 + 1
    nyquist_freq = sample_rate_hertz / 2.0
    
    min_freq_idx = tf.cast(tf.round(min_frequency_hertz * (max_num_samples_f // 2 + 1) / (sample_rate_hertz / 2)), tf.int32)
    max_freq_idx = tf.cast(tf.round(max_frequency_hertz * (max_num_samples_f // 2 + 1) / (sample_rate_hertz / 2)), tf.int32)
    
    # Create a binary mask to keep only the desired frequencies for each waveform
    def create_mask(min_freq_idx, max_freq_idx, max_num_samples):
        mask = tf.concat([
            tf.zeros(min_freq_idx, dtype=tf.complex64),
            tf.ones(max_freq_idx - min_freq_idx, dtype=tf.complex64),
            tf.zeros((max_num_samples // 2 + 1) - max_freq_idx, dtype=tf.complex64)
        ], axis=0)
        return mask
    
    # Assuming min_frequency_hertz and max_frequency_hertz are 1D tensors of the same length
    num_elements = min_frequency_hertz.shape[0]

    # Use a python list to accumulate the results
    masks_list = []

    for i in range(num_elements):
        mask_result = create_mask(min_freq_idx[i], max_freq_idx[i], max_num_samples)
        masks_list.append(mask_result)

    # Convert the list of tensors back to a single tensor
    masks = tf.stack(masks_list)
    
    #masks = tf.vectorized_map(
       # lambda args: create_mask(args[0], args[1], max_num_samples), (min_freq_idx, max_freq_idx))

    # Multiply the frequency domain signal by the mask to zero out undesired frequencies
    filtered_noise_freq = noise_freq_domain * masks

    # Transform back to the time domain using IRFFT
    filtered_noise = tf.signal.irfft(filtered_noise_freq)
        
    #If the enveloped flag is set, apply the envelope
    if enveloped:
        # Create a Hann window for each waveform based on its duration
        envelope_list = [
            tf.pad(
                tf.signal.hann_window(num_samples), 
                [[max_num_samples - num_samples, 0]]
            ) for num_samples in tf.unstack(num_samples_array)]        
        
        envelope = tf.stack(envelope_list)
        
        filtered_noise = filtered_noise * envelope
            
    return filtered_noise
        