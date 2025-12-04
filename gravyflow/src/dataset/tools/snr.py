import keras
from keras import ops
import jax.numpy as jnp
import numpy as np

from gravyflow.src.dataset.tools.psd import psd

import jax

@jax.jit(static_argnames=["scalar"])
def find_closest(
        tensor, 
        scalar : float
    ):
    
    tensor = ops.convert_to_tensor(tensor)
    # Calculate the absolute differences between the tensor and the scalar
    diffs = ops.abs(tensor - scalar)
    
    # Get the index of the minimum difference
    closest_index = ops.argmin(diffs)
    
    return closest_index

@jax.jit(static_argnames=["sample_rate_hertz", "fft_duration_seconds", "overlap_duration_seconds", "lower_frequency_cutoff"])
def snr(
        injection, 
        background,
        sample_rate_hertz: float, 
        fft_duration_seconds: float = 4.0, 
        overlap_duration_seconds: float = 2.0,
        lower_frequency_cutoff: float = 20.0
    ):
    """
    Calculate the signal-to-noise ratio (SNR) of a given signal.
    """
    injection = ops.convert_to_tensor(injection)
    background = ops.convert_to_tensor(background)
    
    injection_num_samples = ops.shape(injection)[-1]
    injection_duration_seconds = ops.cast(injection_num_samples, "float32") / sample_rate_hertz
    
    overlap_num_samples = int(sample_rate_hertz * overlap_duration_seconds)
    fft_num_samples = int(sample_rate_hertz * fft_duration_seconds)
    
    # Set the frequency integration limits
    upper_frequency_cutoff = int(sample_rate_hertz / 2.0)

    # Calculate and normalize the Fourier transform of the signal
    # Use jnp.fft.rfft directly
    inj_fft = jnp.fft.rfft(injection) / sample_rate_hertz
    
    df = 1.0 / injection_duration_seconds
    
    # fsamples range
    num_freqs = injection_num_samples // 2 + 1
    fsamples = ops.arange(0, num_freqs, dtype="float32") * df

    # Get rid of DC
    # Slicing along last dimension
    inj_fft_no_dc  = inj_fft[..., 1:]
    fsamples_no_dc = fsamples[1:]

    # Calculate PSD of the background noise
    freqs, psd_val = psd(
        background, 
        sample_rate_hertz = sample_rate_hertz, 
        nperseg           = fft_num_samples, 
        noverlap          = overlap_num_samples,
        mode="mean"
    )
            
    # Interpolate ASD to match the length of the original signal    
    freqs = ops.cast(freqs, "float32")
    
    # ...
    
    def interp_fn(p):
        return jnp.interp(fsamples_no_dc, freqs, p)
        
    # If psd has batch dim, map over it.
    # Check rank
    rank = len(ops.shape(psd_val))

    if rank > 1:
        # Assuming batch is first dim(s). Flatten to (Batch, Freqs)
        psd_flat = ops.reshape(psd_val, (-1, ops.shape(psd_val)[-1]))
        # vmap
        psd_interp = jnp.vectorize(interp_fn, signature='(n)->(m)')(psd_flat)
        # Reshape back
        target_len = ops.shape(fsamples_no_dc)[0]
        psd_interp = ops.reshape(psd_interp, (*ops.shape(psd_val)[:-1], target_len))
    else:
        psd_interp = jnp.interp(fsamples_no_dc, freqs, psd_val)
        
    # Compute the frequency window for SNR calculation
    start_freq_num_samples = find_closest(
        fsamples_no_dc, 
        lower_frequency_cutoff
    )
    end_freq_num_samples = find_closest(
        fsamples_no_dc, 
        upper_frequency_cutoff
    )
    
    # Compute the SNR numerator in the frequency window
    # conj in JAX/Keras
    inj_fft_conj = jnp.conj(inj_fft_no_dc)
    inj_fft_squared = ops.abs(inj_fft_no_dc * inj_fft_conj)   

    # Slicing
    # start/end are tensors (indices).
    # Dynamic slicing in JAX requires lax.dynamic_slice or similar if indices are not static.
    # find_closest returns a tensor index.
    
    # We can use a mask instead of slicing if indices are dynamic.
    # Or ops.slice?
    
    # Let's try to use a mask for robustness in JAX.
    freq_indices = ops.arange(ops.shape(fsamples_no_dc)[0])
    mask = (freq_indices >= start_freq_num_samples) & (freq_indices < end_freq_num_samples)
    mask = ops.cast(mask, dtype=inj_fft_squared.dtype)
    
    snr_numerator = inj_fft_squared * mask
    snr_denominator = psd_interp # Assuming same shape or broadcastable
    
    # If we use mask, we sum over all, but masked out values are 0.
    # Denominator might be 0 where mask is 0? No, denominator is PSD.
    # But we only want sum in the window.
    
    # Avoid division by zero if denominator is zero (unlikely for PSD but possible)
    ratio = snr_numerator / (snr_denominator + 1e-10)
    
    # Apply mask again to ratio to be sure
    ratio = ratio * mask

    # Calculate the SNR
    # Sum over frequency axis (-1)
    integral = ops.sum(ratio, axis=-1)
    
    SNR = ops.sqrt(
        (4.0 / injection_duration_seconds) * integral
    )    
    
    # Handle infs
    SNR = ops.where(ops.isinf(SNR), ops.zeros_like(SNR), SNR)
    
    # Combine if multi-channel (not handled explicitly in original but implied by reduce_sum(SNR**2))
    # Original: SNR = tf.sqrt(tf.reduce_sum(SNR**2, axis = -1))
    # If input was (Batch, Channel, Time), then SNR is (Batch, Channel).
    # Then reduce sum over Channel?
    # Let's assume input is (Batch, Time) -> SNR is (Batch,).
    # If (Batch, Channel, Time) -> SNR is (Batch, Channel).
    
    # The original code had:
    # SNR = tf.sqrt(tf.reduce_sum(SNR**2, axis = -1))
    # This implies SNR was not scalar yet?
    # Ah, `tf.reduce_sum(snr_numerator / snr_denominator, axis = -1)` reduces frequency dim.
    # So if input was (Batch, Channel, Time), SNR is (Batch, Channel).
    # Then reduce_sum(SNR**2, axis=-1) reduces Channel dim.
    # If input was (Batch, Time), SNR is (Batch,). reduce_sum axis=-1 would reduce Batch? No, that would be wrong.
    # It likely expects (Batch, Detectors, Time).
    
    # Let's keep the final reduction logic.
    SNR = ops.sqrt(ops.sum(SNR**2, axis=-1))

    return SNR

from gravyflow.src.utils.tensor import crop_samples

@jax.jit(static_argnames=["sample_rate_hertz", "fft_duration_seconds", "overlap_duration_seconds", "lower_frequency_cutoff", "onsource_duration_seconds"])
def scale_to_snr(
        injection, 
        background,
        desired_snr,
        sample_rate_hertz: float, 
        fft_duration_seconds: float = 4.0, 
        overlap_duration_seconds: float = 2.0,
        lower_frequency_cutoff: float = 20.0,
        onsource_duration_seconds: float = None
    ):
    
    injection = ops.convert_to_tensor(injection)
    background = ops.convert_to_tensor(background)
    desired_snr = ops.convert_to_tensor(desired_snr)
    
    epsilon = 1.0E-7
    
    # If onsource_duration_seconds is provided, crop the signals to that duration
    # to calculate SNR on the "visible" part.
    if onsource_duration_seconds is not None:
        snr_injection = crop_samples(injection, onsource_duration_seconds, sample_rate_hertz)
        snr_background = crop_samples(background, onsource_duration_seconds, sample_rate_hertz)
    else:
        snr_injection = injection
        snr_background = background

    # Calculate the current SNR
    current_snr = snr(
        snr_injection, 
        snr_background,
        sample_rate_hertz, 
        fft_duration_seconds=fft_duration_seconds, 
        overlap_duration_seconds=overlap_duration_seconds,
        lower_frequency_cutoff=lower_frequency_cutoff
    )
    
    # Ensure `desired_snr` and `current_snr` have compatible shapes
    desired_snr = ops.reshape(desired_snr, [-1])  # Shape: [batch_size]
    current_snr = ops.reshape(current_snr, [-1])  # Shape: [batch_size]
    
    # Calculate scale factor
    scale_factor = desired_snr / (current_snr + epsilon)  # Shape: [batch_size]
    
    # Reshape scale_factor to [batch_size, 1, 1] for broadcasting
    # Assuming injection is (Batch, Channel, Time) or similar.
    # If injection is (Batch, Time), we need [batch_size, 1].
    
    rank = len(ops.shape(injection))
    if rank == 2:
        scale_factor = ops.reshape(scale_factor, [-1, 1])
    elif rank == 3:
        scale_factor = ops.reshape(scale_factor, [-1, 1, 1])
    
    # Multiply using broadcasting
    scaled_injection = injection * scale_factor
    
    return scaled_injection