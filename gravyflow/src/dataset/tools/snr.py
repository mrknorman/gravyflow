import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow.signal as tfs
import numpy as np

import gravyflow as gf

@tf.function(jit_compile=True)
def find_closest(tensor, scalar):
    # Calculate the absolute differences between the tensor and the scalar
    diffs = tf.abs(tensor - scalar)
    
    # Get the index of the minimum difference
    closest_index = tf.argmin(diffs)
    
    return closest_index

@tf.function(jit_compile=True)
def snr(
        injection: tf.Tensor, 
        background: tf.Tensor,
        sample_rate_hertz: float, 
        fft_duration_seconds: float = 4.0, 
        overlap_duration_seconds: float = 2.0,
        lower_frequency_cutoff: float = 20.0
    ) -> tf.Tensor:
    """
    Calculate the signal-to-noise ratio (SNR) of a given signal.

    Parameters:
    injection : tf.Tensor
        The input signal.
    background : tf.Tensor
        The time series to use to calculate the asd.
    sample_rate_hertz : float
        The sampling frequency.
    fft_duration_seconds : int, optional
        Length of the FFT window, default is 4.
    overlap_duration_seconds : int, optional
        Overlap of the FFT windows, default is 2.
    lower_frequency_cutoff : float, optional
        Lower cutoff frequency for the SNR calculation, default is 20Hz.

    Returns:
    SNR : tf.Tensor
        The computed signal-to-noise ratio.
    """
    
    injection_num_samples = injection.shape[-1]
    injection_duration_seconds = injection_num_samples / sample_rate_hertz
        
    # Check if input is 1D or 2D
    is_1d = len(injection.shape) == 1
    if is_1d:
        # If 1D, add an extra dimension
        injection = tf.expand_dims(injection, axis=0)
        background = tf.expand_dims(background, axis=0)
        
    overlap_num_samples = int(sample_rate_hertz*overlap_duration_seconds)
    fft_num_samples = int(sample_rate_hertz*fft_duration_seconds)
    
    # Set the frequency integration limits
    upper_frequency_cutoff = int(sample_rate_hertz / 2.0)

    # Calculate and normalize the Fourier transform of the signal
    inj_fft = tf.signal.rfft(injection) / sample_rate_hertz
    df = 1.0 / injection_duration_seconds
    fsamples = tf.range(
        0, (injection_num_samples // 2 + 1), dtype=tf.float32
    ) * df

    # Get rid of DC
    inj_fft_no_dc  = inj_fft[...,1:]
    fsamples_no_dc = fsamples[1:]

    # Calculate PSD of the background noise
    freqs, psd = gf.psd(
        background, 
        sample_rate_hertz = sample_rate_hertz, 
        nperseg           = fft_num_samples, 
        noverlap          = overlap_num_samples,
        mode="mean"
    )
            
    # Interpolate ASD to match the length of the original signal    
    freqs = tf.cast(freqs, tf.float32)
    psd_interp = tfp.math.interp_regular_1d_grid(
        fsamples_no_dc, freqs[0], freqs[-1], psd, axis=-1
    )
        
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
    inj_fft_squared = tf.abs(
        inj_fft_no_dc*tf.math.conj(inj_fft_no_dc)
    )   

    if len(injection.shape) == 2:
        # Use the interpolated ASD in the frequency window for SNR calculation
        snr_numerator = inj_fft_squared[
            :,start_freq_num_samples:end_freq_num_samples
        ]
        snr_denominator = psd_interp[
            :,start_freq_num_samples:end_freq_num_samples
        ]

    elif len(injection.shape) == 3: 
        snr_numerator = inj_fft_squared[
            :, :, start_freq_num_samples:end_freq_num_samples
        ]
        snr_denominator = psd_interp[
            :, :, start_freq_num_samples:end_freq_num_samples
        ]

    # Calculate the SNR
    SNR = tf.math.sqrt(
        (4.0 / injection_duration_seconds) 
        * tf.reduce_sum(snr_numerator / snr_denominator, axis = -1)
    )    
    SNR = tf.where(tf.math.is_inf(SNR), 0.0, SNR)
    
    # If input was 1D, return 1D
    if is_1d:
        SNR = SNR[0]
    elif len(injection.shape) == 3:
        # Calculate network SNR: 
        #print(SNR)
        SNR = tf.sqrt(tf.reduce_sum(SNR**2, axis = -1))

    return SNR

@tf.function(jit_compile=True)
def scale_to_snr(
    injection: tf.Tensor, 
    background: tf.Tensor,
    desired_snr: float,
    sample_rate_hertz: float, 
    fft_duration_seconds: float = 4.0, 
    overlap_duration_seconds: float = 2.0,
    lower_frequency_cutoff: float = 20.0
    ) -> tf.Tensor:
        
    # Small value to prevent divide by zero errors:
    epsilon : float = 1.0E-7
    
    # Calculate the current SNR of the injection in the background, so that
    # it can be scaled to the desired value:
    current_snr = snr(
        injection, 
        background,
        sample_rate_hertz, 
        fft_duration_seconds = fft_duration_seconds, 
        overlap_duration_seconds = overlap_duration_seconds,
        lower_frequency_cutoff = lower_frequency_cutoff
    )

    # Calculate factor required to scale injection to desired SNR:
    scale_factor : tf.Tensor = desired_snr/(current_snr + epsilon)
        
    # Reshape tensor to allow for compatible shapes in the multiplication
    # operation:
    if len(injection.shape) == 2: 
        scale_factor = tf.reshape(scale_factor, (-1, 1))
    elif len(injection.shape) == 3:
        scale_factor = tf.reshape(scale_factor, (-1, 1, 1)) 

    # Scale by scale factor:
    return injection*scale_factor
    
    