from dataclasses import dataclass
from enum import Enum, auto

import keras
from keras import ops
import jax.numpy as jnp
import numpy as np

@dataclass 
class ConditioningMethod:
    index : str
    shape: tuple = (1,)
    
class ConditioningMethods(Enum):
    WHITEN = ConditioningMethod(0)
    ROLLING_PEARSON = ConditioningMethod(1)
    SPECTROGRM = ConditioningMethod(2)
    
class DataStreams(Enum):
    ONSOURCE = auto()
    OFFSOURCE = auto()

@dataclass
class DataConditioner():
    type_ : ConditioningMethods
    data_stream : DataStreams = DataStreams.ONSOURCE
    
class DataWhitener(DataConditioner):
    None
    
class PearsonCorrelator(DataConditioner):
    None
    
class CreateSpectrogram(DataConditioner):
    num_frame_samples : int = 256
    num_step_samples : int = 128
    num_fft_samples : int = 256
    
    def apply(self, timeseries):
        
        return spectrogram(
            timeseries,
            self.num_frame_samples,
            self.num_step_samples,
            self.num_fft_samples
        )
    
def spectrogram(
        timeseries, 
        num_frame_samples : int = 256, 
        num_step_samples : int = 128, 
        num_fft_samples : int = 256
    ):
    
    """
    Compute the spectrogram for a given time-series tensor.
    """
    timeseries = ops.convert_to_tensor(timeseries)
    
    # STFT implementation using JAX/Keras Ops
    # 1. Frame the signal
    signal_len = ops.shape(timeseries)[-1]
    num_frames = (signal_len - num_frame_samples) // num_step_samples + 1
    
    # Indices: (num_frames, num_frame_samples)
    indices = ops.arange(num_frame_samples)[None, :] + ops.arange(num_frames)[:, None] * num_step_samples
    
    # Handle batch dimensions
    # If timeseries is (Batch, T)
    # Flatten to (Batch, T)
    original_shape = ops.shape(timeseries)
    if len(original_shape) > 1:
        flat_signal = ops.reshape(timeseries, (-1, signal_len))
    else:
        flat_signal = ops.reshape(timeseries, (1, signal_len))
        
    frames = ops.take(flat_signal, indices, axis=-1)
    # frames: (Batch, NumFrames, FrameLen)
    
    # 2. Apply window (Hann)
    window = jnp.hanning(num_frame_samples)
    window = ops.convert_to_tensor(window, dtype="float32")
    
    windowed_frames = frames * window
    
    # 3. FFT
    # tf.signal.stft uses rfft
    # fft_length defaults to smallest power of 2 >= frame_length if not provided, 
    # but here num_fft_samples is provided.
    
    # Use jnp.fft.rfft directly
    stfts = jnp.fft.rfft(windowed_frames, n=num_fft_samples)
    
    # Compute the magnitude squared (power) spectrogram
    spectrograms = ops.abs(stfts) ** 2
    
    # Reshape back if needed
    # If original was (Batch, T), output is (Batch, NumFrames, Freqs)
    # If original was 1D (T,), output is (1, NumFrames, Freqs) -> maybe squeeze?
    # tf.signal.stft returns [..., frames, fft_unique_bins]
    
    if len(original_shape) == 1:
        spectrograms = ops.squeeze(spectrograms, axis=0)
    else:
        # Reshape batch dims
        batch_dims = original_shape[:-1]
        spectrogram_shape = tuple(batch_dims) + (num_frames, num_fft_samples // 2 + 1)
        spectrograms = ops.reshape(spectrograms, spectrogram_shape)
    
    return spectrograms

def spectrogram_shape(
        input_shape, 
        num_frame_samples = 256, 
        num_step_samples = 128, 
        num_fft_samples = 256
    ):
    
    # Extract the last dimension of the input shape which is num_samples
    num_samples = input_shape[-1]

    if (num_frame_samples > num_samples):
        num_frame_samples = num_samples
    
    # Calculate the number of time frames (T)
    num_time_frames = 1 + (num_samples - num_frame_samples) // num_step_samples
    
    # Calculate the number of frequency bins (F)
    num_frequency_bins = num_fft_samples // 2 + 1
    
    # The resulting shape includes the original dimensions and the new spectrogram dimensions
    output_shape = input_shape[:-1] + (num_time_frames, num_frequency_bins)
    
    return output_shape