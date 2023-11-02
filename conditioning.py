from dataclasses import dataclass
from enum import Enum, auto

import tensorflow as tf

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
    
    def apply(self, timeseries : tf.Tensor):
        
        return spectrogram(
            timeseries,
            self.num_frame_samples,
            self.num_step_samples,
            self.num_fft_samples
        )
    
@tf.function(jit_compile=True)
def spectrogram(
        timeseries : tf.Tensor, 
        num_frame_samples : int = 256, 
        num_step_samples : int = 128, 
        num_fft_samples : int = 256
    ):
    
    """
    Compute the spectrogram for a given time-series tensor.
    
    Parameters:
    - timeseries: A 1D tensor containing the time-series data.
    - num_frame_samples: The length of each frame for STFT.
    - num_step_samples: The step size (stride) between frames.
    - num_fft_samples: The number of FFT bins to use.
    
    Returns:
    - A 2D tensor representing the spectrogram.
    """
    # Compute the short-time fourier transform (STFT)
    stfts = tf.signal.stft(
        timeseries, 
        frame_length=num_frame_samples, 
        frame_step=num_step_samples, 
        fft_length=num_fft_samples
    )
    
    # Compute the magnitude squared (power) spectrogram
    spectrograms = tf.abs(stfts) ** 2
    
    return spectrograms

def spectrogram_shape(
        input_shape, 
        num_frame_samples = 256, 
        num_step_samples = 128, 
        num_fft_samples = 256
    ):
    
    # Extract the last dimension of the input shape which is num_samples
    num_samples = input_shape[-1]
    
    # Calculate the number of time frames (T)
    num_time_frames = 1 + (num_samples - num_frame_samples) // num_step_samples
    
    # Calculate the number of frequency bins (F)
    num_frequency_bins = num_fft_samples // 2 + 1
    
    # The resulting shape includes the original dimensions and the new spectrogram dimensions
    output_shape = input_shape[:-1] + (num_time_frames, num_frequency_bins)
    
    return output_shape