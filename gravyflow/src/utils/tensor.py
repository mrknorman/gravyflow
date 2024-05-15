from enum import Enum, auto
from typing import Union, Type, List
from dataclasses import dataclass
import random

import numpy as np
import tensorflow as tf
from numpy.random import default_rng  

from scipy.stats import truncnorm
import gravyflow as gf

class DistributionType(Enum):
    CONSTANT = auto()         
    UNIFORM = auto()
    NORMAL = auto()
    CHOICE = auto()
    LOG = auto()
    POW_TWO = auto()

@dataclass
class Distribution:
    value : Union[int, float] = None
    dtype : Type = float
    min_ : Union[int, float] = None
    max_ : Union[int, float] = None
    mean : float = None
    std : float = None
    possible_values : List = None
    type_ : DistributionType = DistributionType.CONSTANT
    seed : int = None

    def __post_init__(self):

        if self.seed is None:
            self.rng = default_rng(gf.Defaults.seed)
        else:
            self.rng = default_rng(self.seed)

        # Check and adjust min and max values
        if self.min_ is not None or self.max_ is not None:
            if self.min_ is None or self.max_ is None:
                raise ValueError("Both min and max must be provided if one is provided.")
            
            if self.min_ > self.max_:
                self.min_, self.max_ = self.max_, self.min_  # Swap values

    def reseed(self, seed):

        self.seed = seed
        self.rng = default_rng(self.seed)            

    def sample(
        self, 
        num_samples : int = 1
        ) -> Union[List[Union[int, float]], Union[int, float]]:

        match self.type_:
            
            case DistributionType.CONSTANT:

                if self.value is None:
                    raise ValueError(
                        "No constant value given in constant distribution."
                    )
                else:
                    samples = [self.value] * num_samples
            
            case DistributionType.UNIFORM:

                # Check and adjust min and max values
                if self.min_ is not None or self.max_ is not None:
                    if self.min_ is None or self.max_ is None:
                        raise ValueError("Both min and max must be provided if one is provided.")
                    
                    if self.min_ > self.max_:
                        self.min_ = self.max_

                                        
                if self.min_ is None:
                     raise ValueError(
                        "No minumum value given in uniform distribution."
                    )
                elif self.max_ is None:
                    raise ValueError(
                        "No maximum value given in uniform distribution."
                    )
                elif self.min_ == self.max_:
                    samples = [self.min_] * num_samples
                else:                
                    samples = self.rng.uniform(
                            self.min_, 
                            self.max_, 
                            num_samples
                        )
                    
            case DistributionType.NORMAL:

                if self.mean is None:
                    raise ValueError(
                        "No mean value given in normal distribution."
                    )
                elif self.std is None:
                    raise ValueError(
                        "No std value given in normal distribution."
                    )
                else:
                        
                    if self.min_ is None:
                        self.min_ = float("-inf")
                    if self.max_ is None:
                        self.max_ = float("inf")
                       
                    # Check and adjust min and max values
                    if self.min_ is not None or self.max_ is not None:
                        if self.min_ is None or self.max_ is None:
                            raise ValueError("Both min and max must be provided if one is provided.")
                        
                        if self.min_ > self.max_:
                            self.min_ = self.max_

                    if self.min_ == self.max_:
                        samples = [self.min_] * num_samples
                    else:
                        samples = truncnorm.rvs(
                                (self.min_ - self.mean) / self.std,
                                (self.max_ - self.mean) / self.std,
                                loc=self.mean,
                                scale=self.std,
                                size=num_samples,
                                random_state=self.rng.integers(2**32 - 1)
                            )
            case DistributionType.CHOICE:
                if self.possible_values is None:
                    raise ValueError(
                        "No possible values given in choice distribution."
                    )
                samples = self.rng.choice(
                    self.possible_values,
                    size=num_samples
                )
            case DistributionType.LOG:
                if self.min_ is None:
                    raise ValueError(
                        "No minumum value given in log distribution."
                    )
                elif self.max_ is None:
                    raise ValueError(
                        "No maximum value given in log distribution."
                    )
                samples =self.rng.uniform(
                            self.min_, 
                            self.max_, 
                            num_samples
                        )
                samples = 10 ** samples

            case DistributionType.POW_TWO:
                power_low, power_high = map(int, np.log2((self.min_, self.max_)))
                power = self.rng.integers(power_low, high=power_high + 1, size=num_samples)
                samples = 2**power
            
            case _:
                raise ValueError(f'Unsupported distribution type {self.type_}')

        if self.dtype == int:

            if self.type_ == DistributionType.LOG:
                raise ValueError(
                    "Cannot convert log values to ints."
                )
            elif self.type_ == DistributionType.CHOICE:
                raise ValueError(
                    "Cannot convert choice values to ints."
                )

            samples = [int(sample) for sample in samples]
        
        return samples
    
def randomise_arguments(input_dict, func):
    output_dict = {}
    for key, value in input_dict.items():
        output_dict[key] = randomise_dict(value)

    return func(**output_dict), output_dict

@tf.function(jit_compile=True)
def replace_nan_and_inf_with_zero(tensor):
    tensor = tf.where(tf.math.is_nan(tensor), tf.zeros_like(tensor), tensor)
    tensor = tf.where(tf.math.is_inf(tensor), tf.zeros_like(tensor), tensor)
    return tensor    

@tf.function
def expand_tensor(
        signal: tf.Tensor, 
        mask: tf.Tensor, 
        group_size: int = 1
    ) -> tf.Tensor:
    """
    This function expands a tensor (or zero arrays) along the X axis by 
    inserting zeros wherever a corresponding boolean in a 1D tensor is False, 
    and elements from the original tensor where the boolean is True. It works 
    for both 1D and 2D tensors.
    It works for both 1D and 2D tensors.

    Parameters
    ----------
    signal : tf.Tensor
        A 1D, 2D or 3D tensor representing signal injections, where the length 
        of the tensor's first dimension equals the number of True values in the 
        mask.
    mask : tf.Tensor
        A 1D boolean tensor. Its length will determine the length of the 
        expanded tensor.
    group_size : int
        The number of elements in a group in the signal tensor.

    Returns
    -------
    tf.Tensor
        The expanded tensor.
    """
    
    # Validation checks:
    tf.debugging.assert_greater(
        group_size, 
        0, 
        message='Group size must be greater than 0'
    )
    tf.debugging.assert_rank(
        mask, 
        1,
        message='Mask must be a 1D tensor'
    )

    tf.debugging.assert_rank_in(
        signal, 
        [1, 2, 3, 4], 
        message='Signal must be a 1, 2, 3, or 4D tensor in this case'
    )

    # Number of true values in the mask
    num_true_in_mask = tf.reduce_sum(tf.cast(mask, tf.int32))

    # Expected true values
    expected_true_values = signal.shape[0] // group_size

    # TensorFlow assertion for equality check
    tf.debugging.assert_equal(
        num_true_in_mask, 
        expected_true_values,
        message=('Number of groups in signal must match number of True values '
                 'in mask')
    )
        
    # Get static shape if available, otherwise use dynamic shape
    signal_shape = signal.shape
    ndim = len(signal.shape)
    if ndim == 1:
        expanded_signal = tf.zeros(
            tf.shape(mask)[0] * group_size, 
            dtype=signal.dtype
        )
    elif ndim == 2:
        expanded_signal = tf.zeros(
            (tf.shape(mask)[0] * group_size, signal_shape[1]), 
            dtype=signal.dtype
        )
    elif ndim == 3:
        expanded_signal = tf.zeros(
            (tf.shape(mask)[0] * group_size, signal_shape[1], signal_shape[2]), 
            dtype=signal.dtype
        )
    elif ndim == 4:
        expanded_signal = tf.zeros(
            (tf.shape(mask)[0] * group_size, 
             signal_shape[1], 
             signal_shape[2], 
             signal_shape[3]
            ), 
            dtype=signal.dtype
        )
    else:
        raise ValueError("Unsupported shape for signal tensor")
    
    # Get the indices where mask is True:
    true_indices = tf.where(mask) * group_size
    
    # Create a range based on group_size for each index:
    true_indices = tf.cast(true_indices, dtype=tf.int32)
    indices = tf.reshape(
        tf.range(group_size, dtype=tf.int32) 
        + tf.reshape(true_indices, (-1, 1)), (-1, 1)
    )
    
    # Split the signal into the right groups and reshape:
    scattered_values = tf.reshape(signal, (-1, group_size, *signal.shape[1:]))
    
    # Scatter the groups into the expanded_signal tensor:
    expanded_signal = tf.tensor_scatter_nd_update(
        expanded_signal, 
        indices, 
        tf.reshape(scattered_values, (-1, *signal.shape[1:]))
    )
    
    return expanded_signal

@tf.function(jit_compile=True)
def batch_tensor(
        tensor: tf.Tensor, 
        batch_size: int,
    ) -> tf.Tensor:
    
    """
    Batches a tensor into batches of a specified size. If the first dimension
    of the tensor is not exactly divisible by the batch size, remaining elements
    are discarded.

    Parameters
    ----------
    tensor : tf.Tensor
        The tensor to be batched.
    batch_size : int
        The size of each batch.

    Returns
    -------
    tf.Tensor
        The reshaped tensor in batches.
    """
    # Calculate the number of full batches that can be created
    num_batches = tensor.shape[0] // batch_size
    
    # Slice the tensor to only include enough elements for the full batches
    tensor = tensor[:num_batches * batch_size]
    
    original_shape = tf.shape(tensor)

    if len(tensor.shape) == 1:
        # Reshape the 1D tensor into batches
        batched_tensor = tf.reshape(tensor, (num_batches, batch_size))
    elif len(tensor.shape) == 2:
        # Reshape the 2D tensor into batches
        batched_tensor = tf.reshape(tensor, (num_batches, batch_size, original_shape[-1]))
    elif len(tensor.shape) == 3:
        batched_tensor = tf.reshape(
            tensor, 
            (num_batches, batch_size, original_shape[-2], original_shape[-1])
        )
    elif len(tensor.shape) == 4:  
        batched_tensor = tf.reshape(
            tensor, 
            (num_batches, batch_size, original_shape[-3], original_shape[-2], original_shape[-1])
        )
    
    else:
        raise ValueError("Unsupported num dimensions when batching!")

    return batched_tensor

def set_random_seeds(
    seed : int = 1000
    ):
    
    """
    Set random seeds for Tensorflow, Numpy, and Core Python to ensure 
    deterministic results with the same seed. This means that if the seed is the 
    concerved the dataset produced will be identical.
    
    Args
    ---
    
    seed : int
        Random seed which will be used to set both Numpy and TensorFlow seeds
    
    """
    
    # Set tensorflow random seed:
    tf.random.set_seed(seed)
    
    # Set Numpy random seed:
    np.random.seed(seed)
    
    # Set core Python.random seed just in case, I don't think its used:
    random.seed(10)
    
@tf.function(jit_compile=True)
def crop_samples(
        batched_onsource: tf.Tensor, 
        onsource_duration_seconds: float, 
        sample_rate_hertz: float
    ) -> tf.Tensor:
    """
    Crop to remove edge effects and ensure same data is retrieved in all cases.
    
    This function calculates the desired number of samples based on the duration 
    of examples in seconds and the sample rate, then it finds the start and end 
    index for cropping. It then crops the batched_onsource using these indices.
    
    Parameters
    ----------
    batched_onsource : tf.Tensor
        The batch of examples to be cropped.
    onsource_duration_seconds : float
        The duration of an example in seconds.
    sample_rate_hertz : float
        The sample rate in hertz.
    
    Returns
    -------
    tf.Tensor
        The cropped batched_onsource.
    """
    
    dims = len(batched_onsource.shape)
    if dims == 1:
        batched_onsource = tf.expand_dims(batched_onsource, 0) 
    
    # Calculate the desired number of samples based on example duration and 
    # sample rate:
    desired_num_samples = int(onsource_duration_seconds * sample_rate_hertz)
    
    # Calculate the start and end index for cropping
    start = (batched_onsource.shape[-1] - desired_num_samples) // 2
    end = start + desired_num_samples
    
    # Crop the batched_onsource
    batched_onsource = batched_onsource[..., start:end]
    
    if dims == 1:
        batched_onsource = tf.squeeze(batched_onsource) 
    
    return batched_onsource

@tf.function(jit_compile=True)
def rfftfreq(
        num_samples: int, 
        frequency_interval_hertz: Union[float, int] = 1.0
    ) -> tf.Tensor:
    """
    Return the Discrete Fourier Transform sample frequencies
    (for usage with rfft, irfft) using TensorFlow operations.

    Parameters
    ----------
    n : int
        Window length.
    d : scalar, optional
        Sample spacing (inverse of the sampling rate). Defaults to 1.

    Returns
    -------
    f : tf.Tensor
        Tensor of shape ``(n//2 + 1,)`` containing the sample frequencies.

    Examples
    --------
    >>> n = 10
    >>> sample_rate = 100
    >>> freq = rfftfreq_tf(n, d=1./sample_rate)
    """
    
    num_samples = tf.cast(num_samples, tf.float32)
    
    val = 1.0 / (num_samples * frequency_interval_hertz)
    num_frequency_samples = num_samples // 2 + 1

    # Create a range tensor and scale it
    results = tf.range(0, num_frequency_samples, dtype=tf.int32)
    frequency_tensor = tf.cast(results, dtype=tf.float32) * val
    
    return frequency_tensor

def get_element_shape(dataset):
    for element in dataset.take(1):
        return element[0].shape

def print_active_gpu_memory_info():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently active GPU is usually at index 0 when there's only one
            memory_info = tf.config.experimental.get_memory_info('GPU:0')
            print(f"Active GPU memory info: {memory_info}")
        except RuntimeError as e:
            print(e)
    else:
        print("No GPU devices found.")

@tf.function(jit_compile=True)
def pad_if_odd(tensor):
    # Compute whether the size is odd using TensorFlow operations
    size_is_odd = tf.equal(tf.size(tensor) % 2, 1)
    
    # Use tf.cond to conditionally pad the tensor
    tensor = tf.cond(
        size_is_odd,
        lambda: tf.concat([tensor, [0]], 0),  # Pad with one zero if size is odd
        lambda: tensor  # Return the tensor as-is if size is even
    )
    return tensor

@tf.function(jit_compile=True)
def round_to_even(tensor):
    # Assuming tensor is of integer type and 1-D
    # Check if each element is odd by looking at the least significant bit
    is_odd = tf.math.mod(tensor, 2) == 1
    
    # Subtract 1 from all odd elements to make them even
    nearest_even = tensor - tf.cast(is_odd, tensor.dtype)
    return nearest_even    

@tf.function(jit_compile=True)
def pad_to_power_of_two(tensor):
    # Get the current size of the tensor
    current_size = tf.size(tensor)

    # Calculate the next power of two
    next_power_of_two = 2**tf.math.ceil(tf.math.log(tf.cast(current_size, tf.float32)) / tf.math.log(2.0))

    # Calculate how much padding is needed
    padding_size = tf.cast(next_power_of_two, tf.int32) - current_size

    # Pad the tensor to the next power of two
    tensor_padded = tf.pad(tensor, [[0, padding_size]], mode='CONSTANT')

    return tensor_padded

@tf.function(jit_compile=True)
def resample(x, original_size, original_sample_rate_hertz, new_sample_rate_hertz):
    """
    Resample `x` to `num` samples using the Fourier method in TensorFlow, then cut to original size.

    Parameters
    ----------
    x : 1D-Tensor
        The data to be resampled.
    original_size : int
        The original size of the tensor before padding.
    original_sample_rate_hertz : float
        The sample rate of the original signal.
    new_sample_rate_hertz : float
        The desired sample rate after resampling.

    Returns
    -------
    resampled_x : Tensor
        The resampled tensor, cut to the original size.
    """

    fraction = original_sample_rate_hertz / new_sample_rate_hertz

    new_num_samples = tf.math.floordiv(original_size, tf.cast(tf.round(fraction), tf.int32))

    # Round to even for FFT
    new_num_samples = round_to_even(new_num_samples)

    # Perform the FFT
    X = tf.signal.rfft(x)
    
    # Create the new frequency space
    new_X = tf.signal.fftshift(X)
    
    # Slice out the central part of the spectrum to the new size
    new_X = new_X[(original_size - new_num_samples) // 2:(original_size + new_num_samples) // 2]
    
    # Shift back the zero frequency to the beginning
    new_X = tf.signal.ifftshift(new_X)
    
    # Perform the inverse FFT
    resampled_x = tf.signal.irfft(new_X)
    
    # Normalize the amplitude
    resampled_x *= float(new_num_samples) / float(original_size)

    # Cut the resampled tensor back to the original size
    resampled_x_cut = resampled_x[:original_size]
    
    return resampled_x_cut

def check_tensor_integrity(tensor, ndims, min_size):
    # Check if the tensor is 1D
    if tensor.ndim != ndims:
        return False

    # Check if the size of the tensor is greater than 1
    if tensor.shape[0] <= min_size:
        return False

    # Check if the datatype of the tensor is float32
    if tensor.dtype != tf.float32:
        return False

    # Check if the tensor contains any NaN or inf values
    if tf.reduce_any(tf.math.is_nan(tensor)) or tf.reduce_any(tf.math.is_inf(tensor)):
        return False

    return True
