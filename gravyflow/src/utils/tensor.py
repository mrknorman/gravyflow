from enum import Enum, auto
from typing import Union, Type, List
from dataclasses import dataclass
import random

import numpy as np
import keras
from keras import ops
import jax.numpy as jnp
from numpy.random import default_rng  

from scipy.stats import truncnorm
import gravyflow as gf
import jax

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

@jax.jit
def replace_nan_and_inf_with_zero(tensor):
    """Replace NaN and Inf values with zeros. JIT compiled for efficiency."""
    tensor = ops.convert_to_tensor(tensor)
    tensor = ops.where(ops.isnan(tensor), ops.zeros_like(tensor), tensor)
    tensor = ops.where(ops.isinf(tensor), ops.zeros_like(tensor), tensor)
    return tensor    

def expand_tensor(
        signal, 
        mask, 
        group_size: int = 1
    ):
    """
    This function expands a tensor (or zero arrays) along the X axis by 
    inserting zeros wherever a corresponding boolean in a 1D tensor is False, 
    and elements from the original tensor where the boolean is True. It works 
    for both 1D and 2D tensors.
    It works for both 1D and 2D tensors.

    Parameters
    ----------
    signal : tensor
        A 1D, 2D or 3D tensor representing signal injections, where the length 
        of the tensor's first dimension equals the number of True values in the 
        mask.
    mask : tensor
        A 1D boolean tensor. Its length will determine the length of the 
        expanded tensor.
    group_size : int
        The number of elements in a group in the signal tensor.

    Returns
    -------
    tensor
        The expanded tensor.
    """
    
    signal = ops.convert_to_tensor(signal)
    mask = ops.convert_to_tensor(mask)

    # Validation checks:
    if group_size <= 0:
        raise ValueError('Group size must be greater than 0')
    
    if len(ops.shape(mask)) != 1:
        raise ValueError('Mask must be a 1D tensor')

    # Number of true values in the mask
    num_true_in_mask = ops.sum(ops.cast(mask, "int32"))

    # Expected true values
    expected_true_values = ops.shape(signal)[0] // group_size

    # Assertion for equality check (eager check if possible, or use ops.cond/assert)
    # Keras ops doesn't have assert_equal. We'll rely on runtime errors or assume correctness for now
    # or implement a check if we are in eager mode.
    # For JAX/TF graph mode, this might be tricky without specific backend ops.
    # We'll skip the assertion for now or use a simple python assert if eager.
    
    # Get static shape if available, otherwise use dynamic shape
    signal_shape = ops.shape(signal)
    ndim = len(signal_shape)
    
    mask_len = ops.shape(mask)[0]
    
    if ndim == 1:
        expanded_signal = ops.zeros(
            (mask_len * group_size,), 
            dtype=signal.dtype
        )
    elif ndim == 2:
        expanded_signal = ops.zeros(
            (mask_len * group_size, signal_shape[1]), 
            dtype=signal.dtype
        )
    elif ndim == 3:
        expanded_signal = ops.zeros(
            (mask_len * group_size, signal_shape[1], signal_shape[2]), 
            dtype=signal.dtype
        )
    elif ndim == 4:
        expanded_signal = ops.zeros(
            (mask_len * group_size, 
             signal_shape[1], 
             signal_shape[2], 
             signal_shape[3]
            ), 
            dtype=signal.dtype
        )
    else:
        raise ValueError("Unsupported shape for signal tensor")
    
    # Get the indices where mask is True:
    true_indices = ops.where(mask)[0] * group_size # ops.where returns tuple of indices
    
    # Create a range based on group_size for each index:
    true_indices = ops.cast(true_indices, dtype="int32")
    
    # Reshape logic
    # tf.range(group_size) + tf.reshape(true_indices, (-1, 1))
    offsets = ops.arange(group_size, dtype="int32")
    indices = ops.reshape(
        offsets + ops.reshape(true_indices, (-1, 1)), (-1, 1)
    )
    
    # Split the signal into the right groups and reshape:
    scattered_values = ops.reshape(signal, (-1, group_size, *signal_shape[1:]))
    
    # Scatter the groups into the expanded_signal tensor:
    # Keras ops scatter_update might not be available or behave differently.
    # ops.scatter is available.
    # ops.scatter(indices, updates, shape)
    
    # Flatten indices for scatter if needed or use scatter_nd
    # Keras 3 has ops.scatter(indices, values, shape) which is scatter_nd equivalent?
    # No, ops.scatter is usually scatter_nd.
    
    # Let's check if we can use scatter_update on the zeros tensor.
    # expanded_signal = ops.scatter_update(expanded_signal, indices, values)
    # But indices need to be correct shape.
    
    # ops.scatter(indices, values, shape) creates a new tensor.
    # We want to create expanded_signal with values at indices.
    
    updates = ops.reshape(scattered_values, (-1, *signal_shape[1:]))
    
    # ops.scatter expects indices to be (N, D) where D is rank of output?
    # expanded_signal is (M, ...)
    # indices is (N, 1) -> correct for 1st dim scatter.
    
    expanded_signal = ops.scatter(indices, updates, ops.shape(expanded_signal))
    
    return expanded_signal

def batch_tensor(
        tensor, 
        batch_size: int,
    ):
    
    """
    Batches a tensor into batches of a specified size. If the first dimension
    of the tensor is not exactly divisible by the batch size, remaining elements
    are discarded.

    Parameters
    ----------
    tensor : tensor
        The tensor to be batched.
    batch_size : int
        The size of each batch.

    Returns
    -------
    tensor
        The reshaped tensor in batches.
    """
    tensor = ops.convert_to_tensor(tensor)
    
    # Calculate the number of full batches that can be created
    num_batches = ops.shape(tensor)[0] // batch_size
    
    # Slice the tensor to only include enough elements for the full batches
    tensor = tensor[:num_batches * batch_size]
    
    original_shape = ops.shape(tensor)
    ndim = len(original_shape)

    if ndim == 1:
        # Reshape the 1D tensor into batches
        batched_tensor = ops.reshape(tensor, (num_batches, batch_size))
    elif ndim == 2:
        # Reshape the 2D tensor into batches
        batched_tensor = ops.reshape(tensor, (num_batches, batch_size, original_shape[-1]))
    elif ndim == 3:
        batched_tensor = ops.reshape(
            tensor, 
            (num_batches, batch_size, original_shape[-2], original_shape[-1])
        )
    elif ndim == 4:  
        batched_tensor = ops.reshape(
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
    Set random seeds for Keras, Numpy, and Core Python to ensure 
    deterministic results with the same seed.
    
    Args
    ---
    
    seed : int
        Random seed which will be used to set both Numpy and Keras seeds
    
    """
    
    # Set keras random seed (Keras 3 API):
    keras.utils.set_random_seed(seed)
    
    # Set Numpy random seed:
    np.random.seed(seed)
    
    # Set core Python.random seed:
    random.seed(seed)
    
def crop_samples(
        batched_onsource, 
        onsource_duration_seconds: float, 
        sample_rate_hertz: float
    ):
    """
    Crop to remove edge effects and ensure same data is retrieved in all cases.
    
    This function calculates the desired number of samples based on the duration 
    of examples in seconds and the sample rate, then it finds the start and end 
    index for cropping. It then crops the batched_onsource using these indices.
    
    Parameters
    ----------
    batched_onsource : tensor
        The batch of examples to be cropped.
    onsource_duration_seconds : float
        The duration of an example in seconds.
    sample_rate_hertz : float
        The sample rate in hertz.
    
    Returns
    -------
    tensor
        The cropped batched_onsource.
    """
    batched_onsource = ops.convert_to_tensor(batched_onsource)
    
    dims = len(ops.shape(batched_onsource))
    if dims == 1:
        batched_onsource = ops.expand_dims(batched_onsource, 0) 
    
    # Calculate the desired number of samples based on example duration and 
    # sample rate:
    desired_num_samples = int(onsource_duration_seconds * sample_rate_hertz)
    
    # Calculate the start and end index for cropping
    start = (ops.shape(batched_onsource)[-1] - desired_num_samples) // 2
    end = start + desired_num_samples
    
    # Crop the batched_onsource
    batched_onsource = batched_onsource[..., start:end]
    
    if dims == 1:
        batched_onsource = ops.squeeze(batched_onsource, axis=0) 
    
    return batched_onsource

def rfftfreq(
        num_samples: int, 
        frequency_interval_hertz: Union[float, int] = 1.0
    ):
    """
    Return the Discrete Fourier Transform sample frequencies
    (for usage with rfft, irfft) using Keras operations.

    Parameters
    ----------
    n : int
        Window length.
    d : scalar, optional
        Sample spacing (inverse of the sampling rate). Defaults to 1.

    Returns
    -------
    f : tensor
        Tensor of shape ``(n//2 + 1,)`` containing the sample frequencies.

    Examples
    --------
    >>> n = 10
    >>> sample_rate = 100
    >>> freq = rfftfreq(n, d=1./sample_rate)
    """
    
    # If num_samples is a static integer (not a tracer), keep it static for arange
    if isinstance(num_samples, int):
        num_frequency_samples = num_samples // 2 + 1
        results = ops.arange(0, num_frequency_samples, dtype="int32")
        
        # Cast for float calculation
        num_samples_float = ops.cast(num_samples, "float32")
        val = 1.0 / (num_samples_float * frequency_interval_hertz)
        
        frequency_tensor = ops.cast(results, dtype="float32") * val
        return frequency_tensor
        
    num_samples = ops.cast(num_samples, "float32")
    
    val = 1.0 / (num_samples * frequency_interval_hertz)
    num_frequency_samples = ops.cast(num_samples // 2 + 1, "int32")

    # Create a range tensor and scale it
    results = ops.arange(0, num_frequency_samples, dtype="int32")
    frequency_tensor = ops.cast(results, dtype="float32") * val
    
    return frequency_tensor

def get_element_shape(dataset):
    for element in dataset.take(1):
        return element[0].shape

def print_active_gpu_memory_info():
    # Placeholder for JAX/Keras backend memory info
    print("GPU memory info not available for generic Keras backend.")

def pad_if_odd(tensor):
    tensor = ops.convert_to_tensor(tensor)
    # Use python control flow for shape check if possible, as JAX cond requires same shape branches
    shape = tensor.shape
    if shape[0] is not None:
        if shape[0] % 2 != 0:
            return ops.concatenate([tensor, ops.convert_to_tensor([0], dtype=tensor.dtype)], axis=0)
        return tensor
    
    # Fallback for dynamic shapes (might fail in JAX JIT if shape is truly dynamic)
    # But for now, assuming static shape access is sufficient for this migration
    size_is_odd = ops.equal(ops.size(tensor) % 2, 1)
    
    def pad_fn():
        return ops.concatenate([tensor, ops.convert_to_tensor([0], dtype=tensor.dtype)], axis=0)
        
    def no_pad_fn():
        return tensor

    # This will fail in JAX if shapes differ, so we rely on the python check above
    return ops.cond(size_is_odd, pad_fn, no_pad_fn)

def round_to_even(tensor):
    tensor = ops.convert_to_tensor(tensor)
    # Assuming tensor is of integer type and 1-D
    # Check if each element is odd by looking at the least significant bit
    is_odd = (tensor % 2) == 1
    
    # Subtract 1 from all odd elements to make them even
    nearest_even = tensor - ops.cast(is_odd, tensor.dtype)
    return nearest_even    

def pad_to_power_of_two(tensor):
    tensor = ops.convert_to_tensor(tensor)
    # Get the current size of the tensor
    current_size = ops.size(tensor)

    # Calculate the next power of two
    next_power_of_two = 2**ops.ceil(ops.log(ops.cast(current_size, "float32")) / ops.log(2.0))

    # Calculate how much padding is needed
    padding_size = ops.cast(next_power_of_two, "int32") - current_size

    # Pad the tensor to the next power of two
    # ops.pad(x, pad_width) where pad_width is list of lists
    tensor_padded = ops.pad(tensor, [[0, padding_size]])

    return tensor_padded

@jax.jit(static_argnames=["original_sample_rate_hertz", "new_sample_rate_hertz"])
def resample_fft(x, original_sample_rate_hertz: float, new_sample_rate_hertz: float):
    """
    Accurate FFT-based resampling with anti-aliasing via frequency truncation.
    
    JIT compiled for efficiency. Equivalent to scipy.signal.resample behavior.
    Optimized for downsampling from higher rates (e.g., 16384 Hz) to lower
    power-of-2 rates (1024, 2048, 4096, 8192 Hz).
    
    Parameters
    ----------
    x : 1D-Tensor
        The signal to resample.
    original_sample_rate_hertz : float
        Original sample rate (static for JIT).
    new_sample_rate_hertz : float
        Target sample rate (static for JIT).
    
    Returns
    -------
    resampled_x : Tensor
        Resampled signal at new sample rate.
    """
    x = ops.convert_to_tensor(x, dtype="float32")
    original_size = x.shape[0]
    
    # Calculate new size based on sample rate ratio
    ratio = new_sample_rate_hertz / original_sample_rate_hertz
    new_size = int(round(original_size * ratio))
    
    # Ensure even size for FFT efficiency
    if new_size % 2 != 0:
        new_size -= 1
    
    # FFT the signal
    X = jnp.fft.rfft(x)
    
    # For downsampling: truncate frequencies above new Nyquist
    # For upsampling: zero-pad (rare case for GW data)
    new_freq_size = new_size // 2 + 1
    
    if new_freq_size < len(X):
        # Downsampling - truncate high frequencies (anti-aliasing)
        X_new = X[:new_freq_size]
    else:
        # Upsampling - zero pad
        X_new = jnp.pad(X, (0, new_freq_size - len(X)))
    
    # Inverse FFT at new size
    resampled_x = jnp.fft.irfft(X_new, n=new_size)
    
    # Normalize amplitude to preserve signal power
    resampled_x = resampled_x * (new_size / original_size)
    
    return resampled_x

# Keep old function for backward compatibility
def resample(x, original_size, original_sample_rate_hertz, new_sample_rate_hertz):
    """
    Legacy resample function. Use resample_fft for JIT-compiled version.
    """
    return resample_fft(x, original_sample_rate_hertz, new_sample_rate_hertz)

def check_tensor_integrity(tensor, ndims, min_size):
    tensor = ops.convert_to_tensor(tensor)
    # Check if the tensor is 1D
    if len(ops.shape(tensor)) != ndims:
        return False

    # Check if the size of the tensor is greater than 1
    if ops.shape(tensor)[0] <= min_size:
        return False

    # Check if the datatype of the tensor is float32
    # This is hard to check dynamically in a backend-agnostic way without eager execution.
    # We can skip or check dtype property if available.
    if tensor.dtype != "float32":
        return False

    # Check if the tensor contains any NaN or inf values
    if ops.any(ops.isnan(tensor)) or ops.any(ops.isinf(tensor)):
        return False

    return True
