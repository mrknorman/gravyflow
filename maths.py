from enum import Enum, auto
from typing import Union, Type, List
from dataclasses import dataclass

import numpy as np

import tensorflow as tf

class DistributionType(Enum):
    CONSTANT = auto()
    UNIFORM = auto()
    NORMAL = auto()

@dataclass
class Distribution:
    value : Union[int, float] = None
    dtype : Type = float
    min_ : Union[int, float] = None
    max_ : Union[int, float] = None
    mean : float = None
    std : float = None
    type_ : DistributionType = DistributionType.CONSTANT
    num_samples : int = 1

    def sample(self) -> Union[List[Union[int, float]], Union[int, float]]:
        
        match self.type_:
            
            case DistributionType.CONSTANT:

                if self.value is None:
                    raise ValueError(
                        "No constant value given in constant distribution."
                    )
                else:
                    samples = [self.value] * self.num_samples
            
            case DistributionType.UNIFORM:
                
                if self.min_ is None:
                     raise ValueError(
                        "No minumum value given in uniform distribution."
                    )
                elif self.max_ is None:
                    raise ValueError(
                        "No maximum value given in uniform distribution."
                    )
                else:                
                    samples = \
                        np.random.uniform(
                            self.min_, 
                            self.max_, 
                            self.num_samples
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
                    elif self.max_ is None:
                        self.max_ = float("inf")
                    
                    samples = \
                        truncnorm.rvs(
                            (self.min_ - self.mean) / self.std,
                            (self.max_ - self.mean) / self.std,
                            loc=self.mean_value,
                            scale=self.std,
                            size=self.num_samples
                        )
            
            case _:
                raise ValueError('Unsupported distribution type')

        if self.dtype == int:
            samples = [int(sample) for sample in samples]
        
        samples = samples if self.num_samples > 1 else samples[0]
        return samples
    
def randomise_arguments(input_dict, func):
    output_dict = {}
    for key, value in input_dict.items():
        output_dict[key] = randomise_dict(value)

    return func(**output_dict), output_dict

@tf.function
def replace_nan_and_inf_with_zero(tensor):
    tensor = tf.where(tf.math.is_nan(tensor), tf.zeros_like(tensor), tensor)
    tensor = tf.where(tf.math.is_inf(tensor), tf.zeros_like(tensor), tensor)
    return tensor    

@tf.function
def expand_tensor(signal: tf.Tensor, mask: tf.Tensor) -> tf.Tensor:
    """
    This function expands a tensor along the X axis by inserting zeros wherever a 
    corresponding boolean in a 1D tensor is False, and elements from the original 
    tensor where the boolean is True. It works for both 1D and 2D tensors.

    Parameters
    ----------
    signal : tf.Tensor
        A 1D or 2D tensor representing signal injections, where the length of 
        the tensor's first dimension equals the number of True values in the mask.
    mask : tf.Tensor
        A 1D boolean tensor. Its length will determine the length of the expanded tensor.

    Returns
    -------
    tf.Tensor
        The expanded tensor.

    """
    # Ensure that the signal tensor is 1D or 2D
    assert signal.ndim in (1, 2), 'Signal must be a 1D or 2D tensor'
    
    # Ensure that the mask is 1D
    assert mask.ndim == 1, 'Mask must be a 1D tensor'
    
    # Ensure that the length of the signal tensor matches the number of True 
    # values in the mask
    assert tf.reduce_sum(tf.cast(mask, tf.int32)) == signal.shape[0], \
        'Signal tensor length must match number of True values in mask'
    
    # Create a tensor full of zeros with the final shape
    if signal.ndim == 1:
        expanded_signal = tf.zeros(mask.shape[0], dtype=signal.dtype)
    else: # signal.ndim == 2
        N = signal.shape[1]
        expanded_signal = tf.zeros((mask.shape[0], N), dtype=signal.dtype)

    # Get the indices where mask is True
    indices = tf.where(mask)

    # Scatter the signal tensor elements/rows into the expanded_signal tensor at the 
    # positions where mask is True
    expanded_signal = tf.tensor_scatter_nd_update(expanded_signal, indices, signal)

    return expanded_signal