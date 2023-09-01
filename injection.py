# Built-In imports:
from dataclasses import dataclass
from typing import Union

# Library imports
import numpy as np
import tensorflow as tf

from .cuphenom.py.cuphenom import generate_phenom_d
from .maths import Distribution, DistributionType, expand_tensor

@dataclass
class WNBConfig:
    nothing_yet : int = 1.0

@dataclass
class WaveformGenerator:
    snr : Union[float, np.ndarray] = None
    injection_chance : float = 1.0
    front_padding_duration_seconds : float = 0.3
    back_padding_duration_seconds : float = 0.0
    
LENGTH_THREE_PARAMTERS = ["spin_1_in", "spin_2_in"]

@dataclass
class cuPhenomDGenerator(WaveformGenerator):
    mass_1_msun : Distribution = \
        Distribution(min_=5.0, max_=95.0, type_=DistributionType.UNIFORM)
    mass_2_msun : Distribution = \
        Distribution(min_=5.0, max_=95.0, type_=DistributionType.UNIFORM)
    inclination_radians : Distribution = \
        Distribution(min_=0.0, max_=np.pi, type_=DistributionType.UNIFORM)
    distance_mpc : Distribution = \
        Distribution(value=1000.0, type_=DistributionType.CONSTANT)
    reference_orbital_phase_in : Distribution = \
        Distribution(value=0.0, type_=DistributionType.CONSTANT)
    ascending_node_longitude : Distribution = \
        Distribution(value=0.0, type_=DistributionType.CONSTANT)
    eccentricity : Distribution = \
        Distribution(value=0.0, type_=DistributionType.CONSTANT)
    mean_periastron_anomaly : Distribution = \
        Distribution(value=0.0, type_=DistributionType.CONSTANT)
    spin_1_in : Distribution = \
        Distribution(value=0.0, type_=DistributionType.CONSTANT)
    spin_2_in : Distribution = \
        Distribution(value=0.0, type_=DistributionType.CONSTANT)
    
    def generate(
            self,
            num_waveforms : int,
            sample_rate_hertz : float,
            duration_seconds : float
        ):
        
        # Draw parameter samples from distributions:
        parameters = {}
        for attribute, value in self.__dict__.items():        
            if is_not_inherited(self, attribute):
                value.num_samples = num_waveforms
                
                if attribute in LENGTH_THREE_PARAMTERS:
                    value.num_samples *= 3
                
                parameters[attribute] = value.sample()
        
        # Generate phenom_d waveform:
        waveforms = \
            generate_phenom_d(num_waveforms, sample_rate_hertz, **parameters) 
        
        # At the moment take only one polarization: 
        waveforms = waveforms[:, 0]
        
        # Reshape to split injections:
        num_samples = int(sample_rate_hertz*duration_seconds)
        waveforms = waveforms.reshape(-1, num_samples)
        
        return waveforms, parameters
    
@dataclass
class InjectionGenerator:
    config : Union[cuPhenomDGenerator, WNBConfig]
    
    def get_generator(
            self,
            sample_rate_hertz : float,
            onsource_duration_seconds : float,
            crop_duration_seconds : float,
            num_examples_per_generation_batch : int,
            num_examples_per_batch : int,
            scale_factor : float = 10.0E20
        ):
        
        total_duration_seconds : float = \
            onsource_duration_seconds + (crop_duration_seconds * 2.0)
        total_duration_num_samples : int = \
            int(total_duration_seconds * sample_rate_hertz)
        
        # Calculate roll boundaries:
        min_roll_num_samples = \
            int(
                (self.config.back_padding_duration_seconds + crop_duration_seconds)
                *sample_rate_hertz
            ) 

        max_roll_num_samples = \
              total_duration_num_samples \
            - int(
                (crop_duration_seconds + self.config.front_padding_duration_seconds)
                * sample_rate_hertz
            )
        
        while 1:
            mask = \
                generate_mask(
                    num_examples_per_generation_batch,  
                    self.config.injection_chance
                )
            num_waveforms = tf.reduce_sum(tf.cast(mask, tf.int32)).numpy()
            
            waveforms, parameters = \
                self.config.generate(
                    num_waveforms, 
                    sample_rate_hertz,
                    total_duration_seconds
                )
            
            # Convert to tensorflow tensor:
            waveforms = \
                tf.convert_to_tensor(waveforms, dtype = tf.float32)
            
            # Scale by arbitrary factor to reduce chance of precision errors:
            waveforms *= scale_factor
            
            #Roll Tensor to randomise start time:
            waveforms = \
                roll_vector_zero_padding( 
                    waveforms, 
                    min_roll_num_samples, 
                    max_roll_num_samples
                )
            
            # Create zero filled injections to fill spots where injection did 
            # not generate due to injection masks
            injections = expand_tensor(waveforms, mask)
            injections = batch_tensor(injections, num_examples_per_batch)
            
            mask = batch_tensor(mask, num_examples_per_batch)
            
            return injections, mask, parameters
    
@tf.function
def roll_vector_zero_padding_(vector, min_roll, max_roll):
    roll_amount = tf.random.uniform(
        shape=(), minval=min_roll, maxval=max_roll, dtype=tf.int32
    )

    # Create a zero vector of the same size as the input
    zeros = tf.zeros_like(vector)

    # Create the rolled vector by concatenating the sliced vector and zeros
    rolled_vector = tf.concat(
        [vector[roll_amount:], zeros[:roll_amount]], axis=-1
    )

    return rolled_vector

@tf.function
def roll_vector_zero_padding(tensor, min_roll, max_roll):
    return tf.map_fn(
        lambda vec: roll_vector_zero_padding_(
            vec, min_roll, max_roll), tensor
    )

@tf.function
def generate_mask(
    num_injections: int, 
    injection_chance: float
) -> tf.Tensor:

    """
    Generate injection masks using TensorFlow.

    Parameters
    ----------
    num_injections : int
        The number of injection masks to generate.
    injection_chance : float
        The probability of an injection being True.

    Returns
    -------
    tf.Tensor
        A tensor of shape (num_injections,) containing the injection masks.
    """
    # Logits for [False, True] categories
    logits = tf.math.log([1.0 - injection_chance, injection_chance])

    # Generate categorical random variables based on logits
    sampled_indices = tf.random.categorical(
        logits=tf.reshape(logits, [1, -1]), 
        num_samples=num_injections
    )

    # Reshape to match the desired output shape
    injection_masks = tf.reshape(sampled_indices, [num_injections])

    # Convert indices to boolean: False if 0, True if 1
    injection_masks = tf.cast(injection_masks, tf.bool)

    return injection_masks

            
def is_not_inherited(instance, attr: str) -> bool:
    """
    Check if an attribute is inherited from a base class.

    Parameters
    ----------
    cls : type
        The class in which to check for the attribute.
    attr : str
        The name of the attribute.

    Returns
    -------
    bool
        True if the attribute is inherited, False otherwise.
    """
    
    # Check if the attribute exists in any of the base classes
    for base in instance.__class__.__bases__:
        if hasattr(base, attr):
            return False
    
    # Check if the attribute exists in the class itself
    if attr in instance.__dict__:
        return True

    return True
            
            
            
