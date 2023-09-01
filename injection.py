# Built-In imports:
from enum import Enum, auto
from dataclasses import dataclass
from typing import Union, List, Dict

# Library imports
import numpy as np
import tensorflow as tf

from .cuphenom.py.cuphenom import generate_phenom_d
from .maths import Distribution, DistributionType, expand_tensor, batch_tensor

@dataclass
class WNBConfig:
    nothing_yet : int = 1.0

@dataclass
class WaveformGenerator:
    snr : Union[float, np.ndarray] = None
    injection_chance : float = 1.0
    front_padding_duration_seconds : float = 0.3
    back_padding_duration_seconds : float = 0.0

@dataclass 
class WaveformParameter:
    index : str
    shape: tuple = (1,)
    
class WaveformParameters(Enum):
    MASS_1_MSUN = WaveformParameter(auto())
    MASS_2_MSUN = WaveformParameter(auto())
    INCLINATION_RADIANS = WaveformParameter(auto())
    DISTANCE_MPC = WaveformParameter(auto())
    REFERENCE_ORBITAL_PHASE_IN = WaveformParameter(auto())
    ASCENDING_NODE_LONGITUDE = WaveformParameter(auto())
    ECCENTRICITY = WaveformParameter(auto())
    MEAN_PERIASTRON_ANOMALY = WaveformParameter(auto())
    SPIN_1_IN = WaveformParameter(auto(), (3,))
    SPIN_2_IN = WaveformParameter(auto(), (3,))
    
    @classmethod
    def get(cls, key):
        member = cls.__members__.get(key.upper()) 
        
        if member is None:
            raise ValueError(f"{key} not found in WaveformParameters")
        
        return member
    
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
                parameter = \
                    WaveformParameters.get(attribute)
                shape = parameter.value.shape[-1]
                
                value.num_samples = num_waveforms * shape
                
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
    configs : List[Union[cuPhenomDGenerator, WNBConfig]]
    sample_rate_hertz : float
    onsource_duration_seconds : float
    crop_duration_seconds : float
    num_examples_per_generation_batch : int
    num_examples_per_batch : int
    scale_factor : float = 10.0E20
    parameters_to_return : List[WaveformParameters] = None
    
    def generate(self):
        
        if not isinstance(self.configs, list):
            self.configs = [self.configs]
                    
        injections = []
        mask = []
        parameters = {key : [] for key in self.parameters_to_return}
        for config in self.configs:
                        
            injection_, mask_, parameters_ = \
                next(self.generate_one(config))
            
            injections.append(injection_)
            mask.append(mask_)
            
            for key in parameters:
                if key in parameters_:
                    parameters[key].append(parameters_[key])
                else:
                    parameters[key].append(tf.zeros([key.shape[-1] * num_examples_per_batch]))
                    
        injections = tf.stack(injections)
        mask = tf.stack(mask)
        
        for key, value in parameters.items():
            parameters[key] = tf.stack(value)
            
        yield injections, mask, parameters
    
    def generate_one(self, config):
        # Create default empty list for requested parameter returns:
        if self.parameters_to_return is None:
            self.parameters_to_return = []
        
        total_duration_seconds : float = \
            self.onsource_duration_seconds + (self.crop_duration_seconds * 2.0)
        total_duration_num_samples : int = \
            int(total_duration_seconds * self.sample_rate_hertz)
        
        # Calculate roll boundaries:
        min_roll_num_samples = \
            int(
                (config.back_padding_duration_seconds + self.crop_duration_seconds)
                * self.sample_rate_hertz
            ) 

        max_roll_num_samples = \
              total_duration_num_samples \
            - int(
                (self.crop_duration_seconds + config.front_padding_duration_seconds)
                * self.sample_rate_hertz
            )
        
        num_batches : int = self.num_examples_per_generation_batch // self.num_examples_per_batch
        
        while 1:
            mask = \
                generate_mask(
                    self.num_examples_per_generation_batch,  
                    config.injection_chance
                )
            num_waveforms = tf.reduce_sum(tf.cast(mask, tf.int32)).numpy()
            
            waveforms, parameters = \
                config.generate(
                    num_waveforms, 
                    self.sample_rate_hertz,
                    total_duration_seconds
                )
            
            # Convert to tensorflow tensor:
            waveforms = \
                tf.convert_to_tensor(waveforms, dtype = tf.float32)
            
            # Scale by arbitrary factor to reduce chance of precision errors:
            waveforms *= self.scale_factor
            
            #Roll Tensor to randomise start time:
            waveforms = \
                roll_vector_zero_padding( 
                    waveforms, 
                    min_roll_num_samples, 
                    max_roll_num_samples
                )
            
            # Create zero filled injections to fill spots where injection did 
            # not generate due to injection masks:
            injections = expand_tensor(waveforms, mask)
            
            # If no parameters requested, skip parameter processing and return
            # empty dict:
            if self.parameters_to_return:
                
                # Retrive parameters that are requested to reduce unneccisary
                # post processing:
                reduced_parameters = {
                    WaveformParameters.get(key) : value 
                        for key, value in parameters.items() 
                        if WaveformParameters.get(key) in self.parameters_to_return
                }
                
                # Conver to tensor and expand parameter dims for remaining parameters:
                expanded_parameters = {}
                for key, parameter in reduced_parameters.items():
                    
                    # Currenly only possible for parameters with same lenght as 
                    # num_waveforms:
                    if key.value.shape[-1] == 1:
                        parameter = tf.convert_to_tensor(parameter)

                        expanded_parameters[key] = \
                            expand_tensor(
                                parameter, 
                                mask
                            )
                                    
                parameters = batch_injection_parameters(
                    expanded_parameters,
                    self.num_examples_per_batch,
                    num_batches
                )
                
            else:
                parameters = [{}] * num_batches
                
            # Split generation batches into smaller batches of size 
            # num_examples_per_batch:
            injections = batch_tensor(injections, self.num_examples_per_batch)
            mask = batch_tensor(mask, self.num_examples_per_batch)
            
            for injections_, mask_, parameters_ in zip(injections, mask, parameters):
                yield injections_, mask_, parameters_
    
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

def batch_injection_parameters(
        injection_parameters: Dict[str, Union[List[float], List[int]]],
        num_injections_per_batch: int,
        num_batches: int
    ) -> List[Dict[str, Union[List[float], List[int]]]]:
    """
    Splits the given dictionary into smaller dictionaries containing N waveforms.

    Parameters
    ----------
    injection_parameters : Dict[str, Union[List[float], List[int]]]
        The input dictionary containing waveforms data.
    num_injections_per_batch : int
        The number of waveforms for each smaller dictionary.
    num_batches : int
        The total number of batches to split into.

    Returns
    -------
    List[Dict[str, Union[List[float], List[int]]]]
        A list of dictionaries containing the split waveforms.
    """

    result = [{} for _ in range(num_batches)]

    for key, value in injection_parameters.items():
        len_multiplier = key.value.shape[-1]

        for i in range(num_batches):
            start_idx = i * num_injections_per_batch * len_multiplier
            end_idx = (i + 1) * num_injections_per_batch * len_multiplier
            result[i][key] = value[start_idx:end_idx]

    return result