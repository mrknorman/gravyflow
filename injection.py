from __future__ import annotations

# Built-In imports:
from pathlib import Path
from enum import Enum, auto
from dataclasses import dataclass
from typing import Union, List, Dict, Type
import json
from copy import deepcopy
from warnings import warn

# Library imports
import numpy as np
import tensorflow as tf

from .cuphenom.py.cuphenom import generate_phenom_d
from .wnb import generate_white_noise_burst
from .maths import (Distribution, DistributionType, expand_tensor, batch_tensor,
                    set_random_seeds, crop_samples, replace_nan_and_inf_with_zero)
from .snr import scale_to_snr

def replace_placeholders(
        value: dict, 
        replacements: dict
    ) -> None:
        
    """Replace placeholders in the config dictionary with actual values."""
    for k in ["value", "max_", "type_"]:

        if isinstance(value, dict):
            if k in value:
                value[k] = replacements.get(value[k], value[k])

@dataclass 
class ReturnVariable:
    index : int
    shape: tuple = (1,)

class ReturnVariables(Enum):
    ONSOURCE = ReturnVariable(0)
    WHITENED_ONSOURCE = ReturnVariable(1)
    OFFSOURCE = ReturnVariable(2)
    GPS_TIME = ReturnVariable(3)
    INJECTIONS = ReturnVariable(4)
    WHITENED_INJECTIONS = ReturnVariable(5)
    INJECTION_MASKS = ReturnVariable(6)
    SNR = ReturnVariable(7)
    AMPLITUDE = ReturnVariable(8)
    
    def __lt__(self, other):
        # Implement less-than logic
        return self.value.index < other.value.index
    
@dataclass
class WaveformGenerator:
    snr : Union[float, np.ndarray] = None
    injection_chance : float = 1.0
    front_padding_duration_seconds : float = 0.3
    back_padding_duration_seconds : float = 0.0
    
    def copy(self):
        return deepcopy(self)
    
    @classmethod
    def load(
        cls,
        config_path: Path, 
        sample_rate_hertz: float, 
        onsource_duration_seconds: float, 
        snr: Union[np.ndarray, Distribution] = None
    ) -> Type[cls]:
    
        # Define replacement mapping
        replacements = {
            "pi": np.pi,
            "2*pi": 2.0 * np.pi,
            "constant": DistributionType.CONSTANT,
            "uniform": DistributionType.UNIFORM,
            "normal": DistributionType.NORMAL
        }

        # Load injection config
        with config_path.open("r") as file:
            config = json.load(file)

        # Replace placeholders
        for value in config.values():
            replace_placeholders(value, replacements)
        
        generator = None
        # Construct generator based on type
        match config["type"]:
            case 'PhenomD': 
                generator = cuPhenomDGenerator(
                    **{k: Distribution(**v) for k, v in config.items() if k not in 
                       ["type", "injection_chance", "front_padding_duration_seconds", "back_padding_duration_seconds"]},
                    injection_chance=config["injection_chance"],
                    front_padding_duration_seconds=config["front_padding_duration_seconds"],
                    back_padding_duration_seconds=config["back_padding_duration_seconds"]
                )
            case 'WNB':
                generator = WNBGenerator(
                    **{k: Distribution(**v) for k, v in config.items() if k not in 
                       ["type", "injection_chance", "front_padding_duration_seconds", "back_padding_duration_seconds"]},
                    injection_chance=config["injection_chance"],
                    front_padding_duration_seconds=config["front_padding_duration_seconds"],
                    back_padding_duration_seconds=config["back_padding_duration_seconds"]
                )
            case _:
                raise ValueError("This waveform type is not implemented.")
                
        if snr is not None:
            config["snr"] = snr

        return generator

@dataclass 
class WaveformParameter:
    index : str
    shape: tuple = (1,)
    
class WaveformParameters(Enum):
    
    # CBC parameters:
    MASS_1_MSUN = WaveformParameter(100)
    MASS_2_MSUN = WaveformParameter(101)
    INCLINATION_RADIANS = WaveformParameter(102)
    DISTANCE_MPC = WaveformParameter(103)
    REFERENCE_ORBITAL_PHASE_IN = WaveformParameter(104)
    ASCENDING_NODE_LONGITUDE = WaveformParameter(105)
    ECCENTRICITY = WaveformParameter(106)
    MEAN_PERIASTRON_ANOMALY = WaveformParameter(107)
    SPIN_1_IN = WaveformParameter(108, (3,))
    SPIN_2_IN = WaveformParameter(109, (3,))
    
    # WNB paramters:
    DURATION_SECONDS = WaveformParameter(201)
    MIN_FREQUENCY_HERTZ = WaveformParameter(202)
    MAX_FREQUENCY_HERTZ = WaveformParameter(202)
    
    @classmethod
    def get(cls, key):
        member = cls.__members__.get(key.upper()) 
        
        if member is None:
            raise ValueError(f"{key} not found in WaveformParameters")
        
        return member
    
    def __lt__(self, other):
        # Implement less-than logic
        return self.value.index < other.value.index

@dataclass
class WNBGenerator(WaveformGenerator):
    duration_seconds : Distribution = \
        Distribution(min_=0.1, max_=1.0, type_=DistributionType.UNIFORM)
    min_frequency_hertz: Distribution = \
        Distribution(min_=5.0, max_=95.0, type_=DistributionType.UNIFORM)
    max_frequency_hertz: Distribution = \
        Distribution(min_=5.0, max_=95.0, type_=DistributionType.UNIFORM)

    def generate(
            self,
            num_waveforms: int,
            sample_rate_hertz: float,
            max_duration_seconds: float
    ):
        
        if (num_waveforms > 0):
            
            if self.duration_seconds.max_ > max_duration_seconds \
                and self.duration_seconds.type_ is not DistributionType.CONSTANT:
                
                warn("Max duration distibution is greater than requested "
                     "injection duration. Adjusting", UserWarning)
                self.duration_seconds.max_ = max_duration_seconds
            
            if self.duration_seconds.min_ < 0.0 and \
                self.duration_seconds.type_ is not DistributionType.CONSTANT:
                
                warn("Min duration distibution is less than zero "
                     "injection duration. Adjusting", UserWarning)
                
                self.duration_seconds.min_ = 0.0
            
            # Draw parameter samples from distributions:
            parameters = {}
            for attribute, value in self.__dict__.items():        
                if is_not_inherited(self, attribute):
                    parameter = \
                        WaveformParameters.get(attribute)
                    shape = parameter.value.shape[-1]                
                    parameters[attribute] = tf.convert_to_tensor(value.sample(num_waveforms * shape))
                    
            parameters["max_frequency_hertz"], parameters["min_frequency_hertz"] = \
                np.maximum(
                    parameters["max_frequency_hertz"], 
                    parameters["min_frequency_hertz"]
                ), \
                np.minimum(
                    parameters["max_frequency_hertz"],
                    parameters["min_frequency_hertz"]
                )
                                    
            # Generate WNB waveform at the moment take only one polarization: 
            waveforms = generate_white_noise_burst(
                num_waveforms,
                sample_rate_hertz, 
                max_duration_seconds, 
                **parameters
            )
            
        return waveforms, parameters
    
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
        
        if (num_waveforms > 0):
            
            # Draw parameter samples from distributions:
            parameters = {}
            for attribute, value in self.__dict__.items():        
                if is_not_inherited(self, attribute):
                    parameter = \
                        WaveformParameters.get(attribute)
                    shape = parameter.value.shape[-1]                
                    parameters[attribute] = value.sample(num_waveforms * shape)

            # Generate phenom_d waveform:
        
            waveforms = \
                generate_phenom_d(
                    num_waveforms, 
                    sample_rate_hertz, 
                    duration_seconds,
                    **parameters
                ) 

            # At the moment take only one polarization: 
            waveforms = waveforms[:, 0]

            # Reshape to split injections:
            num_samples = int(sample_rate_hertz*duration_seconds)
            waveforms = waveforms.reshape(-1, num_samples)
        
        return waveforms, parameters
    
@dataclass
class InjectionGenerator:
    configs : List[Union[cuPhenomDGenerator, WNBGenerator]]
    sample_rate_hertz : float
    onsource_duration_seconds : float
    crop_duration_seconds : float
    num_examples_per_generation_batch : int
    num_examples_per_batch : int
    scale_factor : float = 10.0E20
    variables_to_return : List[WaveformParameters] = None
    index : int = 0
    
    def generate(self):
        
        self.variables_to_return = \
            [item for item in self.variables_to_return if \
             isinstance(item.value, WaveformParameter)]
        
        if self.configs is False:
            yield None, None, None
            
        elif not isinstance(self.configs, list):
            self.configs = [self.configs]
            
        iterators = [self.generate_one(config) for config in self.configs]
        
        while 1: 
            injections = []
            mask = []
            parameters = {key : [] for key in self.variables_to_return}
            for iterator in iterators:

                injection_, mask_, parameters_ = \
                    next(iterator)

                injections.append(injection_)
                mask.append(mask_)

                for key in parameters:
                    if key in parameters_:
                        parameters[key].append(parameters_[key])
                    else:
                        parameters[key].append(
                            tf.zeros([key.shape[-1] * num_examples_per_batch])
                        )

            injections = tf.stack(injections)
            mask = tf.stack(mask)

            for key, value in parameters.items():
                parameters[key] = tf.stack(value)

            yield injections, mask, parameters
    
    def generate_one(self, config):
        # Create default empty list for requested parameter returns:
        if self.variables_to_return is None:
            self.variables_to_return = []
        
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
        
        num_batches : int = \
            self.num_examples_per_generation_batch // self.num_examples_per_batch
        
        while 1:
            mask = \
                generate_mask(
                    self.num_examples_per_generation_batch,  
                    config.injection_chance
                )
            num_waveforms = tf.reduce_sum(tf.cast(mask, tf.int32)).numpy()
            
            if num_waveforms > 0:
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
            else:
                injections = tf.zeros(shape=(num_batches, total_duration_num_samples))
                parameters = { }

            # If no parameters requested, skip parameter processing and return
            # empty dict:
            if self.variables_to_return:

                # Retrive parameters that are requested to reduce unneccisary
                # post processing:
                reduced_parameters = {
                    WaveformParameters.get(key) : value 
                        for key, value in parameters.items() 
                        if WaveformParameters.get(key) in self.variables_to_return
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
                
    def generate_snrs_(
        self,
        mask: tf.Tensor,
        config : WaveformGenerator
    ) -> tf.Tensor:
    
        """
        Generate Signal-to-Noise Ratios (SNRs) given a mask, generator and example 
        index.

        Parameters
        ----------
        mask : Tensor
            A tensor representing the injection mask.
            
        Returns
        -------
        Tensor
            A tensor representing generated SNRs.
        """

        mask = mask.numpy()

        num_injections = np.sum(mask)
        
        match config.snr:
            case np.ndarray():

                snrs = []
                for index in range(self.index, self.index + num_injections):
                    if index < len(config.snr):
                        snrs.append(config.snr[index])
                    else:
                        snrs.append(config.snr[-1])
                        
                    self.index += 1

            case Distribution():
                snrs = config.snr.sample(num_injections)

            case _:
                raise ValueError(f"Unsupported SNR type: {type(config.snr)}!") 

        snrs = tf.convert_to_tensor(snrs, dtype = tf.float32)
                
        snrs = \
            expand_tensor(
                snrs,
                mask
            )

        return snrs
    
    def generate_snrs(
        self,
        masks : tf.Tensor
        ):
        
        snrs = []
        for mask, config in zip(masks, self.configs):
            snrs.append(
                self.generate_snrs_(
                    mask,
                    config
                )
            )
            
        return tf.stack(snrs)
    
    def add_injections_to_onsource(
            self,
            injections : tf.Tensor,
            mask : tf.Tensor,
            onsource : tf.Tensor,
            variables_to_return : List
        ) -> tf.Tensor:
        
        # Generate SNR values for injections based on inputed config values:
        snrs = self.generate_snrs(mask)
        
        amplitudes = []
        cropped_injections = []
        for injections_, mask_, snrs_ in zip(injections, mask, snrs):

            # Scale injections to SNR:
            scaled_injections = scale_to_snr(
                injections_, 
                onsource,
                snrs_,
                self.sample_rate_hertz,
                fft_duration_seconds=1.0,
                overlap_duration_seconds=0.5
            )
            
            scaled_injections = replace_nan_and_inf_with_zero(scaled_injections)
            
            tf.debugging.check_numerics(scaled_injections, f"NaN detected in scaled_injections'.")

            # Add scaled injections to onsource:
            onsource += scaled_injections
            
            if ReturnVariables.AMPLITUDE in variables_to_return:
                # Calculate amplitude of scaled injections:
                amplitudes.append(
                    tf.reduce_max(tf.abs(scaled_injections), axis=1)
                )
            if (ReturnVariables.INJECTIONS in variables_to_return) or \
                (ReturnVariables.WHITENED_INJECTIONS in variables_to_return):
                # Crop injections so that they appear the same size as output 
                # onsource:
                cropped_injections.append(
                    crop_samples(
                        scaled_injections, 
                        self.onsource_duration_seconds, 
                        self.sample_rate_hertz
                    )
                )
        
        if ReturnVariables.AMPLITUDE in variables_to_return:
            amplitudes = tf.stack(amplitudes)
        else: 
            amplitudes = None
        
        if (ReturnVariables.INJECTIONS in variables_to_return) or \
            (ReturnVariables.WHITENED_INJECTIONS in variables_to_return):
            cropped_injections = tf.stack(cropped_injections)
        else:
            cropped_injections = None
            
        if (ReturnVariables.SNR not in variables_to_return):
            snrs = None
        
        return onsource, cropped_injections, amplitudes, snrs
    
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