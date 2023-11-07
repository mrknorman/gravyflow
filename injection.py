from __future__ import annotations

# Built-In imports:
from dataclasses import dataclass
import logging
from pathlib import Path
from enum import Enum, auto
from dataclasses import dataclass
from typing import Union, List, Dict, Type, List
import json
from copy import deepcopy
from warnings import warn

# Library imports:
import numpy as np
import tensorflow as tf

# Local imports:
import gravyflow as gf

class ScalingOrdinality(Enum):
    BEFORE_PROJECTION = auto()
    AFTER_PROJECTION = auto()

@dataclass
class ScalingType:
    index : int
    ordinality : ScalingOrdinality
    shape: tuple = (1,)

class ScalingTypes(Enum):
    SNR = ScalingType(1, ScalingOrdinality.AFTER_PROJECTION)
    HRSS = ScalingType(2,  ScalingOrdinality.BEFORE_PROJECTION)
    HPEAK = ScalingType(3, ScalingOrdinality.BEFORE_PROJECTION)
    
    @classmethod
    def get(cls, key):
        member = cls.__members__.get(key.upper()) 
        
        if member is None:
            raise ValueError(f"{key} not found in WaveformParameters")
        
        return member

@dataclass
class ScalingMethod:
    value : Union[gf.Distribution, np.ndarray]
    type_ : ScalingTypes
    
    def scale(
        self,
        injections : tf.Tensor, 
        onsource : tf.Tensor,
        scaling_parameters : tf.Tensor,
        sample_rate_hertz : float
        ):
        
        scaled_injections = None
        
        match self.type_:
            
            case ScalingTypes.SNR:
                scaled_injections = gf.scale_to_snr(
                    injections, 
                    onsource,
                    scaling_parameters,
                    sample_rate_hertz,
                    fft_duration_seconds=1.0,
                    overlap_duration_seconds=0.5
                )

            case ScalingTypes.HRSS:
                scaled_injections = scale_to_hrss(
                    injections,
                    scaling_parameters,
                )
                
            case ScalingTypes.HPEAK:
                scaled_injections = scale_to_hpeak(
                    injections,
                    scaling_parameters,
                )

            case _:
                raise ValueError(f"Scaling type {method.type_} not recognised.")
        
        if scaled_injections is not None:
            scaled_injections = gf.replace_nan_and_inf_with_zero(
                scaled_injections
            )    
            tf.debugging.check_numerics(
                scaled_injections, 
                f"NaN detected in scaled_injections'."
            )
        
        return scaled_injections

@tf.function(jit_compile=True)
def calculate_hrss(
    injection: tf.Tensor
    ):
    
    # Return the root sum sqaure of the inputted injections:
    return tf.sqrt(
        tf.reduce_sum(
            tf.reduce_sum(injection*injection, axis = 1), 
            axis = -1
        )
    )

@tf.function(jit_compile=True)
def calculate_hpeak(
    injection: tf.Tensor
    ):
    
    # Return the root sum sqaure of the inputted injections:
    return tf.reduce_max(tf.abs(injection), axis=-1)

@tf.function(jit_compile=True)
def scale_to_hrss(
    injection: tf.Tensor, 
    desired_hrss: float
    ) -> tf.Tensor:
    
    # Small value to prevent divide by zero errors:
    epsilon = 1.0E-7
    
    # Calculate the current HRSS of the injection in the background, so that
    # it can be scaled to the desired value:
    current_hrss = calculate_hrss(
        injection
    )
    
    # Calculate factor required to scale injection to desired HRSS:
    scale_factor = desired_hrss/(current_hrss + epsilon)
    
    # Reshape tensor to allow for compatible shapes in the multiplication
    # operation:
    if len(scale_factor.shape) == 1: 
        scale_factor = tf.reshape(scale_factor, (-1, 1))
        
    scale_factor = tf.expand_dims(scale_factor, axis = 1)
    
    # Return injection scaled by scale factor:
    return injection*scale_factor

@tf.function(jit_compile=True)
def scale_to_hpeak(
    injection: tf.Tensor, 
    desired_hrss: float
    ) -> tf.Tensor:
    
    # Small value to prevent divide by zero errors:
    epsilon = 1.0E-7
    
    # Calculate the current HRSS of the injection in the background, so that
    # it can be scaled to the desired value:
    current_hpeak = calculate_hrss(
        injection
    )
    
    # Calculate factor required to scale injection to desired HRSS:
    scale_factor = desired_hrss/(current_hrss + epsilon)
    
    # Reshape tensor to allow for compatible shapes in the multiplication
    # operation:
    if len(scale_factor.shape) == 1: 
        scale_factor = tf.reshape(scale_factor, (-1, 1))
    
    # Return injection scaled by scale factor:
    return injection*scale_factor

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
    ROLLING_PEARSON_ONSOURCE = ReturnVariable(7)
    SPECTROGRAM_ONSOURCE = ReturnVariable(8)
    
    def __lt__(self, other):
        # Implement less-than logic
        return self.value.index < other.value.index

@dataclass
class WaveformGenerator:
    scaling_method : ScalingMethod = None
    injection_chance : float = 1.0
    front_padding_duration_seconds : float = 0.3
    back_padding_duration_seconds : float = 0.0
    scale_factor : float = None
    network : Union[List[IFOs], gf.Network, Path] = None
        
    def __post_init__(self):
        self.network = self.init_network(self.network)
    
    @classmethod
    def init_network(cls, network):
        
        match network:
            case list():
                network = gf.Network(network)

            case Path():
                network = gf.Network.load(network)
                
            case None | gf.Network():
                pass
            
            case _:
                raise ValueError(
                    ("Unable to initiate network with this type: "
                    f"{type(network)}.")
                )
                
        return network
    
    def copy(self):
        return deepcopy(self)
    
    @classmethod
    def load(
        cls,
        config_path: Path, 
        sample_rate_hertz: float = None, 
        onsource_duration_seconds: float = None, 
        scaling_method: ScalingMethod = None,
        scale_factor : float = None,
        network : Union[List[IFOs], gf.Network, Path] = None
    ) -> Type[cls]:

        if sample_rate_hertz is None:
            sample_rate_hertz = gf.Defaults.sample_rate_hertz
        if onsource_duration_seconds is None:
            onsource_duration_seconds = gf.Defaults.onsource_duration_seconds
        
        # Define replacement mapping
        replacements = {
            "pi": np.pi,
            "2*pi": 2.0 * np.pi,
            "constant": gf.DistributionType.CONSTANT,
            "uniform": gf.DistributionType.UNIFORM,
            "normal": gf.DistributionType.NORMAL,
            "hrss" : ScalingTypes.HRSS,
            "hpeak" : ScalingTypes.HPEAK,
            "snr" : ScalingTypes.SNR
        }

        # Load injection config
        with config_path.open("r") as file:
            config = json.load(file)

        # Replace placeholders
        for value in config.values():
            gf.replace_placeholders(value, replacements)
            
        if scaling_method is not None:
            config["scaling_method"] = scaling_method
            
            if "scaling_distribution" in config:
                config.pop("scaling_distribution")
            if "scaling_type" in config:
                config.pop("scaling_type")
            
        elif "scaling_type" and "scaling_distribution" in config:
            config["scaling_method"] = ScalingMethod(
                gf.Distribution(
                    **config.pop("scaling_distribution"),
                ),
                config.pop("scaling_type")
            )
        else:
            raise ValueError("Missing Scaling Type!")
                        
        if scale_factor is not None:
            config["scale_factor"] = scale_factor
        
        generator = None
        # Construct generator based on type:
        
        waveform_cls = None
        match config.pop("type"):
            case 'PhenomD': 
                waveform_cls = cuPhenomDGenerator
            case 'WNB':
                waveform_cls = WNBGenerator
            case _:
                raise ValueError("This waveform type is not implemented.")
                
        generator = waveform_cls(
            scaling_method=config.pop("scaling_method"),
            scale_factor=config.pop("scale_factor"),
            network=cls.init_network(network),
            injection_chance=config.pop("injection_chance"),
            front_padding_duration_seconds=config.pop(
                "front_padding_duration_seconds"
            ),
            back_padding_duration_seconds=config.pop(
                "back_padding_duration_seconds"
            ),
            **{k: gf.Distribution(**v) for k, v in config.items()},
        )

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
    MAX_FREQUENCY_HERTZ = WaveformParameter(203)
    
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
    duration_seconds : gf.Distribution = \
        gf.Distribution(min_=0.1, max_=1.0, type_=gf.DistributionType.UNIFORM)
    min_frequency_hertz: gf.Distribution = \
        gf.Distribution(min_=5.0, max_=95.0, type_=gf.DistributionType.UNIFORM)
    max_frequency_hertz: gf.Distribution = \
        gf.Distribution(min_=5.0, max_=95.0, type_=gf.DistributionType.UNIFORM)

    def generate(
        self,
        num_waveforms: int,
        sample_rate_hertz: float,
        max_duration_seconds: float
    ):
        
        if (num_waveforms > 0):
            
            if self.duration_seconds.type_ is not gf.DistributionType.CONSTANT:
                if self.duration_seconds.max_ > max_duration_seconds:                
                    warn("Max duration distibution is greater than requested "
                         "injection duration. Adjusting", UserWarning)
                    self.duration_seconds.max_ = max_duration_seconds

                if self.duration_seconds.min_ < 0.0 and \
                    self.duration_seconds.type_ is not gf.DistributionType.CONSTANT:

                    warn("Min duration distibution is less than zero "
                         "injection duration. Adjusting", UserWarning)

                    self.duration_seconds.min_ = 0.0
            
            # Draw parameter samples from distributions:
            parameters = {}
            for attribute, value in self.__dict__.items():    
                if is_not_inherited(self, attribute):
                    
                    parameter = None
                    try:
                        parameter = WaveformParameters.get(attribute)
                    except:
                        parameter = ScalingTypes.get(attribute)
                    
                    shape = parameter.value.shape[-1]                
                    parameters[attribute] = tf.convert_to_tensor(
                        value.sample(num_waveforms * shape)
                    )
                    
            parameters["max_frequency_hertz"] = np.maximum(
                parameters["max_frequency_hertz"], 
                parameters["min_frequency_hertz"]
            )
            parameters["min_frequency_hertz"] = np.minimum(
                parameters["max_frequency_hertz"],
                parameters["min_frequency_hertz"]
            )
                                    
            # Generate WNB waveform:
            waveforms = gf.wnb(
                num_waveforms,
                sample_rate_hertz, 
                max_duration_seconds, 
                **parameters
            )
            
            # Scale by arbitrary factor to reduce chance of precision errors:
            waveforms *= self.scale_factor
            
            return waveforms, parameters

@dataclass
class cuPhenomDGenerator(WaveformGenerator):
    mass_1_msun : gf.Distribution = \
        gf.Distribution(min_=5.0, max_=95.0, type_=gf.DistributionType.UNIFORM)
    mass_2_msun : gf.Distribution = \
        gf.Distribution(min_=5.0, max_=95.0, type_=gf.DistributionType.UNIFORM)
    inclination_radians : gf.Distribution = \
        gf.Distribution(min_=0.0, max_=np.pi, type_=gf.DistributionType.UNIFORM)
    distance_mpc : gf.Distribution = \
        gf.Distribution(value=1000.0, type_=gf.DistributionType.CONSTANT)
    reference_orbital_phase_in : gf.Distribution = \
        gf.Distribution(value=0.0, type_=gf.DistributionType.CONSTANT)
    ascending_node_longitude : gf.Distribution = \
        gf.Distribution(value=0.0, type_=gf.DistributionType.CONSTANT)
    eccentricity : gf.Distribution = \
        gf.Distribution(value=0.0, type_=gf.DistributionType.CONSTANT)
    mean_periastron_anomaly : gf.Distribution = \
        gf.Distribution(value=0.0, type_=gf.DistributionType.CONSTANT)
    spin_1_in : gf.Distribution = \
        gf.Distribution(value=0.0, type_=gf.DistributionType.CONSTANT)
    spin_2_in : gf.Distribution = \
        gf.Distribution(value=0.0, type_=gf.DistributionType.CONSTANT)
    
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
                    
                    parameter = None
                    try:
                        parameter = WaveformParameters.get(attribute)
                    except:
                        parameter = ScalingTypes.get(attribute) 
                    
                    shape = parameter.value.shape[-1]                
                    parameters[attribute] = value.sample(num_waveforms * shape)

            # Generate phenom_d waveform:
            waveforms = gf.imrphenomd(
                    num_waveforms, 
                    sample_rate_hertz, 
                    duration_seconds,
                    **parameters
                )
                        
            waveforms *= self.scale_factor
        
            return waveforms, parameters
    
class IncoherentGenerator(WaveformGenerator):
    component_generators : List(WaveformGenerator)
    
    def __init__(self, component_generators):
        self.component_generators = component_generators
        self.scaling_method = component_generators[0].scaling_method
        self.injection_chance = component_generators[0].injection_chance
        self.front_padding_duration_seconds = component_generators[0].front_padding_duration_seconds
        self.back_padding_duration_seconds = component_generators[0].back_padding_duration_seconds
        self.scale_factor = component_generators[0].scale_factor
        self.network = component_generators[0].network
            
    def generate(
        self,
        num_waveforms: int,
        sample_rate_hertz: float,
        duration_seconds : float
    ):
                    
        if len(self.component_generators) > 0:
            
            waveforms, parameters = [], []
            for generator in self.component_generators: 
                waveforms_, parameters_ = generator.generate(
                    num_waveforms,
                    sample_rate_hertz, 
                    duration_seconds
                )
                
                waveforms.append(waveforms_)
                parameters.append(parameters_)
        
        waveforms = tf.stack(waveforms, axis = 1)
        parameters = parameters[0]

        return waveforms, parameters
    
@dataclass
class InjectionGenerator:
    configs : Union[List[Union[cuPhenomDGenerator, WNBGenerator]], List[List[Union[cuPhenomDGenerator, WNBGenerator]]]]
    sample_rate_hertz : float = None
    onsource_duration_seconds : float = None
    crop_duration_seconds : float = None
    num_examples_per_generation_batch : int = None
    num_examples_per_batch : int = None
    variables_to_return : List[WaveformParameters] = None
    index : int = 0
    
    def __post_init__(self):
        
        if self.sample_rate_hertz is None:
            self.sample_rate_hertz = gf.Defaults.sample_rate_hertz
        if self.onsource_duration_seconds is None:
            self.onsource_duration_seconds = gf.Defaults.onsource_duration_seconds
        if self.crop_duration_seconds is None:
            self.crop_duration_seconds = gf.Defaults.crop_duration_seconds
        if self.num_examples_per_generation_batch is None:
            self.num_examples_per_generation_batch = gf.Defaults.num_examples_per_generation_batch
        if self.num_examples_per_batch is None:
            self.num_examples_per_batch = gf.Defaults.num_examples_per_batch
        
        if (self.num_examples_per_batch > self.num_examples_per_generation_batch):
            logging.warning(
                ("num_injections_per_batch must be less than "
                 "num_examples_per_generation_batch adjusting to compensate")
            )
            
            self.num_examples_per_generation_batch = self.num_examples_per_batch
    
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
                            tf.zeros(
                                [key.value.shape[-1] * self.num_examples_per_batch], 
                                dtype = tf.float64
                            )
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

                #Roll Tensor to randomise start time:
                waveforms = \
                    roll_vector_zero_padding( 
                        waveforms, 
                        min_roll_num_samples, 
                        max_roll_num_samples
                    )
                
                # Create zero filled injections to fill spots where injection 
                # did not generate due to injection masks:
                injections = gf.expand_tensor(waveforms, mask)
            else:
                injections = tf.zeros(
                    shape=(num_batches, 2, total_duration_num_samples)
                )
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

                # Conver to tensor and expand parameter dims for remaining 
                # parameters:
                expanded_parameters = {}
                for key, parameter in reduced_parameters.items():

                    parameter = tf.convert_to_tensor(parameter)

                    expanded_parameters[key] = \
                        gf.expand_tensor(
                            parameter, 
                            mask,
                            group_size=key.value.shape[-1] 
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
            injections = gf.batch_tensor(injections, self.num_examples_per_batch)
            mask = gf.batch_tensor(mask, self.num_examples_per_batch)

            for injections_, mask_, parameters_ in zip(
                injections, mask, parameters
                ):
                
                yield injections_, mask_, parameters_
                
    def generate_scaling_parameters_(
        self,
        mask: tf.Tensor,
        config : WaveformGenerator
    ) -> tf.Tensor:
    
        """
        Generate scaling parameter (SNRs or HRSS) given a mask, generator and 
        example index.

        Parameters
        ----------
        mask : Tensor
            A tensor representing the injection mask.
            
        Returns
        -------
        Tensor
            A tensor representing generated scaling parameters.
        """

        mask = mask.numpy()

        num_injections = np.sum(mask)
        
        match config.scaling_method.value:
            case np.ndarray():

                scaling_parameters = []
                for index in range(self.index, self.index + num_injections):
                    if index < len(config.scaling_method.value):
                        scaling_parameters.append(
                            config.scaling_method.value[index]
                        )
                    else:
                        scaling_parameters.append(
                            config.scaling_method.value[-1]
                        )
                        
                    self.index += 1

            case gf.Distribution():
                scaling_parameters = config.scaling_method.value.sample(
                    num_injections
                )

            case _:
                raise ValueError("Unsupported scaling method value type: "
                                 f"{type(config.scaling_method.value)}!") 

        scaling_parameters = tf.convert_to_tensor(
            scaling_parameters, 
            dtype = tf.float32
        )
                
        scaling_parameters = gf.expand_tensor(
            scaling_parameters,
            mask
        )

        return scaling_parameters
    
    def generate_scaling_parameters(
        self,
        masks : tf.Tensor
        ):
        
        scaling_parameters = []
        for mask, config in zip(masks, self.configs):
            scaling_parameters.append(
                self.generate_scaling_parameters_(
                    mask,
                    config
                )
            )
            
        return tf.stack(scaling_parameters)
    
    def add_injections_to_onsource(
            self,
            injections : tf.Tensor,
            mask : tf.Tensor,
            onsource : tf.Tensor,
            variables_to_return : List,
        ) -> Tuple[tf.Tensor, Union[tf.Tesnor, None], Dict]:
        
        # Generate SNR or HRSS values for injections based on inputed config 
        # values:
        scaling_parameters = self.generate_scaling_parameters(mask)
        
        return_variables = {
            key : [] for key in ScalingTypes if key in variables_to_return
        }
        cropped_injections = []    
        for injections_, mask_, scaling_parameters_, config in \
            zip(injections, mask, scaling_parameters, self.configs):
            
            network = config.network
            
            match config.scaling_method.type_.value.ordinality:
                
                case ScalingOrdinality.BEFORE_PROJECTION:
                
                    scaled_injections = \
                        config.scaling_method.scale(
                            injections_,
                            onsource,
                            scaling_parameters_,
                            self.sample_rate_hertz
                        )

                    if network is not None:
                        scaled_injections = network.project_wave(
                            scaled_injections, self.sample_rate_hertz
                        )
                    else:
                        scaled_injections = scaled_injections[:, 0, :]
            
                case ScalingOrdinality.AFTER_PROJECTION:
                                        
                    if network is not None:
                        injections_ = network.project_wave(
                            injections_, self.sample_rate_hertz
                        )
                    else:
                        injections_ = injections_[:, 0, :]
                    
                    # Scale injections with selected scaling method:
                    scaled_injections = \
                        config.scaling_method.scale(
                            injections_,
                            onsource,
                            scaling_parameters_,
                            self.sample_rate_hertz
                        )
                
                case _:
                    
                    raise ValueError(
                        ("Scaling ordinality "
                        f"{config.scaling_method.type_.value.order} not "
                         " recognised.")
                    )

            # Add scaled injections to onsource:
            onsource += scaled_injections
                        
            if ScalingTypes.HPEAK in variables_to_return:
                # Calculate hpeak of scaled injections:
                
                if config.scaling_method.type_ is not ScalingTypes.HPEAK:
                    return_variables[ScalingTypes.HPEAK].append(
                        calculate_hpeak(injections_)
                    )
                    
            if ScalingTypes.SNR in variables_to_return:
                # Calculate snr of scaled injections:
                
                if config.scaling_method.type_ is not ScalingTypes.SNR:
                    return_variables[ScalingTypes.SNR].append(
                        gf.snr(
                            scaled_injections, 
                            onsource,
                            self.sample_rate_hertz, 
                            fft_duration_seconds=1.0,
                            overlap_duration_seconds=0.5
                        ) 
                    )
                    
            if ScalingTypes.HRSS in variables_to_return:
                # Calculate hrss of scaled injections:
                
                if config.scaling_method.type_ is not ScalingTypes.HRSS:
                    return_variables[ScalingTypes.HRSS].append(
                        calculate_hrss(injections_) 
                    )
            
            if (ReturnVariables.INJECTIONS in variables_to_return) or \
                (ReturnVariables.WHITENED_INJECTIONS in variables_to_return):
                # Crop injections so that they appear the same size as output 
                # onsource:
                cropped_injections.append(
                    gf.crop_samples(
                        scaled_injections, 
                        self.onsource_duration_seconds, 
                        self.sample_rate_hertz
                    )
                )
                                
        if (ReturnVariables.INJECTIONS in variables_to_return) or \
            (ReturnVariables.WHITENED_INJECTIONS in variables_to_return):
            cropped_injections = tf.stack(cropped_injections)
        else:
            cropped_injections = None
            
        # Add scaling parameters to return dictionary
        for scaling_type in ScalingTypes:
            if scaling_type in variables_to_return:
                if config.scaling_method.type_ is not scaling_type:
                    return_variables[scaling_type] = \
                        tf.stack(return_variables[scaling_type])
                else:
                    return_variables[scaling_type] = scaling_parameters
        
        return onsource, cropped_injections, return_variables

@tf.function(jit_compile=True)
def roll_vector_zero_padding_(vector, roll_amount):
    # Create zeros tensor with the same shape as vector
    zeros = tf.zeros_like(vector)

    # Roll the vector along the last dimension and replace the end values with zeros
    rolled_vector = tf.concat([vector[..., roll_amount:], zeros[..., :roll_amount]], axis=-1)
    
    return rolled_vector

@tf.function
def roll_vector_zero_padding(tensor, min_roll, max_roll):
    # Generate an array of roll amounts
    roll_amounts = tf.random.uniform(
        shape=[tensor.shape[0]], minval=min_roll, maxval=max_roll, dtype=tf.int32
    )

    # Define a function to apply rolling to each sub_tensor with corresponding roll_amount
    def map_fn_outer(idx):
        sub_tensor = tensor[idx]
        roll_amount = roll_amounts[idx]

        # Apply the roll_vector_zero_padding_ function along the last dimension for each element in the sub_tensor
        return roll_vector_zero_padding_(sub_tensor, roll_amount)

    # Create an index tensor and map over it
    indices = tf.range(start=0, limit=tensor.shape[0], dtype=tf.int32)
    result = tf.map_fn(map_fn_outer, indices, fn_output_signature=tf.TensorSpec(shape=tensor.shape[1:], dtype=tensor.dtype))

    return result

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
    Splits the given dictionary into smaller dictionaries containing N 
    waveforms.

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