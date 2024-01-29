from __future__ import annotations

# Built-In imports:
from dataclasses import dataclass
from typing import Tuple
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
from numpy.random import default_rng  

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
                raise ValueError(f"Scaling type {self.type_} not recognised.")
        
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
    scale_factor = desired_hrss/(current_hpeak + epsilon)
    
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
    scaling_method : Union[ScalingMethod, None] = None
    injection_chance : float = 1.0
    front_padding_duration_seconds : float = 0.3
    back_padding_duration_seconds : float = 0.0
    scale_factor : Union[float, None] = None
    network : Union[List[IFO], gf.Network, Path, None] = None

    distributed_attributes : Union[Tuple, None] = None
        
    def __post_init__(self):
        self.network = self.init_network(self.network)
        
        # Convert all waveform attributes to gf.Distribution if not already:
        self.convert_all_distributions()

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
                raise TypeError(
                    ("Unable to initiate network with this type: "
                    f"{type(network)}.")
                )
                
        return network
    
    def copy(self):
        return deepcopy(self)
    
    @classmethod
    def load(
        cls,
        path: Path, 
        scaling_method: ScalingMethod = None,
        scale_factor : float = None,
        network : Union[List[IFO], gf.Network, Path] = None
    ) -> Type[cls]:

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
        with path.open("r") as file:
            config = json.load(file)

        # Replace placeholders
        gf.replace_placeholders(
            config, 
            replacements
        )
            
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

    def convert_to_distribution(
        self, 
        attribute : Union[int, float, str, gf.Distribution]
    ):
        if not isinstance(attribute, gf.Distribution):
            if isinstance(attribute, list) or isinstance(attribute, tuple): 
                value = []
                for element in attribute:
                    value.append(
                        gf.Distribution(value=element, type_=gf.DistributionType.CONSTANT)
                    )
                return value
            else:
                return gf.Distribution(value=attribute, type_=gf.DistributionType.CONSTANT)
        else:
            return attribute

    def convert_all_distributions(self):
        for attribute in self.distributed_attributes:
            converted = self.convert_to_distribution(
                getattr(self, attribute)
            )
            setattr(self, attribute, converted)
    
    def reseed(self, seed):

        rng = default_rng(seed)
        for attribute in self.distributed_attributes:
            distribution = getattr(self, attribute)

            # Placing this outside the loop ensures that
            # seed will be the same for each parameter 
            # independent of which are set to distributions
            seed = rng.integers(1E10)
            if isinstance(distribution, gf.Distribution):
                distribution.reseed(seed)
                setattr(self, attribute, distribution)

    def ensure_list_of_floats(
        self,
        name : str,
        value : Union[int, float, gf.Distribution],
    ):
        if not isinstance(value, gf.Distribution):
            if isinstance(value, list) or isinstance(value, tuple):
                if (len(value) == 3):
                    new_value = []
                    for element, label in zip(value, ("x", "y", "z")):
                        new_value.append(self.ensure_float(
                            f"{name}_{label}",
                            element
                        ))
                    return tuple(new_value)
                else:
                    raise ValueError(f"{name} should contain three elements, x, y, and z.")
            else:
                raise TypeError(f"{name} should be list or tuple.")
        else:
            if value.dtype == int:
                logging.warn(f"{name} should not have dtype = int, automatically adjusting.")
                value.dtype = float
                return value
            else:
                return value


    def ensure_float(
        self,
        name : str,
        value : Union[int, float, gf.Distribution],
    ):
        if not isinstance(value, gf.Distribution):
            if isinstance(value, int):
                logging.warn(f"{name} should be float not int, automatically adjusting.")
                return float(value)
            elif isinstance(value, float):
                return value
            else:
                raise TypeError(f"{name} should be float or gf.Distribution object.")    
        else:
            if value.dtype == int:
                logging.warn(f"{name} should not have dtype = int, automatically adjusting.")
                value.dtype = float
                return value
            else:
                return value

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
    duration_seconds : Union[float, gf.Distribution] = 0.5
    min_frequency_hertz: Union[float, gf.Distribution] = 16.0
    max_frequency_hertz: Union[float, gf.Distribution] = 1024.0

    distributed_attributes : Tuple[str] = (
        "duration_seconds",
        "min_frequency_hertz",
        "max_frequency_hertz"
    )

    def __post_init__(self):
        
        self.duration_seconds = self.ensure_float("duration_seconds", self.duration_seconds)
        self.min_frequency_hertz = self.ensure_float("min_frequency_hertz", self.min_frequency_hertz)
        self.max_frequency_hertz = self.ensure_float("max_frequency_hertz", self.max_frequency_hertz)

        if self.scale_factor is None:
            self.scale_factor=1.0

        super().__post_init__()
    
    def generate(
        self,
        num_waveforms: int,
        sample_rate_hertz: float,
        max_duration_seconds: float,
        seed : int
    ):
        
        if (num_waveforms > 0):
            
            if self.duration_seconds.type_ is not gf.DistributionType.CONSTANT:
                if self.duration_seconds.max_ > max_duration_seconds:                
                    warn("Max duration distibution is greater than requested "
                         "injection duration. Adjusting", UserWarning)
                    self.duration_seconds.max_ = max_duration_seconds

                if self.duration_seconds.min_ < 0.0 and self.duration_seconds.type_ is not gf.DistributionType.CONSTANT:

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
                **parameters,
                seed=int(seed)
            )
            
            # Scale by arbitrary factor to reduce chance of precision errors:
            waveforms *= self.scale_factor
            
            return waveforms, parameters

@dataclass
class cuPhenomDGenerator(WaveformGenerator):
    mass_1_msun : Union[float, gf.Distribution] = 30.0
    mass_2_msun : Union[float, gf.Distribution] = 30.0
    inclination_radians : Union[float, gf.Distribution] = 0.0
    distance_mpc : Union[float, gf.Distribution] = 50.0
    reference_orbital_phase_in : Union[float, gf.Distribution] = 0.0
    ascending_node_longitude : Union[float, gf.Distribution] = 0.0
    eccentricity : Union[float, gf.Distribution] = 0.0
    mean_periastron_anomaly : Union[float, gf.Distribution] = 0.0
    spin_1_in : Union[Tuple[float], gf.Distribution] = (0.0, 0.0, 0.0,)
    spin_2_in : Union[Tuple[float], gf.Distribution] = (0.0, 0.0, 0.0,)
    
    distributed_attributes : Tuple[str] = (
        "mass_1_msun",
        "mass_2_msun",
        "inclination_radians",
        "distance_mpc",
        "reference_orbital_phase_in",
        "ascending_node_longitude",
        "eccentricity",
        "mean_periastron_anomaly",
        "spin_1_in",
        "spin_2_in"
    )

    def __post_init__(self):
        self.mass_1_msun = self.ensure_float("mass_1_msun", self.mass_1_msun)
        self.mass_2_msun = self.ensure_float("mass_2_msun", self.mass_2_msun)

        self.inclination_radians = self.ensure_float("inclination_radians", self.inclination_radians)
        self.distance_mpc = self.ensure_float("distance_mpc", self.distance_mpc)
        self.reference_orbital_phase_in = self.ensure_float("reference_orbital_phase_in", self.reference_orbital_phase_in)
        self.ascending_node_longitude = self.ensure_float("ascending_node_longitude", self.ascending_node_longitude)
        self.eccentricity = self.ensure_float("eccentricity", self.eccentricity)
        self.mean_periastron_anomaly = self.ensure_float("mean_periastron_anomaly", self.mean_periastron_anomaly)

        self.spin_1_in = self.ensure_list_of_floats("spin_1_in", self.spin_1_in)
        self.spin_2_in = self.ensure_list_of_floats("spin_2_in", self.spin_2_in)

        if self.scale_factor is None:
            self.scale_factor=gf.Defaults.scale_factor

        super().__post_init__()
    
    def generate(
            self,
            num_waveforms : int,
            sample_rate_hertz : float,
            duration_seconds : float,
            _ : int
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
                    
                    if not isinstance(value, tuple) and  not isinstance(value, list): 
                        parameters[attribute] = value.sample(num_waveforms * shape)
                    else:
                        combined_array = []
                        for element in value:
                            combined_array += element.sample(num_waveforms)
                        parameters[attribute] = combined_array

            # Generate phenom_d waveform:
            waveforms = None
            while waveforms is None:
                
                mass_1 = parameters["mass_1_msun"]
                mass_2 = parameters["mass_2_msun"]
                
                parameters["mass_1_msun"] = np.maximum(mass_1, mass_2)
                parameters["mass_2_msun"] = np.minimum(mass_1, mass_2)

                try:
                    waveforms = gf.imrphenomd(
                            num_waveforms, 
                            sample_rate_hertz, 
                            duration_seconds,
                            **parameters
                        )
                except Exception as e:
                    # If an exception occurs, increment the attempts counter
                    attempts += 1
                    # Check if the maximum number of retries has been reached
                    waveforms = None
                    if attempts >= max_retries:
                        raise Exception(f"Max waveform generation retries reached: {max_retries}") from e
            
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
        duration_seconds : float,
        seed : int
    ):
                    
        if len(self.component_generators) > 0:
            
            waveforms, parameters = [], []
            for generator in self.component_generators: 
                waveforms_, parameters_ = generator.generate(
                    num_waveforms,
                    sample_rate_hertz, 
                    duration_seconds,
                    seed
                )
                
                waveforms.append(waveforms_)
                parameters.append(parameters_)
        
        try:
            waveforms = tf.stack(waveforms, axis = 1)
        except: 
            logging.error("Failed to stack waveforms!")
            return None, None
        
        parameters = parameters[0]

        return waveforms, parameters
    
@dataclass
class InjectionGenerator:
    waveform_generators : Union[
        List[gf.WaveformGenerator], 
        Dict[str, gf.WaveformGenerator]
    ]
    parameters_to_return : Union[List[WaveformParameters], None] = None
    seed : int = None
    index : int = 0

    def __post_init__(self):
        if self.parameters_to_return is None: 
            self.parameters_to_return = []
        if self.seed is None:
            self.seed = gf.Defaults.seed

    def __call__(
            self,
            sample_rate_hertz : float = None,
            onsource_duration_seconds : float = None,
            crop_duration_seconds : float = None,
            num_examples_per_generation_batch : int = None,
            num_examples_per_batch : int = None,
        ):

        self.sample_rate_hertz = sample_rate_hertz
        self.onsource_duration_seconds = onsource_duration_seconds
        self.crop_duration_seconds = crop_duration_seconds
        self.num_examples_per_generation_batch = num_examples_per_generation_batch
        self.num_examples_per_batch = num_examples_per_batch


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

        rng = default_rng(self.seed)
        self.generator_rng = default_rng(rng.integers(1E10))
        self.parameter_rng = default_rng(rng.integers(1E10))

        if (self.num_examples_per_batch > self.num_examples_per_generation_batch):
            logging.warning(
                ("num_injections_per_batch must be less than "
                 "num_examples_per_generation_batch adjusting to compensate")
            )
            
            self.num_examples_per_generation_batch = self.num_examples_per_batch
        
        self.parameters_to_return = [
            item for item in self.parameters_to_return if isinstance(item.value, WaveformParameter)
        ]
        
        if self.waveform_generators is False:
            yield None, None, None
            
        elif not isinstance(self.waveform_generators, list) and not isinstance(self.waveform_generators, dict):
            self.waveform_generators = [self.waveform_generators]

        # Process waveform generators based on their type
        if isinstance(self.waveform_generators, list):
            # Reseeding generators for list type
            for generator in self.waveform_generators:
                generator.reseed(self.parameter_rng.integers(1E10))

            # Creating iterators for each generator in list
            self.iterators = [
                self.generate_one(generator, seed=self.generator_rng.integers(1E10))
                for generator in self.waveform_generators
            ]

        elif isinstance(self.waveform_generators, dict):
            # Reseeding generators for dictionary type
            for generator in self.waveform_generators.values():
                generator["generator"].reseed(self.parameter_rng.integers(1E10))

            # Creating iterators with additional 'name' parameter for each generator in dictionary
            self.iterators = [
                self.generate_one(config["generator"], seed=self.generator_rng.integers(1E10), name=key)
                for key, config in self.waveform_generators.items()
            ]

            # Initializing an empty mask dictionary
            self.mask_dict = {}

        else:
            # Handling invalid waveform generator types
            raise TypeError("Waveform generators must be a dictionary or a list!")

        injections = []
        mask = []
        parameters = {key : [] for key in self.parameters_to_return}
        for iterator in self.iterators:

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
        try:
            injections = tf.stack(injections)
            mask = tf.stack(mask)
        except Exception as e:
            logging.error(f"Failed to stack injections or mask because {e}")
            return None, None, None

        for key, value in parameters.items():
            try:
                parameters[key] = tf.stack(value)
            except Exception as e:
                logging.error(f"Failed to stack injections parameters because {e}")
                return None, None, None

        while 1: 
            yield injections, mask, parameters
    
    def generate_one(
            self, 
            generator : gf.WaveformGenerator, 
            seed : int,
            name : str = None
        ):

        # Create default empty list for requested parameter returns:
        if self.parameters_to_return is None:
            self.parameters_to_return = []
        
        total_duration_seconds : float = self.onsource_duration_seconds + (self.crop_duration_seconds * 2.0)
        total_duration_num_samples : int = int(
            total_duration_seconds * self.sample_rate_hertz
        )

        rng = default_rng(seed)
        position_rng = default_rng(rng.integers(1E10))
        waveform_rng = default_rng(rng.integers(1E10))
        mask_rng = default_rng(rng.integers(1E10))
        
        # Calculate roll boundaries:
        min_roll_num_samples = int(
                (generator.back_padding_duration_seconds + self.crop_duration_seconds)
                * self.sample_rate_hertz
            ) 

        max_roll_num_samples = total_duration_num_samples - int(
                (self.crop_duration_seconds + generator.front_padding_duration_seconds)
                * self.sample_rate_hertz
            )
        
        num_batches : int = self.num_examples_per_generation_batch // self.num_examples_per_batch

        while 1:
            mask = generate_mask(
                    self.num_examples_per_generation_batch,  
                    generator.injection_chance,
                    mask_rng.integers(1E10, size=2)
                )

            if name is not None:
                self.mask_dict[name] = mask

                if self.waveform_generators[name]["excluded"] is not None:

                    excluded = self.waveform_generators[name]["excluded"]

                    mask = tf.math.logical_and(
                        mask,
                        tf.math.logical_not(self.mask_dict[excluded])
                    )

            num_waveforms = tf.reduce_sum(tf.cast(mask, tf.int32)).numpy()
            
            if num_waveforms > 0:
                
                waveforms, parameters = generator.generate(
                        num_waveforms, 
                        self.sample_rate_hertz,
                        total_duration_seconds,
                        waveform_rng.integers(1E10)
                    )
                
                # Convert to tensorflow tensor:
                waveforms = tf.convert_to_tensor(waveforms, dtype = tf.float32)
                #Roll Tensor to randomise start time:
                waveforms = roll_vector_zero_padding( 
                        waveforms, 
                        min_roll_num_samples, 
                        max_roll_num_samples, 
                        mask_rng.integers(1E10, size=2)
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
            if self.parameters_to_return:

                # Retrive parameters that are requested to reduce unneccisary
                # post processing:
                reduced_parameters = {
                    WaveformParameters.get(key) : value 
                        for key, value in parameters.items() 
                        if WaveformParameters.get(key) in self.parameters_to_return
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

            # Split into list to avoid iterating over tensors, which TensorFlow seems 
            # to dislike:

            try:
                injections_list = tf.unstack(injections, axis=0)
                mask_list = tf.unstack(mask, axis=0)
            except Exception as e:
                logging.error(f"Failed to unstack because: {e}")
                continue

            # Now you can zip them together and iterate over them
            for injection_, mask_, parameter_ in zip(injections_list, mask_list, parameters):
                try:
                    # Your processing logic here
                    yield injection_, mask_, parameter_
                except Exception as e:
                    logging.error(f"Failed to yeild because: {e}")
                    continue
    
    def generate_scaling_parameters_(
        self,
        mask: tf.Tensor,
        generator : WaveformGenerator
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
        
        match generator.scaling_method.value:
            case np.ndarray():

                scaling_parameters = []
                for index in range(self.index, self.index + num_injections):
                    if index < len(generator.scaling_method.value):
                        scaling_parameters.append(
                            generator.scaling_method.value[index]
                        )
                    else:
                        scaling_parameters.append(
                            generator.scaling_method.value[-1]
                        )
                        
                    self.index += 1

            case gf.Distribution():
                scaling_parameters = generator.scaling_method.value.sample(
                    num_injections
                )

            case _:
                raise TypeError("Unsupported scaling method value type: "
                                 f"{type(generator.scaling_method.value)}!") 

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

        if isinstance(self.waveform_generators, dict):
            for mask, config in zip(masks, self.waveform_generators.values()):
                scaling_parameters.append(
                    self.generate_scaling_parameters_(
                        mask,
                        config["generator"]
                    )
                )
        else: 
            for mask, generator in zip(masks, self.waveform_generators):
                scaling_parameters.append(
                    self.generate_scaling_parameters_(
                        mask,
                        generator
                    )
                )
            
        return scaling_parameters
    
    def add_injections_to_onsource(
            self,
            injections : tf.Tensor,
            mask : tf.Tensor,
            onsource : tf.Tensor,
            parameters_to_return : List,
        ) -> Tuple[tf.Tensor, Union[tf.Tesnor, None], Dict]:
        
        # Generate SNR or HRSS values for injections based on inputed mask 
        # values:
        scaling_parameters : List = self.generate_scaling_parameters(mask)
        
        return_variables = {
            key : [] for key in ScalingTypes if key in parameters_to_return
        }
        cropped_injections = []    

        # Unstack because TensorFlow hates tensor iteration: 
        try:
            injections_list = tf.unstack(injections, axis=0)
            mask_list = tf.unstack(mask, axis=0)
        except:
            logging.error("Unstacking failed for some reason")
            return None, None, None

        if isinstance(self.waveform_generators, dict):
            waveform_generators = [config["generator"] for config in self.waveform_generators.values()]
        else:
            waveform_generators = self.waveform_generators

        for injections_, _, scaling_parameters_, waveform_generator in zip(
                injections_list, 
                mask_list, 
                scaling_parameters, 
                waveform_generators
            ):
            
            network = waveform_generator.network
            
            match waveform_generator.scaling_method.type_.value.ordinality:
                
                case ScalingOrdinality.BEFORE_PROJECTION:

                    try:
                        scaled_injections = waveform_generator.scaling_method.scale(
                                injections_,
                                onsource,
                                scaling_parameters_,
                                self.sample_rate_hertz
                            )
                    except Exception as e:
                        logger.error(f+-"Failed to scale injections because {e}")

                    try:
                        scaled_injections = network.project_wave(
                            scaled_injections, sample_rate_hertz=self.sample_rate_hertz
                        )
                    except Exception as e:
                        logger.error(f+-"Failed to project injections because {e}")

                    if scaled_injections is None:
                        logging.error("Error scaling injections before projection!")
                        return None, None, None
            
                case ScalingOrdinality.AFTER_PROJECTION:
                    
                    try:
                        if network.num_detectors > 1:
                            injections_ = network.project_wave(
                                injections_, sample_rate_hertz=self.sample_rate_hertz
                            )
                        else:
                            injections_ = tf.reduce_sum(injections_, axis=1, keepdims = True)
                    except Exception as e:
                        logging.error(f"Failed to project injections because {e}")

                    if injections_ is not None:
                        if injections_.shape != onsource.shape:
                            logging.error(f"Shape mismatch {injections_.shape} {onsource.shape}")
                            return None, None, None

                        try:
                            # Scale injections with selected scaling method:
                            scaled_injections = waveform_generator.scaling_method.scale(
                                    injections_,
                                    onsource,
                                    scaling_parameters_,
                                    self.sample_rate_hertz
                                )
                        except Exception as e:
                            logger.error(f+-"Failed to scale injections because {e}")

                    else:
                        logging.error("Error scaling injections after projection!")
                        return None, None, None
                
                case _:
                    
                    raise ValueError(
                        ("Scaling ordinality "
                        f"{waveform_generator.scaling_method.type_.value.order} not "
                         " recognised.")
                    )

            # Add scaled injections to onsource:
            try:
                onsource += scaled_injections
            except Exception as e:
                logger.error(f+-"Failed to add injections because {e}")

            if ScalingTypes.HPEAK in parameters_to_return:
                # Calculate hpeak of scaled injections:
                
                if waveform_generator.scaling_method.type_ is not ScalingTypes.HPEAK:
                    return_variables[ScalingTypes.HPEAK].append(
                        calculate_hpeak(injections_)
                    )
                    
            if ScalingTypes.SNR in parameters_to_return:
                # Calculate snr of scaled injections:
                
                if waveform_generator.scaling_method.type_ is not ScalingTypes.SNR:
                    return_variables[ScalingTypes.SNR].append(
                        gf.snr(
                            scaled_injections, 
                            onsource,
                            self.sample_rate_hertz, 
                            fft_duration_seconds=1.0,
                            overlap_duration_seconds=0.5
                        ) 
                    )
                    
            if ScalingTypes.HRSS in parameters_to_return:
                # Calculate hrss of scaled injections:
                
                if waveform_generator.scaling_method.type_ is not ScalingTypes.HRSS:
                    return_variables[ScalingTypes.HRSS].append(
                        calculate_hrss(injections_) 
                    )
            
            if (ReturnVariables.INJECTIONS in parameters_to_return) or \
                (ReturnVariables.WHITENED_INJECTIONS in parameters_to_return):
                # Crop injections so that they appear the same size as output 
                # onsource:
                cropped_injections.append(
                    gf.crop_samples(
                        scaled_injections, 
                        self.onsource_duration_seconds, 
                        self.sample_rate_hertz
                    )
                )
                                
        if (ReturnVariables.INJECTIONS in parameters_to_return) or \
            (ReturnVariables.WHITENED_INJECTIONS in parameters_to_return):
            try:
                cropped_injections = tf.stack(cropped_injections)
            except:
                logging.error("Failed to stack injections!")
                return None, None, None
        else:
            cropped_injections = None
            
        # Add scaling parameters to return dictionary
        for scaling_type in ScalingTypes:
            if scaling_type in parameters_to_return:
                if waveform_generator.scaling_method.type_ is not scaling_type:
                    try:
                        return_variables[scaling_type] = tf.stack(return_variables[scaling_type])
                    except:
                        logging.error("Failed to stack return variables")
                        return None, None, None
                else:
                    return_variables[scaling_type] = scaling_parameters
                
        return onsource, cropped_injections, return_variables

@tf.function
def roll_vector_zero_padding_(vector, roll_amount):
    # Create zeros tensor with the same shape as vector
    zeros = tf.zeros_like(vector)

    # Roll the vector along the last dimension and replace the end values with zeros
    rolled_vector = tf.concat([vector[..., roll_amount:], zeros[..., :roll_amount]], axis=-1)
    
    return rolled_vector

@tf.function
def roll_vector_zero_padding(tensor, min_roll, max_roll, seed):

    # Ensure the seed is of the correct shape [2] and dtype int32
    seed_tensor = tf.cast(seed, tf.int32)

    # Generate an array of roll amounts
    roll_amounts = tf.random.stateless_uniform(
        shape=[tensor.shape[0]], seed=seed_tensor, minval=min_roll, maxval=max_roll, dtype=tf.int32
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
    injection_chance: float,
    seed
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

    # Ensure the seed is of the correct shape [2] and dtype int32
    seed_tensor = tf.cast(seed, tf.int32)

    # Logits for [False, True] categories
    logits = tf.math.log([1.0 - injection_chance, injection_chance])

    # Generate categorical random variables based on logits
    sampled_indices = tf.random.stateless_categorical(
        seed=seed_tensor,
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