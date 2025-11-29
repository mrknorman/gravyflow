from __future__ import annotations

# Built-In imports:
from dataclasses import dataclass
from typing import Tuple, Optional
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
from gravyflow.src.dataset.features.waveforms.ripple import generate_ripple_waveform

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

@tf.function(jit_compile=True, reduce_retracing=True)
def scale(
        injections: tf.Tensor,
        onsource: tf.Tensor,
        scaling_parameters: tf.Tensor,
        sample_rate_hertz: float,
        scaling_type: str
    ) -> tf.Tensor:
    
    """
    TensorFlow function to scale injections based on different scaling types.
    
    Args:
        injections: Input tensor of injections to be scaled
        onsource: Tensor containing onsource data
        scaling_parameters: Tensor of parameters used for scaling
        sample_rate_hertz: Sampling rate in Hz
        scaling_type: String indicating the type of scaling ('SNR', 'HRSS', or 'HPEAK')
    
    Returns:
        tf.Tensor: Scaled injections
    """
    
    if scaling_type == "SNR":
        return gf.scale_to_snr(
            injections,
            onsource,
            scaling_parameters,
            sample_rate_hertz,
            fft_duration_seconds=1.0,
            overlap_duration_seconds=0.5
        )
    elif scaling_type == "HRSS":
        return scale_to_hrss(
            injections,
            scaling_parameters,
        )
    elif scaling_type == "HPEAK":
        return scale_to_hpeak(
            injections,
            scaling_parameters,
        )

@tf.function(jit_compile=True, reduce_retracing=True)
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

@tf.function(jit_compile=True, reduce_retracing=True)
def calculate_hpeak(
    injection: tf.Tensor
    ):
    
    # Return the root sum sqaure of the inputted injections:
    return tf.reduce_max(tf.abs(injection), axis=-1)

@tf.function(jit_compile=True, reduce_retracing=True)
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

@tf.function(jit_compile=True, reduce_retracing=True)
def scale_to_hpeak(
    injection: tf.Tensor, 
    desired_hpeak: float
    ) -> tf.Tensor:
    
    # Small value to prevent divide by zero errors:
    epsilon = 1.0E-7
    
    # Calculate the current HRSS of the injection in the background, so that
    # it can be scaled to the desired value:
    current_hpeak = calculate_hpeak(
        injection
    )
    
    # Calculate factor required to scale injection to desired HRSS:
    scale_factor = desired_hpeak/(current_hpeak + epsilon)
    
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
                waveform_cls = PhenomDGenerator
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
            waveforms = ensure_last_dim_even(waveforms)
            
            return waveforms, parameters

@tf.function(jit_compile=True, reduce_retracing=True)
def ensure_last_dim_even(tensor):
    # Get the shape of the tensor
    last_dim = tensor.shape[-1]
    
    # Determine if the last dimension is even
    is_even = tf.equal(last_dim % 2, 0)
    
    # Pad the tensor with zeros in the last dimension if needed
    result = tf.cond(
        is_even,
        lambda: tensor,
        lambda: tensor[...,:-1]
    )
    
    return result

@dataclass
class PhenomDGenerator(WaveformGenerator):
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

            # Generate phenom_d waveform using ripple:
            waveforms = None
            
            mass_1 = parameters["mass_1_msun"]
            mass_2 = parameters["mass_2_msun"]
            
            parameters["mass_1_msun"] = np.maximum(mass_1, mass_2)
            parameters["mass_2_msun"] = np.minimum(mass_1, mass_2)

            waveforms = generate_ripple_waveform(
                    num_waveforms=num_waveforms, 
                    sample_rate_hertz=sample_rate_hertz, 
                    duration_seconds=duration_seconds,
                    **parameters
                )
            
            waveforms *= self.scale_factor

            waveforms = ensure_last_dim_even(waveforms)
        
            return waveforms, parameters

# Alias for backward compatibility
cuPhenomDGenerator = PhenomDGenerator
    
class IncoherentGenerator(WaveformGenerator):
    component_generators : List[WaveformGenerator]
    
    def __init__(self, component_generators):
        self.component_generators = component_generators
        self.scaling_method = component_generators[0].scaling_method
        self.injection_chance = component_generators[0].injection_chance
        self.front_padding_duration_seconds = component_generators[0].front_padding_duration_seconds
        self.back_padding_duration_seconds = component_generators[0].back_padding_duration_seconds
        self.scale_factor = component_generators[0].scale_factor
        self.network = component_generators[0].network
        
        if self.network is not None:
            if self.network.num_detectors != len(self.component_generators):
                raise ValueError("When using component generators num ifos must equal num generators!")

    def reseed(self, seed):
        rng = default_rng(seed)
        for generator in self.component_generators:
            seed = rng.integers(1E10)
            generator.reseed(seed)
    
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

@tf.function(jit_compile=True, reduce_retracing=True)
def handle_before_projection(
        injection,
        onsource, 
        scaling_parameters,
        sample_rate_hertz,
        scaling_type,
        seed,
        x_vector,
        y_vector,
        x_length_meters,
        y_length_meters,
        x_response,
        y_response,
        response,
        location
    ):

    scaled = scale(
        injection,
        onsource,
        scaling_parameters,
        sample_rate_hertz,
        scaling_type
    )
    return gf.project_wave(
        seed,
        scaled,
        sample_rate_hertz,
        x_vector,
        y_vector,
        x_length_meters,
        y_length_meters,
        x_response,
        y_response,
        response,
        location
    )

@tf.function(jit_compile=True, reduce_retracing=True)
def handle_after_projection(
        injection,
        onsource, 
        scaling_parameters,
        sample_rate_hertz,
        scaling_type,
        seed,
        x_vector,
        y_vector,
        x_length_meters,
        y_length_meters,
        x_response,
        y_response,
        response,
        location,
        num_detectors
    ):

    projected = tf.cond(
        tf.greater(num_detectors, 1),
        lambda: gf.project_wave(
            seed,
            injection,
            sample_rate_hertz,
            x_vector,
            y_vector,
            x_length_meters,
            y_length_meters,
            x_response,
            y_response,
            response,
            location
        ),
        lambda: tf.reduce_sum(injection, axis=1, keepdims=True)
    )
    
    return scale(
        projected,
        onsource,
        scaling_parameters,
        sample_rate_hertz,
        scaling_type
    )

@tf.function(jit_compile=True, reduce_retracing=True)
def add_to_onsource(
        injection,
        onsource, 
        scaling_parameters,
        sample_rate_hertz,
        scaling_type,
        seed,
        x_vector,
        y_vector,
        x_length_meters,
        y_length_meters,
        x_response,
        y_response,
        response,
        location,
        num_detectors, 
        scaling_ordinality
    ):

    if scaling_ordinality == "BEFORE_PROJECTION":
        scaled_injection = handle_before_projection(
            injection,
            onsource, 
            scaling_parameters,
            sample_rate_hertz,
            scaling_type,
            seed,
            x_vector,
            y_vector,
            x_length_meters,
            y_length_meters,
            x_response,
            y_response,
            response,
            location
        )
    else: 
        scaled_injection = handle_after_projection(
            injection,
            onsource, 
            scaling_parameters,
            sample_rate_hertz,
            scaling_type,
            seed,
            x_vector,
            y_vector,
            x_length_meters,
            y_length_meters,
            x_response,
            y_response,
            response,
            location,
            num_detectors
        )
    
    # Add scaled injection to onsource
    return onsource + scaled_injection, scaled_injection


#@tf.function
def process_single_injection(
    injection: tf.Tensor,
    onsource: tf.Tensor,
    scaling_parameters: tf.Tensor,
    scaling_ordinality,
    scaling_type,
    num_detectors,
    network,
    sample_rate_hertz: float,
    parameters_to_return: list,
    onsource_duration_seconds: float
) -> Tuple[tf.Tensor, Optional[tf.Tensor], Dict[str, tf.Tensor]]:

    injection = ensure_last_dim_even(injection)
    return_variables = {}

    onsource, scaled_injection = add_to_onsource(
        injection,
        onsource, 
        scaling_parameters,
        sample_rate_hertz,
        scaling_type,
        network.rng.integers(1E10, size=2),
        network.x_vector,
        network.y_vector,
        network.x_length_meters,
        network.y_length_meters,
        network.x_response,
        network.y_response,
        network.response,
        network.location,
        num_detectors,
        scaling_ordinality
    )

    if 'HPEAK' in parameters_to_return:
        hpeak_type_value = ScalingTypes.HPEAK.name
        return_variables['HPEAK'] = tf.cond(
            tf.not_equal(
                scaling_type,
                hpeak_type_value
            ),
            lambda: calculate_hpeak(injection),
            lambda: tf.constant(0.0)
        )
    
    if 'SNR' in parameters_to_return:
        snr_type_value = ScalingTypes.SNR.name
        return_variables['SNR'] = tf.cond(
            tf.not_equal(
                tf.constant(scaling_type, dtype=tf.int32),
                tf.constant(snr_type_value, dtype=tf.int32)
            ),
            lambda: gf.snr(
                scaled_injection,
                onsource,
                sample_rate_hertz,
                fft_duration_seconds=1.0,
                overlap_duration_seconds=0.5
            ),
            lambda: tf.constant(0.0)
        )
    
    if 'HRSS' in parameters_to_return:
        hrss_type_value = ScalingTypes.HRSS.name
        return_variables['HRSS'] = tf.cond(
            tf.not_equal(
                tf.constant(scaling_type, dtype=tf.int32),
                tf.constant(hrss_type_value, dtype=tf.int32)
            ),
            lambda: calculate_hrss(injection),
            lambda: tf.constant(0.0)
        )
    
    cropped_injection = None
    if ('INJECTIONS' in parameters_to_return) or ('WHITENED_INJECTIONS' in parameters_to_return):
        cropped_injection = gf.crop_samples(
            scaled_injection,
            onsource_duration_seconds,
            sample_rate_hertz
        )
    
    return onsource, cropped_injection, return_variables

@dataclass
class InjectionGenerator:
    waveform_generators: Union[
        List[gf.WaveformGenerator], 
        Dict[str, gf.WaveformGenerator]
    ]
    parameters_to_return: Optional[List[WaveformParameters]] = None
    seed: Optional[int] = None
    index: int = 0

    def __post_init__(self):
        # Set default seed if not provided.
        if self.seed is None:
            self.seed = gf.Defaults.seed

        # Use a single RNG to derive all necessary seeds.
        self.rng = default_rng(self.seed)
        seeds = self.rng.integers(0, 1E10, size=5)
        self.generator_rng = default_rng(seeds[0])
        self.parameter_rng = default_rng(seeds[1])
        self.position_rng = default_rng(seeds[2])
        self.waveform_rng = default_rng(seeds[3])
        self.mask_rng = default_rng(seeds[4])

        # Ensure parameters_to_return is a list of waveform parameters.
        if self.parameters_to_return is None:
            self.parameters_to_return = []
        self.parameters_to_return = [
            item for item in self.parameters_to_return 
            if isinstance(item.value, WaveformParameter)
        ]

        # Normalize waveform_generators to a list.
        if not isinstance(self.waveform_generators, list) and not isinstance(self.waveform_generators, dict):
            self.waveform_generators = [self.waveform_generators]

        # Reseed each waveform generator once.
        if isinstance(self.waveform_generators, list):
            for generator in self.waveform_generators:
                generator.reseed(self.parameter_rng.integers(1E10))
            # Create and store iterators once.
            self.iterators = [
                self.generate_one(generator, seed=self.generator_rng.integers(1E10))
                for generator in self.waveform_generators
            ]
        elif isinstance(self.waveform_generators, dict):
            for config in self.waveform_generators.values():
                config["generator"].reseed(self.parameter_rng.integers(1E10))
            self.iterators = [
                self.generate_one(config["generator"], seed=self.generator_rng.integers(1E10), name=key)
                for key, config in self.waveform_generators.items()
            ]
            self.mask_dict = {}
        else:
            raise TypeError("Waveform generators must be a dictionary or a list!")
        
    def __call__(
        self,
        sample_rate_hertz: float = None,
        onsource_duration_seconds: float = None,
        crop_duration_seconds: float = None,
        num_examples_per_generation_batch: int = None,
        num_examples_per_batch: int = None,
    ):
        self.sample_rate_hertz = sample_rate_hertz or gf.Defaults.sample_rate_hertz
        self.onsource_duration_seconds = onsource_duration_seconds or gf.Defaults.onsource_duration_seconds
        self.crop_duration_seconds = crop_duration_seconds or gf.Defaults.crop_duration_seconds
        self.num_examples_per_generation_batch = num_examples_per_generation_batch or gf.Defaults.num_examples_per_generation_batch
        self.num_examples_per_batch = num_examples_per_batch or gf.Defaults.num_examples_per_batch

        if self.num_examples_per_batch > self.num_examples_per_generation_batch:
            logging.warning(
                "num_injections_per_batch must be less than num_examples_per_generation_batch; adjusting."
            )
            self.num_examples_per_generation_batch = self.num_examples_per_batch

        # If no waveform generators provided, yield a single empty result.
        if not self.waveform_generators:
            yield None, None, None

        # Loop indefinitely, yielding batches continuously.
        while True:
            injections = []
            mask = []
            parameters = {key: [] for key in self.parameters_to_return}

            # Get one batch from each iterator.
            for iterator in self.iterators:
                injection_, mask_, parameters_ = next(iterator)
                injections.append(injection_)
                mask.append(mask_)
                for key in parameters:
                    if key in parameters_:
                        parameters[key].append(parameters_[key])
                    else:
                        parameters[key].append(
                            tf.zeros([key.value.shape[-1] * self.num_examples_per_batch], dtype=tf.float64)
                        )

            try:
                injections = tf.stack(injections)
            except Exception as e:
                raise Exception(f"Failed to stack injections because {e}.")
            try:
                mask = tf.stack(mask)
            except Exception as e:
                raise Exception(f"Failed to stack mask because {e}.")
            for key, value in parameters.items():
                try:
                    parameters[key] = tf.stack(value)
                except Exception as e:
                    logging.error(f"Failed to stack injection parameters because {e}")
                    return  # or yield None, None, None as appropriate

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

        if self.rng == None:
            self.rng = default_rng(seed)
            self.position_rng = default_rng(self.rng.integers(1E10))
            self.waveform_rng = default_rng(self.rng.integers(1E10))
            self.mask_rng = default_rng(self.rng.integers(1E10))
        
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
                    self.mask_rng.integers(1E10, size=2)
                )

            if name is not None:
                self.mask_dict[name] = mask

                if self.waveform_generators[name]["excluded"] is not None:

                    excluded = self.waveform_generators[name]["excluded"]

                    mask = tf.math.logical_and(
                        mask,
                        tf.math.logical_not(self.mask_dict[excluded])
                    )

            # Convert mask to integer type.
            mask_int = tf.cast(mask, tf.int32)

            # Determine the rank of the mask.
            if mask.shape.ndims is None:
                # If static rank is not available, compute it dynamically.
                mask_rank = tf.rank(mask_int).numpy()
            else:
                mask_rank = mask.shape.ndims

            # Compute num_waveforms depending on the rank.
            if mask_rank == 0:
                # If mask is a scalar, use it directly.
                num_waveforms_val = mask_int.numpy()
            else:
                try:
                    # When mask is non-scalar, sum all entries.
                    num_waveforms_val = tf.reduce_sum(mask_int).numpy()
                except Exception as e:
                    raise ValueError(
                        f"Error reducing mask with shape {mask.shape}: {e}"
                    ) from e

            # Ensure that the reduced result is a scalar.
            if not np.isscalar(num_waveforms_val):
                try:
                    num_waveforms_val = num_waveforms_val.item()
                except Exception as e:
                    raise ValueError(
                        f"Expected scalar from reduction but got array with shape {np.shape(num_waveforms_val)}"
                    ) from e

            # Convert to an integer.
            num_waveforms = int(num_waveforms_val)

            # Convert to an int explicitly (if it isn't already) and check the condition.
            num_waveforms = int(num_waveforms)
            if num_waveforms > 0:
                waveforms, parameters = generator.generate(
                    num_waveforms, 
                    self.sample_rate_hertz,
                    total_duration_seconds,
                    self.waveform_rng.integers(1E10)
                )
                
                # Convert to tensorflow tensor:
                waveforms = tf.convert_to_tensor(waveforms, dtype = tf.float32)
                #Roll Tensor to randomise start time:
                waveforms = roll_vector_zero_padding( 
                        waveforms, 
                        min_roll_num_samples, 
                        max_roll_num_samples, 
                        self.position_rng.integers(1E10, size=2)
                    )
                
                # Create zero filled injections to fill spots where injection 
                # did not generate due to injection masks:
                injections = gf.expand_tensor(waveforms, mask)
            else:
                shape = (num_batches, 2, total_duration_num_samples)
                assert isinstance(shape, tuple), f"Shape should be a tuple {shape}"
                assert all(isinstance(dim, int) for dim in shape), f"All dimensions in shape should be integers {shape}"
                injections = tf.zeros(
                    shape=shape
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

                    expanded_parameters[key] = gf.expand_tensor(
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

           # Unstack the batched tensors to get a list of smaller tensors (one per batch)
            injections_list = tf.unstack(injections, axis=0)
            mask_list = tf.unstack(mask, axis=0)

            # Iterate over the resulting list of tensors.
            for injection_, mask_, parameter_ in zip(injections_list, mask_list, parameters):
                try:
                    yield injection_, mask_, parameter_
                except Exception as e:
                    logging.error(f"Failed to yield because: {e}")
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

        mask = mask.numpy()  # Converts to NumPy (can be memory intensive for large tensors)
        num_injections = np.sum(mask)
        
        match generator.scaling_method.value:
            case np.ndarray():
                scaling_parameters = []
                for index in range(self.index, self.index + num_injections):
                    if index < len(generator.scaling_method.value):
                        scaling_parameters.append(generator.scaling_method.value[index])
                    else:
                        scaling_parameters.append(generator.scaling_method.value[-1])
                    self.index += 1
            case gf.Distribution():
                scaling_parameters = generator.scaling_method.value.sample(num_injections)
            case _:
                raise TypeError("Unsupported scaling method value type: " 
                                f"{type(generator.scaling_method.value)}!")
        
        scaling_parameters = tf.convert_to_tensor(scaling_parameters, dtype=tf.float32)
        scaling_parameters = gf.expand_tensor(scaling_parameters, mask)
        
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
                        tf.squeeze(mask),
                        config["generator"]
                    )
                )
        else: 
            for mask, generator in zip(masks, self.waveform_generators):
                scaling_parameters.append(
                    self.generate_scaling_parameters_(
                        tf.squeeze(mask),
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
        
        onsource = ensure_last_dim_even(onsource)

        try:
            # Unstack the batched tensors to get a list of smaller tensors (one per batch)
            injections_list = tf.unstack(injections, axis=0)
            mask_list = tf.unstack(mask, axis=0)
        except Exception as e:
            logging.error("Splitting failed for some reason: %s", e)
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

            onsource, cropped_injections, return_variables = process_single_injection(
                injections_,
                onsource,
                scaling_parameters_,
                waveform_generator.scaling_method.type_.value.ordinality.name,
                waveform_generator.scaling_method.type_.name,
                waveform_generator.network.num_detectors,
                waveform_generator.network,
                self.sample_rate_hertz,
                [param.name for param in parameters_to_return],
                self.onsource_duration_seconds
            )
                                        
        if (ReturnVariables.INJECTIONS in parameters_to_return) or \
            (ReturnVariables.WHITENED_INJECTIONS in parameters_to_return):
            try:
                cropped_injections = tf.stack(cropped_injections)
                cropped_injections = gf.replace_nan_and_inf_with_zero(
                    cropped_injections
                )
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


@tf.function(
    input_signature=[
        # Expect num_injections as a scalar int32.
        tf.TensorSpec(shape=[], dtype=tf.int32, name="num_injections"),
        # Expect injection_chance as a scalar float32.
        tf.TensorSpec(shape=[], dtype=tf.float32, name="injection_chance"),
        # Expect seed as a vector of length 2 of type int32.
        tf.TensorSpec(shape=[2], dtype=tf.int32, name="seed")
    ],
    reduce_retracing=True
)
def generate_mask(num_injections, injection_chance, seed) -> tf.Tensor:
    """
    Generate injection masks using TensorFlow.

    Parameters:
      num_injections: A scalar int32 tensor representing the number of masks to generate.
      injection_chance: A scalar float32 tensor giving the probability an injection is True.
      seed: A 1-D int32 tensor of shape [2] used for the stateless random op.

    Returns:
      A boolean tensor of shape (num_injections,) containing the injection masks.
    """
    # Compute logits for [False, True] categories.
    logits = tf.math.log([1.0 - injection_chance, injection_chance])
    
    # Generate categorical random variables.
    # tf.random.stateless_categorical returns a tensor of shape [batch_size, num_samples]
    sampled_indices = tf.random.stateless_categorical(
        seed=seed,
        logits=tf.reshape(logits, [1, -1]),
        num_samples=num_injections
    )
    
    # Reshape the output to [num_injections] and convert to booleans.
    injection_masks = tf.reshape(sampled_indices, [num_injections])
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