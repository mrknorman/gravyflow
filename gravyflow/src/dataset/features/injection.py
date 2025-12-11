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
import keras
from keras import ops
import jax.numpy as jnp
import jax
from numpy.random import default_rng  

# Local imports:
import gravyflow as gf
from gravyflow.src.dataset.features.waveforms.cbc import (
    generate_cbc_waveform, 
    calc_duration_from_f_min,
    Approximant
)

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
    value : Union['gf.Distribution', 'gf.Curriculum', np.ndarray]
    type_ : ScalingTypes
    
    def scale(
        self,
        injections, 
        onsource,
        scaling_parameters,
        sample_rate_hertz : float,
        onsource_duration_seconds: float = None
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
                    overlap_duration_seconds=0.5,
                    onsource_duration_seconds=onsource_duration_seconds
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
            pass
        
        return scaled_injections

def scale(
        injections,
        onsource,
        scaling_parameters,
        sample_rate_hertz: float,
        scaling_type: str
    ):
    
    """
    Function to scale injections based on different scaling types.
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

def calculate_hrss(
    injection
    ):
    
    return ops.sqrt(
        ops.sum(
            ops.sum(injection*injection, axis = 1), 
            axis = -1
        )
    )

def calculate_hpeak(
    injection
    ):
    
    return ops.max(ops.abs(injection), axis=-1)

def scale_to_hrss(
    injection, 
    desired_hrss
    ):
    
    epsilon = 1.0E-7
    
    current_hrss = calculate_hrss(
        injection
    )
    
    scale_factor = desired_hrss/(current_hrss + epsilon)
    
    if len(ops.shape(scale_factor)) == 1: 
        scale_factor = ops.reshape(scale_factor, (-1, 1))
        
    scale_factor = ops.expand_dims(scale_factor, axis = 1)
    
    return injection*scale_factor

def scale_to_hpeak(
    injection, 
    desired_hpeak
    ):
    
    epsilon = 1.0E-7
    
    current_hpeak = calculate_hpeak(
        injection
    )
    
    scale_factor = desired_hpeak/(current_hpeak + epsilon)
    
    if len(ops.shape(scale_factor)) == 1: 
        scale_factor = ops.reshape(scale_factor, (-1, 1))
    
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
    CENTRAL_TIME = ReturnVariable(9)
    
    def __lt__(self, other):
        return self.value.index < other.value.index

@dataclass
class WaveformGenerator:
    scaling_method : Union[ScalingMethod, None] = None
    injection_chance : float = 1.0
    front_padding_duration_seconds : float = 0.0
    back_padding_duration_seconds : float = 0.3
    scale_factor : Union[float, None] = None
    network : Union[List[gf.IFO], gf.Network, Path, None] = None

    distributed_attributes : Union[Tuple, None] = None
        
    def __post_init__(self):
        self.network = self.init_network(self.network)
        self.convert_all_distributions()
        
        # Deep copy all distributed attributes to ensure they are independent instances
        # This prevents shared state if the same Distribution object was passed to multiple arguments
        # Deep copy all distributed attributes to ensure they are independent instances
        # This prevents shared state if the same Distribution object was passed to multiple arguments
        if self.distributed_attributes is not None:
            for attribute in self.distributed_attributes:
                val = getattr(self, attribute)
                if isinstance(val, gf.Distribution):
                    setattr(self, attribute, deepcopy(val))
                
        # Ensure attributes are decorrelated by default
        self.reseed(gf.Defaults.seed)

    def get_max_generated_duration(self):
        return 0.0

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
        network : Union[List[gf.IFO], gf.Network, Path] = None
    ) -> Type[cls]:

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

        with path.open("r") as file:
            config = json.load(file)

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
            
        elif "scaling_type" in config and "scaling_distribution" in config:
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
        waveform_cls = None
        match config.pop("type"):
            case 'PhenomD': 
                waveform_cls = CBCGenerator
            case 'WNB':
                waveform_cls = WNBGenerator
            case _:
                raise ValueError("This waveform type is not implemented.")
                
        generator = waveform_cls(
            scaling_method=config.pop("scaling_method"),
            scale_factor=config.pop("scale_factor", None),
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
        if self.distributed_attributes is not None:
            for attribute in self.distributed_attributes:
                converted = self.convert_to_distribution(
                    getattr(self, attribute)
                )
                setattr(self, attribute, converted)
    
    def reseed(self, seed):

        rng = default_rng(seed)
        if self.distributed_attributes is not None:
            for attribute in self.distributed_attributes:
                distribution = getattr(self, attribute)
                seed = rng.integers(1000000000)
                if isinstance(distribution, gf.Distribution):
                    distribution.reseed(seed)
                    setattr(self, attribute, distribution)
                
    def apply_injection_chance(self, waveforms, seed):
        """
        Randomly zeros out waveforms based on injection_chance.
        """
        if self.injection_chance >= 1.0:
            return waveforms
            
        num_waveforms = ops.shape(waveforms)[0]
        
        # Use JAX random if possible for JIT compatibility, or numpy if eager
        # Since we are likely inside a JITted context or using Keras ops, 
        # we should try to use Keras ops or JAX.
        
        # We need a key for JAX
        key = jax.random.PRNGKey(seed)
        
        # Split key to avoid correlation with other uses of seed
        key, subkey = jax.random.split(key)
        
        mask = jax.random.bernoulli(subkey, p=self.injection_chance, shape=(num_waveforms,))
        mask = ops.cast(mask, waveforms.dtype)
        
        # Reshape for broadcasting: (Batch, 1, 1)
        # Assuming waveforms is (Batch, Channels, Time)
        mask = ops.reshape(mask, (num_waveforms, 1, 1))
        
        return waveforms * mask

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
    LAMBDA_1 = WaveformParameter(110)
    LAMBDA_2 = WaveformParameter(111)
    
    # WNB paramters:
    DURATION_SECONDS = WaveformParameter(201)
    MIN_FREQUENCY_HERTZ = WaveformParameter(202)
    MAX_FREQUENCY_HERTZ = WaveformParameter(203)
    
    # Simple waveform parameters:
    FREQUENCY_HERTZ = WaveformParameter(301)
    AMPLITUDE = WaveformParameter(302)
    PHASE_RADIANS = WaveformParameter(303)
    QUALITY_FACTOR = WaveformParameter(304)
    TAU_SECONDS = WaveformParameter(305)
    START_FREQUENCY_HERTZ = WaveformParameter(306)
    END_FREQUENCY_HERTZ = WaveformParameter(307)
    CHIRP_RATE = WaveformParameter(308)
    DAMPING_TIME_SECONDS = WaveformParameter(309)
    
    @classmethod
    def get(cls, key):
        member = cls.__members__.get(key.upper()) 
        
        if member is None:
            raise ValueError(f"{key} not found in WaveformParameters")
        
        return member
    
    def __lt__(self, other):
        return self.value.index < other.value.index

def is_not_inherited(obj, attribute):
    return attribute in obj.__dict__ or attribute in obj.__class__.__annotations__

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
    
    def get_max_generated_duration(self):
        if isinstance(self.duration_seconds, gf.Distribution):
            if self.duration_seconds.type_ == gf.DistributionType.CONSTANT:
                return self.duration_seconds.value
            return self.duration_seconds.max_
        return self.duration_seconds
    
    def generate(
        self,
        num_waveforms: int,
        sample_rate_hertz: float,
        duration_seconds: float,
        seed : int
    ):
        
        if (num_waveforms > 0):
            
            if seed is not None:
                self.reseed(seed)
            
            if self.duration_seconds.type_ is not gf.DistributionType.CONSTANT:
                if self.duration_seconds.max_ > duration_seconds:                
                    warn("Max duration distibution is greater than requested "
                         "injection duration. Adjusting", UserWarning)
                    self.duration_seconds.max_ = duration_seconds

                if self.duration_seconds.min_ < 0.0 and self.duration_seconds.type_ is not gf.DistributionType.CONSTANT:

                    warn("Min duration distibution is less than zero "
                         "injection duration. Adjusting", UserWarning)

                    self.duration_seconds.min_ = 0.0
            
            parameters = {}
            for attribute in self.distributed_attributes:
                value = getattr(self, attribute)
                
                parameter = None
                try:
                    parameter = WaveformParameters.get(attribute)
                except:
                    try:
                        parameter = ScalingTypes.get(attribute)
                    except:
                        continue # Skip if not found
                
                shape = parameter.value.shape[-1]                
                parameters[attribute] = ops.convert_to_tensor(
                    value.sample(num_waveforms * shape)
                )
                    
            parameters["max_frequency_hertz"] = ops.maximum(
                parameters["max_frequency_hertz"], 
                parameters["min_frequency_hertz"]
            )
            parameters["min_frequency_hertz"] = ops.minimum(
                parameters["max_frequency_hertz"],
                parameters["min_frequency_hertz"]
            )
                                    
            waveforms = gf.wnb(
                num_waveforms,
                sample_rate_hertz, 
                duration_seconds, 
                **parameters,
                seed=int(seed)
            )
            
            waveforms *= self.scale_factor
            waveforms = ensure_last_dim_even(waveforms)
            
            # Apply injection chance (randomly zero out waveforms)
            if seed is not None:
                waveforms = self.apply_injection_chance(waveforms, seed)
            
            # Convert parameters to Enum keys for return
            enum_parameters = {}
            for key, value in parameters.items():
                try:
                    enum_key = WaveformParameters.get(key)
                    enum_parameters[enum_key] = value
                except ValueError:
                    try:
                        enum_key = ScalingTypes.get(key)
                        enum_parameters[enum_key] = value
                    except ValueError:
                        # Fallback for keys that don't map to Enums
                        enum_parameters[key] = value
            
            return waveforms, enum_parameters

def ensure_last_dim_even(tensor):
    last_dim = ops.shape(tensor)[-1]
    is_even = (last_dim % 2) == 0
    
    if is_even:
        return tensor
    else:
        return tensor[..., :-1]

@dataclass
class CBCGenerator(WaveformGenerator):
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
    lambda_1 : Union[float, gf.Distribution] = 0.0
    lambda_2 : Union[float, gf.Distribution] = 0.0
    approximant : Approximant = Approximant.IMRPhenomD
    min_frequency_hertz : Union[float, gf.Distribution] = 20.0
    
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
        "spin_2_in",
        "lambda_1",
        "lambda_2",
        "min_frequency_hertz"
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
        self.lambda_1 = self.ensure_float("lambda_1", self.lambda_1)
        self.lambda_2 = self.ensure_float("lambda_2", self.lambda_2)
        self.min_frequency_hertz = self.ensure_float("min_frequency_hertz", self.min_frequency_hertz)

        self.spin_1_in = self.ensure_list_of_floats("spin_1_in", self.spin_1_in)
        self.spin_2_in = self.ensure_list_of_floats("spin_2_in", self.spin_2_in)

        # Parameter validation warnings
        self._validate_parameters()

        if self.scale_factor is None:
            self.scale_factor=gf.Defaults.scale_factor

        super().__post_init__()
    
    def _validate_parameters(self):
        """Validate CBC parameters and warn if outside sensible ranges."""
        def get_range(val):
            """Get min/max from a value or distribution."""
            if isinstance(val, gf.Distribution):
                if val.type_ == gf.DistributionType.CONSTANT:
                    return val.value, val.value
                return val.min_, val.max_
            return val, val
        
        # Mass validation (sensible range: 1-200 Msun for CBC)
        m1_min, m1_max = get_range(self.mass_1_msun)
        m2_min, m2_max = get_range(self.mass_2_msun)
        
        if m1_max > 200 or m2_max > 200:
            warn(
                f"CBCGenerator: Mass values >200 Msun may produce NaN waveforms. "
                f"mass_1 range: [{m1_min}, {m1_max}], mass_2 range: [{m2_min}, {m2_max}]"
            )
        if m1_min < 1 or m2_min < 1:
            warn(
                f"CBCGenerator: Mass values <1 Msun may produce invalid waveforms. "
                f"mass_1 range: [{m1_min}, {m1_max}], mass_2 range: [{m2_min}, {m2_max}]"
            )
        
        # Distance validation (sensible range: 1-10000 Mpc)
        d_min, d_max = get_range(self.distance_mpc)
        if d_max > 10000:
            warn(
                f"CBCGenerator: Distance >10000 Mpc may produce very weak signals. "
                f"distance range: [{d_min}, {d_max}] Mpc"
            )
        if d_min < 1:
            warn(
                f"CBCGenerator: Distance <1 Mpc is unrealistically close. "
                f"distance range: [{d_min}, {d_max}] Mpc"
            )
        
        # Inclination validation (sensible range: 0 to pi)
        inc_min, inc_max = get_range(self.inclination_radians)
        if inc_max > 3.15 or inc_min < 0:
            warn(
                f"CBCGenerator: Inclination should be in [0, pi] radians. "
                f"inclination range: [{inc_min}, {inc_max}]"
            )
        
        # Eccentricity validation (sensible range: 0 to 1)
        ecc_min, ecc_max = get_range(self.eccentricity)
        if ecc_max >= 1 or ecc_min < 0:
            warn(
                f"CBCGenerator: Eccentricity should be in [0, 1). "
                f"eccentricity range: [{ecc_min}, {ecc_max}]"
            )
        
        # Minimum frequency validation (sensible range: 5-100 Hz)
        f_min, f_max = get_range(self.min_frequency_hertz)
        if f_min < 5:
            warn(
                f"CBCGenerator: min_frequency <5 Hz may require very long waveforms. "
                f"min_frequency range: [{f_min}, {f_max}] Hz"
            )
    
    def get_max_generated_duration(self):
        # Calculate max duration based on min frequency and min masses
        
        def get_min(val):
            if isinstance(val, gf.Distribution):
                if val.type_ == gf.DistributionType.CONSTANT:
                    return val.value
                return val.min_
            return val
            
        m1 = get_min(self.mass_1_msun)
        m2 = get_min(self.mass_2_msun)
        f_min = get_min(self.min_frequency_hertz)
        
        # calc_duration_from_f_min expects tensors or floats
        # It returns a tensor if inputs are tensors, or float if floats (if using numpy/python math, but it uses ops)
        # We need a float return.
        
        # Create dummy tensors for calculation
        m1_t = ops.convert_to_tensor(m1, dtype="float32")
        m2_t = ops.convert_to_tensor(m2, dtype="float32")
        f_min_t = ops.convert_to_tensor(f_min, dtype="float32")
        
        duration = calc_duration_from_f_min(m1_t, m2_t, f_min_t)
        
        return float(duration)
    
    def generate(
            self,
            num_waveforms : int,
            sample_rate_hertz : float,
            duration_seconds : float,
            seed : int = None
        ):
        
        if (num_waveforms > 0):
            
            if seed is not None:
                self.reseed(seed)
            
            parameters = {}
            for attribute in self.distributed_attributes:
                value = getattr(self, attribute)
                
                parameter = None
                try:
                    parameter = WaveformParameters.get(attribute)
                except:
                    try:
                        parameter = ScalingTypes.get(attribute) 
                    except:
                        continue

                shape = parameter.value.shape[-1]         
                
                if not isinstance(value, tuple) and  not isinstance(value, list): 
                    parameters[attribute] = value.sample(num_waveforms * shape)
                else:
                    combined_array = []
                    for element in value:
                        combined_array += element.sample(num_waveforms)
                    parameters[attribute] = combined_array

            waveforms = None
            
            mass_1 = parameters["mass_1_msun"]
            mass_2 = parameters["mass_2_msun"]
            
            parameters["mass_1_msun"] = ops.maximum(mass_1, mass_2)
            parameters["mass_2_msun"] = ops.minimum(mass_1, mass_2)
            
            # Wrap inclination to [0, pi] to avoid LAL waveform issues with values like 20.0
            if "inclination_radians" in parameters:
                inc = parameters["inclination_radians"]
                inc = ops.convert_to_tensor(inc, dtype="float32")
                # Modulo 2*pi
                inc = inc % (2.0 * np.pi)
                # Reflect if > pi
                inc = ops.where(inc > np.pi, 2.0 * np.pi - inc, inc)
                parameters["inclination_radians"] = inc
            
            # Add duration to parameters
            parameters["duration_seconds"] = ops.full((num_waveforms,), duration_seconds)

            # Calculate required duration for f_min
            f_min = parameters.get("min_frequency_hertz", 20.0)
            f_min = ops.convert_to_tensor(f_min, dtype="float32")
            
            required_durations = calc_duration_from_f_min(
                parameters["mass_1_msun"],
                parameters["mass_2_msun"],
                f_min
            )
            
            # Find max required duration in batch
            max_required_duration = ops.max(required_durations)
            
            # Convert to float for JIT compatibility (triggers sync)
            max_required_duration = float(max_required_duration)
            
            # Use the larger of requested duration or required duration + half duration (to accommodate merger placement)
            # We need start_time >= 0.
            # start_time = tc - max_required_duration
            # tc = gen_duration - duration_seconds / 2.0
            # So gen_duration - duration_seconds / 2.0 - max_required_duration >= 0
            # gen_duration >= max_required_duration + duration_seconds / 2.0
            # We add a 2.0s safety buffer to account for approximations in calc_duration_from_f_min
            gen_duration = max(duration_seconds, max_required_duration + (duration_seconds / 2.0) + 2.0)
            
            # Snap to next power of 2 to reduce recompilation
            gen_duration = 2**np.ceil(np.log2(gen_duration))
            
            # Calculate coalescence time to place merger in the center of the *requested* duration
            # relative to the end of the generated waveform.
            # We want to keep the last `duration_seconds`.
            # And we want the merger to be at `duration_seconds / 2.0` from the start of that chunk.
            # So `tc` relative to start of generated waveform is:
            # (gen_duration - duration_seconds) + (duration_seconds / 2.0)
            # = gen_duration - duration_seconds / 2.0
            
            coalescence_time = gen_duration - (duration_seconds / 2.0)
            
            # Create kwargs for CBC generator, removing parameters it doesn't accept
            cbc_kwargs = parameters.copy()
            if "min_frequency_hertz" in cbc_kwargs:
                cbc_kwargs.pop("min_frequency_hertz")
            if "duration_seconds" in cbc_kwargs:
                cbc_kwargs.pop("duration_seconds")

            waveforms = generate_cbc_waveform(
                    num_waveforms=num_waveforms, 
                    sample_rate_hertz=sample_rate_hertz, 
                    duration_seconds=gen_duration,
                    approximant=self.approximant,
                    coalescence_time=coalescence_time,
                    **cbc_kwargs
            )
            
            # Crop to the last `duration_seconds`
            num_samples_keep = int(duration_seconds * sample_rate_hertz)
            waveforms = waveforms[..., -num_samples_keep:]
            
            waveforms *= self.scale_factor

            waveforms = ensure_last_dim_even(waveforms)
            
            # Apply injection chance (randomly zero out waveforms)
            if seed is not None:
                waveforms = self.apply_injection_chance(waveforms, seed)
            
            # Convert parameters to Enum keys for return
            enum_parameters = {}
            for key, value in parameters.items():
                try:
                    enum_key = WaveformParameters.get(key)
                    enum_parameters[enum_key] = value
                except ValueError:
                    try:
                        enum_key = ScalingTypes.get(key)
                        enum_parameters[enum_key] = value
                    except ValueError:
                        # Fallback for keys that don't map to Enums
                        enum_parameters[key] = value
        
            return waveforms, enum_parameters


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
            seed = rng.integers(1000000000)
            generator.reseed(seed)
    
    def generate(
        self,
        num_waveforms: int,
        sample_rate_hertz: float,
        duration_seconds : float,
        seed: int = None
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
            waveforms = ops.stack(waveforms, axis = 1)
        except: 
            logging.error("Failed to stack waveforms!")
            return None, None
        
        parameters = parameters[0]

        return waveforms, parameters

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
        location
    ):
    
    projected = gf.project_wave(
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
    )
    
    return scale(
        projected,
        onsource,
        scaling_parameters,
        sample_rate_hertz,
        scaling_type
    )

def roll_vector_zero_padding(vector, shift):
    """
    Rolls a vector by shift, filling with zeros instead of wrapping.
    vector: (Batch, Time) or (Batch, Channels, Time)
    shift: (Batch,) or int
    """
    def roll_one(vec, s):
        s = s.astype("int32")
        length = vec.shape[-1]
        indices = jnp.arange(length)
        shifted_indices = indices - s
        valid = (shifted_indices >= 0) & (shifted_indices < length)
        clamped_indices = jnp.clip(shifted_indices, 0, length - 1)
        # jnp.take is slightly different from ops.take?
        # ops.take(vec, indices, axis=-1)
        # jnp.take(vec, indices, axis=-1)
        val = jnp.take(vec, clamped_indices, axis=-1)
        if len(vec.shape) > 1:
            valid = jnp.expand_dims(valid, 0)
        return val * valid.astype(vec.dtype)

    return jax.vmap(roll_one)(vector, shift)

def generate_mask(vector):
    """
    Generates a mask where vector is non-zero.
    """
    return ops.cast(
        ops.logical_and(
            ops.not_equal(vector, 0.0),
            ops.logical_not(ops.isnan(vector))
        ),
        "float32"
    )

def batch_injection_parameters(parameters_list):
    """
    Combines a list of parameter dictionaries into a single dictionary.
    Handles missing keys by padding with NaNs.
    """
    if not parameters_list:
        return {}
        
    # Get union of all keys
    all_keys = set().union(*(d.keys() for d in parameters_list))
    batched = {}
    
    for key in all_keys:
        values = []
        for p in parameters_list:
            if key in p:
                values.append(p[key])
            else:
                # If key is missing, we need to append a placeholder.
                # We need to know the shape and dtype of the other values to create a compatible placeholder.
                # We can look at other values in the list (forward or backward).
                
                # Find a valid example
                valid_example = next((d[key] for d in parameters_list if key in d), None)
                
                if valid_example is not None:
                    # Create NaN tensor of same shape/dtype
                    if ops.is_tensor(valid_example):
                        shape = ops.shape(valid_example)
                        # Create full of NaNs
                        # ops.full might not support NaN for all backends/dtypes directly, 
                        # but we can multiply ones by NaN.
                        nan_val = float('nan')
                        placeholder = ops.ones_like(valid_example) * nan_val
                    else:
                        placeholder = None # Or float('nan')?
                    
                    values.append(placeholder)
                else:
                    # Should not happen if key is in union
                    values.append(None)

        try:
            # Check if any values are None (non-tensor placeholders)
            if any(v is None for v in values):
                 batched[key] = values
            else:
                batched[key] = ops.stack(values, axis=0)
        except:
            batched[key] = values
            
    return batched

class InjectionGenerator:
    def __init__(
        self,
        waveform_generators: List[WaveformGenerator],
        parameters_to_return: List = None,
        seed: int = None
    ):
        self.waveform_generators = waveform_generators
        if not isinstance(self.waveform_generators, list) and not isinstance(self.waveform_generators, dict):
            self.waveform_generators = [self.waveform_generators]

        if parameters_to_return is None:
            self.parameters_to_return = []
        else:
            self.parameters_to_return = parameters_to_return
        
        # Use gf.Defaults.seed if no seed provided for determinism
        if seed is None:
            seed = gf.Defaults.seed
        self.seed = seed
        self.rng = default_rng(seed)
        self.sample_rate_hertz = gf.Defaults.sample_rate_hertz # Default
        
    def __call__(
        self,
        sample_rate_hertz: float = None,
        onsource_duration_seconds: float = None,
        crop_duration_seconds: float = None,
        num_examples_per_generation_batch: int = None,
        num_examples_per_batch: int = None
    ):
        if sample_rate_hertz is None:
            sample_rate_hertz = gf.Defaults.sample_rate_hertz
        if onsource_duration_seconds is None:
            onsource_duration_seconds = gf.Defaults.onsource_duration_seconds
        if crop_duration_seconds is None:
            crop_duration_seconds = gf.Defaults.crop_duration_seconds
        if num_examples_per_generation_batch is None:
            num_examples_per_generation_batch = gf.Defaults.num_examples_per_generation_batch
        if num_examples_per_batch is None:
            num_examples_per_batch = gf.Defaults.num_examples_per_batch

        self.sample_rate_hertz = sample_rate_hertz
        
        while True:
            injections_list = []
            masks_list = []
            parameters_list = []
            
            total_duration = onsource_duration_seconds + 2 * crop_duration_seconds
            
            # Calculate max padding and delay across generators
            max_front_padding = 0.0
            max_back_padding = 0.0
            max_delay = 0.0
            
            for generator in self.waveform_generators:
                max_front_padding = max(max_front_padding, generator.front_padding_duration_seconds)
                max_back_padding = max(max_back_padding, generator.back_padding_duration_seconds)
                if generator.network is not None:
                     max_delay = max(max_delay, float(generator.network.max_arrival_time_difference_seconds))
            
            # Add safety buffer
            # We need enough extra duration to cover the random shifts (padding) and detector delays
            # The shift can be up to max_back_padding (or max_front_padding in negative direction).
            # To be safe, we need the valid signal to be larger than the crop window by at least 2 * max_shift.
            # extra_duration = 2.0 * (max(max_front_padding, max_back_padding) + max_delay) + 1.0
            
            max_shift = max(max_front_padding, max_back_padding)
            extra_duration = 2.0 * (max_shift + max_delay) + 1.0
            
            # Calculate max signal duration required by generators
            max_signal_duration = 0.0
            for generator in self.waveform_generators:
                max_signal_duration = max(max_signal_duration, generator.get_max_generated_duration())
            
            # We need to generate at least enough to cover the signal + extra buffer
            # But we also need to cover the requested total_duration (analysis window)
            # Actually, total_duration is the analysis window + crop buffers.
            # If the signal is longer than total_duration, we need to generate the full signal length
            # to avoid cropping the start.
            
            gen_duration = max(total_duration, max_signal_duration) + extra_duration
            
            for generator in self.waveform_generators:
                seed = self.rng.integers(1000000000)
                
                min_shift = -int(generator.front_padding_duration_seconds * sample_rate_hertz)
                max_shift = int(generator.back_padding_duration_seconds * sample_rate_hertz)
                
                # Calculate required extended duration to allow centering the merger
                # We need enough room on both sides to shift the window.
                # extended_duration = total_duration + 2 * max(|min_shift|, |max_shift|)
                max_abs_shift = max(abs(min_shift), abs(max_shift))
                extended_duration = max(total_duration, max_signal_duration) + extra_duration + 2.0 * max_abs_shift / sample_rate_hertz
                
                waveforms, params = generator.generate(
                    num_waveforms=num_examples_per_batch,
                    sample_rate_hertz=sample_rate_hertz,
                    duration_seconds=extended_duration,
                    seed=seed
                )
                
                mask = generate_mask(waveforms)
                
                # Check for NaN injections that silently zero the mask
                num_nan_injections = int(ops.sum(ops.any(ops.isnan(waveforms), axis=(1, 2))))
                if num_nan_injections > 0:
                    logging.warning(
                        f"InjectionGenerator: {num_nan_injections}/{num_examples_per_batch} "
                        f"waveforms contain NaN values and will have mask=0. "
                        f"This may indicate an issue with waveform parameters."
                    )
                
                # Reduce mask to a single boolean flag per example (0.0 or 1.0)
                mask = ops.max(mask, axis=(1, 2))
                masks_list.append(mask)

                # Ensure min < max
                if min_shift >= max_shift:
                    shifts = jnp.zeros((num_examples_per_batch,), dtype="int32")
                else:
                    # Generate random shifts for each example in the batch
                    shifts = self.rng.integers(min_shift, max_shift, size=(num_examples_per_batch,))
                    shifts = jnp.array(shifts, dtype="int32")
                
                # Apply shift by slicing
                # We generated a waveform of length `extended_duration`.
                # The merger is centered in this waveform (at extended_duration / 2).
                # We want to slice a window of length `total_duration` such that the merger ends up at `total_duration / 2 + shift`.
                # Let `start` be the slice start index.
                # `tc_sliced = tc_gen - start`.
                # `tc_gen = extended_duration / 2`.
                # `tc_sliced = total_duration / 2 + shift`.
                # `extended_duration / 2 - start = total_duration / 2 + shift`.
                # `start = extended_duration / 2 - total_duration / 2 - shift`.
                # `start = (extended_duration - total_duration) / 2 - shift`.
                
                extended_samples = waveforms.shape[-1]
                target_length = int(total_duration * sample_rate_hertz)
                
                # Center offset
                center_offset = (extended_samples - target_length) // 2
                
                slice_starts = center_offset - shifts
                
                def slice_one(w, s):
                    # w: (..., Time)
                    # s: scalar int
                    # We slice along the last axis
                    rank = len(w.shape)
                    zeros = [jnp.array(0, dtype="int32")] * (rank - 1)
                    start_indices = tuple(zeros + [s.astype("int32")])
                    
                    slice_sizes = w.shape[:-1] + (target_length,)
                    
                    return jax.lax.dynamic_slice(w, start_indices, slice_sizes)

                waveforms = jax.vmap(slice_one)(waveforms, slice_starts)
                
                # Update coalescence time
                # The merger time in the generated waveform is `coalescence_time`.
                # In the sliced waveform, the time axis is shifted.
                # `t_sliced = t_gen - slice_start`.
                # So `tc_sliced = tc_gen - slice_start`.
                # slice_start is in samples. Convert to seconds.
                if "coalescence_time" in params:
                    # params["coalescence_time"] is (Batch,)
                    # slice_starts is (Batch,)
                    params["coalescence_time"] -= slice_starts / sample_rate_hertz
                
                params["shift"] = shifts
                injections_list.append(waveforms)
                parameters_list.append(params)

            try:
                injections = ops.stack(injections_list, axis=0)
                masks = ops.stack(masks_list, axis=0)
            except:
                injections = None
                masks = None
            
            batched_params = batch_injection_parameters(parameters_list)
            
            yield injections, masks, batched_params

    def add_injections_to_onsource(
        self,
        injections,
        mask,
        onsource,
        offsource,
        parameters_to_return,
        onsource_duration_seconds: float = None,
        injection_parameters: Optional[Dict] = None,
    ):
        scaled_injections_list = []
        scaling_params_list = []
        
        final_onsource = onsource
        
        # Determine if we need to return injection times
        return_injection_times = False
        if parameters_to_return and ReturnVariables.CENTRAL_TIME in parameters_to_return:
            return_injection_times = True

        for i, generator in enumerate(self.waveform_generators):
            raw_waveforms = injections[i] 
            
            scaling_method = generator.scaling_method
            network = generator.network
            
            num_examples = ops.shape(raw_waveforms)[0]
            
            # Extract parameters for this batch/generator if available
            # Note: injection_parameters usually has batched params for all generators?
            # Or is it a list of dicts? 
            # In generate(), batched_params is a single dict combining all.
            # But params might be arrays of shape (Batch*NumGens)?
            # No, generate() combines them.
            # However, if we have multiple generators, the parameters for each might be mixed or stacked?
            # Wait, `generate` returns `batched_params` which merges lists of params.
            # If keys conflict (e.g. mass_1 for gen1 and gen2), `batch_injection_parameters` handles it?
            # `batch_injection_parameters` iterates keys and stacks values if they are lists.
            # If `InjectionGenerator` has multiple `WNBGenerator`s, `min_frequency` might be a list of tensors?
            # Or if keys are unique? No, they share keys.
            # If `batched_params` contains stacked tensors, we need to index `i`.
            # BUT `shifts` (key "shift") was stacked in `generate` (line 1309: injections_list.append, parameters_list.append).
            # `batch_injection_parameters` (line 1320) combines them.
            # Let's verify `batch_injection_parameters` logic. 
            # It seems it stacks if key exists in multiple params.
            # So `injection_parameters["shift"]` should be (NumGenerators, Batch) or (Batch, NumGenerators)?
            # `ops.stack` usually stacks on axis 0.
            # If parameters_list is [p1, p2], stack([v1, v2]) -> (2, Batch).
            # So we can access `injection_parameters[key][i]`.
            
            current_shifts = None
            if injection_parameters and "shift" in injection_parameters:
                shifts_all = injection_parameters["shift"]
                # shifts_all should be (NumGenerators, Batch)
                if len(ops.shape(shifts_all)) > 1 and ops.shape(shifts_all)[0] == len(self.waveform_generators):
                    current_shifts = shifts_all[i]
                else:
                    # Fallback if only 1 generator or not stacked as expected
                    current_shifts = shifts_all
            
            # Generate or Retrieve Direction Vectors
            input_ra = None
            input_dec = None
            input_pol = None
            
            # Check if RA/Dec/Pol are in injection_parameters
            # Keys: WaveformParameters.ASCENDING_NODE_LONGITUDE ?? No, RA/Dec are usually implicit or specific params.
            # If they are not in params, we generate them.
            
            # Use seed logic consistent with project_wave or generate
            # project_wave uses self.rng.integers to get a seed for JAX
            # We can do the same here to ensure deterministic behavior relative to the generator state
            proj_seed = network.rng.integers(1000000000, size=2)
            
            # We generate RA/Dec/Pol explicitly so we can calculate time delays
            # Using the same logic as project_wave ensures consistency if we pass them in.
            
            # project_wave expects tensor inputs or None.
            # If we pass None, it generates them inside. But then we don't know them.
            # So we MUST generate them here and pass them.
            
            # Use JAX/Ops generation matching `generate_direction_vectors`
            # But we are in python methods here. 
            # We can use the helper function from detector.py if we import it, or just call network.project_wave with our own values.
            # To generate them, we need to call `generate_direction_vectors` or similar?
            # Actually, `detector.py` exposes `generate_direction_vectors`.
            # But it uses JAX keys.
            # Let's check if parameters have them.
            
            # If we are calculating injection times, we force generation here.
            
            # RA
            if injection_parameters and "right_ascension" in injection_parameters:
                 # Ensure shape
                 input_ra = injection_parameters["right_ascension"]
                 # Index if needed
                 if len(ops.shape(input_ra)) > 1 and ops.shape(input_ra)[0] == len(self.waveform_generators):
                     input_ra = input_ra[i]
            
            # Dec
            if injection_parameters and "declination" in injection_parameters:
                 input_dec = injection_parameters["declination"]
                 if len(ops.shape(input_dec)) > 1 and ops.shape(input_dec)[0] == len(self.waveform_generators):
                     input_dec = input_dec[i]

            # Pol
            if injection_parameters and "polarization" in injection_parameters:
                 input_pol = injection_parameters["polarization"]
                 if len(ops.shape(input_pol)) > 1 and ops.shape(input_pol)[0] == len(self.waveform_generators):
                     input_pol = input_pol[i]

            # If missing, generate utilizing the seed we just drew.
            # Note: project_wave takes `seed` argument (tensor).
            # We can use that seed to generate them using JAX functions if we want to be 100% consistent with implicit generation.
            # `gf.generate_direction_vectors` is available? It's in detector.py.
            # It's not imported in injection.py. 
            # But we can import it or use `network` methods?
            # `network` doesn't have generation methods exposed easily.
            # Let's import it or re-implement simply since we want to pass explicit values.
            
            # Actually, simplest way: let's stick to randomness driven by `proj_seed`.
            # We need to use JAX random to match `project_wave`'s internal logic?
            # Or just generate using `network.rng` (numpy) and convert to tensor.
            # `project_wave` uses JAX random.
            # If we use numpy `network.rng`, it will be deterministic given the seed.
            # And since we pass explicit values to `project_wave`, it won't re-generate.
            # So let's use numpy generation.
            
            num_inj = ops.shape(raw_waveforms)[0]
            
            if input_ra is None or input_dec is None:
                # Uniform sphere
                # u, v ~ U(0, 1)
                # ra = 2pi * u
                # dec = arcsin(2v - 1)
                u = network.rng.random(num_inj).astype("float32")
                v = network.rng.random(num_inj).astype("float32")
                
                if input_ra is None:
                    input_ra = ops.convert_to_tensor(2.0 * np.pi * u)
                if input_dec is None:
                    input_dec = ops.arcsin(2.0 * v - 1.0)
            
            if input_pol is None:
                p = network.rng.random(num_inj).astype("float32")
                input_pol = ops.convert_to_tensor(2.0 * np.pi * p)
            
            # Calculate Time Delays
            # network.get_time_delay returns (Batch, NumDetectors)
            time_delays = network.get_time_delay(input_ra, input_dec)
            
            # Calculate Injection Time
            if return_injection_times and current_shifts is not None:
                # t0_raw = target_duration / 2 + shift
                # Final time = t0_raw + delay
                # target_duration is `onsource_duration_seconds`? 
                # Wait, onsource passed to this method is raw onsource data. 
                # `onsource_duration_seconds` arg is available.
                
                dur = onsource_duration_seconds
                if dur is None:
                    # Fallback to calculating from samples
                     dur = ops.shape(onsource)[-1] / self.sample_rate_hertz
                
                # Convert everything to seconds
                # shift is in samples? Yes, `shifts = self.rng.integers(...)` in generate
                
                shift_seconds = ops.cast(current_shifts, "float32") / self.sample_rate_hertz
                center_seconds = dur / 2.0
                
                t_geocenter = center_seconds + shift_seconds
                
                # Broadcast t_geocenter (Batch,) to (Batch, NumDetectors)
                t_geocenter = ops.expand_dims(t_geocenter, -1)
                
                final_time = t_geocenter + time_delays
                
                # Normalize by duration (0 to 1)
                normalized_time = final_time / dur
                
                scaling_params_list.append({ReturnVariables.CENTRAL_TIME: normalized_time})


            scaling_dist = scaling_method.value

            if not hasattr(self, "_scaling_indices"):
                self._scaling_indices = [0] * len(self.waveform_generators)

            if isinstance(scaling_dist, (np.ndarray, list)):
                current_idx = self._scaling_indices[i]
                end_idx = current_idx + num_examples
                
                if end_idx > len(scaling_dist):
                     # Wrap around
                     current_idx = 0
                     end_idx = num_examples
                     self._scaling_indices[i] = 0
                
                target_val = scaling_dist[current_idx : end_idx]
                self._scaling_indices[i] = end_idx
            else:
                target_val = scaling_dist.sample(num_examples)
            target_val = ops.convert_to_tensor(target_val, dtype="float32")
            
            sample_rate = self.sample_rate_hertz
            
            # Use offsource for scaling if available, otherwise fallback to onsource
            scaling_background = offsource if offsource is not None else onsource
            
            # Determine target length from onsource
            target_num_samples = ops.shape(onsource)[-1]
            
            if scaling_method.type_.value.ordinality == ScalingOrdinality.BEFORE_PROJECTION:
                scaled_raw = scaling_method.scale(
                    raw_waveforms,
                    scaling_background, 
                    target_val,
                    sample_rate,
                    onsource_duration_seconds=None # Do not crop
                )
                
                projected = network.project_wave(
                    scaled_raw,
                    sample_rate_hertz=sample_rate,
                    right_ascension=input_ra,
                    declination=input_dec,
                    polarization=input_pol
                )
                
                # Crop to target length
                start = (ops.shape(projected)[-1] - target_num_samples) // 2
                end = start + target_num_samples
                projected = projected[..., start:end]
                
            else:
                projected = network.project_wave(
                    raw_waveforms,
                    sample_rate_hertz=sample_rate,
                    right_ascension=input_ra,
                    declination=input_dec,
                    polarization=input_pol
                )
                
                # Crop to target length
                start = (ops.shape(projected)[-1] - target_num_samples) // 2
                end = start + target_num_samples
                projected = projected[..., start:end]
                
                scaled_raw = scaling_method.scale(
                    projected,
                    scaling_background,
                    target_val,
                    sample_rate,
                    onsource_duration_seconds=onsource_duration_seconds
                )
                projected = scaled_raw
            
            final_onsource = final_onsource + projected
            
            scaled_injections_list.append(projected)
            
            key = scaling_method.type_ 
            scaling_params_list.append({key: target_val})

        try:
            scaled_injections = ops.stack(scaled_injections_list, axis=0)
        except:
            scaled_injections = None
            
        combined_scaling_params = {}
        for d in scaling_params_list:
            for k, v in d.items():
                if k not in combined_scaling_params:
                    combined_scaling_params[k] = []
                combined_scaling_params[k].append(v)
        
        for k, v in combined_scaling_params.items():
            combined_scaling_params[k] = ops.stack(v, axis=0)

        return final_onsource, scaled_injections, combined_scaling_params