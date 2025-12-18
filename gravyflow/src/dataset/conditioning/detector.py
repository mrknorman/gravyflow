from pathlib import Path
from typing import Union, List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import json
import os
import logging

import keras
from keras import ops
import jax.numpy as jnp
import jax
import numpy as np
from astropy import coordinates, units
from numpy.random import default_rng  

import gravyflow as gf

# Define the speed of light constant (in m/s)
C = 299792458.0

@jax.jit
def get_time_delay_(
        right_ascension, 
        declination,
        location
    ):

    """
    Calculate the time delay for various combinations of right ascension,
    declination, and detector locations.
    """
    right_ascension = ops.convert_to_tensor(right_ascension)
    declination = ops.convert_to_tensor(declination)
    location = ops.convert_to_tensor(location)

    cos_declination = ops.cos(declination)
    sin_declination = ops.sin(declination)
    cos_ra_angle = ops.cos(right_ascension)
    sin_ra_angle = ops.sin(right_ascension)

    e0 = cos_declination * cos_ra_angle
    e1 = cos_declination * -sin_ra_angle
    e2 = sin_declination

    ehat = ops.stack([e0, e1, e2], axis=0)  # Shape (3, N)
    ehat = ops.expand_dims(ehat, 1)  # Shape (3, 1, N) to allow broadcasting

    # Compute the dot product using tensordot
    # location: (X, 3)
    # ehat: (3, 1, N)
    # axes=[[1], [0]] -> sum over axis 1 of location (3) and axis 0 of ehat (3)
    # Result: (X, 1, N)
    time_delay = ops.tensordot(location, ehat, axes=[[1], [0]]) 
    time_delay = time_delay / C  # Normalize by speed of light
    
    # squeeze axis 1 -> (X, N)
    time_delay = ops.squeeze(time_delay, axis=1)
    # transpose -> (N, X)
    time_delay = ops.transpose(time_delay)
    
    return ops.cast(time_delay, dtype="float32")
    
@jax.jit
def get_antenna_pattern_(
        right_ascension, 
        declination, 
        polarization,
        x_vector,
        y_vector,
        x_length_meters,
        y_length_meters,
        x_response,
        y_response,
        response
    ):
    
    right_ascension = ops.expand_dims(right_ascension, 1)
    declination = ops.expand_dims(declination, 1)
    polarization = ops.expand_dims(polarization, 1)
    
    x_vector = ops.expand_dims(x_vector, 0)   
    y_vector = ops.expand_dims(y_vector, 0)   
    x_length_meters = ops.expand_dims(x_length_meters, 0)  
    y_length_meters = ops.expand_dims(y_length_meters, 0)  
    x_response = ops.expand_dims(x_response, 0)  
    y_response = ops.expand_dims(y_response, 0)  
    response = ops.expand_dims(response, 0)

    cos_ra = ops.cos(right_ascension)
    sin_ra = ops.sin(right_ascension)
    cos_dec = ops.cos(declination)
    sin_dec = ops.sin(declination)
    cos_psi = ops.cos(polarization)
    sin_psi = ops.sin(polarization)
    
    x = ops.stack([
        -cos_psi * sin_ra - sin_psi * cos_ra * sin_dec,
        -cos_psi * cos_ra + sin_psi * sin_ra * sin_dec,
        sin_psi * cos_dec
    ], axis=-1)

    y = ops.stack([
        sin_psi * sin_ra - cos_psi * cos_ra * sin_dec,
        sin_psi * cos_ra + cos_psi * sin_ra * sin_dec,
        cos_psi * cos_dec
    ], axis=-1)
    
    # Calculate dx and dy via tensordot
    # response is (3, 3) but expanded to (1, 3, 3)
    # x is (N, 1, 3)
    
    # tf.tensordot(response, x, axes=[[2], [2]])
    # response: (1, 3, 3)
    # x: (N, 1, 3)
    # Sum over last axis of response and last axis of x.
    # Result: (1, 3, N, 1)
    
    tensor_product_dx = ops.tensordot(response, x, axes=[[2], [2]])
    
    # Remove the first and last singleton dimensions
    # Result: (D, 3, N)
    tensor_product_dx_squeezed = ops.squeeze(tensor_product_dx, axis=[0, 4]) 
    
    # Apply transpose to get the shape (N, D, 3)
    dx = ops.transpose(tensor_product_dx_squeezed, (2, 0, 1))
    
    tensor_product_dy = ops.tensordot(response, y, axes=[[2], [2]])
    tensor_product_dy_squeezed = ops.squeeze(tensor_product_dy, axis=[0, 4])
    dy = ops.transpose(tensor_product_dy_squeezed, (2, 0, 1))

    # Expand dimensions for x, y, dx, dy along axis 0
    x = ops.expand_dims(x, axis=0) # (1, N, 1, 3)
    y = ops.expand_dims(y, axis=0)

    dx = ops.expand_dims(dx, axis=0) # (1, N, D, 3)
    dy = ops.expand_dims(dy, axis=0)
    
    # Function to compute final response
    def compute_response(
            dx, 
            dy, 
            a, 
            b
        ):
        
        return ops.squeeze(ops.sum(a * dx + b * dy, axis=-1), axis=0)
    
    antenna_pattern = ops.stack(
        [
            compute_response(dx, -dy, x, y), 
            compute_response(dy, dx, x, y)
        ], 
        axis=-1
    )

    return antenna_pattern

@jax.jit(static_argnames=["num_injections"])
def generate_direction_vectors(
        num_injections: int, 
        seed_tensor: Tuple[int, int],
        right_ascension = None,
        declination = None
    ):
    """
    Generate random direction vectors uniformly distributed over the sphere.
    """
    PI = 3.14159

    # Right Ascension
    if right_ascension is None:
        # JAX random
        key = jax.random.PRNGKey(seed_tensor[0]) # Use first seed
        right_ascension = jax.random.uniform(
            key,
            shape=[num_injections], 
            minval=0.0, 
            maxval=2.0 * PI, 
            dtype="float32"
        )

    # Declination
    if declination is None:
        key = jax.random.PRNGKey(seed_tensor[1]) # Use second seed (or seed+1)
        sin_declination = jax.random.uniform(
            key,
            shape=[num_injections], 
            minval=-1.0, 
            maxval=1.0, 
            dtype="float32"
        )
        declination = ops.arcsin(sin_declination)

    return right_ascension, declination

@jax.jit(static_argnames=["sample_rate_hertz"])
def shift_waveform(
        strain, 
        sample_rate_hertz : float, 
        time_shift_seconds
    ):
    
    frequency_axis = gf.rfftfreq(
        strain.shape[-1],
        1.0/sample_rate_hertz
    )
    
    frequency_axis = ops.expand_dims(
        ops.expand_dims(frequency_axis, axis=0), 
        axis=0
    )
    time_shift_seconds = ops.expand_dims(time_shift_seconds, axis=-1)
    
    PI = 3.14159

    # Use jnp.fft.rfft directly
    strain_fft = jnp.fft.rfft(strain) 

    imaj_part = -2.0*PI*frequency_axis*time_shift_seconds
    
    # tf.complex(real, imag) -> jax.lax.complex(real, imag) or real + 1j * imag
    phase_factor = ops.exp(
        1j * imaj_part
    )
    
    # jnp.fft.irfft
    shitfted_strain = jnp.fft.irfft(phase_factor * strain_fft, n=strain.shape[-1])
    
    # Masking to prevent wrapping
    # shift_waveform expects time_shift_seconds to be (Batch, D, 1) here
    # time_shift_seconds was expanded in line 216.
    
    shift_samples = time_shift_seconds * sample_rate_hertz # (Batch, D, 1)
    
    num_samples = strain.shape[-1]
    indices = ops.arange(num_samples)
    # Expand indices to (1, 1, Time)
    indices = ops.expand_dims(ops.expand_dims(indices, 0), 0)
    
    # Mask logic: Keep if (indices >= shift) AND (indices < N + shift)
    # Use ceil/floor to be conservative and avoid fractional boundary artifacts
    lower_bound = ops.ceil(shift_samples)
    upper_bound = ops.floor(num_samples + shift_samples)
    
    # Create a soft mask (taper) to avoid harsh cutoffs
    # We want a transition from 0 to 1 at lower_bound and 1 to 0 at upper_bound.
    # Taper width in samples
    taper_width = 16.0 
    
    # Sigmoid-like or linear taper? Sine-squared is standard.
    # We can define a continuous mask function based on indices.
    
    # distance from lower bound (positive = inside)
    dist_lower = indices - lower_bound
    # distance from upper bound (positive = inside)
    dist_upper = upper_bound - indices
    
    # We want min(dist_lower, dist_upper) to determine the taper
    # If dist < 0: 0
    # If dist > width: 1
    # If 0 < dist < width: taper
    
    min_dist = ops.minimum(dist_lower, dist_upper)
    
    # Clamp to [0, width]
    clamped_dist = ops.clip(min_dist, 0.0, taper_width)
    
    # Normalize to [0, 1]
    x = clamped_dist / taper_width
    
    # Sine-squared taper: sin^2(pi/2 * x)
    taper = ops.sin(3.14159 / 2.0 * x) ** 2
    
    mask = taper
    mask = ops.cast(mask, "float32")
    
    return ops.real(shitfted_strain) * mask

@jax.jit(static_argnames=["sample_rate_hertz"])
def project_wave(
        seed,
        strain,
        sample_rate_hertz : float,
        x_vector,
        y_vector,
        x_length_meters,
        y_length_meters,
        x_response,
        y_response,
        response,
        location,
        right_ascension = None,
        declination = None,
        polarization = None
    ):

    # Ensure the seed is of the correct shape [2] and dtype int32
    seed_tensor = ops.cast(seed, dtype="int32")
    
    num_injections = strain.shape[0]
    PI = 3.14159
    
    # We need to handle seed carefully. 
    # generate_direction_vectors expects a tuple or something usable as PRNGKey.
    # seed_tensor is a tensor.
    
    random_right_ascension, random_declination = generate_direction_vectors(
        num_injections, 
        seed_tensor, # Pass tensor, handle inside?
        right_ascension,
        declination
    )

    if right_ascension is None:
        right_ascension = random_right_ascension
    
    if declination is None:
        declination = random_declination

    if polarization is None:
        key = jax.random.PRNGKey(seed_tensor[0] + 2) # Offset seed
        polarization = jax.random.uniform(
            key,
            shape=[num_injections], 
            minval=0.0, 
            maxval=2 * PI, 
            dtype="float32"
        )
    
    antenna_patern = get_antenna_pattern_(
        right_ascension, 
        declination, 
        polarization,
        x_vector,
        y_vector,
        x_length_meters,
        y_length_meters,
        x_response,
        y_response,
        response
    ) 
    
    antenna_patern = ops.expand_dims(antenna_patern, axis=-1)
    
    # Deal with non-incoherent case:
    if (len(ops.shape(strain)) == 3):
        strain = ops.expand_dims(strain, axis=1)
    
    injection = ops.sum(strain*antenna_patern, axis = -2)

    time_shift_seconds = get_time_delay_(
        right_ascension, 
        declination,
        location
    )
    
    shifted_waveoform = shift_waveform(
        injection, 
        sample_rate_hertz, 
        time_shift_seconds
    )

    return shifted_waveoform

def rotation_matrix_x(angle):
    c = ops.cos(angle)
    s = ops.sin(angle)

    ones = ops.ones_like(c)
    zeros = ops.zeros_like(c)

    row1 = ops.stack([ones, zeros, zeros], axis=-1)
    row2 = ops.stack([zeros, c, -s], axis=-1)
    row3 = ops.stack([zeros, s, c], axis=-1)
    
    return ops.stack([row1, row2, row3], axis=-2)

def rotation_matrix_y(angle):
    c = ops.cos(angle)
    s = ops.sin(angle)

    ones = ops.ones_like(c)
    zeros = ops.zeros_like(c)

    row1 = ops.stack([c, zeros, -s], axis=-1)
    row2 = ops.stack([zeros, ones, zeros], axis=-1)
    row3 = ops.stack([s, zeros, c], axis=-1)

    return ops.stack([row1, row2, row3], axis=-2)

def rotation_matrix_z(angle):
    c = ops.cos(angle)
    s = ops.sin(angle)

    ones = ops.ones_like(c)
    zeros = ops.zeros_like(c)

    row1 = ops.stack([c, s, zeros], axis=-1)
    row2 = ops.stack([-s, c, zeros], axis=-1)
    row3 = ops.stack([zeros, zeros, ones], axis=-1)

    return ops.stack([row1, row2, row3], axis=-2)

@dataclass
class IFO_:
    """Data class to represent information about an Interferometer."""

    name: str
    optimal_psd_path : Path
    longitude_radians : float
    latitude_radians : float
    y_angle_radians : float
    x_angle_radians : float
    height_meters : float
    x_length_meters : float
    y_length_meters : float

NOISE_PROFILE_DIRECTORY_PATH : Path = gf.PATH / "res/noise_profiles/"
ifo_data : Dict = {
    "livingston" : {
        "name": "Livingston",
        "optimal_psd_path" : NOISE_PROFILE_DIRECTORY_PATH / "livingston.csv",
        "longitude_radians" : -1.5843093707829257, 
        "latitude_radians" : 0.5334231350225018,
        "x_angle_radians" : 4.403177738189697,
        "y_angle_radians" : 2.8323814868927,
        "x_length_meters" : 4000.0,
        "y_length_meters" : 4000.0,
        "height_meters" : -6.573999881744385
    },
    "hanford" : {
        "name": "Hanford",
        "optimal_psd_path" : NOISE_PROFILE_DIRECTORY_PATH / "hanford.csv",
        "longitude_radians" : -2.08405676916594, 
        "latitude_radians" : 0.810795263791696,
        "x_angle_radians": 5.654877185821533,
        "y_angle_radians": 4.084080696105957,
        "x_length_meters": 4000.0,
        "y_length_meters": 4000.0,
        "height_meters": 142.5540008544922
    },
    "virgo" : {
        "name": "Virgo",
        "optimal_psd_path" : NOISE_PROFILE_DIRECTORY_PATH / "virgo.csv",
        "longitude_radians" : 0.1833380521285067, 
        "latitude_radians" : 0.7615118398044829,
        "x_angle_radians": 0.3391628563404083,
        "y_angle_radians": 5.051551818847656,
        "x_length_meters": 3000.0,
        "y_length_meters": 3000.0,
        "height_meters": 51.88399887084961
    }
}
    
class IFO(Enum):
    L1 = IFO_(**ifo_data["livingston"])
    H1 = IFO_(**ifo_data["hanford"])
    V1 = IFO_(**ifo_data["virgo"])

class Network:
    def __init__ (
            self,
            parameters : Union[List[IFO], Dict],
            seed : Optional[int] = None
        ):

        if seed is None:
            seed = gf.Defaults.seed

        self.rng = default_rng(seed)
        
        arguments = {}
        # ... (Parameter parsing logic mostly unchanged, just ops.convert_to_tensor)
        
        if isinstance(parameters, dict):
            if "num_detectors" in parameters:
                num_detectors = parameters.pop("num_detectors")
                if isinstance(num_detectors, gf.Distribution):  # pragma: no cover
                    num_detectors = num_detectors.sample(1)
            else:
                num_detectors = None
                
            for key, value in parameters.items():
                if isinstance(value, (float, int)):
                    arguments[key] = ops.convert_to_tensor([value], dtype="float32")
                elif isinstance(value, (list, np.ndarray)):  # pragma: no cover
                    arguments[key] = ops.convert_to_tensor(value, dtype="float32")
                elif ops.is_tensor(value):  # pragma: no cover
                     arguments[key] = ops.cast(value, "float32")
                elif isinstance(value, gf.Distribution):  # pragma: no cover
                    if num_detectors is None:
                        raise ValueError("Num detectors not specified")
                    else:
                        arguments[key] = ops.convert_to_tensor(
                            value.sample(num_detectors), 
                            dtype="float32"
                        )
                        
        elif isinstance(parameters, list):
            attributes = [
                "latitude_radians", 
                "longitude_radians", 
                "x_angle_radians", 
                "y_angle_radians", 
                "x_length_meters", 
                "y_length_meters", 
                "height_meters"
            ]
            
            num_detectors = len(parameters)
            
            for attribute in attributes:
                attribute_list = [
                    getattr(ifo.value, attribute) for ifo in parameters if isinstance(ifo, IFO)]
                if len(attribute_list) != len(parameters):  # pragma: no cover
                    raise ValueError(
                        "When initializing a network from a list, all "
                        "elements must be IFO Enums.")

                tensor = ops.convert_to_tensor(
                    attribute_list, 
                    dtype="float32"
                )
                arguments[attribute] = tensor
                
        else:  # pragma: no cover
             raise ValueError(
                f"Unsuported type {type(parameters)} for Network "
                "initilisation."
            )
                
        self.num_detectors = num_detectors
                
        self.init_parameters(
            **arguments
        )
    
    def init_parameters(
        self,
        longitude_radians = None,  # Batched tensor
        latitude_radians = None,   # Batched tensor
        y_angle_radians = None,  # Batched tensor
        x_angle_radians = None,  # Batched tensor or None
        height_meters = None,  # Batched tensor
        x_length_meters = None,  # Batched tensor
        y_length_meters = None   # Batched tensor
    ):
        
        PI = np.pi
    
        if x_angle_radians is None:
            x_angle_radians = \
                y_angle_radians + ops.convert_to_tensor(PI / 2.0, dtype="float32")

        # Rotation matrices using the provided functions
        rm1 = rotation_matrix_z(longitude_radians)
        rm2 = rotation_matrix_y(PI / 2.0 - latitude_radians)    
        rm = ops.matmul(rm2, rm1)

        # Calculate response in earth centered coordinates
        responses = []
        vecs = []

        for angle in [y_angle_radians, x_angle_radians]:
            a, b = ops.cos(2 * angle), ops.sin(2 * angle)
            
            response = ops.stack([
                ops.stack([-a, b, ops.zeros_like(a)], axis=-1), 
                ops.stack([b, a, ops.zeros_like(a)], axis=-1), 
                ops.stack(
                    [
                        ops.zeros_like(a), 
                        ops.zeros_like(a), 
                        ops.zeros_like(a)
                    ], 
                    axis=-1
                )
            ], axis=1)

            response = ops.matmul(response, rm)
            response = ops.matmul(
                ops.transpose(rm, axes=[0, 2, 1]), 
                response
            ) / 4.0
            
            responses.append(response)
            
            angle_vector = ops.stack([
                -ops.cos(angle),
                ops.sin(angle),
                ops.zeros_like(angle)
            ], axis=1)

            angle_vector = ops.reshape(angle_vector, [-1, 3, 1])
            
            vec = ops.matmul(ops.transpose(rm, axes=[0, 2, 1]), angle_vector)
            vec = ops.squeeze(vec, axis=-1)
            vecs.append(vec)

        full_response = responses[0] - responses[1]

        # Handling the coordinates.EarthLocation method
        # This part requires numpy/cpu execution as astropy is not differentiable/JAX-native
        # We can use jax.pure_callback if needed, or just run it eagerly since init is usually once.
        
        # Convert tensors to numpy for astropy
        long_np = np.array(longitude_radians)
        lat_np = np.array(latitude_radians)
        h_np = np.array(height_meters)
        
        locations = []
        for long, lat, h in zip(long_np, lat_np, h_np):
            loc = coordinates.EarthLocation.from_geodetic(
                long * units.rad, lat * units.rad, h*units.meter
            )
            locations.append([loc.x.value, loc.y.value, loc.z.value])
        loc = ops.convert_to_tensor(locations, dtype="float32")

        self.location = loc
        self.response = full_response
        self.x_response = responses[1]
        self.y_response = responses[0]
        self.x_vector = vecs[1]
        self.y_vector = vecs[0]
        self.y_angle_radians = y_angle_radians
        self.x_angle_radians = x_angle_radians
        self.height_meters = height_meters
        self.x_altitude_meters = ops.zeros_like(height_meters)
        self.y_altitude_meters = ops.zeros_like(height_meters)
        self.y_length_meters = y_length_meters
        self.x_length_meters = x_length_meters
        
        self.calculate_max_arrival_time_difference()
    
    def get_antenna_pattern(
            self,
            right_ascension, 
            declination, 
            polarization
        ):
        
        return get_antenna_pattern_(
            right_ascension, 
            declination, 
            polarization,
            self.x_vector,
            self.y_vector,
            self.x_length_meters,
            self.y_length_meters,
            self.x_response,
            self.y_response,
            self.response
        )
    
    @classmethod
    def load(
        cls,
        config_path: Path, 
        ) -> "Network":   
        
        # Define replacement mapping
        replacements = {
            "constant": gf.DistributionType.CONSTANT,
            "uniform": gf.DistributionType.UNIFORM,
            "normal": gf.DistributionType.NORMAL,
            "2pi" : np.pi*2.0,
            "pi/2" : np.pi/2.0,
            "-pi/2" : -np.pi/2.0
        }

        # Load injection config
        with config_path.open("r") as file:
            config = json.load(file)
        
        # Replace placeholders
        gf.replace_placeholders(config, replacements)

        arguments = {}
        if "num_detectors" in config:
            arguments.update(
                {"num_detectors" : config.pop("num_detectors")}
            ) 
        
        for key, value in config.items():
            if isinstance(value, dict):
                value = gf.Distribution(**value)
            arguments.update({key: value})
        
        return Network(arguments)
    
    def get_time_delay(
        self,
        right_ascension, 
        declination
    ):
        
        return get_time_delay_(
            right_ascension, 
            declination,
            self.location
        )

    def check_and_convert(
            self, 
            input, 
            name, 
            tensor_length : int
        ):

        if ops.is_tensor(input):
            if ops.shape(input)[0] != tensor_length:
                 raise ValueError(
                    f"Tensor, {name}, must be equal to num injections, {tensor_length}."
                )
            if input.dtype != "float32":
                return ops.cast(input, "float32")
            return input
            
        elif isinstance(input, (list, tuple)):
            if len(input) != tensor_length:
                raise ValueError(
                    f"List/tuple, {name}, must be equal to num injections, {tensor_length}."
                )
            return ops.convert_to_tensor(input, dtype="float32")
            
        elif isinstance(input, (float, int)):
            # ops.full not always available? ops.ones * val
            return ops.ones((tensor_length,), dtype="float32") * float(input)
            
        elif input is None:
            return None
            
        else:  # pragma: no cover
            raise TypeError(
                f"Input, {name}, must be a float, list, tuple, or tensor."
            )

        return input
    
    def project_wave(
        self,
        strain,
        sample_rate_hertz = None,
        right_ascension = None,
        declination  = None,
        polarization  = None
    ):

        if sample_rate_hertz is None:
            sample_rate_hertz = gf.Defaults.sample_rate_hertz

        strain_length = ops.shape(strain)[0]
        right_ascension = self.check_and_convert(
            right_ascension, "right_ascension", tensor_length=strain_length
        )
        declination = self.check_and_convert(
            declination, "declination", tensor_length=strain_length
        )
        polarization = self.check_and_convert(
            polarization, "polarization", tensor_length=strain_length
        )
        
        # rng.integers(1E10, size=2)
        seed = self.rng.integers(1000000000, size=2)
        
        args = (
            seed,
            strain,
            sample_rate_hertz,
            self.x_vector,
            self.y_vector,
            self.x_length_meters,
            self.y_length_meters,
            self.x_response,
            self.y_response,
            self.response,
            self.location,
        )
        kwargs = dict(
            right_ascension=right_ascension,
            declination=declination,
            polarization=polarization
        )
        
        try:
            return project_wave(*args, **kwargs)
        except jax.errors.JaxRuntimeError as e:  # pragma: no cover
            # GPU graph capture may fail on some hardware configurations
            # Fall back to non-JIT execution
            if "Failed to capture gpu graph" in str(e):
                logging.warning(
                    "GPU graph capture failed, falling back to non-JIT execution. "
                    "This may be slower but ensures compatibility."
                )
                with jax.disable_jit():
                    return project_wave(*args, **kwargs)
            raise
    
    def calculate_max_arrival_time_difference(self):
        """
        Compute pairwise distances between each points.
        """
        # Expand dimensions to compute pairwise distances
        p1 = ops.expand_dims(self.location, 1)  # Shape: [N, 1, 3]
        p2 = ops.expand_dims(self.location, 0)  # Shape: [1, N, 3]

        # Compute pairwise differences
        diff = p1 - p2  # Shape: [N, N, 3]

        # Compute pairwise Euclidean distances
        # ops.norm(diff, axis=2)
        # Keras 3 ops.norm might not exist or be different.
        # ops.sqrt(ops.sum(ops.square(diff), axis=2))
        self.distances = ops.sqrt(ops.sum(ops.square(diff), axis=2))
        
        max_distance = ops.max(self.distances)
        
        self.max_arrival_time_difference_seconds = max_distance/C
