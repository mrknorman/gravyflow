from pathlib import Path
from typing import Union, List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import json
import os
import logging

import tensorflow as tf
import numpy as np
from astropy import coordinates, units
from numpy.random import default_rng  

import gravyflow as gf

# Define the speed of light constant (in m/s)
C = 299792458.0

@tf.function(jit_compile=True)
def get_time_delay_(
    right_ascension: tf.Tensor, 
    declination: tf.Tensor,
    location : tf.Tensor
) -> tf.Tensor:
    """
    Calculate the time delay for various combinations of right ascension,
    declination, and detector locations.

    Parameters
    ----------
    right_ascension : tf.Tensor, shape (N,)
        The right ascension (in rad) of the signal.
    declination : tf.Tensor, shape (N,)
        The declination (in rad) of the signal.
    location : tf.Tensor, shape (X, 3)
        Array of detector location coordinates.

    Returns
    -------
    tf.Tensor, shape (X, N)
        The arrival time difference for each combination of detector 
        location and sky signal.
    """

    cos_declination = tf.math.cos(declination)
    sin_declination = tf.math.sin(declination)
    cos_ra_angle = tf.math.cos(right_ascension)
    sin_ra_angle = tf.math.sin(right_ascension)

    e0 = cos_declination * cos_ra_angle
    e1 = cos_declination * -sin_ra_angle
    e2 = sin_declination

    ehat = tf.stack([e0, e1, e2], axis=0)  # Shape (3, N)
    ehat = tf.expand_dims(ehat, 1)  # Shape (3, 1, N) to allow broadcasting

    # Compute the dot product using tensordot
    time_delay = tf.tensordot(location, ehat, axes=[[1], [0]]) 
    time_delay = time_delay / C  # Normalize by speed of light
    
    time_delay = tf.transpose(tf.squeeze(time_delay, axis = 1))
    
    return tf.cast(time_delay, dtype=tf.float32)
    
@tf.function(jit_compile=True)
def get_antenna_pattern_(
    right_ascension: tf.Tensor, 
    declination: tf.Tensor, 
    polarization: tf.Tensor,
    x_vector: tf.Tensor,
    y_vector: tf.Tensor,
    x_length_meters: tf.Tensor,
    y_length_meters: tf.Tensor,
    x_response : tf.Tensor,
    y_response : tf.Tensor,
    response : tf.Tensor
) -> (tf.Tensor, tf.Tensor):
    
    right_ascension = tf.expand_dims(right_ascension, 1)
    declination = tf.expand_dims(declination, 1)
    polarization = tf.expand_dims(polarization, 1)
    
    x_vector = tf.expand_dims(x_vector, 0)   
    y_vector = tf.expand_dims(y_vector, 0)   
    x_length_meters = tf.expand_dims(x_length_meters, 0)  
    y_length_meters = tf.expand_dims(y_length_meters, 0)  
    x_response = tf.expand_dims(x_response, 0)  
    y_response = tf.expand_dims(y_response, 0)  
    response = tf.expand_dims(response, 0)

    cos_ra = tf.math.cos(right_ascension)
    sin_ra = tf.math.sin(right_ascension)
    cos_dec = tf.math.cos(declination)
    sin_dec = tf.math.sin(declination)
    cos_psi = tf.math.cos(polarization)
    sin_psi = tf.math.sin(polarization)
    
    x = tf.stack([
        -cos_psi * sin_ra - sin_psi * cos_ra * sin_dec,
        -cos_psi * cos_ra + sin_psi * sin_ra * sin_dec,
        sin_psi * cos_dec
    ], axis=-1)

    y = tf.stack([
        sin_psi * sin_ra - cos_psi * cos_ra * sin_dec,
        sin_psi * cos_ra + cos_psi * sin_ra * sin_dec,
        cos_psi * cos_dec
    ], axis=-1)
    
    # Calculate dx and dy via tensordot, and immediately remove singleton 
    # dimensions and transpose them
    tensor_product_dx = tf.tensordot(response, x, axes=[[2], [2]])
    # Remove the first and last singleton dimensions
    tensor_product_dx_squeezed = tf.squeeze(tensor_product_dx, axis=[0, -1])
    # Apply transpose to get the shape (100000, 1, 3)
    dx = tf.transpose(tensor_product_dx_squeezed, perm=[2, 0, 1])

    tensor_product_dy = tf.tensordot(response, y, axes=[[2], [2]])
     # Remove the first and last singleton dimensions
    tensor_product_dy_squeezed = tf.squeeze(tensor_product_dy, axis=[0, -1])
    # Apply transpose to get the shape (100000, 1, 3)
    dy = tf.transpose(tensor_product_dy_squeezed, perm=[2, 0, 1])

    # Expand dimensions for x, y, dx, dy along axis 0
    x = tf.expand_dims(x, axis=0)
    y = tf.expand_dims(y, axis=0)

    dx = tf.expand_dims(dx, axis=0)
    dy = tf.expand_dims(dy, axis=0)
    
    # Function to compute final response
    def compute_response(
            dx: tf.Tensor, 
            dy: tf.Tensor, 
            a: tf.Tensor, 
            b: tf.Tensor
        ) -> tf.Tensor:
        
        return tf.squeeze(tf.reduce_sum(a * dx + b * dy, axis=-1), axis=0)
    
    antenna_pattern = tf.stack(
        [
            compute_response(dx, -dy, x, y), 
            compute_response(dy, dx, x, y)
        ], 
        axis=-1
    )

    return antenna_pattern

@tf.function(jit_compile=True)
def generate_direction_vectors(
        num_injections: int, 
        seed_tensor: Tuple[int, int],
        right_ascension: Optional[tf.Tensor] = None,
        declination: Optional[tf.Tensor] = None
    ) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Generate random direction vectors uniformly distributed over the sphere.

    Parameters
    ----------
    num_injections : int
        The number of direction vectors to generate.
    seed : Tuple[int, int]
        A tuple used for random number generation to ensure reproducibility.
    right_ascension : Optional[tf.Tensor]
        Pre-defined right ascension values or None.
    declination : Optional[tf.Tensor]
        Pre-defined declination values or None.

    Returns
    -------
    Tuple[tf.Tensor, tf.Tensor]
        The right ascension and declination tensors.

    """
    PI = tf.constant(3.14159, dtype=tf.float32)

    # Right Ascension
    if right_ascension is None:
        right_ascension = tf.random.stateless_uniform(
            seed=seed_tensor,
            shape=[num_injections], 
            minval=0.0, 
            maxval=2.0 * PI, 
            dtype=tf.float32
        )

    # Declination
    if declination is None:
        sin_declination = tf.random.stateless_uniform(
            seed=seed_tensor+1,
            shape=[num_injections], 
            minval=-1.0, 
            maxval=1.0, 
            dtype=tf.float32
        )
        declination = tf.asin(sin_declination)

    return right_ascension, declination

@tf.function(jit_compile=True)
def project_wave_(
    seed,
    strain : tf.Tensor,
    sample_rate_hertz : float,
    x_vector: tf.Tensor,
    y_vector: tf.Tensor,
    x_length_meters: tf.Tensor,
    y_length_meters: tf.Tensor,
    x_response : tf.Tensor,
    y_response : tf.Tensor,
    response : tf.Tensor,
    location : tf.Tensor,
    right_ascension: Optional[tf.Tensor] = None,
    declination: Optional[tf.Tensor] = None,
    polarization: Optional[tf.Tensor] = None
):

    # Ensure the seed is of the correct shape [2] and dtype int32
    seed_tensor = tf.cast(seed, dtype=tf.int32)
    
    num_injections = tf.shape(strain)[0]
    PI = tf.constant(3.14159, dtype=tf.float32)
    
    right_ascension, declination = generate_direction_vectors(
        num_injections, 
        seed_tensor,
        right_ascension,
        declination
    )

    if polarization is None:
        polarization = tf.random.stateless_uniform(
            seed=seed_tensor+2,
            shape=[num_injections], 
            minval=0.0, 
            maxval=2 * PI, 
            dtype=tf.float32
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
    
    antenna_patern = tf.expand_dims(antenna_patern, axis=-1)
    
    # Deal with non-incoherent case:
    if (len(tf.shape(strain)) == 3):
        strain = tf.expand_dims(strain, axis=1)
    injection = tf.reduce_sum(strain*antenna_patern, axis = -2)

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
        match parameters:
            
            case dict():
                                
                if "num_detectors" in parameters:
                    num_detectors = parameters.pop("num_detectors")
                    
                    if isinstance(num_detectors, gf.Distribution):
                        num_detectors = num_detectors.sample(1)
                    
                else:
                    num_detectors = None
                    
                for key, value in parameters.items():
                                        
                    match value:
                        
                        case float() | int():
                            arguments[key] = tf.convert_to_tensor(
                                [value], dtype=tf.float32
                            )
                            
                        case list() | np.ndarray():
                            arguments[key] = tf.convert_to_tensor(
                                value, dtype=tf.float32
                            )
                            
                        case tf.Tensor():
                            arguments[key] = tf.case(
                                value, dtype=tf.float32
                            )
                            
                        case gf.Distribution():
                            if num_detectors is None:
                                raise ValueError("Num detectors not specified")
                            else:
                                arguments[key] =  tf.convert_to_tensor(
                                    value.sample(num_detectors), 
                                    dtype=tf.float32
                                )
                    
            case list():
                
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
                    if len(attribute_list) != len(parameters):
                        raise ValueError(
                            "When initializing a network from a list, all "
                            "elements must be IFO Enums.")

                    tensor = tf.convert_to_tensor(
                        attribute_list, 
                        dtype=tf.float32
                    )
                    arguments[attribute] = tensor

            case _:
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
        longitude_radians: tf.Tensor = None,  # Batched tensor
        latitude_radians: tf.Tensor = None,   # Batched tensor
        y_angle_radians: tf.Tensor = None,  # Batched tensor
        x_angle_radians: tf.Tensor = None,  # Batched tensor or None
        height_meters: tf.Tensor = None,  # Batched tensor
        x_length_meters: tf.Tensor = None,  # Batched tensor
        y_length_meters: tf.Tensor = None   # Batched tensor
    ):
        
        PI = tf.constant(np.pi, dtype=tf.float32)
    
        if x_angle_radians is None:
            x_angle_radians = \
                y_angle_radians + tf.constant(PI / 2.0, dtype=tf.float32)

        # Rotation matrices using the provided functions
        rm1 = rotation_matrix_z(longitude_radians)
        rm2 = rotation_matrix_y(PI / 2.0 - latitude_radians)    
        rm = tf.matmul(rm2, rm1)

        # Calculate response in earth centered coordinates
        responses = []
        vecs = []

        for angle in [y_angle_radians, x_angle_radians]:
            a, b = tf.cos(2 * angle), tf.sin(2 * angle)
            
            response = tf.stack([
                tf.stack([-a, b, tf.zeros_like(a)], axis=-1), 
                tf.stack([b, a, tf.zeros_like(a)], axis=-1), 
                tf.stack(
                    [
                        tf.zeros_like(a), 
                        tf.zeros_like(a), 
                        tf.zeros_like(a)
                    ], 
                    axis=-1
                )
            ], axis=1)

            response = tf.matmul(response, rm)
            response = tf.matmul(
                tf.transpose(rm, perm=[0, 2, 1]), 
                response
            ) / 4.0
            
            responses.append(response)
            
            angle_vector = tf.stack([
                -tf.cos(angle),
                tf.sin(angle),
                tf.zeros_like(angle)
            ], axis=1)

            angle_vector = tf.reshape(angle_vector, [-1, 3, 1])
            
            vec = tf.matmul(tf.transpose(rm, perm=[0, 2, 1]), angle_vector)
            vec = tf.squeeze(vec, axis=-1)
            vecs.append(vec)

        full_response = responses[0] - responses[1]

        # Handling the coordinates.EarthLocation method
        locations = []
        for long, lat, h in zip(
                longitude_radians, latitude_radians, height_meters
            ):
            
            loc = coordinates.EarthLocation.from_geodetic(
                long * units.rad, lat * units.rad, h*units.meter
            )
            locations.append([loc.x.value, loc.y.value, loc.z.value])
        loc = tf.constant(locations, dtype=tf.float32)

        self.location = loc
        self.response = full_response
        self.x_response = responses[1]
        self.y_response = responses[0]
        self.x_vector = vecs[1]
        self.y_vector = vecs[0]
        self.y_angle_radians = y_angle_radians
        self.x_angle_radians = x_angle_radians
        self.height_meters = height_meters
        self.x_altitude_meters = tf.zeros_like(height_meters)
        self.y_altitude_meters = tf.zeros_like(height_meters)
        self.y_length_meters = y_length_meters
        self.x_length_meters = x_length_meters
        
        self.calculate_max_arrival_time_difference()
    
    def get_antenna_pattern(
            self,
            right_ascension: tf.Tensor, 
            declination: tf.Tensor, 
            polarization: tf.Tensor
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
        ):   
        
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
        right_ascension: tf.Tensor, 
        declination: tf.Tensor
    ) -> tf.Tensor:
        
        return get_time_delay_(
            right_ascension, 
            declination,
            self.location
        )

    def check_and_convert(
            self, 
            input : Union[tf.Tensor, Tuple, List, float, int, None], 
            name : Union[tf.Tensor, Tuple, List, float, int, None], 
            tensor_length : int
        ):

        match input:
            case tf.Tensor() as tensor:
                if tensor.shape[0] != N:
                    raise ValueError(f"Tensor, {name}, must be equal to num injections, {tensor_length}.")
                if tensor.dtype != tf.float32:
                    try:
                        return tf.cast(tensor, tf.float32)
                    except Exception as e:
                        raise ValueError(
                            f"Failed to convert tensor, {name}, to float32"
                        ) from e

            case (list() | tuple()) as lst:
                if len(lst) != N:
                    raise ValueError(f"List/tuple, {name}, must be equal to num injections, {tensor_length}.")
                if any(isinstance(x, int) for x in lst):
                    logging.warn(
                        f"List or tuple, {name}, contains integers, which will be converted to floats."
                    )
                try:
                    return tf.constant(lst, dtype=tf.float32)
                except Exception as e:
                    raise ValueError(
                        f"Failed to convert list/tuple, {name}, to float32 tensor"
                    ) from e

            case (float() | int()):
                if isinstance(input, int):
                    logging.warn(
                        f"Input, {name}, is an integer and will be converted to float."
                    )
                return tf.fill([tensor_length], float(input))

            case None:
                return None

            case _:
                raise TypeError(
                    f"Input, {name}, must be a float, list, tuple, or tensor."
                )

        return input
    
    def project_wave(
        self,
        strain : tf.Tensor,
        sample_rate_hertz : float = None,
        right_ascension: Optional[Union[tf.Tensor, List[float], float]] = None,
        declination: Optional[Union[tf.Tensor, List[float], float]]  = None,
        polarization: Optional[Union[tf.Tensor, List[float], float]]  = None
    ):
        if sample_rate_hertz is None:
            sample_rate_hertz = gf.Defaults.sample_rate_hertz

        strain_length : int = tf.shape(strain)[0]
        right_ascension = self.check_and_convert(
            right_ascension, "right_ascension", tensor_length=strain_length
        )
        declination = self.check_and_convert(
            declination, "declination", tensor_length=strain_length
        )
        polarization = self.check_and_convert(
            polarization, "polarization", tensor_length=strain_length
        )
        
        return project_wave_(
            self.rng.integers(1E10, size=2),
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
            right_ascension=right_ascension,
            declination=declination,
            polarization=polarization
        )
    
    def calculate_max_arrival_time_difference(self):
        """
        Compute pairwise distances between each points.

        Args:
        - points: A tensor of shape [N, 3] representing N 3D points.

        Returns:
        - A tensor of shape [N, N] where entry (i, j) is the Euclidean distance
          between points[i] and points[j].
        """
        # Expand dimensions to compute pairwise distances
        p1 = tf.expand_dims(self.location, 1)  # Shape: [N, 1, 3]
        p2 = tf.expand_dims(self.location, 0)  # Shape: [1, N, 3]

        # Compute pairwise differences
        diff = p1 - p2  # Shape: [N, N, 3]

        # Compute pairwise Euclidean distances
        self.distances = tf.norm(diff, axis=2)  # Shape: [N, N]
        
        max_distance = tf.reduce_max(self.distances)
        
        self.max_arrival_time_difference_seconds = max_distance/C
        
@tf.function(jit_compile=True)
def shift_waveform(
        strain : tf.Tensor, 
        sample_rate_hertz : float, 
        time_shift_seconds : tf.Tensor
    ):
    
    frequency_axis = gf.rfftfreq(
        tf.shape(strain)[-1],
        1.0/sample_rate_hertz
    )
    
    frequency_axis = tf.expand_dims(
        tf.expand_dims(frequency_axis, axis=0), 
        axis=0
    )
    time_shift_seconds = tf.expand_dims(time_shift_seconds, axis=-1)
    
    PI = tf.constant(3.14159, dtype=tf.float32)

    strain_fft = tf.signal.rfft(strain) 

    imaj_part = -2.0*PI*frequency_axis*time_shift_seconds
    phase_factor = tf.exp(
        tf.complex(
            tf.zeros_like(imaj_part),
            imaj_part
        )
    )
    shitfted_strain = tf.signal.irfft(phase_factor * strain_fft)
    
    return tf.math.real(shitfted_strain)

@tf.function(jit_compile=True)
def rotation_matrix_x(angle: tf.Tensor) -> tf.Tensor:
    """
    Generate a 3D rotation matrix around the X-axis using 
    TensorFlow operations.
    
    Parameters:
    - angle: Rotation angle in radians, shape [...].
    
    Returns:
    - 3x3 rotation matrix, shape [..., 3, 3].
    """
    c = tf.cos(angle)
    s = tf.sin(angle)

    ones = tf.ones_like(c)
    zeros = tf.zeros_like(c)

    row1 = tf.stack([ones, zeros, zeros], axis=-1)
    row2 = tf.stack([zeros, c, -s], axis=-1)
    row3 = tf.stack([zeros, s, c], axis=-1)
    
    return tf.stack([row1, row2, row3], axis=-2)

@tf.function(jit_compile=True)
def rotation_matrix_y(angle: tf.Tensor) -> tf.Tensor:
    """
    Generate a 3D rotation matrix around the Y-axis using 
    TensorFlow operations.
    
    Parameters:
    - angle: Rotation angle in radians, shape [...].
    
    Returns:
    - 3x3 rotation matrix, shape [..., 3, 3].
    """
    c = tf.cos(angle)
    s = tf.sin(angle)

    ones = tf.ones_like(c)
    zeros = tf.zeros_like(c)

    row1 = tf.stack([c, zeros, -s], axis=-1)
    row2 = tf.stack([zeros, ones, zeros], axis=-1)
    row3 = tf.stack([s, zeros, c], axis=-1)

    return tf.stack([row1, row2, row3], axis=-2)


@tf.function(jit_compile=True)
def rotation_matrix_z(angle: tf.Tensor) -> tf.Tensor:
    """
    Generate a 3D rotation matrix around the Z-axis using 
    TensorFlow operations.
    
    Parameters:
    - angle: Rotation angle in radians, shape [...].
    
    Returns:
    - 3x3 rotation matrix, shape [..., 3, 3].
    """
    c = tf.cos(angle)
    s = tf.sin(angle)

    ones = tf.ones_like(c)
    zeros = tf.zeros_like(c)

    row1 = tf.stack([c, s, zeros], axis=-1)
    row2 = tf.stack([-s, c, zeros], axis=-1)
    row3 = tf.stack([zeros, zeros, ones], axis=-1)

    return tf.stack([row1, row2, row3], axis=-2)

@tf.function(jit_compile=True)
def single_arm_frequency_response(
        frequency: tf.Tensor, 
        n: tf.Tensor, 
        arm_length_meters: tf.Tensor
    ) -> tf.Tensor:
    
    """
    Compute the relative amplitude factor of the arm response due to signal 
    delay.
    """
    
    # Cast inputs to complex128 for compatibility
    frequency = tf.cast(frequency, dtype=tf.complex64)
    arm_length_meters = tf.cast(arm_length_meters, dtype=tf.complex64)
        
    # Calculate the complex phase term
    phase = 2.0j * tf.constant(np.pi, dtype=tf.complex64) * \
        (arm_length_meters / C) * frequency
    
    # Manually clip the real and imaginary parts
    n_real = tf.math.real(n)
    n_imag = tf.math.imag(n)
    lower_bound = tf.constant(-0.999, dtype=tf.float32)
    upper_bound = tf.constant(0.999, dtype=tf.float32)
    n_real = tf.maximum(lower_bound, tf.minimum(n_real, upper_bound))
    n_imag = tf.maximum(lower_bound, tf.minimum(n_imag, upper_bound))
    
    # Reassemble into a complex tensor
    n = tf.complex(n_real, n_imag)
            
    # Compute components a, b, and c
    a = 1.0 / (4.0 * phase)
    b = (1 - tf.exp(-phase * (1 - n))) / (1 - n)
    c = tf.exp(-2.0 * phase) * (1 - tf.exp(phase * (1 + n))) / (1 + n)
    
    # Compute and return the single arm frequency response
    return tf.math.real(a * (b - c) * 2.0)
