from pathlib import Path
from typing import Union, List, Dict
from dataclasses import dataclass
from enum import Enum
import json

import tensorflow as tf
import numpy as np
from astropy import coordinates, units
from astropy.coordinates.matrix_utilities import rotation_matrix
from astropy.units import meter

from .maths import Distribution, DistributionType, rfftfreq
from .setup import replace_placeholders


# Define the speed of light constant (in m/s)
C = 299792458.0

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

NOISE_PROFILE_DIRECTORY_PATH : Path = Path("./py_ml_tools/res/noise_profiles/")

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
        parameters : Union[List[IFO], Dict]
        ):
        
        arguments = {}
        match parameters:
            
            case dict():
                                
                if "num_detectors" in parameters:
                    num_detectors = parameters.pop("num_detectors")
                    
                    if isinstance(num_detectors, Distribution):
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
                            
                        case Distribution():
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
                
                for attribute in attributes:
                    attribute_list = [
                        getattr(ifo.value, attribute) for ifo in parameters \
                        if isinstance(ifo, IFO)]
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
        resps = []
        vecs = []

        for angle in [y_angle_radians, x_angle_radians]:
            a, b = tf.cos(2 * angle), tf.sin(2 * angle)

            batch_size = tf.shape(a)[0]
            resp = tf.stack([
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

            resp = tf.matmul(resp, rm)
            resp = tf.matmul(tf.transpose(rm, perm=[0, 2, 1]), resp) / 4.0
            
            resps.append(resp)
            
            angle_vector = tf.stack([
                -tf.cos(angle),
                tf.sin(angle),
                tf.zeros_like(angle)
            ], axis=1)

            angle_vector = tf.reshape(angle_vector, [-1, 3, 1])
            
            vec = tf.matmul(tf.transpose(rm, perm=[0, 2, 1]), angle_vector)
            vec = tf.squeeze(vec, axis=-1)
            vecs.append(vec)

        full_resp = resps[0] - resps[1]

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
        self.response = full_resp
        self.xresp = resps[1]
        self.yresp = resps[0]
        self.xvec = vecs[1]
        self.yvec = vecs[0]
        self.y_angle_radians = y_angle_radians
        self.x_angle_radians = x_angle_radians
        self.height_meters = height_meters
        self.x_altitude_meters = tf.zeros_like(height_meters)
        self.y_altitude_meters = tf.zeros_like(height_meters)
        self.y_length_meters = y_length_meters
        self.x_length_meters = x_length_meters
    
    @tf.function
    def get_antenna_pattern_(
        self,
        right_ascension: tf.Tensor, 
        declination: tf.Tensor, 
        polarization: tf.Tensor,
        xvec: tf.Tensor,
        yvec: tf.Tensor,
        x_length_meters: tf.Tensor,
        y_length_meters: tf.Tensor,
        xresp : tf.Tensor,
        yresp : tf.Tensor,
        resp : tf.Tensor
    ) -> (tf.Tensor, tf.Tensor):
        
        right_ascension = tf.expand_dims(right_ascension, 1)
        declination = tf.expand_dims(declination, 1)
        polarization = tf.expand_dims(polarization, 1)
        
        xvec = tf.expand_dims(xvec, 0)   
        yvec = tf.expand_dims(yvec, 0)   
        x_length_meters = tf.expand_dims(x_length_meters, 0)  
        y_length_meters = tf.expand_dims(y_length_meters, 0)  
        xresp = tf.expand_dims(xresp, 0)  
        yresp = tf.expand_dims(yresp, 0)  
        resp = tf.expand_dims(resp, 0)

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
        
        # Calculate dx and dy via tensordot, and immediately squeeze and 
        # transpose them
        dx = tf.transpose(
            tf.squeeze(tf.tensordot(resp, x, axes=[[2], [2]])), perm=[2, 0, 1])
        dy = tf.transpose(
            tf.squeeze(tf.tensordot(resp, y, axes=[[2], [2]])), perm=[2, 0, 1])

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
            
            return tf.squeeze(tf.reduce_sum(a * dx + b * dy, axis=-1))

        return (
            compute_response(dx, -dy, x, y), 
            compute_response(dy, dx, x, y)
        )
    
    def get_antenna_pattern(
            self,
            right_ascension: tf.Tensor, 
            declination: tf.Tensor, 
            polarization: tf.Tensor
        ):
        
        return self.get_antenna_pattern_(
            right_ascension, 
            declination, 
            polarization,
            self.xvec,
            self.yvec,
            self.x_length_meters,
            self.y_length_meters,
            self.xresp,
            self.yresp,
            self.response
        )
    
    @classmethod
    def load(
        cls,
        config_path: Path, 
        ):   
        
        # Define replacement mapping
        replacements = {
            "constant": DistributionType.CONSTANT,
            "uniform": DistributionType.UNIFORM,
            "normal": DistributionType.NORMAL,
            "2pi" : np.pi*2.0,
            "pi/2" : np.pi/2.0,
            "-pi/2" : -np.pi/2.0
        }

        # Load injection config
        with config_path.open("r") as file:
            config = json.load(file)
        
        # Replace placeholders
        for value in config.values():
            replace_placeholders(value, replacements)
        
        num_detectors = config.pop("num_detectors")        
        arguments = {k: Distribution(**v) for k, v in config.items()}
        arguments["num_detectors"] = num_detectors
        
        return Network(arguments)
    
    @tf.function
    def get_time_delay_(
        self,
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
        
        time_delay = tf.transpose(tf.squeeze(time_delay))
        
        return tf.cast(time_delay, dtype=tf.float32)
    
    def get_time_delay(
        self,
        right_ascension: tf.Tensor, 
        declination: tf.Tensor
    ) -> tf.Tensor:
        
        return self.get_time_delay_(
            right_ascension, 
            declination,
            self.location
        )
    
    def project_wave(
        self,
        hc : tf.Tensor,
        hp : tf.Tensor,
        sample_frequency_hertz : float,
        right_ascension: tf.Tensor = None,
        declination: tf.Tensor = None,
        polarization: tf.Tensor = None
    ):
        
        num_injections = tf.shape(hc)[0]
        
        if right_ascension == None:
            right_ascension = tf.constant(
                np.random.uniform(
                    0, 
                    2 * np.pi, 
                    size=(num_injections,)
                ), 
                dtype=tf.float32
            )
        if declination == None:
            declination = tf.constant(
                np.random.uniform(
                    -np.pi / 2, 
                    np.pi / 2, 
                    size=(num_injections,)
                ), 
                dtype=tf.float32
            )
        if polarization == None:
            polarization = tf.constant(
                np.random.uniform(
                    0, 
                    2 * np.pi, 
                    size=(num_injections,)
                ), 
                dtype=tf.float32
                )
        
        fc, fp = self.get_antenna_pattern(
            right_ascension, 
            declination, 
            polarization
        ) 
        
        hc = tf.expand_dims(hc, axis=1)
        hp = tf.expand_dims(hp, axis=1)
        fc = tf.expand_dims(fc, axis=-1)
        fp = tf.expand_dims(fp, axis=-1)
                
        strain = hc*fc + hp*fp
        
        time_shift_seconds = self.get_time_delay(
            right_ascension, 
            declination
        ) 
                
        return shift_waveform(
            strain, 
            sample_frequency_hertz, 
            time_shift_seconds
        )
        
#@tf.function
def shift_waveform(
        strain : tf.Tensor, 
        sample_frequency_hertz : float, 
        time_shift_seconds : tf.Tensor
    ):

    frequency_axis = rfftfreq(
        tf.shape(strain)[-1],
        1.0/sample_frequency_hertz
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

@tf.function
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

@tf.function
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


@tf.function
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
