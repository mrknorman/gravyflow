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

from .maths import Distribution, DistributionType
from .setup import replace_placeholders

@dataclass
class IFO_:
    name: str
    optimal_psd_path : Path
    longitude_radians : float
    latitude_radians : float
    response : np.ndarray
    location : np.ndarray

noise_profile_directory_path : Path = Path("./py_ml_tools/res/noise_profiles/")

ifo_data : Dict = {
    "livingston" : {
        "name": "Livingston",
        "optimal_psd_path" : noise_profile_directory_path / "livingston.csv",
        "longitude_radians" : -1.5843093707829257, 
        "latitude_radians" : 0.5334231350225018,
        "response" : np.array(
            [
                [ 0.41128087,  0.14021027,  0.24729459],
                [ 0.14021027, -0.10900569, -0.18161564],
                [ 0.24729459 ,-0.18161564, -0.30227515]
            ]
        ),
        "location" : np.array(
            [-74276.0447238, -5496283.71971, 3224257.01744]
        )
    },
    "hanford" : {
        "name": "Hanford",
        "optimal_psd_path" : noise_profile_directory_path / "handford.csv",
        "longitude_radians" : -2.08405676916594, 
        "latitude_radians" : 0.810795263791696,
        "response" : np.array(
            [
                [-0.3926141,  -0.07761341, -0.24738905],
                [-0.07761341,  0.31952408,  0.22799784],
                [-0.24738905,  0.22799784,  0.07309003]
            ]
        ),
        "location" : np.array(
            [-2161414.92636, -3834695.17889,  4600350.22664]
        )     
    },
    "virgo" : {
        "name": "Virgo",
        "optimal_psd_path" : noise_profile_directory_path / "virgo.csv",
        "longitude_radians" : 0.1833380521285067, 
        "latitude_radians" : 0.7615118398044829,
        "response" : np.array(
            [
                [ 0.24387404, -0.09908378, -0.23257622],
                [-0.09908378, -0.44782585,  0.1878331 ],
                [-0.23257622,  0.1878331,   0.2039518 ]
            ]
        ),
        "location" : np.array(
            [4546374.099, 842989.697626, 4378576.96241]
        )    
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
        
        match parameters:
            
            case dict():
                                
                if "num_detectors" in parameters:
                    num_detectors = parameters.pop("num_detectors")
                    
                    if isinstance(num_detectors, Distribution):
                        num_detectors = num_detectors.sample(1)
                    
                else:
                    num_detectors = None
                    
                arguments = {}
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
                                
                self.init_parameters(
                    **arguments
                )
                
            case list():
                
                latitude_radians = []
                longitude_radians = []
                response = []
                location = []
                for ifo in parameters:
                    
                    if not isinstance(ifo, IFO):
                        raise ValueError(
                            "When initilising a network from a list, all "
                            "elements must be IFO Enums."
                        )
                        
                    ifo = ifo.value
                    
                    latitude_radians.append(
                        ifo.latitude_radians
                    )
                    longitude_radians.append(
                        ifo.longitude_radians
                    )
                    response.append(
                        ifo.response
                    )
                    location.append(
                        ifo.response
                    )
                
                self.latitude_radians = tf.convert_to_tensor(
                    latitude_radians, dtype=tf.float32
                )
                self.longitude_radians = tf.convert_to_tensor(
                    longitude_radians, dtype=tf.float32
                )
                self.response = tf.convert_to_tensor(
                    response, dtype=tf.float32
                )
                self.location = tf.convert_to_tensor(
                    location, dtype=tf.float32
                )
                
            case _:
                raise ValueError(
                    f"Unsuported type {type(parameters)} for Network "
                    "initilisation."
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
        self.x_altitude = tf.zeros_like(height_meters)
        self.y_altitude = tf.zeros_like(height_meters)
        self.y_length_meters = y_length_meters
        self.x_length_meters = x_length_meters
    
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

def add_detectors_on_earth(
        longitude_radians: tf.Tensor,  # Batched tensor
        latitude_radians: tf.Tensor,   # Batched tensor
        y_angle_radians: tf.Tensor = None,  # Batched tensor
        x_angle_radians: tf.Tensor = None,  # Batched tensor or None
        height_meters: tf.Tensor = None,  # Batched tensor
        x_length_meters: tf.Tensor = None,  # Batched tensor
        y_length_meters: tf.Tensor = None   # Batched tensor
    ) -> dict:

    """Add a new detector on the earth using TensorFlow operations."""
        
    PI = tf.constant(np.pi, dtype=tf.float32)
    
    if x_angle_radians is None:
        x_angle_radians = y_angle_radians + tf.constant(PI / 2.0, dtype=tf.float32)
    
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
                [tf.zeros_like(a), tf.zeros_like(a), tf.zeros_like(a)], 
                axis=-1
            )
        ], axis=1)

        resp = tf.matmul(resp, rm)
        resp = tf.matmul(tf.transpose(rm, perm=[0, 2, 1]), resp) / 4.0
        resps.append(resp)
                
        vec = tf.matmul(
            tf.transpose(rm, perm=[0, 2, 1]), 
            tf.stack([
                -tf.cos(angle),
                tf.sin(angle),
                tf.zeros_like(angle)
            ], axis=0)
        )
        vec = tf.squeeze(vec, axis=-1)
        vecs.append(vec)

    full_resp = resps[0] - resps[1]
    
    # Handling the coordinates.EarthLocation method
    locations = []
    for long, lat, h in zip(longitude_radians, latitude_radians, height_meters):
        loc = coordinates.EarthLocation.from_geodetic(
            long * units.rad, lat * units.rad, h*units.meter
        )
        locations.append([loc.x.value, loc.y.value, loc.z.value])
    loc = tf.constant(locations, dtype=tf.float32)

    return {
        'location': loc,
        'response': full_resp,
        'xresp': resps[1],
        'yresp': resps[0],
        'xvec': vecs[1],
        'yvec': vecs[0],
        'y_angle_radians': y_angle_radians,
        'x_angle_radians': x_angle_radians,
        'height_meters': height_meters,
        'xaltitude': tf.zeros_like(height_meters),
        'yaltitude': tf.zeros_like(height_meters),
        'y_length_meters': y_length_meters,
        'x_length_meters': x_length_meters
    }