import tensorflow as tf
from astropy import coordinates, units
from astropy.units import meter

import numpy as np

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

    row1 = tf.stack([c, zeros, s], axis=-1)
    row2 = tf.stack([zeros, ones, zeros], axis=-1)
    row3 = tf.stack([-s, zeros, c], axis=-1)

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

    row1 = tf.stack([c, -s, zeros], axis=-1)
    row2 = tf.stack([s, c, zeros], axis=-1)
    row3 = tf.stack([zeros, zeros, ones], axis=-1)

    return tf.stack([row1, row2, row3], axis=-2)

def add_detectors_on_earth(
        longitude: tf.Tensor,  # Batched tensor
        latitude: tf.Tensor,   # Batched tensor
        yangle: tf.Tensor = None,  # Batched tensor
        xangle: tf.Tensor = None,  # Batched tensor or None
        height: tf.Tensor = None,  # Batched tensor
        xlength: tf.Tensor = None,  # Batched tensor
        ylength: tf.Tensor = None   # Batched tensor
    ) -> dict:

    """Add a new detector on the earth using TensorFlow operations."""
        
    PI = tf.constant(np.pi, dtype=tf.float32)
    
    if xangle is None:
        xangle = yangle + tf.constant(PI / 2.0, dtype=tf.float32)
        
    print(longitude)
    print(longitude * units.rad)

    # Rotation matrices using the provided functions
    rm1 = rotation_matrix_z(longitude * units.rad)
    rm2 = rotation_matrix_y(PI / 2.0 - latitude)
    rm = tf.matmul(rm2, rm1)

    # Calculate response in earth centered coordinates
    resps = []
    vecs = []

    for angle in [yangle, xangle]:
        a, b = tf.cos(2 * angle), tf.sin(2 * angle)
        
        batch_size = tf.shape(a)[0]
        resp = tf.stack([
            tf.stack([-a, b, tf.zeros_like(a)], axis=-1), 
            tf.stack([b, a, tf.zeros_like(a)], axis=-1), 
            tf.stack([tf.zeros_like(a), tf.zeros_like(a), tf.zeros_like(a)], axis=-1)
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
    for long, lat, h in zip(longitude, latitude, height):
        loc = coordinates.EarthLocation.from_geodetic(long * units.rad, lat * units.rad, h*units.meter)
        locations.append([loc.x.value, loc.y.value, loc.z.value])
    loc = tf.constant(locations, dtype=tf.float32)

    return {
        'location': loc,
        'response': full_resp,
        'xresp': resps[1],
        'yresp': resps[0],
        'xvec': vecs[1],
        'yvec': vecs[0],
        'yangle': yangle,
        'xangle': xangle,
        'height': height,
        'xaltitude': tf.zeros_like(height),
        'yaltitude': tf.zeros_like(height),
        'ylength': ylength,
        'xlength': xlength
    }