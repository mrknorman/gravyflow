from pathlib import Path
from typing import Union, List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum, IntEnum
import json
import os
import logging

import keras
from keras import ops
import jax.numpy as jnp
import jax
import numpy as np
from astropy import coordinates, units
from gravyflow.src.resources import RESOURCES
from gravyflow.src.utils.tensor import rfftfreq

import gravyflow as gf

# Define the speed of light constant (in m/s)
C = 299792458.0

# The GPS time at which sky positions are defined when none is given. Right
# ascension is fixed to the sky; the Earth turns under it, so mapping a source
# to the detectors needs a time. Any time gives an isotropic population; this
# one is a round number that lal reproduces directly.
REFERENCE_GPS_TIME_SECONDS = 1_000_000_000.0

# GPS seconds at which each GPS - UTC offset (leap seconds since 1980) begins.
_LEAP_SECOND_GPS_TIMES = np.array([
    46828800, 78364801, 109900802, 173059203, 252028804, 315187205,
    346723206, 393984007, 425520008, 457056009, 504489610, 551750411,
    599184012, 820108813, 914803214, 1025136015, 1119744016, 1167264017,
], dtype=np.float64)

# GPS - UTC at J2000.0 (2000-01-01 12:00:00 UTC) is 13 s, so J2000.0 is GPS
# 630763213 and 630763200 UTC seconds after the GPS epoch.
_J2000_UTC_SECONDS_SINCE_GPS_EPOCH = 630763200.0


def gps_leap_seconds(gps_time_seconds):
    """GPS - UTC, in seconds, at ``gps_time_seconds``."""
    gps = np.asarray(gps_time_seconds, dtype=np.float64)
    return np.searchsorted(_LEAP_SECOND_GPS_TIMES, gps, side="right").astype(np.float64)


def greenwich_mean_sidereal_time(gps_time_seconds):
    """
    Greenwich mean sidereal time, in radians in [0, 2 pi), at a GPS time.

    The IAU 1982 expression on UTC, as lal's ``GreenwichMeanSiderealTime``
    evaluates it (UT1 is taken as UTC). Computed in float64 with numpy.
    """
    gps = np.asarray(gps_time_seconds, dtype=np.float64)
    utc_seconds = gps - gps_leap_seconds(gps) - _J2000_UTC_SECONDS_SINCE_GPS_EPOCH
    t = utc_seconds / (86400.0 * 36525.0)  # Julian centuries since J2000.0
    sidereal_seconds = (
        67310.54841
        + (3155760000.0 + 8640184.812866) * t
        + 0.093104 * t ** 2
        - 6.2e-6 * t ** 3
    )
    return np.mod(sidereal_seconds * np.pi / 43200.0, 2.0 * np.pi)


def _greenwich_hour_angle(right_ascension, gmst_radians):
    right_ascension = ops.convert_to_tensor(right_ascension, dtype="float32")
    if gmst_radians is None:
        return -right_ascension
    return ops.convert_to_tensor(gmst_radians, dtype="float32") - right_ascension


@jax.jit
def get_time_delay_(
        right_ascension, 
        declination,
        location,
        gmst_radians = None
    ):

    """
    The arrival time at each detector minus the arrival time at the Earth's
    centre, in seconds, as lal's ``TimeDelayFromEarthCenter``.

    The source is at ``right_ascension`` and ``declination`` (radians) when the
    Greenwich mean sidereal time is ``gmst_radians``. A detector displaced
    towards the source hears it first, so its delay is negative.

    Returns an array of shape (num_sources, num_detectors).
    """
    hour_angle = _greenwich_hour_angle(right_ascension, gmst_radians)
    declination = ops.convert_to_tensor(declination)
    location = ops.convert_to_tensor(location)

    cos_declination = ops.cos(declination)
    sin_declination = ops.sin(declination)

    # The unit vector towards the source, in Earth-fixed coordinates.
    e0 = cos_declination * ops.cos(hour_angle)
    e1 = cos_declination * -ops.sin(hour_angle)
    e2 = sin_declination

    ehat = ops.stack([e0, e1, e2], axis=0)  # Shape (3, N)

    # location: (X, 3); result: (X, N)
    time_delay = -ops.tensordot(location, ehat, axes=[[1], [0]]) / C

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
        response,
        gmst_radians = None
    ):
    """
    Antenna patterns (F+, Fx) of each detector, as lal's ``ComputeDetAMResponse``.

    The source is at ``right_ascension`` and ``declination`` with polarisation
    angle ``polarization`` (radians, the LAL/Anderson et al. convention) when the
    Greenwich mean sidereal time is ``gmst_radians``.

    Returns an array of shape (num_sources, num_detectors, 2).
    """
    # The expressions below take the Greenwich hour angle.
    right_ascension = _greenwich_hour_angle(right_ascension, gmst_radians)
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

@jax.jit(static_argnames=["sample_rate_hertz"])
def shift_waveform(
        strain, 
        sample_rate_hertz : float, 
        time_shift_seconds
    ):
    
    frequency_axis = rfftfreq(
        strain.shape[-1],
        1.0/sample_rate_hertz
    )
    
    frequency_axis = ops.expand_dims(
        ops.expand_dims(frequency_axis, axis=0), 
        axis=0
    )
    time_shift_seconds = ops.expand_dims(time_shift_seconds, axis=-1)

    strain_fft = jnp.fft.rfft(strain) 
    phase_factor = ops.exp(-2j * np.pi * frequency_axis * time_shift_seconds)
    shifted_strain = jnp.fft.irfft(phase_factor * strain_fft, n=strain.shape[-1])

    # The shift is circular: zero only the samples that wrapped round from the
    # other end. A delay of s samples (s > 0) wraps into indices below ceil(s);
    # an advance (s < 0) wraps into indices at or above floor(N + s).
    shift_samples = time_shift_seconds * sample_rate_hertz  # (Batch, D, 1)
    num_samples = strain.shape[-1]
    indices = ops.expand_dims(ops.expand_dims(ops.arange(num_samples), 0), 0)
    keep = (indices >= ops.ceil(shift_samples)) & (
        indices < ops.floor(num_samples + shift_samples)
    )

    return ops.real(shifted_strain) * ops.cast(keep, "float32")

@jax.jit(static_argnames=["sample_rate_hertz"])
def project_wave(
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
        right_ascension,
        declination,
        polarization,
        gmst_radians = None
    ):
    """
    Project polarisations onto the detectors, delayed to each detector's arrival.

    Sky angles follow LAL: right ascension and declination are equatorial, and
    ``gmst_radians`` (the Greenwich mean sidereal time) fixes the Earth's
    orientation. The angles are (Batch,) arrays; nothing is drawn here.
    """

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
        response,
        gmst_radians
    ) 
    
    antenna_patern = ops.expand_dims(antenna_patern, axis=-1)
    
    # Deal with non-incoherent case:
    if (len(ops.shape(strain)) == 3):
        strain = ops.expand_dims(strain, axis=1)
    
    injection = ops.sum(strain*antenna_patern, axis = -2)

    time_shift_seconds = get_time_delay_(
        right_ascension, 
        declination,
        location,
        gmst_radians
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
    # The arms' tilt above the local horizontal, as in lal's frDetector.
    x_altitude_radians : float = 0.0
    y_altitude_radians : float = 0.0

# lal's arm altitudes (its frDetector values) for the bundled detectors.
_ARM_ALTITUDE_RADIANS : Dict = {
    "livingston" : {"x_altitude_radians": -3.1210e-4, "y_altitude_radians": -6.1070e-4},
    "hanford" : {"x_altitude_radians": -6.1950e-4, "y_altitude_radians": 1.2500e-5},
    "virgo" : {"x_altitude_radians": 0.0, "y_altitude_radians": 0.0},
}

NOISE_PROFILE_DIRECTORY_PATH : Path = RESOURCES / "noise_profiles"

# Bundled amplitude spectral densities (strain / sqrt(Hz)); each file's header
# names its source. The O3 measured curves are LIGO-T2000012-v2's tables, every
# row as published; the others were generated from lalsimulation's curves.
_ALIGO_O3_ACTUAL_H1 = "aligo_o3actual_h1_t2000012.csv"  # LIGO-T2000012-v2, aligo_O3actual_H1.txt
_ALIGO_O3_ACTUAL_L1 = "aligo_o3actual_l1_t2000012.csv"  # LIGO-T2000012-v2, aligo_O3actual_L1.txt
_ADVIRGO_O3_ACTUAL = "advirgo_o3actual_t2000012.csv"    # LIGO-T2000012-v2, avirgo_O3actual.txt
_ALIGO_O3_LOW = "aligo_o3low_t1800545.csv"      # LIGO-T1800545, aLIGOO3LowT1800545
_ADVIRGO_O3_LOW = "advirgo_o3low_t1800545.csv"  # LIGO-T1800545, AdVO3LowT1800545
_ALIGO_DESIGN = "aligo_design_t1800044.csv"     # LIGO-T1800044, aLIGODesignSensitivityT1800044
_ADVIRGO_DESIGN = "advirgo_design_p1200087.csv" # LIGO-P1200087, AdVDesignSensitivityP1200087


class NoiseProfile(Enum):
    """
    A named set of per-detector noise curves for simulated (coloured) noise.

    O3_ACTUAL, the default (``DEFAULT_NOISE_PROFILE``), is each detector's
    measured sensitivity in the third observing run, from LIGO-T2000012-v2:
    H1, L1 and V1 each have their own curve, spectral lines included, from
    10 Hz to 5 kHz (H1, L1) and to 10 kHz (V1). O3_LOW is the O3 low-noise
    projection of the observing scenarios (Living Rev. Relativ. 21, 3 (2018),
    doi:10.1007/s41114-018-0012-9), as lalsimulation tabulates it in
    https://git.ligo.org/lscsoft/lalsuite/-/tree/lalsimulation-v6.2.1/lalsimulation/lib
    (aLIGO for H1 and L1, Advanced Virgo for V1). DESIGN is the aLIGO design curve
    of Barsotti, Fritschel, Evans and Gras, "Updated Advanced LIGO sensitivity
    design curve" (https://dcc.ligo.org/LIGO-T1800044/public) for H1 and L1, and
    the Advanced Virgo design curve (Living Rev. Relativ. 19, 1 (2016),
    doi:10.1007/lrr-2016-1) for V1. Each file's header names its exact source.
    A curve is used only on its own band (``read_noise_curve``).
    """
    O3_ACTUAL = {"Hanford": _ALIGO_O3_ACTUAL_H1, "Livingston": _ALIGO_O3_ACTUAL_L1, "Virgo": _ADVIRGO_O3_ACTUAL}
    O3_LOW = {"Hanford": _ALIGO_O3_LOW, "Livingston": _ALIGO_O3_LOW, "Virgo": _ADVIRGO_O3_LOW}
    DESIGN = {"Hanford": _ALIGO_DESIGN, "Livingston": _ALIGO_DESIGN, "Virgo": _ADVIRGO_DESIGN}

    def path(self, ifo: "IFO") -> Path:
        """The ASD file of this profile for ``ifo``."""
        return NOISE_PROFILE_DIRECTORY_PATH / self.value[ifo.value.name]


#: The profile of coloured noise when none is named; each IFO's
#: ``optimal_psd_path`` is its curve.
DEFAULT_NOISE_PROFILE = NoiseProfile.O3_ACTUAL


def read_noise_curve(path: Path, *, up_to_hertz: float, from_hertz: Optional[float] = None
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """
    The rows (frequency in Hz, ASD in strain / sqrt(Hz)) of a bundled noise
    curve, for a reader that needs it up to ``up_to_hertz`` (and from
    ``from_hertz``, if given). A curve is never extrapolated: a reader that
    needs it beyond its band is refused, by name. Its last row holds for one
    row's spacing, the curve's own resolution, which is where the curves made
    with lalsimulation end: one evaluation step (1/64 Hz) short of 8192 Hz, the
    Nyquist frequency of 16384 Hz. Coloured noise has no power below a curve,
    as at DC, so it passes no ``from_hertz``.
    """
    frequencies, asd = np.loadtxt(path, delimiter=",").T
    reach = frequencies[-1] + (frequencies[-1] - frequencies[-2])
    if up_to_hertz > reach or (from_hertz is not None and from_hertz < frequencies[0]):
        needed = f"{up_to_hertz:g} Hz" if from_hertz is None else f"{from_hertz:g}-{up_to_hertz:g} Hz"
        raise ValueError(
            f"The noise curve {Path(path).name} covers {frequencies[0]:g}-{frequencies[-1]:g} Hz and "
            f"{needed} is needed: a curve is never extrapolated. Use a sample rate or band inside it, "
            "or a noise profile that covers it."
        )
    return frequencies, asd


ifo_data : Dict = {
    "livingston" : {
        "name": "Livingston",
        "optimal_psd_path" : NOISE_PROFILE_DIRECTORY_PATH / DEFAULT_NOISE_PROFILE.value["Livingston"],
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
        "optimal_psd_path" : NOISE_PROFILE_DIRECTORY_PATH / DEFAULT_NOISE_PROFILE.value["Hanford"],
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
        "optimal_psd_path" : NOISE_PROFILE_DIRECTORY_PATH / DEFAULT_NOISE_PROFILE.value["Virgo"],
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
    L1 = IFO_(**ifo_data["livingston"], **_ARM_ALTITUDE_RADIANS["livingston"])
    H1 = IFO_(**ifo_data["hanford"], **_ARM_ALTITUDE_RADIANS["hanford"])
    V1 = IFO_(**ifo_data["virgo"], **_ARM_ALTITUDE_RADIANS["virgo"])

    @property
    def index(self) -> int:
        """Get integer index for this IFO (L1=0, H1=1, V1=2)."""
        return list(IFO).index(self)

class Network:
    def __init__ (
            self,
            parameters : Union[List[IFO], Dict],
            rng : Optional[np.random.Generator] = None
        ):
        """
        A detector network: a list of IFOs, or a dict of geometry. A dict's
        values may be gf.Distribution objects, drawn once here from ``rng`` (a NumPy
        Generator, such as a gf.SeedTree stream; required for random
        geometry); the Network then holds that fixed geometry. A Network is
        stateless geometry: it holds no random state, and ``project_wave``
        takes every sky angle from its caller.
        """
        def geometry_rng():
            if rng is None:
                raise ValueError(
                    "A Network with random geometry needs an rng (a NumPy Generator, "
                    "such as a gf.SeedTree stream) to draw it from."
                )
            return rng

        arguments = {}
        
        if isinstance(parameters, dict):
            parameters = dict(parameters)
            if "num_detectors" in parameters:
                num_detectors = parameters.pop("num_detectors")
                if isinstance(num_detectors, gf.Distribution):
                    num_detectors = int(num_detectors.sample(1, geometry_rng())[0])
            else:
                num_detectors = None
                
            for key, value in parameters.items():
                if isinstance(value, (float, int)):
                    arguments[key] = ops.convert_to_tensor([value], dtype="float32")
                elif isinstance(value, (list, np.ndarray)):  # pragma: no cover
                    arguments[key] = ops.convert_to_tensor(value, dtype="float32")
                elif ops.is_tensor(value):  # pragma: no cover
                     arguments[key] = ops.cast(value, "float32")
                elif isinstance(value, gf.Distribution):
                    if num_detectors is None:
                        raise ValueError("Num detectors not specified")
                    else:
                        arguments[key] = ops.convert_to_tensor(
                            value.sample(num_detectors, geometry_rng()),
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
                "height_meters",
                "x_altitude_radians",
                "y_altitude_radians"
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
        # The detectors this network projects onto, in channel order; None when
        # it was built from geometry rather than from IFOs.
        self.ifos = list(parameters) if isinstance(parameters, list) else None
                
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
        y_length_meters = None,  # Batched tensor
        x_altitude_radians = None,  # Batched tensor or None (level arms)
        y_altitude_radians = None   # Batched tensor or None (level arms)
    ):
        
        PI = np.pi

        if x_angle_radians is None:
            x_angle_radians = \
                y_angle_radians + ops.convert_to_tensor(PI / 2.0, dtype="float32")
        if x_altitude_radians is None:
            x_altitude_radians = ops.zeros_like(y_angle_radians)
        if y_altitude_radians is None:
            y_altitude_radians = ops.zeros_like(y_angle_radians)

        # The arm unit vectors and the response tensor in Earth-fixed
        # coordinates, as lal's XLALCreateDetector builds them: each arm has an
        # azimuth (clockwise from North) and an altitude above the local
        # horizontal, and the response is (x x^T - y y^T) / 2.
        def f64(value):
            return np.asarray(value, dtype=np.float64).reshape(-1)

        lon, lat = f64(longitude_radians), f64(latitude_radians)

        def arm(azimuth, altitude):
            u_north = np.cos(altitude) * np.cos(azimuth)
            u_east = np.cos(altitude) * np.sin(azimuth)
            u_up = np.sin(altitude)
            u_rho = -np.sin(lat) * u_north + np.cos(lat) * u_up
            return np.stack([
                np.cos(lon) * u_rho - np.sin(lon) * u_east,
                np.sin(lon) * u_rho + np.cos(lon) * u_east,
                np.cos(lat) * u_north + np.sin(lat) * u_up,
            ], axis=-1)

        x_arm = arm(f64(x_angle_radians), f64(x_altitude_radians))
        y_arm = arm(f64(y_angle_radians), f64(y_altitude_radians))
        x_arm_response = 0.5 * np.einsum("ni,nj->nij", x_arm, x_arm)
        y_arm_response = 0.5 * np.einsum("ni,nj->nij", y_arm, y_arm)
        full_response = ops.convert_to_tensor(
            x_arm_response - y_arm_response, dtype="float32"
        )

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
        self.x_response = ops.convert_to_tensor(x_arm_response, dtype="float32")
        self.y_response = ops.convert_to_tensor(y_arm_response, dtype="float32")
        self.x_vector = ops.convert_to_tensor(x_arm, dtype="float32")
        self.y_vector = ops.convert_to_tensor(y_arm, dtype="float32")
        self.y_angle_radians = y_angle_radians
        self.x_angle_radians = x_angle_radians
        self.x_altitude_radians = x_altitude_radians
        self.y_altitude_radians = y_altitude_radians
        self.height_meters = height_meters
        self.y_length_meters = y_length_meters
        self.x_length_meters = x_length_meters
        
        self.calculate_max_arrival_time_difference()
    
    @staticmethod
    def gmst_radians(gps_time_seconds = None, num_sources : int = None):
        """
        The Greenwich mean sidereal time at ``gps_time_seconds`` (scalar or per
        source), as a float32 tensor of length ``num_sources`` when given.
        ``None`` means ``REFERENCE_GPS_TIME_SECONDS``.
        """
        if gps_time_seconds is None:
            gps_time_seconds = REFERENCE_GPS_TIME_SECONDS
        gmst = greenwich_mean_sidereal_time(np.asarray(gps_time_seconds, dtype=np.float64))
        if num_sources is not None:
            gmst = np.broadcast_to(gmst, (num_sources,))
        return ops.convert_to_tensor(np.asarray(gmst, dtype=np.float32))

    def get_antenna_pattern(
            self,
            right_ascension, 
            declination, 
            polarization,
            gps_time_seconds = None
        ):
        """
        F+ and Fx of each detector, shape (num_sources, num_detectors, 2), for
        equatorial sky angles at ``gps_time_seconds`` (LAL conventions).
        """
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
            self.response,
            self.gmst_radians(gps_time_seconds, ops.shape(right_ascension)[0])
        )
    
    @classmethod
    def load(
        cls,
        config_path: Path,
        rng: Optional[np.random.Generator] = None,
        ) -> "Network":
        """
        The Network a JSON config describes. Geometry given as a distribution
        is drawn once, from ``rng`` (required for it), as for ``Network(dict, rng=)``.
        """

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
        
        return Network(arguments, rng=rng)

    def get_time_delay(
        self,
        right_ascension, 
        declination,
        gps_time_seconds = None
    ):
        """
        Arrival time at each detector minus arrival at the Earth's centre, shape
        (num_sources, num_detectors), as lal's ``TimeDelayFromEarthCenter``.
        """
        return get_time_delay_(
            right_ascension, 
            declination,
            self.location,
            self.gmst_radians(gps_time_seconds, ops.shape(right_ascension)[0])
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
        polarization  = None,
        gps_time_seconds = None
    ):
        """
        Project (Batch, 2, Time) polarisations onto the network, each detector's
        copy delayed to its arrival time. Sky angles follow LAL and are defined
        at ``gps_time_seconds`` (default ``REFERENCE_GPS_TIME_SECONDS``).
        """

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
        
        missing = [
            name for name, value in (
                ("right_ascension", right_ascension),
                ("declination", declination),
                ("polarization", polarization),
            ) if value is None
        ]
        if missing:
            raise ValueError(
                f"project_wave needs every sky angle; missing: {', '.join(missing)}. "
                "Draw them from a stream (an injection generator does)."
            )

        args = (
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
            polarization=polarization,
            gmst_radians=self.gmst_radians(gps_time_seconds, strain_length)
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
