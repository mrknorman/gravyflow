from pathlib import Path
from typing import Union, List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum, IntEnum
from functools import cached_property
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
from gravyflow.src.rng.consumers import check_name
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

@jax.jit
def get_time_delay_per_example(right_ascension, declination, location, gmst_radians):
    """
    ``get_time_delay_`` with each source seen by its own detectors: angles and
    ``gmst_radians`` (Batch,), ``location`` (Batch, Detectors, 3). Returns
    (Batch, Detectors).
    """
    def one(ra, dec, loc, gmst):
        return get_time_delay_(ra[None], dec[None], loc, gmst[None])[0]

    return jax.vmap(one)(right_ascension, declination, location, gmst_radians)


@jax.jit(static_argnames=["sample_rate_hertz"])
def project_wave_per_example(
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
        gmst_radians
    ):
    """
    ``project_wave`` with each example projected onto its own detectors: every
    geometry argument is (Batch, Detectors, ...), the angles and
    ``gmst_radians`` are (Batch,), and ``strain`` is (Batch, 2, Time), or
    (Batch, Detectors, 2, Time) for a waveform per detector. Returns (Batch,
    Detectors, Time).
    """
    def one(s, xv, yv, xl, yl, xr, yr, r, loc, ra, dec, pol, gmst):
        return project_wave(
            s[None], sample_rate_hertz, xv, yv, xl, yl, xr, yr, r, loc,
            ra[None], dec[None], pol[None], gmst_radians=gmst[None]
        )[0]

    return jax.vmap(one)(
        strain, x_vector, y_vector, x_length_meters, y_length_meters, x_response,
        y_response, response, location, right_ascension, declination, polarization, gmst_radians
    )


def arm_tensors(longitude, latitude, x_azimuth, y_azimuth, x_altitude, y_altitude):
    """
    A detector's arm unit vectors and response tensors in Earth-fixed
    coordinates, as lal's XLALCreateDetector builds them, in float64: each arm
    has an azimuth (radians clockwise from North) and an altitude above the
    local horizontal at the vertex's geodetic ``longitude`` and ``latitude``.

    Every argument is an array of one shape, one value per detector. Returns
    the x and y arms (shape + (3,)), their responses x x^T / 2 and y y^T / 2,
    and the detector's response (x x^T - y y^T) / 2 (shape + (3, 3)).
    """
    lon, lat = longitude, latitude

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

    x_arm = arm(x_azimuth, x_altitude)
    y_arm = arm(y_azimuth, y_altitude)
    x_response = 0.5 * np.einsum("...i,...j->...ij", x_arm, x_arm)
    y_response = 0.5 * np.einsum("...i,...j->...ij", y_arm, y_arm)
    return x_arm, y_arm, x_response, y_response, x_response - y_response


# WGS84, the reference ellipsoid of lal's detector vertices and of astropy's
# geodetic coordinates: its equatorial radius (m) and flattening.
WGS84_EQUATORIAL_RADIUS_METERS = 6378137.0
WGS84_FLATTENING = 1.0 / 298.257223563


def geodetic_to_earth_fixed(longitude, latitude, height):
    """
    Earth-fixed (geocentric Cartesian) coordinates, in metres, of points at
    WGS84 geodetic ``longitude`` and ``latitude`` (radians) and ``height``
    above the ellipsoid (metres): arrays of one shape, computed in float64.
    Returns shape + (3,). The closed form of astropy's
    ``EarthLocation.from_geodetic`` (ERFA's gd2gc), vectorised.
    """
    longitude, latitude, height = (np.asarray(value, dtype=np.float64) for value in (longitude, latitude, height))
    eccentricity_squared = WGS84_FLATTENING * (2.0 - WGS84_FLATTENING)
    sin_latitude = np.sin(latitude)
    # The prime vertical radius of curvature.
    normal = WGS84_EQUATORIAL_RADIUS_METERS / np.sqrt(1.0 - eccentricity_squared * sin_latitude ** 2)
    return np.stack([
        (normal + height) * np.cos(latitude) * np.cos(longitude),
        (normal + height) * np.cos(latitude) * np.sin(longitude),
        (normal * (1.0 - eccentricity_squared) + height) * sin_latitude,
    ], axis=-1)


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
_ALIGO_O4_HIGH = "aligo_o4high_t2000012.csv"     # LIGO-T2000012-v2, aligo_O4high.txt
_ALIGO_O4_LOW = "aligo_o4low_t2000012.csv"       # LIGO-T2000012-v2, aligo_O4low.txt
_ADVIRGO_O4_HIGH = "advirgo_o4high_t2000012.csv" # LIGO-T2000012-v2, avirgo_O4high_NEW.txt
_ALIGO_O3_LOW = "aligo_o3low_t1800545.csv"      # LIGO-T1800545, aLIGOO3LowT1800545
_ADVIRGO_O3_LOW = "advirgo_o3low_t1800545.csv"  # LIGO-T1800545, AdVO3LowT1800545
_ALIGO_DESIGN = "aligo_design_t1800044.csv"     # LIGO-T1800044, aLIGODesignSensitivityT1800044
_ADVIRGO_DESIGN = "advirgo_design_p1200087.csv" # LIGO-P1200087, AdVDesignSensitivityP1200087
_ALIGO_ZERO_DET_HIGH_POWER = "aligo_zerodet_highpower_t0900288.csv"  # LIGO-T0900288-v3, aLIGOZeroDetHighPower


class NoiseProfile(Enum):
    """
    A named set of per-detector noise curves for simulated (coloured) noise.

    O3_ACTUAL, the default (``DEFAULT_NOISE_PROFILE``), is each detector's
    measured sensitivity in the third observing run, from LIGO-T2000012-v2:
    H1, L1 and V1 each have their own curve, spectral lines included, from
    10 Hz to 5 kHz (H1, L1) and to 10 kHz (V1). O4_HIGH and O4_LOW are the
    fourth run's projections for the updated observing scenarios (Living Rev.
    Relativ. 23, 3 (2020), doi:10.1007/s41114-020-00026-9), LIGO-T2000012-v2's
    tables as published: projections, not measurements. O4_HIGH is aLIGO's high
    projection for H1 and L1 (a 1.4 + 1.4 solar-mass range near 190 Mpc) and Advanced
    Virgo's for V1; O4_LOW is aLIGO's low projection (near 170 Mpc), for H1 and L1
    only, since the document gives Virgo no low projection. No measured O4 curve is
    published as a table; O4's real strain is read with real noise. O3_LOW is the O3
    low-noise projection of the observing scenarios (Living Rev. Relativ. 21, 3 (2018),
    doi:10.1007/s41114-018-0012-9), as lalsimulation tabulates it in
    https://git.ligo.org/lscsoft/lalsuite/-/tree/lalsimulation-v6.2.1/lalsimulation/lib
    (aLIGO for H1 and L1, Advanced Virgo for V1). DESIGN is the aLIGO design curve
    of Barsotti, Fritschel, Evans and Gras, "Updated Advanced LIGO sensitivity
    design curve" (https://dcc.ligo.org/LIGO-T1800044/public) for H1 and L1, and
    the Advanced Virgo design curve (Living Rev. Relativ. 19, 1 (2016),
    doi:10.1007/lrr-2016-1) for V1. ZERO_DET_HIGH_POWER is Advanced LIGO's earlier
    zero-detuning, high-power design curve (LIGO-T0900288-v3), as lalsimulation
    evaluates it (aLIGOZeroDetHighPower), for H1 and L1 only. Each file's header
    names its exact source. A curve is used only on its own band
    (``read_noise_curve``), and only for a detector its profile covers.

    No curve is extrapolated. The aLIGO O3_LOW and DESIGN curves are lalsimulation's
    functions, whose tables stop at 4995.38 Hz and 5000 Hz; above that lalsimulation
    (and PyCBC through it) extrapolates them, to 1.9 times the last tabulated ASD at
    8191 Hz. Their bundled rows stop at the table's end, so a sample rate whose Nyquist
    frequency is past it (16384 Hz) is refused by name; their headers say so.
    """
    O3_ACTUAL = {"Hanford": _ALIGO_O3_ACTUAL_H1, "Livingston": _ALIGO_O3_ACTUAL_L1, "Virgo": _ADVIRGO_O3_ACTUAL}
    O4_HIGH = {"Hanford": _ALIGO_O4_HIGH, "Livingston": _ALIGO_O4_HIGH, "Virgo": _ADVIRGO_O4_HIGH}
    O4_LOW = {"Hanford": _ALIGO_O4_LOW, "Livingston": _ALIGO_O4_LOW}
    O3_LOW = {"Hanford": _ALIGO_O3_LOW, "Livingston": _ALIGO_O3_LOW, "Virgo": _ADVIRGO_O3_LOW}
    DESIGN = {"Hanford": _ALIGO_DESIGN, "Livingston": _ALIGO_DESIGN, "Virgo": _ADVIRGO_DESIGN}
    ZERO_DET_HIGH_POWER = {"Hanford": _ALIGO_ZERO_DET_HIGH_POWER, "Livingston": _ALIGO_ZERO_DET_HIGH_POWER}

    def covers(self, ifo: Union["IFO", "DetectorSlot"]) -> bool:
        """Whether this profile has a curve for ``ifo`` (a slot: its ``noise_curve``'s)."""
        return noise_curve_of(ifo).value.name in self.value

    def path(self, ifo: Union["IFO", "DetectorSlot"]) -> Path:
        """
        The ASD file of this profile for ``ifo``, a bundled detector or a slot (its
        ``noise_curve``'s); ValueError, by name, for a detector it does not cover.
        """
        if not self.covers(ifo):
            raise ValueError(
                f"The noise profile {self.name} has no curve for {ifo.name}: it covers "
                f"{', '.join(sorted(self.value))}. Choose a profile that covers every detector."
            )
        return NOISE_PROFILE_DIRECTORY_PATH / self.value[noise_curve_of(ifo).value.name]


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

    def specification(self) -> Dict:
        """
        The network's geometry by content (``utils.fingerprint.canonical`` reads
        it): every attribute, its arrays as NumPy arrays. A network is stateless
        geometry, so this is all a projection depends on.
        """
        return {name: np.asarray(value) if hasattr(value, "shape") else value
                for name, value in sorted(vars(self).items())}

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

        def f64(value):
            return np.asarray(value, dtype=np.float64).reshape(-1)

        x_arm, y_arm, x_arm_response, y_arm_response, response = arm_tensors(
            f64(longitude_radians), f64(latitude_radians),
            f64(x_angle_radians), f64(y_angle_radians),
            f64(x_altitude_radians), f64(y_altitude_radians),
        )
        full_response = ops.convert_to_tensor(response, dtype="float32")

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
    
    def max_geocentre_delay_seconds(self) -> float:
        """The largest delay between the geocentre and a detector of the network, in seconds."""
        locations = np.asarray(self.location, dtype=np.float64)
        return float(np.max(np.linalg.norm(locations, axis=-1))) / 299792458.0

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


def noise_curve_of(detector: Union[IFO, "DetectorSlot"]) -> IFO:
    """The bundled detector whose noise curve ``detector`` has: itself, or a slot's ``noise_curve``."""
    return detector.noise_curve if isinstance(detector, DetectorSlot) else detector


#: The geometry parameters of a detector slot, as a Network's dict names them
#: (``opening_angle_radians`` gives its y arm's azimuth). Those that default to
#: None are drawn isotropically.
SLOT_GEOMETRY = (
    "latitude_radians", "longitude_radians", "x_angle_radians", "opening_angle_radians",
    "height_meters", "x_length_meters", "y_length_meters", "x_altitude_radians", "y_altitude_radians",
)


def _isotropic_draw(parameter: str, num_examples: int, rng: np.random.Generator) -> np.ndarray:
    """A slot's default draw of ``parameter``: the vertex uniform on the sphere, the azimuth uniform."""
    if parameter == "latitude_radians":
        return np.arcsin(2.0 * rng.random(num_examples) - 1.0)
    if parameter == "longitude_radians":
        return np.pi * (2.0 * rng.random(num_examples) - 1.0)
    return 2.0 * np.pi * rng.random(num_examples)


def _upper_bound(value, name: str) -> float:
    """The largest value ``value`` (a float or a gf.Distribution) can take; ValueError if it has none."""
    if isinstance(value, gf.Distribution):
        if value.type_ == gf.DistributionType.CONSTANT:
            return float(value.value)
        if value.type_ == gf.DistributionType.CHOICE:
            return float(np.max(value.possible_values))
        if value.max_ is None:
            raise ValueError(
                f"{name} needs a largest value: injections are placed so that a signal delayed to the "
                "farthest detector stays in the window. Give its distribution a max_."
            )
        return float(value.max_)
    return float(value)


@dataclass(frozen=True)
class DetectorSlot:
    """
    A detector that is none of the bundled sites: its geometry is drawn for
    each example, and it has noise of its own.

    Noise: simulated (WHITE or COLORED) only, from the slot's own streams,
    ``noise/{name}/{on,off}``, so no two detectors share noise; COLORED noise
    has the curve that ``noise_curve`` (a bundled detector) has in the noise
    profile. Real noise has no slots.

    Geometry, drawn per example: each parameter is a float, a gf.Distribution
    drawn from the slot's stream ``network/{name}/{parameter}``, or (the
    site and the azimuth) None, the default. By default the vertex is uniform
    on the sphere (sin latitude uniform in [-1, 1), longitude in [-pi, pi)), on
    the WGS84 ellipsoid (height 0 m); the x arm's azimuth (radians clockwise
    from North) is uniform in [0, 2 pi); the y arm lies
    ``opening_angle_radians`` anticlockwise of it seen from above, as the
    bundled detectors' arms do (pi / 2; pi / 3 for a triangle's detector); the
    arms are level. Arm lengths enter no projection.

    A slot's ``name`` is its own: none of the bundled detectors' (L1, H1, V1).
    """
    name : str
    noise_curve : IFO
    latitude_radians : Union[float, "gf.Distribution", None] = None
    longitude_radians : Union[float, "gf.Distribution", None] = None
    x_angle_radians : Union[float, "gf.Distribution", None] = None
    opening_angle_radians : Union[float, "gf.Distribution"] = np.pi / 2.0
    height_meters : Union[float, "gf.Distribution"] = 0.0
    x_length_meters : Union[float, "gf.Distribution"] = 4000.0
    y_length_meters : Union[float, "gf.Distribution"] = 4000.0
    x_altitude_radians : Union[float, "gf.Distribution"] = 0.0
    y_altitude_radians : Union[float, "gf.Distribution"] = 0.0

    def __post_init__(self):
        check_name(self.name, "detector slot name")
        if self.name in IFO.__members__:
            raise ValueError(
                f"A detector slot cannot be named {self.name}: L1, H1 and V1 are the bundled detectors, "
                "and a slot's noise streams are its own. Give the slot another name."
            )
        if not isinstance(self.noise_curve, IFO):
            raise TypeError(
                f"A detector slot's noise_curve is the bundled detector (a gf.IFO) whose noise curve it "
                f"has, not {self.noise_curve!r}."
            )
        for parameter in SLOT_GEOMETRY:
            value = getattr(self, parameter)
            if isinstance(value, gf.Distribution) or (value is None and parameter in SLOT_GEOMETRY[:3]):
                continue
            if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
                object.__setattr__(self, parameter, float(value))
                continue
            raise TypeError(
                f"A detector slot's {parameter} is a float or a gf.Distribution"
                f"{', or None for its default draw' if parameter in SLOT_GEOMETRY[:3] else ''}; got {value!r}."
            )

    def draw(self, num_examples : int, streams) -> Dict[str, np.ndarray]:
        """
        The slot's geometry in ``num_examples`` examples, from the batch's
        ``streams`` (a gf.BatchStreams): each parameter (B,), float32, as a
        Network's dict takes them, with the y arm's azimuth,
        ``y_angle_radians``, from the x arm's and the opening angle.
        """
        drawn = {}
        for parameter in SLOT_GEOMETRY:
            value = getattr(self, parameter)
            if isinstance(value, float):
                values = np.full(num_examples, value)
            else:
                rng = streams.stream(f"network/{self.name}/{parameter}")
                values = _isotropic_draw(parameter, num_examples, rng) if value is None \
                    else value.sample(num_examples, rng)
            drawn[parameter] = np.asarray(values, dtype=np.float32)
        opening = drawn.pop("opening_angle_radians")
        drawn["y_angle_radians"] = (
            drawn["x_angle_radians"].astype(np.float64) - opening.astype(np.float64)
        ).astype(np.float32)
        return drawn

    def max_geocentre_distance_meters(self) -> float:
        """The farthest the slot's vertex can be from the geocentre: the equatorial radius plus its greatest height."""
        return WGS84_EQUATORIAL_RADIUS_METERS + max(_upper_bound(self.height_meters, f"{self.name}'s height_meters"), 0.0)


def detector_geometry(parameters : Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    The geometry a projection reads, float32, from a detector's geometry
    parameters (arrays of one shape, as a Network's dict names them, angles in
    radians): its Earth-fixed ``location`` (m), arm unit vectors, arm lengths
    and response tensors, as a Network computes them, with the vertex placed
    by the vectorised WGS84 conversion (``geodetic_to_earth_fixed``).
    """
    def f64(name):
        return np.asarray(parameters[name], dtype=np.float64)

    x_arm, y_arm, x_response, y_response, response = arm_tensors(
        f64("longitude_radians"), f64("latitude_radians"),
        f64("x_angle_radians"), f64("y_angle_radians"),
        f64("x_altitude_radians"), f64("y_altitude_radians"),
    )
    location = geodetic_to_earth_fixed(f64("longitude_radians"), f64("latitude_radians"), f64("height_meters"))
    geometry = {
        "x_vector": x_arm, "y_vector": y_arm,
        "x_length_meters": f64("x_length_meters"), "y_length_meters": f64("y_length_meters"),
        "x_response": x_response, "y_response": y_response, "response": response,
        "location": location,
    }
    return {name: value.astype(np.float32) for name, value in geometry.items()}


#: The Earth radius the geometry features measure positions in: WGS84's equatorial radius.
GEOMETRY_FEATURE_RADIUS_METERS = WGS84_EQUATORIAL_RADIUS_METERS

#: The response tensor's six independent components, in the order of the geometry features.
_RESPONSE_COMPONENTS = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))


@dataclass(frozen=True, eq=False)
class ExampleNetworks:
    """
    The network of each example of one batch: each detector's geometry, as a
    Network holds it but per example, (Batch, Detectors, ...) float32, and
    ``present`` (Batch, Detectors), 1.0 where the detector is in the example's
    network and 0.0 where it is not.

    ``project_wave`` and ``get_time_delay`` take a Network's arguments, one
    source per example: each example is projected onto its own detectors, and
    an absent detector receives no signal.
    """
    x_vector : jnp.ndarray
    y_vector : jnp.ndarray
    x_length_meters : jnp.ndarray
    y_length_meters : jnp.ndarray
    x_response : jnp.ndarray
    y_response : jnp.ndarray
    response : jnp.ndarray
    location : jnp.ndarray
    present : jnp.ndarray

    #: The geometry arrays, in project_wave's order.
    GEOMETRY = ("x_vector", "y_vector", "x_length_meters", "y_length_meters",
                "x_response", "y_response", "response", "location")

    @classmethod
    def stack(cls, columns : List[Dict[str, np.ndarray]], present) -> "ExampleNetworks":
        """Each detector's geometry (a dict of (Batch, ...) arrays, ``detector_geometry``'s), stacked on axis 1."""
        return cls(
            **{name: jnp.asarray(np.stack([column[name] for column in columns], axis=1), dtype=jnp.float32)
               for name in cls.GEOMETRY},
            present=jnp.asarray(present, dtype=jnp.float32),
        )

    @classmethod
    def of_network(cls, network : "Network", num_examples : int) -> "ExampleNetworks":
        """``network``'s fixed geometry in each of ``num_examples`` examples, every detector present."""
        def tiled(value):
            value = jnp.asarray(value, dtype=jnp.float32)
            return jnp.broadcast_to(value, (num_examples,) + value.shape)

        return cls(
            **{name: tiled(getattr(network, name)) for name in cls.GEOMETRY},
            present=jnp.ones((num_examples, int(network.num_detectors)), dtype=jnp.float32),
        )

    @property
    def num_examples(self) -> int:
        return int(self.location.shape[0])

    @property
    def num_detectors(self) -> int:
        return int(self.location.shape[1])

    def _per_example(self, **angles) -> List:
        """Each angle as float32, one per example; ValueError, by name, for any other shape."""
        converted = []
        for name, angle in angles.items():
            angle = ops.convert_to_tensor(angle, dtype="float32")
            if tuple(ops.shape(angle)) != (self.num_examples,):
                raise ValueError(
                    f"{name} must be one angle per example, ({self.num_examples},); got {tuple(ops.shape(angle))}."
                )
            converted.append(angle)
        return converted

    def get_time_delay(self, right_ascension, declination, gps_time_seconds=None):
        """Arrival time at each example's detectors minus at the geocentre, (Batch, Detectors), as Network's."""
        right_ascension, declination = self._per_example(right_ascension=right_ascension, declination=declination)
        return get_time_delay_per_example(
            right_ascension, declination, self.location, Network.gmst_radians(gps_time_seconds, self.num_examples))

    def project_wave(
        self,
        strain,
        sample_rate_hertz = None,
        right_ascension = None,
        declination = None,
        polarization = None,
        gps_time_seconds = None
    ):
        """
        Project each example's (2, Time) polarisations (or (Detectors, 2,
        Time), a waveform per detector) onto its own detectors, as
        Network.project_wave does onto one network: (Batch, Detectors, Time),
        zero in the detectors absent from the example.
        """
        if sample_rate_hertz is None:
            sample_rate_hertz = gf.Defaults.sample_rate_hertz
        if any(angle is None for angle in (right_ascension, declination, polarization)):
            raise ValueError("project_wave needs every sky angle, one per example.")
        if ops.shape(strain)[0] != self.num_examples:
            raise ValueError(
                f"The strain has {ops.shape(strain)[0]} examples and the networks {self.num_examples}."
            )
        right_ascension, declination, polarization = self._per_example(
            right_ascension=right_ascension, declination=declination, polarization=polarization)
        projected = project_wave_per_example(
            ops.cast(strain, "float32"), sample_rate_hertz,
            *(getattr(self, name) for name in self.GEOMETRY),
            right_ascension, declination, polarization,
            Network.gmst_radians(gps_time_seconds, self.num_examples),
        )
        return projected * self.present[..., None]

    def features(self) -> jnp.ndarray:
        """
        Each detector's geometry as a model reads it, (Batch, Detectors, 9)
        float32: its Earth-fixed position in Earth radii
        (``GEOMETRY_FEATURE_RADIUS_METERS``), then the six independent
        components of its response tensor (x x^T - y y^T) / 2: xx, yy, zz,
        xy, xz, yz.
        """
        position = self.location / np.float32(GEOMETRY_FEATURE_RADIUS_METERS)
        response = jnp.stack([self.response[..., i, j] for i, j in _RESPONSE_COMPONENTS], axis=-1)
        return jnp.concatenate([position, response], axis=-1).astype(jnp.float32)


def check_detectors(detectors) -> List:
    """
    ``detectors`` as a list, if each is a bundled detector (gf.IFO) or a
    DetectorSlot and no two share a name; else ValueError, by name. A
    detector's name keys its noise streams, so two detectors of one name would
    have identical noise.
    """
    detectors = list(detectors)
    for detector in detectors:
        if not isinstance(detector, (IFO, DetectorSlot)):
            raise TypeError(f"A detector is a gf.IFO or a gf.DetectorSlot, not {detector!r}.")
    names = [detector.name for detector in detectors]
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        raise ValueError(
            f"The detectors {names} repeat {', '.join(repeated)}: a detector's noise is drawn from streams "
            "named for it, so repeated detectors would have identical noise. For several independent "
            "detectors with one noise curve, use gf.DetectorSlots, each with its own name."
        )
    return detectors


@dataclass(frozen=True)
class Detectors:
    """
    A dataset's detectors when each example has a network of its own: bundled
    detectors (gf.IFO) at their sites, and DetectorSlots whose geometry is
    drawn per example; and how many are present in each example.

    ``num_present`` (a gf.Distribution of integers in [1, len(detectors)];
    None: all of them) is drawn per example from the batch's
    ``network/count`` stream, and the present detectors are a uniformly random
    subset of that size, from ``network/subset``. An absent detector keeps its
    noise but receives no signal, so it adds nothing to the network SNR a
    signal is scaled to. Each detector is one channel of the noise, in order.

    A dataset given Detectors (``ifos=``) projects every example onto its own
    network (``sample``), and its waveform generators take the Detectors as
    their network. Placement and buffers allow for the farthest any detector
    can be: the WGS84 equatorial radius plus a slot's greatest height.
    """
    detectors : Tuple[Union[IFO, DetectorSlot], ...]
    num_present : Optional["gf.Distribution"] = None

    def __post_init__(self):
        object.__setattr__(self, "detectors", tuple(check_detectors(self.detectors)))
        if not self.detectors:
            raise ValueError("Detectors needs at least one detector.")
        if self.num_present is not None and not isinstance(self.num_present, gf.Distribution):
            raise TypeError(
                f"num_present is a gf.Distribution of the number of detectors present, not {self.num_present!r}."
            )

    @property
    def num_detectors(self) -> int:
        return len(self.detectors)

    @property
    def ifos(self) -> List[Union[IFO, DetectorSlot]]:
        """The detectors, in channel order."""
        return list(self.detectors)

    @property
    def slots(self) -> List[DetectorSlot]:
        return [detector for detector in self.detectors if isinstance(detector, DetectorSlot)]

    @property
    def varies(self) -> bool:
        """Whether examples differ in their networks: a slot's geometry or the detectors present."""
        return bool(self.slots) or self.num_present is not None

    @property
    def _bundled_ifos(self) -> List[IFO]:
        return [detector for detector in self.detectors if isinstance(detector, IFO)]

    @cached_property
    def _bundled(self) -> Optional["Network"]:
        """The Network of the bundled detectors, in order; None without any."""
        return Network(self._bundled_ifos) if self._bundled_ifos else None

    def _bundled_column(self, ifo : IFO, num_examples : int) -> Dict[str, np.ndarray]:
        """A bundled detector's geometry, its Network's, in each of ``num_examples`` examples."""
        index = self._bundled_ifos.index(ifo)
        column = {}
        for name in ExampleNetworks.GEOMETRY:
            value = np.asarray(getattr(self._bundled, name), dtype=np.float32)[index]
            column[name] = np.broadcast_to(value, (num_examples,) + value.shape)
        return column

    def _geocentre_distances_meters(self) -> np.ndarray:
        """Each detector's greatest distance from the geocentre, metres."""
        bundled = iter(np.linalg.norm(np.asarray(self._bundled.location, dtype=np.float64), axis=-1)) \
            if self._bundled is not None else iter(())
        return np.array([
            next(bundled) if isinstance(detector, IFO) else detector.max_geocentre_distance_meters()
            for detector in self.detectors
        ])

    def max_geocentre_delay_seconds(self) -> float:
        """The largest delay there can be between the geocentre and a detector, in seconds."""
        return float(np.max(self._geocentre_distances_meters())) / 299792458.0

    @property
    def max_arrival_time_difference_seconds(self) -> float:
        """
        The largest light travel time there can be between two of the
        detectors, in seconds: between bundled detectors their baseline, and
        with a slot, the sum of the two greatest distances from the geocentre.
        """
        if not self.slots:
            return float(self._bundled.max_arrival_time_difference_seconds)
        distances = self._geocentre_distances_meters()
        if len(distances) < 2:
            return 0.0
        bound = distances[:, None] + distances[None, :]
        if self._bundled is not None:
            fixed = [j for j, detector in enumerate(self.detectors) if isinstance(detector, IFO)]
            bound[np.ix_(fixed, fixed)] = np.asarray(self._bundled.distances, dtype=np.float64)
        # A detector's delay from itself is no baseline.
        np.fill_diagonal(bound, 0.0)
        return float(np.max(bound)) / C

    def draw_present(self, num_examples : int, streams) -> np.ndarray:
        """Which detectors each example has, (Batch, Detectors) float32, from ``network/count`` and ``network/subset``."""
        if self.num_present is None:
            return np.ones((num_examples, self.num_detectors), dtype=np.float32)
        count = np.asarray(self.num_present.sample(num_examples, streams.stream("network/count")))
        if count.shape != (num_examples,) or not np.all(count == np.round(count)) \
                or np.any(count < 1) or np.any(count > self.num_detectors):
            raise ValueError(
                f"num_present drew {sorted(set(np.ravel(count).tolist()))}: each example's detector count "
                f"is a whole number from 1 to the {self.num_detectors} detectors."
            )
        keys = streams.stream("network/subset").random((num_examples, self.num_detectors))
        rank = np.argsort(np.argsort(keys, axis=1), axis=1)
        return (rank < count[:, None]).astype(np.float32)

    def sample(self, num_examples : int, streams) -> ExampleNetworks:
        """
        The network of each of ``num_examples`` examples, drawn from the batch's
        ``streams`` (a gf.BatchStreams): a bundled detector's fixed geometry, a
        slot's drawn (``DetectorSlot.draw``), and the detectors present.
        """
        columns = [
            self._bundled_column(detector, num_examples) if isinstance(detector, IFO)
            else detector_geometry(detector.draw(num_examples, streams))
            for detector in self.detectors
        ]
        return ExampleNetworks.stack(columns, self.draw_present(num_examples, streams))
