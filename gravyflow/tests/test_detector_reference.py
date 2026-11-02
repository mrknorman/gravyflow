"""
The bundled detectors' geometry against lal's, recorded: their Earth-fixed
vertices and baselines, and the Network's arrival-time delays and antenna
patterns for a set of sky positions.

The reference is ``fixtures/lal_geometry/reference.json``, written by the
script beside it with the lal version it records; these tests read it and
need no lal. A Network computes and holds its geometry in float64, so its
vertices are lal's to well under a millimetre; its projection computes in
float32, so the delays and patterns are lal's to float32's representation.
"""
import json
from pathlib import Path

import numpy as np
import pytest

import gravyflow as gf
from gravyflow.src.dataset.conditioning.detector import C, geodetic_to_earth_fixed
from gravyflow.tests._helpers import representation_tolerance

REFERENCE = json.loads((Path(__file__).parent / "fixtures" / "lal_geometry" / "reference.json").read_text())
IFOS = (gf.IFO.L1, gf.IFO.H1, gf.IFO.V1)
LAL = [REFERENCE["detectors"][ifo.name] for ifo in IFOS]
LAL_LOCATIONS = np.array([detector["location_meters"] for detector in LAL])

#: lal places each vertex from the same geodetic site on the same ellipsoid
#: (WGS84), in float64: the two agree to about 0.1 mm.
VERTEX_TOLERANCE_METERS = 1e-3

#: The largest detector distance from the geocentre, in light seconds.
EARTH_RADIUS_LIGHT_SECONDS = float(np.max(np.linalg.norm(LAL_LOCATIONS, axis=-1))) / C

#: Representation: the projection's float32 hour angle (GMST minus RA, each
#: up to 2 pi) carries two roundings, and a delay changes by at most R/c per
#: radian of it (about 6 ns is seen).
DELAY_TOLERANCE_SECONDS = representation_tolerance(np.float32, 2.0 * np.pi * EARTH_RADIUS_LIGHT_SECONDS, c=2)

#: Representation: an antenna pattern changes by at most 2 per radian of hour
#: angle, which carries the two float32 roundings above.
PATTERN_TOLERANCE = representation_tolerance(np.float32, 2.0 * 2.0 * np.pi, c=2)


@pytest.fixture(scope="module")
def network() -> gf.Network:
    return gf.Network(list(IFOS))


def _sky():
    return tuple(np.asarray(REFERENCE[name], dtype=np.float64) for name in (
        "right_ascension_radians", "declination_radians", "polarization_radians", "gps_time_seconds"))


def test_the_reference_records_its_lal_version():
    assert REFERENCE["lal_version"] == "7.7.1"


def test_bundled_vertices_are_lals(network):
    location = np.asarray(network.location)
    assert location.dtype == np.float64
    np.testing.assert_allclose(location, LAL_LOCATIONS, rtol=0, atol=VERTEX_TOLERANCE_METERS)


def test_bundled_baselines_are_lals(network):
    baselines = np.linalg.norm(LAL_LOCATIONS[:, None] - LAL_LOCATIONS[None], axis=-1)
    np.testing.assert_allclose(network.distances, baselines, rtol=0, atol=2.0 * VERTEX_TOLERANCE_METERS)
    assert network.max_arrival_time_difference_seconds == pytest.approx(
        np.max(baselines) / C, abs=2.0 * VERTEX_TOLERANCE_METERS / C)


def test_time_delays_are_lals(network):
    right_ascension, declination, _, gps = _sky()
    delay = np.asarray(network.get_time_delay(
        right_ascension.astype(np.float32), declination.astype(np.float32), gps_time_seconds=gps))
    expected = np.stack([detector["time_delay_seconds"] for detector in LAL], axis=-1)
    np.testing.assert_allclose(delay, expected, rtol=0, atol=DELAY_TOLERANCE_SECONDS)


def test_antenna_patterns_are_lals(network):
    right_ascension, declination, polarization, gps = _sky()
    pattern = np.asarray(network.get_antenna_pattern(
        right_ascension.astype(np.float32), declination.astype(np.float32), polarization.astype(np.float32),
        gps_time_seconds=gps))
    expected = np.stack([detector["antenna_pattern"] for detector in LAL], axis=1)
    np.testing.assert_allclose(pattern, expected, rtol=0, atol=PATTERN_TOLERANCE)


def test_a_networks_vertices_are_its_float64_sites():
    """A Network built from geometry places each vertex from its site as given, in float64."""
    rng = np.random.default_rng(3)
    for _ in range(20):
        longitude, latitude, height = rng.uniform(-np.pi, np.pi), np.arcsin(rng.uniform(-1, 1)), rng.uniform(0, 500)
        network = gf.Network({
            "longitude_radians": longitude, "latitude_radians": latitude, "height_meters": height,
            "y_angle_radians": 0.4, "x_length_meters": 4000.0, "y_length_meters": 4000.0,
        })
        np.testing.assert_allclose(
            np.asarray(network.location)[0], geodetic_to_earth_fixed(longitude, latitude, height),
            rtol=0, atol=VERTEX_TOLERANCE_METERS)
