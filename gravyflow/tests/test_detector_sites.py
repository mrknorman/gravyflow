"""
The bundled sites (gf.Site: LIGO-India, KAGRA and the Einstein Telescope's
triangle) against lal's, recorded: their values, the Earth-fixed vertices and
response tensors they give, and a detector at each (``DetectorSlot.at``) in a
dataset's network, beside the bundled detectors.

The reference is ``fixtures/lal_geometry/reference.json``, written by the
script beside it with the lal version it records; these tests read it and need
no lal.
"""
import json
from pathlib import Path

import numpy as np
import pytest

import gravyflow as gf
from gravyflow.src.dataset.conditioning.detector import C, GEOMETRY_FEATURE_RADIUS_METERS, detector_geometry
from gravyflow.tests._helpers import representation_tolerance
from gravyflow.tests.test_detector_reference import DELAY_TOLERANCE_SECONDS

REFERENCE = json.loads((Path(__file__).parent / "fixtures" / "lal_geometry" / "reference.json").read_text())
SITES = list(gf.Site)
RV = gf.ReturnVariables

#: lal's site fields, by the name a gf.Site gives each.
LAL_FIELDS = {
    "latitude_radians": "vertexLatitudeRadians", "longitude_radians": "vertexLongitudeRadians",
    "height_meters": "vertexElevation", "x_angle_radians": "xArmAzimuthRadians",
    "y_angle_radians": "yArmAzimuthRadians", "x_altitude_radians": "xArmAltitudeRadians",
    "y_altitude_radians": "yArmAltitudeRadians",
}

#: Representation: lal builds a response tensor from its float32 arms and holds it in float32,
#: two roundings of components at most 1/2.
RESPONSE_TOLERANCE = representation_tolerance(np.float32, 0.5, c=2)

#: lal places a site's vertex from Cartesian constants of its own, and gives the site's latitude
#: and longitude to nine or ten decimal places: half the ninth's last place, 5e-10 rad, is 3 mm
#: at the Earth's radius (KAGRA's two lie 1.3 mm apart).
SITE_VERTEX_TOLERANCE_METERS = 0.5e-9 * GEOMETRY_FEATURE_RADIUS_METERS

#: Representation: the geometry features and the delays read a vertex in float32, so it carries
#: a few float32 roundings of the Earth's radius (about 0.5 m each).
EXAMPLE_VERTEX_TOLERANCE_METERS = representation_tolerance(np.float32, GEOMETRY_FEATURE_RADIUS_METERS, c=4)


def lal(site):
    return REFERENCE["detectors"][site.name]


def slot_geometry(site):
    """The geometry a slot at ``site`` gives, in float64: its fixed parameters, as a slot draws them."""
    slot = gf.DetectorSlot.at(site, noise_curve=gf.IFO.H1)
    parameters = {name: np.float64(getattr(slot, name)) for name in (
        "latitude_radians", "longitude_radians", "x_angle_radians", "height_meters", "x_length_meters",
        "y_length_meters", "x_altitude_radians", "y_altitude_radians")}
    parameters["y_angle_radians"] = parameters["x_angle_radians"] - np.float64(slot.opening_angle_radians)
    return detector_geometry(parameters)


def test_the_reference_records_lal_7_7_1():
    assert REFERENCE["lal_version"] == "7.7.1"


@pytest.mark.parametrize("site", SITES, ids=lambda site: site.name)
def test_each_site_is_lals_cached_detector(site):
    """The values bundled are lal's, by its detector name and prefix; the arms are near twice lal's midpoints."""
    recorded = lal(site)["site"]
    assert (recorded["name"], recorded["prefix"]) == (site.value.lal_name, site.name)
    for name, field in LAL_FIELDS.items():
        assert getattr(site.value, name) == recorded[field], name
    # Nominal lengths (as the bundled detectors'); lal's midpoints lie within 1 % of half of them.
    for arm in ("x", "y"):
        assert getattr(site.value, f"{arm}_length_meters") == pytest.approx(2.0 * recorded[f"{arm}ArmMidpoint"], rel=0.01)


@pytest.mark.parametrize("site", SITES, ids=lambda site: site.name)
def test_a_sites_vertex_and_response_are_lals(site):
    geometry = slot_geometry(site)
    np.testing.assert_allclose(geometry["location"], lal(site)["location_meters"], rtol=0,
                               atol=SITE_VERTEX_TOLERANCE_METERS)
    np.testing.assert_allclose(geometry["response"], lal(site)["response"], rtol=0, atol=RESPONSE_TOLERANCE)


def test_the_triangles_arms_are_at_sixty_degrees_and_the_others_at_right_angles():
    """The opening angle wraps (E1's y azimuth exceeds its x's); KAGRA's arms are 1.4e-5 rad from square."""
    for site in SITES:
        opening = gf.DetectorSlot.at(site, noise_curve=gf.IFO.H1).opening_angle_radians
        assert opening == pytest.approx(np.pi / 3 if site.name.startswith("E") else np.pi / 2, abs=1e-4)


def test_a_detector_at_a_site_is_a_fixed_slot_named_for_it():
    slot = gf.DetectorSlot.at(gf.Site.K1, noise_curve=gf.IFO.L1)
    assert slot.name == "K1" and slot.noise_curve is gf.IFO.L1 and not slot.drawn
    assert gf.DetectorSlot.at(gf.Site.E1, noise_curve=gf.IFO.V1, name="ET_A").name == "ET_A"
    with pytest.raises(TypeError, match="takes a site, a gf.Site"):
        gf.DetectorSlot.at(gf.IFO.H1, noise_curve=gf.IFO.H1)
    with pytest.raises(ValueError, match="repeat I1"):
        gf.Detectors([gf.DetectorSlot.at(gf.Site.I1, gf.IFO.H1), gf.DetectorSlot.at(gf.Site.I1, gf.IFO.L1)])


def _dataset(ifos):
    return gf.Dataset(
        seed=4, group="train",
        noise_obtainer=gf.NoiseObtainer(noise_type=gf.NoiseType.COLORED),
        ifos=ifos,
        waveform_generators=[gf.WNBGenerator(
            scaling_method=gf.ScalingMethod(
                gf.Distribution(value=12.0, type_=gf.DistributionType.CONSTANT), gf.ScalingTypes.SNR),
            duration_seconds=gf.Distribution(value=0.2, type_=gf.DistributionType.CONSTANT),
            min_frequency_hertz=gf.Distribution(value=40.0, type_=gf.DistributionType.CONSTANT),
            max_frequency_hertz=gf.Distribution(value=200.0, type_=gf.DistributionType.CONSTANT))],
        sample_rate_hertz=512.0, onsource_duration_seconds=1.0, offsource_duration_seconds=4.0,
        num_examples_per_batch=4, input_variables=[RV.ONSOURCE, RV.DETECTOR_GEOMETRY],
        output_variables=[RV.INJECTIONS, RV.DETECTOR_SNR, gf.ScalingTypes.SNR],
    )


#: A test network: the bundled detectors and a detector at every site, each with a bundled curve.
NETWORK = [gf.IFO.H1, gf.IFO.L1, gf.IFO.V1, *(
    gf.DetectorSlot.at(site, noise_curve=gf.IFO.V1 if site.name.startswith("E") else gf.IFO.H1) for site in SITES)]


@pytest.mark.parametrize("ifos", [NETWORK, gf.Detectors(NETWORK)], ids=["plain list", "Detectors"])
def test_sites_in_a_network_beside_the_bundled_detectors(ifos):
    """
    Every example has each site's detector where lal places it, with noise of
    its curve and the signal scaled to the network SNR of all eight.
    """
    x, y = _dataset(ifos)[0]
    names = [detector.name for detector in NETWORK]
    position = np.asarray(x["DETECTOR_GEOMETRY"], np.float64)[..., :3] * GEOMETRY_FEATURE_RADIUS_METERS
    expected = np.array([REFERENCE["detectors"][name]["location_meters"] for name in names])
    np.testing.assert_allclose(position, np.broadcast_to(expected, position.shape), rtol=0,
                               atol=EXAMPLE_VERTEX_TOLERANCE_METERS)
    assert np.all(np.std(np.asarray(x["ONSOURCE"]), axis=-1) > 0)
    each = np.asarray(y["DETECTOR_SNR"], np.float64)[:, 0]
    assert each.shape == (4, len(NETWORK)) and np.all(each > 0)
    np.testing.assert_allclose(np.sqrt(np.sum(each ** 2, axis=-1)), np.asarray(y["SNR"])[:, 0],
                               rtol=representation_tolerance(np.float32, 1.0, 2 * np.sqrt(512.0)))


def test_a_site_slots_per_example_vertex_is_lals():
    """
    A fixed site slot's geometry, as each example draws it, is float64 and
    sits at lal's vertex to the site's own precision, as a bundled detector's
    does in a Network; the projection alone reads it in float32.
    """
    detectors = gf.Detectors([gf.DetectorSlot.at(site, noise_curve=gf.IFO.H1) for site in SITES])
    streams = gf.SeedTree(1).batch("train", 0, 0)
    for slot in detectors.slots:
        assert all(value.dtype == np.float64 for value in slot.draw(3, streams).values()), slot.name
    networks = detectors.sample(3, streams)
    for name in gf.ExampleNetworks.GEOMETRY:
        assert np.asarray(getattr(networks, name)).dtype == np.float64, name
    expected = np.array([lal(site)["location_meters"] for site in SITES])
    np.testing.assert_allclose(np.asarray(networks.location), np.broadcast_to(expected, (3,) + expected.shape),
                               rtol=0, atol=SITE_VERTEX_TOLERANCE_METERS)


def test_a_sites_arrival_times_are_lals():
    """Each example's delays at the sites, by the per-example projection, against lal's TimeDelayFromEarthCenter."""
    detectors = gf.Detectors([gf.DetectorSlot.at(site, noise_curve=gf.IFO.H1) for site in SITES])
    count = len(REFERENCE["gps_time_seconds"])
    networks = detectors.sample(count, gf.SeedTree(1).batch("train", 0, 0))
    delay = np.asarray(networks.get_time_delay(
        np.asarray(REFERENCE["right_ascension_radians"], np.float32),
        np.asarray(REFERENCE["declination_radians"], np.float32),
        gps_time_seconds=np.asarray(REFERENCE["gps_time_seconds"], np.float64)))
    expected = np.stack([lal(site)["time_delay_seconds"] for site in SITES], axis=-1)
    np.testing.assert_allclose(delay, expected, rtol=0,
                               atol=DELAY_TOLERANCE_SECONDS + EXAMPLE_VERTEX_TOLERANCE_METERS / C)
