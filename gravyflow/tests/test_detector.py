"""
Detector geometry and projection: the Network against lal and pycbc.

Antenna patterns and the response tensor of the three cached detectors are
checked against lal in test_waveform_physics.py; here: the rotation helpers,
the network's locations and baselines, a network built from a geometry dict,
time delays, projection of polarisations onto one and three detectors, and the
Network's construction and input checks.

lal is a required input of these tests: if it cannot be imported, they fail.
"""
import json
import warnings

import numpy as np
import pytest
from bokeh.io import output_file, save
from bokeh.layouts import gridplot
from keras import ops
from pycbc.detector import _ground_detectors, add_detector_on_earth

import lal

import gravyflow as gf
from gravyflow.tests._helpers import representation_tolerance

FLOAT32_EPS = float(np.finfo(np.float32).eps)

IFOS = (gf.IFO.L1, gf.IFO.H1, gf.IFO.V1)
LAL_DETECTORS = (lal.LLO_4K_DETECTOR, lal.LHO_4K_DETECTOR, lal.VIRGO_DETECTOR)
LAL_LOCATIONS = np.array([lal.CachedDetectors[name].location for name in LAL_DETECTORS])

#: The largest detector distance from the geocentre, in light seconds.
EARTH_RADIUS_LIGHT_SECONDS = float(np.max(np.linalg.norm(LAL_LOCATIONS, axis=-1))) / lal.C_SI

#: Representation: a delay changes by at most R/c per radian of hour angle, and
#: the float32 hour angle (GMST minus RA, each up to 2 pi) and its trig carry
#: four roundings.
DELAY_TOLERANCE_SECONDS = representation_tolerance(
    np.float32, 2.0 * np.pi * EARTH_RADIUS_LIGHT_SECONDS, c=4
)

#: Representation: an antenna pattern changes by at most 2 per radian of hour
#: angle, which carries the four float32 roundings above.
PATTERN_TOLERANCE = representation_tolerance(np.float32, 2.0 * 2.0 * np.pi, c=4)


def isotropic_sky(rng, count):
    """Right ascension, declination and polarisation, float32, isotropic."""
    right_ascension = rng.uniform(0.0, 2.0 * np.pi, count).astype(np.float32)
    declination = np.arcsin(rng.uniform(-1.0, 1.0, count)).astype(np.float32)
    polarization = rng.uniform(0.0, np.pi, count).astype(np.float32)
    return right_ascension, declination, polarization


@pytest.fixture
def network() -> gf.Network:
    return gf.Network(list(IFOS))


def test_network_locations_and_baselines_match_lal(network):
    assert network.num_detectors == 3
    location = np.asarray(network.location, np.float64)
    # Representation: the geodetic angles and each coordinate are rounded to
    # float32, about one half-ulp each at the Earth's radius.
    tolerance = representation_tolerance(np.float32, np.max(np.abs(LAL_LOCATIONS)), c=2)
    np.testing.assert_allclose(location, LAL_LOCATIONS, rtol=0, atol=tolerance)

    # The longest baseline carries both ends' errors and its float32 norm's
    # rounding, a baseline being shorter than two radii.
    baselines = np.linalg.norm(LAL_LOCATIONS[:, None] - LAL_LOCATIONS[None], axis=-1)
    np.testing.assert_allclose(
        float(network.max_arrival_time_difference_seconds),
        np.max(baselines) / lal.C_SI,
        rtol=0, atol=3.0 * tolerance / lal.C_SI,
    )


def test_network_from_geometry_matches_pycbc():
    longitude, latitude = 0.5, 0.3
    network = gf.Network({
        "longitude_radians": longitude,
        "latitude_radians": latitude,
        "y_angle_radians": 0.0,
        "x_angle_radians": None,
        "height_meters": 0.0,
        "x_length_meters": 4000.0,
        "y_length_meters": 4000.0,
    })
    add_detector_on_earth("gravyflow-test", longitude, latitude)
    reference = _ground_detectors["gravyflow-test"]

    # Representation: the x arm's angle (y + fl(pi/2)) and each component are
    # rounded to float32, unit-scale values.
    tolerance = representation_tolerance(np.float32, 1.0, c=2)
    for name, mine in (("xvec", network.x_vector), ("yvec", network.y_vector), ("response", network.response)):
        np.testing.assert_allclose(
            np.asarray(mine)[0], reference[name], rtol=0, atol=tolerance, err_msg=name
        )


def test_time_delay_matches_lal(network):
    right_ascension, declination, _ = isotropic_sky(np.random.default_rng(11), 100)
    gps = 1126259462.4
    delay = np.asarray(network.get_time_delay(
        ops.convert_to_tensor(right_ascension), ops.convert_to_tensor(declination), gps_time_seconds=gps
    ))
    assert delay.shape == (100, 3)

    # The arrival time at the detector minus at the geocentre, negative for a
    # detector displaced towards the source.
    for index, name in enumerate(LAL_DETECTORS):
        location = lal.CachedDetectors[name].location
        expected = [
            lal.TimeDelayFromEarthCenter(location, float(ra), float(dec), gps)
            for ra, dec in zip(right_ascension, declination)
        ]
        np.testing.assert_allclose(delay[:, index], expected, rtol=0, atol=DELAY_TOLERANCE_SECONDS)


def _polarisations(sample_rate_hertz, num_samples):
    """A centred circularly polarised sine-Gaussian, (2, T) float32, zero at the edges."""
    t = np.arange(num_samples) / sample_rate_hertz - 0.5 * num_samples / sample_rate_hertz
    envelope = np.exp(-(t / 0.02) ** 2)
    phase = 2.0 * np.pi * 150.0 * t
    return np.stack([envelope * np.sin(phase), envelope * np.cos(phase)]).astype(np.float32)


@pytest.mark.parametrize("ifos", [IFOS[:1], IFOS], ids=["one_detector", "three_detectors"])
def test_project_wave_matches_lal(ifos, pytestconfig, tmp_path):
    sample_rate_hertz, num_samples, num_sources = 2048.0, 2048, 16
    gps = 1187008882.4
    polarisations = _polarisations(sample_rate_hertz, num_samples)
    strain = np.broadcast_to(polarisations, (num_sources, 2, num_samples)).copy()
    sky = isotropic_sky(np.random.default_rng(7), num_sources)

    projected = np.asarray(gf.Network(list(ifos)).project_wave(
        strain, sample_rate_hertz, *(ops.convert_to_tensor(angle) for angle in sky), gps_time_seconds=gps
    ), np.float64)
    assert projected.shape == (num_sources, len(ifos), num_samples)

    # lal's pattern and delay applied in float64; the delay is a phase ramp,
    # exact for a signal that is zero at the window's edges.
    gmst = lal.GreenwichMeanSiderealTime(gps)
    frequencies = np.fft.rfftfreq(num_samples, 1.0 / sample_rate_hertz)
    plus, cross = polarisations.astype(np.float64)
    expected = np.empty_like(projected)
    for source, (ra, dec, psi) in enumerate(zip(*sky)):
        for index, ifo in enumerate(ifos):
            detector = lal.CachedDetectors[LAL_DETECTORS[IFOS.index(ifo)]]
            f_plus, f_cross = lal.ComputeDetAMResponse(
                detector.response, float(ra), float(dec), float(psi), gmst
            )
            delay = lal.TimeDelayFromEarthCenter(detector.location, float(ra), float(dec), gps)
            spectrum = np.fft.rfft(f_plus * plus + f_cross * cross)
            expected[source, index] = np.fft.irfft(
                spectrum * np.exp(-2j * np.pi * frequencies * delay), n=num_samples
            )

    # The pattern's and the delay's float32 errors, and the FFT's rounding.
    peak = float(np.max(np.abs(polarisations)))
    rate = float(np.max(np.abs(np.gradient(polarisations, 1.0 / sample_rate_hertz, axis=-1))))
    tolerance = (
        PATTERN_TOLERANCE * 2.0 * peak
        + DELAY_TOLERANCE_SECONDS * 2.0 * rate
        + representation_tolerance(np.float32, 2.0 * peak, c=np.log2(num_samples))
    )
    np.testing.assert_allclose(projected, expected, rtol=0, atol=tolerance)

    if pytestconfig.getoption("plot"):
        output_file(tmp_path / f"projection_{len(ifos)}.html")
        save(gridplot([[
            gf.generate_strain_plot({"Projected": row}, sample_rate_hertz, title=f"{ifo.name} projection")
            for row, ifo in zip(projected[0], ifos)
        ]]))


def test_network_load(tmp_path):
    """A JSON config gives the network its geometry dict gives."""
    config = {
        "num_detectors": 1,
        "latitude_radians": 0.5,
        "longitude_radians": 0.3,
        "x_angle_radians": 0.1,
        "y_angle_radians": 0.2,
        "x_length_meters": 4000.0,
        "y_length_meters": 4000.0,
        "height_meters": 0.0,
    }
    config_path = tmp_path / "network_config.json"
    config_path.write_text(json.dumps(config))

    loaded = gf.Network.load(config_path)
    direct = gf.Network({key: value for key, value in config.items() if key != "num_detectors"})
    assert loaded.num_detectors == 1
    for attribute in ("location", "x_vector", "y_vector", "response"):
        np.testing.assert_array_equal(
            np.asarray(getattr(loaded, attribute)), np.asarray(getattr(direct, attribute)), err_msg=attribute
        )


def test_check_and_convert_list_input():
    network = gf.Network([gf.IFO.L1])
    result = network.check_and_convert([1.0, 2.0], "test_param", tensor_length=2)
    assert ops.shape(result) == (2,)
    np.testing.assert_array_equal(result, [1.0, 2.0])


def test_check_and_convert_tensor_dtype_cast():
    network = gf.Network([gf.IFO.L1])

    # A tensor of another dtype is cast to float32 (an integer one: without
    # x64, jax holds no float64 tensor).
    tensor_int = ops.convert_to_tensor([1, 2], dtype="int32")
    result = network.check_and_convert(tensor_int, "test_param", tensor_length=2)
    assert result.dtype == "float32"
    np.testing.assert_array_equal(result, [1.0, 2.0])

    # A float32 tensor passes through unchanged.
    tensor_f32 = ops.convert_to_tensor([1.0, 2.0], dtype="float32")
    result = network.check_and_convert(tensor_f32, "test_param", tensor_length=2)
    assert result is tensor_f32


def test_network_load_distribution(tmp_path):
    config = {
        "num_detectors": 1,
        "latitude_radians": {
            "type_": "uniform",
            "min_": 0.0,
            "max_": 1.0
        },
        "longitude_radians": 0.0,
        "x_angle_radians": 0.0,
        "y_angle_radians": 0.0,
        "x_length_meters": 4000.0,
        "y_length_meters": 4000.0,
        "height_meters": 0.0
    }
    config_path = tmp_path / "network_dist_config.json"
    config_path.write_text(json.dumps(config))

    # Random geometry is drawn from the rng given, with no deprecated path.
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        network = gf.Network.load(config_path, rng=np.random.default_rng(1))
    assert network.num_detectors == 1
    assert ops.shape(network.location) == (1, 3)
    again = gf.Network.load(config_path, rng=np.random.default_rng(1))
    np.testing.assert_array_equal(np.asarray(again.location), np.asarray(network.location))

    # Without an rng, random geometry is refused by name: there is no hidden stream.
    with pytest.raises(ValueError, match="rng"):
        gf.Network.load(config_path)

    # A generator given the config's path draws the geometry from its own
    # stream of the seed tree: the same network every time, with no warning.
    def generator():
        return gf.WNBGenerator(
            duration_seconds=0.2, min_frequency_hertz=40.0, max_frequency_hertz=200.0,
            network=config_path,
        )
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        first, second = generator().network, generator().network
    np.testing.assert_array_equal(np.asarray(first.location), np.asarray(second.location))
    tree_stream = gf.SeedTree(None).direct().sub("wave/WNBGenerator").stream("param/network")
    np.testing.assert_array_equal(
        np.asarray(first.location), np.asarray(gf.Network.load(config_path, rng=tree_stream).location)
    )


def test_network_holds_no_random_state():
    network = gf.Network([gf.IFO.L1])
    assert not hasattr(network, "rng")
    with pytest.raises(TypeError, match="seed"):
        gf.Network([gf.IFO.L1], seed=1)


def test_project_wave_has_no_seed():
    network = gf.Network([gf.IFO.L1])
    strain = np.zeros((2, 2, 64), np.float32)
    with pytest.raises(TypeError, match="seed"):
        network.project_wave(strain, 512.0, 0.1, 0.2, 0.3, seed=1)


def test_check_and_convert_errors():
    network = gf.Network([gf.IFO.L1])

    tensor = ops.convert_to_tensor([1.0, 2.0, 3.0], dtype="float32")
    with pytest.raises(ValueError, match="must be equal to num injections"):
        network.check_and_convert(tensor, "test", tensor_length=2)

    with pytest.raises(ValueError, match="must be equal to num injections"):
        network.check_and_convert([1.0, 2.0, 3.0], "test", tensor_length=2)

    with pytest.raises(TypeError, match="must be a float, list, tuple, or tensor"):
        network.check_and_convert("invalid", "test", tensor_length=2)


def test_check_and_convert_scalar_none():
    network = gf.Network([gf.IFO.L1])

    result = network.check_and_convert(5.0, "test", tensor_length=3)
    assert ops.shape(result) == (3,)
    np.testing.assert_array_equal(result, [5.0, 5.0, 5.0])

    assert network.check_and_convert(None, "test", tensor_length=3) is None
