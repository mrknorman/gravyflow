"""
Networks drawn per example: detector slots, gf.Detectors and ExampleNetworks.

A dataset given detector slots (gf.DetectorSlot), or a per-example detector
count (gf.Detectors), projects each example onto its own network, drawn from
the batch's ``network/...`` streams. Here: the vectorised geodetic conversion
against astropy and lal; a slot's geometry against a Network's; the per-example
projection against one Network per example; each example's own, reproducible
geometry; delays within the light travel time; independent slot noise with its
curve; absent detectors carrying no signal and no SNR; the geometry and mask
outputs of fixed networks; and the refusals (repeated names, slots named as a
bundled detector, slots with real noise).

lal is a required input of these tests: if it cannot be imported, they fail.
"""
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.signal
from astropy import coordinates, units

import lal

import gravyflow as gf
from gravyflow.src.dataset.conditioning.detector import (
    C, WGS84_EQUATORIAL_RADIUS_METERS, detector_geometry, geodetic_to_earth_fixed,
)
from gravyflow.src.dataset.noise import noise as gf_noise
from gravyflow.src.rng import check_consumer
from gravyflow.tests._helpers import representation_tolerance
from gravyflow.tests.test_colored_noise_amplitude import _assert_median_matches_the_profile

RV = gf.ReturnVariables
SAMPLE_RATE = 512.0
BATCH = 8

IFOS = (gf.IFO.L1, gf.IFO.H1, gf.IFO.V1)
LAL_DETECTORS = (lal.LLO_4K_DETECTOR, lal.LHO_4K_DETECTOR, lal.VIRGO_DETECTOR)


def slots(count=3, **geometry):
    return [gf.DetectorSlot(f"D{j}", noise_curve=gf.IFO.H1, **geometry) for j in range(count)]


def count_between(low, high):
    return gf.Distribution(min_=low, max_=high, type_=gf.DistributionType.UNIFORM, dtype=int)


def wnb(**kwargs):
    return gf.WNBGenerator(
        scaling_method=gf.ScalingMethod(
            gf.Distribution(min_=8.0, max_=20.0, type_=gf.DistributionType.UNIFORM), gf.ScalingTypes.SNR),
        duration_seconds=gf.Distribution(min_=0.1, max_=0.3, type_=gf.DistributionType.UNIFORM),
        min_frequency_hertz=gf.Distribution(min_=40.0, max_=80.0, type_=gf.DistributionType.UNIFORM),
        max_frequency_hertz=gf.Distribution(min_=120.0, max_=200.0, type_=gf.DistributionType.UNIFORM),
        **kwargs,
    )


def dataset(ifos, generators=None, inputs=(RV.ONSOURCE,), outputs=(RV.INJECTIONS,), seed=5, group="train",
            noise_type=gf.NoiseType.WHITE, **kwargs):
    return gf.Dataset(
        seed=seed, group=group,
        noise_obtainer=gf.NoiseObtainer(noise_type=noise_type),
        ifos=ifos,
        waveform_generators=[wnb()] if generators is None else generators,
        sample_rate_hertz=SAMPLE_RATE, onsource_duration_seconds=1.0, offsource_duration_seconds=4.0,
        num_examples_per_batch=BATCH, steps_per_epoch=4,
        input_variables=list(inputs), output_variables=list(outputs), **kwargs,
    )


def isotropic_sky(rng, count):
    return (rng.uniform(0.0, 2.0 * np.pi, count).astype(np.float32),
            np.arcsin(rng.uniform(-1.0, 1.0, count)).astype(np.float32),
            rng.uniform(0.0, np.pi, count).astype(np.float32))


# --- refusals ---------------------------------------------------------------

def test_a_dataset_whose_detectors_repeat_a_name_is_refused():
    """Noise streams are keyed by name: H1 twice would be one noise twice, so it is refused, by name."""
    with pytest.raises(ValueError, match="repeat H1"):
        dataset([gf.IFO.H1, gf.IFO.H1])
    with pytest.raises(ValueError, match="repeat H1"):
        gf.Dataset(seed=5, group="train",
                   noise_obtainer=gf.NoiseObtainer(noise_type=gf.NoiseType.COLORED, ifos=[gf.IFO.H1, gf.IFO.H1]),
                   input_variables=[RV.ONSOURCE])
    with pytest.raises(ValueError, match="repeat D0"):
        gf.Detectors([gf.DetectorSlot("D0", noise_curve=gf.IFO.H1)] * 2)


def test_noise_of_repeated_detectors_is_refused():
    batch = gf_noise.white_noise_batcher(BATCH, [gf.IFO.L1, gf.IFO.L1], 1.0, 0.5, 4.0, SAMPLE_RATE)
    with pytest.raises(ValueError, match="repeat L1"):
        batch(gf.SeedTree(5).batch("train", 0, 0))


def test_a_slot_is_named_as_no_bundled_detector():
    for name in ("L1", "H1", "V1"):
        with pytest.raises(ValueError, match="cannot be named"):
            gf.DetectorSlot(name, noise_curve=gf.IFO.H1)
    with pytest.raises(TypeError, match="noise_curve"):
        gf.DetectorSlot("D0", noise_curve="H1")
    with pytest.raises(TypeError, match="height_meters"):
        gf.DetectorSlot("D0", noise_curve=gf.IFO.H1, height_meters=None)



def uniform(low, high):
    return gf.Distribution(min_=low, max_=high, type_=gf.DistributionType.UNIFORM)


@pytest.mark.parametrize("parameter,value", [
    ("latitude_radians", 1.6), ("latitude_radians", uniform(-1.0, 1.6)),
    ("latitude_radians", gf.Distribution(mean=0.0, std=0.3, type_=gf.DistributionType.NORMAL)),
    ("opening_angle_radians", 0.0), ("opening_angle_radians", np.pi), ("opening_angle_radians", uniform(0.5, 3.5)),
    ("height_meters", 10_001.0), ("height_meters", -10_001.0), ("height_meters", uniform(0.0, 2e4)),
    ("height_meters", gf.Distribution(mean=0.0, std=1.0, type_=gf.DistributionType.NORMAL)),
    ("x_altitude_radians", -1.6), ("y_altitude_radians", gf.Distribution(possible_values=[0.0, 2.0],
                                                                         type_=gf.DistributionType.CHOICE)),
])
def test_a_slots_geometry_outside_its_range_is_refused_by_name(parameter, value):
    with pytest.raises(ValueError, match=f"D0's {parameter} can lie"):
        gf.DetectorSlot("D0", noise_curve=gf.IFO.H1, **{parameter: value})


NON_FINITE = {
    "nan": float("nan"), "inf": float("inf"), "-inf": -float("inf"),
    "uniform to nan": uniform(0.0, float("nan")), "uniform from -inf": uniform(-float("inf"), 0.0),
    "normal mean nan": gf.Distribution(mean=float("nan"), std=0.1, type_=gf.DistributionType.NORMAL),
    "normal std inf": gf.Distribution(mean=0.0, std=float("inf"), type_=gf.DistributionType.NORMAL),
    "constant inf": gf.Distribution(value=float("inf"), type_=gf.DistributionType.CONSTANT),
    "choice of nan": gf.Distribution(possible_values=[0.0, float("nan")], type_=gf.DistributionType.CHOICE),
}


@pytest.mark.parametrize("value", NON_FINITE.values(), ids=NON_FINITE.keys())
@pytest.mark.parametrize("parameter", gf.src.dataset.conditioning.detector.SLOT_GEOMETRY)
def test_a_slots_non_finite_geometry_is_refused_by_name(parameter, value):
    """Every geometry parameter, longitude and the azimuths included: a NaN or infinite value or bound."""
    with pytest.raises(ValueError, match=f"D0's {parameter} must be finite"):
        gf.DetectorSlot("D0", noise_curve=gf.IFO.H1, **{parameter: value})


def test_an_unbounded_normal_of_a_wrapping_angle_is_kept():
    """Its bounds are absent, not infinite: every value it draws is finite, and longitude wraps."""
    gf.DetectorSlot("D0", noise_curve=gf.IFO.H1,
                    longitude_radians=gf.Distribution(mean=0.0, std=10.0, type_=gf.DistributionType.NORMAL))


def test_a_slots_geometry_at_its_ranges_ends_is_kept():
    """Latitude and altitude to plus or minus pi / 2, an opening angle inside (0, pi), a height within 10 km."""
    slot = gf.DetectorSlot("D0", noise_curve=gf.IFO.H1, latitude_radians=np.pi / 2, opening_angle_radians=3.0,
                           height_meters=uniform(-10_000.0, 10_000.0), x_altitude_radians=-np.pi / 2,
                           longitude_radians=7.0, x_angle_radians=-20.0)  # longitude and azimuth wrap
    assert slot.max_geocentre_distance_meters() == WGS84_EQUATORIAL_RADIUS_METERS + 10_000.0


def test_a_plain_list_with_a_drawn_slot_is_refused():
    """A plain list of detectors is a fixed network; a slot whose geometry is drawn goes in gf.Detectors."""
    drawn = [gf.IFO.H1, gf.DetectorSlot("D0", noise_curve=gf.IFO.H1, latitude_radians=0.3, longitude_radians=1.0,
                                        x_angle_radians=uniform(0.0, 1.0))]
    with pytest.raises(ValueError, match=r"geometry of the detector slots \['D0'\] is drawn"):
        dataset(drawn)
    with pytest.raises(ValueError, match=r"geometry of the detector slots \['D0', 'D1'\] is drawn"):
        dataset(slots(2))
    with pytest.raises(ValueError, match=r"geometry of the detector slots \['D0'\] is drawn"):
        gf.Dataset(seed=5, group="train", noise_obtainer=gf.NoiseObtainer(noise_type=gf.NoiseType.WHITE, ifos=slots(1)),
                   waveform_generators=[wnb()], sample_rate_hertz=SAMPLE_RATE, onsource_duration_seconds=1.0,
                   offsource_duration_seconds=4.0, num_examples_per_batch=BATCH, input_variables=[RV.ONSOURCE])


def test_a_plain_list_with_a_fixed_slot_is_a_fixed_network():
    """A slot with all its geometry fixed is a custom detector: every example has it, in a plain list too."""
    fixed = gf.DetectorSlot("D0", noise_curve=gf.IFO.H1, latitude_radians=0.3, longitude_radians=1.0,
                            x_angle_radians=0.2)
    assert not fixed.drawn and gf.DetectorSlot("D1", noise_curve=gf.IFO.H1).drawn
    ds = dataset([gf.IFO.H1, fixed], inputs=(RV.ONSOURCE, RV.DETECTOR_GEOMETRY))
    geometry = np.asarray(ds[0][0]["DETECTOR_GEOMETRY"])
    np.testing.assert_array_equal(geometry, np.broadcast_to(geometry[:1], geometry.shape))
    np.testing.assert_array_equal(ds[0][1]["INJECTIONS"], dataset(gf.Detectors([gf.IFO.H1, fixed]))[0][1]["INJECTIONS"])

@pytest.mark.parametrize("noise_type", [gf.NoiseType.REAL, gf.NoiseType.PSEUDO_REAL])
def test_slots_have_no_real_noise(noise_type):
    with pytest.raises(ValueError, match="have no real data"):
        dataset(gf.Detectors(slots(2)), noise_type=noise_type)
    with pytest.raises(ValueError, match="slots"):
        gf.NoiseObtainer(noise_type=noise_type).batcher(
            gf.SeedTree(5), "train", sample_rate_hertz=SAMPLE_RATE, onsource_duration_seconds=1.0,
            crop_duration_seconds=0.5, offsource_duration_seconds=4.0, num_examples_per_batch=BATCH,
            ifos=slots(2))


@pytest.mark.parametrize("num_present", [
    count_between(0, 2), count_between(2, 3),
    gf.Distribution(value=3, type_=gf.DistributionType.CONSTANT),
    gf.Distribution(possible_values=[1, 3], type_=gf.DistributionType.CHOICE),
    gf.Distribution(mean=1.5, std=0.5, type_=gf.DistributionType.NORMAL, dtype=int),
], ids=["below", "above", "constant", "choice", "unbounded"])
def test_a_detector_count_outside_the_detectors_is_refused_at_construction(num_present):
    with pytest.raises(ValueError, match=r"num_present can draw .* outside the 1 to 2 detectors"):
        gf.Detectors(slots(2), num_present=num_present)


@pytest.mark.parametrize("num_present", [
    gf.Distribution(min_=1.0, max_=2.0, type_=gf.DistributionType.UNIFORM),
    gf.Distribution(value=1.5, type_=gf.DistributionType.CONSTANT),
    gf.Distribution(possible_values=[1, 1.5], type_=gf.DistributionType.CHOICE),
], ids=["uniform_float", "constant", "choice"])
def test_a_detector_count_that_is_not_whole_is_refused_at_construction(num_present):
    with pytest.raises(ValueError, match="num_present draws whole numbers"):
        gf.Detectors(slots(2), num_present=num_present)


def test_a_detector_count_inside_the_detectors_is_drawn():
    for num_present in (count_between(1, 2), gf.Distribution(value=2, type_=gf.DistributionType.CONSTANT),
                        gf.Distribution(possible_values=[1, 2], type_=gf.DistributionType.CHOICE)):
        present = gf.Detectors(slots(2), num_present=num_present).draw_present(64, gf.SeedTree(5).batch("train", 0, 0))
        assert set(np.sum(present, axis=1).tolist()) <= {1.0, 2.0}


def test_the_network_streams_are_registered():
    for consumer in ("network/D0/latitude_radians", "network/count", "network/subset",
                     "dataset/a/network/D0/x_angle_radians", "pool/p/network/count"):
        assert check_consumer(consumer) == consumer


# --- geometry ---------------------------------------------------------------

def test_vectorised_geodetic_matches_astropy():
    """The closed form agrees with astropy's EarthLocation.from_geodetic to under 1 mm."""
    rng = np.random.default_rng(3)
    longitude = rng.uniform(-np.pi, np.pi, 2000)
    latitude = np.arcsin(rng.uniform(-1.0, 1.0, 2000))
    height = rng.uniform(-500.0, 5000.0, 2000)
    reference = coordinates.EarthLocation.from_geodetic(longitude * units.rad, latitude * units.rad, height * units.m)
    expected = np.stack([reference.x.value, reference.y.value, reference.z.value], axis=-1)
    np.testing.assert_allclose(geodetic_to_earth_fixed(longitude, latitude, height), expected, rtol=0, atol=1e-3)


def test_vectorised_geodetic_places_the_bundled_detectors_at_lals_vertices():
    """From each bundled detector's geodetic site, its vertex is lal's to under 1 mm."""
    for ifo, name in zip(IFOS, LAL_DETECTORS):
        site = ifo.value
        location = geodetic_to_earth_fixed(site.longitude_radians, site.latitude_radians, site.height_meters)
        np.testing.assert_allclose(location, lal.CachedDetectors[name].location, rtol=0, atol=1e-3)


def test_a_slots_geometry_is_a_networks():
    """The geometry of a slot's drawn parameters is a Network's of them, bit for bit, in float64."""
    slot = gf.DetectorSlot("D0", noise_curve=gf.IFO.H1, opening_angle_radians=np.pi / 3.0)
    parameters = slot.draw(16, gf.SeedTree(5).batch("train", 0, 0))
    geometry = detector_geometry(parameters)
    network = gf.Network({**{key: np.asarray(value) for key, value in parameters.items()}, "num_detectors": 16})
    for name in gf.ExampleNetworks.GEOMETRY:
        assert geometry[name].dtype == np.float64, name
        np.testing.assert_array_equal(geometry[name], np.asarray(getattr(network, name)), err_msg=name)
    expected = geodetic_to_earth_fixed(parameters["longitude_radians"], parameters["latitude_radians"],
                                       parameters["height_meters"])
    np.testing.assert_array_equal(geometry["location"], expected)
    opening = np.arccos(np.clip(np.sum(geometry["x_vector"] * geometry["y_vector"], axis=-1), -1.0, 1.0))
    np.testing.assert_allclose(opening, np.pi / 3.0, atol=1e-6)


def test_default_slot_geometry_is_isotropic_and_level():
    """Sites uniform on the sphere, azimuths uniform, arms level and at right angles, on the ellipsoid."""
    parameters = gf.DetectorSlot("D0", noise_curve=gf.IFO.H1).draw(20000, gf.SeedTree(5).batch("train", 0, 0))
    n = 20000
    for name, samples, low, high in (
            ("sin latitude", np.sin(parameters["latitude_radians"]), -1.0, 1.0),
            ("longitude", parameters["longitude_radians"], -np.pi, np.pi),
            ("x azimuth", parameters["x_angle_radians"], 0.0, 2.0 * np.pi)):
        assert low <= samples.min() and samples.max() <= high, name
        # Statistical: the mean of n uniforms is within 5 standard errors of the centre.
        assert abs(samples.mean() - 0.5 * (low + high)) < 5.0 * (high - low) / np.sqrt(12.0 * n), name
    np.testing.assert_array_equal(parameters["height_meters"], 0.0)
    np.testing.assert_array_equal(parameters["x_altitude_radians"], 0.0)
    geometry = detector_geometry(parameters)
    np.testing.assert_allclose(np.sum(geometry["x_vector"] * geometry["y_vector"], axis=-1), 0.0, atol=1e-6)
    radius = np.linalg.norm(geometry["location"].astype(np.float64), axis=-1)
    polar = WGS84_EQUATORIAL_RADIUS_METERS * (1.0 - 1.0 / 298.257223563)
    assert polar - 1.0 < radius.min() and radius.max() < WGS84_EQUATORIAL_RADIUS_METERS + 1.0


def test_per_example_projection_equals_a_network_per_example():
    """Each example projected onto its own geometry is its own Network's projection, to float32 rounding."""
    rng = np.random.default_rng(4)
    detectors = gf.Detectors(slots(3))
    networks = detectors.sample(BATCH, gf.SeedTree(5).batch("train", 0, 0))
    per_detector = [slot.draw(BATCH, gf.SeedTree(5).batch("train", 0, 0)) for slot in detectors.slots]
    references = [
        gf.Network({**{key: np.array([p[key][b] for p in per_detector]) for key in per_detector[0]},
                    "num_detectors": 3})
        for b in range(BATCH)
    ]
    # The projection, given each example's own Network's geometry.
    own = gf.ExampleNetworks(
        **{name: np.stack([np.asarray(getattr(network, name)) for network in references])
           for name in gf.ExampleNetworks.GEOMETRY},
        present=np.ones((BATCH, 3), np.float32))
    for shape in ((BATCH, 2, 1024), (BATCH, 3, 2, 1024)):
        strain = rng.normal(size=shape).astype(np.float32)
        ra, dec, pol = isotropic_sky(rng, BATCH)
        projected = np.asarray(own.project_wave(strain, SAMPLE_RATE, ra, dec, pol))
        expected = np.concatenate([
            np.asarray(network.project_wave(*(jnp.asarray(v[b:b + 1]) for v in (strain,)), SAMPLE_RATE,
                                            *(jnp.asarray(v[b:b + 1]) for v in (ra, dec, pol))))
            for b, network in enumerate(references)
        ])
        assert projected.shape == expected.shape == (BATCH, 3, 1024)
        # Representation: a few float32 epsilons of the projection's scale (its FFT sums).
        np.testing.assert_allclose(projected, expected, rtol=0, atol=64 * np.finfo(np.float32).eps * np.abs(expected).max())
        np.testing.assert_allclose(
            np.asarray(own.get_time_delay(ra, dec)),
            np.concatenate([np.asarray(network.get_time_delay(jnp.asarray(ra[b:b + 1]), jnp.asarray(dec[b:b + 1])))
                            for b, network in enumerate(references)]),
            rtol=0, atol=1e-9)
    # Drawn geometry is each Network's, which the projection reads in float32.
    for name in gf.ExampleNetworks.GEOMETRY:
        np.testing.assert_array_equal(getattr(networks, name), getattr(own, name).astype(np.float32), err_msg=name)


def test_each_example_has_its_own_reproducible_geometry():
    detectors = gf.Detectors(slots(3))
    tree = gf.SeedTree(5)
    first = detectors.sample(BATCH, tree.batch("train", 0, 0)).features()
    again = gf.Detectors(slots(3)).sample(BATCH, tree.batch("train", 0, 0)).features()
    np.testing.assert_array_equal(first, again)
    # Every example and every slot differ from every other.
    flat = np.asarray(first).reshape(-1, 9)
    assert len(np.unique(flat, axis=0)) == len(flat)
    for other in (tree.batch("train", 0, 1), tree.batch("train", 1, 0), tree.batch("validate", 0, 0),
                  gf.SeedTree(6).batch("train", 0, 0)):
        assert not np.any(np.all(np.asarray(detectors.sample(BATCH, other).features()) == first, axis=-1))
    # A slot's draws are its own: adding a slot moves no other slot's geometry.
    np.testing.assert_array_equal(gf.Detectors(slots(4)).sample(BATCH, tree.batch("train", 0, 0)).features()[:, :3], first)


def test_a_datasets_examples_have_their_own_networks():
    ds = dataset(gf.Detectors(slots(3)), inputs=(RV.ONSOURCE, RV.DETECTOR_GEOMETRY, RV.DETECTOR_MASK))
    x0, _ = ds[0]
    x1, _ = ds[1]
    assert x0["DETECTOR_GEOMETRY"].shape == (BATCH, 3, 9) and x0["DETECTOR_MASK"].shape == (BATCH, 3)
    np.testing.assert_array_equal(x0["DETECTOR_MASK"], 1.0)
    np.testing.assert_array_equal(ds[0][0]["DETECTOR_GEOMETRY"], x0["DETECTOR_GEOMETRY"])
    assert not np.array_equal(x0["DETECTOR_GEOMETRY"], x1["DETECTOR_GEOMETRY"])
    np.testing.assert_array_equal(
        x0["DETECTOR_GEOMETRY"], ds.detectors.sample(BATCH, ds.streams(0)).features())
    # Within an example the features are its detectors' own: positions on the Earth, traceless responses.
    radius = np.linalg.norm(np.asarray(x0["DETECTOR_GEOMETRY"])[..., :3], axis=-1)
    assert np.all((radius > 0.99) & (radius < 1.0 + 1e-6))
    np.testing.assert_allclose(np.asarray(x0["DETECTOR_GEOMETRY"])[..., 3:6].sum(axis=-1), 0.0, atol=1e-6)


def test_delays_lie_within_the_light_travel_time():
    """Each example's inter-detector delays are within its own baselines' light travel time and the bound."""
    rng = np.random.default_rng(6)
    detectors = gf.Detectors([gf.IFO.H1, *slots(3)])
    networks = detectors.sample(256, gf.SeedTree(5).batch("train", 0, 0))
    ra, dec, _ = isotropic_sky(rng, 256)
    delays = np.asarray(networks.get_time_delay(ra, dec), np.float64)
    location = np.asarray(networks.location, np.float64)
    baselines = np.linalg.norm(location[:, :, None] - location[:, None], axis=-1) / C
    differences = np.abs(delays[:, :, None] - delays[:, None])
    tolerance = 1e-9  # Representation: float32 delays of up to R/c.
    assert np.all(differences <= baselines + tolerance)
    assert np.max(differences) <= detectors.max_arrival_time_difference_seconds + tolerance
    assert np.max(np.abs(delays)) <= detectors.max_geocentre_delay_seconds() + tolerance
    # The bounds: the farthest a slot can be is the equatorial radius.
    assert detectors.max_geocentre_delay_seconds() == WGS84_EQUATORIAL_RADIUS_METERS / 299792458.0
    np.testing.assert_allclose(detectors.max_arrival_time_difference_seconds, 2.0 * WGS84_EQUATORIAL_RADIUS_METERS / C)


def test_placement_allows_for_the_farthest_detector():
    ds = dataset(gf.Detectors(slots(2, height_meters=uniform(0.0, 3000.0))))
    generator = ds.waveform_generators[0]
    assert generator.network is ds.detectors
    assert generator.detector_margin_seconds() == (WGS84_EQUATORIAL_RADIUS_METERS + 3000.0) / 299792458.0


def test_central_times_lie_within_each_examples_light_travel_time():
    ds = dataset(gf.Detectors(slots(3), num_present=count_between(2, 3)),
                 inputs=(RV.DETECTOR_GEOMETRY,), outputs=(RV.CENTRAL_TIME, RV.INJECTION_MASKS))
    x, y = ds[0]
    times = np.asarray(y["CENTRAL_TIME"], np.float64)[:, 0] * 1.0  # (B, D), seconds: the window is 1 s
    location = np.asarray(x["DETECTOR_GEOMETRY"], np.float64)[..., :3] * WGS84_EQUATORIAL_RADIUS_METERS
    baselines = np.linalg.norm(location[:, :, None] - location[:, None], axis=-1) / C
    injected = np.asarray(y["INJECTION_MASKS"])[:, 0] > 0
    differences = np.abs(times[:, :, None] - times[:, None])[injected]
    assert injected.any() and np.all(differences <= baselines[injected] + 1e-6)


# --- noise ------------------------------------------------------------------

def test_slot_noise_is_independent_and_has_its_curve():
    """Slots with one curve have independent noise, each with that curve's PSD; a slot of another curve has its own."""
    detectors = [gf.DetectorSlot("D0", noise_curve=gf.IFO.H1), gf.DetectorSlot("D1", noise_curve=gf.IFO.H1),
                 gf.DetectorSlot("D2", noise_curve=gf.IFO.V1)]
    examples, duration = 32, 16.0
    batch = gf_noise.colored_noise_batcher(
        num_examples_per_batch=examples, onsource_duration_seconds=duration, crop_duration_seconds=0.0,
        offsource_duration_seconds=duration, ifos=detectors, sample_rate_hertz=2048.0,
        noise_profile=gf.NoiseProfile.O3_LOW,
    )
    x = np.asarray(batch(gf.SeedTree(7).batch("train", 0, 0))[RV.ONSOURCE], np.float64)
    for j, detector in enumerate(detectors):
        frequencies, asd = np.loadtxt(gf.NoiseProfile.O3_LOW.path(detector.noise_curve), delimiter=",").T
        f, p = scipy.signal.welch(x[:, j:j + 1], fs=2048.0, nperseg=int(4 * 2048.0))
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = p / np.interp(f, frequencies, asd)[None, None] ** 2
        _assert_median_matches_the_profile(ratio, f, examples, duration)
    # Independent: each slot's noise is coloured from its own unit normal stream, and any two
    # streams' correlation is within 5 standard errors (1 / sqrt(samples)) of zero.
    white = np.asarray(gf_noise.unit_normal_noise(gf.SeedTree(7).batch("train", 0, 0), detectors, "on", examples, 8192))
    for i in range(3):
        for j in range(i + 1, 3):
            correlation = np.corrcoef(white[:, i].ravel(), white[:, j].ravel())[0, 1]
            assert abs(correlation) < 5.0 / np.sqrt(white[:, i].size), (i, j, correlation)


def test_bundled_detectors_keep_their_noise_beside_slots():
    """L1's and H1's noise streams are as they were: a slot beside them moves neither."""
    streams = gf.SeedTree(7).batch("train", 0, 0)
    alone = gf_noise.unit_normal_noise(streams, [gf.IFO.L1, gf.IFO.H1], "on", BATCH, 256)
    beside = gf_noise.unit_normal_noise(streams, [gf.IFO.L1, gf.DetectorSlot("D0", noise_curve=gf.IFO.H1), gf.IFO.H1],
                                        "on", BATCH, 256)
    np.testing.assert_array_equal(beside[:, [0, 2]], alone)
    assert not np.array_equal(beside[:, 1], alone[:, 1])


# --- presence ---------------------------------------------------------------

def test_detector_counts_and_subsets_are_drawn_per_example():
    detectors = gf.Detectors(slots(4), num_present=count_between(1, 3))
    present = np.concatenate([
        np.asarray(detectors.draw_present(64, gf.SeedTree(5).batch("train", 0, k))) for k in range(8)])
    counts = present.sum(axis=1)
    assert set(np.unique(counts)) == {1.0, 2.0, 3.0}
    # No slot is always present, and each is present in about count / D of the examples.
    fraction = present.mean(axis=0)
    assert np.all(fraction > 0.3) and np.all(fraction < 0.7)
    np.testing.assert_array_equal(
        present[:64], detectors.draw_present(64, gf.SeedTree(5).batch("train", 0, 0)))


def test_absent_detectors_carry_no_signal_and_no_snr():
    """The SNR target is the network SNR of the present detectors; an absent one has noise and no signal."""
    ds = dataset(gf.Detectors(slots(3), num_present=count_between(1, 2)), noise_type=gf.NoiseType.COLORED,
                 inputs=(RV.ONSOURCE, RV.DETECTOR_MASK),
                 outputs=(RV.INJECTIONS, RV.OFFSOURCE, gf.ScalingTypes.SNR, RV.INJECTION_MASKS))
    x, y = ds[0]
    mask = np.asarray(x["DETECTOR_MASK"])
    injections = np.asarray(y["INJECTIONS"])[:, 0]
    offsource = np.asarray(y["OFFSOURCE"])
    assert (mask == 0).any() and (mask == 1).any()
    np.testing.assert_array_equal(injections[mask == 0], 0.0)
    assert np.all(np.std(np.asarray(x["ONSOURCE"]), axis=-1) > 0)  # every detector keeps its noise
    for b in np.flatnonzero(np.asarray(y["INJECTION_MASKS"])[:, 0] > 0):
        present = mask[b] > 0
        snr = float(gf.snr(injections[b:b + 1, present], offsource[b:b + 1, present], SAMPLE_RATE)[0])
        np.testing.assert_allclose(snr, float(np.asarray(y["SNR"])[b, 0]), rtol=1e-4)


#: Representation: float32 rounding of the SNR's sum of N squares, twice (scale, measure); N = one onsource window.
SNR_TOLERANCE = representation_tolerance(np.float32, 1.0, 2 * np.sqrt(SAMPLE_RATE * 1.0))


@pytest.mark.parametrize("ifos", [
    [gf.IFO.H1, gf.IFO.L1],
    gf.Detectors([gf.IFO.H1, gf.IFO.L1, gf.IFO.V1]),
    gf.Detectors([gf.IFO.H1, *slots(2)], num_present=count_between(1, 2)),
], ids=["fixed", "Detectors", "Detectors with absent detectors"])
def test_each_detectors_snr_sums_to_the_network_snr_targeted(ifos):
    """
    DETECTOR_SNR, each detector's optimal SNR of the scaled injection by the
    scaling's own measure: over the detectors its root-sum-square is the SNR
    target; an absent detector's is 0; NaN where nothing was injected.
    """
    generators = [wnb(injection_chance=0.6), wnb()]
    ds = dataset(ifos, generators=generators, noise_type=gf.NoiseType.COLORED,
                 outputs=(RV.DETECTOR_SNR, gf.ScalingTypes.SNR, RV.INJECTION_MASKS, RV.DETECTOR_MASK))
    _, y = ds[0]
    each, target = np.asarray(y["DETECTOR_SNR"], np.float64), np.asarray(y["SNR"], np.float64)
    injected = np.asarray(y["INJECTION_MASKS"]) > 0
    mask = np.asarray(y["DETECTOR_MASK"])
    assert each.shape == (BATCH, 2, mask.shape[1])
    assert (~injected).any() and injected.all(axis=0)[1]
    assert np.isnan(each[~injected]).all() and np.isfinite(each[injected]).all()
    absent = np.broadcast_to(mask[:, None] == 0, each.shape) & injected[..., None]
    assert absent.any() == (getattr(ifos, "num_present", None) is not None)
    np.testing.assert_array_equal(each[absent], 0.0)
    assert np.all(each[injected[..., None] & (mask[:, None] > 0)] > 0)
    np.testing.assert_allclose(np.sqrt(np.sum(each[injected] ** 2, axis=-1)), target[injected], rtol=SNR_TOLERANCE)


def test_incoherent_components_reach_only_present_detectors():
    generator = gf.IncoherentGenerator([wnb(), wnb(), wnb()])
    ds = dataset(gf.Detectors(slots(3), num_present=count_between(1, 2)), generators=[generator],
                 inputs=(RV.DETECTOR_MASK,), outputs=(RV.INJECTIONS,))
    x, y = ds[0]
    mask, injections = np.asarray(x["DETECTOR_MASK"]), np.asarray(y["INJECTIONS"])[:, 0]
    np.testing.assert_array_equal(injections[mask == 0], 0.0)
    assert np.all(np.abs(injections[mask == 1]).max(axis=-1) > 0)


def test_the_injection_layer_needs_the_batchs_networks():
    ds = dataset(gf.Detectors(slots(2)))
    injector = gf.InjectionGenerator(waveform_generators=ds.waveform_generators)
    streams = gf.SeedTree(5).direct(0)
    injections, mask, parameters = injector.generate(
        streams, sample_rate_hertz=SAMPLE_RATE, onsource_duration_seconds=1.0,
        crop_duration_seconds=0.5, num_examples_per_batch=BATCH)
    onsource = np.zeros((BATCH, 2, 1024), np.float32)
    with pytest.raises(ValueError, match="networks="):
        injector.add_injections_to_onsource(
            injections, mask, onsource, np.ones((BATCH, 2, 2048), np.float32), None,
            onsource_duration_seconds=1.0, injection_parameters=parameters, streams=streams)


# --- fixed networks ---------------------------------------------------------

def test_a_fixed_networks_geometry_is_its_own_in_every_example():
    ds = dataset([gf.IFO.H1, gf.IFO.L1], inputs=(RV.ONSOURCE, RV.DETECTOR_GEOMETRY, RV.DETECTOR_MASK),
                 outputs=(RV.INJECTIONS,))
    assert not ds.detectors.varies
    network = gf.Network([gf.IFO.H1, gf.IFO.L1])
    location = np.asarray(network.location, np.float64) / WGS84_EQUATORIAL_RADIUS_METERS
    response = np.asarray(network.response)
    expected = np.concatenate([location, np.stack(
        [response[:, i, j] for i, j in ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))], axis=-1)], axis=-1)
    for k in (0, 1):
        x, _ = ds[k]
        np.testing.assert_allclose(x["DETECTOR_GEOMETRY"], np.broadcast_to(expected, (BATCH, 2, 9)), rtol=0, atol=1e-7)
        np.testing.assert_array_equal(x["DETECTOR_MASK"], 1.0)
    # Raw windows (the validator's) get the same constant geometry.
    onsource = np.zeros((3, 2, ds.window_spec.num_onsource_samples), np.float32)
    offsource = np.random.default_rng(1).normal(size=(3, 2, ds.window_spec.num_offsource_samples)).astype(np.float32)
    inputs = ds.model_inputs(onsource, offsource)
    np.testing.assert_array_equal(inputs["DETECTOR_GEOMETRY"], np.asarray(ds[0][0]["DETECTOR_GEOMETRY"])[:3])
    np.testing.assert_array_equal(inputs["DETECTOR_MASK"], 1.0)


def test_a_fixed_network_projects_as_it_did():
    """Bundled detectors all present keep the fixed Network and its projection; the geometry outputs change no batch."""
    plain = dataset([gf.IFO.H1, gf.IFO.L1])
    detectors = dataset(gf.Detectors([gf.IFO.H1, gf.IFO.L1]))
    with_geometry = dataset([gf.IFO.H1, gf.IFO.L1], inputs=(RV.ONSOURCE, RV.DETECTOR_GEOMETRY))
    assert isinstance(plain.waveform_generators[0].network, gf.Network)
    assert isinstance(detectors.waveform_generators[0].network, gf.Network)
    for other in (detectors, with_geometry):
        np.testing.assert_array_equal(other[0][0]["ONSOURCE"], plain[0][0]["ONSOURCE"])
        np.testing.assert_array_equal(other[0][1]["INJECTIONS"], plain[0][1]["INJECTIONS"])


def test_raw_windows_have_no_drawn_network():
    ds = dataset(gf.Detectors(slots(2)), inputs=(RV.ONSOURCE, RV.DETECTOR_GEOMETRY))
    with pytest.raises(ValueError, match="no network of their own"):
        ds.model_inputs(np.zeros((1, 2, ds.window_spec.num_onsource_samples), np.float32),
                        np.zeros((1, 2, ds.window_spec.num_offsource_samples), np.float32))


def test_a_generator_with_geometry_of_its_own_gives_no_geometry_output():
    own = gf.Network({"latitude_radians": [0.1, 0.2], "longitude_radians": [0.3, 0.4], "y_angle_radians": [0.0, 1.0],
                      "height_meters": [0.0, 0.0], "x_length_meters": [4e3, 4e3], "y_length_meters": [4e3, 4e3]})
    with pytest.raises(ValueError, match="geometry of its own"):
        dataset([gf.IFO.H1, gf.IFO.L1], generators=[wnb(network=own)], inputs=(RV.DETECTOR_GEOMETRY,))



@pytest.mark.parametrize("detectors,other", [
    (gf.Detectors([gf.IFO.H1, gf.IFO.L1], num_present=gf.Distribution(value=2, type_=gf.DistributionType.CONSTANT)),
     gf.Detectors([gf.IFO.H1, gf.IFO.L1])),
    (gf.Detectors([gf.DetectorSlot("A", noise_curve=gf.IFO.H1), gf.DetectorSlot("B", noise_curve=gf.IFO.H1)]),
     gf.Detectors([gf.DetectorSlot("A", noise_curve=gf.IFO.H1, height_meters=100.0),
                   gf.DetectorSlot("B", noise_curve=gf.IFO.H1)])),
], ids=["num_present", "slots"])
def test_a_generator_with_geometry_of_its_own_is_refused_where_examples_draw_their_networks(detectors, other):
    """Each example is projected onto the network it draws, so a generator's own geometry would be dropped."""
    own = gf.Network({"latitude_radians": [0.1, -0.7], "longitude_radians": [0.3, 2.0], "y_angle_radians": [0.2, 1.1],
                      "height_meters": [0.0, 0.0], "x_length_meters": [4e3, 4e3], "y_length_meters": [4e3, 4e3]})
    with pytest.raises(ValueError, match="WNBGenerator's network is geometry of its own"):
        dataset(detectors, generators=[wnb(network=own)])
    with pytest.raises(ValueError, match="WNBGenerator's network is not the dataset's gf.Detectors"):
        dataset(detectors, generators=[wnb(network=other)])


def test_a_generator_with_the_datasets_own_detectors_draws_their_networks():
    """A generator given the dataset's detectors (its Detectors, or a Network of its bundled detectors) keeps them."""
    detectors = gf.Detectors([gf.IFO.H1, gf.IFO.L1], num_present=count_between(1, 2))
    reference = dataset(detectors)[0][1]["INJECTIONS"]
    for network in (detectors, gf.Network([gf.IFO.H1, gf.IFO.L1])):
        ds = dataset(detectors, generators=[wnb(network=network)])
        assert ds.waveform_generators[0].network is ds.detectors
        np.testing.assert_array_equal(ds[0][1]["INJECTIONS"], reference)

def test_a_composed_dataset_draws_each_examples_network():
    pool = gf.FeaturePool(name="signal", label=1, probability=1.0,
                          noise_obtainer=gf.NoiseObtainer(noise_type=gf.NoiseType.WHITE), injection_generators=[wnb()])
    ds = gf.ComposedDataset(pools=[pool], seed=5, group="train", ifos=gf.Detectors(slots(2), num_present=count_between(1, 2)),
                            sample_rate_hertz=SAMPLE_RATE, onsource_duration_seconds=1.0, offsource_duration_seconds=4.0,
                            num_examples_per_batch=BATCH, steps_per_epoch=2,
                            input_variables=[RV.DETECTOR_GEOMETRY, RV.DETECTOR_MASK], output_variables=[RV.INJECTIONS])
    x, y = ds[0]
    assert x["DETECTOR_GEOMETRY"].shape == (BATCH, 2, 9)
    np.testing.assert_array_equal(np.asarray(y["INJECTIONS"])[:, 0][np.asarray(x["DETECTOR_MASK"]) == 0], 0.0)


def test_a_dataset_of_slots_is_copied_by_its_specification():
    """A worker's copy rebuilds the dataset from its specification, Detectors included, and serves its batches."""
    import pickle
    ds = dataset(gf.Detectors(slots(2), num_present=count_between(1, 2)), inputs=(RV.DETECTOR_GEOMETRY,))
    assert ds._copy_refusal() is None
    copy = pickle.loads(pickle.dumps(ds))
    np.testing.assert_array_equal(copy[1][0]["DETECTOR_GEOMETRY"], ds[1][0]["DETECTOR_GEOMETRY"])


def test_a_change_of_a_bundled_site_changes_a_varying_datasets_fingerprint(monkeypatch):
    """Detectors are fingerprinted by the geometry they project with: their bundled detectors' by content."""
    def fingerprint():
        return dataset(gf.Detectors([gf.IFO.H1, gf.IFO.L1, *slots(1)])).fingerprint

    before = fingerprint()
    assert fingerprint() == before
    monkeypatch.setattr(gf.IFO.H1.value, "height_meters", gf.IFO.H1.value.height_meters + 1.0)
    assert fingerprint() != before
