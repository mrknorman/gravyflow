"""
Injections: amplitude measures and scaling, the waveform generators' contracts,
and the InjectionGenerator's batches.

The waveforms themselves are checked against lal and lalsimulation elsewhere
(test_cbc.py, test_cbc_lal_parity.py, test_wnb.py, test_waveform_physics.py);
these tests check what the injection layer does with them.
"""
import json
import logging
import warnings
from itertools import islice
from pathlib import Path

import numpy as np
import pytest
from keras import ops

import gravyflow as gf
from gravyflow.src.dataset.features import injection as gf_inj
from gravyflow.tests._helpers import representation_tolerance, statistical_bound

EXAMPLES = Path(gf.tests.PATH / "example_injection_parameters")


def constant(value):
    return gf.Distribution(value=value, type_=gf.DistributionType.CONSTANT)


def uniform(low, high):
    return gf.Distribution(min_=low, max_=high, type_=gf.DistributionType.UNIFORM)


def wnb(**overrides):
    """A WNB of fixed duration and band, cheap at 100 Hz."""
    arguments = dict(
        duration_seconds=constant(0.5),
        min_frequency_hertz=constant(10.0),
        max_frequency_hertz=constant(50.0),
    )
    arguments.update(overrides)
    return gf_inj.WNBGenerator(**arguments)


def streams(seed, label="g"):
    """A generator's streams, as a dataset gives them, from a root of its own."""
    return gf.SeedTree(seed).direct().sub(f"wave/{label}")


# --------------------------------------------------------------------------
# Amplitude measures and scaling
# --------------------------------------------------------------------------

def test_calculate_hrss():
    sample_rate_hertz, num_samples = 128.0, 1024
    t = np.arange(num_samples) / sample_rate_hertz

    # sqrt(sum_t (h+^2 + hx^2) dt): a sine over whole cycles has hrss A sqrt(T / 2),
    # its two polarisations add in quadrature, and physical strain does not underflow.
    amplitude = 3.0e-21
    sine = amplitude * np.sin(2.0 * np.pi * 8.0 * t)
    duration = num_samples / sample_rate_hertz
    cases = {
        "one channel": (sine[None, None], amplitude * np.sqrt(duration / 2.0)),
        "polarisations": (np.stack([sine, 2.0 * sine])[None], amplitude * np.sqrt(5.0 * duration / 2.0)),
        # (Batch, Detectors, 2, Time): the root-sum-square over the detectors too.
        "per detector": (np.stack([np.stack([sine, 0 * sine])] * 3)[None], amplitude * np.sqrt(3.0 * duration / 2.0)),
        "constant": (np.ones((1, 1, num_samples)), np.sqrt(duration)),
    }
    for name, (injection, expected) in cases.items():
        hrss = np.asarray(gf_inj.calculate_hrss(ops.convert_to_tensor(injection, "float32"), sample_rate_hertz))
        assert hrss.shape == (1,), name
        # Representation: a float32 sum of N squares, relative.
        np.testing.assert_allclose(
            hrss, [expected], atol=0,
            rtol=representation_tolerance(np.float32, 1.0, c=np.sqrt(num_samples)), err_msg=name,
        )


def test_calculate_hpeak():
    # max_t sqrt(h+^2 + hx^2), per example; one channel is |h|.
    def hpeak(injection):
        return np.asarray(gf_inj.calculate_hpeak(ops.convert_to_tensor(np.asarray(injection, np.float32))))

    # Representation: a normalisation by the peak, two squares, a root and the
    # scale restored, relative.
    rtol = representation_tolerance(np.float32, 1.0, c=4)
    np.testing.assert_allclose(hpeak([[[1.0, 2.0, 3.0, 2.0, 1.0]]]), [3.0], atol=0, rtol=rtol)
    np.testing.assert_allclose(hpeak([[[1.0, -5.0, 2.0, 1.0]]]), [5.0], atol=0, rtol=rtol)
    # Polarisations 3 and 4 at one time are a peak of 5, larger than either.
    np.testing.assert_allclose(hpeak([[[0.0, 3.0, 4.5], [0.0, 4.0, 0.0]]]), [5.0], atol=0, rtol=rtol)
    # Per detector, the loudest detector's peak; per example, its own.
    per_detector = np.zeros((2, 3, 2, 4))
    per_detector[0, 2, 1, 3] = 8.0
    per_detector[1, 0, 0, 0] = -2.0
    np.testing.assert_allclose(hpeak(per_detector), [8.0, 2.0], atol=0, rtol=rtol)


@pytest.mark.parametrize("target_shape", [(), (2,), (2, 1)], ids=["scalar", "vector", "column"])
@pytest.mark.parametrize("measure", ["hrss", "hpeak"])
def test_scale_to_target(measure, target_shape):
    sample_rate_hertz = 100.0
    injection = np.zeros((3, 2, 100), np.float32)
    injection[0, 0] = np.sin(np.linspace(0.0, 6.0, 100))
    injection[1, 1, 40:60] = -2.0
    # Example 2 is empty: it stays zero, never divided by zero.
    targets = np.array([6.0, 0.5], np.float32)
    if target_shape == ():
        targets = np.float32(6.0)
    elif target_shape == (2, 1):
        targets = targets.reshape(2, 1)
    expected = np.broadcast_to(np.ravel(targets), (2,))

    if measure == "hrss":
        scaled = gf_inj.scale_to_hrss(injection[:2], targets, sample_rate_hertz)
        measured = gf_inj.calculate_hrss(scaled, sample_rate_hertz)
        empty = gf_inj.scale_to_hrss(injection[2:], 6.0, sample_rate_hertz)
    else:
        scaled = gf_inj.scale_to_hpeak(injection[:2], targets)
        measured = gf_inj.calculate_hpeak(scaled)
        empty = gf_inj.scale_to_hpeak(injection[2:], 6.0)

    # Representation: the measure (a sum of 200 squares) taken twice, and a ratio.
    np.testing.assert_allclose(
        measured, expected, atol=0, rtol=representation_tolerance(np.float32, 1.0, c=2.0 * np.sqrt(200))
    )
    np.testing.assert_array_equal(empty, np.zeros_like(injection[2:]))
    # Scaling changes the amplitude, never the shape.
    for index in range(2):
        np.testing.assert_allclose(
            np.asarray(scaled[index]) * np.max(np.abs(injection[index])) / np.max(np.abs(np.asarray(scaled[index]))),
            injection[index], rtol=0, atol=representation_tolerance(np.float32, 2.0, c=4),
        )


@pytest.mark.parametrize("type_", [gf_inj.ScalingTypes.HRSS, gf_inj.ScalingTypes.HPEAK])
def test_scaling_method_scales_to_the_target(type_):
    sample_rate_hertz = 1024.0
    t = np.arange(1024) / sample_rate_hertz
    injection = np.stack([np.sin(2 * np.pi * 30 * t), 0.5 * np.cos(2 * np.pi * 30 * t)])[None].astype(np.float32)
    measure = {
        gf_inj.ScalingTypes.HRSS: lambda x: gf_inj.calculate_hrss(x, sample_rate_hertz),
        gf_inj.ScalingTypes.HPEAK: gf_inj.calculate_hpeak,
    }[type_]
    scaling = gf_inj.ScalingMethod(value=constant(5.0), type_=type_)

    scaled, factor = scaling.scale(injection, None, np.array([5.0], np.float32), sample_rate_hertz=sample_rate_hertz)

    # Representation: a sum of 2048 squares, taken twice, and a ratio.
    rtol = representation_tolerance(np.float32, 1.0, c=2.0 * np.sqrt(2048))
    np.testing.assert_allclose(measure(scaled), [5.0], atol=0, rtol=rtol)
    np.testing.assert_allclose(factor, 5.0 / np.asarray(measure(injection)), atol=0, rtol=rtol)
    np.testing.assert_allclose(scaled, injection * np.asarray(factor)[:, None, None], atol=0, rtol=rtol)


def test_scaling_method_scale_snr():
    sample_rate_hertz = 1024.0
    t = np.arange(1024) / sample_rate_hertz
    signal = np.sin(2 * np.pi * 100 * t)
    injection = np.stack([signal, 0.5 * signal])[None].astype(np.float32)
    # Four seconds of background: the PSD needs three Welch segments.
    background = (0.1 * np.random.default_rng(4).standard_normal((1, 1, 4096))).astype(np.float32)
    scaling = gf_inj.ScalingMethod(value=constant(20.0), type_=gf_inj.ScalingTypes.SNR)

    scaled, factor = scaling.scale(
        injection, background, np.array([20.0], np.float32),
        sample_rate_hertz=sample_rate_hertz, onsource_duration_seconds=1.0,
    )

    assert ops.shape(scaled) == ops.shape(injection)
    # The SNR of what is injected, measured again on the same background, is the
    # target. Representation: two float32 SNRs, each a sum over 1024 samples' bins.
    rtol = representation_tolerance(np.float32, 1.0, c=2.0 * np.sqrt(1024))
    np.testing.assert_allclose(gf.snr(scaled, background, sample_rate_hertz), [20.0], atol=0, rtol=rtol)
    np.testing.assert_allclose(
        factor, 20.0 / np.asarray(gf.snr(injection, background, sample_rate_hertz)), atol=0, rtol=rtol
    )


def test_scaling_method_invalid_type():
    scaling = gf_inj.ScalingMethod(value=constant(1.0), type_=gf_inj.ScalingTypes.SNR)
    scaling.type_ = "INVALID_TYPE"

    with pytest.raises(ValueError, match="Scaling type INVALID_TYPE not recognised"):
        scaling.scale(
            injections=ops.zeros((1, 100)),
            onsource=None,
            scaling_parameters=None,
            sample_rate_hertz=1024.0
        )


# --------------------------------------------------------------------------
# Array helpers
# --------------------------------------------------------------------------

def test_ensure_last_dim_even():
    values = ops.convert_to_tensor(np.arange(10.0).reshape(2, 5))
    # An odd length loses its last sample; an even one is unchanged.
    np.testing.assert_array_equal(gf_inj.ensure_last_dim_even(values), np.arange(10.0).reshape(2, 5)[:, :4])
    even = ops.ones((1, 4))
    assert gf_inj.ensure_last_dim_even(even) is even


def test_generate_mask():
    """1.0 where a value is non-zero and finite, else 0.0."""
    mask = gf_inj.generate_mask(ops.convert_to_tensor(np.array([1.0, 0.0, -2.0, np.nan, 0.5], np.float32)))
    np.testing.assert_array_equal(mask, [1.0, 0.0, 1.0, 0.0, 1.0])
    assert mask.dtype == "float32"


def test_roll_vector_zero_padding():
    """Each row shifts right by its own count, zero-filled, never wrapped."""
    vector = np.array([[1.0, 2.0, 3.0, 4.0, 5.0], [1.0, 2.0, 3.0, 4.0, 5.0]], np.float32)
    rolled = gf_inj.roll_vector_zero_padding(vector, np.array([2, -1]))
    np.testing.assert_array_equal(rolled, [[0.0, 0.0, 1.0, 2.0, 3.0], [2.0, 3.0, 4.0, 5.0, 0.0]])

    # (Batch, Channels, Time): every channel of a row by that row's shift.
    channels = np.stack([vector, 10 * vector], axis=1)
    rolled = gf_inj.roll_vector_zero_padding(channels, np.array([1, 0]))
    np.testing.assert_array_equal(rolled[0], [[0, 1, 2, 3, 4], [0, 10, 20, 30, 40]])
    np.testing.assert_array_equal(rolled[1], channels[1])


def test_batch_injection_parameters():
    """Per-generator parameters stacked generator-first; a key a generator lacks is NaN for it."""
    assert gf_inj.batch_injection_parameters([]) == {}

    first = {gf_inj.WaveformParameters.MASS_1_MSUN: ops.convert_to_tensor([30.0, 31.0])}
    second = {
        gf_inj.WaveformParameters.MASS_1_MSUN: ops.convert_to_tensor([25.0, 26.0]),
        gf_inj.WaveformParameters.MASS_2_MSUN: ops.convert_to_tensor([20.0, 21.0]),
    }
    batched = gf_inj.batch_injection_parameters([first, second])
    assert set(batched) == {gf_inj.WaveformParameters.MASS_1_MSUN, gf_inj.WaveformParameters.MASS_2_MSUN}
    np.testing.assert_array_equal(batched[gf_inj.WaveformParameters.MASS_1_MSUN], [[30.0, 31.0], [25.0, 26.0]])
    np.testing.assert_array_equal(
        batched[gf_inj.WaveformParameters.MASS_2_MSUN], [[np.nan, np.nan], [20.0, 21.0]]
    )


# --------------------------------------------------------------------------
# Enumerations
# --------------------------------------------------------------------------

def test_waveform_parameters_get():
    assert gf_inj.WaveformParameters.get("mass_1_msun") is gf_inj.WaveformParameters.MASS_1_MSUN
    assert gf_inj.WaveformParameters.get("MASS_2_MSUN") is gf_inj.WaveformParameters.MASS_2_MSUN
    with pytest.raises(ValueError, match="invalid_param not found in WaveformParameters"):
        gf_inj.WaveformParameters.get("invalid_param")


def test_scaling_types_get():
    assert gf_inj.ScalingTypes.get("snr") is gf_inj.ScalingTypes.SNR
    assert gf_inj.ScalingTypes.get("HRSS") is gf_inj.ScalingTypes.HRSS
    with pytest.raises(ValueError, match="invalid not found in ScalingTypes"):
        gf_inj.ScalingTypes.get("invalid")


@pytest.mark.parametrize("enumeration", [gf_inj.ReturnVariables, gf_inj.WaveformParameters])
def test_enumerations_order_by_index(enumeration):
    members = list(enumeration)
    by_index = sorted(members, key=lambda member: member.value.index)
    assert sorted(reversed(members)) == by_index
    assert (by_index[0] < by_index[1]) is True
    assert (by_index[1] < by_index[0]) is False


def test_is_not_inherited():
    """True for an attribute the instance or its own class declares."""
    generator = wnb()
    assert gf_inj.is_not_inherited(generator, "duration_seconds") is True
    # A class variable of the base class is neither.
    assert gf_inj.is_not_inherited(generator, "sky_attributes") is False


# --------------------------------------------------------------------------
# WaveformGenerator
# --------------------------------------------------------------------------

def test_waveform_generator_init_network():
    network = gf_inj.WaveformGenerator.init_network([gf.IFO.L1, gf.IFO.H1])
    assert isinstance(network, gf.Network)
    np.testing.assert_array_equal(network.location, gf.Network([gf.IFO.L1, gf.IFO.H1]).location)

    assert gf_inj.WaveformGenerator.init_network(None) is None
    existing = gf.Network([gf.IFO.L1])
    assert gf_inj.WaveformGenerator.init_network(existing) is existing

    with pytest.raises(TypeError, match="Unable to initiate network"):
        gf_inj.WaveformGenerator.init_network("invalid_string")


def test_waveform_generator_init_network_path(tmp_path):
    """A config path's random geometry is drawn from the rng given."""
    path = tmp_path / "network.json"
    path.write_text(json.dumps({
        "num_detectors": 2,
        "latitude_radians": {"type_": "uniform", "min_": 0.0, "max_": 1.0},
        "longitude_radians": 0.0, "x_angle_radians": 0.0, "y_angle_radians": 0.0,
        "x_length_meters": 4000.0, "y_length_meters": 4000.0, "height_meters": 0.0,
    }))
    network = gf_inj.WaveformGenerator.init_network(path, np.random.default_rng(0))
    expected = gf.Network.load(path, rng=np.random.default_rng(0))
    assert network.num_detectors == 2
    np.testing.assert_array_equal(network.location, expected.location)


def test_waveform_generator_copy():
    generator = wnb()
    duplicate = generator.copy()
    assert duplicate is not generator
    assert duplicate.duration_seconds is not generator.duration_seconds
    assert duplicate.duration_seconds == generator.duration_seconds
    assert duplicate.injection_chance == generator.injection_chance


@pytest.mark.parametrize("generator, expected", [
    (lambda: gf_inj.WaveformGenerator(), 0.0),
    (lambda: wnb(), 0.5),
    (lambda: wnb(duration_seconds=2.5), 2.5),
    (lambda: wnb(duration_seconds=uniform(0.3, 0.8)), 0.8),
    # CBC generation is sized from the requested window alone, so a CBC source
    # needs no longer window, whatever its masses and minimum frequency.
    (lambda: gf_inj.CBCGenerator(mass_1_msun=30.0, mass_2_msun=20.0, min_frequency_hertz=20.0), 0.0),
    (lambda: gf_inj.CBCGenerator(
        mass_1_msun=constant(30.0), mass_2_msun=constant(25.0),
        distance_mpc=constant(100.0), min_frequency_hertz=constant(20.0),
    ), 0.0),
], ids=["base", "wnb-constant", "wnb-float", "wnb-uniform", "cbc-float", "cbc-constant"])
def test_get_max_generated_duration(generator, expected):
    duration = generator().get_max_generated_duration()
    assert isinstance(duration, float)
    assert duration == expected


def test_generate_seed_is_deprecated():
    """A generator has no reseed, and generate(seed=) draws the direct streams of that root."""
    generator = wnb()
    assert not hasattr(generator, "reseed")
    with pytest.warns(DeprecationWarning, match="generate"):
        waveforms, _ = generator.generate(2, 100.0, 1.0, seed=42)
    np.testing.assert_array_equal(
        waveforms, generator.generate(2, 100.0, 1.0, gf.SeedTree(42).direct().sub("wave/WNBGenerator"))[0]
    )


def test_waveform_generator_load_example_configs():
    """The example configs load as their type, with their distributions and scaling."""
    burst = gf.WaveformGenerator.load(EXAMPLES / "wnb_parameters.json")
    assert isinstance(burst, gf_inj.WNBGenerator)
    assert (burst.duration_seconds.min_, burst.duration_seconds.max_) == (0.1, 1.0)
    assert burst.duration_seconds.type_ is gf.DistributionType.UNIFORM
    assert (burst.min_frequency_hertz.min_, burst.max_frequency_hertz.max_) == (20.0, 500.0)
    assert burst.scaling_method.type_ is gf_inj.ScalingTypes.SNR
    assert (burst.scaling_method.value.min_, burst.scaling_method.value.max_) == (8.0, 20.0)
    assert burst.injection_chance == 1.0

    binary = gf.WaveformGenerator.load(EXAMPLES / "phenom_d_parameters.json", network=[gf.IFO.L1, gf.IFO.H1])
    assert isinstance(binary, gf_inj.CBCGenerator)
    assert (binary.mass_1_msun.min_, binary.mass_1_msun.max_) == (5, 95)
    assert binary.distance_mpc.value == 1000
    assert binary.injection_chance == 0.5
    assert binary.network.num_detectors == 2


def test_waveform_generator_load_with_scaling_method():
    custom = gf_inj.ScalingMethod(value=constant(15.0), type_=gf_inj.ScalingTypes.HRSS)
    generator = gf.WaveformGenerator.load(EXAMPLES / "phenom_d_parameters.json", scaling_method=custom)
    assert generator.scaling_method is custom


@pytest.mark.parametrize("config, message", [
    ({"type": "WNB", "scaling_distribution": {"value": 1.0, "type_": "constant"}}, "Missing Scaling Type!"),
    ({"type": "INVALID_WAVEFORM", "scaling_distribution": {"value": 1.0, "type_": "constant"},
      "scaling_type": "snr"}, "This waveform type is not implemented"),
], ids=["missing-scaling-type", "unknown-waveform-type"])
def test_waveform_generator_load_refuses(tmp_path, config, message):
    path = tmp_path / "generator.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match=message):
        gf.WaveformGenerator.load(path)


def test_ensure_list_of_floats_refusals():
    generator = gf_inj.WaveformGenerator()
    with pytest.raises(ValueError, match="test_param should contain three elements"):
        generator.ensure_list_of_floats("test_param", [1, 2])
    with pytest.raises(TypeError, match="test_param should be list or tuple"):
        generator.ensure_list_of_floats("test_param", "not_a_list")


def test_ensure_float_refusal():
    with pytest.raises(TypeError, match="test_param should be float or gf.Distribution object"):
        gf_inj.WaveformGenerator().ensure_float("test_param", "string")


@pytest.mark.parametrize("convert, value, message", [
    ("ensure_list_of_floats", gf.Distribution(value=1, type_=gf.DistributionType.CONSTANT, dtype=int),
     "should not have dtype = int"),
    ("ensure_float", gf.Distribution(value=1, type_=gf.DistributionType.CONSTANT, dtype=int),
     "should not have dtype = int"),
    ("ensure_float", 123, "should be float not int"),
], ids=["list-distribution", "float-distribution", "float-int"])
def test_integers_are_converted_with_a_warning(caplog, convert, value, message):
    # The warning is logged through the logging API that is not deprecated.
    with caplog.at_level(logging.WARNING), warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        result = getattr(gf_inj.WaveformGenerator(), convert)("test_param", value)
    assert f"test_param {message}, automatically adjusting." in caplog.text
    if isinstance(value, gf.Distribution):
        assert result.dtype is float and result.value == 1
    else:
        assert isinstance(result, float) and result == 123.0


# --------------------------------------------------------------------------
# The generators' output contract
# --------------------------------------------------------------------------

GENERATOR_CASES = {
    "wnb": (lambda network: wnb(network=network), 100.0, 1.0, {
        gf_inj.WaveformParameters.DURATION_SECONDS: (0.5, 0.5),
        gf_inj.WaveformParameters.MIN_FREQUENCY_HERTZ: (10.0, 10.0),
        gf_inj.WaveformParameters.MAX_FREQUENCY_HERTZ: (50.0, 50.0),
    }),
    # An odd sample count is cut to the even one below it.
    "wnb-odd-length-floats": (lambda network: gf_inj.WNBGenerator(
        duration_seconds=0.5, min_frequency_hertz=20.0, max_frequency_hertz=40.0, network=network
    ), 100.0, 1.01, {
        gf_inj.WaveformParameters.DURATION_SECONDS: (0.5, 0.5),
        gf_inj.WaveformParameters.MIN_FREQUENCY_HERTZ: (20.0, 20.0),
    }),
    "wnb-uniform": (lambda network: wnb(
        duration_seconds=uniform(0.5, 1.0), min_frequency_hertz=uniform(20.0, 50.0),
        max_frequency_hertz=uniform(100.0, 200.0), network=network,
    ), 1024.0, 1.0, {
        gf_inj.WaveformParameters.DURATION_SECONDS: (0.5, 1.0),
        gf_inj.WaveformParameters.MIN_FREQUENCY_HERTZ: (20.0, 50.0),
        gf_inj.WaveformParameters.MAX_FREQUENCY_HERTZ: (100.0, 200.0),
    }),
    "cbc-uniform": (lambda network: gf_inj.CBCGenerator(
        mass_1_msun=uniform(20.0, 40.0), mass_2_msun=uniform(10.0, 30.0),
        distance_mpc=constant(100.0), network=network,
    ), 1024.0, 1.0, {
        gf_inj.WaveformParameters.MASS_1_MSUN: (20.0, 40.0),
        gf_inj.WaveformParameters.MASS_2_MSUN: (10.0, 30.0),
        gf_inj.WaveformParameters.DISTANCE_MPC: (100.0, 100.0),
    }),
    "cbc-floats": (lambda network: gf_inj.CBCGenerator(
        mass_1_msun=30.0, mass_2_msun=20.0, min_frequency_hertz=20.0, network=network,
    ), 2048.0, 2.0, {
        gf_inj.WaveformParameters.MASS_1_MSUN: (30.0, 30.0),
        gf_inj.WaveformParameters.MASS_2_MSUN: (20.0, 20.0),
    }),
}


@pytest.mark.parametrize("network", [None, [gf.IFO.L1]], ids=["no-network", "network"])
@pytest.mark.parametrize("case", list(GENERATOR_CASES))
def test_generator_output_contract(case, network):
    """(Batch, 2, even Time) polarisations, and each drawn parameter, keyed by
    WaveformParameters, one finite value per example inside its support."""
    make, sample_rate_hertz, duration_seconds, supports = GENERATOR_CASES[case]
    num_waveforms = 5
    waveforms, parameters = make(network).generate(
        num_waveforms, sample_rate_hertz, duration_seconds, streams(1)
    )

    num_samples = int(duration_seconds * sample_rate_hertz)
    assert ops.shape(waveforms) == (num_waveforms, 2, num_samples - num_samples % 2)
    assert np.all(np.isfinite(waveforms)) and np.all(np.max(np.abs(waveforms), axis=(1, 2)) > 0)
    assert all(isinstance(key, gf_inj.WaveformParameters) for key in parameters)
    for key, (low, high) in supports.items():
        values = np.asarray(parameters[key])
        assert values.shape == (num_waveforms,), key
        assert np.all((values >= np.float32(low)) & (values <= np.float32(high))), (key, values)
    # The same streams give the same batch.
    again, _ = make(network).generate(num_waveforms, sample_rate_hertz, duration_seconds, streams(1))
    np.testing.assert_array_equal(waveforms, again)


def test_cbc_generator_converts_int_inputs(caplog):
    with caplog.at_level(logging.WARNING):
        generator = gf_inj.CBCGenerator(
            mass_1_msun=gf.Distribution(value=30, type_=gf.DistributionType.CONSTANT, dtype=int),
            mass_2_msun=20,
            distance_mpc=constant(100.0),
        )
    assert "automatically adjusting" in caplog.text
    _, parameters = generator.generate(2, 1024.0, 1.0, streams(2))
    np.testing.assert_array_equal(parameters[gf_inj.WaveformParameters.MASS_1_MSUN], [30.0, 30.0])
    np.testing.assert_array_equal(parameters[gf_inj.WaveformParameters.MASS_2_MSUN], [20.0, 20.0])


def test_wnb_durations_beyond_the_buffer_are_clipped():
    generator = wnb(duration_seconds=uniform(0.1, 2.0))
    with pytest.warns(UserWarning, match="durations above the 1.0 s buffer are clipped to it"):
        waveforms, parameters = generator.generate(64, 100.0, 1.0, streams(3))
    assert ops.shape(waveforms) == (64, 2, 100)
    durations = np.asarray(parameters[gf_inj.WaveformParameters.DURATION_SECONDS])
    # The labels are the realised, clipped durations.
    assert np.max(durations) == 1.0 and np.min(durations) >= np.float32(0.1)


def test_wnb_generator_negative_duration_raises():
    """
    A duration distribution reaching below zero is refused at construction.

    It used to be clamped inside generate(), which rewrote the user's
    distribution for every later call.
    """
    with pytest.raises(ValueError, match="duration_seconds must be positive"):
        wnb(duration_seconds=uniform(-0.5, 1.0))


def test_apply_injection_chance():
    """Each example is kept whole with probability injection_chance, or zeroed whole."""
    num_examples, chance = 4000, 0.3
    generator = gf_inj.WaveformGenerator(injection_chance=chance)
    kept = np.asarray(generator.apply_injection_chance(ops.ones((num_examples, 2, 8)), streams(5)))
    present = kept.reshape(num_examples, -1).max(axis=1)
    np.testing.assert_array_equal(kept, np.broadcast_to(present[:, None, None], kept.shape))
    # Statistical: the kept fraction is binomial, sigma = sqrt(p (1 - p) / n).
    sigma = np.sqrt(chance * (1.0 - chance) / num_examples)
    assert abs(np.mean(present) - chance) < statistical_bound(sigma)
    # The draw is the streams': the same streams keep the same examples.
    np.testing.assert_array_equal(
        kept, generator.apply_injection_chance(ops.ones((num_examples, 2, 8)), streams(5))
    )


def test_apply_injection_chance_always():
    generator = gf_inj.WaveformGenerator(injection_chance=1.0)
    injections = ops.ones((2, 2, 100))
    assert generator.apply_injection_chance(injections, streams(123)) is injections


def test_apply_injection_chance_never():
    generator = gf_inj.WaveformGenerator(injection_chance=0.0)
    result = generator.apply_injection_chance(ops.ones((2, 2, 100)), streams(123))
    np.testing.assert_array_equal(result, np.zeros((2, 2, 100)))


def test_cbc_generator_injection_chance_zero():
    generator = gf_inj.CBCGenerator(
        mass_1_msun=constant(30.0), mass_2_msun=constant(25.0), distance_mpc=constant(100.0),
        injection_chance=0.0,
    )
    waveforms, _ = generator.generate(2, 1024.0, 1.0, streams(42))
    np.testing.assert_array_equal(waveforms, np.zeros((2, 2, 1024)))


# --------------------------------------------------------------------------
# IncoherentGenerator
# --------------------------------------------------------------------------

def test_incoherent_generator():
    """Component j's own waveform in detector j, from the component's own streams."""
    network = [gf.IFO.L1, gf.IFO.H1]
    first, second = wnb(network=network), wnb(duration_seconds=constant(0.4), network=network)
    incoherent = gf_inj.IncoherentGenerator([first, second])
    assert incoherent.injection_chance == first.injection_chance

    batch = streams(42)
    waveforms, parameters = incoherent.generate(3, 100.0, 1.0, batch)
    assert ops.shape(waveforms) == (3, 2, 2, 100)
    for j, component in enumerate((first, second)):
        own, _ = component.generate(3, 100.0, 1.0, batch.sub(f"comp/{j}"))
        np.testing.assert_array_equal(np.asarray(waveforms)[:, j], own)
    # One value per example: the components' common value, NaN where they differ.
    np.testing.assert_array_equal(parameters[gf_inj.WaveformParameters.MIN_FREQUENCY_HERTZ], [10.0] * 3)
    assert np.all(np.isnan(parameters[gf_inj.WaveformParameters.DURATION_SECONDS]))


def test_incoherent_generator_init_mismatch():
    first, second = gf_inj.WNBGenerator(), gf_inj.WNBGenerator()
    first.network = gf.Network([gf.IFO.L1])
    with pytest.raises(ValueError, match="components need a network of as many detectors"):
        gf_inj.IncoherentGenerator([first, second])


def test_incoherent_generator_refuses_components_of_different_shapes():
    """Components whose waveforms differ in shape cannot be one injection."""
    class Fixed(gf_inj.WaveformGenerator):
        def __init__(self, samples):
            super().__init__()
            self.samples = samples

        def generate(self, num_waveforms, sample_rate_hertz, duration_seconds, streams=None, *, seed=None):
            return ops.zeros((num_waveforms, 2, self.samples)), {}

    incoherent = gf_inj.IncoherentGenerator([Fixed(100), Fixed(200)])
    with pytest.raises(ValueError, match="different shapes"):
        incoherent.generate(2, 1024.0, 1.0)


def test_incoherent_generator_has_no_reseed():
    """An IncoherentGenerator holds no random state, so it has nothing to reseed."""
    network = [gf.IFO.L1, gf.IFO.H1]
    incoherent = gf_inj.IncoherentGenerator([wnb(network=network), wnb(network=network)])
    assert not hasattr(incoherent, "reseed")


def test_incoherent_injection_slicing_regression():
    """
    IncoherentGenerator produces waveforms with shape (Batch, Components, Channels, Time);
    InjectionGenerator's slicing once assumed (Batch, Channels, Time) and raised.
    """
    incoherent = gf_inj.IncoherentGenerator([
        wnb(duration_seconds=constant(0.7), min_frequency_hertz=constant(50.0), max_frequency_hertz=constant(100.0)),
        gf_inj.CBCGenerator(
            mass_1_msun=constant(50.0), mass_2_msun=constant(50.0), inclination_radians=constant(0.0)
        ),
    ])
    injections, masks, _ = next(gf_inj.InjectionGenerator(incoherent)(
        num_examples_per_batch=2,
        sample_rate_hertz=1024.0,
        onsource_duration_seconds=1.0,
        crop_duration_seconds=0.5
    ))
    # (Generators, Batch, Components, Polarisations, Time): Time is
    # (onsource + 2 crop) * sample rate.
    assert ops.shape(injections) == (1, 2, 2, 2, 2048)
    # One mask value per example, (Generators, Batch), whatever the waveform's layout.
    assert ops.shape(masks) == (1, 2)


# --------------------------------------------------------------------------
# InjectionGenerator
# --------------------------------------------------------------------------

def test_injection_generator_batches():
    """(G, B, 2, T) injections, a mask that says which examples hold a signal,
    and each generator's parameters, (G, B), from its own streams."""
    generators = [wnb(injection_chance=0.5), wnb(injection_chance=0.5)]
    injector = gf_inj.InjectionGenerator(
        generators, parameters_to_return=[gf_inj.WaveformParameters.DURATION_SECONDS]
    )
    assert injector.labels == ["0:WNBGenerator", "1:WNBGenerator"]
    injections, masks, parameters = injector.generate(
        injector.streams(0), sample_rate_hertz=100.0, onsource_duration_seconds=1.0,
        crop_duration_seconds=0.5, num_examples_per_batch=16,
    )

    assert ops.shape(injections) == (2, 16, 2, 200)
    injections, masks = np.asarray(injections), np.asarray(masks)
    np.testing.assert_array_equal(masks, np.any(injections != 0.0, axis=(2, 3)).astype(np.float32))
    assert 0 < masks.sum() < masks.size
    # Two generators of one configuration draw their own waveforms: their
    # sources, measured before placement, differ.
    source_hrss = np.asarray(parameters[gf_inj.SOURCE_HRSS])
    assert np.all(source_hrss[masks > 0] > 0)
    assert not np.array_equal(source_hrss[0], source_hrss[1])
    np.testing.assert_array_equal(parameters[gf_inj.WaveformParameters.DURATION_SECONDS], np.full((2, 16), 0.5))


def test_injection_generator_iterates_reproducibly():
    """Batch k of the direct iterator is drawn from streams(k): batches differ,
    and a new iterator gives the same batches again."""
    generator = gf.WaveformGenerator.load(EXAMPLES / "phenom_d_parameters.json")
    injector = gf.InjectionGenerator(
        waveform_generators=generator,
        parameters_to_return=[gf.WaveformParameters.MASS_1_MSUN, gf.WaveformParameters.MASS_2_MSUN],
    )
    batches = list(islice(injector(num_examples_per_batch=4), 2))
    again = next(injector(num_examples_per_batch=4))

    for (injections, masks, parameters) in batches:
        assert ops.shape(masks) == (1, 4)
        np.testing.assert_array_equal(masks, np.any(np.asarray(injections) != 0.0, axis=(2, 3)))
        for key in (gf.WaveformParameters.MASS_1_MSUN, gf.WaveformParameters.MASS_2_MSUN):
            values = np.asarray(parameters[key])
            assert values.shape == (1, 4) and np.all((values >= 5.0) & (values <= 95.0))
    assert not np.array_equal(
        batches[0][2][gf.WaveformParameters.MASS_1_MSUN], batches[1][2][gf.WaveformParameters.MASS_1_MSUN]
    )
    np.testing.assert_array_equal(batches[0][0], again[0])
    np.testing.assert_array_equal(
        batches[0][2][gf.WaveformParameters.MASS_1_MSUN], again[2][gf.WaveformParameters.MASS_1_MSUN]
    )


def test_injection_generator_refuses_generators_disagreeing_on_detectors():
    """Per-detector waveforms for 2 and for 3 detectors cannot be stacked."""
    two = [gf.IFO.L1, gf.IFO.H1]
    three = [gf.IFO.L1, gf.IFO.H1, gf.IFO.V1]
    injector = gf_inj.InjectionGenerator([
        gf_inj.IncoherentGenerator([wnb(network=two), wnb(network=two)]),
        gf_inj.IncoherentGenerator([wnb(network=three), wnb(network=three), wnb(network=three)]),
    ])
    with pytest.raises(ValueError, match=r"disagree on the detectors: \[2, 3\]"):
        injector.generate(injector.streams(0), 100.0, 1.0, 0.5, 2)


def test_injection_generator_seed_is_deprecated():
    """The seed tree replaces InjectionGenerator(seed=)."""
    with pytest.warns(DeprecationWarning):
        gf_inj.InjectionGenerator(waveform_generators=[], seed=3)
