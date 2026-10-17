"""
Composition: an IncoherentGenerator is one injection with a waveform per
detector, labelled and masked per example, beside other generators.
"""
import warnings

import jax.numpy as jnp
import numpy as np
import pytest

import gravyflow as gf
from gravyflow.src.dataset.features.injection import calculate_hpeak

RV, WP, ST = gf.ReturnVariables, gf.WaveformParameters, gf.ScalingTypes

FS = 256.0
B = 16
T = int(FS)                # 1 s onsource
T_OFF = int(4 * FS)        # 4 s offsource
IFOS = [gf.IFO.H1, gf.IFO.L1]
WINDOW = dict(
    sample_rate_hertz=FS, onsource_duration_seconds=1.0, offsource_duration_seconds=4.0,
    crop_duration_seconds=0.5, num_examples_per_batch=B, ifos=IFOS, steps_per_epoch=4,
)


def constant(value):
    return gf.Distribution(value=value, type_=gf.DistributionType.CONSTANT)


def uniform(low, high):
    return gf.Distribution(min_=low, max_=high, type_=gf.DistributionType.UNIFORM)


def wnb(snr=None, duration=None, chance=1.0):
    return gf.WNBGenerator(
        scaling_method=gf.ScalingMethod(value=uniform(8.0, 20.0) if snr is None else snr, type_=ST.SNR),
        injection_chance=chance,
        duration_seconds=uniform(0.1, 0.3) if duration is None else duration,
        min_frequency_hertz=uniform(25.0, 40.0), max_frequency_hertz=uniform(60.0, 100.0),
    )


def sine_gaussian(type_=ST.SNR, value=12.0):
    return gf.SineGaussianGenerator(
        scaling_method=gf.ScalingMethod(value=constant(value), type_=type_),
        frequency_hertz=uniform(40.0, 90.0), quality_factor=constant(9.0),
    )


def colored():
    return gf.NoiseObtainer(noise_type=gf.NoiseType.COLORED, ifos=IFOS)


def composed(pools, outputs, inputs=(RV.ONSOURCE,), seed=7, **overrides):
    return gf.ComposedDataset(
        pools=pools, seed=seed, group="train",
        input_variables=list(inputs), output_variables=list(outputs), **{**WINDOW, **overrides},
    )


def plain(generators, outputs, inputs=(RV.ONSOURCE,), seed=7, **overrides):
    return gf.GravyflowDataset(
        seed=seed, group="train", noise_obtainer=colored(), waveform_generators=generators,
        input_variables=list(inputs), output_variables=list(outputs), **{**WINDOW, **overrides},
    )


# --------------------------------------------------------- IncoherentGenerator

def incoherent(*components):
    return gf.IncoherentGenerator(list(components) or [wnb(), wnb()])


def direct_batch(generators, k=0):
    injector = gf.InjectionGenerator(generators)
    return injector.generate(injector.streams(k), sample_rate_hertz=FS, onsource_duration_seconds=1.0,
                             crop_duration_seconds=0.5, num_examples_per_batch=B)


def test_an_incoherent_mask_has_one_value_per_example():
    injections, masks, _ = direct_batch([incoherent(wnb(chance=0.5), wnb(chance=0.5))])
    assert np.shape(masks) == (1, B)
    assert np.shape(injections)[:4] == (1, B, 2, 2)  # (generators, batch, detectors, polarisations, time)
    present = np.any(np.asarray(injections) != 0.0, axis=(2, 3, 4))
    np.testing.assert_array_equal(np.asarray(masks) > 0, present)
    assert 0 < present.sum() < B


def test_incoherent_parameters_describe_the_whole_injection():
    _, same = incoherent(wnb(duration=constant(0.2)), wnb(duration=constant(0.2))).generate(B, FS, 2.0)
    np.testing.assert_allclose(same[WP.DURATION_SECONDS], 0.2)
    # Components that differ have no one value: component 0's is not the injection's.
    _, differ = incoherent(wnb(duration=constant(0.2)), wnb(duration=constant(0.3))).generate(B, FS, 2.0)
    assert np.all(np.isnan(np.asarray(differ[WP.DURATION_SECONDS])))


def test_an_incoherent_generator_is_sized_and_placed_by_its_longest_component():
    generator = incoherent(wnb(duration=constant(0.1)), wnb(duration=constant(0.6)))
    assert generator.get_max_generated_duration() == pytest.approx(0.6)
    _, parameters = generator.generate(B, FS, 2.0)
    before, after = generator.get_signal_extent_seconds(parameters)
    np.testing.assert_allclose(np.broadcast_to(before, (B,)), 0.3)
    np.testing.assert_allclose(np.broadcast_to(after, (B,)), 0.3)


def test_an_incoherent_generator_refuses_a_network_of_another_size():
    generator = incoherent(wnb(), wnb())
    with pytest.raises(ValueError, match="detectors"):
        generator.network = gf.Network([gf.IFO.H1, gf.IFO.L1, gf.IFO.V1])
    with pytest.raises(ValueError, match="detectors"):
        plain([incoherent(wnb(), wnb(), wnb())], [RV.INJECTION_MASKS])


def test_an_incoherent_generator_mixes_with_coherent_ones():
    outputs = [RV.INJECTION_MASKS, ST.SNR, WP.DURATION_SECONDS]
    x, y = plain([wnb(snr=constant(12.0)), incoherent(wnb(snr=constant(15.0)), wnb())], outputs,
                 [RV.ONSOURCE, RV.INJECTIONS])[0]
    assert np.shape(x["INJECTIONS"]) == (B, 2, len(IFOS), T)
    np.testing.assert_array_equal(y["INJECTION_MASKS"], 1.0)
    np.testing.assert_allclose(y["SNR"], np.broadcast_to([12.0, 15.0], (B, 2)), rtol=1e-5)
    # The incoherent signal differs between detectors in shape, not only in amplitude and delay.
    h1, l1 = np.asarray(x["INJECTIONS"])[0, 1]
    assert abs(np.corrcoef(h1, l1)[0, 1]) < 0.5


def test_a_coherent_waveform_repeated_per_detector_projects_as_itself():
    network = gf.Network(IFOS)
    polarisations = np.asarray(direct_batch([wnb()])[0])[0]  # (B, 2, time)
    sky = {name: jnp.full(B, value, jnp.float32)
           for name, value in (("right_ascension", 1.0), ("declination", 0.3), ("polarization", 0.7))}
    repeated = jnp.broadcast_to(polarisations[:, None], (B, len(IFOS)) + polarisations.shape[1:])
    np.testing.assert_allclose(network.project_wave(repeated, **sky), network.project_wave(polarisations, **sky),
                               rtol=1e-6, atol=1e-6 * np.abs(polarisations).max())


def test_hpeak_of_per_detector_polarisations_is_the_loudest_detectors():
    polarisations = np.zeros((1, 2, 2, 8), np.float32)  # (batch, detectors, polarisations, time)
    polarisations[0, 0, :, 3] = (3.0, 4.0)              # detector 0: |h| = 5
    polarisations[0, 1, 0, 5] = 1.0                     # detector 1: |h| = 1
    np.testing.assert_allclose(calculate_hpeak(polarisations), [5.0])
