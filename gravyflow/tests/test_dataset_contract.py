"""
The dataset's preprocessing contract: detector identity, window validation,
one PSD per batch, float32 outputs and finite iteration.
"""
import itertools

import numpy as np
import pytest

import gravyflow as gf
from gravyflow.src.dataset.tools import psd as psd_module

FS = 2048.0


def sine_gaussian_generator(frequency=150.0, network=None, snr=20.0):
    return gf.SineGaussianGenerator(
        scaling_method=gf.ScalingMethod(
            value=gf.Distribution(value=snr, type_=gf.DistributionType.CONSTANT),
            type_=gf.ScalingTypes.SNR,
        ),
        frequency_hertz=gf.Distribution(value=frequency, type_=gf.DistributionType.CONSTANT),
        quality_factor=gf.Distribution(value=9.0, type_=gf.DistributionType.CONSTANT),
        injection_chance=1.0,
        front_padding_duration_seconds=0.1,
        back_padding_duration_seconds=0.1,
        network=network,
    )


def dataset(**overrides):
    settings = dict(
        seed=100,
        sample_rate_hertz=FS,
        onsource_duration_seconds=1.0,
        offsource_duration_seconds=16.0,
        crop_duration_seconds=0.5,
        num_examples_per_batch=4,
        noise_obtainer=gf.NoiseObtainer(noise_type=gf.NoiseType.COLORED),
        input_variables=[gf.ReturnVariables.WHITENED_ONSOURCE],
        output_variables=[],
    )
    settings.update(overrides)
    return gf.GravyflowDataset(**settings)


def test_networks_are_built_from_the_datasets_detectors():
    ds = dataset(ifos=[gf.IFO.H1, gf.IFO.L1], waveform_generators=[sine_gaussian_generator()],
                 input_variables=[gf.ReturnVariables.INJECTIONS, gf.ReturnVariables.WHITENED_ONSOURCE])
    assert [ifo.name for ifo in ds.waveform_generators[0].network.ifos] == ["H1", "L1"]
    inputs, _ = ds[0]
    injections = np.asarray(inputs["INJECTIONS"])
    injections = injections[:, 0]  # (B, G, I, T): the one generator
    assert injections.shape[1] == 2 == np.asarray(inputs["WHITENED_ONSOURCE"]).shape[1]
    # Two detectors see different projections, never one broadcast copy.
    assert not np.allclose(injections[:, 0], injections[:, 1])


def test_building_a_dataset_leaves_the_callers_objects_unchanged():
    generator = sine_gaussian_generator()
    obtainer = gf.NoiseObtainer(noise_type=gf.NoiseType.COLORED)
    dataset(noise_obtainer=obtainer, ifos=[gf.IFO.H1, gf.IFO.L1], waveform_generators=[generator])
    assert generator.network is None
    assert [ifo.name for ifo in obtainer.ifos] == ["L1"]
    # The same objects then serve a one-detector dataset.
    single = dataset(noise_obtainer=obtainer, waveform_generators=[generator],
                     input_variables=[gf.ReturnVariables.INJECTIONS])
    inputs, _ = single[0]
    assert np.asarray(inputs["INJECTIONS"]).shape[-2] == 1


def test_a_network_that_does_not_match_the_noise_detectors_is_refused():
    for network in ([gf.IFO.H1, gf.IFO.L1], [gf.IFO.H1]):
        with pytest.raises(ValueError, match="network"):
            dataset(noise_obtainer=gf.NoiseObtainer(noise_type=gf.NoiseType.COLORED, ifos=[gf.IFO.L1]),
                    waveform_generators=[sine_gaussian_generator(network=network)])
    with pytest.raises(ValueError, match="network"):
        dataset(ifos=[gf.IFO.L1, gf.IFO.H1],
                waveform_generators=[sine_gaussian_generator(network=[gf.IFO.H1, gf.IFO.L1])])
    # The matching network is accepted.
    dataset(ifos=[gf.IFO.H1, gf.IFO.L1], waveform_generators=[sine_gaussian_generator(network=[gf.IFO.H1, gf.IFO.L1])])


def test_a_crop_shorter_than_half_the_whitening_filter_is_refused():
    with pytest.raises(ValueError, match="half of the whitening filter"):
        dataset(crop_duration_seconds=0.25)
    # A shorter filter makes the same crop valid; without whitening no crop is needed.
    dataset(crop_duration_seconds=0.25, spectral_config=gf.SpectralConfig(filter_duration_seconds=0.5))
    dataset(crop_duration_seconds=0.1, input_variables=[gf.ReturnVariables.ONSOURCE])


def test_an_offsource_too_short_for_the_psd_is_refused():
    for offsource in (0.5, 1.0, 1.9):
        with pytest.raises(ValueError, match="Welch segment"):
            dataset(offsource_duration_seconds=offsource)
    with pytest.raises(ValueError, match="Welch segment"):
        dataset(offsource_duration_seconds=1.0, input_variables=[gf.ReturnVariables.ONSOURCE],
                waveform_generators=[sine_gaussian_generator()])


def test_a_generator_reaching_below_the_band_warns():
    with pytest.warns(UserWarning, match="below the band"):
        dataset(waveform_generators=[sine_gaussian_generator(frequency=12.0)])


def test_one_psd_is_estimated_per_batch(monkeypatch):
    calls = []
    original = psd_module.estimate_psd

    def counting(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    import gravyflow.src.dataset.dataset as dataset_module
    monkeypatch.setattr(dataset_module, "estimate_psd", counting)
    ds = dataset(waveform_generators=[sine_gaussian_generator()],
                 input_variables=[gf.ReturnVariables.WHITENED_ONSOURCE, gf.ReturnVariables.WHITENED_INJECTIONS])
    ds[0]
    assert len(calls) == 1


def test_whitened_injections_carry_the_target_snr():
    ds = dataset(waveform_generators=[sine_gaussian_generator(snr=10.0)], num_examples_per_batch=8,
                 input_variables=[gf.ReturnVariables.WHITENED_INJECTIONS])
    inputs, _ = ds[0]
    whitened = np.asarray(inputs["WHITENED_INJECTIONS"], dtype=np.float64)
    whitened = whitened[:, 0]  # (B, G, I, T): the one generator
    norm = np.sqrt(np.sum(whitened ** 2, axis=(-1, -2)))
    np.testing.assert_allclose(norm, 10.0, rtol=0.02)


def test_outputs_are_float32():
    ds = dataset(waveform_generators=[sine_gaussian_generator()],
                 input_variables=[gf.ReturnVariables.WHITENED_ONSOURCE, gf.ReturnVariables.ONSOURCE,
                                  gf.ReturnVariables.INJECTIONS, gf.ReturnVariables.WHITENED_INJECTIONS,
                                  gf.ReturnVariables.SPECTROGRAM_ONSOURCE])
    inputs, _ = ds[0]
    for name, value in inputs.items():
        assert np.asarray(value).dtype == np.float32, name


def test_snr_labels_of_hrss_scaled_injections_match_their_whitened_norm():
    generator = sine_gaussian_generator()
    generator.scaling_method = gf.ScalingMethod(
        value=gf.Distribution(value=2e-22, type_=gf.DistributionType.CONSTANT),
        type_=gf.ScalingTypes.HRSS,
    )
    ds = dataset(waveform_generators=[generator], num_examples_per_batch=8,
                 input_variables=[gf.ReturnVariables.WHITENED_INJECTIONS],
                 output_variables=[gf.ScalingTypes.SNR])
    inputs, outputs = ds[0]
    whitened = np.asarray(inputs["WHITENED_INJECTIONS"], dtype=np.float64)
    whitened = whitened[:, 0]  # (B, G, I, T): the one generator
    norm = np.sqrt(np.sum(whitened ** 2, axis=(-1, -2)))
    label = np.ravel(np.asarray(outputs["SNR"]))
    # Every injection is present; a sky near a detector's null gives a small
    # norm (0.56 for one example of this batch's sky draw).
    assert np.all(norm > 0.1)
    np.testing.assert_allclose(label, norm, rtol=0.02)


def test_iteration_stops_after_len_batches():
    ds = dataset(steps_per_epoch=2, input_variables=[gf.ReturnVariables.ONSOURCE])
    assert sum(1 for _ in itertools.islice(iter(ds), 5)) == 2
