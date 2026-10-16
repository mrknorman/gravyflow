"""
The batch contract: every output a dataset returns is batch-first, float32
(GPS times float64), in its documented shape, and every per-generator label is
NaN where its generator injected nothing.
"""
import numpy as np
import pytest

import gravyflow as gf
from gravyflow.src.dataset.features.injection import calculate_hpeak

FS = 2048.0
B = 8
T = int(FS)               # 1 s onsource
T_OFF = int(16 * FS)      # 16 s offsource
CONSTANT = gf.DistributionType.CONSTANT
RV, WP, ST = gf.ReturnVariables, gf.WaveformParameters, gf.ScalingTypes


def constant(value):
    return gf.Distribution(value=value, type_=CONSTANT)


def scaling(type_=ST.SNR, value=12.0):
    return gf.ScalingMethod(value=constant(value), type_=type_)


def cbc(chance=1.0, method=None, distance=constant(500.0)):
    return gf.CBCGenerator(
        scaling_method=method or scaling(), injection_chance=chance,
        mass_1_msun=constant(30.0), mass_2_msun=constant(25.0), distance_mpc=distance,
    )


def sine_gaussian(chance=1.0):
    return gf.SineGaussianGenerator(
        scaling_method=scaling(), injection_chance=chance,
        frequency_hertz=gf.Distribution(min_=100.0, max_=300.0, type_=gf.DistributionType.UNIFORM),
        quality_factor=constant(9.0),
    )


def wnb(chance=1.0):
    return gf.WNBGenerator(
        scaling_method=scaling(), injection_chance=chance, duration_seconds=constant(0.2),
        min_frequency_hertz=constant(50.0), max_frequency_hertz=constant(400.0),
    )


DATA = [RV.ONSOURCE, RV.WHITENED_ONSOURCE, RV.OFFSOURCE, RV.INJECTIONS, RV.WHITENED_INJECTIONS,
        RV.START_GPS_TIME]
LABELS = [RV.INJECTION_MASKS, RV.INJECTION_PRESENT, RV.CENTRAL_TIME, ST.SNR, ST.HRSS, ST.HPEAK,
          WP.MASS_1_MSUN, WP.DISTANCE_MPC, WP.SPIN_1_IN, WP.FREQUENCY_HERTZ, WP.DURATION_SECONDS,
          WP.RIGHT_ASCENSION_RADIANS, WP.DECLINATION_RADIANS, WP.POLARIZATION_RADIANS]

# The labels each generator draws itself; a generator that does not draw one has NaN.
DRAWN = {
    gf.CBCGenerator: {WP.MASS_1_MSUN, WP.DISTANCE_MPC, WP.SPIN_1_IN},
    gf.SineGaussianGenerator: {WP.FREQUENCY_HERTZ},
    gf.WNBGenerator: set(),
}
# Labels every injected example has, whichever generator injected it.
ALWAYS = [RV.CENTRAL_TIME, ST.SNR, ST.HRSS, ST.HPEAK, WP.RIGHT_ASCENSION_RADIANS,
          WP.DECLINATION_RADIANS, WP.POLARIZATION_RADIANS]

CONFIGURATIONS = {
    "noise-only": ([], [gf.IFO.L1]),
    "cbc": (lambda: [cbc(0.5)], [gf.IFO.L1]),
    "bursts": (lambda: [sine_gaussian(0.5), wnb(0.5)], [gf.IFO.L1]),
    "multi-generator, multi-ifo": (lambda: [cbc(0.5), sine_gaussian(0.5)], [gf.IFO.H1, gf.IFO.L1]),
    "three detectors": (lambda: [sine_gaussian(0.5)], [gf.IFO.H1, gf.IFO.L1, gf.IFO.V1]),
}


def dataset(generators, ifos, outputs=LABELS, inputs=DATA, seed=7, **overrides):
    settings = dict(
        seed=seed, sample_rate_hertz=FS, onsource_duration_seconds=1.0,
        offsource_duration_seconds=16.0, crop_duration_seconds=0.5, num_examples_per_batch=B,
        noise_obtainer=gf.NoiseObtainer(noise_type=gf.NoiseType.COLORED), ifos=ifos,
        waveform_generators=generators, input_variables=inputs, output_variables=outputs,
    )
    settings.update(overrides)
    return gf.GravyflowDataset(**settings)


def documented_shape(key, g, i):
    shapes = {
        RV.ONSOURCE: (B, i, T), RV.WHITENED_ONSOURCE: (B, i, T), RV.OFFSOURCE: (B, i, T_OFF),
        RV.INJECTIONS: (B, g, i, T), RV.WHITENED_INJECTIONS: (B, g, i, T),
        RV.START_GPS_TIME: (B, i), RV.INJECTION_MASKS: (B, g), RV.INJECTION_PRESENT: (B,),
        RV.CENTRAL_TIME: (B, g, i), WP.SPIN_1_IN: (B, g, 3),
    }
    return shapes.get(key, (B, g))


@pytest.mark.parametrize("name", list(CONFIGURATIONS))
def test_every_output_is_batch_first_float32_in_its_documented_shape(name):
    make, ifos = CONFIGURATIONS[name]
    generators = make() if callable(make) else make
    x, y = dataset(generators, ifos)[0]
    g, i = len(generators), len(ifos)

    assert set(x) == {key.name for key in DATA} and set(y) == {key.name for key in LABELS}
    for key in DATA + LABELS:
        value = (x if key in DATA else y)[key.name]
        assert tuple(np.shape(value)) == documented_shape(key, g, i), key.name
        assert str(value.dtype) == ("float64" if key is RV.START_GPS_TIME else "float32"), key.name

    # Simulated noise has no GPS time.
    assert np.all(np.isnan(x["START_GPS_TIME"]))

    masks = np.asarray(y["INJECTION_MASKS"])
    np.testing.assert_array_equal(y["INJECTION_PRESENT"], np.any(masks > 0, axis=1).astype(np.float32))
    if g:
        assert 0 < masks.sum() < masks.size, "the configuration must hold absent injections"
    else:
        assert not np.any(np.asarray(x["INJECTIONS"]).size)

    for key in [k for k in LABELS if k not in (RV.INJECTION_MASKS, RV.INJECTION_PRESENT)]:
        label = np.asarray(y[key.name])
        for generator_index, generator in enumerate(generators):
            values = label[:, generator_index].reshape(B, -1)
            present = masks[:, generator_index] > 0
            # Absent injections are NaN in every injection-parameter label.
            assert np.all(np.isnan(values[~present])), (name, key.name, generator_index)
            if key in ALWAYS or key in DRAWN[type(generator)]:
                assert np.all(np.isfinite(values[present])), (name, key.name, generator_index)
            elif key is not WP.DURATION_SECONDS:
                assert np.all(np.isnan(values)), (name, key.name, generator_index)


def test_one_generator_keeps_its_generator_axis():
    _, y = dataset([sine_gaussian()], [gf.IFO.L1], outputs=[ST.SNR, RV.INJECTION_MASKS])[0]
    assert np.shape(y["SNR"]) == (B, 1) and np.shape(y["INJECTION_MASKS"]) == (B, 1)
    np.testing.assert_allclose(y["SNR"], 12.0, rtol=1e-6)


def test_signal_present_counts_any_generator():
    """B128: an example whose only injection is from the second generator is a signal."""
    ds = dataset([cbc(0.5), sine_gaussian(0.5)], [gf.IFO.L1], inputs=[RV.WHITENED_ONSOURCE],
                 outputs=[RV.INJECTION_MASKS, RV.INJECTION_PRESENT])
    second_only = 0
    for index in range(3):
        _, y = ds[index]
        masks = np.asarray(y["INJECTION_MASKS"])
        present = np.asarray(y["INJECTION_PRESENT"])
        np.testing.assert_array_equal(present, masks.max(axis=1))
        second_only += int(np.sum((masks[:, 0] == 0) & (masks[:, 1] == 1)))
        assert np.all(present[masks[:, 1] == 1] == 1)
    assert second_only > 0


def test_noise_only_dataset_returns_empty_masks_and_noise_labels():
    """B132: requesting the injection labels of a noise-only dataset does not crash."""
    x, y = dataset([], [gf.IFO.H1, gf.IFO.L1], inputs=[RV.WHITENED_ONSOURCE, RV.INJECTIONS],
                   outputs=[RV.INJECTION_MASKS, RV.INJECTION_PRESENT, ST.SNR])[0]
    assert np.shape(y["INJECTION_MASKS"]) == (B, 0)
    assert np.shape(y["SNR"]) == (B, 0)
    assert np.shape(x["INJECTIONS"]) == (B, 0, 2, T)
    np.testing.assert_array_equal(y["INJECTION_PRESENT"], np.zeros(B, np.float32))


def test_central_time_is_nan_where_absent():
    """B160: CENTRAL_TIME is not set for an injection zeroed by injection_chance."""
    _, y = dataset([cbc(0.5)], [gf.IFO.H1, gf.IFO.L1], inputs=[RV.WHITENED_ONSOURCE],
                   outputs=[RV.INJECTION_MASKS, RV.CENTRAL_TIME])[0]
    absent = np.asarray(y["INJECTION_MASKS"])[:, 0] == 0
    times = np.asarray(y["CENTRAL_TIME"])[:, 0]
    assert absent.any() and (~absent).any()
    assert np.all(np.isnan(times[absent]))
    assert np.all((times[~absent] > 0.0) & (times[~absent] < 1.0))


def _source_hpeak(distance_mpc):
    """The physical hpeak of the test's CBC source generated at ``distance_mpc``."""
    waveforms, _ = cbc(distance=constant(float(distance_mpc))).generate(
        num_waveforms=1, sample_rate_hertz=FS, duration_seconds=4.0, seed=1)
    return float(np.asarray(calculate_hpeak(waveforms))[0])


def test_distance_label_is_the_distance_the_injected_amplitude_implies():
    """B140: under HPEAK scaling, the source at the labelled distance has the injected hpeak."""
    target = 2e-21
    _, y = dataset([cbc(method=scaling(ST.HPEAK, target))], [gf.IFO.L1], inputs=[RV.WHITENED_ONSOURCE],
                   outputs=[WP.DISTANCE_MPC, ST.HPEAK])[0]
    distance = np.asarray(y["DISTANCE_MPC"])[:, 0]
    np.testing.assert_allclose(distance, 500.0 * _source_hpeak(500.0) / target, rtol=1e-4)
    assert _source_hpeak(distance[0]) == pytest.approx(target, rel=1e-4)


def test_distance_label_follows_the_snr_target():
    """B140: the same source scaled to twice the SNR is labelled at half the distance."""
    distances = []
    for snr in (10.0, 20.0):
        _, y = dataset([cbc(method=scaling(ST.SNR, snr))], [gf.IFO.L1], inputs=[RV.WHITENED_ONSOURCE],
                       outputs=[WP.DISTANCE_MPC, ST.HPEAK])[0]
        distances.append(np.asarray(y["DISTANCE_MPC"])[:, 0])
        # The labelled distance and the injected hpeak describe one source.
        np.testing.assert_allclose(_source_hpeak(distances[-1][0]), np.asarray(y["HPEAK"])[0, 0], rtol=1e-4)
    np.testing.assert_allclose(distances[0] / distances[1], 2.0, rtol=1e-4)


AMPLITUDE_IS_HRSS = {
    "sine-Gaussian": lambda method: gf.SineGaussianGenerator(
        scaling_method=method, amplitude=constant(3.0), frequency_hertz=constant(150.0),
        quality_factor=constant(9.0)),
    "chirplet": lambda method: gf.ChirpletGenerator(scaling_method=method, amplitude=constant(3.0)),
    "ringdown": lambda method: gf.RingdownGenerator(scaling_method=method, amplitude=constant(3.0)),
}


@pytest.mark.parametrize("method", [scaling(ST.HRSS, 2e-22), scaling(ST.HPEAK, 1e-21), scaling(ST.SNR, 12.0)],
                         ids=["HRSS", "HPEAK", "SNR"])
@pytest.mark.parametrize("make", AMPLITUDE_IS_HRSS.values(), ids=AMPLITUDE_IS_HRSS.keys())
def test_amplitude_label_is_the_injected_amplitude(make, method):
    """B140's class: a burst's AMPLITUDE (its hrss) is the hrss injected, not the one drawn."""
    _, y = dataset([make(method)], [gf.IFO.L1], inputs=[RV.WHITENED_ONSOURCE],
                   outputs=[WP.AMPLITUDE, ST.HRSS])[0]
    np.testing.assert_allclose(np.asarray(y["AMPLITUDE"]), np.asarray(y["HRSS"]), rtol=1e-4)


def test_the_old_sky_keys_are_refused_by_name():
    injector = gf.InjectionGenerator([cbc()], seed=1)
    injector.waveform_generators[0].network = gf.Network([gf.IFO.L1])
    injections, mask, parameters = next(injector(
        sample_rate_hertz=FS, onsource_duration_seconds=1.0, crop_duration_seconds=0.5,
        num_examples_per_batch=2))
    for old, new in gf.RENAMED_SKY_KEYS.items():
        with pytest.raises(KeyError, match=new.name):
            injector.add_injections_to_onsource(
                injections, mask, np.zeros((2, 1, 2 * T), np.float32),
                np.random.default_rng(0).normal(size=(2, 1, T_OFF)).astype(np.float32),
                parameters_to_return=[], onsource_duration_seconds=1.0,
                injection_parameters={**parameters, old: np.zeros((1, 2), np.float32)})


def test_a_batch_off_its_contract_is_refused_with_its_documented_shape():
    contract = gf.BatchContract(batch=4, generators=2, ifos=1, onsource_samples=16)
    good = {"SNR": np.zeros((4, 2), np.float32), "CENTRAL_TIME": np.zeros((4, 2, 1), np.float32)}
    contract.check([[ST.SNR, RV.CENTRAL_TIME]], good)
    with pytest.raises(gf.BatchContractError, match=r"SNR is \(2, 4\) float32; its contract is \(B=4, G=2\)"):
        contract.check([[ST.SNR]], {"SNR": np.zeros((2, 4), np.float32)})
    with pytest.raises(gf.BatchContractError, match="float64"):
        contract.check([[ST.SNR]], {"SNR": np.zeros((4, 2), np.float64)})
    with pytest.raises(gf.BatchContractError, match="GLITCH_TYPE was requested"):
        contract.check([[RV.GLITCH_TYPE]], {})
