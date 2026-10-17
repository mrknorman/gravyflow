"""
Composition: a ComposedDataset's batches meet the batch contract, with every
label row at its example's position; an IncoherentGenerator is one injection
with a waveform per detector, labelled and masked per example, beside other
generators.
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


# ------------------------------------------------------------- ComposedDataset

DATA = [RV.ONSOURCE, RV.WHITENED_ONSOURCE, RV.OFFSOURCE, RV.INJECTIONS, RV.WHITENED_INJECTIONS,
        RV.START_GPS_TIME]
LABELS = [RV.POOL_LABEL, RV.INJECTION_MASKS, RV.INJECTION_PRESENT, RV.CENTRAL_TIME, ST.SNR, ST.HRSS,
          ST.HPEAK, WP.DURATION_SECONDS, WP.FREQUENCY_HERTZ, WP.RIGHT_ASCENSION_RADIANS, RV.DATA_LABEL,
          RV.SUB_TYPE, RV.GLITCH_TYPE, RV.SOURCE_TYPE, RV.COINCIDENT]


def mixed_pools():
    """Noise (label 0), WNBs (label 1, generator 0) and sine-Gaussians (label 2, generator 1)."""
    return [
        gf.FeaturePool(name="noise", label=0, probability=0.4, noise_obtainer=colored()),
        gf.FeaturePool(name="bursts", label=1, probability=0.3, noise_obtainer=colored(),
                       injection_generators=[wnb()]),
        gf.FeaturePool(name="sines", label=2, probability=0.3, noise_obtainer=colored(),
                       injection_generators=[sine_gaussian()]),
    ]


def documented_shape(key, g=2, i=len(IFOS)):
    shapes = {
        RV.ONSOURCE: (B, i, T), RV.WHITENED_ONSOURCE: (B, i, T), RV.OFFSOURCE: (B, i, T_OFF),
        RV.INJECTIONS: (B, g, i, T), RV.WHITENED_INJECTIONS: (B, g, i, T), RV.START_GPS_TIME: (B, i),
        RV.INJECTION_MASKS: (B, g), RV.INJECTION_PRESENT: (B,), RV.POOL_LABEL: (B,),
        RV.CENTRAL_TIME: (B, g, i), RV.DATA_LABEL: (B, i), RV.SUB_TYPE: (B, i), RV.GLITCH_TYPE: (B, i),
        RV.SOURCE_TYPE: (B, i), RV.COINCIDENT: (B,),
    }
    return shapes.get(key, (B, g))


def test_a_composed_batch_meets_the_batch_contract():
    x, y = composed(mixed_pools(), LABELS, DATA)[0]
    assert set(x) == {key.name for key in DATA} and set(y) == {key.name for key in LABELS}
    for key in DATA + LABELS:
        value = (x if key in DATA else y)[key.name]
        assert tuple(np.shape(value)) == documented_shape(key), key.name
        assert str(value.dtype) == ("float64" if key is RV.START_GPS_TIME else "float32"), key.name
    # Simulated noise has no GPS time, and every example here is noise to DATA_LABEL.
    assert np.all(np.isnan(x["START_GPS_TIME"]))
    np.testing.assert_array_equal(y["DATA_LABEL"], 0.0)
    for key in (RV.SUB_TYPE, RV.GLITCH_TYPE, RV.SOURCE_TYPE):
        np.testing.assert_array_equal(y[key.name], -1.0)
    # Simulated noise reads no detectors at one time.
    np.testing.assert_array_equal(y["COINCIDENT"], 0.0)


def test_every_label_row_is_its_examples():
    x, y = composed(mixed_pools(), LABELS, DATA)[0]
    pool = np.asarray(y["POOL_LABEL"])
    assert set(pool.tolist()) == {0.0, 1.0, 2.0}, "the batch must hold every pool"
    masks = np.asarray(y["INJECTION_MASKS"])
    # Generator 0 is the bursts pool's, generator 1 the sines pool's: each injects
    # into its own pool's examples and no other.
    np.testing.assert_array_equal(masks[:, 0], pool == 1.0)
    np.testing.assert_array_equal(masks[:, 1], pool == 2.0)
    np.testing.assert_array_equal(y["INJECTION_PRESENT"], pool > 0.0)
    injections = np.asarray(x["INJECTIONS"])
    for g in range(2):
        present = masks[:, g] > 0
        assert np.all(injections[~present, g] == 0.0)
        assert np.all(np.any(injections[present, g] != 0.0, axis=(-2, -1)))
        for key in (ST.SNR, ST.HRSS, ST.HPEAK, WP.RIGHT_ASCENSION_RADIANS, RV.CENTRAL_TIME):
            values = np.asarray(y[key.name])[:, g].reshape(B, -1)
            assert np.all(np.isnan(values[~present])), (key.name, g)
            assert np.all(np.isfinite(values[present])), (key.name, g)
    np.testing.assert_allclose(np.asarray(y["SNR"])[masks[:, 1] > 0, 1], 12.0, rtol=1e-5)
    # A label only one pool's generator draws is NaN for the other's.
    assert np.all(np.isfinite(np.asarray(y["DURATION_SECONDS"])[masks[:, 0] > 0, 0]))
    assert np.all(np.isnan(np.asarray(y["DURATION_SECONDS"])[:, 1]))


@pytest.mark.parametrize("type_, value, noise", [(ST.SNR, 12.0, "white"), (ST.HRSS, 2e-22, "colored"),
                                                 (ST.HPEAK, 4e-22, "colored")])
def test_one_pool_serves_what_a_plain_dataset_of_its_streams_serves(type_, value, noise):
    """A pool is a plain dataset scoped to pool/{name}: noise, injection, scale and labels alike."""
    def obtainer():
        if noise == "white":
            return gf.NoiseObtainer(noise_type=gf.NoiseType.WHITE, ifos=IFOS)
        return colored()

    outputs = [ST.SNR, ST.HRSS, ST.HPEAK, RV.INJECTION_MASKS, WP.FREQUENCY_HERTZ, WP.AMPLITUDE,
               RV.CENTRAL_TIME]
    inputs = [RV.ONSOURCE, RV.WHITENED_ONSOURCE, RV.INJECTIONS, RV.WHITENED_INJECTIONS, RV.OFFSOURCE]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        pool = gf.FeaturePool(name="signal", label=1, probability=1.0, noise_obtainer=obtainer(),
                              injection_generators=[sine_gaussian(type_, value)])
        pooled = composed([pool], outputs, inputs)
        alone = gf.GravyflowDataset(
            seed=gf.SeedTree(7).sub("pool/signal"), group="train", noise_obtainer=obtainer(),
            waveform_generators=[sine_gaussian(type_, value)], input_variables=inputs,
            output_variables=outputs, **WINDOW,
        )
        for k in range(2):
            (x_pooled, y_pooled), (x_alone, y_alone) = pooled[k], alone[k]
            assert set(x_pooled) == set(x_alone) and set(y_pooled) == set(y_alone)
            for name in x_alone:
                np.testing.assert_array_equal(x_pooled[name], x_alone[name], err_msg=name)
            for name in y_alone:
                np.testing.assert_array_equal(y_pooled[name], y_alone[name], err_msg=name)
    np.testing.assert_allclose(np.asarray(y_pooled[type_.name]), value, rtol=1e-5)


def test_a_scale_factor_is_ignored():
    pool = gf.FeaturePool(name="signal", label=1, probability=1.0, noise_obtainer=colored(),
                          injection_generators=[sine_gaussian(ST.HRSS, 2e-22)])
    outputs = [ST.HRSS, WP.AMPLITUDE]
    reference = composed([pool], outputs, [RV.INJECTIONS])[0]
    with pytest.warns(DeprecationWarning, match="scale_factor"):
        scaled = composed([pool], outputs, [RV.INJECTIONS], scale_factor=1e21)
    for got, want in zip(scaled[0], reference):
        for name in want:
            np.testing.assert_array_equal(got[name], want[name], err_msg=name)


def test_a_pools_target_list_is_read_by_example_across_batches():
    # A new InjectionGenerator every batch restarted the list at index 0, and
    # its cursor dropped the list's tail when it wrapped.
    targets = 10.0 + np.arange(11.0)  # not a multiple of the batch
    pool = gf.FeaturePool(name="sweep", label=1, probability=1.0, noise_obtainer=colored(),
                          injection_generators=[wnb(snr=targets)])
    ds = composed([pool], [ST.SNR])
    injector = ds.injection_generator
    for k in range(3):
        expected = targets[(k * B + np.arange(B)) % len(targets)]
        np.testing.assert_allclose(np.asarray(ds[k][1]["SNR"])[:, 0], expected, rtol=1e-5)
    assert ds.injection_generator is injector


def test_pools_take_the_datasets_detectors():
    pools = [gf.FeaturePool(name="noise", label=0, probability=1.0,
                            noise_obtainer=gf.NoiseObtainer(noise_type=gf.NoiseType.WHITE, ifos=IFOS)),
             gf.FeaturePool(name="bursts", label=1, probability=1.0,
                            noise_obtainer=gf.NoiseObtainer(noise_type=gf.NoiseType.WHITE, ifos=IFOS),
                            injection_generators=[wnb()])]
    settings = {key: value for key, value in WINDOW.items() if key != "ifos"}
    ds = gf.ComposedDataset(pools=pools, group="train", input_variables=[RV.ONSOURCE],
                            output_variables=[RV.INJECTION_MASKS], **settings)
    assert np.shape(ds[0][0]["ONSOURCE"])[1] == len(IFOS)
    assert [ifo.name for ifo in ds.waveform_generators[0].network.ifos] == [ifo.name for ifo in IFOS]
    mismatched = [gf.FeaturePool(name="h1", label=0, probability=1.0,
                                 noise_obtainer=gf.NoiseObtainer(noise_type=gf.NoiseType.WHITE, ifos=[gf.IFO.H1])),
                  gf.FeaturePool(name="l1", label=1, probability=1.0,
                                 noise_obtainer=gf.NoiseObtainer(noise_type=gf.NoiseType.WHITE, ifos=[gf.IFO.L1]))]
    with pytest.raises(ValueError, match="detectors"):
        gf.ComposedDataset(pools=mismatched, group="train", input_variables=[RV.ONSOURCE],
                           output_variables=[RV.POOL_LABEL], **settings)


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


def test_a_composed_pool_takes_an_incoherent_generator():
    pools = [gf.FeaturePool(name="noise", label=0, probability=0.5, noise_obtainer=colored()),
             gf.FeaturePool(name="incoherent", label=1, probability=0.5, noise_obtainer=colored(),
                            injection_generators=[incoherent(wnb(), wnb())])]
    x, y = composed(pools, [RV.POOL_LABEL, RV.INJECTION_MASKS, ST.SNR], [RV.INJECTIONS])[0]
    assert np.shape(x["INJECTIONS"]) == (B, 1, len(IFOS), T)
    np.testing.assert_array_equal(np.asarray(y["INJECTION_MASKS"])[:, 0], np.asarray(y["POOL_LABEL"]) == 1.0)
