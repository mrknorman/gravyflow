"""
Amplitude, units and labels of injections.

Units: noise, injections, HRSS and HPEAK targets and labels are physical strain
throughout; SNR is a ratio and has no unit.

The references are independent of gravyflow: lalsimulation's burst measures
and pycbc's ``sigma``.
"""
import numpy as np
import pytest
from keras import ops

import gravyflow as gf
from gravyflow.src.dataset.features import injection as gf_inj

lal = pytest.importorskip("lal")
lalsim = pytest.importorskip("lalsimulation")
pycbc_types = pytest.importorskip("pycbc.types")
pycbc_filter = pytest.importorskip("pycbc.filter")

FS = 2048.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _profile(ifo, profile=None):
    path = ifo.value.optimal_psd_path if profile is None else profile.path(ifo)
    frequencies, asd = np.loadtxt(path, delimiter=",").T
    return frequencies, asd


def _profile_psd(ifo, frequencies, profile=None):
    f, asd = _profile(ifo, profile)
    return np.interp(frequencies, f, asd) ** 2


def _lal_series(values, delta_t):
    series = lal.CreateREAL8TimeSeries(
        "h", lal.LIGOTimeGPS(0), 0.0, delta_t, lal.DimensionlessUnit, len(values)
    )
    series.data.data = np.asarray(values, np.float64)
    return series


def _lal_hpeak(hp, hc):
    """|h+ + i hx| at the peak: lalsimulation's MeasureHPeak returns the complex
    strain there (and its index)."""
    value = lalsim.MeasureHPeak(hp, hc)
    if isinstance(value, (tuple, list)):
        value = value[0]
    return abs(value)


def _lal_measures(polarisations, sample_rate_hertz):
    """lalsimulation's hrss and hpeak of (B, 2, T) polarisations."""
    hrss, hpeak = [], []
    for hp, hc in np.asarray(polarisations, np.float64):
        hp_ = _lal_series(hp, 1.0 / sample_rate_hertz)
        hc_ = _lal_series(hc, 1.0 / sample_rate_hertz)
        hrss.append(lalsim.MeasureHrss(hp_, hc_))
        hpeak.append(_lal_hpeak(hp_, hc_))
    return np.array(hrss), np.array(hpeak)


def _sine_gaussians(num, num_samples, sample_rate_hertz, seed=3):
    """(num, 2, T) elliptical sine-Gaussians with random shapes and amplitudes."""
    rng = np.random.default_rng(seed)
    t = (np.arange(num_samples) - num_samples // 2) / sample_rate_hertz
    out = np.zeros((num, 2, num_samples))
    for b in range(num):
        f0 = rng.uniform(70.0, 500.0)
        tau = rng.uniform(3.0, 30.0) / (np.sqrt(2.0) * np.pi * f0)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        ellipticity = rng.uniform(0.0, 1.0)
        envelope = rng.uniform(0.1, 10.0) * np.exp(-t ** 2 / (2.0 * tau ** 2))
        out[b, 0] = envelope * np.cos(2.0 * np.pi * f0 * t + phase)
        out[b, 1] = ellipticity * envelope * np.sin(2.0 * np.pi * f0 * t + phase)
    return out.astype(np.float32)


def _pycbc_snr(strain, ifo, sample_rate_hertz, low_frequency_cutoff=20.0):
    """pycbc's optimal SNR of one detector's physical strain against its profile."""
    h = pycbc_types.TimeSeries(
        np.asarray(strain, np.float64), delta_t=1.0 / sample_rate_hertz
    )
    num_frequencies = len(h) // 2 + 1
    delta_f = sample_rate_hertz / len(h)
    frequencies = np.arange(num_frequencies) * delta_f
    psd = _profile_psd(ifo, frequencies)
    psd[frequencies < low_frequency_cutoff] = 0.0
    psd = pycbc_types.FrequencySeries(psd, delta_f=delta_f)
    return float(pycbc_filter.sigma(
        h, psd=psd, low_frequency_cutoff=low_frequency_cutoff,
        high_frequency_cutoff=sample_rate_hertz / 2.0 - delta_f
    ))


def _sine_gaussian_generator(scaling_type, value, injection_chance=1.0, **kwargs):
    return gf.SineGaussianGenerator(
        frequency_hertz=200.0,
        quality_factor=9.0,
        amplitude=1.0,
        phase_radians=0.0,
        scaling_method=gf.ScalingMethod(value, scaling_type),
        injection_chance=injection_chance,
        front_padding_duration_seconds=0.0,
        back_padding_duration_seconds=0.0,
        **kwargs
    )


def _constant(value):
    return gf.Distribution(value=value, type_=gf.DistributionType.CONSTANT)


def _batch(generators, outputs, num_examples=8, seed=1234, ifos=(gf.IFO.H1, gf.IFO.L1)):
    if not isinstance(generators, list):
        generators = [generators]
    dataset = gf.GravyflowDataset(
        seed=seed,
        sample_rate_hertz=FS,
        onsource_duration_seconds=1.0,
        offsource_duration_seconds=16.0,
        crop_duration_seconds=0.5,
        noise_obtainer=gf.NoiseObtainer(
            noise_type=gf.NoiseType.COLORED, ifos=list(ifos)
        ),
        waveform_generators=generators,
        num_examples_per_batch=num_examples,
        input_variables=[gf.ReturnVariables.ONSOURCE],
        output_variables=list(outputs),
    )
    x, y = dataset[0]
    return {key: np.asarray(value, np.float64) for key, value in {**x, **y}.items()}


def _whitened_norm(batch):
    whitened = batch["WHITENED_INJECTIONS"][:, 0]  # (B, G, I, T): the first generator
    return np.sqrt(np.sum(whitened ** 2, axis=(-2, -1)))


def _sine_gaussian_shape_ratio(frequency_hertz=200.0, quality_factor=9.0, offset_samples=0.0):
    """hrss / hpeak of gravyflow's linearly polarised sine-Gaussian (LAL's form,
    cos(2 pi f t) exp(-t^2 / tau^2)), sampled with its peak ``offset_samples``
    off the sample grid, in float64."""
    t = (np.arange(-16384, 16384) + offset_samples) / FS
    tau = quality_factor / (frequency_hertz * np.sqrt(2.0) * np.pi)
    h = np.cos(2.0 * np.pi * frequency_hertz * t) * np.exp(-t ** 2 / tau ** 2)
    return np.sqrt(np.sum(h ** 2) / FS) / np.max(np.abs(h))


def _assert_sine_gaussian_shape_ratio(ratio, rtol=1e-3):
    """``ratio`` is the sampled shape's hrss / hpeak. A generated buffer of odd
    length puts the peak, at sample N / 2, half a sample off the grid, which
    lowers the sampled hpeak; both placements are the source's measures."""
    ratio = np.asarray(ratio, dtype=np.float64)
    candidates = [_sine_gaussian_shape_ratio(offset_samples=o) for o in (0.0, 0.5)]
    errors = [np.max(np.abs(ratio / c - 1.0)) for c in candidates]
    assert min(errors) <= rtol, (ratio, candidates)


# ---------------------------------------------------------------------------
# HRSS and HPEAK
# ---------------------------------------------------------------------------

LAL_CASES = [
    # (Q, frequency, eccentricity, phase)
    (3.0, 70.0, 0.0, 0.0),
    (9.0, 235.0, 0.5, 0.7),
    (9.0, 1053.0, 1.0, 0.0),
    (30.0, 153.0, 0.9, 1.3),
    (100.0, 554.0, 0.0, 2.1),
    (8.9, 849.0, 0.5, 0.3),
]


@pytest.mark.parametrize("quality_factor,frequency,eccentricity,phase", LAL_CASES)
def test_hrss_and_hpeak_match_lal(quality_factor, frequency, eccentricity, phase):
    """calculate_hrss and calculate_hpeak agree with lalsimulation's MeasureHrss
    and MeasureHPeak within 1e-3, on lal's own sine-Gaussians, in float32."""
    sample_rate_hertz = 16384.0
    hp, hc = lalsim.SimBurstSineGaussian(
        quality_factor, frequency, 1e-21, eccentricity, phase, 1.0 / sample_rate_hertz
    )
    polarisations = np.stack([hp.data.data, hc.data.data])[None].astype(np.float32)
    hrss_lal = lalsim.MeasureHrss(hp, hc)
    hpeak_lal = _lal_hpeak(hp, hc)

    hrss = np.asarray(gf.calculate_hrss(polarisations, sample_rate_hertz))
    hpeak = np.asarray(gf.calculate_hpeak(polarisations))

    assert hrss.shape == (1,) and hpeak.shape == (1,)
    np.testing.assert_allclose(hrss[0], hrss_lal, rtol=1e-3)
    np.testing.assert_allclose(hpeak[0], hpeak_lal, rtol=1e-3)


def test_scale_to_hrss_gives_physical_hrss_in_float32():
    """Shapes of any amplitude scaled to hrss 1e-21 in float32 measure 1e-21 by lal."""
    polarisations = _sine_gaussians(16, 4096, FS)
    target = np.full((16,), 1e-21, np.float32)
    scaled = np.asarray(gf_inj.scale_to_hrss(polarisations, target, FS))
    assert scaled.dtype == np.float32
    hrss, _ = _lal_measures(scaled, FS)
    np.testing.assert_allclose(hrss, 1e-21, rtol=1e-3)


def test_scale_to_hpeak_gives_physical_hpeak_per_example():
    """A batch of (B, 2, T) polarisations scales to a (B,) hpeak target."""
    polarisations = _sine_gaussians(16, 4096, FS)
    target = np.geomspace(1e-22, 1e-20, 16).astype(np.float32)
    scaled = np.asarray(gf_inj.scale_to_hpeak(polarisations, target))
    assert scaled.shape == polarisations.shape
    _, hpeak = _lal_measures(scaled, FS)
    np.testing.assert_allclose(hpeak, target, rtol=1e-3)
    assert np.asarray(gf.calculate_hpeak(scaled)).shape == (16,)


def test_scaling_a_zero_injection_gives_zero():
    """An absent (all-zero) injection scales to zero, never NaN or inf."""
    polarisations = np.zeros((2, 2, 512), np.float32)
    for scaled in (
        gf_inj.scale_to_hrss(polarisations, np.full((2,), 1e-21, np.float32), FS),
        gf_inj.scale_to_hpeak(polarisations, np.full((2,), 1e-21, np.float32)),
    ):
        assert np.all(np.asarray(scaled) == 0.0)


def test_scaling_method_returns_its_factor():
    """ScalingMethod.scale returns the scaled injection and the factor applied."""
    polarisations = _sine_gaussians(4, 2048, FS)
    target = np.full((4,), 2e-21, np.float32)
    for scaling_type in (gf.ScalingTypes.HRSS, gf.ScalingTypes.HPEAK):
        method = gf.ScalingMethod(_constant(2e-21), scaling_type)
        scaled, factor = method.scale(polarisations, None, target, FS)
        expected = polarisations * np.asarray(factor)[:, None, None]
        np.testing.assert_allclose(
            np.asarray(scaled), expected, rtol=1e-6, atol=1e-6 * np.abs(expected).max()
        )


def test_scale_to_snr_can_return_its_factor():
    """scale_to_snr returns its factor k on request, and h * k has the target SNR."""
    rng = np.random.default_rng(0)
    background = rng.normal(size=(4, 2, int(16 * FS))).astype(np.float32)
    injection = _sine_gaussians(4, int(2 * FS), FS)
    target = np.array([5.0, 8.0, 10.0, 20.0], np.float32)

    scaled, factor = gf.scale_to_snr(
        injection, background, target, FS,
        fft_duration_seconds=1.0, overlap_duration_seconds=0.5,
        return_factor=True,
    )
    expected = injection * np.asarray(factor)[:, None, None]
    np.testing.assert_allclose(
        np.asarray(scaled), expected, rtol=1e-6, atol=1e-6 * np.abs(expected).max()
    )
    snr = gf.snr(scaled, background, FS, fft_duration_seconds=1.0, overlap_duration_seconds=0.5)
    np.testing.assert_allclose(np.asarray(snr), target, rtol=1e-4)


# ---------------------------------------------------------------------------
# Datasets: amplitude and labels
# ---------------------------------------------------------------------------

LABELS = [gf.ScalingTypes.SNR, gf.ScalingTypes.HRSS, gf.ScalingTypes.HPEAK]
WAVES = [
    gf.ReturnVariables.INJECTIONS,
    gf.ReturnVariables.WHITENED_INJECTIONS,
    gf.ReturnVariables.INJECTION_MASKS,
]


def test_hrss_injection_has_its_physical_snr():
    """An hrss 1e-21 sine-Gaussian is injected at its physical size: each
    detector's whitened injection matches pycbc's SNR of that detector's strain
    (INJECTIONS) within 10 %, and the network SNR lies between 5 % and
    sqrt(2) of the optimally oriented SNR of an hrss 1e-21 source."""
    batch = _batch(
        _sine_gaussian_generator(gf.ScalingTypes.HRSS, _constant(1e-21)),
        WAVES + LABELS, num_examples=16,
    )
    injections = batch["INJECTIONS"][:, 0]  # (B, G, I, T): the one generator
    whitened = batch["WHITENED_INJECTIONS"][:, 0]
    ifos = [gf.IFO.H1, gf.IFO.L1]

    expected = np.array([
        [_pycbc_snr(injections[b, i], ifo, FS) for i, ifo in enumerate(ifos)]
        for b in range(len(injections))
    ])
    measured = np.sqrt(np.sum(whitened ** 2, axis=-1))
    ratio = measured / expected
    assert np.all(np.abs(ratio - 1.0) < 0.10), f"whitened/pycbc ratios {ratio}"
    assert abs(np.median(ratio) - 1.0) < 0.05

    t = (np.arange(int(4 * FS)) - int(2 * FS)) / FS
    tau = 9.0 / (200.0 * np.sqrt(2.0) * np.pi)
    h_plus = np.sin(2.0 * np.pi * 200.0 * t) * np.exp(-t ** 2 / (2.0 * tau ** 2))
    h_plus *= 1e-21 / np.sqrt(np.sum(h_plus ** 2) / FS)
    optimal = _pycbc_snr(h_plus, gf.IFO.H1, FS)
    network = np.sqrt(np.sum(expected ** 2, axis=-1))
    assert np.all(network <= np.sqrt(2.0) * optimal * 1.01)
    assert np.all(network >= 0.05 * optimal)

    np.testing.assert_array_equal(batch["HRSS"], np.float32(1e-21))
    np.testing.assert_allclose(batch["SNR"][:, 0], _whitened_norm(batch), rtol=0.05)


def test_hrss_labels_are_the_targets():
    """The HRSS label is the target drawn for each example, exactly."""
    targets = np.geomspace(5e-22, 4e-21, 8).astype(np.float32)
    batch = _batch(
        _sine_gaussian_generator(gf.ScalingTypes.HRSS, targets), [gf.ScalingTypes.HRSS]
    )
    assert batch["HRSS"].shape == (8, 1)  # (B, G)
    np.testing.assert_array_equal(batch["HRSS"][:, 0], targets.astype(np.float64))


def test_snr_label_is_the_target_and_the_whitened_norm_matches():
    """SNR-scaled injections: the label is the target, and the whitened
    injection's norm is within 2 % of it."""
    batch = _batch(
        _sine_gaussian_generator(gf.ScalingTypes.SNR, _constant(10.0)), WAVES + LABELS
    )
    np.testing.assert_array_equal(batch["SNR"], 10.0)
    np.testing.assert_allclose(_whitened_norm(batch), 10.0, rtol=0.02)


def test_hpeak_scaled_dataset():
    """HPEAK scaling runs in a dataset; its label is the target, and the
    derived HRSS is the source's hrss/hpeak ratio times it."""
    batch = _batch(
        _sine_gaussian_generator(gf.ScalingTypes.HPEAK, _constant(1e-21)), WAVES + LABELS
    )
    np.testing.assert_array_equal(batch["HPEAK"], np.float32(1e-21))
    _assert_sine_gaussian_shape_ratio(batch["HRSS"] / batch["HPEAK"])
    np.testing.assert_allclose(batch["SNR"][:, 0], _whitened_norm(batch), rtol=0.05)


def test_derived_labels_agree_across_scaling_types():
    """The labels of the types not applied are the values the applied scaling
    implies: an SNR-scaled set's HRSS, rescaled by an HRSS-scaled set's SNR per
    unit hrss on the same draws, gives back the SNR target."""
    snr_batch = _batch(
        _sine_gaussian_generator(gf.ScalingTypes.SNR, _constant(10.0)), WAVES + LABELS
    )
    hrss_batch = _batch(
        _sine_gaussian_generator(gf.ScalingTypes.HRSS, _constant(1e-21)), WAVES + LABELS
    )
    _assert_sine_gaussian_shape_ratio(snr_batch["HRSS"] / snr_batch["HPEAK"])
    snr_per_hrss = hrss_batch["SNR"] / hrss_batch["HRSS"]
    np.testing.assert_allclose(snr_batch["HRSS"] * snr_per_hrss, 10.0, rtol=0.02)


def test_absent_injections_have_nan_labels():
    """Examples without an injection carry NaN for every scaling label; the
    others carry their labels."""
    batch = _batch(
        _sine_gaussian_generator(gf.ScalingTypes.SNR, _constant(10.0), injection_chance=0.5),
        WAVES + LABELS, num_examples=16,
    )
    mask = batch["INJECTION_MASKS"].reshape(-1)
    assert 0 < mask.sum() < len(mask)
    for name in ("SNR", "HRSS", "HPEAK"):
        label = batch[name].reshape(-1)
        assert np.all(np.isnan(label[mask == 0])), name
        assert np.all(np.isfinite(label[mask == 1])), name
    np.testing.assert_array_equal(batch["SNR"].reshape(-1)[mask == 1], 10.0)


def test_labels_of_two_generators_are_batch_by_generator():
    """With two generators, each label is (B, G), one column per generator."""
    batch = _batch(
        [
            _sine_gaussian_generator(gf.ScalingTypes.SNR, _constant(10.0)),
            _sine_gaussian_generator(gf.ScalingTypes.HRSS, _constant(2e-21)),
        ],
        [gf.ScalingTypes.SNR, gf.ScalingTypes.HRSS],
    )
    assert batch["SNR"].shape == (8, 2)
    assert batch["HRSS"].shape == (8, 2)
    np.testing.assert_array_equal(batch["SNR"][:, 0], 10.0)
    np.testing.assert_array_equal(batch["HRSS"][:, 1], np.float32(2e-21))


@pytest.mark.parametrize("scaling_type", [gf.ScalingTypes.HRSS, gf.ScalingTypes.HPEAK])
def test_injection_is_the_projected_physical_strain(scaling_type):
    """onsource - noise is the projection of the polarisations scaled to the
    physical target, in strain: no other factor reaches the injection."""
    num_examples = 4
    target = np.float32(1e-21)
    generator = _sine_gaussian_generator(scaling_type, _constant(float(target)))
    network = gf.Network([gf.IFO.H1, gf.IFO.L1])
    generator.network = network
    injection_generator = gf.InjectionGenerator([generator], seed=21)
    injections, mask, parameters = next(injection_generator(
        sample_rate_hertz=FS, onsource_duration_seconds=1.0, crop_duration_seconds=0.5,
        num_examples_per_batch=num_examples,
    ))

    sky = {
        "right_ascension": ops.convert_to_tensor([0.3, 1.7, 3.1, 5.0], "float32"),
        "declination": ops.convert_to_tensor([-0.9, 0.1, 0.6, 1.2], "float32"),
        "polarization": ops.convert_to_tensor([0.2, 2.0, 4.1, 5.9], "float32"),
    }
    # The injection layer reads a given sky from these parameters.
    given_sky = {
        gf.WaveformParameters.RIGHT_ASCENSION_RADIANS: sky["right_ascension"],
        gf.WaveformParameters.DECLINATION_RADIANS: sky["declination"],
        gf.WaveformParameters.POLARIZATION_RADIANS: sky["polarization"],
    }
    num_samples = int(2.0 * FS)
    rng = np.random.default_rng(2)
    # Strain-sized noise, so the injection is resolved in float32 beside it.
    noise = (rng.normal(size=(num_examples, 2, num_samples)) * 1e-21).astype(np.float32)
    offsource = (rng.normal(size=(num_examples, 2, int(16 * FS))) * 1e-21).astype(np.float32)

    onsource, _, _ = injection_generator.add_injections_to_onsource(
        injections, mask, noise, offsource, parameters_to_return=[scaling_type],
        onsource_duration_seconds=1.0, injection_parameters={**parameters, **given_sky},
        streams=injection_generator.streams(0),
    )

    raw = np.asarray(injections[0], np.float64)
    if scaling_type == gf.ScalingTypes.HRSS:
        measure = np.sqrt(np.sum(raw ** 2, axis=(1, 2)) / FS)
    else:
        measure = np.max(np.sqrt(np.sum(raw ** 2, axis=1)), axis=-1)
    physical = (raw * (float(target) / measure)[:, None, None]).astype(np.float32)
    projected = np.asarray(network.project_wave(
        physical, sample_rate_hertz=FS, **sky
    ), np.float64)
    start = (projected.shape[-1] - num_samples) // 2
    projected = projected[..., start:start + num_samples]

    injected = np.asarray(onsource, np.float64) - noise
    scale = np.max(np.abs(projected))
    assert scale > 1e-23
    np.testing.assert_allclose(injected / scale, projected / scale, atol=2e-4)


def test_generators_have_no_scale_factor():
    """Waveform generators take no scale factor: injections are scaled to their
    targets by the InjectionGenerator alone."""
    with pytest.raises(TypeError, match="scale_factor"):
        _sine_gaussian_generator(gf.ScalingTypes.SNR, _constant(10.0), scale_factor=3.0)
    with pytest.raises(TypeError, match="scale_factor"):
        gf.CBCGenerator(scale_factor=3.0)
    assert not hasattr(_sine_gaussian_generator(gf.ScalingTypes.SNR, _constant(10.0)), "scale_factor")
