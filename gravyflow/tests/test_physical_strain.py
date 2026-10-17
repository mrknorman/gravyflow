"""
The data path works in physical strain, and is scale free: whitened outputs and
SNRs are the same, to float32 rounding, whether the strain is given as it is or
multiplied by the internal scale that earlier releases carried (OLD_SCALE).

The square of strain (~1e-42) is below float32's smallest normal number
(1.2e-38), so every step that squares data normalises it first (see
gravyflow.src.dataset.tools.psd). Two scales test that:

- EXACT_SCALE, a power of two near OLD_SCALE: multiplying by it is exact in
  float32, so the scaled input is the same signal, and every output must agree
  to a few float32 roundings, PSDs on the estimator's floor included.
- OLD_SCALE itself: x * 1e21 rounds each sample (relative 6e-8 of the sample),
  so the scaled input differs from x by that rounding, which whitening and
  PSD weighting amplify in quiet bands. Outputs agree to INPUT_ROUNDING.
"""
import numpy as np
import pytest

import gravyflow as gf
from gravyflow.src.dataset.noise import noise as gf_noise

#: The internal scale of earlier releases: strain x OLD_SCALE was O(1).
OLD_SCALE = np.float32(1e21)
#: A power of two near it: scaling by it is exact in float32.
EXACT_SCALE = np.float32(2.0 ** 70)
#: The relative difference x * OLD_SCALE's own rounding makes on coloured noise.
INPUT_ROUNDING = 1e-5
FS = 2048.0
RV = gf.ReturnVariables
ULP = float(np.finfo(np.float32).eps)


def _close(a, b, tolerance=8 * ULP):
    """a and b agree to ``tolerance`` of their largest value."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    assert a.shape == b.shape
    assert np.all(np.isfinite(a)) and np.all(np.isfinite(b))
    peak = np.max(np.abs(a))
    assert peak > 0
    error = np.max(np.abs(a - b)) / peak
    assert error <= tolerance, f"relative difference {error:.3g} > {tolerance:.3g}"


def _coloured_strain(seed, shape, seconds, floor_above_hertz=None):
    """Gaussian noise with the O3 curves, in strain; with ``floor_above_hertz``,
    almost no power above it (1e-12 of the curve), so the PSD estimate sits on
    its floor there."""
    batch = gf_noise.colored_noise_batcher(
        shape[0], seconds, 0.0, seconds, [gf.IFO.H1, gf.IFO.L1][:shape[1]], FS
    )
    x = np.asarray(batch(gf.SeedTree(seed).direct(0))[RV.ONSOURCE], np.float64)
    if floor_above_hertz is not None:
        spectrum = np.fft.rfft(x, axis=-1)
        frequencies = np.fft.rfftfreq(x.shape[-1], 1.0 / FS)
        spectrum[..., frequencies > floor_above_hertz] *= 1e-6
        x = np.fft.irfft(spectrum, n=x.shape[-1], axis=-1)
    return x.astype(np.float32)


CASES = [
    pytest.param((None, EXACT_SCALE, 8 * ULP), id="coloured-exact"),
    pytest.param((700.0, EXACT_SCALE, 8 * ULP), id="on-the-floor-exact"),
    pytest.param((None, OLD_SCALE, INPUT_ROUNDING), id="coloured-1e21"),
]


@pytest.fixture(params=CASES)
def case(request):
    floor_above_hertz, scale, tolerance = request.param
    offsource = _coloured_strain(26, (4, 2), 16.0, floor_above_hertz)
    onsource = _coloured_strain(27, (4, 2), 2.0, floor_above_hertz)
    return onsource, offsource, scale, tolerance, floor_above_hertz


def test_noise_is_strain_sized(case):
    onsource, offsource, *_ = case
    for x in (onsource, offsource):
        std = float(np.std(x, dtype=np.float64))
        assert 1e-24 < std < 1e-17, std


def test_the_spectrum_is_scale_free(case):
    _, offsource, scale, tolerance, floor_above_hertz = case
    physical = gf.estimate_psd(offsource, FS)
    scaled = gf.estimate_psd(offsource * scale, FS)
    _close(physical.normalised, scaled.normalised, tolerance)
    _close(np.asarray(physical.scale) * np.float64(scale), scaled.scale, tolerance)
    normalised = np.asarray(physical.normalised)
    assert np.all(normalised > 0)
    if floor_above_hertz is not None:
        # The quiet band is held at the floor: many bins share the row's minimum.
        assert np.all(np.sum(normalised == normalised.min(axis=-1, keepdims=True), axis=-1) > 100)


def test_whitening_is_identical_with_and_without_the_old_scale(case):
    onsource, offsource, scale, tolerance, _ = case
    physical = gf.whiten(onsource, gf.estimate_psd(offsource, FS), FS)
    scaled = gf.whiten(onsource * scale, gf.estimate_psd(offsource * scale, FS), FS)
    _close(physical, scaled, tolerance)
    # And from the timeseries, as the obtainers whiten (with whiten's own defaults).
    _close(gf.whiten(onsource, offsource, FS), gf.whiten(onsource * scale, offsource * scale, FS), tolerance)
    std = float(np.std(np.asarray(physical)[..., 1024:-1024]))
    assert 0.3 < std < 2.0, std


def test_snr_is_identical_with_and_without_the_old_scale(case):
    _, offsource, scale, tolerance, _ = case
    generator = gf.SineGaussianGenerator(frequency_hertz=150.0, quality_factor=9.0, amplitude=1e-21)
    streams = gf.SeedTree(2).direct(0).sub(f"wave/{generator.stream_label}")
    polarisations, _ = generator.generate(4, FS, 2.0, streams)
    injection = np.repeat(np.asarray(polarisations)[:, :1], 2, axis=1)  # (B, I, T), strain
    spectrum = gf.estimate_psd(offsource, FS)
    scaled_spectrum = gf.estimate_psd(offsource * scale, FS)

    physical = np.asarray(gf.snr(injection, spectrum, FS))
    assert np.all(physical > 0.1), physical
    _close(physical, gf.snr(injection * scale, scaled_spectrum, FS), tolerance)

    scaled_to_ten, factor = gf.scale_to_snr(injection, spectrum, 10.0, FS, return_factor=True)
    old, old_factor = gf.scale_to_snr(injection * scale, scaled_spectrum, 10.0, FS, return_factor=True)
    _close(scaled_to_ten, np.asarray(old, np.float64) / np.float64(scale), tolerance)
    _close(factor, old_factor, tolerance)
    _close(gf.snr(scaled_to_ten, spectrum, FS), np.full(4, 10.0), 8 * ULP)
    _close(gf.whiten(scaled_to_ten, spectrum, FS), gf.whiten(old, scaled_spectrum, FS), tolerance)


def test_matched_filter_is_identical_with_and_without_the_old_scale(case):
    onsource, offsource, scale, tolerance, _ = case
    spectrum = gf.estimate_psd(offsource[:, 0], FS)
    scaled_spectrum = gf.estimate_psd(offsource[:, 0] * scale, FS)
    t = (np.arange(int(FS)) - FS / 2) / FS
    templates = (np.sin(2 * np.pi * 120.0 * t) * np.exp(-(t / 0.05) ** 2))[None].astype(np.float32) * 1e-21
    data = onsource[:, 0, :int(FS)]
    physical = gf.matched_filter_fft(data, templates, spectrum, FS)
    scaled = gf.matched_filter_fft(data * scale, templates, scaled_spectrum, FS)
    _close(physical, scaled, tolerance)


def test_dataset_outputs_are_identical_with_and_without_the_old_scale():
    """A dataset whose noise arrives multiplied by the old scale, as its noise
    did, gives the same whitened onsource, whitened injections and SNR labels,
    and its raw onsource is the strain times that scale."""
    def dataset():
        return gf.GravyflowDataset(
            seed=26, sample_rate_hertz=FS, onsource_duration_seconds=1.0,
            offsource_duration_seconds=16.0, crop_duration_seconds=0.5,
            num_examples_per_batch=4, ifos=[gf.IFO.H1, gf.IFO.L1],
            noise_obtainer=gf.NoiseObtainer(noise_type=gf.NoiseType.COLORED, ifos=[gf.IFO.H1, gf.IFO.L1]),
            waveform_generators=gf.SineGaussianGenerator(
                frequency_hertz=200.0, quality_factor=9.0,
                scaling_method=gf.ScalingMethod(
                    gf.Distribution(value=12.0, type_=gf.DistributionType.CONSTANT), gf.ScalingTypes.SNR),
                front_padding_duration_seconds=0.0, back_padding_duration_seconds=0.0,
            ),
            input_variables=[RV.ONSOURCE, RV.WHITENED_ONSOURCE, RV.WHITENED_INJECTIONS],
            output_variables=[gf.ScalingTypes.SNR],
        )

    physical = dataset()
    old = dataset()
    serve = old.noise

    def scaled_noise(streams):
        batch = dict(serve(streams))
        for key in (RV.ONSOURCE, RV.OFFSOURCE):
            batch[key] = np.asarray(batch[key]) * OLD_SCALE
        return batch

    old.noise = scaled_noise
    (x, y), (x_old, y_old) = physical[0], old[0]
    raw = np.asarray(x[RV.ONSOURCE.name], np.float64)
    assert 1e-24 < np.std(raw) < 1e-17
    _close(raw * np.float64(OLD_SCALE), x_old[RV.ONSOURCE.name])
    _close(x[RV.WHITENED_ONSOURCE.name], x_old[RV.WHITENED_ONSOURCE.name], INPUT_ROUNDING)
    _close(x[RV.WHITENED_INJECTIONS.name], x_old[RV.WHITENED_INJECTIONS.name], INPUT_ROUNDING)
    _close(y["SNR"], y_old["SNR"])
    np.testing.assert_allclose(np.asarray(y["SNR"]), 12.0, rtol=1e-6)


def test_scale_factor_arguments_are_deprecated_and_ignored():
    with pytest.warns(DeprecationWarning, match="scale_factor is deprecated"):
        gf.InjectionGenerator([], scale_factor=3.0)
    with pytest.warns(DeprecationWarning, match="return_scale_factor"):
        gf.scale_to_snr(np.ones((1, 1, 4096), np.float32), np.random.default_rng(0).normal(
            size=(1, 1, 8192)).astype(np.float32), 5.0, FS, return_scale_factor=True)
    with pytest.raises(AttributeError, match="physical strain"):
        gf.Defaults.set(scale_factor=3.0)
    assert not hasattr(gf.Defaults, "scale_factor")
