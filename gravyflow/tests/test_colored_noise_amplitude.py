"""
Amplitude of simulated (coloured) noise: its PSD is its profile's, on the
right frequency grid, in internal units (strain x ``gf.Defaults.scale_factor``)
by default, from per-detector profiles that name their source. The reference
is scipy's Welch estimate.
"""
import numpy as np
import pytest
import scipy.signal

import gravyflow as gf
from gravyflow.src.dataset.noise import noise as gf_noise

S = gf.Defaults.scale_factor
FS = 2048.0
ALL_IFOS = [gf.IFO.H1, gf.IFO.L1, gf.IFO.V1]


def _profile(ifo, profile=None):
    path = ifo.value.optimal_psd_path if profile is None else profile.path(ifo)
    frequencies, asd = np.loadtxt(path, delimiter=",").T
    return frequencies, asd


def _profile_psd(ifo, frequencies, profile=None):
    f, asd = _profile(ifo, profile)
    return np.interp(frequencies, f, asd) ** 2


def _welch_ratio(x, ifos, sample_rate_hertz, profile=None):
    """Per-example Welch PSD divided by the profile PSD: (B, IFO, F), and F."""
    f, p = scipy.signal.welch(
        np.asarray(x, np.float64), fs=sample_rate_hertz,
        nperseg=int(4 * sample_rate_hertz)
    )
    reference = np.stack([_profile_psd(ifo, f, profile) for ifo in ifos])
    with np.errstate(divide="ignore", invalid="ignore"):
        return p / reference[None], f


# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sample_rate_hertz", [2048.0, 16384.0])
def test_colored_noise_psd_matches_its_profile(sample_rate_hertz):
    """The coloured noise has its profile's PSD: median within 1 % over 30-900 Hz
    and every bin within 5 standard errors, for every detector. Each bin
    averages over 7 half-overlapping Hann segments of each example, at least
    112 in all, so the median's own bias is under 0.4 %; a bin's standard error
    is sqrt(1.056 / segments), 1.056 being the variance excess of Welch's
    estimate for 50 %-overlapped Hann windows."""
    duration = 16.0
    num_examples = 32 if sample_rate_hertz == 2048.0 else 16
    batch = gf_noise.colored_noise_batcher(
        num_examples_per_batch=num_examples,
        onsource_duration_seconds=duration,
        crop_duration_seconds=0.0,
        offsource_duration_seconds=duration,
        ifos=ALL_IFOS,
        sample_rate_hertz=sample_rate_hertz,
        scale_factor=1.0,
    )
    x = batch(gf.SeedTree(7).batch("train", 0, 0))[gf.ReturnVariables.ONSOURCE]
    ratio, f = _welch_ratio(x, ALL_IFOS, sample_rate_hertz)
    band = (f >= 30.0) & (f <= 900.0)

    mean = ratio[..., band].mean(axis=0)  # (IFO, F)
    segments_per_example = int((duration - 4.0) / 2.0) + 1
    standard_error = np.sqrt(1.056 / (num_examples * segments_per_example))
    for i, ifo in enumerate(ALL_IFOS):
        median = np.median(mean[i])
        assert abs(median - 1.0) < 0.01, f"{ifo.name}: median Welch/profile = {median:.5g}"
        worst = np.max(np.abs(mean[i] - 1.0) / standard_error)
        assert worst < 5.0, f"{ifo.name}: worst bin is {worst:.2f} standard errors from the profile"


def test_psd_interpolation_uses_the_rfft_grid():
    """Each rfft bin gets the profile at its own frequency, k * fs / N, with no
    power at DC or below the profile, and the last value held above it."""
    sample_rate_hertz = 16384.0
    num_samples = 16384
    frequencies = np.linspace(10.0, 5000.0, 500).astype(np.float32)
    values = np.linspace(2.0, 1.0, 500).astype(np.float32)

    onsource, _ = gf_noise.interpolate_onsource_offsource_psd(
        [num_samples, num_samples], sample_rate_hertz, frequencies, values
    )
    onsource = np.real(np.asarray(onsource))
    grid = np.fft.rfftfreq(num_samples, 1.0 / sample_rate_hertz)
    expected = np.interp(grid, frequencies, values)
    expected[grid < frequencies[0]] = 0.0

    assert onsource.shape == grid.shape
    np.testing.assert_allclose(onsource, expected, rtol=1e-5, atol=1e-7)
    assert onsource[0] == 0.0
    assert onsource[grid > 5000.0].min() == pytest.approx(1.0)


def test_noise_obtainer_defaults_to_internal_units():
    """A NoiseObtainer called without a scale factor returns internal units."""
    obtainer = gf.NoiseObtainer(noise_type=gf.NoiseType.COLORED, ifos=[gf.IFO.H1])
    batch = next(obtainer(
        sample_rate_hertz=FS, onsource_duration_seconds=16.0,
        crop_duration_seconds=0.0, offsource_duration_seconds=16.0,
        num_examples_per_batch=8, seed=11,
    ))
    ratio, f = _welch_ratio(
        np.asarray(batch[gf.ReturnVariables.ONSOURCE], np.float64) / S, [gf.IFO.H1], FS
    )
    band = (f >= 30.0) & (f <= 900.0)
    assert abs(np.median(ratio[:, 0, band].mean(axis=0)) - 1.0) < 0.02


def test_noise_profiles_are_per_detector_with_provenance():
    """Each bundled profile names its source, Virgo's curve is Virgo's, and the
    aLIGO design curve stays available as a named profile."""
    for ifo, source in [
        (gf.IFO.H1, "aLIGOO3LowT1800545"),
        (gf.IFO.L1, "aLIGOO3LowT1800545"),
        (gf.IFO.V1, "AdVO3LowT1800545"),
    ]:
        with open(ifo.value.optimal_psd_path) as file:
            header = "".join(line for line in file if line.startswith("#"))
        assert source in header, f"{ifo.name} profile does not name {source}"

    f_h, asd_h = _profile(gf.IFO.H1)
    f_v, asd_v = _profile(gf.IFO.V1)
    grid = np.geomspace(20.0, 2000.0, 200)
    assert np.max(np.abs(np.log(np.interp(grid, f_v, asd_v) / np.interp(grid, f_h, asd_h)))) > 0.1

    design = gf.NoiseProfile.DESIGN
    for ifo in (gf.IFO.H1, gf.IFO.L1):
        with open(design.path(ifo)) as file:
            assert "T1800044" in file.read(4096)
    assert gf.NoiseProfile.O3_LOW.path(gf.IFO.H1) == gf.IFO.H1.value.optimal_psd_path


def test_colored_noise_uses_the_named_profile():
    """NoiseObtainer(noise_profile=...) colours the noise with that profile."""
    obtainer = gf.NoiseObtainer(
        noise_type=gf.NoiseType.COLORED, ifos=[gf.IFO.H1],
        noise_profile=gf.NoiseProfile.DESIGN,
    )
    batch = next(obtainer(
        sample_rate_hertz=FS, onsource_duration_seconds=16.0,
        crop_duration_seconds=0.0, offsource_duration_seconds=16.0,
        num_examples_per_batch=8, scale_factor=1.0, seed=5,
    ))
    ratio, f = _welch_ratio(
        batch[gf.ReturnVariables.ONSOURCE], [gf.IFO.H1], FS, gf.NoiseProfile.DESIGN
    )
    band = (f >= 30.0) & (f <= 900.0)
    assert abs(np.median(ratio[:, 0, band].mean(axis=0)) - 1.0) < 0.02
