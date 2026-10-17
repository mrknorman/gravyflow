"""
Every public function and layer that can take raw strain, at every amplitude.

gravyflow's arrays are physical strain (~1e-21), whose square (~1e-42) is below
float32's smallest normal number (1.2e-38); XLA on CPU flushes it to zero. A
function that squares, normalises or takes the power of its input must
therefore either be scale free (normalise inside, and restore the scale its
units require) or refuse an input it cannot represent, by name.

CENSUS lists each such public function and layer with the relation its units
require between f(k x) and f(x): the case returns f(k x) taken back to k = 1 (for
example a PSD divided by k^2), so every case is compared with itself at k = 1.
Each case runs for k from 1e-23 to 1 on O(1) data. A case that may refuse may
raise ValueError instead, naming float32, at the amplitudes whose result float32 cannot hold.

`test_every_public_numeric_function_is_in_the_census` lists the public names of
the modules that compute on data (PSD, SNR, whitening, Pearson, spectrogram, the
matched filter, plotting and the augmentations): a new public function there
fails until it joins CENSUS, or EXEMPT with the reason it takes no raw strain.
"""
import functools

import numpy as np
import pytest

import gravyflow as gf
from gravyflow.src.dataset.acquisition.augmentations import (
    AddNoise, RandomShift, SignReversal, TimeReversal, augment
)
from gravyflow.src.dataset.tools.psd import estimate_psd
from gravyflow.src.dataset.tools.snr import template_sigma
from gravyflow.src.detection import MatchedFilter, TemplateBank

FS = 2048.0
#: Every k but 1, the reference itself.
SCALES = [1e-23, 1e-22, 1e-21, 1e-20, 1e-18, 1e-15, 1e-10, 1e-5, 1e-2]


def _coloured(seed, shape, seconds):
    """O(1) Gaussian noise with a steep low-frequency wall, like strain's, float64."""
    rng = np.random.default_rng(seed)
    n = int(seconds * FS)
    f = np.fft.rfftfreq(n, 1.0 / FS)
    g = np.maximum(f, 10.0)
    asd = np.where(f > 10.0, (g / 100.0) ** -2 + 1.0 + (g / 500.0) ** 2, 0.0)
    x = np.fft.irfft(np.fft.rfft(rng.standard_normal(shape + (n,))) * asd, n=n)
    return x / np.std(x)


ONSOURCE = _coloured(1, (2, 2), 2.0)
OFFSOURCE = _coloured(2, (2, 2), 16.0)
_t = (np.arange(int(2 * FS)) - FS) / FS
TEMPLATE = np.exp(-(_t / 0.01) ** 2) * np.sin(2 * np.pi * 150.0 * _t)
INJECTION = np.tile(TEMPLATE, (2, 2, 1))
WHITE_WINDOW = np.random.default_rng(3).standard_normal((1, int(FS))).astype(np.float32)


def at(x, k):
    """x scaled by k, rounded once to float32."""
    return (np.asarray(x, np.float64) * k).astype(np.float32)


def _augmented(augmentation, k):
    strain = (at(ONSOURCE, k), at(OFFSOURCE, k))
    augmented = augment(strain, gf.SeedTree(0).batch("train", 0, 0), [augmentation], "train",
                        [gf.DataLabel.GLITCHES])
    return np.asarray(augmented.onsource) / k


def _spectrogram_image(k):
    plot = gf.generate_spectrogram(at(ONSOURCE[0, 0], k), sample_rate_hertz=FS)
    return np.asarray(plot.renderers[0].data_source.data["image"][0], np.float64) / k ** 2


@functools.cache
def _psd():
    """A one-sided PSD of OFFSOURCE[0, 0] on ONSOURCE's rfft grid, float64."""
    spectrum = gf.psd(OFFSOURCE[0, 0].astype(np.float32), nperseg=int(FS), sample_rate_hertz=FS)
    return np.interp(np.fft.rfftfreq(int(2 * FS), 1.0 / FS), spectrum.frequencies, spectrum.value())


@functools.cache
def _matched_filter():
    return MatchedFilter(bank=TemplateBank.from_masses([30.0], [25.0]), sample_rate_hertz=FS)

ELEMENT, PEAK = "element", "peak"

#: name -> (f(k x) taken back to k = 1, comparison, whether it may refuse). The
#: augmentations run as a dataset applies them (``augment``), to glitches.
CENSUS = {
    "gf.psd": (lambda k: gf.psd(at(OFFSOURCE, k), nperseg=1024, sample_rate_hertz=FS).value() / k ** 2,
               ELEMENT, False),
    "estimate_psd": (lambda k: estimate_psd(at(OFFSOURCE, k), FS).value() / k ** 2, ELEMENT, False),
    "gf.whiten": (lambda k: gf.whiten(at(ONSOURCE, k), at(OFFSOURCE, k), FS), PEAK, False),
    "gf.Whiten": (lambda k: gf.Whiten(sample_rate_hertz=FS, onsource_duration_seconds=1.0)(
        [at(ONSOURCE, k), at(OFFSOURCE, k)]), PEAK, False),
    "gf.WhitenPass": (lambda k: gf.WhitenPass(sample_rate_hertz=FS, onsource_duration_seconds=1.0)(
        at(ONSOURCE, k)), PEAK, False),
    "gf.spectrogram": (lambda k: np.asarray(gf.spectrogram(at(ONSOURCE, k)), np.float64) / k ** 2,
                       ELEMENT, True),
    "gf.generate_spectrogram": (_spectrogram_image, ELEMENT, True),
    "gf.rolling_pearson": (lambda k: gf.rolling_pearson(at(ONSOURCE, k), 0.011, FS), PEAK, False),
    "gf.snr": (lambda k: gf.snr(at(INJECTION, k), at(OFFSOURCE, k), FS), PEAK, False),
    "gf.scale_to_snr": (lambda k: np.asarray(gf.scale_to_snr(at(INJECTION, k), at(OFFSOURCE, k), 8.0, FS)) / k,
                        PEAK, False),
    "gf.matched_filter_fft (PSD array)": (lambda k: gf.matched_filter_fft(
        at(ONSOURCE[:, 0], k), TEMPLATE[None], psd=_psd() * k ** 2, sample_rate_hertz=FS), PEAK, False),
    "gf.matched_filter_fft (PowerSpectrum)": (lambda k: gf.matched_filter_fft(
        at(ONSOURCE[:, 0], k), TEMPLATE[None], psd=estimate_psd(at(OFFSOURCE[:, 0], k), FS),
        sample_rate_hertz=FS), PEAK, False),
    "gf.matched_filter_fft (white)": (lambda k: np.asarray(gf.matched_filter_fft(
        at(ONSOURCE[:, 0], k), TEMPLATE[None], sample_rate_hertz=FS)) / k, PEAK, False),
    "gf.optimal_snr": (lambda k: gf.optimal_snr(at(TEMPLATE, k), psd=_psd() * k ** 2, sample_rate_hertz=FS),
                       PEAK, False),
    "template_sigma": (lambda k: template_sigma(at(TEMPLATE[None], k), psd=_psd() * k ** 2, sample_rate_hertz=FS),
                       PEAK, False),
    "gf.calculate_hrss": (lambda k: np.asarray(gf.calculate_hrss(at(INJECTION, k), FS)) / k, PEAK, False),
    "gf.calculate_hpeak": (lambda k: np.asarray(gf.calculate_hpeak(at(INJECTION, k))) / k, PEAK, False),
    "gf.MatchedFilter": (lambda k: _matched_filter().snr_series(WHITE_WINDOW, background=at(OFFSOURCE[:1, 0], k)),
                         PEAK, False),
    "gf.SignReversal": (lambda k: _augmented(SignReversal(probability=1.0), k), PEAK, False),
    "gf.TimeReversal": (lambda k: _augmented(TimeReversal(probability=1.0), k), PEAK, False),
    "gf.RandomShift": (lambda k: _augmented(RandomShift(probability=1.0), k), PEAK, False),
    "gf.AddNoise": (lambda k: _augmented(AddNoise(probability=1.0, amplitude=0.1), k), PEAK, False),
}

#: Measured: each case's worst error over SCALES, by its comparison's metric (relative
#: to each element of the k = 1 result, or to its peak), on CPU (macOS arm64) at gravyflow
#: df4acab, rounded up. Scaling by k rounds each sample once (relative 6e-8); a case's
#: error is that rounding as it amplifies it: PSD weighting and whitening most, in quiet
#: bands, and the spectrogram most of all, element by element in its quietest pixels.
WORST_ERROR = {
    "estimate_psd": 6.4e-5,
    "gf.AddNoise": 1.1e-7,
    "gf.MatchedFilter": 3.4e-7,
    "gf.RandomShift": 1.1e-7,
    "gf.SignReversal": 1.1e-7,
    "gf.TimeReversal": 1.1e-7,
    "gf.Whiten": 1.1e-6,
    "gf.WhitenPass": 2.4e-7,
    "gf.calculate_hpeak": 8.8e-8,
    "gf.calculate_hrss": 6.7e-8,
    "gf.generate_spectrogram": 3.1e-4,
    "gf.matched_filter_fft (PSD array)": 7.4e-7,
    "gf.matched_filter_fft (PowerSpectrum)": 1.2e-6,
    "gf.matched_filter_fft (white)": 7.0e-7,
    "gf.optimal_snr": 6.3e-8,
    "gf.psd": 7.6e-7,
    "gf.rolling_pearson": 2.8e-7,
    "gf.scale_to_snr": 3.4e-7,
    "gf.snr": 1.8e-7,
    "gf.spectrogram": 9.0e-5,
    "gf.whiten": 1.1e-6,
    "template_sigma": 6.3e-8,
}
#: The margin each case's bound keeps over its measured worst error.
MARGIN = 3.0

#: Public names of the numeric modules that take no raw strain, and why.
EXEMPT = {
    "SpectralConfig": "configuration; takes no data",
    "PowerSpectrum": "the estimate gf.psd and estimate_psd return; value() is checked through them",
    "spectrogram_shape": "shapes only",
    "MatchedFilterLayer": "runs gf.MatchedFilter's search (_run) on the model's inputs",
    "TemplateBank": "templates, generated at unit scale",
    "TemplateGrid": "a grid of template parameters",
    "generate_strain_plot": "plots the values it is given; forms no square",
    "generate_psd_plot": "plots PSD values it is given, in float64",
    "generate_correlation_plot": "plots gf.rolling_pearson's output",
    "generate_segment_timeline_plot": "plots segment times",
    "generate_example_extraction_plot": "plots the values it is given; forms no square",
}

NUMERIC_MODULES = (
    "gravyflow.src.dataset.tools.psd",
    "gravyflow.src.dataset.tools.snr",
    "gravyflow.src.dataset.conditioning.conditioning",
    "gravyflow.src.dataset.conditioning.pearson",
    "gravyflow.src.dataset.conditioning.whiten",
    "gravyflow.src.detection",
    "gravyflow.src.utils.plotting",
    "gravyflow.src.dataset.acquisition.augmentations",
)


def _compare(name, comparison, got, reference, tolerance):
    got, reference = np.asarray(got, np.float64), np.asarray(reference, np.float64)
    assert got.shape == reference.shape, name
    assert np.all(np.isfinite(got)), f"{name}: not finite"
    if comparison == ELEMENT:
        assert np.all(reference > 0), f"{name}: the k = 1 reference has empty bins"
        error = np.max(np.abs(got - reference) / reference)
    else:
        peak = np.max(np.abs(reference))
        assert peak > 0, f"{name}: the k = 1 reference is zero"
        error = np.max(np.abs(got - reference)) / peak
    assert error <= tolerance, f"{name}: relative difference {error:.3g} > {tolerance:.3g}"


@pytest.mark.parametrize("name", sorted(CENSUS))
def test_f_of_k_x_relates_to_f_of_x_as_its_units_require_or_refuses(name):
    case, comparison, may_refuse = CENSUS[name]
    tolerance = MARGIN * WORST_ERROR[name]
    reference = case(1.0)
    refused = []
    for k in SCALES:
        try:
            got = case(k)
        except ValueError as error:
            assert may_refuse, f"{name} refused k = {k:g}: {error}"
            assert "float32" in str(error), f"{name} refused k = {k:g} without naming why: {error}"
            refused.append(k)
            continue
        _compare(f"{name} at k = {k:g}", comparison, got, reference, tolerance)
    if may_refuse:
        # It refuses physical strain, and only below some amplitude.
        assert 1e-21 in refused
        assert refused == SCALES[:len(refused)], refused


def test_every_public_numeric_function_is_in_the_census():
    # "gf.name" entries are public names; others (estimate_psd, template_sigma) are module functions.
    listed = {name.split(" ")[0].removeprefix("gf.") for name in CENSUS if name.startswith("gf.")}
    listed |= set(EXEMPT)
    public = {
        name for name, value in ((name, getattr(gf, name)) for name in gf.__all__)
        if callable(value) and getattr(value, "__module__", "").startswith(NUMERIC_MODULES)
    }
    missing = sorted(public - listed)
    assert not missing, f"public numeric functions missing from CENSUS or EXEMPT: {missing}"
    stale = sorted(name for name in listed if not hasattr(gf, name))
    assert not stale, f"CENSUS or EXEMPT names no public function: {stale}"
