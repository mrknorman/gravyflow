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
import types

import numpy as np
import pytest

import gravyflow as gf
from gravyflow.src.dataset.acquisition.augmentations import (
    AddNoise, RandomShift, SignReversal, TimeReversal
)
from gravyflow.src.dataset.acquisition.transient import TransientDataObtainer
from gravyflow.src.dataset.tools.snr import template_sigma
from gravyflow.src.detection import MatchedFilter, TemplateBank

FS = 2048.0
SCALES = [1e-23, 1e-22, 1e-21, 1e-20, 1e-18, 1e-15, 1e-10, 1e-5, 1e-2, 1.0]
#: Scaling O(1) data by k rounds each sample (relative 6e-8), which whitening and
#: PSD weighting amplify in quiet bands.
TOLERANCE = 1e-5
#: Element-wise tolerance of a power or PSD: its quiet bins are compared too.
ELEMENT_TOLERANCE = 1e-3


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
#: A one-sided PSD of OFFSOURCE[0, 0] on ONSOURCE's rfft grid, float64.
_spectrum = gf.psd(OFFSOURCE[0, 0].astype(np.float32), nperseg=int(FS), sample_rate_hertz=FS)
PSD = np.interp(np.fft.rfftfreq(int(2 * FS), 1.0 / FS), _spectrum.frequencies, _spectrum.value())
WHITE_WINDOW = np.random.default_rng(3).standard_normal((1, int(FS))).astype(np.float32)


def at(x, k):
    """x scaled by k, rounded once to float32."""
    return (np.asarray(x, np.float64) * k).astype(np.float32)


def _augmented(augmentation, k):
    fake = types.SimpleNamespace(rng=np.random.default_rng(0), augmentations=[augmentation])
    return np.asarray(TransientDataObtainer._apply_augmentation(fake, at(ONSOURCE, k))) / k


def _spectrogram_image(k):
    plot = gf.generate_spectrogram(at(ONSOURCE[0, 0], k), sample_rate_hertz=FS)
    return np.asarray(plot.renderers[0].data_source.data["image"][0], np.float64) / k ** 2


_matched_filter = MatchedFilter(bank=TemplateBank.from_masses([30.0], [25.0]), sample_rate_hertz=FS)

ELEMENT, PEAK = "element", "peak"

#: name -> (f(k x) taken back to k = 1, comparison, whether it may refuse). The
#: augmentations run as TransientDataObtainer applies them.
CENSUS = {
    "gf.psd": (lambda k: gf.psd(at(OFFSOURCE, k), nperseg=1024, sample_rate_hertz=FS).value() / k ** 2,
               ELEMENT, False),
    "gf.estimate_psd": (lambda k: gf.estimate_psd(at(OFFSOURCE, k), FS).value() / k ** 2, ELEMENT, False),
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
        at(ONSOURCE[:, 0], k), TEMPLATE[None], psd=PSD * k ** 2, sample_rate_hertz=FS), PEAK, False),
    "gf.matched_filter_fft (PowerSpectrum)": (lambda k: gf.matched_filter_fft(
        at(ONSOURCE[:, 0], k), TEMPLATE[None], psd=gf.estimate_psd(at(OFFSOURCE[:, 0], k), FS),
        sample_rate_hertz=FS), PEAK, False),
    "gf.matched_filter_fft (white)": (lambda k: np.asarray(gf.matched_filter_fft(
        at(ONSOURCE[:, 0], k), TEMPLATE[None], sample_rate_hertz=FS)) / k, PEAK, False),
    "gf.optimal_snr": (lambda k: gf.optimal_snr(at(TEMPLATE, k), psd=PSD * k ** 2, sample_rate_hertz=FS),
                       PEAK, False),
    "template_sigma": (lambda k: template_sigma(at(TEMPLATE[None], k), psd=PSD * k ** 2, sample_rate_hertz=FS),
                       PEAK, False),
    "gf.calculate_hrss": (lambda k: np.asarray(gf.calculate_hrss(at(INJECTION, k), FS)) / k, PEAK, False),
    "gf.calculate_hpeak": (lambda k: np.asarray(gf.calculate_hpeak(at(INJECTION, k))) / k, PEAK, False),
    "gf.MatchedFilter": (lambda k: _matched_filter.snr_series(WHITE_WINDOW, background=at(OFFSOURCE[:1, 0], k)),
                         PEAK, False),
    "gf.SignReversal": (lambda k: _augmented(SignReversal(probability=1.0), k), PEAK, False),
    "gf.TimeReversal": (lambda k: _augmented(TimeReversal(probability=1.0), k), PEAK, False),
    "gf.RandomShift": (lambda k: _augmented(RandomShift(probability=1.0), k), PEAK, False),
    "gf.AddNoise": (lambda k: _augmented(AddNoise(probability=1.0, amplitude=0.1), k), PEAK, False),
}

#: Public names of the numeric modules that take no raw strain, and why.
EXEMPT = {
    "SpectralConfig": "configuration; takes no data",
    "PowerSpectrum": "the estimate gf.psd and gf.estimate_psd return; value() is checked through them",
    "spectrogram_shape": "shapes only",
    "MatchedFilterLayer": "runs gf.MatchedFilter's search (_run) on the model's inputs",
    "TemplateBank": "templates, generated at unit scale",
    "TemplateGrid": "a grid of template parameters",
    "generate_strain_plot": "plots the values it is given; forms no square",
    "generate_psd_plot": "plots PSD values it is given, in float64",
    "generate_correlation_plot": "plots gf.rolling_pearson's output",
    "generate_segment_timeline_plot": "plots segment times",
    "generate_example_extraction_plot": "plots the values it is given; forms no square",
    "default_augmentations": "returns augmentation settings",
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


def _compare(name, comparison, got, reference):
    got, reference = np.asarray(got, np.float64), np.asarray(reference, np.float64)
    assert got.shape == reference.shape, name
    assert np.all(np.isfinite(got)), f"{name}: not finite"
    if comparison == ELEMENT:
        assert np.all(reference > 0), f"{name}: the k = 1 reference has empty bins"
        np.testing.assert_allclose(got, reference, rtol=ELEMENT_TOLERANCE, err_msg=name)
    else:
        peak = np.max(np.abs(reference))
        assert peak > 0, f"{name}: the k = 1 reference is zero"
        error = np.max(np.abs(got - reference)) / peak
        assert error <= TOLERANCE, f"{name}: relative difference {error:.3g} > {TOLERANCE:.3g}"


@pytest.mark.parametrize("name", sorted(CENSUS))
def test_f_of_k_x_relates_to_f_of_x_as_its_units_require_or_refuses(name):
    case, comparison, may_refuse = CENSUS[name]
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
        _compare(f"{name} at k = {k:g}", comparison, got, reference)
    if may_refuse:
        # It refuses physical strain, and only below some amplitude.
        assert 1e-21 in refused
        assert refused == SCALES[:len(refused)], refused


def test_every_public_numeric_function_is_in_the_census():
    # "gf.name" entries are public names; others (template_sigma) are module functions.
    listed = {name.split(" ")[0].removeprefix("gf.") for name in CENSUS if name.startswith("gf.")}
    listed |= set(EXEMPT)
    public = {
        name for name, value in vars(gf).items()
        if callable(value) and getattr(value, "__module__", "").startswith(NUMERIC_MODULES)
    }
    missing = sorted(public - listed)
    assert not missing, f"public numeric functions missing from CENSUS or EXEMPT: {missing}"
    stale = sorted(name for name in listed if not hasattr(gf, name))
    assert not stale, f"CENSUS or EXEMPT names no public function: {stale}"
