"""
The matched-filter search: whitened templates, the in-window complex SNR, the
network statistic, the reported time, the bank's coverage, per-example results
and the baseline model's input contract.

Signals come from the dataset's own CBC generator and are whitened by the data
path's `whiten`, so these tests filter exactly what the dataset produces.
"""
import numpy as np
import pytest

import gravyflow as gf
from gravyflow.src.dataset.conditioning.whiten import whiten
from gravyflow.src.dataset.tools import snr as gf_snr
from gravyflow.src.detection import MatchedFilter, TemplateBank, find_triggers
from gravyflow.src.detection.template_grid import cbc_templates, noise_curve_psd

FS = 2048.0
N = int(FS)            # the 1 s window
CROP = int(0.5 * FS)   # the dataset's default crop


def coloured_noise(rng, shape, seconds):
    """Gaussian noise with L1's bundled noise curve, in strain."""
    n = int(seconds * FS)
    f = np.fft.rfftfreq(n, 1 / FS)
    amp = np.sqrt(noise_curve_psd(gf.IFO.L1, np.maximum(f, 10.0)) * n * FS / 4.0)
    spec = amp * (rng.standard_normal(shape + (len(f),)) + 1j * rng.standard_normal(shape + (len(f),)))
    spec[..., 0] = 0.0
    return np.fft.irfft(spec, n=n)


def raw_onsource(m1, m2, merger_seconds, phase=0.0):
    """A CBC signal in a raw onsource (window plus both crops), merger `merger_seconds` into the window."""
    from gravyflow.src.dataset.features.waveforms.cbc import _generate_cbc_window
    h = np.asarray(_generate_cbc_window(
        num_waveforms=1, sample_rate_hertz=FS, window_seconds=8.0, min_frequency_hertz=np.array([20.0]),
        mass_1_msun=np.array([m1]), mass_2_msun=np.array([m2]), reference_orbital_phase_in=np.array([phase]),
        distance_mpc=np.array([100.0]))[0, 0], dtype=np.float64)
    p = CROP + int(round(merger_seconds * FS))
    c = len(h) // 2
    return h[c - p: c - p + N + 2 * CROP]


@pytest.fixture(scope="module")
def spectrum():
    rng = np.random.default_rng(18)
    return gf.estimate_psd(coloured_noise(rng, (1,), 16.0).astype(np.float32), FS, gf.SpectralConfig())


def whitened(raw, spectrum, target=None):
    """(whitened window, label SNR) of a raw onsource; scaled to `target` if given."""
    raw = np.asarray(raw, dtype=np.float32).reshape(1, -1)
    label = float(np.asarray(gf.snr(gf.crop_samples(raw, 1.0, FS), spectrum, FS))[0])
    w = np.asarray(whiten(raw, spectrum, FS))[:, CROP:-CROP]
    if target is not None:
        w, label = w * (target / label), target
    return w, label


def test_whitened_templates_recover_the_label_snr(spectrum):
    # A noise-free injection, whitened by the data path, against its own
    # template: the matched filter reads the SNR the dataset labels it with.
    mf = MatchedFilter(bank=TemplateBank.from_masses([30.0], [25.0]), sample_rate_hertz=FS)
    for m1, m2, merger in [(30.0, 25.0, 0.7), (30.0, 25.0, 0.2)]:
        w, label = whitened(raw_onsource(m1, m2, merger), spectrum, target=10.0)
        out = mf.max_snr(w, spectrum)
        assert abs(float(out["snr"][0]) - label) < 0.01 * label, (merger, out["snr"])
        # ... and exactly the norm of the whitened window, the SNR module's definition.
        norm = float(np.asarray(gf_snr.optimal_snr(w[0], None, FS)))
        assert abs(float(out["snr"][0]) - norm) < 2e-3 * norm


def test_a_long_inspiral_is_normalised_by_its_part_in_the_window(spectrum):
    # A 12 + 8 binary is in band for several seconds before merger; only its
    # end is in the window, and the filter recovers the window's SNR.
    mf = MatchedFilter(bank=TemplateBank.from_masses([12.0], [8.0]), sample_rate_hertz=FS)
    for merger in (0.1, 0.6, 0.95):
        w, label = whitened(raw_onsource(12.0, 8.0, merger), spectrum, target=10.0)
        out = mf.max_snr(w, spectrum)
        assert abs(float(out["snr"][0]) - label) < 0.01 * label, (merger, out["snr"])


@pytest.mark.parametrize("phase", [0.0, np.pi / 4, np.pi / 2, np.pi])
def test_the_snr_is_maximised_over_phase(spectrum, phase):
    mf = MatchedFilter(bank=TemplateBank.from_masses([30.0], [25.0]), sample_rate_hertz=FS)
    # The orbital phase moves the GW phase by twice as much.
    w, label = whitened(raw_onsource(30.0, 25.0, 0.6, phase=phase / 2), spectrum, target=10.0)
    assert abs(float(mf.max_snr(w, spectrum)["snr"][0]) - 10.0) < 0.1


def test_noise_rho_squared_has_two_degrees_of_freedom():
    # Unit white noise: rho^2 is chi-squared(2) at every template and merger time.
    rng = np.random.default_rng(2)
    mf = MatchedFilter(bank=TemplateBank.from_masses([30.0, 10.0], [25.0, 6.0]), sample_rate_hertz=FS)
    series = np.asarray(mf.snr_series(rng.standard_normal((96, N)).astype(np.float32)))
    assert series.shape == (96, 2, N)
    assert abs(np.mean(series ** 2) / 2.0 - 1.0) < 0.03


def test_whitened_noise_gives_unit_noise_statistics(spectrum):
    # Coloured noise whitened by the data path with the true PSD, against
    # templates whitened with the same filter: still chi-squared(2).
    f = np.arange(N // 2 + 1, dtype=np.float64)
    true = noise_curve_psd(gf.IFO.L1, np.maximum(f, 10.0)) * FS
    exact = gf.PowerSpectrum(np.asarray(true / true.max(), np.float32)[None],
                             np.array([[np.sqrt(true.max())]], np.float32), FS,
                             gf.SpectralConfig(correct_inverse_bias=False), 1.0, 31)
    rng = np.random.default_rng(3)
    raw = coloured_noise(rng, (64,), 2.0).astype(np.float32)
    mf = MatchedFilter(bank=TemplateBank.from_masses([30.0], [25.0]), sample_rate_hertz=FS)
    w = np.asarray(whiten(raw, exact, FS))[:, CROP:-CROP]
    assert abs(np.mean(np.asarray(mf.snr_series(w, exact)) ** 2) / 2.0 - 1.0) < 0.04
    # With an estimated PSD the noise is a few per cent louder: the templates'
    # weights and the noise's excess both follow the estimate's low bins.
    w = np.asarray(whiten(raw, spectrum, FS))[:, CROP:-CROP]
    assert abs(np.mean(np.asarray(mf.snr_series(w, spectrum)) ** 2) / 2.0 - 1.0) < 0.15


def test_reported_time_is_the_merger_time(spectrum):
    mf = MatchedFilter(bank=TemplateBank.from_masses([30.0], [25.0]), sample_rate_hertz=FS)
    for merger in (0.25, 0.6):
        w, _ = whitened(raw_onsource(30.0, 25.0, merger), spectrum, target=12.0)
        out = mf.max_snr(w, spectrum)
        assert abs(float(out["time_seconds"][0]) - merger) <= 2.0 / FS
        trig = mf.detect(w, spectrum, threshold=8.0)
        assert len(trig) == 1 and abs(trig[0]["time_seconds"] - merger) <= 2.0 / FS


def test_max_snr_is_per_example(spectrum):
    rng = np.random.default_rng(5)
    mf = MatchedFilter(bank=TemplateBank.from_masses([30.0], [25.0]), sample_rate_hertz=FS)
    raw = coloured_noise(rng, (4,), 2.0)
    signal = raw_onsource(30.0, 25.0, 0.5)
    _, label = whitened(signal, spectrum)
    raw[2] += signal * 15.0 / label
    w = np.asarray(whiten(raw.astype(np.float32), spectrum, FS))[:, CROP:-CROP]
    out = mf.max_snr(w, spectrum)
    assert out["snr"].shape == (4,) and out["time_seconds"].shape == (4,)
    assert int(np.argmax(out["snr"])) == 2 and out["snr"][2] > 12.0
    assert np.all(np.delete(out["snr"], 2) < 6.0)


def test_one_spectrum_per_example(spectrum):
    # Each example is whitened with its own spectrum; the filter must use each
    # row for its own example, not broadcast rows against the templates.
    rng = np.random.default_rng(6)
    loud = gf.estimate_psd((coloured_noise(rng, (1,), 16.0) * 3.0).astype(np.float32), FS, gf.SpectralConfig())
    both = gf.PowerSpectrum(np.concatenate([np.asarray(spectrum.normalised), np.asarray(loud.normalised)]),
                            np.concatenate([np.asarray(spectrum.scale), np.asarray(loud.scale)]),
                            spectrum.sample_rate_hertz, spectrum.config, spectrum.inverse_bias, spectrum.num_segments)
    raw = raw_onsource(30.0, 25.0, 0.5)
    w0, l0 = whitened(raw, spectrum)
    w1, l1 = whitened(raw, loud)
    mf = MatchedFilter(bank=TemplateBank.from_masses([30.0], [25.0]), sample_rate_hertz=FS)
    out = mf.max_snr(np.concatenate([w0, w1]), both)
    np.testing.assert_allclose(out["snr"], [l0, l1], rtol=0.01)


def test_a_psd_array_is_not_mistaken_for_a_spectrum():
    mf = MatchedFilter(bank=TemplateBank.from_masses([30.0], [25.0]), sample_rate_hertz=FS)
    psd = np.ones((1, N // 2 + 1), dtype=np.float32)
    with pytest.raises(ValueError):
        mf.max_snr(np.zeros((1, N), dtype=np.float32), psd)
    with pytest.raises(ValueError):  # a raw onsource (with its crops) is not a window
        mf.max_snr(np.zeros((1, N + 2 * CROP), dtype=np.float32))


def white_signal(mf, merger_samples):
    """A unit-SNR (30, 25) signal in white data, merger `merger_samples` into the window."""
    t = cbc_templates([30.0], [25.0], FS, 3.0, 20.0)[0].astype(np.float64)
    g = np.fft.irfft(mf._white_band_spectra(np.fft.rfft(t, n=mf.n_fft)), n=mf.n_fft)
    s = mf.template_merger_index - merger_samples
    w = g[s:s + N]
    return w / np.sqrt(np.sum(w ** 2))


def test_network_snr_is_coincident_not_averaged():
    # H1 and L1 see the signal with opposite signs (their antenna patterns) and
    # 8 ms apart: the network SNR is the quadrature sum, not the channel mean's.
    mf = MatchedFilter(bank=TemplateBank.from_masses([30.0], [25.0]), sample_rate_hertz=FS,
                       ifos=[gf.IFO.H1, gf.IFO.L1])
    at = lambda merger_samples: white_signal(mf, merger_samples)
    shift = int(0.008 * FS)
    x = np.stack([10.0 * at(1000), -8.0 * at(1000 + shift)])[None].astype(np.float32)
    out = mf.max_snr(x)
    assert abs(float(out["snr"][0]) - np.hypot(10.0, 8.0)) < 0.05
    assert abs(float(out["time_seconds"][0]) - 1000 / FS) <= 1.0 / FS
    # Beyond the light travel time the second detector's peak is not coincident.
    far = np.stack([10.0 * at(1000), -8.0 * at(1000 + int(0.05 * FS))])[None].astype(np.float32)
    assert float(mf.max_snr(far)["snr"][0]) < 10.6
    with pytest.raises(ValueError):  # two channels need the detectors' geometry
        MatchedFilter(bank=TemplateBank.from_masses([30.0], [25.0]), sample_rate_hertz=FS).max_snr(x)


def test_the_coincidence_window_is_the_networks_light_travel_time():
    mf = MatchedFilter(bank=TemplateBank.from_masses([30.0], [25.0]), sample_rate_hertz=FS,
                       ifos=[gf.IFO.H1, gf.IFO.L1, gf.IFO.V1])
    seconds = float(gf.Network([gf.IFO.H1, gf.IFO.L1, gf.IFO.V1]).max_arrival_time_difference_seconds)
    assert mf.max_time_offset_samples(3) == int(np.ceil((seconds + 0.002) * FS))


def test_network_snr_at_the_edge_of_the_search_window():
    # The first detector's merger is just inside the searched merger times and
    # the second's just outside them: coincidence still finds the second.
    mf = MatchedFilter(bank=TemplateBank.from_masses([30.0], [25.0]), sample_rate_hertz=FS,
                       ifos=[gf.IFO.H1, gf.IFO.L1], search_window_seconds=(0.3, 0.6))
    h1, l1 = int(round(0.595 * FS)), int(round(0.603 * FS))
    x = np.stack([10.0 * white_signal(mf, h1), 8.0 * white_signal(mf, l1)])[None].astype(np.float32)
    out = mf.max_snr(x)
    assert abs(float(out["snr"][0]) - np.hypot(10.0, 8.0)) < 0.05, out["snr"]
    assert abs(float(out["time_seconds"][0]) - h1 / FS) <= 1.0 / FS
    # ... and the same at the window's lower edge.
    mf = MatchedFilter(bank=TemplateBank.from_masses([30.0], [25.0]), sample_rate_hertz=FS,
                       ifos=[gf.IFO.H1, gf.IFO.L1], search_window_seconds=(0.3, 0.6))
    h1, l1 = int(round(0.305 * FS)), int(round(0.297 * FS))
    x = np.stack([10.0 * white_signal(mf, h1), 8.0 * white_signal(mf, l1)])[None].astype(np.float32)
    assert abs(float(mf.max_snr(x)["snr"][0]) - np.hypot(10.0, 8.0)) < 0.05


def test_a_spectrum_is_refused_only_when_it_would_change_the_search():
    # The templates' alignment comes from this filter's tap count and their
    # start from its band's lower edge; their whitening filter is designed from
    # the spectrum. A spectrum whose filter has another tap count, or whose band
    # starts elsewhere, would misalign or mis-start every template, so it is
    # refused by name. Any other difference (the upper band edge, a filter
    # length that rounds to the same taps, the estimator) leaves the search
    # exactly as a filter built with the spectrum's own config.
    rng = np.random.default_rng(10)
    noise = coloured_noise(rng, (1,), 16.0).astype(np.float32)
    bank = TemplateBank.from_masses([30.0], [25.0])
    mf = MatchedFilter(bank=bank, sample_rate_hertz=FS)
    w = rng.standard_normal((1, N)).astype(np.float32)
    for name, config in [("filter_duration_seconds", gf.SpectralConfig(filter_duration_seconds=0.5)),
                         ("lower_frequency_hertz", gf.SpectralConfig(lower_frequency_hertz=30.0))]:
        with pytest.raises(ValueError, match=name):
            mf.max_snr(w, gf.estimate_psd(noise, FS, config))
    for config in [gf.SpectralConfig(upper_frequency_hertz=800.0), gf.SpectralConfig(filter_duration_seconds=1.0001),
                   gf.SpectralConfig(psd_method="mean")]:
        spectrum = gf.estimate_psd(noise, FS, config)
        own = MatchedFilter(bank=bank, sample_rate_hertz=FS, spectral_config=config).max_snr(w, spectrum)
        out = mf.max_snr(w, spectrum)
        assert (out["snr"][0], out["time_seconds"][0]) == (own["snr"][0], own["time_seconds"][0]), config


def test_band_and_templates_start_at_f_low():
    for f_low in (20.0, 30.0):
        t = cbc_templates([95.0], [95.0], FS, 3.0, f_low)[0]
        a = np.abs(np.fft.rfft(t))
        f = np.fft.rfftfreq(len(t), 1 / FS)
        lowest = f[np.argmax(a > 1e-2 * a.max())]
        assert f_low - 3.0 <= lowest <= f_low + 8.0, (f_low, lowest)
    mf = MatchedFilter(bank=TemplateBank.from_masses([30.0], [25.0]), sample_rate_hertz=FS,
                       spectral_config=gf.SpectralConfig(lower_frequency_hertz=30.0))
    f = np.fft.rfftfreq(mf.n_fft, 1 / FS)
    band = np.asarray(mf._white_band_spectra(np.ones(len(f), dtype=np.complex64)))
    assert np.all(band[f < 29.9] == 0) and np.all(band[(f > 30.1) & (f < FS / 2 - 1)] == 1)


def test_chunking_does_not_change_the_result(spectrum):
    rng = np.random.default_rng(7)
    raw = coloured_noise(rng, (10,), 2.0).astype(np.float32)
    w = np.asarray(whiten(raw, spectrum, FS))[:, CROP:-CROP]
    bank = TemplateBank.from_masses([30.0, 20.0, 12.0], [25.0, 15.0, 8.0])
    a = MatchedFilter(bank=bank, sample_rate_hertz=FS, batch_chunk=3, template_chunk=1).max_snr(w, spectrum)
    b = MatchedFilter(bank=bank, sample_rate_hertz=FS, batch_chunk=16, template_chunk=4).max_snr(w, spectrum)
    np.testing.assert_allclose(a["snr"], b["snr"], rtol=1e-5)
    np.testing.assert_array_equal(a["template_index"], b["template_index"])


def test_detect_does_not_build_the_full_series(spectrum, monkeypatch):
    # detect maximises over templates as it scans them, so its memory does not
    # grow with the bank; its triggers are those of the full series.
    rng = np.random.default_rng(11)
    raw = coloured_noise(rng, (3,), 2.0)
    raw[1] += raw_onsource(30.0, 25.0, 0.6) * 12.0 / whitened(raw_onsource(30.0, 25.0, 0.6), spectrum)[1]
    w = np.asarray(whiten(raw.astype(np.float32), spectrum, FS))[:, CROP:-CROP]
    mf = MatchedFilter(bank=TemplateBank.from_masses([30.0, 20.0, 12.0], [25.0, 15.0, 8.0]), sample_rate_hertz=FS,
                       template_chunk=1)
    expected = find_triggers(np.asarray(mf.snr_series(w, spectrum)), 5.0, int(0.1 * FS))
    monkeypatch.setattr(MatchedFilter, "snr_series", lambda *args, **kwargs: pytest.fail("built the full series"))
    triggers = mf.detect(w, spectrum, threshold=5.0)
    assert len(triggers) == len(expected) and any(t["batch_index"] == 1 for t in triggers)
    for t, e in zip(triggers, expected):
        assert (t["batch_index"], t["template_index"]) == (e["batch_index"], e["template_index"])
        assert t["time_seconds"] == e["time_index"] / FS and abs(t["snr"] - e["snr"]) < 1e-4 * e["snr"]


def test_find_triggers_clusters_each_example():
    # Maximised over templates first: the loudest template at each time.
    series = np.zeros((3, 1000), dtype=np.float32)
    series[0, 200] = 10.0
    series[1, 600] = 12.0
    series[2, 601] = 11.0
    trig = find_triggers(series, threshold=8.0, cluster_window=50)
    assert [(t["time_index"], t["snr"], t["template_index"]) for t in trig] == [(200, 10.0, 0), (600, 12.0, 1)]
    assert "batch_index" not in trig[0]
    # Each example is clustered on its own.
    peak = np.zeros((1, 1000), dtype=np.float32)
    peak[0, 400:461] = 20.0 * np.hanning(61)
    assert len(find_triggers(peak, threshold=8.0, cluster_window=200)) == 1
    batched = find_triggers(np.stack([peak, peak]), threshold=8.0, cluster_window=200)
    assert [(t["batch_index"], t["time_index"]) for t in batched] == [(0, 430), (1, 430)]
    two = np.zeros((2, 2, 500), dtype=np.float32)
    two[0, 0, 100] = 9.0
    two[1, 1, 300] = 11.0
    trig = find_triggers(two, threshold=8.0, cluster_window=20)
    assert [(t["batch_index"], t["time_index"], t["template_index"]) for t in trig] == [(0, 100, 0), (1, 300, 1)]
    assert find_triggers(np.full((2, 100), 5.0), threshold=8.0) == []
    run = np.zeros((1, 2000), dtype=np.float32)
    run[0, 100:1100] = 9.0
    run[0, 600] = 12.0
    trig = find_triggers(run, 8.0, 200)  # one long excursion is one trigger, at its peak
    assert len(trig) == 1 and trig[0]["time_index"] == 600
    two = np.zeros((1, 2000), dtype=np.float32)
    two[0, 100] = two[0, 500] = 9.0
    assert len(find_triggers(two, 8.0, 200)) == 2


def test_stochastic_placement_reaches_its_minimal_match():
    # The placement algorithm, in the metric it places with: the stopping rule
    # leaves little of the range below the minimal match. Whether that metric
    # is the search's is tested below, independently of it.
    bank = TemplateBank.stochastic(mass_range=(20.0, 40.0), convergence_candidates=512, use_bundled=False)
    rng = np.random.default_rng(8)
    m = rng.uniform(20.0, 40.0, size=(40, 2))
    ff = bank.fitting_factors(m.max(axis=1), m.min(axis=1))
    assert np.median(ff) >= 0.97 and np.min(ff) >= 0.95, (np.median(ff), np.min(ff))


def test_the_default_bank_covers_the_search(spectrum):
    # The fitting factor through the search itself, not the metric the bank
    # was placed with: the default bank's SNR over each injection's own
    # template's, for noise-free whitened injections across the mass range and
    # the window.
    rng = np.random.default_rng(19)
    m = np.exp(rng.uniform(np.log(5.0), np.log(95.0), size=(10, 2)))
    m1, m2 = m.max(axis=1), m.min(axis=1)
    mergers = rng.uniform(0.02, 0.98, size=len(m1))
    w = np.concatenate([whitened(raw_onsource(a, b, t), spectrum, target=20.0)[0] for a, b, t in zip(m1, m2, mergers)])
    bank = MatchedFilter(sample_rate_hertz=FS).max_snr(w, spectrum)
    own = np.array([MatchedFilter(bank=TemplateBank.from_masses([a], [b]), sample_rate_hertz=FS)
                    .max_snr(w[i:i + 1], spectrum)["snr"][0] for i, (a, b) in enumerate(zip(m1, m2))])
    ff = bank["snr"] / own
    assert ff.min() >= 0.97, list(zip(m1.round(1), m2.round(1), ff.round(4)))
    assert np.all(np.abs(bank["time_seconds"] - mergers) <= 0.01)


VISIBLE_SECONDS = 1.5  # what the default search sees of a signal: the window plus half the filter
PYCBC_SEGMENT = int(4 * FS)


def visible_spectra(mass_1, mass_2):
    """
    pycbc frequency series of what the default search sees of each waveform,
    built here rather than by the bank's MatchSpace: the VISIBLE_SECONDS before
    merger, Hann-ramped over their first tenth, and everything after.
    """
    import pycbc.types
    h = cbc_templates(mass_1, mass_2, FS, 2 * VISIBLE_SECONDS, 20.0).astype(np.float64)
    h = h[:, h.shape[-1] // 2 - int(VISIBLE_SECONDS * FS):]
    width = int(round(0.1 * VISIBLE_SECONDS * FS))
    h[:, :width] *= 0.5 - 0.5 * np.cos(np.pi * np.arange(width) / width)
    return [pycbc.types.TimeSeries(np.pad(row, (0, PYCBC_SEGMENT - len(row))), delta_t=1 / FS)
            .to_frequencyseries() for row in h]


def chirp_times(mass_1, mass_2, f_low=20.0):
    """(tau0, tau3) in seconds at f_low: the Newtonian and 1.5PN chirp times."""
    m1, m2 = np.asarray(mass_1, dtype=np.float64), np.asarray(mass_2, dtype=np.float64)
    eta = m1 * m2 / (m1 + m2) ** 2
    x = np.pi * (m1 + m2) * 4.925490947e-6 * f_low
    return 5.0 / (256.0 * np.pi * f_low * eta) * x ** (-5.0 / 3.0), 1.0 / (8.0 * f_low * eta) * x ** (-2.0 / 3.0)


def pycbc_fitting_factors(bank, mass_1, mass_2, nearest=None):
    """
    Each signal's best pycbc.filter.match (phase and time maximised, with
    LALSimulation's analytic aLIGOO3LowT1800545 from 20 Hz) with a template of
    `bank`: an overlap independent of the metric the bank is placed with. With
    `nearest`, each signal is matched only against that many templates nearest
    to it in chirp times, which can only lower its fitting factor.
    """
    import pycbc.filter
    import pycbc.psd
    psd = pycbc.psd.from_string("aLIGOO3LowT1800545", PYCBC_SEGMENT // 2 + 1, FS / PYCBC_SEGMENT, 10.0)
    templates = visible_spectra(bank.mass_1, bank.mass_2)
    t0, t3 = chirp_times(bank.mass_1, bank.mass_2)
    s0, s3 = chirp_times(mass_1, mass_2)
    ff = []
    for signal, a0, a3 in zip(visible_spectra(mass_1, mass_2), s0, s3):
        candidates = np.argsort(np.hypot(t0 / a0 - 1.0, t3 / a3 - 1.0))[:nearest]
        ff.append(max(pycbc.filter.match(signal, templates[j], psd=psd, low_frequency_cutoff=20.0)[0]
                      for j in candidates))
    return np.array(ff)


def test_the_bank_metric_is_pycbcs_overlap():
    # pycbc's match (its own overlap, LALSimulation's analytic noise curve) on
    # the search's waveforms equals the fitting factor the bank is placed with.
    from gravyflow.src.detection.template_grid import MatchSpace
    bank = TemplateBank.from_masses([50.0, 13.0, 28.0, 30.0], [11.0, 8.5, 8.0, 25.0])
    signals = ([51.7, 12.2, 26.7], [10.4, 9.1, 8.3])
    own = bank.fitting_factors(*signals, MatchSpace())
    np.testing.assert_allclose(own, pycbc_fitting_factors(bank, *signals), atol=1e-3)
    assert np.min(own) < 0.97  # the comparison spans matches either side of the minimal match


def coverage_injections(n):
    """n log-uniform (m1, m2) over 5-95, the measure the bank is placed in, from a fixed seed."""
    m = np.exp(np.random.default_rng(187).uniform(np.log(5.0), np.log(95.0), size=(n, 2)))
    return m.max(axis=1), m.min(axis=1)


def test_the_bundled_bank_covers_its_range_by_an_independent_overlap():
    # The shipped bank's coverage of its interior, measured with pycbc's match
    # rather than the placement metric; its boundary is the next test's.
    # Placement stops when about 1% of the space is left below the minimal
    # match: three placements with different seeds leave 0.72-1.05% of 20,000
    # log-uniform injections there, so a sound bank puts more than 22 of these
    # 1000 below it with probability 2e-6 to 5e-4 (binomial; 3e-2 at 1.5%).
    # Removing 10% of the templates leaves 3.9% below, which passes with
    # probability 2e-3; removing 25% leaves 10%, which passes with probability
    # below 1e-21. The minimum is not bounded: it is the deepest of a few
    # random holes, so it depends on where the injections fall.
    bank = MatchedFilter(sample_rate_hertz=FS).bank
    ff = pycbc_fitting_factors(bank, *coverage_injections(1000), nearest=48)
    assert np.sum(ff < 0.97) <= 22, (np.sum(ff < 0.97), ff.min())


def test_the_bundled_bank_covers_the_boundary_of_its_range():
    # Placement covers every point of the boundary (one component at 5 or 95
    # solar masses, corners included) by construction, so this bound holds for
    # any sound placement, not only for this one. Points between the
    # placement's own boundary candidates, in the metric it places with (equal
    # to pycbc's overlap, above). A bank shifted by 2% in mass leaves the edge
    # m2 = 5 uncovered, which the interior's fraction above cannot see.
    bank = MatchedFilter(sample_rate_hertz=FS).bank
    edge = np.exp(np.linspace(np.log(5.0), np.log(95.0), 97))
    m1, m2 = np.concatenate([edge, np.full(97, 95.0)]), np.concatenate([np.full(97, 5.0), edge])
    ff = bank.fitting_factors(m1, m2)
    worst = np.argsort(ff)[:3]
    assert ff.min() >= 0.965, [(m1[i].round(2), m2[i].round(2), ff[i].round(4)) for i in worst]


def test_the_bundled_bank_is_the_default_filters_and_names_what_it_depends_on():
    import hashlib
    import json
    from gravyflow.src.detection.matched_filter import default_bank_placement
    from gravyflow.src.detection.template_grid import BUNDLED_BANK_DIRECTORY
    key = json.loads(json.dumps(default_bank_placement().key()))
    curve = gf.IFO.L1.value.optimal_psd_path
    with open(curve, "rb") as f:
        assert key["space"]["noise_curve"]["sha256"] == hashlib.sha256(f.read()).hexdigest()
    assert key["space"]["waveform_model"]["approximant"] == "IMRPhenomD"
    assert key["code"]["files"] == ["cbc.py", "template_grid.py"]
    records = [json.loads(p.read_text()) for p in sorted((gf.PATH / BUNDLED_BANK_DIRECTORY).glob("*.json"))]
    # After any change to what the key names, the bundled bank is placed again.
    assert [r["arguments"] for r in records] == [key], \
        "a stale bundled bank: place it again with python -m gravyflow.src.detection.bundle_banks"
    bank = MatchedFilter(sample_rate_hertz=FS).bank
    assert bank.minimal_match == 0.97 and bank.num_templates == len(records[0]["mass_1"])
    assert bank.to_dict() == records[0]  # the library's own serialiser writes the loadable record


def test_a_bank_record_is_found_by_its_key_and_a_stale_one_is_not(monkeypatch):
    from gravyflow.src.detection import template_grid
    from gravyflow.src.detection.matched_filter import default_bank_placement
    placement = default_bank_placement(mass_range=(30.0, 40.0))
    record = TemplateBank(np.array([35.0, 31.0]), np.array([33.0, 30.0]), placement, 256, 0.004).to_dict()
    monkeypatch.setattr(template_grid, "_bundled_banks", lambda: (record,))
    monkeypatch.setattr(template_grid, "_stochastic_masses",
                        lambda *args: (np.array([40.0]), np.array([30.0]), 128, 0.0))
    found = TemplateBank.place(placement)
    np.testing.assert_array_equal(found.mass_1, [35.0, 31.0])
    assert found.placement == placement and found.to_dict() == record
    # The same arguments with another noise curve: not loaded, and said so.
    stale = json_copy(record)
    stale["arguments"]["space"]["noise_curve"]["sha256"] = "0" * 64
    monkeypatch.setattr(template_grid, "_bundled_banks", lambda: (stale,))
    with pytest.warns(UserWarning, match="noise_curve"):
        placed = TemplateBank.place(placement)
    np.testing.assert_array_equal(placed.mass_1, [40.0])


def test_the_bank_key_names_the_placement_code(tmp_path, monkeypatch):
    # A bank depends on gravyflow's own code too (the CBC generator, the match
    # space, the proposal and acceptance rule): changing the ramp width changes
    # the key, and the bundled bank placed before the change is not loaded.
    from gravyflow.src.detection import template_grid
    from gravyflow.src.detection.matched_filter import default_bank_placement
    placement = default_bank_placement()
    key = placement.key()
    copies = []
    for path in template_grid.PLACEMENT_SOURCES:
        copies.append(tmp_path / path.name)
        copies[-1].write_bytes(path.read_bytes())
    monkeypatch.setattr(template_grid, "PLACEMENT_SOURCES", tuple(copies))
    assert placement.key() == key  # the code's content, not its location
    module = next(path for path in copies if path.name == "template_grid.py")
    source = module.read_text()
    assert source.count("0.1 * self.visible_seconds") == 1
    module.write_text(source.replace("0.1 * self.visible_seconds", "0.2 * self.visible_seconds"))
    changed = placement.key()
    assert changed["code"] != key["code"] and {**changed, "code": None} == {**key, "code": None}
    monkeypatch.setattr(template_grid, "_stochastic_masses",
                        lambda *args: (np.array([40.0]), np.array([30.0]), 128, 0.0))
    with pytest.warns(UserWarning, match="code"):
        placed = TemplateBank.place(placement)
    np.testing.assert_array_equal(placed.mass_1, [40.0])


def test_max_templates_holds_for_a_bundled_bank():
    # The limit is the same whether the bank is placed or loaded.
    with pytest.raises(ValueError, match="max_templates"):
        TemplateBank.stochastic(max_templates=100)
    size = MatchedFilter(sample_rate_hertz=FS).bank.num_templates
    assert TemplateBank.stochastic(max_templates=size).num_templates == size


def json_copy(data):
    import json
    return json.loads(json.dumps(data))


def test_baseline_takes_the_example_input_contract():
    from gravyflow.src.model.examples.matched_filter_baseline import (
        MatchedFilterBaseline, MatchedFilterBaselineConfig)
    from gravyflow.src.validate.bank import require_logit_model
    cfg = MatchedFilterBaselineConfig(mass_range_msun=(25.0, 35.0), batch_size=4, snr_min=15.0, snr_max=15.0,
                                      injection_chance=1.0)
    model = MatchedFilterBaseline.model(config=cfg)
    require_logit_model(model)
    assert [t.name for t in model.inputs] == ["WHITENED_ONSOURCE", "OFFSOURCE"]
    x, y = MatchedFilterBaseline.dataset(config=cfg, seed=11)[0]
    assert x["WHITENED_ONSOURCE"].shape == (4, 1, N) and x["WHITENED_ONSOURCE"].dtype == np.float32
    # The labels every example returns: one presence flag per example.
    assert list(y) == ["INJECTION_PRESENT"]
    np.testing.assert_array_equal(np.asarray(y["INJECTION_PRESENT"]), np.ones(4, np.float32))
    assert np.asarray(y["INJECTION_PRESENT"]).dtype == np.float32
    logits = np.asarray(model.predict_on_batch(x))
    assert logits.shape == (4, 2) and logits.dtype == np.float32
    snr = logits[:, 1] * cfg.temperature + cfg.snr_threshold
    assert np.all(snr > 9.0), snr  # every example holds an SNR-15 injection


def test_the_baseline_searches_the_datasets_placement_range():
    from gravyflow.src.model.examples.matched_filter_baseline import (
        MatchedFilterBaseline, MatchedFilterBaselineConfig)
    cfg = MatchedFilterBaselineConfig(mass_range_msun=(25.0, 35.0))
    args = MatchedFilterBaseline.dataset_args(config=cfg)
    # The default placement puts mergers anywhere in the window.
    assert MatchedFilterBaseline.search_window_seconds(args) == (0.0, 1.0)
    # A narrower placement narrows the search, widened by the detector's delay
    # from the geocentre (a detector sees the merger that much off the placed time).
    args["waveform_generators"].front_padding_duration_seconds = 0.0
    args["waveform_generators"].back_padding_duration_seconds = 0.3
    margin = float(np.max(np.linalg.norm(np.asarray(gf.Network([gf.IFO.L1]).location), axis=-1))) / 299792458.0
    lo, hi = MatchedFilterBaseline.search_window_seconds(args)
    assert lo == pytest.approx(0.5 - margin) and hi == pytest.approx(0.8 + margin) and 0.02 < margin < 0.022
    layer = MatchedFilterBaseline.model(config=cfg, dataset_args=args).get_layer("matched_filter")
    assert layer.search_window_seconds == (lo, hi)


def test_a_noise_only_dataset_has_no_placement_range_to_search():
    # Without waveform generators (none, an empty list, or no entry: gf.Dataset
    # takes all three) there is no placement range, so the window must be set.
    from gravyflow.src.model.examples.matched_filter_baseline import (
        MatchedFilterBaseline, MatchedFilterBaselineConfig)
    args = MatchedFilterBaseline.dataset_args(config=MatchedFilterBaselineConfig(mass_range_msun=(25.0, 35.0)))
    missing = {k: v for k, v in args.items() if k != "waveform_generators"}
    for noise_only in (missing, {**args, "waveform_generators": []}, {**args, "waveform_generators": None}):
        with pytest.raises(ValueError, match="no waveform generators.*search_window_seconds"):
            MatchedFilterBaseline.search_window_seconds(noise_only)


def test_layer_round_trips_through_its_config_and_a_saved_model(tmp_path):
    import keras
    from gravyflow.src.detection import MatchedFilterLayer
    from gravyflow.src.model.examples.matched_filter_baseline import (
        MatchedFilterBaseline, MatchedFilterBaselineConfig)
    cfg = MatchedFilterBaselineConfig(mass_range_msun=(25.0, 35.0), batch_size=2)
    model = MatchedFilterBaseline.model(config=cfg)
    layer = model.get_layer("matched_filter")
    clone = MatchedFilterLayer.from_config(layer.get_config())
    np.testing.assert_array_equal(clone.matched_filter.bank.mass_1, layer.matched_filter.bank.mass_1)
    x, _ = MatchedFilterBaseline.dataset(config=cfg, seed=12)[0]
    path = tmp_path / "baseline.keras"
    model.save(path)
    reloaded = keras.models.load_model(path)
    np.testing.assert_allclose(reloaded.predict_on_batch(x), model.predict_on_batch(x), rtol=1e-6)


def test_the_validator_scores_the_baseline():
    from gravyflow.src.model.examples.base import ValidationDatasetConfig
    from gravyflow.src.model.examples.matched_filter_baseline import (
        MatchedFilterBaseline, MatchedFilterBaselineConfig)
    from gravyflow.src.validate import ValidationConfig, Validator
    cfg = MatchedFilterBaselineConfig(mass_range_msun=(25.0, 35.0))
    args = MatchedFilterBaseline.validation_dataset_args(
        config=ValidationDatasetConfig(mass_range_msun=(25.0, 35.0), batch_size=8), baseline=cfg)
    validator = Validator.validate(
        model=MatchedFilterBaseline.model(config=cfg), name="matched_filter", dataset_args=args,
        config=ValidationConfig(num_examples=16, batch_size=8, snr_range=(12.0, 20.0), snr_bin_width=8.0))
    noise = validator.far_scores * cfg.temperature + cfg.snr_threshold
    signal = validator.efficiency_data["scores"] * cfg.temperature + cfg.snr_threshold
    assert noise.shape == (16,) and signal.shape == (16,)
    assert np.all(noise < 7.0) and np.median(signal) > 11.0, (noise, signal)
