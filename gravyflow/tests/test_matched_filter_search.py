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
    """Gaussian noise with L1's bundled noise curve, x 1e21 (the dataset's internal units)."""
    n = int(seconds * FS)
    f = np.fft.rfftfreq(n, 1 / FS)
    amp = np.sqrt(noise_curve_psd(gf.IFO.L1, np.maximum(f, 10.0)) * n * FS / 4.0) * 1e21
    spec = amp * (rng.standard_normal(shape + (len(f),)) + 1j * rng.standard_normal(shape + (len(f),)))
    spec[..., 0] = 0.0
    return np.fft.irfft(spec, n=n)


def raw_onsource(m1, m2, merger_seconds, phase=0.0):
    """A CBC signal in a raw onsource (window plus both crops), merger `merger_seconds` into the window."""
    from gravyflow.src.dataset.features.waveforms.cbc import _generate_cbc_window
    h = np.asarray(_generate_cbc_window(
        num_waveforms=1, sample_rate_hertz=FS, window_seconds=8.0, min_frequency_hertz=np.array([20.0]),
        mass_1_msun=np.array([m1]), mass_2_msun=np.array([m2]), reference_orbital_phase_in=np.array([phase]),
        distance_mpc=np.array([100.0]))[0, 0], dtype=np.float64) * 1e21
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
    true = noise_curve_psd(gf.IFO.L1, np.maximum(f, 10.0)) * 1e42 * FS
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


def test_network_snr_is_coincident_not_averaged():
    # H1 and L1 see the signal with opposite signs (their antenna patterns) and
    # 8 ms apart: the network SNR is the quadrature sum, not the channel mean's.
    mf = MatchedFilter(bank=TemplateBank.from_masses([30.0], [25.0]), sample_rate_hertz=FS,
                       ifos=[gf.IFO.H1, gf.IFO.L1])
    t = cbc_templates([30.0], [25.0], FS, 3.0, 20.0)[0].astype(np.float64)
    band = mf._white_band_spectra(np.fft.rfft(t, n=mf.n_fft))
    g = np.fft.irfft(band, n=mf.n_fft)

    def at(merger_samples):
        s = mf.template_merger_index - merger_samples
        w = g[s:s + N]
        return w / np.sqrt(np.sum(w ** 2))

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


def test_find_triggers_clusters_each_example():
    peak = np.zeros((1, 1000), dtype=np.float32)
    peak[0, 400:460] = 20.0 * np.hanning(60)
    assert len(find_triggers(peak, threshold=8.0, cluster_window=200)) == 1
    batched = find_triggers(np.stack([peak, peak]), threshold=8.0, cluster_window=200)
    assert [t["batch_index"] for t in batched] == [0, 1]
    run = np.zeros((1, 2000), dtype=np.float32)
    run[0, 100:1100] = 9.0
    run[0, 600] = 12.0
    trig = find_triggers(run, 8.0, 200)  # one long excursion is one trigger, at its peak
    assert len(trig) == 1 and trig[0]["time_index"] == 600
    two = np.zeros((1, 2000), dtype=np.float32)
    two[0, 100] = two[0, 500] = 9.0
    assert len(find_triggers(two, 8.0, 200)) == 2


def test_stochastic_bank_covers_its_range():
    bank = TemplateBank.stochastic(mass_range=(20.0, 40.0), convergence_candidates=512, use_bundled=False)
    rng = np.random.default_rng(8)
    m = rng.uniform(20.0, 40.0, size=(40, 2))
    ff = bank.fitting_factors(m.max(axis=1), m.min(axis=1))
    assert np.median(ff) >= 0.97 and np.min(ff) >= 0.95, (np.median(ff), np.min(ff))


def test_default_bank_covers_light_binaries():
    # The default bank (bundled) over 5-95 Msun: light binaries are covered too.
    bank = MatchedFilter(sample_rate_hertz=FS).bank
    assert bank.minimal_match == 0.97 and bank.num_templates > 136
    rng = np.random.default_rng(9)
    m = np.exp(rng.uniform(np.log(5.0), np.log(95.0), size=(48, 2)))
    m1, m2 = m.max(axis=1), m.min(axis=1)
    ff = bank.fitting_factors(m1, m2)
    light = (m1 + m2) < 30.0
    assert np.median(ff) >= 0.97 and np.percentile(ff, 5) >= 0.95, (np.median(ff), np.percentile(ff, 5))
    assert light.sum() >= 5 and np.median(ff[light]) >= 0.97


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
