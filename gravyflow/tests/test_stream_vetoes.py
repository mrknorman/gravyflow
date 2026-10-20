"""
The streaming filter's noise-robustness methods, each as PyCBC publishes it:
autogating, the PSD-variation statistic, Nitz's sine-Gaussian veto, complete
stored triggers whose ranking is recomputed without filtering, and
per-template exponential background fits.
"""
import numpy as np
import pytest

from gravyflow.src.detection import TemplateBank
from gravyflow.src.detection.background import exponential_fit, fit_templates, coincident_ranking
from gravyflow.src.detection.stream import DetectorTriggers, coincide
from gravyflow.src.detection.vetoes import (
    PSDVariation, gate, newsnr, psd_variation, ranking_statistic, value_at)
from gravyflow.tests.test_stream_search import (
    FS, coloured_noise, exact_spectrum, optimal_snr, search, signal, sine_gaussian)

HEAVY = ([60.0, 40.0], [45.0, 30.0], [0.0, 0.5], [0.0, 0.5])


def test_gating_is_pycbcs_gate_data():
    import pycbc.strain
    import pycbc.types
    rng = np.random.default_rng(40)
    strain = rng.standard_normal(int(20 * FS))
    gates = [int(5.3 * FS), int(19.9 * FS)]  # the second runs off the end
    mine = gate(strain, gates, FS, 0.25, 0.25)
    theirs = pycbc.strain.gate_data(pycbc.types.TimeSeries(strain.copy(), delta_t=1 / FS),
                                    [(g / FS, 0.25, 0.25) for g in gates])
    np.testing.assert_allclose(mine, theirs.numpy(), atol=1e-12)
    assert np.all(mine[int(5.1 * FS):int(5.5 * FS)] == 0.0)


def test_gating_follows_its_stated_sample_convention():
    # Widths in whole samples and gate times on whole nanoseconds (at 2048
    # Hz, samples that are multiples of 4): gate_data exactly, at any epoch.
    # At other samples gate_data, truncating its nanosecond time, starts one
    # sample earlier. A width that is not whole samples is floored.
    import pycbc.strain
    import pycbc.types
    ones = np.ones(int(20 * FS))
    theirs = lambda epoch, c, width, taper: pycbc.strain.gate_data(
        pycbc.types.TimeSeries(ones.copy(), delta_t=1 / FS, epoch=epoch), [(epoch + c / FS, width, taper)]).numpy()
    shifted = 0
    for epoch in (0.0, 1126259462.0):
        for width, taper in ((0.25, 0.25), (0.5, 0.125)):
            for c in range(int(5 * FS), int(5 * FS) + 200):
                mine = gate(ones, [c], FS, width, taper)
                if c % 4 == 0:
                    np.testing.assert_array_equal(mine, theirs(epoch, c, width, taper))
                elif not np.array_equal(mine, theirs(epoch, c, width, taper)):
                    np.testing.assert_array_equal(np.roll(mine, -1), theirs(epoch, c, width, taper))
                    shifted += 1
    assert shifted > 0
    c, width, taper = int(5 * FS), 0.1, 0.1  # 204.8 samples each
    mine = gate(ones, [c], FS, width, taper)
    assert np.all(mine[c - 204:c + 204] == 0.0) and mine[c - 205] > 0.0 and mine[c + 204] > 0.0
    assert np.all(mine[:c - 204 - 204] == 1.0) and np.all(mine[c + 204 + 204:] == 1.0)


def test_a_loud_glitch_is_gated_and_a_signal_elsewhere_is_not():
    # A glitch hundreds of whitened standard deviations loud is found, gated
    # and recorded; the triggers it made are gone and a signal 70 s away
    # keeps its SNR.
    rng = np.random.default_rng(41)
    seconds = 200.0
    noise = coloured_noise(rng, seconds)
    blip = sine_gaussian(60.0, seconds, frequency=150.0, q=6.0)
    blip *= 2000.0 / optimal_snr(blip, exact_spectrum())
    one = ([40.0], [30.0], [0.0], [0.2])
    chirp = signal(one, [130.0], seconds)
    chirp *= 12.0 / optimal_snr(chirp, exact_spectrum())
    bank = TemplateBank.from_masses(*one)
    ungated = search(bank, autogating=None).filter(noise + blip + chirp, 0.0, "L1")
    gated = search(bank).filter(noise + blip + chirp, 0.0, "L1")
    clean = search(bank).filter(noise + chirp, 0.0, "L1")
    near = lambda out: np.abs(out.time - 60.0) < 1.0
    assert np.max(ungated.snr[near(ungated)]) > 50.0
    assert len(gated.gates) >= 1 and np.min(np.abs(gated.gates - 60.0)) < 0.05
    assert not np.any(gated.snr[near(gated)] > 8.0)
    at = lambda out: np.max(out.snr[np.abs(out.time - 130.0) < 0.1])
    assert abs(at(gated) / at(clean) - 1.0) < 0.02 and len(clean.gates) == 0


def test_psd_variation_is_pycbcs():
    import pycbc.psd.variation
    import pycbc.types
    rng = np.random.default_rng(42)
    strain = coloured_noise(rng, 256.0)
    times, mine = psd_variation(strain, FS)
    theirs = pycbc.psd.variation.calc_filt_psd_variation(
        pycbc.types.TimeSeries(strain, delta_t=1 / FS, epoch=0.0), 8.0, 0.25, 256.0, 8.0, 4.0, "median", 20.0, 480.0)
    count = len(mine)
    np.testing.assert_allclose(times, theirs.sample_times.numpy()[:count])
    np.testing.assert_allclose(mine, theirs.numpy()[:count], rtol=1e-3)
    assert abs(np.median(mine) - 1.0) < 0.1


def test_a_louder_stretch_raises_the_psd_variation_and_lowers_the_ranking():
    # Noise twice as loud for 30 s: v there is about 4, near 1 elsewhere, and
    # its noise triggers' ranking falls to about half their SNR.
    rng = np.random.default_rng(43)
    seconds = 200.0
    noise = coloured_noise(rng, seconds)
    t = np.arange(len(noise)) / FS
    noise *= np.where((t > 90.0) & (t < 120.0), 2.0, 1.0)
    times, values = psd_variation(noise, FS)
    assert 2.5 < np.max(value_at(times, values, np.array([110.0, 115.0]))) < 6.0
    assert abs(np.median(value_at(times, values, np.array([40.0, 50.0, 160.0]))) - 1.0) < 0.3
    out = search(TemplateBank.from_masses(*HEAVY), snr_threshold=5.0, autogating=None).filter(noise, 0.0, "L1")
    loud = (out.time > 105.0) & (out.time < 118.0)
    assert np.any(loud) and np.all(out.psd_variation[loud] > 2.0)
    assert np.all(out.reweighted_snr[loud] < 0.75 * out.snr[loud])


def test_the_sine_gaussian_veto_follows_its_distribution_and_cuts_a_blip():
    # In Gaussian noise the sine-Gaussian chi-squared is a reduced chi-squared
    # with 14 degrees of freedom (mean 1); at a signal it stays near 1; a short
    # glitch above the template's peak frequency makes it large and lowers the
    # ranking below newsnr's.
    rng = np.random.default_rng(44)
    seconds = 400.0
    noise = coloured_noise(rng, seconds)
    bank = TemplateBank.from_masses(*HEAVY)
    out = search(bank, snr_threshold=3.5, autogating=None, psd_variation=None).filter(noise, 0.0, "L1")
    applied = out.sg_chisq != 1.0
    assert np.sum(applied) > 100 and abs(np.mean(out.sg_chisq[applied]) - 1.0) < 0.15
    chirp = signal(([60.0], [45.0], [0.0], [0.0]), [200.0], seconds)
    chirp *= 20.0 / optimal_snr(chirp, exact_spectrum())
    out = search(bank, autogating=None, psd_variation=None).filter(noise + chirp, 0.0, "L1")
    peak = np.argmax(out.snr)
    # (A loud signal's ringdown leaks some power into the lowest tiles; below
    # the threshold of 4 the ranking is newsnr's.)
    assert out.snr[peak] > 18.0 and out.sg_chisq[peak] < 4.0
    assert out.reweighted_snr[peak] == pytest.approx(newsnr(out.snr[peak], out.reduced_chisq[peak]), rel=1e-6)
    blip = sine_gaussian(200.0, seconds, frequency=350.0, q=5.0)
    blip *= 40.0 / optimal_snr(blip, exact_spectrum())
    out = search(bank, autogating=None, psd_variation=None).filter(noise + blip, 0.0, "L1")
    near = np.abs(out.time - 200.0) < 0.2
    loud = np.flatnonzero(near)[np.argmax(out.snr[near])]
    assert out.sg_chisq[loud] > 4.0, out.sg_chisq[loud]
    assert out.reweighted_snr[loud] < newsnr(out.snr[loud], out.reduced_chisq[loud])


def test_stored_triggers_recompute_their_ranking(tmp_path):
    # Every input of the ranking is stored: saved and loaded, the triggers
    # derive their ranking exactly, and without the corrections it is newsnr.
    rng = np.random.default_rng(45)
    noise = coloured_noise(rng, 200.0)
    bank = TemplateBank.from_masses(*HEAVY)
    out = search(bank, snr_threshold=5.0).filter(noise, 1000.0, "L1")
    out.save(tmp_path / "l1.npz")
    back = DetectorTriggers.load(tmp_path / "l1.npz", bank)
    for name in ("time", "template", "snr", "reduced_chisq", "reweighted_snr", "sg_chisq", "psd_variation", "gates"):
        np.testing.assert_array_equal(getattr(back, name), getattr(out, name))
    assert back.span == out.span and back.detector == "L1"
    np.testing.assert_array_equal(back.ranked().reweighted_snr, out.reweighted_snr)
    plain = back.ranked(sine_gaussian=False, psd_variation=False).reweighted_snr
    np.testing.assert_allclose(plain, newsnr(out.snr, out.reduced_chisq), rtol=1e-6)
    np.testing.assert_array_equal(
        out.reweighted_snr, ranking_statistic(out.snr, out.reduced_chisq, out.sg_chisq, out.psd_variation))


def test_the_ranking_is_pycbcs_newsnr_sgveto_psdvar():
    from pycbc.events.ranking import newsnr_sgveto_psdvar
    rng = np.random.default_rng(46)
    snr = rng.uniform(5, 30, 500)
    chisq = rng.uniform(0.2, 10, 500)
    sg = rng.uniform(0.2, 12, 500)
    v = rng.uniform(0.3, 3, 500)
    np.testing.assert_allclose(ranking_statistic(snr, chisq, sg, v), newsnr_sgveto_psdvar(snr, chisq, sg, v),
                               rtol=1e-6)


def background_triggers(rng, alphas, live_seconds, bank, threshold=6.0):
    """Synthetic background: per template, Poisson counts of exponential excesses above the threshold."""
    template, value = [], []
    for k, alpha in enumerate(alphas):
        n = rng.poisson(400)
        template.append(np.full(n, k))
        value.append(threshold + rng.exponential(1.0 / alpha, n))
    value = np.concatenate(value).astype(np.float32)
    return DetectorTriggers("H1", np.sort(rng.uniform(0, live_seconds, len(value))), np.concatenate(template),
                            value, np.ones_like(value), (0.0, live_seconds), bank=bank)  # ranked = value


def test_exponential_fits_are_pycbcs_and_recover_their_slopes():
    from pycbc.events.trigger_fits import fit_above_thresh
    rng = np.random.default_rng(47)
    values = 6.0 + rng.exponential(1 / 3.0, 2000)
    alpha, error, count = exponential_fit(np.concatenate([values, [5.0, 5.5]]), 6.0)
    theirs = fit_above_thresh("exponential", values, 6.0)
    np.testing.assert_allclose([alpha, error], theirs, rtol=1e-9)
    assert count == 2000 and abs(alpha - 3.0) < 4 * error
    # Per template, unsmoothed (a width below the templates' spacing), the
    # fits find each template's own slope and rate.
    bank = TemplateBank.from_masses([10.0, 30.0, 80.0], [8.0, 25.0, 70.0])
    triggers = background_triggers(rng, [2.0, 4.0, 6.0], 1000.0, bank)
    fits = fit_templates(triggers, bank, threshold=6.0, smoothing_width=1e-6, lower_frequency_hertz=20.0)
    np.testing.assert_allclose(fits.alpha, [2.0, 4.0, 6.0], rtol=0.2)
    np.testing.assert_allclose(fits.rate, fits.count / 1000.0)
    # Smoothed over all three, they pool: one slope, the mean rate.
    wide = fit_templates(triggers, bank, threshold=6.0, smoothing_width=10.0, lower_frequency_hertz=20.0)
    pooled = np.sum(fits.count) / np.sum(fits.count / fits.alpha_raw)
    np.testing.assert_allclose(wide.alpha, pooled, rtol=1e-9)
    np.testing.assert_allclose(wide.rate, np.mean(fits.count) / 1000.0)


def test_the_fit_statistic_ranks_by_each_detectors_background():
    # ln r = -alpha (x - x_t) + ln alpha + ln rate (ExpFitStatistic); a
    # coincidence of a template with a steep background outranks one of
    # equal SNRs in a template with a long tail.
    rng = np.random.default_rng(48)
    bank = TemplateBank.from_masses([10.0, 80.0], [8.0, 70.0])
    fits = {d: fit_templates(background_triggers(rng, [5.0, 1.0], 1000.0, bank), bank, 6.0, 1e-6, 20.0)
            for d in ("H1", "L1")}
    x = np.array([8.0, 8.0])
    expected = -fits["H1"].alpha * 2.0 + np.log(fits["H1"].alpha) + np.log(fits["H1"].rate)
    np.testing.assert_allclose(fits["H1"].log_noise_rate([0, 1], x), expected)
    pair = {d: DetectorTriggers(d, [100.0, 300.0], [0, 1], x, [1.0, 1.0], (0.0, 1000.0), bank=bank) for d in ("H1", "L1")}
    network = coincide(pair)
    ranking = coincident_ranking(network, fits)
    assert ranking[network.template == 0][0] > ranking[network.template == 1][0]
