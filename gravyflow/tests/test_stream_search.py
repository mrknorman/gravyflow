"""
The streaming matched filter: per-detector triggers over continuous strain,
the chi-squared veto and reweighted SNR, coincidence, time slides and the
network triggers it emits.

Signals are the bank's own templates (`MatchSpace.templates`, from the
dataset's CBC generator), in Gaussian noise coloured by aLIGO's O4 high
projection, the curve the streaming filter's bank is placed on.
"""
import numpy as np
import pytest
from scipy import stats

import gravyflow as gf
from gravyflow.src.dataset.conditioning.whiten import whiten
from gravyflow.src.detection import TemplateBank
from gravyflow.src.detection.stream import (
    Coincidences, DetectorTriggers, StreamingMatchedFilter, bank_identity, coincide, reweighted_snr,
    stream_bank_placement)
from gravyflow.src.detection.template_grid import noise_curve_psd

FS = 2048.0
# A shorter spectrum and filter than the default (16 s), to keep the tests fast.
CONFIG = gf.SpectralConfig(lower_frequency_hertz=20.0, fft_duration_seconds=4.0, overlap_duration_seconds=2.0,
                           filter_duration_seconds=4.0)
SPACE = stream_bank_placement().space


def coloured_noise(rng, seconds):
    """Gaussian noise with aLIGO's O4 high curve (its table starts at 10.2 Hz), in strain."""
    n = int(seconds * FS)
    f = np.fft.rfftfreq(n, 1 / FS)
    amp = np.sqrt(noise_curve_psd(gf.IFO.L1, np.maximum(f, 10.3), gf.NoiseProfile.O4_HIGH) * n * FS / 4.0)
    spec = amp * (rng.standard_normal(len(f)) + 1j * rng.standard_normal(len(f)))
    spec[0] = 0.0
    return np.fft.irfft(spec, n=n)


def exact_spectrum(config=CONFIG):
    """The O4 high curve as a PowerSpectrum on the config's Welch grid, with no estimator to correct."""
    config = gf.SpectralConfig(**{**config.__dict__, "correct_inverse_bias": False})
    nperseg = config.nperseg(FS)
    f = np.fft.rfftfreq(nperseg, 1 / FS)
    true = noise_curve_psd(gf.IFO.L1, np.maximum(f, 10.3), gf.NoiseProfile.O4_HIGH) * FS
    return gf.PowerSpectrum(np.asarray(true / true.max(), np.float32)[None],
                            np.array([[np.sqrt(true.max())]], np.float32), FS, config, 1.0, 31)


def search(bank, config=CONFIG, **kwargs):
    return StreamingMatchedFilter(bank=bank, spectral_config=config, block_seconds=kwargs.pop("block_seconds", 64.0),
                                  **kwargs)


def signal(params, mergers_seconds, seconds, amplitude=1.0):
    """The bank-space templates of `params` (one row each) with their mergers at `mergers_seconds`, in a series."""
    out = np.zeros(int(seconds * FS))
    templates = SPACE.templates(*params).astype(np.float64)
    centre = templates.shape[-1] // 2
    for template, merger in zip(templates, mergers_seconds):
        start = int(round(merger * FS)) - centre  # the template's first sample in the series
        lo, hi = max(start, 0), min(start + len(template), len(out))
        out[lo:hi] += amplitude * template[lo - start:hi - start]
    return out


def optimal_snr(strain, spectrum):
    """The SNR a signal carries: the norm of the data path's whitening of it (away from the edges)."""
    white = np.asarray(whiten(strain.astype(np.float32), spectrum, FS), dtype=np.float64)
    edge = int(spectrum.config.filter_duration_seconds * FS)
    return float(np.sqrt(np.sum(white[edge:-edge] ** 2)))


SPINNING = ([14.0], [9.0], [0.6], [-0.4])


def test_the_reweighted_snr_is_pycbcs_newsnr():
    from pycbc.events.ranking import newsnr
    snr = np.array([6.0, 8.0, 10.0, 12.0, 30.0, 9.0])
    chisq = np.array([0.5, 1.0, 1.5, 3.0, 10.0, 0.0])
    np.testing.assert_allclose(reweighted_snr(snr, chisq), newsnr(snr, chisq), rtol=1e-6)
    assert reweighted_snr(10.0, 0.9) == 10.0 and reweighted_snr(10.0, 4.0) < 0.6 * 10.0


def test_a_template_recovers_its_signals_snr_at_its_merger_time():
    # A noise-free aligned-spin signal against its own template: the SNR the
    # data path's whitening gives it, at its merger sample, with chi-squared 0.
    # (Noise-free strain has no PSD variation to measure.)
    spectrum = exact_spectrum()
    mf = search(TemplateBank.from_masses(*SPINNING), psd_variation=None)
    strain = signal(SPINNING, [97.3], 200.0)
    strain *= 20.0 / optimal_snr(strain, spectrum)
    out = mf.filter(strain, 1000.0, "L1", psd=spectrum)
    assert len(out) == 1, out.snr
    assert abs(out.snr[0] / 20.0 - 1.0) < 2e-3, out.snr
    assert out.time[0] == 1000.0 + round(97.3 * FS) / FS
    assert out.reduced_chisq[0] < 1e-3 and out.reweighted_snr[0] == out.snr[0]


def test_each_merger_time_is_searched_once():
    # Overlapping blocks search consecutive spans: the searched span is one
    # stretch from the first block's first searchable merger time to the last
    # block's last, and a signal at any merger time, block edges included, is
    # one trigger. (Noise-free strain has no PSD variation to measure.)
    spectrum = exact_spectrum()
    mf = search(TemplateBank.from_masses(*SPINNING), psd_variation=None)
    seconds = 301.0
    n, first, stride = mf._layout(int(seconds * FS))
    starts = np.arange(0, int(seconds * FS) - n + 1, stride)
    edges = (starts + first + stride) / FS  # where each block's search ends and the next one's begins
    mergers = [edges[0] - 0.5 / FS, edges[1], edges[2] + 2.0 / FS, seconds - (n - first - stride) / FS - 2.0]
    amplitude = 20.0 / optimal_snr(signal(SPINNING, [mergers[1]], seconds), spectrum)
    strain = sum(signal(SPINNING, [merger], seconds, amplitude) for merger in mergers)
    out = mf.filter(strain, 0.0, "L1", psd=spectrum)
    assert out.span == (first / FS, seconds - (n - first - stride) / FS)
    np.testing.assert_allclose(np.sort(out.time), np.round(np.array(mergers) * FS) / FS, atol=0.5 / FS)
    np.testing.assert_allclose(out.snr, 20.0, rtol=2e-3)


def test_the_strain_must_hold_a_searchable_span():
    with pytest.raises(ValueError, match="no searchable merger time"):
        search(TemplateBank.from_masses(*SPINNING)).filter(np.zeros(int(10 * FS)), 0.0, "L1")
    with pytest.raises(ValueError, match="NaN"):
        search(TemplateBank.from_masses(*SPINNING)).filter(np.full(int(100 * FS), np.nan), 0.0, "L1")


def test_chisq_is_pycbcs_power_chisq():
    # PyCBC's power chi-squared (its bins, its formula, double precision) on
    # the same whitened data and whitened template, at every trigger: noise
    # and a signal. The filter weighs each bin by its own power where PyCBC
    # takes them as equal, and works in float32: measured agreement 0.2%.
    import pycbc.types
    from pycbc.vetoes import power_chisq
    from gravyflow.src.dataset.conditioning.whiten import whitening_filter
    rng = np.random.default_rng(21)
    spectrum = exact_spectrum()
    mf = search(TemplateBank.from_masses(*SPINNING), snr_threshold=4.0)
    n = int(64 * FS)  # one block: the strain is whitened as a whole
    chirp = signal(SPINNING, [40.0], 64.0)
    strain = coloured_noise(rng, 64.0) + chirp * (10.0 / optimal_snr(chirp, spectrum))
    out = mf.filter(strain, 0.0, "L1", psd=spectrum)
    assert len(out) >= 8 and np.max(out.snr) > 10.0
    white = np.asarray(whiten(strain.astype(np.float32), spectrum, FS), dtype=np.float64)
    fir = np.asarray(whitening_filter(spectrum, n), dtype=np.float64).reshape(-1)
    k = np.arange(n // 2 + 1)
    # The whitened template, its merger at index 0: whiten's alignment of the filter, and the template's merger.
    shift = (len(fir) - 1) // 2 + mf.pre
    template = np.fft.irfft(np.fft.rfft(np.asarray(mf.templates()[0], np.float64), n) * np.fft.rfft(fir, n)
                            * np.exp(2j * np.pi * k * shift / n), n)
    white_psd = pycbc.types.FrequencySeries(np.full(n // 2 + 1, 2.0 / FS), delta_f=FS / n)  # unit-variance noise
    theirs = power_chisq(pycbc.types.TimeSeries(template, delta_t=1 / FS), pycbc.types.TimeSeries(white, delta_t=1 / FS),
                         mf.chisq_bins, white_psd, low_frequency_cutoff=20.0)
    samples = np.round(out.time * FS).astype(int)
    np.testing.assert_allclose(out.reduced_chisq * mf.chisq_dof, np.asarray(theirs)[samples], rtol=0.01)


def test_chisq_follows_its_distribution_for_signals_in_gaussian_noise():
    # Allen's chi-squared for a signal its template matches, in Gaussian
    # noise with each block's PSD estimated from the block, is chi-squared
    # with 2p - 2 degrees of freedom. At a trigger the SNR is a peak, which
    # lowers chi-squared slightly (measured: mean 0.96 of its degrees of
    # freedom over 240 injections at SNR 15); the test bounds the mean to 10%
    # and the shape by Kolmogorov-Smirnov.
    rng = np.random.default_rng(15)
    spectrum = exact_spectrum()
    mf = search(TemplateBank.from_masses(*SPINNING))
    amplitude = 12.0 / optimal_snr(signal(SPINNING, [50.0], 100.0), spectrum)
    chisq, snr = [], []
    for _ in range(4):
        mergers = np.arange(30.0, 380.0, 25.0) + rng.uniform(0.0, 1.0, size=14)
        out = mf.filter(coloured_noise(rng, 400.0) + sum(signal(SPINNING, [m], 400.0, amplitude) for m in mergers),
                        0.0, "L1")
        found = [np.argmin(np.abs(out.time - m)) for m in mergers if np.min(np.abs(out.time - m)) < 0.01]
        chisq.extend(out.reduced_chisq[found] * mf.chisq_dof)
        snr.extend(out.snr[found])
    assert len(chisq) >= 54, len(chisq)  # SNR 12 against a threshold of 5.5: every one found
    assert abs(np.mean(snr) / 12.0 - 1.0) < 0.05, np.mean(snr)
    assert abs(np.mean(chisq) / mf.chisq_dof - 1.0) < 0.1, np.mean(chisq) / mf.chisq_dof
    assert stats.kstest(chisq, stats.chi2(mf.chisq_dof).cdf).pvalue > 1e-3


def sine_gaussian(centre_seconds, seconds, frequency=120.0, q=8.0):
    t = np.arange(int(seconds * FS)) / FS - centre_seconds
    tau = q / (np.sqrt(2.0) * np.pi * frequency)
    return np.exp(-(t / tau) ** 2) * np.sin(2.0 * np.pi * frequency * t)


def test_a_glitch_is_vetoed_and_a_signal_is_not():
    # A loud sine-Gaussian rings a high-mass template, whose SNR it passes;
    # its power is in one frequency bin, so chi-squared is large and the
    # reweighted SNR falls well below the SNR. A signal of the same SNR
    # keeps its SNR.
    rng = np.random.default_rng(16)
    params = ([60.0, 40.0], [45.0, 30.0], [0.0, 0.5], [0.0, 0.5])
    mf = search(TemplateBank.from_masses(*params))
    noise = coloured_noise(rng, 200.0)
    spectrum = exact_spectrum()
    glitch = sine_gaussian(100.0, 200.0)
    out = mf.filter(noise + glitch * (60.0 / optimal_snr(glitch, spectrum)), 0.0, "L1")
    loud = np.argmax(out.snr)
    assert out.snr[loud] > 10.0 and abs(out.time[loud] - 100.0) < 0.1, (out.snr[loud], out.time[loud])
    assert out.reweighted_snr[loud] < 0.5 * out.snr[loud], (out.snr[loud], out.reduced_chisq[loud])
    one = ([60.0], [45.0], [0.0], [0.0])
    chirp = signal(one, [100.0], 200.0)
    out = mf.filter(noise + chirp * (out.snr[loud] / optimal_snr(chirp, spectrum)), 0.0, "L1")
    loud_signal = np.argmax(out.snr)
    assert out.reweighted_snr[loud_signal] > 0.9 * out.snr[loud_signal]


#: The identity of a bank of eight templates, for synthetic triggers' indices.
EIGHT = bank_identity(TemplateBank.from_masses(np.arange(8) + 20.0, np.arange(8) + 10.0))


def triggers(detector, time, template, snr, span=(0.0, 1000.0)):
    snr = np.asarray(snr, dtype=np.float32)
    return DetectorTriggers(detector, time, template, snr, np.ones_like(snr), span, bank=EIGHT)  # ranked = snr


def test_a_coincidence_is_one_template_within_the_light_travel_time():
    # H1-L1: 10.0 ms of light travel plus the 2 ms margin.
    h1 = triggers("H1", [100.0, 200.0, 300.0, 400.0], [3, 3, 5, 7], [8.0, 9.0, 10.0, 11.0])
    l1 = triggers("L1", [100.0115, 200.0125, 300.001, 400.0, 400.005], [3, 3, 6, 7, 7], [6.0, 7.0, 8.0, 5.0, 9.0])
    found = coincide({"H1": h1, "L1": l1})
    # 200 s is 12.5 ms apart; 300 s has another template in L1; 400 s takes L1's loudest.
    np.testing.assert_array_equal(found.time, [100.0, 400.0])
    np.testing.assert_array_equal(found.template, [3, 7])
    np.testing.assert_allclose(found.stat, [np.hypot(8.0, 6.0), np.hypot(11.0, 9.0)], rtol=1e-6)
    np.testing.assert_array_equal(found.reweighted_snr["L1"], [6.0, 9.0])


def test_coincidences_are_clustered_in_time():
    h1 = triggers("H1", [100.0, 100.4, 101.9, 300.0], [1, 2, 3, 4], [8.0, 12.0, 9.0, 7.0])
    l1 = triggers("L1", [100.0, 100.4, 101.9, 300.0], [1, 2, 3, 4], [8.0, 12.0, 9.0, 7.0])
    found = coincide({"H1": h1, "L1": l1}, cluster_seconds=1.0)
    # 101.9 s is more than 1 s from the loudest (100.4 s), and loudest within its own +-1 s.
    np.testing.assert_array_equal(found.time, [100.4, 101.9, 300.0])


def test_a_time_slide_shifts_one_detectors_triggers_only():
    h1 = triggers("H1", [100.0, 250.0], [1, 2], [8.0, 9.0], span=(0.0, 1000.0))
    l1 = triggers("L1", [900.0, 50.0], [1, 2], [8.0, 9.0], span=(0.0, 1000.0))
    assert len(coincide({"H1": h1, "L1": l1})) == 0
    slid = l1.shifted(200.0, wrap=True)
    np.testing.assert_allclose(slid.time, [100.0, 250.0])
    assert slid.span == l1.span
    np.testing.assert_array_equal(coincide({"H1": h1, "L1": slid}).template, [1, 2])
    assert l1.shifted(200.0).span == (200.0, 1200.0)


def test_the_network_search_emits_mlgwsc_triggers(tmp_path):
    # Two detectors, one signal arriving 6 ms later at L1: one network
    # trigger at H1's merger time, its statistic the quadrature sum.
    rng = np.random.default_rng(17)
    spectrum = exact_spectrum()
    params = ([30.0], [22.0], [0.2], [0.1])
    mf = search(TemplateBank.from_masses(*params))
    chirp = signal(params, [80.0], 160.0)
    amplitude = 10.0 / optimal_snr(chirp, spectrum)
    strain = {gf.IFO.H1: coloured_noise(rng, 160.0) + amplitude * chirp,
              gf.IFO.L1: coloured_noise(rng, 160.0) + amplitude * signal(params, [80.006], 160.0)}
    result = mf.search(strain, 1e9)
    assert set(result.detectors) == {"H1", "L1"}
    network = result.coincidences
    loud = np.argmax(network.stat)
    assert abs(network.time[loud] - (1e9 + 80.0)) < 2e-3
    np.testing.assert_allclose(network.stat[loud], np.hypot(network.reweighted_snr["H1"][loud],
                                                            network.reweighted_snr["L1"][loud]), rtol=1e-5)
    assert network.stat[loud] > 11.0
    path = result.triggers(0.1).write(tmp_path / "triggers.hdf")
    read = gf.Triggers.read(path)
    np.testing.assert_array_equal(read.time, network.time)
    np.testing.assert_array_equal(read.var, 0.1)


def test_injections_scored_locally_equal_the_full_pass():
    # Each injection scored on a short stretch around it equals the full
    # pass over the strain with that injection added, trigger for trigger,
    # to float32 precision; a stretch shorter than the rule does not.
    from gravyflow.src.detection.stream import Injection
    rng = np.random.default_rng(30)
    spectrum = exact_spectrum()
    params = ([14.0, 40.0, 25.0], [9.0, 30.0, 6.0], [0.6, 0.0, 0.3], [-0.4, 0.2, -0.5])
    mf = search(TemplateBank.from_masses(*params), snr_threshold=5.0)
    seconds = 260.0
    noise = {"H1": coloured_noise(rng, seconds), "L1": coloured_noise(rng, seconds)}
    n, first, blocks = mf._blocks(int(seconds * FS))
    boundary = blocks[1][1] / FS  # where the second block's search begins
    mergers = [round(m * FS) / FS for m in (boundary + 0.05, 150.3, 200.7)]  # on the sample grid
    templates = SPACE.templates(*params).astype(np.float64)
    centre = templates.shape[-1] // 2
    injections = []
    for row, merger in zip(templates, mergers):
        wave = row * (rng.uniform(9.0, 15.0) / optimal_snr(row, spectrum))
        injections.append(Injection(merger, {"H1": merger - centre / FS, "L1": merger + (8 - centre) / FS},
                                    {"H1": wave, "L1": wave}))
    local = mf.score_injections(noise, 0.0, injections, window_seconds=0.5)
    for injection, result in zip(injections, local):
        full = {d: mf.filter(injection.add_to(noise[d].copy(), d, 0.0, FS), 0.0, d) for d in noise}
        for d, mine in result.detectors.items():
            near = (full[d].time >= mine.span[0]) & (full[d].time < mine.span[1])
            np.testing.assert_array_equal(mine.time, full[d].time[near])
            np.testing.assert_array_equal(mine.template, full[d].template[near])
            np.testing.assert_allclose(mine.snr, full[d].snr[near], rtol=1e-4)
            np.testing.assert_allclose(mine.reduced_chisq, full[d].reduced_chisq[near], rtol=1e-3)
            assert np.max(mine.snr) > 8.0
        network = coincide(full)
        loud = np.argmax(result.coincidences.stat)
        at = np.argmin(np.abs(network.time - result.coincidences.time[loud]))
        np.testing.assert_allclose(result.coincidences.stat[loud], network.stat[at], rtol=1e-4)
    # The stretch rule is tight: half a template span shorter, the SNR near the merger is not the full pass's.
    injection = injections[1]
    start, lo, hi = next(b for b in blocks if b[1] <= int(150.3 * FS) < b[2])
    block = injection.add_to(np.array(noise["H1"][start:start + n]), "H1", start / FS, FS)
    window = (int(149.8 * FS) - start, int(150.8 * FS) - start)
    chunked = mf._chunked(mf.templates())
    _, need = mf.stretch_samples(*window, first)
    exact = mf._search_block(block, chunked, None, first, *window)
    short = mf._search_block(block, chunked, None, first, *window, length=need - mf.pre // 2)
    assert len(short[0]) != len(exact[0]) or not np.allclose(short[2], exact[2], rtol=1e-4)


def assert_same_triggers(mine, theirs, snr_rtol=1e-5, chisq_rtol=1e-4):
    """Two runs' triggers are the same triggers, with SNR and chi-squared equal to float32 rounding."""
    np.testing.assert_array_equal(mine.time, theirs.time)
    np.testing.assert_array_equal(mine.template, theirs.template)
    np.testing.assert_allclose(mine.snr, theirs.snr, rtol=snr_rtol)
    np.testing.assert_allclose(mine.reduced_chisq, theirs.reduced_chisq, rtol=chisq_rtol, atol=1e-6)


def test_the_kernels_are_float32_whatever_the_processes_64_bit_mode():
    # ripplegw's XP models switch 64-bit mode on for the process at import:
    # traced then, chi-squared's bin edges would be float64, which an Apple
    # GPU cannot run. Under float32_on the kernels hold no 64-bit type.
    import jax
    import jax.numpy as jnp
    from gravyflow.src.detection.stream import _chisq, _snr_peaks, float32_on
    n = 4096
    spectrum = jnp.zeros(n // 2 + 1, jnp.complex64)
    chisq_args = (spectrum, spectrum, jnp.zeros((2, 512), jnp.float32), jnp.zeros((2, n // 2 + 1), jnp.float32))
    peak_args = (spectrum, spectrum, jnp.zeros((1, 2, 512), jnp.float32))
    lower = lambda: (_chisq.lower(*chisq_args, n=n, bins=4, band=(40, 1000)).as_text(),
                     _snr_peaks.lower(*peak_args, n=n, first=600, windows=4, width=256).as_text())
    with jax.enable_x64(True):
        bare = lower()
        with float32_on(jax.devices("cpu")[0]):
            scoped = lower()
    import re
    # A float64 or complex128 tensor anywhere in the program.
    wide = lambda text: re.findall(r"tensor<[0-9x]*(?:f64|complex<f64>)>", text)
    assert wide(bare[0])  # what the process-wide switch does to an unscoped kernel
    for text in scoped:
        assert not wide(text), wide(text)[:3]


def test_the_device_placed_path_equals_the_host_path():
    # The kernels on a device the caller names give the host path's
    # triggers, with SNR and chi-squared equal to float32 rounding, also with
    # the process in 64-bit mode. (The CPU stands in for an accelerator here;
    # the evidence holds the same comparison on an Apple GPU.)
    import jax
    rng = np.random.default_rng(31)
    params = ([14.0, 40.0, 25.0], [9.0, 30.0, 6.0], [0.6, 0.0, 0.3], [-0.4, 0.2, -0.5])
    bank = TemplateBank.from_masses(*params)
    strain = coloured_noise(rng, 200.0)
    wave = signal(([40.0], [30.0], [0.0], [0.2]), [120.4], 200.0)
    strain += wave * (12.0 / optimal_snr(wave, exact_spectrum()))
    host = search(bank, snr_threshold=5.0).filter(strain, 0.0, "L1")
    cpu = jax.devices("cpu")[0]
    placed = search(bank, snr_threshold=5.0, device=cpu)
    assert placed.templates().devices() == {cpu}
    assert_same_triggers(placed.filter(strain, 0.0, "L1"), host)
    with jax.enable_x64(True):
        wide = search(bank, snr_threshold=5.0, device=cpu).filter(strain, 0.0, "L1")
    assert_same_triggers(wide, host)
    assert len(host) >= 3 and np.max(host.snr) > 11.0


def test_a_bank_from_another_space_is_refused():
    bank = TemplateBank(np.array([30.0]), np.array([25.0]), placement=stream_bank_placement(sample_rate_hertz=4096.0))
    with pytest.raises(ValueError, match="4096"):
        StreamingMatchedFilter(bank=bank)
