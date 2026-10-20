"""
CBC windows from ``_generate_cbc_window``: batching, input forms, refusal, and
where the merger, the envelope's peak, sits against LALSimulation's waveforms.

How the waveforms match LALSimulation's is ``test_cbc_lal_parity.py``'s, and
the chirp-time helpers are ``test_cbc_units.py``'s.
"""
import lal
import lalsimulation
import numpy as np
import pytest
from keras import ops
from scipy.signal import correlate, correlation_lags, hilbert

from gravyflow.src.dataset.features.waveforms import cbc
from gravyflow.src.dataset.features.waveforms.cbc import _generate_cbc_window

SAMPLE_RATE = 2048.0
F_LOW = 20.0
F_REF = 20.0


def test_each_row_of_a_batch_is_its_source_generated_alone():
    """
    Row i of a batch is source i generated on its own, whatever form each input
    takes: per-row arrays, a flat spin list, a (num_waveforms, 3) spin array, a
    tensor, a Python float.
    """
    mass_1 = np.array([30.0, 12.0, 45.0])
    mass_2 = np.array([25.0, 8.0, 20.0])
    spin_1 = np.array([[0.1, 0.0, 0.3], [0.0, 0.2, -0.4], [0.3, -0.1, 0.5]])
    spin_2 = np.array([[0.0, 0.0, 0.2], [0.1, 0.0, 0.0], [-0.2, 0.2, 0.1]])
    inclination = np.array([0.3, 1.1, 2.0])
    distance = np.array([300.0, 500.0, 900.0])
    phase = np.array([0.1, 2.0, 4.0])

    batch = np.asarray(_generate_cbc_window(
        num_waveforms=3,
        sample_rate_hertz=SAMPLE_RATE,
        window_seconds=2.0,
        min_frequency_hertz=None,
        approximant="IMRPhenomPv2",
        mass_1_msun=mass_1,
        mass_2_msun=mass_2,
        spin_1_in=spin_1.ravel().tolist(),
        spin_2_in=spin_2,
        inclination_radians=inclination,
        distance_mpc=distance,
        reference_orbital_phase_in=phase,
    ))
    assert batch.shape == (3, 2, int(2.0 * SAMPLE_RATE))

    for i in range(3):
        alone = np.asarray(_generate_cbc_window(
            num_waveforms=1,
            sample_rate_hertz=SAMPLE_RATE,
            window_seconds=2.0,
            min_frequency_hertz=None,
            approximant="IMRPhenomPv2",
            mass_1_msun=ops.convert_to_tensor(mass_1[i]),
            mass_2_msun=float(mass_2[i]),
            spin_1_in=spin_1[i].tolist(),
            spin_2_in=spin_2[i],
            inclination_radians=float(inclination[i]),
            distance_mpc=float(distance[i]),
            reference_orbital_phase_in=float(phase[i]),
        ))[0]
        # Representation: one float32 result through an inverse FFT of
        # 2 ** 12 bins, so a few ulp per log2 N of the peak. Rows mixed up, or a
        # spin or scalar broadcast wrongly, differ at order the peak itself.
        tolerance = 4 * np.log2(alone.shape[-1]) * np.finfo(np.float32).eps * np.abs(alone).max()
        np.testing.assert_allclose(batch[i], alone, rtol=0.0, atol=tolerance, err_msg=f"row {i}")


def test_an_unknown_approximant_is_refused():
    with pytest.raises(ValueError, match="not supported"):
        _generate_cbc_window(
            num_waveforms=1,
            sample_rate_hertz=SAMPLE_RATE,
            window_seconds=1.0,
            min_frequency_hertz=None,
            approximant="InvalidApproximant",
        )


@pytest.mark.parametrize("name, spin_1, spin_2", [
    ("spin_1_in", [0.8, 0.0, 0.8], [0.0, 0.0, 0.0]),
    ("spin_2_in", [[0.0, 0.0, 0.1], [0.0, 0.0, 0.2]], [[0.0, 0.0, 0.0], [0.0, 0.0, -1.01]]),
])
def test_a_spin_above_one_is_refused_by_name(name, spin_1, spin_2):
    """A spin past the Kerr bound, |chi| > 1, is refused, naming the spin and its magnitude."""
    with pytest.raises(ValueError, match=rf"{name}.*\|chi\| <= 1"):
        _generate_cbc_window(
            num_waveforms=2 if np.ndim(spin_1) == 2 else 1,
            sample_rate_hertz=SAMPLE_RATE, window_seconds=1.0, min_frequency_hertz=None,
            approximant="IMRPhenomPv2", spin_1_in=spin_1, spin_2_in=spin_2,
        )
    # time_to_merger_seconds names its spins spin_1 and spin_2.
    with pytest.raises(ValueError, match=rf"{name[:6]}.*\|chi\| <= 1"):
        n = 2 if np.ndim(spin_1) == 2 else 1
        cbc.time_to_merger_seconds(
            20.0, "IMRPhenomPv2", np.full(n, 30.0), np.full(n, 30.0),
            np.reshape(spin_1, (n, 3)), np.reshape(spin_2, (n, 3)),
            np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n),
        )


def test_spins_at_the_kerr_bound_generate():
    """
    Spins of magnitude 1, as built from components (a unit vector rotated, a float32
    copy of one), are physical and generate: the bound allows their rounding.
    """
    phi = np.linspace(0.0, 2.0 * np.pi, 6, endpoint=False)
    x = np.sqrt(1.0 - 0.3 ** 2)
    spin_1 = np.stack([x * np.cos(phi), x * np.sin(phi), np.full(6, 0.3)], axis=-1).astype(np.float32)
    spin_2 = np.tile([0.0, 0.0, -1.0], (6, 1))
    waveforms = np.asarray(_generate_cbc_window(
        num_waveforms=6, sample_rate_hertz=SAMPLE_RATE, window_seconds=1.0, min_frequency_hertz=None,
        approximant="IMRPhenomPv2", spin_1_in=spin_1, spin_2_in=spin_2,
    ))
    assert np.all(np.isfinite(waveforms))


# The merger is the peak of the envelope, |analytic h+|^2 + |analytic hx|^2, at
# sample K // 2 of every K-sample window, for every approximant.
#
# Sources: (m1, m2, spin_1, spin_2, lambda_1, lambda_2, inclination). The
# binary-black-hole grids span the bundled banks' masses, 5 to 95 solar masses,
# with their corners, and spins to 0.8; the neutron-star grid spans binary
# neutron stars' masses and tidal deformabilities.
ALIGNED = [
    (5.0, 5.0, (0, 0, 0.0), (0, 0, 0.0), 0, 0, 0.0),
    (95.0, 5.0, (0, 0, 0.8), (0, 0, 0.0), 0, 0, 0.6),
    (95.0, 95.0, (0, 0, 0.8), (0, 0, 0.8), 0, 0, 0.3),
    (95.0, 95.0, (0, 0, -0.8), (0, 0, -0.8), 0, 0, 1.4),
    (30.0, 30.0, (0, 0, 0.0), (0, 0, 0.0), 0, 0, 1.0),
    (40.0, 21.0, (0, 0, -0.6), (0, 0, 0.3), 0, 0, 0.8),
    (60.0, 50.0, (0, 0, 0.2), (0, 0, -0.4), 0, 0, 1.2),
    (12.0, 5.0, (0, 0, 0.5), (0, 0, -0.8), 0, 0, 2.0),
]
# IMRPhenomPv2 reaches mass ratio 19 with aligned spin only: with in-plane spin
# (0.3 to 0.6) at mass ratio 18 to 19, LAL's Pv2 and ripple's alike have their
# envelope's largest value in the inspiral, 0.3 to 0.9 s before the merger and
# 2.5 to 50 times the merger's, so there the model has no merger peak to place.
PRECESSING = [
    (5.0, 5.0, (0.5, 0.2, 0.4), (0.1, -0.6, 0.3), 0, 0, 1.0),
    (95.0, 5.0, (0.0, 0.0, 0.5), (0.0, 0.0, 0.0), 0, 0, 0.7),
    (95.0, 10.0, (0.6, 0.0, 0.5), (0.0, 0.0, 0.0), 0, 0, 0.7),
    (95.0, 95.0, (0.3, 0.1, 0.5), (-0.2, 0.3, 0.4), 0, 0, 0.4),
    (40.0, 21.0, (0.7, 0.2, 0.3), (0.1, -0.3, 0.2), 0, 0, 0.8),
    (60.0, 50.0, (0.0, 0.0, 0.2), (0.0, 0.0, -0.4), 0, 0, 1.2),
]
NEUTRON_STARS = [
    (1.4, 1.3, (0, 0, 0.0), (0, 0, 0.0), 400, 600, 0.5),
    (2.0, 1.2, (0, 0, 0.05), (0, 0, -0.05), 100, 1500, 1.1),
    (1.2, 1.2, (0, 0, 0.0), (0, 0, 0.0), 2000, 2000, 0.0),
]
SOURCES = {
    "IMRPhenomD": ALIGNED,
    "IMRPhenomXAS": ALIGNED,
    "IMRPhenomPv2": PRECESSING,
    "IMRPhenomD_NRTidalv2": NEUTRON_STARS,
    # An inspiral-only model, so inspiral-dominated sources: at 95 + 5 it has
    # no merger, and its envelope's peak moves with the kept window.
    "TaylorF2": [ALIGNED[0], ALIGNED[4], ALIGNED[7], NEUTRON_STARS[0]],
}
WINDOWS_SECONDS = (1.0, 1.5)
# At 95 + 95 solar masses LAL's time-domain conditioning, which starts lower
# for heavier binaries than gravyflow's taper, reaches the merger and reshapes
# it, so these are compared with LAL's model (below) but not with
# ChooseTDWaveform's t = 0.
LAL_TD_CONDITIONED_APART = {(95.0, 95.0)}


def _envelope_and_power_peaks(plus, cross):
    """
    The samples where the envelope, |analytic h+|^2 + |analytic hx|^2, and the
    power, h+^2 + hx^2, peak. The analytic signal is scipy's, independent of
    the generator's.
    """
    plus, cross = np.asarray(plus, np.float64), np.asarray(cross, np.float64)
    envelope = np.abs(hilbert(plus)) ** 2 + np.abs(hilbert(cross)) ** 2
    return int(np.argmax(envelope)), int(np.argmax(plus ** 2 + cross ** 2))


def _lal_arguments(approximant, source):
    m1, m2, spin_1, spin_2, lambda_1, lambda_2, inclination = source
    parameters = lal.CreateDict()
    lalsimulation.SimInspiralWaveformParamsInsertTidalLambda1(parameters, float(lambda_1))
    lalsimulation.SimInspiralWaveformParamsInsertTidalLambda2(parameters, float(lambda_2))
    masses_spins = (m1 * lal.MSUN_SI, m2 * lal.MSUN_SI, *map(float, spin_1), *map(float, spin_2),
                    400e6 * lal.PC_SI, inclination, 0.0, 0.0, 0.0, 0.0)
    return masses_spins, parameters, lalsimulation.GetApproximantFromString(approximant)


def _lal_time_domain(approximant, source, window_seconds):
    """
    SimInspiralChooseTDWaveform at SAMPLE_RATE, with LAL's own conditioning:
    h+, hx and the sample of its t = 0.
    """
    masses_spins, parameters, model = _lal_arguments(approximant, source)
    hp, hc = lalsimulation.SimInspiralChooseTDWaveform(
        *masses_spins, 1.0 / SAMPLE_RATE, F_LOW, F_REF, parameters, model)
    # The epoch is a whole number of samples before t = 0, to LAL's rounding.
    return hp.data.data, hc.data.data, -int(round(float(hp.epoch) * SAMPLE_RATE))


def _lal_model(approximant, source, window_seconds):
    """
    SimInspiralChooseFDWaveform with gravyflow's conditioning (its taper below
    the band edge, on the window's buffer), inverse-transformed: LAL's model
    as gravyflow conditions it. h+, hx and the sample of the model's t = 0.
    """
    masses_spins, parameters, model = _lal_arguments(approximant, source)
    buffer_seconds = cbc._cbc_buffer_seconds(window_seconds)
    num_samples = int(buffer_seconds * SAMPLE_RATE)
    num_kept = int(window_seconds * SAMPLE_RATE)
    merger_seconds = buffer_seconds - (num_kept - num_kept // 2) / SAMPLE_RATE
    frequencies = np.fft.rfftfreq(num_samples, 1.0 / SAMPLE_RATE)
    hp, hc = lalsimulation.SimInspiralChooseFDWaveform(
        *masses_spins, 1.0 / buffer_seconds, 1.0, SAMPLE_RATE / 2, F_REF, parameters, model)
    start, edge = (float(f) for f in cbc._taper_band(source[0], source[1], merger_seconds, F_LOW))
    rise = np.clip((frequencies - start) / (edge - start), 0.0, 1.0)
    taper = np.where(frequencies <= start, 0.0, 0.5 - 0.5 * np.cos(np.pi * rise))
    # The model's t = 0 is sample 0; rolled mid-buffer so that nothing wraps.
    plus, cross = (
        np.roll(np.fft.irfft(h.data.data[: frequencies.size] * taper, n=num_samples), num_samples // 2)
        for h in (hp, hc)
    )
    return plus, cross, num_samples // 2


def _lal_sample_of(window, lal_plus, lal_cross):
    """
    The sample of LAL's waveform that sample 0 of ``window`` lies at, from the
    peak of their correlation maximised over a common phase: the lag of the
    largest |sum over polarisations of <analytic window, analytic LAL>|. It
    depends on neither waveform's time convention, and not on their phases.
    """
    correlation = sum(
        correlate(hilbert(np.asarray(lal_h, np.float64)), hilbert(np.asarray(ours, np.float64)), mode="full")
        for lal_h, ours in ((lal_plus, window[0]), (lal_cross, window[1]))
    )
    lags = correlation_lags(len(lal_plus), window.shape[-1], mode="full")
    return int(lags[np.argmax(np.abs(correlation))])


def _windows(approximant, sources, window_seconds):
    m1, m2, spin_1, spin_2, lambda_1, lambda_2, inclination = (
        np.asarray(values, dtype=float) for values in zip(*sources)
    )
    return np.asarray(_generate_cbc_window(
        num_waveforms=len(sources),
        sample_rate_hertz=SAMPLE_RATE,
        window_seconds=window_seconds,
        min_frequency_hertz=F_LOW,
        approximant=approximant,
        f_ref=F_REF,
        mass_1_msun=m1,
        mass_2_msun=m2,
        spin_1_in=spin_1,
        spin_2_in=spin_2,
        lambda_1=lambda_1,
        lambda_2=lambda_2,
        inclination_radians=inclination,
    ))


def merger_against_lal(approximant, window_seconds):
    """
    For each source of SOURCES[approximant], in samples: where the window's
    envelope peaks from its centre; and, against each LAL reference ("model":
    LAL's model as gravyflow conditions it; "td": ChooseTDWaveform), where the
    window's centre, its merger, lies from the reference's t = 0, and where
    the reference's envelope and power peak from it. TaylorF2 has no LAL
    time-domain reference.
    """
    sources = SOURCES[approximant]
    windows = _windows(approximant, sources, window_seconds)
    centre = windows.shape[-1] // 2
    references = {"model": _lal_model}
    if approximant != "TaylorF2":
        references["td"] = _lal_time_domain
    rows = []
    for source, window in zip(sources, windows):
        row = {"source": source, "envelope": _envelope_and_power_peaks(*window)[0] - centre}
        for name, reference in references.items():
            plus, cross, zero = reference(approximant, source, window_seconds)
            envelope, power = _envelope_and_power_peaks(plus, cross)
            row[name] = {
                "merger": _lal_sample_of(window, plus, cross) + centre - zero,
                "envelope": envelope - zero,
                "power": power - zero,
            }
        rows.append(row)
    return rows


@pytest.mark.parametrize("approximant", list(SOURCES))
def test_the_merger_is_the_envelope_peak_at_the_window_centre(approximant):
    """
    The envelope peaks at sample K // 2, exactly, for every source and window,
    and that sample is where the envelope of LAL's model peaks, to the sample.

    Against ChooseTDWaveform's t = 0, which LAL defines per model:
    - IMRPhenomD: LAL moves t = 0 to the peak sample of the face-on power,
      which is the envelope's, so the merger is LAL's t = 0, to the sample.
    - IMRPhenomXAS: LAL moves t = 0 to the power's peak sample at the source's
      inclination. The power, h+^2 + hx^2, oscillates at twice the phase, so
      its peak sits off the envelope's by LAL's own jitter (up to 3 samples on
      this grid), and the merger is within that jitter of LAL's t = 0.
    - IMRPhenomPv2 and IMRPhenomD_NRTidalv2: LAL keeps each model's own t = 0,
      which is not the peak, so the merger deliberately differs from LAL's
      t = 0, by as much as LAL's envelope peak does (up to 91 samples, 44 ms,
      for Pv2 on this grid).
    """
    failures = []
    for window_seconds in WINDOWS_SECONDS:
        for row in merger_against_lal(approximant, window_seconds):
            m1, m2, spin_1, spin_2 = row["source"][:4]
            name = f"{m1}+{m2} chi_z {spin_1[2]:+}/{spin_2[2]:+}, {window_seconds} s"
            if row["envelope"] != 0:
                failures.append(f"{name}: envelope peak {row['envelope']:+d} samples from the centre")
            # Each peak is the sample nearest the same continuous peak, on
            # grids aligned on different samples: one sample apart at most.
            model = row["model"]
            if abs(model["merger"] - model["envelope"]) > 1:
                failures.append(f"{name}: merger {model['merger']:+d}, LAL model's envelope peak {model['envelope']:+d}")
            if "td" not in row or (m1, m2) in LAL_TD_CONDITIONED_APART:
                continue
            td = row["td"]
            if abs(td["merger"] - td["envelope"]) > 1:
                failures.append(f"{name}: merger {td['merger']:+d}, ChooseTDWaveform's envelope peak {td['envelope']:+d}")
            if approximant == "IMRPhenomD" and abs(td["merger"]) > 1:
                failures.append(f"{name}: merger {td['merger']:+d} samples from ChooseTDWaveform's t = 0")
            if approximant == "IMRPhenomXAS" and abs(td["merger"]) > abs(td["envelope"] - td["power"]) + 1:
                failures.append(
                    f"{name}: merger {td['merger']:+d} from ChooseTDWaveform's t = 0, beyond LAL's "
                    f"jitter {td['envelope'] - td['power']:+d}"
                )
    assert not failures, f"{approximant}: {failures}"
