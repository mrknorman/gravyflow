"""
LAL parity of gravyflow's CBC waveforms.

For each approximant, the generator (``_generate_cbc_waveform_jit``) produces a
few sources end to end, and each polarisation is matched against
``lalsimulation.SimInspiralChooseFDWaveform`` for the same source. The match is
maximised over time and phase and weighted by the aLIGO zero-detuned
high-power PSD above the band's lower frequency (and below the ISCO frequency
for TaylorF2, LAL's cutoff for that model).

In the time domain, the waveforms the dataset injects are matched the same way
against ``lalsimulation.SimInspiralTD``, LAL's own time-domain waveforms, across
total mass, mass ratio and spin.

lal is a required input of this test: if it cannot be imported, the test
fails and says so. It is never skipped.
"""
import math

import numpy as np
import pytest

from gravyflow.src.dataset.features.waveforms.cbc import (
    Approximant,
    _generate_cbc_waveform_jit,
    _generate_cbc_window,
    _taper_band,
)

SAMPLE_RATE = 4096.0
F_REF = 20.0
PHASE = 0.3
DISTANCE_MPC = 400.0
MINIMUM_MATCH = 0.9999

# A band: the window, the merger time within it, and the match's lower frequency.
BINARY_BLACK_HOLE = dict(duration=32.0, coalescence_time=30.0, f_low=20.0)
BINARY_NEUTRON_STAR = dict(duration=64.0, coalescence_time=60.0, f_low=40.0)

Z = (0.0, 0.0, 0.0)
# (label, m1, m2, spin1, spin2, lambda1, lambda2, inclination)
ALIGNED = [
    ("36+29, aligned spins", 36.0, 29.0, (0, 0, 0.3), (0, 0, -0.2), 0.0, 0.0, 0.4),
    ("10+10, spins .8/.8", 10.0, 10.0, (0, 0, 0.8), (0, 0, 0.8), 0.0, 0.0, 1.0),
    ("50+10, spins -.8/.5", 50.0, 10.0, (0, 0, -0.8), (0, 0, 0.5), 0.0, 0.0, 0.8),
]
PRECESSING = [
    ("36+29, precessing", 36.0, 29.0, (0.2, 0.1, 0.3), (-0.1, 0.2, -0.2), 0.0, 0.0, 0.4),
    ("10+10, |spins| .78/.76", 10.0, 10.0, (0.6, 0.0, 0.5), (0.0, 0.7, -0.3), 0.0, 0.0, 1.0),
    ("50+10, |spin| .79", 50.0, 10.0, (0.7, 0.2, 0.3), (0.1, -0.3, 0.2), 0.0, 0.0, 0.8),
]
TIDAL = [
    ("1.4+1.4, lambdas 400", 1.4, 1.4, Z, Z, 400.0, 400.0, 0.4),
    ("1.6+1.2, lambdas 200/800", 1.6, 1.2, (0, 0, 0.05), (0, 0, -0.02), 200.0, 800.0, 1.0),
    ("2.0+1.4, lambdas 50/600, spins .3/.2", 2.0, 1.4, (0, 0, 0.3), (0, 0, 0.2), 50.0, 600.0, 1.2),
]

CASES = {
    Approximant.IMRPhenomD: (ALIGNED, BINARY_BLACK_HOLE),
    Approximant.IMRPhenomXAS: (ALIGNED, BINARY_BLACK_HOLE),
    Approximant.IMRPhenomPv2: (PRECESSING, BINARY_BLACK_HOLE),
    Approximant.TaylorF2: (TIDAL, BINARY_NEUTRON_STAR),
    Approximant.IMRPhenomD_NRTidalv2: (TIDAL, BINARY_NEUTRON_STAR),
}

# The time domain. gravyflow's waveforms come from the generator the dataset
# injects with, starting at the band's lower frequency, on a window that holds
# each source's whole inspiral from it. The reference is
# lalsimulation.SimInspiralTD, LAL's own time-domain waveform for the same source.
TD_SAMPLE_RATE = 2048.0
TD_MINIMUM_MATCH = 0.999
TD_BINARY_BLACK_HOLE = dict(window=64.0, f_low=20.0)
TD_BINARY_NEUTRON_STAR = dict(window=128.0, f_low=40.0)

# Total masses 12 to 190, mass ratios 1 to 8 and spins up to 0.8, within the
# bundled template bank's 5 to 95 solar masses per component.
TD_ALIGNED = [
    ("6+6, spins .8/.8", 6.0, 6.0, (0, 0, 0.8), (0, 0, 0.8), 0.0, 0.0, 0.4),
    ("15+5, spins .3/0", 15.0, 5.0, (0, 0, 0.3), Z, 0.0, 0.0, 1.0),
    ("30+10, spins -.8/.5", 30.0, 10.0, (0, 0, -0.8), (0, 0, 0.5), 0.0, 0.0, 0.8),
    ("30+30, no spins", 30.0, 30.0, Z, Z, 0.0, 0.0, 0.0),
    ("51.7+10.4, no spins", 51.7, 10.4, Z, Z, 0.0, 0.0, 0.0),
    ("88+11, spins .5/-.3", 88.0, 11.0, (0, 0, 0.5), (0, 0, -0.3), 0.0, 0.0, 1.2),
    ("95+95, spins .5/.5", 95.0, 95.0, (0, 0, 0.5), (0, 0, 0.5), 0.0, 0.0, 0.4),
]
TD_PRECESSING = [
    ("6+6, |spins| .71/.68", 6.0, 6.0, (0.5, 0.2, 0.46), (0.1, -0.6, 0.3), 0.0, 0.0, 1.0),
    ("30+10, |spin| .79", 30.0, 10.0, (0.7, 0.2, 0.3), (0.1, -0.3, 0.2), 0.0, 0.0, 0.8),
    ("51.7+10.4, |spin| .78", 51.7, 10.4, (0.6, 0.0, 0.5), (0.0, 0.4, -0.3), 0.0, 0.0, 0.5),
    ("88+11, |spin| .64", 88.0, 11.0, (0.4, 0.4, 0.3), Z, 0.0, 0.0, 1.2),
    ("95+95, precessing", 95.0, 95.0, (0.3, 0.1, 0.5), (-0.2, 0.3, 0.4), 0.0, 0.0, 0.4),
]

TD_CASES = {
    Approximant.IMRPhenomD: (TD_ALIGNED, TD_BINARY_BLACK_HOLE),
    Approximant.IMRPhenomXAS: (TD_ALIGNED, TD_BINARY_BLACK_HOLE),
    Approximant.IMRPhenomPv2: (TD_PRECESSING, TD_BINARY_BLACK_HOLE),
    Approximant.TaylorF2: (TIDAL, TD_BINARY_NEUTRON_STAR),
    Approximant.IMRPhenomD_NRTidalv2: (TIDAL, TD_BINARY_NEUTRON_STAR),
}


def _import_lal():
    try:
        import lal
        import lalsimulation
    except ImportError as error:
        pytest.fail(
            "lal and lalsimulation are required inputs of the LAL parity test and "
            f"cannot be imported ({error}). Install lalsuite in this environment."
        )
    return lal, lalsimulation


def _lal_source(lal, lalsimulation, point):
    """LAL's leading arguments for ``point``, through the phase, and its tidal settings."""
    _, m1, m2, s1, s2, lambda_1, lambda_2, inclination = point
    settings = lal.CreateDict()
    if lambda_1 or lambda_2:
        lalsimulation.SimInspiralWaveformParamsInsertTidalLambda1(settings, lambda_1)
        lalsimulation.SimInspiralWaveformParamsInsertTidalLambda2(settings, lambda_2)
    arguments = (
        m1 * lal.MSUN_SI, m2 * lal.MSUN_SI,
        *map(float, s1), *map(float, s2),
        DISTANCE_MPC * 1e6 * lal.PC_SI, inclination, PHASE, 0.0, 0.0, 0.0,
    )
    return arguments, settings


def _lal_polarisations(lal, lalsimulation, approximant, point, f_low, df, n_freq):
    arguments, settings = _lal_source(lal, lalsimulation, point)
    hp, hc = lalsimulation.SimInspiralChooseFDWaveform(
        *arguments, df, f_low, SAMPLE_RATE / 2, F_REF, settings,
        lalsimulation.GetApproximantFromString(approximant.value),
    )
    polarisations = []
    for series in (hp, hc):
        data = np.zeros(n_freq, dtype=complex)
        length = min(n_freq, series.data.length)
        data[:length] = series.data.data[:length]
        polarisations.append(data)
    return polarisations


def _weight(lal, lalsimulation, approximant, point, freqs, f_low):
    _, m1, m2 = point[:3]
    keep = freqs >= f_low
    if approximant is Approximant.TaylorF2:
        f_isco = 1.0 / (6.0 ** 1.5 * math.pi * (m1 + m2) * lal.MTSUN_SI)
        keep &= freqs < f_isco
    weight = np.zeros_like(freqs)
    weight[keep] = [1.0 / lalsimulation.SimNoisePSDaLIGOZeroDetHighPower(f) for f in freqs[keep]]
    return weight


def _match(a, b, freqs, weight):
    """Match of ``a`` and ``b``, maximised over time and phase."""
    df = freqs[1] - freqs[0]
    integrand = np.conj(a) * b * weight
    norm = math.sqrt(np.sum(np.abs(a) ** 2 * weight) * np.sum(np.abs(b) ** 2 * weight))
    padding = 8
    n = padding * len(freqs)
    padded = np.zeros(n, dtype=complex)
    padded[: len(freqs)] = integrand
    series = np.abs(np.fft.ifft(padded) * n)
    peak = int(np.argmax(series))
    step = 1.0 / (n * df)
    # Refine the peak between samples with a golden-section search.
    lower, upper = (peak - 1) * step, (peak + 1) * step
    golden = (math.sqrt(5.0) - 1.0) / 2.0

    def overlap(t):
        return abs(np.sum(integrand * np.exp(2j * np.pi * freqs * t)))

    best = series[peak]
    for _ in range(40):
        left = upper - golden * (upper - lower)
        right = lower + golden * (upper - lower)
        if overlap(left) > overlap(right):
            upper = right
        else:
            lower = left
    best = max(best, overlap(0.5 * (lower + upper)))
    return best / norm


@pytest.mark.parametrize("approximant", list(CASES), ids=lambda a: a.value)
def test_cbc_waveforms_match_lal(approximant):
    lal, lalsimulation = _import_lal()
    points, band = CASES[approximant]
    duration, coalescence_time, f_low = band["duration"], band["coalescence_time"], band["f_low"]

    m1 = np.array([p[1] for p in points])
    m2 = np.array([p[2] for p in points])
    # With no minimum frequency, each waveform is tapered from the frequency it has
    # coalescence_time before merger, at the buffer's start; the taper must end
    # below the band.
    _, edge = _taper_band(m1, m2, np.full(len(points), coalescence_time))
    assert np.all(np.asarray(edge) < f_low), edge

    waveforms = _generate_cbc_waveform_jit(
        num_waveforms=len(points),
        sample_rate_hertz=SAMPLE_RATE,
        duration_seconds=duration,
        mass_1_msun=m1,
        mass_2_msun=m2,
        inclination_radians=np.array([p[7] for p in points]),
        distance_mpc=DISTANCE_MPC,
        reference_orbital_phase_in=PHASE,
        spin_1_in=np.array([p[3] for p in points], dtype=float),
        spin_2_in=np.array([p[4] for p in points], dtype=float),
        lambda_1=np.array([p[5] for p in points]),
        lambda_2=np.array([p[6] for p in points]),
        approximant=approximant,
        f_ref=F_REF,
        coalescence_time=coalescence_time,
    )
    waveforms = np.asarray(waveforms, dtype=np.float64)
    assert np.all(np.isfinite(waveforms))

    n_freq = int(SAMPLE_RATE * duration) // 2 + 1
    freqs = np.arange(n_freq) / duration
    failures = []
    for index, point in enumerate(points):
        reference = _lal_polarisations(lal, lalsimulation, approximant, point, f_low, 1.0 / duration, n_freq)
        weight = _weight(lal, lalsimulation, approximant, point, freqs, f_low)
        for polarisation, name in enumerate(("plus", "cross")):
            spectrum = np.fft.rfft(waveforms[index, polarisation]) / SAMPLE_RATE
            match = _match(reference[polarisation], spectrum, freqs, weight)
            if not match >= MINIMUM_MATCH:
                failures.append(f"{point[0]}, {name}: match {match:.7f}")
    assert not failures, f"{approximant.value} below {MINIMUM_MATCH} against LAL: {failures}"


def _lal_time_domain(lal, lalsimulation, approximant, point, f_low):
    """LAL's time-domain plus and cross polarisations of ``point``, shape (2, samples)."""
    arguments, settings = _lal_source(lal, lalsimulation, point)
    hp, hc = lalsimulation.SimInspiralTD(
        *arguments, 1.0 / TD_SAMPLE_RATE, f_low, F_REF, settings,
        lalsimulation.GetApproximantFromString(approximant.value),
    )
    return np.stack([hp.data.data, hc.data.data])


@pytest.mark.parametrize("approximant", list(TD_CASES), ids=lambda a: a.value)
def test_cbc_time_domain_waveforms_match_lal(approximant):
    lal, lalsimulation = _import_lal()
    points, band = TD_CASES[approximant]
    window, f_low = band["window"], band["f_low"]
    count = len(points)

    waveforms = _generate_cbc_window(
        num_waveforms=count,
        sample_rate_hertz=TD_SAMPLE_RATE,
        window_seconds=window,
        min_frequency_hertz=np.full(count, f_low),
        approximant=approximant,
        f_ref=F_REF,
        mass_1_msun=np.array([p[1] for p in points]),
        mass_2_msun=np.array([p[2] for p in points]),
        inclination_radians=np.array([p[7] for p in points]),
        distance_mpc=np.full(count, DISTANCE_MPC),
        reference_orbital_phase_in=np.full(count, PHASE),
        spin_1_in=np.array([p[3] for p in points], dtype=float),
        spin_2_in=np.array([p[4] for p in points], dtype=float),
        lambda_1=np.array([p[5] for p in points]),
        lambda_2=np.array([p[6] for p in points]),
    )
    waveforms = np.asarray(waveforms, dtype=np.float64)
    assert np.all(np.isfinite(waveforms))

    failures = []
    for index, point in enumerate(points):
        reference = _lal_time_domain(lal, lalsimulation, approximant, point, f_low)
        # One grid, long enough that no time shift between the two wraps.
        size = 2 ** math.ceil(math.log2(waveforms.shape[-1] + reference.shape[-1]))
        freqs = np.fft.rfftfreq(size, 1.0 / TD_SAMPLE_RATE)
        weight = _weight(lal, lalsimulation, approximant, point, freqs, f_low)
        for polarisation, name in enumerate(("plus", "cross")):
            match = _match(
                np.fft.rfft(reference[polarisation], size),
                np.fft.rfft(waveforms[index, polarisation], size),
                freqs,
                weight,
            )
            if not match >= TD_MINIMUM_MATCH:
                failures.append(f"{point[0]}, {name}: match {match:.5f}")
    assert not failures, (
        f"{approximant.value} below {TD_MINIMUM_MATCH} against LAL's SimInspiralTD: {failures}"
    )
