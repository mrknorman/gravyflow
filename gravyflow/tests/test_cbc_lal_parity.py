"""
LAL parity of gravyflow's CBC waveforms.

For each approximant, ``generate_cbc_waveform`` produces a few sources end to
end, and each polarisation is matched against
``lalsimulation.SimInspiralChooseFDWaveform`` for the same source. The match is
maximised over time and phase and weighted by the aLIGO zero-detuned
high-power PSD above the band's lower frequency (and below the ISCO frequency
for TaylorF2, LAL's cutoff for that model).

lal is a required input of this test: if it cannot be imported, the test
fails and says so. It is never skipped.
"""
import math

import numpy as np
import pytest

from gravyflow.src.dataset.features.waveforms.cbc import (
    Approximant,
    calc_minimum_frequency,
    generate_cbc_waveform,
)

SAMPLE_RATE = 4096.0
F_REF = 20.0
PHASE = 0.3
DISTANCE_MPC = 400.0
TAPER_WIDTH = 8.0  # Hz, gravyflow's low-frequency taper
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


def _lal_polarisations(lal, lalsimulation, approximant, point, f_low, df, n_freq):
    _, m1, m2, s1, s2, lambda_1, lambda_2, inclination = point
    settings = lal.CreateDict()
    if lambda_1 or lambda_2:
        lalsimulation.SimInspiralWaveformParamsInsertTidalLambda1(settings, lambda_1)
        lalsimulation.SimInspiralWaveformParamsInsertTidalLambda2(settings, lambda_2)
    hp, hc = lalsimulation.SimInspiralChooseFDWaveform(
        m1 * lal.MSUN_SI, m2 * lal.MSUN_SI,
        *map(float, s1), *map(float, s2),
        DISTANCE_MPC * 1e6 * lal.PC_SI, inclination, PHASE, 0.0, 0.0, 0.0,
        df, f_low, SAMPLE_RATE / 2, F_REF, settings,
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
    # gravyflow starts each waveform at the frequency it has coalescence_time before
    # merger and tapers it over TAPER_WIDTH; the taper must end below the band.
    f_min = np.asarray(calc_minimum_frequency(m1, m2, np.full(len(points), coalescence_time)))
    assert np.all(f_min + TAPER_WIDTH < f_low), f_min

    waveforms = generate_cbc_waveform(
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
