"""
MLGWSC-1's dataset 1, made by gravyflow and streamed (``gf.MLGWSC1Stream``): the challenge's
segments, its noise, its injection population and schedule, its injections' conditioning,
and a stream that is a function of its seed and GPS time alone.

The comparison with the challenge's own generate_data.py (its noise, its injection file and
its injected strain, run by PyCBC) is an instrument outside the suite; the values these
tests hold from it are named where they appear.
"""

import numpy as np
import pytest
from scipy import signal, stats

import gravyflow as gf
from gravyflow.src.dataset.acquisition import mlgwsc1, mlgwsc1_stream
from gravyflow.src.dataset.conditioning.detector import read_noise_curve
from gravyflow.src.dataset.noise.stationary import StationaryNoise
from gravyflow.src.rng import SeedTree

RATE = 2048


@pytest.fixture(scope="module")
def ds1():
    return gf.MLGWSC1Stream(1, seed=7)


def test_the_segments_are_the_challenges():
    """restrict_segments gives what generate_data.py's restrict_segments gives (its output, 2026-10-19)."""
    month = mlgwsc1.restrict_segments()
    assert len(month) == 129 and np.sum(np.diff(month, axis=1)) == 2_592_000
    assert month[0].tolist() == [1238205077, 1238215138] and month[-1].tolist() == [1244486004, 1244492948]
    assert mlgwsc1.restrict_segments(duration=4 * 3600).tolist() == [
        [1238205077, 1238215138], [1238400368, 1238404707]]
    assert mlgwsc1.restrict_segments(start_offset=86400, duration=6 * 3600).tolist() == [
        [1238664625, 1238672696], [1238677592, 1238687593], [1238731472, 1238735000]]
    with pytest.raises(ValueError, match="not the 31536000 s asked for"):
        mlgwsc1.restrict_segments(duration=31_536_000)


def test_the_population_is_ds1_ini():
    """Every parameter has ds1.ini's distribution (KS against its closed form), and the distance
    is the chirp distance scaled by the chirp mass to the 5 / 6."""
    sources = mlgwsc1_stream.Population().draw(SeedTree(3), 100_000)
    u = lambda x, lo, hi: (x - lo) / (hi - lo)  # noqa: E731
    cdfs = {
        "mass1": lambda x: u(x, 10, 50) ** 2,  # the larger of two uniform draws
        "mass2": lambda x: 1 - (1 - u(x, 10, 50)) ** 2,  # the smaller
        "ra": lambda x: u(x, 0, 2 * np.pi),
        "dec": lambda x: (np.sin(x) + 1) / 2,  # uniform on the sphere
        "inclination": lambda x: (1 - np.cos(x)) / 2,  # uniform in its cosine
        "coa_phase": lambda x: u(x, 0, 2 * np.pi),
        "polarization": lambda x: u(x, 0, 2 * np.pi),
        "chirp_distance": lambda x: u(x ** 3, 130 ** 3, 350 ** 3),  # uniform in volume
    }
    for name, cdf in cdfs.items():
        assert stats.kstest(sources[name], cdf).pvalue > 1e-3, name
    assert np.all(sources["mass2"] <= sources["mass1"])
    chirp = (sources["mass1"] * sources["mass2"]) ** 0.6 / (sources["mass1"] + sources["mass2"]) ** 0.2
    np.testing.assert_allclose(sources["distance"],
                               sources["chirp_distance"] * (chirp / (2 ** -0.2 * 1.4)) ** (5 / 6), rtol=1e-12)


def test_injections_come_every_24_to_30_seconds_inside_the_padded_segments(ds1):
    """The schedule spans the challenge's 362 segments (pycbc_create_injections' renewal: 24 s
    plus a uniform draw up to 6 s), and only the sources 30 s or more inside a segment are injected."""
    tc = ds1.schedule["tc"]
    start, end = mlgwsc1.segments()[0, 0], mlgwsc1.segments()[-1, 1]
    spacing = np.diff(tc)
    assert start <= tc[0] <= start + 6 and tc[-1] < end
    assert 24 <= spacing.min() and spacing.max() <= 30 and abs(spacing.mean() - 27) < 0.02
    assert stats.kstest(spacing - 24, "uniform", args=(0, 6)).pvalue > 1e-3
    inside = ds1.injections["tc"]
    held = np.zeros(len(inside), bool)
    for lo, hi in ds1.segments:
        held |= (inside >= lo + 30) & (inside <= hi - 30)
    assert held.all() and abs(len(inside) - 2_592_000 / 27) < 0.01 * 2_592_000 / 27
    # The table is the same whatever the extent: one injection file for the challenge's span.
    assert np.array_equal(gf.MLGWSC1Stream(1, seed=7, duration_seconds=7200).schedule, ds1.schedule)


def test_the_noise_has_the_curve_from_15_hz_and_none_below(ds1):
    """The background's Welch PSD is aLIGOZeroDetHighPower within its scatter from 16 Hz, and
    the colouring filter passes nothing below 15 Hz: under 1e-8 of its power at 15.3 Hz from
    14.9 Hz down (the Hann truncation spreads the edge over 1 / 128 s)."""
    kernel = ds1.spec.noise.kernel(gf.IFO.H1)
    response = np.abs(np.fft.rfft(kernel, n=8 * len(kernel))) ** 2
    grid = np.fft.rfftfreq(8 * len(kernel), 1.0 / RATE)
    assert np.max(response[grid < 14.9]) < 1e-8 * np.median(response[(grid > 15.2) & (grid < 15.4)])
    start = float(ds1.segments[0, 0])
    strain = ds1.read(start + 1000, start + 2024, injections=False).strain
    frequencies, power = signal.welch(strain, fs=RATE, nperseg=4 * RATE, window="hann")
    curve_f, curve_asd = read_noise_curve(gf.NoiseProfile.ZERO_DET_HIGH_POWER.path(gf.IFO.H1), up_to_hertz=1024)
    curve = np.interp(frequencies, curve_f, curve_asd) ** 2
    for lo, hi in ((16, 32), (32, 128), (128, 512), (512, 1000)):
        band = (frequencies >= lo) & (frequencies < hi)
        ratio = np.mean(power[:, band] / curve[band], axis=1)
        # 511 segments of 4 s: each bin's mean scatters by 4.5 %; a band of n bins by 4.5 % / sqrt(n).
        assert np.all(np.abs(ratio - 1) < 5 * 0.045 / np.sqrt(band.sum())), (lo, hi, ratio)
    assert abs(np.corrcoef(strain)[0, 1]) < 0.02


def test_a_stream_is_a_function_of_its_seed_and_time():
    """Separate streams of one seed read the same bytes, however the span is cut and in any
    order; another seed reads others, and the fingerprint says which."""
    first = gf.MLGWSC1Stream(1, seed=11, duration_seconds=7200)
    start = float(first.segments[0, 0]) + 400.0  # the span crosses a 512 s noise block
    whole = first.read(start, start + 300.0).strain
    again = gf.MLGWSC1Stream(1, seed=11, duration_seconds=7200)
    cuts = [0.0, 0.5, 121.25, 300.0]
    pieces = [again.read(start + a, start + b).strain for a, b in zip(cuts[:-1], cuts[1:])][::-1]
    assert np.array_equal(np.concatenate(pieces[::-1], axis=1), whole)
    other = gf.MLGWSC1Stream(1, seed=12, duration_seconds=7200)
    assert np.std(other.read(start, start + 300.0).strain - whole) > np.std(whole)
    assert first.fingerprint() == again.fingerprint() != other.fingerprint()
    assert first.fingerprint() != gf.MLGWSC1Stream(1, seed=11).fingerprint()  # the extent is named
    assert first.mode == mlgwsc1_stream.SIMULATED and "no reservation" in first.mode


def test_each_injection_arrives_at_its_tc_on_each_detector(ds1):
    """The foreground less the background is the injections: each peaks at its tc plus the
    detector's delay, and nothing is there away from them. Over 20 s before a source's tc to
    4 s after, no other source has strain: injections are at least 24 s apart, and the
    longest (10 + 10 solar masses) starts under 10 s before its merger."""
    tc = ds1.injections["tc"]
    source = ds1.injections[np.argmax(tc > ds1.segments[0, 0] + 200)]
    start = np.floor(source["tc"]) - 20.0
    injected = ds1.read(start, start + 24.0).strain - ds1.read(start, start + 24.0, injections=False).strain
    network = gf.Network([gf.IFO.H1, gf.IFO.L1])
    angles = [np.array([source[n]], np.float32) for n in ("ra", "dec")]
    delays = np.asarray(network.get_time_delay(*angles, gps_time_seconds=np.array([source["tc"]])))[0]
    for d in range(2):
        envelope = np.abs(signal.hilbert(injected[d]))
        peak = start + np.argmax(envelope) / RATE
        # IMRPhenomXAS peaks within a few milliseconds of its t = 0.
        assert abs(peak - (source["tc"] + delays[d])) < 0.01, (d, peak, source["tc"], delays[d])
    # Before the source's waveform starts, only the float32 fractional shift's rounding
    # (about 2e-5 of the peak) is there.
    times = start + np.arange(injected.shape[1]) / RATE
    assert np.max(np.abs(injected[:, times < source["tc"] - 12.0])) < 1e-4 * np.max(np.abs(injected))


def test_injections_are_conditioned_as_pycbc_conditions_the_challenges():
    """The injector's spectrum is the model's, whole from 20 Hz, rolled up by the rising half of
    a Kaiser window (beta 8) from the challenge's taper start, and nothing below it."""
    injector = mlgwsc1_stream.DATASETS[1].injector
    start = mlgwsc1_stream.taper_start_hertz(20.0)
    assert abs(start - 20.0 * 0.99 ** 17) < 1e-9  # 1.5 times as long, by 1 % steps
    sources = {name: np.full(injector.BATCH, value) for name, value in dict(
        mass1=10.0, mass2=10.0, distance=500.0, inclination=0.3, coa_phase=0.4).items()}
    plus = injector.polarisations(sources, float(RATE))[0, 0]
    spectrum = np.abs(np.fft.rfft(plus))
    f = np.fft.rfftfreq(len(plus), 1.0 / RATE)
    level = np.median(spectrum[(f > 20.5) & (f < 21.5)] * f[(f > 20.5) & (f < 21.5)] ** (7 / 6))
    at = lambda hz: np.median(spectrum[np.abs(f - hz) < 0.1] * hz ** (7 / 6)) / level  # noqa: E731
    kaiser = lambda hz: np.i0(8 * np.sqrt(1 - ((20 - hz) / (20 - start)) ** 2)) / np.i0(8)  # noqa: E731
    assert at(16.0) < 1e-3 and at(25.0) == pytest.approx(1.0, abs=0.03)
    for hz in (18.0, 18.5, 19.0):
        assert at(hz) == pytest.approx(kaiser(hz), abs=0.03), hz


def test_refusals():
    """Another dataset, a read outside the segments or off the sample grid, and a noise filter
    of an odd length are each refused by name."""
    with pytest.raises(ValueError, match="not dataset 2"):
        gf.MLGWSC1Stream(2)
    stream = gf.MLGWSC1Stream(1, seed=0, duration_seconds=7200)
    start = float(stream.segments[0, 0])
    with pytest.raises(ValueError, match="not inside one of"):
        stream.read(start - 1.0, start + 1.0)
    with pytest.raises(ValueError, match="sample grid"):
        stream.read(start + 0.0001, start + 1.0)
    with pytest.raises(ValueError, match="even number of samples"):
        StationaryNoise(gf.NoiseProfile.ZERO_DET_HIGH_POWER, filter_seconds=1.0 / 2048)
    with pytest.raises(ValueError, match="no curve for V1"):
        StationaryNoise(gf.NoiseProfile.ZERO_DET_HIGH_POWER).kernel(gf.IFO.V1)
