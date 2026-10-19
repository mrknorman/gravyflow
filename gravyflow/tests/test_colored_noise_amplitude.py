"""
Amplitude of simulated (coloured) noise: its PSD is its profile's, on the
right frequency grid, in strain, from per-detector profiles that name their
source. The references are scipy's Welch estimate for the smooth curves, and
the whole series' periodogram for the measured curves, whose narrow lines a
windowed estimate smears.
"""
import hashlib

import numpy as np
import pytest
import scipy.signal

import gravyflow as gf
from gravyflow.src.dataset.noise import noise as gf_noise

FS = 2048.0
ALL_IFOS = [gf.IFO.H1, gf.IFO.L1, gf.IFO.V1]

# The LIGO-T2000012-v2 table each detector's O3_ACTUAL curve was written from,
# and the SHA-256 of the table as downloaded.
O3_ACTUAL_SOURCES = {
    gf.IFO.H1: ("aligo_O3actual_H1.txt", "553d04ee6f8e4d737e004f57d383382f6bdda6774f56b8ceb16c317a38c3b632"),
    gf.IFO.L1: ("aligo_O3actual_L1.txt", "730b33339283a8a677af2f52d4aa9876f09ffe28a52c45627f1e57687c6b869b"),
    gf.IFO.V1: ("avirgo_O3actual.txt", "951c0ebce8e994ce5a8a85b06794a481b0f4d465e1321a37288462c5bd19c8ae"),
}
O3_ACTUAL_URL = "https://dcc.ligo.org/public/0165/T2000012/002/"


def _profile(ifo, profile=None):
    path = ifo.value.optimal_psd_path if profile is None else profile.path(ifo)
    frequencies, asd = np.loadtxt(path, delimiter=",").T
    return frequencies, asd


def _profile_psd(ifo, frequencies, profile=None):
    f, asd = _profile(ifo, profile)
    return np.interp(frequencies, f, asd) ** 2


def _welch_ratio(x, ifos, sample_rate_hertz, profile=None):
    """Per-example Welch PSD divided by the profile PSD: (B, IFO, F), and F."""
    f, p = scipy.signal.welch(
        np.asarray(x, np.float64), fs=sample_rate_hertz,
        nperseg=int(4 * sample_rate_hertz)
    )
    reference = np.stack([_profile_psd(ifo, f, profile) for ifo in ifos])
    with np.errstate(divide="ignore", invalid="ignore"):
        return p / reference[None], f


#: Examples a NoiseObtainer test draws: 16 s each, 7 Welch segments, 224 in all.
OBTAINER_EXAMPLES = 32


def median_bound(segments, bins, k=4.0):
    """
    Statistical: the bound on |median over ``bins`` bins of Welch/profile - 1|, each
    bin the mean over ``segments`` half-overlapping Hann segments. A bin is about
    chi-square with nu = 2 segments / 1.056 degrees of freedom over nu (1.056: the
    variance excess of 50 %-overlapped Hann windows), whose median lies 2 / (3 nu)
    below its mean; the median of n independent bins has a standard error of
    1.2533 sigma / sqrt(n), and Hann's equivalent noise bandwidth of 1.5 bins
    leaves bins / 1.5 independent ones. The bound is the bias plus k such errors.
    """
    nu = 2.0 * segments / 1.056
    sigma = np.sqrt(1.056 / segments)
    return 2.0 / (3.0 * nu) + k * 1.2533 * sigma / np.sqrt(bins / 1.5)


def _assert_median_matches_the_profile(ratio, f, examples, duration):
    band = (f >= 30.0) & (f <= 900.0)
    segments = examples * (int((duration - 4.0) / 2.0) + 1)
    bound = median_bound(segments, int(band.sum()))
    median = np.median(ratio[:, 0, band].mean(axis=0))
    assert abs(median - 1.0) < bound, f"median Welch/profile = {median:.5g}, bound {bound:.4f}"


# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sample_rate_hertz", [2048.0, 8192.0])
def test_colored_noise_psd_matches_its_profile(sample_rate_hertz):
    """The coloured noise has its profile's PSD (O3_LOW, a smooth curve, to the
    4096 Hz Nyquist frequency of 8192 Hz): median within 1 % over 30-900 Hz
    and every bin within 5 standard errors, for every detector. Each bin
    averages over 7 half-overlapping Hann segments of each example, at least
    112 in all, so the median's own bias is under 0.4 %; a bin's standard error
    is sqrt(1.056 / segments), 1.056 being the variance excess of Welch's
    estimate for 50 %-overlapped Hann windows."""
    duration = 16.0
    num_examples = 32 if sample_rate_hertz == 2048.0 else 16
    batch = gf_noise.colored_noise_batcher(
        num_examples_per_batch=num_examples,
        onsource_duration_seconds=duration,
        crop_duration_seconds=0.0,
        offsource_duration_seconds=duration,
        ifos=ALL_IFOS,
        sample_rate_hertz=sample_rate_hertz,
        noise_profile=gf.NoiseProfile.O3_LOW,
    )
    x = batch(gf.SeedTree(7).batch("train", 0, 0))[gf.ReturnVariables.ONSOURCE]
    ratio, f = _welch_ratio(x, ALL_IFOS, sample_rate_hertz, gf.NoiseProfile.O3_LOW)
    band = (f >= 30.0) & (f <= 900.0)

    mean = ratio[..., band].mean(axis=0)  # (IFO, F)
    segments_per_example = int((duration - 4.0) / 2.0) + 1
    standard_error = np.sqrt(1.056 / (num_examples * segments_per_example))
    for i, ifo in enumerate(ALL_IFOS):
        median = np.median(mean[i])
        assert abs(median - 1.0) < 0.01, f"{ifo.name}: median Welch/profile = {median:.5g}"
        worst = np.max(np.abs(mean[i] - 1.0) / standard_error)
        assert worst < 5.0, f"{ifo.name}: worst bin is {worst:.2f} standard errors from the profile"


@pytest.mark.parametrize("sample_rate_hertz", [2048.0, 8192.0])
def test_default_noise_has_each_detectors_measured_o3_curve(sample_rate_hertz):
    """With no profile named, each detector's noise has its own measured O3
    curve (O3_ACTUAL, LIGO-T2000012), lines included: every rfft bin's power,
    averaged over the batch, is that detector's file's PSD at the bin. The
    periodogram of the whole series, unwindowed, has expectation exactly the
    PSD on the grid the noise is coloured on, so the check holds through the
    curves' narrow lines. A bin's ratio averages 32 exponential draws; a 1 Hz
    band's, 512 (standard error 0.044)."""
    duration, num_examples = 16.0, 32
    batch = gf_noise.colored_noise_batcher(
        num_examples_per_batch=num_examples,
        onsource_duration_seconds=duration,
        crop_duration_seconds=0.0,
        offsource_duration_seconds=duration,
        ifos=ALL_IFOS,
        sample_rate_hertz=sample_rate_hertz,
    )
    x = np.asarray(batch(gf.SeedTree(7).batch("train", 0, 0))[gf.ReturnVariables.ONSOURCE], np.float64)
    n = x.shape[-1]
    f = np.fft.rfftfreq(n, 1.0 / sample_rate_hertz)
    periodogram = 2.0 * np.abs(np.fft.rfft(x)) ** 2 / (sample_rate_hertz * n)
    band = (f >= 10.0) & (f < sample_rate_hertz / 2.0)
    per_hertz = int(round(duration))
    for i, ifo in enumerate(ALL_IFOS):
        frequencies, asd = np.loadtxt(gf.NoiseProfile.O3_ACTUAL.path(ifo), delimiter=",").T
        ratio = periodogram[:, i, band].mean(axis=0) / np.interp(f[band], frequencies, asd) ** 2
        assert abs(np.mean(ratio) - 1.0) < 0.01, f"{ifo.name}: mean periodogram/curve = {np.mean(ratio):.5g}"
        hertz = ratio[: len(ratio) // per_hertz * per_hertz].reshape(-1, per_hertz).mean(axis=1)
        worst = np.max(np.abs(hertz - 1.0)) / np.sqrt(1.0 / (num_examples * per_hertz))
        assert worst < 6.0, f"{ifo.name}: worst 1 Hz band is {worst:.2f} standard errors from the curve"
    # The detectors' curves are their own: H1's is not L1's.
    _, asd_h = _profile(gf.IFO.H1, gf.NoiseProfile.O3_ACTUAL)
    _, asd_l = _profile(gf.IFO.L1, gf.NoiseProfile.O3_ACTUAL)
    assert np.median(np.abs(np.log(asd_h / asd_l))) > 0.1


def test_a_curve_is_never_extrapolated():
    """A curve colours noise only on its own band: below it the noise has no
    power, as at DC, and a sample rate whose Nyquist frequency lies beyond it
    is refused, naming the band. The aLIGO O3 curves end at 5 kHz, Virgo's at
    10 kHz; Advanced Virgo's O3 low curve reaches the 8192 Hz Nyquist frequency of
    16384 Hz, and aLIGO's stops where its table does
    (``test_a_sample_rate_past_a_curves_table_is_refused``)."""
    def batcher(ifos, sample_rate_hertz, noise_profile=None):
        return gf_noise.colored_noise_batcher(
            num_examples_per_batch=1, onsource_duration_seconds=1.0, crop_duration_seconds=0.0,
            offsource_duration_seconds=1.0, ifos=ifos, sample_rate_hertz=sample_rate_hertz,
            noise_profile=noise_profile,
        )
    for ifo in (gf.IFO.H1, gf.IFO.L1):
        with pytest.raises(ValueError, match="10-5000 Hz"):
            batcher([ifo], 16384.0)
        batcher([ifo], 8192.0)
    batcher([gf.IFO.V1], 16384.0)
    batcher([gf.IFO.V1], 16384.0, gf.NoiseProfile.O3_LOW)

    x = np.asarray(batcher([gf.IFO.H1], FS)(gf.SeedTree(3).batch("train", 0, 0))[gf.ReturnVariables.ONSOURCE])
    spectrum = np.abs(np.fft.rfft(x.astype(np.float64)))
    f = np.fft.rfftfreq(x.shape[-1], 1.0 / FS)
    assert np.all(spectrum[..., f < 10.0] < 1e-6 * spectrum[..., f >= 10.0].max())


def test_psd_interpolation_uses_the_rfft_grid():
    """Each rfft bin gets the profile at its own frequency, k * fs / N, with no
    power at DC or below the profile, and the last value held above it."""
    sample_rate_hertz = 16384.0
    num_samples = 16384
    frequencies = np.linspace(10.0, 5000.0, 500).astype(np.float32)
    values = np.linspace(2.0, 1.0, 500).astype(np.float32)

    onsource, _ = gf_noise.interpolate_onsource_offsource_psd(
        [num_samples, num_samples], sample_rate_hertz, frequencies, values
    )
    onsource = np.real(np.asarray(onsource))
    grid = np.fft.rfftfreq(num_samples, 1.0 / sample_rate_hertz)
    expected = np.interp(grid, frequencies, values)
    expected[grid < frequencies[0]] = 0.0

    assert onsource.shape == grid.shape
    np.testing.assert_allclose(onsource, expected, rtol=1e-5, atol=1e-7)
    assert onsource[0] == 0.0
    assert onsource[grid > 5000.0].min() == pytest.approx(1.0)


def test_noise_obtainer_returns_strain():
    """A NoiseObtainer returns strain: its PSD is the profile's, unscaled."""
    obtainer = gf.NoiseObtainer(noise_type=gf.NoiseType.COLORED, ifos=[gf.IFO.H1])
    batch = next(obtainer(
        group="train", sample_rate_hertz=FS, onsource_duration_seconds=16.0,
        crop_duration_seconds=0.0, offsource_duration_seconds=16.0,
        num_examples_per_batch=OBTAINER_EXAMPLES, seed=11,
    ))
    ratio, f = _welch_ratio(
        np.asarray(batch[gf.ReturnVariables.ONSOURCE], np.float64), [gf.IFO.H1], FS
    )
    _assert_median_matches_the_profile(ratio, f, OBTAINER_EXAMPLES, 16.0)


def _header(path):
    with open(path) as file:
        return "".join(line for line in file if line.startswith("#"))


def test_the_default_profile_is_o3_actual():
    """Each detector's default curve, its ``optimal_psd_path``, which coloured
    noise with no profile named uses, is its own O3_ACTUAL curve."""
    for ifo in ALL_IFOS:
        assert ifo.value.optimal_psd_path == gf.NoiseProfile.O3_ACTUAL.path(ifo), ifo.name
    assert len({gf.NoiseProfile.O3_ACTUAL.path(ifo) for ifo in ALL_IFOS}) == 3


def test_the_o3_actual_headers_name_their_source_and_its_sha256():
    """Each O3_ACTUAL file names LIGO-T2000012-v2, the table it was written
    from, where that table is published, and the table's SHA-256."""
    for ifo, (source, sha256) in O3_ACTUAL_SOURCES.items():
        header = _header(gf.NoiseProfile.O3_ACTUAL.path(ifo))
        assert "LIGO-T2000012-v2" in header, ifo.name
        assert f"Source: {O3_ACTUAL_URL}{source}\n" in header, ifo.name
        assert f"Source sha256: {sha256}\n" in header, ifo.name
        assert "Not resampled" in header, ifo.name


def test_noise_profiles_are_per_detector_with_provenance():
    """Each bundled profile names its source, Virgo's curve is Virgo's where a
    profile has one, and the O3 low and aLIGO design curves stay available as
    named profiles. A LIGO-only profile refuses Virgo by name."""
    with pytest.raises(ValueError, match="ZERO_DET_HIGH_POWER has no curve for V1"):
        gf.NoiseProfile.ZERO_DET_HIGH_POWER.path(gf.IFO.V1)
    for profile, ifo, source in [
        (gf.NoiseProfile.ZERO_DET_HIGH_POWER, gf.IFO.H1, "aLIGOZeroDetHighPower"),
        (gf.NoiseProfile.ZERO_DET_HIGH_POWER, gf.IFO.L1, "LIGO-T0900288-v3"),
        (gf.NoiseProfile.O3_LOW, gf.IFO.H1, "aLIGOO3LowT1800545"),
        (gf.NoiseProfile.O3_LOW, gf.IFO.L1, "aLIGOO3LowT1800545"),
        (gf.NoiseProfile.O3_LOW, gf.IFO.V1, "AdVO3LowT1800545"),
        (gf.NoiseProfile.DESIGN, gf.IFO.H1, "T1800044"),
        (gf.NoiseProfile.DESIGN, gf.IFO.L1, "T1800044"),
        (gf.NoiseProfile.DESIGN, gf.IFO.V1, "P1200087"),
    ]:
        assert source in _header(profile.path(ifo)), f"{profile.name} {ifo.name} does not name {source}"

    grid = np.geomspace(20.0, 2000.0, 200)
    for profile in (p for p in gf.NoiseProfile if p.covers(gf.IFO.V1)):
        f_h, asd_h = _profile(gf.IFO.H1, profile)
        f_v, asd_v = _profile(gf.IFO.V1, profile)
        assert np.max(np.abs(np.log(np.interp(grid, f_v, asd_v) / np.interp(grid, f_h, asd_h)))) > 0.1


@pytest.mark.parametrize("profile, ifo", [
    (gf.NoiseProfile.DESIGN, gf.IFO.H1),
    (gf.NoiseProfile.O4_HIGH, gf.IFO.L1),
    (gf.NoiseProfile.O4_HIGH, gf.IFO.V1),
    (gf.NoiseProfile.O4_LOW, gf.IFO.H1),
], ids=lambda value: value.name)
def test_colored_noise_uses_the_named_profile(profile, ifo):
    """NoiseObtainer(noise_profile=...) colours the noise with that profile."""
    obtainer = gf.NoiseObtainer(
        noise_type=gf.NoiseType.COLORED, ifos=[ifo],
        noise_profile=profile,
    )
    batch = next(obtainer(
        group="train", sample_rate_hertz=FS, onsource_duration_seconds=16.0,
        crop_duration_seconds=0.0, offsource_duration_seconds=16.0,
        num_examples_per_batch=OBTAINER_EXAMPLES, seed=5,
    ))
    ratio, f = _welch_ratio(batch[gf.ReturnVariables.ONSOURCE], [ifo], FS, profile)
    _assert_median_matches_the_profile(ratio, f, OBTAINER_EXAMPLES, 16.0)


# The LIGO-T2000012-v2 table each O4 curve was written from, and its SHA-256 as downloaded.
O4_SOURCES = {
    (gf.NoiseProfile.O4_HIGH, gf.IFO.H1): ("aligo_O4high.txt",
                                           "eb5ec9b081c3d86d2f4257b9aff6a57566d168b8a95e5e57b7909eebad021780"),
    (gf.NoiseProfile.O4_HIGH, gf.IFO.L1): ("aligo_O4high.txt",
                                           "eb5ec9b081c3d86d2f4257b9aff6a57566d168b8a95e5e57b7909eebad021780"),
    (gf.NoiseProfile.O4_HIGH, gf.IFO.V1): ("avirgo_O4high_NEW.txt",
                                           "07eae64166a762db30cda3f757b6546723ce6209be633f7915efc7f270ae0a00"),
    (gf.NoiseProfile.O4_LOW, gf.IFO.H1): ("aligo_O4low.txt",
                                          "9a7449cda558e4f5c124c5693cf5149d464a9d9c5383acd24030fac1400b88bd"),
    (gf.NoiseProfile.O4_LOW, gf.IFO.L1): ("aligo_O4low.txt",
                                          "9a7449cda558e4f5c124c5693cf5149d464a9d9c5383acd24030fac1400b88bd"),
}


def test_the_o4_curves_name_their_source_and_are_projections():
    """
    Each O4 curve names its LIGO-T2000012-v2 table, where it is published and its
    SHA-256, and says it is a projection; O4_LOW has no Virgo curve and refuses V1 by
    name. The projections order as published: aLIGO's high below its low, both below
    O3's measured curves through the bucket.
    """
    for (profile, ifo), (source, sha256) in O4_SOURCES.items():
        header = _header(profile.path(ifo))
        assert "LIGO-T2000012-v2" in header and "projection" in header, (profile.name, ifo.name)
        assert f"Source: {O3_ACTUAL_URL}{source}\n" in header, (profile.name, ifo.name)
        assert f"Source sha256: {sha256}\n" in header, (profile.name, ifo.name)
    with pytest.raises(ValueError, match="O4_LOW has no curve for V1"):
        gf.NoiseProfile.O4_LOW.path(gf.IFO.V1)
    bucket = np.geomspace(60.0, 400.0, 50)
    for ifo in (gf.IFO.H1, gf.IFO.L1):
        high, low, o3 = (np.interp(bucket, *_profile(ifo, profile))
                         for profile in (gf.NoiseProfile.O4_HIGH, gf.NoiseProfile.O4_LOW, gf.NoiseProfile.O3_ACTUAL))
        assert np.median(high / low) < 1.0 and np.median(low / o3) < 1.0, ifo.name


#: The aLIGO curves made with lalsimulation from a table that ends below 8192 Hz: the
#: lalsimulation function, and where the table it reads ends (Hz).
LALSIMULATION_TABLE_ENDS = {
    gf.NoiseProfile.O3_LOW: ("SimNoisePSDaLIGOO3LowT1800545", 4995.378),
    gf.NoiseProfile.DESIGN: ("SimNoisePSDaLIGODesignSensitivityT1800044", 5000.0),
}


def _lalsimulation_asd(function_name, frequencies):
    """lalsimulation's ASD at ``frequencies``, from its PSD evaluated every 1/64 Hz to
    8192 Hz and interpolated log-log, as the bundled curves were made."""
    import lal
    import lalsimulation

    step = 1.0 / 64.0
    series = lal.CreateREAL8FrequencySeries("", 0, 0.0, step, lal.DimensionlessUnit, int(8192 / step) + 1)
    getattr(lalsimulation, function_name)(series, 0.0)
    grid, psd = np.arange(series.data.length) * step, series.data.data
    positive = psd > 0
    return np.exp(np.interp(np.log(frequencies), np.log(grid[positive]), 0.5 * np.log(psd[positive])))


@pytest.mark.parametrize("profile", list(LALSIMULATION_TABLE_ENDS), ids=lambda v: v.name)
def test_the_lalsimulation_curves_stop_where_their_table_ends(profile):
    """
    The aLIGO O3_LOW and DESIGN curves are never extrapolated: lalsimulation's tables
    stop at 4995.38 Hz and 5000 Hz, beyond which it extrapolates them, so the bundled
    rows stop there too, and the header says so. Every row kept is lalsimulation's
    value, as before the cut.
    """
    function_name, table_end = LALSIMULATION_TABLE_ENDS[profile]
    header = _header(profile.path(gf.IFO.H1))
    assert f"stops at {table_end:g} Hz" in header and "never extrapolated" in header, header
    frequencies, asd = _profile(gf.IFO.H1, profile)
    assert table_end - 15.0 < frequencies[-1] <= table_end
    np.testing.assert_allclose(asd, _lalsimulation_asd(function_name, frequencies), rtol=1e-4)


@pytest.mark.parametrize("profile", list(LALSIMULATION_TABLE_ENDS), ids=lambda v: v.name)
def test_a_sample_rate_past_a_curves_table_is_refused(profile):
    """
    Coloured noise at 16384 Hz needs the curve to its 8192 Hz Nyquist frequency, past
    the end of the aLIGO O3_LOW and DESIGN tables: refused, by name. 8192 Hz (to
    4096 Hz) is served.
    """
    def batcher(sample_rate_hertz):
        return gf_noise.colored_noise_batcher(
            num_examples_per_batch=1, onsource_duration_seconds=1.0, crop_duration_seconds=0.0,
            offsource_duration_seconds=1.0, ifos=[gf.IFO.H1], sample_rate_hertz=sample_rate_hertz,
            noise_profile=profile,
        )
    with pytest.raises(ValueError, match=r"8192 Hz is needed: a curve is never extrapolated"):
        batcher(16384.0)
    batcher(8192.0)
