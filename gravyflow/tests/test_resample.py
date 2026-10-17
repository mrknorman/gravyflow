"""
The one resampler, ``gf.resample``, and every read of real strain that goes through it.

Its contract: alias suppression of at least 100 dB above the new Nyquist frequency,
passband ripple of at most 1e-3 up to 0.9 times it, output sample n at exactly
``t0 + n / rate_out``, and a linear ramp kept. Every window read (real-noise segments,
transient windows and their partners) serves a tone above the new Nyquist frequency
suppressed when its source serves another rate than the read's. The census: each site
where strain changes rate serves exactly the samples ``gf.resample`` gives, every
transient window sample exact to both ends of its stored row; no other code names a
resampler or slices with a stride, but the few functions that stride what is never strain.
"""

import ast
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import freqz

import gravyflow as gf
from gravyflow.src.dataset.acquisition.sources import SegmentStrain
from gravyflow.src.dataset.config import TransientDefaults
from gravyflow.src.dataset.features.glitch_cache import TransientCache
from gravyflow.src.utils.resample import (
    PASSBAND_FRACTION, resample, resample_filter, resample_margin, resample_ratio, whole_hertz,
)
from gravyflow.src.utils.tensor import centre_index
from gravyflow.tests import test_segment_provenance as noise_reads
from gravyflow.tests import test_transient_windows as transient_reads

#: Rate pairs: integer and rational ratios down, and one up.
RATES = [(4096, 2048), (16384, 2048), (8192, 2048), (16384, 4096), (4096, 1000), (2048, 4096)]
SUPPRESSION_DB = 100.0
RIPPLE = 1e-3
#: The tone the reads must suppress, and the rate they serve it at. A whole number of
#: seconds holds no whole number of its cycles, which an FFT resampler would pass by luck.
TONE_HERTZ, RATE = 1500.1, 2048.0


def interior(rate_in, rate_out):
    """Output samples within the resampler's margin of either end of a whole input."""
    return int(np.ceil(resample_margin(rate_in, rate_out) * rate_out / rate_in)) + 1


def rms(x):
    return float(np.sqrt(np.mean(np.square(np.asarray(x, dtype=np.float64)))))


class TestContract:

    @pytest.mark.parametrize("rate_in, rate_out", RATES)
    def test_filter_suppresses_and_keeps_its_bands(self, rate_in, rate_out):
        up, down = resample_ratio(rate_in, rate_out)
        taps = resample_filter(up, down)
        frequency, response = freqz(taps, worN=2**21, fs=2.0)  # 1.0 is the upsampled Nyquist
        new_nyquist = 1.0 / max(up, down)
        stop = 20 * np.log10(np.max(np.abs(response[frequency >= new_nyquist])))
        ripple = np.max(np.abs(np.abs(response[frequency <= PASSBAND_FRACTION * new_nyquist]) - 1.0))
        assert stop <= -SUPPRESSION_DB, f"{rate_in}->{rate_out}: stopband at {stop:.2f} dB"
        assert ripple <= RIPPLE, f"{rate_in}->{rate_out}: passband ripple {ripple:.2e}"

    @pytest.mark.parametrize("rate_in, rate_out", [rates for rates in RATES if rates[1] < rates[0]])
    def test_tones_above_the_new_nyquist_are_suppressed(self, rate_in, rate_out):
        times = np.arange(int(4 * rate_in)) / rate_in
        edge = interior(rate_in, rate_out)
        for tone in np.linspace(rate_out / 2, rate_in / 2, 7)[1:-1]:
            resampled, _ = resample(np.sin(2 * np.pi * tone * times + 0.4), 0.0, rate_in, rate_out)
            assert rms(resampled[edge:-edge]) <= 1e-5, f"{tone:.1f} Hz at {rate_in}->{rate_out}"

    @pytest.mark.parametrize("extra", [0, 1, 3])
    @pytest.mark.parametrize("rate_in, rate_out", RATES)
    def test_sample_n_is_at_t0_plus_n_over_the_rate(self, rate_in, rate_out, extra):
        t0 = 1238175000.3  # off the sample grid of every rate
        tones = np.array([37.0, 211.5, 0.85 * min(rate_in, rate_out) / 2])

        def signal(times):
            return np.sin(2 * np.pi * tones * (times[:, None] - 1238175000.0)).sum(axis=1)

        num_samples = int(2 * rate_in) + extra
        resampled, start = resample(signal(t0 + np.arange(num_samples) / rate_in), t0, rate_in, rate_out)
        assert start == t0
        assert len(resampled) == -(-num_samples * rate_out // rate_in)
        edge = interior(rate_in, rate_out)
        n = np.arange(len(resampled))[edge:-edge]
        error = np.max(np.abs(resampled[edge:-edge] - signal(t0 + n / rate_out)))
        # The passband ripple bounds each tone's error; a shift of 0.002 samples would not fit.
        assert error <= len(tones) * RIPPLE, f"{num_samples} samples at {rate_in}->{rate_out}: {error:.2e}"

    @pytest.mark.parametrize("rate_in, rate_out", RATES)
    def test_a_linear_ramp_is_kept_to_its_ends(self, rate_in, rate_out):
        times = np.arange(int(rate_in) + 3) / rate_in
        resampled, _ = resample(2.5 - 40.0 * times, 0.0, rate_in, rate_out)
        expected = 2.5 - 40.0 * np.arange(len(resampled)) / rate_out
        np.testing.assert_allclose(resampled, expected, rtol=0, atol=1e-5)

    def test_equal_rates_return_the_input(self):
        x = np.random.default_rng(1).normal(size=(2, 999)).astype(np.float32)
        same, start = resample(x, 7.5, 2048, 2048.0)
        assert start == 7.5 and same.dtype == np.float32
        np.testing.assert_array_equal(same, x)
        assert resample_margin(2048, 2048) == 0

    def test_computes_in_float64_and_returns_float32(self):
        x = 1e-21 * np.random.default_rng(2).normal(size=4096)
        from_double, _ = resample(x, 0.0, 4096, 2048)
        from_single, _ = resample(x.astype(np.float32), 0.0, 4096, 2048)
        assert from_double.dtype == from_single.dtype == np.float32
        assert np.all(np.isfinite(from_double)) and rms(from_double) > 0.5e-21
        assert np.max(np.abs(from_double - from_single)) <= 1e-6 * np.max(np.abs(from_double))

    def test_interior_samples_do_not_depend_on_the_read(self):
        """Beyond the margin, an output is bit for bit what any longer read gives."""
        rate_in, rate_out = 4096, 1000
        up, down = resample_ratio(rate_in, rate_out)
        x = np.random.default_rng(3).normal(size=20 * rate_in)
        whole, _ = resample(x, 0.0, rate_in, rate_out)
        offset = 3 * down  # a read starting on an output sample: output 3 * up
        part, _ = resample(x[offset:offset + 6 * rate_in], 0.0, rate_in, rate_out)
        edge = interior(rate_in, rate_out)
        np.testing.assert_array_equal(part[edge:-edge], whole[3 * up + edge:3 * up + len(part) - edge])

    def test_resamples_the_last_axis_of_each_row_alike(self):
        x = np.random.default_rng(4).normal(size=(3, 2, 8192))
        rows, _ = resample(x, 0.0, 4096, 2048)
        for index in np.ndindex(x.shape[:-1]):
            np.testing.assert_array_equal(rows[index], resample(x[index], 0.0, 4096, 2048)[0])

    @pytest.mark.parametrize("rate", [2048.5, 0, -2048, float("nan"), float("inf"), "1e3.5"])
    def test_refuses_a_rate_that_is_not_whole_hertz(self, rate):
        with pytest.raises(ValueError):
            resample(np.zeros(16), 0.0, 4096, rate)
        if isinstance(rate, float) and np.isfinite(rate):
            with pytest.raises(ValueError, match=f"{rate!r} Hz is not a positive whole number"):
                whole_hertz(rate)

    def test_the_filter_is_designed_once_per_reduced_ratio(self):
        assert resample_ratio(4096, 2048) == resample_ratio(16384.0, 8192) == (1, 2)
        assert resample_ratio(4096, 1000) == (125, 512)
        taps = resample_filter(*resample_ratio(4096, 2048))
        assert taps is resample_filter(*resample_ratio(16384, 8192))
        assert len(taps) % 2 == 1 and taps.sum() == pytest.approx(1.0, abs=1e-12)
        with pytest.raises(ValueError):
            taps[0] = 0.0


# -------------------------------------------------- every read goes through it

class ToneFetch:
    """
    A fake GWOSC fetch publishing ``served_rate`` only: [start, end) on the absolute
    grid, holding one tone at ``TONE_HERTZ``. Records the rates served.
    """

    source_id = "gwosc-fake-tone"

    def __init__(self, served_rate, tone=TONE_HERTZ):
        self.served_rate, self.tone, self.served = float(served_rate), tone, []
        self.rates = (self.served_rate,)

    def __call__(self, ifo, sample_rate_hertz, interval):
        (start, end), rate = interval, float(sample_rate_hertz)
        first, stop = int(np.ceil(start * rate - 1e-6)), int(np.ceil(end * rate - 1e-6))
        times = np.arange(first, stop, dtype=np.float64) / rate
        self.served.append(rate)
        return SegmentStrain(np.sin(2 * np.pi * self.tone * (times - transient_reads.REF) + 0.4),
                             first / rate, rate, self.source_id)


def read_piece(fetch):
    """ROW's first piece as the noise path reads it (``acquire``), and its first sample's index."""
    (start, end), ifo = noise_reads.ROW[0], noise_reads.IFOS[0]
    with noise_reads.fetching(fetch):
        [row] = noise_reads.obtainer().acquire(RATE, np.array([[[start, end]]]), [ifo])
    first = int(round(row.start_gps_time[0] * RATE))
    return np.asarray(row.data[0]), first


@pytest.mark.parametrize("served_rate", [4096.0, 16384.0])
def test_noise_segments_are_resampled(served_rate):
    fetch = ToneFetch(served_rate)
    data, first = read_piece(fetch)
    (start, end) = noise_reads.ROW[0]
    assert fetch.served and set(fetch.served) != {RATE}, "the source served the output rate"
    assert start <= first / RATE and first / RATE + len(data) / RATE <= end
    assert len(data) % int(RATE) == 0 and len(data) >= (int(end - start) - 1) * RATE
    # Every sample is exact: the read reaches the margin past each end of the piece.
    assert rms(data) <= 1e-5, f"{TONE_HERTZ} Hz served at {RATE} Hz from {served_rate} Hz"


@pytest.mark.parametrize("served_rate, tone", [
    (4096.0, TONE_HERTZ),   # resampled as the cache is read
    (16384.0, TONE_HERTZ),  # resampled as it is downloaded (to 4096 Hz), and as it is read
    (16384.0, 3596.1),      # suppressed only by the download's resampling: it would alias to 500 Hz
])
def test_transient_and_partner_windows_are_resampled(tmp_path, served_rate, tone):
    # The glitches are seen in L1; H1 is read at partner times.
    with transient_reads.offline(ToneFetch(served_rate, tone)) as fetch:
        batch = transient_reads.first_batch(tmp_path, [gf.IFO.H1, gf.IFO.L1])
    assert fetch.served and set(fetch.served) != {RATE}, "the source served the output rate"
    assert transient_reads.RATE == RATE
    for column, read in enumerate(("partner", "transient")):
        for key in (gf.ReturnVariables.ONSOURCE, gf.ReturnVariables.OFFSOURCE):
            windows = np.asarray(batch[key])[:, column]
            assert rms(windows) <= 1e-5, f"{read} {key.name}: {tone} Hz from {served_rate} Hz"


def test_transient_windows_at_a_rate_that_does_not_divide_the_stored_one(tmp_path):
    """1000 Hz from the stored 4096 Hz: each sample is its own time, the transient at the centre."""
    rate = 1000.0
    with transient_reads.offline():  # every sample valued its GPS time
        batch = transient_reads.first_batch(tmp_path, [gf.IFO.L1], sample_rate_hertz=rate)
    onsource = np.asarray(batch[gf.ReturnVariables.ONSOURCE], dtype=np.float64)
    offsource = np.asarray(batch[gf.ReturnVariables.OFFSOURCE], dtype=np.float64)
    error = np.abs(np.sort(transient_reads.transient_times(batch)[:, 0]) - transient_reads.TIMES)
    assert np.max(error) <= 1.0 / rate, f"centre minus transient time: {error} s"
    for window in (onsource, offsource):
        np.testing.assert_allclose(np.diff(window, axis=-1), 1.0 / rate, atol=1e-4)
    np.testing.assert_allclose(onsource[..., 0] - offsource[..., -1], 1.0 / rate, atol=1e-4)


# ------------------------------------------------------------- the census
#
# The census is semantic where strain changes rate: each site serves, bit for bit, what
# gf.resample gives the strain it read, on white noise, which any other filter, stride,
# gather, reshape or interpolation changes. Elsewhere, a syntax census refuses the
# resamplers it can name and strided slices.

class NoiseFetch:
    """
    A fake GWOSC fetch publishing ``served_rate`` only: [start, end) on the absolute
    grid, holding white noise. Records each strain served.
    """

    source_id = "gwosc-fake-noise"

    def __init__(self, served_rate):
        self.served_rate, self.served = float(served_rate), []
        self.rates = (self.served_rate,)

    def __call__(self, ifo, sample_rate_hertz, interval):
        (start, end), rate = interval, float(sample_rate_hertz)
        first, stop = int(np.ceil(start * rate - 1e-6)), int(np.ceil(end * rate - 1e-6))
        noise = np.random.default_rng(len(self.served)).normal(size=stop - first)
        self.served.append(SegmentStrain(noise, first / rate, rate, self.source_id))
        return self.served[-1]


@pytest.mark.parametrize("served_rate", [4096.0, 16384.0])
def test_a_noise_segment_is_the_resamplers_samples(served_rate):
    fetch = NoiseFetch(served_rate)
    data, first = read_piece(fetch)
    (served,) = fetch.served  # one whole tile, stored as float32
    tile = resample(np.asarray(served.data, dtype=np.float32), 0.0, served_rate, RATE)[0]
    offset = first - int(round(served.t0 * RATE))
    np.testing.assert_array_equal(data, tile[offset:offset + len(data)])


def test_a_transient_download_is_the_resamplers_samples(tmp_path):
    reader = transient_reads.obtainer(tmp_path)
    span = (gf.IFO.L1, transient_reads.TIMES[0] - 48.2, transient_reads.TIMES[0] + 48.2)
    stored_rate = TransientDefaults.CACHE_SAMPLE_RATE_HERTZ
    with transient_reads.offline(NoiseFetch(4 * stored_rate)) as fetch:
        samples, t0 = reader._download({"read": span}, stored_rate)["read"]
    (served,) = fetch.served
    assert t0 == served.t0
    np.testing.assert_array_equal(samples, resample(served.data, 0.0, 4 * stored_rate, stored_rate)[0])


@pytest.mark.parametrize("rate", [2048, 1000])
def test_transient_windows_are_a_longer_reads_samples(rate):
    """
    Every sample of a transient's windows is exact: ``read_windows`` serves, bit for bit,
    what the resampler gives a longer stretch of the same strain, at a ratio of one half
    and at 4096 to 1000, with windows that reach both ends of the stored row. Noise, not
    a ramp: the line-padded filter keeps a ramp exact even inside its margin.
    """
    stored_rate = TransientDefaults.CACHE_SAMPLE_RATE_HERTZ
    stored = int(32 * stored_rate)  # samples of the stored offsource, and of the onsource
    up, down = resample_ratio(stored_rate, rate)
    before, after = TransientCache.window_reach(stored, stored, stored_rate, rate)
    num_onsource, num_offsource = 2 * after, before - after - (before - after) % 2
    durations = ((num_onsource + 0.5) / rate, (num_offsource + 0.5) / rate)

    pad = 2 * (resample_margin(stored_rate, rate) + down)  # strain beyond each end of the row
    strain = np.random.default_rng(5).normal(size=(2, pad + 2 * stored + pad)).astype(np.float32)
    row = strain[:, pad:pad + 2 * stored]
    onsource, offsource = TransientCache.read_windows(
        row[:, stored:], row[:, :stored], stored_rate, rate, *durations,
    )
    # The longer read starts a whole number of down samples before the transient's
    # sample, as every read does, so its samples fall on the windows'.
    transient = pad + stored + centre_index(stored)
    first = transient % down
    longer, _ = resample(strain[:, first:], 0.0, stored_rate, rate)
    centre = (transient - first) // down * up
    onsource_start = centre - centre_index(num_onsource)
    np.testing.assert_array_equal(offsource, longer[:, onsource_start - num_offsource:onsource_start])
    np.testing.assert_array_equal(onsource, longer[:, onsource_start:onsource_start + num_onsource])
    read = TransientCache.window_read(stored, stored, stored_rate, rate, *durations)
    assert read.stored.start < down and read.stored.stop > 2 * stored - down, (
        f"the windows read stored samples {read.stored}, not both ends of {2 * stored}"
    )


#: Names whose use resamples outside the one resampler.
RESAMPLERS = {"resample_fft", "resample_poly", "upfirdn", "decimate"}
#: The functions allowed a strided slice, by module and qualified name, and why: each
#: strides what is never strain data. A slice cannot say what it slices, so every other
#: function, strain's modules' included, keeps the census; an allowance no code needs
#: any more fails (`test_every_stride_allowance_is_needed`).
STRIDE_ALLOWED = {
    ("validate/plotting.py", "generate_waveform_plot"): "thins lines for display; never data",
    ("validate/plotting.py", "generate_parameter_space_plot"): "thins a curve for display; never data",
    ("dataset/store/tile.py", "_runs"): "pairs the edges of a mask's runs: indices, never samples",
}


def _source_files():
    root = Path(gf.__file__).parent / "src"
    return root, sorted(root.rglob("*.py"))


def _is_reversal_or_unit(step):
    value = step.operand if isinstance(step, ast.UnaryOp) and isinstance(step.op, ast.USub) else step
    return isinstance(value, ast.Constant) and value.value == 1


def _strided_slices(tree):
    """(The qualified name of the enclosing function, the line) of each slice with another stride than ±1."""
    found = []

    def visit(node, scope):
        for child in ast.iter_child_nodes(node):
            inner = scope
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                inner = f"{scope}.{child.name}" if scope else child.name
            if isinstance(child, ast.Slice) and child.step is not None and not _is_reversal_or_unit(child.step):
                found.append((inner, child.lineno))
            visit(child, inner)

    visit(tree, "")
    return found


def test_no_other_code_resamples_strain():
    """
    Nothing in the library resamples but ``utils/resample.py``: no FFT resampler, no
    SciPy resampler or decimator, no gwpy ``.resample`` and no strided slice outside the
    functions allowed one.
    """
    root, files = _source_files()
    found = []
    for path in files:
        name = path.relative_to(root).as_posix()
        if name == "utils/resample.py":
            continue
        tree = ast.parse(path.read_text(), filename=name)
        found.extend(f"{name}:{line} slices with a stride (in {scope or 'the module'})"
                     for scope, line in _strided_slices(tree) if (name, scope) not in STRIDE_ALLOWED)
        for node in ast.walk(tree):
            where = f"{name}:{getattr(node, 'lineno', '?')}"
            if isinstance(node, ast.Name) and node.id in RESAMPLERS:
                found.append(f"{where} uses {node.id}")
            elif isinstance(node, ast.Attribute) and node.attr in RESAMPLERS:
                found.append(f"{where} uses .{node.attr}")
            elif isinstance(node, ast.alias) and node.name.split(".")[-1] in RESAMPLERS:
                found.append(f"{where} imports {node.name}")
            elif isinstance(node, ast.FunctionDef) and (node.name in RESAMPLERS or node.name == "resample"):
                found.append(f"{where} defines {node.name}")
            elif isinstance(node, ast.ImportFrom) and node.module == "scipy.signal" \
                    and any(alias.name == "resample" for alias in node.names):
                found.append(f"{where} imports scipy.signal.resample")
            elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                  and node.func.attr == "resample"
                  and not (isinstance(node.func.value, ast.Name) and node.func.value.id == "gf")):
                found.append(f"{where} calls .resample")
    assert not found, "Resampling outside utils/resample.py:\n" + "\n".join(found)


def test_every_stride_allowance_is_needed():
    """Each allowed function still slices with a stride, so the allowances stay the fewest the code needs."""
    root, files = _source_files()
    strided = {(path.relative_to(root).as_posix(), scope)
               for path in files for scope, _ in _strided_slices(ast.parse(path.read_text()))}
    assert not set(STRIDE_ALLOWED) - strided, (
        f"allowances no code needs: {sorted(set(STRIDE_ALLOWED) - strided)}"
    )
