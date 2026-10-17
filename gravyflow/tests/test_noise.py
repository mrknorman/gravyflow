import logging
from itertools import islice

import pytest
import numpy as np
from keras import ops
import jax.numpy as jnp

import gravyflow as gf
from gravyflow.src.dataset.noise import noise as gf_noise
from gravyflow.src.dataset.features.injection import ReturnVariables as RV
from gravyflow.tests._helpers import representation_tolerance, statistical_bound

@pytest.mark.parametrize("lowest", [0.0, 10.0])
def test_interpolate_psd_puts_each_bin_at_its_own_frequency(lowest):
    """
    Each rfft bin k of each length N gets the spectrum at k fs / N: a linear
    spectrum is reproduced to float32 rounding, DC and bins below the spectrum's
    lowest frequency are 0, and bins above its highest hold its last value.
    """
    sample_rate = 1024.0
    freqs = np.linspace(lowest, 400.0, 81, dtype=np.float32)
    values = 3.0 + 0.5 * freqs
    on, off = gf_noise.interpolate_onsource_offsource_psd([2048, 1024], sample_rate, freqs, values)
    for interpolated, num_samples in ((on, 2048), (off, 1024)):
        interpolated = np.asarray(interpolated)
        assert interpolated.shape == (num_samples // 2 + 1,) and interpolated.dtype == np.complex64
        assert np.all(interpolated.imag == 0.0)
        grid = np.fft.rfftfreq(num_samples, 1.0 / sample_rate)
        expected = np.where((grid == 0.0) | (grid < freqs[0]), 0.0, 3.0 + 0.5 * np.clip(grid, None, freqs[-1]))
        # Representation: a few float32 roundings of values up to the spectrum's peak.
        np.testing.assert_allclose(interpolated.real, expected, rtol=0.0,
                                   atol=representation_tolerance(np.float32, values.max(), 4))

def test_unsupported_noise_type():
    """Verify error is raised for invalid noise type."""
    # This might need to test NoiseObtainer validation
    
    noise = gf.NoiseObtainer(noise_type="INVALID_TYPE")
    
    with pytest.raises(ValueError, match="NoiseType .* not recognised"):
        # Validation happens in __call__
        next(noise(group="train"))

def test_noise_obtainer_defaults():
    """Test NoiseObtainer default parameter handling."""
    noise = gf.NoiseObtainer(
        noise_type=gf.NoiseType.WHITE,
        ifos=gf.IFO.L1
    )
    
    gen = noise(group="train")
    batch = next(gen)
    onsource = batch[RV.ONSOURCE]
    offsource = batch[RV.OFFSOURCE]
    
    # Should use defaults from gf.Defaults
    expected_onsource_dur = gf.Defaults.onsource_duration_seconds + 2 * gf.Defaults.crop_duration_seconds
    expected_onsource_samples = gf_noise.ensure_even(int(expected_onsource_dur * gf.Defaults.sample_rate_hertz))
    expected_offsource_samples = gf_noise.ensure_even(int(gf.Defaults.offsource_duration_seconds * gf.Defaults.sample_rate_hertz))
    
    assert ops.shape(onsource)[-1] == expected_onsource_samples
    assert ops.shape(offsource)[-1] == expected_offsource_samples


def test_noise_obtainer_single_ifo_conversion():
    """Test that single IFO is converted to list."""
    noise = gf.NoiseObtainer(
        noise_type=gf.NoiseType.WHITE,
        ifos=gf.IFO.L1  # Single, not list
    )
    
    # After __post_init__, ifos should be a list
    assert isinstance(noise.ifos, list)
    assert len(noise.ifos) == 1


def test_noise_obtainer_has_no_split_of_its_own():
    """The split is the data obtainer's (its SplitSpec); NoiseObtainer takes no fractions."""
    with pytest.raises(TypeError):
        gf.NoiseObtainer(noise_type=gf.NoiseType.WHITE, ifos=gf.IFO.L1, groups={"train": 1.0})
    assert not hasattr(gf.NoiseObtainer(noise_type=gf.NoiseType.WHITE, ifos=gf.IFO.L1), "groups")


def test_noise_obtainer_requires_a_named_split():
    noise = gf.NoiseObtainer(noise_type=gf.NoiseType.WHITE, ifos=gf.IFO.L1)
    with pytest.raises(ValueError, match="Name the split"):
        next(noise())
    with pytest.raises(KeyError, match="a split is one of"):
        next(noise(group="all"))


# Removed: test_noise_obtainer_generator_none_error
# This was an empty test that couldn't be triggered through normal usage.
# The error path it was meant to test is covered by type checking.


def test_noise_obtainer_real_without_ifo_obtainer():
    """REAL noise without an ifo_data_obtainer defaults to public GWOSC data;
    removing the obtainer afterwards raises ValueError."""
    noise = gf.NoiseObtainer(
        noise_type=gf.NoiseType.REAL,
        ifos=gf.IFO.L1,
        ifo_data_obtainer=None  # No obtainer: the public-data default is built
    )
    assert noise.ifo_data_obtainer is not None
    assert noise.ifo_data_obtainer.data_labels == [gf.DataLabel.NOISE]

    noise.ifo_data_obtainer = None
    with pytest.raises(ValueError, match="No IFO obtainer object present"):
        next(noise(group="train"))


def coloured_variance(asd, num_samples, sample_rate_hertz):
    """
    The per-sample variance of ``color_noise``'s output for an ASD on the rfft
    grid: x = irfft(c W) with c = sqrt(fs / 2) ASD, so by Parseval
    var = sum over the full spectrum of c_k^2 / N, each interior bin counted
    twice (k and N - k), DC and the Nyquist bin once.
    """
    weights = np.full(len(asd), 2.0)
    weights[0] = 1.0
    if num_samples % 2 == 0:
        weights[-1] = 1.0
    power = (sample_rate_hertz / 2.0) * np.abs(np.asarray(asd, np.float64)) ** 2
    return float(np.sum(weights * power) / num_samples), weights * power


def test_noise_obtainer_scale_factor_is_deprecated_and_ignored():
    """
    A scale_factor passed to a NoiseObtainer warns and changes nothing: COLORED
    noise is strain, with the variance its detector's curve integrates to.
    """
    from gravyflow.src.dataset.conditioning.detector import DEFAULT_NOISE_PROFILE, read_noise_curve

    noise = gf.NoiseObtainer(noise_type=gf.NoiseType.COLORED, ifos=gf.IFO.L1)
    rate, batch = 1024.0, 32
    window = dict(group="train", sample_rate_hertz=rate, onsource_duration_seconds=1.0,
                  crop_duration_seconds=0.5, offsource_duration_seconds=2.0,
                  num_examples_per_batch=batch, seed=42)
    expected = np.asarray(next(noise(**window))[RV.ONSOURCE], np.float64)
    with pytest.warns(DeprecationWarning, match="scale_factor is deprecated.*physical strain"):
        served = next(noise(scale_factor=3.0, **window))[RV.ONSOURCE]
    np.testing.assert_array_equal(np.asarray(served), expected)

    # The curve's ASD on the onsource's rfft grid (np.interp, independent of the
    # library's interpolation), no power below the curve or at DC.
    num_samples = expected.shape[-1]
    frequencies, asd = read_noise_curve(DEFAULT_NOISE_PROFILE.path(gf.IFO.L1), up_to_hertz=rate / 2.0)
    grid = np.fft.rfftfreq(num_samples, 1.0 / rate)
    asd = np.where((grid > 0) & (grid >= frequencies[0]), np.interp(grid, frequencies, asd), 0.0)
    variance, power = coloured_variance(asd, num_samples, rate)
    # Statistical: the sample variance over B rows of N samples is a sum of
    # |W_k|^2 terms, each exponential with its own mean, so its relative standard
    # deviation is sqrt(sum p_k^2) / sum p_k over the rows' independent bins.
    relative_sigma = np.sqrt(np.sum(power ** 2) / batch) / np.sum(power)
    measured = np.mean(expected ** 2)
    assert abs(measured / variance - 1.0) < statistical_bound(relative_sigma), (
        f"coloured variance {measured:.3e}, where the curve integrates to {variance:.3e} "
        f"(relative sigma {relative_sigma:.3f})"
    )


def test_noise_obtainer_pseudo_real_without_obtainer():
    """Test that PSEUDO_REAL noise without ifo_data_obtainer raises ValueError."""
    noise = gf.NoiseObtainer(
        noise_type=gf.NoiseType.PSEUDO_REAL,
        ifos=gf.IFO.L1,
        ifo_data_obtainer=None
    )
    
    with pytest.raises(ValueError, match="No IFO obtainer object present"):
        gen = noise(group="train")
        next(gen)


# --- Synthetic noise batchers: one batch per address -------------------------

def _streams(seed=7, split="train", epoch=0, k=0):
    return gf.SeedTree(seed).batch(split, epoch, k)


@pytest.mark.parametrize("kind", ["white", "colored"])
def test_noise_batchers_shapes_and_addresses(kind):
    ifos = [gf.IFO.L1, gf.IFO.H1]
    if kind == "white":
        batch = gf_noise.white_noise_batcher(4, ifos, 1.0, 0.5, 4.0, 1024.0)
    else:
        batch = gf_noise.colored_noise_batcher(4, 1.0, 0.5, 4.0, ifos, 1024.0)
    first = batch(_streams())
    onsource = np.asarray(first[gf.ReturnVariables.ONSOURCE])
    offsource = np.asarray(first[gf.ReturnVariables.OFFSOURCE])
    assert onsource.shape == (4, 2, 2048) and offsource.shape == (4, 2, 4096)
    assert np.all(np.isfinite(onsource)) and np.all(np.isfinite(offsource))
    # The same address gives the same batch; any other address another.
    np.testing.assert_array_equal(np.asarray(batch(_streams())[gf.ReturnVariables.ONSOURCE]), onsource)
    for other in (_streams(k=1), _streams(epoch=1), _streams(split="validate"), _streams(seed=8)):
        assert not np.array_equal(np.asarray(batch(other)[gf.ReturnVariables.ONSOURCE]), onsource)


def test_each_detector_has_its_own_noise_stream():
    one = gf_noise.white_noise_batcher(2, [gf.IFO.L1], 1.0, 0.5, 4.0, 512.0)(_streams())
    two = gf_noise.white_noise_batcher(2, [gf.IFO.L1, gf.IFO.H1], 1.0, 0.5, 4.0, 512.0)(_streams())
    # Adding H1 leaves L1's noise as it was.
    np.testing.assert_array_equal(np.asarray(two[gf.ReturnVariables.ONSOURCE])[:, 0],
                                  np.asarray(one[gf.ReturnVariables.ONSOURCE])[:, 0])


def test_white_noise_is_unit_normal():
    batch = gf_noise.white_noise_batcher(64, [gf.IFO.L1], 1.0, 0.0, 1.0, 1024.0)(_streams())
    x = np.asarray(batch[gf.ReturnVariables.ONSOURCE], np.float64)
    n = x.size
    # Statistical: the mean of n unit normals has sigma 1 / sqrt(n), their sample
    # standard deviation 1 / sqrt(2 n).
    assert abs(x.mean()) < statistical_bound(1.0 / np.sqrt(n))
    assert abs(x.std() - 1.0) < statistical_bound(1.0 / np.sqrt(2.0 * n))


def test_color_noise_scales_each_bin_by_its_asd():
    """
    Colouring is x = irfft(sqrt(fs / 2) ASD rfft(w)): a bin-centred sine passes
    scaled by sqrt(fs / 2) times its own bin's ASD, so a sum of two sines comes
    out as the same sum with each sine scaled by its own factor.
    """
    rate, n = 64.0, 64
    t = np.arange(n) / rate
    sines = np.stack([np.sin(2 * np.pi * 5.0 * t), np.cos(2 * np.pi * 12.0 * t)]).astype(np.float32)
    asd = np.linspace(0.5, 3.0, n // 2 + 1).astype(np.float32)
    white = jnp.asarray((sines[0] + sines[1])[None, None, :])
    coloured = np.asarray(gf_noise.color_noise(white, jnp.asarray(asd, jnp.complex64)[None, None, :],
                                               sample_rate_hertz=rate))
    assert coloured.shape == (1, 1, n)
    expected = np.sqrt(rate / 2.0) * (asd[5] * sines[0] + asd[12] * sines[1])
    # Representation: an rfft and an irfft of n samples, scaled by the output's peak.
    np.testing.assert_allclose(coloured[0, 0], expected, rtol=0.0,
                               atol=representation_tolerance(np.float32, np.abs(expected).max(), 2 * np.log2(n)))


def test_obtainer_iterator_walks_the_batches_of_epoch_zero():
    obtainer = gf.NoiseObtainer(noise_type=gf.NoiseType.WHITE, ifos=[gf.IFO.L1])
    kwargs = dict(sample_rate_hertz=512.0, onsource_duration_seconds=1.0, crop_duration_seconds=0.5,
                  offsource_duration_seconds=4.0, num_examples_per_batch=2)
    iterator = obtainer(seed=3, group="validate", **kwargs)
    batch = obtainer.batcher(gf.SeedTree(3), "validate", **kwargs)
    for k in range(3):
        np.testing.assert_array_equal(
            np.asarray(next(iterator)[gf.ReturnVariables.ONSOURCE]),
            np.asarray(batch(gf.SeedTree(3).batch("validate", 0, k))[gf.ReturnVariables.ONSOURCE]),
        )


# --- Real noise addressed by index, offline: GRID and PSEUDO_REAL --------------
#
# The strain is a fake GWOSC fetch of hashed uniform noise: each sample's value is a
# function of its absolute index alone, scaled by the amplitude of the science
# segment it lies in, so an example's level names the kind of segment it came from.

from gravyflow.src.dataset.acquisition.sources import SegmentStrain  # noqa: E402
from gravyflow.src.dataset.acquisition import noise as acquisition_noise  # noqa: E402
from gravyflow.tests.conftest import gwosc_serving  # noqa: E402

OFFLINE_RATE = 256.0
OFFLINE_WINDOW = dict(sample_rate_hertz=OFFLINE_RATE, onsource_duration_seconds=1.0,
                      crop_duration_seconds=0.5, offsource_duration_seconds=4.0)
#: A window's footprint: its padded onsource and its offsource (s).
FOOTPRINT_SECONDS = 6.0
#: The first split block (8192 s) inside O3.
BLOCK_SECONDS = 8192.0
FIRST_BLOCK = float(np.ceil(1238170000.0 / BLOCK_SECONDS) * BLOCK_SECONDS)
#: Equal splits, so a few blocks give every split some time.
THIRDS = gf.SplitSpec(fractions={"test": 1 / 3, "validate": 1 / 3, "train": 1 / 3})


def science_layout(lengths, amplitudes, blocks, offset=200.0):
    """
    One science segment a block, ``offset`` s into it: block i's lasts
    ``lengths[i % len]`` seconds at amplitude ``amplitudes[i % len]``. Returns
    (start, end, amplitude) rows.
    """
    return [(FIRST_BLOCK + BLOCK_SECONDS * i + offset,
             FIRST_BLOCK + BLOCK_SECONDS * i + offset + lengths[i % len(lengths)],
             amplitudes[i % len(amplitudes)]) for i in range(blocks)]


class HashedFetch:
    """A fake GWOSC fetch at OFFLINE_RATE: hashed uniform noise, scaled by each sample's segment's amplitude."""

    rates = (OFFLINE_RATE,)
    source_id = "gwosc-fake-hashed"

    def __init__(self, layout):
        self.layout = layout

    def __call__(self, ifo, sample_rate_hertz, interval):
        start, end = interval
        rate = float(sample_rate_hertz)
        first, stop = int(np.ceil(start * rate - 1e-6)), int(np.ceil(end * rate - 1e-6))
        x = np.arange(first, stop, dtype=np.uint64) + np.uint64(sum(map(ord, ifo.name)) << 40)
        with np.errstate(over="ignore"):  # splitmix64
            x = x * np.uint64(0x9E3779B97F4A7C15)
            x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
            x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
            x = x ^ (x >> np.uint64(31))
        values = (x >> np.uint64(11)).astype(np.float64) / float(2 ** 53) - 0.5
        gps = np.arange(first, stop, dtype=np.float64) / rate
        for segment_start, segment_end, amplitude in self.layout:
            values[(gps >= segment_start) & (gps < segment_end)] *= amplitude
        return SegmentStrain(values.astype(np.float32), first / rate, rate, self.source_id)


def offline_noise(layout, **kwargs):
    """A noise data obtainer whose science time is ``layout``'s segments, with nothing vetoed."""

    class OfflineNoise(gf.NoiseDataObtainer):
        def get_segment_times(self, start, end, ifo, state_flags):
            rows = [(s, e) for s, e, _ in layout if s >= start and e <= end]
            return np.asarray(rows, dtype=np.float64).reshape(-1, 2)

        def remove_unwanted_segments(self, ifo, valid_segments):
            return valid_segments

    arguments = dict(observing_runs=gf.ObservingRun.O3, data_quality=gf.DataQuality.BEST,
                     data_labels=[gf.DataLabel.NOISE], augmentations=[])
    arguments.update(kwargs)
    return OfflineNoise(**arguments)


#: Thirty blocks of 40 s and 30 s pieces, in three equal splits: every split has some.
GRID_LAYOUT = science_layout([40.0, 30.0], [1.0], blocks=30)
GRID_BATCH = 4


def test_real_noise_serves_the_strain_at_its_labelled_times():
    """
    REAL noise (RANDOM order) reads each window where START_GPS_TIME says: the
    (padded) onsource is the source's strain from START_GPS_TIME minus the crop,
    the offsource the strain just before it, sample for sample, and every window
    of every batch starts somewhere new (an obtainer whose random state were
    copied between batches would serve the same windows again).
    """
    ifos, batch_size, num_batches = [gf.IFO.L1, gf.IFO.H1], 4, 5
    crop = OFFLINE_WINDOW["crop_duration_seconds"]
    num_onsource = int((OFFLINE_WINDOW["onsource_duration_seconds"] + 2 * crop) * OFFLINE_RATE)
    num_offsource = int(OFFLINE_WINDOW["offsource_duration_seconds"] * OFFLINE_RATE)
    fetch = HashedFetch(GRID_LAYOUT)
    with gwosc_serving(fetch):
        noise = gf.NoiseObtainer(noise_type=gf.NoiseType.REAL, ifos=ifos,
                                 ifo_data_obtainer=offline_noise(GRID_LAYOUT))
        batches = list(islice(noise(**OFFLINE_WINDOW, num_examples_per_batch=batch_size, group="train", seed=3),
                              num_batches))
    assert len(batches) == num_batches
    starts = []
    for batch in batches:
        onsource, offsource = np.asarray(batch[RV.ONSOURCE]), np.asarray(batch[RV.OFFSOURCE])
        start = np.asarray(batch[RV.START_GPS_TIME])
        assert onsource.shape == (batch_size, len(ifos), num_onsource)
        assert offsource.shape == (batch_size, len(ifos), num_offsource)
        assert start.shape == (batch_size, len(ifos))
        for example in range(batch_size):
            for d, ifo in enumerate(ifos):
                first = start[example, d] - crop
                # Exact: the served samples are the source's, unchanged.
                np.testing.assert_array_equal(
                    onsource[example, d], fetch(ifo, OFFLINE_RATE, (first, first + num_onsource / OFFLINE_RATE)).data)
                np.testing.assert_array_equal(
                    offsource[example, d], fetch(ifo, OFFLINE_RATE, (first - num_offsource / OFFLINE_RATE, first)).data)
        starts.append(start)
    starts = np.concatenate(starts)
    for d in range(len(ifos)):
        assert len(np.unique(starts[:, d])) == num_batches * batch_size, "a window was served twice"


def grid_dataset(group):
    return gf.Dataset(
        noise_obtainer=gf.NoiseObtainer(noise_type=gf.NoiseType.REAL, ifos=[gf.IFO.L1],
                                        ifo_data_obtainer=offline_noise(GRID_LAYOUT, split_spec=THIRDS)),
        **OFFLINE_WINDOW, num_examples_per_batch=GRID_BATCH, group=group, seed=11, ifos=[gf.IFO.L1],
        sampling_mode=gf.SamplingMode.GRID,
        input_variables=[gf.ReturnVariables.ONSOURCE], output_variables=[gf.ReturnVariables.START_GPS_TIME],
    )


def grid_starts_of(ds, indices):
    """START_GPS_TIME of the batches ``indices``, in that order: (n, B, I)."""
    return np.stack([np.asarray(ds[k][1][gf.ReturnVariables.START_GPS_TIME.name]) for k in indices])


class TestGridByIndex:
    """GRID is a finite set addressed by index: floor(N / B) batches, each window once, the same every epoch."""

    @pytest.mark.parametrize("group", ["train", "validate", "test"])
    def test_length_is_the_whole_batches_of_the_set(self, group, caplog):
        with gwosc_serving(HashedFetch(GRID_LAYOUT)), caplog.at_level(logging.INFO):
            ds = grid_dataset(group)
        plan = ds.noise.plan
        assert isinstance(plan, acquisition_noise.GridPlan)
        assert len(plan.starts) > 2 * GRID_BATCH, "the split serves too few GRID windows to test"
        assert len(ds) == len(plan.starts) // GRID_BATCH
        assert plan.remainder == len(plan.starts) % GRID_BATCH
        assert f"remainder of {plan.remainder} example" in caplog.text, "the remainder is not reported"
        with pytest.raises(IndexError):
            ds.streams(len(ds))

    def test_every_window_once(self):
        with gwosc_serving(HashedFetch(GRID_LAYOUT)):
            ds = grid_dataset("train")
            starts = grid_starts_of(ds, range(len(ds)))
        assert starts.shape[:2] == (len(ds), GRID_BATCH)
        assert len(np.unique(starts[..., 0])) == len(ds) * GRID_BATCH, "a GRID window was served twice"

    def test_a_batch_is_a_function_of_its_index(self):
        with gwosc_serving(HashedFetch(GRID_LAYOUT)):
            ds = grid_dataset("test")
            backwards = grid_starts_of(ds, reversed(range(len(ds))))[::-1]
            fresh = grid_starts_of(grid_dataset("test"), range(len(ds)))
        np.testing.assert_array_equal(backwards, fresh)

    @pytest.mark.parametrize("group", ["train", "validate", "test"])
    def test_every_epoch_serves_the_same_set(self, group):
        with gwosc_serving(HashedFetch(GRID_LAYOUT)):
            ds = grid_dataset(group)
            first = grid_starts_of(ds, range(len(ds)))
            ds.on_epoch_end()
            ds.epoch = 3
            again = grid_starts_of(ds, range(len(ds)))
        np.testing.assert_array_equal(first, again)

    def test_the_obtainer_iterator_ends_with_the_set(self):
        with gwosc_serving(HashedFetch(GRID_LAYOUT)):
            length = len(grid_dataset("train"))
            noise = gf.NoiseObtainer(noise_type=gf.NoiseType.REAL, ifos=[gf.IFO.L1],
                                     ifo_data_obtainer=offline_noise(GRID_LAYOUT, split_spec=THIRDS))
            served = list(islice(noise(**OFFLINE_WINDOW, num_examples_per_batch=GRID_BATCH, group="train", seed=11,
                                       sampling_mode=gf.SamplingMode.GRID), 3 * length))
        assert len(served) == length


class TestGridPairs:
    """GRID's pairs: slots one stride apart, each once a pass; detector k's the next free one from its middle, 2S away."""

    ON, OFF = 64, 256  # samples

    def starts(self, counts, firsts, separation, coincident=False):
        return acquisition_noise.grid_examples(np.asarray([firsts], np.int64), np.asarray([counts], np.int64),
                             np.zeros((1, len(counts), 2)), self.ON, self.OFF, separation, coincident)

    def test_each_slot_once_and_separated(self):
        starts = self.starts([6400, 6400], [0, 0], 640)
        assert len(starts) > 0
        for d in range(2):
            assert len(np.unique(starts[:, d])) == len(starts)
            assert np.all((starts[:, d] - self.OFF) % self.ON == 0), "a window off its slot"
        assert np.min(np.abs(starts[:, 0] - starts[:, 1])) >= 640

    @pytest.mark.parametrize("counts, firsts", [
        ([2560, 2560], [0, 0]),          # equal short pieces
        ([65536, 131072], [65536, 0]),   # one piece inside the other
    ])
    def test_adversarial_layouts_stay_apart(self, counts, firsts):
        starts = self.starts(counts, firsts, 768)
        assert len(starts) > 0
        assert np.min(np.abs(starts[:, 0] - starts[:, 1])) >= 768

    def test_an_inseparable_row_serves_nothing(self):
        assert len(self.starts([512, 512], [0, 0], 768)) == 0

    def test_coincident_slots_start_together(self):
        starts = self.starts([6400, 6400], [0, 0], 0, coincident=True)
        assert len(starts) == len(np.unique(starts[:, 0])) > 0
        np.testing.assert_array_equal(starts[:, 0], starts[:, 1])

    def test_a_piece_in_two_rows_serves_each_slot_once(self):
        firsts = np.array([[0, 100000], [0, 300000]], np.int64)
        counts = np.array([[6400, 6400], [6400, 6400]], np.int64)
        rows = np.array([[[0, 25], [390, 415]], [[0, 25], [1171, 1196]]], np.float64)
        starts = acquisition_noise.grid_examples(firsts, counts, rows, self.ON, self.OFF, 640, False)
        assert len(starts) > 0
        assert len(np.unique(starts[:, 0])) == len(starts), "detector 0's piece served a slot twice"

    def test_a_plan_has_floor_n_over_b_batches(self):
        starts = np.arange(30, dtype=np.int64).reshape(15, 2)
        plan = acquisition_noise.GridPlan("test", starts, 4)
        assert (plan.batches, plan.remainder) == (3, 3)
        np.testing.assert_array_equal(plan.examples(2), starts[8:12])
        with pytest.raises(IndexError):
            plan.examples(3)


SHORT, LONG = 200.0, 600.0
QUIET, LOUD = 1.0, 100.0
#: Short quiet pieces and long loud ones, three times as long, one a block.
PSEUDO_LAYOUT = science_layout([SHORT, LONG], [QUIET, LOUD], blocks=24)


def pseudo_real(layout, **kwargs):
    return gf.NoiseObtainer(noise_type=gf.NoiseType.PSEUDO_REAL, ifos=[gf.IFO.L1],
                            ifo_data_obtainer=offline_noise(layout, **kwargs))


def loud_share(batches):
    """The share of examples coloured by a loud segment: their level is far above the quietest's."""
    levels = np.concatenate([np.std(np.asarray(b[RV.ONSOURCE]), axis=-1).ravel() for b in batches])
    return float(np.mean(levels > np.sqrt(LOUD / QUIET) * np.min(levels)))


class TestPseudoRealByIndex:
    """PSEUDO_REAL colours each example by the PSD of a row of the RANDOM plan: endless, each row by its length."""

    BATCH = 64

    def batches(self, noise, n, group="train"):
        return list(islice(noise(**OFFLINE_WINDOW, num_examples_per_batch=self.BATCH, group=group, seed=5), n))

    def test_rows_serve_in_proportion_to_their_length(self):
        with gwosc_serving(HashedFetch(PSEUDO_LAYOUT)):
            noise = pseudo_real(PSEUDO_LAYOUT, saturation=0.25)
            served = self.batches(noise, 120)
            rows = noise.batcher(gf.SeedTree(5), "train", **OFFLINE_WINDOW,
                                 num_examples_per_batch=self.BATCH).plan.rows[:, 0, :]
        # The split's own rows (not the layout's: the split need not hold as many
        # long rows as short), each weighted by its window starts.
        starts = (rows[:, 1] - rows[:, 0]) - FOOTPRINT_SECONDS
        expected = np.sum(starts[rows[:, 1] - rows[:, 0] == LONG]) / np.sum(starts)
        # Measured: the share's standard deviation over seeds 0-11 is 0.0053
        # (gravyflow 164a167, this host; binomial alone gives 0.0043, the rows'
        # mixing windows the rest), times k = 4.
        assert abs(loud_share(served) - expected) < 4 * 0.0053, \
            f"loud share {loud_share(served):.3f}, where the rows' lengths give {expected:.3f}"

    def test_serves_past_one_pass(self):
        with gwosc_serving(HashedFetch(PSEUDO_LAYOUT)):
            served = self.batches(pseudo_real(PSEUDO_LAYOUT, saturation=0.25), 400)
        assert len(served) == 400, f"the reader ended after {len(served)} batches"

    def test_a_batch_is_a_function_of_its_address(self):
        kwargs = dict(**OFFLINE_WINDOW, num_examples_per_batch=8, steps_per_epoch=10)
        tree = gf.SeedTree(5)
        with gwosc_serving(HashedFetch(PSEUDO_LAYOUT)):
            noise = pseudo_real(PSEUDO_LAYOUT, saturation=0.25)
            batch = noise.batcher(tree, "train", **kwargs)
            late = np.asarray(batch(tree.batch("train", 3, 7))[RV.ONSOURCE])
            early = np.asarray(batch(tree.batch("train", 0, 0))[RV.ONSOURCE])
            again = noise.batcher(tree, "train", **kwargs)
            np.testing.assert_array_equal(np.asarray(again(tree.batch("train", 0, 0))[RV.ONSOURCE]), early)
            np.testing.assert_array_equal(np.asarray(again(tree.batch("train", 3, 7))[RV.ONSOURCE]), late)
        assert not np.array_equal(early, late)

    def test_a_batch_colours_by_the_rows_of_its_place_in_the_passes(self):
        # Each block's segment has its own amplitude, so an example's level names its row.
        amplitudes = [1.5 ** i for i in range(24)]
        layout = science_layout([LONG], amplitudes, blocks=24)
        steps, address = 10, (3, 7)
        tree = gf.SeedTree(5)
        with gwosc_serving(HashedFetch(layout)):
            batch = pseudo_real(layout, saturation=0.25).batcher(
                tree, "train", **OFFLINE_WINDOW, num_examples_per_batch=32, steps_per_epoch=steps)
            levels = np.std(np.asarray(batch(tree.batch("train", *address))[RV.ONSOURCE]), axis=-1)[:, 0]
        plan = batch.plan
        row_amplitude = np.array([amplitudes[int((start - FIRST_BLOCK) // BLOCK_SECONDS)]
                                  for start in plan.rows[:, 0, 0]])
        # Uniform noise of amplitude a has standard deviation a / sqrt(12); colouring keeps its power.
        drawn = np.argmin(np.abs(np.log(levels[:, None] * np.sqrt(12.0) / row_amplitude[None, :])), axis=1)
        g = address[0] * steps + address[1]
        assert set(plan.mixed(g).tolist()) != set(plan.mixed(address[1]).tolist())
        assert set(drawn.tolist()) <= set(plan.mixed(g).tolist()), \
            f"batch {g} drew rows {sorted(set(drawn.tolist()))}, outside its mixing window {sorted(set(plan.mixed(g).tolist()))}"

    def test_a_short_piece_is_left_out_not_read(self):
        # Every block also holds a 1 s science segment, whose interior is shorter than a window.
        layout = science_layout([1.0], [QUIET], blocks=24, offset=150.0) + science_layout([LONG], [QUIET], blocks=24)
        with gwosc_serving(HashedFetch(layout)):
            served = self.batches(pseudo_real(layout, segment_order=gf.SegmentOrder.SHORTEST_FIRST), 3)
        assert len(served) == 3
        assert all(np.all(np.isfinite(np.asarray(b[RV.ONSOURCE]))) for b in served)

    def test_a_row_shorter_than_a_psd_segment_is_left_out(self):
        # A window so short that 3 s pieces (768 samples) are long enough for it, but not for one PSD segment.
        window = dict(sample_rate_hertz=OFFLINE_RATE, onsource_duration_seconds=0.25, crop_duration_seconds=0.0,
                      offsource_duration_seconds=0.25)
        layout = science_layout([3.0], [QUIET], blocks=24, offset=150.0) + science_layout([LONG], [QUIET], blocks=24)
        with gwosc_serving(HashedFetch(layout)):
            served = list(islice(pseudo_real(layout, segment_order=gf.SegmentOrder.SHORTEST_FIRST)(
                **window, num_examples_per_batch=self.BATCH, group="train", seed=5), 3))
        assert len(served) == 3

    def test_the_spectra_kept_are_bounded_by_the_rows_not_the_windows(self):
        tree = gf.SeedTree(5)
        with gwosc_serving(HashedFetch(PSEUDO_LAYOUT)):
            noise = pseudo_real(PSEUDO_LAYOUT, saturation=0.25)
            reader = noise._reader([gf.IFO.L1], tree, "train", OFFLINE_RATE)
            batch = gf_noise.pseudo_real_batcher(reader, tree, gf.WindowSpec.from_params(**OFFLINE_WINDOW), 16,
                                                 [gf.IFO.L1])
            for k in range(60):
                batch(tree.batch("train", 0, k))
        bins = gf_noise.PSEUDO_REAL_NPERSEG // 2 + 1
        assert 0 < len(batch.spectra) <= len(batch.plan.rows)
        for normalised, scales in batch.spectra.values():
            assert isinstance(normalised, np.ndarray) and normalised.shape == (1, bins) and scales.shape == (1,)
        assert sum(n.nbytes + s.nbytes for n, s in batch.spectra.values()) <= len(batch.plan.rows) * (bins + 1) * 4

    def test_each_split_is_coloured_by_its_own_rows_only(self):
        # Each block's segment has its own amplitude, so an example's level names its block, and so its split.
        amplitudes = [1.5 ** i for i in range(24)]
        layout = science_layout([LONG], amplitudes, blocks=24)
        noise = gf.NoiseObtainer(noise_type=gf.NoiseType.PSEUDO_REAL, ifos=[gf.IFO.L1],
                                 ifo_data_obtainer=offline_noise(layout, split_spec=THIRDS, saturation=0.25))
        tree = gf.SeedTree(5)
        with gwosc_serving(HashedFetch(layout)):
            batchers = {split: noise.batcher(tree, split, **OFFLINE_WINDOW, num_examples_per_batch=32)
                        for split in ("train", "validate", "test")}
            levels = {split: [] for split in batchers}
            for k in range(6):
                for split, batch in batchers.items():
                    levels[split].append(np.std(np.asarray(batch(tree.batch(split, 0, k))[RV.ONSOURCE]), axis=-1))
        first_block = int(FIRST_BLOCK // BLOCK_SECONDS)
        for split, drawn in levels.items():
            blocks = np.argmin(np.abs(np.log(np.concatenate(drawn).ravel()[:, None] * np.sqrt(12.0)
                                             / np.asarray(amplitudes)[None, :])), axis=1)
            splits = {THIRDS.block_split(first_block + int(b)) for b in blocks}
            assert splits == {split}, f"the {split} batches were coloured by rows of {splits}"

    def test_grid_is_refused(self):
        with pytest.raises(ValueError, match="no GRID mode"):
            pseudo_real(PSEUDO_LAYOUT).batcher(gf.SeedTree(5), "train", **OFFLINE_WINDOW,
                                               sampling_mode=gf.SamplingMode.GRID)

    def test_a_row_without_power_is_refused(self):
        layout = science_layout([LONG], [0.0, QUIET, QUIET], blocks=24)
        with gwosc_serving(HashedFetch(layout)):
            with pytest.raises(ValueError, match="no power"):
                self.batches(pseudo_real(layout, saturation=0.25), 200)

    def test_precomputed_segments_are_served(self):
        with gwosc_serving(HashedFetch(PSEUDO_LAYOUT)):
            obtainer = offline_noise(PSEUDO_LAYOUT, saturation=0.25)
            obtainer.get_valid_segments([gf.IFO.L1], 5, "train")
            noise = gf.NoiseObtainer(noise_type=gf.NoiseType.PSEUDO_REAL, ifos=[gf.IFO.L1],
                                     ifo_data_obtainer=obtainer)
            served = self.batches(noise, 2)
        assert len(served) == 2

    def test_both_are_index_addressed(self):
        with gwosc_serving(HashedFetch(PSEUDO_LAYOUT)):
            pseudo = gf.Dataset(noise_obtainer=pseudo_real(PSEUDO_LAYOUT, saturation=0.25), **OFFLINE_WINDOW,
                                num_examples_per_batch=4, group="train", seed=5, steps_per_epoch=3,
                                input_variables=[gf.ReturnVariables.ONSOURCE],
                                output_variables=[gf.ReturnVariables.INJECTION_MASKS])
        with gwosc_serving(HashedFetch(GRID_LAYOUT)):
            grid = grid_dataset("test")
        assert pseudo.seed_metadata["index_addressed"] and grid.seed_metadata["index_addressed"]
