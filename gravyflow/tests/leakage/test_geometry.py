"""
Window geometry and per-row independence, decoded from the data.

Every returned window is read back from the fake strain's provenance code: where it
was cut, from which fetched piece, in which split's time, and which raw samples each
output row depends on.
"""

import numpy as np
import pytest

import gravyflow as gf
from gravyflow.src.dataset import dataset as dataset_module

from . import audit
from . import fixtures as fx
from .provenance import decode_rows

RV = gf.ReturnVariables
IFOS = ["H1", "L1"]
MODES = [gf.SamplingMode.RANDOM, gf.SamplingMode.GRID]

#: Layouts whose pieces touch a block edge between two splits, and pieces of awkward
#: lengths (512.2 s, 1030 s, 2048 s, and a short piece with an odd sample count).
LAYOUTS = {
    "train|test": (lambda: fx.straddle("train", "test", half_width=512.0), ("train", "test")),
    "validate|train": (lambda: fx.straddle("validate", "train", half_width=512.0), ("validate", "train")),
    "odd pieces": (fx.odd_pieces, ("train",)),
}


def _sets(fixture, directory, splits, mode, path):
    sets = {}
    for split in splits:
        if path == "obtainer":
            batches = fx.noise_batches(fixture, directory / split, split, mode=mode)
        else:
            batches = fx.dataset_batches(fx.dataset(fixture, directory / split, split, mode=mode))
        sets[f"{path} {split}"] = (split, batches)
    return sets


@pytest.mark.parametrize("path", ["obtainer", "dataset"])
@pytest.mark.parametrize("mode", MODES, ids=lambda m: m.name)
@pytest.mark.parametrize("layout", list(LAYOUTS))
def test_window_inside_own_piece(world, tmp_path, layout, mode, path):
    """
    Each decoded window lies inside one fetched piece of its own detector, inside the
    library's served time for its split, and inside a block of its split; its offsource
    ends where its padded onsource starts; and every sample is its own GPS time (the
    decoder refuses any drop, duplicate, reversal or rescale). The served time itself
    is judged against the golden table, every segment of it, so a split edge that
    moves by any amount fails even where no sampled window lands.
    """
    build, splits = LAYOUTS[layout]
    fixture = build()
    with world(fixture) as harness:
        records = audit.collect(
            harness, _sets(fixture, tmp_path, splits, mode, path), 2, IFOS, fx.window(),
            onsource_cropped=(path == "dataset"),
        )
        served = {split: fx.served_segments(fixture, IFOS, split) for split in splits}
    for split, segments in served.items():
        audit.assert_served_in_split(segments, split, fixture.origin)
    audit.assert_inside_one_fetch(records.values(), harness)
    audit.assert_in_split(records.values(), fixture.origin)
    for record in records.values():
        audit.assert_inside([record], served[record.split])


# =============================================================================
# EVERY OUTPUT ROW DEPENDS ONLY ON ITS OWN RAW ROW
# =============================================================================
#
# The perturbation is applied to raw rows as acquisition hands them on (after the
# piece's resampling), so the test isolates what the per-row stages (PSD, whitening,
# SNR scaling and labels) read. Every row is perturbed in turn: a dependency from any
# row to any other is seen, whichever pair it joins. It is a gain rising along the
# row, never a uniform scale, which the scale-free whitening rightly ignores.

PERTURBATION = 2.0 ** -3


def _perturbed(values):
    return values * (1.0 + PERTURBATION * np.linspace(0.0, 1.0, values.shape[-1], dtype=np.float32))


def _snr_generator():
    return gf.WNBGenerator(
        duration_seconds=gf.Distribution(value=0.5, type_=gf.DistributionType.CONSTANT),
        min_frequency_hertz=gf.Distribution(value=40.0, type_=gf.DistributionType.CONSTANT),
        max_frequency_hertz=gf.Distribution(value=100.0, type_=gf.DistributionType.CONSTANT),
        scaling_method=gf.ScalingMethod(
            value=gf.Distribution(value=12.0, type_=gf.DistributionType.CONSTANT),
            type_=gf.ScalingTypes.SNR,
        ),
    )


def _copy_batch(batch, rows=()):
    """
    A copy of a raw batch dict, with ONSOURCE and OFFSOURCE perturbed at each index in
    ``rows``: (j, d) is one row, (j,) every detector of example j.
    """
    out = dict(batch)
    for key in (RV.ONSOURCE, RV.OFFSOURCE):
        values = np.array(batch[key], dtype=np.float32, copy=True)
        for row in rows:
            values[row] = _perturbed(values[row])
        out[key] = values
    return out


class PsdSpy:
    """Records what the Dataset hands its PSD estimator, and what it gets back."""

    def __init__(self):
        self.inputs, self.outputs = [], []
        self._estimate = dataset_module.estimate_psd

    def __call__(self, background, *args, **kwargs):
        spectrum = self._estimate(background, *args, **kwargs)
        self.inputs.append(np.asarray(background))
        self.outputs.append({"normalised": np.asarray(spectrum.normalised),
                             "scale": np.asarray(spectrum.scale)})
        return spectrum


def _dataset_batch(world, fixture, directory, rows=(), generators=False):
    """
    One fresh Dataset batch (noise only, or with an SNR-scaled burst) with the raw
    rows ``rows`` perturbed, the raw noise batch it was made from, and the PSD the
    Dataset estimated (its input and output). Each build draws the same raw data and
    the same injection streams, so builds differ only where the perturbation reaches.
    """
    with world(fixture) as harness:
        spy = PsdSpy()
        harness.monkeypatch.setattr(dataset_module, "estimate_psd", spy)
        ds = fx.dataset(
            fixture, directory, "train",
            input_variables=[RV.ONSOURCE, RV.OFFSOURCE, RV.WHITENED_ONSOURCE],
            output_variables=[RV.INJECTIONS, RV.WHITENED_INJECTIONS, gf.ScalingTypes.SNR]
            if generators else [RV.START_GPS_TIME],
            waveform_generators=[_snr_generator()] if generators else None,
        )
        noise, raws = ds.noise, []

        def perturbed(streams):
            raws.append(noise(streams))
            return _copy_batch(raws[-1], rows)

        ds.noise = perturbed
        batch = {key: np.asarray(value) for part in ds[0] for key, value in part.items()}
        (raw,) = raws
        assert harness.source.snapshot()[0], "the batch fetched nothing"
    (received,), (psd,) = spy.inputs, spy.outputs
    return batch, raw, received, psd


def _assert_only_row_changed(control, perturbed, row, what):
    """Row ``row`` of the (B, I, ...) arrays changed, and no other row did."""
    j, d = row
    assert not np.array_equal(control[j, d], perturbed[j, d]), \
        f"{what}: the perturbed row [{j}, {d}] did not change"
    for b, k in np.ndindex(control.shape[:2]):
        if (b, k) != (j, d):
            assert np.array_equal(control[b, k], perturbed[b, k]), \
                f"{what}: row [{b}, {k}] changed with row [{j}, {d}]'s data"


def test_psd_is_per_row(world, tmp_path):
    """
    The PSD estimator is handed exactly each row's own offsource (decoded from what it
    receives). Perturbing each raw row [j, d] in turn changes that row's PSD and
    whitened onsource and no other row's. With an SNR-scaled injection, perturbing
    each example in turn changes no other example's whitened onsource, injections or
    SNR label (the detectors of one example share its network SNR scale by design).
    """
    fixture = fx.straddle("train", "test", half_width=256.0)
    whitened = RV.WHITENED_ONSOURCE.name
    control, raw, received, psd = _dataset_batch(world, fixture, tmp_path / "noise")
    own = decode_rows(raw[RV.OFFSOURCE], IFOS, fixture.encoding, what="raw offsource")
    assert np.array_equal(decode_rows(received, IFOS, fixture.encoding, what="PSD input"), own)
    assert np.all(own[..., 1] - own[..., 0] == fx.window().offsource)

    rows = list(np.ndindex(control[whitened].shape[:2]))
    assert len(rows) == 16
    for row in rows:
        perturbed, _, _, psd_b = _dataset_batch(world, fixture, tmp_path / f"noise-{row}", [row])
        _assert_only_row_changed(control[whitened], perturbed[whitened], row, "whitened onsource")
        for part in ("normalised", "scale"):
            _assert_only_row_changed(psd[part], psd_b[part], row, f"PSD {part}")

    injected, *_ = _dataset_batch(world, fixture, tmp_path / "injected", generators=True)
    assert np.array_equal(injected[RV.OFFSOURCE.name], control[RV.OFFSOURCE.name])
    keys = (whitened, RV.INJECTIONS.name, RV.WHITENED_INJECTIONS.name, gf.ScalingTypes.SNR.name)
    examples = injected[RV.INJECTIONS.name].shape[0]
    assert examples == 8
    for j in range(examples):
        perturbed, *_ = _dataset_batch(world, fixture, tmp_path / f"injected-{j}", [(j,)], True)
        assert not np.array_equal(injected[RV.INJECTIONS.name][j], perturbed[RV.INJECTIONS.name][j]), \
            f"example {j}'s injections did not change with its noise"
        others = np.arange(examples) != j
        for key in keys:
            assert np.array_equal(injected[key][others], perturbed[key][others]), \
                f"{key}: another example changed with example {j}'s data"


def test_obtainer_whitening_is_per_row(world, tmp_path):
    """
    The obtainer's own whiten=True path: perturbing each raw row in turn changes that
    row's whitened onsource and no other row's. The raw batch is captured where the
    obtainer hands it to its post-processing, then replayed through it.
    """
    fixture = fx.straddle("train", "test", half_width=256.0)
    with world(fixture) as harness:
        obtainer = fx.ifo_data_obtainer(fixture)
        post_process, captured = obtainer._post_process_batch, []

        def capturing(batch, *args, **kwargs):
            captured.append((_copy_batch(batch), args, kwargs))
            return post_process(batch, *args, **kwargs)

        obtainer._post_process_batch = capturing
        ifos = [gf.IFO[name] for name in IFOS]
        obtainer.get_valid_segments(ifos, 7, "train")
        generator = obtainer.get_onsource_offsource_chunks(
            256.0, fx.ONSOURCE_SECONDS, fx.CROP_SECONDS, fx.OFFSOURCE_SECONDS, 8, ifos,
            1.0, 7, gf.SamplingMode.RANDOM, whiten=True,
        )
        served = np.asarray(next(generator)[RV.ONSOURCE])
        assert harness.source.snapshot()[0], "the whitened batch fetched nothing"
    raw, args, kwargs = captured[0]

    def run(rows=()):
        return np.asarray(post_process(_copy_batch(raw, rows), *args, **kwargs)[RV.ONSOURCE])

    control = run()
    assert np.array_equal(control, served), "the replay differs from the served batch"
    rows = list(np.ndindex(control.shape[:2]))
    assert len(rows) == 16
    for row in rows:
        _assert_only_row_changed(control, run([row]), row, "whitened onsource")


# L7: START_GPS_TIME is the GPS time of the first returned (cropped) onsource sample.
@pytest.mark.parametrize("path", ["obtainer", "dataset"])
@pytest.mark.parametrize("mode", [gf.SamplingMode.RANDOM, gf.SamplingMode.GRID], ids=lambda m: m.name)
@pytest.mark.parametrize("rate", [1024.0, 2048.0, 4096.0])
def test_start_gps_is_first_sample(world, tmp_path, rate, mode, path):
    fixture = fx.odd_pieces(rate=rate)
    window = fx.window(rate)
    with world(fixture) as harness:
        if path == "obtainer":
            batches = fx.noise_batches(fixture, tmp_path, "train", rate=rate, mode=mode)
        else:
            batches = fx.dataset_batches(fx.dataset(fixture, tmp_path, "train", rate=rate, mode=mode))
        record = audit.collect(harness, {path: ("train", batches)}, 2, IFOS, window,
                               onsource_cropped=(path == "dataset"))[path]
    assert record.start_labels is not None
    cropped_start = (record.footprints[..., 0] + window.offsource + window.crop) / rate
    error = record.start_labels - cropped_start
    assert np.max(np.abs(error)) <= 1.0 / rate, (
        f"START_GPS_TIME minus the decoded first onsource sample: min {error.min():.6f} s, "
        f"max {error.max():.6f} s"
    )


#: The resampling code: three in-band tones, each phased by absolute GPS time.
TONE_EPOCH = 1238175000.0
TONES_HERTZ = np.array([61.25, 211.75, 487.5])
TONE_PHASES = np.array([0.3, 1.9, 4.4])


def _tones(times):
    return np.sin(2 * np.pi * TONES_HERTZ * (np.asarray(times)[:, None] - TONE_EPOCH) + TONE_PHASES).sum(axis=1)


def _decoded_drift(samples, times):
    """
    How late ``samples`` are against their claimed ``times``, in seconds: each tone's
    phase fitted by least squares, against the phase the code gives those times.
    """
    phase = 2 * np.pi * TONES_HERTZ * (np.asarray(times)[:, None] - TONE_EPOCH)
    design = np.concatenate([np.sin(phase), np.cos(phase)], axis=1)
    (sines, cosines), *_ = np.linalg.lstsq(design, samples, rcond=None)[0].reshape(1, 2, -1)
    offset = np.angle(np.exp(1j * (np.arctan2(cosines, sines) - TONE_PHASES)))
    return float(np.mean(offset / (2 * np.pi * TONES_HERTZ)))


# Resampling keeps time: sample n of a resampled piece is at t0 + n / rate, to a tenth
# of a sample, along the whole piece, for any piece length.
@pytest.mark.parametrize("extra", [0, 1, 3], ids=lambda k: f"N+{k}")
@pytest.mark.parametrize("rate_in", [4096.0, 2048.0, 16384.0])
def test_resample_keeps_time(rate_in, extra):
    rate_out = 2048.0
    t0 = TONE_EPOCH + 100.3  # off the sample grid of either rate
    num_samples = int(2 * rate_in) + extra
    resampled, start = gf.resample(_tones(t0 + np.arange(num_samples) / rate_in), t0, rate_in, rate_out)
    assert start == t0
    width = 128
    for fraction in (0.4, 0.5, 0.9):
        centre = int(fraction * len(resampled))
        n = np.arange(centre - width, centre + width)
        drift = _decoded_drift(np.asarray(resampled[n], dtype=np.float64), t0 + n / rate_out) * rate_out
        assert abs(drift) <= 0.1, (
            f"{num_samples} samples at {rate_in:g} Hz: sample {centre} of {len(resampled)} "
            f"is {drift:+.3f} samples off t0 + n / {rate_out:g}"
        )
