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
    decoder refuses any drop, duplicate, reversal or rescale).
    """
    build, splits = LAYOUTS[layout]
    fixture = build()
    with world(fixture) as harness:
        records = audit.collect(
            harness, _sets(fixture, tmp_path, splits, mode, path), 2, IFOS, fx.window(),
            onsource_cropped=(path == "dataset"),
        )
        served = {split: fx.served_segments(fixture, IFOS, split) for split in splits}
    audit.assert_inside_one_fetch(records.values(), harness)
    audit.assert_in_split(records.values(), fixture.origin)
    for record in records.values():
        audit.assert_inside([record], served[record.split])


# =============================================================================
# EVERY OUTPUT ROW DEPENDS ONLY ON ITS OWN RAW ROW
# =============================================================================
#
# The perturbation is applied to one raw row as acquisition hands it on (after the
# piece's resampling, whose float32 FFT couples every sample of a piece through its
# rounding), so the test isolates what the per-row stages (PSD, whitening, SNR
# scaling and labels) read.

PERTURBATION = 1.0 + 2.0 ** -3


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


def _perturb_row(batch, row):
    """A copy of a raw batch dict with ONSOURCE and OFFSOURCE of row (j, d) scaled."""
    out = dict(batch)
    for key in (RV.ONSOURCE, RV.OFFSOURCE):
        values = np.array(batch[key], dtype=np.float32, copy=True)
        values[row] *= PERTURBATION
        out[key] = values
    return out


def _perturbing(iterator, row):
    for batch in iterator:
        yield _perturb_row(batch, row)


class PsdSpy:
    """Records what the Dataset hands its PSD estimator, and what it gets back."""

    def __init__(self):
        self.inputs, self.outputs = [], []
        self._estimate = dataset_module.estimate_psd

    def __call__(self, background, *args, **kwargs):
        spectrum = self._estimate(background, *args, **kwargs)
        self.inputs.append(np.asarray(background))
        self.outputs.append((np.asarray(spectrum.normalised), np.asarray(spectrum.scale)))
        return spectrum


def _drawing(draw, drawn):
    """Yield one batch, drawn lazily (so collect sees its fetches), and keep it."""
    drawn.append(draw())
    yield drawn[-1]


def _dataset_batch(world, fixture, directory, generators, row=None):
    """One Dataset batch (noise-only or with an SNR-scaled burst) and the PSD spy."""
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
        if row is not None:
            ds.noise = _perturbing(ds.noise, row)
        drawn = []
        records = audit.collect(harness, {"train": ("train", _drawing(lambda: ds[0], drawn))}, 1,
                                IFOS, fx.window(), onsource_cropped=True) \
            if not generators and row is None else None
        if not drawn:
            drawn.append(ds[0])
    batch = {key: np.asarray(value) for part in drawn[0] for key, value in part.items()}
    return batch, spy, (records["train"].footprints if records else None)


def _assert_only_row_changed(control, perturbed, key, row, what):
    j, d = row
    assert not np.array_equal(control[key][j, d], perturbed[key][j, d]), \
        f"{what}: the perturbed row did not change"
    for b, k in np.ndindex(control[key].shape[:2]):
        if (b, k) != (j, d):
            assert np.array_equal(control[key][b, k], perturbed[key][b, k]), \
                f"{what}: row [{b}, {k}] changed with row [{j}, {d}]'s data"


def test_psd_is_per_row(world, tmp_path):
    """
    The PSD estimator is handed exactly each row's own offsource (decoded from what it
    receives). Perturbing one raw row [j, d] changes no other row's PSD or whitened
    onsource, and no other example's injections or SNR labels.
    """
    fixture = fx.straddle("train", "test", half_width=256.0)
    noise, spy, prints = _dataset_batch(world, fixture, tmp_path / "noise", generators=False)
    assert len(spy.inputs) == 1
    received = decode_rows(spy.inputs[0], IFOS, fixture.encoding, what="PSD input")
    assert np.array_equal(received[..., 0], prints[..., 0])
    assert np.all(received[..., 1] - received[..., 0] == fx.window().offsource)

    row = (1, 0)
    # Noise only: one PSD and one whitened onsource per row.
    noise_b, spy_b, _ = _dataset_batch(world, fixture, tmp_path / "noise-b", generators=False, row=row)
    psd_changed = False
    for index, part in enumerate(("normalised", "scale")):
        a, b = spy.outputs[0][index], spy_b.outputs[0][index]
        for other in np.ndindex(a.shape[:2]):
            if other != row:
                assert np.array_equal(a[other], b[other]), f"PSD {part} of row {other} changed"
        psd_changed |= not np.array_equal(a[row], b[row])
    assert psd_changed, "the perturbed row's PSD did not change"
    _assert_only_row_changed(noise, noise_b, RV.WHITENED_ONSOURCE.name, row, "whitened onsource")

    # With an SNR-scaled injection: detectors of one example share the network SNR
    # scale by design; every other example is bit-identical.
    control, _, _ = _dataset_batch(world, fixture, tmp_path / "a", generators=True)
    perturbed, _, _ = _dataset_batch(world, fixture, tmp_path / "b", generators=True, row=row)
    assert np.array_equal(control[RV.OFFSOURCE.name], noise[RV.OFFSOURCE.name])
    others = np.arange(control[RV.INJECTIONS.name].shape[0]) != row[0]
    for key in (RV.WHITENED_ONSOURCE.name, RV.INJECTIONS.name, RV.WHITENED_INJECTIONS.name,
                gf.ScalingTypes.SNR.name):
        assert np.array_equal(control[key][others], perturbed[key][others]), key
    assert not np.array_equal(control[RV.INJECTIONS.name][row[0]], perturbed[RV.INJECTIONS.name][row[0]])


def test_obtainer_whitening_is_per_row(world, tmp_path):
    """The obtainer's own whiten=True path: a perturbed raw row changes no other row."""
    fixture = fx.straddle("train", "test", half_width=256.0)
    row = (2, 1)

    def run(perturb):
        with world(fixture) as harness:
            obtainer = fx.ifo_data_obtainer(fixture)
            if perturb:
                post_process = obtainer._post_process_batch

                def perturbed_post_process(batch, *args, **kwargs):
                    return post_process(_perturb_row(batch, row), *args, **kwargs)

                obtainer._post_process_batch = perturbed_post_process
            ifos = [gf.IFO[name] for name in IFOS]
            obtainer.get_valid_segments(ifos, 7, dict(fx.GOLDEN_GROUPS), "train")
            generator = obtainer.get_onsource_offsource_chunks(
                256.0, fx.ONSOURCE_SECONDS, fx.CROP_SECONDS, fx.OFFSOURCE_SECONDS, 8, ifos,
                1.0, 7, gf.SamplingMode.RANDOM, whiten=True,
            )
            batch = next(generator)
            assert harness.source.snapshot()[0], "the whitened batch fetched nothing"
        return {key: np.asarray(value) for key, value in batch.items()}

    control, perturbed = run(False), run(True)
    _assert_only_row_changed(control, perturbed, RV.ONSOURCE, row, "whitened onsource")
