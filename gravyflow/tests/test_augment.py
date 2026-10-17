"""
Augmentation is opt-in and train only. Each example draws each augmentation once,
from that augmentation's own stream, and the draw applies alike to its onsource
and offsource; AUGMENTATIONS flags what each example had. Real events refuse time
reversal and random shift, and noise refuses random shift.

The fake world's checks (evaluation, the default batch, windows unmoved by
augmentation, injection after augmentation) are in ``leakage/test_streams.py``.
"""
import logging

import numpy as np
import pytest

import gravyflow as gf
from gravyflow.src.dataset.acquisition import augmentations as module
from gravyflow.src.dataset.acquisition.augmentations import augment

B, I, T, T_OFF = 64, 2, 128, 512
GLITCHES, NOISE, EVENTS = [gf.DataLabel.GLITCHES], [gf.DataLabel.NOISE], [gf.DataLabel.EVENTS]


def streams(split="train", k=0, seed=11):
    return gf.SeedTree(seed).batch(split, 0, k)


def ramps(scale=1.0):
    """B identical examples, rising ramps, so any difference between examples is augmentation's."""
    on = np.broadcast_to(np.arange(1, T + 1, dtype=np.float32), (B, I, T)) * np.float32(scale)
    off = np.broadcast_to(np.arange(1, T_OFF + 1, dtype=np.float32), (B, I, T_OFF)) * np.float32(scale)
    return on.copy(), off.copy()


def reversals(row) -> int:
    """The SignReversal and TimeReversal bits a ramp row carries."""
    row = np.asarray(row, np.float64)
    bits = gf.SignReversal.bit if row[0] < 0 else 0
    steps = np.diff(np.abs(row))
    assert np.all(steps > 0) or np.all(steps < 0), "not a ramp"
    return bits | (gf.TimeReversal.bit if steps[0] < 0 else 0)


def test_none_by_default():
    for labels in (NOISE, GLITCHES, EVENTS):
        obtainer = gf.IFODataObtainer(data_quality=gf.DataQuality.BEST, data_labels=labels,
                                      observing_runs=[gf.ObservingRun.O3])
        assert obtainer.augmentations == ()
    assert not hasattr(gf, "default_augmentations")


def test_each_example_draws_once_for_both_windows():
    """B identical examples flip independently; each one's windows and detectors flip together."""
    on, off = ramps()
    spec = [gf.SignReversal(probability=0.5), gf.TimeReversal(probability=0.5)]
    out = augment((on, off), streams(), spec, "train", GLITCHES)
    flags = np.asarray(out.applied)
    assert flags.dtype == np.float32 and flags.shape == (B,)
    for b in range(B):
        bits = {reversals(row) for rows in (out.onsource, out.offsource) for row in np.asarray(rows)[b]}
        assert bits == {int(flags[b])}, (b, bits, flags[b])
    assert set(flags.astype(int)) == {0, 1, 2, 3}
    # The inputs are never written.
    for given, original in zip((on, off), ramps()):
        np.testing.assert_array_equal(given, original)


def test_each_augmentation_draws_from_its_own_stream():
    """
    An augmentation's choice is the first B draws of its own ``augment/{name}``
    stream, whatever else the spec holds and in whatever order.
    """
    on, off = ramps()
    sign = gf.SignReversal(probability=0.5)
    alone = augment((on, off), streams(), [sign], "train", GLITCHES).applied
    expected = streams().stream("augment/SignReversal").random(B) < 0.5
    np.testing.assert_array_equal(alone == sign.bit, expected)
    others = [gf.AddNoise(probability=0.5), gf.TimeReversal(probability=0.5)]
    for spec in ([sign, *others], [*others, sign]):
        flags = augment((on, off), streams(), spec, "train", GLITCHES).applied
        np.testing.assert_array_equal(flags.astype(int) & sign.bit, alone.astype(int))
    # Another batch draws afresh.
    assert not np.array_equal(augment((on, off), streams(k=1), [sign], "train", GLITCHES).applied, alone)


@pytest.mark.parametrize("split, mode", [("validate", None), ("test", None),
                                         ("train", gf.SamplingMode.GRID)])
def test_evaluation_is_never_augmented(split, mode, monkeypatch, caplog):
    """Validate, test and a GRID pass come back as they were, flagged 0, which is logged once."""
    monkeypatch.setattr(module, "_LOGGED", set())
    on, off = ramps()
    spec = [gf.SignReversal(probability=1.0), gf.AddNoise(probability=1.0)]
    with caplog.at_level(logging.INFO, logger=module.__name__):
        for _ in range(2):
            out = augment((on, off), streams(split), spec, split, GLITCHES, sampling_mode=mode)
            assert out.onsource is on and out.offsource is off
            np.testing.assert_array_equal(out.applied, np.zeros(B, np.float32))
    assert len([r for r in caplog.records if "never augmented" in r.getMessage()]) == 1


def test_an_empty_spec_returns_the_strain_as_it_is():
    on, off = ramps()
    out = augment((on, off), streams(), (), "train")
    assert out.onsource is on and out.offsource is off
    np.testing.assert_array_equal(out.applied, np.zeros(B, np.float32))


def test_add_noise_scales_each_example_by_its_own_background():
    """
    AddNoise adds noise of ``amplitude`` times each example's own offsource std,
    per detector, alike to both windows: an example's noise never depends on the
    batch, and an example with no variance gets none.
    """
    rng = np.random.default_rng(3)
    scales = np.array([1e-21, 1e-18, 0.0, 1e-21])
    on = (rng.standard_normal((4, I, T)) * scales[:, None, None]).astype(np.float32)
    off = (rng.standard_normal((4, I, 4 * T_OFF)) * scales[:, None, None]).astype(np.float32)
    out = augment((on, off), gf.SeedTree(5).batch("train", 0, 0), [gf.AddNoise(probability=1.0, amplitude=0.1)],
                  "train", GLITCHES)
    np.testing.assert_array_equal(out.applied, np.full(4, gf.AddNoise.bit, np.float32))
    varied = scales > 0
    for windows, original in ((out.onsource, on), (out.offsource, off)):
        added = np.asarray(windows, np.float64) - original
        assert np.all(added[~varied] == 0.0)
        ratio = np.std(added[varied], axis=-1) / np.std(off[varied].astype(np.float64), axis=-1)
        np.testing.assert_allclose(ratio, 0.1, rtol=0.2)


def test_random_shift_rolls_both_windows_alike():
    on, off = ramps()
    shift = gf.RandomShift(probability=1.0, shift_fraction=0.25)
    out = augment((on, off), streams(), [shift], "train", GLITCHES)

    def shifts(rows, n):
        moved = (np.argmax(np.asarray(rows), axis=-1) - (n - 1)) % n
        return np.where(moved > n // 2, moved - n, moved)

    on_shift, off_shift = shifts(out.onsource, T), shifts(out.offsource, T_OFF)
    np.testing.assert_array_equal(on_shift, off_shift)
    assert np.all(on_shift == on_shift[:, :1]) and np.all(np.abs(on_shift) <= T // 4)
    assert len(np.unique(on_shift[:, 0])) > 1


@pytest.mark.parametrize("labels", [EVENTS, [gf.EventConfidence.CONFIDENT], [*GLITCHES, *EVENTS]])
def test_real_events_are_never_reversed_or_shifted(labels):
    for augmentation in (gf.TimeReversal(), gf.RandomShift()):
        with pytest.raises(ValueError, match="real events"):
            gf.TransientDataObtainer(data_quality=gf.DataQuality.BEST, data_labels=labels,
                                     augmentations=[augmentation])
        # Belt and braces: a spec changed after configuration is refused where it applies.
        with pytest.raises(ValueError, match="real events"):
            augment(ramps(), streams(), [augmentation], "train", labels)
    allowed = [gf.SignReversal(), gf.AddNoise()]
    obtainer = gf.TransientDataObtainer(data_quality=gf.DataQuality.BEST, data_labels=labels,
                                        augmentations=allowed)
    assert obtainer.augmentations == tuple(allowed)


def test_noise_refuses_random_shift():
    """Noise windows are drawn uniformly already: a shift would only wrap them."""
    with pytest.raises(ValueError, match="noise"):
        gf.IFODataObtainer(data_quality=gf.DataQuality.BEST, data_labels=NOISE,
                           augmentations=[gf.RandomShift()])
    with pytest.raises(ValueError, match="noise"):
        augment(ramps(), streams(), [gf.RandomShift()], "train", NOISE)


@pytest.mark.parametrize("spec, match", [
    (lambda: [gf.SignReversal(), gf.SignReversal(probability=0.1)], "more than once"),
    (lambda: [gf.TimeReversal(probability=1.5)], "probability"),
    (lambda: [gf.AddNoise(amplitude=-0.1)], "amplitude"),
    (lambda: [gf.RandomShift(shift_fraction=1.0)], "shift_fraction"),
    (lambda: ["SignReversal"], "not an augmentation"),
])
def test_a_malformed_spec_is_refused(spec, match):
    with pytest.raises(ValueError, match=match):
        gf.IFODataObtainer(data_quality=gf.DataQuality.BEST, data_labels=GLITCHES, augmentations=spec())


def test_saturation_is_independent_of_augmentation():
    plain, augmented = (gf.IFODataObtainer(data_quality=gf.DataQuality.BEST, data_labels=NOISE, augmentations=spec)
                        for spec in ([], [gf.SignReversal(), gf.TimeReversal()]))
    assert plain.saturation == augmented.saturation == 8.0
    assert not hasattr(augmented, "_get_effective_saturation")
