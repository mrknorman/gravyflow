"""
The dataset tap's own checks: a batch served again at an address it has seen
is compared with the first, never dropped unread.
"""

import numpy as np
import pytest

import gravyflow as gf
from gravyflow.tests.leakage.tap import DatasetTap

RV = gf.ReturnVariables


def _dataset():
    return gf.Dataset(
        seed=7,
        group="train",
        noise_obtainer=gf.NoiseObtainer(noise_type=gf.NoiseType.WHITE, ifos=[gf.IFO.L1]),
        sample_rate_hertz=256.0,
        onsource_duration_seconds=1.0,
        offsource_duration_seconds=2.0,
        crop_duration_seconds=0.5,
        num_examples_per_batch=2,
        steps_per_epoch=3,
        input_variables=[RV.ONSOURCE],
        output_variables=[],
    )


def test_a_batch_served_again_identically_is_recorded_once(monkeypatch):
    tap = DatasetTap(monkeypatch, ())
    ds = _dataset()
    first = ds[1][0][RV.ONSOURCE.name]
    again = ds[1][0][RV.ONSOURCE.name]
    np.testing.assert_array_equal(np.asarray(first), np.asarray(again))
    (record,) = tap.take()
    assert record.addresses == [(0, 1)]
    assert tap.serves == [("train", 0, 1), ("train", 0, 1)]


def test_other_data_served_at_a_seen_address_is_refused(monkeypatch):
    # Keras trains on the batches it asks for after its structure pass: a
    # dataset that served other data at that address must not pass unread.
    tap = DatasetTap(monkeypatch, ())
    ds = _dataset()
    ds[0]
    ds.tree = gf.SeedTree(8)  # the plant: the same address now draws other noise
    with pytest.raises(AssertionError, match=r"\(epoch 0, batch 0\).*ONSOURCE"):
        ds[0]
    assert len(tap.take()[0].batches) == 1
