"""
Self-tests of the content fingerprints: each check catches a planted leak of
the kind it exists for, passes independent data, and refuses input it cannot
judge. A fingerprint that passed everything would make every leakage test
built on it vacuous.
"""

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import pytest

from gravyflow.tests.leakage.fingerprints import (
    OpaqueStateError, assert_rows_disjoint, assert_streams_independent, assert_unchanged,
    concatenate_streams, label_streams, shared_rows, state_digest, state_items, strain_prints,
)


def _noise(seed, shape):
    return np.random.default_rng(seed).standard_normal(shape).astype(np.float32)


# ------------------------------------------------------------------ strain

def test_independent_strain_shares_nothing():
    a = strain_prints({"ONSOURCE": _noise(1, (8, 2, 384)), "OFFSOURCE": _noise(2, (8, 2, 1024))})
    b = strain_prints({"ONSOURCE": _noise(3, (8, 2, 384)), "OFFSOURCE": _noise(4, (8, 2, 1024))})
    assert a.num_rows == b.num_rows == 32
    assert shared_rows(a, b) == {}
    assert_rows_disjoint({"train": a, "test": b})


def test_shifted_partial_copy_is_caught():
    off = _noise(1, (4, 2, 1024))
    on = _noise(2, (4, 2, 384))
    on[3, 1, 100:164] = off[0, 0, 500:564]          # 64 samples, at another offset, key and detector
    a = strain_prints({"OFFSOURCE": off})
    b = strain_prints({"ONSOURCE": on})
    assert shared_rows(b, a) == {("ONSOURCE", "OFFSOURCE"): 1}
    with pytest.raises(AssertionError, match=r"OFFSOURCE->ONSOURCE: 1"):
        assert_rows_disjoint({"train": a, "validate": b})


def test_overlap_shorter_than_a_window_is_not_a_window():
    off = _noise(1, (2, 1, 512))
    on = _noise(2, (2, 1, 256))
    on[0, 0, :63] = off[0, 0, 200:263]
    assert shared_rows(strain_prints({"ON": on}), strain_prints({"OFF": off})) == {}


def test_identical_rows_are_counted_per_row():
    x = _noise(5, (8, 2, 256))
    y = _noise(6, (8, 2, 256))
    y[:3] = x[:3]
    assert shared_rows(strain_prints({"ONSOURCE": y}), strain_prints({"ONSOURCE": x})) == {
        ("ONSOURCE", "ONSOURCE"): 6}


def test_strain_refuses_rows_it_cannot_judge():
    x = _noise(1, (4, 1, 256))
    x[2] = 0.0
    with pytest.raises(ValueError, match=r"ONSOURCE\[2, 0\]: the row is all zeros"):
        strain_prints({"ONSOURCE": x})
    y = _noise(1, (4, 1, 256))
    y[1, 0, 7] = np.nan
    with pytest.raises(ValueError, match="not finite"):
        strain_prints({"ONSOURCE": y})
    with pytest.raises(TypeError, match="float32"):
        strain_prints({"ONSOURCE": _noise(1, (4, 1, 256)).astype(np.float64)})
    with pytest.raises(ValueError, match="at least 64 samples"):
        strain_prints({"ONSOURCE": _noise(1, (4, 1, 32))})
    with pytest.raises(ValueError, match="every row was excluded"):
        strain_prints({"INJECTIONS": x}, absent={"INJECTIONS": np.ones((4, 1), bool)})


def test_absent_rows_and_constant_windows_are_excluded_and_counted():
    inj = np.zeros((4, 1, 256), np.float32)
    inj[0, 0, 100:200] = _noise(1, 100)
    prints = strain_prints({"INJECTIONS": inj}, absent={"INJECTIONS": np.array([[False], [True], [True], [True]])})
    assert prints.num_rows == 1
    # The zero head [0, 100) holds 37 all-zero windows; the 56-sample tail holds none.
    assert prints.constant_windows == 37
    assert prints.windows.size == 193 - 37
    other = np.zeros((2, 1, 256), np.float32)
    other[:, 0, 0:50] = _noise(2, (2, 50))
    assert shared_rows(prints, strain_prints({"INJECTIONS": other})) == {}


def test_disjoint_refuses_too_few_sets_or_rows():
    a = strain_prints({"ONSOURCE": _noise(1, (1, 1, 256))})
    with pytest.raises(ValueError, match="at least two sets"):
        assert_rows_disjoint({"train": a})
    with pytest.raises(ValueError, match="has 1 rows; at least 2 are needed"):
        assert_rows_disjoint({"train": a, "test": a})


# ------------------------------------------------------------------ labels

def _uniform(seed, n=256):
    return np.random.default_rng(seed).uniform(size=n)


def test_independent_streams_pass():
    report = assert_streams_independent({
        "train": {"SNR": _uniform(1), "RA": _uniform(2), "MASK": (_uniform(3) > 0.5).astype(float)},
        "validate": {"SNR": _uniform(4), "RA": _uniform(5), "MASK": (_uniform(6) > 0.5).astype(float)},
        "test": {"SNR": _uniform(7), "RA": _uniform(8), "MASK": (_uniform(9) > 0.5).astype(float)},
    })
    assert len(report["tested"]) == 27 and report["excluded"] == []


def test_same_stream_mapped_to_another_range_is_caught():
    u = _uniform(1)
    with pytest.raises(AssertionError, match=r"train/SNR vs validator/SNR: rho \+1.0000 at lag 0"):
        assert_streams_independent({"train": {"SNR": 8 + 12 * u}, "validator": {"SNR": 15 * u}})


def test_lagged_monotone_stream_is_caught():
    u = _uniform(1, 300)
    lagged = np.exp(u[3:259])           # the same draws, three examples later, transformed
    with pytest.raises(AssertionError, match=r"train/RA vs test/SNR: rho \+1.0000 at lag 3"):
        assert_streams_independent({"train": {"RA": u[:256]}, "test": {"SNR": lagged}})


def test_one_shared_float64_value_is_caught():
    a, b = _uniform(1), _uniform(2)
    b[17] = a[200]
    with pytest.raises(AssertionError, match=r"1 of 256 values of train appear exactly in test \(chance allows 0\)"):
        assert_streams_independent({"train": {"X": a}, "test": {"X": b}})


def test_a_stream_served_in_another_order_is_caught():
    a = (8 + 12 * _uniform(1, 64)).astype(np.float32)
    b = np.random.default_rng(3).permutation(a)
    with pytest.raises(AssertionError, match="64 of 64 values of train appear exactly in test"):
        assert_streams_independent({"train": {"SNR": a}, "test": {"SNR": b}})


def test_float32_chance_collisions_are_allowed():
    # 4096 independent float32 draws in [8, 20]: about 1.4 chance collisions per pair.
    a = (8 + 12 * _uniform(1, 4096)).astype(np.float32)
    b = (8 + 12 * _uniform(2, 4096)).astype(np.float32)
    assert np.count_nonzero(np.isin(a, b)) > 0, "the case must hold chance collisions"
    assert_streams_independent({"train": {"SNR": a}, "test": {"SNR": b}})


def test_discrete_labels_are_not_compared_for_equality():
    mask_a = (_uniform(1) > 0.5).astype(float)
    mask_b = (_uniform(2) > 0.5).astype(float)
    assert_streams_independent({"train": {"MASK": mask_a}, "test": {"MASK": mask_b}})


def test_constant_streams_are_excluded_by_name():
    report = assert_streams_independent({
        "train": {"X": _uniform(1), "C": np.full(256, 3.0)},
        "test": {"X": _uniform(2), "C": np.full(256, 3.0)},
    })
    assert report["excluded"] == ["train/C (constant 3)", "test/C (constant 3)"]
    with pytest.raises(ValueError, match="No stream left"):
        assert_streams_independent({"train": {"C": np.zeros(256)}, "test": {"C": np.ones(256)}})


def test_streams_refuse_short_sequences():
    with pytest.raises(ValueError, match="at least 64 examples"):
        assert_streams_independent({"train": {"X": _uniform(1, 63)}, "test": {"X": _uniform(2)}})


def test_label_streams_split_extra_axes_and_join_batches():
    batches = [label_streams({"RA": np.arange(6.0).reshape(3, 2) + 10 * k, "SNR": np.arange(3.0)}, 3)
               for k in range(2)]
    joined = concatenate_streams(batches)
    assert sorted(joined) == ["RA[0]", "RA[1]", "SNR"]
    np.testing.assert_array_equal(joined["RA[1]"], [1, 3, 5, 11, 13, 15])
    with pytest.raises(ValueError, match="expected 3 examples first"):
        label_streams({"RA": np.zeros((2, 3))}, 3)


# ------------------------------------------------------------- object state

@dataclass
class _Source:
    name: str
    rng: np.random.Generator
    values: np.ndarray
    children: list = field(default_factory=list)


def test_state_digest_sees_a_drawn_generator():
    source = _Source("a", np.random.default_rng(1), np.arange(4.0))
    before = {"source": state_items(source, "source")}
    assert_unchanged(before, {"source": source})
    source.rng.uniform()
    with pytest.raises(AssertionError, match=r"source.rng.bit_generator.state"):
        assert_unchanged(before, {"source": source})


def test_state_digest_sees_arrays_attributes_and_lazy_state():
    source = _Source("a", np.random.default_rng(1), np.arange(4.0))
    digest = state_digest(source)
    assert state_digest(_Source("a", np.random.default_rng(1), np.arange(4.0))) == digest
    source.values[2] = -1.0
    assert state_digest(source) != digest
    other = _Source("a", np.random.default_rng(1), np.arange(4.0))
    other.cache = {"k": 1}            # an attribute that appears on use
    assert state_digest(other) != digest


def test_state_digest_follows_cycles():
    source = _Source("a", np.random.default_rng(1), np.arange(4.0))
    source.children.append(source)
    items = state_items(source, "s")
    assert items["s.children[0]"] == b"ref:s"


@pytest.mark.parametrize("opaque, name", [
    (threading.Lock(), "lock"),
    ((x for x in range(3)), "generator"),
    (ThreadPoolExecutor(1), "ThreadPoolExecutor"),
])
def test_state_digest_refuses_opaque_types_by_name(opaque, name):
    source = _Source("a", np.random.default_rng(1), np.arange(4.0), children=[opaque])
    with pytest.raises(OpaqueStateError, match=rf"s.children\[0\]: .*{name}"):
        state_items(source, "s")
    if isinstance(opaque, ThreadPoolExecutor):
        opaque.shutdown()
