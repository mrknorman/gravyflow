"""
Triggers and the MLGWSC-1 trigger file: what gravyflow writes is the file the
challenge's examples write, byte for byte, and its scorer reads it as written.

The challenge's scorer (``evaluate.py``, gwastro/ml-mock-data-challenge-1 at
6d5e62f) reads each trigger file as ``np.vstack([fp['time'], fp['stat'],
fp['var']])``, and its examples write one with ``create_dataset`` of ``time``,
``stat`` and ``var``, in that order; ``challenge_write`` and ``challenge_read`` are
those lines.
"""

import h5py
import numpy as np
import pytest

import gravyflow as gf
from gravyflow.src.detection.triggers import FIELDS


def challenge_write(path, time, stat, var):
    """A trigger file as the challenge's examples write it (examples/example_torch.py)."""
    with h5py.File(path, "w") as outfile:
        outfile.create_dataset("time", data=time)
        outfile.create_dataset("stat", data=stat)
        outfile.create_dataset("var", data=var)


def challenge_read(path):
    """A trigger file as the challenge's scorer reads it (evaluate.py)."""
    with h5py.File(path, "r") as fp:
        return np.vstack([fp["time"], fp["stat"], fp["var"]])


def example():
    rng = np.random.default_rng(3)
    time = 1238166018.0 + np.sort(rng.uniform(0, 2592000, 50))
    return time, rng.normal(size=50), np.full(50, 0.3)


def test_the_file_is_the_challenges_byte_for_byte(tmp_path):
    time, stat, var = example()
    challenge_write(tmp_path / "challenge.hdf", time, stat, var)
    gf.Triggers(time, stat, var).write(tmp_path / "gravyflow.hdf")
    assert (tmp_path / "gravyflow.hdf").read_bytes() == (tmp_path / "challenge.hdf").read_bytes()
    read = challenge_read(tmp_path / "gravyflow.hdf")
    assert read.dtype == np.float64 and read.shape == (3, 50)
    np.testing.assert_array_equal(read, np.vstack([time, stat, var]))


def test_a_challenge_file_reads_back_exactly(tmp_path):
    time, stat, var = example()
    challenge_write(tmp_path / "challenge.hdf", time, stat, var)
    triggers = gf.Triggers.read(tmp_path / "challenge.hdf")
    assert len(triggers) == 50 and FIELDS == ("time", "stat", "var")
    for name, expected in zip(FIELDS, (time, stat, var)):
        assert getattr(triggers, name).dtype == np.float64
        np.testing.assert_array_equal(getattr(triggers, name), expected)
    # Round trip, and an empty search is a valid file too.
    triggers.write(tmp_path / "again.hdf")
    assert (tmp_path / "again.hdf").read_bytes() == (tmp_path / "challenge.hdf").read_bytes()
    empty = gf.Triggers([], [], [])
    assert len(gf.Triggers.read(empty.write(tmp_path / "empty.hdf"))) == 0


def test_triggers_refuse_what_the_scorer_cannot_rank(tmp_path):
    with pytest.raises(ValueError, match="one length"):
        gf.Triggers([1.0, 2.0], [1.0], [0.1, 0.1])
    with pytest.raises(ValueError, match="stat holds a NaN"):
        gf.Triggers([1.0], [np.nan], [0.1])
    with pytest.raises(ValueError, match="one-dimensional"):
        gf.Triggers([[1.0]], [[1.0]], [[0.1]])
    with pytest.raises(ValueError, match="cannot be negative"):
        gf.Triggers([1.0], [1.0], [-0.1])
    triggers = gf.Triggers([1.0], [2.0], [0.1])
    with pytest.raises(ValueError, match="read-only"):
        triggers.time[0] = 5.0
    # A file is never overwritten unless asked; a file without the three datasets is refused.
    triggers.write(tmp_path / "t.hdf")
    with pytest.raises(FileExistsError):
        triggers.write(tmp_path / "t.hdf")
    gf.Triggers([3.0], [4.0], [0.2]).write(tmp_path / "t.hdf", overwrite=True)
    assert gf.Triggers.read(tmp_path / "t.hdf").time.tolist() == [3.0]
    with h5py.File(tmp_path / "partial.hdf", "w") as file:
        file.create_dataset("time", data=[1.0])
    with pytest.raises(ValueError, match=r"no \['stat', 'var'\] dataset"):
        gf.Triggers.read(tmp_path / "partial.hdf")
