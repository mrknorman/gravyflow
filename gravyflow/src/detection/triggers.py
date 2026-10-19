"""
A search's triggers, and the MLGWSC-1 trigger file they are written as.

A trigger is a candidate event: its GPS time, a ranking statistic and a timing
tolerance. The first machine-learning gravitational-wave search mock data challenge
(MLGWSC-1, Schäfer et al. 2023, arXiv:2209.11146) fixed a file for them that its
scorer, ``evaluate.py``, reads: an HDF5 file holding three one-dimensional datasets
of equal length at its root,

- ``time``: each trigger's GPS time, in seconds;
- ``stat``: its ranking statistic, larger meaning more likely astrophysical;
- ``var``: the largest distance, in seconds, from an injection's merger at which
  the trigger still counts as finding it.

The scorer reads each file as ``np.vstack([fp['time'], fp['stat'], fp['var']])``
and ignores anything else in it. ``Triggers.write`` writes exactly that file, as
the challenge's examples write it (float64 datasets, created in that order), and
``Triggers.read`` reads one, so a search scored here and by the challenge's script
is scored on the same bytes.

The file has no field for which way a statistic ranks: a search whose statistic
ranks the other way (smaller meaning more likely) declares it beside its triggers,
as an external model's card does (``external.OutputContract.direction``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union

import h5py
import numpy as np

#: The datasets of an MLGWSC-1 trigger file, in the order the challenge writes them.
FIELDS = ("time", "stat", "var")


@dataclass(frozen=True, eq=False)
class Triggers:
    """
    A search's triggers: three float64 arrays of equal length, read-only.

    Args:
        time: Each trigger's GPS time, in seconds.
        stat: Its ranking statistic.
        var: The largest distance, in seconds, from an injection's merger at which
            it counts as finding it; not negative.

    Each is taken as a one-dimensional float64 array; a NaN or infinity anywhere is
    refused by name, since the challenge's scorer ranks and windows on them.
    """

    time: np.ndarray
    stat: np.ndarray
    var: np.ndarray

    def __post_init__(self):
        arrays = {}
        for name in FIELDS:
            array = np.array(getattr(self, name), dtype=np.float64)
            if array.ndim != 1:
                raise ValueError(f"Triggers' {name} must be one-dimensional, got shape {array.shape}.")
            if not np.isfinite(array).all():
                raise ValueError(f"Triggers' {name} holds a NaN or an infinity.")
            array.setflags(write=False)
            arrays[name] = array
        lengths = {name: len(array) for name, array in arrays.items()}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"Triggers' time, stat and var must have one length, got {lengths}.")
        if (arrays["var"] < 0).any():
            raise ValueError("Triggers' var is a distance in seconds and cannot be negative.")
        for name, array in arrays.items():
            object.__setattr__(self, name, array)

    def __len__(self) -> int:
        return len(self.time)

    @classmethod
    def read(cls, path: Union[str, Path]) -> "Triggers":
        """
        The triggers in the MLGWSC-1 trigger file at ``path``. A file missing one of
        the three datasets is refused by name; other contents are ignored, as the
        challenge's scorer ignores them.
        """
        with h5py.File(path, "r") as file:
            missing = [name for name in FIELDS if name not in file]
            if missing:
                raise ValueError(f"{path} is not an MLGWSC-1 trigger file: it has no {missing} dataset.")
            return cls(*(file[name][()] for name in FIELDS))

    def write(self, path: Union[str, Path], overwrite: bool = False) -> Path:
        """
        Write the triggers to ``path`` as an MLGWSC-1 trigger file and return it. An
        existing file is refused unless ``overwrite``.
        """
        path = Path(path)
        with h5py.File(path, "w" if overwrite else "x") as file:
            for name in FIELDS:
                file.create_dataset(name, data=getattr(self, name))
        return path


__all__ = ["FIELDS", "Triggers"]
