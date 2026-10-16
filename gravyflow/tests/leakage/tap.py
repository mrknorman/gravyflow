"""
A tap on every dataset a piece of code builds: an example script's build
functions, a training run, the validator. It records what each dataset serves,
including outputs the code never asked for, so a leakage test can compare the
noise under a signal and every label stream of data it could not otherwise see.
"""

from __future__ import annotations

import hashlib
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

import gravyflow as gf
from gravyflow.src.dataset import dataset as dataset_module
from gravyflow.tests.leakage.fingerprints import concatenate_streams, label_streams

RV = gf.ReturnVariables

#: The whitened data, and the whitened injections that make its noise recoverable.
STRAIN = (RV.WHITENED_ONSOURCE, RV.WHITENED_INJECTIONS)
#: Every per-example label a dataset can return about its injections.
INJECTION_LABELS = (RV.INJECTION_PRESENT, *gf.ScalingTypes, *gf.WaveformParameters, RV.CENTRAL_TIME)

Batch = Dict[str, np.ndarray]


class DatasetTap:
    """
    Records the batches every `GravyflowDataset` built while it is installed
    serves.

    Each dataset is built asking for ``outputs`` besides its caller's own.
    Asking for an output selects it and draws nothing more. Every batch served
    is recorded, its inputs and outputs together, then returned with only the
    outputs the caller asked for, so the code under test sees exactly the
    batches it would have seen. A batch served twice (Keras peeks before an
    epoch) is recorded once.
    """

    def __init__(self, monkeypatch, outputs: Sequence):
        built: List[Tuple[str, List[Batch]]] = []
        self._built = built
        extra = list(outputs)

        class Tapped(dataset_module.GravyflowDataset):
            def __init__(self, *, output_variables=None, **kwargs):
                asked = list(output_variables or [])
                super().__init__(output_variables=asked + [v for v in extra if v not in asked], **kwargs)
                self._asked = {v.name for v in asked}
                self._digests, self._served = set(), []
                built.append((self.group, self._served))

            def __getitem__(self, index):
                x, y = super().__getitem__(index)
                served = {name: np.asarray(value) for name, value in {**y, **x}.items()}
                digest = hashlib.sha256(b"".join(
                    name.encode() + served[name].tobytes() for name in sorted(served))).hexdigest()
                if digest not in self._digests:
                    self._digests.add(digest)
                    self._served.append(served)
                return x, {name: value for name, value in y.items() if name in self._asked}

        # Example models build `gf.GravyflowDataset`; `gf.Dataset` builds the module's.
        monkeypatch.setattr(dataset_module, "GravyflowDataset", Tapped)
        monkeypatch.setattr(gf, "GravyflowDataset", Tapped)

    def take(self) -> List[Tuple[str, List[Batch]]]:
        """
        ``(group, batches served)`` for each dataset built since the last take,
        in the order they were built. A dataset that served nothing is refused:
        its checks would be empty.
        """
        taken = list(self._built)
        self._built.clear()
        for group, served in taken:
            if not served:
                raise ValueError(f"A {group!r} dataset was built but served no batch.")
        return taken


def whitened(batches: Iterable[Batch]) -> np.ndarray:
    """The whitened onsource of every example, (examples, detectors, samples)."""
    return np.concatenate([np.asarray(b[RV.WHITENED_ONSOURCE.name], np.float32) for b in batches])


def whitened_noise(batches: Iterable[Batch]) -> np.ndarray:
    """
    The whitened noise of every example: its whitened onsource less every
    generator's whitened injection. Whitening is linear in the onsource for
    the batch's one spectrum, so this is the noise alone, whatever the signal.
    """
    return np.concatenate([
        np.asarray(b[RV.WHITENED_ONSOURCE.name], np.float64)
        - np.asarray(b[RV.WHITENED_INJECTIONS.name], np.float64).sum(axis=1)
        for b in batches])


def injection_label_streams(batches: Iterable[Batch]) -> Dict[str, np.ndarray]:
    """Every injection label's per-example stream, joined over the batches in order."""
    names = [v.name for v in INJECTION_LABELS]
    per_batch = []
    for batch in batches:
        labels = {name: batch[name] for name in names}
        per_batch.append(label_streams(labels, len(batch[RV.INJECTION_PRESENT.name])))
    return concatenate_streams(per_batch)
