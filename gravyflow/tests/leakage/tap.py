"""
A tap on every dataset a piece of code builds: an example script's build
functions, a training run, the validator. It records what each dataset serves,
including outputs the code never asked for, so a leakage test can compare the
noise under a signal and every label stream of data it could not otherwise see.
"""

from __future__ import annotations

import hashlib
from typing import Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple

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
#: A batch's (epoch, k).
Address = Tuple[int, int]


class Tapped(NamedTuple):
    """What one tapped dataset served, in the order it served it."""

    #: The dataset's split.
    group: str
    #: Each distinct batch, its inputs and outputs together.
    batches: List[Batch]
    #: The (epoch, k) address of each of ``batches``; None for a dataset that
    #: is not index-addressed, whose batches depend on the order asked.
    addresses: List[Optional[Address]]
    #: The raw windows it conditioned for its caller (`prepare`), in the order
    #: given: each call's padded onsource and offsource, as ONSOURCE and
    #: OFFSOURCE. These are windows it scores but never serves: time-slid
    #: noise, a real event.
    prepared: List[Batch]


def _digest(batch: Batch) -> str:
    return hashlib.sha256(b"".join(name.encode() + batch[name].tobytes() for name in sorted(batch))).hexdigest()


def _require_same(first: Batch, again: Batch, group: str, address: Address) -> None:
    """A batch served again at ``address`` is the bytes first served there."""
    differ = sorted(name for name in first
                    if first[name].shape != again[name].shape or first[name].tobytes() != again[name].tobytes())
    if differ:
        raise AssertionError(
            f"The {group!r} dataset served other data at (epoch {address[0]}, batch {address[1]}) "
            f"than it first served there, in {', '.join(differ)}."
        )


class DatasetTap:
    """
    Records the batches every `GravyflowDataset` built while it is installed
    serves, a subclass's (`ComposedDataset`) included, and the raw windows each
    conditions for its caller (`prepare`).

    Each dataset is built asking for ``outputs`` besides its caller's own.
    Asking for an output selects it and draws nothing more. Every batch served
    is recorded, its inputs and outputs together, then returned with only the
    outputs the caller asked for, so the code under test sees exactly the
    batches it would have seen.

    A batch served twice (Keras makes a pass before its first epoch) is
    recorded once. An index-addressed dataset's batch is its address, (epoch,
    k), and a batch served again at an address must be the bytes first served
    there, or the tap fails the test: the session's kernels are bit-reproducible
    (the root conftest.py), so any difference is the dataset serving other
    data at one address, which Keras would train on unseen. Any other
    dataset's batch is recognised by its bytes.

    ``serves`` logs every batch asked of any tapped dataset, repeats
    included, as (split, epoch, k), in the order asked; `take` leaves it.
    """

    def __init__(self, monkeypatch, outputs: Sequence):
        built: List[Tapped] = []
        self._built = built
        serves: List[Tuple[str, int, int]] = []
        self.serves = serves
        extra = list(outputs)

        class Dataset(dataset_module.GravyflowDataset):
            def __init__(self, *, output_variables=None, **kwargs):
                asked = list(output_variables or [])
                super().__init__(output_variables=asked + [v for v in extra if v not in asked], **kwargs)
                self._asked = {v.name for v in asked}
                self._seen: Dict = {}
                self._record = Tapped(self.group, [], [], [])
                built.append(self._record)

            def __getitem__(self, index):
                x, y = super().__getitem__(index)
                streams = self.streams(index)
                address = (streams.epoch, streams.k)
                serves.append((self.group, *address))
                served = {name: np.asarray(value) for name, value in {**y, **x}.items()}
                key = address if self.index_addressed else _digest(served)
                if key not in self._seen:
                    self._seen[key] = len(self._record.batches)
                    self._record.batches.append(served)
                    self._record.addresses.append(address if self.index_addressed else None)
                else:
                    _require_same(self._record.batches[self._seen[key]], served, self.group, address)
                return x, {name: value for name, value in y.items() if name in self._asked}

            def prepare(self, onsource, offsource):
                self._record.prepared.append({RV.ONSOURCE.name: np.asarray(onsource),
                                              RV.OFFSOURCE.name: np.asarray(offsource)})
                return super().prepare(onsource, offsource)

        # Every dataset class is tapped, each subclass as its own: example models build
        # `gf.GravyflowDataset`, `gf.Dataset` builds the module's, and composed data
        # `gf.ComposedDataset`, which a tap of the base class alone never saw. gravyflow
        # imports its names lazily, so its dataset names are read from the dataset
        # module, and patched under the same name where gravyflow exports one.
        base, tapped = dataset_module.GravyflowDataset, {}
        for name, cls in list(vars(dataset_module).items()):
            if isinstance(cls, type) and issubclass(cls, base):
                if cls not in tapped:
                    tapped[cls] = Dataset if cls is base else type(cls.__name__, (Dataset, cls), {})
                monkeypatch.setattr(dataset_module, name, tapped[cls])
                if name in gf.__all__:
                    monkeypatch.setattr(gf, name, tapped[cls])

    def take(self) -> List[Tapped]:
        """
        What each dataset built since the last take served and prepared, in
        the order they were built. A dataset that served no batch and prepared
        no window is refused: its checks would be empty.
        """
        taken = list(self._built)
        self._built.clear()
        for record in taken:
            if not record.batches and not record.prepared:
                raise ValueError(f"A {record.group!r} dataset was built but served no batch and prepared no window.")
        return taken


class EpochWindows:
    """
    Cuts a tap's serve log at Keras' epochs. An epoch's window runs from its
    ``on_epoch_begin`` to its ``on_epoch_end``: its training steps and its
    validation. Keras' epochs reach it through `callback`, or through a
    callback class a fit already uses (`hook`), as that class sees them.
    """

    def __init__(self, tap: DatasetTap):
        self._log = tap.serves
        self._start = len(self._log)
        self._first = None
        self._begin = None
        self._windows: Dict[int, List[Tuple[str, int, int]]] = {}

    def begin(self, epoch: int):
        self._begin = len(self._log)
        if self._first is None:
            self._first = self._begin

    def end(self, epoch: int):
        if epoch in self._windows:
            raise ValueError(f"Keras epoch {epoch} ran twice before a take.")
        self._windows[epoch] = self._log[self._begin:]

    def take(self) -> Tuple[List[Tuple[str, int, int]], Dict[int, List[Tuple[str, int, int]]]]:
        """
        Since the last take: the serves before the first epoch began, and
        each epoch's serves, as (split, epoch, k). Refuses when no epoch ran.
        """
        if self._first is None:
            raise ValueError("No Keras epoch ran since the last take.")
        before, windows = self._log[self._start:self._first], self._windows
        self._start, self._first, self._windows = len(self._log), None, {}
        return before, windows

    def callback(self):
        """A Keras callback that reports its fit's epochs here."""
        import keras

        windows = self

        class Report(keras.callbacks.Callback):
            def on_epoch_begin(self, epoch, logs=None):
                windows.begin(epoch)

            def on_epoch_end(self, epoch, logs=None):
                windows.end(epoch)

        return Report()

    def hook(self, monkeypatch, callback_class):
        """Report the epochs of every fit that uses ``callback_class``, as it sees them."""
        begin, end = callback_class.on_epoch_begin, callback_class.on_epoch_end

        def on_epoch_begin(callback, epoch, logs=None):
            self.begin(epoch)
            begin(callback, epoch, logs)

        def on_epoch_end(callback, epoch, logs=None):
            self.end(epoch)
            end(callback, epoch, logs)

        monkeypatch.setattr(callback_class, "on_epoch_begin", on_epoch_begin)
        monkeypatch.setattr(callback_class, "on_epoch_end", on_epoch_end)


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
