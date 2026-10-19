from typing import List, Tuple, Union, Dict, Any, Iterator, Optional, Set
from enum import Enum, auto
import copy
import inspect
import os
from pathlib import Path
from dataclasses import dataclass
from warnings import warn
import logging

logger = logging.getLogger(__name__)

import numpy as np
import keras
from keras import ops
import jax
import gravyflow as gf
from gravyflow.src.dataset.conditioning.whiten import whiten
from gravyflow.src.dataset.tools.psd import SpectralConfig, estimate_psd, MIN_PSD_SEGMENTS
from gravyflow.src.utils.tensor import (
    crop_samples, replace_nan_and_inf_with_zero, window_samples
)
from gravyflow.src.utils.shapes import ANY, OFFSOURCE_SAMPLES, BatchContract, BatchContractError
from gravyflow.src.dataset.config import Defaults, WindowSpec, ignore_scale_factor
from gravyflow.src.dataset.noise.noise import NoiseBatcher, addressed_by_index, obtainer_batcher, reads_real_data
from gravyflow.src.dataset.acquisition.augmentations import Augmented, augment
from gravyflow.src.dataset.acquisition.base import BaseDataObtainer
from gravyflow.src.dataset.features.injection import InjectionGenerator, check_scaling_matches_noise
from gravyflow.src.rng import TREE_VERSION, BatchStreams, SeedTree, check_coordinate, check_name, check_split
from gravyflow.src.rng import platform
from gravyflow.src.dataset.acquisition.versions import DATA_VERSIONS
from gravyflow.src.dataset.materialise import (
    DEFAULT_SHARD_SIZE, FORMAT, ContractMismatch, MaterialisedSet, NotIndexAddressed, write_set,
)
from gravyflow.src.utils.fingerprint import (
    UnfingerprintableError, canonical, config_fingerprint, library_versions,
)
from gravyflow.src.dataset.conditioning.pearson import rolling_pearson
from gravyflow.src.dataset.conditioning.conditioning import spectrogram
from gravyflow.src.utils.numerics import ensure_even, ensure_list
from gravyflow.src.utils.gps import as_gps_times

def validate_noise_settings(
        noise_obtainer: gf.NoiseObtainer, 
        variables_to_return: List[
            Union[
                gf.WaveformParameters, 
                gf.ReturnVariables
            ]
        ]
    ) -> None:
    """
    Validates noise settings and emits warnings or errors as appropriate.
    """
    
    # Validate noise type and ifo_data_obtainer
    # Skip validation for TransientObtainer which doesn't have noise_type
    if not hasattr(noise_obtainer, 'noise_type'):
        return  # TransientObtainer or similar - skip validation
    
    if noise_obtainer.noise_type in [gf.NoiseType.WHITE, gf.NoiseType.COLORED]:
        
        if noise_obtainer.ifo_data_obtainer is not None:
            warn(
                ("Noise is not real or PSEUDO-REAL, yet data obtainer is"
                 " defined."), 
                UserWarning
            )
            
        if gf.ReturnVariables.START_GPS_TIME in variables_to_return:
            warn(
                "Simulated noise has no GPS time: START_GPS_TIME is NaN.",
                UserWarning
            )
    
    # Validate whitening for white noise
    if noise_obtainer.noise_type is gf.NoiseType.WHITE:
    
        if gf.ReturnVariables.WHITENED_INJECTIONS in variables_to_return or \
           gf.ReturnVariables.WHITENED_ONSOURCE in variables_to_return:
            
            warn(
                "Whitening requested for WHITE NOISE.", 
                UserWarning
            )
            
def noise_ifos(ifos, noise_obtainer) -> List:
    """
    The detectors a dataset's noise has, and so its generators' network: the
    dataset's own ``ifos``, else its noise obtainer's, else L1.
    """
    if ifos is not None:
        return ensure_list(ifos)
    obtainer_ifos = getattr(noise_obtainer, "ifos", None)
    return ensure_list(obtainer_ifos) if obtainer_ifos is not None else [gf.IFO.L1]


def split_spec_of(obtainers) -> Optional["gf.SplitSpec"]:
    """
    How the real data read through ``obtainers`` is split: the spec of the IFO
    data obtainers they read through (a NoiseObtainer's, or a transient obtainer
    itself); None (no obtainer named) is the default obtainer, real noise split
    by the default spec. An obtainer that reads no real data (`reads_real_data`:
    synthetic noise) is split-neutral: data read through such obtainers alone
    names no spec (None, `split_neutral`), and combines with real data of any
    spec.

    This is the one computation of a split, a dataset's
    (`IndexedBatches.split_spec`) and the validator's alike, and it reads only
    the obtainers' specifications, so a split is known before any data is
    fetched. Everything else that needs a split, the training runs included,
    reads it from data.
    """
    specs = {getattr(getattr(obtainer, "ifo_data_obtainer", obtainer), "split_spec", None) or gf.DEFAULT_SPLIT
             for obtainer in obtainers if reads_real_data(obtainer)}
    if len(specs) > 1:
        raise ValueError(f"A dataset's obtainers split time differently: {sorted(s.fingerprint() for s in specs)}.")
    return specs.pop() if specs else None


def split_neutral(data) -> bool:
    """
    Whether ``data`` is split-neutral: gravyflow data none of whose obtainers
    reads real data (`reads_real_data`), which is why its ``split_spec`` is
    None. Only gravyflow's own data is: a dataset (`IndexedBatches`), or batches
    that name the dataset they serve as their ``dataset`` (the run recipe's do).
    Anything else whose ``split_spec`` is None names no split.
    """
    dataset = getattr(data, "dataset", data)
    return isinstance(dataset, IndexedBatches) and not any(map(reads_real_data, dataset._obtainers()))


def augment_batch(onsource, offsource, streams: BatchStreams, obtainer, split: str,
                  sampling_mode: Optional["gf.SamplingMode"] = None) -> Augmented:
    """
    A batch's raw strain with the augmentations of the IFO data obtainer it was
    read through applied (``augment``): training data only, before injection
    and whitening. Simulated noise (WHITE, COLORED) reads through none, and a
    reader of the caller's own carries none, so neither is augmented.
    """
    if isinstance(obtainer, gf.NoiseObtainer):
        obtainer = obtainer.ifo_data_obtainer if reads_real_data(obtainer) else None
    if not isinstance(obtainer, BaseDataObtainer):
        return augment((onsource, offsource), streams, (), split)
    return augment((onsource, offsource), streams, obtainer.augmentations, split, obtainer.data_labels,
                   sampling_mode)


def augmented_batcher(obtainer, tree: SeedTree, group: str, **window) -> NoiseBatcher:
    """
    The batcher of ``obtainer`` (``obtainer_batcher``), each batch's raw strain
    augmented for its split ``group`` (``augment_batch``) and flagged in
    AUGMENTATIONS. A dataset reads its noise through one, and a composed dataset
    one per pool, so every example is augmented once, by its own source and from
    its own streams, after its noise is read and before injection.
    """
    read = obtainer_batcher(obtainer, tree, group, **window)
    sampling_mode = window.get("sampling_mode")

    def batch(streams: BatchStreams) -> Dict:
        data = dict(read(streams))
        RV = gf.ReturnVariables
        data[RV.ONSOURCE], data[RV.OFFSOURCE], data[RV.AUGMENTATIONS] = augment_batch(
            data[RV.ONSOURCE], data[RV.OFFSOURCE], streams, obtainer, group, sampling_mode)
        return data

    batch.prepare = getattr(read, "prepare", None)
    batch.plan = getattr(read, "plan", None)
    batch.batches = getattr(read, "batches", None)
    return batch


class Unprepared(RuntimeError):
    """
    ``prepare`` met contradictions between a source's timeline and what it delivers:
    ``contradictions`` lists each ``TimelineContradiction``, and ``exclusions`` the
    entries that plan around them all.
    """

    def __init__(self, contradictions):
        self.contradictions = tuple(contradictions)
        self.exclusions = tuple(dict.fromkeys(error.exclusion for error in self.contradictions))
        listed = "\n".join(f"- {error}" for error in self.contradictions)
        super().__init__(
            f"{len(self.contradictions)} contradictions between the timeline and the source:\n{listed}\n"
            f"To plan around them all, add exclusions={self.exclusions!r} to the obtainer."
        )


def freeze_availability(obtainers, ifos) -> None:
    """
    Freeze the availability of every IFO data obtainer ``obtainers`` read through
    (``BaseDataObtainer.freeze_availability``) for ``ifos``: computed once, as the
    dataset is built, and part of its obtainers' specification from then on.
    """
    for obtainer in obtainers:
        if not reads_real_data(obtainer):
            continue
        reader = obtainer.ifo_data_obtainer if isinstance(obtainer, gf.NoiseObtainer) else obtainer
        if isinstance(reader, BaseDataObtainer):
            reader.freeze_availability(ifos)


def get_max_arrival_time_difference(ifos: List[gf.IFO]) -> float:
    """
    The default lag range of ``ROLLING_PEARSON_ONSOURCE`` (s): the largest light
    travel time between ``ifos``, the detectors the dataset uses (``noise_ifos``;
    every generator's network is bound to them), never the generators', so
    noise-only and injection batches have the same shape.

    A Python float, because `rolling_pearson` takes it as a static argument.
    """
    return float(gf.Network(ensure_list(ifos)).max_arrival_time_difference_seconds)

WHITENED_VARIABLES = (
    gf.ReturnVariables.WHITENED_ONSOURCE,
    gf.ReturnVariables.WHITENED_INJECTIONS,
    gf.ReturnVariables.ROLLING_PEARSON_ONSOURCE,
    gf.ReturnVariables.SPECTROGRAM_ONSOURCE,
)


def _ifo_names(ifos) -> List[str]:
    return [ifo.name if hasattr(ifo, "name") else str(ifo) for ifo in ensure_list(ifos)]


def check_network_matches_ifos(network: "gf.Network", ifos) -> None:
    """
    Raise ValueError unless `network` projects onto exactly `ifos`, in order. A
    network built from geometry alone (no IFO list) is checked by its detector
    count.
    """
    expected = _ifo_names(ifos)
    network_ifos = getattr(network, "ifos", None)
    if network_ifos is not None:
        if _ifo_names(network_ifos) != expected:
            raise ValueError(
                f"A waveform generator's network projects onto {_ifo_names(network_ifos)}, "
                f"but the noise has detectors {expected}; they must be the same, in order."
            )
        return
    num_detectors = getattr(network, "num_detectors", None)
    if num_detectors is not None and int(np.asarray(num_detectors).ravel()[0]) != len(expected):
        raise ValueError(
            f"A waveform generator's network has {int(np.asarray(num_detectors).ravel()[0])} detectors, "
            f"but the noise has {len(expected)} ({expected})."
        )


def lowest_generator_frequency(generator) -> Optional[float]:
    """The lowest frequency a generator's parameters allow, where it declares one."""
    lowest = None
    for name in ("min_frequency_hertz", "frequency_hertz", "start_frequency_hertz", "end_frequency_hertz"):
        value = getattr(generator, name, None)
        if value is None:
            continue
        if isinstance(value, gf.Distribution):
            if value.type_ == gf.DistributionType.CONSTANT:
                value = value.value
            elif value.min_ is not None:
                value = value.min_
            elif value.possible_values is not None:
                value = min(value.possible_values)
            else:
                continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        lowest = value if lowest is None else min(lowest, value)
    return lowest


class EpochMode(Enum):
    """How a dataset's batches change from epoch to epoch."""
    FRESH = auto()   # New draws every epoch: batch k of epoch e is addressed by (e, k).
    REPEAT = auto()  # The same batches every epoch: epoch 0's.


class _RefusedCopy:
    """
    A dataset's copy that cannot serve: every use raises why. A refusal is never
    raised as the dataset is pickled (``IndexedBatches.__reduce__``) or
    unpickled (``_rebuilt``), but through the first task that uses the copy.
    Keras' process enqueuer pickles every dataset it still holds, not only the
    one it serves, into each pool it starts, so a refusal raised at pickling
    would fail every later call in the session; and a pool, or that enqueuer,
    answers a worker that dies unpickling its dataset by starting another,
    forever.
    """

    __slots__ = ("_refusal",)

    def __init__(self, refusal: str):
        object.__setattr__(self, "_refusal", refusal)

    def __reduce__(self):
        return (_RefusedCopy, (self._refusal,))

    def _refuse(self, *args):
        raise RuntimeError(self._refusal)

    __getitem__ = __len__ = __iter__ = __setattr__ = _refuse

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        self._refuse()


def _rebuilt(cls, arguments: dict, state: dict, fingerprint: str):
    """
    A dataset rebuilt in this process from its original's resolved ``arguments``
    with its serving ``state`` (``IndexedBatches.__reduce__``) when its
    fingerprint is the original's ``fingerprint``; otherwise a copy that refuses
    to serve, saying why (``_RefusedCopy``).
    """
    rebuilt = f"A {cls.__name__} rebuilt in process {os.getpid()}"
    try:
        dataset = cls(**arguments)
        rebuilt_fingerprint = dataset.fingerprint
    except Exception as error:
        return _RefusedCopy(f"{rebuilt} could not be built: {type(error).__name__}: {error}")
    if rebuilt_fingerprint != fingerprint:
        return _RefusedCopy(
            f"{rebuilt} has fingerprint {rebuilt_fingerprint}, not its original's {fingerprint}: this "
            "process's libraries, gravyflow or data versions are not the original's (fingerprint_record), "
            "so it would serve other batches."
        )
    dataset.__dict__.update(state)
    return dataset


class _KerasEpoch(keras.callbacks.Callback):
    """Sets a dataset's epoch to Keras' epoch, before Keras reads its batches."""

    def __init__(self, dataset: "IndexedBatches"):
        super().__init__()
        self.dataset = dataset

    def on_train_begin(self, logs=None):
        # With fit(steps_per_epoch=) other than the dataset's length, Keras
        # keeps one iterator across epochs and fetches ahead of
        # on_epoch_begin, so an epoch would train batches addressed to the
        # epoch before.
        steps = self.params.get("steps")
        if steps is not None and steps != len(self.dataset):
            raise ValueError(
                f"fit(steps_per_epoch={steps}) differs from the dataset's {len(self.dataset)} batches "
                "per epoch, which would serve one epoch's batches in another. Give the dataset "
                "steps_per_epoch instead, and leave fit's unset."
            )
        # Keras' enqueuer, under these settings (its PyDatasetAdapter's test), asks for
        # batches from concurrent workers, so the order fit asks in cannot be checked.
        workers, processes = self.dataset.workers, self.dataset.use_multiprocessing
        if workers > 1 or (workers > 0 and processes):
            raise ValueError(
                f"fit under epoch_callback checks that it asks for each epoch's batches in order "
                f"(its shuffle is unseeded); {workers} workers ask for them concurrently, so the "
                "check cannot hold. Set the dataset's workers to 0 for fit; workers serve evaluate, "
                "predict and a plain loop."
            )
        self.dataset._keras_fitting = True

    def on_epoch_begin(self, epoch, logs=None):
        self.dataset.epoch = epoch
        self.dataset._keras_next = 0

    def on_train_end(self, logs=None):
        # Only fit shuffles: evaluate, predict and plain indexing after it may ask
        # for batches in any order.
        self.dataset._keras_fitting = False


class IndexedBatches:
    """
    Batches addressed by index: ``ds[k]`` is a pure function of the dataset's
    specification, its seed tree, its split, its effective epoch and k.

    The training split (EpochMode.FRESH, the default for ``group="train"``)
    draws fresh batches every epoch. Other splits (EpochMode.REPEAT) always
    serve epoch 0, so an evaluation set is the same in every epoch and every
    run.

    The epoch moves on with ``on_epoch_end``: a loop over ``ds[k]`` or
    ``iter(ds)`` calls it after each pass, and ``next(ds)``, a cursor over
    (epoch, k), calls it itself. Under ``model.fit`` the epoch is Keras' own,
    from ``ds.epoch_callback()``.

    Real noise drawn at random (REAL, ``SamplingMode.RANDOM``) and PSEUDO_REAL
    noise are addressed by index too: batch k of epoch e is batch e * steps + k
    of its split's passes, from availability frozen when the dataset was built
    (``freeze_availability``). So are transients: training batch k of epoch e
    is batch e * steps + k of the split's passes over its records. GRID noise
    and a transient evaluation split are finite sets: their length is their
    whole batches, floor(N / B), whatever ``steps_per_epoch`` says, the
    remainder reported, and every epoch serves the same set. Only a reader of
    the caller's own is read in order (``seed_metadata["index_addressed"]`` is
    False): its batches depend on the order they are asked for.

    Under ``model.fit`` the batches are asked for in order: Keras' shuffle
    (``fit``'s default) reorders them with Python's unseeded global random,
    so a dataset given ``epoch_callback()`` refuses a batch asked for out of
    order. Pass ``shuffle=False`` to ``fit``.
    """

    def _init_batches(self, seed, group: str, steps_per_epoch: int, epoch_mode: Optional[EpochMode],
                      name: Optional[str] = None):
        self.name = None if name is None else check_name(name, "dataset name")
        # The tree as given, resolved, before the name's scope: what a copy is rebuilt from.
        self._seed_tree = SeedTree.of(seed)
        self.tree = self._seed_tree
        if self.name is not None:
            self.tree = self.tree.sub(f"dataset/{self.name}")
        self.seed = self.tree.root
        self._group = check_split(group)
        self.steps_per_epoch = steps_per_epoch
        if epoch_mode is None:
            epoch_mode = EpochMode.FRESH if group == "train" else EpochMode.REPEAT
        self.epoch_mode = epoch_mode
        self._epoch = 0
        self._cursor = 0
        self._epoch_from_keras = False
        self._warned_keras_epoch = False
        # Whether fit is running under epoch_callback, and the batch it asks for next
        # (``_check_keras_order``).
        self._keras_fitting = False
        self._keras_next = 0
        # The materialised set batches are served from, if any (``load_materialised``).
        self._materialised: Optional[MaterialisedSet] = None
        # The process that built (or rebuilt) the dataset, the only one it serves in.
        self._process = os.getpid()

    # ---- workers

    #: What a copy in another process carries beside its specification (``__reduce__``):
    #: the epoch and cursor, Keras' fitting state and worker settings, a loaded set.
    _SERVING_STATE = ("_epoch", "_cursor", "_epoch_from_keras", "_warned_keras_epoch", "_keras_fitting",
                      "_keras_next", "_workers", "_use_multiprocessing", "_max_queue_size", "_materialised")

    #: Constructor parameters a copy is rebuilt without: deprecated, and ignored.
    _IGNORED_PARAMETERS = ("scale_factor",)

    def _built(self) -> None:
        """
        Record the specification the dataset was built from, as its constructor's
        last step: everything it serves was derived from it then, so a copy is
        refused once it has changed (``__reduce__``). A subclass sets its own
        settings before the base constructor runs.
        """
        try:
            self._built_specification = canonical(self.specification())
        except UnfingerprintableError:
            self._built_specification = None  # a copy refuses it by that name (``_copy_refusal``)

    def _rebuild_parameters(self) -> Tuple[List[inspect.Parameter], List[str]]:
        """
        The constructor's parameters, and those a copy cannot be rebuilt with:
        anything the specification, the seed and the split do not hold, such as a
        subclass's own parameter, which, rebuilt at its default, would serve other
        batches under the same fingerprint. A subclass declares such a parameter
        in its specification (``_SPECIFICATION``), which fingerprints it too.
        """
        held = {*self.specification(), "seed", "group", *self._IGNORED_PARAMETERS}
        parameters = list(inspect.signature(type(self).__init__).parameters.values())[1:]
        unexpressed = [
            str(parameter) for parameter in parameters
            if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD) or parameter.name not in held
        ]
        return parameters, unexpressed

    def _copy_refusal(self) -> Optional[str]:
        """
        Why a copy of this dataset in another process would not serve its
        batches, or None: a dataset read in order; a constructor the
        specification cannot rebuild (``_rebuild_parameters``); a specification
        no fingerprint can read, so the copy could not be checked; or one changed
        after the dataset was built.
        """
        if not self.index_addressed:
            return self._read_in_order("which a copy in another process would ask for again from the first")
        _, unexpressed = self._rebuild_parameters()
        if unexpressed:
            return (
                f"A {type(self).__name__} cannot be copied into another process: its constructor takes "
                f"{', '.join(unexpressed)}, which its specification does not hold, so a copy rebuilt from "
                "the specification would not be this dataset. Declare each in the class's _SPECIFICATION "
                "(it then decides the fingerprint too), or serve the dataset in this process (workers=0)."
            )
        try:
            specification = canonical(self.specification())
        except UnfingerprintableError as error:
            return (
                f"A {type(self).__name__} is copied into another process only with its fingerprint, which "
                f"checks the copy serves its batches, and its settings have none: {error} Or serve the "
                "dataset in this process (workers=0)."
            )
        built = self._built_specification or {}
        changed = [name for name in sorted({*specification, *built}) if specification.get(name) != built.get(name)]
        if changed:
            return (
                f"This {type(self).__name__} changed after it was built: {', '.join(changed)}. What it serves "
                "was derived from the specification it was built with, so a copy rebuilt from the one it has "
                "now would serve other batches. Build a new dataset with the settings it should have; a "
                "subclass sets its own settings before calling super().__init__(), which records them."
            )
        return None

    def __reduce__(self):
        """
        A copy for another process (a spawned worker's) is rebuilt there from the
        resolved specification (its own copies of what it was given, the defaults
        filled in), its seed tree before its name's scope and its split, so the
        worker's defaults change nothing; with this dataset's serving state, it
        serves only if its fingerprint is this dataset's (``_rebuilt``). A copy
        that would not serve this dataset's batches (``_copy_refusal``) is a
        ``_RefusedCopy``: pickling itself never raises.
        """
        refusal = self._copy_refusal()
        if refusal is not None:
            return (_RefusedCopy, (refusal,))
        resolved = {**self.specification(), "seed": self._seed_tree, "group": self.group}
        parameters, _ = self._rebuild_parameters()
        arguments = {parameter.name: resolved[parameter.name] for parameter in parameters if parameter.name in resolved}
        state = {name: self.__dict__[name] for name in self._SERVING_STATE if name in self.__dict__}
        return (_rebuilt, (type(self), arguments, state, self.fingerprint))

    def _read_in_order(self, consequence: str) -> str:
        """Why a dataset read in order cannot be served by workers, which ``consequence``."""
        return ("This dataset is read in order (index_addressed is False): its batches depend on the order "
                f"they are asked for, {consequence}. Keep workers=0.")

    def _require_own_process(self) -> None:
        """Refuse to serve in a process fork copied the dataset into: every batch served, written or read checks."""
        if os.getpid() != self._process:
            raise RuntimeError(
                "This dataset was copied into another process by fork, which shares JAX's state with "
                "its parent and can deadlock it. Use threads, or the spawn start method "
                "(multiprocessing.set_start_method('spawn')), whose workers rebuild the dataset."
            )

    @property
    def workers(self) -> int:
        """Keras' worker count (``keras.utils.PyDataset``): 0, the default, serves in the caller."""
        return self._workers

    @workers.setter
    def workers(self, value: int):
        if value and not self.index_addressed:
            raise ValueError(self._read_in_order(f"which {value} workers would race"))
        self._workers = value

    @property
    def group(self) -> str:
        """
        The split this dataset serves, fixed at construction: its batchers are
        bound to it, and augmentation asks it whether a batch is training data,
        so a split set later would leave the two disagreeing.
        """
        return self._group

    @property
    def epoch(self) -> int:
        """The training epoch: it selects the batches and any Curriculum's distribution."""
        return self._epoch

    @epoch.setter
    def epoch(self, epoch: int):
        """Set the epoch, e.g. to resume a run (Keras' ``initial_epoch``)."""
        self._epoch = check_coordinate(epoch, "epoch")

    @property
    def _batch_epoch(self) -> int:
        return self._epoch if self.epoch_mode is EpochMode.FRESH else 0

    @property
    def _schedule_epoch(self) -> Optional[int]:
        """The epoch a Curriculum is read at: training's, else its end (None)."""
        return self._epoch if self.epoch_mode is EpochMode.FRESH else None

    def streams(self, index: int) -> BatchStreams:
        """The streams of batch ``index`` at the current epoch."""
        index = check_coordinate(index, "batch index")
        if index >= len(self):
            raise IndexError(f"Batch {index} is outside the {len(self)} batches of an epoch.")
        return self.tree.batch(self.group, self._batch_epoch, index)

    def __len__(self):
        # A finite set (GRID noise, an evaluation set of transients) is its whole
        # batches, served once (``batch.batches``); every other split is steps_per_epoch long.
        batches = getattr(getattr(self, "noise", None), "batches", None)
        return self.steps_per_epoch if batches is None else batches

    def epoch_callback(self) -> keras.callbacks.Callback:
        """
        The callback that gives this dataset Keras' epoch:
        ``model.fit(ds, ..., callbacks=[ds.epoch_callback()], shuffle=False)``.
        Under it, a batch ``fit`` asks for out of order is refused: its default
        shuffle draws from Python's unseeded global random. Outside ``fit``
        (``evaluate``, ``predict``, ``ds[k]``) batches are served in any order.

        Keras also calls ``on_epoch_end`` when ``fit`` starts, before its
        first epoch, so counting those calls would serve Keras epoch e the
        batches (and Curriculum) of epoch e + 1. Only a callback is told
        Keras' epoch: with it, Keras epoch e, ``initial_epoch`` included,
        serves epoch e. So a second ``fit`` without ``initial_epoch`` starts
        again at epoch 0 and serves the first's batches; continue a run with
        ``initial_epoch``. ``fit(steps_per_epoch=)`` must be unset or the
        dataset's length (it is refused otherwise).
        """
        self._epoch_from_keras = True
        return _KerasEpoch(self)

    def _check_keras_order(self, index: int) -> None:
        """
        Refuse a batch ``fit`` asks for out of order: its shuffle (its default)
        reorders a PyDataset's batches with Python's unseeded global random, so the
        run's order would be repeatable by no seed. Only while ``fit`` runs under
        epoch_callback: evaluate, predict and a plain loop may ask in any order.
        """
        if not self._keras_fitting:
            return
        if index != self._keras_next:
            raise ValueError(
                f"Keras asked for batch {index} where batch {self._keras_next} comes next: fit is "
                "shuffling the batches with Python's unseeded global random. Pass shuffle=False to "
                "fit; every batch is already a fresh, seeded draw."
            )
        self._keras_next = index + 1

    def on_epoch_begin(self):
        """Keras calls this before each epoch's batches; a plain loop never does."""
        if (
            self.epoch_mode is EpochMode.FRESH
            and not self._epoch_from_keras
            and not self._warned_keras_epoch
        ):
            self._warned_keras_epoch = True
            warn(
                f"The '{self.group}' split draws fresh batches each epoch, but Keras "
                "is driving it without its epoch: Keras calls on_epoch_end once "
                "before the first epoch, so each Keras epoch would serve the next "
                "epoch's batches. Pass callbacks=[dataset.epoch_callback()] and shuffle=False to fit.",
                UserWarning,
                stacklevel=2,
            )

    def on_epoch_end(self):
        if self.epoch_mode is EpochMode.FRESH:
            self._epoch += 1

    def __next__(self):
        if self._cursor >= len(self):
            self._cursor = 0
            self.on_epoch_end()
        batch = self[self._cursor]
        self._cursor += 1
        return batch

    @property
    def seed_metadata(self) -> dict:
        """
        Where the data's randomness came from, to record beside a model, with
        the platform that turned it into bits (`platform.platform_record`).
        """
        return {
            **self.tree.metadata(),
            "split": self.group,
            "split_fingerprint": None if self.split_spec is None else self.split_spec.fingerprint(),
            "split_mode": None if self.split_spec is None else self.split_spec.mode,
            "epoch_mode": self.epoch_mode.name,
            "index_addressed": self.index_addressed,
            "platform": platform.platform_record(),
            "materialised": None if self._materialised is None else {
                "set_id": self._materialised.set_id, "epoch": self._materialised.epoch,
                "k_range": [self._materialised.k_range.start, self._materialised.k_range.stop]},
        }

    @property
    def split_spec(self) -> Optional["gf.SplitSpec"]:
        """
        How the real data this dataset reads is split: `split_spec_of` its
        obtainers; None for synthetic noise alone, which is split-neutral
        (`split_neutral`).
        """
        return split_spec_of(self._obtainers())

    @property
    def index_addressed(self) -> bool:
        """
        Whether every batch is a pure function of its address (see the class
        docstring): data that reads no real data (`reads_real_data`), real
        noise drawn at random, and transients.
        """
        sampling_mode = getattr(self, "sampling_mode", None)
        return all(addressed_by_index(obtainer, sampling_mode) for obtainer in self._obtainers())

    def __getitem__(self, index) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Batch ``index`` of the current epoch: a pure function of its address
        when the noise is (see the class docstring), read from the loaded
        materialised set when the set holds it (``load_materialised``), and
        computed otherwise.
        """
        self._require_own_process()
        self.streams(index)  # refuses an index outside the epoch
        self._check_keras_order(index)
        if self._materialised is not None and self._materialised.serves(self._batch_epoch, index):
            return self._read_materialised(index)
        return self._compute_batch(index)

    def _compute_batch(self, index: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        raise NotImplementedError

    # ---- identity and materialised sets

    def fingerprint_record(self) -> dict:
        """
        Everything that decides this dataset's batches but their address (split,
        effective epoch, k): its canonical specification, the seed tree's root
        and scope, the split spec's fingerprint, the seed tree's and the data's
        versions, and the version of every library whose code produces batch
        bytes, with Python's (`fingerprint.library_versions`). What the platform
        decides (``XLA_FLAGS``, the backend) is recorded beside it, never in it
        (`platform.platform_record`).
        """
        tree = self.tree.metadata()
        return {
            "specification": canonical(self.specification()),
            "root": tree["root"],
            "prefix": tree["prefix"],
            "split_fingerprint": None if self.split_spec is None else self.split_spec.fingerprint(),
            "tree_version": TREE_VERSION,
            "data_versions": dict(DATA_VERSIONS),
            "versions": library_versions(),
        }

    @property
    def fingerprint(self) -> str:
        """
        The content fingerprint (`config_fingerprint`) of `fingerprint_record`:
        batch k of a split and epoch is a pure function of it.
        """
        return config_fingerprint(self.fingerprint_record())

    def _set_identity(self) -> dict:
        """The fields of a materialised set's manifest that a reader of this dataset must match."""
        record = self.fingerprint_record()
        return {"split": self.group, "stage": "batch",
                "fingerprint": {**record, "sha256": config_fingerprint(record)},
                "contract": self.contract_record()}

    def _require_index_addressed(self, action: str) -> None:
        if not self.index_addressed:
            raise NotIndexAddressed(
                f"Cannot {action} a materialised set for this dataset: it reads a reader of the caller's "
                "own, so it is not addressed by index (seed_metadata['index_addressed'] is False). Its "
                "batch k depends on the batches asked for before it, so a set written, resumed or read "
                "out of that order would hold some batches twice and others never.")

    def materialise(self, root, k_range=None, *, batches: Optional[int] = None,
                    shard_size: int = DEFAULT_SHARD_SIZE) -> MaterialisedSet:
        """
        Write batches ``k_range`` (a range, or (start, stop)) or the first
        ``batches`` of the current epoch, by default the whole epoch, as a
        materialised set under the local directory ``root``, and return it
        (`gravyflow.src.dataset.materialise`).

        The set's directory is named by its manifest's sha256, which names the
        dataset's fingerprint, the platform, the split, the effective epoch and
        the range: a set already complete is returned as it is, and one an
        interrupted run left incomplete is resumed, its committed shards kept.
        Writers agree only on bit-reproducible kernels (`platform`); writers
        that disagree are refused, never merged.

        Like training, it warns on kernels not known to be pure
        (`platform.warn_if_not_pure`), whose bytes no other writer reproduces.

        Raises:
            NotIndexAddressed: a dataset read in order (``index_addressed`` False).
            IndexError: a batch outside the epoch.
            MaterialiseConflict: a file of the set exists with other bytes.
            gravyflow.src.utils.fingerprint.UnfingerprintableError: a part of
                the spec that has no content to fingerprint.
        """
        self._require_own_process()
        self._require_index_addressed("write")
        if k_range is not None and batches is not None:
            raise ValueError("Give materialise one of k_range or batches=N, not both.")
        if batches is not None:
            k_range = range(check_coordinate(batches, "batches"))
        elif k_range is None:
            k_range = range(len(self))
        elif not isinstance(k_range, range):
            k_range = range(*k_range)
        if k_range.step != 1 or not 0 <= k_range.start < k_range.stop <= len(self):
            raise IndexError(f"Batches {k_range} are outside the {len(self)} batches of an epoch.")
        if check_coordinate(shard_size, "shard_size") < 1:
            raise ValueError(f"shard_size must be at least 1, got {shard_size}.")
        platform.warn_if_not_pure()
        manifest = {
            **self._set_identity(),
            "format": FORMAT,
            "platform": platform.platform_record(),
            "shard_size": shard_size,
            "epoch": self._batch_epoch,
            "k_range": [k_range.start, k_range.stop],
        }
        return write_set(root, manifest, self._compute_batch)

    def load_materialised(self, source) -> MaterialisedSet:
        """
        Serve this dataset's batches from a complete materialised set (a
        `MaterialisedSet` or its directory) wherever the set holds the address:
        its epoch and k range; every other address is computed. Each read is
        verified, and ``seed_metadata["materialised"]`` names the set. A set
        written on another platform serves its own bytes.

        Raises:
            NotIndexAddressed: a dataset read in order (``index_addressed`` False).
            SplitMismatch, FingerprintMismatch, ContractMismatch: another split,
                spec, stage or contract.
            MissingShard: an incomplete set (``materialise`` resumes it).
            UnknownFormat, ForeignPlatform, IntegrityError: a set that is not
                what its files say.
        """
        self._require_own_process()
        self._require_index_addressed("read")
        the_set = source if isinstance(source, MaterialisedSet) else MaterialisedSet(source)
        the_set.check_reader(self._set_identity())
        the_set.require_complete()
        self._materialised = the_set
        return the_set

    def _read_materialised(self, index: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        inputs, outputs = self._materialised.read(index)
        try:
            self.batch_contract.check((self.input_variables, self.output_variables), inputs, outputs)
        except BatchContractError as error:
            raise ContractMismatch(f"Batch {index} of {self._materialised.path}: {error}") from None
        return inputs, outputs

    def prepare(self, *, batches: Optional[int] = None, epochs: Optional[int] = None) -> int:
        """
        Read, ahead of a run, every tile the first ``batches`` batches, or the batches
        of the first ``epochs`` epochs, need, fetching what the store lacks, and list
        every contradiction between the source's timeline and what it delivers at
        once. After a clean ``prepare`` those batches read nothing from the network.
        An evaluation split serves the same batches every epoch, so its epochs are
        one. Returns the number of batches prepared.

        Raises:
            Unprepared: naming every ``TimelineContradiction`` met, each with the
                ``exclusions`` entry that plans around it. Adding them changes the
                availability, so every batch: a run that adds them starts again.
        """
        self._require_own_process()
        if (batches is None) == (epochs is None):
            raise ValueError("Give prepare one of batches=N or epochs=E.")
        if epochs is not None:
            epochs = check_coordinate(epochs, "epochs")
            epochs = 1 if self.epoch_mode is EpochMode.REPEAT else epochs
            addresses = [(e, k) for e in range(epochs) for k in range(len(self))]
        else:
            batches = check_coordinate(batches, "batches")
            per_epoch = len(self)
            addresses = [(n // per_epoch if self.epoch_mode is EpochMode.FRESH else 0, n % per_epoch)
                         for n in range(batches)]
        prepare = getattr(self.noise, "prepare", None)
        if prepare is None:
            raise ValueError(
                f"Only REAL noise (RANDOM or GRID) and transients are prepared by address, not PSEUDO_REAL "
                f"noise (each row's spectrum is read when an example first draws it) or a reader of the "
                f"caller's own; this dataset's noise ({[type(o).__name__ for o in self._obtainers()]}, "
                f"{self.sampling_mode.name}) is not."
            )
        contradictions = prepare([self.tree.batch(self.group, e, k) for e, k in addresses])
        if contradictions:
            raise Unprepared(contradictions)
        return len(addresses)


class GravyflowDataset(IndexedBatches, keras.utils.PyDataset):
    """
    A Keras PyDataset to generate data for gravitational wave detection training.

    Randomness: one seed tree, rooted at ``seed`` (default ``gf.Defaults.seed``),
    with every draw addressed by (split ``group``, epoch, batch k, consumer);
    see IndexedBatches, which also says which noise is not yet read by
    address. Under ``model.fit``, pass ``callbacks=[ds.epoch_callback()]`` and ``shuffle=False`` for
    a training split. Different splits never share a stream, so train,
    validate and test built from one seed are independent.

    Within one split, datasets of one seed draw the same streams (common
    random numbers): a noise-only dataset and a signal dataset share their
    noise, so an ablation compares like with like. Datasets that must be
    independent, such as the classes of a classifier, take distinct
    ``name``s, each a scope of the seed tree, or are the pools of one
    ComposedDataset. The dataset works
    on its own copies of the obtainer and generators it is given, so the
    caller's objects are never changed and can serve other datasets; so are
    the lists it is given (its variables are tuples).

    Workers: a copy in another process (a spawned worker's) is rebuilt from the
    dataset's resolved specification and serves only if its fingerprint is
    this dataset's (``IndexedBatches.__reduce__``); a copy made by fork refuses
    to serve.

    Augmentation: a training split drawn at random applies its IFO data
    obtainer's augmentations to each batch's raw strain, per example, before
    injection and whitening, as its noise is read (``augmented_batcher``);
    evaluation data never does. AUGMENTATIONS gives what each example had.
    """

    def __init__(
        self,
        seed: Optional[int] = None,
        sample_rate_hertz: Optional[float] = None,
        onsource_duration_seconds: Optional[float] = None,
        offsource_duration_seconds: Optional[float] = None,
        crop_duration_seconds: Optional[float] = None,
        scale_factor: None = None,
        noise_obtainer: Optional[gf.NoiseObtainer] = None,
        group: str = None,
        waveform_generators: Optional[List[Union[gf.CBCGenerator, gf.WNBGenerator]]] = None,
        num_examples_per_generation_batch: Optional[int] = None,
        num_examples_per_batch: Optional[int] = None,
        ifos: Optional[List[gf.IFO]] = None,
        input_variables: Optional[List[Union[gf.WaveformParameters, gf.ReturnVariables]]] = None,
        output_variables: Optional[List[Union[gf.WaveformParameters, gf.ReturnVariables]]] = None,
        mask_history: Optional[List] = None,
        steps_per_epoch: int = 1000,
        sampling_mode: Optional["gf.SamplingMode"] = None,
        spectral_config: Optional[SpectralConfig] = None,
        epoch_mode: Optional[EpochMode] = None,
        name: Optional[str] = None,
        pearson_max_lag_seconds: Optional[float] = None,
    ):
        """
        seed: the root of the seed tree, an integer >= 0 or a gf.SeedTree.
        name: the dataset's scope in the seed tree (letters, digits and
            _.:+-). Datasets of one seed and split with different names draw
            independent streams; unnamed, they share them.
        epoch_mode: EpochMode.FRESH (default for group="train") or REPEAT.
        spectral_config: the band, PSD estimator and whitening filter shared by
            SNR scaling, SNR labels and whitening (gf.SpectralConfig; default
            band 20 Hz to Nyquist, 1 s / 0.5 s median Welch PSD, 1 s filter).
            One PSD is estimated per batch from the offsource and reused by all
            three.
        pearson_max_lag_seconds: the lag range of ``ROLLING_PEARSON_ONSOURCE``,
            every lag from minus to plus this (rounded out to whole samples); by
            default the largest light travel time between the noise's detectors.
        scale_factor: deprecated and ignored. Every input, output and label is
            physical strain (or derived from it: whitened data, SNRs).
        """
        ignore_scale_factor(scale_factor)
        # Initialize PyDataset (single-threaded to avoid JAX/fork deadlocks)
        super().__init__(workers=0)
        self._init_batches(seed, group, steps_per_epoch, epoch_mode, name)

        # Set default values
        self.sample_rate_hertz = sample_rate_hertz if sample_rate_hertz is not None else gf.Defaults.sample_rate_hertz
        self.onsource_duration_seconds = onsource_duration_seconds if onsource_duration_seconds is not None else gf.Defaults.onsource_duration_seconds
        self.offsource_duration_seconds = offsource_duration_seconds if offsource_duration_seconds is not None else gf.Defaults.offsource_duration_seconds
        self.crop_duration_seconds = crop_duration_seconds if crop_duration_seconds is not None else gf.Defaults.crop_duration_seconds
        self.num_examples_per_generation_batch = num_examples_per_generation_batch if num_examples_per_generation_batch is not None else gf.Defaults.num_examples_per_generation_batch
        self.num_examples_per_batch = num_examples_per_batch if num_examples_per_batch is not None else gf.Defaults.num_examples_per_batch
        
        # The dataset's own copies: the caller's objects are never changed.
        self.noise_obtainer = copy.deepcopy(noise_obtainer) if noise_obtainer is not None else gf.NoiseObtainer()
        self.ifos = list(ensure_list(ifos)) if ifos is not None else None
        
        # Default to RANDOM mode if not specified
        self.sampling_mode = sampling_mode if sampling_mode is not None else gf.SamplingMode.RANDOM
        self.input_variables = tuple(input_variables or ())
        self.output_variables = tuple(output_variables or ())
        self.mask_history = mask_history
        self.spectral_config = spectral_config if spectral_config is not None else SpectralConfig()
        self.pearson_max_lag_seconds = (
            get_max_arrival_time_difference(self._noise_ifos()) if pearson_max_lag_seconds is None
            else float(pearson_max_lag_seconds)
        )
        if not self.pearson_max_lag_seconds >= 0.0:
            raise ValueError(f"pearson_max_lag_seconds must be >= 0, got {pearson_max_lag_seconds}.")

        # Initialize waveform generators
        self.waveform_generators = copy.deepcopy(waveform_generators) if waveform_generators is not None else []
        if not isinstance(self.waveform_generators, list) and not isinstance(self.waveform_generators, dict):
            self.waveform_generators = [self.waveform_generators]

        # Set network for waveform generators if not provided
        self._set_generator_networks()

        # Validate settings
        self._validate_settings()
        self.batch_contract = self._batch_contract()

        # Create noise and injection generators
        self._create_generators()
        self._built()

    def _noise_ifos(self) -> List:
        """The detectors the noise will have (``noise_ifos``)."""
        return noise_ifos(self.ifos, self.noise_obtainer)

    def _set_generator_networks(self):
        """
        Give every generator a network of exactly the noise's detectors, in order.

        A generator without a network (the dataset's own copy) gets one built
        from the noise's detectors. A generator with its own network must match
        the noise's detectors; a mismatch would broadcast one projection into
        several channels or mislabel them, so it raises.
        """
        noise_ifos = self._noise_ifos()

        def bind(generator):
            if generator.network is None:
                generator.network = gf.Network(noise_ifos)
            else:
                check_network_matches_ifos(generator.network, noise_ifos)
            return generator

        if isinstance(self.waveform_generators, list):
            self.waveform_generators = [bind(generator) for generator in self.waveform_generators]
        elif isinstance(self.waveform_generators, dict):
            bound = {}
            for name, entry in self.waveform_generators.items():
                entry = dict(entry)
                entry["generator"] = bind(entry["generator"])
                bound[name] = entry
            self.waveform_generators = bound

    def _validate_settings(self):
        """Validate the generator settings."""
        self.variables_to_return = set(self.input_variables + self.output_variables)
        if not self.variables_to_return:
            raise ValueError("No return variables requested. What's the point?")
        validate_noise_settings(self.noise_obtainer, self.variables_to_return)
        check_scaling_matches_noise(self.noise_obtainer, self._generator_list())
        self._validate_padding_settings()
        self._validate_spectral_settings()

    def _generator_list(self) -> List:
        if isinstance(self.waveform_generators, list):
            return list(self.waveform_generators)
        if isinstance(self.waveform_generators, dict):
            return [g["generator"] for g in self.waveform_generators.values()]
        return []

    def _validate_spectral_settings(self):
        """
        Check the windows against the PSD estimate and the whitening filter, once,
        at construction:
        - whitening needs a crop of at least half the filter on each side of the
          onsource (so the padded onsource is at least as long as the filter),
          since the filter corrupts that much at each edge;
        - SNR scaling and whitening need an offsource holding at least
          MIN_PSD_SEGMENTS Welch segments.
        """
        config = self.spectral_config
        fs = self.sample_rate_hertz
        config.validate_band(fs)

        self._needs_whitening = any(
            variable in self.variables_to_return for variable in WHITENED_VARIABLES
        )
        has_snr_scaling = any(
            getattr(getattr(generator, "scaling_method", None), "type_", None) == gf.ScalingTypes.SNR
            for generator in self._generator_list()
        )
        # SNR labels of HRSS/HPEAK-scaled injections are measured against the
        # batch's spectrum too, so they share the one definition.
        has_snr_label = self._generator_list() and gf.ScalingTypes.SNR in self.variables_to_return
        self._needs_spectrum = self._needs_whitening or has_snr_scaling or bool(has_snr_label)

        if self._needs_whitening:
            ntaps = config.num_taps(fs)
            crop_samples_each_side = int(round(self.crop_duration_seconds * fs))
            if crop_samples_each_side < ntaps // 2:
                raise ValueError(
                    f"crop_duration_seconds ({self.crop_duration_seconds}) must be at least half of "
                    f"the whitening filter ({config.filter_duration_seconds} s): the filter corrupts "
                    f"{config.filter_duration_seconds / 2:g} s at each edge of the window."
                )
            if config.fft_duration_seconds > config.filter_duration_seconds:
                warn(
                    f"The PSD segments ({config.fft_duration_seconds} s) are longer than the "
                    f"whitening filter ({config.filter_duration_seconds} s), which cannot realise "
                    "the PSD's finer features (narrow lines are not removed).",
                    UserWarning
                )

        if self._needs_spectrum:
            num_offsource = int(round(self.offsource_duration_seconds * fs))
            nperseg = config.nperseg(fs)
            noverlap = config.noverlap(fs)
            num_segments = 0 if num_offsource < nperseg else (num_offsource - nperseg) // (nperseg - noverlap) + 1
            if num_segments < MIN_PSD_SEGMENTS:
                raise ValueError(
                    f"offsource_duration_seconds ({self.offsource_duration_seconds}) gives "
                    f"{num_segments} Welch segment(s) of {config.fft_duration_seconds} s; the PSD "
                    f"needs at least {MIN_PSD_SEGMENTS}."
                )

        if config.lower_frequency_hertz:
            for generator in self._generator_list():
                lowest = lowest_generator_frequency(generator)
                if lowest is not None and lowest < config.lower_frequency_hertz:
                    warn(
                        f"{type(generator).__name__} can produce power at {lowest:g} Hz, below the "
                        f"band's lower edge ({config.lower_frequency_hertz:g} Hz). That power is "
                        "removed by whitening and not counted in the SNR, so SNR scaling can "
                        "make such signals very loud in the raw onsource.",
                        UserWarning
                    )

    def _batch_contract(self) -> BatchContract:
        """The shapes and dtypes every batch of this dataset is checked against."""
        return BatchContract(
            batch=self.num_examples_per_batch,
            generators=len(self._generator_list()),
            ifos=len(self._noise_ifos()),
            onsource_samples=window_samples(self.onsource_duration_seconds, self.sample_rate_hertz),
        )

    def _batch_spectrum(self, offsource):
        """The batch's one PSD estimate, shared by scaling, labels and whitening."""
        return estimate_psd(offsource, self.sample_rate_hertz, self.spectral_config)

    def _validate_padding_settings(self):
        """Validate that padding does not exceed half of onsource duration."""
        if self.waveform_generators:
            generators = []
            if isinstance(self.waveform_generators, list):
                generators = self.waveform_generators
            elif isinstance(self.waveform_generators, dict):
                generators = [g["generator"] for g in self.waveform_generators.values()]
            
            for generator in generators:
                if getattr(generator, "scaling_method", None) is None:
                    raise ValueError(
                        f"{type(generator).__name__} has no scaling_method; give it a "
                        "gf.ScalingMethod to inject it into a dataset."
                    )
                # None means "as far as the window allows".
                front = generator.front_padding_duration_seconds or 0.0
                back = generator.back_padding_duration_seconds or 0.0

                # Condition: padding <= onsource / 2
                # If padding is larger, the signal center (placed at center of window)
                # could be shifted outside the cropped onsource window.
                max_padding = max(front, back)
                limit = self.onsource_duration_seconds / 2.0
                
                if max_padding > limit:
                    warn(
                        f"Padding ({max_padding}s) exceeds half of onsource duration ({limit}s). "
                        "This may cause the signal center to be shifted outside the cropped view.",
                        UserWarning
                    )

    def _create_generators(self):
        """Create noise and injection generators."""
        # Reading the data starts here: first the availability every plan reads is frozen.
        freeze_availability(self._obtainers(), self._noise_ifos())
        # Build base kwargs for noise obtainer
        noise_kwargs = {
            'sample_rate_hertz': self.sample_rate_hertz,
            'onsource_duration_seconds': self.onsource_duration_seconds,
            'crop_duration_seconds': self.crop_duration_seconds,
            'offsource_duration_seconds': self.offsource_duration_seconds,
            'num_examples_per_batch': self.num_examples_per_batch,
            'sampling_mode': self.sampling_mode,
            'steps_per_epoch': self.steps_per_epoch,
        }
        # Add ifos to noise_kwargs if provided
        if self.ifos is not None:
            noise_kwargs['ifos'] = self.ifos

        self.noise = augmented_batcher(self.noise_obtainer, self.tree, self.group, **noise_kwargs)

        waveform_parameters_to_return = [
            item for item in self.variables_to_return if isinstance(
                item.value, gf.WaveformParameter
            )
        ]
        self.injection_generator = gf.InjectionGenerator(
            waveform_generators=self.waveform_generators,
            parameters_to_return=waveform_parameters_to_return,
        )

    def _obtainers(self) -> List:
        return [self.noise_obtainer]

    #: What the specification is made of (`specification`): every setting that decides the batches.
    _SPECIFICATION = (
        "name", "sample_rate_hertz", "onsource_duration_seconds", "offsource_duration_seconds",
        "crop_duration_seconds", "num_examples_per_generation_batch", "num_examples_per_batch",
        "noise_obtainer", "waveform_generators", "ifos", "sampling_mode", "input_variables",
        "output_variables", "mask_history", "spectral_config", "pearson_max_lag_seconds", "steps_per_epoch",
        "epoch_mode",
    )

    def specification(self) -> Dict[str, Any]:
        """
        Every setting that decides the dataset's batches, by name: its own
        copies of what it was given, with the defaults filled in. Not the seed,
        split or epoch, which address batches (`fingerprint_record`).
        """
        unset = [name for name in self._SPECIFICATION if not hasattr(self, name)]
        if unset:
            raise TypeError(
                f"A {type(self).__name__} has no {', '.join(unset)}, which its _SPECIFICATION declares: the "
                "base constructor records the specification the dataset is built from as its last step, so "
                "a subclass sets its own settings before calling super().__init__()."
            )
        return {name: getattr(self, name) for name in self._SPECIFICATION}

    def contract_record(self) -> Dict[str, List]:
        """
        The batch contract (`BatchContract`) as data: [name, shape, dtype] of
        each input and output, in order. A size the settings do not fix keeps
        its symbol (the offsource length, "T_off"; trailing axes, "...").
        """
        def entry(key):
            shape = self.batch_contract.shape(key, OFFSOURCE_SAMPLES)
            return [key.name, ["..." if axis is ANY else axis for axis in shape], self.batch_contract.dtype(key)]

        return {"inputs": [entry(key) for key in self.input_variables],
                "outputs": [entry(key) for key in self.output_variables]}

    def _compute_batch(self, index: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Batch ``index`` of the current epoch, computed from its streams."""
        streams = self.streams(index)
        # Errors pass through as they were raised: a refusal keeps its type and what it
        # names (a TimelineContradiction its exclusion).
        batch_data = self.noise(streams)
        onsource = batch_data[gf.ReturnVariables.ONSOURCE]
        offsource = batch_data[gf.ReturnVariables.OFFSOURCE]
        # The first returned onsource sample's time, in every mode.
        gps_times = batch_data.get(gf.ReturnVariables.START_GPS_TIME)
        feature_labels = batch_data.get(gf.ReturnVariables.SUB_TYPE)
        # Transient-specific outputs
        glitch_type = batch_data.get(gf.ReturnVariables.GLITCH_TYPE)
        source_type = batch_data.get(gf.ReturnVariables.SOURCE_TYPE)
        data_label = batch_data.get(gf.ReturnVariables.DATA_LABEL)
        # A composed dataset's noise names each example's pool.
        pool_label = batch_data.get(gf.ReturnVariables.POOL_LABEL)
        # The augmentations each example's noise had as it was read.
        augmentations = batch_data.get(gf.ReturnVariables.AUGMENTATIONS)
        # Whether each example's detectors were read at one time.
        coincident = batch_data.get(gf.ReturnVariables.COINCIDENT)

        injections, mask, parameters = self.injection_generator.generate(
            streams,
            sample_rate_hertz=self.sample_rate_hertz,
            onsource_duration_seconds=self.onsource_duration_seconds,
            crop_duration_seconds=self.crop_duration_seconds,
            num_examples_per_batch=self.num_examples_per_batch,
        )

        # One PSD per batch: scaling, labels and whitening all read this estimate.
        spectrum = self._batch_spectrum(offsource) if self._needs_spectrum else None

        onsource, scaled_injections, labels = self.injection_generator.add_injections_to_onsource(
            injections,
            mask,
            onsource,
            spectrum if spectrum is not None else offsource,
            parameters_to_return=self.variables_to_return,
            onsource_duration_seconds=self.onsource_duration_seconds,
            injection_parameters=parameters,
            streams=streams,
            epoch=self._schedule_epoch,
            steps_per_epoch=len(self),
        )

        whitened_injections = self._process_whitened_injections(scaled_injections, spectrum)
        injections = self._process_injections(scaled_injections)
        labels = self._batch_labels(labels, mask)

        conditioned = self._condition(onsource, offsource, spectrum)
        gps_times = self._process_gps_times(gps_times)

        input_dict, output_dict = self._create_output_dictionaries(
            conditioned["onsource"], conditioned["whitened_onsource"], conditioned["offsource"], gps_times,
            injections, whitened_injections, labels[gf.ReturnVariables.INJECTION_MASKS],
            conditioned["rolling_pearson_onsource"], conditioned["spectrogram_onsource"], labels,
            feature_labels=self._process_transient_label(feature_labels),
            glitch_type=self._process_transient_label(glitch_type),
            source_type=self._process_transient_label(source_type),
            data_label=self._process_transient_label(data_label),
            pool_label=self._process_transient_label(pool_label),
            augmentations=augmentations,
            coincident=self._process_transient_label(coincident),
        )
        self.batch_contract.check(
            (self.input_variables, self.output_variables), input_dict, output_dict
        )

        return input_dict, output_dict

    @property
    def window_spec(self) -> WindowSpec:
        """The geometry of the raw windows the dataset conditions (``model_inputs``)."""
        return WindowSpec(self.sample_rate_hertz, self.onsource_duration_seconds,
                          self.offsource_duration_seconds, self.crop_duration_seconds)

    def model_inputs(self, onsource, offsource) -> Dict[str, Any]:
        """
        The model inputs (``input_variables``) of raw windows that hold nothing
        injected: each padded onsource (B, I, ``window_spec.num_onsource_samples``)
        with the offsource just before it (B, I, ``window_spec.num_offsource_samples``).

        They are conditioned by the code that conditions a batch's (``_condition``):
        each example's spectrum from its offsource, then whitening, cropping and the
        derived inputs, so windows of another origin (a real event, time-slid noise)
        score exactly as the dataset's own examples do.
        """
        spectrum = self._batch_spectrum(offsource) if self._needs_spectrum else None
        conditioned = self._condition(onsource, offsource, spectrum)
        inputs = create_variable_dictionary(
            self.input_variables, conditioned["onsource"], conditioned["whitened_onsource"],
            conditioned["offsource"], None, None, None, None,
            conditioned["rolling_pearson_onsource"], conditioned["spectrogram_onsource"], None,
        )
        missing = [variable.name for variable in self.input_variables if inputs.get(variable.name) is None]
        if missing:
            raise ValueError(
                f"Raw windows cannot give the inputs {missing}: only the conditioned strain and "
                "what derives from it."
            )
        return inputs

    def _condition(self, onsource, offsource, spectrum) -> Dict[str, Any]:
        """
        The conditioned strain of a batch: its whitened onsource and what derives from
        it (rolling Pearson, spectrogram), and its cropped onsource and offsource, each
        only if requested (None otherwise). The one conditioning of every window the
        dataset serves or turns into ``model_inputs``.
        """
        whitened_onsource, rolling_pearson_onsource, spectrogram_onsource = self._process_onsource(
            onsource, spectrum)
        return {
            "onsource": self._process_raw_onsource(onsource),
            "whitened_onsource": whitened_onsource,
            "offsource": self._process_offsource(offsource),
            "rolling_pearson_onsource": rolling_pearson_onsource,
            "spectrogram_onsource": spectrogram_onsource,
        }

    def _process_whitened_injections(self, scaled_injections, spectrum):
        """Whiten the injections (G, B, I, T) with the batch's PSD, if required."""
        if gf.ReturnVariables.WHITENED_INJECTIONS in self.variables_to_return:
            whitened_injections = whiten(scaled_injections, spectrum, self.sample_rate_hertz)
            whitened_injections = gf.crop_samples(
                whitened_injections,
                self.onsource_duration_seconds,
                self.sample_rate_hertz
            )
            whitened_injections = ops.cast(whitened_injections, "float32")
            return _batch_first(gf.replace_nan_and_inf_with_zero(
                whitened_injections, gf.ReturnVariables.WHITENED_INJECTIONS.name
            ))
        return None

    def _process_injections(self, scaled_injections):
        """Process raw injections if required."""
        if gf.ReturnVariables.INJECTIONS in self.variables_to_return:
            # Crop raw injections to match onsource duration
            injections = gf.crop_samples(
                scaled_injections,
                self.onsource_duration_seconds,
                self.sample_rate_hertz
            )
            injections = ops.cast(injections, "float32")
            return _batch_first(gf.replace_nan_and_inf_with_zero(
                injections, gf.ReturnVariables.INJECTIONS.name
            ))
        return None

    def _process_onsource(self, onsource, spectrum):
        """Process onsource data, including whitening, rolling Pearson, and spectrogram generation."""
        if self._needs_whitening:
            whitened_onsource = whiten(onsource, spectrum, self.sample_rate_hertz)
            whitened_onsource = gf.crop_samples(
                whitened_onsource,
                self.onsource_duration_seconds,
                self.sample_rate_hertz
            )
            # float32, scrubbed before any cast: a float16 cast turned values
            # above 65504 into inf and then silently into 0.
            whitened_onsource = gf.replace_nan_and_inf_with_zero(
                ops.cast(whitened_onsource, "float32"), gf.ReturnVariables.WHITENED_ONSOURCE.name
            )

            rolling_pearson_onsource = self._calculate_rolling_pearson(whitened_onsource)
            spectrogram_onsource = self._calculate_spectrogram(whitened_onsource)

            return whitened_onsource, rolling_pearson_onsource, spectrogram_onsource
        return None, None, None

    def _calculate_rolling_pearson(self, whitened_onsource):
        """Calculate rolling Pearson correlation if required."""
        if gf.ReturnVariables.ROLLING_PEARSON_ONSOURCE in self.variables_to_return:
            return gf.rolling_pearson(whitened_onsource, self.pearson_max_lag_seconds, self.sample_rate_hertz)
        return None

    def _calculate_spectrogram(self, whitened_onsource):
        """Calculate spectrogram if required."""
        if gf.ReturnVariables.SPECTROGRAM_ONSOURCE in self.variables_to_return:
            return gf.spectrogram(whitened_onsource)
        return None

    def _process_raw_onsource(self, onsource):
        """Process raw onsource data if required."""
        if gf.ReturnVariables.ONSOURCE in self.variables_to_return:
            onsource = gf.crop_samples(
                onsource, 
                self.onsource_duration_seconds, 
                self.sample_rate_hertz
            )
            onsource = ops.cast(onsource, "float32")
            return gf.replace_nan_and_inf_with_zero(onsource, gf.ReturnVariables.ONSOURCE.name)
        return None

    def _process_offsource(self, offsource):
        """Process offsource data if required."""
        if gf.ReturnVariables.OFFSOURCE in self.variables_to_return:
            offsource = ops.cast(offsource, "float32")
            return gf.replace_nan_and_inf_with_zero(offsource, gf.ReturnVariables.OFFSOURCE.name)
        return None

    def _process_gps_times(self, gps_times):
        """GPS times (B, I), float64, if required; NaN where the noise has none."""
        if gf.ReturnVariables.START_GPS_TIME in self.variables_to_return:
            if gps_times is None:
                return np.full(self.batch_contract.shape(gf.ReturnVariables.START_GPS_TIME), np.nan)
            return as_gps_times(gps_times)
        return None

    @staticmethod
    def _process_transient_label(label):
        """A class label (a transient's ids per detector, a pool's) as float32, like every label."""
        return None if label is None else ops.cast(label, "float32")

    def _batch_labels(self, labels, mask):
        """
        The batch's labels, batch-first: INJECTION_MASKS (B, G),
        INJECTION_PRESENT (B,), and every per-generator label requested,
        (B, G) + its own shape, float32. A label is NaN where generator g
        injected nothing into example b (the injection layer's rule), and
        wholly NaN when no generator draws it.
        """
        mask = ops.transpose(ops.cast(mask, "float32"))
        batch = {
            gf.ReturnVariables.INJECTION_MASKS: mask,
            gf.ReturnVariables.INJECTION_PRESENT: ops.cast(ops.any(mask > 0, axis=1), "float32"),
        }
        for key in self.variables_to_return:
            if isinstance(key, (gf.WaveformParameters, gf.ScalingTypes)) \
                    or key is gf.ReturnVariables.CENTRAL_TIME:
                value = labels.get(key)
                batch[key] = _batch_first(value) if value is not None else \
                    ops.full(self.batch_contract.shape(key), np.nan, dtype="float32")
        return batch

    def _create_output_dictionaries(self, *args, **labels) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        input_dict = create_variable_dictionary(self.input_variables, *args, **labels)
        output_dict = create_variable_dictionary(self.output_variables, *args, **labels)
        
        # Remove keys with None values
        input_dict = {k: v for k, v in input_dict.items() if v is not None}
        output_dict = {k: v for k, v in output_dict.items() if v is not None}
        
        return input_dict, output_dict

def _batch_first(value):
    """A generator-first (G, B, ...) array as batch-first (B, G, ...)."""
    return ops.moveaxis(ops.convert_to_tensor(value), 0, 1)


def create_variable_dictionary(
    return_variables: List[Union[gf.ReturnVariables, gf.WaveformParameters, gf.ScalingTypes]],
    onsource,
    whitened_onsource,
    offsource,
    gps_times,
    injections,
    whitened_injections,
    mask,
    rolling_pearson_onsource,
    spectrogram_onsource,
    injection_parameters : Optional[Dict],
    feature_labels = None,
    glitch_type = None,
    source_type = None,
    data_label = None,
    pool_label = None,
    augmentations = None,
    coincident = None,
    ) -> Dict:
    """
    Create dictionary of requested return variables.
    
    Supports ReturnVariables, WaveformParameters, and ScalingTypes. Every
    WaveformParameter and ScalingType comes from ``injection_parameters``; the
    ScalingTypes (SNR, HRSS, HPEAK) are the labels the injection layer
    recorded, never recomputed here.
    
    Args:
        feature_labels: Labels for SUB_TYPE classification (from TransientObtainer).
        glitch_type: GLITCH_TYPE labels (from TransientObtainer).
        source_type: SOURCE_TYPE labels (from TransientObtainer).
        data_label: DATA_LABEL values (from TransientObtainer).
        pool_label: POOL_LABEL values (a ComposedDataset's pools).
        augmentations: AUGMENTATIONS, each example's augmentation flags.
        coincident: COINCIDENT, 1 where an example's detectors were read at one time.
    """
    from keras import ops
    
    operations = {
        gf.ReturnVariables.ONSOURCE: onsource,
        gf.ReturnVariables.WHITENED_ONSOURCE: whitened_onsource,
        gf.ReturnVariables.OFFSOURCE: offsource,
        gf.ReturnVariables.START_GPS_TIME: gps_times,
        gf.ReturnVariables.INJECTIONS: injections,
        gf.ReturnVariables.WHITENED_INJECTIONS: whitened_injections,
        gf.ReturnVariables.INJECTION_MASKS: mask,
        gf.ReturnVariables.ROLLING_PEARSON_ONSOURCE: rolling_pearson_onsource,
        gf.ReturnVariables.SPECTROGRAM_ONSOURCE: spectrogram_onsource,
        gf.ReturnVariables.SUB_TYPE: feature_labels,
        # Transient-specific outputs
        gf.ReturnVariables.GLITCH_TYPE: glitch_type,
        gf.ReturnVariables.SOURCE_TYPE: source_type,
        gf.ReturnVariables.DATA_LABEL: data_label,
        gf.ReturnVariables.POOL_LABEL: pool_label,
        gf.ReturnVariables.AUGMENTATIONS: augmentations,
        gf.ReturnVariables.COINCIDENT: coincident,
    }

    # Add WaveformParameters from injection_parameters
    if injection_parameters:
        operations.update(
            {
                key: value for key, value in injection_parameters.items() \
                if key in return_variables
            }
        )
    
    return {
        key.name: operations[key] for key in return_variables \
        if key in operations
    }

# Legacy wrapper for compatibility
def Dataset(**kwargs):
    return GravyflowDataset(**kwargs)


# =============================================================================
# COMPOSED DATASET
# =============================================================================

def _noise_defaults() -> Dict:
    """
    What an example carries in a composed batch where its pool's source gives
    no such output (synthetic noise): no GPS time, and the labels a transient
    obtainer gives a partner detector, noise with no type, its detectors read
    apart.
    """
    return {
        gf.ReturnVariables.START_GPS_TIME: (np.nan, np.float64),
        gf.ReturnVariables.DATA_LABEL: (gf.DataLabel.NOISE.value, np.int32),
        gf.ReturnVariables.SUB_TYPE: (-1, np.int32),
        gf.ReturnVariables.GLITCH_TYPE: (-1, np.int32),
        gf.ReturnVariables.SOURCE_TYPE: (-1, np.int32),
        gf.ReturnVariables.COINCIDENT: (0, np.int32),
    }


class _PoolInjections(InjectionGenerator):
    """
    A composed dataset's injection layer: the generators of every pool, in
    pool order, each drawing from its pool's streams (``pool/{name}/``) and
    injecting only into the examples assigned to its pool. Elsewhere its
    waveform is zero and its mask 0, so its labels there are NaN.
    """

    def __init__(self, pools, pool_of, generators, assignment, parameters_to_return):
        super().__init__(
            waveform_generators=generators,
            parameters_to_return=parameters_to_return,
            scopes=[f"pool/{pools[index].name}" for index in pool_of],
        )
        self._pool_of = np.asarray(pool_of, dtype=int)
        self._assignment = assignment

    def generate(self, streams, **window):
        injections, masks, parameters = super().generate(streams, **window)
        # (G, B): generator g's pool holds example b.
        assigned = (self._pool_of[:, None] == self._assignment(streams)[None, :]).astype(np.float32)
        rows = assigned.reshape(assigned.shape + (1,) * (len(ops.shape(injections)) - 2))
        return injections * rows, masks * assigned, parameters


class ComposedDataset(GravyflowDataset):
    """
    A dataset composed of FeaturePools, each a source of examples with its
    own class label (POOL_LABEL) and sampling probability.

    Each example's pool is drawn from the batch's ``pool/assign`` stream, and
    example b is row b of its pool's batch. Every draw of a pool sits under
    ``pool/{name}/``, so pools never share a stream, and a pool's injection
    generators inject into its own examples only. A pool's noise is augmented
    as a GravyflowDataset's is, by its own source, for the dataset's split, as
    it is read: example b has its pool's augmentation of row b, and
    AUGMENTATIONS says so.

    Otherwise a composed batch is a GravyflowDataset's, built by the same code
    (see GravyflowDataset and IndexedBatches): one PSD per batch for scaling,
    labels and whitening, and the batch contract, where the generators G are
    every pool's, in pool order; a generator is absent (mask 0, labels NaN)
    from the other pools' examples. Where a pool's source gives no GPS time or
    transient labels (synthetic noise), its examples have NaN, DATA_LABEL
    NOISE and -1 for the types, as a transient obtainer labels a partner
    detector. With one pool, a composed dataset serves exactly what a
    GravyflowDataset of seed ``gf.SeedTree(seed).sub(f"pool/{name}")`` does.

    Example:
        pools = [
            gf.FeaturePool(
                name="pure_noise",
                label=0,
                probability=0.5,
                noise_obtainer=gf.NoiseObtainer(noise_type=gf.NoiseType.REAL, ...)
            ),
            gf.FeaturePool(
                name="glitches",
                label=1,
                probability=0.3,
                transient_obtainer=gf.TransientDataObtainer(...)
            ),
            gf.FeaturePool(
                name="cbc_signals",
                label=2,
                probability=0.2,
                noise_obtainer=gf.NoiseObtainer(...),
                injection_generators=[cbc_generator]
            ),
        ]

        dataset = gf.ComposedDataset(
            pools=pools,
            sample_rate_hertz=2048.0,
            onsource_duration_seconds=1.0,
            num_examples_per_batch=32,
            input_variables=[gf.ReturnVariables.WHITENED_ONSOURCE],
            output_variables=[gf.ReturnVariables.POOL_LABEL]
        )
    """

    def __init__(
        self,
        pools: List["gf.FeaturePool"],
        seed: Optional[int] = None,
        sample_rate_hertz: Optional[float] = None,
        onsource_duration_seconds: Optional[float] = None,
        offsource_duration_seconds: Optional[float] = None,
        crop_duration_seconds: Optional[float] = None,
        scale_factor: None = None,
        num_examples_per_generation_batch: Optional[int] = None,
        num_examples_per_batch: Optional[int] = None,
        ifos: Optional[List[gf.IFO]] = None,
        input_variables: Optional[List[Union[gf.WaveformParameters, gf.ReturnVariables]]] = None,
        output_variables: Optional[List[Union[gf.WaveformParameters, gf.ReturnVariables]]] = None,
        steps_per_epoch: int = 1000,
        group: str = None,
        epoch_mode: Optional[EpochMode] = None,
        spectral_config: Optional[SpectralConfig] = None,
    ):
        """
        Args:
            pools: the FeaturePools the examples are drawn from.
            seed: The root of the seed tree (an integer >= 0 or a gf.SeedTree).
            sample_rate_hertz: Sample rate in Hz.
            onsource_duration_seconds: Duration of onsource window.
            offsource_duration_seconds: Duration of offsource window.
            crop_duration_seconds: Duration to crop from onsource edges.
            scale_factor: Deprecated and ignored: the data are physical strain.
            num_examples_per_generation_batch: Batch size for waveform generation.
            num_examples_per_batch: Batch size for returned data.
            ifos: the detectors of every pool; default the detectors the
                pools' sources share, else L1.
            input_variables: Variables to include in input dict.
            output_variables: Variables to include in output dict.
            steps_per_epoch: Number of batches per epoch.
            group: Dataset split (train/val/test).
            epoch_mode: EpochMode.FRESH (default for group="train") or REPEAT.
            spectral_config: as GravyflowDataset's.
        """
        from gravyflow.src.dataset.pools import PoolSampler

        ignore_scale_factor(scale_factor)
        # PyDataset's own initialiser: GravyflowDataset's builds from one obtainer.
        keras.utils.PyDataset.__init__(self, workers=0)
        self._init_batches(seed, group, steps_per_epoch, epoch_mode)

        self.sample_rate_hertz = sample_rate_hertz if sample_rate_hertz is not None else Defaults.sample_rate_hertz
        self.onsource_duration_seconds = onsource_duration_seconds if onsource_duration_seconds is not None else Defaults.onsource_duration_seconds
        self.offsource_duration_seconds = offsource_duration_seconds if offsource_duration_seconds is not None else Defaults.offsource_duration_seconds
        self.crop_duration_seconds = crop_duration_seconds if crop_duration_seconds is not None else Defaults.crop_duration_seconds
        self.num_examples_per_generation_batch = num_examples_per_generation_batch if num_examples_per_generation_batch is not None else Defaults.num_examples_per_generation_batch
        self.num_examples_per_batch = num_examples_per_batch if num_examples_per_batch is not None else Defaults.num_examples_per_batch
        self.ifos = list(ensure_list(ifos)) if ifos is not None else None
        self.sampling_mode = gf.SamplingMode.RANDOM
        self.input_variables = tuple(input_variables or ())
        self.output_variables = tuple(output_variables or ())
        self.mask_history = None
        self.spectral_config = spectral_config if spectral_config is not None else SpectralConfig()

        # The dataset's own copies: the caller's pools are never changed.
        self.pools = copy.deepcopy(pools)
        self.pool_sampler = PoolSampler(self.pools)
        self._generator_pools = [
            index for index, pool in enumerate(self.pools) for _ in pool.injection_generators or []
        ]
        self.waveform_generators = [
            generator for pool in self.pools for generator in pool.injection_generators or []
        ]

        self._set_generator_networks()
        self._validate_settings()
        self.batch_contract = self._batch_contract()
        self._create_generators()
        self._built()

    _SPECIFICATION = (
        "pools", "sample_rate_hertz", "onsource_duration_seconds", "offsource_duration_seconds",
        "crop_duration_seconds", "num_examples_per_generation_batch", "num_examples_per_batch", "ifos",
        "input_variables", "output_variables", "spectral_config", "steps_per_epoch", "epoch_mode",
    )

    def _obtainers(self) -> List:
        return [
            pool.noise_obtainer if pool.is_noise_pool else pool.transient_obtainer
            for pool in self.pools
        ]

    def _noise_ifos(self) -> List:
        """
        The detectors of every pool's noise: the dataset's ``ifos``, else the
        detectors the pools' sources share, else L1.
        """
        if self.ifos is not None:
            return self.ifos
        sources = {}
        for obtainer in self._obtainers():
            ifos = getattr(obtainer, "ifos", None)
            if ifos is not None:
                sources[tuple(_ifo_names(ifos))] = ensure_list(ifos)
        if len(sources) > 1:
            raise ValueError(
                f"The pools' sources have different detectors, {sorted(sources)}: give the "
                "dataset the ifos every pool should have."
            )
        return next(iter(sources.values()), [gf.IFO.L1])

    def _validate_settings(self):
        """GravyflowDataset's checks, with each pool's source checked as its noise."""
        self.variables_to_return = set(self.input_variables + self.output_variables)
        if not self.variables_to_return:
            raise ValueError("No return variables requested. What's the point?")
        for obtainer in self._obtainers():
            validate_noise_settings(obtainer, self.variables_to_return)
        self._validate_padding_settings()
        self._validate_spectral_settings()

    def _create_generators(self):
        """Each pool's noise batcher, the mixture of them, and the pools' injection layer."""
        # Reading the data starts here: first the availability every plan reads is frozen.
        freeze_availability(self._obtainers(), self._noise_ifos())
        window = {
            'sample_rate_hertz': self.sample_rate_hertz,
            'onsource_duration_seconds': self.onsource_duration_seconds,
            'crop_duration_seconds': self.crop_duration_seconds,
            'offsource_duration_seconds': self.offsource_duration_seconds,
            'num_examples_per_batch': self.num_examples_per_batch,
            'ifos': self._noise_ifos(),
            'sampling_mode': self.sampling_mode,
            'steps_per_epoch': self.steps_per_epoch,
        }
        self._pool_noise = [
            augmented_batcher(obtainer, self.tree.sub(f"pool/{pool.name}"), self.group, **window)
            for pool, obtainer in zip(self.pools, self._obtainers())
        ]
        self.noise = self._mixed_noise
        self.injection_generator = _PoolInjections(
            self.pools, self._generator_pools, self.waveform_generators, self._assignment,
            parameters_to_return=[
                item for item in self.variables_to_return if isinstance(item.value, gf.WaveformParameter)
            ],
        )

    def _assignment(self, streams: BatchStreams) -> np.ndarray:
        """Each example's pool index, drawn from the batch's ``pool/assign`` stream."""
        return self.pool_sampler.sample_pool_indices(
            self.num_examples_per_batch, streams.stream("pool/assign")
        )

    def _mixed_noise(self, streams: BatchStreams) -> Dict:
        """
        The batch's noise, as an obtainer's batch: example b is row b of its
        pool's batch (drawn from ``pool/{name}/``, augmented), with its pool's
        label. Only the pools the batch holds are read.
        """
        assignment = self._assignment(streams)
        defaults = _noise_defaults()
        mixed = {}
        for index in np.unique(assignment):
            pool = self.pools[index]
            batch = self._pool_noise[index](streams.sub(f"pool/{pool.name}"))
            rows = assignment == index
            per_detector = tuple(np.shape(batch[gf.ReturnVariables.ONSOURCE])[:2])
            for key in (gf.ReturnVariables.ONSOURCE, gf.ReturnVariables.OFFSOURCE,
                        gf.ReturnVariables.AUGMENTATIONS, *defaults):
                if key in batch:
                    value = np.asarray(batch[key])
                else:
                    fill, dtype = defaults[key]
                    # Per example, or per example and detector, as the key's contract is.
                    value = np.full(per_detector[:len(key.value.shape)], fill, dtype=dtype)
                if key not in mixed:
                    mixed[key] = np.empty((len(assignment),) + value.shape[1:], dtype=value.dtype)
                mixed[key][rows] = value[rows]
        labels = np.array([pool.label for pool in self.pools], dtype=np.float32)
        mixed[gf.ReturnVariables.POOL_LABEL] = labels[assignment]
        return mixed
