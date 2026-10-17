"""
The run recipe of the example models: how a training run is laid out, resumed
and evaluated.

- **Presets.** `TrainingSchedule.cpu` sizes a run for a laptop CPU;
  `TrainingSchedule.gpu` is the scale the papers used.
- **Seeds.** One root seed for every split: each dataset draws from its split's
  streams of the seed tree (its ``group``), so the training, validation and
  test data never share a stream. The training data takes Keras' epoch
  (``epoch_callback``), so each epoch, and a resumed run's first, draws that
  epoch's batches rather than replaying earlier ones.
- **Reproducible from the seed.** The dataset owns the order: ``fit`` serves
  each epoch's batches in index order (Keras' shuffle would draw from Python's
  unseeded global random), and the model's initial weights and dropout seeds
  come from the seed tree (`keras_draws`). Two runs of one seed train the same
  batches in the same order from the same weights, and a resumed run trains as
  an uninterrupted one: bit for bit when XLA runs single-threaded on a CPU
  (``XLA_FLAGS=--xla_cpu_multi_thread_eigen=false``). Its multi-threaded
  kernels round the whitened strain differently from call to call, in the last
  bits, and training amplifies the difference.
- **A run directory outside the package.** `RunDirectory` refuses any path
  inside the installed library, so a run can never overwrite shipped files.
- **Resume.** Each epoch ends with the model and its optimiser saved
  (``last.keras``), the best model so far (``best.keras``), and the run's state
  (``state.json``: epoch, early-stopping counter, history). Training started
  again in the same directory continues from the last epoch.
- **Provenance.** Every checkpoint records the sha256 of its weights and of the
  run's configuration. Loading one recomputes both and refuses a mismatch:
  a checkpoint from another configuration, or one whose file changed, is never
  used silently.
- **The split.** A run records the split its data reads, never one declared
  beside it: the fingerprint of the training and validation data's
  ``SplitSpec`` and the validation data's split, which selects the model
  (``run.json``'s ``split``). The record belongs to the weights: every model
  the run returns or its directory loads carries it (`with_run_record`), and
  the validator reads it there. Data split by two specs, a resumed run's data
  split otherwise, or a model to train whose weights record another split, is
  refused.
- **A fixed validation set.** `materialise` draws the validation batches once,
  so every epoch, and every resumed run, scores the same data. Anything that
  serves the same batches each epoch fits the same place; a dataset that
  addresses its batches by index can replace it.
"""

from __future__ import annotations

import json
import os
import random
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import keras
import numpy as np

import gravyflow as gf
from gravyflow.src.validate.bank import per_example
from gravyflow.src.validate.config import require_same_split, run_record_of, split_record, with_run_record
from gravyflow.src.validate.provenance import (
    StaleCheckpointError, canonical, config_fingerprint, model_fingerprint, require_match,
)

SPLITS = ("train", "validate", "test")

#: The name Keras gives the state variable of each seed generator (dropout's).
_SEED_STATE = "seed_generator_state"


@contextmanager
def keras_draws(seed: int):
    """
    Give Keras' unseeded draws ``seed``, leaving the global random state alone.

    Keras seeds every layer built without a seed (its initialisers, a dropout
    layer's seed generator) from Python's global random, when the layer is
    built or loaded. Within this context those draws come from ``seed``, and
    Python's random state is restored afterwards: a model built here is the
    same for the same seed, and building or loading one changes no global
    random state.
    """
    state = random.getstate()
    random.seed(seed)
    try:
        yield
    finally:
        random.setstate(state)


def _seed_generators(model: keras.Model) -> list:
    # In model order; their names are numbered per session, so order identifies them.
    return [v for v in model.variables if v.name == _SEED_STATE]


def _seed_states(model: keras.Model) -> List[List[int]]:
    """The state of each seed generator in ``model`` (dropout's), in model order."""
    return [np.asarray(v.value).tolist() for v in _seed_generators(model)]


def _restore_seed_states(model: keras.Model, states: List[List[int]], source: Path) -> None:
    variables = _seed_generators(model)
    if len(variables) != len(states):
        raise StaleCheckpointError(
            f"{source} records the state of {len(states)} seed generators, but the model has "
            f"{len(variables)}: it was written by an older recipe and cannot resume exactly.")
    for variable, value in zip(variables, states):
        variable.assign(np.asarray(value, dtype=variable.dtype))


# ------------------------------------------------------------------ presets

@dataclass(frozen=True)
class TrainingSchedule:
    """How long a run trains, and when it stops early."""
    epochs: int
    steps_per_epoch: int
    validation_batches: int
    # Stop after this many epochs without a lower validation loss...
    patience: int
    # ...counted from this epoch on (after a curriculum, for instance).
    monitor_from_epoch: int = 0

    @classmethod
    def cpu(cls) -> "TrainingSchedule":
        """About 3000 steps: minutes on a laptop CPU at 2048 Hz."""
        return cls(epochs=10, steps_per_epoch=300, validation_batches=32, patience=4)

    @classmethod
    def gpu(cls) -> "TrainingSchedule":
        """The papers' scale: 360,000 examples per epoch, until convergence."""
        return cls(epochs=1000, steps_per_epoch=11250, validation_batches=625, patience=5)


# ---------------------------------------------------------------- batches

def _float32(features: Dict[str, Any]) -> Dict[str, np.ndarray]:
    return {k: np.asarray(v, dtype=np.float32) for k, v in features.items()}


class _DatasetBatches(keras.utils.PyDataset):
    """
    A gravyflow dataset's batches, converted for Keras: their number, epoch and
    split (``split_spec``, ``group``) are the dataset's.
    """

    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    @property
    def split_spec(self) -> gf.SplitSpec:
        return self.dataset.split_spec

    @property
    def group(self) -> str:
        return self.dataset.group

    def epoch_callback(self) -> keras.callbacks.Callback:
        return self.dataset.epoch_callback()

    def on_epoch_begin(self):
        self.dataset.on_epoch_begin()

    def on_epoch_end(self):
        self.dataset.on_epoch_end()


class DetectionBatches(_DatasetBatches):
    """A gravyflow dataset as (float32 inputs, one-hot [noise, signal] labels)."""

    def __getitem__(self, index):
        x, y = self.dataset[index]
        present = np.asarray(y[gf.ReturnVariables.INJECTION_PRESENT.name]) > 0.5
        return _float32(x), np.eye(2, dtype=np.float32)[present.astype(int)]


class RegressionBatches(_DatasetBatches):
    """A gravyflow dataset as (float32 inputs, (examples, labels) float32 targets)."""

    def __init__(self, dataset, labels: Sequence):
        super().__init__(dataset)
        self.labels = [getattr(label, "name", label) for label in labels]

    def __getitem__(self, index):
        x, y = self.dataset[index]
        x = _float32(x)
        n = next(iter(x.values())).shape[0]
        targets = np.stack([per_example(y[name], n, name) for name in self.labels], axis=-1)
        return x, targets.astype(np.float32)


class FixedBatches(keras.utils.PyDataset):
    """Batches held in memory, served identically every epoch, with the split they were drawn from."""

    def __init__(self, batches: List, split_spec: Optional[gf.SplitSpec], group: Optional[str]):
        super().__init__()
        self.batches = batches
        self.split_spec, self.group = split_spec, group

    def __len__(self) -> int:
        return len(self.batches)

    def __getitem__(self, index):
        return self.batches[index]


def materialise(batches, num_batches: int) -> FixedBatches:
    """
    Draw ``num_batches`` batches once: a fixed validation set, naming the split
    of ``batches`` (None for batches that name none, which a run refuses).
    """
    return FixedBatches([batches[k] for k in range(num_batches)],
                        getattr(batches, "split_spec", None), getattr(batches, "group", None))


# ---------------------------------------------------------- run directory

class RunDirectory:
    """
    Where one training run keeps its checkpoints, state and final model.

    Layout: ``run.json`` (the configuration and its sha256), ``state.json``,
    ``checkpoints/{last,best}.keras`` with their ``.json`` provenance, and
    ``model.keras`` with ``model.json`` once training ends.
    """

    def __init__(self, path):
        path = Path(path).expanduser().resolve()
        package = Path(gf.__file__).resolve().parent
        if path == package or package in path.parents:
            raise ValueError(f"A run directory must be outside the gravyflow package ({package}); got {path}.")
        self.path = path
        self.checkpoints = path / "checkpoints"

    # ---- json files, written atomically

    def _read(self, path: Path) -> Optional[dict]:
        return json.loads(path.read_text()) if path.exists() else None

    def _write(self, path: Path, data: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True))
        os.replace(tmp, path)

    @property
    def run(self) -> Optional[dict]:
        return self._read(self.path / "run.json")

    @property
    def state(self) -> Optional[dict]:
        return self._read(self.path / "state.json")

    def start(self, description: Dict[str, Any], split: Dict[str, str]) -> str:
        """
        Record a new run's configuration and its data's split (`split_record`);
        returns the configuration's sha256.
        """
        sha = config_fingerprint(description)
        self._write(self.path / "run.json",
                    {"config_sha256": sha, "description": canonical(description), "split": split})
        return sha

    def write_state(self, state: Dict[str, Any]) -> None:
        self._write(self.path / "state.json", state)

    # ---- models with provenance

    def save_model(self, model: keras.Model, name: str, config_sha256: str, **record) -> None:
        """Save ``name``.keras (with optimiser state) and its provenance, atomically."""
        target = self.checkpoints / f"{name}.keras" if name in ("last", "best") else self.path / f"{name}.keras"
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(f"{target.stem}.tmp.keras")
        model.save(tmp)
        os.replace(tmp, target)
        # Keras saves no seed generator state, so a reloaded dropout layer would
        # draw new masks: the record keeps it, and loading restores it.
        self._write(target.with_suffix(".json"), {
            "weights_sha256": model_fingerprint(model), "config_sha256": config_sha256,
            "seed_states": _seed_states(model), **record})

    def load_model(self, name: str, config_sha256: Optional[str] = None) -> keras.Model:
        """
        Load a saved model and verify its provenance: its file must still hold
        the recorded weights and, given ``config_sha256``, it must belong to
        that configuration. Its seed generators resume from their saved state,
        and it carries the run's record (``run.json``; an empty one when the
        directory has none), which the validator reads.

        Raises:
            StaleCheckpointError: a mismatch.
        """
        target = self.checkpoints / f"{name}.keras" if name in ("last", "best") else self.path / f"{name}.keras"
        record = self._read(target.with_suffix(".json"))
        if record is None or not target.exists():
            raise FileNotFoundError(f"No {name} model with provenance in {self.path}.")
        if config_sha256 is not None:
            require_match("run configuration", record["config_sha256"], config_sha256, target)
        with keras_draws(0):  # what a load draws is replaced by the saved state
            model = keras.models.load_model(target)
        require_match("model", record["weights_sha256"], model_fingerprint(model), target)
        _restore_seed_states(model, record.get("seed_states", []), target)
        return with_run_record(model, self.run or {})


# --------------------------------------------------------------- training

class _EpochCheckpoint(keras.callbacks.Callback):
    """Saves the last and best models and the run state at the end of each epoch."""

    def __init__(self, directory: RunDirectory, config_sha256: str, schedule: TrainingSchedule,
                 state: Dict[str, Any]):
        super().__init__()
        self.directory, self.config_sha256 = directory, config_sha256
        self.schedule, self.state = schedule, state

    def on_epoch_end(self, epoch, logs=None):
        logs = {k: float(v) for k, v in (logs or {}).items()}
        state = self.state
        state["history"].append({"epoch": epoch, **logs})
        val_loss = logs.get("val_loss")
        if val_loss is None:
            raise ValueError("The run recipe monitors val_loss: pass validation data.")
        if state["best_val_loss"] is None or val_loss < state["best_val_loss"]:
            state.update(best_val_loss=val_loss, best_epoch=epoch, wait=0)
            self.directory.save_model(self.model, "best", self.config_sha256, epoch=epoch, val_loss=val_loss)
        elif epoch >= self.schedule.monitor_from_epoch:
            state["wait"] += 1
        state["stopped_early"] = state["wait"] >= self.schedule.patience
        self.directory.save_model(self.model, "last", self.config_sha256, epoch=epoch, val_loss=val_loss)
        state["epoch"] = epoch
        self.directory.write_state(state)  # last: the state never points at a missing checkpoint
        if state["stopped_early"]:
            self.model.stop_training = True


@dataclass
class TrainingRun:
    """
    One training run of an example model, in a run directory.

    ``description`` is everything that defines the run (model and data
    configuration, schedule, seed); its sha256 is the configuration a
    checkpoint must match to be resumed. Raising ``schedule.epochs`` extends a
    finished run.

    The run's split is its data's, never declared beside it: ``train`` reads
    it from the training and validation data (`split_record`) and records it
    in ``run.json``, which every model of the run carries, so a validator never
    reports on data that trained or selected the model.
    """
    directory: RunDirectory
    schedule: TrainingSchedule
    seed: int
    description: Dict[str, Any] = field(default_factory=dict)

    @property
    def config_sha256(self) -> str:
        return config_fingerprint(self._full_description())

    @property
    def model_seed(self) -> int:
        """The seed Keras builds the run's model from (`keras_draws`): a stream of the seed tree."""
        return int(gf.SeedTree(self.seed).stream("train", "model/build").integers(2**63))

    def _full_description(self) -> Dict[str, Any]:
        # The epoch count is where the run stops, not what it does: leaving it
        # out lets a finished run be extended.
        schedule = {k: v for k, v in asdict(self.schedule).items() if k != "epochs"}
        return {**self.description, "schedule": schedule, "seed": self.seed}

    def train(
        self,
        build_model: Callable[[], keras.Model],
        training_data: Callable[[int], keras.utils.PyDataset],
        validation_data: keras.utils.PyDataset,
        verbose: int = 2,
    ) -> keras.Model:
        """
        Train, or resume the run in ``directory``, and return the best model
        (also saved as ``model.keras`` with its provenance), carrying the run's
        record.

        Args:
            build_model: a new compiled model. It is built under `keras_draws`,
                so its initial weights and dropout seeds follow from the run's
                seed. Weights that already carry a record (another run's, say)
                keep to its split.
            training_data: the training batches, given the run's root seed:
                the training split of gravyflow data (`DetectionBatches`, say).
                Its ``epoch_callback()``, when it has one, gives it Keras'
                epoch: a resumed run draws its first epoch's batches, and a
                curriculum reads that epoch.
            validation_data: fixed validation batches (see `materialise`),
                of the same split spec.

        Raises:
            ValueError: data that names no split, or splits disagreeing
                (`split_record`).
            StaleCheckpointError: the directory holds a run of another
                configuration, or of data split otherwise; or the model built
                records another split.
        """
        data = training_data(self.seed)
        split = split_record(data, validation_data)
        run = self.directory.run
        sha = self.config_sha256
        state = self.directory.state
        if run is not None:
            require_match("run configuration", run["config_sha256"], sha, self.directory.path)
            require_same_split(run.get("split"), split, self.directory.path)
        if state is None:
            with keras_draws(self.model_seed):
                model = build_model()
            given = run_record_of(model)
            if given is not None:
                require_same_split(given.get("split"), split, "The model build_model returned")
            self.directory.start(self._full_description(), split)
            state = {"epoch": -1, "best_val_loss": None, "best_epoch": None, "wait": 0,
                     "stopped_early": False, "history": []}
        else:
            model = self.directory.load_model("last", sha)
        first_epoch = state["epoch"] + 1
        if first_epoch < self.schedule.epochs and not state["stopped_early"]:
            callbacks = [_EpochCheckpoint(self.directory, sha, self.schedule, state)]
            if hasattr(data, "epoch_callback"):
                callbacks.insert(0, data.epoch_callback())
            model.fit(
                data,
                validation_data=validation_data,
                epochs=self.schedule.epochs,
                initial_epoch=first_epoch,
                callbacks=callbacks,
                shuffle=False,  # the dataset owns the order (see the module docstring)
                verbose=verbose,
            )
        best = self.directory.load_model("best", sha)
        state = self.directory.state
        self.directory.save_model(best, "model", sha, epoch=state["best_epoch"],
                                  val_loss=state["best_val_loss"])
        return best


__all__ = [
    "DetectionBatches", "FixedBatches", "RegressionBatches", "RunDirectory", "StaleCheckpointError",
    "TrainingRun", "TrainingSchedule", "keras_draws", "materialise",
]
