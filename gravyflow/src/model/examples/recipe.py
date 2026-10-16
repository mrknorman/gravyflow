"""
The run recipe of the example models: how a training run is laid out, resumed
and evaluated.

- **Presets.** `TrainingSchedule.cpu` sizes a run for a laptop CPU;
  `TrainingSchedule.gpu` is the scale the papers used.
- **Seeds.** One root seed gives each split its own seed (`split_seeds`), so the
  training, validation and test data never share a stream.
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
- **A fixed validation set.** `materialise` draws the validation batches once,
  so every epoch, and every resumed run, scores the same data. Anything that
  serves the same batches each epoch fits the same place; a dataset that
  addresses its batches by index can replace it.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import keras
import numpy as np

import gravyflow as gf
from gravyflow.src.validate.bank import derived_seed, per_example
from gravyflow.src.validate.provenance import (
    StaleCheckpointError, canonical, config_fingerprint, model_fingerprint, require_match,
)

SPLITS = ("train", "validate", "test")


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


def split_seeds(root: int) -> Dict[str, int]:
    """A seed for each split, derived from one root."""
    return {split: derived_seed(root, "split", split) for split in SPLITS}


# ---------------------------------------------------------------- batches

def _float32(features: Dict[str, Any]) -> Dict[str, np.ndarray]:
    return {k: np.asarray(v, dtype=np.float32) for k, v in features.items()}


def injection_present(masks, batch_size: int) -> np.ndarray:
    """
    Whether each example holds an injection, from ``INJECTION_MASKS`` shaped
    (examples,) or (generators, examples): present if any generator injected.
    """
    m = np.squeeze(np.asarray(masks))
    if m.shape == (batch_size,) or (batch_size == 1 and m.ndim == 0):
        return m.reshape(batch_size) > 0.5
    if m.ndim == 2 and m.shape[1] == batch_size:
        return np.max(m, axis=0) > 0.5
    raise ValueError(f"INJECTION_MASKS of shape {np.shape(masks)} for a batch of {batch_size}.")


class DetectionBatches(keras.utils.PyDataset):
    """A gravyflow dataset as (float32 inputs, one-hot [noise, signal] labels)."""

    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index):
        x, y = self.dataset[index]
        x = _float32(x)
        n = next(iter(x.values())).shape[0]
        present = injection_present(y[gf.ReturnVariables.INJECTION_MASKS.name], n)
        return x, np.eye(2, dtype=np.float32)[present.astype(int)]


class RegressionBatches(keras.utils.PyDataset):
    """A gravyflow dataset as (float32 inputs, (examples, labels) float32 targets)."""

    def __init__(self, dataset, labels: Sequence):
        super().__init__()
        self.dataset = dataset
        self.labels = [getattr(label, "name", label) for label in labels]

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index):
        x, y = self.dataset[index]
        x = _float32(x)
        n = next(iter(x.values())).shape[0]
        targets = np.stack([per_example(y[name], n, name) for name in self.labels], axis=-1)
        return x, targets.astype(np.float32)


class FixedBatches(keras.utils.PyDataset):
    """Batches held in memory, served identically every epoch."""

    def __init__(self, batches: List):
        super().__init__()
        self.batches = batches

    def __len__(self) -> int:
        return len(self.batches)

    def __getitem__(self, index):
        return self.batches[index]


def materialise(batches, num_batches: int) -> FixedBatches:
    """Draw ``num_batches`` batches once: a fixed validation set."""
    return FixedBatches([batches[k] for k in range(num_batches)])


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

    def start(self, description: Dict[str, Any]) -> str:
        """Record a new run's configuration; returns its sha256."""
        sha = config_fingerprint(description)
        self._write(self.path / "run.json", {"config_sha256": sha, "description": canonical(description)})
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
        self._write(target.with_suffix(".json"), {
            "weights_sha256": model_fingerprint(model), "config_sha256": config_sha256, **record})

    def load_model(self, name: str, config_sha256: Optional[str] = None) -> keras.Model:
        """
        Load a saved model and verify its provenance: its file must still hold
        the recorded weights and, given ``config_sha256``, it must belong to
        that configuration.

        Raises:
            StaleCheckpointError: a mismatch.
        """
        target = self.checkpoints / f"{name}.keras" if name in ("last", "best") else self.path / f"{name}.keras"
        record = self._read(target.with_suffix(".json"))
        if record is None or not target.exists():
            raise FileNotFoundError(f"No {name} model with provenance in {self.path}.")
        if config_sha256 is not None:
            require_match("run configuration", record["config_sha256"], config_sha256, target)
        model = keras.models.load_model(target)
        require_match("model", record["weights_sha256"], model_fingerprint(model), target)
        return model


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
    """
    directory: RunDirectory
    schedule: TrainingSchedule
    seed: int
    description: Dict[str, Any] = field(default_factory=dict)

    @property
    def seeds(self) -> Dict[str, int]:
        return split_seeds(self.seed)

    @property
    def config_sha256(self) -> str:
        return config_fingerprint(self._full_description())

    def _full_description(self) -> Dict[str, Any]:
        # The epoch count is where the run stops, not what it does: leaving it
        # out lets a finished run be extended.
        schedule = {k: v for k, v in asdict(self.schedule).items() if k != "epochs"}
        return {**self.description, "schedule": schedule, "seed": self.seed}

    def train_seed(self, first_epoch: int) -> int:
        """
        The training data's seed for a run (re)started at ``first_epoch``: a
        resumed run draws new data rather than replaying the first epochs.
        """
        return derived_seed(self.seeds["train"], "from-epoch", str(first_epoch))

    def train(
        self,
        build_model: Callable[[], keras.Model],
        training_data: Callable[[int, int], keras.utils.PyDataset],
        validation_data: keras.utils.PyDataset,
        verbose: int = 2,
    ) -> keras.Model:
        """
        Train, or resume the run in ``directory``, and return the best model
        (also saved as ``model.keras`` with its provenance).

        Args:
            build_model: a new compiled model.
            training_data: the training batches, given a seed (see `train_seed`) and
                the first epoch they serve (for a schedule, such as a curriculum,
                that depends on the epoch).
            validation_data: fixed validation batches (see `materialise`).

        Raises:
            StaleCheckpointError: the directory holds a run of another configuration.
        """
        run = self.directory.run
        sha = self.config_sha256
        state = self.directory.state
        if run is not None:
            require_match("run configuration", run["config_sha256"], sha, self.directory.path)
        if state is None:
            self.directory.start(self._full_description())
            model = build_model()
            state = {"epoch": -1, "best_val_loss": None, "best_epoch": None, "wait": 0,
                     "stopped_early": False, "history": []}
        else:
            model = self.directory.load_model("last", sha)
        first_epoch = state["epoch"] + 1
        if first_epoch < self.schedule.epochs and not state["stopped_early"]:
            model.fit(
                training_data(self.train_seed(first_epoch), first_epoch),
                validation_data=validation_data,
                epochs=self.schedule.epochs,
                initial_epoch=first_epoch,
                callbacks=[_EpochCheckpoint(self.directory, sha, self.schedule, state)],
                verbose=verbose,
            )
        best = self.directory.load_model("best", sha)
        state = self.directory.state
        self.directory.save_model(best, "model", sha, epoch=state["best_epoch"],
                                  val_loss=state["best_val_loss"])
        return best


__all__ = [
    "DetectionBatches", "FixedBatches", "RegressionBatches", "RunDirectory", "StaleCheckpointError",
    "TrainingRun", "TrainingSchedule", "injection_present", "materialise", "split_seeds",
]
