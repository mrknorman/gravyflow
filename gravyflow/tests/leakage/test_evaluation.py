"""
Evaluation wiring: the data the validator scores never meets the data a model
trains or is selected on, through the shipped scripts' own build functions;
and the validator draws from the test split unless told otherwise.
"""

import hashlib

import keras
import numpy as np

import gravyflow as gf
from gravyflow.src.model.examples import train_gabbard_2017 as train_script
from gravyflow.src.model.examples import validate_gabbard_2017 as validate_script
from gravyflow.src.validate import ValidationConfig, Validator
from gravyflow.src.validate.bank import ValidationBank
from gravyflow.tests.leakage.fingerprints import (
    assert_noise_independent, assert_rows_disjoint, assert_streams_independent, strain_prints,
)

MODEL_INPUT = gf.ReturnVariables.WHITENED_ONSOURCE.name
EPOCHS, STEPS, BATCH = 2, 2, 32          # 64 examples per epoch: enough for label streams


class Recording(keras.utils.PyDataset):
    """Serves another PyDataset's batches unchanged, and keeps each one served."""

    def __init__(self, inner, served):
        super().__init__()
        self.inner, self.served = inner, served

    def __len__(self):
        return len(self.inner)

    def __getitem__(self, index):
        batch = self.inner[index]
        self.served.append(batch)
        return batch

    def on_epoch_end(self):
        self.inner.on_epoch_end()

    def __getattr__(self, name):
        if name == "inner":
            raise AttributeError(name)
        return getattr(self.inner, name)


def logit_model(samples):
    """A fixed two-class model returning logits; its weights never matter here."""
    model = keras.Sequential([
        keras.Input(shape=(1, samples), name=MODEL_INPUT),
        keras.layers.Flatten(),
        keras.layers.Dense(2, kernel_initializer="zeros", bias_initializer="zeros"),
    ])
    model.compile(optimizer="sgd", loss=keras.losses.CategoricalCrossentropy(from_logits=True))
    return model


def train(directory, epochs, served):
    """Train (or resume) the shipped Gabbard 2017 run in ``directory`` on its own data."""
    args = train_script.parse_args([
        "--output-dir", str(directory), "--epochs", str(epochs), "--steps-per-epoch", str(STEPS),
        "--validation-batches", str(STEPS), "--verbose", "0",
    ])
    config, run = train_script.build_run(args)
    assert config.batch_size == BATCH, "the cpu preset's batch size changed; resize the check"
    training_data, validation = train_script.build_datasets(config, run)
    samples = int(config.sample_rate_hertz * config.onsource_duration_seconds)
    run.train(
        build_model=lambda: logit_model(samples),
        training_data=lambda *args: Recording(training_data(*args), served),
        validation_data=validation,
        verbose=0,
    )
    return validation, run


def distinct(served):
    """
    The distinct batches served, in order. Keras may request a batch more than
    once (it peeks before an epoch); a batch that is a pure function of its
    address is then served twice, identically, and counted once.
    """
    seen, batches = set(), []
    for batch in served:
        key = digest([batch])
        if key not in seen:
            seen.add(key)
            batches.append(batch)
    return batches


def digest(batches):
    h = hashlib.sha256()
    for x, y in batches:
        for value in (*[x[k] for k in sorted(x)], y):
            h.update(np.ascontiguousarray(np.asarray(value)).tobytes())
    return h.hexdigest()


def inputs(batches):
    return {MODEL_INPUT: np.concatenate([np.asarray(x[MODEL_INPUT], np.float32) for x, _ in batches])}


def present(batches):
    """The signal-or-noise label of each example: the one-hot target's signal column."""
    return {"INJECTION_PRESENT": np.concatenate([np.asarray(y)[:, 1] for _, y in batches])}


def test_validator_sets_disjoint(tmp_path):
    """
    Every training batch of two epochs, the fixed validation set, and the
    validator's noise background and injection set, all built by the shipped
    scripts from their default seeds, share no strain and no label stream.
    """
    served = []
    validation, _ = train(tmp_path / "run", EPOCHS, served)
    training = distinct(served)
    assert len(training) >= EPOCHS * STEPS, f"{len(training)} distinct training batches for {EPOCHS} epochs"

    args = validate_script.parse_args([
        "--run-dir", str(tmp_path / "run"), "--num-examples", str(STEPS * BATCH),
        "--batch-size", str(BATCH),
    ])
    run_record = train_script.RunDirectory(tmp_path / "run").run
    dataset_args, config = validate_script.build_validation(args, run_record)
    model = logit_model(int(run_record["description"]["config"]["sample_rate_hertz"]))
    scored = []
    score = model.predict_on_batch

    def recording_score(x):
        scored.append(np.asarray(x[MODEL_INPUT], np.float32))
        return score(x)

    model.predict_on_batch = recording_score
    validator = Validator.validate(model=model, name="leakage", dataset_args=dataset_args, config=config)
    assert len(scored) == 2 * STEPS, f"the validator scored {len(scored)} batches, expected {2 * STEPS}"

    whitened = {f"train batch {k}": inputs([batch]) for k, batch in enumerate(training)}
    whitened.update({
        "validation": inputs(validation.batches),
        "validator noise": {MODEL_INPUT: np.concatenate(scored[:STEPS])},
        "validator injections": {MODEL_INPUT: np.concatenate(scored[STEPS:])},
    })
    assert_rows_disjoint({name: strain_prints(arrays) for name, arrays in whitened.items()})
    # The injection set adds a signal to its noise, so shared noise there is
    # never byte-equal: it shows as correlation instead.
    assert_noise_independent({name: arrays[MODEL_INPUT] for name, arrays in whitened.items()})

    efficiency = validator.efficiency_data
    assert_streams_independent({
        "train": present(training),
        "validation": present(validation.batches),
        "validator injections": {key: np.asarray(efficiency[key]) for key in ("snrs", "hrss", "hpeak",
                                                                              "mass1", "mass2", "central_times")},
    })


def test_validator_default_is_test(monkeypatch):
    """
    By default the validator scores the test split: its configuration, the
    shipped validation script's, and every dataset it builds.
    """
    assert ValidationConfig().group == "test"
    run_record = {"description": {"config": {
        "sample_rate_hertz": 256.0, "onsource_duration_seconds": 1.0, "offsource_duration_seconds": 4.0}}}
    dataset_args, config = validate_script.build_validation(
        validate_script.parse_args(["--run-dir", "unused", "--num-examples", "8", "--batch-size", "8"]),
        run_record)
    assert config.group == "test"

    groups = []
    build = gf.Dataset

    def spy(**kwargs):
        groups.append(kwargs.get("group"))
        return build(**kwargs)

    monkeypatch.setattr(gf, "Dataset", spy)
    bank = ValidationBank(logit_model(256), dataset_args, config)
    bank.generate_noise()
    bank.generate()
    assert groups == ["test", "test"], f"the validator built datasets in {groups}"
