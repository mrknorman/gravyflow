"""
Evaluation wiring: the data the validator scores never meets the data a model
trains or is selected on, through the shipped scripts' own build functions
(Gabbard et al. 2017, and both phases of George & Huerta 2017 over their SNR
curriculum); and the validator draws from the test split unless told otherwise.

Every dataset those functions build is tapped (`tap.DatasetTap`), so the
strain compared is what each dataset served: its whitened windows byte for
byte, and its noise, the whitened onsource less its whitened injections, by
cross-correlation, which sees noise shared under any added signal. Every
injection label stream is compared too, at every offset; and a run resumed
part-way trains on the batches the uninterrupted run did.
"""

from unittest import mock

import keras
import numpy as np
import pytest

import gravyflow as gf
from gravyflow.src.model.examples import train_gabbard_2017 as train_script
from gravyflow.src.model.examples import train_george_huerta_2017 as george_huerta_script
from gravyflow.src.model.examples import validate_gabbard_2017 as validate_script
from gravyflow.src.model.examples import recipe
from gravyflow.src.model.examples.recipe import RunDirectory
from gravyflow.src.validate import ValidationConfig, Validator
from gravyflow.src.validate.bank import ValidationBank
from gravyflow.tests.leakage.fingerprints import (
    assert_noise_independent, assert_rows_disjoint, assert_streams_independent, strain_prints,
)
from gravyflow.tests.leakage.tap import (
    INJECTION_LABELS, STRAIN, DatasetTap, EpochWindows, injection_label_streams, whitened, whitened_noise,
)

MODEL_INPUT = gf.ReturnVariables.WHITENED_ONSOURCE.name
EPOCHS, STEPS, BATCH = 2, 2, 32
# George & Huerta's curriculum needs a run of at least four epochs.
CURRICULUM_EPOCHS = 4
VALIDATOR_STAGES = ("validator noise", "validator injections")


def stub_model(samples, task="classification"):
    """A fixed linear model on the whitened onsource, two logits or two masses out; its weights never matter."""
    model = keras.Sequential([
        keras.Input(shape=(1, samples), name=MODEL_INPUT),
        keras.layers.Flatten(),
        keras.layers.Dense(2, kernel_initializer="zeros", bias_initializer="zeros"),
    ])
    loss = keras.losses.CategoricalCrossentropy(from_logits=True) if task == "classification" else "mae"
    model.compile(optimizer="sgd", loss=loss)
    return model


def samples_of(config):
    return int(config["sample_rate_hertz"] * config["onsource_duration_seconds"])


def train(directory, epochs, steps=STEPS, validation_batches=STEPS):
    """Train (or resume) the shipped Gabbard 2017 run in ``directory`` on its own data."""
    args = train_script.parse_args([
        "--output-dir", str(directory), "--epochs", str(epochs), "--steps-per-epoch", str(steps),
        "--validation-batches", str(validation_batches), "--verbose", "0",
    ])
    config, run = train_script.build_run(args)
    assert config.batch_size == BATCH, "the cpu preset's batch size changed; resize the check"
    training_data, validation = train_script.build_datasets(config, run)
    run.train(build_model=lambda: stub_model(samples_of(vars(config))), training_data=training_data,
              validation_data=validation, verbose=0)
    return run


def train_george_huerta(directory, task, steps=1, validation_batches=2):
    """
    One phase of the shipped George & Huerta 2017 run (the shallow network,
    on the cpu preset, over the SNR curriculum), in ``directory/task``.
    """
    args = george_huerta_script.parse_args([
        "--output-dir", str(directory), "--model", "shallow", "--task", task,
        "--epochs", str(CURRICULUM_EPOCHS), "--steps-per-epoch", str(steps),
        "--validation-batches", str(validation_batches), "--verbose", "0",
    ])
    model_class = george_huerta_script.MODELS[args.model]
    config, schedule = george_huerta_script.configure(args)
    assert config.use_curriculum and config.batch_size == BATCH
    run = george_huerta_script.build_run(model_class, task, config, schedule, args)
    training_data, validation = george_huerta_script.build_datasets(model_class, task, config, run)
    run.train(build_model=lambda: stub_model(samples_of(vars(config)), task), training_data=training_data,
              validation_data=validation, verbose=0)
    return run


def validate(run_directory, num_examples):
    """Score the shipped validation script's sets for the run in ``run_directory``, with a stub model."""
    args = validate_script.parse_args([
        "--run-dir", str(run_directory), "--num-examples", str(num_examples), "--batch-size", str(BATCH),
    ])
    run_record = RunDirectory(run_directory).run
    dataset_args, config = validate_script.build_validation(args, run_record)
    model = stub_model(samples_of(run_record["description"]["config"]))
    return Validator.validate(model=model, name="leakage", dataset_args=dataset_args, config=config)


def by_split(taken, stages=(), batches_per_stage=None, epochs=False):
    """
    The tapped datasets' batches by split: train and validate each joined over
    their datasets, then the validator's stages, which the caller names in the
    order the validator builds them. Every stage named must have been built,
    and nothing more in the test split: a stage that escaped the tap would
    otherwise drop out of the comparison unseen. With ``epochs``, training is
    one set per epoch, ``train epoch e``.
    """
    sets, expected = {}, list(stages)
    for record in taken:
        if record.group == "test":
            if not expected:
                raise AssertionError(f"a test-split dataset was built beyond the stages {list(stages)}")
            name = expected.pop(0)
            if batches_per_stage is not None:
                assert len(record.batches) == batches_per_stage, f"{name} served {len(record.batches)} batches"
        elif record.group == "train" and epochs:
            if None in record.addresses:
                raise AssertionError("the training data is not index-addressed, so it has no epochs")
            for (epoch, _), batch in zip(record.addresses, record.batches):
                sets.setdefault(f"train epoch {epoch}", []).append(batch)
            continue
        else:
            name = record.group
        sets.setdefault(name, []).extend(record.batches)
    if expected:
        raise AssertionError(f"the tap saw no dataset for the stages {expected}")
    return sets


def assert_strain_disjoint(sets):
    """No two sets of served batches share a whitened window, or their noise."""
    assert_rows_disjoint({name: strain_prints({MODEL_INPUT: whitened(b)}) for name, b in sets.items()})
    # Shared noise under an added signal is never byte-equal, and a loud signal
    # correlates with another on its own: the noise alone is cross-correlated.
    assert_noise_independent({name: whitened_noise(b) for name, b in sets.items()})


@pytest.mark.global_rng
def test_validator_sets_disjoint(tmp_path, monkeypatch):
    """
    Every training batch of two epochs, the fixed validation set, and the
    validator's noise background and injection set, all built by the shipped
    Gabbard 2017 scripts from their default seeds, share no strain.
    """
    tap = DatasetTap(monkeypatch, STRAIN)
    train(tmp_path / "run", EPOCHS)
    validate(tmp_path / "run", STEPS * BATCH)
    sets = by_split(tap.take(), VALIDATOR_STAGES, batches_per_stage=STEPS)
    training = sets.pop("train")
    assert len(training) >= EPOCHS * STEPS, f"{len(training)} distinct training batches for {EPOCHS} epochs"
    sets.update({f"train batch {k}": [batch] for k, batch in enumerate(training)})
    assert_strain_disjoint(sets)


@pytest.mark.global_rng
def test_george_huerta_sets_disjoint(tmp_path, monkeypatch):
    """
    Both phases of the shipped George & Huerta 2017 run, the mass regression
    and then the detector, each trained over the SNR curriculum, with their
    fixed validation sets (the regression's carrying the mass targets), and
    the validator's sets for the detector, share no strain: within a phase,
    between every training batch and the validation set; across the phases
    and the validator, between splits.

    The two phases draw from one root, so a split's noise is common to both
    (common random numbers, as ruled) and is not compared phase against phase.
    """
    tap = DatasetTap(monkeypatch, STRAIN)
    splits = {}
    for task in ("regression", "classification"):
        train_george_huerta(tmp_path, task)
        phase = by_split(tap.take())
        assert len(phase["train"]) >= CURRICULUM_EPOCHS, f"{task}: {len(phase['train'])} training batches"
        assert_strain_disjoint({"validation": phase["validate"],
                                **{f"train batch {k}": [b] for k, b in enumerate(phase["train"])}})
        for split, batches in phase.items():
            splits.setdefault(split, []).extend(batches)
    validate(tmp_path / "classification", BATCH)
    splits.update(by_split(tap.take(), VALIDATOR_STAGES, batches_per_stage=1))
    assert_strain_disjoint(splits)


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
    bank = ValidationBank(stub_model(256), dataset_args, config)
    bank.generate_noise()
    bank.generate()
    assert groups == ["test", "test"], f"the validator built datasets in {groups}"


# A label stream is compared over the examples present in both sets at an
# offset: with half the examples injected on each side, a quarter of them. The
# sets are therefore larger than the strain tests', so that every pair of
# streams has an offset long enough to test (the check refuses otherwise).
# Gabbard's training epochs are sets of their own, so they are twice as long.
LABEL_STEPS, LABEL_VALIDATION_BATCHES, LABEL_VALIDATOR_BATCHES = 6, 12, 6
EPOCH_LABEL_STEPS = 12


def label_sets(taken_by_split):
    return {name: injection_label_streams(batches) for name, batches in taken_by_split.items()}


@pytest.mark.global_rng
def test_validator_label_streams_independent(tmp_path, monkeypatch):
    """
    No injection label stream (presence, SNR, HRSS, HPEAK, every waveform
    parameter, sky, polarisation, central time) of either epoch of training,
    the fixed validation set or the validator's injection set, all built by
    the shipped Gabbard 2017 scripts, is shared with another's, in value, in
    rank or at any offset.

    Each epoch is its own set: joined, epochs that repeated their draws would
    hide the repeat from the exact-value check, which compares only streams
    whose values are all distinct.
    """
    tap = DatasetTap(monkeypatch, INJECTION_LABELS)
    train(tmp_path / "run", EPOCHS, steps=EPOCH_LABEL_STEPS, validation_batches=LABEL_VALIDATION_BATCHES)
    validate(tmp_path / "run", LABEL_VALIDATOR_BATCHES * BATCH)
    assert_streams_independent(label_sets(by_split(
        tap.take(), VALIDATOR_STAGES, batches_per_stage=LABEL_VALIDATOR_BATCHES, epochs=True)))


@pytest.mark.global_rng
def test_george_huerta_label_streams_independent(tmp_path, monkeypatch):
    """
    As above for both phases of the shipped George & Huerta 2017 run over the
    SNR curriculum: each phase's training (the regression's mass targets among
    its labels) and fixed validation set, and the validator's injection set.
    The phases share one root, so a split is compared with every other split,
    within a phase and across the two, never with itself in the other phase.

    Each phase is its own set: joined, the phases' streams would change
    distribution part-way (the regression injects every example), and two
    such streams correlate at the offset that aligns their changes. A
    phase's epochs are joined, which keeps the comparison's cost bounded;
    the Gabbard test above compares epochs one by one through the same
    library.
    """
    tap = DatasetTap(monkeypatch, INJECTION_LABELS)
    phases = {}
    for task in ("regression", "classification"):
        train_george_huerta(tmp_path, task, steps=LABEL_STEPS, validation_batches=LABEL_VALIDATION_BATCHES)
        phases[task] = {f"{task} {split}": streams for split, streams in label_sets(by_split(tap.take())).items()}
    validate(tmp_path / "classification", LABEL_VALIDATOR_BATCHES * BATCH)
    validator = label_sets(by_split(tap.take(), VALIDATOR_STAGES, batches_per_stage=LABEL_VALIDATOR_BATCHES))
    regression, classification = phases["regression"], phases["classification"]
    for view in ({**regression, **validator}, {**classification, **validator},
                 {"regression train": regression["regression train"],
                  "classification validate": classification["classification validate"]},
                 {"classification train": classification["classification train"],
                  "regression validate": regression["regression validate"]}):
        assert_streams_independent(view)


# Rebuilt on CPU, a batch's whitened strain may differ in the last bits of float32.
REBUILD_RTOL = 1e-5


def training_batches(taken):
    """Every training batch served, by its (epoch, k) address."""
    return {address: batch for record in taken if record.group == "train"
            for address, batch in zip(record.addresses, record.batches)}


def training_epochs(windows):
    """Each Keras epoch's training addresses, (epoch, k), sorted."""
    return {epoch: sorted(serve[1:] for serve in served if serve[0] == "train") for epoch, served in windows.items()}


@pytest.mark.global_rng
def test_resumed_run_trains_on_the_same_epochs(tmp_path, monkeypatch):
    """
    A run stopped after epoch 0 and resumed trains each epoch it resumes on
    the batches the uninterrupted run trained that epoch on, each the same,
    and Keras epoch e of either run on epoch e's batches. The epochs are as
    the run's own checkpoint callback sees them.
    """
    tap = DatasetTap(monkeypatch, ())
    windows = EpochWindows(tap)
    windows.hook(monkeypatch, recipe._EpochCheckpoint)
    train(tmp_path / "run", EPOCHS)
    uninterrupted, (_, uninterrupted_epochs) = training_batches(tap.take()), windows.take()
    train(tmp_path / "resumed", 1)
    tap.take()
    windows.take()
    train(tmp_path / "resumed", EPOCHS)
    resumed, (_, resumed_epochs) = training_batches(tap.take()), windows.take()

    assert None not in uninterrupted, "the training data is not index-addressed"
    expected = {epoch: [(epoch, k) for k in range(STEPS)] for epoch in range(EPOCHS)}
    assert training_epochs(uninterrupted_epochs) == expected
    assert training_epochs(resumed_epochs) == {epoch: expected[epoch] for epoch in range(1, EPOCHS)}, (
        "a run resumed at epoch 1 trained on other batches than the uninterrupted run")
    for address, batch in resumed.items():
        for name, value in batch.items():
            np.testing.assert_allclose(value, uninterrupted[address][name], rtol=REBUILD_RTOL,
                                       atol=REBUILD_RTOL * float(np.max(np.abs(value))),
                                       err_msg=f"{name} of batch {address}")


def _scores_split(group, run_record):
    """Whether the validator goes on to score ``group`` for a model with ``run_record``."""
    class Scored(Exception):
        pass

    def scored(*args, **kwargs):
        raise Scored

    with mock.patch("gravyflow.src.validate.validator.model_fingerprint", scored):
        try:
            Validator.validate(None, "model", {"onsource_duration_seconds": 1.0},
                               ValidationConfig(group=group), run_record=run_record)
        except Scored:
            return True
    return False


def test_validator_refuses_the_selection_split():
    """
    The validator refuses to score the split that selected the model (run.json's
    selection_split), and scores the other held-out split.
    """
    record = {"split": {"fingerprint": gf.DEFAULT_SPLIT.fingerprint(), "selection_split": "validate"}}
    assert _scores_split("test", record)
    with pytest.raises(ValueError, match="selected on the 'validate' split"):
        _scores_split("validate", record)


def test_selection_check_sees_a_validator_that_scores_it(monkeypatch):
    """The planted leak: a validator that ignores the record scores the selection split."""
    monkeypatch.setattr("gravyflow.src.validate.validator.check_held_out", lambda *args: None)
    record = {"split": {"fingerprint": gf.DEFAULT_SPLIT.fingerprint(), "selection_split": "validate"}}
    assert _scores_split("validate", record)
