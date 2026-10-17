"""
Evaluation wiring: the data the validator scores never meets the data a model
trains or is selected on, through the shipped scripts' own build functions
(Gabbard et al. 2017, and both phases of George & Huerta 2017 over their SNR
curriculum); and the validator draws from the test split unless told otherwise.

Every dataset those functions build is tapped (`tap.DatasetTap`), so the
strain compared is what each dataset served: its whitened windows byte for
byte, and its noise, the whitened onsource less its whitened injections, by
cross-correlation, which sees noise shared under any added signal.
"""

import keras
import pytest

import gravyflow as gf
from gravyflow.src.model.examples import train_gabbard_2017 as train_script
from gravyflow.src.model.examples import train_george_huerta_2017 as george_huerta_script
from gravyflow.src.model.examples import validate_gabbard_2017 as validate_script
from gravyflow.src.model.examples.recipe import RunDirectory
from gravyflow.src.validate import ValidationConfig, Validator
from gravyflow.src.validate.bank import ValidationBank
from gravyflow.tests.leakage.fingerprints import assert_noise_independent, assert_rows_disjoint, strain_prints
from gravyflow.tests.leakage.tap import STRAIN, DatasetTap, whitened, whitened_noise

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


def by_split(taken, stages=(), batches_per_stage=None):
    """
    The tapped datasets' batches by split: train and validate each joined over
    their datasets, then the validator's stages, which the caller names in the
    order the validator builds them. Every stage named must have been built,
    and nothing more in the test split: a stage that escaped the tap would
    otherwise drop out of the comparison unseen.
    """
    sets, expected = {}, list(stages)
    for record in taken:
        if record.group == "test":
            if not expected:
                raise AssertionError(f"a test-split dataset was built beyond the stages {list(stages)}")
            name = expected.pop(0)
            if batches_per_stage is not None:
                assert len(record.batches) == batches_per_stage, f"{name} served {len(record.batches)} batches"
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
