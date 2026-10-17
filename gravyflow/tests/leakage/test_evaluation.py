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
from gravyflow.src.dataset.dataset import split_spec_of
from gravyflow.src.model.examples import train_gabbard_2017 as train_script
from gravyflow.src.model.examples import train_george_huerta_2017 as george_huerta_script
from gravyflow.src.model.examples import validate_gabbard_2017 as validate_script
from gravyflow.src.model.examples import recipe
from gravyflow.src.model.examples.recipe import RunDirectory
from gravyflow.src.validate import StaleCheckpointError, ValidationConfig, Validator
from gravyflow.src.validate.bank import ValidationBank
from gravyflow.src.validate.config import run_record_of, with_run_record
from gravyflow.tests.leakage import fixtures as fx
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
    model = with_run_record(stub_model(samples_of(run_record["description"]["config"])), run_record)
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


class _Scored(Exception):
    """The validator went on to score: past its held-out checks."""


def _scores(call):
    """Whether ``call`` reaches the validator's scoring, which it stops there."""
    def scored(*args, **kwargs):
        raise _Scored

    with mock.patch("gravyflow.src.validate.validator.model_fingerprint", scored):
        try:
            call()
        except _Scored:
            return True
    return False


def _scores_split(group, run_record, dataset_args=None, **config):
    """
    Whether the validator goes on to score ``group`` for a model whose weights carry
    ``run_record``, with the validation settings ``config``.
    """
    dataset_args = dataset_args or {"noise_obtainer": gf.NoiseObtainer(noise_type=gf.NoiseType.WHITE),
                                    "onsource_duration_seconds": 1.0,
                                    "input_variables": [gf.ReturnVariables.WHITENED_ONSOURCE]}
    model = with_run_record(stub_model(256), run_record)
    return _scores(lambda: Validator.validate(model, "model", dataset_args, ValidationConfig(group=group, **config)))


FOLD_2 = gf.SplitSpec(version="fold-2")


class StandingBatches(recipe.DetectionBatches):
    """
    A gravyflow dataset's batches as the recipe wraps them, served as small arrays: a
    run reads the dataset's split, never its strain.
    """

    def __getitem__(self, index):
        rng = np.random.default_rng(index)
        x = rng.normal(size=(4, 1, 16)).astype(np.float32)
        return {"X": x}, np.eye(2, dtype=np.float32)[rng.integers(0, 2, 4)]


def tiny_model():
    inputs = keras.Input(shape=(1, 16), name="X")
    model = keras.Model({"X": inputs}, keras.layers.Dense(2)(keras.layers.Flatten()(inputs)))
    model.compile(optimizer="sgd", loss=keras.losses.CategoricalCrossentropy(from_logits=True))
    return model


def _split_world(specs):
    """O3 data inside one block of each spec's train, validate and test splits, away from its edges."""
    first = gf.DEFAULT_SPLIT.block_of(fx.origin_gps("O3")) + 2
    blocks = set()
    for spec in specs:
        for split in ("train", "validate", "test"):
            blocks.add(next(k for k in range(first, first + 1000) if spec.block_split(k) == split))
    intervals = [(k * fx.BLOCK_SECONDS + 1024.0, k * fx.BLOCK_SECONDS + 4096.0) for k in sorted(blocks)]
    return fx._fixture(intervals, 256.0, "O3", ("O3",))


@pytest.mark.parametrize("split", ["validate", "test"])
def test_a_grid_evaluation_set_is_finite_and_the_same_every_epoch(world, split):
    """
    L12 (GRID): a GRID evaluation set is floor(N / B) batches of its N windows, each
    window served at most once, and every epoch serves the same batches, in whatever
    order they are asked for.
    """
    fixture = _split_world([gf.DEFAULT_SPLIT])
    with world(fixture):
        ds = fx.dataset(fixture, split, mode=gf.SamplingMode.GRID, batch=8)
        windows = len(ds.noise.plan.starts)
        assert len(ds) == windows // 8 > 0, f"{len(ds)} batches of {windows} windows"

        def starts(order):
            return {k: np.asarray(ds[k][1][gf.ReturnVariables.START_GPS_TIME.name]) for k in order}

        first = starts(range(len(ds)))
        ds.on_epoch_end()
        ds.epoch = 2
        later = starts(reversed(range(len(ds))))
    for k in range(len(ds)):
        np.testing.assert_array_equal(later[k], first[k], err_msg=f"batch {k} changed between epochs")
    served = np.concatenate([first[k] for k in range(len(ds))])
    for d in range(served.shape[1]):
        assert len(np.unique(served[:, d])) == len(served), f"detector {d} served a window twice"


def test_a_grid_evaluation_set_is_prepared_once(world):
    """``prepare`` commits every tile a GRID set reads, once for all epochs; its batches then fetch nothing."""
    fixture = _split_world([gf.DEFAULT_SPLIT])
    with world(fixture) as harness:
        ds = fx.dataset(fixture, "test", mode=gf.SamplingMode.GRID, batch=8)
        assert ds.prepare(epochs=3) == len(ds) > 0
        prepared = len(harness.source.snapshot()[0])
        assert prepared > 0
        for k in range(len(ds)):
            ds[k]
        assert len(harness.source.snapshot()[0]) == prepared, "a prepared GRID batch fetched"


def test_a_run_records_the_split_its_data_reads(world, tmp_path):
    """
    A run records the split its data reads, never one declared beside it: a default run
    on fold-2 data records fold-2's fingerprint (in run.json, which its models carry), so
    the validator refuses the default test split, which shares time with fold-2's training
    split, and scores fold-2's. Resuming the run on data split otherwise is refused.
    """
    fixture = _split_world([FOLD_2, gf.DEFAULT_SPLIT])
    with world(fixture):
        def data(split, spec=FOLD_2):
            return StandingBatches(fx.dataset(fixture, split, steps_per_epoch=2,
                                              split_spec=spec))

        def run(epochs):
            return recipe.TrainingRun(RunDirectory(tmp_path / "run"), seed=1, schedule=recipe.TrainingSchedule(
                epochs=epochs, steps_per_epoch=2, validation_batches=1, patience=1))

        run(1).train(tiny_model, lambda seed: data("train"), recipe.materialise(data("validate"), 1), verbose=0)
        directory = RunDirectory(tmp_path / "run")
        record = directory.run
        assert record["split"] == {"fingerprint": FOLD_2.fingerprint(), "selection_split": "validate"}
        assert run_record_of(directory.load_model("model")) == record

        def args(spec):
            return {"noise_obtainer": fx.noise_obtainer(fixture, split_spec=spec),
                    "onsource_duration_seconds": 1.0, "input_variables": [gf.ReturnVariables.WHITENED_ONSOURCE]}

        with pytest.raises(ValueError, match="SplitSpec"):
            _scores_split("test", record, args(gf.DEFAULT_SPLIT))
        assert _scores_split("test", record, args(FOLD_2))

        def resume(spec):
            run(2).train(tiny_model, lambda seed: data("train", spec), recipe.materialise(data("validate", spec), 1),
                         verbose=0)

        with pytest.raises(StaleCheckpointError, match="records the split"):
            resume(gf.DEFAULT_SPLIT)
        directory._write(directory.path / "run.json", {k: v for k, v in record.items() if k != "split"})
        with pytest.raises(StaleCheckpointError, match="records the split none"):
            resume(FOLD_2)


@pytest.mark.global_rng  # gf.Model builds its Keras model from Keras' global seed
def test_a_gf_model_records_its_split_for_the_validator(tmp_path):
    """
    gf.Model records the split its data reads and the split that selected it, and its
    weights carry that record to the validator through Model.validate: it refuses to
    score the selection split ('validate') and scores 'test'.
    """
    from gravyflow.tests.test_model_train import dataset_args, tiny_model as model_of

    model = model_of(tmp_path)
    model.train(validate_args={**dataset_args(steps_per_epoch=2), "group": "validate"})

    def scores(group):
        args = {key: value for key, value in dataset_args().items() if key != "group"}
        return _scores(lambda: model.validate(args, ValidationConfig(group=group)))

    with pytest.raises(ValueError, match="selected on the 'validate' split"):
        scores("validate")
    assert scores("test")
    assert model.run_record["split"] == {
        "fingerprint": gf.DEFAULT_SPLIT.fingerprint(), "selection_split": "validate"}


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


def _selected_on_validate(spec):
    """The record of a model trained on ``spec``'s train split and selected on its validate split."""
    return {"split": {"fingerprint": spec.fingerprint(), "selection_split": "validate"}}


def test_the_validator_reads_the_split_of_the_data_it_scores():
    """
    The validator checks a model's record against the split of the data it scores,
    whatever obtainer reads that data. With fold-2 glitches as its noise (a transient
    obtainer, not a NoiseObtainer) it refuses a model trained on the default split, whose
    training time fold-2's test split shares, and scores a model trained on fold-2.
    """
    glitches = gf.TransientDataObtainer(data_quality=gf.DataQuality.BEST, data_labels=[gf.DataLabel.GLITCHES],
                                        observing_runs=gf.ObservingRun.O3, split_spec=FOLD_2)
    args = {"noise_obtainer": glitches, "onsource_duration_seconds": 1.0,
            "input_variables": [gf.ReturnVariables.WHITENED_ONSOURCE]}
    with pytest.raises(ValueError, match="SplitSpec"):
        _scores_split("test", _selected_on_validate(gf.DEFAULT_SPLIT), args)
    assert _scores_split("test", _selected_on_validate(FOLD_2), args)


def test_synthetic_data_is_split_neutral(world):
    """
    Data that reads no real data (WHITE or COLORED noise) is split-neutral: it names
    no SplitSpec, so it combines with real data of any spec. Synthetic training data
    selected on fold-2's real validate split records fold-2's spec; synthetic data
    alone records the default, as it did. A validator scoring synthetic data scores a
    model of any spec, though never on its selection split. Real noise with no
    obtainer named is the default spec's, and is never neutral.
    """
    from gravyflow.src.validate.config import split_record
    from gravyflow.tests.test_model_train import dataset_args

    def synthetic(group, noise_type):
        args = {**dataset_args(), "group": group, "steps_per_epoch": 1,
                "noise_obtainer": gf.NoiseObtainer(noise_type=noise_type, ifos=[gf.IFO.L1])}
        return StandingBatches(gf.Dataset(**args))

    fixture = _split_world([FOLD_2, gf.DEFAULT_SPLIT])
    with world(fixture):
        real = StandingBatches(fx.dataset(fixture, "validate", steps_per_epoch=2,
                                          split_spec=FOLD_2))
        for noise_type in (gf.NoiseType.WHITE, gf.NoiseType.COLORED):
            training = synthetic("train", noise_type)
            assert training.split_spec is None
            assert training.dataset.seed_metadata["split_fingerprint"] is None
            assert split_record(training, real) == {"fingerprint": FOLD_2.fingerprint(),
                                                    "selection_split": "validate"}
            assert split_record(training, synthetic("validate", noise_type)) == {
                "fingerprint": gf.DEFAULT_SPLIT.fingerprint(), "selection_split": "validate"}
    assert split_spec_of([None]) == gf.DEFAULT_SPLIT
    assert split_spec_of([gf.NoiseObtainer(noise_type=gf.NoiseType.REAL)]) == gf.DEFAULT_SPLIT
    assert _scores_split("test", _selected_on_validate(FOLD_2))
    with pytest.raises(ValueError, match="selected on the 'validate' split"):
        _scores_split("validate", _selected_on_validate(FOLD_2))


def test_only_gravyflow_data_that_reads_no_real_data_is_split_neutral():
    """
    A run took a split_spec of None from any object as split-neutral data. Only
    gravyflow data none of whose obtainers reads real data is (`split_neutral`): a
    synthetic dataset, the recipe's batches of one, and a fixed set drawn from them.
    Anything else whose split_spec is None names no split, and a run refuses it.
    """
    from types import SimpleNamespace

    from gravyflow.src.dataset.dataset import split_neutral
    from gravyflow.src.validate.config import split_record
    from gravyflow.tests.test_model_train import dataset_args

    def synthetic(group):  # white noise
        return gf.Dataset(**{**dataset_args(), "group": group, "steps_per_epoch": 1})

    training = synthetic("train")
    for data in (training, recipe.DetectionBatches(training),
                 recipe.materialise(StandingBatches(synthetic("validate")), 1)):
        assert data.split_spec is None and split_neutral(data)
    assert split_record(training, synthetic("validate")) == {
        "fingerprint": gf.DEFAULT_SPLIT.fingerprint(), "selection_split": "validate"}
    for role, impostor in (("training", SimpleNamespace(split_spec=None, group="train")),
                           ("validation", SimpleNamespace(split_spec=None, group="validate"))):
        assert not split_neutral(impostor)
        pair = (impostor, synthetic("validate")) if role == "training" else (training, impostor)
        with pytest.raises(ValueError, match=f"The {role} data .* names no split"):
            split_record(*pair)


def test_real_events_beside_synthetic_noise_are_checked_against_the_model():
    """
    Real events are real data, whatever the noise beside them. Beside synthetic
    noise, which names no split, the validator read them under the default split but
    checked the model against none, so a model trained on fold-2, whose training time
    the default test split shares, was scored on real events it may have trained on.
    The bank's split counts its real-events stage, which beside synthetic noise reads
    under the default split: that model is refused before any data is read, a model
    of the default split is scored, and with no real events the noise is
    split-neutral again.
    """
    first = gf.DEFAULT_SPLIT.block_of(fx.origin_gps("O3"))
    assert any(gf.DEFAULT_SPLIT.block_split(k) == "test" and FOLD_2.block_split(k) == "train"
               for k in range(first, first + 1000))
    with pytest.raises(ValueError, match="SplitSpec"):
        _scores_split("test", _selected_on_validate(FOLD_2), score_real_events=True)
    assert _scores_split("test", _selected_on_validate(gf.DEFAULT_SPLIT), score_real_events=True)
    assert _scores_split("test", _selected_on_validate(FOLD_2))


class _Read(Exception):
    pass


def _events_split(bank):
    """The split the bank's real-event stage reads its events under, stopped before it reads them."""
    asked = {}

    class Recording(gf.TransientDataObtainer):
        def __init__(self, *args, split_spec=None, **kwargs):
            asked["split_spec"] = split_spec
            raise _Read

    bank.far_scores = np.zeros(4)
    with mock.patch.object(gf, "TransientDataObtainer", Recording), pytest.raises(_Read):
        bank.generate_real_events([gf.ObservingRun.O3])
    return asked["split_spec"]


@pytest.mark.parametrize("noise,real_events,spec", [
    (gf.NoiseType.WHITE, True, gf.DEFAULT_SPLIT),
    (gf.NoiseType.WHITE, False, None),
    (gf.NoiseType.REAL, True, FOLD_2),
    (gf.NoiseType.REAL, False, FOLD_2),
], ids=["synthetic noise", "synthetic noise, real events off", "real noise", "real noise, real events off"])
def test_the_real_events_stage_reads_under_the_banks_split(offline_events, noise, real_events, spec):
    """
    The real-event stage reads its events under the bank's split, the one the model is
    checked against: real noise's, or the default beside synthetic noise. A bank whose
    split names none (synthetic noise, real events off) refuses to read them by name.
    """
    reader = gf.IFODataObtainer(data_quality=gf.DataQuality.BEST, data_labels=[gf.DataLabel.NOISE],
                                split_spec=FOLD_2) if noise is gf.NoiseType.REAL else None
    args = {"noise_obtainer": gf.NoiseObtainer(noise_type=noise, ifo_data_obtainer=reader),
            "onsource_duration_seconds": 1.0, "input_variables": [gf.ReturnVariables.WHITENED_ONSOURCE]}
    bank = ValidationBank(stub_model(256), args, ValidationConfig(score_real_events=real_events))
    assert bank.split_spec == spec
    if spec is None:
        with pytest.raises(ValueError, match="this bank's names none"):
            _events_split(bank)
    else:
        assert _events_split(bank) == spec


#: The one computation of a split, where gravyflow's source defines it.
COMPUTATION = ("dataset/dataset.py", split_spec_of.__name__)


def _split_spec_reads():
    """
    Where gravyflow's source reads a ``split_spec`` from another object than itself
    (``x.split_spec`` or ``getattr(x, "split_spec", ...)``, x not ``self``) or computes
    one (a call of `split_spec_of`), as (module, enclosing class and function, how):
    "obtainer" when x names an obtainer, "data" when it does not, "computes" for a call.
    """
    import ast
    from pathlib import Path

    source = Path(gf.__file__).parent / "src"
    reads = set()

    def names_an_obtainer(receiver):
        return any("obtainer" in (getattr(node, "id", None) or getattr(node, "attr", None)
                                  or (node.value if isinstance(node, ast.Constant) and isinstance(node.value, str)
                                      else ""))
                   for node in ast.walk(receiver))

    def visit(node, scope, module):
        for child in ast.iter_child_nodes(node):
            inner = scope
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                inner = f"{scope}.{child.name}" if scope else child.name
            receiver = None
            if isinstance(child, ast.Attribute) and child.attr == "split_spec" and isinstance(child.ctx, ast.Load):
                receiver = child.value
            elif (isinstance(child, ast.Call) and isinstance(child.func, ast.Name)
                  and child.func.id in ("getattr", "hasattr") and len(child.args) >= 2
                  and isinstance(child.args[1], ast.Constant) and child.args[1].value == "split_spec"):
                receiver = child.args[0]
            elif (isinstance(child, ast.Call)
                  and COMPUTATION[1] in (getattr(child.func, "id", None), getattr(child.func, "attr", None))):
                reads.add((module, inner, "computes"))
            if receiver is not None and not (isinstance(receiver, ast.Name) and receiver.id == "self"):
                reads.add((module, inner, "obtainer" if names_an_obtainer(receiver) else "data"))
            visit(child, inner, module)

    for path in sorted(source.rglob("*.py")):
        visit(ast.parse(path.read_text()), "", path.relative_to(source).as_posix())
    return reads


#: Every place that reads another object's split or computes one, and why it may. Only
#: the one computation reads obtainers; everything else reads its split from data, or
#: has the computation read the obtainers its data is read through.
SPLIT_READERS = {
    COMPUTATION: "the one computation: the obtainers' spec",
    ("dataset/dataset.py", "IndexedBatches.split_spec"): "a dataset's: the computation over its obtainers",
    ("validate/bank.py", "ValidationBank.split_spec"): (
        "the validator's data: the computation over its datasets' obtainer, before they are built"),
    ("validate/validator.py", "Validator.validate"): "the validator's check: its data's (the bank's)",
    ("validate/config.py", "split_record"): "a training run's: its training and validation data's",
    ("model/examples/recipe.py", "_DatasetBatches.split_spec"): "a recipe's batches: their dataset's",
    ("model/examples/recipe.py", "materialise"): "a fixed validation set: the batches it was drawn from",
}


def test_only_a_dataset_derives_its_split_from_its_obtainers():
    """
    A split has one computation, `split_spec_of`, over the obtainers data is read
    through, and only a dataset and the validator's bank (over its datasets' obtainer,
    before it builds them) call it. Everything else reads the split of its data, never
    computing it again: a second computation could see other obtainers, or the same
    ones otherwise (one ignored a transient obtainer used as noise), and score a model
    on the time it trained on.
    """
    reads = _split_spec_reads()
    readers = {(module, scope) for module, scope, _ in reads}
    assert readers == set(SPLIT_READERS), (
        f"split read outside the known readers: {sorted(readers - set(SPLIT_READERS))}; "
        f"known readers gone: {sorted(set(SPLIT_READERS) - readers)}"
    )
    from_obtainers = {(module, scope) for module, scope, how in reads if how == "obtainer"}
    assert from_obtainers == {COMPUTATION}, (
        f"a split derived from an obtainer outside {COMPUTATION[1]}: {sorted(from_obtainers)}"
    )
    computing = {(module, scope) for module, scope, how in reads if how == "computes"}
    assert computing == {("dataset/dataset.py", "IndexedBatches.split_spec"),
                         ("validate/bank.py", "ValidationBank.split_spec")}, (
        f"the split computed again outside a dataset and the validator's bank: {sorted(computing)}"
    )


@pytest.mark.parametrize("noise,spec", [
    (lambda: {}, gf.DEFAULT_SPLIT),
    (lambda: {"noise_obtainer": gf.NoiseObtainer(noise_type=gf.NoiseType.WHITE)}, None),
    (lambda: {"noise_obtainer": gf.NoiseObtainer(noise_type=gf.NoiseType.REAL, ifo_data_obtainer=gf.IFODataObtainer(
        data_quality=gf.DataQuality.BEST, data_labels=[gf.DataLabel.NOISE], split_spec=FOLD_2))}, FOLD_2),
    (lambda: {"noise_obtainer": gf.TransientDataObtainer(
        data_quality=gf.DataQuality.BEST, data_labels=[gf.DataLabel.GLITCHES],
        observing_runs=gf.ObservingRun.O3, split_spec=FOLD_2)}, FOLD_2),
], ids=["the default noise", "white noise", "real noise", "glitches as noise"])
def test_the_banks_split_is_its_datasets(noise, spec):
    """
    The bank reads its split before it builds a dataset, and it is the split of the
    dataset it then builds: its noise's, a glitch obtainer used as noise included, the
    default noise's (real) when none is named, and none for synthetic noise. The dataset is built here without its
    generators, which read the data, since only its split is compared.
    """
    args = {"onsource_duration_seconds": 1.0, "input_variables": [gf.ReturnVariables.WHITENED_ONSOURCE], **noise()}
    bank = ValidationBank(stub_model(256), args)
    with mock.patch.object(gf.GravyflowDataset, "_create_generators", lambda self: None):
        assert bank.split_spec == bank._noise_dataset.split_spec == spec


class _EventsObtainerBuilt(Exception):
    pass


@pytest.mark.parametrize("noise,source", [
    (lambda: {}, None),
    (lambda: {"noise_obtainer": gf.NoiseObtainer(noise_type=gf.NoiseType.REAL, ifo_data_obtainer=gf.IFODataObtainer(
        data_quality=gf.DataQuality.BEST, data_labels=[gf.DataLabel.NOISE], source=gf.LocalFrames()))},
     gf.LocalFrames()),
    (lambda: {"noise_obtainer": gf.TransientDataObtainer(
        data_quality=gf.DataQuality.BEST, data_labels=[gf.DataLabel.GLITCHES],
        observing_runs=gf.ObservingRun.O3, source=gf.LocalFrames())}, gf.LocalFrames()),
], ids=["the default noise", "real noise", "glitches as noise"])
def test_the_banks_events_read_its_backgrounds_source(noise, source, monkeypatch):
    """
    Real events are read from the source the noise background reads, so an event's
    score and the background its FAR comes from are measured on the same strain; with
    synthetic noise, from the default source.
    """
    args = {"onsource_duration_seconds": 1.0, "input_variables": [gf.ReturnVariables.WHITENED_ONSOURCE], **noise()}
    built = []

    def events_obtainer(**kwargs):
        built.append(kwargs.get("source"))
        raise _EventsObtainerBuilt

    from gravyflow.src.dataset.features import event
    monkeypatch.setattr(event, "get_events_with_params",
                        lambda *args, **kwargs: [{"name": "GW190521", "gps": 1242442967.4}])
    monkeypatch.setattr(gf, "TransientDataObtainer", events_obtainer)
    bank = ValidationBank(stub_model(256), args)
    bank.far_scores = np.zeros(1)
    with pytest.raises(_EventsObtainerBuilt):
        bank.generate_real_events()
    assert built == [source]


class _ModelBatches(recipe.DetectionBatches):
    """A gravyflow dataset's split, served as the small gf.Model's batches: training reads its split."""

    def __getitem__(self, index):
        rng = np.random.default_rng(index)
        x = rng.normal(size=(4, 1, 256)).astype(np.float32)
        return {"ONSOURCE": x}, {"INJECTION_PRESENT": rng.integers(0, 2, (4, 1)).astype(np.float32)}


def _gf_model(path, load=False):
    """A small gf.Model in ``path`` with no dataset of its own (each `train` names its data), or loaded from it."""
    from gravyflow.tests.test_model_train import BATCH as MODEL_BATCH, ONSOURCE, SAMPLE_RATE, TRAINING

    shape = dict(input_configs=[{"name": "ONSOURCE", "shape": (1, int(SAMPLE_RATE * ONSOURCE))}],
                 output_config={"name": "INJECTION_PRESENT", "type": "binary"},
                 training_config=dict(TRAINING), optimizer="sgd", loss=None)
    if load:
        return gf.Model.load("tiny", path, hidden_layers=[gf.DenseLayer(4)], **shape)
    return gf.Model(name="tiny", layers=[gf.DenseLayer(4)], batch_size=MODEL_BATCH, model_path=path, **shape)


def _default_data():
    """A gf.Model's training data and validation arguments on synthetic noise: the default split."""
    from gravyflow.tests.test_model_train import dataset_args

    return {"training_dataset": gf.Dataset(**dataset_args(steps_per_epoch=3)),
            "validate_args": {**dataset_args(steps_per_epoch=2), "group": "validate"}}


def _fold_2_data(fixture):
    """A gf.Model's training data and validation arguments on the fixture's real noise, split by fold-2."""
    from gravyflow.tests.test_model_train import dataset_args

    real = dict(dataset_args(), noise_obtainer=fx.noise_obtainer(fixture, ifos=("L1",), split_spec=FOLD_2),
                ifos=[gf.IFO.L1], waveform_generators=None)
    return {"training_dataset": _ModelBatches(gf.Dataset(**{**real, "steps_per_epoch": 3})),
            "validate_args": {**real, "steps_per_epoch": 2, "group": "validate"}}


@pytest.mark.global_rng  # gf.Model builds its Keras model from Keras' global seed
@pytest.mark.parametrize("force_retrain", [True, False])
def test_trained_weights_keep_to_their_split(world, tmp_path, force_retrain):
    """
    A gf.Model's split record belongs to its weights. Weights trained on the default
    split, still in memory or loaded from their checkpoint, are never trained on fold-2's
    data, whatever force_retrain says: their record would then name fold-2, and the
    validator would score fold-2's test, which their first training read.
    """
    fixture = _split_world([FOLD_2, gf.DEFAULT_SPLIT])
    trained = _gf_model(tmp_path / "model")
    trained.train(**_default_data())
    loaded = _gf_model(tmp_path / "model", load=True)
    with world(fixture):
        for model in (trained, loaded):
            with pytest.raises(StaleCheckpointError, match="records the split"):
                model.train(**_fold_2_data(fixture), force_retrain=force_retrain)
            assert model.run_record == _selected_on_validate(gf.DEFAULT_SPLIT)


@pytest.mark.global_rng
def test_a_resumed_gf_model_keeps_to_its_directorys_split(world, tmp_path):
    """
    Resuming (force_retrain=False) continues the run saved in the model's directory,
    whose best weights it may restore: new weights resumed there on another split's data
    are refused, as weights carrying that run's record would be.
    """
    fixture = _split_world([FOLD_2, gf.DEFAULT_SPLIT])
    _gf_model(tmp_path / "model").train(**_default_data())
    fresh = _gf_model(tmp_path / "model")
    with world(fixture):
        with pytest.raises(StaleCheckpointError, match="records the split"):
            fresh.train(**_fold_2_data(fixture), force_retrain=False)


class _Interrupted(Exception):
    pass


class _InterruptTraining(keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        raise _Interrupted


@pytest.mark.global_rng
def test_a_new_run_never_leaves_an_earlier_runs_weights_beside_its_record(world, tmp_path):
    """
    New weights trained in a directory holding a run of another split start a new run
    there: the earlier run's checkpoint goes before the new record is written, so a run
    interrupted before its first checkpoint never leaves default-split weights beside a
    fold-2 record, for a load to label fold-2's.
    """
    fixture = _split_world([FOLD_2, gf.DEFAULT_SPLIT])
    _gf_model(tmp_path / "model").train(**_default_data())
    with world(fixture):
        with pytest.raises(_Interrupted):
            _gf_model(tmp_path / "model").train(**_fold_2_data(fixture),
                                                callbacks=[_InterruptTraining()])
    assert _gf_model(tmp_path / "model", load=True).run_record is None, "loaded an earlier run's weights"


@pytest.mark.global_rng
def test_gravyflow_models_carry_their_record(world, tmp_path):
    """
    Every model gravyflow trains or loads carries its training record, so the validator
    refuses the split that selected it without being handed the record: a run's model
    (TrainingRun.train's, and RunDirectory.load_model's), and a gf.Model's.
    """
    from gravyflow.tests.test_model_train import dataset_args

    fixture = _split_world([FOLD_2, gf.DEFAULT_SPLIT])

    def selected_on_validate_is_refused(model, scored_args):
        with pytest.raises(ValueError, match="selected on the 'validate' split"):
            _scores(lambda: Validator.validate(model, "model", scored_args, ValidationConfig(group="validate")))

    with world(fixture):
        def data(split):
            return StandingBatches(fx.dataset(fixture, split, steps_per_epoch=2,
                                              split_spec=FOLD_2))

        run = recipe.TrainingRun(RunDirectory(tmp_path / "run"), seed=1, schedule=recipe.TrainingSchedule(
            epochs=1, steps_per_epoch=2, validation_batches=1, patience=1))
        model = run.train(tiny_model, lambda seed: data("train"), recipe.materialise(data("validate"), 1),
                          verbose=0)
        args = {"noise_obtainer": fx.noise_obtainer(fixture, split_spec=FOLD_2),
                "onsource_duration_seconds": 1.0, "input_variables": [gf.ReturnVariables.WHITENED_ONSOURCE]}
        for loaded in (model, *(run.directory.load_model(name) for name in ("model", "best", "last"))):
            selected_on_validate_is_refused(loaded, args)

    trained = _gf_model(tmp_path / "gf-model")
    trained.train(**_default_data())
    selected_on_validate_is_refused(trained.model, {k: v for k, v in dataset_args().items() if k != "group"})


def test_a_run_never_relabels_the_weights_it_is_given(world, tmp_path):
    """
    A run whose build_model returns weights already trained (another run's model, or
    one's weights transferred) keeps to their split: trained on the default split, they
    are never trained again on fold-2's and recorded as fold-2's.
    """
    fixture = _split_world([FOLD_2, gf.DEFAULT_SPLIT])
    with world(fixture):
        def data(split, spec):
            return StandingBatches(fx.dataset(fixture, split, steps_per_epoch=2,
                                              split_spec=spec))

        def train_run(directory, spec, build):
            run = recipe.TrainingRun(RunDirectory(tmp_path / directory), seed=1, schedule=recipe.TrainingSchedule(
                epochs=1, steps_per_epoch=2, validation_batches=1, patience=1))
            return run.train(build, lambda seed: data("train", spec), recipe.materialise(data("validate", spec), 1),
                             verbose=0)

        trained = train_run("default", gf.DEFAULT_SPLIT, tiny_model)
        with pytest.raises(StaleCheckpointError, match="records the split"):
            train_run("fold-2", FOLD_2, lambda: trained)


def test_a_transient_evaluation_set_is_served_at_most_once(world):
    """
    A transient test split read through a dataset is one fixed set: floor(N / B) whole
    batches, each record at most once, the same at every epoch; a batch past the set is
    refused by name, never a second pass or a partial batch.
    """
    from gravyflow.tests.leakage.fakes import GlitchRecord

    edge = fx.boundary("O3", "train", "test")
    fixture = fx._fixture([(edge, edge + fx.BLOCK_SECONDS)], 256.0, "O3", ("O3",))
    peaks = [edge + fx.BLOCK_SECONDS / 2 + 300.0 * k for k in range(-2, 3)]
    fixture = fixture.with_catalogues(glitches=[GlitchRecord("L1", t, t - 0.5, 1.0, "Blip", 0.95) for t in peaks])
    with world(fixture):
        ds = gf.Dataset(
            noise_obtainer=fx.ifo_data_obtainer(fixture, data_labels=[gf.DataLabel.GLITCHES]),
            sample_rate_hertz=256.0, onsource_duration_seconds=1.0, crop_duration_seconds=0.5,
            offsource_duration_seconds=4.0, num_examples_per_batch=2, steps_per_epoch=3, group="test",
            seed=5, ifos=[gf.IFO.L1], input_variables=[gf.ReturnVariables.ONSOURCE],
            output_variables=[gf.ReturnVariables.START_GPS_TIME],
        )
        epochs = []
        for _ in range(2):
            epochs.append([np.asarray(ds[k][1][gf.ReturnVariables.START_GPS_TIME.name])[:, 0] for k in range(2)])
            ds.on_epoch_end()
        # The set is the dataset's length: a batch past it is refused by the dataset, and
        # by the batcher itself, by name.
        assert len(ds) == 2
        with pytest.raises(IndexError, match="outside the 2 batches"):
            ds[2]
        with pytest.raises(IndexError, match="never are"):
            ds.noise(ds.tree.batch("test", 0, 2))
    served = np.concatenate(epochs[0])
    assert len(set(served.tolist())) == len(served) == 4
    assert all(min(abs(start - t) for t in peaks) < 2.0 for start in served), "a window is no record's"
    assert all(np.array_equal(a, b) for a, b in zip(epochs[0], epochs[1])), "an evaluation epoch differs"
