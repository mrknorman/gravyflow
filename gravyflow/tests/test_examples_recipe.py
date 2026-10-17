"""
The example models and their run recipe: the input contract, the run directory,
resume with checkpoint provenance, and the training scripts' settings.
"""

import importlib
import os
import warnings
from types import SimpleNamespace

import keras
import numpy as np
import pytest


import gravyflow as gf
from gravyflow.src.resources import PACKAGE
from gravyflow.src.model.examples import (
    Gabbard2017, Gabbard2017Config, GeorgeHuerta2017Config, GeorgeHuerta2017Deep, GeorgeHuerta2017Shallow,
)
from gravyflow.src.model.examples.george_huerta_2017 import curriculum_epochs
from gravyflow.src.model.examples.recipe import (
    DetectionBatches, FixedBatches, RegressionBatches, RunDirectory, TrainingRun, TrainingSchedule, materialise,
)
from gravyflow.src.validate import StaleCheckpointError


# ------------------------------------------------------------- the models

@pytest.mark.parametrize("model_class,config", [
    (Gabbard2017, Gabbard2017Config.cpu()),
    (GeorgeHuerta2017Shallow, GeorgeHuerta2017Config.cpu()),
    (GeorgeHuerta2017Deep, GeorgeHuerta2017Config()),
])
@pytest.mark.global_rng
def test_examples_take_whitened_onsource_and_return_logits(model_class, config):
    model = model_class.model(config)
    assert [t.name for t in model.inputs] == ["WHITENED_ONSOURCE"]
    assert model.layers[-1].activation.__name__ == "linear"
    samples = int(config.sample_rate_hertz * config.onsource_duration_seconds)
    x = np.random.default_rng(0).normal(size=(3, 1, samples)).astype(np.float16)
    out = model.predict_on_batch({"WHITENED_ONSOURCE": x})
    assert out.shape == (3, 2) and out.dtype == np.float32


@pytest.mark.global_rng
def test_gabbard_cpu_preset_size():
    assert Gabbard2017.model(Gabbard2017Config.cpu()).count_params() == 41_842


@pytest.mark.global_rng
def test_shipped_weights_for_the_old_contract_are_refused():
    with pytest.raises(ValueError, match="older input contract"):
        Gabbard2017.pretrained()


@pytest.mark.global_rng
def test_george_huerta_transfer_copies_every_layer_but_the_output():
    config = GeorgeHuerta2017Config.cpu()
    predictor = GeorgeHuerta2017Shallow.model(config, task="regression")
    detector = GeorgeHuerta2017Shallow.model(config, task="classification")
    copied = GeorgeHuerta2017Shallow.transfer_weights(predictor, detector)
    assert copied == 4  # three convolutions and one dense layer
    for name in ("Conv1D_1", "Dense_1"):
        for a, b in zip(predictor.get_layer(name).get_weights(), detector.get_layer(name).get_weights()):
            np.testing.assert_array_equal(a, b)


# ------------------------------------------------------- the scripts' settings

def test_train_script_leaves_the_platform_alone():
    # GF-B65: the script forced JAX_PLATFORMS=cuda, which fails on a Mac.
    before = {k: os.environ.get(k) for k in ("JAX_PLATFORMS", "XLA_FLAGS")}
    module = importlib.import_module("gravyflow.src.model.examples.train_gabbard_2017")
    try:
        importlib.reload(module)
        assert {k: os.environ.get(k) for k in before} == before
    finally:
        for k, v in before.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_train_scripts_need_an_output_directory():
    # GF-B66: checkpoints went to the shipped weights in the package.
    for name in ("train_gabbard_2017", "train_george_huerta_2017"):
        module = importlib.import_module(f"gravyflow.src.model.examples.{name}")
        with pytest.raises(SystemExit):
            module.parse_args([])


def test_run_directory_refuses_the_package(tmp_path):
    for inside in (PACKAGE, PACKAGE / "res" / "model", PACKAGE / "res" / "model" / "run"):
        with pytest.raises(ValueError, match="outside the gravyflow package"):
            RunDirectory(inside)
    assert RunDirectory(tmp_path / "run").path == (tmp_path / "run").resolve()


@pytest.mark.parametrize("epochs", [4, 5, 10, 20, 81])
def test_curriculum_reaches_the_target_before_the_run_ends(epochs):
    # GF-B77: an 80-epoch curriculum in a 20-epoch run ended training at SNR 62-80.
    train = importlib.import_module("gravyflow.src.model.examples.train_george_huerta_2017")
    args = train.parse_args(["--output-dir", "unused", "--epochs", str(epochs)])
    config, schedule = train.configure(args)
    assert config.snr_curriculum_num_epochs == curriculum_epochs(epochs) <= epochs
    assert schedule.monitor_from_epoch == config.snr_curriculum_num_epochs
    last = config.snr(True).at(epochs - 1)
    assert (last.min_, last.max_) == (config.snr_min, config.snr_max)
    first = config.snr(True).at(0)
    assert (first.min_, first.max_) == (config.snr_curriculum_start_min, config.snr_curriculum_start_max)


def test_curriculum_needs_room():
    with pytest.raises(ValueError):
        curriculum_epochs(3)


# -------------------------------------------------------------- batches

def test_detection_labels_are_one_hot_injection_present():
    class Batches:
        def __len__(self):
            return 1

        def __getitem__(self, index):
            x = {"WHITENED_ONSOURCE": np.zeros((4, 1, 16), np.float64)}
            return x, {"INJECTION_PRESENT": np.array([1.0, 0.0, 1.0, 0.0], np.float32)}

    x, y = DetectionBatches(Batches())[0]
    assert x["WHITENED_ONSOURCE"].dtype == np.float32
    np.testing.assert_array_equal(y, [[0, 1], [1, 0], [0, 1], [1, 0]])


class CountingBatches(keras.utils.PyDataset):
    """Random batches standing for a gravyflow split's (``group``, default spec), counting draws."""

    split_spec = gf.DEFAULT_SPLIT

    def __init__(self, seed, n=3, group="train"):
        super().__init__()
        self.rng, self.n, self.calls = np.random.default_rng(seed), n, 0
        self.group = group

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        self.calls += 1
        x = self.rng.normal(size=(8, 1, 16)).astype(np.float32)
        y = np.eye(2, dtype=np.float32)[self.rng.integers(0, 2, 8)]
        return {"WHITENED_ONSOURCE": x}, y


def test_materialise_serves_identical_batches():
    source = CountingBatches(0)
    fixed = materialise(source, 2)
    assert isinstance(fixed, FixedBatches) and len(fixed) == 2 and source.calls == 2
    np.testing.assert_array_equal(fixed[0][0]["WHITENED_ONSOURCE"], fixed[0][0]["WHITENED_ONSOURCE"])
    assert fixed[1] is fixed[1]


def test_batches_name_their_datasets_split():
    fold = gf.SplitSpec(version="fold-3")
    dataset = SimpleNamespace(split_spec=fold, group="validate")  # a dataset of real data split by fold
    for batches in (DetectionBatches(dataset), RegressionBatches(dataset, []),
                    FixedBatches([], dataset.split_spec, dataset.group)):
        assert (batches.split_spec, batches.group) == (fold, "validate")
    fixed = materialise(CountingBatches(0, group="test"), 1)
    assert (fixed.split_spec, fixed.group) == (gf.DEFAULT_SPLIT, "test")


@pytest.mark.parametrize("training,validation,refusal", [
    (FixedBatches([], None, None), CountingBatches(0, group="validate"), "training data .* names no split"),
    (CountingBatches(0), materialise(range(1), 1), "validation data .* names no split"),
    (CountingBatches(0, group="validate"), CountingBatches(0, group="validate"), "training data is the 'validate'"),
    (CountingBatches(0), CountingBatches(0), "validation data is the 'train'"),
    (CountingBatches(0), FixedBatches([], gf.SplitSpec(version="fold-3"), "validate"), "split by SplitSpec"),
], ids=["training unnamed", "validation unnamed", "training held out", "selection on train", "two specs"])
def test_a_run_refuses_data_that_does_not_name_one_split(tmp_path, training, validation, refusal):
    with pytest.raises(ValueError, match=refusal):
        make_run(tmp_path, 1).train(tiny_model, lambda seed: training, validation, verbose=0)
    assert not (tmp_path / "run.json").exists()


# ------------------------------------------------------ resume and provenance

def tiny_model():
    inputs = keras.Input(shape=(1, 16), name="WHITENED_ONSOURCE")
    x = keras.layers.Dense(2, name="output")(keras.layers.Flatten()(inputs))
    model = keras.Model({"WHITENED_ONSOURCE": inputs}, x)
    model.compile(optimizer=keras.optimizers.Adam(1e-2),
                  loss=keras.losses.CategoricalCrossentropy(from_logits=True))
    return model


def make_run(path, epochs, seed=7):
    return TrainingRun(RunDirectory(path), TrainingSchedule(epochs=epochs, steps_per_epoch=3,
                                                            validation_batches=2, patience=100),
                       seed=seed, description={"model": "tiny"})


class EpochBatches(CountingBatches):
    """Counting batches that record the epoch of the gravyflow training split they stand for."""

    def __init__(self, seed, served):
        super().__init__(seed)
        self.served = served
        self.dataset = gf.Dataset(
            seed=seed, noise_obtainer=gf.NoiseObtainer(noise_type=gf.NoiseType.WHITE),
            steps_per_epoch=self.n, input_variables=[gf.ReturnVariables.ONSOURCE],
            group="train",
        )

    def epoch_callback(self):
        return self.dataset.epoch_callback()

    def __getitem__(self, index):
        self.served.append(self.dataset.epoch)
        return super().__getitem__(index)


def test_a_run_on_multi_threaded_cpu_kernels_warns(tmp_path, monkeypatch):
    # Multi-threaded kernels round the whitened strain differently from call to
    # call, and training amplifies it: such a run is not reproducible, and the
    # CPU preset sometimes fails to train. The test session runs single-threaded.
    from gravyflow.src.rng import platform

    validation = materialise(CountingBatches(99, group="validate"), 2)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        make_run(tmp_path / "single", 1).train(tiny_model, lambda seed: CountingBatches(seed), validation, verbose=0)
    monkeypatch.setattr(platform, "multi_threaded_cpu", lambda: True)
    with pytest.warns(UserWarning, match="XLA_FLAGS=--xla_cpu_multi_thread_eigen=false"):
        make_run(tmp_path / "multi", 1).train(tiny_model, lambda seed: CountingBatches(seed), validation, verbose=0)


def test_training_resumes_from_the_last_epoch(tmp_path):
    builds, served = [], []

    def build():
        builds.append(1)
        return tiny_model()

    def data(seed):
        return EpochBatches(seed, served)

    validation = materialise(CountingBatches(99, group="validate"), 2)
    make_run(tmp_path, 2).train(build, data, validation, verbose=0)
    directory = RunDirectory(tmp_path)
    assert directory.state["epoch"] == 1 and len(directory.state["history"]) == 2
    last_weights = directory.load_model("last").get_weights()

    resumed = make_run(tmp_path, 4)  # extend the finished run
    resumed.train(build, data, validation, verbose=0)
    assert len(builds) == 1  # resumed from the checkpoint
    # Each Keras epoch, the resumed ones too, served that epoch's batches. Each
    # fit first peeks at two batches to build the model (epoch 0's: shapes only).
    peek = [0, 0]
    assert served == peek + [0, 0, 0, 1, 1, 1] + peek + [2, 2, 2, 3, 3, 3]
    state = directory.state
    assert state["epoch"] == 3 and [h["epoch"] for h in state["history"]] == [0, 1, 2, 3]
    assert (tmp_path / "model.keras").exists()
    assert not all(np.array_equal(a, b) for a, b in
                   zip(last_weights, directory.load_model("last").get_weights()))


class AddressedBatches(keras.utils.PyDataset):
    """Batches addressed by (epoch, index), as a gravyflow training split's are, recording the order served."""

    def __init__(self, seed, served, n=4, group="train"):
        super().__init__()
        self.seed, self.served, self.n = seed, served, n
        self.dataset = gf.Dataset(
            seed=seed, noise_obtainer=gf.NoiseObtainer(noise_type=gf.NoiseType.WHITE),
            steps_per_epoch=n, input_variables=[gf.ReturnVariables.ONSOURCE],
            group=group,
        )
        self.split_spec, self.group = self.dataset.split_spec, self.dataset.group

    def __len__(self):
        return self.n

    def epoch_callback(self):
        return self.dataset.epoch_callback()

    def __getitem__(self, index):
        epoch = self.dataset.epoch
        self.served.append((epoch, index))
        rng = np.random.default_rng([self.seed, epoch, index])
        x = rng.normal(size=(8, 1, 16)).astype(np.float32)
        return {"WHITENED_ONSOURCE": x}, np.eye(2, dtype=np.float32)[(x.mean(axis=(1, 2)) > 0).astype(int)]


def dropout_model():
    """A model with every kind of Keras draw: initialisers and dropout."""
    inputs = keras.Input(shape=(1, 16), name="WHITENED_ONSOURCE")
    x = keras.layers.Dense(8, activation="elu")(keras.layers.Flatten()(inputs))
    x = keras.layers.Dropout(0.5)(x)
    model = keras.Model({"WHITENED_ONSOURCE": inputs}, keras.layers.Dense(2, name="output")(x))
    model.compile(optimizer=keras.optimizers.Adam(1e-2),
                  loss=keras.losses.CategoricalCrossentropy(from_logits=True))
    return model


def test_training_is_reproducible_from_the_seed(tmp_path):
    # GF-B215: fit shuffled each epoch with Python's unseeded global random, and
    # Keras seeded the initialisers and dropout from it too: neither the order
    # nor the weights followed from the run's seed. No global_rng mark: the
    # recipe must leave the global random state alone.
    runs = []
    for name in ("a", "b"):
        served = []
        validation = materialise(AddressedBatches(99, [], group="validate"), 2)
        run = make_run(tmp_path / name, 3)
        model = run.train(dropout_model, lambda seed: AddressedBatches(seed, served), validation, verbose=0)
        runs.append((served, model.get_weights()))
    (served_a, weights_a), (served_b, weights_b) = runs
    peek = [(0, 0), (0, 1)]  # fit's structure pass, before the first epoch
    expected = peek + [(epoch, k) for epoch in range(3) for k in range(4)]
    assert served_a == expected, f"served {served_a}"
    assert served_b == served_a
    for a, b in zip(weights_a, weights_b):
        np.testing.assert_array_equal(a, b)

    another = make_run(tmp_path / "c", 1, seed=8).train(
        dropout_model, lambda seed: AddressedBatches(seed, []),
        materialise(AddressedBatches(99, [], group="validate"), 2), verbose=0)
    assert not np.array_equal(another.get_weights()[0], weights_a[0])


def test_a_resumed_run_trains_as_an_uninterrupted_one(tmp_path):
    # GF-B215: a resumed run's epochs came in another order.
    validation = materialise(AddressedBatches(99, [], group="validate"), 2)
    whole = make_run(tmp_path / "whole", 3).train(
        dropout_model, lambda seed: AddressedBatches(seed, []), validation, verbose=0)
    make_run(tmp_path / "parts", 2).train(dropout_model, lambda seed: AddressedBatches(seed, []), validation, verbose=0)
    resumed = make_run(tmp_path / "parts", 3).train(
        dropout_model, lambda seed: AddressedBatches(seed, []), validation, verbose=0)
    whole_state = RunDirectory(tmp_path / "whole").state
    parts_state = RunDirectory(tmp_path / "parts").state
    assert whole_state["best_epoch"] == parts_state["best_epoch"]
    for a, b in zip(RunDirectory(tmp_path / "whole").load_model("last").get_weights(),
                    RunDirectory(tmp_path / "parts").load_model("last").get_weights()):
        np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-7)
    for a, b in zip(whole.get_weights(), resumed.get_weights()):
        np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-7)


def test_resume_refuses_another_configuration(tmp_path):
    validation = materialise(CountingBatches(99, group="validate"), 2)
    make_run(tmp_path, 1).train(tiny_model, CountingBatches, validation, verbose=0)
    with pytest.raises(StaleCheckpointError, match="run configuration"):
        make_run(tmp_path, 2, seed=8).train(tiny_model, CountingBatches, validation, verbose=0)


@pytest.mark.global_rng
def test_a_replaced_checkpoint_is_refused(tmp_path):
    validation = materialise(CountingBatches(99, group="validate"), 2)
    run = make_run(tmp_path, 1)
    run.train(tiny_model, CountingBatches, validation, verbose=0)
    keras.utils.set_random_seed(123)
    tiny_model().save(tmp_path / "checkpoints" / "last.keras")  # a stale or foreign file
    with pytest.raises(StaleCheckpointError, match="another model"):
        RunDirectory(tmp_path).load_model("last", run.config_sha256)
