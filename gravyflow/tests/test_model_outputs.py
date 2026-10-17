"""
gf.Model's output contract and its checkpoints: a detection model outputs a
logit, which the validator ranks; every checkpoint it saves loads again, and a
checkpoint that cannot load is an error, never an untrained model.
"""

import inspect

import keras
import numpy as np
import pytest

import gravyflow as gf
from gravyflow.src.model.examples.base import ValidationDatasetConfig, validation_dataset_args
from gravyflow.src.validate import ValidationConfig
from gravyflow.src.validate.bank import require_logit_model

SAMPLES = 4096  # WhitenPass crops the default onsource, 2048 samples, from the middle
INPUT = {"name": gf.ReturnVariables.ONSOURCE.name, "shape": (1, SAMPLES)}
BINARY = {"name": gf.ReturnVariables.INJECTION_PRESENT.name, "type": "binary"}
NORMAL = {"name": "SNR", "type": "normal"}


def model(tmp_path, layers, output_config=BINARY, input_config=INPUT):
    return gf.Model(
        name="tiny", layers=layers, optimizer="sgd", loss=None,
        input_configs=[input_config], output_config=output_config,
        model_path=tmp_path / "tiny",
    )


def load(tmp_path, layers, output_config=BINARY):
    return gf.Model.load(
        "tiny", tmp_path / "tiny", input_configs=[INPUT], output_config=output_config,
        hidden_layers=layers,
    )


def save(m, tmp_path):
    gf.ensure_directory_exists(tmp_path / "tiny")
    m.model.save(gf.model_file(tmp_path / "tiny"))


def predict(m, seed=0):
    x = np.random.default_rng(seed).normal(size=(8, 1, SAMPLES)).astype(np.float32)
    return m.model.predict_on_batch({INPUT["name"]: x})


# ------------------------------------------------------------------ logits

@pytest.mark.global_rng
def test_a_binary_model_outputs_a_logit(tmp_path):
    # GF-B205: the output was a sigmoid probability, which the validator refuses
    # (and which saturates, so scores tie).
    m = model(tmp_path, [gf.DenseLayer(4)])
    require_logit_model(m.model)
    assert m.model.layers[-1].activation.__name__ == "linear"
    loss = m.model.loss[BINARY["name"]]
    assert isinstance(loss, keras.losses.BinaryCrossentropy) and loss.get_config()["from_logits"]


@pytest.mark.global_rng
def test_binary_accuracy_reads_a_logit(tmp_path):
    m = model(tmp_path, [gf.DenseLayer(4)])
    # With a zero kernel the output is the bias: a logit of 0.3, so signal
    # (a probability threshold of 0.5 would call it noise).
    output = m.model.get_layer(BINARY["name"])
    kernel, _ = output.get_weights()
    output.set_weights([np.zeros_like(kernel), np.array([0.3], np.float32)])
    x = np.random.default_rng(0).normal(size=(4, 1, SAMPLES)).astype(np.float32)
    y = np.ones((4, 1), np.float32)
    logs = m.model.evaluate({INPUT["name"]: x}, {BINARY["name"]: y}, return_dict=True, verbose=0)
    assert logs["binary_accuracy"] == pytest.approx(1.0)


@pytest.mark.global_rng
def test_the_validator_scores_a_gf_model(tmp_path):
    # Model.validate called the validator with arguments it no longer takes.
    samples = 2048
    m = model(tmp_path, [gf.DenseLayer(4)],
              input_config={"name": gf.ReturnVariables.WHITENED_ONSOURCE.name, "shape": (1, samples)})
    validator = m.validate(
        validation_dataset_args(config=ValidationDatasetConfig(sample_rate_hertz=2048.0)),
        ValidationConfig(num_examples=64, batch_size=32, snr_range=(0.0, 15.0)),
        tmp_path / "validation",
    )
    assert validator.far_scores.shape == (64,)
    # Logits, not probabilities: a ReLU layer's scores are unbounded, never all in [0, 1].
    scores = np.concatenate([validator.far_scores, validator.efficiency_data["scores"]])
    assert not np.all((scores >= 0.0) & (scores <= 1.0))
    assert (tmp_path / "validation" / "validation_plots.html").stat().st_size > 0


# ------------------------------------------------------------- checkpoints

@pytest.mark.global_rng
@pytest.mark.parametrize("layers,output_config", [
    ([gf.DenseLayer(4)], BINARY),
    ([gf.DenseLayer(4)], NORMAL),  # IndependentFoldedNormal and folded_normal_nll
    ([gf.WhitenPassLayer(), gf.DenseLayer(4)], BINARY),  # gf.Reshape
], ids=["binary", "normal", "whitenpass"])
def test_a_saved_model_loads_with_its_weights(tmp_path, layers, output_config):
    # GF-B206: the custom layers and the loss were unregistered, so the load
    # failed, and Model.load returned an untrained model instead.
    trained = model(tmp_path, layers, output_config)
    save(trained, tmp_path)
    loaded = load(tmp_path, layers, output_config)
    assert loaded.loaded
    for a, b in zip(trained.model.get_weights(), loaded.model.get_weights()):
        np.testing.assert_array_equal(a, b)
    out_a, out_b = predict(trained), predict(loaded)
    for name in out_a:
        np.testing.assert_allclose(out_a[name], out_b[name], rtol=1e-6)


@pytest.mark.global_rng
def test_a_checkpoint_that_cannot_load_is_an_error(tmp_path):
    # GF-B206: a failed load logged an error and returned the untrained blueprint.
    gf.ensure_directory_exists(tmp_path / "tiny")
    gf.model_file(tmp_path / "tiny").write_bytes(b"not a keras archive")
    with pytest.raises(Exception) as raised:
        load(tmp_path, [gf.DenseLayer(4)])
    assert "model.keras" in str(raised.value)


def test_every_gravyflow_keras_object_is_registered():
    # The class of GF-B206: a custom Keras object without registration saves
    # but cannot load.
    import gravyflow.src.model.examples  # noqa: F401 (defines the examples' layers)

    def subclasses(cls):
        for sub in cls.__subclasses__():
            yield sub
            yield from subclasses(sub)

    bases = (keras.layers.Layer, keras.losses.Loss, keras.optimizers.schedules.LearningRateSchedule)
    ours = {cls for base in bases for cls in subclasses(base) if cls.__module__.startswith("gravyflow")}
    assert ours, "no gravyflow Keras objects found"
    unregistered = sorted(cls.__qualname__ for cls in ours
                          if keras.saving.get_registered_object(keras.saving.get_registered_name(cls)) is not cls)
    assert not unregistered, f"unregistered Keras objects: {unregistered}"
    from gravyflow.src.model import model as model_module
    losses = [f for name, f in inspect.getmembers(model_module, inspect.isfunction) if name.endswith("_nll")]
    assert losses
    assert all(keras.saving.get_registered_object(keras.saving.get_registered_name(f)) is f for f in losses)
