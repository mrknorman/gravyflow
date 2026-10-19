"""
Building a gf.Model draws from its seed, never from a global random state.

Keras seeds every layer built without a seed (its initialisers, a dropout
layer's seed generator) from Python's global random. A model that builds under
no draw scope changes that state and gets weights that depend on it; built
under `keras_draws` with a stream of its seed, it does neither. None of these
tests carries ``global_rng``: the guard in conftest fails each one whose build
moves NumPy's, Python's or Keras' global random state, and the first test says
so itself.
"""

import copy
import random

import numpy as np

import gravyflow as gf

SAMPLES = 64
INPUT = {"name": gf.ReturnVariables.ONSOURCE.name, "shape": (1, SAMPLES)}
BINARY = {"name": gf.ReturnVariables.INJECTION_PRESENT.name, "type": "binary"}


def build(tmp_path, seed=None):
    return gf.Model(
        name="tiny", layers=[gf.DenseLayer(4), gf.DropLayer(0.5), gf.DenseLayer(3)],
        optimizer="sgd", loss=None, input_configs=[INPUT], output_config=BINARY,
        model_path=tmp_path / "tiny", seed=seed,
    )


def weights(model):
    return [np.asarray(w) for w in model.model.get_weights()]


def seed_states(model):
    # Dropout's seed generators: their state is drawn when the layer is built.
    return [np.asarray(v.value) for v in model.model.variables if v.name == "seed_generator_state"]


def same(a, b):
    return len(a) == len(b) and all(np.array_equal(x, y) for x, y in zip(a, b))


def test_building_a_model_leaves_python_randoms_state_alone(tmp_path):
    before = random.getstate()
    build(tmp_path)
    assert random.getstate() == before


def test_two_builds_of_one_seed_are_the_same_model(tmp_path):
    a, b = build(tmp_path / "a", seed=5), build(tmp_path / "b", seed=5)
    assert len(weights(a)) > 0 and same(weights(a), weights(b))
    assert len(seed_states(a)) > 0 and same(seed_states(a), seed_states(b))


def test_the_default_seed_is_one_model_too(tmp_path):
    assert same(weights(build(tmp_path / "a")), weights(build(tmp_path / "b")))


def test_another_seed_is_another_model(tmp_path):
    a, b = build(tmp_path / "a", seed=5), build(tmp_path / "b", seed=6)
    assert not same(weights(a), weights(b))
    assert not same(seed_states(a), seed_states(b))


def test_a_build_does_not_follow_the_global_random_state(tmp_path):
    state = random.getstate()
    try:
        random.seed(1)
        a = build(tmp_path / "a", seed=5)
        random.seed(2)
        b = build(tmp_path / "b", seed=5)
    finally:
        random.setstate(state)
    assert same(weights(a), weights(b)) and same(seed_states(a), seed_states(b))


def test_keras_draws_restores_the_state_and_follows_its_seed():
    import keras
    from gravyflow.src.rng import keras_draws

    def draw(seed):
        with keras_draws(seed):
            return np.asarray(keras.initializers.GlorotUniform()((2, 3)))

    before = random.getstate()
    a, b, c = draw(1), draw(1), draw(2)
    assert random.getstate() == before
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_loading_a_checkpoint_leaves_python_randoms_state_alone(tmp_path):
    # Keras seeds a loaded dropout layer's generator from the global random too.
    built = build(tmp_path, seed=5)
    gf.ensure_directory_exists(tmp_path / "tiny")
    built.model.save(gf.model_file(tmp_path / "tiny"))
    before = random.getstate()
    loaded = gf.Model.load(
        "tiny", tmp_path / "tiny", input_configs=[INPUT], output_config=BINARY,
        hidden_layers=[gf.DenseLayer(4), gf.DropLayer(0.5), gf.DenseLayer(3)], seed=5,
    )
    assert loaded.loaded and random.getstate() == before
    assert same(weights(loaded), weights(built))


def test_a_hyperparameter_draws_from_its_seed_and_changes_no_distribution():
    distribution = gf.Distribution(min_=1.0, max_=9.0, type_=gf.DistributionType.UNIFORM)
    before = copy.deepcopy(distribution)
    first, second = gf.HyperParameter(distribution, seed=5), gf.HyperParameter(distribution, seed=5)
    assert first.value == second.value != gf.HyperParameter(distribution, seed=6).value
    first.reseed(6)
    first.randomize()
    assert first.value == gf.HyperParameter(distribution, seed=6).value
    assert distribution == before
