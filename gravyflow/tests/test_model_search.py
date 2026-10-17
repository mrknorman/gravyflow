"""
The model search scores each model of a generation from the files its training
left: its history and its validator's results.
"""

import copy

import numpy as np
import pytest

import gravyflow as gf
from gravyflow.src.model.model import Population, calculate_fitness


def _population(path, members):
    # The search's state alone: building a Population draws and saves genomes.
    population = Population.__new__(Population)
    population.population_directory_path, population.num_population_members = path, members
    return population


def _trained(directory, far_scores, scores, snrs, val_loss):
    validator = gf.Validator("model", dataset_args={"onsource_duration_seconds": 1.0})
    validator.far_scores = np.asarray(far_scores, np.float32)
    validator.efficiency_data = {"scores": np.asarray(scores, np.float32), "snrs": np.asarray(snrs, np.float32)}
    validator.save(directory / "validation_data.h5")
    gf.save_dict_to_hdf5({"val_loss": np.asarray(val_loss)}, directory / "history.hdf5", True)
    return validator


def test_a_generation_is_scored_from_its_validation_results(tmp_path):
    generation = tmp_path / "generation_1"
    rng = np.random.default_rng(3)
    snrs = np.linspace(0.0, 15.0, 200)
    scores = snrs / 15.0 + rng.normal(0.0, 0.1, 200)
    validator = _trained(generation / "model_3", rng.uniform(0.0, 0.9, 5000), scores, snrs, [0.6, 0.25, 0.3])
    (generation / "model_2").mkdir()  # trained, never validated

    fitnesses = _population(tmp_path, 2).load_and_calculate_fitness(1)

    threshold = validator.background().threshold_for_far(1e-3).threshold
    expected = calculate_fitness(scores[snrs > 8].astype(np.float32), threshold, {"val_loss": [0.6, 0.25, 0.3]})
    assert expected > 0
    assert fitnesses == pytest.approx([0.0, expected])


def test_the_search_draws_from_its_seed_and_changes_no_distribution():
    # The global RNG guard (conftest) fails this test if a draw came from a global RNG.
    distribution = gf.Distribution(min_=1.0, max_=9.0, type_=gf.DistributionType.UNIFORM)
    before = copy.deepcopy(distribution)
    first, second = gf.HyperParameter(distribution, seed=5), gf.HyperParameter(distribution, seed=5)
    for parameter in (first, second):
        parameter.mutate(1.0)
    assert first.value == second.value != gf.HyperParameter(distribution, seed=6).value
    assert distribution == before


def test_a_broken_result_is_an_error(tmp_path):
    model = tmp_path / "generation_0" / "model_0"
    model.mkdir(parents=True)
    (model / "validation_data.h5").write_bytes(b"not hdf5")
    with pytest.raises(OSError, match="file signature not found"):
        _population(tmp_path, 1).load_and_calculate_fitness(0)


def test_an_injection_gene_sets_a_new_scaling_distribution():
    # The genes once mutated the generator's scaling Distribution in place,
    # which a frozen Distribution refuses; they build a new one.
    from gravyflow.src.model.genetics import HyperInjectionGenerator, HyperParameter

    original = gf.Distribution(min_=8.0, max_=20.0, type_=gf.DistributionType.UNIFORM)
    generator = gf.WNBGenerator(
        scaling_method=gf.ScalingMethod(original, gf.ScalingTypes.SNR),
        duration_seconds=0.2, min_frequency_hertz=40.0, max_frequency_hertz=200.0,
    )
    hyper = HyperInjectionGenerator(
        min_=HyperParameter(5.0), max_=HyperParameter(30.0), mean=HyperParameter(10.0),
        std=HyperParameter(2.0), distribution=HyperParameter("uniform"), chance=HyperParameter(0.5),
        generator=HyperParameter(generator),
    )
    built = hyper.return_generator()
    value = built.scaling_method.value
    assert (value.min_, value.max_, value.mean, value.std) == (5.0, 30.0, 10.0, 2.0)
    assert built.injection_chance == 0.5
    assert original == gf.Distribution(min_=8.0, max_=20.0, type_=gf.DistributionType.UNIFORM)
