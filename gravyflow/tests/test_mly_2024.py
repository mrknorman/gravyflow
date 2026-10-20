"""
The MLy example (Skliris, Norman & Sutton 2024): its two networks, their data,
the training script, and the rolling Pearson correlation the coherence network
reads.
"""

import importlib
from dataclasses import replace

import keras
import numpy as np
import pytest

import gravyflow as gf
from gravyflow.src.dataset.conditioning.pearson import num_lags
from gravyflow.src.model.examples import MLy, MLyCoherence, MLyCoincidence, MLyConfig
from gravyflow.src.model.examples.mly_2024 import CyclicLearningRate, mly_dataset_args
from gravyflow.src.model.examples.recipe import DetectionBatches
from gravyflow.src.validate import ValidationConfig, Validator

HL = (gf.IFO.H1, gf.IFO.L1)
HLV = (gf.IFO.H1, gf.IFO.L1, gf.IFO.V1)


def small(ifos=HL, **kwargs) -> MLyConfig:
    return replace(MLyConfig(), ifos=ifos, batch_size=4, **kwargs)


def wnb(snr=20.0, injection_chance=1.0):
    return gf.WNBGenerator(
        duration_seconds=0.2, min_frequency_hertz=40.0, max_frequency_hertz=400.0,
        scaling_method=gf.ScalingMethod(
            value=gf.Distribution(value=snr, type_=gf.DistributionType.CONSTANT), type_=gf.ScalingTypes.SNR),
        injection_chance=injection_chance)


def pearson_dataset(ifos, generator, sample_rate_hertz=1024.0, on_dataset=False, **kwargs):
    """A Pearson dataset whose detectors are set on the noise obtainer, or on the dataset."""
    if on_dataset:
        where = dict(ifos=list(ifos), noise_obtainer=gf.NoiseObtainer(noise_type=gf.NoiseType.COLORED))
    else:
        where = dict(noise_obtainer=gf.NoiseObtainer(noise_type=gf.NoiseType.COLORED, ifos=list(ifos)))
    return gf.Dataset(
        sample_rate_hertz=sample_rate_hertz, onsource_duration_seconds=1.0, offsource_duration_seconds=16.0,
        **where,
        waveform_generators=generator, num_examples_per_batch=2, seed=1, steps_per_epoch=1,
        input_variables=[gf.ReturnVariables.ROLLING_PEARSON_ONSOURCE], output_variables=[], group="train",
        **kwargs)


# ------------------------------------------------------- rolling Pearson

def test_rolling_pearson_runs_with_a_generator():
    # The lag range reached rolling_pearson as a JAX array, which a static jit argument cannot be.
    x, _ = pearson_dataset(HL, wnb())[0]
    p = np.asarray(x["ROLLING_PEARSON_ONSOURCE"])
    # H1-L1 is 10.01 ms, 10.25 samples at 1024 Hz: lags -11 to +11.
    assert p.shape == (2, 1, 23) and np.all(np.isfinite(p)) and np.all(np.abs(p) <= 1.0 + 1e-3)


@pytest.mark.parametrize("on_dataset", [False, True])
@pytest.mark.parametrize("ifos,sample_rate_hertz", [(HLV, 1024.0), (HL, 4096.0)])
def test_rolling_pearson_lags_do_not_depend_on_the_generators(ifos, sample_rate_hertz, on_dataset):
    # Without a generator the lag range was a fixed 10 ms, not the detectors' light travel time,
    # so a noise-only batch and an injection batch differed in shape. With the detectors set on
    # the dataset rather than its noise obtainer, it read the obtainer's default L1: no lags.
    shapes = {np.shape(pearson_dataset(ifos, g, sample_rate_hertz, on_dataset)[0][0]
                       ["ROLLING_PEARSON_ONSOURCE"])
              for g in (wnb(), None)}
    lags = num_lags(gf.Network(list(ifos)).max_arrival_time_difference_seconds, sample_rate_hertz)
    pairs = len(ifos) * (len(ifos) - 1) // 2
    assert shapes == {(2, pairs, lags)}


def test_the_dataset_takes_a_lag_range():
    # The lag range was always the detectors' largest light travel time: MLy could not
    # correlate over its paper's 30 ms.
    x, _ = pearson_dataset(HL, wnb(), pearson_max_lag_seconds=0.03)[0]
    # 30 ms is 30.72 samples at 1024 Hz: lags -31 to +31.
    assert np.shape(x["ROLLING_PEARSON_ONSOURCE"]) == (2, 1, 63)


@pytest.mark.parametrize("ifos,pairs", [(HL, 1), (HLV, 3)])
def test_mly_correlates_over_its_papers_30_ms(ifos, pairs):
    config = replace(MLyConfig(), ifos=ifos)
    assert config.correlation_shape == (pairs, num_lags(0.03, config.sample_rate_hertz)) == (pairs, 63)
    args = mly_dataset_args(config, (10.0, 50.0), 0.5, [gf.ReturnVariables.ROLLING_PEARSON_ONSOURCE])
    assert args["pearson_max_lag_seconds"] == 0.03


# ------------------------------------------------------------ the networks

@pytest.mark.global_rng
@pytest.mark.parametrize("model_class", [MLyCoincidence, MLyCoherence])
@pytest.mark.parametrize("ifos", [HL, HLV])
def test_mly_takes_whitened_onsource_and_returns_logits(model_class, ifos):
    config = small(ifos)
    model = model_class.model(config)
    names = [t.name for t in model.inputs]
    expected = ["WHITENED_ONSOURCE"] + (["ROLLING_PEARSON_ONSOURCE"] if model_class is MLyCoherence else [])
    assert sorted(names) == sorted(expected)
    assert model.layers[-1].activation.__name__ == "linear"
    rng = np.random.default_rng(0)
    x = {"WHITENED_ONSOURCE": rng.normal(size=(3, len(ifos), 1024)).astype(np.float16)}
    if model_class is MLyCoherence:
        x["ROLLING_PEARSON_ONSOURCE"] = rng.uniform(-1, 1, size=(3, *config.correlation_shape)).astype(np.float32)
    out = model.predict_on_batch(x)
    assert out.shape == (3, 2) and out.dtype == np.float32


# The parameter counts of the paper's networks, from the authors' training code
# (github.com/VasSkliris/mly-methods-paper, virgo_rescaling/model{1,2}.py) for
# H, L and V: Conv1D k·c_in·f + f, BatchNormalization 4f, Dense n_in·n + n.
#   Model 1: blocks (64, 42, 36), (128, 42, 36), (128, 42, 36), each conv
#     followed by batch normalisation; dense 256, 256 with batch normalisation; 2.
#   Model 2: strain convs 56×32/4, 128×24/2, 214×16/2; correlation convs 64×8,
#     64×5, each followed by batch normalisation; dense 256 with batch
#     normalisation; 2.
@pytest.mark.global_rng
def test_coincidence_layers_match_the_paper():
    model = MLyCoincidence.model(small(HLV))
    assert model.count_params() == 2_471_938
    convs = [layer for layer in model.layers if isinstance(layer, keras.layers.Conv1D)]
    assert [(c.filters, c.kernel_size[0], c.strides[0], c.padding, c.activation.__name__) for c in convs] == [
        (f, k, 1, "same", "relu") for f in (64, 128, 128) for k in (42, 36)]
    # The skip adds each block's first convolution (before its batch
    # normalisation) to the block's output.
    adds = [layer for layer in model.layers if isinstance(layer, keras.layers.Add)]
    assert len(adds) == 3
    for block, add in enumerate(adds):
        assert any(t is convs[2 * block].output for t in add.input)


@pytest.mark.global_rng
def test_coherence_layers_match_the_paper():
    config = small(HLV)
    model = MLyCoherence.model(config)
    names = [f"strain_conv{n}" for n in (1, 2, 3)] + [f"correlation_conv{n}" for n in (1, 2)]
    convs = [model.get_layer(name) for name in names]
    assert len([layer for layer in model.layers if isinstance(layer, keras.layers.Conv1D)]) == 5
    assert [(c.filters, c.kernel_size[0], c.strides[0], c.padding, c.activation.__name__) for c in convs] == [
        (56, 32, 4, "valid", "relu"), (128, 24, 2, "valid", "relu"), (214, 16, 2, "valid", "relu"),
        (64, 8, 1, "valid", "relu"), (64, 5, 1, "valid", "relu")]
    # 713,288 with the paper's 60 lags; the library's ±30 ms at 1024 Hz (30.72
    # samples: lags -31 to +31) is 63 lags, which changes no weight count.
    assert model.count_params() == 713_288
    assert config.correlation_shape == (3, 63)


def test_combined_score_is_the_log_odds_of_the_product():
    rng = np.random.default_rng(1)
    a, b = rng.normal(scale=3, size=(50, 2)), rng.normal(scale=3, size=(50, 2))
    s1, s2 = a[:, 1] - a[:, 0], b[:, 1] - b[:, 0]
    p = 1 / (1 + np.exp(-s1)) / (1 + np.exp(-s2))
    np.testing.assert_allclose(MLy.combined_score(a, b), np.log(p / (1 - p)), rtol=1e-9, atol=1e-9)
    # Where the product of probabilities saturates at 1.0, the log-odds still rank.
    loud = np.array([[0.0, 60.0], [0.0, 80.0]])
    scores = MLy.combined_score(loud, loud)
    assert np.all(np.isfinite(scores)) and scores[1] > scores[0]


def test_cyclic_learning_rate_is_triangular_and_serialises():
    schedule = CyclicLearningRate(base=1e-5, peak=7e-3, half_cycle_steps=10)
    values = [float(schedule(step)) for step in (0, 5, 10, 15, 20, 30)]
    np.testing.assert_allclose(values, [1e-5, (1e-5 + 7e-3) / 2, 7e-3, (1e-5 + 7e-3) / 2, 1e-5, 7e-3], rtol=1e-5)
    again = CyclicLearningRate.from_config(schedule.get_config())
    assert float(again(15)) == pytest.approx(values[3])
    # The paper's half cycle: four passes over 135,000 examples in batches of 16.
    assert MLyConfig().coincidence_half_cycle_steps == 4 * 135_000 // 16


# ---------------------------------------------------------------- data

@pytest.mark.global_rng
@pytest.mark.parametrize("model_class", [MLyCoincidence, MLyCoherence])
def test_mly_trains_on_its_dataset(model_class, tmp_path):
    # The networks read an input named "strain" shaped (samples, detectors),
    # and the coherence dataset passed an argument the dataset rejects.
    config = small(HL)
    batches = DetectionBatches(model_class.dataset(config=config, seed=3, group="train", steps_per_epoch=2))
    x, y = batches[0]
    assert set(x) == {t.name for t in model_class.model(config).inputs} and y.shape == (4, 2)
    model = model_class.compile_model(model_class.model(config), config)
    history = model.fit(batches, epochs=1, verbose=0)
    assert np.isfinite(history.history["loss"][0])
    model.save(tmp_path / "model.keras")  # the recipe's checkpoints: the optimiser and its schedule
    loaded = keras.models.load_model(tmp_path / "model.keras")
    np.testing.assert_allclose(loaded.predict_on_batch(x), model.predict_on_batch(x), rtol=1e-5, atol=1e-5)


def test_mly_dataset_follows_the_paper():
    config = MLyConfig()
    for model_class, snr, chance in [(MLyCoincidence, (12.0, 30.0), 1 / 3), (MLyCoherence, (10.0, 50.0), 0.5)]:
        dataset = model_class.dataset(config=config, group="train", steps_per_epoch=1)
        generator = dataset.waveform_generators[0]
        value = generator.scaling_method.value
        assert generator.scaling_method.type_ is gf.ScalingTypes.SNR
        assert (value.min_, value.max_) == snr and generator.injection_chance == pytest.approx(chance)
        assert (generator.duration_seconds.min_, generator.duration_seconds.max_) == (0.05, 0.9)
        # The band: 40-480 Hz wide, inside 20-512 Hz.
        lo, hi = generator.min_frequency_hertz, generator.max_frequency_hertz
        assert hi.min_ - lo.max_ == 40.0 and hi.max_ - lo.min_ == 480.0 and lo.min_ >= 20.0 and hi.max_ < 512.0
        assert dataset.sample_rate_hertz == 1024.0 and dataset.onsource_duration_seconds == 1.0
        assert dataset.offsource_duration_seconds == 16.0
        assert list(dataset.noise_obtainer.ifos) == list(HLV)


@pytest.mark.global_rng
def test_validator_scores_a_burst_model():
    # The validator asked every injection set for CBC masses, which a WNB set lacks.
    config = small(HL)
    model = MLyCoherence.model(config)
    # Stream mode scores each injection on a stretch of injection_stretch_seconds (40 s),
    # placed only where the background scored every window its found test reads: 192
    # windows at the default quarter-window stride span enough of one stream for that.
    validation = ValidationConfig(num_examples=192, batch_size=32, snr_range=(0.0, 20.0))
    validator = Validator.validate(
        model=model, name="mly", dataset_args=MLyCoherence.validation_dataset_args(config), config=validation)
    assert validator.far_scores.shape == (validation.num_examples,)
    assert validator.efficiency_data["scores"].shape == (validation.num_examples,)
    assert "mass1" not in validator.efficiency_data


# ------------------------------------------------------------- the script

def test_train_script_settings():
    train = importlib.import_module("gravyflow.src.model.examples.train_mly_2024")
    with pytest.raises(SystemExit):
        train.parse_args([])
    cpu = train.parse_args(["--output-dir", "unused", "--model", "coherence"])
    config, schedule = train.configure(cpu)
    assert config.ifos == HL and schedule == train.TrainingSchedule.cpu()
    gpu = train.parse_args(["--output-dir", "unused", "--model", "coincidence", "--preset", "gpu"])
    config, schedule = train.configure(gpu)
    # The paper: 135,000 training examples in batches of 16, 15 epochs (model 1).
    assert config.ifos == HLV and schedule.epochs == 15 and schedule.steps_per_epoch == 8438


# ------------------------------------------------------------ real events

@pytest.mark.global_rng
def test_real_event_inputs_feed_the_coherence_network():
    # Real events were given only the whitened onsource, the onsource and the offsource,
    # so a model reading ROLLING_PEARSON_ONSOURCE raised KeyError. They are now prepared
    # by the background's dataset, over the model's own lag range.
    from gravyflow.src.validate.bank import ValidationBank

    config = small(HLV)
    model = MLyCoherence.model(config)
    bank = ValidationBank(model, MLyCoherence.validation_dataset_args(config), ValidationConfig())
    rng = np.random.default_rng(2)
    on = rng.normal(size=(3, 3, 2048)).astype(np.float32)
    off = rng.normal(size=(3, 3, 16 * 1024)).astype(np.float32)
    x = bank.stage_dataset("far").model_inputs(on, off)
    assert sorted(x) == ["ROLLING_PEARSON_ONSOURCE", "WHITENED_ONSOURCE"]
    assert x["ROLLING_PEARSON_ONSOURCE"].shape == (3, *config.correlation_shape)
    assert x["WHITENED_ONSOURCE"].shape == (3, 3, 1024)
    assert np.all(np.isfinite(logit_scores_of(model, x)))


def logit_scores_of(model, x):
    from gravyflow.src.validate.bank import logit_scores
    return logit_scores(model.predict(x, verbose=0))


@pytest.mark.parametrize("shape,n,expected", [
    ((1, 1, 2), 1, [0.0]),      # one example, one generator, two detectors
    ((1, 2), 1, [0.0]),
    ((4, 2), 4, [0, 2, 4, 6]),   # (B, I)
    ((4, 1, 2), 4, [0, 2, 4, 6]),  # (B, G, I)
])
def test_per_detector_labels_give_the_first_detector(shape, n, expected):
    from gravyflow.src.validate.bank import per_example
    value = np.arange(np.prod(shape), dtype=float).reshape(shape)
    np.testing.assert_array_equal(per_example(value, n, "ct", per_detector=True), expected)
