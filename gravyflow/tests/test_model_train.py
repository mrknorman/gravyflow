"""
gf.Model trains on its dataset with one definition of an epoch's length: the
training dataset's, which the model sizes from its training config.
"""

import pytest

import gravyflow as gf

SAMPLE_RATE, ONSOURCE, OFFSOURCE, BATCH = 256.0, 1.0, 2.0, 4

TRAINING = dict(num_examples_per_epoc=3 * BATCH, num_validation_examples=64, max_epochs=2, patience=10)


def dataset_args(**extra):
    wnb = gf.WaveformGenerator.load(
        path=gf.tests.PATH / "example_injection_parameters" / "wnb_parameters.json",
        scaling_method=gf.ScalingMethod(
            value=gf.Distribution(min_=10.0, max_=20.0, type_=gf.DistributionType.UNIFORM),
            type_=gf.ScalingTypes.SNR,
        ),
        network=[gf.IFO.L1],
    )
    wnb.injection_chance = 0.5
    return dict(
        group="train",
        noise_obtainer=gf.NoiseObtainer(noise_type=gf.NoiseType.WHITE, ifos=[gf.IFO.L1]),
        sample_rate_hertz=SAMPLE_RATE,
        onsource_duration_seconds=ONSOURCE,
        offsource_duration_seconds=OFFSOURCE,
        crop_duration_seconds=0.5,
        num_examples_per_batch=BATCH,
        waveform_generators=[wnb],
        input_variables=[gf.ReturnVariables.ONSOURCE],
        output_variables=[gf.ReturnVariables.INJECTION_PRESENT],
        **extra,
    )


def tiny_model(tmp_path, **extra):
    return gf.Model(
        name="tiny",
        layers=[gf.DenseLayer(4)],
        optimizer="sgd",
        loss=None,
        input_configs=[{
            "name": gf.ReturnVariables.ONSOURCE.name,
            "shape": (1, int(SAMPLE_RATE * ONSOURCE)),
        }],
        output_config={"name": gf.ReturnVariables.INJECTION_PRESENT.name, "type": "binary"},
        training_config=dict(TRAINING),
        dataset_args=dataset_args(**extra),
        batch_size=BATCH,
        model_path=tmp_path / "tiny",
    )


class TestOneEpochLength:
    """Each test builds a Keras model, whose initialisers draw from Keras' global seed."""

    @pytest.mark.global_rng
    def test_the_dataset_takes_its_epoch_length_from_the_training_config(self, tmp_path):
        assert len(tiny_model(tmp_path).training_dataset) == 3

    @pytest.mark.global_rng
    def test_the_same_explicit_epoch_length_is_accepted(self, tmp_path):
        assert len(tiny_model(tmp_path, steps_per_epoch=3).training_dataset) == 3

    @pytest.mark.global_rng
    def test_a_different_explicit_epoch_length_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="steps_per_epoch=5"):
            tiny_model(tmp_path, steps_per_epoch=5)

    @pytest.mark.global_rng
    def test_train_runs_its_epochs(self, tmp_path):
        model = tiny_model(tmp_path)
        with pytest.raises(ValueError, match="validates on the 'validate' split"):
            model.train(validate_args=dataset_args(steps_per_epoch=2))
        assert model.train(validate_args={**dataset_args(steps_per_epoch=2), "group": "validate"})
        history = model.metrics[0].history
        assert len(history["loss"]) == TRAINING["max_epochs"]
        assert len(history["val_loss"]) == TRAINING["max_epochs"]
