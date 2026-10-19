"""
gf.Model trains on its dataset with one definition of an epoch's length: the
training dataset's, which the model sizes from its training config.
"""

import warnings

import pytest

import gravyflow as gf
from gravyflow.src.validate.provenance import StaleCheckpointError

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
    """A gf.Model's training dataset takes its epoch's length from the training config."""

    def test_the_dataset_takes_its_epoch_length_from_the_training_config(self, tmp_path):
        assert len(tiny_model(tmp_path).training_dataset) == 3

    def test_the_same_explicit_epoch_length_is_accepted(self, tmp_path):
        assert len(tiny_model(tmp_path, steps_per_epoch=3).training_dataset) == 3

    def test_a_different_explicit_epoch_length_is_refused(self, tmp_path):
        with pytest.raises(ValueError, match="steps_per_epoch=5"):
            tiny_model(tmp_path, steps_per_epoch=5)

    def test_train_runs_its_epochs(self, tmp_path):
        model = tiny_model(tmp_path)
        with pytest.raises(ValueError, match="validates on the 'validate' split"):
            model.train(validate_args=dataset_args(steps_per_epoch=2))
        assert model.train(validate_args={**dataset_args(steps_per_epoch=2), "group": "validate"})
        history = model.metrics[0].history
        assert len(history["loss"]) == TRAINING["max_epochs"]
        assert len(history["val_loss"]) == TRAINING["max_epochs"]


class TestResume:
    """force_retrain=False resumes the run in the model's directory, and never discards it."""

    @pytest.mark.parametrize("record", ["kept", "removed"])
    def test_a_checkpoint_with_no_history_is_kept_and_refused(self, tmp_path, record):
        # A run whose history is empty cannot be resumed; starting a new one would
        # delete its checkpoint (or overwrite it at the first epoch). Only
        # force_retrain=True may.
        validate = {**dataset_args(steps_per_epoch=2), "group": "validate"}
        model = tiny_model(tmp_path)
        assert model.train(validate_args=validate)
        checkpoint = gf.model_file(model.model_path)
        saved = checkpoint.read_bytes()
        gf.save_dict_to_hdf5({}, model.model_path / "history.hdf5", True)
        if record == "removed":
            (model.model_path / "run.json").unlink()
        with pytest.raises(StaleCheckpointError, match="force_retrain=True"):
            tiny_model(tmp_path).train(validate_args=validate, force_retrain=False)
        assert checkpoint.read_bytes() == saved
        assert tiny_model(tmp_path).train(validate_args=validate, force_retrain=True)


class TestPlatform:

    def test_training_on_multi_threaded_cpu_kernels_warns(self, tmp_path, monkeypatch):
        # gf.Model.train ran on multi-threaded CPU kernels without a word, unlike
        # a recipe run: such a run is not reproducible from its seed.
        from gravyflow.src.rng import platform

        validate = {**dataset_args(steps_per_epoch=2), "group": "validate"}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert tiny_model(tmp_path / "single").train(validate_args=validate)
        assert not [w for w in caught if "XLA_FLAGS" in str(w.message)]
        monkeypatch.setattr(platform, "pure", lambda backend=None: False)
        with pytest.warns(UserWarning, match="XLA_FLAGS=--xla_cpu_multi_thread_eigen=false"):
            assert tiny_model(tmp_path / "multi").train(validate_args=validate)
