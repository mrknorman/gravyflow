"""
The data versions: the numerics that make served samples, in every fingerprint and
every cache key.

Every data obtainer's specification carries ``DATA_VERSIONS``, so a checkpoint, a
validation or a training run made before a version changed is refused after it. The
store keeps raw strain, so a change of the resampler cuts no tile; the store's own
format is a data version, and names its root. A transient cache of an earlier format
is never read.
"""

from contextlib import contextmanager
from unittest import mock

import h5py
import numpy as np
import pytest
import scipy

import gravyflow as gf
from gravyflow.src.dataset.acquisition import base
from gravyflow.src.dataset.acquisition.base import BaseDataObtainer
from gravyflow.src.dataset.store import STORE_FORMAT
from gravyflow.src.dataset.acquisition.versions import DATA_VERSIONS
from gravyflow.src.dataset.features.glitch_cache import (
    TRANSIENT_CACHE_FORMAT, TransientCache, generate_transient_cache_path,
)
from gravyflow.src.model.examples.base import ValidationDatasetConfig, validation_dataset_args
from gravyflow.src.model.examples.recipe import RunDirectory, TrainingRun, TrainingSchedule
from gravyflow.src.validate import StaleCheckpointError, ValidationConfig, Validator
from gravyflow.src.validate.bank import ValidationBank
from gravyflow.src.validate.config import with_run_record
from gravyflow.src.validate.provenance import config_fingerprint, model_fingerprint
from gravyflow.tests import test_examples_recipe as recipe_tests
from gravyflow.tests.conftest import refuse_internet
from gravyflow.tests import test_segment_provenance as noise_reads

#: The transient cache format gravyflow wrote before its resampler was the FIR
#: (``utils/resample.py``).
TRANSIENT_FORMAT_BEFORE_THE_FIR = 3


@contextmanager
def before_the_cut():
    """Specifications as gravyflow made them before it had data versions."""
    specification = BaseDataObtainer.specification

    def without_versions(self):
        made = dict(specification(self))
        made.pop("data_versions", None)
        return made

    with mock.patch.object(BaseDataObtainer, "specification", without_versions):
        yield


def real_noise():
    return gf.NoiseObtainer(noise_type=gf.NoiseType.REAL, ifos=[gf.IFO.H1],
                            ifo_data_obtainer=noise_reads.obtainer())


def test_every_obtainer_specification_carries_the_data_versions():
    assert dict(DATA_VERSIONS) == {"resampler": "fir-1", "scipy": scipy.__version__, "store": STORE_FORMAT}
    with pytest.raises(TypeError):
        DATA_VERSIONS["resampler"] = "another"  # one home, read-only
    transient = gf.TransientDataObtainer(data_quality=gf.DataQuality.BEST,
                                         data_labels=[gf.DataLabel.GLITCHES],
                                         observing_runs=gf.ObservingRun.O3)
    for obtainer in (noise_reads.obtainer(), transient):
        assert obtainer.specification()["data_versions"] == dict(DATA_VERSIONS)


def test_a_data_version_changes_every_fingerprint(monkeypatch):
    references = [config_fingerprint(noise_reads.obtainer()), config_fingerprint(real_noise())]
    monkeypatch.setattr(base, "DATA_VERSIONS", {**DATA_VERSIONS, "resampler": "another"})
    changed = [config_fingerprint(noise_reads.obtainer()), config_fingerprint(real_noise())]
    assert all(a != b for a, b in zip(references, changed))
    # White noise reads no strain: no data version is in its fingerprint.
    white = gf.NoiseObtainer(noise_type=gf.NoiseType.WHITE)
    assert "data_versions" not in str(white.specification())


class TestCachesAreCut:

    def test_the_store_format_is_a_data_version_and_names_the_root(self, tmp_path):
        assert DATA_VERSIONS["store"] == STORE_FORMAT
        with noise_reads.fetching(noise_reads.GridFetch()):
            noise_reads.batches(noise_reads.obtainer(), gf.SamplingMode.GRID, num_batches=1)
        store = gf.src.dataset.acquisition.gwosc_data.user_cache_dir("store")
        assert sorted(path.name for path in store.iterdir()) == [f"v{STORE_FORMAT}"]

    def test_a_transient_cache_from_before_the_cut_is_never_read(self, tmp_path):
        path = generate_transient_cache_path(data_directory=tmp_path, prefix="glitch", source=gf.GWOSC())
        assert TRANSIENT_CACHE_FORMAT > TRANSIENT_FORMAT_BEFORE_THE_FIR
        assert path.name.startswith(f"glitch_cache_v{TRANSIENT_CACHE_FORMAT}_")

        cache = TransientCache(path, mode="w")  # a file of the old format at the new name
        cache.initialize_file(4096.0, 32.0, 32.0, ["H1", "L1"], 2, 8, 8)
        with h5py.File(path, "a") as f:
            f["glitches"].attrs["version"] = TRANSIENT_FORMAT_BEFORE_THE_FIR
        with pytest.raises(ValueError, match=f"format {TRANSIENT_FORMAT_BEFORE_THE_FIR}; "
                                             f"this gravyflow reads only format {TRANSIENT_CACHE_FORMAT}"):
            TransientCache(path).get_metadata()


# ----------------------------------------------- stale checkpoints are refused

def logit_model():
    import keras
    inputs = keras.Input(shape=(1, 2048), name="WHITENED_ONSOURCE")
    outputs = keras.layers.Dense(2, name="output")(keras.layers.Flatten()(inputs))
    return keras.Model({"WHITENED_ONSOURCE": inputs}, outputs)


def write_validation(path, model, dataset_args, config):
    """A finished validation's checkpoint, as ``Validator.validate`` saves one."""
    saved = Validator("tiny", config, dataset_args)
    saved.model_sha256 = model_fingerprint(model)
    saved.config_sha256 = config_fingerprint(saved.config, dataset_args)
    saved.far_scores, saved.noise_gps_times = np.zeros(4), np.zeros(4)
    saved.efficiency_data = {"snrs": np.zeros(4), "scores": np.zeros(4)}
    saved.save(path)


@pytest.mark.global_rng
def test_validator_refuses_a_checkpoint_from_before_the_cut(tmp_path, monkeypatch):
    model = logit_model()
    dataset_args = validation_dataset_args(noise_type=gf.NoiseType.REAL,
                                           config=ValidationDatasetConfig(sample_rate_hertz=2048.0))
    config = ValidationConfig(num_examples=64, batch_size=32)

    def fail(self):
        raise AssertionError("a matching checkpoint must not be scored again")

    monkeypatch.setattr(ValidationBank, "generate_noise", fail)
    monkeypatch.setattr(ValidationBank, "generate", fail)

    write_validation(tmp_path / "now.h5", model, dataset_args, config)
    resumed = Validator.validate(model, "tiny", dataset_args, config, tmp_path / "now.h5")
    np.testing.assert_array_equal(resumed.far_scores, np.zeros(4))  # the control: resumed

    with before_the_cut():
        write_validation(tmp_path / "before.h5", model, dataset_args, config)
    with pytest.raises(StaleCheckpointError, match="another validation configuration"):
        Validator.validate(model, "tiny", dataset_args, config, tmp_path / "before.h5")


@pytest.mark.global_rng
def test_the_validator_refuses_before_it_reads_any_data(tmp_path, monkeypatch):
    """
    The validator refuses from its arguments alone, before it builds a dataset or
    fetches any strain: with every connection refused and recorded (with --network
    too), a model selected on the split it would score, a model trained on another
    split, and a checkpoint from before the cut are each refused on real-noise
    arguments, and nothing tried to connect.
    """
    attempts = []
    refuse_internet(monkeypatch, attempts, loopback=False)
    model = logit_model()
    dataset_args = validation_dataset_args(noise_type=gf.NoiseType.REAL,
                                           config=ValidationDatasetConfig(sample_rate_hertz=2048.0))
    config = ValidationConfig(num_examples=64, batch_size=32)

    def trained(fingerprint, selection_split):
        return with_run_record(model, {"split": {"fingerprint": fingerprint, "selection_split": selection_split}})

    with pytest.raises(ValueError, match="selected on the 'test' split"):
        Validator.validate(trained(gf.DEFAULT_SPLIT.fingerprint(), "test"), "tiny", dataset_args, config)
    with pytest.raises(ValueError, match="SplitSpec"):
        Validator.validate(trained(gf.SplitSpec(version="fold-2").fingerprint(), "validate"), "tiny",
                           dataset_args, config)
    held_out = trained(gf.DEFAULT_SPLIT.fingerprint(), "validate")
    with before_the_cut():
        write_validation(tmp_path / "before.h5", held_out, dataset_args, config)
    with pytest.raises(StaleCheckpointError, match="another validation configuration"):
        Validator.validate(held_out, "tiny", dataset_args, config, tmp_path / "before.h5")
    assert attempts == []


def test_training_run_refuses_a_checkpoint_from_before_the_cut(tmp_path):
    def run(path, epochs):
        return TrainingRun(RunDirectory(path),
                           TrainingSchedule(epochs=epochs, steps_per_epoch=3, validation_batches=2,
                                            patience=100),
                           seed=7, description={"model": "tiny", "noise": real_noise()})

    validation = recipe_tests.materialise(recipe_tests.CountingBatches(99, group="validate"), 2)
    run(tmp_path / "now", 1).train(recipe_tests.tiny_model, recipe_tests.CountingBatches, validation, verbose=0)
    run(tmp_path / "now", 2).train(recipe_tests.tiny_model, recipe_tests.CountingBatches, validation, verbose=0)
    assert RunDirectory(tmp_path / "now").state["epoch"] == 1  # the control: resumed

    with before_the_cut():
        run(tmp_path / "before", 1).train(recipe_tests.tiny_model, recipe_tests.CountingBatches, validation,
                                          verbose=0)
    with pytest.raises(StaleCheckpointError, match="another run configuration"):
        run(tmp_path / "before", 2).train(recipe_tests.tiny_model, recipe_tests.CountingBatches, validation,
                                          verbose=0)
