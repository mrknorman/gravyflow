"""
The data versions: the numerics that make served samples, in every fingerprint and
every cache key.

Every data obtainer's specification carries ``DATA_VERSIONS``, so a checkpoint, a
validation or a training run made before a version changed is refused after it. Caches
of strain the resampler made are cut with it: a real-noise segment stored before the
resampler changed is fetched again, and a transient cache of an earlier format is
never read.
"""

from contextlib import closing, contextmanager
from unittest import mock

import h5py
import numpy as np
import pytest
import scipy

import gravyflow as gf
from gravyflow.src.dataset.acquisition import base
from gravyflow.src.dataset.acquisition.base import BaseDataObtainer, segment_key
from gravyflow.src.dataset.acquisition.versions import DATA_VERSIONS
from gravyflow.src.dataset.features.glitch_cache import (
    TRANSIENT_CACHE_FORMAT, TransientCache, generate_transient_cache_path,
)
from gravyflow.src.model.examples.base import ValidationDatasetConfig, validation_dataset_args
from gravyflow.src.model.examples.recipe import RunDirectory, TrainingRun, TrainingSchedule
from gravyflow.src.validate import StaleCheckpointError, ValidationConfig, Validator
from gravyflow.src.validate.bank import ValidationBank
from gravyflow.src.validate.provenance import config_fingerprint, model_fingerprint
from gravyflow.tests import test_examples_recipe as recipe_tests
from gravyflow.tests import test_segment_provenance as noise_reads

#: The segment layout and transient cache format gravyflow wrote before its resampler
#: was the FIR (``utils/resample.py``).
LAYOUT_BEFORE_THE_FIR = "segment-layout-3"
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
    assert dict(DATA_VERSIONS) == {"resampler": "fir-1", "scipy": scipy.__version__}
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

    def test_a_segment_stored_before_the_cut_is_fetched_again(self, tmp_path, monkeypatch, caplog):
        (start, end), ifo = noise_reads.ROW[0], noise_reads.IFOS[0]
        key = segment_key(ifo, start, end)
        path = noise_reads.obtainer().generate_file_path(noise_reads.RATE, "train", tmp_path)

        with monkeypatch.context() as before:  # stored by the code before the cut
            before.setattr(base, "SEGMENT_LAYOUT_VERSION", LAYOUT_BEFORE_THE_FIR)
            stale = noise_reads.obtainer(cache_segments=True)
            stale_path = stale.generate_file_path(noise_reads.RATE, "train", tmp_path)
            with noise_reads.fetching(noise_reads.GridFetch()):
                noise_reads.batches(stale, gf.SamplingMode.GRID, num_batches=1)
        with closing(h5py.File(stale_path, "r")) as stored:
            assert stored[key].attrs["layout"] == LAYOUT_BEFORE_THE_FIR

        assert path != stale_path, "the segment file is keyed by its layout"
        fresh = noise_reads.obtainer(cache_segments=True)
        fresh.file_path = stale_path  # even at the old file's path
        with noise_reads.fetching(noise_reads.GridFetch()) as fetch:
            fresh.get_segment(start, end, noise_reads.RATE, ifo, key)
        assert fetch.calls == [(ifo.name, start, end)], "a segment stored before the cut was served"
        assert "fetching it again" in caplog.text

    def test_a_segment_file_is_keyed_by_the_data_versions(self, tmp_path, monkeypatch):
        path = noise_reads.obtainer().generate_file_path(noise_reads.RATE, "train", tmp_path)
        monkeypatch.setattr(base, "DATA_VERSIONS", {**DATA_VERSIONS, "scipy": "another"})
        assert noise_reads.obtainer().generate_file_path(noise_reads.RATE, "train", tmp_path) != path

    def test_a_transient_cache_from_before_the_cut_is_never_read(self, tmp_path):
        path = generate_transient_cache_path(data_directory=tmp_path, prefix="glitch")
        assert TRANSIENT_CACHE_FORMAT > TRANSIENT_FORMAT_BEFORE_THE_FIR
        assert path.name == f"glitch_cache_v{TRANSIENT_CACHE_FORMAT}.h5"

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
