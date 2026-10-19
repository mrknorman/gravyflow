import h5py
import numpy as np
import pytest

import gravyflow as gf
from gravyflow.tests import _helpers


def test_io_utils(tmp_path):
    """ensure_directory_exists makes every missing level."""
    target_dir = tmp_path / "subdir" / "subsubdir"
    assert not target_dir.exists()

    gf.ensure_directory_exists(target_dir)

    assert target_dir.is_dir()


# ---------------------------------------------------------------------------
# Goldens
# ---------------------------------------------------------------------------

@pytest.fixture
def goldens(tmp_path, monkeypatch):
    """A temporary goldens directory, and no blessing pending."""
    monkeypatch.setattr(_helpers, "GOLDENS", tmp_path)
    monkeypatch.setattr(_helpers, "pending_blessings", [])
    monkeypatch.delenv(_helpers.BLESS_ENVIRONMENT_VARIABLE, raising=False)
    return tmp_path


def _arrays(seed=7):
    rng = np.random.default_rng(seed)
    return {"strain": rng.normal(scale=1e-21, size=64), "labels": np.arange(4)}


def _bless(monkeypatch, arrays, path, **declared):
    """Bless as ``gravyflow.tests.bless`` does: collect in the test, write after it."""
    monkeypatch.setenv(_helpers.BLESS_ENVIRONMENT_VARIABLE, _helpers._current_test())
    _helpers.assert_matches_golden(arrays, path, **declared)
    monkeypatch.delenv(_helpers.BLESS_ENVIRONMENT_VARIABLE)
    (blessing,) = _helpers.pending_blessings
    _helpers.write_golden(blessing["path"], blessing["arrays"], {
        "test": blessing["test"], "source": blessing["source"], "seed": blessing["seed"],
        "arrays": sorted(blessing["arrays"]), "commit": "0" * 40, "dirty": False,
        "versions": {}, "date": "2026-01-01T00:00:00+00:00",
    })


def test_a_missing_golden_fails_naming_its_path(goldens):
    path = goldens / "absent.hdf5"
    with pytest.raises(AssertionError, match=f"Golden {path} is missing"):
        _helpers.assert_matches_golden(_arrays(), path, c=1.0)


def test_a_golden_without_its_sidecar_fails_naming_both_paths(goldens):
    path = goldens / "bare.hdf5"
    with h5py.File(path, "w") as golden:
        golden.create_dataset("strain", data=np.zeros(4))
    with pytest.raises(AssertionError, match=f"Golden {path} has no sidecar {goldens / 'bare.json'}"):
        _helpers.assert_matches_golden({"strain": np.zeros(4)}, path, c=1.0)


def test_a_golden_outside_the_goldens_directory_is_refused(goldens, tmp_path_factory):
    path = tmp_path_factory.mktemp("elsewhere") / "stray.hdf5"
    with pytest.raises(AssertionError, match=f"Golden {path} is not under {goldens}"):
        _helpers.assert_matches_golden(_arrays(), path, c=1.0)


def test_a_blessed_golden_round_trips_at_the_data_scale(goldens, monkeypatch):
    path = goldens / "round_trip.hdf5"
    arrays = _arrays()
    _bless(monkeypatch, arrays, path, source="synthetic", seed=7)

    _helpers.assert_matches_golden(arrays, path, c=4.0)

    # The tolerance follows the strain's 1e-21 scale (an absolute 1e-6 would pass anything).
    tolerance = _helpers.representation_tolerance(np.float64, np.max(np.abs(arrays["strain"])), 4.0)
    within = dict(arrays, strain=arrays["strain"] + 0.5 * tolerance)
    _helpers.assert_matches_golden(within, path, c=4.0)
    beyond = dict(arrays, strain=arrays["strain"] + 2.0 * tolerance)
    with pytest.raises(AssertionError, match="round_trip.hdf5:strain"):
        _helpers.assert_matches_golden(beyond, path, c=4.0)
    with pytest.raises(AssertionError, match="round_trip.hdf5:labels"):
        _helpers.assert_matches_golden(dict(arrays, labels=np.arange(1, 5)), path, c=4.0)
    with pytest.raises(AssertionError, match="holds"):
        _helpers.assert_matches_golden({"strain": arrays["strain"]}, path, c=4.0)


@pytest.mark.parametrize("declared, refusal", [
    ({"source": "real", "seed": 7}, "blessed only from"),
    ({"source": None, "seed": None}, "blessed only from"),
    ({"source": "synthetic", "seed": None}, "only with its seed"),
])
def test_blessing_refuses_real_or_unseeded_data(goldens, monkeypatch, declared, refusal):
    with pytest.raises(AssertionError, match=refusal):
        _bless(monkeypatch, _arrays(), goldens / "refused.hdf5", **declared)
    assert _helpers.pending_blessings == []


def test_blessing_applies_only_to_the_test_being_blessed(goldens, monkeypatch):
    monkeypatch.setenv(_helpers.BLESS_ENVIRONMENT_VARIABLE, "gravyflow/tests/test_other.py::test_other")
    with pytest.raises(AssertionError, match="is missing"):
        _helpers.assert_matches_golden(_arrays(), goldens / "other.hdf5", source="synthetic", seed=7)


# Kind (b): the batch passes through about eight FFT-length reductions (the
# Welch estimate, whitening forward and back, SNR scaling), each ~log2(N).
SEEDED_BATCH_SAMPLES = 1024
SEEDED_BATCH_C = 8 * np.log2(SEEDED_BATCH_SAMPLES)


def test_seeded_dataset_batch_matches_its_golden():
    """A seeded white-noise batch with one WNB injection, as blessed."""
    seed = 2026
    dataset = gf.Dataset(
        noise_obtainer=gf.NoiseObtainer(noise_type=gf.NoiseType.WHITE, ifos=[gf.IFO.L1]),
        sample_rate_hertz=float(SEEDED_BATCH_SAMPLES),
        onsource_duration_seconds=1.0,
        offsource_duration_seconds=4.0,
        crop_duration_seconds=0.5,
        num_examples_per_batch=2,
        waveform_generators=[gf.WNBGenerator(
            scaling_method=gf.ScalingMethod(
                value=gf.Distribution(min_=8.0, max_=12.0, type_=gf.DistributionType.UNIFORM),
                type_=gf.ScalingTypes.SNR,
            ),
            network=[gf.IFO.L1],
            duration_seconds=0.2,
            min_frequency_hertz=40.0,
            max_frequency_hertz=300.0,
        )],
        input_variables=[
            gf.ReturnVariables.WHITENED_ONSOURCE,
            gf.ReturnVariables.INJECTIONS,
            gf.ScalingTypes.SNR,
        ],
        seed=seed,
        group="train",
    )
    inputs, _ = dataset[0]
    arrays = {name: np.asarray(value) for name, value in inputs.items()}
    gf.tests.assert_matches_golden(
        arrays, gf.tests.GOLDENS / "seeded_dataset_batch.hdf5",
        c=SEEDED_BATCH_C, source="synthetic", seed=seed,
    )
