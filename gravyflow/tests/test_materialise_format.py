"""
The materialised-set format: a dataset's batches written once, as content-named
files under a manifest, and read back verified.

Every way a set can be wrong is planted here (`PLANTS`) and must be refused by
name or, for the races a store must tolerate, verified harmless. A set read back
serves exactly the bytes the dataset computes, for every split.
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from unittest import mock

import numpy as np
import pytest

import gravyflow as gf
from gravyflow.src.dataset.materialise import (
    FORMAT, ContractMismatch, FingerprintMismatch, ForeignPlatform, IntegrityError, LocalBackend,
    MaterialiseConflict, MaterialisedSet, MissingShard, NotIndexAddressed, SplitMismatch, UnknownFormat,
    encode_batch, json_bytes,
)
from gravyflow.src.rng import platform
from gravyflow.src.utils import fingerprint as fingerprint_module
from gravyflow.src.utils.fingerprint import (
    BATCH_LIBRARIES, UnfingerprintableError, config_fingerprint, library_versions,
)

SHARD = 2
BATCHES = 4


def dataset(group="validate", seed=3, **overrides):
    """A small synthetic dataset: white noise with a white-noise burst, whitened, at 512 Hz."""
    snr = gf.Distribution(min_=8.0, max_=20.0, type_=gf.DistributionType.UNIFORM)
    args = dict(
        seed=seed, group=group, steps_per_epoch=BATCHES, sample_rate_hertz=512.0, num_examples_per_batch=4,
        noise_obtainer=gf.NoiseObtainer(noise_type=gf.NoiseType.WHITE),
        waveform_generators=[gf.WNBGenerator(
            scaling_method=gf.ScalingMethod(snr, gf.ScalingTypes.SNR),
            duration_seconds=gf.Distribution(min_=0.1, max_=0.4, type_=gf.DistributionType.UNIFORM),
            min_frequency_hertz=gf.Distribution(min_=40.0, max_=80.0, type_=gf.DistributionType.UNIFORM),
            max_frequency_hertz=gf.Distribution(min_=120.0, max_=200.0, type_=gf.DistributionType.UNIFORM),
        )],
        input_variables=[gf.ReturnVariables.WHITENED_ONSOURCE],
        output_variables=[gf.ReturnVariables.INJECTION_MASKS, gf.ReturnVariables.START_GPS_TIME],
    )
    return gf.Dataset(**{**args, **overrides})


def batch_bytes(batch):
    """A batch's arrays, by name, as (dtype, shape, bytes): equal only when bit-identical."""
    x, y = batch
    return {f"{side}/{name}": (np.asarray(v).dtype.str, np.shape(v), np.asarray(v).tobytes())
            for side, d in (("inputs", x), ("outputs", y)) for name, v in d.items()}


def read_all(ds):
    return [ds[k] for k in range(BATCHES)]


@pytest.fixture(scope="module")
def pristine(tmp_path_factory):
    """One materialised validation set, copied for each plant."""
    root = tmp_path_factory.mktemp("pristine")
    the_set = dataset().materialise(root, batches=BATCHES, shard_size=SHARD)
    return the_set.path


@pytest.fixture
def world(pristine, tmp_path):
    """A fresh copy of the pristine set, and a reader of the spec that wrote it."""
    path = tmp_path / pristine.name
    shutil.copytree(pristine, path)
    return SimpleWorld(path)


@dataclass
class SimpleWorld:
    path: Path

    @property
    def manifest(self) -> dict:
        return json.loads((self.path / "manifest.json").read_text())

    def shard(self, s) -> dict:
        return json.loads((self.path / "shards" / f"{s}.json").read_text())

    def write_shard(self, s, shard) -> None:
        (self.path / "shards" / f"{s}.json").write_bytes(json_bytes(shard))

    def batch_file(self, k) -> Path:
        s = k // SHARD
        sha = dict(self.shard(s)["batches"])[k]
        return self.path / "batches" / f"{sha}.npz"

    def rename_consistently(self, manifest: dict) -> None:
        """Write ``manifest`` and move the set to its new id, as a writer of that manifest would have."""
        data = json_bytes(manifest)
        set_id = hashlib.sha256(data).hexdigest()
        (self.path / "manifest.json").write_bytes(data)
        for s in range(BATCHES // SHARD):
            self.write_shard(s, {**self.shard(s), "set_id": set_id})
        self.path = self.path.rename(self.path.with_name(set_id))

    def load(self, reader=None):
        ds = reader if reader is not None else dataset()
        ds.load_materialised(self.path)
        return read_all(ds)


# ------------------------------------------------------------------ the plants

def plant_truncated_batch(w):
    f = w.batch_file(1)
    f.write_bytes(f.read_bytes()[:-100])
    return w.load


def plant_swapped_batch(w):
    a, b = w.batch_file(0), w.batch_file(1)
    data_a, data_b = a.read_bytes(), b.read_bytes()
    a.write_bytes(data_b)
    b.write_bytes(data_a)
    return w.load


def plant_edited_shard(w):
    # Batch 0's entry names batch 1's file: every file is intact and correctly named.
    shard = w.shard(0)
    entries = dict(shard["batches"])
    shard["batches"] = [[0, entries[1]], [1, entries[1]]]
    w.write_shard(0, shard)
    return w.load


def plant_wrong_fingerprint(w):
    return lambda: w.load(dataset(seed=4))


def plant_wrong_contract(w):
    manifest = w.manifest
    manifest["contract"]["inputs"][0][2] = "float16"
    w.rename_consistently(manifest)
    return w.load


def plant_batch_breaks_contract(w):
    # A batch, correctly named and listed, whose onsource is float64.
    with np.load(w.batch_file(1), allow_pickle=False) as npz:
        arrays = {name: npz[name] for name in npz.files}
    arrays["inputs/WHITENED_ONSOURCE"] = arrays["inputs/WHITENED_ONSOURCE"].astype(np.float64)
    data = encode_batch(arrays)
    sha = hashlib.sha256(data).hexdigest()
    (w.path / "batches" / f"{sha}.npz").write_bytes(data)
    shard = w.shard(0)
    shard["batches"][1][1] = sha
    w.write_shard(0, shard)
    return w.load


def plant_foreign_platform(w):
    # Shard 1 as a writer on another platform would have appended it.
    shard = w.shard(1)
    shard["platform"] = {**shard["platform"], "backend": "tpu"}
    w.write_shard(1, shard)
    return w.load


def plant_missing_shard(w):
    (w.path / "shards" / "1.json").unlink()
    return w.load


def plant_unknown_format(w):
    manifest = w.manifest
    manifest["format"] = FORMAT + 1
    w.rename_consistently(manifest)  # as a later gravyflow would have written it
    return w.load


def plant_edited_manifest(w):
    manifest = w.manifest
    manifest["k_range"] = [0, 2]
    (w.path / "manifest.json").write_bytes(json_bytes(manifest))
    return w.load


def plant_split_mismatch(w):
    return lambda: w.load(dataset(group="test"))


def _race(w, payloads):
    """Two writers commit ``payloads`` under one name at once; returns what each raised."""
    backend = LocalBackend(w.path.parent)
    name = f"{w.path.name}/batches/{'0' * 64}.npz"
    barrier, raised = threading.Barrier(len(payloads)), [None] * len(payloads)

    def write(i):
        barrier.wait()
        try:
            backend.create(name, payloads[i])
        except Exception as error:  # recorded for the check
            raised[i] = error

    threads = [threading.Thread(target=write, args=(i,)) for i in range(len(payloads))]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    return backend.read(name), raised


def plant_race_equal_bytes(w):
    def act():
        stored, raised = _race(w, [b"same bytes", b"same bytes"])
        assert stored == b"same bytes" and raised == [None, None]
    return act


def plant_race_different_bytes(w):
    def act():
        stored, raised = _race(w, [b"first writer", b"second writer"])
        losers = [e for e in raised if e is not None]
        assert stored in (b"first writer", b"second writer") and len(losers) == 1
        raise losers[0]
    return act


# ------------------------------------------------- datasets read in order

class _InOrderDataset(gf.GravyflowDataset):
    """A dataset read in order, as one over a reader of the caller's own is (``index_addressed`` False)."""

    @property
    def index_addressed(self) -> bool:
        return False


def read_in_order(ds):
    ds.__class__ = _InOrderDataset
    return ds


def plant_edited_versions_record(w):
    # The record says another keras wrote the set; its sha256 was left as it was.
    manifest = w.manifest
    manifest["fingerprint"]["versions"]["keras"] = "0.0.0-other"
    w.rename_consistently(manifest)
    return w.load


def plant_reader_on_other_library(w):
    def act():
        real = fingerprint_module._distribution_version
        with mock.patch.object(fingerprint_module, "_distribution_version",
                               lambda name: "0.0.0-other" if name == "jaxlib" else real(name)):
            w.load()
    return act


def plant_reader_in_order(w):
    return lambda: w.load(read_in_order(dataset()))


def plant_other_stage(w):
    manifest = w.manifest
    manifest["stage"] = "tile"
    w.rename_consistently(manifest)
    return w.load


def plant_manifest_not_an_object(w):
    data = b"[]\n"
    (w.path / "manifest.json").write_bytes(data)
    w.path = w.path.rename(w.path.with_name(hashlib.sha256(data).hexdigest()))
    return w.load


def plant_null_shard(w):
    (w.path / "shards" / "0.json").write_bytes(b"null\n")
    return w.load


def plant_empty_shard(w):
    (w.path / "shards" / "1.json").write_bytes(b"")
    return w.load


def plant_fixed_set_epoch_ends(w):
    # Keras ends each epoch on its validation data: a fixed set of a training
    # (fresh-each-epoch) dataset serves its own bytes every epoch.
    def act():
        from gravyflow.src.model.examples.recipe import DetectionBatches, materialise
        ds = dataset(group="train", output_variables=[gf.ReturnVariables.INJECTION_PRESENT])
        fixed = materialise(DetectionBatches(ds), BATCHES, w.path.parent / "fixed")
        first = [fixed[k] for k in range(BATCHES)]
        for _ in range(2):
            fixed.on_epoch_begin()
            fixed.on_epoch_end()
        with mock.patch.object(gf.GravyflowDataset, "_compute_batch", side_effect=AssertionError("computed")):
            again = [fixed[k] for k in range(BATCHES)]
        for (x, y), (first_x, first_y) in zip(again, first):
            assert batch_bytes((x, {"labels": y})) == batch_bytes((first_x, {"labels": first_y}))
    return act


@dataclass(frozen=True)
class Plant:
    name: str
    plant: Callable
    refusal: type = None   # None: the planted state is harmless and verified so
    match: str = ""


PLANTS = [
    Plant("truncated batch", plant_truncated_batch, IntegrityError, "sha256"),
    Plant("swapped batch", plant_swapped_batch, IntegrityError, "sha256"),
    Plant("edited shard", plant_edited_shard, IntegrityError, "holds batch 1"),
    Plant("wrong fingerprint", plant_wrong_fingerprint, FingerprintMismatch, "fingerprint"),
    Plant("wrong contract", plant_wrong_contract, ContractMismatch, "contract"),
    Plant("batch breaks the contract", plant_batch_breaks_contract, ContractMismatch, "float64"),
    Plant("foreign platform appended", plant_foreign_platform, ForeignPlatform, "tpu"),
    Plant("missing shard", plant_missing_shard, MissingShard, r"k in \[2, 4\)"),
    Plant("unknown FORMAT", plant_unknown_format, UnknownFormat, f"format {FORMAT + 1}"),
    Plant("edited manifest", plant_edited_manifest, IntegrityError, "manifest"),
    Plant("split mismatch", plant_split_mismatch, SplitMismatch, "'test'"),
    Plant("racing writers, equal bytes", plant_race_equal_bytes),
    Plant("racing writers, different bytes", plant_race_different_bytes, MaterialiseConflict, "other bytes"),
    Plant("reader read in order", plant_reader_in_order, NotIndexAddressed, "index"),
    Plant("fixed set across epochs", plant_fixed_set_epoch_ends),
    Plant("versions record edited", plant_edited_versions_record, IntegrityError, "fingerprint record"),
    Plant("reader on another jaxlib", plant_reader_on_other_library, FingerprintMismatch, "fingerprint"),
    Plant("another stage", plant_other_stage, ContractMismatch, "stage"),
    Plant("manifest not an object", plant_manifest_not_an_object, IntegrityError, "not a JSON object"),
    Plant("null shard", plant_null_shard, IntegrityError, "shards/0.json"),
    Plant("empty shard", plant_empty_shard, IntegrityError, "empty"),
]


def check_plant(plant: Plant, w) -> None:
    """Plant ``plant`` in the set of ``w`` and require its refusal (or, for a harmless one, success)."""
    act = plant.plant(w)
    if plant.refusal is None:
        act()
    else:
        with pytest.raises(plant.refusal, match=plant.match):
            act()


@pytest.mark.parametrize("plant", PLANTS, ids=[p.name for p in PLANTS])
def test_each_plant_is_refused_or_verified(plant, world):
    check_plant(plant, world)


def test_the_pristine_set_reads(world):
    world.load()  # the plants' control: unplanted, every batch reads


# ------------------------------------------------------------- reading back

@pytest.mark.parametrize("group,epoch", [("train", 0), ("train", 2), ("validate", 0), ("test", 0)])
def test_a_set_serves_the_bytes_the_dataset_computes(tmp_path, group, epoch):
    writer = dataset(group=group)
    writer.epoch = epoch
    the_set = writer.materialise(tmp_path, batches=BATCHES, shard_size=SHARD)
    reference = dataset(group=group)
    reference.epoch = epoch
    computed = [batch_bytes(b) for b in read_all(reference)]

    reader = dataset(group=group)
    reader.epoch = epoch
    assert reader.seed_metadata["materialised"] is None
    reader.load_materialised(the_set)
    assert [batch_bytes(b) for b in read_all(reader)] == computed
    assert reader.seed_metadata["materialised"] == {
        "set_id": the_set.set_id, "epoch": epoch, "k_range": [0, BATCHES]}


def test_addresses_outside_the_set_are_computed(tmp_path):
    the_set = dataset(group="train").materialise(tmp_path, k_range=range(1, 3), shard_size=SHARD)
    reader = dataset(group="train")
    reader.load_materialised(the_set)
    reference = dataset(group="train")
    reads = []
    original = the_set.read
    reader._materialised.read = lambda k: reads.append(k) or original(k)
    for epoch in (0, 1):
        reader.epoch = reference.epoch = epoch
        for k in range(BATCHES):
            assert batch_bytes(reader[k]) == batch_bytes(reference[k])
    assert reads == [1, 2]  # epoch 0's k in [1, 3) only


def test_materialise_returns_a_complete_set_without_computing(tmp_path, monkeypatch):
    ds = dataset()
    first = ds.materialise(tmp_path, batches=BATCHES, shard_size=SHARD)
    monkeypatch.setattr(type(ds), "_compute_batch", lambda self, k: pytest.fail(f"computed batch {k}"))
    again = ds.materialise(tmp_path, batches=BATCHES, shard_size=SHARD)
    assert again.set_id == first.set_id and again.missing() == []


def test_materialise_resumes_an_interrupted_run(tmp_path, monkeypatch):
    ds = dataset()
    compute = type(ds)._compute_batch
    computed = []

    def interrupted(self, k):
        if k == 3:
            raise KeyboardInterrupt("interrupted at batch 3")
        computed.append(k)
        return compute(self, k)

    monkeypatch.setattr(type(ds), "_compute_batch", interrupted)
    with pytest.raises(KeyboardInterrupt):
        ds.materialise(tmp_path, batches=BATCHES, shard_size=SHARD)
    assert computed == [0, 1, 2]
    (set_dir,) = [p for p in tmp_path.iterdir() if p.is_dir()]
    partial = MaterialisedSet(set_dir)
    assert partial.missing() == [range(2, 4)]
    with pytest.raises(MissingShard, match=r"k in \[2, 4\).*materialise"):
        dataset().load_materialised(partial)

    def resumed(self, k):
        computed.append(k)
        return compute(self, k)

    monkeypatch.setattr(type(ds), "_compute_batch", resumed)
    the_set = ds.materialise(tmp_path, batches=BATCHES, shard_size=SHARD)
    assert computed == [0, 1, 2, 2, 3]  # only the incomplete shard again
    fresh = dataset().materialise(tmp_path / "fresh", batches=BATCHES, shard_size=SHARD)
    assert fresh.set_id == the_set.set_id
    assert [batch_bytes(the_set.read(k)) for k in range(BATCHES)] == [
        batch_bytes(fresh.read(k)) for k in range(BATCHES)]


def test_writers_that_disagree_conflict(tmp_path, monkeypatch):
    # Two writers of one set computing different bytes for one batch: the second shard commit
    # finds other bytes and is refused, and the first writer's set stands.
    first = dataset().materialise(tmp_path, batches=BATCHES, shard_size=SHARD)
    for shard in (tmp_path / first.set_id / "shards").iterdir():
        if shard.name == "1.json":
            saved = shard.read_bytes()
            shard.unlink()
    compute = gf.GravyflowDataset._compute_batch

    def drifted(self, k):
        x, y = compute(self, k)
        return {name: np.asarray(v) + np.float32(1e-6) for name, v in x.items()}, y

    shard_path = tmp_path / first.set_id / "shards" / "1.json"
    real_create = LocalBackend.create

    def racing_create(self, name, data):
        if name.endswith("shards/1.json"):
            shard_path.write_bytes(saved)  # the other writer committed shard 1 first
        return real_create(self, name, data)

    monkeypatch.setattr(gf.GravyflowDataset, "_compute_batch", drifted)
    monkeypatch.setattr(LocalBackend, "create", racing_create)
    with pytest.raises(MaterialiseConflict, match="shards/1.json"):
        dataset().materialise(tmp_path, batches=BATCHES, shard_size=SHARD)
    assert shard_path.read_bytes() == saved


_WRITER = """
import sys
from gravyflow.tests.test_materialise_format import dataset, BATCHES, SHARD
the_set = dataset().materialise(sys.argv[1], batches=BATCHES, shard_size=SHARD)
print(the_set.set_id)
"""


def test_two_processes_materialise_one_set(tmp_path):
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}
    procs = [subprocess.Popen([sys.executable, "-c", _WRITER, str(tmp_path)], env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for _ in range(2)]
    outputs = [p.communicate(timeout=240) for p in procs]
    assert [p.returncode for p in procs] == [0, 0], [err[-2000:] for _, err in outputs]
    (set_id,) = {out.strip().splitlines()[-1] for out, _ in outputs}
    reader = dataset()
    reader.load_materialised(tmp_path / set_id)
    assert [batch_bytes(b) for b in read_all(reader)] == [batch_bytes(b) for b in read_all(dataset())]


# ------------------------------------------------------------- the manifest

def test_the_manifest_records_identity_and_platform(tmp_path):
    ds = dataset()
    the_set = ds.materialise(tmp_path, batches=BATCHES, shard_size=SHARD)
    manifest = the_set.manifest
    assert manifest["format"] == FORMAT
    assert manifest["fingerprint"]["sha256"] == ds.fingerprint
    assert manifest["fingerprint"]["root"] == 3
    assert manifest["platform"] == platform.platform_record()
    assert "--xla_cpu_multi_thread_eigen=false" in manifest["platform"]["xla_flags"]
    assert (manifest["split"], manifest["epoch"], manifest["k_range"], manifest["shard_size"]) == (
        "validate", 0, [0, BATCHES], SHARD)
    assert manifest["contract"]["inputs"] == [["WHITENED_ONSOURCE", [4, 1, 512], "float32"]]
    assert the_set.set_id == hashlib.sha256((the_set.path / "manifest.json").read_bytes()).hexdigest()


def test_the_fingerprint_follows_the_spec_not_the_platform(monkeypatch):
    base = dataset().fingerprint
    assert dataset().fingerprint == base
    assert dataset(group="test").fingerprint == base     # the split is the address's, not the spec's
    for other in (dataset(seed=4), dataset(num_examples_per_batch=8), dataset(name="other")):
        assert other.fingerprint != base
    monkeypatch.setenv("XLA_FLAGS", "--xla_some_other_flag")  # recorded, never hashed
    assert dataset().fingerprint == base


def test_materialise_refuses_addresses_outside_the_epoch(tmp_path):
    ds = dataset()
    with pytest.raises(IndexError, match="outside the 4 batches"):
        ds.materialise(tmp_path, batches=BATCHES + 1)
    with pytest.raises(ValueError, match="one of k_range or batches"):
        ds.materialise(tmp_path, k_range=range(2), batches=2)
    with pytest.raises(ValueError, match="local"):
        ds.materialise("gs://bucket/prefix", batches=1)


def test_materialise_warns_on_multi_threaded_kernels(tmp_path, monkeypatch):
    monkeypatch.setattr(platform, "pure", lambda backend=None: False)
    with pytest.warns(UserWarning, match="not deterministic"):
        dataset().materialise(tmp_path, batches=1)


def test_a_dataset_read_in_order_is_never_materialised(tmp_path, pristine):
    # Its batch k depends on the batches asked for before it: a set written or
    # resumed from it would hold some batches twice and others never.
    with pytest.raises(NotIndexAddressed, match="index"):
        read_in_order(dataset()).materialise(tmp_path, batches=BATCHES, shard_size=SHARD)
    assert list(tmp_path.iterdir()) == []
    with pytest.raises(NotIndexAddressed, match="index"):
        read_in_order(dataset()).load_materialised(pristine)


# ------------------------------------------------------------- the identity

def test_the_versions_record_every_library_that_makes_batch_bytes():
    versions = dataset().fingerprint_record()["versions"]
    assert set(BATCH_LIBRARIES) == {"numpy", "scipy", "jax", "jaxlib", "keras", "ripplegw"}
    assert set(versions) == {"python", "gravyflow", *BATCH_LIBRARIES}
    assert versions["gravyflow"] == gf.__version__ and versions["numpy"] == np.__version__
    assert versions == library_versions()


@pytest.mark.parametrize("library", ["gravyflow", *BATCH_LIBRARIES])
def test_each_library_version_changes_the_fingerprint(library, monkeypatch):
    base = dataset().fingerprint
    if library == "gravyflow":
        monkeypatch.setattr(gf, "__version__", "0.0.0-other")
    else:
        real = fingerprint_module._distribution_version
        monkeypatch.setattr(fingerprint_module, "_distribution_version",
                            lambda name: "0.0.0-other" if name == library else real(name))
    assert dataset().fingerprint != base


@pytest.mark.parametrize("field,value", [("python_version", "2.7.18"), ("python_implementation", "PyPy")])
def test_the_python_implementation_and_version_change_the_fingerprint(field, value, monkeypatch):
    import platform as python_platform
    base = dataset().fingerprint
    monkeypatch.setattr(python_platform, field, lambda: value)
    assert dataset().fingerprint != base


def test_each_identity_input_changes_the_fingerprint(monkeypatch):
    from gravyflow.src.dataset import dataset as dataset_module
    from gravyflow.src.dataset.acquisition.versions import DATA_VERSIONS

    base = dataset().fingerprint
    changed = {"root": dataset(seed=4).fingerprint, "spec": dataset(sample_rate_hertz=256.0).fingerprint}
    with monkeypatch.context() as patch:
        patch.setattr(gf.GravyflowDataset, "split_spec", property(lambda self: gf.SplitSpec(version="fold-3")))
        changed["SplitSpec"] = dataset().fingerprint
    with monkeypatch.context() as patch:
        patch.setattr(dataset_module, "TREE_VERSION", "tree-other")
        changed["TREE_VERSION"] = dataset().fingerprint
    for key in DATA_VERSIONS:
        with monkeypatch.context() as patch:
            patch.setattr(dataset_module, "DATA_VERSIONS", {**DATA_VERSIONS, key: "data-other"})
            changed[f"DATA_VERSIONS[{key}]"] = dataset().fingerprint
    assert len(changed) > 4 and dataset().fingerprint == base
    assert [name for name, fp in changed.items() if fp == base] == []


def test_prefetching_and_use_leave_the_fingerprint(tmp_path):
    ds = dataset(group="train")
    base = ds.fingerprint
    ds.workers, ds.use_multiprocessing, ds.max_queue_size = 4, True, 99  # Keras' prefetch settings
    ds[0], ds[1]
    ds.on_epoch_end()
    ds.epoch = 5
    ds.load_materialised(ds.materialise(tmp_path, batches=1))
    ds[0]
    assert ds.fingerprint == base


def test_the_fingerprint_is_config_fingerprint_of_its_record():
    # One canonical encoding: a dataset's fingerprint and a configuration's are made alike.
    ds = dataset()
    assert ds.fingerprint == config_fingerprint(ds.fingerprint_record())


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), np.float32("nan")])
def test_a_non_finite_setting_is_refused_by_name(value):
    with pytest.raises(UnfingerprintableError, match="not finite"):
        config_fingerprint({"threshold": value})
    ds = dataset()
    ds.crop_duration_seconds = value
    with pytest.raises(UnfingerprintableError, match="not finite"):
        ds.fingerprint


# ------------------------------------------------------------- malformed files

def _rewrite_manifest(w, data: bytes) -> None:
    (w.path / "manifest.json").write_bytes(data)
    w.path = w.path.rename(w.path.with_name(hashlib.sha256(data).hexdigest()))


@pytest.mark.parametrize("data", [b"[]\n", b"null\n", b"3\n"])
def test_a_manifest_that_is_not_an_object_is_refused(world, data):
    _rewrite_manifest(world, data)
    with pytest.raises(IntegrityError, match="not a JSON object"):
        MaterialisedSet(world.path)


def test_the_manifest_is_hashed_before_it_is_parsed(world):
    (world.path / "manifest.json").write_bytes(b"\x00not json")
    with pytest.raises(IntegrityError, match="sha256"):
        MaterialisedSet(world.path)


@pytest.mark.parametrize("data", [b"null\n", b"[]\n", b'{"batches": null}\n', b'{"batches": [[0]]}\n'],
                         ids=["null", "list", "null batches", "short entry"])
def test_a_malformed_shard_is_refused(world, data):
    (world.path / "shards" / "0.json").write_bytes(data)
    with pytest.raises(IntegrityError, match="shards/0.json"):
        world.load()


def test_a_zero_length_shard_says_how_to_recover(world):
    (world.path / "shards" / "1.json").write_bytes(b"")
    with pytest.raises(IntegrityError, match=r"empty.*[Dd]elete it.*materialise"):
        world.load()


def test_a_resumed_write_verifies_committed_batches(tmp_path):
    ds = dataset()
    the_set = ds.materialise(tmp_path, batches=BATCHES, shard_size=SHARD)
    batch = the_set.path / "batches" / f"{the_set.shard(0)[1]}.npz"
    batch.unlink()
    with pytest.raises(IntegrityError, match="does not exist"):
        ds.materialise(tmp_path, batches=BATCHES, shard_size=SHARD)
    batch.write_bytes(b"truncated")
    with pytest.raises(IntegrityError, match="sha256"):
        ds.materialise(tmp_path, batches=BATCHES, shard_size=SHARD)


# --------------------------------------------------------------- the writes

def test_a_commit_syncs_a_uniquely_named_temporary(tmp_path, monkeypatch):
    synced, linked = [], []
    real_fsync, real_link = os.fsync, os.link
    monkeypatch.setattr(os, "fsync", lambda fd: synced.append(fd) or real_fsync(fd))
    monkeypatch.setattr(os, "link", lambda src, dst: linked.append(Path(src).name) or real_link(src, dst))
    backend = LocalBackend(tmp_path)
    backend.create("a/x.npz", b"one")
    backend.create("a/x.npz", b"one")  # an equal second writer
    assert len(synced) >= 2 and len(set(linked)) == 2
    assert all(name.startswith(f".x.npz.{os.getpid()}.") for name in linked)
    assert [p.name for p in (tmp_path / "a").iterdir()] == ["x.npz"]


def test_a_stale_temporary_is_cleaned_on_the_next_write(tmp_path):
    dead = subprocess.run([sys.executable, "-c", "import os; print(os.getpid())"],
                          capture_output=True, text=True, check=True).stdout.strip()
    (tmp_path / "a").mkdir()
    stale = tmp_path / "a" / f".x.npz.{dead}.0123abcd.tmp"
    stale.write_bytes(b"half a batch")
    live = tmp_path / "a" / f".x.npz.{os.getpid()}.feedface.tmp"  # a live writer's, mid-commit
    live.write_bytes(b"in progress")
    LocalBackend(tmp_path).create("a/x.npz", b"whole")
    assert not stale.exists() and live.exists()
    assert (tmp_path / "a" / "x.npz").read_bytes() == b"whole"
