"""
The platform: what, besides a batch's address, decides its bits.

The record (`platform.platform_record`) is kept beside data, in
``seed_metadata`` and in every materialised set's manifest, and never in a
dataset's fingerprint. Each backend is judged pure or not by its own flag, and
the CBC stage, which runs on the host's CPU, refuses by name in a process
without a CPU backend.

Only the CPU backend runs here: the GPU and TPU cases replace
``jax.default_backend`` and read the flags, so they prove the judgement and the
warning, not that a GPU's kernels are deterministic under its flag.
"""

import json
import platform as host_platform
import subprocess
import sys
import textwrap
import warnings
from pathlib import Path

import jax
import pytest

import gravyflow as gf
from gravyflow.src.dataset.materialise import ForeignPlatform, MaterialisedSet
from gravyflow.src.rng import platform
from gravyflow.tests.conftest import child_env

CPU_FLAG = "--xla_cpu_multi_thread_eigen=false"
OTHER_FLAG = "--xla_force_host_platform_device_count=1"   # a known flag: XLA aborts on unknown ones
GPU_FLAG = "--xla_gpu_deterministic_ops=true"


def dataset(**overrides):
    """A small synthetic dataset: white noise alone, at 256 Hz, 4 batches."""
    args = dict(
        seed=5, group="validate", steps_per_epoch=4, sample_rate_hertz=256.0, num_examples_per_batch=2,
        noise_obtainer=gf.NoiseObtainer(noise_type=gf.NoiseType.WHITE),
        input_variables=[gf.ReturnVariables.ONSOURCE],
        output_variables=[gf.ReturnVariables.START_GPS_TIME],
    )
    return gf.Dataset(**{**args, **overrides})


# ------------------------------------------------------------------ the record

def test_the_record_names_what_decides_the_bits(monkeypatch):
    monkeypatch.setenv("XLA_FLAGS", f"{OTHER_FLAG} {CPU_FLAG}")
    assert platform.platform_record() == {
        "backend": "cpu",
        "device_kind": jax.devices()[0].device_kind,
        "xla_flags": f"{OTHER_FLAG} {CPU_FLAG}",
        "matmul_precision": jax.config.jax_default_matmul_precision,
        "cbc": {"device": jax.devices("cpu")[0].device_kind, "pure": True},
        "architecture": host_platform.machine(),
    }
    monkeypatch.setenv("XLA_FLAGS", OTHER_FLAG)
    assert platform.platform_record()["cbc"]["pure"] is False


def test_seed_metadata_and_the_manifest_record_the_platform(tmp_path):
    ds = dataset()
    assert ds.seed_metadata["platform"] == platform.platform_record()
    the_set = ds.materialise(tmp_path, batches=2, shard_size=1)
    assert the_set.manifest["platform"] == platform.platform_record()
    assert ds.seed_metadata["platform"] == the_set.manifest["platform"]


def test_the_record_is_never_in_the_fingerprint(monkeypatch):
    base = dataset().fingerprint
    monkeypatch.setattr(platform.host_platform, "machine", lambda: "x86_64")
    monkeypatch.setenv("XLA_FLAGS", GPU_FLAG)
    assert platform.platform_record()["architecture"] == "x86_64"
    assert dataset().fingerprint == base


# ------------------------------------------------------------------ purity

@pytest.mark.parametrize("flags, cpu, gpu", [
    ("", False, False),
    (CPU_FLAG, True, False),
    (GPU_FLAG, False, True),
    (f"{OTHER_FLAG} {GPU_FLAG} {CPU_FLAG}", True, True),
    ("--xla_cpu_multi_thread_eigen=true --xla_gpu_deterministic_ops=false", False, False),
])
def test_each_backend_is_pure_by_its_own_flag(monkeypatch, flags, cpu, gpu):
    monkeypatch.setenv("XLA_FLAGS", flags)
    assert platform.pure("cpu") is cpu
    assert platform.pure("gpu") is gpu
    assert platform.pure("tpu") is None     # no flag is known to make a TPU's kernels pure


def test_purity_defaults_to_the_default_backend(monkeypatch):
    monkeypatch.setenv("XLA_FLAGS", GPU_FLAG)
    assert platform.pure() is False                      # this process runs on the CPU
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    assert platform.pure() is True
    monkeypatch.setattr(jax, "default_backend", lambda: "tpu")
    assert platform.pure() is None


@pytest.mark.parametrize("backend, flags, match", [
    ("cpu", "", f"XLA_FLAGS={CPU_FLAG}"),
    ("gpu", CPU_FLAG, f"XLA_FLAGS={GPU_FLAG}"),
    ("tpu", f"{CPU_FLAG} {GPU_FLAG}", "No flag is known to make XLA's tpu kernels"),
])
def test_a_backend_that_is_not_pure_warns(monkeypatch, backend, flags, match):
    monkeypatch.setenv("XLA_FLAGS", flags)
    monkeypatch.setattr(jax, "default_backend", lambda: backend)
    with pytest.warns(UserWarning, match=match):
        platform.warn_if_not_pure()


@pytest.mark.parametrize("backend, flags", [("cpu", CPU_FLAG), ("gpu", GPU_FLAG)])
def test_a_pure_backend_does_not_warn(monkeypatch, backend, flags):
    monkeypatch.setenv("XLA_FLAGS", flags)
    monkeypatch.setattr(jax, "default_backend", lambda: backend)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        platform.warn_if_not_pure()


# ------------------------------------------------------------------ the CBC stage's host

def test_cbc_without_a_cpu_backend_refuses_by_name(tmp_path):
    # A process started with JAX_PLATFORMS=tpu has no CPU backend. No TPU runs
    # here, so the child starts on the CPU and then files its one backend under
    # "tpu" in JAX's registry (a private table of jax 0.12) and clears its cache:
    # jax.devices("cpu") then refuses as JAX does in a TPU-only process, while
    # the default backend still computes everything else.
    env = child_env(KERAS_BACKEND="jax", JAX_PLATFORMS="cpu", GRAVYFLOW_CACHE_DIR=str(tmp_path / "cache"))
    child = textwrap.dedent("""
        import json
        import jax
        from jax._src import xla_bridge
        jax.devices()
        xla_bridge._backends = {"tpu": xla_bridge._backends["cpu"]}
        xla_bridge.get_backend.cache_clear()
        import gravyflow as gf
        from gravyflow.src.rng import platform
        constant = lambda value: gf.Distribution(value=value, type_=gf.DistributionType.CONSTANT)
        ds = gf.Dataset(
            seed=5, group="validate", steps_per_epoch=1, sample_rate_hertz=256.0, num_examples_per_batch=2,
            noise_obtainer=gf.NoiseObtainer(noise_type=gf.NoiseType.WHITE),
            waveform_generators=[gf.CBCGenerator(
                scaling_method=gf.ScalingMethod(value=constant(12.0), type_=gf.ScalingTypes.SNR),
                mass_1_msun=constant(30.0), mass_2_msun=constant(25.0))],
            input_variables=[gf.ReturnVariables.WHITENED_ONSOURCE],
            output_variables=[gf.ReturnVariables.INJECTION_MASKS],
        )
        record = platform.platform_record()
        try:
            ds[0]
            refusal = None
        except platform.NoHostBackend as error:
            refusal = str(error)
        print(json.dumps({"file": gf.__file__, "cbc": record["cbc"], "refusal": refusal}))
    """)
    process = subprocess.run([sys.executable, "-c", child], env=env, capture_output=True, text=True, timeout=300)
    assert process.returncode == 0, process.stderr[-4000:]
    report = json.loads(process.stdout.splitlines()[-1])
    assert Path(report["file"]).resolve() == Path(gf.__file__).resolve()
    assert report["cbc"] == {"device": None, "pure": None}
    assert report["refusal"] is not None, "the CBC batch ran without a CPU backend"
    assert "no CPU backend" in report["refusal"] and "include cpu" in report["refusal"]


# ------------------------------------------------------------------ materialised sets

def test_a_set_resumed_on_another_architecture_is_its_own(tmp_path, monkeypatch):
    # An interrupted set on this host, then the same spec under the same root on
    # a host of another architecture: its batches never join this host's set.
    here = dataset().materialise(tmp_path, batches=4, shard_size=2)
    (here.path / "shards" / "1.json").unlink()
    monkeypatch.setattr(platform.host_platform, "machine", lambda: "x86_64")
    there = dataset().materialise(tmp_path, batches=4, shard_size=2)
    assert there.set_id != here.set_id
    assert there.manifest["platform"]["architecture"] == "x86_64"
    assert MaterialisedSet(here.path).missing() == [range(2, 4)]

    # A shard of the other host's written into this host's set is refused by name.
    shard = json.loads((there.path / "shards" / "1.json").read_text())
    shard["set_id"] = here.set_id
    (here.path / "shards" / "1.json").write_text(json.dumps(shard))
    monkeypatch.undo()
    with pytest.raises(ForeignPlatform, match="x86_64"):
        dataset().load_materialised(here.path)
