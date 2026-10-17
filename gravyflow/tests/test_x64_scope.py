"""
JAX's 64-bit mode is on only for waveform generation and matched-filter templates.

- Importing gravyflow, and the ripplegw import inside it, leaves the mode as it was.
- CBC waveforms match a reference generated wholly in 64-bit mode, and are float32.
- GPS times keep their float64 precision through the acquisition and dataset path.
"""
import subprocess
import sys
import textwrap

import jax
import numpy as np
import pytest
from keras import ops

import gravyflow as gf
from gravyflow.src.dataset.acquisition import base
from gravyflow.src.dataset.features.waveforms import cbc
from gravyflow.src.utils.shapes import ShapeEnforcer

GPS = 1238167546.7832
SAMPLE_RATE = 2048.0


def _run(code):
    """Run ``code`` in a fresh interpreter and return its last line of output."""
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)], capture_output=True, text=True, timeout=300
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip().splitlines()[-1]


def test_importing_gravyflow_leaves_x64_off():
    assert _run("""
        import jax
        import gravyflow
        print(jax.config.jax_enable_x64)
    """) == "False"


def test_the_ripplegw_import_does_not_leave_x64_on():
    # gravyflow imports ripplegw on first use of its CBC code, and ripplegw's models switch x64 on when imported.
    assert _run("""
        import sys
        import jax
        import gravyflow as gf
        assert "ripplegw" not in sys.modules
        gf.CBCGenerator
        assert "ripplegw" in sys.modules
        import ripplegw
        from ripplegw.waveforms.cbc.IMRPhenomD import IMRPhenomD_QNMdata
        print(jax.config.jax_enable_x64, IMRPhenomD_QNMdata.QNMData_fRD.dtype)
    """) == "False float64"


def test_importing_gravyflow_keeps_x64_on_when_the_caller_set_it():
    assert _run("""
        import jax
        jax.config.update("jax_enable_x64", True)
        import gravyflow
        print(jax.config.jax_enable_x64)
    """) == "True"


def _sources(approximant, count, seed):
    rng = np.random.default_rng(seed)
    masses = rng.uniform(3.0, 60.0, (2, count))
    spin_1 = np.zeros((count, 3))
    spin_2 = np.zeros((count, 3))
    if approximant == "IMRPhenomPv2":
        spin_1 = rng.uniform(-0.45, 0.45, (count, 3))
        spin_2 = rng.uniform(-0.45, 0.45, (count, 3))
    else:
        spin_1[:, 2] = rng.uniform(-0.8, 0.8, count)
        spin_2[:, 2] = rng.uniform(-0.8, 0.8, count)
    return dict(
        mass_1_msun=masses.max(0),
        mass_2_msun=masses.min(0),
        spin_1_in=spin_1,
        spin_2_in=spin_2,
        inclination_radians=rng.uniform(0.0, np.pi, count),
        distance_mpc=rng.uniform(100.0, 1000.0, count),
        reference_orbital_phase_in=rng.uniform(0.0, 2.0 * np.pi, count),
    )


@pytest.mark.parametrize("approximant", ["IMRPhenomD", "IMRPhenomXAS", "IMRPhenomPv2"])
def test_cbc_waveforms_match_a_64_bit_reference(approximant):
    source = _sources(approximant, count=8, seed=7)
    with jax.enable_x64(False):
        # Arrays made outside the scope are float32, as the dataset path makes them.
        waveforms = cbc._generate_cbc_window(
            num_waveforms=8,
            sample_rate_hertz=SAMPLE_RATE,
            window_seconds=2.0,
            min_frequency_hertz=None,
            approximant=approximant,
            **{name: ops.convert_to_tensor(value, dtype="float32") for name, value in source.items()},
        )
        assert not jax.config.jax_enable_x64
    with jax.enable_x64(True):
        reference = cbc._generate_cbc_window(
            num_waveforms=8,
            sample_rate_hertz=SAMPLE_RATE,
            window_seconds=2.0,
            min_frequency_hertz=None,
            approximant=approximant,
            **{name: jax.numpy.asarray(value, dtype=jax.numpy.float64) for name, value in source.items()},
        )

    assert waveforms.dtype == np.float32
    a = np.asarray(waveforms, dtype=np.float64)
    b = np.asarray(reference, dtype=np.float64)
    overlap = np.sum(a * b, axis=-1) / np.sqrt(np.sum(a * a, axis=-1) * np.sum(b * b, axis=-1))
    assert overlap.min() >= 0.99999


def test_matched_filter_templates_are_float32_and_leave_x64_off():
    with jax.enable_x64(False):
        matched_filter = gf.MatchedFilter(
            bank=gf.TemplateBank.from_masses([20.0, 30.0, 30.0], [20.0, 20.0, 30.0]),
            sample_rate_hertz=SAMPLE_RATE,
            onsource_duration_seconds=1.0,
        )
        templates = matched_filter.generate_templates()
        assert not jax.config.jax_enable_x64
    assert templates.dtype == np.float32
    assert np.isfinite(np.asarray(templates)).all()
    assert np.abs(np.asarray(templates)).max() > 0.0


# GPS times: synthetic segments whose samples are their own indices, so a
# window's first sample gives its start index exactly.
NUM_SAMPLES = int(64 * SAMPLE_RATE)
ONSOURCE = int(1.0 * SAMPLE_RATE)
OFFSOURCE = int(16.0 * SAMPLE_RATE)
INTERVAL = np.float64(np.float32(1.0 / SAMPLE_RATE))  # IFOData keeps the interval in float32


def _segment():
    data = np.arange(NUM_SAMPLES, dtype=np.float32)
    return base.IFOData(data=[data, data], sample_rate_hertz=SAMPLE_RATE, start_gps_time=[GPS, GPS + 100.0])


def test_gps_times_survive_random_subsection():
    data = np.arange(NUM_SAMPLES, dtype=np.float32)
    with jax.enable_x64(False):
        subarrays, _, start_times = base.random_subsection(
            [data], [GPS], np.float32(1.0 / SAMPLE_RATE), ONSOURCE, OFFSOURCE, 8, 1234
        )
    assert start_times.dtype == np.float64
    expected = GPS + np.asarray(subarrays)[:, 0, 0].astype(np.float64) * INTERVAL
    start_times = start_times[:, 0]
    np.testing.assert_allclose(start_times, expected, rtol=0.0, atol=1e-6)


def test_gps_times_survive_grid_windows():
    with jax.enable_x64(False):
        onsource, _, gps_times, _ = _segment().grid_windows(ONSOURCE, OFFSOURCE, 0.0, [None, None])
    assert gps_times.dtype == np.float64
    expected = np.array([GPS, GPS + 100.0])[None, :] + np.asarray(onsource)[:, :, 0].astype(np.float64) * INTERVAL
    np.testing.assert_allclose(gps_times, expected, rtol=0.0, atol=1e-6)


class _SegmentNoise:
    """A noise obtainer that samples the synthetic segment as the real-noise path does."""

    ifos = [gf.IFO.H1, gf.IFO.L1]

    def __call__(self, num_examples_per_batch, seed, **kwargs):
        segment = _segment()
        rng = np.random.default_rng(seed)

        def batches():
            while True:
                onsource, offsource, gps_times = segment.random_subsection(
                    ONSOURCE + 2 * int(0.5 * SAMPLE_RATE), OFFSOURCE, num_examples_per_batch, rng.integers(1e9)
                )
                yield {
                    gf.ReturnVariables.ONSOURCE: onsource,
                    gf.ReturnVariables.OFFSOURCE: offsource,
                    gf.ReturnVariables.START_GPS_TIME: gps_times,
                }

        return ShapeEnforcer.wrap_generator(batches(), num_ifos=len(self.ifos))


def test_gps_times_survive_the_dataset_path():
    with jax.enable_x64(False):
        dataset = gf.GravyflowDataset(
            seed=42,
            sample_rate_hertz=SAMPLE_RATE,
            onsource_duration_seconds=1.0,
            offsource_duration_seconds=16.0,
            crop_duration_seconds=0.5,
            num_examples_per_batch=4,
            noise_obtainer=_SegmentNoise(),
            input_variables=[gf.ReturnVariables.ONSOURCE],
            output_variables=[gf.ReturnVariables.START_GPS_TIME],
            group="train",
        )
        inputs, outputs = dataset[0]
    gps_times = outputs[gf.ReturnVariables.START_GPS_TIME.name]
    assert isinstance(gps_times, np.ndarray) and gps_times.dtype == np.float64

    # The onsource output is cropped by 0.5 s at each end.
    first = np.asarray(inputs[gf.ReturnVariables.ONSOURCE.name])[:, :, 0].astype(np.float64)
    start = first - int(0.5 * SAMPLE_RATE)
    expected = np.array([GPS, GPS + 100.0])[None, :] + start * INTERVAL
    np.testing.assert_allclose(gps_times, expected, rtol=0.0, atol=1e-6)
