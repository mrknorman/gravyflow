"""
Tests for Simple Waveform Generators

Tests for PeriodicWaveGenerator, SineGaussianGenerator, ChirpletGenerator, and RingdownGenerator.
"""

from pathlib import Path

import pytest
import numpy as np
import jax.numpy as jnp
from bokeh.io import output_file, save
from bokeh.layouts import gridplot
from _pytest.config import Config

import gravyflow as gf


# =============================================================================
# PeriodicWaveGenerator Tests
# =============================================================================

def _test_periodic_wave_generation(
    wave_shape: gf.WaveShape,
    num_tests: int = 4,
    plot_results: bool = False
) -> None:
    """Test periodic wave generation for a given shape."""
    output_directory_path = gf.PATH.parent.parent / "gravyflow_data/tests/"
    
    gen = gf.PeriodicWaveGenerator(
        wave_shape=wave_shape,
        frequency_hertz=gf.Distribution(value=100.0, type_=gf.DistributionType.CONSTANT),
        amplitude=gf.Distribution(value=1.0, type_=gf.DistributionType.CONSTANT),
        duration_seconds=gf.Distribution(value=0.5, type_=gf.DistributionType.CONSTANT),
        network=[gf.IFO.L1],
    )
    
    waveforms, params = gen.generate(
        num_waveforms=num_tests,
        sample_rate_hertz=2048.0,
        duration_seconds=1.0,
        seed=42
    )
    
    # Check shape and validity
    assert waveforms.shape[0] == num_tests
    assert waveforms.shape[1] == 2
    assert not jnp.any(jnp.isnan(waveforms))
    assert not jnp.any(jnp.isinf(waveforms))
    
    # Check parameters returned with correct keys
    assert gf.WaveformParameters.FREQUENCY_HERTZ in params
    assert gf.WaveformParameters.AMPLITUDE in params
    assert gf.WaveformParameters.DURATION_SECONDS in params
    
    if plot_results:
        layout = [
            [gf.generate_strain_plot(
                {"Plus": waveform[0], "Cross": waveform[1]},
                title=f"{wave_shape.name} wave: freq={float(freq):.1f}Hz"
            )]
            for waveform, freq in zip(
                waveforms, 
                params[gf.WaveformParameters.FREQUENCY_HERTZ]
            )
        ]
        
        gf.ensure_directory_exists(output_directory_path)
        output_file(output_directory_path / f"periodic_{wave_shape.name.lower()}_plots.html")
        grid = gridplot(layout)
        save(grid)


def test_sine_wave_generation(pytestconfig: Config) -> None:
    """Test sine wave generation."""
    _test_periodic_wave_generation(
        wave_shape=gf.WaveShape.SINE,
        plot_results=pytestconfig.getoption("plot")
    )


def test_square_wave_generation(pytestconfig: Config) -> None:
    """Test square wave generation."""
    _test_periodic_wave_generation(
        wave_shape=gf.WaveShape.SQUARE,
        plot_results=pytestconfig.getoption("plot")
    )


def test_sawtooth_wave_generation(pytestconfig: Config) -> None:
    """Test sawtooth wave generation."""
    _test_periodic_wave_generation(
        wave_shape=gf.WaveShape.SAWTOOTH,
        plot_results=pytestconfig.getoption("plot")
    )


def test_triangle_wave_generation(pytestconfig: Config) -> None:
    """Test triangle wave generation."""
    _test_periodic_wave_generation(
        wave_shape=gf.WaveShape.TRIANGLE,
        plot_results=pytestconfig.getoption("plot")
    )


def test_periodic_frequency_distribution() -> None:
    """Test that frequency distribution produces varying frequencies."""
    gen = gf.PeriodicWaveGenerator(
        wave_shape=gf.WaveShape.SINE,
        frequency_hertz=gf.Distribution(min_=50.0, max_=200.0, type_=gf.DistributionType.UNIFORM),
        network=[gf.IFO.L1],
    )
    
    _, params = gen.generate(
        num_waveforms=10,
        sample_rate_hertz=2048.0,
        duration_seconds=1.0,
        seed=42
    )
    
    freqs = np.array(params[gf.WaveformParameters.FREQUENCY_HERTZ])
    
    # Should have variation
    assert np.std(freqs) > 10.0
    # Should be within bounds
    assert np.all(freqs >= 50.0)
    assert np.all(freqs <= 200.0)


# =============================================================================
# SineGaussianGenerator Tests
# =============================================================================

def _test_sine_gaussian_generation(
    num_tests: int = 4,
    plot_results: bool = False
) -> None:
    """Test sine-Gaussian generation."""
    output_directory_path = gf.PATH.parent.parent / "gravyflow_data/tests/"
    
    gen = gf.SineGaussianGenerator(
        frequency_hertz=gf.Distribution(value=150.0, type_=gf.DistributionType.CONSTANT),
        quality_factor=gf.Distribution(value=10.0, type_=gf.DistributionType.CONSTANT),
        amplitude=gf.Distribution(value=1.0, type_=gf.DistributionType.CONSTANT),
        network=[gf.IFO.L1],
    )
    
    waveforms, params = gen.generate(
        num_waveforms=num_tests,
        sample_rate_hertz=2048.0,
        duration_seconds=1.0,
        seed=42
    )
    
    assert waveforms.shape[0] == num_tests
    assert waveforms.shape[1] == 2
    assert not jnp.any(jnp.isnan(waveforms))
    
    # Check parameters
    assert gf.WaveformParameters.QUALITY_FACTOR in params
    assert gf.WaveformParameters.TAU_SECONDS in params
    
    if plot_results:
        layout = [
            [gf.generate_strain_plot(
                {"Plus": waveform[0], "Cross": waveform[1]},
                title=f"Sine-Gaussian: Q={float(q):.1f}, f={float(freq):.1f}Hz"
            )]
            for waveform, freq, q in zip(
                waveforms, 
                params[gf.WaveformParameters.FREQUENCY_HERTZ],
                params[gf.WaveformParameters.QUALITY_FACTOR]
            )
        ]
        
        gf.ensure_directory_exists(output_directory_path)
        output_file(output_directory_path / "sine_gaussian_plots.html")
        grid = gridplot(layout)
        save(grid)


def test_sine_gaussian_generation(pytestconfig: Config) -> None:
    """Test sine-Gaussian generation."""
    _test_sine_gaussian_generation(
        plot_results=pytestconfig.getoption("plot")
    )


def test_sine_gaussian_q_affects_width() -> None:
    """Test that higher Q produces narrower signals."""
    gen_low_q = gf.SineGaussianGenerator(
        frequency_hertz=100.0,
        quality_factor=5.0,
        network=[gf.IFO.L1],
    )
    
    gen_high_q = gf.SineGaussianGenerator(
        frequency_hertz=100.0,
        quality_factor=20.0,
        network=[gf.IFO.L1],
    )
    
    wf_low, params_low = gen_low_q.generate(1, 2048.0, 2.0, seed=42)
    wf_high, params_high = gen_high_q.generate(1, 2048.0, 2.0, seed=42)
    
    # Higher Q = higher tau = wider signal (Q ∝ tau)
    tau_low = float(params_low[gf.WaveformParameters.TAU_SECONDS][0])
    tau_high = float(params_high[gf.WaveformParameters.TAU_SECONDS][0])
    assert tau_high > tau_low


def test_sine_gaussian_centered_peak() -> None:
    """Test that sine-Gaussian peaks near center."""
    gen = gf.SineGaussianGenerator(
        frequency_hertz=100.0,
        quality_factor=10.0,
        network=[gf.IFO.L1],
    )
    
    waveform, _ = gen.generate(
        num_waveforms=1,
        sample_rate_hertz=2048.0,
        duration_seconds=1.0,
        seed=42
    )
    
    h_plus = np.abs(np.array(waveform[0, 0, :]))
    peak_idx = np.argmax(h_plus)
    center = len(h_plus) // 2
    
    # Peak should be within 10% of center
    assert abs(peak_idx - center) < len(h_plus) * 0.1


# =============================================================================
# ChirpletGenerator Tests
# =============================================================================

def _test_chirplet_generation(
    num_tests: int = 4,
    plot_results: bool = False
) -> None:
    """Test chirplet generation."""
    output_directory_path = gf.PATH.parent.parent / "gravyflow_data/tests/"
    
    gen = gf.ChirpletGenerator(
        start_frequency_hertz=gf.Distribution(value=50.0, type_=gf.DistributionType.CONSTANT),
        end_frequency_hertz=gf.Distribution(value=200.0, type_=gf.DistributionType.CONSTANT),
        duration_seconds=gf.Distribution(value=0.5, type_=gf.DistributionType.CONSTANT),
        network=[gf.IFO.L1],
    )
    
    waveforms, params = gen.generate(
        num_waveforms=num_tests,
        sample_rate_hertz=2048.0,
        duration_seconds=1.0,
        seed=42
    )
    
    assert waveforms.shape[0] == num_tests
    assert not jnp.any(jnp.isnan(waveforms))
    
    # Check parameters
    assert gf.WaveformParameters.START_FREQUENCY_HERTZ in params
    assert gf.WaveformParameters.END_FREQUENCY_HERTZ in params
    assert gf.WaveformParameters.CHIRP_RATE in params
    
    if plot_results:
        layout = [
            [gf.generate_strain_plot(
                {"Plus": waveform[0], "Cross": waveform[1]},
                title=f"Chirplet: {float(f0):.1f}Hz → {float(f1):.1f}Hz"
            )]
            for waveform, f0, f1 in zip(
                waveforms, 
                params[gf.WaveformParameters.START_FREQUENCY_HERTZ],
                params[gf.WaveformParameters.END_FREQUENCY_HERTZ]
            )
        ]
        
        gf.ensure_directory_exists(output_directory_path)
        output_file(output_directory_path / "chirplet_plots.html")
        grid = gridplot(layout)
        save(grid)


def test_chirplet_generation(pytestconfig: Config) -> None:
    """Test chirplet generation."""
    _test_chirplet_generation(
        plot_results=pytestconfig.getoption("plot")
    )


def test_chirplet_frequency_sweep() -> None:
    """Test that chirp rate is correctly computed."""
    gen = gf.ChirpletGenerator(
        start_frequency_hertz=100.0,
        end_frequency_hertz=500.0,
        duration_seconds=1.0,
        network=[gf.IFO.L1],
    )
    
    _, params = gen.generate(
        num_waveforms=1,
        sample_rate_hertz=2048.0,
        duration_seconds=2.0,
        seed=42
    )
    
    k = float(params[gf.WaveformParameters.CHIRP_RATE][0])
    f0 = float(params[gf.WaveformParameters.START_FREQUENCY_HERTZ][0])
    f1 = float(params[gf.WaveformParameters.END_FREQUENCY_HERTZ][0])
    dur = float(params[gf.WaveformParameters.DURATION_SECONDS][0])
    
    # k = (f1 - f0) / dur
    expected_k = (f1 - f0) / dur
    assert abs(k - expected_k) < 1.0


# =============================================================================
# RingdownGenerator Tests
# =============================================================================

def _test_ringdown_generation(
    num_tests: int = 4,
    plot_results: bool = False
) -> None:
    """Test ringdown generation."""
    output_directory_path = gf.PATH.parent.parent / "gravyflow_data/tests/"
    
    gen = gf.RingdownGenerator(
        frequency_hertz=gf.Distribution(value=200.0, type_=gf.DistributionType.CONSTANT),
        damping_time_seconds=gf.Distribution(value=0.05, type_=gf.DistributionType.CONSTANT),
        amplitude=gf.Distribution(value=1.0, type_=gf.DistributionType.CONSTANT),
        network=[gf.IFO.L1],
    )
    
    waveforms, params = gen.generate(
        num_waveforms=num_tests,
        sample_rate_hertz=2048.0,
        duration_seconds=1.0,
        seed=42
    )
    
    assert waveforms.shape[0] == num_tests
    assert not jnp.any(jnp.isnan(waveforms))
    
    # Check parameters
    assert gf.WaveformParameters.DAMPING_TIME_SECONDS in params
    
    if plot_results:
        layout = [
            [gf.generate_strain_plot(
                {"Plus": waveform[0], "Cross": waveform[1]},
                title=f"Ringdown: f={float(freq):.1f}Hz, τ={float(tau)*1000:.1f}ms"
            )]
            for waveform, freq, tau in zip(
                waveforms, 
                params[gf.WaveformParameters.FREQUENCY_HERTZ],
                params[gf.WaveformParameters.DAMPING_TIME_SECONDS]
            )
        ]
        
        gf.ensure_directory_exists(output_directory_path)
        output_file(output_directory_path / "ringdown_plots.html")
        grid = gridplot(layout)
        save(grid)


def test_ringdown_generation(pytestconfig: Config) -> None:
    """Test ringdown generation."""
    _test_ringdown_generation(
        plot_results=pytestconfig.getoption("plot")
    )


def test_ringdown_exponential_decay() -> None:
    """Test that ringdown decays exponentially."""
    gen = gf.RingdownGenerator(
        frequency_hertz=100.0,
        damping_time_seconds=0.1,  # 100ms decay
        amplitude=1.0,
        network=[gf.IFO.L1],
    )
    
    waveform, _ = gen.generate(
        num_waveforms=1,
        sample_rate_hertz=2048.0,
        duration_seconds=1.0,
        seed=42
    )
    
    h_plus = np.abs(np.array(waveform[0, 0, :]))
    
    # Signal should decay: amplitude at t=0 > amplitude at t=0.5s
    sample_mid = int(0.5 * 2048)
    
    # Use envelope approximation
    start_amp = np.max(h_plus[:100])
    mid_amp = np.max(h_plus[sample_mid:sample_mid+100])
    
    assert start_amp > mid_amp * 2  # Should have decayed significantly


def test_ringdown_max_duration() -> None:
    """Test get_max_generated_duration calculation."""
    gen = gf.RingdownGenerator(
        damping_time_seconds=0.05,  # tau = 50ms
        network=[gf.IFO.L1],
    )
    
    max_dur = gen.get_max_generated_duration()
    
    # Should be ~5*tau = 0.25s
    assert abs(max_dur - 0.25) < 0.01


# =============================================================================
# Integration Tests with GravyflowDataset
# =============================================================================

def test_periodic_wave_in_dataset() -> None:
    """Test PeriodicWaveGenerator works in dataset pipeline."""
    from gravyflow.src.dataset.dataset import GravyflowDataset
    
    scaling_method = gf.ScalingMethod(
        value=gf.Distribution(value=10.0, type_=gf.DistributionType.CONSTANT),
        type_=gf.ScalingTypes.SNR
    )
    
    gen = gf.PeriodicWaveGenerator(
        wave_shape=gf.WaveShape.SINE,
        frequency_hertz=100.0,
        duration_seconds=0.5,
        scaling_method=scaling_method,
        network=[gf.IFO.L1],
        injection_chance=1.0,
    )
    
    noise_obtainer = gf.NoiseObtainer(
        noise_type=gf.NoiseType.WHITE,
        ifos=[gf.IFO.L1]
    )
    
    dataset = GravyflowDataset(
        noise_obtainer=noise_obtainer,
        sample_rate_hertz=2048.0,
        onsource_duration_seconds=2.0,
        offsource_duration_seconds=8.0,
        num_examples_per_batch=2,
        waveform_generators=[gen],
        input_variables=[gf.ReturnVariables.WHITENED_ONSOURCE],
        seed=42,
    )
    
    batch = next(iter(dataset))
    X, y = batch
    
    data = X[gf.ReturnVariables.WHITENED_ONSOURCE.name]
    
    assert data.shape[0] == 2
    assert not np.any(np.isnan(data))


def test_sine_gaussian_in_dataset() -> None:
    """Test SineGaussianGenerator works in dataset pipeline."""
    from gravyflow.src.dataset.dataset import GravyflowDataset
    
    scaling_method = gf.ScalingMethod(
        value=gf.Distribution(value=15.0, type_=gf.DistributionType.CONSTANT),
        type_=gf.ScalingTypes.SNR
    )
    
    gen = gf.SineGaussianGenerator(
        frequency_hertz=150.0,
        quality_factor=10.0,
        scaling_method=scaling_method,
        network=[gf.IFO.L1],
        injection_chance=1.0,
    )
    
    noise_obtainer = gf.NoiseObtainer(
        noise_type=gf.NoiseType.WHITE,
        ifos=[gf.IFO.L1]
    )
    
    dataset = GravyflowDataset(
        noise_obtainer=noise_obtainer,
        sample_rate_hertz=2048.0,
        onsource_duration_seconds=2.0,
        offsource_duration_seconds=8.0,
        num_examples_per_batch=2,
        waveform_generators=[gen],
        input_variables=[gf.ReturnVariables.WHITENED_ONSOURCE],
        seed=42,
    )
    
    batch = next(iter(dataset))
    X, _ = batch
    
    data = X[gf.ReturnVariables.WHITENED_ONSOURCE.name]
    
    assert data.shape[0] == 2
    assert not np.any(np.isnan(data))
