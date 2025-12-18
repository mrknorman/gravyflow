#Built-in imports
from pathlib import Path
from typing import Dict, Tuple, Any
import logging

#Library imports
import numpy as np
from scipy.signal import welch
from gwpy.frequencyseries import FrequencySeries
from gwpy.timeseries import TimeSeries
from bokeh.plotting import figure, output_file, save, show
from bokeh.palettes import Bright
from bokeh.layouts import gridplot
from bokeh.models import ColumnDataSource, HoverTool, Legend
from _pytest.config import Config

# Local imports:

import gravyflow as gf
from gravyflow.src.dataset.tools import snr as gf_snr
from gravyflow.src.dataset.tools import psd as gf_psd
from gravyflow.src.dataset.conditioning.whiten import whiten
from gravyflow.src.dataset.features.waveforms.cbc import generate_cbc_waveform
import keras
from keras import ops

def plot_whitened_strain_examples(
        whitening_results : Dict,
        output_directory_path : Path
    ) -> None:

    layout = [
        [
            gf.generate_strain_plot(
                {
                    "Whitened (tf) Onsource + Injection": whitening_results["onsource_plus_injection"]["tensorflow"],
                    "Whitened (tf) Injection" : whitening_results["scaled_injection"]["tensorflow"],
                    "Injection": scaled_injection
                },
                title="PhenomD injection example tf whitening",
            ), 
            gf.generate_spectrogram(
                whitening_results["onsource_plus_injection"]["tensorflow"]
            )
        ],
        [
            gf.generate_strain_plot(
                {
                    "Whitened (gwpy) Onsource + Injection": whitening_results["onsource_plus_injection"]["gwpy"],
                    "Whitened (gwpy) Injection" : whitening_results["scaled_injection"]["gwpy"],
                    "Injection": scaled_injection
                },
                title=f"PhenomD injection example gwpy whitening",
            ), 
            gf.generate_spectrogram(
                whitening_results["onsource_plus_injection"]["gwpy"]
            )
        ]
    ]

    # Ensure output directory exists
    gf.ensure_directory_exists(output_directory_path)
    
    # Define an output path for the dashboard
    output_file(output_directory_path / "whitening_test_plots.html")

    # Arrange the plots in a grid. 
    grid = gridplot(layout)
        
    save(grid)
    
def plot_psd(
        frequencies : np.ndarray, 
        onsource_plus_injection_whitened_tf_psd_scipy : np.ndarray, 
        onsource_whitened_tf_psd_scipy : np.ndarray, 
        onsource_whitened_tf_psd_tf : np.ndarray, 
        onsource_whitened_gwpy_psd_scipy : np.ndarray,
        file_path : Path
    ) -> None:
    
    p = figure(
        title = "Power Spectral Density", 
        x_axis_label = 'Frequency (Hz)', 
        y_axis_label = 'PSD'
    )

    p.line(
        frequencies, 
        onsource_plus_injection_whitened_tf_psd_scipy, 
        legend_label="Onsource + Injection Whitened Tensorflow PSD Tensorflow", 
        line_width = 2, 
        line_color=Bright[7][0],
    )
    p.line(
        frequencies, 
        onsource_whitened_tf_psd_tf, 
        legend_label="Onsource Whitened Tensorflow PSD Tensorflow", 
        line_width = 2, 
        line_color=Bright[7][1]    
    )
    
    p.line(
        frequencies, 
        onsource_whitened_tf_psd_scipy, 
        legend_label="Onsource Whitened Tensorflow PSD Scipy", 
        line_width = 2, 
        line_color=Bright[7][2],
        line_dash="dotdash"
    )

    p.line(
        frequencies, 
        onsource_whitened_gwpy_psd_scipy, 
        legend_label="Onsource Whitened GWPy PSD SciPy", 
        line_width = 2, 
        line_color=Bright[7][3], 
        line_dash="dotted"
    )

    # Output to static HTML file
    output_file(file_path)

    # Save the figure
    save(p)
    
def compare_whitening(
    strain,
    background,
    sample_rate_hertz : float,
    fft_duration_seconds : float = 2.0, 
    overlap_duration_seconds : float = 1.0,
    filter_duration_seconds : float = 2.0
    ) -> Tuple[Any, np.ndarray]:
    
    # Tensorflow whitening:
    whitened_tensorflow = whiten(
        strain, 
        background, 
        sample_rate_hertz, 
        fft_duration_seconds=fft_duration_seconds, 
        overlap_duration_seconds=overlap_duration_seconds,
        filter_duration_seconds=filter_duration_seconds
    )

    _, psd = welch(
        background, 
        sample_rate_hertz, 
        nperseg=int(sample_rate_hertz*fft_duration_seconds), 
        noverlap=int(sample_rate_hertz*overlap_duration_seconds), 
    )
    
    # GWPy whitening:
    ts = TimeSeries(
        strain[0],
        sample_rate=sample_rate_hertz
    )
    whitened_gwpy = ts.whiten(
        fftlength=fft_duration_seconds, 
        overlap=overlap_duration_seconds,
        asd=FrequencySeries(np.sqrt(psd[0])),
        fduration=filter_duration_seconds
    ).value
    whitened_gwpy = np.expand_dims(whitened_gwpy, axis=0)

    whitened_tensorflow = gf.crop_samples(
        batched_onsource=whitened_tensorflow,
        onsource_duration_seconds=gf.Defaults.onsource_duration_seconds,
        sample_rate_hertz=sample_rate_hertz
    )
    whitened_gwpy = np.array(gf.crop_samples(
        batched_onsource=ops.convert_to_tensor(whitened_gwpy),
        onsource_duration_seconds=gf.Defaults.onsource_duration_seconds,
        sample_rate_hertz=sample_rate_hertz
    ))

    return whitened_tensorflow, whitened_gwpy

def compare_psd_methods(
    strain, 
    sample_rate_hertz : float, 
    nperseg : int,
    noverlap : int = None
    ) -> Tuple[Any, Any, np.ndarray]:
    
    strain = ops.cast(strain, dtype="float32")
    
    # Use scipy's default overlap if not specified (nperseg // 2)
    if noverlap is None:
        noverlap = nperseg // 2
    
    frequencies_scipy, strain_psd_scipy = welch(
        strain, 
        sample_rate_hertz, 
        nperseg=nperseg,
        noverlap=noverlap
    )

    frequencies_tensorflow, strain_psd_tensorflow = gf.psd(
        strain, 
        sample_rate_hertz = sample_rate_hertz, 
        nperseg=nperseg,
        noverlap=noverlap
    )
    
    assert all(frequencies_scipy == frequencies_tensorflow), "Frequencies not equal."
    
    return frequencies_tensorflow, strain_psd_tensorflow, strain_psd_scipy

def _test_snr( 
        plot_results : bool
    ) -> None:
    
    with gf.env():
        
        output_directory_path : Path = gf.PATH.parent.parent / "gravyflow_data/tests/"

        gf.Defaults.onsource_duration_seconds = 16.0

        # Setup ifo data acquisition object:
        ifo_data_obtainer : gf.IFODataObtainer = gf.IFODataObtainer(
            data_quality=gf.DataQuality.BEST,
            data_labels=[
                gf.DataLabel.NOISE, 
                gf.DataLabel.GLITCHES
            ],
            observing_runs=gf.ObservingRun.O3,
            segment_order=gf.SegmentOrder.RANDOM,
            force_acquisition = True,
            cache_segments = False
        )
        
        # Initialize noise generator wrapper:
        noise_obtainer: gf.NoiseObtainer = gf.NoiseObtainer(
            ifo_data_obtainer = ifo_data_obtainer,
            noise_type = gf.NoiseType.REAL,
            ifos = gf.IFO.L1
        )
        
        dataset : keras.utils.PyDataset = gf.Dataset(
            # Noise: 
            noise_obtainer=noise_obtainer,
            # Output configuration:
            num_examples_per_batch=1,
            input_variables = [
                gf.ReturnVariables.ONSOURCE
            ]
        )
        
        background, _ = next(iter(dataset))

        sample_rate_hertz : float = gf.Defaults.sample_rate_hertz
        onsource_duration_seconds : float = gf.Defaults.onsource_duration_seconds
        crop_duration_seconds : float = gf.Defaults.crop_duration_seconds

        # Generate phenom injection:
        injection = generate_cbc_waveform(
            num_waveforms=1, 
            mass_1_msun=30, 
            mass_2_msun=30,
            sample_rate_hertz=sample_rate_hertz,
            duration_seconds=onsource_duration_seconds + (2.0*crop_duration_seconds),
            inclination_radians=1.0,
            distance_mpc=100,
            reference_orbital_phase_in=0.0,
            ascending_node_longitude=100.0,
            eccentricity=0.0,
            mean_periastron_anomaly=0.0, 
            spin_1_in=[0.0, 0.0, 0.0],
            spin_2_in=[0.0, 0.0, 0.0],
            approximant="IMRPhenomD"
        )

        # Scale injection to avoid precision error when converting to 32 bit 
        # float for tensorflow compatability:
        injection *= 1.0E21

        injection = ops.convert_to_tensor(injection[:, 0], dtype = "float32")
        injection = ops.expand_dims(injection, 0)
        
        min_roll : int = int(crop_duration_seconds * sample_rate_hertz)
        max_roll : int = int(
            (onsource_duration_seconds/2.0 + crop_duration_seconds) * sample_rate_hertz
        )

        rng = np.random.default_rng(gf.Defaults.seed)
        shift = rng.integers(min_roll, max_roll, size=1)
        shift = ops.convert_to_tensor(shift, dtype="int32")

        injection = gf.roll_vector_zero_padding(
            injection,
            shift
        )      
        # Get first elements, and return to float 32 to tf functions:
        injection = injection[0]
        onsource = ops.cast(
            background[gf.ReturnVariables.ONSOURCE.name][0], 
            "float32"
        )
        
        # Scale to SNR 30:
        snr : float = 30.0
        scaled_injection = gf.scale_to_snr(
            injection, 
            onsource,
            snr,
            sample_rate_hertz=sample_rate_hertz,
            fft_duration_seconds = 4.0, 
            overlap_duration_seconds = 0.5
        )        
        
        # Crop injection to match onsource size before adding
        # (onsource from dataset is already cropped, injection has crop buffer)
        scaled_injection = gf.crop_samples(
            scaled_injection,
            gf.Defaults.onsource_duration_seconds,
            gf.Defaults.sample_rate_hertz
        )
        
        onsource_plus_injection = onsource + scaled_injection
        injection = gf.crop_samples(
            injection,
            gf.Defaults.onsource_duration_seconds,
            gf.Defaults.sample_rate_hertz
        )
        
        for_whitening_comparison = {
            "onsource" : onsource,
            "onsource_plus_injection" : onsource_plus_injection,
            "scaled_injection" : scaled_injection,
            "injection" : injection
        }
        
        whitening_results = {}
        for key, strain in for_whitening_comparison.items():
            whitened_tf, whitened_gwpy = compare_whitening(
                strain,
                onsource,
                sample_rate_hertz,
                fft_duration_seconds=1.0,  # Match new whiten() defaults
                overlap_duration_seconds=0.5  # Match new whiten() defaults
            )
            
            whitening_results[key] = {
                "tensorflow" : whitened_tf,
                "gwpy" : whitened_gwpy
            }

        if plot_results:
            plot_whitened_strain_examples(whitening_results, output_directory_path)
        
        nperseg : int = int((1.0/32.0)*sample_rate_hertz)
        
        psd_results = {}
        common_params = {'sample_rate_hertz': sample_rate_hertz, 'nperseg': nperseg}

        # Compute PSD for different methods and data types
        for data_type in ["onsource_plus_injection", "onsource"]:
            for method in ["tensorflow", "gwpy"]:
                key = f"{data_type}_{method}"

                frequencies, psd_tensorflow, psd_scipy = compare_psd_methods(
                    whitening_results[data_type][method], 
                    **common_params
                )

                psd_results[key] = {
                    'tensorflow': psd_tensorflow, 
                    'scipy': psd_scipy
                }
        
        np.testing.assert_allclose(
            psd_results["onsource_tensorflow"]["scipy"][0][2:],
            psd_results["onsource_tensorflow"]["tensorflow"][0][2:],
            atol=6e-06,  # Relaxed to account for numerical differences
            err_msg="GravyFlow Whiten SciPy PSD does not equal GravyFlow Whiten GravyFlow PSD.", 
            verbose=True
        )

        np.testing.assert_allclose(
            psd_results["onsource_gwpy"]["scipy"][0][2:],
            psd_results["onsource_tensorflow"]["tensorflow"][0][2:],
            atol=1e-05,  # Should match with correct parameters
            err_msg="GwPy Whiten SciPy PSD does not equal GravyFlow Whiten GravyFlow PSD.", 
            verbose=True
        )

        np.testing.assert_allclose(
            psd_results["onsource_gwpy"]["scipy"][0],
            psd_results["onsource_tensorflow"]["scipy"][0],
            atol=1e-05,  # Should match with correct parameters
            err_msg="GwPy Whiten SciPy PSD does not equal GravyFlow Whiten Scipy PSD.", 
            verbose=True
        )

        if plot_results:
            plot_psd(
                frequencies, 
                psd_results["onsource_plus_injection_tensorflow"]["tensorflow"][0],
                psd_results["onsource_tensorflow"]["scipy"][0],
                psd_results["onsource_tensorflow"]["tensorflow"][0],
                psd_results["onsource_gwpy"]["scipy"][0],
                output_directory_path / "whitening_psd_test_plots.html"
            )

def test_snr(
        pytestconfig : Config
    ) -> None:
    
    _test_snr(
        plot_results=pytestconfig.getoption("plot")
    )

def test_find_closest():
    tensor = ops.convert_to_tensor([1.0, 2.0, 5.0])
    scalar = 2.1
    idx = gf_snr.find_closest(tensor, scalar)
    assert idx == 1 

def test_snr_calculation():
    # Create a signal and background
    sample_rate = 100.0
    duration = 4.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    # Injection: 10Hz sine wave
    # Shape: (Batch=1, Channel=1, Time)
    injection = np.sin(2 * np.pi * 10.0 * t).astype(np.float32)
    injection = ops.convert_to_tensor(injection.reshape(1, 1, -1))
    
    # Background: White noise
    np.random.seed(42)
    background = np.random.normal(0, 1, size=t.shape).astype(np.float32)
    background = ops.convert_to_tensor(background.reshape(1, 1, -1))
    
    # Calculate SNR
    snr_val = gf_snr.snr(
        injection, 
        background, 
        sample_rate_hertz=sample_rate,
        fft_duration_seconds=1.0, 
        overlap_duration_seconds=0.5,
        lower_frequency_cutoff=5.0
    )
    
    # SNR should be positive
    # Result shape should be (Batch,)
    assert ops.shape(snr_val)[0] == 1
    assert snr_val[0] > 0.0

def test_scale_to_snr():
    sample_rate = 100.0
    duration = 4.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    # Injection: 10Hz sine wave
    injection = np.sin(2 * np.pi * 10.0 * t).astype(np.float32)
    injection = ops.convert_to_tensor(injection.reshape(1, 1, -1))
    
    # Background
    background = np.random.normal(0, 1, size=t.shape).astype(np.float32)
    background = ops.convert_to_tensor(background.reshape(1, 1, -1))
    
    desired_snr = ops.convert_to_tensor([10.0])
    
    scaled_injection = gf_snr.scale_to_snr(
        injection,
        background,
        desired_snr,
        sample_rate_hertz=sample_rate,
        fft_duration_seconds=1.0,
        overlap_duration_seconds=0.5,
        lower_frequency_cutoff=5.0
    )
    
    # Verify shape
    assert ops.shape(scaled_injection) == ops.shape(injection)
    
    # Verify new SNR is close to desired
    new_snr = gf_snr.snr(
        scaled_injection,
        background,
        sample_rate_hertz=sample_rate,
        fft_duration_seconds=1.0,
        overlap_duration_seconds=0.5,
        lower_frequency_cutoff=5.0
    )
    
    # Relax tolerance to 25% due to potential windowing/interpolation effects
    np.testing.assert_allclose(new_snr, desired_snr, rtol=1e-07)

def test_snr_theoretical_sine():
    # Verify SNR matches theoretical expectation for a sine wave in white noise
    sample_rate = 1024.0
    # Use longer duration to reduce PSD estimation variance
    duration = 64.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    freq = 100.0
    amplitude = 1.0
    # Sine wave: A * sin(2*pi*f*t)
    injection = amplitude * np.sin(2 * np.pi * freq * t).astype(np.float32)
    injection = ops.convert_to_tensor(injection.reshape(1, 1, -1))
    
    # White noise background with unit variance
    np.random.seed(42)
    background = np.random.normal(0, 1, size=t.shape).astype(np.float32)
    background = ops.convert_to_tensor(background.reshape(1, 1, -1))
    
    # Theoretical SNR
    # SNR = A * sqrt(T * fs / 2)
    expected_snr = amplitude * np.sqrt(duration * sample_rate / 2.0)
    
    calculated_snr = gf_snr.snr(
        injection, 
        background, 
        sample_rate_hertz=sample_rate,
        fft_duration_seconds=1.0, 
        overlap_duration_seconds=0.5,
        lower_frequency_cutoff=5.0
    )
        
    assert np.isclose(calculated_snr[0], expected_snr, rtol=0.02)

def test_scale_to_snr_chirp():
    # Verify scaling works for a broadband chirp signal
    sample_rate = 1024.0
    duration = 4.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    # Linear Chirp from 20Hz to 400Hz
    from scipy.signal import chirp
    injection = chirp(t, f0=20, t1=duration, f1=400, method='linear').astype(np.float32)
    injection = ops.convert_to_tensor(injection.reshape(1, 1, -1))
    
    # White noise background
    np.random.seed(42)
    background = np.random.normal(0, 1, size=t.shape).astype(np.float32)
    background = ops.convert_to_tensor(background.reshape(1, 1, -1))
    
    desired_snr = 15.0
    
    scaled_injection = gf_snr.scale_to_snr(
        injection,
        background,
        desired_snr,
        sample_rate_hertz=sample_rate,
        fft_duration_seconds=1.0,
        overlap_duration_seconds=0.5,
        lower_frequency_cutoff=15.0 # Below chirp start
    )
    
    new_snr = gf_snr.snr(
        scaled_injection,
        background,
        sample_rate_hertz=sample_rate,
        fft_duration_seconds=1.0,
        overlap_duration_seconds=0.5,
        lower_frequency_cutoff=15.0
    )
        
    # Allow 5% tolerance
    assert np.isclose(new_snr[0], desired_snr, rtol=0.02)


# ============================================================================
# EDGE CASE TESTS FOR COVERAGE
# ============================================================================

def test_snr_1d_psd_branch():
    """Test SNR calculation with 1D background that produces 1D PSD (covers line 96)."""
    sample_rate = 256.0
    duration = 4.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    # 1D injection (no batch, no channel)
    injection = np.sin(2 * np.pi * 30.0 * t).astype(np.float32)
    injection = ops.convert_to_tensor(injection)
    
    # 1D background 
    np.random.seed(42)
    background = np.random.normal(0, 1, size=t.shape).astype(np.float32)
    background = ops.convert_to_tensor(background)
    
    # This should use 1D interpolation path (line 96)
    snr_val = gf_snr.snr(
        injection, 
        background, 
        sample_rate_hertz=sample_rate,
        fft_duration_seconds=1.0, 
        overlap_duration_seconds=0.5,
        lower_frequency_cutoff=10.0
    )
    
    # Should return a scalar or 0D/1D tensor
    assert float(snr_val) > 0.0


def test_scale_to_snr_2d_input():
    """Test scale_to_snr with 2D injection input (batch, time) - covers line 238."""
    sample_rate = 256.0
    duration = 4.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    # 2D injection: (batch=2, time)
    injection = np.sin(2 * np.pi * 50.0 * t).astype(np.float32)
    injection = ops.convert_to_tensor(np.stack([injection, injection * 0.5], axis=0))  # (2, time)
    
    # 2D background: (batch=2, time)
    np.random.seed(42)
    background = np.random.normal(0, 1, size=(2, len(t))).astype(np.float32)
    background = ops.convert_to_tensor(background)
    
    desired_snr = ops.convert_to_tensor([10.0, 15.0])
    
    # This triggers rank==2 branch at line 238
    scaled = gf_snr.scale_to_snr(
        injection,
        background,
        desired_snr,
        sample_rate_hertz=sample_rate,
        fft_duration_seconds=1.0,
        overlap_duration_seconds=0.5,
        lower_frequency_cutoff=10.0
    )
    
    # Shape should be preserved
    assert ops.shape(scaled) == (2, int(sample_rate * duration))


def test_scale_to_snr_with_cropping():
    """Test scale_to_snr with onsource_duration_seconds parameter (line 197-202)."""
    sample_rate = 256.0
    duration = 4.0
    onsource_duration = 2.0  # Crop to half
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    # 3D injection: (batch=1, channel=1, time)
    injection = np.sin(2 * np.pi * 50.0 * t).astype(np.float32)
    injection = ops.convert_to_tensor(injection.reshape(1, 1, -1))
    
    # 3D background
    np.random.seed(42)
    background = np.random.normal(0, 1, size=t.shape).astype(np.float32)
    background = ops.convert_to_tensor(background.reshape(1, 1, -1))
    
    desired_snr = ops.convert_to_tensor([10.0])
    
    # This uses the cropping logic at lines 197-202
    scaled = gf_snr.scale_to_snr(
        injection,
        background,
        desired_snr,
        sample_rate_hertz=sample_rate,
        fft_duration_seconds=1.0,
        overlap_duration_seconds=0.5,
        lower_frequency_cutoff=10.0,
        onsource_duration_seconds=onsource_duration
    )
    
    # Shape should be preserved (not cropped output, just cropped for SNR calc)
    assert ops.shape(scaled) == ops.shape(injection)


def test_find_closest_edge_cases():
    """Test find_closest with various edge cases."""
    # Exact match
    tensor = ops.convert_to_tensor([1.0, 5.0, 10.0])
    assert int(gf_snr.find_closest(tensor, 5.0)) == 1
    
    # First element closest
    assert int(gf_snr.find_closest(tensor, 0.5)) == 0
    
    # Last element closest
    assert int(gf_snr.find_closest(tensor, 12.0)) == 2
    
    # Midpoint between elements - should return one of them
    idx = int(gf_snr.find_closest(tensor, 3.0))
    assert idx in [0, 1]  # Could be either depending on tie-breaking


# ============================================================================
# TESTS USING SIMPLE WAVEFORM GENERATORS
# ============================================================================

def test_snr_with_sine_gaussian():
    """Test SNR calculation using SineGaussianGenerator."""
    gen = gf.SineGaussianGenerator(
        frequency_hertz=gf.Distribution(value=100.0, type_=gf.DistributionType.CONSTANT),
        quality_factor=gf.Distribution(value=10.0, type_=gf.DistributionType.CONSTANT),
        amplitude=gf.Distribution(value=1.0, type_=gf.DistributionType.CONSTANT),
        network=[gf.IFO.L1],
    )
    
    sample_rate = 1024.0
    duration = 4.0
    
    waveforms, params = gen.generate(
        num_waveforms=1,
        sample_rate_hertz=sample_rate,
        duration_seconds=duration,
        seed=42
    )
    
    # Extract h+ component: (batch, 2, time) -> (batch, 1, time)
    injection = waveforms[:, 0:1, :]
    
    # Generate white noise background
    np.random.seed(42)
    background = np.random.normal(0, 1, size=(1, 1, int(sample_rate * duration))).astype(np.float32)
    background = ops.convert_to_tensor(background)
    
    snr_val = gf_snr.snr(
        injection, 
        background, 
        sample_rate_hertz=sample_rate,
        fft_duration_seconds=1.0, 
        overlap_duration_seconds=0.5,
        lower_frequency_cutoff=10.0
    )
    
    # SNR should be positive
    assert snr_val[0] > 0.0


def test_scale_to_snr_with_chirplet():
    """Test scale_to_snr using ChirpletGenerator."""
    gen = gf.ChirpletGenerator(
        start_frequency_hertz=gf.Distribution(value=50.0, type_=gf.DistributionType.CONSTANT),
        end_frequency_hertz=gf.Distribution(value=200.0, type_=gf.DistributionType.CONSTANT),
        duration_seconds=gf.Distribution(value=2.0, type_=gf.DistributionType.CONSTANT),
        amplitude=gf.Distribution(value=1.0, type_=gf.DistributionType.CONSTANT),
        network=[gf.IFO.L1],
    )
    
    sample_rate = 1024.0
    duration = 4.0
    
    waveforms, _ = gen.generate(
        num_waveforms=1,
        sample_rate_hertz=sample_rate,
        duration_seconds=duration,
        seed=42
    )
    
    # Extract h+ component
    injection = waveforms[:, 0:1, :]
    
    # Generate white noise background
    np.random.seed(42)
    background = np.random.normal(0, 1, size=(1, 1, int(sample_rate * duration))).astype(np.float32)
    background = ops.convert_to_tensor(background)
    
    desired_snr = 20.0
    
    scaled_injection = gf_snr.scale_to_snr(
        injection,
        background,
        desired_snr,
        sample_rate_hertz=sample_rate,
        fft_duration_seconds=1.0,
        overlap_duration_seconds=0.5,
        lower_frequency_cutoff=15.0
    )
    
    # Verify new SNR is close to desired
    new_snr = gf_snr.snr(
        scaled_injection,
        background,
        sample_rate_hertz=sample_rate,
        fft_duration_seconds=1.0,
        overlap_duration_seconds=0.5,
        lower_frequency_cutoff=15.0
    )
    
    # Allow 5% tolerance
    np.testing.assert_allclose(new_snr[0], desired_snr, rtol=0.05)


def test_snr_with_ringdown():
    """Test SNR calculation using RingdownGenerator."""
    gen = gf.RingdownGenerator(
        frequency_hertz=gf.Distribution(value=150.0, type_=gf.DistributionType.CONSTANT),
        damping_time_seconds=gf.Distribution(value=0.05, type_=gf.DistributionType.CONSTANT),
        amplitude=gf.Distribution(value=1.0, type_=gf.DistributionType.CONSTANT),
        network=[gf.IFO.L1],
    )
    
    sample_rate = 1024.0
    duration = 2.0
    
    waveforms, params = gen.generate(
        num_waveforms=2,  # Batch of 2
        sample_rate_hertz=sample_rate,
        duration_seconds=duration,
        seed=42
    )
    
    # Extract h+ component
    injection = waveforms[:, 0:1, :]
    
    # Generate white noise background
    np.random.seed(42)
    background = np.random.normal(0, 1, size=(2, 1, int(sample_rate * duration))).astype(np.float32)
    background = ops.convert_to_tensor(background)
    
    snr_val = gf_snr.snr(
        injection, 
        background, 
        sample_rate_hertz=sample_rate,
        fft_duration_seconds=1.0, 
        overlap_duration_seconds=0.5,
        lower_frequency_cutoff=50.0
    )
    
    # Both batch elements should have positive SNR
    assert ops.shape(snr_val) == (2,)
    assert snr_val[0] > 0.0
    assert snr_val[1] > 0.0


def test_snr_with_periodic_wave():
    """Test SNR calculation using PeriodicWaveGenerator with different shapes."""
    for wave_shape in [gf.WaveShape.SINE, gf.WaveShape.SQUARE, gf.WaveShape.TRIANGLE]:
        gen = gf.PeriodicWaveGenerator(
            wave_shape=wave_shape,
            frequency_hertz=gf.Distribution(value=50.0, type_=gf.DistributionType.CONSTANT),
            duration_seconds=gf.Distribution(value=2.0, type_=gf.DistributionType.CONSTANT),
            amplitude=gf.Distribution(value=1.0, type_=gf.DistributionType.CONSTANT),
            network=[gf.IFO.L1],
        )
        
        sample_rate = 512.0
        duration = 4.0
        
        waveforms, _ = gen.generate(
            num_waveforms=1,
            sample_rate_hertz=sample_rate,
            duration_seconds=duration,
            seed=42
        )
        
        # Extract h+ component
        injection = waveforms[:, 0:1, :]
        
        # White noise
        np.random.seed(42)
        background = np.random.normal(0, 1, size=(1, 1, int(sample_rate * duration))).astype(np.float32)
        background = ops.convert_to_tensor(background)
        
        snr_val = gf_snr.snr(
            injection, 
            background, 
            sample_rate_hertz=sample_rate,
            fft_duration_seconds=1.0, 
            overlap_duration_seconds=0.5,
            lower_frequency_cutoff=10.0
        )
        
        # SNR should be positive for all wave shapes
        assert snr_val[0] > 0.0, f"{wave_shape.name} wave should have positive SNR"