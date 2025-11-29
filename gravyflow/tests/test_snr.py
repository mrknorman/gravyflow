#Built-in imports
from pathlib import Path
from typing import Dict, Tuple
import logging

#Library imports
import tensorflow as tf
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

def plot_whitened_strain_examples(
        whitening_results : Dict,
        output_directory_path : Path
    ) -> None:

    layout = [
        [
            gf.generate_strain_plot(
                {
                    "Whitened (tf) Onsouce + Injection": whitening_results["onsource_plus_injection"]["tensorflow"],
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
                    "Whitened (gwpy) Onsouce + Injection": whitening_results["onsource_plus_injection"]["gwpy"],
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
    strain : tf.Tensor,
    background: tf.Tensor,
    sample_rate_hertz : float,
    fft_duration_seconds : float = 2.0, 
    overlap_duration_seconds : float = 1.0
    ) -> Tuple[tf.Tensor, np.ndarray]:
    
    # Tensorflow whitening:
    whitened_tensorflow = gf.whiten(
        strain, 
        background, 
        sample_rate_hertz, 
        fft_duration_seconds=fft_duration_seconds, 
        overlap_duration_seconds=overlap_duration_seconds
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
        asd=FrequencySeries(np.sqrt(psd[0]))
    ).value
    whitened_gwpy = np.expand_dims(whitened_gwpy, axis=0)

    whitened_tensorflow = gf.crop_samples(
        batched_onsource=whitened_tensorflow,
        onsource_duration_seconds=gf.Defaults.onsource_duration_seconds,
        sample_rate_hertz=sample_rate_hertz
    )
    whitened_gwpy = gf.crop_samples(
        batched_onsource=tf.convert_to_tensor(whitened_gwpy),
        onsource_duration_seconds=gf.Defaults.onsource_duration_seconds,
        sample_rate_hertz=sample_rate_hertz
    ).numpy()

    return whitened_tensorflow, whitened_gwpy

def compare_psd_methods(
    strain : tf.Tensor, 
    sample_rate_hertz : float, 
    nperseg : int
    ) -> Tuple[tf.Tensor, tf.Tensor, np.ndarray]:
    
    strain = tf.cast(strain, dtype=tf.float32)
    
    frequencies_scipy, strain_psd_scipy = welch(
        strain, 
        sample_rate_hertz, 
        nperseg=nperseg
    )

    frequencies_tensorflow, strain_psd_tensorflow = gf.psd(
        strain, 
        sample_rate_hertz = sample_rate_hertz, 
        nperseg=nperseg
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
            gf.ObservingRun.O3, 
            gf.DataQuality.BEST, 
            [
                gf.DataLabel.NOISE, 
                gf.DataLabel.GLITCHES
            ],
            gf.SegmentOrder.RANDOM,
            force_acquisition = True,
            cache_segments = False
        )
        
        # Initilise noise generator wrapper:
        noise_obtainer: gf.NoiseObtainer = gf.NoiseObtainer(
            ifo_data_obtainer = ifo_data_obtainer,
            noise_type = gf.NoiseType.REAL,
            ifos = gf.IFO.L1
        )
        
        dataset : tf.data.Dataset = gf.Dataset(
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
        injection = gf.imrphenomd(
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
            spin_2_in=[0.0, 0.0, 0.0]
        )

        # Scale injection to avoid precision error when converting to 32 bit 
        # float for tensorflow compatability:
        injection *= 1.0E21

        injection = tf.convert_to_tensor(injection[:, 0], dtype = tf.float32)
        injection = tf.expand_dims(injection, 0)
        
        min_roll : int = int(crop_duration_seconds * sample_rate_hertz)
        max_roll : int = int(
            (onsource_duration_seconds/2.0 + crop_duration_seconds) * sample_rate_hertz
        )

        rng = np.random.default_rng(gf.Defaults.seed)
        injection = gf.roll_vector_zero_padding(
            tensor=injection, 
            min_roll=min_roll, 
            max_roll=max_roll,
            seed=rng.integers(1E10, size=2)
        )
        
        # Get first elements, and return to float 32 to tf functions:
        injection = injection[0]
        onsource = tf.cast(
            background[gf.ReturnVariables.ONSOURCE.name][0], 
            tf.float32
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
        
        onsource_plus_injection = onsource + scaled_injection

        scaled_injection = gf.crop_samples(
            scaled_injection,
            gf.Defaults.onsource_duration_seconds,
            gf.Defaults.sample_rate_hertz
        )
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
                fft_duration_seconds=2.0,
                overlap_duration_seconds=1.0
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
                    'tensorflow': psd_tensorflow.numpy(), 
                    'scipy': psd_scipy
                }
        
        np.testing.assert_allclose(
            psd_results["onsource_tensorflow"]["scipy"][0][2:],
            psd_results["onsource_tensorflow"]["tensorflow"][0][2:],
            atol=1e-07, 
            err_msg="GravyFlow Whiten SciPy PSD does not equal GravyFlow Whiten GravyFlow PSD.", 
            verbose=True
        )

        np.testing.assert_allclose(
            psd_results["onsource_gwpy"]["scipy"][0][2:],
            psd_results["onsource_tensorflow"]["tensorflow"][0][2:],
            atol=1e-05, 
            err_msg="GwPy Whiten SciPy PSD does not equal GravyFlow Whiten GravyFlow PSD.", 
            verbose=True
        )

        np.testing.assert_allclose(
            psd_results["onsource_gwpy"]["scipy"][0],
            psd_results["onsource_tensorflow"]["scipy"][0],
            atol=1e-05, 
            err_msg="GwPy Whiten SciPy PSD does not equal GravyFlow Whiten Scipy PSD.", 
            verbose=True
        )

        if plot_results:
            plot_psd(
                frequencies.numpy(), 
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