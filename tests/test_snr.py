#Built-in imports
from pathlib import Path
import logging

#Library imports
import tensorflow as tf
import numpy as np
from scipy.signal import welch
from gwpy.timeseries import TimeSeries
from bokeh.plotting import figure, output_file, save, show
from bokeh.palettes import Bright
from bokeh.layouts import gridplot
from bokeh.models import ColumnDataSource, HoverTool, Legend

# Local imports:
import gravyflow as gf
    
def plot_psd(
        frequencies, 
        onsource_plus_injection_whitened_tf_psd_scipy, 
        onsource_whitened_tf_psd_scipy, 
        onsource_whitened_tf_psd_tf, 
        onsource_whitened_gwpy_psd_scipy,
        filename
    ):
    
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
    output_file(filename)

    # Save the figure
    save(p)
    
def compare_whitening(
    strain : tf.Tensor,
    sample_rate_hertz : float,
    duration_seconds : float,
    fft_duration_seconds : float = 4.0, 
    overlap_duration_seconds : float = 2.0
    ):
    
    # Tensorflow whitening:
    whitened_tensorflow = \
        gfwhiten(
            strain, 
            strain, 
            sample_rate_hertz, 
            fft_duration_seconds=4.0, 
            overlap_duration_seconds=2.0
        )
    
    # GWPy whitening:
    ts = TimeSeries(
        strain,
        sample_rate=sample_rate_hertz
    )
    whitened_gwpy = \
        ts.whiten(
            fftlength=fft_duration_seconds, 
            overlap=overlap_duration_seconds
        ).value
    
    whitened_tensorflow = \
        gfcrop_samples(
            whitened_tensorflow,
            duration_seconds,
            sample_rate_hertz
        )

    whitened_gwpy = gfcrop_samples(
        whitened_gwpy,
        duration_seconds,
        sample_rate_hertz
        )
    
    return whitened_tensorflow, whitened_gwpy

def compare_psd_methods(
    strain : tf.Tensor, 
    sample_rate_hertz : float, 
    nperseg : int
    ):
    
    strain = tf.cast(strain, dtype=tf.float32)
    
    frequencies_scipy, strain_psd_scipy = \
        welch(
            strain, 
            sample_rate_hertz, 
            nperseg=nperseg
        )
    
    frequencies_tensorflow, strain_psd_tensorflow = \
        gfpsd(
            strain, 
            sample_rate_hertz = sample_rate_hertz, 
            nperseg=nperseg
        )
    
    assert all(frequencies_scipy == frequencies_tensorflow), \
        "Frequencies not equal."
    
    return frequencies_tensorflow, strain_psd_tensorflow, strain_psd_scipy

def test_snr(
    output_diretory_path : Path = Path("./py_ml_data/tests/")
    ):

    # Test Parameters:
    num_examples_per_generation_batch : int = 2048
    num_examples_per_batch : int = 1
    sample_rate_hertz : float = 2048.0
    onsource_duration_seconds : float = 16.0
    offsource_duration_seconds : float = 16.0
    crop_duration_seconds : float = 0.5
    scale_factor : float = 1.0E21
    
    # Setup ifo data acquisition object:
    ifo_data_obtainer : gf.IFODataObtainer = \
        gf.IFODataObtainer(
            gfObservingRun.O3, 
            DataQuality.BEST, 
            [
                gfDataLabel.NOISE, 
                gfDataLabel.GLITCHES
            ],
            gfSegmentOrder.RANDOM,
            force_acquisition = True,
            cache_segments = False
        )
    
    # Initilise noise generator wrapper:
    noise_obtainer: gfNoiseObtainer = \
        gfNoiseObtainer(
            ifo_data_obtainer = ifo_data_obtainer,
            noise_type = gfNoiseType.REAL,
            ifos = gfIFO.L1
        )
    
    dataset : tf.data.Dataset = gfDataset(
            # Random Seed:
            seed= 1000,
            # Temporal components:
            sample_rate_hertz=sample_rate_hertz,   
            onsource_duration_seconds=onsource_duration_seconds,
            offsource_duration_seconds=offsource_duration_seconds,
            crop_duration_seconds=crop_duration_seconds,
            # Noise: 
            noise_obtainer=noise_obtainer,
            # Output configuration:
            num_examples_per_batch=1,
            input_variables = [
                gfReturnVariables.ONSOURCE, 
                gfReturnVariables.OFFSOURCE
            ]
        )
    
    background, _ = next(iter(dataset))
            
    # Generate phenom injection:
    injection = \
        gf.imrphenomd(
            num_waveforms = num_examples_per_batch, 
            mass_1_msun = 30, 
            mass_2_msun = 30,
            sample_rate_hertz = sample_rate_hertz,
            duration_seconds = onsource_duration_seconds,
            inclination_radians = 1.0,
            distance_mpc = 100,
            reference_orbital_phase_in = 0.0,
            ascending_node_longitude = 100.0,
            eccentricity = 0.0,
            mean_periastron_anomaly = 0.0, 
            spin_1_in = [0.0, 0.0, 0.0],
            spin_2_in = [0.0, 0.0, 0.0]
        )

    # Scale injection to avoid precision error when converting to 32 bit 
    # float for tensorflow compatability:
    injection *= 1.0E21

    injection = tf.convert_to_tensor(injection[:, 0], dtype = tf.float32)
    injection = tf.expand_dims(injection, 0)
    
    min_roll : int = int(crop_duration_seconds * sample_rate_hertz)
    max_roll : int = int(
        (onsource_duration_seconds/2 + crop_duration_seconds) * sample_rate_hertz
    )
        
    injection = gfroll_vector_zero_padding(
        injection, 
        min_roll, 
        max_roll
    )
    
    # Get first elements, and return to float 32 to tf functions:
    injection = injection[0]
    offsource = tf.cast(
        background[gfReturnVariables.OFFSOURCE.name][0], tf.float32
    )
    onsource = tf.cast(
        background[gfReturnVariables.ONSOURCE.name][0], tf.float32
    )
    
    # Scale to SNR 30:
    snr : float = 30.0
    scaled_injection = \
        gfscale_to_snr(
            injection, 
            onsource,
            snr,
            sample_rate_hertz = sample_rate_hertz, 
            fft_duration_seconds = 4.0, 
            overlap_duration_seconds = 0.5,
        )        
                
    onsource_plus_injection = onsource + scaled_injection
    
    for_whitening_comparison = {
        "onsource" : onsource,
        "onsource_plus_injection" : onsource_plus_injection,
        "scaled_injection" : scaled_injection,
        "injection" : injection
    }
    
    whitening_results = {}
    for key, strain in for_whitening_comparison.items():
        whitened_tf, whitened_gwpy = \
            compare_whitening(
                strain,
                sample_rate_hertz,
                onsource_duration_seconds,
                fft_duration_seconds=4.0,
                overlap_duration_seconds=2.0
            )
        
        whitening_results[key] = {
            "tensorflow" : whitened_tf,
            "gwpy" : whitened_gwpy
        }
        
    layout = [
        [gfgenerate_strain_plot(
            {
                "Whitened (tf) Onsouce + Injection": \
                    whitening_results["onsource_plus_injection"]["tensorflow"],
                "Whitened (tf) Injection" : \
                    whitening_results["injection"]["tensorflow"],
                "Injection": injection
            },
            sample_rate_hertz,
            onsource_duration_seconds,
            title=f"cuPhenomD injection example tf whitening",
            scale_factor=scale_factor
        ), 
        gfgenerate_spectrogram(
            whitening_results["onsource_plus_injection"]["tensorflow"], 
            sample_rate_hertz,
        )],
        [gfgenerate_strain_plot(
            {
                "Whitened (gwpy) Onsouce + Injection": \
                    whitening_results["onsource_plus_injection"]["gwpy"],
                "Whitened (gwpy) Injection" : \
                    whitening_results["injection"]["gwpy"],
                "Injection": injection
            },
            sample_rate_hertz,
            onsource_duration_seconds,
            title=f"cuPhenomD injection example gwpy whitening",
            scale_factor=scale_factor
        ), 
        gfgenerate_spectrogram(
            whitening_results["onsource_plus_injection"]["gwpy"], 
            sample_rate_hertz,
        )]
    ]
    
    # Ensure output directory exists
    gfensure_directory_exists(output_diretory_path)
    
    # Define an output path for the dashboard
    output_file(output_diretory_path / "whitening_test_plots.html")

    # Arrange the plots in a grid. 
    grid = gridplot(layout)
        
    save(grid)
    
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

    plot_psd(
        frequencies.numpy(), 
        psd_results["onsource_plus_injection_tensorflow"]["tensorflow"],
        psd_results["onsource_tensorflow"]["scipy"],
        psd_results["onsource_tensorflow"]["tensorflow"],
        psd_results["onsource_gwpy"]["scipy"],
        Path("./py_ml_data/tests/whitening_psds.html")
    )

if __name__ == "__main__":    
    # ---- User parameters ---- #
    
    # GPU setup:
    min_gpu_memory_mb : int = 4000
    num_gpus_to_request : int = 1
    memory_to_allocate_tf : int = 2000
    
    # Setup CUDA
    gpus = gffind_available_GPUs(min_gpu_memory_mb, num_gpus_to_request)
    strategy = gfsetup_cuda(
        gpus, 
        max_memory_limit=memory_to_allocate_tf, 
        logging_level=logging.WARNING
    )    
    
    # Set logging level:
    logging.basicConfig(level=logging.INFO)
    
    # Test SNR:
    with strategy.scope():
        test_snr()