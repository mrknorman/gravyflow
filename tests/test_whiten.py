#Built-in imports
from typing import Tuple, Dict
from pathlib import Path
import logging

#Library imports
import tensorflow as tf
import numpy as np
import scipy
from gwpy.timeseries import TimeSeries
from bokeh.plotting import output_file, save
from bokeh.layouts import gridplot
from _pytest.config import Config

# Local imports:
import gravyflow as gf
    
def plot_whiten_functions(
        fft_duration_seconds : float = 1.0, 
        overlap_duration_seconds : float = 0.5
    ) -> None:

    output_diretory_path : Path = gf.PATH.parent / "gravyflow_data/tests"
    
    # Constants
    sample_rate_hertz: int = 8192
    duration_seconds: int = 16
    
    """
    Plot whitening functions for gravitational wave time-series data.
    
    Parameters:
    -----------
    sample_rate_hertz : int
        Sampling rate in Hz.
    duration_seconds : int
        Duration of the time-series in seconds.
    fft_length_seconds : int
        Length of the FFT window in seconds.
    overlap_fraction : float
        Fractional overlap for the FFT.
    """
    
    # Generate random time-series data
    time_axis = np.linspace(
        0, 
        duration_seconds, 
        int(duration_seconds * sample_rate_hertz), 
        endpoint=False
    )
    
    peak_1_frequency : float = 543.0
    peak_1_amplitude : float = 3.0
    
    peak_2_frequency : float = 210.0
    peak_2_amplitude : float = 5.0
    
    np.random.seed(42)
    data = ( 
        peak_1_amplitude * np.sin(2 * np.pi * peak_1_frequency * time_axis)
        + peak_2_amplitude * np.sin(2 * np.pi * peak_2_frequency * time_axis)
        + np.random.normal(size=time_axis.shape)
    )
    data = data.astype(np.float32)
    
    # Resample using GWpy
    ts = TimeSeries(data, sample_rate=sample_rate_hertz).resample(4096)
    sample_rate_hertz = 4096.0
    data = ts.value.astype(np.float32)
    data = tf.convert_to_tensor(data)
    
    # Calculate Power Spectral Density
    def calc_psd(
            series: np.ndarray
        ) -> np.ndarray:

        return scipy.signal.csd(
            series, 
            series, 
            fs=sample_rate_hertz, 
            window="hann", 
            nperseg=int(sample_rate_hertz * fft_duration_seconds), 
            noverlap=int(sample_rate_hertz * overlap_duration_seconds), 
            average="median"
        )
    
    frequencies, data_psd = calc_psd(data)
    whitened_gwpy = ts.whiten(fft_duration_seconds, overlap_duration_seconds).value
    _, gwpy_whitened_noise_psd = calc_psd(whitened_gwpy)
    
    # TensorFlow whitening
    whitened_tensorflow : tf.Tensor = gf.whiten(
        data, 
        data, 
        sample_rate_hertz, 
        fft_duration_seconds=fft_duration_seconds, 
        overlap_duration_seconds=overlap_duration_seconds
    ).numpy()
    
    _, tensorflow_whitened_noise_psd = calc_psd(whitened_tensorflow)
    
    # Calculate residuals
    residuals = np.abs(whitened_tensorflow - whitened_gwpy)
    
    # Plot results
    time_results: Dict[str, np.ndarray] = {
        "Raw Data": data,
        "Data Whitened with Tensorflow": whitened_tensorflow, 
        "Data Whitened with GWPy": whitened_gwpy, 
        "Residuals": residuals
    }

    layout = [
        [
            gf.generate_strain_plot(
                strain={key : value},
                sample_rate_hertz=sample_rate_hertz,
                title=key,
                scale_factor=1.0
            )
        ] for key, value in time_results.items()
    ]
    
    # Specify the output file and save the plot
    output_file(output_diretory_path / "noise_whitening_tests_time.html")
    
    grid = gridplot(layout)
    
    save(grid)
        
    psd_results: Dict[str, np.ndarray] = {
        "PSD of Raw Data": data_psd, 
        "PSD of Data Whitened with Tensorflow": tensorflow_whitened_noise_psd, 
        "PSD of Data Whitened with GWPy": gwpy_whitened_noise_psd
    }
    
    layout = [
        [
            gf.generate_psd_plot(
                psd={key : value},
                frequencies=frequencies,
                title=key
            )
        ] for key, value in psd_results.items()
    ]
    
    # Specify the output file and save the plot
    output_file(output_diretory_path / "noise_whitening_tests_psd.html")
    
    grid = gridplot(layout)
    
    save(grid)
    
def test_whiten_functions() -> None:
    """
    Test the behavior of whitening functions by comparing their outputs.
    """
    # Constants
    sample_rate_hertz: int = 8192
    duration_seconds: int = 16
    fft_duration_seconds: int = 1
    overlap_duration_seconds: float = 0.5
    threshold: float = 1.0E-3
    
    # Generate random time-series data
    time_axis = np.linspace(
        0, 
        duration_seconds, 
        int(duration_seconds * sample_rate_hertz), 
        endpoint=False
    )
    
    peak_1_frequency : float = 543
    peak_1_amplitude : float = 3.0
    
    peak_2_frequency : float = 210
    peak_2_amplitude : float = 5.0
    
    np.random.seed(42)
    data = peak_1_amplitude * np.sin(2 * np.pi * peak_1_frequency * time_axis) \
        + peak_2_amplitude * np.sin(2 * np.pi * peak_2_frequency * time_axis) \
        + np.random.normal(size=time_axis.shape)
    data = data.astype(np.float32)
    
    # Whitening using GWpy
    ts = TimeSeries(data, sample_rate=sample_rate_hertz)
    whitened_gwpy = ts.whiten(fftlength=fft_duration_seconds, overlap=overlap_duration_seconds).value
    
    # Whitening using TensorFlow
    timeseries = tf.convert_to_tensor(data, dtype=tf.float32)
    whitened_tensorflow = gf.whiten(
        timeseries, 
        timeseries,
        sample_rate_hertz, 
        fft_duration_seconds=fft_duration_seconds, 
        overlap_duration_seconds=overlap_duration_seconds
    )
    whitened_tensorflow = whitened_tensorflow.numpy()
    
    # Calculate PSD using Scipy
    def psd_scipy(
            data: np.ndarray
        ) -> Tuple[np.ndarray, np.ndarray]:

        return scipy.signal.csd(
            data, data, fs=sample_rate_hertz, window='hann',
            nperseg=int(sample_rate_hertz * fft_duration_seconds),
            noverlap=int(sample_rate_hertz * overlap_duration_seconds),
            average='median'
        )
    
    f, data_psd = psd_scipy(data)
    _, gwpy_whitened_noise_psd = psd_scipy(whitened_gwpy)
    _, tensorflow_whitened_noise_psd = psd_scipy(whitened_tensorflow)
    
    # Find peaks in PSD
    def find_peaks(
            psd: np.ndarray
        ) -> np.ndarray:

        peaks, _ = scipy.signal.find_peaks(np.abs(psd), threshold=threshold)
        return peaks
    
    gwpy_peaks = find_peaks(gwpy_whitened_noise_psd)
    tensorflow_peaks = find_peaks(tensorflow_whitened_noise_psd)
    data_peaks = find_peaks(data_psd)
    
    # Assert no peaks in PSD of whitened data
    assert len(gwpy_peaks) == 0, \
        "Peaks found in the PSD of the GWpy whitened data"
    assert len(tensorflow_peaks) == 0, \
        "Peaks found in the PSD of the TensorFlow whitened data"
    
def real_noise_test() -> None:

    output_diretory_path : Path = gf.PATH.parent / "gravyflow_data/tests"
    
    # Test Parameters:
    num_examples_per_batch : int = 1
    sample_rate_hertz : float = 2048.0
    onsource_duration_seconds : float = 16.0
    fft_duration_seconds : float = 1.0
    overlap_duration_seconds : float = 0.5
    offsource_duration_seconds : float = 16.0
    crop_duration_seconds : float = 0.5
    scale_factor : float = 1.0E21
    
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
            ifo_data_obtainer=ifo_data_obtainer,
            noise_type=gf.NoiseType.REAL,
            ifos=gf.IFO.L1
        )
    
    generator  : tf.data.Dataset = gf.Dataset(
            # Temporal components:
            sample_rate_hertz=sample_rate_hertz,   
            onsource_duration_seconds=onsource_duration_seconds,
            offsource_duration_seconds=offsource_duration_seconds,
            crop_duration_seconds=crop_duration_seconds,
            # Noise: 
            noise_obtainer=noise_obtainer,
            # Output configuration:
            num_examples_per_batch=num_examples_per_batch,
            input_variables = [
                gf.ReturnVariables.ONSOURCE, 
                gf.ReturnVariables.WHITENED_ONSOURCE
            ]
        )
    
    background, _ = next(iter(generator))
            
    raw_noise : np.ndarray = (
        background[
            gf.ReturnVariables.ONSOURCE.name
        ].numpy()[0]
    )
    whitened_noise : np.ndarray = (
        background[
            gf.ReturnVariables.WHITENED_ONSOURCE.name
        ].numpy()[0]
    )

    # Create a GWpy TimeSeries object
    ts = TimeSeries(
        raw_noise[0], 
        sample_rate=sample_rate_hertz
    )
    noise_whitened_gwpy = ts.whiten(
        fftlength=fft_duration_seconds, 
        overlap=overlap_duration_seconds, 
        fduration=1.0
    ).value
        
    # Whitening using tensorflow-based function
    timeseries  = tf.convert_to_tensor(
        raw_noise, dtype = tf.float32
    )
    noise_whitened_tensorflow = gf.whiten(
        timeseries, 
        timeseries, 
        sample_rate_hertz, 
        fft_duration_seconds=fft_duration_seconds, 
        overlap_duration_seconds=overlap_duration_seconds
    )
    noise_whitened_tensorflow = noise_whitened_tensorflow.numpy()

    # Calculate power spectral densities
    frequencies, raw_noise_psd = scipy.signal.csd(
        raw_noise, 
        raw_noise, 
        fs=sample_rate_hertz,
        window='hann', 
        nperseg=int(sample_rate_hertz*fft_duration_seconds),
        noverlap=int(sample_rate_hertz*overlap_duration_seconds), 
        average='median'
    )
    _, gwpy_whitened_noise_psd = scipy.signal.csd(
        noise_whitened_gwpy, 
        noise_whitened_gwpy, 
        fs=sample_rate_hertz,
        window='hann', 
        nperseg=int(sample_rate_hertz*fft_duration_seconds), 
        noverlap=int(sample_rate_hertz*overlap_duration_seconds), 
        average='median'
    ) 
    _, tensorflow_whitened_noise_psd = scipy.signal.csd(
        noise_whitened_tensorflow, 
        noise_whitened_tensorflow, 
        fs=sample_rate_hertz,
        window='hann', 
        nperseg=int(sample_rate_hertz*fft_duration_seconds), 
        noverlap=int(sample_rate_hertz*overlap_duration_seconds), 
        average='median'
    )

    residuals = np.abs(noise_whitened_tensorflow - noise_whitened_gwpy)

    time_results = {
        "Raw Noise": raw_noise, 
        "Noise Whitened by Tensorflow": noise_whitened_tensorflow,
        "Noise Whitened by GWPY": noise_whitened_gwpy, 
        "Residuals": residuals, 
        "Noise generated by Generator": whitened_noise
    }
    
    layout = [
        [
            gf.generate_strain_plot(
                strain={key : value},
                sample_rate_hertz=sample_rate_hertz,
                title=key,
                scale_factor=scale_factor
            )
        ] for key, value in time_results.items()
    ]
    
    # Specify the output file and save the plot
    output_file(output_diretory_path / "real_noise_whitening_tests_time.html")
    
    grid = gridplot(layout)
    
    save(grid)

    psd_results = {
        "Original": raw_noise_psd, 
        "Tensorflow": tensorflow_whitened_noise_psd, 
        "GWPY": gwpy_whitened_noise_psd
    }
    
    layout = [
        [gf.generate_psd_plot(
            psd={key : value},
            frequencies=frequencies,
            title=key
        )] for key, value in psd_results.items()
    ]
    
    # Specify the output file and save the plot
    output_file(output_diretory_path / "real_noise_whitening_tests_psd.html")
    
    grid = gridplot(layout)
    
    save(grid)

if __name__ == "__main__":
    
    # ---- User parameters ---- #
    
    # GPU setup:
    min_gpu_memory_mb : int = 4000
    num_gpus_to_request : int = 1
    memory_to_allocate_tf : int = 2000
    
    # Setup CUDA
    gpus = gf.find_available_GPUs(min_gpu_memory_mb, num_gpus_to_request)
    strategy = gf.setup_cuda(
        gpus, 
        max_memory_limit=memory_to_allocate_tf, 
        logging_level=logging.WARNING
    )    
    
    # Set logging level:
    logging.basicConfig(level=logging.INFO)
    
    #Run Tests:
    with strategy.scope():    
        test_whiten_functions()
        plot_whiten_functions()
        real_noise_test()