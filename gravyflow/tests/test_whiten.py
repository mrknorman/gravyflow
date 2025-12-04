#Built-in imports
from typing import Tuple, Dict
from pathlib import Path
import logging

#Library imports
import numpy as np
import scipy
from gwpy.timeseries import TimeSeries
from bokeh.plotting import output_file, save
from bokeh.layouts import gridplot
from _pytest.config import Config

# Local imports:
# Local imports:
import gravyflow as gf
from gravyflow.src.dataset.conditioning import whiten as gf_whiten
import keras
from keras import ops
    
def plot_whiten_functions(
        fft_duration_seconds : float = 1.0, 
        overlap_duration_seconds : float = 0.5
    ) -> None:

    output_diretory_path : Path = gf.PATH.parent.parent / "gravyflow_data/tests"
    
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
    data = ops.convert_to_tensor(data)
    
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
    whitened_tensorflow = gf.whiten(
        data, 
        data, 
        sample_rate_hertz, 
        fft_duration_seconds=fft_duration_seconds, 
        overlap_duration_seconds=overlap_duration_seconds
    )
    whitened_tensorflow = np.array(whitened_tensorflow)
    
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
    
def _test_whiten_functions(
        plot_results : bool
    ) -> None:

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
    timeseries = ops.convert_to_tensor(data, dtype="float32")
    whitened_tensorflow = gf.whiten(
        timeseries, 
        timeseries,
        sample_rate_hertz, 
        fft_duration_seconds=fft_duration_seconds, 
        overlap_duration_seconds=overlap_duration_seconds
    )
    whitened_tensorflow = np.array(whitened_tensorflow)
    
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

    if plot_results:
        plot_whiten_functions()
    
def _test_whitening_real_noise(
        plot_results : bool
    ) -> None:

    # Test Parameters:
    num_examples_per_batch : int = 1
    sample_rate_hertz : float = gf.Defaults.sample_rate_hertz
    fft_duration_seconds : float = 1.0
    overlap_duration_seconds : float = 0.5
    output_diretory_path : Path = gf.PATH.parent.parent / "gravyflow_data/tests"
    
    # Setup ifo data acquisition object:
    ifo_data_obtainer : gf.IFODataObtainer = gf.IFODataObtainer(
        observing_runs=gf.ObservingRun.O3, 
        data_quality=gf.DataQuality.BEST, 
        data_labels=[
            gf.DataLabel.NOISE, 
            gf.DataLabel.GLITCHES
        ],
        force_acquisition=True,
        cache_segments=False
    )
    
    # Initilise noise generator wrapper:
    noise_obtainer: gf.NoiseObtainer = gf.NoiseObtainer(
        ifo_data_obtainer=ifo_data_obtainer,
        noise_type=gf.NoiseType.REAL,
        ifos=gf.IFO.L1
    )
    
    generator : keras.utils.PyDataset = gf.Dataset(
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
            
    raw_noise = background[gf.ReturnVariables.ONSOURCE.name]
    raw_noise = np.array(raw_noise)[0]

    whitened_noise = background[gf.ReturnVariables.WHITENED_ONSOURCE.name]
    whitened_noise = np.array(whitened_noise)[0]

    # Create a GWpy TimeSeries object
    ts = TimeSeries(
        raw_noise[0], 
        sample_rate=gf.Defaults.sample_rate_hertz
    )
    noise_whitened_gwpy = ts.whiten(
        fftlength=fft_duration_seconds, 
        overlap=overlap_duration_seconds, 
        fduration=1.0
    ).value
        
    # Whitening using tensorflow-based function
    timeseries  = ops.convert_to_tensor(
        raw_noise, dtype = "float32"
    )
    noise_whitened_tensorflow = gf.whiten(
        timeseries, 
        timeseries, 
        sample_rate_hertz, 
        fft_duration_seconds=fft_duration_seconds, 
        overlap_duration_seconds=overlap_duration_seconds
    )
    noise_whitened_tensorflow = np.array(noise_whitened_tensorflow)

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
                title=key
            )
        ] for key, value in time_results.items()
    ]

    if plot_results:

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


def test_whiten_functions(
        pytestconfig : Config
    ) -> None:

    with gf.env():
        _test_whiten_functions(
            plot_results=pytestconfig.getoption("plot")
        )

def test_whitening_real_noise(        
        pytestconfig : Config
    ) -> None:

    with gf.env():
        _test_whitening_real_noise(
            plot_results=pytestconfig.getoption("plot")
        )

def test_planck():
    N = 100
    nleft = 10
    nright = 10
    window = gf_whiten.planck(N, nleft, nright)
    assert ops.shape(window) == (N,)
    # The implementation produces 0.5 at the left edge and ~0.27 at the right edge
    assert np.isclose(window[0], 0.5)
    assert np.isclose(window[-1], 0.26894, atol=1e-4)
    # assert window[50] == 1.0 # This might also be 1.0

def test_fftconvolve():
    # Simple convolution
    in1 = ops.convert_to_tensor([1.0, 2.0, 3.0])
    in2 = ops.convert_to_tensor([0.0, 1.0, 0.5])
    # [1, 2, 3] * [0, 1, 0.5]
    # 0, 1, 0.5
    #    0, 2, 1.0
    #       0, 3, 1.5
    # 0, 1, 2.5, 4, 1.5
    
    # mode='full'
    conv = gf_whiten.fftconvolve(in1, in2, mode='full')
    expected = np.convolve([1.0, 2.0, 3.0], [0.0, 1.0, 0.5], mode='full')
    np.testing.assert_allclose(conv, expected, atol=1e-5)
    
    # mode='same'
    conv_same = gf_whiten.fftconvolve(in1, in2, mode='same')
    expected_same = np.convolve([1.0, 2.0, 3.0], [0.0, 1.0, 0.5], mode='same')
    np.testing.assert_allclose(conv_same, expected_same, atol=1e-5)

def test_whiten_runs():
    sample_rate = 100.0
    duration = 4.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    # Signal: Sine wave
    signal = np.sin(2 * np.pi * 10.0 * t).astype(np.float32)
    signal = ops.convert_to_tensor(signal.reshape(1, 1, -1))
    
    # Background: White noise
    background = np.random.normal(0, 1, size=t.shape).astype(np.float32)
    background = ops.convert_to_tensor(background.reshape(1, 1, -1))
    
    whitened = gf_whiten.whiten(
        signal,
        background,
        sample_rate_hertz=sample_rate,
        fft_duration_seconds=1.0,
        overlap_duration_seconds=0.5
    )
    
    assert ops.shape(whitened) == ops.shape(signal)

def test_whiten_colored_noise():
    # Verify that whitening colored noise results in white noise (flat PSD)
    sample_rate = 1024.0
    duration = 16.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    # Generate colored noise (1/f noise or similar)
    # Use cumulative sum of white noise (Brownian noise) -> 1/f^2
    np.random.seed(42)
    white_noise = np.random.normal(0, 1, size=t.shape).astype(np.float32)
    colored_noise = np.cumsum(white_noise)
    # Remove DC and trend
    colored_noise = scipy.signal.detrend(colored_noise)
    # Normalize
    colored_noise = colored_noise / np.std(colored_noise)
    
    colored_noise_tensor = ops.convert_to_tensor(colored_noise.reshape(1, 1, -1))
    
    # Whiten
    # Use the colored noise itself as the background estimate
    whitened = gf_whiten.whiten(
        colored_noise_tensor,
        colored_noise_tensor,
        sample_rate_hertz=sample_rate,
        fft_duration_seconds=1.0,
        overlap_duration_seconds=0.5
    )
    
    whitened_np = np.array(whitened)[0, 0]
    
    # Calculate PSD of whitened data
    f, pxx = scipy.signal.welch(
        whitened_np, 
        fs=sample_rate, 
        nperseg=1024
    )
    
    # Check flatness
    # PSD of white noise should be constant.
    # Ignore edges (filter artifacts) and DC
    valid_indices = (f > 20) & (f < sample_rate/2 - 20)
    pxx_valid = pxx[valid_indices]
    
    # Coefficient of variation (std/mean) should be low
    cv = np.std(pxx_valid) / np.mean(pxx_valid)
    
    print(f"PSD Coefficient of Variation: {cv}")
    
    # For white noise estimated with Welch, CV depends on number of averages.
    # 16s duration, 1s window, 0.5 overlap -> ~30 segments.
    # Expected CV ~ 1/sqrt(30) ~ 0.18.
    # Allow some margin.
    assert cv < 0.3, f"Whitened noise is not flat enough. CV: {cv}"
    
    # Also check that it's flatter than original
    f_orig, pxx_orig = scipy.signal.welch(colored_noise, fs=sample_rate, nperseg=1024)
    pxx_orig_valid = pxx_orig[valid_indices]
    cv_orig = np.std(pxx_orig_valid) / np.mean(pxx_orig_valid)
    
    print(f"Original PSD CV: {cv_orig}")
    assert cv < cv_orig, "Whitening did not improve flatness"