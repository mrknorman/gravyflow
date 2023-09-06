# Standard library
from pathlib import Path

# Third-party libraries
import numpy as np
import tensorflow as tf
from bokeh.plotting import figure, output_file, save
from scipy.signal import welch

# Local application imports
from ..psd import calculate_psd

def test_welch_method() -> None:
    """
    Test the Welch method for computing the Power Spectral Density (PSD).

    Steps:
    1. Generate a sinusoidal signal.
    2. Compute the PSD using scipy's Welch method.
    3. Compute the PSD using a TensorFlow-based method.
    4. Plot both PSDs for comparison.
    """

    # Step 1: Generate a sinusoidal signal
    sample_rate_hertz : float  = 8196  # Sampling rate in hertz
    duration_seconds : float = 2.0  # Time duration in seconds
    time_array_seconds : np.ndarray = np.arange(
        0.0, duration_seconds, 1.0 / sample_rate_hertz
    )  # Time array in seconds
    frequency_hertz : float = 123.4  # Frequency in hertz
    amplitude : float = 0.5
    signal_array : np.ndarray = \
        amplitude * np.sin(2 * np.pi * frequency_hertz * time_array_seconds)

    # Step 2: Compute PSD using scipy
    frequencies_scipy_hertz, psd_scipy = \
        welch(
            signal_array, sample_rate_hertz, nperseg=1024
        )

    # Step 3: Compute PSD using TensorFlow
    signal_tensor: tf.Tensor = tf.constant(signal_array, dtype=tf.float32)
    frequencies_tf_hertz, psd_tf = calculate_psd(
        signal_tensor, nperseg=1024, sample_rate_hertz=sample_rate_hertz
    )
    frequencies_tf_hertz, psd_tf = frequencies_tf_hertz.numpy(), psd_tf.numpy()

    # Step 4: Plot results using bokeh
    plot = figure(
        title="PSD: Power Spectral Density",
        x_axis_label="Frequency (hertz)",
        y_axis_label="Power Spectral Density",
        y_axis_type="log",
        width=800,
        height=400,
    )
    plot.line(
        frequencies_scipy_hertz, 
        psd_scipy, 
        legend_label="scipy", 
        line_color="blue"
    )
    plot.line(
        frequencies_tf_hertz, 
        psd_tf, 
        legend_label="tensorflow", 
        line_color="red", 
        line_dash="dashed"
    )

    # Specify output file and save the plot
    output_path = Path("./py_ml_data/tests/psd_plots.html")
    output_file(str(output_path))
    save(plot)

if __name__ == "__main__":
    test_welch_method()