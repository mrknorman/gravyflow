from scipy.signal import welch
from .psd import calculate_psd
import tensorflow as tf
from bokeh.plotting import figure, save, output_file

def test_welch_method():
    # Step 1: Generate a signal
    fs = 8196  # sample rate
    T = 2.0    # seconds
    t = np.arange(0, T, 1/fs)  # time variable
    freq = 123.4  # frequency of the signal
    x = 0.5*np.sin(2*np.pi*freq*t)
    
    # Step 2: Compute the power spectral density using scipy
    f_scipy, Pxx_scipy = welch(x, fs, nperseg=1024)

    # Step 3: Compute the power spectral density using TensorFlow
    x_tf = tf.constant(x, dtype=tf.float32)
    f_tf, Pxx_tf = calculate_psd(x_tf, nperseg=1024, fs=fs)
    f_tf, Pxx_tf = f_tf.numpy(), Pxx_tf.numpy()

    # Step 4: Plot the results using bokeh
    p = figure(title="PSD: power spectral density", x_axis_label='Frequency', y_axis_label='Power Spectral Density', 
               y_axis_type="log", plot_width=800, plot_height=400)
    p.line(f_scipy, Pxx_scipy, legend_label="scipy", line_color="blue")
    p.line(f_tf, Pxx_tf, legend_label="tensorflow", line_color="red", line_dash="dashed")
    
    print(f_tf, Pxx_tf)
    
    # Specify the output file and save the plot
    output_file("./py_ml_data/psd_test.html")
    save(p)

if __name__ == "__main__":
    test_welch_method()