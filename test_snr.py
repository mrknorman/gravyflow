from .dataset import get_ifo_data, O3, roll_vector_zero_padding, crop_samples
from .cuphenom.py.cuphenom import generate_phenom_d
from gwpy.timeseries import TimeSeries
from .whiten import whiten
from .psd import calculate_psd
from .snr import scale_to_snr
from .setup import setup_cuda, find_available_GPUs
from scipy.signal import welch
from itertools import islice
import tensorflow as tf
import numpy as np

from bokeh.plotting import figure, output_file, save, show
from bokeh.palettes import Category10
from bokeh.layouts import gridplot
from bokeh.models import ColumnDataSource, HoverTool, Legend
    
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
        legend_label="Onsource + Injection Whitened Tf PSD", 
        line_width = 2, 
        line_color=Category10[5][1]
    )
    p.line(
        frequencies, 
        onsource_whitened_tf_psd_scipy, 
        legend_label="Onsource Whitened TF PSD Scipy", 
        line_width = 2, 
        line_color=Category10[5][2]
    )
    p.line(
        frequencies, 
        onsource_whitened_tf_psd_tf, 
        legend_label="Onsource Whitened TF PSD TF", 
        line_width = 2, 
        line_color=Category10[5][3], 
        line_dash="dashed"
    )
    p.line(
        frequencies, 
        onsource_whitened_gwpy_psd_scipy, 
        legend_label="Onsource Whitened GWPY PSD SciPy", 
        line_width = 2, 
        line_color=Category10[5][4], 
        line_dash="dashed"
    )

    # Output to static HTML file
    output_file(filename)

    # Save the figure
    save(p)

def plot_and_save(background, injection, o_injection, scaled_injection, signal_duration):
    # convert tensors to numpy arrays
    background = background.numpy()
    injection = injection.numpy()
    scaled_injection = scaled_injection.numpy()
    o_injection = o_injection.numpy()

    # Prepare output file
    output_file("timeseries_plot.html")

    # Create a new plot with a title and axis labels
    p1 = figure(title="Background", x_axis_label='x', y_axis_label='y')
    p2 = figure(title="Injection", x_axis_label='x', y_axis_label='y')
    p3 = figure(title="Scaled Injection", x_axis_label='x', y_axis_label='y')

    # Add a line renderer with legend and line thickness
    
    x = np.linspace(0, signal_duration, len(background))
    p = figure(width=600, height=400)

    # Here x-values are assumed to be the indices. 
    # If you have a separate x-array, use that instead.

    # Plot background
    p.line(x, background, legend_label="Background", line_color=Category10[3][0])

    # Plot injection
    p.line(x, injection, legend_label="Whitened Injection", line_color=Category10[3][1])

    # Plot scaled_injection
    p.line(x, scaled_injection, legend_label="Whitened Scaled Injection", line_color=Category10[3][2])
    
    p.line(x, o_injection, legend_label="Original Injection", line_color=Category10[4][3])


    p.legend.location = "top_left"

    # Save the html
    save(p)
    
def analyse_signal(
    signal,
    noise,
    sample_rate_hertz,
    crop_duration_seconds,
    apply_whitening = True,
    whiten_fft_length_seconds = 4.0,
    whiten_overlap_seconds = 2.0,
    psd_fft_length_seconds = 1/32
    ):
    
    signal_gwpy = TimeSeries(signal, sample_rate=sample_rate_hertz)
    signal_tf = signal
    
    if apply_whitening:
        signal_tf = whiten(
            signal_tf, 
            noise, #offsource, 
            sample_rate_hertz, 
            fftlength=4.0, 
            overlap=2.0
        )
        
        # Create a GWpy TimeSeries object
        signal_gwpy = signal_gwpy.whiten(fftlength=4.0, overlap=2.0).value

    signals = {"tensorflow": signal_tf, "gwpy": signal_gwpy}
    psds = {}
    
    for key, value in signals.items(): 
        signals[key] = crop_samples(
            value,
            crop_duration_seconds,
            sample_rate_hertz
        )
        
        psds[key] = {}
            
        frequencies, psd = \
            welch(
                value, 
                sample_rate_hertz, 
                nperseg=int(psd_fft_length_seconds*sample_rate_hertz)
            )
        
        psds[key]["scipy"] = psd
        
        value = tf.convert_to_tensor(value, dtype = tf.float32)
        
        frequencies, psd = \
            calculate_psd(
                value, 
                sample_rate_hertz = sample_rate_hertz, 
                nperseg=int((1/32)*sample_rate_hertz)
            )
        
        frequencies = frequencies.numpy()
        psd = psd.numpy()
        
        psds[key]["tf"] = psd
    
    return frequencies, psds, signals

def plot_results(
        results_dict, 
        sample_rate_hertz,
        whiten_fft_length_seconds,
        whiten_overlap_seconds,
        psd_fft_length_seconds
    ):
    # create a list of colors for the lines
    colors = ['red', 'blue', 'green', 'purple', 'orange']

    # create empty figures for the time series and the PSD
    p1 = figure(width=800, height=400, title='Time Series')
    p2 = figure(width=800, height=400, title='PSD')
    
    legend1_items = []
    legend2_items = []
    for i, (key, result) in enumerate(results_dict.items()):
        # retrieve the frequencies, PSDs, and signals
        frequencies, psds, signals = result

        for sig_key in signals.keys():
            # add the time series line to the time series figure
            source = ColumnDataSource(data=dict(x=list(range(len(signals[sig_key].numpy()))), y=signals[sig_key].numpy()))
            ts_line = p1.line('x', 'y', source=source, color=colors[i], line_width=2)
            legend1_items.append((f"{key}_{sig_key}", [ts_line]))
            p1.add_tools(HoverTool())

        # add the PSD line to the PSD figure
        for psd_key in psds.keys():
            source = ColumnDataSource(data=dict(x=frequencies, y=psds[psd_key]['scipy']))
            ts_line = p2.line('x', 'y', source=source, color=colors[i], line_width=2)
            legend2_items.append((f"{key}_{psd_key}", [ts_line]))
            p2.add_tools(HoverTool())

    # Add the legends to the figures
    legend1 = Legend(items=legend1_items, location=(10,0))
    legend2 = Legend(items=legend2_items, location=(10,0))

    p1.add_layout(legend1, 'right')
    p2.add_layout(legend2, 'right')
    
    # create a gridplot
    grid = gridplot([[p1], [p2]])
    
    filename = f"psd_comparison_{sample_rate_hertz}_{whiten_fft_length_seconds}_{whiten_overlap_seconds}_{psd_fft_length_seconds}.html"
    
    output_file(filename)

    # show the plot
    show(grid)

if __name__ == "__main__":
    # Call generatePhenom function
    
    gpus = find_available_GPUs(4000, 1)
    setup_cuda(gpus, max_memory_limit = 2000, verbose = True)  
    
    sample_rate_hertz = 2048.0
    onsource_duration_seconds = 8.0
    
    #0.25, 0.5
    
    background_noise_iterator = get_ifo_data(
        time_interval = O3,
        data_labels = ["noise", "glitches"],
        seed = 400,
        force_generation = True,
        ifo = "L1",
        sample_rate_hertz = sample_rate_hertz,
        onsource_duration_seconds = onsource_duration_seconds,
        num_examples_per_batch = 1,
        order = "random",
        apply_whitening = False,
        input_keys = ["onsource", "offsource"],
        output_keys = []
    )
    
    for background, _ in islice(background_noise_iterator, 1):
        
        # Generate phenom injection
        injection = \
            generate_phenom_d(
                num_waveforms = 1, 
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
        injection *= 1.0E20

        #some kind of projection here
        injection = tf.convert_to_tensor(injection[:, 0], dtype = tf.float32)
        injection = tf.expand_dims(injection, 0)
        
        injection = roll_vector_zero_padding(injection, int(1.2 * sample_rate_hertz), int(4.8 * sample_rate_hertz))
        offsource = tf.cast(background["offsource"][0], dtype = tf.float32)
        onsource  = tf.cast(background["onsource"][0], dtype = tf.float32)
        injection = injection[0]
        
        onsource_duration_seconds = 6.0
        
        print(injection.shape)

        scaled_injection = \
            scale_to_snr(
                injection, 
                onsource,
                30.0,
                sample_rate_hertz = sample_rate_hertz, 
                fft_duration_seconds = 4.0, 
                overlap_duration_seconds = 0.5,
            )        
        
        print(scaled_injection, scaled_injection.shape)
        
        onsource_plus_injection = onsource + scaled_injection
        
        signals = {
            "whitened_onsource_plus_injection" : onsource_plus_injection,
            "whitened_onsource" : onsource, 
            "whitened_scaled_injection" : scaled_injection, 
            "whitened_injection": injection, 
            "injection": injection
        }
        
        results_dict = {}
        for key, value in signals.items():
            
            apply_whitening = True,
            whiten_fft_length_seconds = 4.0,
            whiten_overlap_seconds = 2.0,
            psd_fft_length_seconds = 1/32
            
            if (key == "injection"):
                apply_whitening = False
            
            results_dict[key] = \
                analyse_signal(
                    value,
                    onsource,
                    sample_rate_hertz,
                    onsource_duration_seconds,
                    apply_whitening = apply_whitening,
                    whiten_fft_length_seconds = whiten_fft_length_seconds,
                    whiten_overlap_seconds = whiten_overlap_seconds,
                    psd_fft_length_seconds = psd_fft_length_seconds
                )
            
        print(results_dict.keys())
        
        scaled_injection = crop_samples(
            scaled_injection,
            onsource_duration_seconds,
            sample_rate_hertz
            )
                
        plot_results(
            results_dict, 
            sample_rate_hertz,
            whiten_fft_length_seconds,
            whiten_overlap_seconds,
            psd_fft_length_seconds
        )
                    
        onsource_whitened_tf = whiten(
            onsource, 
            onsource, #offsource, 
            sample_rate_hertz, 
            fftlength=4.0, 
            overlap=2.0
        )
        
        onsource_whitened_tf = crop_samples(
            onsource_whitened_tf,
            onsource_duration_seconds,
            sample_rate_hertz
            )
        
         # Create a GWpy TimeSeries object
        ts = TimeSeries(onsource, sample_rate=sample_rate_hertz)
        onsource_whitened_gwpy = ts.whiten(fftlength=4.0, overlap=2.0).value
        
        onsource_whitened_gwpy = crop_samples(
            onsource_whitened_gwpy,
            onsource_duration_seconds,
            sample_rate_hertz
            )
        
        onsource_plus_injection_whitened_tf  = whiten(
            onsource_plus_injection, 
            onsource, #offsource, 
            sample_rate_hertz, 
            fftlength=4.0, 
            overlap=2.0
        )
        
        onsource_plus_injection_whitened_tf = crop_samples(
            onsource_plus_injection_whitened_tf,
            onsource_duration_seconds,
            sample_rate_hertz
            )
                
        injection_whitened_tf = whiten(
            injection, 
            onsource, #offsource, 
            sample_rate_hertz, 
            fftlength=4.0, 
            overlap=2.0
        )
        
        injection_whitened_tf = crop_samples(
            injection_whitened_tf,
            onsource_duration_seconds,
            sample_rate_hertz
            )
        
        print(injection.shape, scaled_injection.shape)
        
        scaled_injection_whitened_tf = whiten(
            scaled_injection, 
            onsource, #offsource, 
            sample_rate_hertz, 
            fftlength=4.0, 
            overlap=2.0
        )
        
        scaled_injection_whitened_tf = crop_samples(
            scaled_injection_whitened_tf,
            onsource_duration_seconds,
            sample_rate_hertz
            )
        
        plot_and_save(
            onsource_plus_injection_whitened_tf, 
            injection_whitened_tf, 
            scaled_injection, 
            scaled_injection_whitened_tf, 
            onsource_duration_seconds
        )
        
        frequencies, onsource_plus_injection_whitened_tf_psd = \
            welch(
                onsource_plus_injection_whitened_tf , 
                sample_rate_hertz, 
                nperseg=(1/32)*sample_rate_hertz
            )
                
        frequencies, onsource_whitened_tf_psd_tf = \
            calculate_psd(
                onsource_whitened_tf, 
                sample_rate_hertz = sample_rate_hertz, 
                nperseg=int((1/32)*sample_rate_hertz)
            )
        
        frequencies, onsource_whitened_gwpy_psd_scipy = \
            welch(
                onsource_whitened_gwpy, 
                sample_rate_hertz, 
                nperseg=int((1/32)*sample_rate_hertz)
            )
                
        frequencies, onsource_whitened_tf_psd_scipy = \
            welch(
                onsource_whitened_tf, 
                sample_rate_hertz, 
                nperseg = (1/32)*sample_rate_hertz
            )
        
        plot_psd(
            frequencies, 
            onsource_plus_injection_whitened_tf_psd, 
            onsource_whitened_tf_psd_scipy, 
            onsource_whitened_tf_psd_tf.numpy(),  
            onsource_whitened_gwpy_psd_scipy,
            "psd_test.html"
        )    
