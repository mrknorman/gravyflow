from typing import Dict, Union

import numpy as np
import tensorflow as tf
import scipy as sp
from scipy.constants import golden
from bokeh.io import save, output_file
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, Legend, ColorBar, LogTicker, LinearColorMapper
from bokeh.palettes import Bright
from bokeh.models import Div
from bokeh.layouts import grid, column

import gravyflow as gf

def create_info_panel(params: dict, height = 200) -> Div:
    style = """
        <style>
            .centered-content {
                display: flex;
                flex-direction: column;
                justify-content: center;
                height: 100%;
                font-family: Arial, sans-serif;
                background-color: #f4f4f4;
                padding: 10px;
                border-radius: 5px;
                border: 1px solid #ddd;

                width: 190px;             /* Set the fixed width */
                max-width: 190px;         /* Ensure it doesn't grow beyond this width */
                min-width: 190px;         /* Ensure it doesn't shrink below this width */
                overflow-wrap: break-word; /* Wrap overflowing text */
            }
            li {
                margin-bottom: 5px;
            }
            strong {
                color: #2c3e50;
            }
        </style>
    """
    html_content = "<div class='centered-content'><ul>" + "".join([f"<li><strong>{key}:</strong> {value}</li>" for key, value in params.items()]) + "</ul></div>"
    return Div(text=style + html_content, width=190, height=height)

def check_ndarrays_same_length(
        my_dict : Dict[str, Union[np.ndarray, tf.Tensor]]
    ):

    """
    Check if all values in the dictionary are np.ndarrays and have the same 
    length.

    Parameters:
        my_dict (dict): The dictionary to check.
    
    Returns:
        bool: True if all conditions are met, False otherwise.
        str: A message describing the result.
    """

    # Check if the dictionary is empty
    if not my_dict:
        raise ValueError(
                f"The dictionary is empty." 
            )

    # Initialize a variable to store the length of the first ndarray
    first_length = None

    for key, value in my_dict.items():
        # Check if the value is an np.ndarray:
        if not (isinstance(value, np.ndarray) or isinstance(value, tf.Tensor)):
            raise ValueError(f"The value for key '{key}' is not an np.ndarray.")

        # Check the length of the ndarray:
        current_length = len(value)

        if first_length is None:
            first_length = current_length

        elif current_length != first_length:
            raise ValueError(
                f"The ndarrays have different lengths: {first_length} and " 
                f"{current_length}."
            )

    return first_length

def generate_strain_plot(
        strain : Dict[str, np.ndarray],
        sample_rate_hertz : float = None,
        title : str = "",
        colors : list = None,
        has_legend : bool = True,
        scale_factor : float = None,
        height : int = 400,
        width : int = None
    ):

    if sample_rate_hertz is None:
        sample_rate_hertz = gf.Defaults.sample_rate_hertz
    if scale_factor is None:
        scale_factor = gf.Defaults.scale_factor

    duration_seconds = next(
            iter(strain.values()), 'default'
        ).shape[-1] / sample_rate_hertz
    
    if colors is None:
        colors = Bright[7] 
        
    if width is None:
        width = int(height * golden)
    
    # Detect if the data has an additional dimension
    first_key = next(iter(strain))
    if len(strain[first_key].shape) == 1:
        strains = [strain]
    else:
        N = strain[first_key].shape[0]
        height = height//N
        strains = [{key: strain[key][i] for key in strain} for i in range(N)]

    plots = []
    for curr_strain in strains:
        # If inputs are tensors, convert to numpy array:
        for key, value in curr_strain.items():
            if isinstance(value, tf.Tensor):
                curr_strain[key] = value.numpy()
                
        # Get num samples and check dictionaries:
        num_samples = check_ndarrays_same_length(curr_strain)

        # Generate time axis for plotting:
        time_axis = np.linspace(0.0, duration_seconds, num_samples)
    
        # Create data dictionary to use as source:
        data = { "time" : time_axis }
        for key, value in curr_strain.items():
            data[key] = value

        source = ColumnDataSource(data)
    
        y_axis_label = f"Strain"
        if scale_factor is not None and scale_factor != 1:
            y_axis_label += f" (scaled by {scale_factor})"
    
        p = figure(
            x_axis_label="Time (seconds)", 
            y_axis_label=y_axis_label,
            #title=title,
            width=width,
            height=height
        )

        for index, (key, value) in enumerate(curr_strain.items()):
            p.line(
                "time", 
                key, 
                source=source, 
                line_width=2, 
                line_color=colors[index % len(colors)],  # Cycle through colors if there are more lines than colors
                legend_label=key
            )
    
        p.legend.location = "top_left"
        p.legend.click_policy = "hide"
        p.legend.visible = has_legend
        p.xgrid.visible = False
        p.ygrid.visible = False

        plots.append(p)

    if len(plots) == 1:
        return plots[0]
    else:
        return column(*plots)

def generate_psd_plot(
    psd : Dict[str, np.ndarray],
    frequencies : float = np.ndarray,
    title : str = "",
    colors : list = Bright[7],
    has_legend : bool = True
    ):
    
    # Parameters:
    height : int = 400
    width : int = int(height*golden)
        
    # Get num samples and check dictionies:
    num_samples = check_ndarrays_same_length(psd)
    
    # If inputs are tensors, convert to numpy array:
    for key, value in psd.items():
        if isinstance(value, tf.Tensor):
            psd[key] = value.numpy()
    
    # Create data dictionary to use as source:
    data : Dict = { "frequency" : frequencies }
    for key, value in psd.items():
        data[key] = value
    
    # Preparing the data:
    source = ColumnDataSource(data)
    
    # Prepare y_axis:
    y_axis_label = f"PSD"
    
    # Create a new plot with a title and axis labels
    p = \
        figure(
            title=title, 
            x_axis_label="Frequency (hertz)", 
            y_axis_label=y_axis_label,
            width=width,
            height=height,
            x_axis_type="log", 
            y_axis_type="log"
        )
    
    # Add lines to figure for every line in psd
    for index, (key, value) in enumerate(psd.items()):
        p.line(
            "frequency", 
            key, 
            source=source, 
            line_width=2, 
            line_color = colors[index],
            legend_label = key
        )
        
    p.legend.location = "top_left"
    p.legend.click_policy = "hide"
    p.legend.visible = has_legend
    
    # Disable x and y grid
    p.xgrid.visible = False
    p.ygrid.visible = False

    return p

# Define the spectrogram visualization function using Bokeh
def generate_spectrogram(
    strain: np.ndarray, 
    sample_rate_hertz: float,
    num_fft_samples: int = 256, 
    height : int = 400,
    width : int = None,
    num_overlap_samples: int = 200
) -> figure:
    
    """
    Plot a spectrogram using Bokeh and return the figure.
    """
        
    if width is None:
        width = int(height * golden)
    
    # Compute the spectrogram using TensorFlow
    tensor_strain = tf.convert_to_tensor(strain, dtype=tf.float32)
    
    num_step_samples = num_fft_samples - num_overlap_samples
    spectrogram = gf.spectrogram(
        tensor_strain, 
        num_frame_samples=num_fft_samples, 
        num_step_samples=num_step_samples, 
        num_fft_samples=num_fft_samples
    )
    
    # Convert the TF output to NumPy arrays for visualization
    Sxx = spectrogram.numpy().T

    # Frequency (since TensorFlow computes only half the FFT output, we need to 
    # adapt the frequency axis accordingly)
    f = np.linspace(0, sample_rate_hertz / 2, num_fft_samples // 2 + 1)

    # Time
    t = np.arange(0, Sxx.shape[1]) * (num_step_samples / sample_rate_hertz)

    # Convert to dB
    #Sxx_dB = 10 * np.log10(Sxx + 1e-10)  # Adding a small number to avoid log of
    # zero
    Sxx_dB = Sxx
    
    f = f[1:]
    Sxx_dB = Sxx_dB[1:]
    
    # Create Bokeh figure
    p = figure(
        x_axis_label='Time (seconds)',
        y_axis_label='Frequency (Hz)',
        y_axis_type="log",
        height=height,
        width=width
    )
    
    # Create color mapper
    mapper = LinearColorMapper(
        palette="Plasma256", 
        low=np.min(Sxx_dB), 
        high=np.max(Sxx_dB)
    )

    # Plotting the spectrogram
    p.image(image=[Sxx_dB], x=0, y=f[0], dw=t[-1], dh=f[-1], color_mapper=mapper)

    # Add color bar
    color_bar = ColorBar(color_mapper=mapper, location=(0, 0), ticker=LogTicker())
    p.add_layout(color_bar, 'right')

    return p

def generate_correlation_plot(
    correlation: np.ndarray,
    sample_rate_hertz: float,
    title: str = "",
    colors: list = None,
    has_legend: bool = True,
    height: int = 400,
    width: int = None
    ):
        
    if colors is None:
        colors = Bright[7]

    if width is None:
        golden = 1.618  # Golden ratio
        width = int(height * golden)
    
    num_pairs, num_samples = correlation.shape

    # Convert tensor to numpy array if needed
    if isinstance(correlation, tf.Tensor):
        correlation = correlation.numpy()
        
    duration_seconds : float = num_samples*(1/sample_rate_hertz)

    # Generate time axis for plotting:
    time_axis = np.linspace(-duration_seconds/2.0, duration_seconds/2.0, num_samples)
    
    # Create data dictionary to use as source:
    data = {"time": time_axis}
    for i in range(num_pairs):
        data[f"pair_{i}"] = correlation[i]

    source = ColumnDataSource(data)
    
    y_axis_label = "Pearson Correlation"
    
    p = figure(
        x_axis_label="Arrival Time Difference (seconds)", 
        y_axis_label=y_axis_label,
        title=str(title),
        width=width,
        height=height
    )

    for i in range(num_pairs):
        p.line(
            "time", 
            f"pair_{i}", 
            source=source, 
            line_width=2, 
            line_color=colors[i % len(colors)],  # Cycle through colors
            legend_label=f"Pair {i}"
        )
    
    p.legend.location = "top_left"
    p.legend.click_policy = "hide"
    p.legend.visible = has_legend
    p.xgrid.visible = False
    p.ygrid.visible = False
    p.y_range.start = -1.0
    p.y_range.end = 1.0

    return p