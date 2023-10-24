from typing import Dict, Union

import numpy as np
import tensorflow as tf
from scipy.signal import spectrogram
from scipy.constants import golden

from bokeh.io import save, output_file
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, Legend, ColorBar, LogTicker, LinearColorMapper
from bokeh.palettes import Bright
from bokeh.models import Div
from bokeh.layouts import grid

def create_info_panel(params: dict) -> Div:
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
    return Div(text=style + html_content, width=190, height=200)

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
    sample_rate_hertz : float,
    duration_seconds : float,
    title : str = "",
    colors : list = None, # We'll handle default value inside the function
    has_legend : bool = True,
    scale_factor : float = None,
    height : int = 300,
    width : int = None
    ):
    
    if colors is None:
        colors = Bright[7]  # Assuming Bright is a known list elsewhere in your code.

    if width is None:
        width = int(height * golden)  # Assuming golden is a known constant elsewhere in your code.
    
    # Detect if the data has an additional dimension
    first_key = next(iter(strain))
    if len(strain[first_key].shape) == 1:
        strains = [strain]
    else:
        N = strain[first_key].shape[0]
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
        if scale_factor is not None or scale_factor == 1.0:
            y_axis_label += f" (scaled by {scale_factor})"
    
        p = figure(
            x_axis_label="Time (seconds)", 
            y_axis_label=y_axis_label,
            title=title,
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
        return plots

def generate_psd_plot(
    psd : Dict[str, np.ndarray],
    frequencies : float = np.ndarray,
    title : str = "",
    colors : list = Bright[7],
    has_legend : bool = True
    ):
    
    # Parameters:
    height : int = 300
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

def generate_spectrogram(
        strain: np.ndarray, 
        sample_rate_hertz: float,
        nperseg: int = 128, 
        noverlap: int = 64
    ) -> figure:
    """
    Plot a spectrogram using Bokeh and return the figure.

    Parameters
    ----------
    strain : np.ndarray
        Strain time-series data.
    sample_rate_hertz : float
        Sample rate in Hz.
    nperseg : int, optional
        Number of samples per segment, default is 128.
    noverlap : int, optional
        Number of samples to overlap, default is 64.

    Returns
    -------
    figure
        Bokeh figure object containing the spectrogram plot.
    """
    
    # Parameters:
    height : int = 300
    width : int = int(height*golden)
    
    # Compute the spectrogram
    f, t, Sxx = spectrogram(strain, fs=sample_rate_hertz, nperseg=nperseg, noverlap=noverlap)
    
    f = f[1:]
    Sxx = Sxx[1:]
    
    # Convert to dB
    Sxx_dB = 10 * np.log10(Sxx)

    # Validate dimensions
    if Sxx_dB.shape != (len(f), len(t)):
        raise ValueError("Dimension mismatch between Sxx_dB and frequency/time vectors.")

    # Create Bokeh figure
    p = figure(
        title="Spectrogram",
        x_axis_label='Time (seconds)',
        y_axis_label='Frequency (Hz)',
        y_axis_type="log",
        width = width,
        height = height
    )

    # Adjust axes range
    p.x_range.start = t[0]
    p.x_range.end = t[-1]
    p.y_range.start = f[0]
    p.y_range.end = f[-1]
        
    # Create color mapper
    mapper = LinearColorMapper(palette="Inferno256", low=Sxx_dB.min(), high=Sxx_dB.max())
        
    # Plotting the spectrogram
    p.image(image=[Sxx_dB], x=t[0], y=f[0], dw=(t[-1] - t[0]), dh=(f[-1] - f[0]), color_mapper=mapper)
    
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
    height: int = 300,
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
    time_axis = np.linspace(0.0, duration_seconds, num_samples)
    
    # Create data dictionary to use as source:
    data = {"time": time_axis}
    for i in range(num_pairs):
        data[f"pair_{i}"] = correlation[i]

    source = ColumnDataSource(data)
    
    y_axis_label = "Correlation"
    
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

    return p