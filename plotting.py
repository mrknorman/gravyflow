from typing import Dict, Union

import numpy as np
import tensorflow as tf
from scipy.signal import spectrogram

from bokeh.io import save, output_file
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, Legend, ColorBar, LogTicker, LinearColorMapper
from bokeh.palettes import Bright

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
    colors : list = Bright[7],
    scale_factor : float = None
    ):
    
    # Parameters:
    width : int = 1600
    height : int = 600
        
    # Get num samples and check dictionies:
    num_samples = check_ndarrays_same_length(strain)
    
    # If inputs are tensors, convert to numpy array:
    for key, value in strain.items():
        if isinstance(value, tf.Tensor):
            strain[key] = value.numpy()
        
    # Generate time axis for plotting:
    time_axis : np.ndarray = \
        np.linspace(0.0, duration_seconds, num_samples)
    
    # Create data dictionary to use as source:
    data : Dict = { "time" : time_axis }
    for key, value in strain.items():
        data[key] = value
    
    # Preparing the data:
    source = ColumnDataSource(data)
    
    # Prepare y_axis:
    y_axis_label = f"Strain"
    if scale_factor is not None:
        y_axis_label += f" (scaled by {scale_factor})"
    
    # Create a new plot with a title and axis labels
    p = \
        figure(
            title=title, 
            x_axis_label="Time (seconds)", 
            y_axis_label=y_axis_label,
            width=width,
            height=height
        )
    
    # Add lines to figure for every line in strain
    for index, (key, value) in enumerate(strain.items()):
        p.line(
            "time", 
            key, 
            source=source, 
            line_width=2, 
            line_color = colors[index],
            legend_label = key
        )
        
    legend = Legend(location="top_left")
    p.add_layout(legend)
    p.legend.click_policy = "hide"

    return p

def generate_psd_plot(
    psd : Dict[str, np.ndarray],
    frequencies : float = np.ndarray,
    title : str = "",
    colors : list = Bright[7]
    ):
    
    # Parameters:
    width : int = 1600
    height : int = 600
        
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
            x_axis_label="Frequency (Hertz)", 
            y_axis_label=y_axis_label,
            width=width,
            height=height
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
        
    legend = Legend(location="top_left")
    p.add_layout(legend)
    p.legend.click_policy = "hide"

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
    width : int = 1600
    height : int = 600

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

    