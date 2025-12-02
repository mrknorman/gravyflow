from typing import Dict, Union, List

import numpy as np
import tensorflow as tf
from scipy.constants import golden
from bokeh.io import save, output_file
from bokeh.plotting import figure
from bokeh.models import (ColumnDataSource, ColorBar, LogTicker, LinearColorMapper, 
                          HoverTool)
from bokeh.palettes import Bright
from bokeh.models import Div
from bokeh.layouts import column

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
    html_content = "<div class='centered-content'><ul>" + "".join(
        [f"<li><strong>{key}:</strong> {value}</li>" for key, value in params.items()]
    ) + "</ul></div>"

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
        # Check if the value is an np.ndarray or can be converted to one:
        if not hasattr(value, '__array__') and not isinstance(value, (list, tuple, np.ndarray)):
             # Try converting to numpy to see if it works (e.g. for Keras tensors)
            try:
                value = np.array(value)
            except:
                raise ValueError(f"The value for key '{key}' is not an np.ndarray or array-like.")

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
        sample_rate_hertz : Union[float, None] = None,
        title : Union[str, List[str]] = "",
        colors : Union[List, None] = None,
        has_legend : bool = True,
        scale_factor : Union[float, None] = None,
        height : int = 400,
        width : Union[int, None] = None
    ):
    
    if sample_rate_hertz is None:
        sample_rate_hertz = gf.Defaults.sample_rate_hertz
    
    # Safely get duration
    first_val = next(iter(strain.values()))
    # Handle case where input is list/tensor before checking shape
    if hasattr(first_val, 'shape'):
        dim = first_val.shape[-1]
    else:
        dim = len(first_val) # Fallback for lists
        
    duration_seconds = dim / sample_rate_hertz
    
    if colors is None:
        colors = Bright[7] 
        
    if width is None:
        width = int(height * golden)
    
    # Detect if the data has an additional dimension
    # Convert first key to array to check shape safely
    first_key = next(iter(strain))
    first_val_arr = np.array(strain[first_key])
    
    if len(first_val_arr.shape) == 1:
        strains = [strain]
    else:
        N = first_val_arr.shape[0]
        # Adjust height for subplots
        height = height // N if N > 0 else height
        strains = [{key: strain[key][i] for key in strain} for i in range(N)]

    if not isinstance(title, list):
        title = [title] * len(strains)

    y_axis_label = f"Strain"
    if scale_factor is not None and scale_factor != 1:
        y_axis_label += f" (scaled by {scale_factor})"

    tooltips = [
        ("Name", "@name"),
        ("Time (seconds)", "@x"),
        (y_axis_label, "@y"),
    ]

    plots = []
    for curr_title, curr_strain in zip(title, strains):
        
        # --- FIX STARTS HERE ---
        # Robustly convert inputs to numpy arrays
        for key, value in curr_strain.items():
            # Handle JAX/Keras/TF tensors by converting to numpy
            if not isinstance(value, np.ndarray):
                try:
                    value = np.array(value)
                except Exception as e:
                    # Fallback for some tensor types if np.array() fails directly
                    if hasattr(value, 'numpy'):
                        value = value.numpy()
                    else:
                        raise ValueError(f"Could not convert {key} to numpy array: {e}")
                
            curr_strain[key] = value
        # --- FIX ENDS HERE ---
                
        # Get num samples and check dictionaries:
        num_samples = check_ndarrays_same_length(curr_strain)

        # Generate time axis for plotting:
        time_axis = np.linspace(0.0, duration_seconds, num_samples)

        p = figure(
            x_axis_label="Time (seconds)", 
            y_axis_label=y_axis_label,
            title=curr_title,
            width=width,
            height=height
        )

        for index, (key, value) in enumerate(curr_strain.items()):

            source = ColumnDataSource(
                {
                    "x" : time_axis,
                    "y" : value,
                    "name" : [key] * len(time_axis)
                }
            )

            p.line(
                "x", 
                "y", 
                source=source, 
                line_width=2, 
                line_color=colors[index % len(colors)],
                legend_label=key
            )

        # Configure legend based on the number of lines
        if len(curr_strain) > 1 and has_legend:
            p.legend.location = "top_left"
            p.legend.click_policy = "hide"
            p.legend.visible = True
        else:
            p.legend.visible = False

        hover = HoverTool()
        hover.tooltips = tooltips
        p.add_tools(hover)

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
        if not isinstance(value, np.ndarray):
             psd[key] = np.array(value)
    
    # Create data dictionary to use as source:
    data : Dict = { "frequency" : frequencies }
    for key, value in psd.items():
        data[key] = value
    
    # Preparing the data:
    source = ColumnDataSource(data)
    
    # Prepare y_axis:
    y_axis_label = f"PSD"
    
    # Create a new plot with a title and axis labels
    p = figure(
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
    sample_rate_hertz: float = None,
    num_fft_samples: int = 256, 
    height: int = 400,
    width: int = None,
    num_overlap_samples: int = 200
):
    """
    Plot a spectrogram using Bokeh and return the figure or figures.
    """

    if sample_rate_hertz is None:
        sample_rate_hertz = gf.Defaults.sample_rate_hertz
        
    if width is None:
        width = int(height * golden)
    
    # Check if strain has an additional dimension
    if len(strain.shape) == 1:
        strains = [strain]
    else:
        N = strain.shape[0]
        height = height // N
        strains = [strain[i] for i in range(N)]
    
    plots = []
    for curr_strain in strains:
        # Compute the spectrogram using Keras Ops or TensorFlow
        # Ensure input is a tensor
        tensor_strain = ops.convert_to_tensor(curr_strain, dtype="float32")
        
        num_step_samples = num_fft_samples - num_overlap_samples
        spectrogram = gf.spectrogram(
            tensor_strain, 
            num_frame_samples=num_fft_samples, 
            num_step_samples=num_step_samples, 
            num_fft_samples=num_fft_samples
        )
        
        # Convert the output to NumPy arrays for visualization
        # Handle JAX/TF output
        if hasattr(spectrogram, 'numpy'):
             Sxx = spectrogram.numpy().T
        else:
             Sxx = np.array(spectrogram).T
        f = np.linspace(0, sample_rate_hertz / 2, num_fft_samples // 2 + 1)
        t = np.arange(0, Sxx.shape[1]) * (num_step_samples / sample_rate_hertz)
        Sxx_dB = Sxx[1:]  # Adjusted for dB if needed

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
        p.image(image=[Sxx_dB], x=0, y=f[1], dw=t[-1], dh=f[-1], color_mapper=mapper)

        # Add color bar
        color_bar = ColorBar(color_mapper=mapper, location=(0, 0), ticker=LogTicker())
        p.add_layout(color_bar, 'right')

        plots.append(p)

    if len(plots) == 1:
        return plots[0]
    else:
        return column(*plots)

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
    if not isinstance(correlation, np.ndarray):
        correlation = np.array(correlation)
        
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