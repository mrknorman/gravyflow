from typing import Dict, Union, List, Optional
from datetime import datetime, timedelta

import numpy as np
from scipy.constants import golden
from bokeh.io import save, output_file
from bokeh.plotting import figure
from bokeh.models import (ColumnDataSource, ColorBar, LogTicker, LinearColorMapper, 
                          HoverTool, BoxAnnotation, Span, Range1d, Label, CustomJSTickFormatter)
from bokeh.palettes import Bright, Category10
from bokeh.models import Div
from bokeh.layouts import column, gridplot

import gravyflow as gf
from keras import ops


def gps_to_datetime(gps_time: float) -> str:
    """Convert GPS time to human-readable datetime string (UTC)."""
    gps_epoch = datetime(1980, 1, 6, 0, 0, 0)
    leap_seconds = 18  # Current leap seconds as of 2021
    dt = gps_epoch + timedelta(seconds=gps_time + leap_seconds)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


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
        my_dict : Dict[str, Union[np.ndarray, object]]
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


def generate_segment_timeline_plot(
    segments: Dict[gf.IFO, np.ndarray],
    observing_runs: Optional[List[gf.ObservingRun]] = None,
    title: str = "Noise Segment Timeline",
    height: int = 400,
    width: Optional[int] = None,
    segment_colors: Optional[Dict[gf.IFO, str]] = None,
    show_observing_runs: bool = True
):
    """
    Generate a Bokeh plot showing noise segments on a timeline with observing runs.
    
    Parameters
    ----------
    segments : Dict[gf.IFO, np.ndarray]
        Dictionary mapping IFO detectors to segment arrays.
        Each array has shape (N, 2) with [start_gps, end_gps].
    observing_runs : List[gf.ObservingRun], optional
        List of observing runs to show (O1, O2, O3). Defaults to all.
    title : str
        Plot title.
    height : int
        Plot height in pixels.
    width : int, optional
        Plot width. Defaults to golden ratio.
    segment_colors : Dict[gf.IFO, str], optional
        Custom colors for each detector's segments.
    show_observing_runs : bool
        Whether to show observing run bands.
        
    Returns
    -------
    bokeh.plotting.figure
        The Bokeh figure object.
    """
    if observing_runs is None:
        observing_runs = [gf.ObservingRun.O1, gf.ObservingRun.O2, gf.ObservingRun.O3]
    
    if width is None:
        width = int(height * golden)
    
    if segment_colors is None:
        segment_colors = {
            gf.IFO.L1: "#1f77b4",  # Blue (Livingston)
            gf.IFO.H1: "#ff7f0e",  # Orange (Hanford)
            gf.IFO.V1: "#2ca02c",  # Green (Virgo)
        }
    
    # Define row indices for each detector
    ifo_positions = {
        gf.IFO.L1: 2,
        gf.IFO.H1: 1,
        gf.IFO.V1: 0,
    }
    
    # Observing run colors (semi-transparent)
    run_colors = {
        gf.ObservingRun.O1: "#e377c2",  # Pink
        gf.ObservingRun.O2: "#bcbd22",  # Yellow-green
        gf.ObservingRun.O3: "#17becf",  # Cyan
    }
    
    # Calculate time range from observing runs and segments
    all_times = []
    for run in observing_runs:
        all_times.extend([run.value.start_gps_time, run.value.end_gps_time])
    for ifo, segs in segments.items():
        if len(segs) > 0:
            all_times.extend(segs.flatten().tolist())
    
    if not all_times:
        raise ValueError("No time data to plot")
    
    min_time = min(all_times)
    max_time = max(all_times)
    padding = (max_time - min_time) * 0.02
    
    # JavaScript code to convert GPS time to readable date string (YYYY-MM-DD)
    formatter_code = """
        var gps_epoch_ms = 315964800000;
        var leap_seconds = 18;
        var unix_ms = (tick + leap_seconds) * 1000 + gps_epoch_ms;
        var date = new Date(unix_ms);
        return date.toISOString().split('T')[0];
    """

    # Create figure
    p = figure(
        title=title,
        x_axis_label="Date (UTC)",
        y_axis_label="Detector",
        width=width,
        height=height,
        x_range=(min_time - padding, max_time + padding),
        y_range=(-0.5, 2.5),
        tools="pan,box_zoom,wheel_zoom,reset,save,hover"
    )
    p.xaxis.formatter = CustomJSTickFormatter(code=formatter_code)
    
    # Add observing run background bands
    if show_observing_runs:
        for run in observing_runs:
            run_data = run.value
            box = BoxAnnotation(
                left=run_data.start_gps_time,
                right=run_data.end_gps_time,
                fill_alpha=0.15,
                fill_color=run_colors.get(run, "#cccccc"),
                line_color=None
            )
            p.add_layout(box)
            
            # Add label for observing run
            label = Label(
                x=run_data.start_gps_time + (run_data.end_gps_time - run_data.start_gps_time) * 0.5,
                y=2.4,
                text=run_data.name,
                text_font_size="10pt",
                text_color=run_colors.get(run, "#333333"),
                text_align="center"
            )
            p.add_layout(label)
    
    # Plot segments for each detector
    for ifo, segs in segments.items():
        if len(segs) == 0:
            continue
            
        y_pos = ifo_positions.get(ifo, 0)
        color = segment_colors.get(ifo, "#333333")
        
        # Prepare data for segments
        starts = segs[:, 0]
        ends = segs[:, 1]
        durations = ends - starts
        
        # Convert GPS times to readable datetime strings
        start_datetimes = [gps_to_datetime(t) for t in starts]
        end_datetimes = [gps_to_datetime(t) for t in ends]
        
        source = ColumnDataSource({
            "left": starts,
            "right": ends,
            "bottom": [y_pos - 0.3] * len(starts),
            "top": [y_pos + 0.3] * len(starts),
            "ifo": [ifo.value.name] * len(starts),
            "start_gps": starts,
            "end_gps": ends,
            "start_datetime": start_datetimes,
            "end_datetime": end_datetimes,
            "duration": durations
        })
        
        p.quad(
            left="left",
            right="right",
            bottom="bottom",
            top="top",
            source=source,
            fill_color=color,
            fill_alpha=0.7,
            line_color=color,
            legend_label=ifo.value.name
        )
    
    # Configure y-axis ticks
    detector_names = ["Virgo (V1)", "Hanford (H1)", "Livingston (L1)"]
    p.yaxis.ticker = [0, 1, 2]
    p.yaxis.major_label_overrides = {0: detector_names[0], 1: detector_names[1], 2: detector_names[2]}
    
    # Configure hover tool
    hover = p.select(dict(type=HoverTool))
    hover.tooltips = [
        ("Detector", "@ifo"),
        ("Start", "@start_datetime"),
        ("End", "@end_datetime"),
        ("Duration", "@duration{0.0}s"),
        ("GPS Start", "@start_gps{0.0}"),
        ("GPS End", "@end_gps{0.0}")
    ]
    
    # Legend
    p.legend.location = "top_right"
    p.legend.click_policy = "hide"
    
    return p


def generate_example_extraction_plot(
    segment_times: Dict[gf.IFO, np.ndarray],
    extraction_points: np.ndarray,
    onsource_duration_seconds: float,
    offsource_duration_seconds: float = 0.0,
    padding_duration_seconds: float = 0.0,
    title: str = "Example Extraction from Segments",
    height: int = 400,
    width: Optional[int] = None,
    detector_colors: Optional[Dict[gf.IFO, str]] = None
):
    """
    Generate a Bokeh plot showing where examples are extracted from segments.
    
    Shows valid time regions in each segment with highlighted extraction windows
    for onsource and offsource data.
    
    Parameters
    ----------
    segment_times : Dict[gf.IFO, np.ndarray]
        Dictionary mapping IFO to single segment [start_gps, end_gps].
    extraction_points : np.ndarray
        GPS times of extraction centers (e.g., merger times).
        Shape: (num_examples,) or (num_examples, num_ifos).
    onsource_duration_seconds : float
        Duration of onsource window in seconds.
    offsource_duration_seconds : float
        Duration of offsource window in seconds.
    padding_duration_seconds : float
        Padding around onsource window.
    title : str
        Plot title.
    height : int
        Plot height.
    width : int, optional
        Plot width.
    detector_colors : Dict[gf.IFO, str], optional
        Custom colors for each detector.
        
    Returns
    -------
    bokeh.layouts.column
        Bokeh layout with subplots for each detector.
    """
    if width is None:
        width = int(height * golden)
    
    if detector_colors is None:
        detector_colors = {
            gf.IFO.L1: "#1f77b4",
            gf.IFO.H1: "#ff7f0e",
            gf.IFO.V1: "#2ca02c",
        }
    
    plots = []
    subplot_height = height // max(1, len(segment_times))
    
    # Handle extraction_points shape
    extraction_points = np.atleast_1d(extraction_points)
    if extraction_points.ndim == 1:
        # Same extraction points for all IFOs
        extraction_dict = {ifo: extraction_points for ifo in segment_times.keys()}
    else:
        # Different extraction points per IFO
        extraction_dict = dict(zip(segment_times.keys(), extraction_points.T))
    
    total_onsource = onsource_duration_seconds + 2 * padding_duration_seconds
    
    for ifo, seg in segment_times.items():
        seg = np.atleast_1d(seg)
        if seg.ndim == 1:
            start_gps, end_gps = seg[0], seg[1]
        else:
            start_gps, end_gps = seg[0, 0], seg[0, 1]
        
        color = detector_colors.get(ifo, "#333333")
        extraction_gps = extraction_dict.get(ifo, extraction_points)
        
        # JavaScript code to convert GPS time to readable time string (HH:MM:SS)
        formatter_code = """
            var gps_epoch_ms = 315964800000;
            var leap_seconds = 18;
            var unix_ms = (tick + leap_seconds) * 1000 + gps_epoch_ms;
            var date = new Date(unix_ms);
            return date.toISOString().split('T')[1].split('.')[0];
        """
        
        p = figure(
            title=f"{ifo.value.name}",
            x_axis_label="Time (UTC)",
            y_axis_label="",
            width=width,
            height=subplot_height,
            tools="pan,box_zoom,wheel_zoom,reset,save"
        )
        p.xaxis.formatter = CustomJSTickFormatter(code=formatter_code)
        
        # Draw full segment as background
        p.quad(
            left=start_gps,
            right=end_gps,
            bottom=-0.3,
            top=0.3,
            fill_color=color,
            fill_alpha=0.2,
            line_color=color,
            legend_label="Valid Segment"
        )
        
        # Draw extraction windows
        onsource_added = False
        offsource_added = False
        
        for i, gps in enumerate(extraction_gps):
            # Onsource window (with padding)
            onsource_start = gps - total_onsource / 2
            onsource_end = gps + total_onsource / 2
            
            if not onsource_added:
                p.quad(
                    left=onsource_start,
                    right=onsource_end,
                    bottom=-0.25,
                    top=0.25,
                    fill_color="#d62728",  # Red
                    fill_alpha=0.6,
                    line_color="#d62728",
                    legend_label="Onsource"
                )
                onsource_added = True
            else:
                p.quad(
                    left=onsource_start,
                    right=onsource_end,
                    bottom=-0.25,
                    top=0.25,
                    fill_color="#d62728",
                    fill_alpha=0.6,
                    line_color="#d62728"
                )
            
            # Extraction center line
            span = Span(
                location=gps,
                dimension='height',
                line_color="#d62728",
                line_width=2,
                line_dash="dashed"
            )
            p.add_layout(span)
            
            # Offsource window (if defined)
            if offsource_duration_seconds > 0:
                # Place offsource before onsource
                offsource_end = onsource_start - 0.1  # Small gap
                offsource_start = offsource_end - offsource_duration_seconds
                
                if not offsource_added:
                    p.quad(
                        left=offsource_start,
                        right=offsource_end,
                        bottom=-0.2,
                        top=0.2,
                        fill_color="#9467bd",  # Purple
                        fill_alpha=0.6,
                        line_color="#9467bd",
                        legend_label="Offsource"
                    )
                    offsource_added = True
                else:
                    p.quad(
                        left=offsource_start,
                        right=offsource_end,
                        bottom=-0.2,
                        top=0.2,
                        fill_color="#9467bd",
                        fill_alpha=0.6,
                        line_color="#9467bd"
                    )
        
        # Configure
        p.yaxis.visible = False
        p.legend.location = "top_right"
        p.legend.click_policy = "hide"
        
        plots.append(p)
    
    # Set common title
    return column(*plots)