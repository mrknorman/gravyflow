import warnings
with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        r"The value of the smallest subnormal for <class \'numpy.float32\'\> "
        "type is zero."
    )
    warnings.filterwarnings(
        "ignore",
        r"The value of the smallest subnormal for <class \'numpy.float64\'\> "
        "type is zero."
    )
    
    from dataclasses import dataclass
    import numpy as np

    import matplotlib
    matplotlib.use("Agg")
    
    import bokeh.plotting as bk
    import bokeh.palettes  as palettes
    
    import matplotlib.pyplot as plt

    import ctypes
    libc = ctypes.CDLL(None)

    from scipy.interpolate import interp1d
    from typing import Callable

    import sys
    import os

def printf(string : str):
    libc.puts(str(string).encode("UTF-8"))
    
def print_error(e):
    exc_type, exc_obj, exc_tb = sys.exc_info()
    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]

    printf(f"{e}")
    printf(f"{exc_type, fname, exc_tb.tb_lineno}")
    
def add_error_except(func: Callable):
    """Decorator that adds error checking."""
    
    def wrap(*args, **kwargs):
    
        result = 1;
        try:
            result = func(*args, **kwargs)
        except Exception as error:
            print_error(error)
        
        return result
    return wrap

@dataclass
class Axis():
    name  : str
    label : str
    
@dataclass
class Axes():
    name : str
    axis : tuple[Axis, ...]
    
@dataclass
class SeriesValues():
    axis_name : str
    values    : np.ndarray
    
@dataclass
class Series():
    label     : str
    axes_name : str
    values    : tuple[SeriesValues, ...]

@dataclass
class Figure():
    axes   : dict[Axes, ...]
    series : tuple[Series, ...]
    
    def __init__(self, input_args : tuple):
                
        axes_dict = {}
        for index, axes in enumerate(input_args[0]):

            axes = Axes(*axes)
            axis_list = []
            for index, axis in enumerate(axes.axis):
                axis_list.append(Axis(*axis))

            axes.axis = tuple(axis_list)
            axes_dict[axes.name] = axes
        
        self.axes = axes_dict

        series_list = []
        for index, series in enumerate(input_args[1]):

            series = Series(*series)
            values_list = []
            for index, values in enumerate(series.values):
                values_list.append(SeriesValues(*values))

            series.values = tuple(values_list)
            series_list.append(series)
        
        self.series = tuple(series_list)

@add_error_except
def plot_figure(verbosity: int, input_args: tuple, output_file_name: str):
    
    # Init Figure Dataclass:    
    figure = Figure(input_args)
    
    # Set output file:
    bk.output_file(f"{output_file_name}.html")
    
    # Set up axes:
    graphs = {}
    for name in figure.axes.keys():
        graph = figure.axes[name]
        
        graphs[name] = bk.figure(        
            x_axis_label = graph.axis[0].label, 
            y_axis_label = graph.axis[1].label
        )
        
    pallet = palettes.Colorblind[8]
    
    for series, color in zip(figure.series, pallet):
        axes_name = series.axes_name
        graph = graphs[axes_name]
                
        values_map = {}
        for index, axis in enumerate(figure.axes[axes_name].axis):
            values_map[axis.name] = index
            
        values = [None] * len(series.values)
        
        for value in series.values:
            index = values_map[value.axis_name]
            
            values[index] = value.values; 
        
        graph.line(
            values[0],
            values[1],
            line_color = color, 
            legend_label = series.label
        )
    
    plot = bk.gridplot([graphs.values(), [None]])
    bk.save(plot)

@add_error_except
def plot_line_graph(x_array, y_array, x_label, y_label, file_name):

    bk.output_file(f"{file_name}.html")
    graph = bk.figure(        
        x_axis_label = x_label, 
        y_axis_label = y_label
    )

    graph.line(
        x_array, 
        y_array
    )
    bk.save(graph)