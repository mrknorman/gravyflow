#!/usr/bin/env python3
"""
Track NVIDIA GPU memory usage over time using nvidia-smi and plot the result
with an interactive Bokeh dashboard.
"""

import subprocess
import time
import datetime
import pandas as pd
from bokeh.plotting import figure, output_file, show
from bokeh.models import ColumnDataSource, HoverTool, Legend
from bokeh.layouts import column

# Parameters for polling
POLL_INTERVAL = 5  # seconds between polls
TOTAL_DURATION = 3600 // 2.0  # total duration in seconds (e.g. 5 minutes)

def poll_nvidia_smi():
    """
    Calls nvidia-smi to get the current GPU memory usage.
    Returns:
        A list of dictionaries with keys: 'timestamp', 'gpu', 'memory_used'
    """
    data_points = []
    # Command to query GPU index and used memory (in MB) without header and units.
    command = "nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits"
    try:
        result = subprocess.check_output(command, shell=True)
        # Decode the result and split into lines.
        lines = result.decode("utf-8").strip().split("\n")
        # Record current timestamp
        now = datetime.datetime.now()
        for line in lines:
            # Each line should be something like: "0, 2350"
            parts = line.split(",")
            if len(parts) >= 2:
                gpu_index = int(parts[0].strip())
                memory_used = float(parts[1].strip())
                data_points.append({
                    "timestamp": now,
                    "gpu": gpu_index,
                    "memory_used": memory_used
                })
    except Exception as e:
        print("Error polling nvidia-smi:", e)
    return data_points

def collect_data(poll_interval, total_duration):
    """
    Collects nvidia-smi data over the given duration.
    Returns:
        A pandas DataFrame containing the collected data.
    """
    all_data = []
    num_polls = int(total_duration / poll_interval)
    print(f"Starting data collection for {total_duration} seconds "
          f"({num_polls} polls every {poll_interval} seconds)...")
    for i in range(num_polls):
        dp = poll_nvidia_smi()
        if dp:
            all_data.extend(dp)
        time.sleep(poll_interval)
    return pd.DataFrame(all_data)

def create_dashboard(df):
    """
    Creates and displays an interactive Bokeh dashboard from the DataFrame.
    
    Args:
        df (pandas.DataFrame): DataFrame with columns 'timestamp', 'gpu', and 'memory_used'
    """
    # Convert timestamp to pandas datetime (if not already)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    # Pivot the DataFrame so that each GPU has its own column.
    pivot_df = df.pivot(index="timestamp", columns="gpu", values="memory_used")
    pivot_df = pivot_df.sort_index()
    
    # Convert the GPU column names (keys) to strings for Bokeh compatibility.
    pivot_df.columns = pivot_df.columns.astype(str)
    
    # Create a ColumnDataSource from the pivoted DataFrame.
    source = ColumnDataSource(pivot_df)
    
    # Create a figure with a datetime x-axis.
    p = figure(title="NVIDIA GPU Memory Usage Over Time",
               x_axis_type="datetime",
               width=900, height=400,
               tools="pan,wheel_zoom,box_zoom,reset,save")
    
    colors = ["blue", "green", "red", "orange", "purple", "brown", "magenta", "cyan"]
    
    # Plot one line per GPU (using the column names from the pivoted DataFrame)
    for i, gpu in enumerate(pivot_df.columns):
        color = colors[i % len(colors)]
        p.line(x="timestamp", y=gpu, source=source,
               line_width=2, color=color, legend_label=f"GPU {gpu}")
        p.circle(x="timestamp", y=gpu, source=source,
                 size=6, color=color, legend_label=f"GPU {gpu}")
    
    # Configure axis labels.
    p.xaxis.axis_label = "Time"
    p.yaxis.axis_label = "Memory Used (MB)"
    p.legend.location = "top_left"
    
    # Add a hover tool.
    hover = HoverTool(
        tooltips=[
            ("Time", "@timestamp{%F %T}"),
            ("Memory (MB)", "$y"),
            ("GPU", "$name")
        ],
        formatters={"@timestamp": "datetime"},
        mode="vline"
    )
    p.add_tools(hover)
    
    # Output to an HTML file and show the plot.
    output_file("nvidia_smi_dashboard.html", title="NVIDIA-SMI Memory Dashboard")
    show(p)

def main():
    df = collect_data(POLL_INTERVAL, TOTAL_DURATION)
    if df.empty:
        print("No data collected.")
    else:
        print("Data collection complete. Creating dashboard...")
        create_dashboard(df)

if __name__ == "__main__":
    main()
