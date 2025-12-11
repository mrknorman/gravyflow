from typing import List, Tuple, Dict, Optional, Union
import numpy as np
from datetime import datetime, timedelta
from itertools import cycle
import pandas as pd
import logging

# Bokeh imports
from bokeh.embed import components, file_html
from bokeh.io import output_file, save
from bokeh.layouts import column, gridplot, row
from bokeh.models import (
    ColumnDataSource, CustomJS, HoverTool,
    Legend, LegendItem, Slider, Select, Div,
    DatetimeTickFormatter, BoxAnnotation, TapTool,
    LinearColorMapper, ColorBar, Label
)
from bokeh.plotting import figure
from bokeh.resources import INLINE
from bokeh.palettes import Bright, Category10, Viridis256
from bokeh.transform import linear_cmap

# Holoviews/Datashader
import holoviews as hv
from holoviews.operation.datashader import datashade
import holoviews.operation.datashader

import gravyflow as gf
from .utils import downsample_data, pad_with_random_values, calculate_far_score_thresholds

logger = logging.getLogger(__name__)

def generate_gps_distribution_plot(
    validators: list,
    colors: List[str] = Bright[7],
    width: int = 1000,
    height: int = 350,
    valid_segments: np.ndarray = None,
    onsource_duration_seconds: float = 1.0,
    offsource_duration_seconds: float = 16.0
):
    """
    Generate an interactive timeline plot showing GPS time distribution of noise samples.
    Features:
    - Top plot: Timeline showing segment durations, extraction times, and valid detector periods
    - Bottom plot: Detail view showing overlaid onsource/offsource/valid windows on single time axis
    - Clickable segments with larger hit areas for small segments
    - Legends on both plots
    
    Args:
        validators: List of validators with noise_gps_times
        colors: Color palette
        width: Plot width in pixels
        height: Plot height in pixels
        valid_segments: Optional array of shape (N, 2) with [start, end] GPS times for detector on-time
        onsource_duration_seconds: Duration of onsource window
        offsource_duration_seconds: Duration of offsource window
    """
    
    # GPS epoch: January 6, 1980 (GPS time = 0)
    GPS_EPOCH = datetime(1980, 1, 6, 0, 0, 0)
    LEAP_SECONDS = 18
    
    def gps_to_datetime(gps_time):
        """Convert GPS time to Python datetime (UTC)."""
        return GPS_EPOCH + timedelta(seconds=float(gps_time) - LEAP_SECONDS)
    
    def gps_to_ms(gps_time):
        """Convert GPS time to milliseconds since epoch for Bokeh datetime axis."""
        dt = gps_to_datetime(gps_time)
        return dt.timestamp() * 1000
    
    # Collect GPS times from first validator with data
    gps_times = None
    for v in validators:
        if hasattr(v, 'noise_gps_times') and v.noise_gps_times is not None and len(v.noise_gps_times) > 0:
            gps_times = np.array(v.noise_gps_times)
            break
    
    if gps_times is None or len(gps_times) == 0:
        p = figure(title="GPS Time Distribution (No Data)", width=width, height=height)
        return p
    
    # Sort GPS times
    gps_times = np.sort(gps_times)
    min_gps = np.min(gps_times)
    max_gps = np.max(gps_times)
    time_span = max_gps - min_gps
    
    # Identify segments by finding gaps > 100 seconds
    time_diffs = np.diff(gps_times)
    gap_threshold = 100.0
    gap_indices = np.where(time_diffs > gap_threshold)[0]
    
    segment_starts = [0] + list(gap_indices + 1)
    segment_ends = list(gap_indices + 1) + [len(gps_times)]
    
    # Minimum segment width for display (60 seconds) to ensure clickability
    MIN_SEGMENT_DISPLAY_WIDTH = 60.0  # seconds
    
    # Build segment data with fixed Y positions (stacked timeline)
    segments = []
    for i, (start_idx, end_idx) in enumerate(zip(segment_starts, segment_ends)):
        seg_gps = gps_times[start_idx:end_idx]
        seg_start = seg_gps[0]
        seg_end = seg_gps[-1]
        actual_duration = seg_end - seg_start
        
        # For display: expand small segments to minimum width for visibility
        display_start = seg_start
        display_end = seg_end
        if actual_duration < MIN_SEGMENT_DISPLAY_WIDTH:
            center = (seg_start + seg_end) / 2
            display_start = center - MIN_SEGMENT_DISPLAY_WIDTH / 2
            display_end = center + MIN_SEGMENT_DISPLAY_WIDTH / 2
            
        segments.append({
            'index': i,
            'start_gps': float(seg_gps[0]),
            'end_gps': float(seg_gps[-1]),
            'seg_start_ms': gps_to_ms(display_start),
            'seg_end_ms': gps_to_ms(display_end),
            # Also store actual bounds for hit area
            'actual_start_ms': gps_to_ms(seg_start),
            'actual_end_ms': gps_to_ms(seg_end),
            'count': len(seg_gps),
            'gps_times': [float(t) for t in seg_gps],
            'y': 1,  # Fixed Y for timeline bar
        })
    
    # Create segment source for visible bars
    segment_source = ColumnDataSource(data=dict(
        left=[s['seg_start_ms'] for s in segments],
        right=[s['seg_end_ms'] for s in segments],
        top=[1.8] * len(segments),
        bottom=[0.2] * len(segments),
        index=[s['index'] for s in segments],
        start_gps=[s['start_gps'] for s in segments],
        end_gps=[s['end_gps'] for s in segments],
        count=[s['count'] for s in segments],
        gps_times=[s['gps_times'] for s in segments],
        date_label=[gps_to_datetime(s['start_gps']).strftime('%Y-%m-%d %H:%M') for s in segments],
        duration=[(s['end_gps'] - s['start_gps']) for s in segments]
    ))
    
    # Create hit area source with wider bounds for clicking
    # Hit areas extend by 12 hours on each side for easy clicking
    HIT_AREA_EXTENSION = 43200.0 * 1000  # 12 hours in ms
    hit_source = ColumnDataSource(data=dict(
        left=[s['actual_start_ms'] - HIT_AREA_EXTENSION for s in segments],
        right=[s['actual_end_ms'] + HIT_AREA_EXTENSION for s in segments],
        top=[2.0] * len(segments),
        bottom=[0.0] * len(segments),
        index=[s['index'] for s in segments],
    ))
    
    # Create extraction times source (less prominent - small ticks at bottom)
    all_extraction_times = []
    for s in segments:
        for t in s['gps_times']:
            all_extraction_times.append({
                'x': gps_to_ms(t),
                'gps': t,
                'segment_idx': s['index']
            })
    
    extraction_source = ColumnDataSource(data=dict(
        x=[e['x'] for e in all_extraction_times],
        top=[0.15] * len(all_extraction_times),
        bottom=[0.0] * len(all_extraction_times),
        gps=[e['gps'] for e in all_extraction_times],
        segment_idx=[e['segment_idx'] for e in all_extraction_times]
    ))
    
    # Main timeline plot
    p_main = figure(
        title=f"Noise Sampling Timeline ({len(segments)} segments, {time_span/86400:.1f} days span)",
        x_axis_type="datetime",
        x_axis_label="Date (UTC)",
        y_axis_label="",
        width=width,
        height=height,
        tools="pan,box_zoom,wheel_zoom,reset,tap",
        y_range=(-0.5, 2.5)
    )
    
    # Hide Y axis (it's just for layering)
    p_main.yaxis.visible = False
    p_main.ygrid.visible = False
    
    # Create valid segments source if provided, otherwise use inferred segments
    if valid_segments is not None and len(valid_segments) > 0:
        # Handle different shapes - valid_segments may be (N, num_ifos, 2) or (N, 2)
        if len(valid_segments.shape) == 3:
            # Take first IFO's segments for display
            segs = valid_segments[:, 0, :]
        else:
            segs = valid_segments
        valid_segment_data = dict(
            left=[gps_to_ms(s[0]) for s in segs],
            right=[gps_to_ms(s[1]) for s in segs],
            top=[2.2] * len(segs),
            bottom=[-0.2] * len(segs)
        )
    else:
        # Fall back to inferred from extraction times
        valid_segment_data = dict(
            left=[s['seg_start_ms'] for s in segments],
            right=[s['seg_end_ms'] for s in segments],
            top=[2.2] * len(segments),
            bottom=[-0.2] * len(segments)
        )
    valid_segment_source = ColumnDataSource(data=valid_segment_data)
    
    # Layer 0 (background): Valid segment periods - light gray
    valid_renderer = p_main.quad(
        left='left', right='right', top='top', bottom='bottom',
        source=valid_segment_source,
        fill_color="#e8e8e8", fill_alpha=0.5,
        line_color="#cccccc", line_width=1
    )
    
    # Layer 1 (middle): All extraction times as thin lines
    extraction_renderer = p_main.segment(
        x0='x', y0='bottom', x1='x', y1='top',
        source=extraction_source,
        line_color=Bright[7][2], line_alpha=0.4, line_width=1
    )
    
    # Layer 2: Invisible hit areas (for clicking small segments)
    hit_renderer = p_main.quad(
        left='left', right='right', top='top', bottom='bottom',
        source=hit_source,
        fill_alpha=0, line_alpha=0  # Completely invisible
    )
    
    # Layer 3 (foreground): Visible segment bars
    segment_renderer = p_main.quad(
        left='left', right='right', top='top', bottom='bottom',
        source=segment_source,
        fill_color=Bright[7][0], fill_alpha=0.8,
        line_color=Bright[7][0], line_width=2,
        selection_fill_color=Bright[7][1], selection_fill_alpha=1.0,
        selection_line_width=3,
        nonselection_fill_alpha=0.6
    )
    
    # Link hit area selection to visible segment selection
    hit_select_callback = CustomJS(args=dict(
        hit_source=hit_source,
        segment_source=segment_source
    ), code="""
        const hit_indices = hit_source.selected.indices;
        if (hit_indices.length > 0) {
            segment_source.selected.indices = hit_indices;
        }
    """)
    hit_source.selected.js_on_change('indices', hit_select_callback)
    
    # Add legend to main plot
    legend = Legend(items=[
        LegendItem(label="Detector On-Time", renderers=[valid_renderer]),
        LegendItem(label="Extraction Times", renderers=[extraction_renderer]),
        LegendItem(label="Used Segments", renderers=[segment_renderer]),
    ], location="top_left", click_policy="hide")
    p_main.add_layout(legend, 'right')
    
    # Hover for segments (apply to both hit area and visible)
    hover = HoverTool(tooltips=[
        ("Date", "@date_label"),
        ("Duration", "@duration{0.1f} seconds"),
        ("Samples", "@count"),
        ("GPS", "@start_gps{0.0f} - @end_gps{0.0f}")
    ], renderers=[segment_renderer])
    p_main.add_tools(hover)
    
    # TapTool for hit areas
    tap_tool = TapTool(renderers=[hit_renderer, segment_renderer])
    p_main.add_tools(tap_tool)
    
    # Date formatter
    p_main.xaxis.formatter = DatetimeTickFormatter(
        days="%Y-%m-%d",
        months="%Y-%m",
        hours="%m-%d %H:%M"
    )
    
    p_main.axis.axis_label_text_font_size = "12pt"
    p_main.axis.major_label_text_font_size = "10pt"
    p_main.title.text_font_size = "14pt"
    
    # Detail plot - shows overlaid onsource/offsource/valid windows on single time axis
    # Y axis has 3 fixed positions: Valid (0), Offsource (1), Onsource (2)
    ONSOURCE_DURATION = onsource_duration_seconds
    OFFSOURCE_DURATION = offsource_duration_seconds
    
    detail_source = ColumnDataSource(data=dict(
        left=[], right=[], top=[], bottom=[], color=[], alpha=[], window_type=[]
    ))
    
    p_detail = figure(
        title="Selected Segment Detail (click a segment above)",
        x_axis_type="datetime",
        x_axis_label="Time (UTC)",
        y_axis_label="",
        width=width,
        height=200,
        tools="pan,box_zoom,wheel_zoom,reset",
        y_range=(-0.5, 2.5)
    )
    
    # Custom Y axis tick labels
    p_detail.yaxis.ticker = [0, 1, 2]
    p_detail.yaxis.major_label_overrides = {0: 'Valid', 1: 'Offsource', 2: 'Onsource'}
    p_detail.ygrid.visible = False
    
    # All bars rendered from same source with per-bar colors
    detail_renderer = p_detail.quad(
        left='left', right='right', top='top', bottom='bottom',
        source=detail_source,
        fill_color='color', fill_alpha='alpha', line_color='color', line_width=1
    )
    
    p_detail.xaxis.formatter = DatetimeTickFormatter(
        days="%Y-%m-%d",
        hours="%H:%M:%S",
        minutes="%H:%M:%S",
        seconds="%H:%M:%S"
    )
    
    # Add legend entries for detail plot (using dummy renderers for legend labels)
    # Create dummy sources for legend
    dummy_valid = p_detail.quad(left=[0], right=[0], top=[0], bottom=[0], 
                                 fill_color="#e8e8e8", fill_alpha=0.5, visible=False)
    dummy_off = p_detail.quad(left=[0], right=[0], top=[0], bottom=[0], 
                               fill_color="#cccccc", fill_alpha=0.5, visible=False)
    dummy_on = p_detail.quad(left=[0], right=[0], top=[0], bottom=[0], 
                              fill_color=Bright[7][0], fill_alpha=0.8, visible=False)
    
    detail_legend = Legend(items=[
        LegendItem(label="Valid", renderers=[dummy_valid]),
        LegendItem(label="Offsource", renderers=[dummy_off]),
        LegendItem(label="Onsource", renderers=[dummy_on]),
    ], location="top_left")
    p_detail.add_layout(detail_legend, 'right')
    
    # JavaScript callback for segment selection - creates overlaid bars
    callback = CustomJS(args=dict(
        segment_source=segment_source,
        detail_source=detail_source,
        detail_plot=p_detail,
        gps_epoch_ms=(GPS_EPOCH.timestamp() - LEAP_SECONDS) * 1000,
        onsource_sec=ONSOURCE_DURATION,
        offsource_sec=OFFSOURCE_DURATION,
        valid_color="#e8e8e8",
        offsource_color="#999999",
        onsource_color=Bright[7][0]
    ), code="""
        const indices = segment_source.selected.indices;
        if (indices.length === 0) {
            detail_source.data = {
                left: [], right: [], top: [], bottom: [], color: [], alpha: [], window_type: []
            };
            detail_plot.title.text = "Selected Segment Detail (click a segment above)";
            detail_source.change.emit();
            return;
        }
        
        const idx = indices[0];
        const gps_times = segment_source.data.gps_times[idx];
        const start_gps = segment_source.data.start_gps[idx];
        const end_gps = segment_source.data.end_gps[idx];
        const start_date = segment_source.data.date_label[idx];
        const count = segment_source.data.count[idx];
        
        const left = [];
        const right = [];
        const top = [];
        const bottom = [];
        const color = [];
        const alpha = [];
        const window_type = [];
        
        // Row 0: Valid segment (full extent)
        // Calculate valid extent from offsource start to onsource end
        const valid_start_ms = gps_epoch_ms + (start_gps - offsource_sec) * 1000;
        const valid_end_ms = gps_epoch_ms + (end_gps + onsource_sec) * 1000;
        left.push(valid_start_ms);
        right.push(valid_end_ms);
        top.push(0.4);
        bottom.push(-0.4);
        color.push(valid_color);
        alpha.push(0.5);
        window_type.push('Valid');
        
        // Row 1: Offsource windows (one bar per extraction time, may overlap)
        // Merge overlapping offsource windows for cleaner display
        const offsource_intervals = [];
        for (let i = 0; i < gps_times.length; i++) {
            const t = gps_times[i];
            const off_start = gps_epoch_ms + (t - offsource_sec) * 1000;
            const off_end = gps_epoch_ms + t * 1000;
            offsource_intervals.push([off_start, off_end]);
        }
        
        // Sort and merge overlapping intervals
        offsource_intervals.sort((a, b) => a[0] - b[0]);
        const merged_offsource = [];
        for (const interval of offsource_intervals) {
            if (merged_offsource.length === 0 || merged_offsource[merged_offsource.length - 1][1] < interval[0]) {
                merged_offsource.push([interval[0], interval[1]]);
            } else {
                merged_offsource[merged_offsource.length - 1][1] = Math.max(
                    merged_offsource[merged_offsource.length - 1][1], interval[1]
                );
            }
        }
        
        for (const interval of merged_offsource) {
            left.push(interval[0]);
            right.push(interval[1]);
            top.push(1.4);
            bottom.push(0.6);
            color.push(offsource_color);
            alpha.push(0.6);
            window_type.push('Offsource');
        }
        
        // Row 2: Onsource windows (one bar per extraction time, may overlap)
        const onsource_intervals = [];
        for (let i = 0; i < gps_times.length; i++) {
            const t = gps_times[i];
            const on_start = gps_epoch_ms + t * 1000;
            const on_end = gps_epoch_ms + (t + onsource_sec) * 1000;
            onsource_intervals.push([on_start, on_end]);
        }
        
        // Sort and merge overlapping intervals
        onsource_intervals.sort((a, b) => a[0] - b[0]);
        const merged_onsource = [];
        for (const interval of onsource_intervals) {
            if (merged_onsource.length === 0 || merged_onsource[merged_onsource.length - 1][1] < interval[0]) {
                merged_onsource.push([interval[0], interval[1]]);
            } else {
                merged_onsource[merged_onsource.length - 1][1] = Math.max(
                    merged_onsource[merged_onsource.length - 1][1], interval[1]
                );
            }
        }
        
        for (const interval of merged_onsource) {
            left.push(interval[0]);
            right.push(interval[1]);
            top.push(2.4);
            bottom.push(1.6);
            color.push(onsource_color);
            alpha.push(0.8);
            window_type.push('Onsource');
        }
        
        detail_source.data = {
            left: left, right: right, top: top, bottom: bottom,
            color: color, alpha: alpha, window_type: window_type
        };
        
        // Auto-adjust x-axis range to fit the selected segment data
        const x_min = valid_start_ms - (valid_end_ms - valid_start_ms) * 0.05;  // 5% padding
        const x_max = valid_end_ms + (valid_end_ms - valid_start_ms) * 0.05;
        detail_plot.x_range.start = x_min;
        detail_plot.x_range.end = x_max;
        
        detail_plot.title.text = "Segment " + idx + ": " + start_date + " (" + count + " samples, " + merged_onsource.length + " onsource, " + merged_offsource.length + " offsource bars)";
        detail_source.change.emit();
    """)
    
    segment_source.selected.js_on_change('indices', callback)
    
    # Info label
    unique_times = len(np.unique(np.round(gps_times, 1)))
    label = Label(
        x=10, y=10, x_units='screen', y_units='screen',
        text=f"Total: {len(gps_times):,} samples | Unique (0.1s): {unique_times:,} | Segments: {len(segments)} | Click segment for detail",
        text_font_size="10pt",
        text_color="#666"
    )
    p_main.add_layout(label)
    
    return column(p_main, p_detail)


def generate_far_curves(
        validators : list,
        colors : List[str] = Bright[7],
        width : int = 800,
        height : int = 600
    ):

    colors = cycle(colors)
    
    tooltips = [
        ("Score Threshold", "@x{0.0000}"),
        ("False Alarm Rate (Hz)", "@y{0.0e}"),
    ]

    p = figure(
        #title = "False Alarm Rate (FAR) curves",
        width=width,
        height=height,
        x_axis_label="Score Threshold",
        y_axis_label="False Alarm Rate (Hz)",
        tooltips=tooltips,
        x_axis_type="log",
        y_axis_type="log"
    )
        
    max_num_points = 2000

    for index, (color, validator) in enumerate(zip(colors, validators)):
        far_scores = validator.far_scores
        
        name = validator.name

        if name is not None:
            title = gf.snake_to_capitalized_spaces(name)
        else:
            title = f"default_{index}"
            name = index
                
        far_scores = np.sort(far_scores)[::-1]
        total_num_seconds = len(far_scores) * validator.input_duration_seconds
        far_axis = np.arange(1, len(far_scores) + 1, dtype=float) / total_num_seconds
        
        downsampled_far_scores, downsampled_far_axis = downsample_data(
            far_scores, far_axis, max_num_points
        )
        
        # Round to 6 decimal places to reduce serialized size
        downsampled_far_scores = np.around(downsampled_far_scores, decimals=6)
        downsampled_far_axis = np.around(downsampled_far_axis, decimals=8)
        
        source = ColumnDataSource(
            data=dict(
                x=downsampled_far_scores, 
                y=downsampled_far_axis
            )
        )
        
        p.line(
            "x", 
            "y", 
            source=source, 
            line_color=color,
            line_width=2,
            legend_label=title
        )

    hover = HoverTool()
    hover.tooltips = tooltips
    p.add_tools(hover)
    
    p.legend.location = "bottom_left"
    p.legend.click_policy = "hide"
    p.legend.click_policy = "hide"
    p.legend.label_text_font_size = "18pt"

    # Increase font sizes
    p.axis.axis_label_text_font_size = "21pt"  # Increase axis label font size
    p.axis.major_label_text_font_size = "18pt"  # Increase tick label font size

    # If you have titles
    p.title.text_font_size = '24pt'

    return p

def generate_roc_curves(
    validators: list,
    colors : List[str] = Bright[7], 
    width : int = 800,
    height : int = 600
    ):

    colors = cycle(colors)
    
    p = figure(
        #title="Receiver Operating Characteristic (ROC) Curves",
        x_axis_label='False Alarm Rate (Hz)',
        y_axis_label='Accuracy (Per Cent)',
        width=width, 
        height=height,
        x_axis_type='log', 
        x_range=[1e-6, 1], 
        y_range=[0.0, 100.0]
    )
    
    max_num_points = 500

    initial_population_key = list(validators[0].roc_data.keys())[0]
    all_sources = {}
    
    for color, validator in zip(colors, validators):
        roc_data = validator.roc_data[initial_population_key]
        name = validator.name
                
        if name is not None:
            title = gf.snake_to_capitalized_spaces(name)
        else:
            title = f"default_{index}"
            name = index
        
        fpr = roc_data["fpr"]
        tpr = roc_data["tpr"]*100
        roc_auc = roc_data["roc_auc"]

        reduced_fpr, reduced_tpr = downsample_data(fpr, tpr, max_num_points)
        source = ColumnDataSource(
            data=dict(
                x=reduced_fpr, 
                y=reduced_tpr, 
                roc_auc=[roc_auc] * len(reduced_fpr))
            )
        all_sources[name] = source
        line = p.line(
            x='x', 
            y='y', 
            source=source,
            color=color, 
            width=2, 
            legend_label=f'{title} (area = {roc_auc:.5f})'
        )
        
        hover = HoverTool(
            tooltips=[
                ("Series", title),
                ("False Positive Rate", "{0.0000}"),
                ("True Positive Rate", "{0.0000}")
            ],
            renderers=[line]
        )
        p.add_tools(hover)

    p.legend.location = "bottom_right"
    p.legend.click_policy = "hide"
    p.legend.click_policy = "hide"
    p.legend.label_text_font_size = "18pt"

    # Increase font sizes
    p.axis.axis_label_text_font_size = "21pt"  # Increase axis label font size
    p.axis.major_label_text_font_size = "18pt"  # Increase tick label font size

    # If you have titles
    p.title.text_font_size = '24pt'
    
    # Dropdown to select the test population
    populations = list(validators[0].roc_data.keys())
    select = Select(
        title="Test Population:", 
        value=initial_population_key, 
        options=populations
    )
    # JS code to update the curves when the test population changes
    update_code = """
        const selected_population = cb_obj.value;
        
        for (let name in all_sources) {
            const source = all_sources[name];
            const new_data = all_data[name][selected_population];
            source.data.x = new_data.fpr;
            source.data.y = new_data.tpr;
            source.data.roc_auc = new Array(new_data.fpr.length).fill(new_data.roc_auc);
            source.change.emit();
        }
    """

    # Organize all data in a structured way for JS to easily pick it
    all_data = {}
    for validator in validators:
        name = validator.name
        all_data[name] = {}
        for population, data in validator.roc_data.items():
            fpr, tpr = downsample_data(data["fpr"], data["tpr"]*100, max_num_points)
            all_data[name][population] = {
                'fpr': fpr,
                'tpr': tpr,
                'roc_auc': data['roc_auc']
            }

    callback = CustomJS(
        args=
        {
            'all_sources': all_sources, 
            'all_data': all_data
        }, 
        code=update_code
    )
    select.js_on_change('value', callback)
    
    return p, select

def generate_waveform_plot(
    data : dict,
    onsource_duration_seconds : float,
    colors : list = Bright[7]
    ):

    from datetime import datetime, timedelta
    from bokeh.layouts import column, row
    import pandas as pd
    
    # Don't use cycle - we index directly into colors list
    if not isinstance(colors, (list, tuple)):
        colors = list(colors)
    
    # Extract and flatten data for plotting
    onsource_data = np.array(data[gf.ReturnVariables.WHITENED_ONSOURCE.name])
    
    # WHITENED_INJECTIONS is optional (not present for noise-only false positives)
    has_injection = gf.ReturnVariables.WHITENED_INJECTIONS.name in data
    if has_injection:
        injection_data = np.array(data[gf.ReturnVariables.WHITENED_INJECTIONS.name])
    else:
        injection_data = None
    
    # Flatten multi-detector data - take first detector if multi-dimensional
    if onsource_data.ndim > 1:
        onsource_data = onsource_data.flatten() if onsource_data.shape[0] == 1 else onsource_data[0]
    if has_injection and injection_data is not None and injection_data.ndim > 1:
        injection_data = injection_data.flatten() if injection_data.shape[0] == 1 else injection_data[0]
    
    # Cast onsource to float32 for Datashader (avoids float16 error)
    onsource_data = onsource_data.astype(np.float32)
    # Cast injection to float64 for Bokeh/JS compatibility (avoids serialization/browser errors)
    if has_injection and injection_data is not None:
        injection_data = injection_data.astype(np.float64)
    
    # Helper to extract scalar value from data
    def get_scalar(key, default=None):
        val = data.get(key, default)
        if val is None:
            return default
        try:
            arr = np.asarray(val)
            if arr.ndim == 0:
                return float(arr)
            elif arr.size > 0:
                return float(arr.flatten()[0])
            else:
                return default
        except:
             return default
    
    # Extract parameters
    mass1 = get_scalar(gf.WaveformParameters.MASS_1_MSUN.name, 0)
    mass2 = get_scalar(gf.WaveformParameters.MASS_2_MSUN.name, 0)
    score = get_scalar('score', 0)
    snr = get_scalar(gf.ScalingTypes.SNR.name)
    GPS_TIME_KEY = gf.ReturnVariables.GPS_TIME.name
    gps_time = get_scalar(GPS_TIME_KEY)
    
    # Convert GPS to human readable (approximate - ignores leap seconds)
    human_time = "N/A"
    if gps_time is not None and gps_time > 0:
        # GPS epoch: Jan 6, 1980. GPS is ~18s ahead of UTC due to leap seconds
        gps_epoch = datetime(1980, 1, 6, 0, 0, 0)
        try:
            dt = gps_epoch + timedelta(seconds=float(gps_time) - 18)  # Approximate UTC
            human_time = dt.strftime("%Y-%m-%d %H:%M:%S UTC")
        except:
            human_time = "Error"
    
    # Create time axis
    num_samples = len(onsource_data)
    time = np.linspace(0, onsource_duration_seconds, num_samples)
    
    p = figure(
        title="Worst Performing Input",
        x_axis_label='Time (seconds)',
        y_axis_label='Whitened Strain',
        width=750, 
        height=300
    )

    # Downsample ONSOURCE data to ~1024 Hz for aggressive size reduction
    TARGET_SAMPLE_RATE = 1024
    if len(onsource_data) > TARGET_SAMPLE_RATE:
        step = int(np.ceil(len(onsource_data) / TARGET_SAMPLE_RATE))
        onsource_plot = onsource_data[::step]
        time_plot_onsource = time[::step]
    else:
        onsource_plot = onsource_data
        time_plot_onsource = time

    source = ColumnDataSource(
        data=dict(
            x=time_plot_onsource, 
            y=onsource_plot
        )
    )
    p.line(
        x='x', 
        y='y', 
        source=source,
        color=colors[0], 
        width=1,  # Thinner line for dense noise
        legend_label='Whitened Strain' + (' + Injection' if has_injection else ' (Noise Only)')
    )

    # Create slider for injection scale (only if injection exists)
    scale_slider = None
    
    if has_injection and injection_data is not None:
        # Now add the interactive lines (Injection) using standard Bokeh
        # These remain as vectors so they are sharp and interact fast with JS
        
        # Downsample injection data for client-side plotting (reduce HTML size)
        MAX_DISPLAY_POINTS = 1500
        if len(injection_data) > MAX_DISPLAY_POINTS:
            # Simple decimation is sufficient for visual line inspection
            step = int(np.ceil(len(injection_data) / MAX_DISPLAY_POINTS))
            injection_plot = injection_data[::step]
            time_plot = time[::step]
        else:
            injection_plot = injection_data
            time_plot = time

        # Injection Line (Static)
        source_inj = ColumnDataSource(data=dict(x=time_plot, y=injection_plot))
        p.line(x='x', y='y', source=source_inj, color=colors[1], width=2, legend_label='Whitened Injection')
        
        # Scaled Injection Line (Interactive)
        scaled_source = ColumnDataSource(
            data=dict(
                x=time_plot, 
                y=injection_plot, 
                y_original=injection_plot.copy()
            )
        )
        scaled_line = p.line(
            x='x', 
            y='y', 
            source=scaled_source,
            color=colors[2], 
            width=2, 
            legend_label='Scaled Injection'
        )
        
        # Add hover tool for scaled injection
        hover = HoverTool(renderers=[scaled_line], tooltips=[("Time", "@x{0.000}"), ("Strain", "@y")])
        p.add_tools(hover)
        
        # Create slider for injection scale
        scale_slider = Slider(start=1, end=2, value=1, step=0.1, title="Injection Scale")
        
        scale_callback = CustomJS(
            args=dict(source=scaled_source, slider=scale_slider),
            code="""
                const data = source.data;
                const scale = slider.value;
                const y_orig = data['y_original'];
                const y = data['y'];
                for (let i = 0; i < y_orig.length; i++) {
                    y[i] = y_orig[i] * scale;
                }
                source.change.emit();
            """
        )
        scale_slider.js_on_change('value', scale_callback)
    
    p.legend.location = "bottom_right"
    p.legend.click_policy = "hide"
    p.legend.label_text_font_size = "10pt"

    # Increase font sizes
    p.axis.axis_label_text_font_size = "12pt"
    p.axis.major_label_text_font_size = "10pt"
    p.title.text_font_size = '14pt'
    
    # Info Panel
    info_items = []
    info_items.append(f"<b>Score:</b> {score:.3f}")
    if snr: info_items.append(f"<b>SNR:</b> {snr:.1f}")
    if gps_time is not None:
        info_items.append(f"<b>GPS:</b> {gps_time:.3f}")
        info_items.append(f"<b>Time:</b> {human_time}")
    if mass1: info_items.append(f"<b>M₁:</b> {mass1:.1f} M☉")
    if mass2: info_items.append(f"<b>M₂:</b> {mass2:.1f} M☉")
    
    info_html = "<br>".join(info_items)
    info_panel = Div(
        text=f"""<div style="padding:12px;font-size:11px;">
            <b>Parameters</b><br>{info_html}</div>""",
        width=200
    )
    
    # Build layout based on whether injection slider exists
    if scale_slider:
        return row(column(scale_slider, p), info_panel)
    else:
        return row(p, info_panel)

def generate_parameter_space_plot(
    validators: list,
    min_snr: float = 5.0,
    max_points: int = 10000,
    width: int = 800,
    height: int = 600
):
    """
    Generate interactive parameter space explorer with selectable axes.
    Uses hybrid approach: datashader for background density + downsampled scatter for hover.
    
    Args:
        validators: List of Validator instances
        min_snr: Minimum SNR to include (default 5.0 for ROC consistency)
        max_points: Maximum points to display as scatter overlay
        width: Plot width in pixels
        height: Plot height in pixels
    
    Returns:
        Tuple of (plot, controls_row) for layout
    """
    
    # Collect data from first validator with data
    data = None
    for v in validators:
        if hasattr(v, 'efficiency_data') and v.efficiency_data:
            data = v.efficiency_data
            break
    
    if data is None or len(data.get('snrs', [])) == 0:
        # Return empty placeholder
        p = figure(title="Parameter Space (No Data)", width=width, height=height)
        return p, None
    
    # Extract and filter by SNR
    snrs = np.array(data['snrs'])
    scores = np.array(data['scores'])
    mass1 = np.array(data.get('mass1', []))
    mass2 = np.array(data.get('mass2', []))
    mass2 = np.array(data.get('mass2', []))
    gps_times = np.array(data.get('gps_times', []))
    central_times = np.array(data.get('central_times', []))
    hpeak = np.array(data.get('hpeak', []))
    hrss = np.array(data.get('hrss', []))
    
    # Filter by minimum SNR
    mask = snrs >= min_snr
    snrs = snrs[mask]
    scores = scores[mask]
    
    if len(mass1) > 0:
        mass1 = mass1[mask]
    if len(mass2) > 0:
        mass2 = mass2[mask]
    if len(gps_times) > 0:
        gps_times = gps_times[mask]
    if len(central_times) > 0:
        central_times = central_times[mask]
    if len(hpeak) > 0:
        hpeak = hpeak[mask]
    if len(hrss) > 0:
        hrss = hrss[mask]
    
    # Compute derived quantities
    if len(mass1) > 0 and len(mass2) > 0:
        # Ensure m1 >= m2 for ratio calculation
        m1_safe = np.maximum(mass1, mass2)
        m2_safe = np.minimum(mass1, mass2)
        mass_ratio = m2_safe / np.maximum(m1_safe, 1e-6)  # q = m2/m1 <= 1
        chirp_mass = (m1_safe * m2_safe) ** (3/5) / (m1_safe + m2_safe) ** (1/5)
        total_mass = m1_safe + m2_safe
    else:
        mass_ratio = np.zeros_like(snrs)
        chirp_mass = np.zeros_like(snrs)
        total_mass = np.zeros_like(snrs)
    
    # Build parameter dict for all available axes
    all_params = {
        'SNR': snrs,
        'Score': scores,
        'M₁ (M☉)': mass1 if len(mass1) > 0 else np.zeros_like(snrs),
        'M₂ (M☉)': mass2 if len(mass2) > 0 else np.zeros_like(snrs),
        'Mass Ratio (q)': mass_ratio,
        'Chirp Mass (M☉)': chirp_mass,
        'Mass Ratio (q)': mass_ratio,
        'Chirp Mass (M☉)': chirp_mass,
        'Total Mass (M☉)': total_mass,
        'GPS Time': gps_times if len(gps_times) > 0 else np.zeros_like(snrs),
        'Central Time (Normalized)': central_times if len(central_times) > 0 else np.zeros_like(snrs),
        'Hpeak': hpeak if len(hpeak) > 0 else np.zeros_like(snrs),
        'Hrss': hrss if len(hrss) > 0 else np.zeros_like(snrs)
    }
    
    # Downsample for scatter overlay if needed
    n_total = len(snrs)
    if n_total > max_points:
        idx = np.random.choice(n_total, max_points, replace=False)
    else:
        idx = np.arange(n_total)
    
    # --- Pre-calculate rolling averages for all parameters ---
    rolling_data_map = {}
    window_fraction = 0.05
    window_size = max(20, int(n_total * window_fraction))
    
    for key, values in all_params.items():
        if key == 'Score': continue
        
        # Sort by parameter value
        sort_p = np.argsort(values)
        sorted_x = values[sort_p]
        sorted_score = scores[sort_p]
        
        # Calculate rolling mean
        series = pd.Series(sorted_score)
        rolling_mean = series.rolling(window=window_size, min_periods=window_size//2, center=True).mean()
        
        # Fill NaN at edges
        rolling_mean = rolling_mean.bfill().ffill()
        
        # Downsample for JS performance if needed
        # Use relatively dense sampling for smooth curves but not full 1M points
        MAX_ROLLING_POINTS = 2000
        if len(sorted_x) > MAX_ROLLING_POINTS:
            step = int(np.ceil(len(sorted_x) / MAX_ROLLING_POINTS))
            x_plot = sorted_x[::step]
            y_plot = rolling_mean.values[::step]
        else:
            x_plot = sorted_x
            y_plot = rolling_mean.values
            
        rolling_data_map[key] = {
            'x': x_plot.tolist(),
            'y': y_plot.tolist()
        }
    # ---------------------------------------------------------
    
    # Create downsampled data source with all parameters
    scatter_data = {k: v[idx].astype(np.float64) for k, v in all_params.items()}
    scatter_data['color'] = scatter_data['Score']  # Color by score initially
    source = ColumnDataSource(data=scatter_data)
    
    # Store full data for JS callback
    full_data = {k: v.tolist() for k, v in all_params.items()}
    
    # Initial axis selection
    initial_x = 'M₁ (M☉)'
    initial_y = 'M₂ (M☉)'
    
    # Create plot
    p = figure(
        title="Parameter Space Explorer",
        x_axis_label=initial_x,
        y_axis_label=initial_y,
        width=width,
        height=height,
        tools="pan,box_zoom,wheel_zoom,reset"
    )
    
    # --- Rolling Average Plot ---
    p_rolling = figure(
        title=f"Rolling Average Score vs {initial_x}",
        x_axis_label=initial_x,
        y_axis_label="Average Score",
        width=width,
        height=200,
        tools="pan,box_zoom,wheel_zoom,reset",
        y_range=(-0.1, 1.1)
    )
    
    # Initial rolling data
    init_roll = rolling_data_map.get(initial_x, {'x': [], 'y': []})
    source_rolling = ColumnDataSource(data=dict(x=init_roll['x'], y=init_roll['y']))
    
    line_rolling = p_rolling.line(
        x='x', y='y',
        source=source_rolling,
        line_width=3,
        color=Viridis256[128], # Mid-range color
        legend_label="Avg Score"
    )
    
    # Hover for rolling plot
    hover_rolling = HoverTool(
        renderers=[line_rolling],
        tooltips=[(f"Param", "@x{0.00}"), ("Avg Score", "@y{0.00}")]
    )
    p_rolling.add_tools(hover_rolling)
    p_rolling.legend.visible = False
    # ----------------------------
    
    # Color mapper for score
    mapper = linear_cmap(
        field_name='color',
        palette=Viridis256,
        low=0.0,
        high=1.0
    )
    
    # Scatter plot
    scatter = p.scatter(
        x=initial_x, y=initial_y,
        source=source,
        size=5,
        color=mapper,
        alpha=0.6,
        marker='circle'
    )
    
    # Add color bar
    color_bar = ColorBar(
        color_mapper=mapper['transform'],
        title='Score',
        width=8,
        location=(0, 0)
    )
    p.add_layout(color_bar, 'right')
    
    # Add hover tool
    hover = HoverTool(
        renderers=[scatter],
        tooltips=[
            ("SNR", "@{SNR}{0.1f}"),
            ("Score", "@{Score}{0.3f}"),
            ("M₁", "@{M₁ (M☉)}{0.1f} M☉"),
            ("M₂", "@{M₂ (M☉)}{0.1f} M☉"),
            ("q", "@{Mass Ratio (q)}{0.2f}"),
            ("M₂", "@{M₂ (M☉)}{0.1f} M☉"),
            ("q", "@{Mass Ratio (q)}{0.2f}"),
            ("Mchirp", "@{Chirp Mass (M☉)}{0.1f} M☉"),
            ("Mtotal", "@{Total Mass (M☉)}{0.1f} M☉"),
            ("GPS", "@{GPS Time}{0.00}"),
            ("t₀", "@{Central Time (Normalized)}{0.000}"),
            ("Hpeak", "@{Hpeak}{0.00e}"),
            ("Hrss", "@{Hrss}{0.00e}")
        ]
    )
    p.add_tools(hover)
    
    # Create axis selectors
    axis_options = list(all_params.keys())
    
    x_select = Select(
        title="X Axis:",
        value=initial_x,
        options=axis_options,
        width=150
    )
    
    y_select = Select(
        title="Y Axis:",
        value=initial_y,
        options=axis_options,
        width=150
    )
    
    # JavaScript callback to update axes
    callback = CustomJS(
        args=dict(
            source=source,
            x_select=x_select,
            y_select=y_select,
            scatter=scatter,
            xaxis=p.xaxis[0],
            yaxis=p.yaxis[0],
            source_rolling=source_rolling,
            rolling_data_map=rolling_data_map,
            p_rolling=p_rolling,
            xaxis_rolling=p_rolling.xaxis[0]
        ),
        code="""
            const x_key = x_select.value;
            const y_key = y_select.value;
            
            // Update scatter glyph x/y field references
            scatter.glyph.x = {field: x_key};
            scatter.glyph.y = {field: y_key};
            
            // Update scatter axis labels
            xaxis.axis_label = x_key;
            yaxis.axis_label = y_key;
            
            source.change.emit();
            
            // --- Update Rolling Plot ---
            const new_roll = rolling_data_map[x_key];
            if (new_roll) {
                source_rolling.data.x = new_roll.x;
                source_rolling.data.y = new_roll.y;
                source_rolling.change.emit();
                
                // Update axis label and title
                xaxis_rolling.axis_label = x_key;
                p_rolling.title.text = "Rolling Average Score vs " + x_key;
            }
            // ---------------------------
        """
    )
    
    x_select.js_on_change('value', callback)
    y_select.js_on_change('value', callback)
    
    # Increase font sizes
    p.axis.axis_label_text_font_size = "14pt"
    p.axis.major_label_text_font_size = "11pt"
    p.title.text_font_size = "16pt"
    
    # Info text
    info_div = Div(
        text=f"""<div style="font-size:11px;color:#666;">
            Showing {len(idx):,} of {n_total:,} points (SNR ≥ {min_snr}).<br>
            Color indicates model score (0=miss, 1=detect).
        </div>""",
        width=300
    )
    
    controls = row(x_select, y_select, info_div)
    
    # Validator expects (plot, controls) tuple
    # We pack everything into 'plot' and return None for 'controls'
    # Note: p already contains color_bar via add_layout
    full_layout = column(p, controls, p_rolling)
    return full_layout, None

def generate_efficiency_plot(
    validators: list,
    fars: List[float] = [1e-1, 1e-2, 1e-3, 1e-4],
    colors: List[str] = Bright[7],
    width: int = 1000,
    height: int = 800
):
    """
    Generate interactive efficiency plot with slider for FAR.
    Calculates efficiency curves (Recall vs SNR) for each FAR.
    """
    colors = cycle(colors)
    
    p = figure(
        #title="Efficiency vs SNR",
        x_axis_label="SNR",
        y_axis_label="Efficiency (Recall) / Score",
        y_range=(0, 1.05),
        width=width,
        height=height,
        tools="pan,box_zoom,wheel_zoom,reset"  # No default hover - we add scoped one later
    )
    
    # Increase font sizes by 50%
    p.axis.axis_label_text_font_size = "21pt"  # 14pt * 1.5
    p.axis.major_label_text_font_size = "18pt"  # 12pt * 1.5
    p.title.text_font_size = "24pt"  # 16pt * 1.5
    # Note: legend font size set after legend is created
    
    # Add Datashaded Scatter of Scores (Background)
    all_snrs = []
    all_scores = []
    # Collect data from all validators
    for validator in validators:
        if validator.efficiency_data:
             all_snrs.append(validator.efficiency_data["snrs"])
             all_scores.append(validator.efficiency_data["scores"])
    
    if all_snrs:
        combined_snrs = np.concatenate(all_snrs).astype(np.float32)
        combined_scores = np.concatenate(all_scores).astype(np.float32)
        
        # Use HoloViews for rasterization (compute 2D histogram on server)
        import holoviews as hv
        import datashader as ds
        import holoviews.operation.datashader
        
        # Points
        points = hv.Points((combined_snrs, combined_scores))
        
        # Rasterize to get a 2D grid of counts
        raster = hv.operation.datashader.rasterize(points, width=400, height=400, dynamic=False)
        
        # Extract data directy from HoloViews element
        bounds = raster.bounds.lbrt()
        x0 = bounds[0]  # left
        y0 = bounds[1]  # bottom
        dw = bounds[2] - bounds[0]  # right - left
        dh = bounds[3] - bounds[1]  # top - bottom
        
        try:
            counts = raster.dimension_values(raster.vdims[0], flat=False)
        except:
            counts = np.zeros((400, 400))
            
        counts = np.nan_to_num(np.array(counts))
        rows, cols = counts.shape
        
        # Initialize RGBA grid (all transparent)
        rgba = np.zeros((rows, cols), dtype=np.uint32) 
        
        heatmap_source = ColumnDataSource(data=dict(
            image=[rgba], 
            counts=[counts], # Pass raw counts grid
            x=[x0], y=[y0], dw=[dw], dh=[dh]
        ))
        
        # Create ImageRGBA glyph
        p.image_rgba(image='image', x='x', y='y', dw='dw', dh='dh', source=heatmap_source)
        
        p.xaxis.axis_label = "SNR"
        p.yaxis.axis_label = "Efficiency (Recall) / Score"
    else:
        heatmap_source = None
    
    # Store curve renderers for hover tool
    curve_renderers = []
    
    all_data = {}
    sources = {}
    all_thresholds = {}
    thresh_sources = {}
    thresh_labels = {}
    
    initial_far = fars[0]
    
    for i, (validator, color) in enumerate(zip(validators, colors)):
        name = validator.name or f"Model {i}"
        
        if validator.efficiency_data is None or validator.far_scores is None:
            continue
            
        snrs = validator.efficiency_data["snrs"]
        scores = validator.efficiency_data["scores"]
        far_scores = validator.far_scores
        
        # Calculate thresholds for requested FARs
        thresholds = calculate_far_score_thresholds(
            far_scores,
            validator.input_duration_seconds,
            np.array(fars)
        )
        
        # Binning setup - use rolling window for smoother curves
        min_snr = np.floor(snrs.min())
        max_snr = np.ceil(snrs.max())
        
        # Sort data by SNR for efficient rolling window
        sort_idx = np.argsort(snrs)
        sorted_snrs = snrs[sort_idx]
        sorted_scores = scores[sort_idx]
        
        model_curves = {}
        
        for far in fars:
            if far not in thresholds:
                continue
                
            thresh = thresholds[far][1]
            
            # Skip if threshold >= 1.0 - no sample can pass
            if thresh >= 1.0:
                model_curves[f"{far:.1e}"] = {"x": [], "y": []}
                continue
            
            # Rolling window approach
            window_half_width = 0.5
            min_samples = 20
            
            eval_snrs = np.linspace(min_snr + window_half_width, max_snr - window_half_width, 50)
            rolling_recalls = []
            valid_snrs = []
            
            for snr_center in eval_snrs:
                mask = (sorted_snrs >= snr_center - window_half_width) & (sorted_snrs < snr_center + window_half_width)
                if np.sum(mask) >= min_samples:
                    recall = np.mean(sorted_scores[mask] > thresh)
                    rolling_recalls.append(recall)
                    valid_snrs.append(snr_center)
            
            if len(valid_snrs) < 5:
                model_curves[f"{far:.1e}"] = {"x": [], "y": []}
                continue
            
            model_curves[f"{far:.1e}"] = {"x": list(valid_snrs), "y": list(rolling_recalls)}

        all_data[name] = model_curves
        
        # Store thresholds for this validator
        thresh_map = {}
        for far in fars:
            if far in thresholds:
                thresh_map[f"{far:.1e}"] = thresholds[far][1]
            else:
                thresh_map[f"{far:.1e}"] = 0.0
        
        all_thresholds[name] = thresh_map
        
        init_curve = model_curves.get(f"{initial_far:.1e}", {"x": [], "y": []})
        source = ColumnDataSource(data=dict(x=init_curve["x"], y=init_curve["y"]))
        sources[name] = source
        
        curve_line = p.line(
            x='x', y='y', source=source,
            line_width=3, color=color, legend_label=name
        )
        curve_renderers.append(curve_line)
        
        # Add invisible scatter markers for hover interaction only
        curve_circles = p.scatter(
            x='x', y='y', source=source,
            size=10, color=color, alpha=0, marker='circle'
        )
        curve_renderers.append(curve_circles)
        
        # Add dynamic threshold line
        init_thresh = thresh_map.get(f"{initial_far:.1e}", 0.0)
        thresh_source = ColumnDataSource(data=dict(x=[0, 20], y=[init_thresh, init_thresh]))
        thresh_sources[name] = thresh_source
        
        p.line(
            x='x', y='y', source=thresh_source,
            line_width=2, color=color, line_dash='dashed', alpha=0.7,
            legend_label=f"{name} Threshold"
        )
        
        # Add threshold label (dynamic text)
        thresh_label = Label(
            x=0.5, y=init_thresh, 
            text=f"{init_thresh}",
            text_font_size="15pt",
            text_color=color,
            x_units='screen', y_units='data',
            x_offset=5
        )
        p.add_layout(thresh_label)
        thresh_labels[name] = thresh_label
    
    p.legend.location = "bottom_right"
    p.legend.click_policy = "hide"
    p.legend.label_text_font_size = "18pt"  # 50% larger fonts
    
    # Add hover tool scoped only to efficiency curve lines
    if curve_renderers:
        curve_hover = HoverTool(
            renderers=curve_renderers,
            tooltips=[("SNR", "@x{0.1f}"), ("Recall", "@y{0.3f}")]
        )
        p.add_tools(curve_hover)
        
    # Filter FAR options
    valid_fars = []
    if validators and all_thresholds and all_data:
        first_model = list(all_thresholds.keys())[0]
        model_curves = all_data.get(first_model, {})
        for far_str, thresh_val in all_thresholds[first_model].items():
            curve_data = model_curves.get(far_str, {"x": [], "y": []})
            if thresh_val < 1.0 and len(curve_data.get("x", [])) > 0:
                valid_fars.append(far_str)
    
    far_options = valid_fars if valid_fars else [f"{f:.1e}" for f in fars]
    
    # Power-of-10 discrete options for dropdown
    power_of_10_options = []
    for far_str in far_options:
        try:
            val = float(far_str)
            import math
            log_val = math.log10(val)
            if abs(log_val - round(log_val)) < 0.01:
                exp = int(round(log_val))
                superscript_digits = {'0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴', 
                                      '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹', '-': '⁻'}
                exp_str = str(exp)
                superscript = ''.join(superscript_digits.get(c, c) for c in exp_str)
                display = f"10{superscript}"
                power_of_10_options.append((far_str, display))
        except:
            pass
    
    if not power_of_10_options:
        power_of_10_options = [(far_options[i], f"Option {i+1}") for i in range(min(4, len(far_options)))]
    
    slider = Slider(
        start=0, 
        end=len(far_options)-1, 
        value=0, 
        step=1, 
        title=f"FAR: {far_options[0]}",
        show_value=False
    )
    
    select_options = [(opt[0], opt[1]) for opt in power_of_10_options]
    select = Select(
        title="FAR (Power of 10):",
        value=power_of_10_options[0][0] if power_of_10_options else far_options[0],
        options=select_options
    )
    
    code = """
        let active_thresh = null;
        const far_index = cb_obj.value;
        const far_val = far_options[far_index];
        cb_obj.title = "FAR: " + far_val;
        
        for (const name in sources) {
            const source = sources[name];
            const data = all_data[name][far_val];
            if (data) {
                source.data.x = data.x;
                source.data.y = data.y;
                source.change.emit();
            }
            
            // Update threshold line
            const thresh_source = thresh_sources[name];
            const thresh = all_thresholds[name][far_val];
            if (thresh_source && thresh !== undefined) {
                thresh_source.data.y = [thresh, thresh];
                thresh_source.change.emit();
            }
            
            // Update threshold label
            const label = thresh_labels[name];
            if (label && thresh !== undefined) {
                label.y = thresh;
                label.text = String(thresh);
            }
            
            if (name === current_model_name && thresh !== undefined) {
                active_thresh = thresh;
            }
        }
        
        // Heatmap Update Logic
        if (heatmap_source && active_thresh !== null) {
            const data = heatmap_source.data;
            const counts_flat = data['counts'][0]; 
            const image = data['image'][0]; 
            const y0 = data['y'][0];
            const dh = data['dh'][0];
            
            const N = 400; // width
            const M = 400; // height
            
            for (let i = 0; i < image.length; i++) {
                const count = counts_flat[i];
                if (count > 0) {
                    const r = Math.floor(i / N);
                    const y_val = y0 + (r / M) * dh;
                    
                    let color = 0;
                    if (y_val >= active_thresh) {
                        color = 0xFF00CC00; // Green
                    } else {
                        color = 0xFFAAAAAA; // Grey
                    }
                    image[i] = color;
                } else {
                    image[i] = 0;
                }
            }
            heatmap_source.change.emit();
        }
    """
    
    primary_name = validators[-1].name if validators else "Model 0"
    
    slider_callback = CustomJS(
        args=dict(
            sources=sources, 
            all_data=all_data, 
            far_options=far_options, 
            thresh_sources=thresh_sources, 
            all_thresholds=all_thresholds,
            thresh_labels=thresh_labels,
            heatmap_source=heatmap_source,
            current_model_name=primary_name
        ), 
        code=code
    )
    slider.js_on_change('value', slider_callback)
    
    select_code = """
        const selected_far = cb_obj.value;
        let best_idx = 0;
        let best_diff = Infinity;
        for (let i = 0; i < far_options.length; i++) {
            const opt = parseFloat(far_options[i]);
            const sel = parseFloat(selected_far);
            const diff = Math.abs(Math.log10(opt) - Math.log10(sel));
            if (diff < best_diff) {
                best_diff = diff;
                best_idx = i;
            }
        }
        slider.value = best_idx;
    """
    select.js_on_change('value', CustomJS(
        args=dict(slider=slider, far_options=far_options),
        code=select_code
    ))
    
    controls = row(select, slider)
    
    return p, controls


def generate_real_events_table(
    events: List[Dict],
    far_scores: np.ndarray = None,
    input_duration_seconds: float = 1.0,
    far_thresholds: List[float] = [1e-1, 1e-2, 1e-3, 1e-4],
    width: int = 1200
):
    """
    Generate interactive table for real GW events with FAR-based coloring.
    
    Args:
        events: List of event dicts with name, gps, mass1, mass2, distance, score, etc.
        far_scores: Noise scores for FAR threshold calculation
        input_duration_seconds: Duration per sample for FAR calculation
        far_thresholds: List of FAR values for dropdown
        width: Table width in pixels
    
    Returns:
        Tuple of (table_layout, far_select)
    """
    from .utils import calculate_far_score_thresholds
    import panel as pn
    
    if not events:
        return pn.pane.Markdown("**No real events data available.**"), None
    
    # Calculate FAR thresholds if we have noise scores
    if far_scores is not None and len(far_scores) > 0:
        thresholds = calculate_far_score_thresholds(
            far_scores, input_duration_seconds, np.array(far_thresholds)
        )
    else:
        thresholds = {}
    
    # Separate events by type
    confident = [e for e in events if e.get("event_type") == "CONFIDENT"]
    marginal = [e for e in events if e.get("event_type") == "MARGINAL"]
    
    # Build DataFrame for display
    rows = []
    for event in confident + marginal:
        score = event.get("score")
        
        # Calculate minimum FAR at which this event would be detected
        min_far = None
        if score is not None and thresholds:
            for far in sorted(far_thresholds):
                far_key = far
                if far_key in thresholds:
                    thresh_score = thresholds[far_key][1]
                    if score >= thresh_score:
                        min_far = far
                        break
        
        # Convert GPS to human-readable date
        gps_time = event.get('gps', 0)
        from datetime import datetime, timedelta
        try:
            # GPS epoch is Jan 6, 1980 00:00:00 UTC
            # GPS has leap seconds - approximate with +18s (as of 2024)
            gps_epoch = datetime(1980, 1, 6, 0, 0, 0)
            dt = gps_epoch + timedelta(seconds=gps_time + 18)
            date_str = dt.strftime("%Y-%m-%d %H:%M")
        except:
            date_str = "-"
        
        row = {
            "Name": event.get("name", "Unknown"),
            "Type": event.get("event_type", "Unknown"),
            "Run": event.get("observing_run", "?"),
            "Date": date_str,
            "M₁ (M☉)": f"{event.get('mass1', np.nan):.1f}" if not np.isnan(event.get('mass1', np.nan)) else "-",
            "M₂ (M☉)": f"{event.get('mass2', np.nan):.1f}" if not np.isnan(event.get('mass2', np.nan)) else "-",
            "D (Mpc)": f"{event.get('distance', np.nan):.0f}" if not np.isnan(event.get('distance', np.nan)) else "-",
            "p(BBH)": f"{event.get('p_bbh', np.nan):.2f}" if not np.isnan(event.get('p_bbh', np.nan)) else "-",
            "p(BNS)": f"{event.get('p_bns', np.nan):.2f}" if not np.isnan(event.get('p_bns', np.nan)) else "-",
            "Score": f"{score:.3f}" if score is not None else "Pending",
            "Min FAR": f"{min_far:.1e}" if min_far is not None else "-",
            "_score_num": score if score is not None else -1,
            "_type": event.get("event_type", "Unknown"),
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Create styled table with color coding
    def style_row(row):
        """Apply color based on event type and detection status."""
        colors = []
        event_type = row.get("_type", "Unknown")
        score = row.get("_score_num", -1)
        
        # Base color by type
        if event_type == "CONFIDENT":
            base_color = "#e6f3ff"  # Light blue background
        else:
            base_color = "#fff3e6"  # Light orange background
        
        return pd.Series([f"background-color: {base_color}"] * len(row))
    
    # Create FAR dropdown
    far_options = [(f"{f:.1e}", f"FAR ≤ {f:.1e}") for f in far_thresholds]
    far_select = pn.widgets.Select(
        name="Detection Threshold (FAR)",
        options=dict([(v[1], v[0]) for v in far_options]),
        value=far_options[0][0],
        width=200
    )
    
    # Display columns (exclude internal columns)
    display_cols = ["Name", "Type", "Run", "Date", "M₁ (M☉)", "M₂ (M☉)", "D (Mpc)", "p(BBH)", "p(BNS)", "Score", "Min FAR"]
    
    # Summary stats
    num_confident = len(confident)
    num_marginal = len(marginal)
    scored_events = [e for e in events if e.get("score") is not None]
    
    summary_html = f"""
    <div style="padding: 10px; background: #2a2a2a; border-radius: 5px; margin-bottom: 10px;">
        <b>Real Event Summary:</b><br>
        • <span style="color: #66b3ff;">Confident Events:</span> {num_confident}<br>
        • <span style="color: #ffb366;">Marginal Events:</span> {num_marginal}<br>
        • Scored: {len(scored_events)} / {len(events)}
    </div>
    """
    summary = pn.pane.HTML(summary_html)
    
    # Create table
    table = pn.widgets.DataFrame(
        df[display_cols],
        width=width,
        height=min(600, 30 * len(df) + 50),
        auto_edit=False,
        show_index=False
    )
    
    return pn.Column(summary, far_select, table), far_select
