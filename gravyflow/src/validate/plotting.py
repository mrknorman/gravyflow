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
from .utils import downsample_data

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


def _display_indices(n: int, max_points: int) -> np.ndarray:
    """Indices of at most ``max_points`` exact points, dense at the start of a ranked curve."""
    if n <= max_points:
        return np.arange(n)
    return np.unique(np.geomspace(1, n, max_points).astype(int) - 1)


def _title(validator, index: int) -> str:
    return gf.snake_to_capitalized_spaces(validator.name) if validator.name else f"Model {index}"


def _style(p, legend_location: str) -> None:
    p.legend.location = legend_location
    p.legend.click_policy = "hide"
    p.legend.label_text_font_size = "18pt"
    p.axis.axis_label_text_font_size = "21pt"
    p.axis.major_label_text_font_size = "18pt"
    p.title.text_font_size = '24pt'


def generate_far_curves(
        validators : list,
        colors : List[str] = Bright[7],
        width : int = 800,
        height : int = 600
    ):
    """FAR (Hz) against the score threshold (logit), with each curve's Poisson band."""
    tooltips = [
        ("Score Threshold", "@x{0.000}"),
        ("False Alarm Rate (Hz)", "@y{0.0e}"),
        ("Interval (Hz)", "@lower{0.0e} – @upper{0.0e}"),
    ]
    p = figure(
        width=width,
        height=height,
        x_axis_label="Score Threshold (logit)",
        y_axis_label="False Alarm Rate (Hz)",
        tooltips=tooltips,
        y_axis_type="log"
    )
    for index, (color, validator) in enumerate(zip(cycle(colors), validators)):
        if validator.far_scores is None:
            continue
        confidence = validator.config.confidence
        curve = validator.background().far_curve(confidence)
        keep = _display_indices(curve.thresholds.size, 2000)
        source = ColumnDataSource(data=dict(
            x=curve.thresholds[keep], y=curve.rate_hz[keep],
            lower=curve.lower_hz[keep], upper=curve.upper_hz[keep],
        ))
        title = _title(validator, index)
        p.varea(x="x", y1="lower", y2="upper", source=source, fill_color=color, fill_alpha=0.2,
                legend_label=f"{title} ({confidence:.0%} interval)")
        p.line("x", "y", source=source, line_color=color, line_width=2, legend_label=title)
    _style(p, "bottom_left")
    return p

#: The sensitive volume plot's y axes: within each curve's reach R; Euclidean volume, detector-frame masses (as
#: MLGWSC-1's evaluate.py).
VOLUME_AXIS_LABELS = ("Sensitive Volume within reach R (Gpc³, Euclidean, detector-frame masses)",
                      "Sensitive Distance within reach R (Mpc, Euclidean, detector-frame masses)")


def generate_sensitive_volume_plot(
        validators: list,
        colors: List[str] = Bright[7],
        width: int = 1000,
        height: int = 600
    ):
    """
    Sensitive volume (Gpc³) and sensitive distance (Mpc) within each curve's reach R
    (named in its legend) against FAR per month: each curve with its joint interval
    (the wider band), its injection interval (the narrower band), the Poisson
    interval of the FAR its threshold achieves (the horizontal bars, the
    background's term), its noise floor (dashed), and, dotted, V within R plus the
    estimated volume beyond R (out to the curve's ``beyond_reach_factor`` times R),
    whose interval the tooltip gives. The points `SensitiveVolumeCurve.quote` will
    not quote are ringed (noise-dominated); the reach-limited ones, where the
    estimate beyond R exceeds its stated fraction of V, are crossed: a warning.
    Volumes are Euclidean, of populations whose masses are in the detector frame, as
    the axes say. A FAR the background cannot resolve, and a validator with no
    volume set, have no points.

    Returns:
        A column of the two figures.
    """
    from .volume import SECONDS_PER_MONTH, SECONDS_PER_YEAR, sensitive_distance

    tooltips = [
        ("FAR (per month)", "@x{0.00e}"), ("FAR (per year)", "@per_year{0.00e}"),
        ("Within reach R", "@y{0.000}"), ("Joint", "@joint_lower{0.000} – @joint_upper{0.000}"),
        ("Injections", "@injection_lower{0.000} – @injection_upper{0.000}"),
        ("Background", "@background_lower{0.000} – @background_upper{0.000}"),
        ("Noise floor", "@noise_floor{0.000}"),
        ("With the estimate beyond R", "@with_beyond{0.000} (@with_beyond_lower{0.000} – @with_beyond_upper{0.000})"),
    ]
    figures = [
        figure(width=width, height=height, x_axis_type="log", x_axis_label="False Alarm Rate (per month)",
               y_axis_label=label, tools="pan,box_zoom,wheel_zoom,reset", tooltips=tooltips)
        for label in VOLUME_AXIS_LABELS
    ]
    for index, (color, validator) in enumerate(zip(cycle(colors), validators)):
        curve = validator.sensitive_volume_curve()
        if curve is None:
            continue
        ok = curve.resolved
        title = f"{_title(validator, index)} (R = {curve.reach_mpc:,.0f} Mpc)"
        within = curve.volume_within_reach_gpc3
        for p, convert in zip(figures, (np.asarray, sensitive_distance)):
            source = ColumnDataSource(data=dict(
                x=curve.far_hz[ok] * SECONDS_PER_MONTH, per_year=curve.far_hz[ok] * SECONDS_PER_YEAR,
                x_lower=curve.far_lower_hz[ok] * SECONDS_PER_MONTH,
                x_upper=curve.far_upper_hz[ok] * SECONDS_PER_MONTH,
                y=convert(within[ok]),
                noise_floor=convert(curve.noise_floor_gpc3[ok]),
                **{f"with_beyond{end}": convert(within[ok] + getattr(curve, f"beyond_reach{end}_gpc3")[ok])
                   for end in ("", "_lower", "_upper")},
                **{f"{term}_{end}": convert(getattr(curve, f"{term}_{end}_gpc3")[ok])
                   for term in ("joint", "injection", "background") for end in ("lower", "upper")},
            ))
            p.varea(x="x", y1="joint_lower", y2="joint_upper", source=source, fill_color=color, fill_alpha=0.15,
                    legend_label=f"{title} (joint {curve.confidence:.0%} interval)")
            p.varea(x="x", y1="injection_lower", y2="injection_upper", source=source, fill_color=color,
                    fill_alpha=0.3, legend_label=f"{title} (injections)")
            p.segment(x0="x_lower", x1="x_upper", y0="y", y1="y", source=source, line_color=color,
                      legend_label=f"{title} (FAR interval)")
            p.line("x", "y", source=source, line_color=color, line_width=2, legend_label=title)
            p.scatter("x", "y", source=source, size=6, color=color, legend_label=title)
            p.line("x", "noise_floor", source=source, line_color=color, line_dash="dashed",
                   legend_label=f"{title} (noise floor)")
            p.line("x", "with_beyond", source=source, line_color=color, line_dash="dotted",
                   legend_label=f"{title} (with the estimate to {curve.beyond_reach_factor:g} R)")
            for marked, marker, what in (
                    (~curve.quotable[ok], "circle", f"not quoted: noise floor > {curve.noise_floor_fraction:.0%} of V"),
                    (curve.reach_limited[ok], "x", f"reach-limited: beyond R > {curve.beyond_reach_fraction:.0%} of V")):
                if marked.any():
                    p.scatter(x=curve.far_hz[ok][marked] * SECONDS_PER_MONTH, y=convert(within[ok][marked]),
                              size=14, marker=marker, fill_alpha=0, line_color=color, line_width=2,
                              legend_label=f"{title} ({what})")
    for p in figures:
        if p.renderers:
            _style(p, "bottom_right")
    return column(*figures)


def generate_roc_curves(
    validators: list,
    colors : List[str] = Bright[7],
    width : int = 800,
    height : int = 600
    ):
    """Exact ROC curves (per window) of noise against an injection population."""
    p = figure(
        x_axis_label='False Positive Rate (per window)',
        y_axis_label='True Positive Rate (%)',
        width=width,
        height=height,
        x_axis_type='log',
        x_range=[1e-6, 1],
        y_range=[0.0, 100.0]
    )
    max_num_points = 500
    all_data, all_sources = {}, {}
    for index, validator in enumerate(validators):
        name = validator.name or f"Model {index}"
        all_data[name] = {}
        for population, roc in validator.roc_curves().items():
            keep = _display_indices(roc["fpr"].size, max_num_points)
            all_data[name][population] = {
                'fpr': roc["fpr"][keep], 'tpr': roc["tpr"][keep] * 100, 'roc_auc': roc["roc_auc"]
            }
    populations = list(next(iter(all_data.values()), {}).keys())
    if not populations:
        return p, Select(title="Test Population:", options=[])
    initial = populations[0]

    for index, (color, validator) in enumerate(zip(cycle(colors), validators)):
        name = validator.name or f"Model {index}"
        data = all_data[name].get(initial)
        if data is None:
            continue
        source = ColumnDataSource(data=dict(x=data['fpr'], y=data['tpr']))
        all_sources[name] = source
        line = p.line(x='x', y='y', source=source, color=color, width=2,
                      legend_label=f'{_title(validator, index)} (area = {data["roc_auc"]:.5f})')
        p.add_tools(HoverTool(
            tooltips=[("Series", _title(validator, index)),
                      ("False Positive Rate", "@x{0.0000e}"),
                      ("True Positive Rate (%)", "@y{0.00}")],
            renderers=[line]
        ))
    _style(p, "bottom_right")

    select = Select(title="Test Population:", value=initial, options=populations)
    select.js_on_change('value', CustomJS(
        args={'all_sources': all_sources, 'all_data': all_data},
        code="""
        const population = cb_obj.value;
        for (const name in all_sources) {
            const data = all_data[name][population];
            if (!data) { continue; }
            all_sources[name].data = {x: data.fpr, y: data.tpr};
        }
        """
    ))
    return p, select

def first_detector(sample: dict):
    """
    A worst performer's whitened onsource and injection in its first detector.

    ``sample`` is one example of batch-first outputs: WHITENED_ONSOURCE is
    (detectors, samples) and WHITENED_INJECTIONS (generators, detectors,
    samples), summed over its generators. The injection is None for a sample
    without one (a noise-only false positive).
    """
    onsource = np.asarray(sample[gf.ReturnVariables.WHITENED_ONSOURCE.name])
    if onsource.ndim != 2:
        raise ValueError(f"WHITENED_ONSOURCE: expected (detectors, samples), got {onsource.shape}.")
    injection = sample.get(gf.ReturnVariables.WHITENED_INJECTIONS.name)
    if injection is None:
        return onsource[0], None
    injection = np.asarray(injection)
    if injection.ndim != 3:
        raise ValueError(
            f"WHITENED_INJECTIONS: expected (generators, detectors, samples), got {injection.shape}.")
    return onsource[0], injection.sum(axis=0)[0]

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
    
    onsource_data, injection_data = first_detector(data)
    has_injection = injection_data is not None

    # Cast onsource to float32 for Datashader (avoids float16 error)
    onsource_data = onsource_data.astype(np.float32)
    # Cast injection to float64 for Bokeh/JS compatibility (avoids serialization/browser errors)
    if has_injection:
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
        except (TypeError, ValueError, OverflowError):
             return default
    
    # Extract parameters
    mass1 = get_scalar(gf.WaveformParameters.MASS_1_MSUN.name, 0)
    mass2 = get_scalar(gf.WaveformParameters.MASS_2_MSUN.name, 0)
    score = get_scalar('score', 0)
    snr = get_scalar(gf.ScalingTypes.SNR.name)
    GPS_TIME_KEY = gf.ReturnVariables.START_GPS_TIME.name
    gps_time = get_scalar(GPS_TIME_KEY)
    
    # Convert GPS to human readable (approximate - ignores leap seconds)
    human_time = "N/A"
    if gps_time is not None and gps_time > 0:
        # GPS epoch: Jan 6, 1980. GPS is ~18s ahead of UTC due to leap seconds
        gps_epoch = datetime(1980, 1, 6, 0, 0, 0)
        try:
            dt = gps_epoch + timedelta(seconds=float(gps_time) - 18)  # Approximate UTC
            human_time = dt.strftime("%Y-%m-%d %H:%M:%S UTC")
        except (TypeError, ValueError, OverflowError):
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
    
    if has_injection:
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
        idx = gf.SeedTree().direct().stream("validator/plot").choice(n_total, max_points, replace=False)
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
        y_axis_label="Average Score (logit)",
        width=width,
        height=200,
        tools="pan,box_zoom,wheel_zoom,reset",
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
    # Scores are logits (unbounded): span the central 98% of them.
    mapper = linear_cmap(
        field_name='color',
        palette=Viridis256,
        low=float(np.percentile(scores, 1)) if scores.size else 0.0,
        high=float(np.percentile(scores, 99)) if scores.size else 1.0
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
    colors: List[str] = Bright[7],
    width: int = 1000,
    height: int = 800
):
    """
    Efficiency against injected SNR at a chosen FAR, one point per SNR bin with
    its binomial interval. A FAR the noise background cannot resolve has no curve.

    Returns:
        ``(figure, select)``: the plot and the FAR selector.
    """
    p = figure(
        x_axis_label="Injected SNR (bin mean)",
        y_axis_label="Efficiency",
        y_range=(0, 1.05),
        width=width,
        height=height,
        tools="pan,box_zoom,wheel_zoom,reset"
    )
    all_data, sources = {}, {}
    far_keys: List[str] = []
    for index, (validator, color) in enumerate(zip(validators, cycle(colors))):
        if validator.efficiency_data is None or validator.far_scores is None:
            continue
        name = validator.name or f"Model {index}"
        curves = {}
        for far, (threshold, curve) in validator.efficiency_curves().items():
            key = f"{far:.1e}"
            ok = np.isfinite(curve.efficiency)
            curves[key] = dict(
                x=curve.snr_mean[ok].tolist(), y=curve.efficiency[ok].tolist(),
                lower=curve.lower[ok].tolist(), upper=curve.upper[ok].tolist(),
                n=curve.total[ok].tolist(),
            )
            if threshold.resolved and key not in far_keys:
                far_keys.append(key)
        all_data[name] = curves
        sources[name] = ColumnDataSource(data=dict(x=[], y=[], lower=[], upper=[], n=[]))
        confidence = validator.config.confidence
        p.varea(x="x", y1="lower", y2="upper", source=sources[name], fill_color=color,
                fill_alpha=0.2, legend_label=f"{name} ({confidence:.0%} interval)")
        line = p.line(x="x", y="y", source=sources[name], line_width=3, color=color, legend_label=name)
        p.scatter(x="x", y="y", source=sources[name], size=8, color=color)
        p.add_tools(HoverTool(renderers=[line], tooltips=[
            ("SNR", "@x{0.0}"), ("Efficiency", "@y{0.000}"),
            ("Interval", "@lower{0.000} – @upper{0.000}"), ("Injections", "@n"),
        ]))

    if not far_keys:
        far_keys = [f"{f:.1e}" for f in (validators[-1].config.far_thresholds if validators else [])][:1]
    initial = far_keys[0] if far_keys else None
    for name, source in sources.items():
        if initial in all_data[name]:
            source.data = all_data[name][initial]
    if sources:
        p.legend.location = "bottom_right"
        p.legend.click_policy = "hide"
        p.legend.label_text_font_size = "18pt"
    p.axis.axis_label_text_font_size = "21pt"
    p.axis.major_label_text_font_size = "18pt"

    select = Select(title="False Alarm Rate (Hz):", value=initial, options=far_keys)
    select.js_on_change('value', CustomJS(
        args=dict(sources=sources, all_data=all_data),
        code="""
        const far = cb_obj.value;
        for (const name in sources) {
            const data = all_data[name][far];
            sources[name].data = data ? data : {x: [], y: [], lower: [], upper: [], n: []};
        }
        """
    ))
    return p, select

def generate_real_events_table(
    events: List[Dict],
    input_duration_seconds: float,
    far_thresholds: List[float] = [1e-1, 1e-2, 1e-3, 1e-4],
    width: int = 1200,
    num_waveform_plots: int = 5
):
    """
    Generate interactive table for real GW events.

    Each scored event carries its FAR (Hz) read from the validator's noise
    background (``far_hz``) and the upper end of its interval (``far_upper_hz``).

    Args:
        events: List of event dicts with name, gps, mass1, mass2, distance, p_astro,
            score, etc. (``get_events_with_params``, scored by the validator)
        input_duration_seconds: The duration of each event's whitened strain
        far_thresholds: List of FAR values for dropdown
        width: Table width in pixels

    Returns:
        Tuple of (table_layout, far_select)
    """
    import panel as pn

    if not events:
        return pn.pane.Markdown("**No real events data available.**"), None
    
    # Separate events by type
    confident = [e for e in events if e.get("event_type") == "CONFIDENT"]
    marginal = [e for e in events if e.get("event_type") == "MARGINAL"]
    
    # Build DataFrame for display
    rows = []
    for event in confident + marginal:
        score = event.get("score")
        
        far_hz, far_upper_hz = event.get("far_hz"), event.get("far_upper_hz")
        
        # Convert GPS to human-readable date
        gps_time = event.get('gps', 0)
        from datetime import datetime, timedelta
        try:
            # GPS epoch is Jan 6, 1980 00:00:00 UTC
            # GPS has leap seconds - approximate with +18s (as of 2024)
            gps_epoch = datetime(1980, 1, 6, 0, 0, 0)
            dt = gps_epoch + timedelta(seconds=gps_time + 18)
            date_str = dt.strftime("%Y-%m-%d %H:%M")
        except (TypeError, ValueError, OverflowError):
            date_str = "-"
        
        # Helper for safe formatting
        def safe_fmt(key, fmt=".1f"):
            val = event.get(key)
            if val is None: 
                return "-"
            try:
                if np.isnan(val): return "-"
                return f"{val:{fmt}}"
            except (TypeError, ValueError, OverflowError):
                return "-"

        row = {
            "Name": event.get("name", "Unknown"),
            "Type": event.get("event_type", "Unknown"),
            "Run": event.get("observing_run", "?"),
            "Date": date_str,
            "M₁ (M☉)": safe_fmt("mass1", ".1f"),
            "M₂ (M☉)": safe_fmt("mass2", ".1f"),
            "D (Mpc)": safe_fmt("distance", ".0f"),
            "p(astro)": safe_fmt("p_astro", ".2f"),
            "Score": f"{score:.3f}" if score is not None else "Pending",
            "FAR (Hz)": f"{far_hz:.1e} (≤ {far_upper_hz:.1e})" if far_hz is not None else "-",
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
    display_cols = ["Name", "Type", "Run", "Date", "M₁ (M☉)", "M₂ (M☉)", "D (Mpc)", "p(astro)", "Score", "FAR (Hz)"]
    
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
    
    # ---------------------------------------------------------
    # Generate Waveform Plots for Top Events
    # ---------------------------------------------------------
    plot_items = []
    
    # Filter events with waveform data and valid score
    events_with_wave = [e for e in events if "whitened_strain" in e and e.get("score") is not None]
    
    # Sort by score descending
    events_with_wave.sort(key=lambda x: float(x["score"]), reverse=True)
    
    # Take top N
    top_events = events_with_wave[:num_waveform_plots]
    
    if top_events:
        plot_items.append(pn.layout.Divider())
        plot_items.append(pn.pane.Markdown(f"### Top {len(top_events)} Detected Events (Whitened Waveforms)"))
        
        from bokeh.plotting import figure
        from bokeh.models import Legend
        
        for i, event in enumerate(top_events):
            strain = event["whitened_strain"] # (IFOs, T)
            if not isinstance(strain, np.ndarray): continue
            
            num_ifos, num_samples = strain.shape
            time = np.linspace(0, input_duration_seconds, num_samples)
            
            # Create figure
            p_wave = figure(
                title=f"Event: {event.get('name', 'Unknown')} (Score: {event.get('score', 0):.3f})",
                x_axis_label="Time (s)",
                y_axis_label="Whitened Strain",
                width=int(width * 0.7),
                height=300,
                tools="pan,box_zoom,reset,save"
            )
            
            colors = ["#1f77b4", "#d62728", "#2ca02c"] # Blue, Red, Green
            legends = []
            
            for ch in range(num_ifos):
                color = colors[ch % len(colors)]
                # Attempt to guess IFO based on index (standard: H1, L1, V1)
                lbl = ["H1", "L1", "V1"][ch] if ch < 3 else f"Ch{ch}"
                
                line = p_wave.line(time, strain[ch], color=color, line_width=1.5)
                legends.append((lbl, [line]))
                
            legend = Legend(items=legends, location="top_right")
            p_wave.add_layout(legend)
            
            # Helper for safe formatting
            def safe_val(key, fmt=".1f"):
                val = event.get(key)
                if val is None: return "-"
                try: 
                    if np.isnan(val): return "-"
                    return f"{val:{fmt}}"
                except (TypeError, ValueError, OverflowError): return "-"

            # Parameter Box (Dark Theme)
            params_html = f"""
            <div style="background-color: #2b2b2b; padding: 15px; border-left: 5px solid #007bff; border-radius: 4px; font-family: monospace; color: #e0e0e0;">
                <h4 style="margin-top:0; color: #ffffff;">Parameters</h4>
                <b>GPS:</b> {safe_val('gps', '.2f')}<br>
                <b>M1:</b> {safe_val('mass1', '.1f')} M☉<br>
                <b>M2:</b> {safe_val('mass2', '.1f')} M☉<br>
                <b>Dist:</b> {safe_val('distance', '.0f')} Mpc<br>
                <b>p(astro):</b> {safe_val('p_astro', '.2f')}<br>
                <b>Status:</b> {event.get('event_type', 'Unknown')}
            </div>
            """
            
            row = pn.Row(p_wave, pn.pane.HTML(params_html, width=int(width * 0.25)))
            plot_items.append(row)
            plot_items.append(pn.layout.Divider())

    return pn.Column(summary, far_select, table, *plot_items), far_select
