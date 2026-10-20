"""
The dashboard's Glitches and Timing tabs.

- **Glitches**: the false-positive rate per glitch class at each false-alarm-rate level,
  with 1σ and 2σ bars; the volume on real data against its Gaussian twin; and which
  glitches make the tail (the classes of the background triggers above each level, and
  the loudest triggers with their nearest glitch).
- **Timing**: throughput and latency (median and tail) with the machine, thread count and
  host load at the time; A-B-B-A rounds where measured; the training cost.

A part with no record says so rather than vanishing.
"""

from __future__ import annotations

from itertools import cycle
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
from bokeh.models import ColumnDataSource
from bokeh.palettes import Bright
from bokeh.plotting import figure

from .glitches import tail_by_class, twin_comparison, validator_glitch_rates
from .plotting import _band_alpha, _sigma_label, _style, _title

#: The loudest background triggers the Glitches tab lists with their nearest glitch.
TAIL_ROWS = 25


def _note(text: str):
    import panel as pn

    return pn.pane.Markdown(text, sizing_mode="stretch_width")


def _table(frame: pd.DataFrame):
    import panel as pn

    return pn.pane.DataFrame(frame, index=False, sizing_mode="stretch_width")


# ------------------------------------------------------------------ Glitches

def glitch_rate_plot(validators: Sequence, colors=Bright[7], width: int = 900, height: int = 500):
    """Each class's false-positive rate at each resolved level, with 1σ and 2σ bars; None without a glitch record."""
    series = [(index, v, validator_glitch_rates(v)) for index, v in enumerate(validators)]
    series = [(index, v, rates) for index, v, rates in series if rates is not None]
    if not series:
        return None
    classes = sorted({label for _, _, rates in series for label in rates.labels})
    position = {label: k for k, label in enumerate(classes)}
    p = figure(width=width, height=height, x_axis_label="Glitch class", y_axis_label="False-positive rate",
               y_range=(-0.02, 1.02), x_range=(-0.6, len(classes) - 0.4),
               tooltips=[("Class", "@label"), ("Level", "@far"), ("Rate", "@rate{0.000}"),
                         ("Passed / glitches", "@passed / @count"), ("1σ", "@lower_0{0.000} – @upper_0{0.000}"),
                         ("2σ", "@lower_1{0.000} – @upper_1{0.000}")])
    groups = [(index, v, rates, f) for index, v, rates in series for f in range(rates.far_hz.size)
              if np.isfinite(rates.threshold[f])]
    width_each = 0.8 / max(len(groups), 1)
    for g, (color, (index, v, rates, f)) in enumerate(zip(cycle(colors), groups)):
        x = np.array([position[label] for label in rates.labels]) - 0.4 + (g + 0.5) * width_each
        source = ColumnDataSource(dict(
            x=x, rate=rates.rate[f], label=list(rates.labels), far=[f"{rates.far_hz[f]:.1e} Hz"] * x.size,
            passed=rates.passed[f], count=rates.counts,
            **{f"lower_{s}": rates.lower[s, f] for s in range(len(rates.sigmas))},
            **{f"upper_{s}": rates.upper[s, f] for s in range(len(rates.sigmas))}))
        legend = f"{_title(v, index)} at {rates.far_hz[f]:.1e} Hz"
        for s, sigma in reversed(list(enumerate(rates.sigmas))):
            p.segment(x0="x", x1="x", y0=f"lower_{s}", y1=f"upper_{s}", source=source, line_color=color,
                      line_width=2 if s == 0 else 1, line_alpha=0.5 + _band_alpha(sigma),
                      legend_label=f"{legend} ({_sigma_label(sigma)})")
        p.scatter("x", "rate", source=source, size=9, color=color, legend_label=legend)
    p.xaxis.ticker = list(range(len(classes)))
    p.xaxis.major_label_overrides = {k: label for k, label in enumerate(classes)}
    p.xaxis.major_label_orientation = 0.8
    _style(p, "top_right")
    p.legend.label_text_font_size = "12pt"
    return p


def twin_plot(validators: Sequence, twins: Dict[str, object], colors=Bright[7], width: int = 900, height: int = 450):
    """The real volume over its Gaussian twin's against FAR, with 1σ and 2σ bands, and its table; None without a twin."""
    rows, p = [], None
    for color, (index, v) in zip(cycle(colors), enumerate(validators)):
        twin = twins.get(v.name)
        real_curve = v.sensitive_volume_curve() if twin is not None else None
        twin_curve = twin.sensitive_volume_curve() if twin is not None else None
        if real_curve is None or twin_curve is None:
            continue
        cmp = twin_comparison(real_curve, twin_curve)
        if p is None:
            p = figure(width=width, height=height, x_axis_type="log", x_axis_label="False Alarm Rate (Hz)",
                       y_axis_label="Volume, real over Gaussian twin")
        title = _title(v, index)
        keep = np.isfinite(cmp.ratio)
        for s, sigma in reversed(list(enumerate(cmp.sigmas))):
            p.varea(x=cmp.far_hz[keep], y1=cmp.ratio_lower[s][keep], y2=cmp.ratio_upper[s][keep], fill_color=color,
                    fill_alpha=_band_alpha(sigma), legend_label=f"{title} ({_sigma_label(sigma)})")
        p.line(cmp.far_hz[keep], cmp.ratio[keep], line_color=color, line_width=2, legend_label=title)
        for f in range(cmp.far_hz.size):
            rows.append({"Model": title, "FAR (Hz)": f"{cmp.far_hz[f]:.1e}", "Real (Gpc³)": cmp.real_gpc3[f],
                         "Twin (Gpc³)": cmp.twin_gpc3[f], "Real / twin": cmp.ratio[f],
                         **{f"{_sigma_label(sig)}": f"{cmp.ratio_lower[s][f]:.3g} – {cmp.ratio_upper[s][f]:.3g}"
                            for s, sig in enumerate(cmp.sigmas)},
                         "Quotable": bool(cmp.quotable[f])})
    if p is None:
        return None, None
    p.line([float(np.min(cmp.far_hz)), float(np.max(cmp.far_hz))], [1.0, 1.0], line_dash="dashed",
           line_color="gray", legend_label="No difference")
    _style(p, "bottom_right")
    return p, pd.DataFrame(rows)


def tail_tables(validator, near_seconds: float = 1.0):
    """The classes of the background triggers above each level, and the loudest triggers with their nearest glitch."""
    record = getattr(validator, "trigger_glitches", None)
    if record is None:
        return None, None
    background = validator.background()
    rows = []
    for far in validator.config.far_thresholds:
        threshold = background.threshold_for_far(far).threshold
        if not np.isfinite(threshold):
            continue
        for label, count in tail_by_class(record, threshold, near_seconds).items():
            rows.append({"FAR (Hz)": f"{far:.1e}", "Threshold": threshold, "Nearest glitch class": label,
                         "Triggers": count})
    order = np.argsort(-np.asarray(record["scores"], dtype=np.float64), kind="stable")[:TAIL_ROWS]
    loudest = pd.DataFrame({
        "Score": np.asarray(record["scores"])[order],
        "GPS (first detector)": np.asarray(record["gps"])[order, 0],
        "Nearest glitch class": np.asarray(record["label"]).astype(str)[order],
        "Detector": np.asarray(record["ifo"]).astype(str)[order],
        "Distance (s)": np.asarray(record["distance_seconds"])[order],
    })
    return pd.DataFrame(rows), loudest


def generate_glitch_tab(validators: Sequence, twins: Optional[Dict[str, object]] = None):
    """The Glitches tab (module docstring)."""
    import panel as pn

    items = [_note("### False positives per glitch class\nEach labelled glitch scored by the model; a class's "
                   "rate is the fraction of its glitches whose loudest trigger within the found window passes "
                   "the level's threshold.")]
    rates = glitch_rate_plot(validators)
    items.append(rates if rates is not None else _note("No glitch record: the model was not run over labelled "
                                                       "glitches (Gravity Spy labels cover O1 to O3b)."))
    items.append(_note("### Real data against its Gaussian twin"))
    plot, table = twin_plot(validators, twins or {})
    items.extend([plot, _table(table)] if plot is not None else
                 [_note("No Gaussian twin given: the volume on real data is not compared with its twin.")])
    items.append(_note("### Which glitches make the tail"))
    self_validator = validators[-1]
    by_class, loudest = tail_tables(self_validator)
    if by_class is None:
        items.append(_note("No trigger record: the background's triggers have not been matched to glitches."))
    else:
        items.extend([_table(by_class), _note(f"The {TAIL_ROWS} loudest background triggers:"), _table(loudest)])
    return pn.Column(*items)


# -------------------------------------------------------------------- Timing

def _format(value) -> str:
    return f"{value:.4g}" if isinstance(value, (int, float)) and not isinstance(value, bool) else str(value)


def timing_tables(validator):
    """A validator's timing as rows of (quantity, statistic, value, units) and its context; None without a record."""
    record = getattr(validator, "timing", None)
    rows = []
    if record is not None:
        if record.get("throughput"):
            for statistic, value in record["throughput"]["summary"].items():
                rows.append({"Quantity": "Throughput (batch)", "Statistic": statistic, "Value": value,
                             "Units": "compute-seconds per second of strain"})
        if record.get("latency"):
            wait = record["latency"]["algorithmic_wait_seconds"]
            for statistic, value in record["latency"]["compute"].items():
                rows.append({"Quantity": "Data-to-trigger latency (streaming)", "Statistic": statistic,
                             "Value": value + wait, "Units": f"s (compute + {wait:g} s algorithmic wait)"})
        if record.get("abba"):
            abba = record["abba"]
            rows.append({"Quantity": "A-B-B-A", "Statistic": "median B / median A", "Value": abba["ratio_b_to_a"],
                         "Units": f"{len(abba['order']) // 4} rounds; distinguishable: {abba['distinguishable']}"})
    cost = getattr(validator, "training_cost", None)
    if cost is not None:
        rows.append({"Quantity": f"Training cost ({cost['kind']})", "Statistic": "total", "Value": cost["value"],
                     "Units": cost["units"] + (f"; source: {cost['source']}" if cost.get("source") else "")
                     + (f"; on {cost['hardware']}" if cost.get("hardware") else "")})
    return pd.DataFrame(rows) if rows else None


def _context(record: Dict) -> str:
    """The machine, thread count and host load a timing was taken at."""
    parts = [f"**Machine:** {record['machine'].get('hardware', 'unknown')}", f"**Threads:** {record['threads']}"]
    for name in ("throughput", "latency", "abba"):
        part = record.get(name)
        if part and part.get("host_load"):
            load = part["host_load"]
            busy = {True: "busy (flagged)", False: "quiet", None: "unmeasured"}[load.get("loaded")]
            parts.append(f"**Host load ({name}):** {load.get('start')} → {load.get('end')} on "
                         f"{load.get('cpu_count')} cores, {busy}")
    return "  \n".join(parts)


def generate_timing_tab(validators: Sequence):
    """The Timing tab (module docstring): one table per validator; timings on different hardware are never divided."""
    import panel as pn

    items = [_note("### Throughput and latency\nMedian and tail on the machine named, at the thread count given; "
                   "no ratio is taken across different hardware.")]
    for index, v in enumerate(validators):
        table = timing_tables(v)
        items.append(_note(f"#### {_title(v, index)}"))
        if table is None:
            items.append(_note("Not timed, and no training cost recorded."))
            continue
        if getattr(v, "timing", None) is not None:
            items.append(_note(_context(v.timing)))
        else:
            items.append(_note("Not timed."))
        items.append(_table(table.assign(Value=[_format(x) for x in table["Value"]])))
    return pn.Column(*items)


__all__ = ["generate_glitch_tab", "generate_timing_tab", "glitch_rate_plot", "tail_tables", "timing_tables",
           "twin_plot"]
