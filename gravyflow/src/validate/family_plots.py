"""
The dashboard's views of the metric family (`family`): the Summary tab's table of the
headline numbers at each false-alarm-rate level, the detection ROC, the false-alarm
calibration, VT and the binned volume, and the real events' counts per level. Every
number carries its 1 and 2 sigma intervals; a point the background cannot quote is
marked, never dropped.
"""

from __future__ import annotations

from itertools import cycle
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from bokeh.layouts import column
from bokeh.models import ColumnDataSource, CustomJS, Div, Select, Span
from bokeh.palettes import Bright
from bokeh.plotting import figure

from . import family
from .metrics import MIN_BACKGROUND_EVENTS
from .plotting import _band_alpha, _initial_far, _legend_beside, _sigma_label, _title
from .volume import SECONDS_PER_MONTH, SECONDS_PER_YEAR, sensitive_distance, sensitive_vt

#: What a level the background does not reach shows in place of a number.
NOT_REACHED = "not reached"


def _with_interval(value, lower, upper, sigmas, fmt: str = ".3g", scale: float = 1.0) -> str:
    """A number with its interval at each level: ``v [a–b (1σ); c–d (2σ)]``; "–" for NaN."""
    if value is None or not np.isfinite(value):
        return "–"
    parts = [f"{lower[j] * scale:{fmt}}–{upper[j] * scale:{fmt}} ({_sigma_label(s)})"
             for j, s in enumerate(sigmas) if np.isfinite(lower[j]) and np.isfinite(upper[j])]
    return f"{value * scale:{fmt}}" + (f" [{'; '.join(parts)}]" if parts else "")


def _per_year(far_hz) -> np.ndarray:
    return np.asarray(far_hz, dtype=np.float64) * SECONDS_PER_YEAR


# ------------------------------------------------------------------- Summary

def summary_rows(validator, matched_filter=None) -> List[Dict[str, str]]:
    """The Summary tab's rows: one per false-alarm-rate level (`family.FAR_LEVELS_HZ`), its headline numbers."""
    levels = list(family.FAR_LEVELS_HZ)
    background = validator.background()
    sigmas = validator.config.interval_sigmas
    targets = [background.threshold_for_far(far) for far in levels]
    volume = validator.sensitive_volume_curve(levels)
    vt = None if volume is None else sensitive_vt(volume, validator.live_time_seconds)
    efficiency = validator.efficiency_curves(levels) if validator.efficiency_data is not None else {}
    counts = validator.recovered_counts(levels)
    ratio = validator.matched_filter_ratio(matched_filter, levels) if matched_filter is not None else None
    timing = validator.timing_accuracy(levels)
    rows = []
    for i, (far, name, target) in enumerate(zip(levels, family.FAR_LEVEL_NAMES, targets)):
        if not target.resolved:
            reached = NOT_REACHED
        elif not target.quotable:
            reached = f"not quoted: {target.independent_events} independent background events (< {MIN_BACKGROUND_EVENTS})"
        else:
            reached = f"quoted ({target.independent_events} independent background events)"
        row = {"Level": name, "Background": reached}
        if volume is not None:
            row["Volume (Gpc³)"] = _with_interval(volume.volume_within_reach_gpc3[i], volume.joint_lower_gpc3[:, i],
                                                  volume.joint_upper_gpc3[:, i], sigmas)
            d = sensitive_distance(np.clip([volume.volume_within_reach_gpc3[i], *volume.joint_lower_gpc3[:, i],
                                            *volume.joint_upper_gpc3[:, i]], 0.0, None))
            k = len(sigmas)
            row["Distance (Mpc)"] = _with_interval(d[0], d[1:1 + k], d[1 + k:], sigmas, ".4g")
            row["VT (Gpc³ yr)"] = _with_interval(vt.vt_gpc3_yr[i], vt.lower_gpc3_yr[:, i], vt.upper_gpc3_yr[:, i], sigmas)
        if efficiency:
            fit = efficiency[float(far)].fit
            row["50% SNR"] = _with_interval(fit.snr_50, fit.lower[:, 0], fit.upper[:, 0], sigmas)
            row["90% SNR"] = _with_interval(fit.snr_90, fit.lower[:, 1], fit.upper[:, 1], sigmas)
        for set_name, count in counts.items():
            row[f"Found ({set_name})"] = (
                "–" if not target.resolved else
                f"{_with_interval(count.found[i], count.lower[:, i], count.upper[:, i], sigmas)} of {count.total} "
                f"(noise alone: {count.by_chance[i]:.2g})")
        if matched_filter is not None:
            row["Ratio to matched filter"] = ("no volume set" if ratio is None else
                                              _with_interval(ratio.ratio[i], ratio.lower[:, i], ratio.upper[:, i], sigmas))
        if timing is not None:
            row["|Δt| median (ms)"] = _with_interval(timing.median_abs_seconds[i], timing.median_abs_lower[:, i],
                                                     timing.median_abs_upper[:, i], sigmas, ".3g", 1e3)
            row["|Δt| 90th percentile (ms)"] = _with_interval(timing.p90_abs_seconds[i], timing.p90_abs_lower[:, i],
                                                              timing.p90_abs_upper[:, i], sigmas, ".3g", 1e3)
        rows.append(row)
    return rows


def generate_summary_tab(validators: list, matched_filter=None, width: int = 1450):
    """
    The Summary tab: per model, a header naming it, how it was measured, the live time
    searched and how deep its background reaches, then the headline numbers at each
    false-alarm-rate level (`summary_rows`), each with its 1σ and 2σ intervals.
    """
    import pandas as pd
    import panel as pn

    items = []
    for index, validator in enumerate(validators):
        if validator.far_scores is None:
            continue
        background = validator.background()
        resolve, quote = family.far_floor_hz(background)
        header = (f"### {_title(validator, index)}\n\n{validator.mode_summary()}  \n"
                  f"Live time searched: {validator.live_time_seconds:,.0f} s "
                  f"({validator.live_time_seconds / SECONDS_PER_YEAR:.3g} yr); background "
                  f"{background.livetime_seconds:,.0f} s over {background.data_reuse} slide(s). "
                  f"Deepest FAR resolved: {resolve * SECONDS_PER_YEAR:.3g} per year; deepest quoted "
                  f"(≥ {MIN_BACKGROUND_EVENTS} independent background events): {quote * SECONDS_PER_YEAR:.3g} per year. "
                  "Intervals are 1σ and 2σ.")
        if matched_filter is None:
            header += " No matched-filter result was supplied, so there is no ratio to it."
        rows = summary_rows(validator, matched_filter)
        items.append(pn.pane.Markdown(header, sizing_mode="stretch_width"))
        items.append(pn.widgets.DataFrame(pd.DataFrame(rows), width=width, height=40 * len(rows) + 50,
                                          auto_edit=False, show_index=False))
    return pn.Column(*items) if items else pn.pane.Markdown("**No results to summarise.**")


# ------------------------------------------------------------------- ROC

POPULATION = "population (volume set)"


def generate_detection_roc_plot(validators: list, colors: List[str] = Bright[7], width: int = 1100,
                                height: int = 600):
    """
    Detection probability against false-alarm rate (per year), per SNR band and for the
    population (`family.detection_roc`, `family.population_detection_probability`), with
    its joint 1σ and 2σ bands; points the background cannot quote are marked. A selector
    picks the band. Returns ``(figure, select)``.
    """
    p = figure(width=width, height=height, x_axis_type="log", x_axis_label="False Alarm Rate (per year)",
               y_axis_label="Detection probability (chance subtracted)", y_range=(-0.05, 1.05),
               tools="pan,box_zoom,wheel_zoom,reset", title="Detection probability against false-alarm rate",
               tooltips=[("FAR (per year)", "@x{0.00e}"), ("Detection probability", "@y{0.000}"),
                         ("1σ", "@lower_0{0.000} – @upper_0{0.000}"), ("2σ", "@lower_1{0.000} – @upper_1{0.000}")])
    all_data, sources, options = {}, {}, []
    for index, validator in enumerate(validators):
        roc = validator.detection_roc()
        if roc is None:
            continue
        name = _title(validator, index)
        all_data[name] = {}
        ok = np.isfinite(roc.threshold)
        for j, label in enumerate(roc.labels):
            all_data[name][label] = dict(
                x=_per_year(roc.far_hz[ok]), y=roc.probability[ok, j], quotable=roc.quotable[ok].astype(int),
                **{f"lower_{k}": roc.lower[k, ok, j] for k in range(len(roc.sigmas))},
                **{f"upper_{k}": roc.upper[k, ok, j] for k in range(len(roc.sigmas))})
        curve = validator.sensitive_volume_curve()
        if curve is not None:
            population = family.population_detection_probability(curve)
            ok = curve.resolved
            all_data[name][POPULATION] = dict(
                x=_per_year(population.far_hz[ok]), y=population.probability[ok],
                quotable=population.quotable[ok].astype(int),
                **{f"lower_{k}": population.lower[k, ok] for k in range(len(population.sigmas))},
                **{f"upper_{k}": population.upper[k, ok] for k in range(len(population.sigmas))})
        options += [key for key in all_data[name] if key not in options]
    if not options:
        return p, Select(title="Injections:", options=[])
    initial = "all SNRs" if "all SNRs" in options else options[0]
    for index, (color, validator) in enumerate(zip(cycle(colors), validators)):
        name = _title(validator, index)
        data = all_data.get(name, {}).get(initial)
        if data is None:
            continue
        sources[name] = ColumnDataSource(data=data)
        sigmas = validator.config.interval_sigmas
        for k, sigma in reversed(list(enumerate(sigmas))):
            p.varea(x="x", y1=f"lower_{k}", y2=f"upper_{k}", source=sources[name], fill_color=color,
                    fill_alpha=_band_alpha(sigma), legend_label=f"{name} ({_sigma_label(sigma)} joint)")
        p.line("x", "y", source=sources[name], line_color=color, line_width=2, legend_label=name)
        p.scatter("x", "y", source=sources[name], color=color, size=5, legend_label=name)
    if p.renderers:
        _legend_beside(p)
    select = Select(title="Injections:", value=initial, options=options)
    select.js_on_change("value", CustomJS(args=dict(sources=sources, all_data=all_data), code="""
        for (const name in sources) {
            const data = all_data[name][cb_obj.value];
            if (data) { sources[name].data = data; }
        }
    """))
    return p, select


def generate_far_calibration_plot(validators: list, colors: List[str] = Bright[7], width: int = 1100,
                                  height: int = 500):
    """
    False-alarm calibration (`family.far_calibration`): the rate measured on held-out
    background over the rate claimed, against the claimed rate (per year), with the
    measured rate's 1σ and 2σ Poisson intervals; 1 is perfect calibration.
    """
    p = figure(width=width, height=height, x_axis_type="log", x_axis_label="Claimed False Alarm Rate (per year)",
               y_axis_label="Held-out rate / claimed rate", tools="pan,box_zoom,wheel_zoom,reset",
               title="False-alarm calibration on held-out background (a half that shares no data)")
    p.add_layout(Span(location=1.0, dimension="width", line_dash="dashed", line_color="gray"))
    notes = []
    for index, (color, validator) in enumerate(zip(cycle(colors), validators)):
        name = _title(validator, index)
        calibration = validator.far_calibration()
        if calibration is None:
            notes.append(f"{name}: the background has fewer than two blocks of data, so none can be held out.")
            continue
        ok = np.isfinite(calibration.ratio)
        x = _per_year(calibration.claimed_hz[ok])
        claimed = calibration.claimed_hz[ok]
        for k, sigma in reversed(list(enumerate(calibration.sigmas))):
            p.segment(x0=x, x1=x, y0=calibration.lower_hz[k, ok] / claimed, y1=calibration.upper_hz[k, ok] / claimed,
                      line_color=color, line_width=6 if k == 0 else 2, line_alpha=0.6,
                      legend_label=f"{name} ({_sigma_label(sigma)})")
        p.scatter(x, calibration.ratio[ok], color=color, size=7, legend_label=name)
        thin = ~calibration.quotable[ok]
        if thin.any():
            p.scatter(x[thin], calibration.ratio[ok][thin], marker="x", size=10, color=color,
                      legend_label=f"{name} (not quoted: < {MIN_BACKGROUND_EVENTS} independent events)")
    if p.renderers:
        _legend_beside(p)
    return column(p, Div(text="<br>".join(notes), width=width)) if notes else column(p)


# ------------------------------------------------------------------- Volume

def generate_vt_plot(validators: list, colors: List[str] = Bright[7], width: int = 1450, height: int = 500):
    """VT (Gpc³ yr, `volume.sensitive_vt`) against false-alarm rate (per month), with its joint 1σ and 2σ bands."""
    p = figure(width=width, height=height, x_axis_type="log", x_axis_label="False Alarm Rate (per month)",
               y_axis_label="VT within the reach (Gpc³ yr)", tools="pan,box_zoom,wheel_zoom,reset",
               title="Volume times live time searched",
               tooltips=[("FAR (per month)", "@x{0.00e}"), ("VT", "@y{0.000}"),
                         ("1σ", "@lower_0{0.000} – @upper_0{0.000}"), ("2σ", "@lower_1{0.000} – @upper_1{0.000}")])
    for index, (color, validator) in enumerate(zip(cycle(colors), validators)):
        vt = validator.sensitive_vt()
        if vt is None:
            continue
        name = _title(validator, index)
        ok = np.isfinite(vt.vt_gpc3_yr)
        source = ColumnDataSource(data=dict(
            x=vt.far_hz[ok] * SECONDS_PER_MONTH, y=vt.vt_gpc3_yr[ok],
            **{f"lower_{k}": np.clip(vt.lower_gpc3_yr[k, ok], 0, None) for k in range(len(vt.sigmas))},
            **{f"upper_{k}": vt.upper_gpc3_yr[k, ok] for k in range(len(vt.sigmas))}))
        for k, sigma in reversed(list(enumerate(vt.sigmas))):
            p.varea(x="x", y1=f"lower_{k}", y2=f"upper_{k}", source=source, fill_color=color,
                    fill_alpha=_band_alpha(sigma), legend_label=f"{name} ({_sigma_label(sigma)} joint)")
        p.line("x", "y", source=source, line_color=color, line_width=2,
               legend_label=f"{name} (live time {vt.live_time_years:.3g} yr)")
        thin = ~vt.quotable[ok]
        if thin.any():
            p.scatter(vt.far_hz[ok][thin] * SECONDS_PER_MONTH, vt.vt_gpc3_yr[ok][thin], marker="square", size=12,
                      fill_alpha=0, line_color=color,
                      legend_label=f"{name} (not quoted: < {MIN_BACKGROUND_EVENTS} independent background events)")
    if p.renderers:
        _legend_beside(p)
    return p


def generate_binned_volume_plot(validators: list, colors: List[str] = Bright[7], width: int = 1450,
                                height: int = 450):
    """
    The sensitive volume within the reach in bins of chirp mass, mass ratio and (where
    recorded) effective spin (`Validator.binned_volumes`), one figure per parameter, each
    bin's point with its joint 1σ and 2σ error bars, at the false-alarm rate a selector picks.
    """
    figures, all_data, sources, far_keys = {}, {}, {}, []
    for index, (color, validator) in enumerate(zip(cycle(colors), validators)):
        name = _title(validator, index)
        for parameter, binned in validator.binned_volumes().items():
            if parameter not in figures:
                figures[parameter] = figure(width=width, height=height, x_axis_label=parameter,
                                            y_axis_label="Sensitive Volume within the reach (Gpc³)",
                                            tools="pan,box_zoom,wheel_zoom,reset",
                                            title=f"Sensitive volume by {parameter}")
            key = f"{name}|{parameter}"
            centres = 0.5 * (binned.edges[1:] + binned.edges[:-1])
            volume, lower, upper = binned.volume_gpc3, binned.lower_gpc3, binned.upper_gpc3
            all_data[key] = {}
            for i, far in enumerate(binned.far_hz):
                if not np.any(np.isfinite(volume[i])):
                    continue
                far_key = f"{far:.1e}"
                all_data[key][far_key] = dict(x=centres, y=volume[i], n=binned.total,
                                              **{f"lower_{k}": lower[k, i] for k in range(lower.shape[0])},
                                              **{f"upper_{k}": upper[k, i] for k in range(upper.shape[0])})
                if far_key not in far_keys:
                    far_keys.append(far_key)
            if not all_data[key]:
                continue
            sources[key] = ColumnDataSource(data=next(iter(all_data[key].values())))
            p = figures[parameter]
            for k, sigma in reversed(list(enumerate(validator.config.interval_sigmas))):
                p.segment(x0="x", x1="x", y0=f"lower_{k}", y1=f"upper_{k}", source=sources[key], line_color=color,
                          line_width=6 if k == 0 else 2, line_alpha=0.6, legend_label=f"{name} ({_sigma_label(sigma)} joint)")
            p.scatter("x", "y", source=sources[key], color=color, size=8, legend_label=name)
    if not figures:
        return column(Div(text="<p>No volume set: no binned volume.</p>"))
    initial = far_keys[_initial_far(far_keys)] if far_keys else None
    for key, source in sources.items():
        if initial in all_data[key]:
            source.data = all_data[key][initial]
    for p in figures.values():
        if p.renderers:
            _legend_beside(p)
    select = Select(title="False Alarm Rate (Hz) of the binned volume:", value=initial, options=far_keys)
    select.js_on_change("value", CustomJS(args=dict(sources=sources, all_data=all_data), code="""
        for (const key in sources) {
            const data = all_data[key][cb_obj.value];
            if (data) { sources[key].data = data; }
        }
    """))
    return column(select, *figures.values())


# --------------------------------------------------------------- Real events

def generate_real_event_levels(validator, training_periods: Optional[Sequence[Tuple[float, float]]] = None):
    """
    The real events' counts at each false-alarm-rate level (`Validator.real_event_summary`),
    and each event's statistic, FAR with its 1σ and 2σ intervals, and whether it lies in the
    model's training period (given for an as-published model).
    """
    import pandas as pd
    import panel as pn

    summary = validator.real_event_summary(training_periods=training_periods)
    if summary is None:
        return pn.pane.Markdown("**No scored real events.**")
    flagged = training_periods is not None
    levels = pd.DataFrame([{
        "Level": name,
        "Threshold": f"{t:.4g}" if np.isfinite(t) else NOT_REACHED,
        "Events found": int(found) if ok else "–",
        **({"Found outside the training period": int(outside) if ok else "–"} if flagged else {}),
    } for name, t, ok, found, outside in zip(family.FAR_LEVEL_NAMES, summary.threshold, summary.resolved,
                                             summary.found, summary.found_outside_training)])
    rows = pd.DataFrame([{
        "Name": row["name"], "GPS": f"{row['gps']:.1f}", "Statistic": f"{row['score']:.4g}",
        "FAR (per year)": _with_interval(row["far_hz"], row["far_interval_hz"][:, 0], row["far_interval_hz"][:, 1],
                                         summary.sigmas, ".3g", SECONDS_PER_YEAR)
                          + ("" if row["background_events"] >= MIN_BACKGROUND_EVENTS else
                             f" (not quoted: {row['background_events']} independent background events)"),
        "In training period": ("yes" if row["in_training"] else "no") if flagged else "no period given",
    } for row in summary.rows])
    note = ("Events inside the model's training period are flagged: an as-published model may have trained on them."
            if flagged else "No training period was given, so no event is flagged.")
    return pn.Column(
        pn.pane.Markdown(f"### Real events at each false-alarm-rate level\n\n{note}", sizing_mode="stretch_width"),
        pn.widgets.DataFrame(levels, height=40 * len(levels) + 50, auto_edit=False, show_index=False),
        pn.widgets.DataFrame(rows, height=min(600, 30 * len(rows) + 50), auto_edit=False, show_index=False))


__all__ = [
    "generate_binned_volume_plot",
    "generate_detection_roc_plot",
    "generate_far_calibration_plot",
    "generate_real_event_levels",
    "generate_summary_tab",
    "generate_vt_plot",
    "summary_rows",
]
