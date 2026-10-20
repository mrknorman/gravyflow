"""
The 3D volume view: the sensitive-volume set's injections in space, found or missed.

Each injection is drawn at its sky direction times its distance (physical, or the
chirp distance the reach is measured in), found or missed at a false-alarm rate
chosen with a slider (found: ``score >= threshold``, as everywhere in `metrics`),
with a sphere at that rate's sensitive distance and the outline of the reach. Points
can be coloured by found state or by chirp mass. Bokeh has no 3D glyphs, so the
view is an orthographic projection computed in the page (azimuth and elevation
sliders): the dashboard stays one self-contained file.

It reads the record only: the volume set's sky positions, distances, masses and
scores, and the sensitive volume curve's thresholds and distances. A record without
sky positions (``right_ascension`` and ``declination``, radians) has no view.
"""
from typing import Dict, Optional

import numpy as np
from bokeh.layouts import column, row
from bokeh.models import (ColumnDataSource, CustomJS, CustomJSTickFormatter, Div, RadioButtonGroup, Range1d,
                          Slider)
from bokeh.palettes import Viridis256
from bokeh.plotting import figure

#: The record's sky-position keys in ``volume_data`` (radians).
SKY_KEYS = ("right_ascension", "declination")

FOUND_COLOR = "#4dd36b"
MISSED_COLOR = "#8c8c8c"
DISTANCE_MODES = ("Physical distance", "Scaled (chirp) distance")
COLOR_MODES = ("Found / missed", "Chirp mass")

# The sphere's wireframe: parallels and meridians of the unit sphere.
_WIRE_STEPS = np.linspace(0.0, 2.0 * np.pi, 73)


def _unit_wireframe():
    lines = []
    for dec in np.radians([-60, -30, 0, 30, 60]):
        lines.append((np.cos(dec) * np.cos(_WIRE_STEPS), np.cos(dec) * np.sin(_WIRE_STEPS),
                      np.full_like(_WIRE_STEPS, np.sin(dec))))
    half = np.linspace(-np.pi / 2, np.pi / 2, 37)
    for ra in np.radians(np.arange(0, 180, 30)):
        lines.append((np.cos(half) * np.cos(ra), np.cos(half) * np.sin(ra), np.sin(half)))
        lines.append((-np.cos(half) * np.cos(ra), -np.cos(half) * np.sin(ra), np.sin(half)))
    return [[a.tolist() for a in line] for line in lines]


def volume_3d_levels(validator) -> Optional[Dict[str, np.ndarray]]:
    """
    What the view reads at each of the record's false-alarm rates: ``far_hz``, the
    background's ``threshold`` (NaN where it cannot resolve the rate), the
    ``sensitive_distance_mpc`` within the reach and the ``reach_mpc`` (chirp distance),
    and the ``found`` and ``missed`` counts. None without a volume set with sky positions.
    """
    data = getattr(validator, "volume_data", None)
    if data is None or any(key not in data for key in SKY_KEYS):
        return None
    curve = validator.sensitive_volume_curve()
    if curve is None:
        return None
    scores = np.asarray(data["scores"], dtype=np.float64)
    thresholds = np.asarray(curve.threshold, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        found = np.array([int(np.sum(scores >= t)) for t in thresholds])
    return dict(far_hz=np.asarray(curve.far_hz, dtype=np.float64), threshold=thresholds,
                sensitive_distance_mpc=np.asarray(curve.distance_within_reach_mpc, dtype=np.float64),
                reach_mpc=float(curve.reach_mpc), found=found, missed=scores.size - found)


_PROJECT = """
// A dict argument arrives as a Map.
const levels = levels_arg instanceof Map ? Object.fromEntries(levels_arg) : levels_arg;
const az = azimuth.value * Math.PI / 180, el = elevation.value * Math.PI / 180;
const ca = Math.cos(az), sa = Math.sin(az), ce = Math.cos(el), se = Math.sin(el);
const proj = (x, y, z) => [-x * sa + y * ca, z * ce - (x * ca + y * sa) * se];
const i = level.value;
const t = levels.threshold[i];
const scaled = distance_mode.active == 1;
const by_mass = color_mode.active == 1;
const d = points.data;
const n = d.score.length;
const px = new Array(n), py = new Array(n), color = new Array(n), alpha = new Array(n);
for (let k = 0; k < n; k++) {
    const r = scaled ? d.chirp_distance[k] : d.distance[k];
    const p = proj(r * d.ux[k], r * d.uy[k], r * d.uz[k]);
    px[k] = p[0]; py[k] = p[1];
    const found = Number.isFinite(t) && d.score[k] >= t;  // an unresolved rate (null) finds nothing
    color[k] = by_mass ? d.mass_color[k] : (found ? found_color : missed_color);
    alpha[k] = found ? 0.9 : (by_mass ? 0.15 : 0.35);
}
points.data = Object.assign({}, d, {px: px, py: py, color: color, alpha: alpha});
const radius = levels.sensitive_distance_mpc[i];
const xs = [], ys = [];
if (Number.isFinite(radius)) {
    for (const line of wire) {
        const lx = [], ly = [];
        for (let k = 0; k < line[0].length; k++) {
            const p = proj(radius * line[0][k], radius * line[1][k], radius * line[2][k]);
            lx.push(p[0]); ly.push(p[1]);
        }
        xs.push(lx); ys.push(ly);
    }
}
sphere.data = {xs: xs, ys: ys};
outline.data = {x: [0], y: [0], radius: [Number.isFinite(radius) ? radius : 0]};
const far = levels.far_hz[i];
statement.text = "<b>FAR " + far.toExponential(1) + " Hz (" + (far * seconds_per_year).toExponential(1)
    + " per year)</b>: " + (Number.isFinite(t)
        ? levels.found[i] + " found, " + levels.missed[i] + " missed; sensitive distance "
          + radius.toFixed(0) + " Mpc"
        : "the background cannot resolve this rate") + "; reach " + levels.reach_mpc.toFixed(0)
    + " Mpc (the sphere and the reach in chirp distance)";
"""


def generate_volume_3d_plot(validator, width: int = 1000, height: int = 900):
    """
    The volume set's injections in 3D: sky direction times distance, found or missed
    at the false-alarm rate a slider chooses among the record's levels, with a sphere
    (wireframe and outline) at that rate's sensitive distance and a dashed outline at
    the reach (both in chirp distance), coloured by found state or by chirp mass, at
    physical or scaled (chirp) distance, turned by azimuth and elevation sliders.

    Returns:
        A column of the controls, the figure and the statement of the level shown,
        or None without a volume set with sky positions (`SKY_KEYS`).
    """
    from gravyflow.src.dataset.features.injection import chirp_mass
    from .plotting import _initial_far
    from .volume import SECONDS_PER_YEAR

    levels = volume_3d_levels(validator)
    if levels is None:
        return None
    data = validator.volume_data
    ra, dec = (np.asarray(data[key], dtype=np.float64) for key in SKY_KEYS)
    mass = chirp_mass(np.asarray(data["mass1"], dtype=np.float64), np.asarray(data["mass2"], dtype=np.float64))
    span = np.ptp(mass) or 1.0
    mass_color = [Viridis256[int(k)] for k in np.clip((mass - mass.min()) / span * 255, 0, 255)]
    distance = np.asarray(data["distances"], dtype=np.float64)
    chirp_dist = np.asarray(data["chirp_distances"], dtype=np.float64)
    n = distance.size
    points = ColumnDataSource(data=dict(
        ux=np.cos(dec) * np.cos(ra), uy=np.cos(dec) * np.sin(ra), uz=np.sin(dec),
        distance=distance, chirp_distance=chirp_dist, score=np.asarray(data["scores"], dtype=np.float64),
        chirp_mass=mass, mass_color=mass_color,
        px=np.zeros(n), py=np.zeros(n), color=[MISSED_COLOR] * n, alpha=np.full(n, 0.35)))
    sphere = ColumnDataSource(data=dict(xs=[], ys=[]))
    outline = ColumnDataSource(data=dict(x=[0], y=[0], radius=[0]))
    extent = 1.05 * max(np.nanmax(distance), np.nanmax(chirp_dist), levels["reach_mpc"])

    p = figure(width=width, height=height, match_aspect=True, x_range=Range1d(-extent, extent),
               y_range=Range1d(-extent, extent), tools="pan,box_zoom,wheel_zoom,reset",
               title=f"Volume set in 3D: {validator.name}",
               tooltips=[("Distance (Mpc)", "@distance{0,0}"), ("Chirp distance (Mpc)", "@chirp_distance{0,0}"),
                         ("Chirp mass (M☉)", "@chirp_mass{0.00}"), ("Score", "@score{0.000}")])
    p.axis.visible = False
    p.grid.visible = False
    p.scatter("px", "py", source=points, size=5, color="color", alpha="alpha", legend_label="Injections")
    p.multi_line("xs", "ys", source=sphere, line_color="#e0e0e0", line_alpha=0.25,
                 legend_label="Sensitive distance (sphere)")
    p.circle(x="x", y="y", radius="radius", source=outline, fill_alpha=0, line_color="#e0e0e0", line_width=2,
             legend_label="Sensitive distance (sphere)")
    p.circle(x=0, y=0, radius=levels["reach_mpc"], fill_alpha=0, line_color="#ffb000", line_dash="dashed",
             line_width=2, legend_label="Reach (outline)")
    p.legend.click_policy = "hide"
    p.legend.location = "top_left"

    keys = [f"{far:.1e}" for far in levels["far_hz"]]
    initial = _initial_far(keys)
    level = Slider(start=0, end=max(len(keys) - 1, 1), value=initial, step=1, title="False Alarm Rate (Hz)",
                   disabled=len(keys) < 2, width=420,
                   format=CustomJSTickFormatter(args=dict(labels=keys), code="return labels[Math.round(tick)] || '';"))
    azimuth = Slider(start=-180, end=180, value=30, step=1, title="Azimuth (°)", width=260)
    elevation = Slider(start=-90, end=90, value=20, step=1, title="Elevation (°)", width=260)
    distance_mode = RadioButtonGroup(labels=list(DISTANCE_MODES), active=0)
    color_mode = RadioButtonGroup(labels=list(COLOR_MODES), active=0)
    statement = Div(text="", width=width)
    js_levels = {key: (value.tolist() if isinstance(value, np.ndarray) else value) for key, value in levels.items()}
    # JSON has no NaN: an unresolved rate's threshold and distance travel as null, and read as not finite.
    for key in ("threshold", "sensitive_distance_mpc"):
        js_levels[key] = [None if not np.isfinite(v) else v for v in levels[key]]
    callback = CustomJS(args=dict(points=points, sphere=sphere, outline=outline, levels_arg=js_levels,
                                  wire=_unit_wireframe(), level=level, azimuth=azimuth, elevation=elevation,
                                  distance_mode=distance_mode, color_mode=color_mode, statement=statement,
                                  found_color=FOUND_COLOR, missed_color=MISSED_COLOR,
                                  seconds_per_year=SECONDS_PER_YEAR), code=_PROJECT)
    for control in (level, azimuth, elevation):
        control.js_on_change("value", callback)
    for control in (distance_mode, color_mode):
        control.js_on_change("active", callback)
    _initial_view(points, sphere, outline, levels, initial, azimuth.value, elevation.value)
    _initial_statement(statement, levels, initial, SECONDS_PER_YEAR)
    return column(row(level, azimuth, elevation), row(distance_mode, color_mode), p, statement)


def _project(x, y, z, azimuth_deg, elevation_deg):
    az, el = np.radians(azimuth_deg), np.radians(elevation_deg)
    return -x * np.sin(az) + y * np.cos(az), z * np.cos(el) - (x * np.cos(az) + y * np.sin(az)) * np.sin(el)


def _initial_view(points, sphere, outline, levels, i, azimuth, elevation) -> None:
    """The view the page opens on, as the page's own projection draws it (physical distance, found / missed)."""
    d = points.data
    r = np.asarray(d["distance"])
    px, py = _project(r * d["ux"], r * d["uy"], r * d["uz"], azimuth, elevation)
    with np.errstate(invalid="ignore"):
        found = np.asarray(d["score"]) >= levels["threshold"][i]
    points.data = {**d, "px": px, "py": py, "color": np.where(found, FOUND_COLOR, MISSED_COLOR).tolist(),
                   "alpha": np.where(found, 0.9, 0.35)}
    radius = levels["sensitive_distance_mpc"][i]
    if np.isfinite(radius):
        lines = [_project(*(radius * np.asarray(c) for c in line), azimuth, elevation) for line in _unit_wireframe()]
        sphere.data = dict(xs=[lx.tolist() for lx, _ in lines], ys=[ly.tolist() for _, ly in lines])
    outline.data = dict(x=[0], y=[0], radius=[float(radius) if np.isfinite(radius) else 0.0])


def _initial_statement(statement, levels, i, seconds_per_year) -> None:
    far = levels["far_hz"][i]
    if np.isfinite(levels["threshold"][i]):
        what = (f"{levels['found'][i]} found, {levels['missed'][i]} missed; sensitive distance "
                f"{levels['sensitive_distance_mpc'][i]:.0f} Mpc")
    else:
        what = "the background cannot resolve this rate"
    statement.text = (f"<b>FAR {far:.1e} Hz ({far * seconds_per_year:.1e} per year)</b>: {what}; reach "
                      f"{levels['reach_mpc']:.0f} Mpc (the sphere and the reach in chirp distance)")
