"""The 3D volume view, built from synthetic records."""
from types import SimpleNamespace

import numpy as np
import pytest
from bokeh.models import CustomJS, MultiLine, RadioButtonGroup, Scatter, Slider

from gravyflow.src.validate.volume import sensitive_distance
from gravyflow.src.validate.volume_3d import (FOUND_COLOR, MISSED_COLOR, generate_volume_3d_plot,
                                              volume_3d_levels)

FARS = np.array([1e-1, 1e-2, 1e-3, 1e-4, 1e-5])
THRESHOLDS = np.array([-1.0, 0.0, 1.0, 2.0, np.nan])  # the last rate the background cannot resolve
REACH = 400.0


def synthetic_validator(n=200, seed=3, sky=True):
    rng = np.random.default_rng(seed)
    mass1, mass2 = rng.uniform(5, 50, n), rng.uniform(5, 50, n)
    chirp = rng.uniform(1, REACH, n)
    data = dict(scores=rng.normal(0, 1.5, n), chirp_distances=chirp, distances=chirp * rng.uniform(1, 10, n),
                snrs=rng.uniform(1, 30, n), mass1=mass1, mass2=mass2, reach_mpc=np.float64(REACH))
    if sky:
        data.update(right_ascension=rng.uniform(0, 2 * np.pi, n), declination=np.arcsin(rng.uniform(-1, 1, n)))
    volumes = np.array([3.0, 1.0, 0.5, 0.1, np.nan]) * 1e-3
    curve = SimpleNamespace(far_hz=FARS, threshold=THRESHOLDS, reach_mpc=REACH,
                            distance_within_reach_mpc=sensitive_distance(volumes))
    return SimpleNamespace(name="tiny", volume_data=data, sensitive_volume_curve=lambda: curve)


def parts(plot):
    controls, modes, figure, statement = plot.children
    level, azimuth, elevation = controls.children
    return level, azimuth, elevation, modes.children, figure, statement


def test_found_and_missed_counts_at_every_level():
    v = synthetic_validator()
    scores = v.volume_data["scores"]
    levels = volume_3d_levels(v)
    expected = [int(np.sum(scores >= t)) if np.isfinite(t) else 0 for t in THRESHOLDS]
    assert levels["found"].tolist() == expected
    assert (levels["found"] + levels["missed"]).tolist() == [scores.size] * FARS.size
    # The view opens on the level nearest 1e-3 Hz, its points coloured by that level's found state.
    level, _, _, _, figure, statement = parts(generate_volume_3d_plot(v))
    points = next(r for r in figure.renderers if isinstance(r.glyph, Scatter)).data_source.data
    assert level.value == 2 and len(points["px"]) == scores.size
    assert points["color"].count(FOUND_COLOR) == expected[2]
    assert points["color"].count(MISSED_COLOR) == scores.size - expected[2]
    assert f"{expected[2]} found, {scores.size - expected[2]} missed" in statement.text
    # The page recomputes found state from the same thresholds and scores.
    callback = level.js_property_callbacks["change:value"][0]
    assert isinstance(callback, CustomJS) and "Number.isFinite(t) && d.score[k] >= t" in callback.code
    assert callback.args["levels_arg"]["found"] == expected


def test_the_sphere_is_at_the_sensitive_distance_and_the_outline_at_the_reach():
    v = synthetic_validator()
    distances = v.sensitive_volume_curve().distance_within_reach_mpc
    level, _, _, _, figure, statement = parts(generate_volume_3d_plot(v))
    circles = [r for r in figure.renderers if type(r.glyph).__name__ == "Circle"]
    sphere_outline = next(r for r in circles if "radius" in r.data_source.data)
    assert sphere_outline.data_source.data["radius"] == [pytest.approx(distances[level.value])]
    reach = [r for r in circles if r is not sphere_outline]
    assert len(reach) == 1 and reach[0].glyph.radius == REACH
    # The wireframe's every point lies on the sphere: its projection never leaves the outline.
    wire = next(r for r in figure.renderers if isinstance(r.glyph, MultiLine)).data_source.data
    radii = np.hypot(np.concatenate(wire["xs"]), np.concatenate(wire["ys"]))
    assert radii.max() == pytest.approx(distances[level.value], rel=1e-9)
    callback = level.js_property_callbacks["change:value"][0]
    assert callback.args["levels_arg"]["sensitive_distance_mpc"][:4] == pytest.approx(distances[:4].tolist())
    assert callback.args["levels_arg"]["sensitive_distance_mpc"][4] is None  # unresolved: no sphere
    labels = {item.label.value for item in figure.legend[0].items}
    assert labels == {"Injections", "Sensitive distance (sphere)", "Reach (outline)"}
    assert figure.legend[0].click_policy == "hide"


def test_the_slider_levels_are_the_records_levels():
    v = synthetic_validator()
    level, azimuth, elevation, modes, _, _ = parts(generate_volume_3d_plot(v))
    assert isinstance(level, Slider) and (level.start, level.end, level.step) == (0, FARS.size - 1, 1)
    assert level.format.args["labels"] == [f"{far:.1e}" for far in FARS]
    callback = level.js_property_callbacks["change:value"][0]
    assert callback.args["levels_arg"]["far_hz"] == FARS.tolist()
    assert [m.labels for m in modes] == [["Physical distance", "Scaled (chirp) distance"],
                                         ["Found / missed", "Chirp mass"]]
    assert all(isinstance(m, RadioButtonGroup) for m in modes)
    for control in (azimuth, elevation):
        assert control.js_property_callbacks["change:value"][0] is callback
    for m in modes:
        assert m.js_property_callbacks["change:active"][0] is callback


def test_no_view_without_sky_positions():
    assert generate_volume_3d_plot(synthetic_validator(sky=False)) is None
    assert generate_volume_3d_plot(SimpleNamespace(name="x", volume_data=None)) is None


def test_the_view_renders_to_a_self_contained_file(tmp_path):
    from bokeh.io import save
    from bokeh.resources import INLINE

    path = tmp_path / "volume_3d.html"
    save(generate_volume_3d_plot(synthetic_validator()), filename=str(path), resources=INLINE, title="3D volume")
    html = path.read_text()
    assert path.stat().st_size > 0 and "<html" in html.lower() and "Bokeh" in html
    assert "cdn.bokeh.org" not in html  # inline resources: one file
