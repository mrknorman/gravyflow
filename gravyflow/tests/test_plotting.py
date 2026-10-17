import numpy as np
import pytest
import gravyflow as gf
from bokeh.models import Plot


def test_generate_strain_plot_draws_the_data_against_time():
    """The plot's sources hold the time axis and the strain it was given."""
    rng = np.random.default_rng(0)
    data = rng.standard_normal(100).astype(np.float32)

    plot = gf.generate_strain_plot({"Test Signal": data}, sample_rate_hertz=50.0, title="Test Plot")

    assert isinstance(plot, Plot)
    assert plot.title.text == "Test Plot"
    sources = [r.data_source.data for r in plot.renderers if "y" in r.data_source.data]
    assert len(sources) == 1
    np.testing.assert_array_equal(sources[0]["y"], data)           # exact: the data, copied
    np.testing.assert_array_equal(sources[0]["x"], np.linspace(0.0, 2.0, 100))  # 100 samples at 50 Hz


def test_generate_spectrogram_puts_a_tone_at_its_frequency():
    """A pure tone's power peaks in the row of its frequency, and the axes span the signal."""
    sample_rate, tone, num_fft = 100.0, 20.0, 256
    t = np.arange(1000) / sample_rate
    data = np.sin(2 * np.pi * tone * t).astype(np.float32)

    plot = gf.generate_spectrogram(data, sample_rate_hertz=sample_rate)

    assert isinstance(plot, Plot)
    image = plot.renderers[0].data_source.data["image"][0]
    frequencies = np.linspace(0, sample_rate / 2, num_fft // 2 + 1)[1:]   # the plot drops the DC row
    assert image.shape[0] == frequencies.size
    # Exact: one FFT bin, the finest the transform can say.
    assert abs(frequencies[np.argmax(image.mean(axis=1))] - tone) <= sample_rate / num_fft


def test_the_real_events_report_shows_each_events_p_astro():
    """The event list gives p_astro; the real-events table and parameter box show it."""
    import panel as pn
    from gravyflow.src.validate.plotting import generate_real_events_table

    event = {
        "name": "GW-REPORT", "gps": 1240000000.4, "catalog": "GWTC-3-confident",
        "observing_run": "O3", "mass1": 30.0, "mass2": 25.0, "distance": 400.0,
        "p_astro": 0.97, "event_type": "CONFIDENT", "score": 0.9,
        "far_hz": 1e-3, "far_upper_hz": 2e-3, "whitened_strain": np.zeros((2, 64)),
    }
    layout, _ = generate_real_events_table([event], input_duration_seconds=1.0)
    table = next(item for item in layout if isinstance(item, pn.widgets.DataFrame))
    assert table.value["p(astro)"].tolist() == ["0.97"]
    boxes = [pane.object for row in layout if isinstance(row, pn.Row)
             for pane in row if isinstance(pane, pn.pane.HTML)]
    assert len(boxes) == 1 and "<b>p(astro):</b> 0.97" in boxes[0]

