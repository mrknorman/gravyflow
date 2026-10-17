
import pytest

import numpy as np
import gravyflow as gf
from bokeh.models import Plot

@pytest.mark.global_rng
def test_generate_strain_plot():
    """Verify that calling with dummy data returns a valid Bokeh object."""
    
    # Create dummy data
    # Dictionary of {label: data}
    # Data should be (Batch, IFOs, Time) or similar?
    # generate_strain_plot expects a dictionary where values are arrays.
    # It usually plots the first element if batch?
    # Let's check signature or usage.
    # In test_dataset.py: gf.generate_strain_plot({"Whitened Onsource": data[0]}, title="Noise example")
    # data[0] is likely (IFOs, Time) or (Time,)?
    
    # Let's assume (Time,) or (IFOs, Time).
    # If it handles multi-channel, it might expect specific shape.
    
    data = np.random.randn(100).astype(np.float32)
    input_dict = {"Test Signal": data}
    
    plot = gf.generate_strain_plot(input_dict, title="Test Plot")
    
    assert isinstance(plot, Plot)
    assert plot.title.text == "Test Plot"

@pytest.mark.global_rng
def test_generate_spectrogram():
    """Verify that calling with dummy data returns a valid Bokeh object."""
    
    # generate_spectrogram(data, sample_rate_hertz, ...)
    # data: (Time,)
    
    data = np.random.randn(1000).astype(np.float32)
    sample_rate = 100.0
    
    plot = gf.generate_spectrogram(
        data,
        sample_rate_hertz=sample_rate
    )

    assert isinstance(plot, Plot)


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

