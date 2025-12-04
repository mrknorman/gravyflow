
import pytest
import numpy as np
import gravyflow as gf
from bokeh.models import Plot

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

