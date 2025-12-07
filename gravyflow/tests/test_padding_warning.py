import pytest
import gravyflow as gf
from warnings import catch_warnings

def test_padding_warning():
    """
    Verify that a warning is issued when padding exceeds half of onsource duration.
    """
    with catch_warnings(record=True) as w:
        # Cause all warnings to always be triggered.
        import warnings
        warnings.simplefilter("always")
        
        # Default padding is back=0.3.
        # Set onsource=0.4. Limit = 0.2.
        # 0.3 > 0.2 -> Should warn.
        
        dataset = gf.Dataset(
            onsource_duration_seconds=0.4,
            crop_duration_seconds=0.5,
            waveform_generators=[
                gf.WNBGenerator()
            ],
            input_variables=[gf.ReturnVariables.INJECTIONS],
            noise_obtainer=gf.NoiseObtainer(noise_type=gf.NoiseType.WHITE)
        )
        
        # Check for the specific warning
        found = False
        for warning in w:
            if "exceeds half of onsource duration" in str(warning.message):
                found = True
                break
        
        assert found, "Expected warning about padding not found!"

def test_no_padding_warning():
    """
    Verify that NO warning is issued when padding is within limits.
    """
    with catch_warnings(record=True) as w:
        import warnings
        warnings.simplefilter("always")
        
        # Default padding is back=0.3.
        # Set onsource=1.0. Limit = 0.5.
        # 0.3 <= 0.5 -> Should NOT warn.
        
        dataset = gf.Dataset(
            onsource_duration_seconds=1.0,
            crop_duration_seconds=0.5,
            waveform_generators=[
                gf.WNBGenerator()
            ],
            input_variables=[gf.ReturnVariables.INJECTIONS],
            noise_obtainer=gf.NoiseObtainer(noise_type=gf.NoiseType.WHITE)
        )
        
        # Check that the specific warning is NOT present
        found = False
        for warning in w:
            if "exceeds half of onsource duration" in str(warning.message):
                found = True
                break
        
        assert not found, "Unexpected warning about padding found!"
