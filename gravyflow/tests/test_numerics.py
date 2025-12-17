
import pytest
from gravyflow.src.utils.numerics import ensure_even

def test_ensure_even():
    """Test ensure_even function."""
    # Even number stays the same
    assert ensure_even(100) == 100
    assert ensure_even(4096) == 4096
    
    # Odd number becomes even (subtract 1)
    assert ensure_even(101) == 100
    assert ensure_even(4097) == 4096
    assert ensure_even(1) == 0
    assert ensure_even(0) == 0
    assert ensure_even(2) == 2
    assert ensure_even(3) == 2
