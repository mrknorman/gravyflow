"""Tests for gravyflow config module."""
import os
os.environ["KERAS_BACKEND"] = "jax"

import pytest
from gravyflow.src.dataset.config import Defaults


def test_defaults_set_valid_attribute():
    """Test Defaults.set() with valid attributes."""
    # Store original value
    original_seed = Defaults.seed
    
    # Set new value
    Defaults.set(seed=12345)
    assert Defaults.seed == 12345
    
    # Restore original
    Defaults.set(seed=original_seed)


def test_defaults_set_multiple_attributes():
    """Test Defaults.set() with multiple valid attributes."""
    original_seed = Defaults.seed
    original_rate = Defaults.sample_rate_hertz
    
    Defaults.set(seed=9999, sample_rate_hertz=4096.0)
    assert Defaults.seed == 9999
    assert Defaults.sample_rate_hertz == 4096.0
    
    # Restore
    Defaults.set(seed=original_seed, sample_rate_hertz=original_rate)


def test_defaults_set_invalid_attribute():
    """Test Defaults.set() raises AttributeError for invalid attribute."""
    with pytest.raises(AttributeError, match="has no attribute named"):
        Defaults.set(nonexistent_attribute=42)
