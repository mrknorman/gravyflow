"""Tests for gravyflow CBC waveform generation."""
import os
os.environ["KERAS_BACKEND"] = "jax"

import pytest
import numpy as np
import jax.numpy as jnp
from keras import ops

from gravyflow.src.dataset.features.waveforms.cbc import (
    generate_cbc_waveform,
    calc_minimum_frequency,
    calc_duration_from_f_min,
    Approximant
)


# ============================================================================
# HELPER FUNCTION TESTS
# ============================================================================

def test_calc_minimum_frequency():
    """Test minimum frequency calculation."""
    m1 = 30.0
    m2 = 30.0
    duration = 4.0
    
    f_min = calc_minimum_frequency(m1, m2, duration)
    
    # Should return a positive frequency
    assert float(f_min) > 0.0
    # For heavy masses and long duration, should be low
    assert float(f_min) < 50.0


def test_calc_minimum_frequency_float32():
    """Test calc_minimum_frequency with float32 dtype (covers line 66)."""
    m1 = jnp.array(30.0, dtype=jnp.float32)
    m2 = jnp.array(30.0, dtype=jnp.float32)
    duration = jnp.array(4.0, dtype=jnp.float32)
    
    f_min = calc_minimum_frequency(m1, m2, duration)
    
    assert float(f_min) > 0.0


def test_calc_duration_from_f_min():
    """Test duration calculation from minimum frequency (covers lines 83-107)."""
    m1 = 30.0
    m2 = 30.0
    f_min = 20.0
    
    duration = calc_duration_from_f_min(m1, m2, f_min)
    
    # Should return positive duration
    assert float(duration) > 0.0


def test_calc_duration_from_f_min_inverse():
    """Verify calc_duration_from_f_min is inverse of calc_minimum_frequency."""
    m1 = 30.0
    m2 = 30.0
    original_duration = 8.0
    
    # Calculate f_min from duration
    f_min = calc_minimum_frequency(m1, m2, original_duration)
    
    # Calculate duration back from f_min
    recovered_duration = calc_duration_from_f_min(m1, m2, f_min)
    
    # Should match
    np.testing.assert_allclose(float(recovered_duration), original_duration, rtol=0.01)


def test_calc_duration_from_f_min_float32():
    """Test calc_duration_from_f_min with float32 dtype (covers lines 97-99)."""
    m1 = jnp.array(30.0, dtype=jnp.float32)
    m2 = jnp.array(30.0, dtype=jnp.float32)
    f_min = jnp.array(20.0, dtype=jnp.float64)  # Different dtype to trigger cast
    
    duration = calc_duration_from_f_min(m1, m2, f_min)
    
    # Duration should be positive
    assert float(duration) > 0.0



# WAVEFORM GENERATION TESTS
# ============================================================================

def test_generate_cbc_waveform_imrphenomd():
    """Test basic IMRPhenomD waveform generation."""
    waveforms = generate_cbc_waveform(
        num_waveforms=2,
        sample_rate_hertz=2048.0,
        duration_seconds=2.0,
        mass_1_msun=30.0,
        mass_2_msun=30.0,
        approximant="IMRPhenomD"
    )
    
    # Shape: (num_waveforms, 2, num_samples)
    expected_samples = int(2048.0 * 2.0)
    assert waveforms.shape == (2, 2, expected_samples)
    
    # Should have non-zero values (actual waveform)
    assert float(jnp.max(jnp.abs(waveforms))) > 0.0


def test_generate_cbc_waveform_imrphenomxas():
    """Test IMRPhenomXAS approximant."""
    waveforms = generate_cbc_waveform(
        num_waveforms=1,
        sample_rate_hertz=2048.0,
        duration_seconds=2.0,
        mass_1_msun=30.0,
        mass_2_msun=30.0,
        approximant="IMRPhenomXAS"
    )
    
    assert waveforms.shape[0] == 1
    assert waveforms.shape[1] == 2


def test_generate_cbc_waveform_imrphenompv2():
    """Test IMRPhenomPv2 approximant (covers lines 252-264)."""
    waveforms = generate_cbc_waveform(
        num_waveforms=1,
        sample_rate_hertz=2048.0,
        duration_seconds=2.0,
        mass_1_msun=30.0,
        mass_2_msun=30.0,
        spin_1_in=[0.1, 0.1, 0.3],  # Precessing spins
        spin_2_in=[0.0, 0.0, 0.2],
        approximant="IMRPhenomPv2"
    )
    
    assert waveforms.shape[0] == 1
    assert waveforms.shape[1] == 2


def test_generate_cbc_waveform_taylorf2():
    """Test TaylorF2 approximant with tidal parameters (covers lines 266-274)."""
    waveforms = generate_cbc_waveform(
        num_waveforms=1,
        sample_rate_hertz=2048.0,
        duration_seconds=4.0,
        mass_1_msun=1.4,  # Neutron star masses
        mass_2_msun=1.4,
        lambda_1=400.0,  # Tidal deformability
        lambda_2=400.0,
        approximant="TaylorF2"
    )
    
    assert waveforms.shape[0] == 1
    assert waveforms.shape[1] == 2


def test_generate_cbc_waveform_nrtidalv2():
    """Test IMRPhenomD_NRTidalv2 approximant."""
    waveforms = generate_cbc_waveform(
        num_waveforms=1,
        sample_rate_hertz=2048.0,
        duration_seconds=4.0,
        mass_1_msun=1.4,
        mass_2_msun=1.4,
        lambda_1=300.0,
        lambda_2=300.0,
        approximant="IMRPhenomD_NRTidalv2"
    )
    
    assert waveforms.shape[0] == 1


def test_generate_cbc_waveform_invalid_approximant():
    """Test error handling for invalid approximant (covers line 172)."""
    with pytest.raises(ValueError, match="not supported"):
        generate_cbc_waveform(
            num_waveforms=1,
            approximant="InvalidApproximant"
        )


# ============================================================================
# SPIN RESHAPING TESTS
# ============================================================================

def test_generate_cbc_waveform_batch_spins():
    """Test waveform generation with batch spin inputs (covers lines 199-200, 204-205)."""
    num_waveforms = 3
    
    # Batch spins: (num_waveforms * 3,) flattened
    spin_1_flat = [0.1, 0.0, 0.3] * num_waveforms  # Will be reshaped to (3, 3)
    spin_2_flat = [0.0, 0.0, 0.2] * num_waveforms
    
    waveforms = generate_cbc_waveform(
        num_waveforms=num_waveforms,
        sample_rate_hertz=2048.0,
        duration_seconds=2.0,
        mass_1_msun=30.0,
        mass_2_msun=30.0,
        spin_1_in=spin_1_flat,
        spin_2_in=spin_2_flat,
        approximant="IMRPhenomD"
    )
    
    assert waveforms.shape[0] == num_waveforms


def test_generate_cbc_waveform_with_coalescence_time():
    """Test explicit coalescence time parameter."""
    waveforms = generate_cbc_waveform(
        num_waveforms=1,
        sample_rate_hertz=2048.0,
        duration_seconds=2.0,
        mass_1_msun=30.0,
        mass_2_msun=30.0,
        coalescence_time=1.5,  # Explicit coalescence time
        approximant="IMRPhenomD"
    )
    
    assert waveforms.shape[0] == 1


def test_generate_cbc_waveform_tensor_input():
    """Test waveform generation with tensor inputs (covers line 375)."""
    # Use ops tensors as inputs
    mass = ops.convert_to_tensor(30.0)
    
    waveforms = generate_cbc_waveform(
        num_waveforms=1,
        sample_rate_hertz=2048.0,
        duration_seconds=2.0,
        mass_1_msun=mass,
        mass_2_msun=mass,
        approximant="IMRPhenomD"
    )
    
    assert waveforms.shape[0] == 1
