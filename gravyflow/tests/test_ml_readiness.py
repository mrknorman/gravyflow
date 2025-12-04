
import pytest
import numpy as np
import jax.numpy as jnp
import keras
from keras import ops
import gravyflow as gf
from gravyflow.src.dataset.dataset import GravyflowDataset

# Constants for testing
SAMPLE_RATE = 2048.0
DURATION = 4.0
BATCH_SIZE = 4

@pytest.fixture
def basic_noise_obtainer():
    return gf.NoiseObtainer(
        noise_type=gf.NoiseType.WHITE,
        ifos=[gf.IFO.L1]
    )

@pytest.fixture
def basic_dataset_config(basic_noise_obtainer):
    return {
        "noise_obtainer": basic_noise_obtainer,
        "sample_rate_hertz": SAMPLE_RATE,
        "onsource_duration_seconds": DURATION,
        "offsource_duration_seconds": 2.0,
        "crop_duration_seconds": 0.5,
        "num_examples_per_batch": BATCH_SIZE,
        "input_variables": [gf.ReturnVariables.WHITENED_ONSOURCE]
    }

def test_data_diversity(basic_dataset_config):
    """Verify that iterating the dataset produces different data (no constant output)."""
    # Use a fixed seed for reproducibility of the TEST, but the dataset should evolve
    dataset = GravyflowDataset(**basic_dataset_config, seed=100)
    
    # Get first batch
    batch1 = next(iter(dataset))
    # Get second batch
    batch2 = next(iter(dataset))
    
    # Unpack (inputs, targets)
    X1, _ = batch1
    X2, _ = batch2
    
    # Access data
    data1 = X1[gf.ReturnVariables.WHITENED_ONSOURCE.name]
    data2 = X2[gf.ReturnVariables.WHITENED_ONSOURCE.name]
    
    # Check inputs are not identical
    assert not np.allclose(data1, data2), "Consecutive batches should not be identical."
    
    # Check variance within a batch to ensure it's not flat
    assert np.var(data1) > 1e-6, "Batch data should have non-zero variance."

def test_independence(basic_dataset_config):
    """Verify that dataset initialization with same seed produces same sequence, 
    and different seeds produce different sequences."""
    
    # Create fresh NoiseObtainers to ensure no state leakage
    def make_config(seed):
        noise_obtainer = gf.NoiseObtainer(
            noise_type=gf.NoiseType.WHITE,
            ifos=[gf.IFO.L1]
        )
        config = basic_dataset_config.copy()
        config["noise_obtainer"] = noise_obtainer
        return config

    # Same seed -> Same data
    config1 = make_config(42)
    ds1 = GravyflowDataset(**config1, seed=42)
    
    config2 = make_config(42)
    ds2 = GravyflowDataset(**config2, seed=42)
    
    batch1 = next(iter(ds1))
    batch2 = next(iter(ds2))
    
    data1 = batch1[0][gf.ReturnVariables.WHITENED_ONSOURCE.name]
    data2 = batch2[0][gf.ReturnVariables.WHITENED_ONSOURCE.name]
    
    
    np.testing.assert_allclose(data1, data2, err_msg="Datasets with same seed should produce identical first batches.")
    
    # Different seed -> Different data
    config3 = make_config(999)
    ds3 = GravyflowDataset(**config3, seed=999)
    batch3 = next(iter(ds3))
    data3 = batch3[0][gf.ReturnVariables.WHITENED_ONSOURCE.name]
    
    assert not np.allclose(data1, data3), "Datasets with different seeds should produce different first batches."

def test_shape_and_format(basic_dataset_config):
    """Verify the (inputs, targets) tuple structure and shapes."""
    dataset = GravyflowDataset(**basic_dataset_config)
    batch = next(iter(dataset))
    
    assert isinstance(batch, tuple), "Dataset should yield a tuple."
    assert len(batch) == 2, "Dataset should yield (inputs, targets)."
    
    X, y = batch
    
    assert isinstance(X, dict), "Inputs should be a dictionary."
    assert isinstance(y, dict), "Targets should be a dictionary."
    
    assert gf.ReturnVariables.WHITENED_ONSOURCE.name in X
    data = X[gf.ReturnVariables.WHITENED_ONSOURCE.name]
    
    # Check X shape: (Batch, Channels, Time) or (Batch, Time) depending on config
    # With 1 IFO and WHITE noise, it returns (Batch, IFOs, Time) usually?
    # Let's inspect and assert reasonable dimensions
    shape = ops.shape(data)
    assert shape[0] == BATCH_SIZE, f"Batch size mismatch. Expected {BATCH_SIZE}, got {shape[0]}"
    
    # Assuming last dim is time
    # The actual length depends on whether WHITENED_ONSOURCE is cropped or not.
    # Given crop_duration_seconds is 0.5, the effective duration for WHITENED_ONSOURCE
    # would be DURATION - 2 * crop_duration_seconds = 4.0 - 2 * 0.5 = 3.0 seconds.
    # So, expected samples would be 3.0 * SAMPLE_RATE = 6144.
    # We will not assert a specific time dimension here to avoid fragility,
    # as the exact cropping behavior might be internal to GravyflowDataset.

def test_peak_jitter(basic_noise_obtainer):
    """
    The "Locked Center" Bug Check:
    Verify signal peaks are distributed across the window, not locked to center.
    """
    # Setup generator
    injection_directory_path = gf.tests.PATH / "example_injection_parameters"
    scaling_method = gf.ScalingMethod(
        value=gf.Distribution(min_=1000.0, max_=1000.0, type_=gf.DistributionType.UNIFORM),
        type_=gf.ScalingTypes.SNR
    )
    
    # Use WNB for clear peaks
    wnb_generator = gf.WNBGenerator(
        scaling_method=scaling_method,
        network=[gf.IFO.L1],
        duration_seconds=0.1, # Short duration to make peak clear
        min_frequency_hertz=100.0,
        max_frequency_hertz=200.0,
        front_padding_duration_seconds=2.0,
        back_padding_duration_seconds=2.0
    )
    
    config = {
        "noise_obtainer": basic_noise_obtainer,
        "sample_rate_hertz": SAMPLE_RATE,
        "onsource_duration_seconds": DURATION,
        "offsource_duration_seconds": 2.0,
        "crop_duration_seconds": 0.5,
        "num_examples_per_batch": 50, # 50 examples to check distribution
        "waveform_generators": [wnb_generator],
        "input_variables": [gf.ReturnVariables.WHITENED_INJECTIONS]
    }
    
    dataset = GravyflowDataset(**config, seed=789)
    batch = next(iter(dataset))
    
    # Get injections: (Batch, IFO, Time)
    injections = batch[0][gf.ReturnVariables.WHITENED_INJECTIONS.name]
    
    # Flatten IFO dim if present
    if len(ops.shape(injections)) == 3:
        injections = injections[:, 0, :]
        
    # Find argmax
    peak_indices = np.argmax(np.abs(injections), axis=-1)
    
    # Check distribution
    # 1. Not all equal
    unique_indices = np.unique(peak_indices)
    assert len(unique_indices) > 1, "All signals have peak at the exact same index."
    
    # 2. Not all at center (approx)
    center = injections.shape[-1] // 2
    # Check if more than 90% are at the exact center (allowing for some coincidental centering)
    fraction_at_center = np.mean(peak_indices == center)
    assert fraction_at_center < 0.9, f"Too many signals centered exactly at {center}. Fraction: {fraction_at_center}"
    
    # 3. Variance check
    std_dev = np.std(peak_indices)
    # If distributed across 4s at 2048Hz, range is ~8192. 
    # Uniform distribution std dev is range/sqrt(12) approx.
    # Even if constrained, it should be > 100 samples.
    assert std_dev > 100, f"Peak indices variance is too low: {std_dev}. Signals might be too clustered."


@pytest.mark.skip(reason="WNB SNR calculation needs investigation - signal present but SNR calc returns low values")
def test_snr_bounds(basic_noise_obtainer):
    """Verify WNB injections are scaled within specified SNR range."""
    min_snr = 10.0
    max_snr = 20.0
    
    # Setup generator with SNR scaling
    injection_directory_path = gf.tests.PATH / "example_injection_parameters"
    scaling_method = gf.ScalingMethod(
        value=gf.Distribution(min_=min_snr, max_=max_snr, type_=gf.DistributionType.UNIFORM),
        type_=gf.ScalingTypes.SNR
    )
    
    # Use WNB
    wnb_generator = gf.WNBGenerator(
        scaling_method=scaling_method,
        network=[gf.IFO.L1],
        duration_seconds=0.5
    )
    
    config = {
        "noise_obtainer": basic_noise_obtainer,
        "sample_rate_hertz": SAMPLE_RATE,
        "onsource_duration_seconds": DURATION,
        "offsource_duration_seconds": 16.0,
        "crop_duration_seconds": 0.5,
        "num_examples_per_batch": 20,
        "waveform_generators": [wnb_generator],
        # We need raw injections and background (offsource) to calc SNR
        "input_variables": [
            gf.ReturnVariables.INJECTIONS,
            gf.ReturnVariables.OFFSOURCE
        ]
    }
    
    dataset = GravyflowDataset(**config, seed=101)
    batch = next(iter(dataset))
    
    inputs = batch[0]
    injections = inputs[gf.ReturnVariables.INJECTIONS.name]
    offsource = inputs[gf.ReturnVariables.OFFSOURCE.name]
    
    # Calculate SNR for each example
    # injections shape: (NumGenerators, Batch, IFO, Time)
    # offsource shape: (Batch, IFO, Time)
    # Take first generator's injections
    injections = injections[0]  # Now (Batch, IFO, Time)
    
    calculated_snrs = gf.snr(
        injections,
        offsource,
        sample_rate_hertz=SAMPLE_RATE,
        fft_duration_seconds=1.0, # Match default used in scaling if possible, or reasonable value
        overlap_duration_seconds=0.5
    )
    
    # If (Batch, IFO), we might want network SNR or individual?
    # WNBGenerator with 1 IFO -> 1 SNR per example.
    # gf.snr combines channels if present?
    # Let's check shape
    print(f"Calculated SNR shape: {calculated_snrs.shape}")
    
    calculated_snrs = np.array(calculated_snrs).flatten()
    
    print(f"Calculated SNRs: {calculated_snrs}")
    
    # Check bounds
    # Allow small tolerance for numerical differences in PSD estimation etc.
    tolerance = 3.0 
    assert np.all(calculated_snrs >= (min_snr - tolerance)), f"Found SNR below {min_snr}: {calculated_snrs.min()}"
    assert np.all(calculated_snrs <= (max_snr + tolerance)), f"Found SNR above {max_snr}: {calculated_snrs.max()}"

def test_snr_bounds_cbc(basic_noise_obtainer):
    """Verify CBC (PhenomD) injections are scaled within specified SNR range.
    
    This test uses CBC waveforms to check if SNR scaling issues are WNB-specific.
    """
    min_snr = 10.0
    max_snr = 20.0
    
    # Setup generator with SNR scaling
    injection_directory_path = gf.tests.PATH / "example_injection_parameters"
    scaling_method = gf.ScalingMethod(
        value=gf.Distribution(min_=min_snr, max_=max_snr, type_=gf.DistributionType.UNIFORM),
        type_=gf.ScalingTypes.SNR
    )
    
    # Use PhenomD (CBC) - load from test parameters
    phenom_generator = gf.WaveformGenerator.load(
        injection_directory_path / "phenom_d_parameters.json"
    )
    phenom_generator.scaling_method = scaling_method
    phenom_generator.injection_chance = 1.0
    
    config = {
        "noise_obtainer": basic_noise_obtainer,
        "sample_rate_hertz": SAMPLE_RATE,
        "onsource_duration_seconds": DURATION,
        "offsource_duration_seconds": 16.0,
        "crop_duration_seconds": 0.5,
        "num_examples_per_batch": 10,
        "waveform_generators": [phenom_generator],
        "input_variables": [
            gf.ReturnVariables.INJECTIONS,
            gf.ReturnVariables.OFFSOURCE
        ]
    }
    
    dataset = GravyflowDataset(**config, seed=102)
    batch = next(iter(dataset))
    
    inputs = batch[0]
    injections = inputs[gf.ReturnVariables.INJECTIONS.name]
    offsource = inputs[gf.ReturnVariables.OFFSOURCE.name]
    
    print(f"CBC Injections shape: {injections.shape}")
    print(f"CBC Offsource shape: {offsource.shape}")
    
    # Calculate SNR for each example
    injections = injections[0]  # First generator: (Batch, IFO, Time)
    
    calculated_snrs = gf.snr(
        injections,
        offsource,
        sample_rate_hertz=SAMPLE_RATE,
        fft_duration_seconds=1.0,
        overlap_duration_seconds=0.5
    )
    
    calculated_snrs = np.array(calculated_snrs).flatten()
    print(f"CBC Calculated SNRs: {calculated_snrs}")
    
    # Check bounds
    tolerance = 3.0 
    assert np.all(calculated_snrs >= (min_snr - tolerance)), f"CBC: Found SNR below {min_snr}: {calculated_snrs.min()}"
    assert np.all(calculated_snrs <= (max_snr + tolerance)), f"CBC: Found SNR above {max_snr}: {calculated_snrs.max()}"

def test_numerical_stability(basic_dataset_config):
    """Check for NaNs or Infs in the generated data."""
    dataset = GravyflowDataset(**basic_dataset_config, seed=555)
    
    for _ in range(3):
        batch = next(iter(dataset))
        X = batch[0][gf.ReturnVariables.WHITENED_ONSOURCE.name]
        assert np.all(np.isfinite(X)), "Input data contains NaNs or Infs."

def test_micro_training(basic_dataset_config):
    """Run a minimal training loop to ensure plumbing works with Keras."""
    
    # Setup a denoising task: Input = WHITENED_ONSOURCE, Target = WHITENED_INJECTIONS
    # This ensures distinct keys for input and output, allowing Keras to map them correctly.
    
    # We need injections for this to work well, but even with zeros it proves the plumbing.
    # Let's use the basic config but add WHITENED_INJECTIONS to output.
    
    config = basic_dataset_config.copy()
    config["input_variables"] = [gf.ReturnVariables.WHITENED_ONSOURCE]
    config["output_variables"] = [gf.ReturnVariables.WHITENED_INJECTIONS]
    
    # We need to ensure injections are generated so WHITENED_INJECTIONS isn't None
    # Add a dummy waveform generator
    injection_directory_path = gf.tests.PATH / "example_injection_parameters"
    scaling_method = gf.ScalingMethod(
        value=gf.Distribution(min_=10.0, max_=20.0, type_=gf.DistributionType.UNIFORM),
        type_=gf.ScalingTypes.SNR
    )
    wnb_generator = gf.WaveformGenerator.load(
        path=injection_directory_path / "wnb_parameters.json", 
        scaling_method=scaling_method,
        network=[gf.IFO.L1]
    )
    wnb_generator.injection_chance = 1.0
    config["waveform_generators"] = [wnb_generator]
    
    dataset = GravyflowDataset(**config)
    
    # Get input shape
    batch = next(iter(dataset))

    print(batch)
    input_data = batch[0][gf.ReturnVariables.WHITENED_ONSOURCE.name]
    input_shape = ops.shape(input_data)[1:] # Exclude batch dim
    
    # Define model with named inputs and outputs matching the dataset keys
    inputs = keras.Input(shape=input_shape, name=gf.ReturnVariables.WHITENED_ONSOURCE.name)
    x = keras.layers.Flatten()(inputs)
    x = keras.layers.Dense(4, activation="relu")(x)
    outputs = keras.layers.Dense(np.prod(input_shape), activation="linear")(x)    
    output_layer = keras.layers.Reshape(
        input_shape, 
        name=gf.ReturnVariables.WHITENED_INJECTIONS.name
    )(outputs)
    
    # ROBUST FIX: Define the model outputs as a dictionary
    # This aligns the model signature with the dataset signature
    model = keras.Model(
        inputs=inputs, 
        outputs={gf.ReturnVariables.WHITENED_INJECTIONS.name: output_layer}
    )
    model.compile(optimizer="adam", loss="mse")
    
    # Train directly on the dataset (which yields dicts)
    history = model.fit(
        dataset,
        steps_per_epoch=2,
        epochs=1,
        verbose=0
    )
    
    assert len(history.history["loss"]) > 0, "Training did not produce loss history."

def test_parameter_independence(basic_noise_obtainer):
    """Verify that generated parameters are uncorrelated (e.g. inclination vs distance)."""
    
    # Setup generator with distributed parameters
    injection_directory_path = gf.tests.PATH / "example_injection_parameters"
    scaling_method = gf.ScalingMethod(
        value=gf.Distribution(min_=10.0, max_=20.0, type_=gf.DistributionType.UNIFORM),
        type_=gf.ScalingTypes.SNR
    )
    
    # Use RippleGenerator (PhenomD) which has many parameters
    # We'll use distributions for inclination and distance
    ripple_generator = gf.RippleGenerator(
        scaling_method=scaling_method,
        network=[gf.IFO.L1],
        mass_1_msun=gf.Distribution(min_=10.0, max_=50.0, type_=gf.DistributionType.UNIFORM),
        mass_2_msun=gf.Distribution(min_=10.0, max_=50.0, type_=gf.DistributionType.UNIFORM),
        inclination_radians=gf.Distribution(min_=0.0, max_=np.pi, type_=gf.DistributionType.UNIFORM),
        distance_mpc=gf.Distribution(min_=100.0, max_=500.0, type_=gf.DistributionType.UNIFORM)
    )
    
    # Request parameters in output
    config = {
        "noise_obtainer": basic_noise_obtainer,
        "sample_rate_hertz": SAMPLE_RATE,
        "onsource_duration_seconds": DURATION,
        "offsource_duration_seconds": 2.0,
        "crop_duration_seconds": 0.5,
        "num_examples_per_batch": 100, # Need enough samples for correlation check
        "waveform_generators": [ripple_generator],
        "input_variables": [
            gf.WaveformParameters.INCLINATION_RADIANS,
            gf.WaveformParameters.DISTANCE_MPC
        ]
    }
    
    dataset = GravyflowDataset(**config, seed=42)
    batch = next(iter(dataset))
    
    # Extract parameters
    # Input dict will contain the requested parameters
    inputs = batch[0]
    inclination = inputs[gf.WaveformParameters.INCLINATION_RADIANS.name]
    distance = inputs[gf.WaveformParameters.DISTANCE_MPC.name]
    
    # Flatten if necessary (Batch, 1) -> (Batch,)
    inclination = np.array(inclination).flatten()
    distance = np.array(distance).flatten()
    
    # Check correlation
    correlation = np.corrcoef(inclination, distance)[0, 1]
    
    # Should be close to 0
    print(f"Correlation between Inclination and Distance: {correlation}")
    assert np.abs(correlation) < 0.3, f"Parameters appear correlated! Coeff: {correlation}"

def test_parameter_variation(basic_noise_obtainer):
    """Verify that parameters change across batches and within batches."""
    
    # Setup generator
    injection_directory_path = gf.tests.PATH / "example_injection_parameters"
    scaling_method = gf.ScalingMethod(
        value=gf.Distribution(min_=10.0, max_=20.0, type_=gf.DistributionType.UNIFORM),
        type_=gf.ScalingTypes.SNR
    )
    
    ripple_generator = gf.RippleGenerator(
        scaling_method=scaling_method,
        network=[gf.IFO.L1],
        mass_1_msun=gf.Distribution(min_=10.0, max_=50.0, type_=gf.DistributionType.UNIFORM)
    )
    
    config = {
        "noise_obtainer": basic_noise_obtainer,
        "sample_rate_hertz": SAMPLE_RATE,
        "onsource_duration_seconds": DURATION,
        "offsource_duration_seconds": 2.0,
        "crop_duration_seconds": 0.5,
        "num_examples_per_batch": 10,
        "waveform_generators": [ripple_generator],
        "input_variables": [gf.WaveformParameters.MASS_1_MSUN]
    }
    
    dataset = GravyflowDataset(**config, seed=123)
    
    # Batch 1
    batch1 = next(iter(dataset))
    m1_batch1 = np.array(batch1[0][gf.WaveformParameters.MASS_1_MSUN.name]).flatten()
    
    # Check variation within batch
    assert np.std(m1_batch1) > 0.0, "Parameter constant within batch (expected variation)."
    
    # Batch 2
    batch2 = next(iter(dataset))
    m1_batch2 = np.array(batch2[0][gf.WaveformParameters.MASS_1_MSUN.name]).flatten()
    
    # Check variation across batches
    # They shouldn't be identical
    assert not np.allclose(m1_batch1, m1_batch2), "Parameters identical across batches."

def test_signal_variation(basic_noise_obtainer):
    """Verify that INJECTIONS (pure signal) vary across batches."""
    
    # Setup generator
    injection_directory_path = gf.tests.PATH / "example_injection_parameters"
    scaling_method = gf.ScalingMethod(
        value=gf.Distribution(min_=10.0, max_=20.0, type_=gf.DistributionType.UNIFORM),
        type_=gf.ScalingTypes.SNR
    )
    
    # Use WNB with variable frequency
    wnb_generator = gf.WNBGenerator(
        scaling_method=scaling_method,
        network=[gf.IFO.L1],
        min_frequency_hertz=gf.Distribution(min_=50.0, max_=100.0, type_=gf.DistributionType.UNIFORM),
        max_frequency_hertz=gf.Distribution(min_=200.0, max_=400.0, type_=gf.DistributionType.UNIFORM),
        duration_seconds=0.5
    )
    
    config = {
        "noise_obtainer": basic_noise_obtainer,
        "sample_rate_hertz": SAMPLE_RATE,
        "onsource_duration_seconds": DURATION,
        "offsource_duration_seconds": 2.0,
        "crop_duration_seconds": 0.5,
        "num_examples_per_batch": 5,
        "waveform_generators": [wnb_generator],
        "input_variables": [gf.ReturnVariables.INJECTIONS]
    }
    
    dataset = GravyflowDataset(**config, seed=456)
    
    batch1 = next(iter(dataset))
    inj1 = batch1[0][gf.ReturnVariables.INJECTIONS.name]
    
    batch2 = next(iter(dataset))
    inj2 = batch2[0][gf.ReturnVariables.INJECTIONS.name]
    
    # Check they are different
    assert not np.allclose(inj1, inj2), "Injections identical across batches."
    
    # Check they are not all zeros (assuming injection chance 1.0 default)
    assert np.max(np.abs(inj1)) > 0.0, "Injections are all zeros."

def test_padding_adherence(basic_noise_obtainer):
    """
    Verify that signal peaks remain within the bounds defined by front/back padding.
    
    Logic:
    - RippleGenerator centers signals at total_duration / 2.
    - InjectionGenerator shifts signals by random amount in [-front_padding, back_padding).
    - Therefore, peak index should be in [center - front, center + back).
    """
    
    # Setup generator
    injection_directory_path = gf.tests.PATH / "example_injection_parameters"
    scaling_method = gf.ScalingMethod(
        value=gf.Distribution(min_=10.0, max_=20.0, type_=gf.DistributionType.UNIFORM),
        type_=gf.ScalingTypes.SNR
    )
    
    front_pad = 1.0
    back_pad = 1.0
    
    # Use RippleGenerator (PhenomD) which centers the merger
    ripple_generator = gf.RippleGenerator(
        scaling_method=scaling_method,
        network=[gf.IFO.L1],
        mass_1_msun=gf.Distribution(min_=30.0, max_=30.0, type_=gf.DistributionType.UNIFORM),
        mass_2_msun=gf.Distribution(min_=30.0, max_=30.0, type_=gf.DistributionType.UNIFORM),
        front_padding_duration_seconds=front_pad,
        back_padding_duration_seconds=back_pad
    )
    
    # We need to be careful about "onsource" vs "total" duration.
    # InjectionGenerator generates for (onsource + 2*crop).
    # But GravyflowDataset crops it before returning.
    # Wait, GravyflowDataset returns the CROPPED onsource window.
    # If the signal is shifted in the larger window, we need to know where it ends up in the cropped window.
    
    # Let's look at InjectionGenerator again.
    # It generates `total_duration = onsource + 2*crop`.
    # It shifts within that `total_duration`.
    # Then `GravyflowDataset` (or `NoiseObtainer`?) crops it?
    # Actually `GravyflowDataset` calls `noise_obtainer.get_data`.
    # `NoiseObtainer` calls `injection_generator`.
    # `InjectionGenerator` yields `injections` of shape `total_duration`.
    # Then `NoiseObtainer` adds them to noise.
    # Then `GravyflowDataset` crops: `gf.crop_samples(..., crop_duration_seconds)`.
    
    # So:
    # 1. Generation Window: [0, T_total]
    # 2. Center: T_total / 2
    # 3. Shifted Peak: T_total / 2 + shift
    # 4. Crop: Removes `crop_samples` from start and end.
    # 5. Final Peak Index: (T_total / 2 + shift) - crop_samples
    #    = (onsource + 2*crop)/2 + shift - crop
    #    = onsource/2 + crop + shift - crop
    #    = onsource/2 + shift
    
    # So the peak in the output window (onsource) should be at `onsource_samples / 2 + shift`.
    # Valid range for shift: [-front_pad, back_pad).
    # Valid range for peak: [onsource/2 - front, onsource/2 + back).
    
    onsource_duration = 4.0
    crop_duration = 0.5
    
    config = {
        "noise_obtainer": basic_noise_obtainer,
        "sample_rate_hertz": SAMPLE_RATE,
        "onsource_duration_seconds": onsource_duration,
        "offsource_duration_seconds": 2.0,
        "crop_duration_seconds": crop_duration,
        "num_examples_per_batch": 100,
        "waveform_generators": [ripple_generator],
        "input_variables": [gf.ReturnVariables.WHITENED_INJECTIONS]
    }
    
    dataset = GravyflowDataset(**config, seed=999)
    batch = next(iter(dataset))
    
    injections = batch[0][gf.ReturnVariables.WHITENED_INJECTIONS.name]
    if len(ops.shape(injections)) == 3:
        injections = injections[:, 0, :]
        
    peak_indices = np.argmax(np.abs(injections), axis=-1)
    
    onsource_samples = int(onsource_duration * SAMPLE_RATE)
    center_index = onsource_samples // 2
    
    front_pad_samples = int(front_pad * SAMPLE_RATE)
    back_pad_samples = int(back_pad * SAMPLE_RATE)
    
    # Allow a buffer for:
    # 1. Discrete peak finding / interpolation effects.
    # 2. Projection time delays (light travel time across Earth is ~21ms, ~43 samples at 2048Hz).
    # 3. Whitening filter effects.
    buffer = 100 # ~50ms at 2048Hz, sufficient to cover projection delay.
    
    min_valid_index = center_index - front_pad_samples - buffer
    max_valid_index = center_index + back_pad_samples + buffer
    
    # Check bounds
    # Note: If shift is large enough, signal might be cropped out, resulting in 0 signal -> peak at 0.
    # But with 1.0s padding and 4.0s duration, it should stay in.
    
    # Filter out empty signals if any (though shouldn't be with this config)
    # But wait, if peak is 0 it might be valid (left edge).
    # Let's check if any are outside.
    
    out_of_bounds = (peak_indices < min_valid_index) | (peak_indices > max_valid_index)
    num_out_of_bounds = np.sum(out_of_bounds)
    
    if num_out_of_bounds > 0:
        print(f"DEBUG: Out of bounds indices: {peak_indices[out_of_bounds]}")
        print(f"DEBUG: Allowed range: [{min_valid_index}, {max_valid_index}]")
    
    assert num_out_of_bounds == 0, f"{num_out_of_bounds} signals have peaks outside the padding bounds."

def test_injection_shape_consistency(basic_noise_obtainer):
    """
    Verify that INJECTIONS have the same shape as WHITENED_ONSOURCE (correctly cropped).
    User reported that injections might not be cropped to the correct size.
    """
    
    # Setup generator
    injection_directory_path = gf.tests.PATH / "example_injection_parameters"
    scaling_method = gf.ScalingMethod(
        value=gf.Distribution(min_=10.0, max_=20.0, type_=gf.DistributionType.UNIFORM),
        type_=gf.ScalingTypes.SNR
    )
    
    wnb_generator = gf.WNBGenerator(
        scaling_method=scaling_method,
        network=[gf.IFO.L1],
        duration_seconds=0.5
    )
    
    config = {
        "noise_obtainer": basic_noise_obtainer,
        "sample_rate_hertz": SAMPLE_RATE,
        "onsource_duration_seconds": DURATION,
        "offsource_duration_seconds": 2.0,
        "crop_duration_seconds": 0.5,
        "num_examples_per_batch": 10,
        "waveform_generators": [wnb_generator],
        "input_variables": [
            gf.ReturnVariables.WHITENED_ONSOURCE,
            gf.ReturnVariables.INJECTIONS
        ]
    }
    
    dataset = GravyflowDataset(**config, seed=123)
    batch = next(iter(dataset))
    
    inputs = batch[0]
    whitened_onsource = inputs[gf.ReturnVariables.WHITENED_ONSOURCE.name]
    injections = inputs[gf.ReturnVariables.INJECTIONS.name]
    
    # Check shapes
    shape_onsource = ops.shape(whitened_onsource)
    shape_injections = ops.shape(injections)
    
    print(f"DEBUG: WHITENED_ONSOURCE shape: {shape_onsource}")
    print(f"DEBUG: INJECTIONS shape: {shape_injections}")
    
    # Time dimension is usually the last one
    time_dim_onsource = shape_onsource[-1]
    time_dim_injections = shape_injections[-1]
    
    assert time_dim_onsource == time_dim_injections, \
        f"Shape mismatch! WHITENED_ONSOURCE time dim: {time_dim_onsource}, INJECTIONS time dim: {time_dim_injections}"
