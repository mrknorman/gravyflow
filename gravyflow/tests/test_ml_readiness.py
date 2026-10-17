"""
The datasets a model trains on: varied, reproducible, finite, correctly shaped
and scaled, with injections placed and labelled as configured.
"""
import keras
import numpy as np
import pytest
from keras import ops
from pycbc.filter import sigma
from pycbc.types import FrequencySeries, TimeSeries

import gravyflow as gf
from gravyflow.src.dataset.dataset import GravyflowDataset
from gravyflow.tests import effective_count, representation_tolerance, statistical_bound


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
    dataset = GravyflowDataset(**basic_dataset_config, seed=100, group="train")
    
    # Batches 0 and 1
    batch1 = dataset[0]
    batch2 = dataset[1]
    
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
    ds1 = GravyflowDataset(**config1, seed=42, group="train")
    
    config2 = make_config(42)
    ds2 = GravyflowDataset(**config2, seed=42, group="train")
    
    batch1 = next(iter(ds1))
    batch2 = next(iter(ds2))
    
    data1 = batch1[0][gf.ReturnVariables.WHITENED_ONSOURCE.name]
    data2 = batch2[0][gf.ReturnVariables.WHITENED_ONSOURCE.name]
    
    
    np.testing.assert_allclose(data1, data2, err_msg="Datasets with same seed should produce identical first batches.")
    
    # Different seed -> Different data
    config3 = make_config(999)
    ds3 = GravyflowDataset(**config3, seed=999, group="train")
    batch3 = next(iter(ds3))
    data3 = batch3[0][gf.ReturnVariables.WHITENED_ONSOURCE.name]
    
    assert not np.allclose(data1, data3), "Datasets with different seeds should produce different first batches."

def test_peak_jitter(basic_noise_obtainer):
    """
    Peaks spread uniformly over the padding, not locked to the centre.

    The injection's centre is shifted uniformly over [-front, back) padding, so
    over n examples the peak index has mean ``samples / 2`` and standard
    deviation ``span / sqrt(12)``. Statistical bounds, k = 4: the mean's standard
    error is ``std / sqrt(n)``; the sample standard deviation's is
    ``std * sqrt(0.2 / n)`` (uniform: Var(s) = (mu4 - std^4) / (4 std^2 n),
    mu4 = 1.8 std^4).
    """
    padding_seconds = 2.0
    num_examples = 50
    wnb_generator = gf.WNBGenerator(
        scaling_method=gf.ScalingMethod(
            value=gf.Distribution(min_=1000.0, max_=1000.0, type_=gf.DistributionType.UNIFORM),
            type_=gf.ScalingTypes.SNR
        ),
        network=[gf.IFO.L1],
        duration_seconds=0.1,  # Short, so the peak marks the centre.
        min_frequency_hertz=100.0,
        max_frequency_hertz=200.0,
        front_padding_duration_seconds=padding_seconds,
        back_padding_duration_seconds=padding_seconds,
    )
    dataset = GravyflowDataset(
        noise_obtainer=basic_noise_obtainer,
        sample_rate_hertz=SAMPLE_RATE,
        onsource_duration_seconds=DURATION,
        offsource_duration_seconds=2.0,
        crop_duration_seconds=0.5,
        num_examples_per_batch=num_examples,
        waveform_generators=[wnb_generator],
        input_variables=[gf.ReturnVariables.WHITENED_INJECTIONS],
        seed=789,
        group="train",
    )
    batch = next(iter(dataset))
    # (Batch, Generators, IFO, Time): the one generator, the first IFO.
    injections = np.asarray(batch[0][gf.ReturnVariables.WHITENED_INJECTIONS.name])[:, 0, 0, :]
    peak_indices = np.argmax(np.abs(injections), axis=-1)

    span = 2 * padding_seconds * SAMPLE_RATE
    expected_std = span / np.sqrt(12.0)
    expected_mean = injections.shape[-1] / 2
    assert abs(np.mean(peak_indices) - expected_mean) <= statistical_bound(
        expected_std / np.sqrt(num_examples)
    ), f"Peak indices are centred at {np.mean(peak_indices)}, not {expected_mean}."
    assert abs(np.std(peak_indices) - expected_std) <= statistical_bound(
        expected_std * np.sqrt(0.2 / num_examples)
    ), f"Peak indices spread {np.std(peak_indices)}, not {expected_std}: clustered or locked."


#: The PSD estimate the dataset scales against (SpectralConfig's defaults).
SPECTRAL = gf.SpectralConfig()
#: A Hann window's equivalent noise bandwidth, in bins: neighbouring bins of a
#: Hann-windowed estimate are correlated, so this many count as one.
HANN_BINS_PER_INDEPENDENT_BIN = 1.5


def _wnb_generator(scaling_method):
    return gf.WNBGenerator(scaling_method=scaling_method, network=[gf.IFO.L1], duration_seconds=0.5)


def _phenom_d_generator(scaling_method):
    generator = gf.WaveformGenerator.load(
        gf.tests.PATH / "example_injection_parameters" / "phenom_d_parameters.json"
    )
    generator.scaling_method = scaling_method
    generator.injection_chance = 1.0
    return generator


def _reference_snr(injection):
    """
    pycbc's sigma of one detector's series against the true one-sided PSD of
    the unit-variance white noise, 2 / fs, over the dataset's band.
    """
    htilde = TimeSeries(
        np.asarray(injection, dtype=np.float64), delta_t=1.0 / SAMPLE_RATE
    ).to_frequencyseries()
    psd = FrequencySeries(np.full(len(htilde), 2.0 / SAMPLE_RATE), delta_f=htilde.delta_f)
    return float(sigma(htilde, psd=psd, low_frequency_cutoff=SPECTRAL.lower_frequency_hertz))


def _reference_snr_sigma(injection, offsource_seconds):
    """
    The relative standard deviation of the scaled SNR against the reference,
    from the noise of the dataset's PSD estimate.

    A median of K independent periodograms (exponential, unit mean) has relative
    standard deviation 1 / (ln 2 sqrt(K)); the K non-overlapping segments of the
    offsource bound the overlapping estimate's count from below. The SNR is the
    square root of a weighted mean of the estimate over the injection's in-band
    power, which averages N_eff independent bins, so its relative deviation is
    half the estimate's over sqrt(N_eff).
    """
    num_segments = int(offsource_seconds // SPECTRAL.fft_duration_seconds)
    estimate_sigma = 1.0 / (np.log(2.0) * np.sqrt(num_segments))
    power = np.abs(np.fft.rfft(np.asarray(injection, dtype=np.float64))) ** 2
    frequencies = np.fft.rfftfreq(len(injection), 1.0 / SAMPLE_RATE)
    in_band = (frequencies >= SPECTRAL.lower_frequency_hertz) & (frequencies < SAMPLE_RATE / 2)
    estimate_bins = np.floor(frequencies[in_band] * SPECTRAL.fft_duration_seconds).astype(int)
    independent_bins = effective_count(
        np.bincount(estimate_bins, weights=power[in_band])
    ) / HANN_BINS_PER_INDEPENDENT_BIN
    return 0.5 * estimate_sigma / np.sqrt(independent_bins)


@pytest.mark.parametrize("make_generator, num_examples, seed", [
    (_wnb_generator, 20, 101),
    (_phenom_d_generator, 10, 102),
], ids=["wnb", "phenom_d"])
def test_injections_carry_their_snr_label(basic_noise_obtainer, make_generator, num_examples, seed):
    """
    Each injection is scaled to the SNR its label reports, against the noise.

    The labels lie in the target range; the SNR of the returned, cropped
    injection in the returned offsource is its label (representation: the
    FFT's log2(N) rounding); and pycbc's SNR against the noise's true PSD
    agrees with it within the PSD estimate's own noise (statistical, k = 4).
    """
    min_snr, max_snr = 10.0, 20.0
    offsource_seconds = 16.0
    generator = make_generator(gf.ScalingMethod(
        value=gf.Distribution(min_=min_snr, max_=max_snr, type_=gf.DistributionType.UNIFORM),
        type_=gf.ScalingTypes.SNR
    ))
    dataset = GravyflowDataset(
        noise_obtainer=basic_noise_obtainer,
        sample_rate_hertz=SAMPLE_RATE,
        onsource_duration_seconds=DURATION,
        offsource_duration_seconds=offsource_seconds,
        crop_duration_seconds=0.5,
        num_examples_per_batch=num_examples,
        waveform_generators=[generator],
        input_variables=[
            gf.ReturnVariables.INJECTIONS, gf.ReturnVariables.OFFSOURCE, gf.ScalingTypes.SNR
        ],
        seed=seed,
        group="train",
    )
    inputs, _ = next(iter(dataset))
    # (Batch, Generators, IFO, Time) and (Batch, Generators): the one generator.
    injections = np.asarray(inputs[gf.ReturnVariables.INJECTIONS.name])[:, 0]
    offsource = np.asarray(inputs[gf.ReturnVariables.OFFSOURCE.name])
    labels = np.asarray(inputs[gf.ScalingTypes.SNR.name])[:, 0]

    assert np.all((labels >= min_snr) & (labels <= max_snr)), f"Labels outside the range: {labels}"

    measured = np.asarray(gf.snr(injections, offsource, sample_rate_hertz=SAMPLE_RATE))
    tolerance = representation_tolerance(np.float32, max_snr, np.log2(injections.shape[-1]))
    np.testing.assert_allclose(measured, labels, rtol=0.0, atol=tolerance)

    for injection, label in zip(injections[:, 0], labels):
        deviation = _reference_snr(injection) / label - 1.0
        bound = statistical_bound(_reference_snr_sigma(injection, offsource_seconds))
        assert abs(deviation) <= bound, (
            f"SNR {label} against the true PSD is off by {deviation:.4f} (bound {bound:.4f})."
        )


def test_numerical_stability(basic_dataset_config):
    """Batches have the configured shape, (Batch, IFO, onsource samples), and are finite."""
    dataset = GravyflowDataset(**basic_dataset_config, seed=555, group="train")
    expected_shape = (BATCH_SIZE, 1, int(DURATION * SAMPLE_RATE))

    for k in range(3):
        inputs, targets = dataset[k]
        assert set(inputs) == {gf.ReturnVariables.WHITENED_ONSOURCE.name}
        assert targets == {}
        X = np.asarray(inputs[gf.ReturnVariables.WHITENED_ONSOURCE.name])
        assert X.shape == expected_shape
        assert np.all(np.isfinite(X)), "Input data contains NaNs or Infs."

def test_micro_training(basic_dataset_config):
    """A minimal Keras model trains on the dataset's dicts with a finite loss."""
    # A denoising task, so the input and target keys differ: WHITENED_ONSOURCE
    # in, WHITENED_INJECTIONS out.
    config = basic_dataset_config.copy()
    config["input_variables"] = [gf.ReturnVariables.WHITENED_ONSOURCE]
    config["output_variables"] = [gf.ReturnVariables.WHITENED_INJECTIONS]
    wnb_generator = gf.WaveformGenerator.load(
        path=gf.tests.PATH / "example_injection_parameters" / "wnb_parameters.json",
        scaling_method=gf.ScalingMethod(
            value=gf.Distribution(min_=10.0, max_=20.0, type_=gf.DistributionType.UNIFORM),
            type_=gf.ScalingTypes.SNR
        ),
        network=[gf.IFO.L1]
    )
    wnb_generator.injection_chance = 1.0
    config["waveform_generators"] = [wnb_generator]
    dataset = GravyflowDataset(**config, seed=314, group="train")

    input_shape = ops.shape(next(iter(dataset))[0][gf.ReturnVariables.WHITENED_ONSOURCE.name])[1:]
    # Seeded initialisers: the weights come from their own seeds, not a global RNG.
    inputs = keras.Input(shape=input_shape, name=gf.ReturnVariables.WHITENED_ONSOURCE.name)
    x = keras.layers.Flatten()(inputs)
    x = keras.layers.Dense(
        4, activation="relu", kernel_initializer=keras.initializers.GlorotUniform(seed=1)
    )(x)
    x = keras.layers.Dense(
        int(np.prod(input_shape)), kernel_initializer=keras.initializers.GlorotUniform(seed=2)
    )(x)
    # The target is (Batch, Generators, IFO, Time), one generator here.
    outputs = keras.layers.Reshape(
        (1,) + tuple(input_shape), name=gf.ReturnVariables.WHITENED_INJECTIONS.name
    )(x)
    model = keras.Model(
        inputs=inputs, outputs={gf.ReturnVariables.WHITENED_INJECTIONS.name: outputs}
    )
    model.compile(optimizer="adam", loss="mse")

    # shuffle=False: the dataset's seeds set its order, and Keras would shuffle a
    # PyDataset's batch order with Python's global random module.
    history = model.fit(dataset, steps_per_epoch=2, epochs=1, verbose=0, shuffle=False)

    losses = history.history["loss"]
    assert len(losses) == 1 and np.isfinite(losses[0]), f"Training loss: {losses}"


def test_parameter_independence(basic_noise_obtainer):
    """Verify that drawn parameters are uncorrelated (e.g. inclination vs mass).

    Not inclination vs distance: under SNR scaling DISTANCE_MPC is the distance
    the injected amplitude implies, which depends on the inclination.

    Statistical, k = 3 (stated): the sample correlation of n independent draws
    has standard deviation 1 / sqrt(n).
    """
    num_examples = 100
    cbc_generator = gf.CBCGenerator(
        scaling_method=gf.ScalingMethod(
            value=gf.Distribution(min_=10.0, max_=20.0, type_=gf.DistributionType.UNIFORM),
            type_=gf.ScalingTypes.SNR
        ),
        network=[gf.IFO.L1],
        mass_1_msun=gf.Distribution(min_=10.0, max_=50.0, type_=gf.DistributionType.UNIFORM),
        mass_2_msun=gf.Distribution(min_=10.0, max_=50.0, type_=gf.DistributionType.UNIFORM),
        inclination_radians=gf.Distribution(min_=0.0, max_=np.pi, type_=gf.DistributionType.UNIFORM),
        distance_mpc=gf.Distribution(min_=100.0, max_=500.0, type_=gf.DistributionType.UNIFORM)
    )
    dataset = GravyflowDataset(
        noise_obtainer=basic_noise_obtainer,
        sample_rate_hertz=SAMPLE_RATE,
        onsource_duration_seconds=DURATION,
        offsource_duration_seconds=2.0,
        crop_duration_seconds=0.5,
        num_examples_per_batch=num_examples,
        waveform_generators=[cbc_generator],
        input_variables=[
            gf.WaveformParameters.INCLINATION_RADIANS,
            gf.WaveformParameters.MASS_1_MSUN
        ],
        seed=42,
        group="train",
    )
    inputs = next(iter(dataset))[0]
    # (Batch, Generators) -> (Batch,)
    inclination = np.asarray(inputs[gf.WaveformParameters.INCLINATION_RADIANS.name]).flatten()
    mass = np.asarray(inputs[gf.WaveformParameters.MASS_1_MSUN.name]).flatten()

    correlation = np.corrcoef(inclination, mass)[0, 1]
    bound = statistical_bound(1.0 / np.sqrt(num_examples), k=3.0)
    assert abs(correlation) <= bound, f"Parameters appear correlated: {correlation} (bound {bound})"


def test_parameter_variation(basic_noise_obtainer):
    """Verify that parameters change across batches and within batches."""
    
    # Setup generator
    scaling_method = gf.ScalingMethod(
        value=gf.Distribution(min_=10.0, max_=20.0, type_=gf.DistributionType.UNIFORM),
        type_=gf.ScalingTypes.SNR
    )
    
    cbc_generator = gf.CBCGenerator(
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
        "waveform_generators": [cbc_generator],
        "input_variables": [gf.WaveformParameters.MASS_1_MSUN]
    }
    
    dataset = GravyflowDataset(**config, seed=123, group="train")
    
    # Batch 0
    batch1 = dataset[0]
    m1_batch1 = np.array(batch1[0][gf.WaveformParameters.MASS_1_MSUN.name]).flatten()
    
    # Check variation within batch
    assert np.std(m1_batch1) > 0.0, "Parameter constant within batch (expected variation)."
    
    # Batch 1
    batch2 = dataset[1]
    m1_batch2 = np.array(batch2[0][gf.WaveformParameters.MASS_1_MSUN.name]).flatten()
    
    # Check variation across batches
    # They shouldn't be identical
    assert not np.allclose(m1_batch1, m1_batch2), "Parameters identical across batches."

def test_signal_variation(basic_noise_obtainer):
    """Verify that INJECTIONS (pure signal) vary across batches."""
    
    # Setup generator
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
    
    dataset = GravyflowDataset(**config, seed=456, group="train")
    
    # next(iter(dataset)) is always batch 0: compare two batches by index.
    inj1 = np.asarray(dataset[0][0][gf.ReturnVariables.INJECTIONS.name])
    inj2 = np.asarray(dataset[1][0][gf.ReturnVariables.INJECTIONS.name])

    # Different by far more than float noise: no example of batch 1 repeats one of batch 0.
    scale = np.max(np.abs(inj1))
    for a in inj1:
        for b in inj2:
            assert np.max(np.abs(a - b)) > 1e-3 * scale, "An injection repeats across batches."
    
    # Check they are not all zeros (assuming injection chance 1.0 default)
    assert np.max(np.abs(inj1)) > 0.0, "Injections are all zeros."

def test_padding_adherence(basic_noise_obtainer):
    """
    Verify that signal peaks remain within the bounds defined by front/back padding.
    
    Logic:
    - CBCGenerator centers signals at total_duration / 2.
    - InjectionGenerator shifts signals by random amount in [-front_padding, back_padding).
    - Therefore, peak index should be in [center - front, center + back).
    """
    
    # Setup generator
    scaling_method = gf.ScalingMethod(
        value=gf.Distribution(min_=10.0, max_=20.0, type_=gf.DistributionType.UNIFORM),
        type_=gf.ScalingTypes.SNR
    )
    
    front_pad = 1.0
    back_pad = 1.0
    
    # Use CBCGenerator (PhenomD) which centers the merger
    cbc_generator = gf.CBCGenerator(
        scaling_method=scaling_method,
        network=[gf.IFO.L1],
        mass_1_msun=gf.Distribution(min_=30.0, max_=30.0, type_=gf.DistributionType.UNIFORM),
        mass_2_msun=gf.Distribution(min_=30.0, max_=30.0, type_=gf.DistributionType.UNIFORM),
        front_padding_duration_seconds=front_pad,
        back_padding_duration_seconds=back_pad
    )
    
    # Logic:
    # 1. Generation Window: [0, T_total] where T_total = onsource + 2*crop
    # 2. Center: T_total / 2
    # 3. Shifted Peak: T_total / 2 + shift
    # 4. Crop: Removes `crop_samples` from start and end.
    # 5. Final Peak Index in Output: (T_total / 2 + shift) - crop_samples
    #    = (onsource + 2*crop)/2 + shift - crop
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
        "waveform_generators": [cbc_generator],
        "input_variables": [gf.ReturnVariables.WHITENED_INJECTIONS]
    }
    
    dataset = GravyflowDataset(**config, seed=999, group="train")
    batch = next(iter(dataset))
    
    injections = batch[0][gf.ReturnVariables.WHITENED_INJECTIONS.name]
    injections = injections[:, 0, 0, :]  # the one generator, the first IFO
        
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
    
    dataset = GravyflowDataset(**config, seed=123, group="train")
    batch = next(iter(dataset))
    
    inputs = batch[0]
    whitened_onsource = inputs[gf.ReturnVariables.WHITENED_ONSOURCE.name]
    injections = inputs[gf.ReturnVariables.INJECTIONS.name]
    
    # Check shapes
    shape_onsource = ops.shape(whitened_onsource)
    shape_injections = ops.shape(injections)
    
    
    # Time dimension is usually the last one
    time_dim_onsource = shape_onsource[-1]
    time_dim_injections = shape_injections[-1]
    
    assert time_dim_onsource == time_dim_injections, \
        f"Shape mismatch! WHITENED_ONSOURCE time dim: {time_dim_onsource}, INJECTIONS time dim: {time_dim_injections}"

def test_injection_mask_consistency(basic_noise_obtainer):
    """
    Regression Test for "Missing CBC Signal" issue.
    Verifies that if INJECTION_MASKS indicates an injection (value 1.0),
    the corresponding INJECTIONS data is not all zeros.
    
    This guards against the bug where NaNs in generated waveforms were treated 
    as valid injections by the mask generation logic (NaN != 0 is True), 
    but were subsequently cleaned to 0.0 by the dataset pipeline.
    """
    # Setup CBCGenerator with IMRPhenomPv2 (problematic waveform)
    scaling_method = gf.ScalingMethod(
        value=gf.Distribution(min_=10.0, max_=20.0, type_=gf.DistributionType.UNIFORM),
        type_=gf.ScalingTypes.SNR
    )
    
    # Use a specific seed that might have triggered issues, or just random
    cbc_generator = gf.CBCGenerator(
        scaling_method=scaling_method,
        network=[gf.IFO.L1],
        approximant="IMRPhenomPv2",
        mass_1_msun=gf.Distribution(min_=10.0, max_=60.0, type_=gf.DistributionType.UNIFORM),
        mass_2_msun=gf.Distribution(min_=10.0, max_=60.0, type_=gf.DistributionType.UNIFORM),
        distance_mpc=gf.Distribution(min_=100.0, max_=2000.0, type_=gf.DistributionType.UNIFORM),
        inclination_radians=gf.Distribution(min_=0.0, max_=3.14, type_=gf.DistributionType.UNIFORM),
    )
    
    config = {
        "noise_obtainer": basic_noise_obtainer,
        "sample_rate_hertz": 2048.0,
        "onsource_duration_seconds": 4.0,
        "offsource_duration_seconds": 4.0,
        "crop_duration_seconds": 0.5,
        "num_examples_per_batch": 16,
        "waveform_generators": [cbc_generator],
        "input_variables": [
            gf.ReturnVariables.INJECTIONS,
            gf.ReturnVariables.INJECTION_MASKS
        ],
        "output_variables": [gf.ReturnVariables.INJECTION_MASKS] # Dummy
    }
    
    # Run for a few batches to ensure coverage
    dataset = GravyflowDataset(**config, seed=0, group="train")
    iterator = iter(dataset)
    
    for _ in range(5):
        batch = next(iterator)
        inputs = batch[0]
        
        # injections: (Batch, Generators, IFO, Time); masks: (Batch, Generators)
        injections = inputs[gf.ReturnVariables.INJECTIONS.name][:, 0]
        masks = inputs[gf.ReturnVariables.INJECTION_MASKS.name][:, 0]
        
        # The masks are the injections that carry signal: each set mask has a
        # non-zero injection, each clear one none, and every batch has some.
        injected = np.max(np.abs(np.asarray(injections)), axis=(-2, -1)) > 0
        masks = np.asarray(masks)
        assert np.any(masks > 0.5), "No injection in the batch: the check would pass vacuously."
        np.testing.assert_array_equal(
            masks > 0.5, injected, err_msg="A mask disagrees with its injection."
        )
