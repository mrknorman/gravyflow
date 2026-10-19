"""
Tests of the matched-filter building blocks: the SNR module's matched_filter_fft,
template_sigma and optimal_snr, and the template grid. The search itself
(MatchedFilter, its layer, trigger clustering and the baseline model) is tested
in test_matched_filter_search.py.
"""

from pathlib import Path

import pytest

import numpy as np
import jax.numpy as jnp

import gravyflow as gf
from gravyflow.src.detection import TemplateGrid, matched_filter_fft, template_sigma
from gravyflow.tests._helpers import representation_tolerance, statistical_bound


def rounding(num_samples: int) -> float:
    """
    Representation: the relative float32 rounding of an SNR, a sum of N
    squares and FFTs of N samples (sqrt N + log2 N roundings).
    """
    return representation_tolerance(np.float32, 1.0, np.sqrt(num_samples) + np.log2(num_samples))


def whole_cycles(frequency_hertz, sample_rate, duration):
    """A unit sine of a whole number of cycles: its energy N / 2 sits in one rfft bin."""
    t = np.arange(int(sample_rate * duration)) / sample_rate
    return jnp.asarray(np.sin(2 * np.pi * frequency_hertz * t), jnp.float32)


class TestTemplateGrid:
    """Tests for the TemplateGrid class."""
    
    def test_grid_creation(self):
        """Test that template grid creates correct number of templates."""
        grid = TemplateGrid(
            mass_1_range=(10.0, 50.0),
            mass_2_range=(10.0, 50.0),
            num_mass_1_points=5,
        )
        
        # For n mass points, we get n*(n+1)/2 templates (upper triangle)
        expected = 5 * (5 + 1) // 2  # = 15
        assert grid.num_templates == expected
        
    def test_mass_constraints(self):
        """Test that m1 >= m2 is enforced."""
        grid = TemplateGrid(
            mass_1_range=(20.0, 40.0),
            mass_2_range=(20.0, 40.0),
            num_mass_1_points=4,
        )
        
        m1, m2 = grid.get_parameters()
        
        # All pairs should satisfy m1 >= m2
        assert jnp.all(m1 >= m2)


class TestMatchedFilterCore:
    """Tests for core matched filter functions."""
    
    def test_template_sigma_unit_signal(self):
        """Test that template sigma is computed correctly for unit sine wave.
        
        For a unit-amplitude sine wave, sigma ≈ 1.0 with our normalization.
        This verifies the sigma calculation is working correctly.
        """
        sample_rate = 8192.0
        n_samples = int(sample_rate)
        templates = whole_cycles(100.0, sample_rate, 1.0)[None, :]

        sigma = template_sigma(templates, psd=None, sample_rate_hertz=sample_rate)

        # psd=None is unit-variance white noise (one-sided PSD 2/fs), the PSD of
        # whitened data: a unit sine of N samples then has SNR sqrt(N / 2).
        np.testing.assert_allclose(float(sigma[0]), np.sqrt(n_samples / 2.0), rtol=rounding(n_samples))
        
    def test_autocorrelation_peak(self):
        """Test that autocorrelation SNR equals optimal SNR (sigma).
        
        For autocorrelation (signal matched against itself):
        - Inner product <h|h> = sigma²
        - Max SNR = <h|h> / sigma = sigma
        """
        sample_rate = 8192.0
        duration = 1.0
        n_samples = int(sample_rate * duration)
        
        t = jnp.linspace(0, duration, n_samples)
        signal = jnp.sin(2 * jnp.pi * 100 * t)
        templates = signal[None, :]
        
        # Compute optimal SNR (sigma)
        sigma = template_sigma(templates, psd=None, sample_rate_hertz=sample_rate)
        expected_snr = float(sigma[0])
        
        # Match filter signal against itself
        snr = matched_filter_fft(signal, templates, psd=None, sample_rate_hertz=sample_rate)
        max_snr = float(jnp.max(snr))
        
        np.testing.assert_allclose(max_snr, expected_snr, rtol=rounding(n_samples),
            err_msg=f"Autocorrelation SNR should equal sigma: got {max_snr}, expected {expected_snr}")
        
    def test_correct_template_identification(self):
        """Test that the correct template is identified as best match."""
        sample_rate = 8192.0
        duration = 1.0
        n_samples = int(sample_rate * duration)
        t = jnp.linspace(0, duration, n_samples)
        
        # Create 3 different frequency templates
        templates = jnp.stack([
            jnp.sin(2 * jnp.pi * 50 * t),   # 50 Hz
            jnp.sin(2 * jnp.pi * 100 * t),  # 100 Hz
            jnp.sin(2 * jnp.pi * 150 * t),  # 150 Hz
        ])
        
        # Signal matches template 1 (100 Hz)
        signal = jnp.sin(2 * jnp.pi * 100 * t)
        
        snr = matched_filter_fft(signal, templates, psd=None, sample_rate_hertz=sample_rate)
        max_snr_per_template = jnp.max(snr, axis=-1)
        best_idx = int(jnp.argmax(max_snr_per_template))
        
        assert best_idx == 1, f"Expected template 1, got {best_idx}"


    def test_snr_recovery_with_return_variable(self):
        """
        Verify matched filter SNR matches the injected SNR returned by the dataset.
        
        This test:
        1. Generates a batch with randomized SNRs (uniform 8-15).
        2. Requests ScalingTypes.SNR and INJECTIONS as return variables.
        3. Uses the exact INJECTIONS as templates (perfect match).
        4. Computes PSD from OFFSOURCE using same settings as dataset.
        5. Runs matched filter and verifies recovered SNR ≈ target SNR.
        """
        from gravyflow.src.dataset.dataset import GravyflowDataset
        from gravyflow.src.dataset.tools.snr import matched_filter_fft
        from gravyflow.src.dataset.tools.psd import psd as gf_psd
        
        # Use a range of SNRs to verify scaling works dynamically
        scaling_method = gf.ScalingMethod(
            value=gf.Distribution(min_=8.0, max_=15.0, type_=gf.DistributionType.UNIFORM),
            type_=gf.ScalingTypes.SNR
        )
        
        cbc_generator = gf.CBCGenerator(
            scaling_method=scaling_method,
            network=[gf.IFO.L1],
            mass_1_msun=gf.Distribution(value=30.0, type_=gf.DistributionType.CONSTANT),
            mass_2_msun=gf.Distribution(value=25.0, type_=gf.DistributionType.CONSTANT),
            injection_chance=1.0,
            # With no crop, the window's edges carry the whitening's edge
            # effects: keep the merger within 0.3 s of the centre.
            front_padding_duration_seconds=0.3,
            back_padding_duration_seconds=0.3,
        )
        
        noise_obtainer = gf.NoiseObtainer(
            noise_type=gf.NoiseType.WHITE,  # Use WHITE for simpler PSD
            ifos=[gf.IFO.L1]
        )
        
        dataset = GravyflowDataset(
            noise_obtainer=noise_obtainer,
            sample_rate_hertz=8192.0,
            onsource_duration_seconds=2.0,
            offsource_duration_seconds=16.0,  # Longer for better PSD estimate
            crop_duration_seconds=0.0,  # No cropping for simplicity
            num_examples_per_batch=4,
            waveform_generators=[cbc_generator],
            input_variables=[
                gf.ReturnVariables.ONSOURCE,
                gf.ReturnVariables.OFFSOURCE,
                gf.ReturnVariables.INJECTIONS,  # Get exact injection waveform
            ],
            output_variables=[
                gf.ReturnVariables.INJECTION_MASKS,
                gf.ScalingTypes.SNR,
            ],
            seed=42,
            group="train",
        )
        
        batch = next(iter(dataset))
        inputs, outputs = batch
        
        onsource = inputs[gf.ReturnVariables.ONSOURCE.name]
        offsource = inputs[gf.ReturnVariables.OFFSOURCE.name]
        injections = inputs[gf.ReturnVariables.INJECTIONS.name]
        injected_snrs = np.array(outputs["SNR"])[:, 0]  # (batch, generators): the one generator
        
        # PSD settings matching dataset's snr function
        fft_duration = 4.0
        overlap_duration = 2.0
        nperseg = int(8192.0 * fft_duration)
        noverlap = int(8192.0 * overlap_duration)
        
        for i in range(4):
            # Extract single example
            data = np.array(onsource[i, 0, :])
            bg = np.array(offsource[i, 0, :])
            injection = np.squeeze(injections[i, 0, :])
            target_snr = float(injected_snrs[i])
            
            # Compute PSD from offsource using gf_psd with same settings as dataset
            spectrum = gf_psd(bg, sample_rate_hertz=8192.0, 
                                     nperseg=nperseg, noverlap=noverlap, mode='mean')
            freqs, psd_val = spectrum.frequencies, spectrum.value()
            
            # Interpolate PSD to match injection length
            n = len(injection)
            fft_freqs = np.fft.rfftfreq(n, d=1/8192.0)
            psd_interp = np.interp(fft_freqs, np.array(freqs), np.squeeze(psd_val))
            
            # Apply 20Hz low-frequency cutoff (same as dataset)
            psd_interp[fft_freqs < 20.0] = 1e10
            psd_jax = jnp.array(psd_interp, dtype=jnp.float32)
            
            # Use exact injection as template
            template = jnp.array(injection[None, :], dtype=jnp.float32)
            data_jax = jnp.array(data, dtype=jnp.float32)
            
            # Run matched filter
            snr_result = matched_filter_fft(data_jax, template, psd=psd_jax, 
                                            sample_rate_hertz=8192.0)
            recovered_snr = float(jnp.max(jnp.abs(snr_result)))

            # Statistical (k = 3, stated): the recovered SNR is the target plus
            # the noise's projection on the template, a unit normal, so 3 sigma
            # is 3 SNR units (plus a small PSD-estimate error, well inside it).
            # It also bounds the recovered SNR below by min target - 3 = 5.
            assert abs(recovered_snr - target_snr) < statistical_bound(1.0, k=3.0), \
                f"Example {i}: recovered SNR {recovered_snr:.2f}, target {target_snr:.2f}"
    
def test_matched_filter_with_sine_gaussian():
    """Test matched filter detection of sine-Gaussian signal.
    
    Uses SineGaussianGenerator to create both signal and template,
    verifying that matched filter can identify the signal.
    """
    # Generate sine-Gaussian as signal
    gen = gf.SineGaussianGenerator(
        frequency_hertz=gf.Distribution(value=100.0, type_=gf.DistributionType.CONSTANT),
        quality_factor=gf.Distribution(value=8.0, type_=gf.DistributionType.CONSTANT),
        amplitude=gf.Distribution(value=1.0, type_=gf.DistributionType.CONSTANT),
        network=[gf.IFO.L1],
    )
    
    sample_rate = 2048.0
    duration = 1.0
    
    waveforms, _ = gen.generate(
        num_waveforms=1,
        sample_rate_hertz=sample_rate,
        duration_seconds=duration
    )
    
    # Use h+ as signal
    signal = waveforms[0, 0, :]
    
    # Generate template bank using same generator with varying Q
    templates = []
    for q in [6.0, 8.0, 10.0]:  # Template 1 (index 1) matches Q=8
        gen_template = gf.SineGaussianGenerator(
            frequency_hertz=100.0, 
            quality_factor=q,
            network=[gf.IFO.L1],
        )
        wf, _ = gen_template.generate(1, sample_rate, duration)
        # Normalize template
        template = wf[0, 0, :]
        template = template / jnp.sqrt(jnp.sum(template**2))
        templates.append(template)
    
    templates = jnp.stack(templates)  # (3, time)
    
    # Add noise
    noise = np.random.default_rng(42).normal(0, 0.1, signal.shape)
    data = signal + jnp.array(noise)
    
    # Run matched filter
    snr = matched_filter_fft(data, templates, psd=None)
    
    # Template at index 1 (Q=8) should have highest SNR
    max_snr_per_template = jnp.max(snr, axis=-1)
    best_idx = int(jnp.argmax(max_snr_per_template))
    
    assert best_idx == 1, f"Expected Q=8 template (index 1) to be best, got {best_idx}"


def test_matched_filter_with_chirplet():
    """Test matched filter detection of chirplet signal.
    
    Verifies matched filter can discriminate between chirps
    with different frequency sweeps.
    """
    sample_rate = 2048.0
    duration = 1.0
    
    # Generate target chirplet: 50Hz -> 200Hz
    gen = gf.ChirpletGenerator(
        start_frequency_hertz=50.0,
        end_frequency_hertz=200.0,
        duration_seconds=0.5,
        network=[gf.IFO.L1],
    )
    
    waveforms, _ = gen.generate(1, sample_rate, duration)
    signal = waveforms[0, 0, :]
    
    # Create template bank with different frequency sweeps
    templates = []
    freq_pairs = [(30.0, 150.0), (50.0, 200.0), (100.0, 300.0)]  # Index 1 matches
    for f0, f1 in freq_pairs:
        gen_t = gf.ChirpletGenerator(
            start_frequency_hertz=f0,
            end_frequency_hertz=f1,
            duration_seconds=0.5,
            network=[gf.IFO.L1],
        )
        wf, _ = gen_t.generate(1, sample_rate, duration)
        template = wf[0, 0, :]
        template = template / jnp.sqrt(jnp.sum(template**2))
        templates.append(template)
    
    templates = jnp.stack(templates)
    
    # Add noise
    noise = np.random.default_rng(42).normal(0, 0.1, signal.shape)
    data = signal + jnp.array(noise)
    
    # Run matched filter
    snr = matched_filter_fft(data, templates, psd=None)
    
    max_snr_per_template = jnp.max(snr, axis=-1)
    best_idx = int(jnp.argmax(max_snr_per_template))
    
    assert best_idx == 1, f"Expected 50-200Hz chirplet (index 1) to be best, got {best_idx}"


def test_matched_filter_with_periodic_waves():
    """Test matched filter can distinguish between periodic wave shapes."""
    sample_rate = 2048.0
    duration = 0.5
    
    # Generate sine wave as signal
    gen = gf.PeriodicWaveGenerator(
        wave_shape=gf.WaveShape.SINE,
        frequency_hertz=100.0,
        duration_seconds=0.3,
        network=[gf.IFO.L1],
    )
    
    waveforms, _ = gen.generate(1, sample_rate, duration)
    signal = waveforms[0, 0, :]
    
    # Templates: sine, square, triangle (sine should match best)
    templates = []
    for shape in [gf.WaveShape.SINE, gf.WaveShape.SQUARE, gf.WaveShape.TRIANGLE]:
        gen_t = gf.PeriodicWaveGenerator(
            wave_shape=shape,
            frequency_hertz=100.0,
            duration_seconds=0.3,
            network=[gf.IFO.L1],
        )
        wf, _ = gen_t.generate(1, sample_rate, duration)
        template = wf[0, 0, :]
        template = template / jnp.sqrt(jnp.sum(template**2))
        templates.append(template)
    
    templates = jnp.stack(templates)
    
    # Add noise
    noise = np.random.default_rng(42).normal(0, 0.05, signal.shape)
    data = signal + jnp.array(noise)
    
    # Run matched filter
    snr = matched_filter_fft(data, templates, psd=None)
    
    max_snr_per_template = jnp.max(snr, axis=-1)
    best_idx = int(jnp.argmax(max_snr_per_template))
    
    # Sine template should have highest SNR with sine signal
    assert best_idx == 0, f"Expected SINE template (index 0) to be best, got {best_idx}"


class TestSNRScientificAccuracy:
    """The SNR's defining properties, each exact up to float32 rounding."""

    RATE, N = 2048.0, 2048

    def test_snr_scales_with_amplitude(self):
        """The matched-filter SNR is linear in the signal's amplitude."""
        t = jnp.arange(self.N) / self.RATE
        base_signal = jnp.sin(2 * jnp.pi * (50 + 50 * t) * t)  # chirp 50-150 Hz
        template = base_signal[None, :]
        amplitudes = np.array([0.5, 1.0, 2.0, 4.0])
        snrs = np.array([float(jnp.max(matched_filter_fft(a * base_signal, template, psd=None,
                                                          sample_rate_hertz=self.RATE)))
                         for a in amplitudes])
        np.testing.assert_allclose(snrs / snrs[0], amplitudes / amplitudes[0], rtol=rounding(self.N))

    def test_snr_inversely_proportional_to_noise(self):
        """sigma scales as PSD^(-1/2): four times the noise power halves it."""
        template = whole_cycles(100.0, self.RATE, 1.0)[None, :]
        n_freq = self.N // 2 + 1
        sigma_base = template_sigma(template, jnp.ones(n_freq) * 1e-20, self.RATE)
        sigma_loud = template_sigma(template, jnp.ones(n_freq) * 4e-20, self.RATE)
        np.testing.assert_allclose(float(sigma_loud[0] / sigma_base[0]), 0.5, rtol=rounding(self.N))

    def test_autocorrelation_snr_equals_optimal_snr(self):
        """A signal filtered with itself recovers its optimal SNR, sqrt(<h|h>)."""
        from gravyflow.src.dataset.tools.snr import optimal_snr

        signal = whole_cycles(100.0, self.RATE, 1.0)
        opt_snr = float(optimal_snr(signal, psd=None, sample_rate_hertz=self.RATE))
        np.testing.assert_allclose(opt_snr, np.sqrt(self.N / 2.0), rtol=rounding(self.N))
        mf_snr = float(jnp.max(matched_filter_fft(signal, signal[None, :], psd=None, sample_rate_hertz=self.RATE)))
        np.testing.assert_allclose(mf_snr, opt_snr, rtol=rounding(self.N))

    def test_matched_filter_time_localization(self):
        """The SNR of a pulse against itself peaks at zero lag, and a delayed pulse at its delay, exactly."""
        t = jnp.arange(self.N) / self.RATE

        def pulse(centre):
            return jnp.exp(-((t - centre) ** 2) / (2 * 0.01 ** 2))

        template = pulse(0.5)[None, :]
        for delay in (0, 100, 517):
            snr = matched_filter_fft(pulse(0.5 + delay / self.RATE), template, psd=None, sample_rate_hertz=self.RATE)
            assert int(jnp.argmax(snr[0])) == delay

    def test_template_mismatch_reduces_snr(self):
        """
        Sines of whole cycles in different bins are orthogonal at every circular
        lag: a 120 Hz template finds nothing of a 100 Hz signal.
        """
        signal = whole_cycles(100.0, self.RATE, 1.0)
        snr_perfect = float(jnp.max(matched_filter_fft(signal, signal[None, :], psd=None,
                                                        sample_rate_hertz=self.RATE)))
        snr_mismatch = float(jnp.max(matched_filter_fft(signal, whole_cycles(120.0, self.RATE, 1.0)[None, :],
                                                         psd=None, sample_rate_hertz=self.RATE)))
        assert snr_mismatch < rounding(self.N) * snr_perfect, (snr_mismatch, snr_perfect)


class TestSNRCoverage:
    """A PSD given on another grid is interpolated onto the data's."""

    RATE, N = 2048.0, 2048

    def test_matched_filter_with_psd_interpolation(self):
        """A flat PSD of the wrong length interpolates to the same flat PSD: the same SNR series."""
        signal = whole_cycles(100.0, self.RATE, 1.0)
        n_freq = self.N // 2 + 1
        on_grid = matched_filter_fft(signal, signal[None, :], psd=jnp.ones(n_freq) * 1e-20,
                                     sample_rate_hertz=self.RATE)
        off_grid = matched_filter_fft(signal, signal[None, :], psd=jnp.ones(n_freq + 100) * 1e-20,
                                      sample_rate_hertz=self.RATE)
        assert off_grid.shape == (1, self.N)
        np.testing.assert_allclose(np.asarray(off_grid), np.asarray(on_grid), rtol=0.0,
                                   atol=rounding(self.N) * float(jnp.max(on_grid)))

    def test_template_sigma_with_psd_interpolation(self):
        """A PSD of the wrong length gives the sigma of the same PSD on the data's grid."""
        template = whole_cycles(100.0, self.RATE, 1.0)[None, :]
        n_freq = self.N // 2 + 1
        # A linear PSD is interpolated exactly: the same line on either grid.
        line = lambda n: np.linspace(1.0, 3.0, n) * 1e-20  # float64: physical PSDs underflow float32
        on_grid = template_sigma(template, psd=line(n_freq), sample_rate_hertz=self.RATE)
        off_grid = template_sigma(template, psd=line(n_freq + 50), sample_rate_hertz=self.RATE)
        assert off_grid.shape == (1,)
        np.testing.assert_allclose(np.asarray(off_grid), np.asarray(on_grid), rtol=rounding(self.N))

    def test_optimal_snr_function(self):
        """optimal_snr of one signal is template_sigma of it as a one-template bank."""
        from gravyflow.src.dataset.tools.snr import optimal_snr

        signal = whole_cycles(100.0, self.RATE, 0.5)
        opt = optimal_snr(signal, psd=None, sample_rate_hertz=self.RATE)
        np.testing.assert_array_equal(np.asarray(opt),
                                      np.asarray(template_sigma(signal[None, :], None, self.RATE))[0])
        np.testing.assert_allclose(float(opt), np.sqrt(signal.size / 2.0), rtol=rounding(signal.size))

class TestTemplateGridCoverage:
    """Tests to cover remaining code paths in template_grid.py."""
    
    def test_get_chirp_masses(self):
        """Test get_chirp_masses method (lines 86-87)."""
        grid = TemplateGrid(
            mass_1_range=(10.0, 30.0),
            mass_2_range=(10.0, 30.0),
            num_mass_1_points=4,
        )
        
        chirp_masses = grid.get_chirp_masses()
        m1, m2 = grid.get_parameters()
        
        # Check shape
        assert chirp_masses.shape == m1.shape
        
        # Verify formula: M_c = (m1 * m2)^(3/5) / (m1 + m2)^(1/5)
        expected = (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2
        np.testing.assert_allclose(chirp_masses, expected, rtol=1e-6)
        
        # Chirp mass should be positive
        assert jnp.all(chirp_masses > 0)
        
        # Chirp mass should be less than geometric mean of component masses
        # For equal masses: M_c = m * 2^(-1/5) ≈ 0.87 * m
        for i in range(len(m1)):
            geometric_mean = jnp.sqrt(m1[i] * m2[i])
            assert chirp_masses[i] <= geometric_mean
    
    def test_asymmetric_mass_grid(self):
        """Test grid with different num_mass_2_points."""
        grid = TemplateGrid(
            mass_1_range=(10.0, 50.0),
            mass_2_range=(5.0, 30.0),
            num_mass_1_points=5,
            num_mass_2_points=3,  # Different from num_mass_1
        )
        
        m1, m2 = grid.get_parameters()
        
        # All should satisfy m1 >= m2
        assert jnp.all(m1 >= m2)
        
        # Mass ranges should be respected
        assert jnp.min(m1) >= 10.0
        assert jnp.max(m1) <= 50.0
        assert jnp.min(m2) >= 5.0
        assert jnp.max(m2) <= 30.0


class TestPsdUnderJit:
    """A PSD array is normalised on the host: the documented limit of the array path."""

    RATE = 1024.0

    def templates(self):
        return whole_cycles(100.0, self.RATE, 1.0)[None, :]

    def array(self):
        # The one-sided PSD of unit-variance white noise, as a float64 array.
        return np.full(int(self.RATE) // 2 + 1, 2.0 / self.RATE)

    def test_a_psd_array_closed_over_traces_and_matches_eager(self):
        import jax
        psd, templates = self.array(), self.templates()
        eager = template_sigma(templates, psd=psd, sample_rate_hertz=self.RATE)
        traced = jax.jit(lambda t: template_sigma(t, psd=psd, sample_rate_hertz=self.RATE))(templates)
        np.testing.assert_allclose(np.asarray(traced), np.asarray(eager), rtol=1e-6)

    def test_a_psd_array_passed_as_a_jit_argument_is_refused(self):
        import jax
        with pytest.raises(jax.errors.TracerArrayConversionError):
            jax.jit(lambda t, p: template_sigma(t, psd=p, sample_rate_hertz=self.RATE))(
                self.templates(), self.array())

    def test_a_power_spectrum_passed_as_a_jit_argument_traces(self):
        import jax
        config = gf.SpectralConfig(lower_frequency_hertz=20.0)
        spectrum = gf.PowerSpectrum(np.full((1, 1, int(self.RATE) // 2 + 1), 2.0, dtype=np.float32),
                                    np.ones((1, 1, 1), dtype=np.float32), self.RATE, config)
        templates = self.templates()
        eager = template_sigma(templates, psd=spectrum, sample_rate_hertz=self.RATE)
        traced = jax.jit(lambda t, p: template_sigma(t, psd=p, sample_rate_hertz=self.RATE))(templates, spectrum)
        np.testing.assert_allclose(np.asarray(traced), np.asarray(eager), rtol=1e-5)

    def test_no_psd_traces_with_the_data_as_arguments(self):
        import jax
        templates = self.templates()
        data = templates[0] + 0.5
        traced = jax.jit(lambda d, t: matched_filter_fft(d, t, psd=None, sample_rate_hertz=self.RATE))(
            data, templates)
        eager = matched_filter_fft(data, templates, psd=None, sample_rate_hertz=self.RATE)
        np.testing.assert_allclose(np.asarray(traced), np.asarray(eager), rtol=1e-5, atol=1e-6)

    def test_the_array_limit_is_documented_where_users_read_it(self):
        from gravyflow.src.dataset.tools import snr as gf_snr
        for function in (gf_snr.template_sigma, gf_snr.optimal_snr, gf_snr.matched_filter_fft):
            assert "jax.jit" in function.__doc__, function.__name__
        api = (Path(__file__).resolve().parents[2] / "docs" / "api.md").read_text()
        for name in ("matched_filter_fft", "optimal_snr"):
            entry = next(line for line in api.splitlines() if line.startswith(f"- `{name}`"))
            assert "jax.jit" in entry, entry
