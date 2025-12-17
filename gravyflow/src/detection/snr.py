"""
SNR Computation for Matched Filtering

JAX-optimized functions for computing signal-to-noise ratio via 
frequency-domain correlation.

Shape Conventions:
- Data: (Samples,) for 1D or (Batch, Samples) for batched
- Templates: (NumTemplates, Samples)
- SNR output: (Batch, NumTemplates, Samples) or (NumTemplates, Samples) if 1D input
"""

import jax
import jax.numpy as jnp
from typing import Optional

# Shape helpers - using explicit ndim checks for JIT compatibility
# Note: Can't use ShapeEnforcer inside JIT, so we use raw ndim checks with comments


@jax.jit
def matched_filter_fft(
    data: jnp.ndarray,
    templates: jnp.ndarray,
    psd: Optional[jnp.ndarray] = None,
    sample_rate_hertz: float = 8192.0,
) -> jnp.ndarray:
    """
    FFT-based matched filtering with GPU parallelization.
    
    Computes the matched filter SNR timeseries for multiple templates:
    
        SNR(t) = 4 * Re[IFFT(d_tilde * h_tilde* / S_n)]
    
    If no PSD is provided, assumes white noise (uniform PSD).
    
    Args:
        data: Strain data, shape (Samples,) or (Batch, Samples)
        templates: Template waveforms, shape (NumTemplates, Samples)
        psd: One-sided PSD, shape (NumFreqs,). If None, uniform weighting.
        sample_rate_hertz: Sample rate for frequency normalization.
    
    Returns:
        SNR timeseries, shape (Batch, NumTemplates, Samples) or 
        (NumTemplates, Samples) if data is 1D
    """
    # Shape: data = (Samples,) for 1D, (Batch, Samples) for batched
    data_1d = data.ndim == 1  # Check if input is 1D (single sample)
    if data_1d:
        # Expand: (Samples,) -> (1, Samples) for batch dimension
        data = data[None, :]
    
    # Shape: data = (Batch, Samples) at this point
    n_samples = data.shape[-1]  # Axis -1 = Samples
    
    # FFT of data and templates - normalize by sample_rate to match dataset convention
    # This gives h_tilde(f) ≈ rfft(h) / sample_rate, consistent with gravyflow.src.dataset.tools.snr
    # Shape: data_fft = (Batch, NumFreqs)
    data_fft = jnp.fft.rfft(data, axis=-1) / sample_rate_hertz
    # Shape: template_fft = (NumTemplates, NumFreqs)
    template_fft = jnp.fft.rfft(templates, axis=-1) / sample_rate_hertz

    
    # Duration in seconds
    duration_seconds = n_samples / sample_rate_hertz
    
    # Frequency bin width (not used directly after normalization simplification)
    df = sample_rate_hertz / n_samples
    
    # Apply PSD weighting if provided
    if psd is not None:
        # Ensure PSD has correct length
        n_freq = data_fft.shape[-1]
        if psd.shape[-1] != n_freq:
            # Interpolate or truncate
            psd = jnp.interp(
                jnp.arange(n_freq) * df,
                jnp.arange(len(psd)) * df,
                psd
            )
        
        # Avoid division by zero
        psd = jnp.maximum(psd, 1e-50)
        weight = 1.0 / psd
    else:
        weight = 1.0
    
    # Cross-correlation in frequency domain
    # (batch, 1, freq) * (1, templates, freq) * weight
    corr_fft = (
        data_fft[:, None, :] * 
        jnp.conj(template_fft)[None, :, :] * 
        weight
    )
    
    # IFFT to get correlation timeseries
    # irfft returns the inverse FFT (real part)
    corr = jnp.fft.irfft(corr_fft, n=n_samples, axis=-1)
    
    # Normalize to compute SNR:
    # The matched filter SNR is: rho = <d|h> / sqrt(<h|h>)
    # 
    # Inner product formula: <d|h> = 4 * integral(d_tilde * conj(h_tilde) / Sn * df)
    # 
    # With FFT normalized by sample_rate (h_tilde = rfft/sr):
    #   corr_fft = d_tilde * h_tilde* = (D/sr) * (H/sr)* = D*H* / sr^2
    #   corr = irfft(corr_fft) = (1/n) * sum(D*H* / sr^2)
    # 
    # The inner product is:
    #   <d|h> = 4 * sum(d_tilde * h_tilde* / Sn) * df
    #         = 4 * sum(D*H*/sr^2 / Sn) * df
    #         = 4 * df * sum(D*H* / Sn) / sr^2
    #         = 4 * (sr/n) * n * corr * sr^2 / 1  (since corr includes 1/n and 1/sr^2)
    #         = 4 * sr * corr * sr^2 / sr^2 = 4 * sr * corr??
    # 
    # Wait, let's be more careful:
    #   corr[t] = irfft(d_tilde * h_tilde*) = (1/n) * sum_k(d_tilde[k] * h_tilde*[k])
    #   <d|h> at lag 0 = 4 * sum_k(d_tilde * h_tilde* / Sn) * df
    #   With Sn=1 (white): <d|h> = 4 * df * sum = 4 * df * n * corr[0]
    #                            = 4 * (sr/n) * n * corr = 4 * sr * corr
    # 
    # Sigma^2 = 4/T * sum(|h_tilde|^2) with h_tilde = rfft/sr
    # For autocorrelation (d=h), <h|h>_peak = 4 * sr * corr_peak
    # But corr = irfft(|h_tilde|^2) = (1/n) * sum(|H/sr|^2) = sum(|H|^2)/(n*sr^2)
    # So 4 * sr * corr = 4 * sr * sum(|H|^2) / (n * sr^2) = 4 * sum(|H|^2) / (n * sr)
    # 
    # Sigma^2 = 4/T * sum(|H/sr|^2) = 4*sr/n * sum(|H|^2)/sr^2 = 4 * sum(|H|^2) / (n * sr)
    # 
    # Great! So 4*sr*corr should equal sigma^2. Let's verify the test showed 2x because
    # of something else... Actually looking at test: 4*sr*corr = 1.999, sigma^2 = 0.999
    # The factor of 2 is because corr_peak = sum(|fft|^2)/n (no *2 for one-sided), but
    # sigma includes both positive and negative frequencies implicitly.
    # 
    # For rfft (one-sided), we need to account for this:
    # Actually the issue is:
    # - rfft gives F[k] for k=0..n/2 
    # - irfft expects these and returns real signal
    # - The power is 2x for k>0 (one-sided), but irfft handles this internally
    # 
    # The fix: use 2*sr*corr for one-sided to match sigma convention (4/T factor)
    inner_product = (2.0 * sample_rate_hertz) * corr
    
    sigma = template_sigma(templates, psd, sample_rate_hertz)
    
    # Avoid division by zero
    sigma = jnp.maximum(sigma, 1e-50)
    
    snr = inner_product / sigma[None, :, None]
    
    if data_1d:
        snr = snr[0]  # Remove batch dimension
    
    return jnp.abs(snr)


@jax.jit
def template_sigma(
    templates: jnp.ndarray,
    psd: Optional[jnp.ndarray] = None,
    sample_rate_hertz: float = 8192.0,
) -> jnp.ndarray:
    """
    Compute optimal SNR normalization (sigma) for templates.
    
    sigma = sqrt(4 * integral(|h_tilde|^2 / S_n * df))
    
    Args:
        templates: Template waveforms, shape (num_templates, samples)
        psd: One-sided PSD. If None, uniform weighting.
        sample_rate_hertz: Sample rate.
    
    Returns:
        Sigma values, shape (num_templates,)
    """
    n_samples = templates.shape[-1]
    duration_seconds = n_samples / sample_rate_hertz
    df = sample_rate_hertz / n_samples
    
    # Normalize FFT by sample_rate to match dataset convention
    # h_tilde(f) ≈ rfft(h) / sample_rate
    template_fft = jnp.fft.rfft(templates, axis=-1) / sample_rate_hertz
    
    if psd is not None:
        n_freq = template_fft.shape[-1]
        if psd.shape[-1] != n_freq:
            psd = jnp.interp(
                jnp.arange(n_freq) * df,
                jnp.arange(len(psd)) * df,
                psd
            )
        psd = jnp.maximum(psd, 1e-50)
        weight = 1.0 / psd
    else:
        weight = 1.0
    
    # Inner product: <h|h> = 4 * integral(|h_tilde(f)|^2 / S_n * df)
    #
    # With normalized FFT (h_tilde = rfft/sample_rate):
    #   <h|h> = 4 * sum(|h_tilde|^2 / Sn) * df
    #         = 4 * sum(...) * (1/duration)
    #         = 4/T * sum(|h_tilde|^2 / Sn)
    #
    # This matches gravyflow.src.dataset.tools.snr.py convention exactly.
    sigma_sq = (4.0 / duration_seconds) * jnp.sum(jnp.abs(template_fft) ** 2 * weight, axis=-1)
    
    return jnp.sqrt(sigma_sq)


@jax.jit
def optimal_snr(
    signal: jnp.ndarray,
    psd: Optional[jnp.ndarray] = None,
    sample_rate_hertz: float = 8192.0,
) -> float:
    """
    Compute optimal (matched filter) SNR for a signal.
    
    This is the SNR you would recover if the template exactly matches
    the signal in the data.
    
    Args:
        signal: Signal waveform, shape (samples,)
        psd: One-sided PSD. If None, uniform weighting.
        sample_rate_hertz: Sample rate.
    
    Returns:
        Optimal SNR (scalar)
    """
    sigma = template_sigma(signal[None, :], psd, sample_rate_hertz)
    return sigma[0]


def find_triggers(
    snr: jnp.ndarray,
    threshold: float = 8.0,
    cluster_window: int = 100,
) -> list:
    """
    Find triggers (peaks) above SNR threshold.
    
    Args:
        snr: SNR timeseries, shape (NumTemplates, Samples) for 2D or 
             (Batch, NumTemplates, Samples) for 3D
        threshold: Minimum SNR for trigger
        cluster_window: Samples to cluster nearby peaks
    
    Returns:
        List of trigger dictionaries with keys:
        - 'snr': Peak SNR value
        - 'time_index': Sample index of peak
        - 'template_index': Which template triggered
    """
    # Shape check: determine if 2D (NumTemplates, Samples) or 3D (Batch, NumTemplates, Samples)
    if snr.ndim == 2:
        # Shape: (NumTemplates, Samples) - single batch
        # Axis 0 = NumTemplates, Axis 1 = Samples
        max_snr_over_templates = jnp.max(snr, axis=0)  # -> (Samples,)
        best_template = jnp.argmax(snr, axis=0)        # -> (Samples,)
    else:
        # Shape: (Batch, NumTemplates, Samples) - batched
        # Axis 0 = Batch, Axis 1 = NumTemplates, Axis 2 = Samples
        max_snr_over_templates = jnp.max(snr, axis=1)  # -> (Batch, Samples)
        best_template = jnp.argmax(snr, axis=1)        # -> (Batch, Samples)
    
    triggers = []
    
    # Simple peak finding (convert to numpy for iteration)
    import numpy as np
    snr_np = np.array(max_snr_over_templates)
    template_np = np.array(best_template)
    
    above_thresh = snr_np > threshold
    # Shape-dependent indexing
    indices = np.where(above_thresh)[0] if snr.ndim == 2 else np.argwhere(above_thresh)
    
    # Cluster nearby peaks
    i = 0
    while i < len(indices):
        if snr.ndim == 2:
            idx = indices[i]
            cluster_end = i + 1
            while cluster_end < len(indices) and indices[cluster_end] - indices[i] < cluster_window:
                cluster_end += 1
            
            # Find peak in cluster
            cluster_indices = indices[i:cluster_end]
            peak_idx = cluster_indices[np.argmax(snr_np[cluster_indices])]
            
            triggers.append({
                'snr': float(snr_np[peak_idx]),
                'time_index': int(peak_idx),
                'template_index': int(template_np[peak_idx]),
            })
            
            i = cluster_end
        else:
            # Handle batch case
            batch_idx, time_idx = indices[i]
            triggers.append({
                'snr': float(snr_np[batch_idx, time_idx]),
                'time_index': int(time_idx),
                'template_index': int(template_np[batch_idx, time_idx]),
                'batch_index': int(batch_idx),
            })
            i += 1
    
    return triggers
