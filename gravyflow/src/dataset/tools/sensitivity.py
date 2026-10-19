"""A detector's sensitivity from its noise spectrum: the BNS range.

The BNS range is LIGO's SenseMonitor range, as LALSimulation's
XLALMeasureStandardSirenSenseMonitorRange computes it: the distance at which a
1.4 + 1.4 solar-mass binary neutron star, averaged over sky position and
orientation, has an optimal SNR of 8 in one detector, with the leading-order
(Newtonian) inspiral spectrum |h~(f)| ~ f^(-7/6) up to the innermost stable
circular orbit of the total mass, and no cosmology. Its optimally located and
oriented distance is the horizon,

    D_h = (1 / 8) * sqrt(4 * integral |h~_1Mpc(f)|^2 / S(f) df),
    |h~_1Mpc(f)| = sqrt(5 / 24) * pi^(-2/3) * (c / 1 Mpc) * (G M_c / c^3)^(5/6) * f^(-7/6),

and the range is D_h / 2.2648, the average over sky and orientation.
"""
import numpy as np

from gravyflow.src.dataset.tools.psd import PowerSpectrum

# LAL's constants (lal.MTSUN_SI, lal.C_SI, lal.PC_SI).
_SOLAR_MASS_SECONDS = 4.925490947641267e-06
_SPEED_OF_LIGHT_M_S = 299792458.0
_MEGAPARSEC_M = 3.085677581491367e22

BNS_MASS_MSUN = 1.4
BNS_RANGE_SNR = 8.0
# The horizon over the range: the sky- and orientation-averaged amplitude.
HORIZON_OVER_RANGE = 2.2648
# Each Welch bin of the PSD is integrated on this many points, so the steep
# f^(-7/3) weight is integrated exactly enough at any PSD resolution.
_POINTS_PER_BIN = 16


def bns_range_mpc(spectrum: PowerSpectrum) -> np.ndarray:
    """
    The BNS range, in Mpc, of each row of ``spectrum``: float32, of the spectrum's
    leading shape (a (B, I) batch's spectrum gives (B, I)).

    The integral runs over the spectrum's band, from its lower edge to its upper
    edge (the Nyquist frequency when unset) or the binary's innermost stable
    circular orbit (1570 Hz), whichever is lower, on the inverse PSD the SNR
    weighs by (``PowerSpectrum.inverse``, without the estimate's inverse bias),
    linearly interpolated within each bin. A band without a lower edge is
    refused: the range diverges there.
    """
    config = spectrum.config
    if config.lower_frequency_hertz is None:
        raise ValueError(
            "The BNS range needs a band with a lower edge; the spectrum's "
            "SpectralConfig has lower_frequency_hertz=None."
        )
    sample_rate = spectrum.sample_rate_hertz
    total_mass_seconds = 2.0 * BNS_MASS_MSUN * _SOLAR_MASS_SECONDS
    isco = 1.0 / (6.0 ** 1.5 * np.pi * total_mass_seconds)
    upper = sample_rate / 2.0 if config.upper_frequency_hertz is None else config.upper_frequency_hertz
    upper = min(upper, isco)

    num_samples = spectrum.nperseg * _POINTS_PER_BIN
    frequencies = np.arange(num_samples // 2 + 1) * (sample_rate / num_samples)
    in_band = (frequencies >= config.lower_frequency_hertz) & (frequencies <= upper)
    frequencies = frequencies[in_band]
    # 1 / S in normalised units: S = normalised * scale^2 / fs (psd.py).
    inverse = np.asarray(spectrum.inverse(num_samples), dtype=np.float64)[..., in_band]
    scale = np.asarray(spectrum.scale, dtype=np.float64)
    integral = np.trapezoid(frequencies ** (-7.0 / 3.0) * inverse, frequencies, axis=-1)
    integral = integral * sample_rate / scale[..., 0] ** 2

    chirp_mass_seconds = BNS_MASS_MSUN * 2.0 ** (-1.0 / 5.0) * _SOLAR_MASS_SECONDS
    amplitude = (
        np.sqrt(5.0 / 24.0) * np.pi ** (-2.0 / 3.0)
        * (_SPEED_OF_LIGHT_M_S / _MEGAPARSEC_M) * chirp_mass_seconds ** (5.0 / 6.0)
    )
    horizon_mpc = amplitude * np.sqrt(4.0 * integral) / BNS_RANGE_SNR
    return (horizon_mpc / HORIZON_OVER_RANGE).astype(np.float32)
