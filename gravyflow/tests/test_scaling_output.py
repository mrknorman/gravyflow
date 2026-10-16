"""
Tests for ScalingTypes (SNR, HPEAK, HRSS) as return variables.

The labels are made by the injection layer (InjectionGenerator.
add_injections_to_onsource) and passed through by create_variable_dictionary,
never recomputed there. Their values are tested in test_amplitude.py.
"""

import numpy as np

import gravyflow as gf
from gravyflow.src.dataset.dataset import create_variable_dictionary


def _dictionary(return_variables, injection_parameters, onsource=None):
    return create_variable_dictionary(
        return_variables=return_variables,
        onsource=onsource,
        whitened_onsource=None,
        offsource=None,
        gps_times=None,
        injections=None,
        whitened_injections=None,
        mask=None,
        rolling_pearson_onsource=None,
        spectrogram_onsource=None,
        injection_parameters=injection_parameters,
    )


class TestScalingTypesOutput:
    """Tests for ScalingTypes as return variables."""

    def test_labels_are_passed_through(self):
        """Each requested ScalingType is the injection layer's label, unchanged."""
        labels = {
            gf.ScalingTypes.SNR: np.array([8.0, np.nan, 12.0], np.float32),
            gf.ScalingTypes.HRSS: np.array([1e-21, np.nan, 3e-21], np.float32),
            gf.ScalingTypes.HPEAK: np.array([2e-21, np.nan, 6e-21], np.float32),
        }
        result = _dictionary(list(labels), labels)

        for scaling_type, value in labels.items():
            assert result[scaling_type.name] is value

    def test_only_requested_labels_are_returned(self):
        """Labels not requested are not returned."""
        labels = {
            gf.ScalingTypes.SNR: np.array([8.0], np.float32),
            gf.ScalingTypes.HRSS: np.array([1e-21], np.float32),
        }
        result = _dictionary([gf.ScalingTypes.SNR], labels)

        assert set(result) == {"SNR"}

    def test_labels_mix_with_return_variables(self):
        """ScalingTypes can be requested with ReturnVariables."""
        onsource = np.zeros((2, 1, 16), np.float32)
        labels = {gf.ScalingTypes.SNR: np.array([8.0, 9.0], np.float32)}
        result = _dictionary(
            [gf.ReturnVariables.ONSOURCE, gf.ScalingTypes.SNR], labels, onsource=onsource
        )

        assert result["ONSOURCE"] is onsource
        np.testing.assert_array_equal(result["SNR"], [8.0, 9.0])

    def test_absent_labels_are_not_computed(self):
        """A ScalingType the injection layer did not label is not invented."""
        result = _dictionary([gf.ScalingTypes.SNR, gf.ScalingTypes.HRSS], {})

        assert "SNR" not in result
        assert "HRSS" not in result
