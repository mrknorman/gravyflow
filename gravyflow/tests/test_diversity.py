"""
Label tracking reads one class per example, in the layout the batch contract
documents for its key.
"""
import numpy as np
import pytest

import gravyflow as gf


class _Batches:
    """Two batches of per-detector class labels: a glitch's own detector holds its class, a partner -1."""

    labels = [np.array([[3.0, -1.0], [-1.0, 5.0], [-1.0, -1.0]], np.float32),
              np.array([[-1.0, 0.0]], np.float32)]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {}, {"SUB_TYPE": self.labels[index], "POOL_LABEL": np.array([2.0, 0.0, 1.0], np.float32)}


@pytest.mark.parametrize("index, expected", [(0, [3.0, 5.0, -1.0]), (1, [0.0])])
def test_label_tracking_reads_the_class_of_each_example(index, expected):
    tracker = gf.LabelTrackingDataset(_Batches(), label_key="SUB_TYPE")
    tracker[index]
    np.testing.assert_array_equal(tracker._last_labels, expected)


def test_label_tracking_keeps_one_label_per_example():
    tracker = gf.LabelTrackingDataset(_Batches(), label_key="POOL_LABEL")
    tracker[0]
    np.testing.assert_array_equal(tracker._last_labels, [2.0, 0.0, 1.0])
