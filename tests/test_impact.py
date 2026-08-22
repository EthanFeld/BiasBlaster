import numpy as np
import pytest

from biasblaster.channel import ObservableImpactBatch
from biasblaster.impact import combine_observable_impacts


def _batch(names, mode_names, mode_means, jacobian, ideal):
    ideal = np.asarray(ideal, dtype=float)
    jacobian = np.asarray(jacobian, dtype=float)
    means = np.asarray(mode_means, dtype=float)
    bias = jacobian @ means
    return ObservableImpactBatch(
        names=tuple(names),
        ideal=ideal,
        bias=bias,
        predicted=ideal + bias,
        jacobian=jacobian,
        mode_names=tuple(mode_names),
        mode_means=means,
        covariance=None,
        mode_covariance=None,
        forward_dropped_l2=0.0,
        backward_dropped_l2=np.zeros(len(names)),
    )


def test_combine_observable_impacts_aligns_shared_calibration_modes():
    h_batch = _batch(
        ("h01",),
        ("drift", "gate_a"),
        (0.01, 0.02),
        [[2.0, 3.0]],
        [0.4],
    )
    s_batch = _batch(
        ("s01",),
        ("drift", "gate_b"),
        (0.01, 0.04),
        [[5.0, 7.0]],
        [0.1],
    )
    sigma = np.array(
        [
            [1e-4, 2e-5, 0.0],
            [2e-5, 3e-4, 0.0],
            [0.0, 0.0, 4e-4],
        ]
    )

    combined = combine_observable_impacts(
        [h_batch, s_batch], mode_covariance=sigma
    )

    assert combined.names == ("h01", "s01")
    assert combined.mode_names == ("drift", "gate_a", "gate_b")
    expected_jacobian = np.array([[2.0, 3.0, 0.0], [5.0, 0.0, 7.0]])
    expected_means = np.array([0.01, 0.02, 0.04])
    assert np.allclose(combined.jacobian, expected_jacobian)
    assert np.allclose(combined.mode_means, expected_means)
    assert np.allclose(combined.bias, expected_jacobian @ expected_means)
    assert np.allclose(
        combined.covariance,
        expected_jacobian @ sigma @ expected_jacobian.T,
    )
    assert combined.covariance[0, 1] != 0.0


def test_combine_observable_impacts_rejects_inconsistent_shared_mean():
    first = _batch(("a",), ("shared",), (0.01,), [[1.0]], [0.0])
    second = _batch(("b",), ("shared",), (0.02,), [[1.0]], [0.0])

    with pytest.raises(ValueError, match="inconsistent calibrated mean"):
        combine_observable_impacts([first, second])
