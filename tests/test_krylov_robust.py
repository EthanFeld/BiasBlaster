import numpy as np

from biasblaster.channel import ObservableImpactBatch
from biasblaster.krylov_robust import shrinkage_weights, with_shrunk_observable_bias


def test_shrinkage_tracks_bias_signal_to_uncertainty():
    weights = shrinkage_weights(
        np.array([1.0, 0.1, 0.0]),
        np.array([0.01, 1.0, 0.0]),
    )
    assert weights[0] > 0.98
    assert weights[1] < 0.02
    assert weights[2] == 0.0


def test_shrunk_impact_retains_covariance_and_reduces_only_bias():
    covariance = np.diag([0.01, 1.0])
    impacts = ObservableImpactBatch(
        names=("a", "b"),
        ideal=np.array([0.0, 0.0]),
        bias=np.array([1.0, 0.1]),
        predicted=np.array([1.0, 0.1]),
        jacobian=np.eye(2),
        mode_names=("m0", "m1"),
        mode_means=np.array([1.0, 0.1]),
        covariance=covariance,
        mode_covariance=np.eye(2),
        forward_dropped_l2=0.0,
        backward_dropped_l2=np.zeros(2),
    )
    adjusted, result = with_shrunk_observable_bias(impacts)
    assert np.allclose(adjusted.covariance, covariance)
    assert np.allclose(adjusted.jacobian, impacts.jacobian)
    assert adjusted.bias[0] > 0.98
    assert adjusted.bias[1] < 0.002
    assert np.allclose(adjusted.bias, result.correction)
