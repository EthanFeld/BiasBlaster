import numpy as np

from biasblaster.channel import (
    ChannelMode,
    NoiseChannelApplication,
    ObservableImpactBatch,
    amplitude_damping_ptm,
    channel_delta,
    computational_readout_ptm,
    dephasing_ptm,
    depolarizing_ptm,
    estimate_observable_impacts,
    pauli_error_ptm,
    propagate_noisy_observables,
)
from biasblaster.krylov import (
    KrylovObservableSpec,
    build_krylov_error_model,
    debias_krylov_matrices,
    estimate_energy_impact,
    optimal_shot_allocation,
    optimize_energy_bias_controls,
    recommended_overlap_floor,
    solve_krylov_generalized_eigenproblem,
)
from biasblaster.model import CircuitOperation


def _h_gate():
    return CircuitOperation(
        "h", (0,), np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    )


def test_dephasing_first_order_matches_single_finite_channel():
    circuit = [_h_gate()]
    initial = {"I": 1.0, "Z": 1.0}
    observables = {"X": {"X": 1.0}}
    probability = 0.07
    ptm = dephasing_ptm(probability)
    impacts = estimate_observable_impacts(
        circuit,
        initial,
        observables,
        [ChannelMode("phase", 1, (0,), channel_delta(ptm))],
        1,
    )
    exact = propagate_noisy_observables(
        circuit,
        initial,
        observables,
        [NoiseChannelApplication("phase", 1, (0,), ptm)],
        1,
    )
    assert np.allclose(impacts.predicted, [exact["X"]], atol=1e-12)
    assert np.allclose(impacts.ideal, [1.0], atol=1e-12)
    assert np.allclose(impacts.bias, [-2 * probability], atol=1e-12)


def test_amplitude_damping_captures_nonunital_shift():
    gamma = 0.1
    initial = {"I": 1.0, "Z": -1.0}
    observables = {"Z": {"Z": 1.0}}
    ptm = amplitude_damping_ptm(gamma)
    impacts = estimate_observable_impacts(
        [],
        initial,
        observables,
        [ChannelMode("t1", 0, (0,), channel_delta(ptm))],
        1,
    )
    assert np.allclose(impacts.predicted, [-1 + 2 * gamma])


def test_mode_covariance_propagates_to_observable_covariance():
    initial = {"I": 1.0, "X": 1.0}
    observables = {"X": {"X": 1.0}}
    derivative = channel_delta(dephasing_ptm(0.01)) / 0.01
    modes = [
        ChannelMode("a", 0, (0,), derivative, mean=0.01),
        ChannelMode("b", 0, (0,), derivative, mean=0.02),
    ]
    sigma = np.array([[1e-6, 5e-7], [5e-7, 4e-6]])
    impacts = estimate_observable_impacts(
        [], initial, observables, modes, 1, mode_covariance=sigma
    )
    jacobian = impacts.jacobian[0]
    assert np.allclose(impacts.covariance[0, 0], jacobian @ sigma @ jacobian)


def test_noise_channel_helpers_have_expected_action():
    depolarizing = depolarizing_ptm(0.03)
    assert depolarizing.shape == (4, 4)
    assert np.isclose(depolarizing[0, 0], 1.0)

    bit_flip = pauli_error_ptm({"X": 0.1})
    assert np.isclose(bit_flip[3, 3], 0.8)

    readout = computational_readout_ptm(0.1, 0.2)
    assert np.isclose(readout[3, 0] + readout[3, 3], 0.8)


def test_krylov_energy_bias_and_matrix_debiasing():
    impacts = ObservableImpactBatch(
        names=("h00", "s00"),
        ideal=np.array([2.0, 1.0]),
        bias=np.array([0.1, 0.02]),
        predicted=np.array([2.1, 1.02]),
        jacobian=np.array([[1.0, 0.0], [0.0, 1.0]]),
        mode_names=("dh", "ds"),
        mode_means=np.array([0.1, 0.02]),
        covariance=None,
        mode_covariance=None,
        forward_dropped_l2=0.0,
        backward_dropped_l2=np.zeros(2),
    )
    specs = [
        KrylovObservableSpec("h00", "H", 0, 0),
        KrylovObservableSpec("s00", "S", 0, 0),
    ]
    model = build_krylov_error_model(impacts, specs, 1)
    impact = estimate_energy_impact(
        np.array([[2.0]]), np.array([[1.0]]), model
    )
    assert np.isclose(impact.predicted_bias, 0.1 - 2 * 0.02)
    h_corrected, s_corrected = debias_krylov_matrices(
        np.array([[2.1]]), np.array([[1.02]]), model
    )
    assert np.allclose(h_corrected, [[2.0]])
    assert np.allclose(s_corrected, [[1.0]])


def test_noise_aware_overlap_floor_and_regularized_solver():
    impacts = ObservableImpactBatch(
        names=("s00",),
        ideal=np.array([1.0]),
        bias=np.array([0.05]),
        predicted=np.array([1.05]),
        jacobian=np.array([[1.0]]),
        mode_names=("s",),
        mode_means=np.array([0.05]),
        covariance=None,
        mode_covariance=None,
        forward_dropped_l2=0.0,
        backward_dropped_l2=np.zeros(1),
    )
    model = build_krylov_error_model(
        impacts,
        [KrylovObservableSpec("s00", "S", 0, 0)],
        2,
        mode_covariance=np.array([[0.0004]]),
    )
    assert recommended_overlap_floor(model) >= 0.07 - 1e-12

    hamiltonian = np.diag([0.5, 1.0])
    overlap = np.diag([1.0, 1e-6])
    result = solve_krylov_generalized_eigenproblem(
        hamiltonian, overlap, overlap_floor=1e-4
    )
    assert result.retained_overlap_rank == 1
    assert np.isclose(result.energy, 0.5)


def test_sensitivity_weighted_shots_and_energy_control_reduce_cost_function():
    allocation = optimal_shot_allocation(
        [10.0, 1.0], [1.0, 1.0], 110, minimum_shots=1
    )
    assert allocation.sum() == 110
    assert allocation[0] > allocation[1]

    result = optimize_energy_bias_controls(
        [0.1, 0.2], np.eye(2), [1.0, 2.0], budget=0.05
    )
    assert np.linalg.norm(result.controls) <= 0.05 + 1e-12
    assert abs(result.predicted_after) < abs(result.predicted_before)
