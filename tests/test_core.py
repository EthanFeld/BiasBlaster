from __future__ import annotations

import numpy as np

from biasblaster import (
    CircuitOperation,
    CliffordTableau,
    estimate_nearest_clifford_transport,
    estimate_pauli_transport,
    local_pauli_transfer,
    named_clifford,
    optimize_bias_controls,
    pauli_matrix,
    simulate_piecewise_constant_pulse,
)


def _rz(angle: float) -> np.ndarray:
    return np.diag([np.exp(-0.5j * angle), np.exp(0.5j * angle)])


def _cx() -> np.ndarray:
    return np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
        dtype=complex,
    )


def test_tableau_conjugation_maps_known_paulis_and_signs() -> None:
    h = named_clifford("H", 1, (0,))
    s = named_clifford("S", 1, (0,))
    cnot = named_clifford("CNOT", 2, (0, 1))
    assert h.conjugate("X").label == "Z"
    assert s.conjugate("X").label == "Y"
    assert s.conjugate("Y").sign == -1
    assert cnot.conjugate("IX").label == "XX"
    assert cnot.conjugate("ZI").label == "ZZ"
    assert isinstance(CliffordTableau.identity(1), CliffordTableau)


def test_pauli_branch_support_cap_keeps_heavy_terms_and_reports_bound() -> None:
    operations = (
        CircuitOperation("source", (0,), np.eye(2)),
        CircuitOperation("rz", (0,), _rz(0.7), 0.7),
    )
    uncapped = estimate_pauli_transport(operations, 1)
    capped = estimate_pauli_transport(operations, 1, support_cap=1)
    assert uncapped.support >= 2
    assert capped.support == 1
    assert capped.truncation_bound > 0.0
    assert capped.truncation_bound_for(np.ones((2, 3))) > 0.0
    assert local_pauli_transfer(operations[1]).shape == (4, 4)


def test_nearest_clifford_is_exact_for_cliffords_and_selects_s_for_rz() -> None:
    identity = CircuitOperation("source", (0,), np.eye(2))
    h = CircuitOperation("h", (0,), np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2))
    exact = estimate_nearest_clifford_transport((identity, h), 1)
    assert exact.mode_labels[0][0] == "Z"
    assert exact.signs[0, 0] == 1.0

    near_s = CircuitOperation("rz", (0,), _rz(np.pi / 2 - 0.01), np.pi / 2 - 0.01)
    estimate = estimate_nearest_clifford_transport((identity, near_s), 1)
    assert estimate.mode_labels[0] == ("Y", "X", "Z")
    assert tuple(estimate.signs[0]) == (1.0, -1.0, 1.0)
    original = near_s.unitary.copy()
    estimate_nearest_clifford_transport((near_s,), 1)
    assert np.array_equal(near_s.unitary, original)


def test_optimizer_reduces_bias_without_exceeding_effort_budget() -> None:
    operation = CircuitOperation("source", (0,), np.eye(2))
    transport = estimate_pauli_transport((operation,), 1)
    generators = np.array([[1.0, 0.0, 0.0]])
    jacobians = np.array([[[1.0], [0.0], [0.0]]])
    result = optimize_bias_controls(generators, transport, jacobians, effort_fraction=0.5)
    assert result.predicted_after_norm < result.predicted_before_norm
    assert result.control_norm <= result.budget + 1e-10
    assert np.allclose(result.updated_generators, [[0.5, 0.0, 0.0]], atol=1e-8)


def test_pulse_derivative_matches_one_qubit_finite_difference() -> None:
    pulse = np.array([[0.2]])
    drift = 0.1 * pauli_matrix("Z")
    control = (pauli_matrix("X"),)
    result = simulate_piecewise_constant_pulse(pulse, drift, control, 0.7)
    epsilon = 1e-6
    plus = simulate_piecewise_constant_pulse(pulse + epsilon, drift, control, 0.7).unitary
    minus = simulate_piecewise_constant_pulse(pulse - epsilon, drift, control, 0.7).unitary
    finite_difference = (plus - minus) / (2 * epsilon)
    assert np.linalg.norm(finite_difference - result.derivatives[0]) < 1e-8


def test_integration_pauli_update_is_no_worse_than_nearest_clifford() -> None:
    operations = (
        CircuitOperation("source", (0,), np.eye(2)),
        CircuitOperation("rz", (0,), _rz(0.42), 0.42),
        CircuitOperation("cx", (0, 1), _cx()),
        CircuitOperation("rz", (1,), _rz(-0.31), -0.31),
    )
    generators = np.array(
        [[0.20, -0.10, 0.05], [0.04, 0.02, -0.03], [0.03, -0.02, 0.01], [-0.07, 0.06, 0.02]],
    )
    jacobians = np.zeros((4, 3, 3), dtype=float)
    jacobians[:, :, :] = np.eye(3)
    pauli_transport = estimate_pauli_transport(operations, 2)
    clifford_transport = estimate_nearest_clifford_transport(operations, 2)
    pauli_result = optimize_bias_controls(generators, pauli_transport, jacobians, effort_fraction=0.2)
    clifford_result = optimize_bias_controls(generators, clifford_transport, jacobians, effort_fraction=0.2)
    exact = estimate_pauli_transport(operations, 2)
    pauli_after = np.linalg.norm(list(exact.contract(pauli_result.updated_generators).values()))
    clifford_after = np.linalg.norm(list(exact.contract(clifford_result.updated_generators).values()))
    assert pauli_result.budget == clifford_result.budget
    assert pauli_after <= clifford_after + 1e-10
