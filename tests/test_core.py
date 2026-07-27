from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import expm, expm_frechet

from biasblaster import (
    CircuitOperation,
    CliffordTableau,
    calibrate_pulse_error_model,
    estimate_hybrid_pauli_transport,
    estimate_nearest_clifford_transport,
    estimate_pauli_transport,
    local_pauli_transfer,
    named_clifford,
    nearest_clifford_process_fidelity,
    optimize_bias_controls,
    optimize_local_error_controls,
    pauli_labels,
    pauli_matrix,
    simulate_piecewise_constant_pulse,
)
from biasblaster.transport import SparsePauliTransport
from biasblaster.optimizer import _trust_region_least_squares


def _rz(angle: float) -> np.ndarray:
    return np.diag([np.exp(-0.5j * angle), np.exp(0.5j * angle)])


def _cx() -> np.ndarray:
    return np.array(
        [[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]],
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
    generators = np.ones((2, 3))
    assert capped.truncation_bound_for(generators) > 0.0
    exact_bias = uncapped.contract(generators)
    capped_bias = capped.contract(generators)
    labels = set(exact_bias) | set(capped_bias)
    omitted_norm = np.linalg.norm([
        exact_bias.get(label, 0.0) - capped_bias.get(label, 0.0)
        for label in labels
    ])
    assert omitted_norm <= capped.truncation_bound_for(generators) + 1.0e-12
    assert local_pauli_transfer(operations[1]).shape == (4, 4)


def test_hybrid_pauli_transport_bounds_nearest_clifford_tail_error() -> None:
    h = CircuitOperation("h", (0,), np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2.0))
    operations = (
        CircuitOperation("t", (0,), _rz(np.pi / 4.0), np.pi / 4.0),
        h,
        CircuitOperation("rz", (0,), _rz(0.31), 0.31),
    )
    generators = np.array([[0.3, -0.2, 0.1], [-0.1, 0.4, 0.2], [0.2, 0.1, -0.3]])
    exact = estimate_pauli_transport(operations, 1, coefficient_tol=0.0)
    capped = estimate_pauli_transport(operations, 1, support_cap=1, coefficient_tol=0.0)
    hybrid = estimate_hybrid_pauli_transport(operations, 1, support_cap=1, coefficient_tol=0.0)
    exact_bias = exact.contract(generators)
    hybrid_bias = hybrid.contract(generators)
    capped_bias = capped.contract(generators)
    labels = set(exact_bias) | set(hybrid_bias) | set(capped_bias)
    error = np.linalg.norm([
        exact_bias.get(label, 0.0) - hybrid_bias.get(label, 0.0)
        for label in labels
    ])
    capped_error = np.linalg.norm([
        exact_bias.get(label, 0.0) - capped_bias.get(label, 0.0)
        for label in labels
    ])
    # The cap applies to exact corrections; the nearest-Clifford backbone is
    # retained independently and has three final Pauli destinations here.
    assert hybrid.support >= 1
    assert error < capped_error
    assert error <= hybrid.truncation_bound_for(generators) + 1.0e-12


def test_hybrid_selection_matrix_prioritizes_optimizer_signal() -> None:
    operations = (CircuitOperation("t", (0,), _rz(np.pi / 4.0), np.pi / 4.0),)
    exact = estimate_pauli_transport(operations, 1, coefficient_tol=0.0)
    unweighted = estimate_hybrid_pauli_transport(operations, 1, support_cap=1, coefficient_tol=0.0)
    weighted = estimate_hybrid_pauli_transport(
        operations, 1, support_cap=1, coefficient_tol=0.0,
        selection_matrix=np.array([[1.0], [0.0], [0.0]]),
    )
    generators = np.array([[1.0, 0.0, 0.0]])

    def bias_error(transport: object) -> float:
        reference = exact.contract(generators)
        estimate = transport.contract(generators)  # type: ignore[union-attr]
        return float(np.linalg.norm([
            reference.get(label, 0.0) - estimate.get(label, 0.0)
            for label in set(reference) | set(estimate)
        ]))

    assert bias_error(weighted) < bias_error(unweighted)


def test_hybrid_zero_correction_support_is_nearest_clifford() -> None:
    operations = (
        CircuitOperation("t", (0,), _rz(np.pi / 4.0), np.pi / 4.0),
        CircuitOperation("h", (0,), np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2.0)),
    )
    generators = np.array([[0.3, -0.2, 0.1], [-0.1, 0.4, 0.2]])
    anchored = estimate_hybrid_pauli_transport(operations, 1, support_cap=0)
    nearest = estimate_nearest_clifford_transport(operations, 1)

    assert anchored.contract(generators) == pytest.approx(nearest.contract(generators))
    assert anchored.truncation_bound is None
    assert anchored.truncation_bound_for(generators) is None


def test_hybrid_full_correction_support_recovers_exact_pauli_transport() -> None:
    h = CircuitOperation("h", (0,), np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2.0))
    operations = (
        CircuitOperation("t", (0,), _rz(np.pi / 4.0), np.pi / 4.0),
        h,
        CircuitOperation("rz", (0,), _rz(0.31), 0.31),
    )
    generators = np.array([[0.3, -0.2, 0.1], [-0.1, 0.4, 0.2], [0.2, 0.1, -0.3]])
    exact = estimate_pauli_transport(operations, 1, coefficient_tol=0.0)
    anchored = estimate_hybrid_pauli_transport(operations, 1, support_cap=3, coefficient_tol=0.0)

    assert anchored.contract(generators) == pytest.approx(exact.contract(generators))
    assert anchored.truncation_bound == pytest.approx(0.0)


def test_coefficient_tolerance_is_included_in_truncation_bound() -> None:
    operations = (CircuitOperation("rz-q0", (0, 1), np.kron(np.eye(2), _rz(0.01)), 0.01),)
    transport = estimate_pauli_transport(
        operations,
        2,
        coefficient_tol=0.1,
    )
    generators = np.array([[1.0, 0.0, 0.0]])
    assert transport.truncation_bound_for(generators) >= np.sin(0.01)
    assert len(transport.contract(generators)) == 1


def test_circuit_operation_rejects_nonunitary_matrix() -> None:
    with pytest.raises(ValueError, match="unitary"):
        CircuitOperation("bad", (0,), np.diag([2.0, 1.0]))


def test_cached_pauli_matrices_and_operation_unitaries_are_immutable() -> None:
    with pytest.raises(ValueError):
        pauli_matrix("X")[0, 0] = 0.0

    operation = CircuitOperation("identity", (0,), np.eye(2))
    with pytest.raises(ValueError):
        operation.unitary[0, 0] = 0.0
    for angle in (np.nan, np.inf, -np.inf):
        with pytest.raises(ValueError, match="angle"):
            CircuitOperation("identity", (0,), np.eye(2), angle)


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
    assert nearest_clifford_process_fidelity(h.unitary) == pytest.approx(1.0)
    assert nearest_clifford_process_fidelity(near_s.unitary) < 1.0


def test_pre_gate_generators_propagate_through_own_gate_and_suffix() -> None:
    identity = CircuitOperation("identity", (0,), np.eye(2, dtype=complex))
    h = CircuitOperation("h", (0,), np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2.0))
    operations = (identity, h)

    exact = estimate_pauli_transport(operations, 1)
    nearest = estimate_nearest_clifford_transport(operations, 1)

    first_gate_x = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    second_gate_x = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    assert exact.contract(first_gate_x) == pytest.approx({"Z": 1.0})
    assert exact.contract(second_gate_x) == pytest.approx({"Z": 1.0})
    assert nearest.contract(first_gate_x) == pytest.approx({"Z": 1.0})
    assert nearest.contract(second_gate_x) == pytest.approx({"Z": 1.0})


def test_two_qubit_qiskit_order_matches_exact_and_clifford_transport() -> None:
    h = CircuitOperation("h", (0,), np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2.0))
    cx = CircuitOperation("cx", (0, 1), _cx())
    generators = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

    # A pre-H X error on q0 becomes Z on q0, which a CX with q0 as control
    # preserves. With q0 rightmost, that final Pauli is IZ.
    expected = {"IZ": 1.0}
    assert estimate_pauli_transport((h, cx), 2).contract(generators) == pytest.approx(expected)
    assert estimate_nearest_clifford_transport((h, cx), 2).contract(generators) == pytest.approx(expected)


def test_full_su4_transport_preserves_all_two_qubit_generator_modes() -> None:
    operation = CircuitOperation("identity", (0, 1), np.eye(4, dtype=complex))
    transport = estimate_pauli_transport((operation,), 2, two_qubit_generator_basis="full-su4")
    generators = np.zeros((1, 15))
    generators[0, pauli_labels(2)[1:].index("XY")] = 0.4
    assert transport.generator_mode_count == 15
    assert transport.contract(generators) == pytest.approx({"YX": 0.4})

    result = optimize_bias_controls(generators, transport, np.eye(15)[None], effort_fraction=0.25)
    assert result.predicted_after_norm < result.predicted_before_norm


def test_full_su4_nearest_clifford_preserves_all_two_qubit_generator_modes() -> None:
    operation = CircuitOperation("identity", (0, 1), np.eye(4, dtype=complex))
    transport = estimate_nearest_clifford_transport(
        (operation,),
        2,
        two_qubit_generator_basis="full-su4",
    )
    generators = np.zeros((1, 15))
    generators[0, pauli_labels(2)[1:].index("XY")] = 0.4
    assert transport.generator_mode_count == 15
    assert transport.contract(generators) == pytest.approx({"YX": 0.4})


def test_pauli_transport_matches_dense_pre_gate_generator_algebra() -> None:
    rz = CircuitOperation("rz", (0,), _rz(0.37), 0.37)
    h = CircuitOperation("h", (0,), np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2.0))
    operations = (rz, h)
    generators = np.array([[0.31, -0.22, 0.17], [-0.14, 0.28, 0.09]])

    expected_generator = np.zeros((2, 2), dtype=complex)
    for gate, operation in enumerate(operations):
        suffix = np.eye(2, dtype=complex)
        for later in operations[gate:]:
            suffix = later.unitary @ suffix
        local_generator = sum(
            coefficient * pauli_matrix(label)
            for coefficient, label in zip(generators[gate], ("X", "Y", "Z"), strict=True)
        )
        expected_generator += suffix @ local_generator @ suffix.conj().T
    expected = {
        label: float((np.trace(pauli_matrix(label) @ expected_generator) / 2).real)
        for label in ("X", "Y", "Z")
    }
    expected = {label: value for label, value in expected.items() if abs(value) > 1e-12}

    assert estimate_pauli_transport(operations, 1).contract(generators) == pytest.approx(expected)


def test_optimizer_reduces_bias_without_exceeding_effort_budget() -> None:
    operation = CircuitOperation("source", (0,), np.eye(2))
    transport = estimate_pauli_transport((operation,), 1)
    generators = np.array([[1.0, 0.0, 0.0]])
    jacobians = np.array([[[1.0], [0.0], [0.0]]])
    result = optimize_bias_controls(generators, transport, jacobians, effort_fraction=0.5)
    assert result.predicted_after_norm < result.predicted_before_norm
    assert result.control_norm <= result.budget + 1e-10
    assert np.allclose(result.updated_generators, [[0.5, 0.0, 0.0]], atol=1e-8)


def test_optimizer_bound_is_evaluated_on_updated_generators() -> None:
    transport = SparsePauliTransport(
        {"X": np.array([1.0, 0.0, 0.0])},
        n_qubits=1,
        gate_count=1,
        active_gate_indices=(0,),
        support_count=1,
        truncation_bound=1.0,
    )
    generators = np.array([[1.0, 0.0, 0.0]])
    jacobians = np.array([[[1.0], [100.0], [0.0]]])
    result = optimize_bias_controls(generators, transport, jacobians, effort_fraction=0.5)
    assert result.truncation_bound == pytest.approx(transport.truncation_bound_for(result.updated_generators))
    assert result.truncation_bound > transport.truncation_bound_for(generators)


def test_local_error_oracle_optimizes_without_transport() -> None:
    generators = np.array([[1.0, 0.0, 0.0]])
    jacobians = np.array([[[1.0], [0.0], [0.0]]])
    result = optimize_local_error_controls(generators, jacobians, effort_fraction=0.5)
    assert result.predicted_after_norm < result.predicted_before_norm
    assert result.control_norm <= result.budget + 1e-10
    assert np.allclose(result.updated_generators, [[0.5, 0.0, 0.0]], atol=1e-8)


def test_local_error_oracle_supports_full_su4_coordinates() -> None:
    generators = np.zeros((1, 15))
    generators[0, 14] = 1.0
    jacobians = np.zeros((1, 15, 1))
    jacobians[0, 14, 0] = 1.0
    result = optimize_local_error_controls(generators, jacobians, effort_fraction=0.5)
    assert result.predicted_after_norm < result.predicted_before_norm
    assert result.control_norm <= result.budget + 1e-10


def test_benchmark_derives_local_coordinates_and_jacobians_from_pulses() -> None:
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("benchmark_tradeoff", root / "scripts" / "benchmark_tradeoff.py")
    assert spec and spec.loader
    benchmark = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark)
    operation = CircuitOperation("identity", (0,), np.eye(2))
    seeded_error = np.array([0.02, -0.03, 0.04])
    model = benchmark._pulse_gate_model(operation, seeded_error)
    generators, jacobians = benchmark._pulse_calibration((model,))
    assert generators.shape == (1, 15)
    assert jacobians.shape == (1, 15, 3)
    assert np.allclose(generators[0, :3], seeded_error, atol=1e-8)
    assert np.allclose(generators[0, 3:], 0.0, atol=1e-8)
    assert np.allclose(jacobians[0, :3], np.eye(3), atol=1e-8)
    assert np.allclose(jacobians[0, 3:], 0.0, atol=1e-8)
    assert model.base_calibration is not None
    assert np.allclose(model.unitary(), model.base_calibration.simulation.unitary)
    # Nonlinear rescore evolves the changed pulse, rather than exponentiating
    # the optimizer's linearized local-error update.
    updated = benchmark._pulse_operations((model,), np.array([[-0.01, 0.0, 0.0]]))
    assert not np.allclose(updated[0].unitary, model.unitary())


def test_trust_region_solver_preserves_weak_well_resolved_control_directions() -> None:
    design = np.diag([1.0, 1.0e-7])
    bias = np.array([1.0, 1.0e-7])
    controls = _trust_region_least_squares(design, bias, budget=2.0)
    assert np.allclose(controls, [-1.0, -1.0], atol=1e-8)
    assert np.linalg.norm(design @ controls + bias) < 1e-12


def test_trust_region_solver_satisfies_active_constraint_kkt_conditions() -> None:
    rng = np.random.default_rng(41)
    design = rng.normal(size=(7, 4))
    bias = rng.normal(size=7)
    budget = 0.15
    controls = _trust_region_least_squares(design, bias, budget)
    gradient = design.T @ (design @ controls + bias)
    multiplier = -float(np.dot(gradient, controls)) / float(np.dot(controls, controls))

    assert np.linalg.norm(controls) == pytest.approx(budget, abs=1e-12)
    assert multiplier >= 0.0
    assert np.linalg.norm(gradient + multiplier * controls) < 1e-10


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


def test_pulse_spectral_frechet_matches_scipy_for_multiple_segments() -> None:
    pulse = np.array([[0.2, -0.1], [0.3, 0.4]])
    drift = 0.1 * pauli_matrix("Z")
    controls = (pauli_matrix("X"), pauli_matrix("Y"))
    result = simulate_piecewise_constant_pulse(pulse, drift, controls, 0.7)
    dt = 0.35
    segment_exponentials = []
    segment_derivatives = []
    for row in pulse:
        hamiltonian = drift + sum(value * control for value, control in zip(row, controls, strict=True))
        argument = -1j * hamiltonian * dt
        segment_exponentials.append(expm(argument))
        segment_derivatives.append(
            [expm_frechet(argument, -1j * control * dt, compute_expm=False) for control in controls]
        )
    assert np.allclose(result.unitary, segment_exponentials[1] @ segment_exponentials[0])
    assert np.allclose(result.derivatives[0], segment_exponentials[1] @ segment_derivatives[0][0])
    assert np.allclose(result.derivatives[1], segment_exponentials[1] @ segment_derivatives[0][1])
    assert np.allclose(result.derivatives[2], segment_derivatives[1][0] @ segment_exponentials[0])
    assert np.allclose(result.derivatives[3], segment_derivatives[1][1] @ segment_exponentials[0])


def test_pulse_error_calibration_log_jacobian_matches_finite_difference() -> None:
    pulse = np.array([[0.2]])
    drift = 0.1 * pauli_matrix("Z")
    controls = (pauli_matrix("X"),)
    calibration = calibrate_pulse_error_model(
        pulse, drift, controls, 0.7, np.eye(2), ("X", "Y", "Z")
    )
    epsilon = 1.0e-6
    plus = calibrate_pulse_error_model(
        pulse + epsilon, drift, controls, 0.7, np.eye(2), ("X", "Y", "Z")
    )
    minus = calibrate_pulse_error_model(
        pulse - epsilon, drift, controls, 0.7, np.eye(2), ("X", "Y", "Z")
    )
    finite_difference = (plus.generators - minus.generators) / (2.0 * epsilon)
    assert np.allclose(calibration.jacobians[:, 0], finite_difference, atol=1e-7)


def test_pulse_calibration_normalizes_global_phase_before_two_qubit_log() -> None:
    pulse = np.array([[0.02, -0.03, 0.04]])
    controls = tuple(pauli_matrix(label) for label in ("XX", "YY", "ZZ"))
    drift = 0.15 * pauli_matrix("IX") + 0.2 * pauli_matrix("XX")
    calibration = calibrate_pulse_error_model(
        pulse, drift, controls, 1.0, np.eye(4), ("XX", "YY", "ZZ")
    )
    assert np.allclose(np.linalg.det(calibration.relative_error), 1.0, atol=1e-10)
    assert calibration.omitted_generator_norm > 0.0


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


def test_tradeoff_benchmark_can_compare_multiple_estimators_by_fidelity(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "per_case_results.json"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/benchmark_tradeoff.py",
            "--corpus",
            "synthetic",
            "--estimator",
            "all",
            "--support-caps",
            "1",
            "8",
            "--widths",
            "1",
            "2",
            "--layers",
            "2",
            "--circuits",
            "1",
            "--seeds",
            "7",
            "11",
            "--error-models",
            "iid-coherent",
            "sparse-outliers",
            "--fidelity-probes",
            "2",
            "--output",
            str(output),
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.count("mean_sampled_haar_output_state_fidelity_after=") == 12
    assert "corpus=synthetic" in completed.stdout
    assert "two_qubit_generator_basis=full-su4" in completed.stdout
    assert "estimator=nearest-clifford" in completed.stdout
    assert "estimator=anchored-pauli" in completed.stdout
    assert "estimator=no-bias" in completed.stdout
    assert "estimator=local-error-oracle" in completed.stdout
    assert "support_cap=1" in completed.stdout
    assert "support_cap=0" in completed.stdout
    assert "support_cap=8" in completed.stdout
    assert "error_model=spatially-correlated" not in completed.stdout
    assert "widths=1,2" in completed.stdout
    assert "layers=2" in completed.stdout
    assert "mean_sampled_haar_output_state_infidelity_reduction=" in completed.stdout
    assert "mean_analytic_simulator_coordinate_model_ms=" in completed.stdout
    assert "mean_baseline_rescore_ms=" in completed.stdout
    assert "mean_post_update_rescore_ms=" in completed.stdout
    assert "mean_transport_cold_build_ms=" in completed.stdout
    assert "mean_optimization_ms=" in completed.stdout
    assert "mean_post_update_truncation_bound=" in completed.stdout
    assert "mean_post_update_truncation_bound=unavailable" in completed.stdout
    assert "truncation_bound_scope=actual-post-update-local-coordinates; cap0=unavailable" in completed.stdout
    assert "fidelity_metric=sampled-haar-output-state-fidelity; not exact-average-gate-fidelity" in completed.stdout
    assert "local_coordinate_source=analytic-simulator-oracle" in completed.stdout
    assert "nearest_clifford_circuit_approximation_bound=not-available" in completed.stdout
    assert f"per_case_output={output}" in completed.stdout
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["corpus"] == "synthetic"
    assert len(payload["records"]) == 48
    assert {row["estimator"] for row in payload["records"]} == {
        "no-bias", "local-error-oracle", "nearest-clifford", "anchored-pauli",
    }


def test_fidelity_rescore_places_local_error_before_its_ideal_operation() -> None:
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("benchmark_tradeoff", root / "scripts" / "benchmark_tradeoff.py")
    assert spec and spec.loader
    benchmark = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark)

    h = CircuitOperation("h", (0,), np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2.0))
    probes = np.array([[1.0], [0.0]], dtype=complex)
    ideal = benchmark._evolve_states((h,), 1, probes)
    # exp(-i * pi/2 * Z) is a global phase on |0> before H, so matching
    # transport convention requires unit fidelity against the ideal output.
    perturbed = benchmark._evolve_states((h,), 1, probes, np.array([[0.0, 0.0, np.pi / 2.0]]))
    assert benchmark._mean_output_state_fidelity(ideal, perturbed) == pytest.approx(1.0)


def test_mqt_stochastic_factories_are_seeded_when_benchmark_extra_is_available() -> None:
    pytest.importorskip("mqt.bench")
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("benchmark_tradeoff", root / "scripts" / "benchmark_tradeoff.py")
    assert spec and spec.loader
    benchmark = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark)

    first = benchmark._mqt_operations("graphstate", 6, 17)
    second = benchmark._mqt_operations("graphstate", 6, 17)
    assert len(first) == len(second)
    assert all(
        left.name == right.name and left.qubits == right.qubits and np.array_equal(left.unitary, right.unitary)
        for left, right in zip(first, second, strict=True)
    )
