from __future__ import annotations

import numpy as np
import pytest

from biasblaster import (
    CircuitOperation,
    estimate_pauli_transport,
    optimize_bias_controls,
    optimize_bias_controls_fpga,
    optimize_bias_controls_fpga_greedy,
)
from biasblaster.fpga_optimizer import (
    FPGA_BIAS_BATCH_OPCODE,
    FPGA_BIAS_PAYLOAD_BYTES,
    FPGA_MAX_CONTROLS,
    FPGA_REPLY_BYTES,
    FPGALink,
    FpgaOptimizerConfig,
    _compress_gate_rows,
    bounded_retraction_step_q,
)
from biasblaster.transport import SparsePauliTransport


class _MockLink:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def pulse_gate_bias_optimize(self, **kwargs):
        self.calls.append(kwargs)
        assert kwargs["gate_id"] == 0
        assert kwargs["gate_kind"] == 1
        assert np.asarray(kwargs["theta_q"]).shape == (FPGA_MAX_CONTROLS,)
        # A one-dimensional negative update, encoded as Q12.
        return 0, 1, np.array([-2048, 0, 0, 0, 0, 0], dtype=np.int16)


class _GreedyMockLink:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def pulse_gate_bias_optimize(self, **kwargs):
        self.calls.append(kwargs)
        # Move each selected gate to its negative coordinate bound. The real
        # FPGA returns the bounded fixed-point retraction in this position.
        return (
            kwargs["gate_id"],
            kwargs["gate_kind"],
            np.asarray(kwargs["lower_q"], dtype=np.int16).copy(),
        )


class _BatchWireLink(FPGALink):
    def __init__(self) -> None:
        self.requests: list[tuple[int, bytes, int]] = []

    def _request(self, opcode: int, payload: bytes, reply_len: int) -> bytes:
        self.requests.append((opcode, payload, reply_len))
        replies = (
            (7).to_bytes(2, "little") + bytes([1]) + np.zeros(6, dtype="<i2").tobytes(),
            (8).to_bytes(2, "little") + bytes([2]) + np.ones(6, dtype="<i2").tobytes(),
        )
        return b"".join(replies)


def _wire_record(gate_id: int, gate_kind: int = 1) -> dict[str, object]:
    return {
        "gate_id": gate_id,
        "gate_kind": gate_kind,
        "alpha_q": 4096,
        "gain_q": 4096,
        "max_step_q": 1024,
        "damping_q": 1,
        "theta_q": np.zeros(6, dtype=np.int16),
        "lower_q": -np.ones(6, dtype=np.int16),
        "upper_q": np.ones(6, dtype=np.int16),
        "residuals_q": np.zeros(1, dtype=np.int16),
        "jac_rows_q": np.zeros((1, 6), dtype=np.int16),
    }


def _problem(control_count: int = 1):
    operation = CircuitOperation("source", (0,), np.eye(2))
    transport = estimate_pauli_transport((operation,), 1)
    generators = np.array([[1.0, 0.0, 0.0]])
    jacobians = np.zeros((1, 3, control_count), dtype=float)
    jacobians[0, 0, 0] = 1.0
    return generators, transport, jacobians


def test_fpga_reference_preserves_biasblaster_contract_and_budget() -> None:
    generators, transport, jacobians = _problem()
    result = optimize_bias_controls_fpga(
        generators, transport, jacobians, effort_fraction=0.5
    )

    assert result.controls.shape == (1, 1)
    assert result.control_norm <= result.budget + 1e-12
    assert result.predicted_after_norm < result.predicted_before_norm
    assert np.allclose(result.updated_generators, [[0.5, 0.0, 0.0]], atol=2e-3)


def test_fpga_reference_matches_repo_trust_region_to_q12() -> None:
    rng = np.random.default_rng(91)
    operation = CircuitOperation("source", (0,), np.eye(2))
    transport = estimate_pauli_transport((operation,), 1)
    for _ in range(12):
        control_count = int(rng.integers(1, 5))
        generators = rng.normal(0.0, 0.35, (1, 3))
        jacobians = rng.normal(0.0, 0.55, (1, 3, control_count))
        effort_fraction = float(rng.uniform(0.1, 0.8))
        cpu = optimize_bias_controls(
            generators, transport, jacobians, effort_fraction=effort_fraction
        )
        fpga = optimize_bias_controls_fpga(
            generators, transport, jacobians, effort_fraction=effort_fraction
        )
        assert np.allclose(fpga.controls, cpu.controls, atol=1.0 / 4096.0)
        assert abs(fpga.predicted_after_norm - cpu.predicted_after_norm) < 1.0e-3
        assert fpga.control_norm <= fpga.budget + 1.0e-12


def test_fpga_reference_reports_bound_for_updated_generators() -> None:
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
    result = optimize_bias_controls_fpga(
        generators, transport, jacobians, effort_fraction=0.5
    )

    assert result.truncation_bound == pytest.approx(
        transport.truncation_bound_for(result.updated_generators)
    )


def test_fpga_link_receives_same_compiled_residual_design_contract() -> None:
    generators, transport, jacobians = _problem()
    link = _MockLink()
    result = optimize_bias_controls_fpga(
        generators, transport, jacobians, effort_fraction=0.5, link=link
    )

    assert len(link.calls) == 1
    assert np.asarray(link.calls[0]["residuals_q"]).size == transport.support
    assert np.asarray(link.calls[0]["jac_rows_q"]).shape == (transport.support, 6)
    assert np.allclose(result.controls, [[-0.5]], atol=1e-12)


def test_fpga_link_batches_only_precomputed_gate_records() -> None:
    link = _BatchWireLink()

    replies = link.pulse_gate_bias_optimize_batch([_wire_record(7), _wire_record(8, 2)])

    assert len(link.requests) == 1
    _, payload, _ = link.requests[0]
    assert payload[0] == 2
    assert len(payload) == 1 + 2 * FPGA_BIAS_PAYLOAD_BYTES
    assert [(gate_id, kind) for gate_id, kind, _ in replies] == [(7, 1), (8, 2)]
    assert np.array_equal(replies[1][2], np.ones(6, dtype=np.int16))


def test_fpga_greedy_prioritizes_larger_transported_generators() -> None:
    operations = (
        CircuitOperation("source-0", (0,), np.eye(2)),
        CircuitOperation("source-1", (0,), np.eye(2)),
    )
    transport = estimate_pauli_transport(operations, 1)
    generators = np.array([[0.1, 0.0, 0.0], [0.4, 0.0, 0.0]])
    jacobians = np.zeros((2, 3, 1), dtype=float)
    jacobians[:, 0, 0] = 1.0
    link = _GreedyMockLink()

    result = optimize_bias_controls_fpga_greedy(
        generators,
        transport,
        jacobians,
        effort_fraction=0.5,
        link=link,
    )

    # First sweep chooses larger gate first. Later calls revisit gates after
    # accepted updates change their transported-bias priority.
    assert [call["gate_id"] for call in link.calls][:2] == [1, 0]
    assert len(link.calls) > 2
    assert np.allclose(result.controls[:, 0], [-0.05, -0.2], atol=1.0 / 4096.0)
    assert result.predicted_after_norm < result.predicted_before_norm
    assert result.control_norm <= result.budget + 1.0e-12


def test_fpga_greedy_compresses_all_bias_rows_into_one_local_record() -> None:
    transport = SparsePauliTransport(
        {label: np.array([1.0, 0.0, 0.0]) for label in ("XI", "YI", "ZI", "II", "XX", "YY")},
        n_qubits=2,
        gate_count=1,
        active_gate_indices=(0,),
        support_count=6,
    )
    generators = np.array([[0.1, 0.0, 0.0]])
    jacobians = np.array([[[1.0], [0.0], [0.0]]])
    link = _GreedyMockLink()

    result = optimize_bias_controls_fpga_greedy(
        generators, transport, jacobians, effort_fraction=0.5, link=link
    )

    assert len(link.calls) == 1
    assert np.asarray(link.calls[0]["residuals_q"]).size == 1
    assert np.asarray(link.calls[0]["jac_rows_q"]).shape == (1, 6)
    assert result.control_norm <= result.budget + 1.0e-12


def test_compressed_rows_preserve_local_normal_equations() -> None:
    rng = np.random.default_rng(73)
    residual = rng.normal(size=11)
    block = rng.normal(size=(11, 3))
    compressed_residual, compressed_block, scale = _compress_gate_rows(
        residual, block, FpgaOptimizerConfig(),
    )

    unscaled_residual = compressed_residual / scale
    unscaled_block = compressed_block / scale
    assert compressed_residual.size <= 3
    assert np.allclose(unscaled_block.T @ unscaled_block, block.T @ block)
    assert np.allclose(unscaled_block.T @ unscaled_residual, block.T @ residual)


def test_fpga_greedy_radially_projects_multi_control_candidate() -> None:
    generators, transport, _ = _problem(control_count=2)
    jacobians = np.zeros((1, 3, 2), dtype=float)
    jacobians[0, 0, :] = 1.0
    link = _GreedyMockLink()
    result = optimize_bias_controls_fpga_greedy(
        generators,
        transport,
        jacobians,
        effort_fraction=0.5,
        link=link,
        config=FpgaOptimizerConfig(max_sweeps=1),
    )

    assert np.linalg.norm(result.controls[0]) <= result.budget + 1.0e-12
    assert np.linalg.norm(result.controls[0]) == pytest.approx(result.budget, abs=1.0 / 4096.0)


def test_fpga_greedy_ranks_expected_decrease_not_generator_size() -> None:
    operations = (
        CircuitOperation("weak-control", (0,), np.eye(2)),
        CircuitOperation("strong-control", (0,), np.eye(2)),
    )
    transport = estimate_pauli_transport(operations, 1)
    generators = np.array([[1.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
    jacobians = np.zeros((2, 3, 1), dtype=float)
    jacobians[0, 0, 0] = 0.01
    jacobians[1, 0, 0] = 1.0
    link = _GreedyMockLink()

    optimize_bias_controls_fpga_greedy(
        generators,
        transport,
        jacobians,
        effort_fraction=0.25,
        link=link,
        config=FpgaOptimizerConfig(max_sweeps=1),
    )

    assert link.calls[0]["gate_id"] == 1


def test_fpga_greedy_uses_cpu_greedy_fallback_for_oversize_gate_controls() -> None:
    generators, transport, jacobians = _problem(control_count=7)
    result = optimize_bias_controls_fpga_greedy(
        generators,
        transport,
        jacobians,
        effort_fraction=0.5,
    )

    assert result.predicted_after_norm < result.predicted_before_norm
    assert result.control_norm <= result.budget + 1.0e-12
    with pytest.raises(ValueError, match="greedy problem"):
        optimize_bias_controls_fpga_greedy(
            generators,
            transport,
            jacobians,
            effort_fraction=0.5,
            fallback="error",
        )


def test_oversize_problem_falls_back_to_exact_cpu_or_errors() -> None:
    generators, transport, jacobians = _problem(control_count=7)
    expected = optimize_bias_controls(
        generators, transport, jacobians, effort_fraction=0.5
    )
    actual = optimize_bias_controls_fpga(
        generators, transport, jacobians, effort_fraction=0.5
    )
    assert np.array_equal(actual.controls, expected.controls)
    with pytest.raises(ValueError, match="fixed payload"):
        optimize_bias_controls_fpga(
            generators,
            transport,
            jacobians,
            effort_fraction=0.5,
            fallback="error",
        )


def test_fixed_point_retraction_matches_q12_motion_and_bounds() -> None:
    result = bounded_retraction_step_q(
        [0, 0, 0, 0, 0, 0],
        [-4096, 0, 0, 0, 0, 0],
        [-1024, -1, -1, -1, -1, -1],
        [1024, 1, 1, 1, 1, 1],
        alpha_q=4096,
        max_step_q=1024,
    )
    assert result[0] == -1024
