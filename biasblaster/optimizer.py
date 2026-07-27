"""One trust-region optimizer shared by both transport representations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .nearest_clifford import SparseCliffordTransport
from .transport import AnchoredPauliTransport, SparsePauliTransport


@dataclass(frozen=True)
class BiasOptimizationResult:
    controls: np.ndarray
    updated_generators: np.ndarray
    predicted_before_norm: float
    predicted_after_norm: float
    control_norm: float
    budget: float
    # ``None`` denotes an unavailable exact-transport certificate.
    truncation_bound: float | None = 0.0


@dataclass(frozen=True)
class _BiasOptimizationProblem:
    """Validated linearized problem shared by CPU and FPGA optimizers."""

    local: np.ndarray
    jacobians: np.ndarray
    labels: tuple[str, ...]
    residual: np.ndarray
    design: np.ndarray
    budget: float


def _transport_rows(transport: SparsePauliTransport | SparseCliffordTransport | AnchoredPauliTransport, generators: np.ndarray) -> tuple[tuple[str, ...], np.ndarray]:
    if isinstance(transport, AnchoredPauliTransport):
        labels = tuple(sorted({
            *transport.corrections,
            *(
                label
                for gate in transport.active_gate_indices
                for label in transport.backbone.mode_labels[gate][:transport.backbone.mode_counts[gate]]
            ),
        }))
        matrix = np.zeros(
            (len(labels), transport.gate_count, transport.generator_mode_count), dtype=float,
        )
        index = {label: position for position, label in enumerate(labels)}
        for gate in transport.active_gate_indices:
            modes = transport.backbone.mode_labels[gate]
            count = transport.backbone.mode_counts[gate]
            for mode, label in enumerate(modes[:count]):
                matrix[index[label], gate, mode] += transport.backbone.signs[gate, mode]
        active = list(transport.active_gate_indices)
        for label, values in transport.corrections.items():
            matrix[index[label], active, :] += values.toarray().reshape(len(active), transport.generator_mode_count)
        return labels, matrix
    if isinstance(transport, SparseCliffordTransport):
        labels = tuple(sorted({
            label
            for modes, count in zip(transport.mode_labels, transport.mode_counts, strict=True)
            for label in modes[:count]
        }))
        matrix = np.zeros(
            (len(labels), transport.gate_count, transport.generator_mode_count),
            dtype=float,
        )
        index = {label: position for position, label in enumerate(labels)}
        for gate, (modes, count) in enumerate(zip(transport.mode_labels, transport.mode_counts, strict=True)):
            for mode, label in enumerate(modes[:count]):
                matrix[index[label], gate, mode] = transport.signs[gate, mode]
        return labels, matrix
    labels = tuple(transport.coefficients)
    matrix = np.zeros((len(labels), transport.gate_count, transport.generator_mode_count), dtype=float)
    active = {gate: index for index, gate in enumerate(transport.active_gate_indices)}
    for row, label in enumerate(labels):
        matrix[row, list(active), :] = transport.coefficients[label].reshape(len(active), transport.generator_mode_count)
    return labels, matrix


def _trust_region_least_squares(design: np.ndarray, bias: np.ndarray, budget: float) -> np.ndarray:
    """Return minimum-norm least-squares controls inside an L2 ball.

    The SVD formulation avoids forming normal equations for the unconstrained
    solve, which would square the condition number of ``design``.
    """

    if budget < 0:
        raise ValueError("budget must be nonnegative")
    if budget == 0 or not np.any(design):
        return np.zeros(design.shape[1], dtype=float)
    values = np.asarray(design, dtype=float)
    residual = np.asarray(bias, dtype=float)
    if values.ndim != 2 or residual.shape != (values.shape[0],):
        raise ValueError("design must be two-dimensional and bias must match its row count")

    # `lstsq` supplies the stable minimum-norm unconstrained solution. Its
    # default rank threshold is scaled to the input, unlike a fixed cutoff on
    # A.T @ A.
    unconstrained, _, _, _ = np.linalg.lstsq(values, -residual, rcond=None)
    if np.linalg.norm(unconstrained) <= budget:
        return unconstrained

    left_vectors, singular_values, right_vectors_t = np.linalg.svd(values, full_matrices=False)
    projected_bias = left_vectors.T @ residual

    def step(regularization: float) -> np.ndarray:
        weights = singular_values / (singular_values * singular_values + regularization)
        return -right_vectors_t.T @ (weights * projected_bias)

    gradient = values.T @ residual
    low = 0.0
    high = max(float(np.linalg.norm(gradient) / budget), 1.0)
    while np.linalg.norm(step(high)) > budget:
        high *= 2.0
    for _ in range(80):
        middle = 0.5 * (low + high)
        if np.linalg.norm(step(middle)) > budget:
            low = middle
        else:
            high = middle
    return step(high)


def _build_bias_optimization_problem(
    local_generators: np.ndarray,
    transport: SparsePauliTransport | SparseCliffordTransport | AnchoredPauliTransport,
    local_jacobians: np.ndarray,
    effort_fraction: float,
) -> _BiasOptimizationProblem:
    """Validate inputs and build the common residual/design representation."""

    local = np.asarray(local_generators, dtype=float)
    if local.ndim != 2 or local.shape[0] == 0 or not np.all(np.isfinite(local)):
        raise ValueError("local_generators must be finite and nonempty")
    if not 0.0 <= effort_fraction <= 1.0:
        raise ValueError("effort_fraction must lie in [0, 1]")
    jacobians = np.asarray(local_jacobians, dtype=float)
    if jacobians.ndim != 3 or jacobians.shape[:2] != local.shape or jacobians.shape[2] < 1 or not np.all(np.isfinite(jacobians)):
        raise ValueError("local_jacobians must have shape (gate_count, 3, controls)")
    if isinstance(transport, SparseCliffordTransport):
        if transport.gate_count != local.shape[0]:
            raise ValueError("transport and local_generators gate counts differ")
        if local.shape[1] != transport.generator_mode_count:
            raise ValueError("local generator mode count differs from transport")
        if any(np.any(local[gate, count:]) for gate, count in enumerate(transport.mode_counts)):
            raise ValueError("inactive generator modes must be zero")
    elif isinstance(transport, (SparsePauliTransport, AnchoredPauliTransport)):
        if transport.gate_count != local.shape[0]:
            raise ValueError("transport and local_generators gate counts differ")
        if local.shape[1] != transport.generator_mode_count:
            raise ValueError("local generator mode count differs from transport")
    else:
        raise TypeError("transport must be a supported Pauli or Clifford transport")

    labels, transfer = _transport_rows(transport, local)
    residual = transfer.reshape(len(labels), -1) @ local.reshape(-1) if labels else np.empty(0)
    design = np.zeros((len(labels), local.shape[0] * jacobians.shape[2]), dtype=float)
    for gate in range(local.shape[0]):
        for control in range(jacobians.shape[2]):
            design[:, gate * jacobians.shape[2] + control] = transfer[:, gate, :] @ jacobians[gate, :, control]
    return _BiasOptimizationProblem(
        local=local,
        jacobians=jacobians,
        labels=labels,
        residual=residual,
        design=design,
        budget=effort_fraction * float(np.linalg.norm(local)),
    )


def optimize_bias_controls(
    local_generators: np.ndarray,
    transport: SparsePauliTransport | SparseCliffordTransport | AnchoredPauliTransport,
    local_jacobians: np.ndarray,
    effort_fraction: float = 0.25,
) -> BiasOptimizationResult:
    """Reduce predicted end bias under a global L2 effort budget.

    ``local_jacobians[g, mode, control]`` is a calibrated local first-order
    response. The same least-squares solve works for sparse and nearest-
    Clifford transport, with no caller-side conversion.
    """

    problem = _build_bias_optimization_problem(
        local_generators, transport, local_jacobians, effort_fraction,
    )
    local = problem.local
    jacobians = problem.jacobians
    before = problem.residual
    design = problem.design
    budget = problem.budget
    flat_controls = _trust_region_least_squares(design, before, budget)
    controls = flat_controls.reshape(local.shape[0], jacobians.shape[2])
    updated = local + np.einsum("gmc,gc->gm", jacobians, controls)
    after = problem.residual + problem.design @ flat_controls
    truncation_bound = transport.truncation_bound_for(updated) if isinstance(transport, (SparsePauliTransport, AnchoredPauliTransport)) else 0.0
    return BiasOptimizationResult(
        controls=controls,
        updated_generators=updated,
        predicted_before_norm=float(np.linalg.norm(before)),
        predicted_after_norm=float(np.linalg.norm(after)),
        control_norm=float(np.linalg.norm(controls)),
        budget=budget,
        truncation_bound=truncation_bound,
    )


def optimize_local_error_controls(
    local_generators: np.ndarray,
    local_jacobians: np.ndarray,
    effort_fraction: float = 0.25,
) -> BiasOptimizationResult:
    """Optimize known local errors directly, without circuit transport.

    This is an oracle baseline for benchmarks: it minimizes the local error
    model itself, ``||theta + J delta||_2``, under the same global effort
    budget as :func:`optimize_bias_controls`.
    """

    local = np.asarray(local_generators, dtype=float)
    if local.ndim != 2 or local.shape[0] == 0 or local.shape[1] < 1 or not np.all(np.isfinite(local)):
        raise ValueError("local_generators must be a finite (gate_count, modes) array")
    if not 0.0 <= effort_fraction <= 1.0:
        raise ValueError("effort_fraction must lie in [0, 1]")
    jacobians = np.asarray(local_jacobians, dtype=float)
    if jacobians.ndim != 3 or jacobians.shape[:2] != local.shape or jacobians.shape[2] < 1 or not np.all(np.isfinite(jacobians)):
        raise ValueError("local_jacobians must have shape (gate_count, 3, controls)")

    gate_count, mode_count, control_count = jacobians.shape
    residual = local.reshape(-1)
    design = np.zeros((residual.size, gate_count * control_count), dtype=float)
    for gate in range(gate_count):
        for control in range(control_count):
            design[
                gate * mode_count : (gate + 1) * mode_count,
                gate * control_count + control,
            ] = jacobians[gate, :, control]
    budget = effort_fraction * float(np.linalg.norm(local))
    flat_controls = _trust_region_least_squares(design, residual, budget)
    controls = flat_controls.reshape(gate_count, control_count)
    updated = local + np.einsum("gmc,gc->gm", jacobians, controls)
    after = residual + design @ flat_controls
    return BiasOptimizationResult(
        controls=controls,
        updated_generators=updated,
        predicted_before_norm=float(np.linalg.norm(residual)),
        predicted_after_norm=float(np.linalg.norm(after)),
        control_norm=float(np.linalg.norm(controls)),
        budget=budget,
    )


OptimizationResult = BiasOptimizationResult
