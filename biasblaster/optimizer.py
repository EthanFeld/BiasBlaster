"""One trust-region optimizer shared by both transport representations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .nearest_clifford import SparseCliffordTransport
from .transport import SparsePauliTransport


@dataclass(frozen=True)
class BiasOptimizationResult:
    controls: np.ndarray
    updated_generators: np.ndarray
    predicted_before_norm: float
    predicted_after_norm: float
    control_norm: float
    budget: float
    truncation_bound: float = 0.0


def _transport_rows(transport: SparsePauliTransport | SparseCliffordTransport, generators: np.ndarray) -> tuple[tuple[str, ...], np.ndarray]:
    if isinstance(transport, SparseCliffordTransport):
        labels = tuple(sorted({label for modes in transport.mode_labels for label in modes}))
        matrix = np.zeros((len(labels), transport.gate_count, 3), dtype=float)
        index = {label: position for position, label in enumerate(labels)}
        for gate, modes in enumerate(transport.mode_labels):
            for mode, label in enumerate(modes):
                matrix[index[label], gate, mode] = transport.signs[gate, mode]
        return labels, matrix
    labels = tuple(transport.coefficients)
    matrix = np.zeros((len(labels), transport.gate_count, 3), dtype=float)
    active = {gate: index for index, gate in enumerate(transport.active_gate_indices)}
    for row, label in enumerate(labels):
        matrix[row, list(active), :] = transport.coefficients[label].reshape(len(active), 3)
    return labels, matrix


def _trust_region_least_squares(design: np.ndarray, bias: np.ndarray, budget: float) -> np.ndarray:
    if budget < 0:
        raise ValueError("budget must be nonnegative")
    if budget == 0 or not np.any(design):
        return np.zeros(design.shape[1], dtype=float)
    hessian = design.T @ design
    gradient = design.T @ bias
    unconstrained = -np.linalg.pinv(hessian, rcond=1e-12) @ gradient
    if np.linalg.norm(unconstrained) <= budget:
        return unconstrained
    eigenvalues, eigenvectors = np.linalg.eigh(hessian)
    projected_gradient = eigenvectors.T @ gradient

    def step(regularization: float) -> np.ndarray:
        return -eigenvectors @ (projected_gradient / (eigenvalues + regularization))

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


def optimize_bias_controls(
    local_generators: np.ndarray,
    transport: SparsePauliTransport | SparseCliffordTransport,
    local_jacobians: np.ndarray,
    effort_fraction: float = 0.25,
) -> BiasOptimizationResult:
    """Reduce predicted end bias under a global L2 effort budget.

    ``local_jacobians[g, mode, control]`` is a calibrated local first-order
    response. The same least-squares solve works for sparse and nearest-
    Clifford transport, with no caller-side conversion.
    """

    local = np.asarray(local_generators, dtype=float)
    if local.ndim != 2 or local.shape[0] == 0 or not np.all(np.isfinite(local)):
        raise ValueError("local_generators must be finite and nonempty")
    if local.shape[1] != 3:
        raise ValueError("local_generators must have shape (gate_count, 3)")
    if not 0.0 <= effort_fraction <= 1.0:
        raise ValueError("effort_fraction must lie in [0, 1]")
    jacobians = np.asarray(local_jacobians, dtype=float)
    if jacobians.ndim != 3 or jacobians.shape[:2] != local.shape or jacobians.shape[2] < 1 or not np.all(np.isfinite(jacobians)):
        raise ValueError("local_jacobians must have shape (gate_count, 3, controls)")
    if isinstance(transport, SparseCliffordTransport):
        if transport.gate_count != local.shape[0]:
            raise ValueError("transport and local_generators gate counts differ")
    elif isinstance(transport, SparsePauliTransport):
        if transport.gate_count != local.shape[0]:
            raise ValueError("transport and local_generators gate counts differ")
    else:
        raise TypeError("transport must be SparsePauliTransport or SparseCliffordTransport")

    labels, transfer = _transport_rows(transport, local)
    before = transfer.reshape(len(labels), -1) @ local.reshape(-1) if labels else np.empty(0)
    design = np.zeros((len(labels), local.shape[0] * jacobians.shape[2]), dtype=float)
    for gate in range(local.shape[0]):
        for control in range(jacobians.shape[2]):
            design[:, gate * jacobians.shape[2] + control] = transfer[:, gate, :] @ jacobians[gate, :, control]
    budget = effort_fraction * float(np.linalg.norm(local))
    flat_controls = _trust_region_least_squares(design, before, budget)
    controls = flat_controls.reshape(local.shape[0], jacobians.shape[2])
    updated = local + np.einsum("gmc,gc->gm", jacobians, controls)
    after = transfer.reshape(len(labels), -1) @ updated.reshape(-1) if labels else np.empty(0)
    truncation_bound = transport.truncation_bound_for(local) if isinstance(transport, SparsePauliTransport) else 0.0
    return BiasOptimizationResult(
        controls=controls,
        updated_generators=updated,
        predicted_before_norm=float(np.linalg.norm(before)),
        predicted_after_norm=float(np.linalg.norm(after)),
        control_norm=float(np.linalg.norm(controls)),
        budget=budget,
        truncation_bound=truncation_bound,
    )


OptimizationResult = BiasOptimizationResult
