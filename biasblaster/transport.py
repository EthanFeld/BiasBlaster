"""Bounded sparse Pauli transport for ordinary ordered circuits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .local_transfer import conjugate_sparse_modes
from .model import CircuitOperation
from .pauli import embed_pauli_label, local_generator_labels


def _validate_operations(operations: Sequence[CircuitOperation], n_qubits: int) -> tuple[CircuitOperation, ...]:
    values = tuple(operations)
    if n_qubits < 1 or not values:
        raise ValueError("operations and a positive n_qubits are required")
    for operation in values:
        if not isinstance(operation, CircuitOperation):
            raise TypeError("operations must contain CircuitOperation values")
        if len(operation.qubits) not in (1, 2):
            raise ValueError("only one- and two-qubit operations are supported")
        if any(qubit >= n_qubits for qubit in operation.qubits):
            raise ValueError("operation qubit outside register")
    return values


def _validate_active(active_gate_indices: Sequence[int] | None, gate_count: int) -> tuple[int, ...]:
    active = tuple(range(gate_count)) if active_gate_indices is None else tuple(int(index) for index in active_gate_indices)
    if not active or len(set(active)) != len(active) or any(index < 0 or index >= gate_count for index in active):
        raise ValueError("active_gate_indices must contain unique valid gate indices")
    return active


def _prune(
    coefficients: dict[str, np.ndarray],
    support_cap: int | None,
    *,
    coefficient_tol: float,
) -> tuple[dict[str, np.ndarray], float]:
    retained = {
        label: values
        for label, values in coefficients.items()
        if np.max(np.abs(values), initial=0.0) > coefficient_tol
    }
    if support_cap is None or len(retained) <= support_cap:
        return retained, 0.0
    ranked = sorted(retained, key=lambda label: (-float(np.linalg.norm(retained[label])), label))
    keep = set(ranked[:support_cap])
    # Sum of omitted vector norms is conservative even if later branches overlap.
    bound = float(sum(np.linalg.norm(values) for label, values in retained.items() if label not in keep))
    return {label: values for label, values in retained.items() if label in keep}, bound


def _merge(target: dict[str, np.ndarray], label: str, values: np.ndarray) -> None:
    if label in target:
        target[label] += values
    else:
        target[label] = values.copy()


@dataclass(frozen=True)
class SparsePauliTransport:
    """Sparse end-Pauli transport columns indexed by local generator modes.

    ``coefficients[label]`` is a vector with one entry per active gate/mode.
    The representation never creates a dense ``4**n`` output array.
    """

    coefficients: Mapping[str, np.ndarray]
    n_qubits: int
    gate_count: int
    active_gate_indices: tuple[int, ...]
    support_count: int
    truncation_bound: float = 0.0

    def __post_init__(self) -> None:
        if self.n_qubits < 1 or self.gate_count < 1:
            raise ValueError("transport dimensions must be positive")
        active = _validate_active(self.active_gate_indices, self.gate_count)
        if self.truncation_bound < 0 or not np.isfinite(self.truncation_bound):
            raise ValueError("truncation_bound must be finite and nonnegative")
        width = len(active) * 3
        copied: dict[str, np.ndarray] = {}
        for label, values in self.coefficients.items():
            label = str(label)
            if len(label) != self.n_qubits or any(character not in "IXYZ" for character in label):
                raise ValueError("invalid Pauli label in transport")
            vector = np.asarray(values, dtype=float)
            if vector.shape != (width,) or not np.all(np.isfinite(vector)):
                raise ValueError("transport columns must have shape (active_gate_count * 3,)")
            if np.any(vector):
                copied[label] = vector.copy()
        if self.support_count != len(copied):
            raise ValueError("support_count must equal the retained Pauli support")
        object.__setattr__(self, "active_gate_indices", active)
        object.__setattr__(self, "coefficients", copied)

    @property
    def support(self) -> int:
        return self.support_count

    @property
    def retained_bytes(self) -> int:
        return int(sum(values.nbytes for values in self.coefficients.values()))

    def _active_generators(self, generators: np.ndarray) -> np.ndarray:
        local = np.asarray(generators, dtype=float)
        if local.shape != (self.gate_count, 3) or not np.all(np.isfinite(local)):
            raise ValueError("generators must have shape (gate_count, 3) and be finite")
        return local[list(self.active_gate_indices)]

    def contract(self, generators: np.ndarray) -> dict[str, float]:
        """Contract local generator coefficients into a sparse end bias."""

        local = self._active_generators(generators).reshape(-1)
        return {
            label: float(np.dot(values, local))
            for label, values in self.coefficients.items()
            if abs(float(np.dot(values, local))) > 1e-12
        }

    def local_priorities(self, generators: np.ndarray) -> np.ndarray:
        """Return ``M_g.T @ b`` for the current transported bias ``b``."""

        self._active_generators(generators)
        bias = self.contract(generators)
        priority = np.zeros((self.gate_count, 3), dtype=float)
        if bias:
            for index, gate in enumerate(self.active_gate_indices):
                for mode in range(3):
                    priority[gate, mode] = sum(
                        float(self.coefficients[label].reshape(-1)[index * 3 + mode]) * value
                        for label, value in bias.items()
                    )
        return priority

    def truncation_bound_for(self, generators: np.ndarray) -> float:
        return float(self.truncation_bound * np.linalg.norm(self._active_generators(generators)))

    def compact_transfer(self) -> np.ndarray:
        """Return a bounded ``(active_gate, support, mode)`` adapter."""

        labels = tuple(self.coefficients)
        return np.stack([values.reshape(len(self.active_gate_indices), 3) for values in self.coefficients.values()], axis=1) if labels else np.empty((len(self.active_gate_indices), 0, 3))


def estimate_pauli_transport(
    operations: Sequence[CircuitOperation],
    n_qubits: int,
    *,
    support_cap: int | None = None,
    coefficient_tol: float = 1e-12,
    active_gate_indices: Sequence[int] | None = None,
) -> SparsePauliTransport:
    """Estimate end transport with exact local Pauli propagation.

    When ``support_cap`` is set, the largest retained vectors are kept and the
    sum of omitted vector norms is accumulated in ``truncation_bound``.
    """

    values = _validate_operations(operations, n_qubits)
    if support_cap is not None and (not isinstance(support_cap, int) or support_cap < 1):
        raise ValueError("support_cap must be a positive integer")
    if coefficient_tol < 0:
        raise ValueError("coefficient_tol must be nonnegative")
    active = _validate_active(active_gate_indices, len(values))
    offsets = {gate: index * 3 for index, gate in enumerate(active)}
    width = len(active) * 3
    coefficients: dict[str, np.ndarray] = {}
    truncation_bound = 0.0

    for gate_index in range(len(values) - 1, -1, -1):
        if gate_index < len(values) - 1:
            coefficients = conjugate_sparse_modes(
                coefficients,
                values[gate_index + 1],
                n_qubits,
                coefficient_tol=coefficient_tol,
            )
            coefficients, dropped = _prune(coefficients, support_cap, coefficient_tol=coefficient_tol)
            truncation_bound += dropped
        if gate_index in offsets:
            for mode, local_label in enumerate(local_generator_labels(len(values[gate_index].qubits))):
                vector = np.zeros(width, dtype=float)
                vector[offsets[gate_index] + mode] = 1.0
                label = embed_pauli_label(local_label, values[gate_index].qubits, n_qubits)
                _merge(coefficients, label, vector)
            coefficients, dropped = _prune(coefficients, support_cap, coefficient_tol=coefficient_tol)
            truncation_bound += dropped

    return SparsePauliTransport(
        coefficients=coefficients,
        n_qubits=n_qubits,
        gate_count=len(values),
        active_gate_indices=active,
        support_count=len(coefficients),
        truncation_bound=truncation_bound,
    )


def propagate_pauli_bias(
    operations: Sequence[CircuitOperation],
    generators: np.ndarray,
    n_qubits: int,
    *,
    support_cap: int | None = None,
    coefficient_tol: float = 1e-12,
) -> tuple[dict[str, float], float]:
    """Convenience scalar path returning bias and its omitted-transport bound."""

    transport = estimate_pauli_transport(
        operations,
        n_qubits,
        support_cap=support_cap,
        coefficient_tol=coefficient_tol,
    )
    return transport.contract(generators), transport.truncation_bound_for(generators)
