"""Exact bounded-local Pauli-transfer algebra."""

from __future__ import annotations

from functools import lru_cache
from typing import Mapping

import numpy as np

from .model import CircuitOperation
from .pauli import (
    embed_pauli_label,
    extract_pauli_label,
    local_generator_labels,
    pauli_labels,
    pauli_matrix,
)


@lru_cache(maxsize=512)
def _local_matrices(arity: int) -> tuple[np.ndarray, ...]:
    return tuple(pauli_matrix(label) for label in pauli_labels(arity))


def local_pauli_transfer(operation: CircuitOperation) -> np.ndarray:
    """Return the real Pauli transfer matrix for one local operation.

    The matrix maps coefficients of ``P`` to coefficients of ``U P U†``.
    Only the operation's local Hilbert space is used.
    """

    arity = len(operation.qubits)
    if arity not in (1, 2):
        raise ValueError("local Pauli transfer supports only one- and two-qubit operations")
    labels = pauli_labels(arity)
    matrices = _local_matrices(arity)
    dimension = 2**arity
    transfer = np.empty((len(labels), len(labels)), dtype=float)
    for source, pauli in enumerate(matrices):
        transformed = operation.unitary @ pauli @ operation.unitary.conj().T
        for target, output in enumerate(matrices):
            coefficient = np.trace(output @ transformed) / dimension
            if abs(coefficient.imag) > 1e-9:
                raise ValueError("operation does not induce a real Pauli transfer")
            transfer[target, source] = float(coefficient.real)
    transfer[np.abs(transfer) < 1e-12] = 0.0
    return transfer


def conjugate_sparse_pauli(
    coefficients: Mapping[str, float],
    operation: CircuitOperation,
    n_qubits: int,
    *,
    coefficient_tol: float = 1e-12,
) -> dict[str, float]:
    """Conjugate a sparse scalar Pauli expansion through one operation."""

    if coefficient_tol < 0:
        raise ValueError("coefficient_tol must be nonnegative")
    labels = pauli_labels(len(operation.qubits))
    transfer = local_pauli_transfer(operation)
    result: dict[str, float] = {}
    for global_label, value in coefficients.items():
        if len(global_label) != n_qubits:
            raise ValueError("global Pauli label has the wrong register size")
        source = labels.index(extract_pauli_label(global_label, operation.qubits))
        for target, factor in enumerate(transfer[:, source]):
            if abs(factor) <= coefficient_tol:
                continue
            output = embed_pauli_label(labels[target], operation.qubits, n_qubits)
            # Preserve untouched spectator factors.
            characters = list(global_label)
            for qubit, character in zip(operation.qubits, labels[target]):
                characters[n_qubits - 1 - qubit] = character
            output = "".join(characters)
            result[output] = result.get(output, 0.0) + float(value) * float(factor)
    return {label: value for label, value in result.items() if abs(value) > coefficient_tol}


def conjugate_sparse_modes(
    coefficients: Mapping[str, np.ndarray],
    operation: CircuitOperation,
    n_qubits: int,
    *,
    coefficient_tol: float = 1e-12,
) -> dict[str, np.ndarray]:
    """Conjugate sparse Pauli labels while retaining source-mode vectors."""

    if coefficient_tol < 0:
        raise ValueError("coefficient_tol must be nonnegative")
    labels = pauli_labels(len(operation.qubits))
    transfer = local_pauli_transfer(operation)
    result: dict[str, np.ndarray] = {}
    for global_label, values in coefficients.items():
        if len(global_label) != n_qubits:
            raise ValueError("global Pauli label has the wrong register size")
        source = labels.index(extract_pauli_label(global_label, operation.qubits))
        for target, factor in enumerate(transfer[:, source]):
            if abs(factor) <= coefficient_tol:
                continue
            characters = list(global_label)
            for qubit, character in zip(operation.qubits, labels[target]):
                characters[n_qubits - 1 - qubit] = character
            output = "".join(characters)
            contribution = np.asarray(values, dtype=float) * float(factor)
            if output in result:
                result[output] += contribution
            else:
                result[output] = contribution.copy()
    return {
        label: values
        for label, values in result.items()
        if np.max(np.abs(values), initial=0.0) > coefficient_tol
    }


def pauli_transfer(operation: CircuitOperation) -> np.ndarray:
    """Alias with a concise name for users building local algebra checks."""

    return local_pauli_transfer(operation)
