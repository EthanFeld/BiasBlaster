"""Pauli labels and matrices using the q0-is-rightmost convention."""

from __future__ import annotations

from functools import lru_cache
from itertools import product

import numpy as np

_PAULI = "IXYZ"
_SINGLE = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}


@lru_cache(maxsize=None)
def pauli_labels(n_qubits: int) -> tuple[str, ...]:
    """Return all labels in lexicographic tensor-product order."""

    if not isinstance(n_qubits, int) or n_qubits < 1:
        raise ValueError("n_qubits must be a positive integer")
    return tuple("".join(label) for label in product(_PAULI, repeat=n_qubits))


@lru_cache(maxsize=512)
def pauli_matrix(label: str) -> np.ndarray:
    """Return the Hermitian matrix for a Pauli label."""

    if not label or any(character not in _PAULI for character in label):
        raise ValueError("invalid Pauli label")
    result = np.array([[1]], dtype=complex)
    for character in label:
        result = np.kron(result, _SINGLE[character])
    # This value is cached and used as an algebraic constant throughout the
    # package.  Do not let one caller corrupt later transport calculations.
    result.setflags(write=False)
    return result


def embed_pauli_label(local_label: str, qubits: tuple[int, ...], n_qubits: int) -> str:
    """Embed a local label into a register, with q0 as the rightmost factor."""

    if len(local_label) != len(qubits) or not qubits:
        raise ValueError("local label and qubit arity must agree")
    if n_qubits < 1 or any(qubit < 0 or qubit >= n_qubits for qubit in qubits):
        raise ValueError("qubit outside register")
    if len(set(qubits)) != len(qubits) or any(character not in _PAULI for character in local_label):
        raise ValueError("invalid local Pauli embedding")
    result = ["I"] * n_qubits
    for qubit, character in zip(qubits, local_label):
        result[n_qubits - 1 - qubit] = character
    return "".join(result)


def extract_pauli_label(global_label: str, qubits: tuple[int, ...]) -> str:
    """Extract local factors from a global label in the requested qubit order."""

    n_qubits = len(global_label)
    if not global_label or any(character not in _PAULI for character in global_label):
        raise ValueError("invalid global Pauli label")
    if not qubits or len(set(qubits)) != len(qubits) or any(qubit < 0 or qubit >= n_qubits for qubit in qubits):
        raise ValueError("qubit outside register")
    return "".join(global_label[n_qubits - 1 - qubit] for qubit in qubits)


def local_generator_labels(arity: int, *, full_two_qubit: bool = False) -> tuple[str, ...]:
    """Return the three supported local coherent-generator modes."""

    if arity == 1:
        return ("X", "Y", "Z")
    if arity == 2:
        if full_two_qubit:
            return pauli_labels(2)[1:]
        return ("XX", "YY", "ZZ")
    raise ValueError("only one- and two-qubit local generators are supported")
