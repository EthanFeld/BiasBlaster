"""Compact nearest-Clifford end transport."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from typing import Sequence

import numpy as np

from .model import CircuitOperation
from .pauli import embed_pauli_label, local_generator_labels, pauli_labels, pauli_matrix
from .tableau import CliffordTableau, SymplecticPauli, named_clifford, tableau_from_unitary


def _tableau_key(tableau: CliffordTableau) -> tuple[int, ...]:
    values: list[int] = []
    for image in tableau.x_images + tableau.z_images:
        values.extend((image.x, image.z, image.sign))
    return tuple(values)


@lru_cache(maxsize=2)
def _clifford_tableaus(arity: int) -> tuple[CliffordTableau, ...]:
    if arity not in (1, 2):
        raise ValueError("nearest-Clifford approximation supports only one- and two-qubit gates")
    generators = [named_clifford("H", arity, (qubit,)) for qubit in range(arity)]
    generators += [named_clifford("S", arity, (qubit,)) for qubit in range(arity)]
    if arity == 2:
        generators += [named_clifford("CNOT", arity, (0, 1)), named_clifford("CNOT", arity, (1, 0))]
    seen: dict[tuple[int, ...], CliffordTableau] = {}
    pending = [CliffordTableau.identity(arity)]
    while pending:
        current = pending.pop()
        key = _tableau_key(current)
        if key in seen:
            continue
        seen[key] = current
        pending.extend(current.compose(generator) for generator in generators)
    return tuple(seen[key] for key in sorted(seen))


def _tableau_transfer(tableau: CliffordTableau) -> np.ndarray:
    labels = pauli_labels(tableau.n_qubits)
    index = {label: position for position, label in enumerate(labels)}
    transfer = np.zeros((len(labels), len(labels)), dtype=float)
    for source, label in enumerate(labels):
        image = tableau.conjugate(label)
        transfer[index[image.label], source] = image.sign
    return transfer


def _pauli_transfer(unitary: np.ndarray) -> np.ndarray:
    value = np.asarray(unitary, dtype=complex)
    if value.ndim != 2 or value.shape[0] != value.shape[1]:
        raise ValueError("unitary must be square")
    arity = int(round(np.log2(value.shape[0])))
    if arity not in (1, 2) or value.shape != (2**arity, 2**arity):
        raise ValueError("operation unitary must be one- or two-qubit")
    if not np.allclose(value.conj().T @ value, np.eye(value.shape[0]), atol=1e-9, rtol=0):
        raise ValueError("operation unitary must be unitary")
    labels = pauli_labels(arity)
    matrices = tuple(pauli_matrix(label) for label in labels)
    transfer = np.empty((len(labels), len(labels)), dtype=float)
    for source, matrix in enumerate(matrices):
        transformed = value @ matrix @ value.conj().T
        for target, output in enumerate(matrices):
            coefficient = np.trace(output @ transformed) / value.shape[0]
            if abs(coefficient.imag) > 1e-9:
                raise ValueError("unitary has a non-real Pauli transfer")
            transfer[target, source] = float(coefficient.real)
    transfer[np.abs(transfer) < 1e-12] = 0.0
    return transfer


def closest_clifford(unitary: np.ndarray) -> CliffordTableau:
    """Return the same-arity Clifford with maximum phase-invariant process fidelity."""

    value = np.asarray(unitary, dtype=complex)
    try:
        return tableau_from_unitary(value)
    except ValueError:
        pass
    arity = int(round(np.log2(value.shape[0]))) if value.ndim == 2 else 0
    if arity not in (1, 2):
        raise ValueError("nearest-Clifford approximation supports only one- and two-qubit gates")
    actual = _pauli_transfer(value)
    candidates = _clifford_tableaus(arity)
    scores = [float(np.trace(_tableau_transfer(candidate).T @ actual)) for candidate in candidates]
    # Candidates are canonically ordered, so argmax is deterministic on ties.
    return candidates[int(np.argmax(scores))]


def _embed_tableau(local: CliffordTableau, qubits: tuple[int, ...], n_qubits: int) -> CliffordTableau:
    result = CliffordTableau.identity(n_qubits)
    x, z = list(result.x_images), list(result.z_images)
    for local_qubit, global_qubit in enumerate(qubits):
        for image, target in ((local.x_images[local_qubit], x), (local.z_images[local_qubit], z)):
            label = embed_pauli_label(image.label, qubits, n_qubits)
            target[global_qubit] = SymplecticPauli.from_label(label, image.sign)
    return CliffordTableau(n_qubits, tuple(x), tuple(z))


@dataclass(frozen=True)
class SparseCliffordTransport:
    """One signed final Pauli destination per gate and generator mode."""

    n_qubits: int
    mode_labels: tuple[tuple[str, str, str], ...]
    signs: np.ndarray

    def __post_init__(self) -> None:
        signs = np.asarray(self.signs, dtype=float)
        if self.n_qubits < 1 or not self.mode_labels or signs.shape != (len(self.mode_labels), 3):
            raise ValueError("mode_labels and signs must have shape (gate_count, 3)")
        if any(len(label) != self.n_qubits or any(character not in "IXYZ" for character in label) for modes in self.mode_labels for label in modes):
            raise ValueError("invalid final Pauli label")
        if not np.all(np.isin(signs, (-1.0, 1.0))):
            raise ValueError("signs must be ±1")
        object.__setattr__(self, "mode_labels", tuple(tuple(modes) for modes in self.mode_labels))
        object.__setattr__(self, "signs", signs.copy())

    @property
    def gate_count(self) -> int:
        return len(self.mode_labels)

    @property
    def support(self) -> int:
        return len({label for modes in self.mode_labels for label in modes})

    @property
    def support_count(self) -> int:
        return self.support

    @property
    def truncation_bound(self) -> float:
        return 0.0

    @property
    def retained_bytes(self) -> int:
        return int(self.gate_count * 3 * self.n_qubits + self.signs.nbytes)

    def truncation_bound_for(self, generators: np.ndarray) -> float:
        local = np.asarray(generators, dtype=float)
        if local.shape != (self.gate_count, 3):
            raise ValueError("generators must have shape (gate_count, 3)")
        return 0.0

    def contract(self, generators: np.ndarray) -> dict[str, float]:
        local = np.asarray(generators, dtype=float)
        if local.shape != (self.gate_count, 3) or not np.all(np.isfinite(local)):
            raise ValueError("generators must have shape (gate_count, 3) and be finite")
        result: dict[str, float] = {}
        for gate, modes in enumerate(self.mode_labels):
            for mode, label in enumerate(modes):
                result[label] = result.get(label, 0.0) + float(self.signs[gate, mode] * local[gate, mode])
        return {label: value for label, value in result.items() if abs(value) > 1e-12}

    def local_priorities(self, generators: np.ndarray) -> np.ndarray:
        local = np.asarray(generators, dtype=float)
        if local.shape != (self.gate_count, 3) or not np.all(np.isfinite(local)):
            raise ValueError("generators must have shape (gate_count, 3) and be finite")
        bias = self.contract(local)
        return np.array(
            [[self.signs[gate, mode] * bias.get(label, 0.0) for mode, label in enumerate(modes)] for gate, modes in enumerate(self.mode_labels)],
            dtype=float,
        )

    def compact_transfer(self) -> np.ndarray:
        labels = tuple(sorted({label for modes in self.mode_labels for label in modes}))
        index = {label: position for position, label in enumerate(labels)}
        result = np.zeros((self.gate_count, len(labels), 3), dtype=float)
        for gate, modes in enumerate(self.mode_labels):
            for mode, label in enumerate(modes):
                result[gate, index[label], mode] = self.signs[gate, mode]
        return result

    def to_dense(self, max_qubits: int) -> np.ndarray:
        """Explicitly adapt to ``(gate, 4**max_qubits - 1, mode)`` when bounded."""

        if max_qubits < self.n_qubits:
            raise ValueError("max_qubits cannot truncate estimator labels")
        labels = pauli_labels(max_qubits)[1:]
        index = {label: position for position, label in enumerate(labels)}
        result = np.zeros((self.gate_count, len(labels), 3), dtype=float)
        prefix = "I" * (max_qubits - self.n_qubits)
        for gate, modes in enumerate(self.mode_labels):
            for mode, label in enumerate(modes):
                result[gate, index[prefix + label], mode] = self.signs[gate, mode]
        return result


def estimate_nearest_clifford_transport(
    operations: Sequence[CircuitOperation], n_qubits: int
) -> SparseCliffordTransport:
    """Replace each ideal operation by its nearest same-arity Clifford."""

    values = tuple(operations)
    if n_qubits < 1 or not values:
        raise ValueError("operations and a positive n_qubits are required")
    suffix = CliffordTableau.identity(n_qubits)
    labels: list[tuple[str, str, str] | None] = [None] * len(values)
    signs = np.empty((len(values), 3), dtype=float)
    for index in range(len(values) - 1, -1, -1):
        operation = values[index]
        if not isinstance(operation, CircuitOperation):
            raise TypeError("operations must contain CircuitOperation values")
        if len(operation.qubits) not in (1, 2) or any(qubit >= n_qubits for qubit in operation.qubits):
            raise ValueError("nearest-Clifford estimator supports valid one- and two-qubit operations only")
        modes = local_generator_labels(len(operation.qubits))
        images = [suffix.conjugate(embed_pauli_label(mode, operation.qubits, n_qubits)) for mode in modes]
        labels[index] = tuple(image.label for image in images)
        signs[index] = [image.sign for image in images]
        suffix = _embed_tableau(closest_clifford(operation.unitary), operation.qubits, n_qubits).compose(suffix)
    return SparseCliffordTransport(n_qubits, tuple(label for label in labels if label is not None), signs)
