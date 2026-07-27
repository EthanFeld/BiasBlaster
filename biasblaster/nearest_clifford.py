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


def nearest_clifford_process_fidelity(unitary: np.ndarray) -> float:
    """Return local process fidelity to the selected nearest Clifford.

    This is a per-operation diagnostic, not a circuit-level approximation
    bound. It does not certify error from replacing several gates.
    """

    value = np.asarray(unitary, dtype=complex)
    candidate = closest_clifford(value)
    dimension = value.shape[0]
    score = float(np.trace(_tableau_transfer(candidate).T @ _pauli_transfer(value)))
    return float(np.clip(score / (dimension * dimension), 0.0, 1.0))


def _embed_tableau(local: CliffordTableau, qubits: tuple[int, ...], n_qubits: int) -> CliffordTableau:
    result = CliffordTableau.identity(n_qubits)
    x, z = list(result.x_images), list(result.z_images)
    for local_qubit, global_qubit in enumerate(qubits):
        for image, target in ((local.x_images[local_qubit], x), (local.z_images[local_qubit], z)):
            # Tableau labels use q0 as their rightmost factor, whereas
            # embed_pauli_label accepts characters in qarg order.
            label = embed_pauli_label(image.label[::-1], qubits, n_qubits)
            target[global_qubit] = SymplecticPauli.from_label(label, image.sign)
    return CliffordTableau(n_qubits, tuple(x), tuple(z))


@dataclass(frozen=True)
class SparseCliffordTransport:
    """One signed final Pauli destination per gate and generator mode."""

    n_qubits: int
    mode_labels: tuple[tuple[str, ...], ...]
    signs: np.ndarray
    generator_mode_count: int = 3
    mode_counts: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        signs = np.asarray(self.signs, dtype=float)
        if self.n_qubits < 1 or not self.mode_labels or self.generator_mode_count not in (3, 15):
            raise ValueError("mode_labels need a positive register and 3 or 15 modes")
        if signs.shape != (len(self.mode_labels), self.generator_mode_count):
            raise ValueError("signs must have shape (gate_count, generator_mode_count)")
        modes = tuple(tuple(str(label) for label in row) for row in self.mode_labels)
        if any(len(row) != self.generator_mode_count for row in modes):
            raise ValueError("every gate must store generator_mode_count labels")
        counts = self.mode_counts or (self.generator_mode_count,) * len(modes)
        if len(counts) != len(modes) or any(count < 1 or count > self.generator_mode_count for count in counts):
            raise ValueError("mode_counts must match gates and generator_mode_count")
        if any(
            len(label) != self.n_qubits or any(character not in "IXYZ" for character in label)
            for row in modes
            for label in row
        ):
            raise ValueError("invalid final Pauli label")
        for gate, count in enumerate(counts):
            if not np.all(np.isin(signs[gate, :count], (-1.0, 1.0))) or np.any(signs[gate, count:]):
                raise ValueError("active signs must be ±1 and inactive signs zero")
        object.__setattr__(self, "mode_labels", modes)
        object.__setattr__(self, "signs", signs.copy())
        object.__setattr__(self, "mode_counts", tuple(counts))

    @property
    def gate_count(self) -> int:
        return len(self.mode_labels)

    @property
    def support(self) -> int:
        return len({
            label
            for modes, count in zip(self.mode_labels, self.mode_counts, strict=True)
            for label in modes[:count]
        })

    @property
    def support_count(self) -> int:
        return self.support

    @property
    def truncation_bound(self) -> float:
        return 0.0

    @property
    def retained_payload_bytes(self) -> int:
        """Return numeric/string payload bytes, not Python object memory."""

        return int(sum(self.mode_counts) * self.n_qubits + self.signs.nbytes)

    @property
    def retained_bytes(self) -> int:
        """Compatibility alias for :attr:`retained_payload_bytes`."""

        return self.retained_payload_bytes

    def _validate_generators(self, generators: np.ndarray) -> np.ndarray:
        local = np.asarray(generators, dtype=float)
        if local.shape != (self.gate_count, self.generator_mode_count) or not np.all(np.isfinite(local)):
            raise ValueError("generators must have shape (gate_count, generator_mode_count) and be finite")
        if any(np.any(local[gate, count:]) for gate, count in enumerate(self.mode_counts)):
            raise ValueError("inactive generator modes must be zero")
        return local

    def truncation_bound_for(self, generators: np.ndarray) -> float:
        self._validate_generators(generators)
        return 0.0

    def contract(self, generators: np.ndarray) -> dict[str, float]:
        local = self._validate_generators(generators)
        result: dict[str, float] = {}
        for gate, (modes, count) in enumerate(zip(self.mode_labels, self.mode_counts, strict=True)):
            for mode, label in enumerate(modes[:count]):
                result[label] = result.get(label, 0.0) + float(self.signs[gate, mode] * local[gate, mode])
        return {label: value for label, value in result.items() if abs(value) > 1e-12}

    def local_priorities(self, generators: np.ndarray) -> np.ndarray:
        local = self._validate_generators(generators)
        bias = self.contract(local)
        priority = np.zeros((self.gate_count, self.generator_mode_count), dtype=float)
        for gate, (modes, count) in enumerate(zip(self.mode_labels, self.mode_counts, strict=True)):
            priority[gate, :count] = [
                self.signs[gate, mode] * bias.get(label, 0.0)
                for mode, label in enumerate(modes[:count])
            ]
        return priority

    def compact_transfer(self) -> np.ndarray:
        labels = tuple(sorted({
            label
            for modes, count in zip(self.mode_labels, self.mode_counts, strict=True)
            for label in modes[:count]
        }))
        index = {label: position for position, label in enumerate(labels)}
        result = np.zeros((self.gate_count, len(labels), self.generator_mode_count), dtype=float)
        for gate, (modes, count) in enumerate(zip(self.mode_labels, self.mode_counts, strict=True)):
            for mode, label in enumerate(modes[:count]):
                result[gate, index[label], mode] = self.signs[gate, mode]
        return result

    def to_dense(self, max_qubits: int) -> np.ndarray:
        """Explicitly adapt to ``(gate, 4**max_qubits - 1, mode)`` when bounded."""

        if max_qubits < self.n_qubits:
            raise ValueError("max_qubits cannot truncate estimator labels")
        labels = pauli_labels(max_qubits)[1:]
        index = {label: position for position, label in enumerate(labels)}
        result = np.zeros((self.gate_count, len(labels), self.generator_mode_count), dtype=float)
        prefix = "I" * (max_qubits - self.n_qubits)
        for gate, (modes, count) in enumerate(zip(self.mode_labels, self.mode_counts, strict=True)):
            for mode, label in enumerate(modes[:count]):
                result[gate, index[prefix + label], mode] = self.signs[gate, mode]
        return result


def estimate_nearest_clifford_transport(
    operations: Sequence[CircuitOperation],
    n_qubits: int,
    *,
    two_qubit_generator_basis: str = "diagonal",
) -> SparseCliffordTransport:
    """Replace each ideal operation by its nearest same-arity Clifford."""

    values = tuple(operations)
    if n_qubits < 1 or not values:
        raise ValueError("operations and a positive n_qubits are required")
    if two_qubit_generator_basis not in ("diagonal", "full-su4"):
        raise ValueError("two_qubit_generator_basis must be 'diagonal' or 'full-su4'")
    mode_count = 15 if two_qubit_generator_basis == "full-su4" else 3
    mode_counts = tuple(
        15 if mode_count == 15 and len(operation.qubits) == 2 else 3
        for operation in values
    )
    suffix = CliffordTableau.identity(n_qubits)
    labels: list[tuple[str, ...] | None] = [None] * len(values)
    signs = np.zeros((len(values), mode_count), dtype=float)
    for index in range(len(values) - 1, -1, -1):
        operation = values[index]
        if not isinstance(operation, CircuitOperation):
            raise TypeError("operations must contain CircuitOperation values")
        if len(operation.qubits) not in (1, 2) or any(qubit >= n_qubits for qubit in operation.qubits):
            raise ValueError("nearest-Clifford estimator supports valid one- and two-qubit operations only")
        modes = local_generator_labels(
            len(operation.qubits),
            full_two_qubit=mode_count == 15,
        )
        suffix = _embed_tableau(closest_clifford(operation.unitary), operation.qubits, n_qubits).compose(suffix)
        images = [suffix.conjugate(embed_pauli_label(mode, operation.qubits, n_qubits)) for mode in modes]
        labels[index] = tuple(image.label for image in images) + ("I" * n_qubits,) * (mode_count - len(images))
        signs[index, :len(images)] = [image.sign for image in images]
    return SparseCliffordTransport(
        n_qubits,
        tuple(label for label in labels if label is not None),
        signs,
        generator_mode_count=mode_count,
        mode_counts=mode_counts,
    )
