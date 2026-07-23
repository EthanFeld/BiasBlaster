"""Signed symplectic tableaux for Clifford Pauli conjugation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np

from .pauli import pauli_labels, pauli_matrix


def _bit_count(value: int) -> int:
    return int(value.bit_count())


@dataclass(frozen=True)
class SymplecticPauli:
    x: int
    z: int
    n_qubits: int
    sign: int = 1

    def __post_init__(self) -> None:
        if self.n_qubits < 1:
            raise ValueError("n_qubits must be positive")
        mask = (1 << self.n_qubits) - 1
        if self.x & ~mask or self.z & ~mask or self.sign not in (-1, 1):
            raise ValueError("invalid symplectic Pauli")

    @property
    def label(self) -> str:
        characters: list[str] = []
        for qubit in range(self.n_qubits - 1, -1, -1):
            x = (self.x >> qubit) & 1
            z = (self.z >> qubit) & 1
            characters.append("Y" if x and z else "X" if x else "Z" if z else "I")
        return "".join(characters)

    @classmethod
    def from_label(cls, label: str, sign: int = 1) -> "SymplecticPauli":
        if not label or any(character not in "IXYZ" for character in label):
            raise ValueError("invalid Pauli label")
        x = z = 0
        for qubit, character in enumerate(reversed(label)):
            if character in "XY":
                x |= 1 << qubit
            if character in "ZY":
                z |= 1 << qubit
        return cls(x, z, len(label), int(sign))


@dataclass(frozen=True)
class _PauliTerm:
    x: int
    z: int
    phase: int = 0


def _multiply(left: _PauliTerm, right: _PauliTerm) -> _PauliTerm:
    x = left.x ^ right.x
    z = left.z ^ right.z
    phase = (
        left.phase
        + right.phase
        + _bit_count(left.x & left.z)
        + _bit_count(right.x & right.z)
        - _bit_count(x & z)
        + 2 * _bit_count(left.z & right.x)
    ) % 4
    return _PauliTerm(x, z, phase)


def _image_term(image: SymplecticPauli) -> _PauliTerm:
    return _PauliTerm(image.x, image.z, 0 if image.sign == 1 else 2)


@dataclass(frozen=True)
class CliffordTableau:
    """Images of each X and Z generator under ``U P U†``."""

    n_qubits: int
    x_images: tuple[SymplecticPauli, ...]
    z_images: tuple[SymplecticPauli, ...]

    def __post_init__(self) -> None:
        if self.n_qubits < 1 or len(self.x_images) != self.n_qubits or len(self.z_images) != self.n_qubits:
            raise ValueError("tableau needs one X and Z image per qubit")
        if any(image.n_qubits != self.n_qubits for image in self.x_images + self.z_images):
            raise ValueError("tableau image register mismatch")

    @classmethod
    def identity(cls, n_qubits: int) -> "CliffordTableau":
        if n_qubits < 1:
            raise ValueError("n_qubits must be positive")
        return cls(
            n_qubits,
            tuple(SymplecticPauli(1 << qubit, 0, n_qubits) for qubit in range(n_qubits)),
            tuple(SymplecticPauli(0, 1 << qubit, n_qubits) for qubit in range(n_qubits)),
        )

    @property
    def symplectic_matrix(self) -> np.ndarray:
        matrix = np.zeros((2 * self.n_qubits, 2 * self.n_qubits), dtype=np.uint8)
        for column, image in enumerate(self.x_images + self.z_images):
            for bit in range(self.n_qubits):
                matrix[bit, column] = (image.x >> bit) & 1
                matrix[self.n_qubits + bit, column] = (image.z >> bit) & 1
        return matrix

    @classmethod
    def from_symplectic(cls, matrix: np.ndarray, signs: Iterable[int] | None = None) -> "CliffordTableau":
        value = np.asarray(matrix, dtype=int)
        if value.ndim != 2 or value.shape[0] != value.shape[1] or value.shape[0] % 2:
            raise ValueError("symplectic matrix must be square with even dimension")
        n_qubits = value.shape[0] // 2
        value = value % 2
        omega = np.block(
            [
                [np.zeros((n_qubits, n_qubits), dtype=int), np.eye(n_qubits, dtype=int)],
                [np.eye(n_qubits, dtype=int), np.zeros((n_qubits, n_qubits), dtype=int)],
            ]
        )
        if not np.array_equal((value.T @ omega @ value) % 2, omega):
            raise ValueError("matrix is not symplectic")
        phase = tuple(1 for _ in range(2 * n_qubits)) if signs is None else tuple(int(sign) for sign in signs)
        if len(phase) != 2 * n_qubits or any(sign not in (-1, 1) for sign in phase):
            raise ValueError("signs must contain ±1 for every generator image")

        def image(column: int) -> SymplecticPauli:
            x = sum(int(value[bit, column]) << bit for bit in range(n_qubits))
            z = sum(int(value[n_qubits + bit, column]) << bit for bit in range(n_qubits))
            return SymplecticPauli(x, z, n_qubits, phase[column])

        return cls(n_qubits, tuple(image(q) for q in range(n_qubits)), tuple(image(n_qubits + q) for q in range(n_qubits)))

    def conjugate(self, pauli: str | SymplecticPauli) -> SymplecticPauli:
        value = pauli if isinstance(pauli, SymplecticPauli) else SymplecticPauli.from_label(pauli)
        if value.n_qubits != self.n_qubits:
            raise ValueError("Pauli register mismatch")
        term = _PauliTerm(0, 0, 0)
        for qubit in range(self.n_qubits):
            if (value.x >> qubit) & 1:
                term = _multiply(term, _image_term(self.x_images[qubit]))
        for qubit in range(self.n_qubits):
            if (value.z >> qubit) & 1:
                term = _multiply(term, _image_term(self.z_images[qubit]))
        term = _PauliTerm(term.x, term.z, (term.phase + _bit_count(value.x & value.z)) % 4)
        if term.phase not in (0, 2):
            raise ValueError("tableau maps a Hermitian Pauli to a non-Hermitian phase")
        return SymplecticPauli(term.x, term.z, self.n_qubits, value.sign * (1 if term.phase == 0 else -1))

    def compose(self, after: "CliffordTableau") -> "CliffordTableau":
        """Return ``after ∘ self`` for application-order composition."""

        if after.n_qubits != self.n_qubits:
            raise ValueError("tableau register mismatch")
        return CliffordTableau(
            self.n_qubits,
            tuple(after.conjugate(image) for image in self.x_images),
            tuple(after.conjugate(image) for image in self.z_images),
        )


def _named_tableau(name: str, n_qubits: int, qubits: Iterable[int]) -> CliffordTableau:
    name = name.upper()
    qs = tuple(int(qubit) for qubit in qubits)
    if name in {"H", "S", "SDG"} and len(qs) == 1:
        qubit = qs[0]
        if qubit < 0 or qubit >= n_qubits:
            raise ValueError("qubit outside register")
        identity = CliffordTableau.identity(n_qubits)
        x, z = list(identity.x_images), list(identity.z_images)
        if name == "H":
            x[qubit] = SymplecticPauli(0, 1 << qubit, n_qubits)
            z[qubit] = SymplecticPauli(1 << qubit, 0, n_qubits)
        elif name == "S":
            x[qubit] = SymplecticPauli(1 << qubit, 1 << qubit, n_qubits)
        else:
            x[qubit] = SymplecticPauli(1 << qubit, 1 << qubit, n_qubits, -1)
        return CliffordTableau(n_qubits, tuple(x), tuple(z))
    if name in {"CZ", "CNOT", "SWAP", "ISWAP"} and len(qs) == 2:
        q0, q1 = qs
        if q0 == q1 or min(qs) < 0 or max(qs) >= n_qubits:
            raise ValueError("invalid two-qubit gate qubits")
        identity = CliffordTableau.identity(n_qubits)
        x, z = list(identity.x_images), list(identity.z_images)
        if name == "CZ":
            x[q0] = SymplecticPauli(1 << q0, 1 << q1, n_qubits)
            x[q1] = SymplecticPauli(1 << q1, 1 << q0, n_qubits)
        elif name == "CNOT":
            x[q0] = SymplecticPauli((1 << q0) | (1 << q1), 0, n_qubits)
            z[q1] = SymplecticPauli(0, (1 << q0) | (1 << q1), n_qubits)
        elif name == "SWAP":
            x[q0] = SymplecticPauli(1 << q1, 0, n_qubits)
            x[q1] = SymplecticPauli(1 << q0, 0, n_qubits)
            z[q0] = SymplecticPauli(0, 1 << q1, n_qubits)
            z[q1] = SymplecticPauli(0, 1 << q0, n_qubits)
        else:
            x[q0] = SymplecticPauli(1 << q1, (1 << q0) | (1 << q1), n_qubits)
            x[q1] = SymplecticPauli(1 << q0, (1 << q0) | (1 << q1), n_qubits)
            z[q0] = SymplecticPauli(0, 1 << q1, n_qubits)
            z[q1] = SymplecticPauli(0, 1 << q0, n_qubits)
        return CliffordTableau(n_qubits, tuple(x), tuple(z))
    raise ValueError(f"unsupported named Clifford {name}")


def named_clifford(name: str, n_qubits: int, qubits: Iterable[int]) -> CliffordTableau:
    return _named_tableau(name, n_qubits, qubits)


def pauli_to_symplectic(label: str) -> tuple[int, int]:
    value = SymplecticPauli.from_label(label)
    return value.x, value.z


def symplectic_to_pauli(x: int, z: int, n_qubits: int | None = None) -> str:
    if n_qubits is None:
        n_qubits = max(1, int(x).bit_length(), int(z).bit_length())
    return SymplecticPauli(int(x), int(z), n_qubits).label


def tableau_from_symplectic(
    matrix: np.ndarray, signs: Iterable[int] | None = None
) -> CliffordTableau:
    return CliffordTableau.from_symplectic(matrix, signs)


def tableau_from_gate_sequence(
    gates: Iterable[CliffordTableau | tuple[str, Iterable[int]]], n_qubits: int
) -> CliffordTableau:
    result = CliffordTableau.identity(n_qubits)
    for gate in gates:
        current = gate if isinstance(gate, CliffordTableau) else _named_tableau(gate[0], n_qubits, gate[1])
        result = result.compose(current)
    return result


def tableau_from_unitary(unitary: np.ndarray, *, tol: float = 1e-9) -> CliffordTableau:
    """Infer a tableau for a small unitary, raising ``ValueError`` if non-Clifford."""

    value = np.asarray(unitary, dtype=complex)
    if value.ndim != 2 or value.shape[0] != value.shape[1]:
        raise ValueError("unitary must be square")
    dimension = value.shape[0]
    n_qubits = int(round(np.log2(dimension)))
    if 2**n_qubits != dimension or not np.allclose(value.conj().T @ value, np.eye(dimension), atol=tol, rtol=0):
        raise ValueError("unitary is not a valid qubit unitary")
    x_images: list[SymplecticPauli] = []
    z_images: list[SymplecticPauli] = []
    for qubit in range(n_qubits):
        for source, target in (("X", x_images), ("Z", z_images)):
            label = ["I"] * n_qubits
            label[n_qubits - 1 - qubit] = source
            transformed = value @ pauli_matrix("".join(label)) @ value.conj().T
            coefficients = np.array(
                [np.trace(pauli_matrix(output) @ transformed) / dimension for output in pauli_labels(n_qubits)]
            )
            indices = np.flatnonzero(np.abs(coefficients) > tol)
            if len(indices) != 1 or not np.isclose(abs(coefficients[indices[0]]), 1.0, atol=tol):
                raise ValueError("unitary is not Clifford")
            coefficient = coefficients[indices[0]]
            if abs(coefficient.imag) > tol:
                raise ValueError("unitary has a non-Hermitian Pauli image")
            target.append(SymplecticPauli.from_label(pauli_labels(n_qubits)[indices[0]], int(np.sign(coefficient.real))))
    return CliffordTableau(n_qubits, tuple(x_images), tuple(z_images))


def conjugate_pauli(pauli: str | SymplecticPauli, tableau: CliffordTableau) -> tuple[str, int]:
    result = tableau.conjugate(pauli)
    return result.label, result.sign


def conjugate_sparse_pauli(coefficients: Mapping[str, float], tableau: CliffordTableau) -> dict[str, float]:
    result: dict[str, float] = {}
    for label, value in coefficients.items():
        image = tableau.conjugate(label)
        result[image.label] = result.get(image.label, 0.0) + image.sign * float(value)
    return {label: value for label, value in result.items() if value != 0.0}


SymplecticTableau = CliffordTableau
PauliString = SymplecticPauli
