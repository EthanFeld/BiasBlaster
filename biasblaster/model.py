"""Small data contracts for ideal circuit operations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CircuitOperation:
    """An ideal one- or two-qubit operation in application order."""

    name: str
    qubits: tuple[int, ...]
    unitary: np.ndarray
    angle: float = 0.0

    def __post_init__(self) -> None:
        qubits = tuple(int(qubit) for qubit in self.qubits)
        if not qubits or len(set(qubits)) != len(qubits) or min(qubits) < 0:
            raise ValueError("operation qubits must be nonempty, distinct, and nonnegative")
        unitary = np.asarray(self.unitary, dtype=complex)
        dimension = 2 ** len(qubits)
        if unitary.shape != (dimension, dimension):
            raise ValueError("unitary dimension does not match operation arity")
        if not np.all(np.isfinite(unitary)):
            raise ValueError("unitary must be finite")
        if not np.allclose(
            unitary.conj().T @ unitary,
            np.eye(dimension, dtype=complex),
            atol=1e-9,
            rtol=0.0,
        ):
            raise ValueError("unitary must be unitary")
        angle = float(self.angle)
        if not np.isfinite(angle):
            raise ValueError("angle must be finite")
        # A frozen dataclass does not freeze ndarray contents.  Keep the
        # validated unitary immutable so later in-place mutation cannot
        # invalidate a CircuitOperation.
        unitary = unitary.copy()
        unitary.setflags(write=False)
        object.__setattr__(self, "qubits", qubits)
        object.__setattr__(self, "unitary", unitary)
        object.__setattr__(self, "angle", angle)
