"""Optional closed-system piecewise-constant pulse validation.

This validates an ideal Hamiltonian model only. It excludes leakage,
decoherence, transfer-function distortion, calibration drift, and hardware
validation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm, expm_frechet


@dataclass(frozen=True)
class PulseSimulationResult:
    pulse: np.ndarray
    unitary: np.ndarray
    derivatives: np.ndarray
    dt: float


def simulate_piecewise_constant_pulse(
    pulse: np.ndarray,
    drift_hamiltonian: np.ndarray,
    control_hamiltonians: list[np.ndarray] | tuple[np.ndarray, ...],
    gate_time: float,
) -> PulseSimulationResult:
    """Return closed-system unitary evolution and exact control derivatives."""

    values = np.asarray(pulse, dtype=float)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.ndim != 2 or values.shape[0] < 1:
        raise ValueError("pulse must be a nonempty vector or segment-by-control matrix")
    drift = np.asarray(drift_hamiltonian, dtype=complex)
    controls = tuple(np.asarray(hamiltonian, dtype=complex) for hamiltonian in control_hamiltonians)
    dimension = drift.shape[0] if drift.ndim == 2 else 0
    if drift.shape != (dimension, dimension) or not controls or any(control.shape != (dimension, dimension) for control in controls):
        raise ValueError("drift and controls must be same-size square matrices")
    if values.shape[1] != len(controls):
        raise ValueError("pulse control count does not match control Hamiltonians")
    if gate_time <= 0:
        raise ValueError("gate_time must be positive")
    if not np.allclose(drift, drift.conj().T, atol=1e-10) or any(not np.allclose(control, control.conj().T, atol=1e-10) for control in controls):
        raise ValueError("Hamiltonians must be Hermitian")

    segments = values.shape[0]
    dt = float(gate_time) / segments
    exponentials: list[np.ndarray] = []
    frechets: list[list[np.ndarray]] = []
    for row in values:
        hamiltonian = drift + sum(value * control for value, control in zip(row, controls))
        argument = -1j * hamiltonian * dt
        exponentials.append(expm(argument))
        frechets.append([expm_frechet(argument, -1j * control * dt, compute_expm=False) for control in controls])

    before = [np.eye(dimension, dtype=complex)]
    for exponential in exponentials:
        before.append(exponential @ before[-1])
    after = [np.eye(dimension, dtype=complex) for _ in range(segments)]
    suffix = np.eye(dimension, dtype=complex)
    for segment in range(segments - 1, -1, -1):
        after[segment] = suffix
        suffix = suffix @ exponentials[segment]
    derivatives = np.empty((values.size, dimension, dimension), dtype=complex)
    cursor = 0
    for segment in range(segments):
        for control in range(len(controls)):
            derivatives[cursor] = after[segment] @ frechets[segment][control] @ before[segment]
            cursor += 1
    return PulseSimulationResult(values.copy(), before[-1], derivatives, dt)
