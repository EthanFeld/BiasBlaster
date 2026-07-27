"""Optional closed-system piecewise-constant pulse validation.

This validates an ideal Hamiltonian model only. It excludes leakage,
decoherence, transfer-function distortion, calibration drift, and hardware
validation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm

from .pauli import pauli_matrix


@dataclass(frozen=True)
class PulseSimulationResult:
    pulse: np.ndarray
    unitary: np.ndarray
    derivatives: np.ndarray
    dt: float


@dataclass(frozen=True)
class PulseCalibrationResult:
    """Local error coordinates and exact pulse-coordinate derivatives.

    ``generators`` are coefficients of the principal, traceless logarithm of
    the phase-normalized ``target_unitary.conj().T @ unitary``.  Therefore the
    modeled operation is ``target_unitary @ exp(-1j * H_error)`` up to global
    phase: the same pre-gate error convention used by circuit transport.
    """

    simulation: PulseSimulationResult
    generators: np.ndarray
    jacobians: np.ndarray
    relative_error: np.ndarray
    omitted_generator_norm: float


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
        eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
        phases = np.exp(-1j * eigenvalues * dt)
        exponentials.append((eigenvectors * phases) @ eigenvectors.conj().T)
        differences = eigenvalues[:, None] - eigenvalues[None, :]
        frechet_factor = (
            -1j
            * dt
            * np.exp(-0.5j * dt * (eigenvalues[:, None] + eigenvalues[None, :]))
            * np.sinc(dt * differences / (2.0 * np.pi))
        )
        frechets.append(
            [
                eigenvectors
                @ (frechet_factor * (eigenvectors.conj().T @ control @ eigenvectors))
                @ eigenvectors.conj().T
                for control in controls
            ]
        )

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


def calibrate_pulse_error_model(
    pulse: np.ndarray,
    drift_hamiltonian: np.ndarray,
    control_hamiltonians: list[np.ndarray] | tuple[np.ndarray, ...],
    gate_time: float,
    target_unitary: np.ndarray,
    generator_labels: tuple[str, ...],
    *,
    atol: float = 1.0e-7,
) -> PulseCalibrationResult:
    """Derive local-error coordinates and Jacobians from pulse evolution.

    This is not a finite-difference calibration.  The pulse propagator returns
    exact Frechet derivatives ``dU/dp``.  They are moved into the phase-
    normalized pre-gate error frame ``E ~ U_target† U`` and converted to derivatives of the
    principal logarithm ``E = exp(-i H)`` by the inverse differential of the
    matrix exponential.  The result is valid while no eigenvalue difference
    of ``H`` crosses a nonzero multiple of ``2π``; such a branch singularity
    is rejected rather than silently producing a bad Jacobian.
    """

    simulation = simulate_piecewise_constant_pulse(
        pulse, drift_hamiltonian, control_hamiltonians, gate_time
    )
    target = np.asarray(target_unitary, dtype=complex)
    dimension = simulation.unitary.shape[0]
    if target.shape != (dimension, dimension) or not np.allclose(
        target.conj().T @ target, np.eye(dimension), atol=atol, rtol=0.0
    ):
        raise ValueError("target_unitary must be unitary with the pulse dimension")
    if not generator_labels or any(len(label) != int(round(np.log2(dimension))) for label in generator_labels):
        raise ValueError("generator_labels must match the pulse Hilbert-space dimension")

    # Global phase has no error content.  Trace phase first selects the branch
    # connected to identity; a determinant correction then puts E exactly in
    # SU(d) before taking a traceless principal logarithm.
    relative = target.conj().T @ simulation.unitary
    # Trace phase is continuous for the small relative errors this local chart
    # admits. Applying a determinant root only after that choice avoids its
    # branch jump for targets such as a CNOT represented up to global phase.
    if abs(np.trace(relative)) <= atol:
        raise ValueError("relative pulse error is too far from identity for local calibration")
    phase = np.angle(np.trace(relative))
    relative = np.exp(-1j * phase) * relative
    determinant_phase = np.angle(np.linalg.det(relative)) / dimension
    phase += determinant_phase
    relative = np.exp(-1j * determinant_phase) * relative
    # ``relative`` is unitary, so its principal logarithm has an exact small
    # spectral form.  This avoids a general Schur/logm solve for every local
    # 2x2 or 4x4 pulse while preserving the principal-log branch.
    eigenphases, eigenvectors = np.linalg.eig(relative)
    if not np.allclose(
        eigenvectors.conj().T @ eigenvectors,
        np.eye(dimension),
        atol=atol,
        rtol=0.0,
    ):
        raise ValueError("relative pulse error has no stable unitary eigensystem")
    generator = eigenvectors @ np.diag(-np.angle(eigenphases)) @ eigenvectors.conj().T
    generator = 0.5 * (generator + generator.conj().T)
    generator -= np.trace(generator) * np.eye(dimension) / dimension
    if not np.allclose(expm(-1j * generator), relative, atol=atol, rtol=0.0):
        raise ValueError("relative pulse error is outside the principal-log chart")

    eigenvalues, eigenvectors = np.linalg.eigh(generator)
    differences = eigenvalues[:, None] - eigenvalues[None, :]
    # integral_0^1 exp(i s delta) ds = exp(i delta / 2) sinc(delta / 2).
    dexp_factor = np.exp(0.5j * differences) * np.sinc(differences / (2.0 * np.pi))
    if np.any(np.abs(dexp_factor) < atol):
        raise ValueError("pulse error is at a logarithm-derivative branch singularity")

    basis = tuple(pauli_matrix(label) for label in generator_labels)
    coordinates = np.array(
        [float((np.trace(matrix @ generator) / dimension).real) for matrix in basis]
    )
    reconstructed = sum(value * matrix for value, matrix in zip(coordinates, basis, strict=True))
    omitted = float(np.linalg.norm(generator - reconstructed))

    derivatives = np.empty((len(generator_labels), simulation.derivatives.shape[0]), dtype=float)
    for parameter, derivative in enumerate(simulation.derivatives):
        d_relative = np.exp(-1j * phase) * target.conj().T @ derivative
        tangent = 1j * relative.conj().T @ d_relative
        tangent = 0.5 * (tangent + tangent.conj().T)
        tangent -= np.trace(tangent) * np.eye(dimension) / dimension
        tangent_eigenbasis = eigenvectors.conj().T @ tangent @ eigenvectors
        d_generator = eigenvectors @ (tangent_eigenbasis / dexp_factor) @ eigenvectors.conj().T
        d_generator = 0.5 * (d_generator + d_generator.conj().T)
        for mode, matrix in enumerate(basis):
            derivatives[mode, parameter] = float((np.trace(matrix @ d_generator) / dimension).real)

    return PulseCalibrationResult(
        simulation=simulation,
        generators=coordinates,
        jacobians=derivatives,
        relative_error=relative,
        omitted_generator_norm=omitted,
    )
