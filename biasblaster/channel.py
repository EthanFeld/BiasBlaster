"""Channel-level Pauli propagation for calibrated hardware noise.

This generalizes BiasBlaster's coherent Pauli-generator transport to arbitrary
one- and two-qubit Pauli-transfer-matrix (PTM) perturbations. Error modes live
on circuit boundaries: 0 is before the first gate, g+1 is after gate g, and
len(circuit) is immediately before measurement.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Mapping, Sequence

import numpy as np

from .local_transfer import local_pauli_transfer
from .model import CircuitOperation
from .pauli import extract_pauli_label, pauli_labels, pauli_matrix


@dataclass(frozen=True)
class ChannelMode:
    """One scalar calibrated error parameter and its local PTM derivative."""

    name: str
    boundary: int
    qubits: tuple[int, ...]
    delta_ptm: np.ndarray
    mean: float = 1.0

    def __post_init__(self) -> None:
        qubits = tuple(int(q) for q in self.qubits)
        if not self.name or self.boundary < 0:
            raise ValueError("mode name must be nonempty and boundary nonnegative")
        if not qubits or len(set(qubits)) != len(qubits) or min(qubits) < 0:
            raise ValueError("mode qubits must be nonempty, distinct, and nonnegative")
        if len(qubits) not in (1, 2):
            raise ValueError("channel modes currently support one- and two-qubit locality")
        delta = np.asarray(self.delta_ptm, dtype=float)
        dim = 4 ** len(qubits)
        if delta.shape != (dim, dim) or not np.all(np.isfinite(delta)):
            raise ValueError("delta_ptm has the wrong shape or contains non-finite values")
        mean = float(self.mean)
        if not np.isfinite(mean):
            raise ValueError("mode mean must be finite")
        delta = delta.copy()
        delta.setflags(write=False)
        object.__setattr__(self, "qubits", qubits)
        object.__setattr__(self, "delta_ptm", delta)
        object.__setattr__(self, "mean", mean)


@dataclass(frozen=True)
class NoiseChannelApplication:
    """A finite local PTM used to validate the first-order model."""

    name: str
    boundary: int
    qubits: tuple[int, ...]
    ptm: np.ndarray

    def __post_init__(self) -> None:
        qubits = tuple(int(q) for q in self.qubits)
        if not self.name or self.boundary < 0:
            raise ValueError("channel name must be nonempty and boundary nonnegative")
        if not qubits or len(set(qubits)) != len(qubits) or min(qubits) < 0:
            raise ValueError("channel qubits must be nonempty, distinct, and nonnegative")
        if len(qubits) not in (1, 2):
            raise ValueError("noise channels currently support one- and two-qubit locality")
        ptm = np.asarray(self.ptm, dtype=float)
        dim = 4 ** len(qubits)
        if ptm.shape != (dim, dim) or not np.all(np.isfinite(ptm)):
            raise ValueError("ptm has the wrong shape or contains non-finite values")
        ptm = ptm.copy()
        ptm.setflags(write=False)
        object.__setattr__(self, "qubits", qubits)
        object.__setattr__(self, "ptm", ptm)


@dataclass(frozen=True)
class ObservableImpactBatch:
    """First-order bias, sensitivities, and covariance for observables."""

    names: tuple[str, ...]
    ideal: np.ndarray
    bias: np.ndarray
    predicted: np.ndarray
    jacobian: np.ndarray
    mode_names: tuple[str, ...]
    mode_means: np.ndarray
    covariance: np.ndarray | None
    mode_covariance: np.ndarray | None
    forward_dropped_l2: float
    backward_dropped_l2: np.ndarray

    @property
    def variances(self) -> np.ndarray | None:
        return None if self.covariance is None else np.diag(self.covariance).copy()


def _check_sparse(values: Mapping[str, float], n_qubits: int, name: str) -> dict[str, float]:
    result: dict[str, float] = {}
    for label, value in values.items():
        if len(label) != n_qubits or any(c not in "IXYZ" for c in label):
            raise ValueError(f"{name} contains an invalid Pauli label")
        value = float(value)
        if not np.isfinite(value):
            raise ValueError(f"{name} coefficients must be finite")
        if value:
            result[label] = value
    return result


def _replace_local(global_label: str, local_label: str, qubits: tuple[int, ...]) -> str:
    chars = list(global_label)
    n_qubits = len(global_label)
    for q, char in zip(qubits, local_label):
        chars[n_qubits - 1 - q] = char
    return "".join(chars)


def _truncate(values: dict[str, float], support_cap: int | None) -> tuple[dict[str, float], float]:
    if support_cap is None or len(values) <= support_cap:
        return values, 0.0
    if support_cap < 1:
        raise ValueError("support_cap must be positive")
    kept = sorted(values.items(), key=lambda item: abs(item[1]), reverse=True)[:support_cap]
    labels = {label for label, _ in kept}
    dropped = sqrt(sum(v * v for label, v in values.items() if label not in labels))
    return dict(kept), dropped


def apply_local_ptm_sparse(
    coefficients: Mapping[str, float],
    qubits: tuple[int, ...],
    ptm: np.ndarray,
    n_qubits: int,
    *,
    adjoint: bool = False,
    coefficient_tol: float = 1e-12,
    support_cap: int | None = None,
) -> tuple[dict[str, float], float]:
    """Apply a local PTM to sparse state moments or observable coefficients."""

    if coefficient_tol < 0:
        raise ValueError("coefficient_tol must be nonnegative")
    qubits = tuple(int(q) for q in qubits)
    if not qubits or len(set(qubits)) != len(qubits) or any(q < 0 or q >= n_qubits for q in qubits):
        raise ValueError("invalid local qubits")
    labels = pauli_labels(len(qubits))
    matrix = np.asarray(ptm, dtype=float)
    if matrix.shape != (len(labels), len(labels)):
        raise ValueError("ptm dimension does not match qubit arity")
    if adjoint:
        matrix = matrix.T
    result: dict[str, float] = {}
    for global_label, value in coefficients.items():
        if len(global_label) != n_qubits:
            raise ValueError("global Pauli label has the wrong register size")
        source = labels.index(extract_pauli_label(global_label, qubits))
        for target, factor in enumerate(matrix[:, source]):
            if abs(factor) <= coefficient_tol:
                continue
            output = _replace_local(global_label, labels[target], qubits)
            result[output] = result.get(output, 0.0) + float(value) * float(factor)
    result = {label: value for label, value in result.items() if abs(value) > coefficient_tol}
    return _truncate(result, support_cap)


def pauli_inner(observable: Mapping[str, float], state: Mapping[str, float]) -> float:
    """Return Tr(O rho) from Pauli expansion coefficients and Pauli moments."""

    return float(sum(value * state.get(label, 0.0) for label, value in observable.items()))


def contract_local_perturbation(
    forward_state: Mapping[str, float],
    backward_observable: Mapping[str, float],
    qubits: tuple[int, ...],
    delta_ptm: np.ndarray,
    n_qubits: int,
    *,
    coefficient_tol: float = 1e-12,
) -> float:
    """Evaluate o^T Delta r without constructing a global superoperator."""

    labels = pauli_labels(len(qubits))
    delta = np.asarray(delta_ptm, dtype=float)
    if delta.shape != (len(labels), len(labels)):
        raise ValueError("delta_ptm dimension does not match qubit arity")
    total = 0.0
    for global_label, state_value in forward_state.items():
        source = labels.index(extract_pauli_label(global_label, qubits))
        for target, factor in enumerate(delta[:, source]):
            if abs(factor) <= coefficient_tol:
                continue
            output = _replace_local(global_label, labels[target], qubits)
            total += backward_observable.get(output, 0.0) * float(factor) * float(state_value)
    return float(total)


def _forward_boundaries(circuit, initial, n_qubits, coefficient_tol, support_cap):
    current, dropped_total = _truncate(dict(initial), support_cap)
    states = [current]
    for operation in circuit:
        current, dropped = apply_local_ptm_sparse(
            current, operation.qubits, local_pauli_transfer(operation), n_qubits,
            coefficient_tol=coefficient_tol, support_cap=support_cap,
        )
        dropped_total += dropped
        states.append(current)
    return states, dropped_total


def _backward_boundaries(circuit, observable, n_qubits, coefficient_tol, support_cap):
    current, dropped_total = _truncate(dict(observable), support_cap)
    boundaries = [None] * (len(circuit) + 1)
    boundaries[-1] = current
    for gate_index in range(len(circuit) - 1, -1, -1):
        operation = circuit[gate_index]
        current, dropped = apply_local_ptm_sparse(
            current, operation.qubits, local_pauli_transfer(operation), n_qubits,
            adjoint=True, coefficient_tol=coefficient_tol, support_cap=support_cap,
        )
        dropped_total += dropped
        boundaries[gate_index] = current
    return boundaries, dropped_total


def estimate_observable_impacts(
    circuit: Sequence[CircuitOperation],
    initial_state: Mapping[str, float],
    observables: Mapping[str, Mapping[str, float]],
    modes: Sequence[ChannelMode],
    n_qubits: int,
    *,
    mode_covariance: np.ndarray | None = None,
    shot_covariance: np.ndarray | None = None,
    coefficient_tol: float = 1e-12,
    support_cap: int | None = None,
) -> ObservableImpactBatch:
    """Propagate calibrated channel bias and covariance to measured observables."""

    circuit = tuple(circuit)
    if n_qubits < 1:
        raise ValueError("n_qubits must be positive")
    if any(q >= n_qubits for op in circuit for q in op.qubits):
        raise ValueError("operation qubit outside register")
    if not observables:
        raise ValueError("at least one observable is required")
    initial = _check_sparse(initial_state, n_qubits, "initial_state")
    names = tuple(observables)
    checked = {name: _check_sparse(observables[name], n_qubits, name) for name in names}
    modes = tuple(modes)
    for mode in modes:
        if mode.boundary > len(circuit) or any(q >= n_qubits for q in mode.qubits):
            raise ValueError("channel mode lies outside the circuit/register")

    forward, forward_dropped = _forward_boundaries(
        circuit, initial, n_qubits, coefficient_tol, support_cap
    )
    ideal = np.empty(len(names), dtype=float)
    jacobian = np.empty((len(names), len(modes)), dtype=float)
    backward_dropped = np.empty(len(names), dtype=float)
    for obs_index, name in enumerate(names):
        observable = checked[name]
        backward, dropped = _backward_boundaries(
            circuit, observable, n_qubits, coefficient_tol, support_cap
        )
        backward_dropped[obs_index] = dropped
        ideal[obs_index] = pauli_inner(observable, forward[-1])
        for mode_index, mode in enumerate(modes):
            jacobian[obs_index, mode_index] = contract_local_perturbation(
                forward[mode.boundary], backward[mode.boundary], mode.qubits,
                mode.delta_ptm, n_qubits, coefficient_tol=coefficient_tol,
            )

    means = np.asarray([mode.mean for mode in modes], dtype=float)
    bias = jacobian @ means if modes else np.zeros(len(names), dtype=float)
    sigma_copy = None
    covariance = None
    if mode_covariance is not None:
        sigma = np.asarray(mode_covariance, dtype=float)
        if sigma.shape != (len(modes), len(modes)) or not np.allclose(sigma, sigma.T, atol=1e-10, rtol=0):
            raise ValueError("mode_covariance must be a symmetric n_modes x n_modes matrix")
        if np.min(np.linalg.eigvalsh(sigma), initial=0.0) < -1e-10:
            raise ValueError("mode_covariance must be positive semidefinite")
        sigma_copy = sigma.copy()
        covariance = jacobian @ sigma @ jacobian.T
    if shot_covariance is not None:
        shot = np.asarray(shot_covariance, dtype=float)
        if shot.shape != (len(names), len(names)) or not np.allclose(shot, shot.T, atol=1e-10, rtol=0):
            raise ValueError("shot_covariance must be a symmetric n_observables x n_observables matrix")
        covariance = shot.copy() if covariance is None else covariance + shot

    return ObservableImpactBatch(
        names=names, ideal=ideal, bias=bias, predicted=ideal + bias,
        jacobian=jacobian, mode_names=tuple(mode.name for mode in modes),
        mode_means=means, covariance=covariance, mode_covariance=sigma_copy,
        forward_dropped_l2=float(forward_dropped), backward_dropped_l2=backward_dropped,
    )


def propagate_noisy_observables(
    circuit: Sequence[CircuitOperation],
    initial_state: Mapping[str, float],
    observables: Mapping[str, Mapping[str, float]],
    channels: Sequence[NoiseChannelApplication],
    n_qubits: int,
    *,
    coefficient_tol: float = 1e-12,
    support_cap: int | None = None,
) -> dict[str, float]:
    """Apply finite local PTMs for nonlinear validation of the first-order model."""

    circuit = tuple(circuit)
    current = _check_sparse(initial_state, n_qubits, "initial_state")
    schedule: dict[int, list[NoiseChannelApplication]] = {}
    for channel in channels:
        if channel.boundary > len(circuit) or any(q >= n_qubits for q in channel.qubits):
            raise ValueError("noise channel lies outside the circuit/register")
        schedule.setdefault(channel.boundary, []).append(channel)
    for boundary in range(len(circuit) + 1):
        for channel in schedule.get(boundary, ()):
            current, _ = apply_local_ptm_sparse(
                current, channel.qubits, channel.ptm, n_qubits,
                coefficient_tol=coefficient_tol, support_cap=support_cap,
            )
        if boundary < len(circuit):
            operation = circuit[boundary]
            current, _ = apply_local_ptm_sparse(
                current, operation.qubits, local_pauli_transfer(operation), n_qubits,
                coefficient_tol=coefficient_tol, support_cap=support_cap,
            )
    return {
        name: pauli_inner(_check_sparse(obs, n_qubits, name), current)
        for name, obs in observables.items()
    }


def kraus_to_ptm(kraus: Sequence[np.ndarray]) -> np.ndarray:
    """Convert arbitrary one- or two-qubit Kraus operators to a real PTM."""

    operators = tuple(np.asarray(k, dtype=complex) for k in kraus)
    if not operators:
        raise ValueError("at least one Kraus operator is required")
    dimension = operators[0].shape[0]
    if dimension not in (2, 4) or any(k.shape != (dimension, dimension) for k in operators):
        raise ValueError("Kraus operators must share a one- or two-qubit square shape")
    arity = 1 if dimension == 2 else 2
    labels = pauli_labels(arity)
    matrices = tuple(pauli_matrix(label[::-1]) for label in labels)
    result = np.empty((len(labels), len(labels)), dtype=float)
    for source, pauli in enumerate(matrices):
        transformed = sum((k @ pauli @ k.conj().T for k in operators), np.zeros_like(pauli))
        for target, output in enumerate(matrices):
            coefficient = np.trace(output @ transformed) / dimension
            if abs(coefficient.imag) > 1e-9:
                raise ValueError("channel does not induce a real Hermiticity-preserving PTM")
            result[target, source] = float(coefficient.real)
    result[np.abs(result) < 1e-12] = 0.0
    return result


def coherent_generator_delta_ptm(generator_label: str) -> np.ndarray:
    """Return dR/dtheta at theta=0 for exp(-i theta P)."""

    arity = len(generator_label)
    if arity not in (1, 2) or generator_label not in pauli_labels(arity) or set(generator_label) == {"I"}:
        raise ValueError("generator_label must be a nonidentity one- or two-qubit Pauli")
    labels = pauli_labels(arity)
    matrices = tuple(pauli_matrix(label[::-1]) for label in labels)
    generator = pauli_matrix(generator_label[::-1])
    dimension = 2 ** arity
    result = np.empty((len(labels), len(labels)), dtype=float)
    for source, pauli in enumerate(matrices):
        derivative = -1j * (generator @ pauli - pauli @ generator)
        for target, output in enumerate(matrices):
            coefficient = np.trace(output @ derivative) / dimension
            result[target, source] = float(coefficient.real)
    result[np.abs(result) < 1e-12] = 0.0
    return result


def unitary_error_ptm(unitary: np.ndarray) -> np.ndarray:
    """Return the PTM for an arbitrary one- or two-qubit coherent error."""

    unitary = np.asarray(unitary, dtype=complex)
    if unitary.shape not in ((2, 2), (4, 4)) or not np.allclose(
        unitary.conj().T @ unitary, np.eye(unitary.shape[0]), atol=1e-9, rtol=0
    ):
        raise ValueError("unitary must be a one- or two-qubit unitary")
    return kraus_to_ptm((unitary,))


def channel_delta(ptm: np.ndarray) -> np.ndarray:
    matrix = np.asarray(ptm, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("ptm must be square")
    return matrix - np.eye(matrix.shape[0], dtype=float)


def pauli_error_ptm(probabilities: Mapping[str, float]) -> np.ndarray:
    """Return a stochastic one- or two-qubit Pauli channel PTM."""

    if not probabilities:
        raise ValueError("at least one Pauli probability is required")
    arities = {len(label) for label in probabilities}
    if len(arities) != 1 or next(iter(arities)) not in (1, 2):
        raise ValueError("Pauli labels must share one- or two-qubit arity")
    arity = next(iter(arities))
    labels = pauli_labels(arity)
    probs: dict[str, float] = {}
    for label, value in probabilities.items():
        if label not in labels or float(value) < 0:
            raise ValueError("invalid Pauli probability")
        probs[label] = probs.get(label, 0.0) + float(value)
    explicit = sum(v for label, v in probs.items() if label != "I" * arity)
    identity_given = probs.get("I" * arity, 0.0)
    if explicit + identity_given > 1.0 + 1e-12:
        raise ValueError("Pauli probabilities exceed one")
    probs["I" * arity] = 1.0 - explicit
    return kraus_to_ptm([
        sqrt(p) * pauli_matrix(label[::-1]) for label, p in probs.items() if p > 0
    ])


def depolarizing_ptm(probability: float, arity: int = 1) -> np.ndarray:
    probability = float(probability)
    if not 0 <= probability <= 1 or arity not in (1, 2):
        raise ValueError("invalid depolarizing probability or arity")
    errors = pauli_labels(arity)[1:]
    return pauli_error_ptm({label: probability / len(errors) for label in errors})


def dephasing_ptm(probability: float) -> np.ndarray:
    return pauli_error_ptm({"Z": float(probability)})


def amplitude_damping_ptm(gamma: float) -> np.ndarray:
    gamma = float(gamma)
    if not 0 <= gamma <= 1:
        raise ValueError("gamma must lie in [0, 1]")
    return kraus_to_ptm((
        np.array([[1.0, 0.0], [0.0, sqrt(1.0 - gamma)]], dtype=complex),
        np.array([[0.0, sqrt(gamma)], [0.0, 0.0]], dtype=complex),
    ))


def generalized_amplitude_damping_ptm(gamma: float, excited_population: float) -> np.ndarray:
    gamma = float(gamma)
    excited_population = float(excited_population)
    if not 0 <= gamma <= 1 or not 0 <= excited_population <= 1:
        raise ValueError("gamma and excited_population must lie in [0, 1]")
    p_ground = 1.0 - excited_population
    s, r = sqrt(1.0 - gamma), sqrt(gamma)
    return kraus_to_ptm((
        sqrt(p_ground) * np.array([[1.0, 0.0], [0.0, s]], complex),
        sqrt(p_ground) * np.array([[0.0, r], [0.0, 0.0]], complex),
        sqrt(1.0 - p_ground) * np.array([[s, 0.0], [0.0, 1.0]], complex),
        sqrt(1.0 - p_ground) * np.array([[0.0, 0.0], [r, 0.0]], complex),
    ))


def reset_to_zero_ptm(probability: float) -> np.ndarray:
    probability = float(probability)
    if not 0 <= probability <= 1:
        raise ValueError("probability must lie in [0, 1]")
    return kraus_to_ptm((
        sqrt(1.0 - probability) * np.eye(2, dtype=complex),
        sqrt(probability) * np.array([[1.0, 0.0], [0.0, 0.0]], complex),
        sqrt(probability) * np.array([[0.0, 1.0], [0.0, 0.0]], complex),
    ))


def computational_readout_ptm(p_1_given_0: float, p_0_given_1: float) -> np.ndarray:
    """Effective final-boundary Z-readout confusion map after basis rotation."""

    p01, p10 = float(p_1_given_0), float(p_0_given_1)
    if not 0 <= p01 <= 1 or not 0 <= p10 <= 1:
        raise ValueError("readout probabilities must lie in [0, 1]")
    result = np.eye(4, dtype=float)
    result[3, 0] = p10 - p01
    result[3, 3] = 1.0 - p01 - p10
    return result


def heralded_erasure_ptm(probability: float, arity: int = 1) -> np.ndarray:
    """Trace-decreasing computational-subspace survival map (1-p) I."""

    probability = float(probability)
    if not 0 <= probability <= 1 or arity not in (1, 2):
        raise ValueError("invalid erasure probability or arity")
    return (1.0 - probability) * np.eye(4 ** arity, dtype=float)
