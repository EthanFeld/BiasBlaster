"""Noise-aware Quantum Krylov analysis built on channel-level Pauli propagation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
from scipy.linalg import eigh

from .channel import ObservableImpactBatch


@dataclass(frozen=True)
class KrylovObservableSpec:
    """Map one measured scalar observable into a Hermitian H or S matrix entry."""

    name: str
    target: Literal["H", "S"]
    row: int
    col: int
    component: Literal["real", "imag"] = "real"
    scale: float = 1.0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("observable spec name must be nonempty")
        if self.target not in ("H", "S"):
            raise ValueError("target must be H or S")
        if self.row < 0 or self.col < 0:
            raise ValueError("matrix indices must be nonnegative")
        if self.component not in ("real", "imag"):
            raise ValueError("component must be real or imag")
        scale = float(self.scale)
        if not np.isfinite(scale):
            raise ValueError("scale must be finite")
        object.__setattr__(self, "scale", scale)


@dataclass(frozen=True)
class KrylovErrorModel:
    """First-order H/S bias and per-mode Jacobians."""

    bias_h: np.ndarray
    bias_s: np.ndarray
    jacobian_h: np.ndarray
    jacobian_s: np.ndarray
    mode_names: tuple[str, ...]
    mode_means: np.ndarray
    mode_covariance: np.ndarray | None
    observable_covariance: np.ndarray | None
    observable_weights_h: np.ndarray
    observable_weights_s: np.ndarray
    observable_names: tuple[str, ...]


@dataclass(frozen=True)
class KrylovEigenResult:
    energy: float
    vector: np.ndarray
    retained_overlap_rank: int
    overlap_floor: float
    overlap_eigenvalues: np.ndarray


@dataclass(frozen=True)
class EnergyImpact:
    energy: float
    mode_sensitivities: np.ndarray
    predicted_bias: float
    variance: float | None
    standard_deviation: float | None
    observable_sensitivities: np.ndarray


@dataclass(frozen=True)
class EnergyControlResult:
    controls: np.ndarray
    predicted_before: float
    predicted_after: float
    budget: float


def _entry_matrix(dimension: int, spec: KrylovObservableSpec) -> np.ndarray:
    if spec.row >= dimension or spec.col >= dimension:
        raise ValueError("observable spec index outside Krylov dimension")
    matrix = np.zeros((dimension, dimension), dtype=complex)
    value = spec.scale if spec.component == "real" else 1j * spec.scale
    matrix[spec.row, spec.col] += value
    if spec.row != spec.col:
        matrix[spec.col, spec.row] += np.conjugate(value)
    elif spec.component == "imag":
        raise ValueError("a Hermitian diagonal entry cannot have an imaginary component")
    return matrix


def build_krylov_error_model(
    impacts: ObservableImpactBatch,
    specs: Sequence[KrylovObservableSpec],
    dimension: int,
    *,
    mode_covariance: np.ndarray | None = None,
) -> KrylovErrorModel:
    """Project observable bias/Jacobians into Krylov H and S matrices."""

    if dimension < 1:
        raise ValueError("dimension must be positive")
    specs = tuple(specs)
    by_name = {spec.name: spec for spec in specs}
    if len(by_name) != len(specs):
        raise ValueError("observable spec names must be unique")
    missing = set(impacts.names) - set(by_name)
    extra = set(by_name) - set(impacts.names)
    if missing or extra:
        raise ValueError(f"spec names must match impact names; missing={sorted(missing)}, extra={sorted(extra)}")

    n_obs = len(impacts.names)
    n_modes = len(impacts.mode_names)
    obs_h = np.zeros((n_obs, dimension, dimension), dtype=complex)
    obs_s = np.zeros_like(obs_h)
    for obs_index, name in enumerate(impacts.names):
        spec = by_name[name]
        entry = _entry_matrix(dimension, spec)
        if spec.target == "H":
            obs_h[obs_index] = entry
        else:
            obs_s[obs_index] = entry

    bias_h = np.tensordot(impacts.bias, obs_h, axes=(0, 0))
    bias_s = np.tensordot(impacts.bias, obs_s, axes=(0, 0))
    jac_h = np.tensordot(impacts.jacobian.T, obs_h, axes=(1, 0)) if n_modes else np.zeros((0, dimension, dimension), complex)
    jac_s = np.tensordot(impacts.jacobian.T, obs_s, axes=(1, 0)) if n_modes else np.zeros((0, dimension, dimension), complex)

    sigma = impacts.mode_covariance if mode_covariance is None else mode_covariance
    if sigma is not None:
        sigma = np.asarray(sigma, dtype=float)
        if sigma.shape != (n_modes, n_modes):
            raise ValueError("mode_covariance has the wrong shape")
    return KrylovErrorModel(
        bias_h=bias_h,
        bias_s=bias_s,
        jacobian_h=jac_h,
        jacobian_s=jac_s,
        mode_names=impacts.mode_names,
        mode_means=impacts.mode_means.copy(),
        mode_covariance=None if sigma is None else sigma.copy(),
        observable_covariance=None if impacts.covariance is None else impacts.covariance.copy(),
        observable_weights_h=obs_h,
        observable_weights_s=obs_s,
        observable_names=impacts.names,
    )


def recommended_overlap_floor(
    error_model: KrylovErrorModel,
    *,
    safety_factor: float = 1.0,
    absolute_floor: float = 1e-12,
) -> float:
    """Estimate a noise-aware S-eigenvalue cutoff from bias plus uncertainty.

    The stochastic term uses expected Frobenius perturbation norm and is thus a
    heuristic scale, not a probabilistic certificate.
    """

    if safety_factor < 0 or absolute_floor < 0:
        raise ValueError("floors and safety_factor must be nonnegative")
    deterministic = float(np.linalg.norm(error_model.bias_s, ord=2))
    stochastic = 0.0
    if error_model.observable_covariance is not None:
        sigma = error_model.observable_covariance
        n_obs = len(error_model.observable_names)
        gram = np.empty((n_obs, n_obs), dtype=float)
        for i in range(n_obs):
            for j in range(n_obs):
                gram[i, j] = float(np.real(np.vdot(error_model.observable_weights_s[i], error_model.observable_weights_s[j])))
        stochastic = float(np.sqrt(max(float(np.sum(sigma * gram)), 0.0)))
    elif error_model.mode_covariance is not None and len(error_model.mode_names):
        sigma = error_model.mode_covariance
        gram = np.empty_like(sigma)
        for i in range(len(error_model.mode_names)):
            for j in range(len(error_model.mode_names)):
                gram[i, j] = float(np.real(np.vdot(error_model.jacobian_s[i], error_model.jacobian_s[j])))
        stochastic = float(np.sqrt(max(float(np.sum(sigma * gram)), 0.0)))
    return max(float(absolute_floor), float(safety_factor) * (deterministic + stochastic))


def solve_krylov_generalized_eigenproblem(
    hamiltonian: np.ndarray,
    overlap: np.ndarray,
    *,
    root: int = 0,
    overlap_floor: float = 1e-10,
) -> KrylovEigenResult:
    """Solve H c = E S c after discarding poorly conditioned S directions."""

    h = np.asarray(hamiltonian, dtype=complex)
    s = np.asarray(overlap, dtype=complex)
    if h.ndim != 2 or h.shape[0] != h.shape[1] or s.shape != h.shape:
        raise ValueError("H and S must be square matrices of the same shape")
    if not np.allclose(h, h.conj().T, atol=1e-9, rtol=0.0):
        raise ValueError("H must be Hermitian")
    if not np.allclose(s, s.conj().T, atol=1e-9, rtol=0.0):
        raise ValueError("S must be Hermitian")
    if overlap_floor < 0:
        raise ValueError("overlap_floor must be nonnegative")

    overlap_values, overlap_vectors = eigh(s)
    keep = overlap_values > overlap_floor
    rank = int(np.count_nonzero(keep))
    if rank == 0:
        raise ValueError("no overlap eigenvalue survives the requested floor")
    if root < 0 or root >= rank:
        raise ValueError("root is outside the retained Krylov rank")
    whitening = overlap_vectors[:, keep] / np.sqrt(overlap_values[keep])[None, :]
    reduced_h = whitening.conj().T @ h @ whitening
    energies, vectors = eigh(reduced_h)
    vector = whitening @ vectors[:, root]
    norm = np.real(vector.conj().T @ s @ vector)
    vector = vector / np.sqrt(norm)
    return KrylovEigenResult(
        energy=float(np.real(energies[root])), vector=vector,
        retained_overlap_rank=rank, overlap_floor=float(overlap_floor),
        overlap_eigenvalues=overlap_values,
    )


def _energy_matrix_derivative(vector: np.ndarray, energy: float, dh: np.ndarray, ds: np.ndarray) -> float:
    value = vector.conj().T @ (dh - energy * ds) @ vector
    if abs(value.imag) > 1e-8:
        raise ValueError("Hermitian perturbation produced a non-real energy derivative")
    return float(value.real)


def estimate_energy_impact(
    hamiltonian: np.ndarray,
    overlap: np.ndarray,
    error_model: KrylovErrorModel,
    *,
    root: int = 0,
    overlap_floor: float = 1e-10,
) -> EnergyImpact:
    """Propagate calibrated H/S noise into first-order generalized-eigenvalue error."""

    eig = solve_krylov_generalized_eigenproblem(
        hamiltonian, overlap, root=root, overlap_floor=overlap_floor
    )
    sensitivities = np.array([
        _energy_matrix_derivative(eig.vector, eig.energy, dh, ds)
        for dh, ds in zip(error_model.jacobian_h, error_model.jacobian_s)
    ], dtype=float)
    predicted_bias = float(sensitivities @ error_model.mode_means) if len(sensitivities) else 0.0
    observable_sensitivities = np.array([
        _energy_matrix_derivative(eig.vector, eig.energy, dh, ds)
        for dh, ds in zip(error_model.observable_weights_h, error_model.observable_weights_s)
    ], dtype=float)
    variance = standard_deviation = None
    if error_model.observable_covariance is not None:
        variance = max(float(observable_sensitivities @ error_model.observable_covariance @ observable_sensitivities), 0.0)
        standard_deviation = float(np.sqrt(variance))
    elif error_model.mode_covariance is not None:
        variance = max(float(sensitivities @ error_model.mode_covariance @ sensitivities), 0.0)
        standard_deviation = float(np.sqrt(variance))
    return EnergyImpact(
        energy=eig.energy, mode_sensitivities=sensitivities,
        predicted_bias=predicted_bias, variance=variance,
        standard_deviation=standard_deviation,
        observable_sensitivities=observable_sensitivities,
    )


def debias_krylov_matrices(
    measured_h: np.ndarray,
    measured_s: np.ndarray,
    error_model: KrylovErrorModel,
) -> tuple[np.ndarray, np.ndarray]:
    """Subtract the predicted first-order systematic H/S bias."""

    h = np.asarray(measured_h, dtype=complex)
    s = np.asarray(measured_s, dtype=complex)
    if h.shape != error_model.bias_h.shape or s.shape != error_model.bias_s.shape:
        raise ValueError("measured matrices and error model dimensions do not agree")
    return h - error_model.bias_h, s - error_model.bias_s


def optimal_shot_allocation(
    observable_sensitivities: Sequence[float],
    per_shot_variances: Sequence[float],
    total_shots: int,
    *,
    minimum_shots: int = 1,
) -> np.ndarray:
    """Allocate shots as N_i proportional to |dE/dx_i| sqrt(v_i)."""

    weights = np.abs(np.asarray(observable_sensitivities, dtype=float))
    variances = np.asarray(per_shot_variances, dtype=float)
    if weights.ndim != 1 or variances.shape != weights.shape:
        raise ValueError("sensitivities and per-shot variances must be same-length vectors")
    if np.any(variances < 0) or not np.all(np.isfinite(variances)):
        raise ValueError("per-shot variances must be finite and nonnegative")
    if minimum_shots < 0 or total_shots < minimum_shots * len(weights):
        raise ValueError("total_shots is smaller than the requested minimum allocation")
    if len(weights) == 0:
        return np.zeros(0, dtype=int)
    remaining = total_shots - minimum_shots * len(weights)
    scores = weights * np.sqrt(variances)
    if np.sum(scores) == 0:
        scores = np.ones_like(scores)
    fractional = remaining * scores / np.sum(scores)
    extra = np.floor(fractional).astype(int)
    leftovers = remaining - int(np.sum(extra))
    if leftovers:
        order = np.argsort(-(fractional - extra))
        extra[order[:leftovers]] += 1
    return extra + minimum_shots


def optimize_energy_bias_controls(
    mode_means: Sequence[float],
    mode_control_jacobian: np.ndarray,
    energy_sensitivities: Sequence[float],
    budget: float,
) -> EnergyControlResult:
    """Minimize first-order absolute energy bias under an L2 control budget."""

    means = np.asarray(mode_means, dtype=float)
    sensitivity = np.asarray(energy_sensitivities, dtype=float)
    control_jac = np.asarray(mode_control_jacobian, dtype=float)
    if means.ndim != 1 or sensitivity.shape != means.shape:
        raise ValueError("mode means and energy sensitivities must be same-length vectors")
    if control_jac.ndim != 2 or control_jac.shape[0] != len(means):
        raise ValueError("mode_control_jacobian has the wrong shape")
    budget = float(budget)
    if budget < 0 or not np.isfinite(budget):
        raise ValueError("budget must be finite and nonnegative")
    before = float(sensitivity @ means)
    gradient = control_jac.T @ sensitivity
    gradient_norm = float(np.linalg.norm(gradient))
    controls = np.zeros(control_jac.shape[1], dtype=float)
    if gradient_norm > 0 and before != 0 and budget > 0:
        exact = -before * gradient / (gradient_norm ** 2)
        controls = exact if np.linalg.norm(exact) <= budget else -np.sign(before) * budget * gradient / gradient_norm
    after = before + float(gradient @ controls)
    return EnergyControlResult(
        controls=controls, predicted_before=before,
        predicted_after=after, budget=budget,
    )
