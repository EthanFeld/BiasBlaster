"""Fair, guarded pilot-stage adaptive policies for Quantum Krylov benchmarks."""

from __future__ import annotations

import numpy as np

from .impact import combine_observable_impacts
from .krylov import build_krylov_error_model, debias_krylov_matrices
from .krylov_regularization import solve_noise_aware_krylov
from .qkrylov_experiment import (
    EffectiveNoiseParameters,
    KrylovBenchmarkResult,
    PolicyResult,
    _calibration_covariance,
    _raw_impacts_and_finite_values,
    _uniform_allocation,
    _values_by_name,
    assemble_krylov_matrices,
    build_tfim_krylov_plan,
    sample_pm1_expectations,
)


def _policy(name, result, plan, allocation: np.ndarray, threshold: float) -> PolicyResult:
    weighted = int(
        sum(
            int(allocation[index]) * estimator.two_qubit_gates
            for index, estimator in enumerate(plan.estimators)
        )
    )
    energy = result.energy
    return PolicyResult(
        name=name,
        energy=energy,
        ground_energy_error=abs(energy - plan.exact_ground_energy),
        deviation_from_ideal_qk=abs(energy - plan.ideal_qk_energy),
        retained_rank=result.retained_overlap_rank,
        overlap_floor=float(threshold),
        shots=int(np.sum(allocation)),
        weighted_two_qubit_executions=weighted,
    )


def _sample_additional(
    expectations: np.ndarray,
    shots: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    result = np.zeros_like(expectations, dtype=float)
    positive = shots > 0
    if np.any(positive):
        result[positive] = sample_pm1_expectations(
            expectations[positive], shots[positive], rng
        )
    return result


def _energy_observable_sensitivities(eigen, model) -> np.ndarray:
    vector = eigen.vector
    energy = eigen.energy
    return np.asarray(
        [
            float(np.real(vector.conj().T @ (dh - energy * ds) @ vector))
            for dh, ds in zip(
                model.observable_weights_h,
                model.observable_weights_s,
                strict=True,
            )
        ],
        dtype=float,
    )


def _overlap_stability_scores(overlap: np.ndarray, model) -> np.ndarray:
    """Score estimators by influence on fragile overlap eigenvalues."""

    values, vectors = np.linalg.eigh(np.asarray(overlap, dtype=complex))
    scores = np.zeros(len(model.observable_names), dtype=float)
    positive = values > 1e-12
    for mode_index in np.flatnonzero(positive):
        vector = vectors[:, mode_index]
        scale = max(float(values[mode_index]), 1e-6)
        gradients = np.asarray(
            [
                abs(float(np.real(vector.conj().T @ ds @ vector)))
                for ds in model.observable_weights_s
            ],
            dtype=float,
        )
        scores = np.maximum(scores, gradients / scale)
    return scores


def _normalize_scores(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    finite = np.where(np.isfinite(values), np.maximum(values, 0.0), 0.0)
    positive = finite[finite > 0]
    if positive.size == 0:
        return np.zeros_like(finite)
    reference = float(np.median(positive))
    return finite / max(reference, 1e-15)


def guarded_shot_allocation(
    energy_sensitivities: np.ndarray,
    overlap_scores: np.ndarray,
    per_shot_variances: np.ndarray,
    total_shots: int,
    *,
    uniform_fraction: float = 0.50,
    overlap_weight: float = 0.50,
    max_weight_ratio: float = 4.0,
) -> np.ndarray:
    """Allocate a budget without allowing one noisy pilot gradient to dominate."""

    energy = np.asarray(energy_sensitivities, dtype=float)
    overlap = np.asarray(overlap_scores, dtype=float)
    variances = np.asarray(per_shot_variances, dtype=float)
    if energy.ndim != 1 or overlap.shape != energy.shape or variances.shape != energy.shape:
        raise ValueError("adaptive score vectors must have the same one-dimensional shape")
    if np.any(variances < 0) or not np.all(np.isfinite(variances)):
        raise ValueError("per_shot_variances must be finite and nonnegative")
    if total_shots < 0:
        raise ValueError("total_shots must be nonnegative")
    if not 0.0 <= uniform_fraction <= 1.0:
        raise ValueError("uniform_fraction must lie in [0, 1]")
    if not 0.0 <= overlap_weight <= 1.0:
        raise ValueError("overlap_weight must lie in [0, 1]")
    if max_weight_ratio < 1.0 or not np.isfinite(max_weight_ratio):
        raise ValueError("max_weight_ratio must be finite and at least one")
    count = len(energy)
    if count == 0:
        return np.zeros(0, dtype=int)

    uniform_total = int(round(total_shots * uniform_fraction))
    uniform_total = min(max(uniform_total, 0), total_shots)
    allocation = np.full(count, uniform_total // count, dtype=int)
    allocation[: uniform_total % count] += 1
    targeted_total = total_shots - int(np.sum(allocation))
    if targeted_total == 0:
        return allocation

    energy_score = _normalize_scores(np.abs(energy))
    overlap_score = _normalize_scores(overlap)
    blended = (1.0 - overlap_weight) * energy_score + overlap_weight * overlap_score
    blended *= np.sqrt(variances)
    positive = blended[blended > 0]
    if positive.size == 0:
        blended = np.ones(count, dtype=float)
    else:
        floor = float(np.median(positive)) / max_weight_ratio
        ceiling = float(np.median(positive)) * max_weight_ratio
        blended = np.clip(blended, floor, ceiling)
    fractional = targeted_total * blended / np.sum(blended)
    extra = np.floor(fractional).astype(int)
    leftovers = targeted_total - int(np.sum(extra))
    if leftovers:
        order = np.argsort(-(fractional - extra))
        extra[order[:leftovers]] += 1
    return allocation + extra


def _minimum_retained_overlap_snr(assessment) -> float:
    """Return min lambda/threshold over retained overlap modes."""

    kept = np.flatnonzero(assessment.keep)
    if kept.size == 0:
        return 0.0
    ratios = assessment.eigenvalues[kept] / np.maximum(
        assessment.thresholds[kept], 1e-15
    )
    return float(np.min(ratios))


def run_tfim_qkrylov_adaptive_benchmark(
    *,
    n_qubits: int = 2,
    dimension: int = 3,
    time_step: float = 0.35,
    trotter_steps: int = 2,
    total_shots: int = 100_000,
    minimum_shots: int = 100,
    pilot_fraction: float = 0.20,
    adaptive_uniform_fraction: float = 0.50,
    adaptive_overlap_weight: float = 0.50,
    adaptive_max_weight_ratio: float = 4.0,
    adaptive_min_overlap_snr: float = 4.0,
    max_condition_number: float | None = 25.0,
    seed: int = 7,
    noise: EffectiveNoiseParameters | None = None,
    calibration_relative_sigma: float = 0.10,
    overlap_safety_factor: float = 1.0,
    support_cap: int | None = None,
) -> KrylovBenchmarkResult:
    """Run a fixed-budget, condition-aware adaptive QK ablation.

    The pilot consumes part of the same shot budget and its samples are reused.
    Targeted allocation is allowed only when all requested Krylov directions
    survive the pilot filter and the weakest retained overlap mode exceeds its
    predicted error threshold by ``adaptive_min_overlap_snr``. Otherwise the
    remaining budget stays uniform. This makes "do not adapt" an explicit
    hardware-aware decision rather than forcing a noisy sensitivity estimate.
    """

    pilot_fraction = float(pilot_fraction)
    adaptive_min_overlap_snr = float(adaptive_min_overlap_snr)
    if not 0.0 < pilot_fraction < 1.0:
        raise ValueError("pilot_fraction must lie strictly between zero and one")
    if adaptive_min_overlap_snr < 1.0 or not np.isfinite(adaptive_min_overlap_snr):
        raise ValueError("adaptive_min_overlap_snr must be finite and at least one")
    plan = build_tfim_krylov_plan(
        n_qubits=n_qubits,
        dimension=dimension,
        time_step=time_step,
        trotter_steps=trotter_steps,
    )
    if noise is None:
        noise = EffectiveNoiseParameters()
    raw_batches, finite_values = _raw_impacts_and_finite_values(
        plan, noise, support_cap=support_cap
    )
    raw_combined = combine_observable_impacts(raw_batches)
    sigma_mode = _calibration_covariance(raw_combined, calibration_relative_sigma)
    calibration_combined = combine_observable_impacts(raw_batches, mode_covariance=sigma_mode)
    specs = tuple(estimator.spec for estimator in plan.estimators)
    calibration_model = build_krylov_error_model(
        calibration_combined, specs, plan.dimension
    )

    first_order_residual = calibration_combined.predicted - finite_values
    first_order_rmse = float(np.sqrt(np.mean(first_order_residual**2)))
    first_order_max = float(np.max(np.abs(first_order_residual), initial=0.0))
    per_shot_variances = np.maximum(
        1.0 - np.clip(finite_values, -1.0, 1.0) ** 2, 1e-12
    )
    uniform = _uniform_allocation(len(plan.estimators), total_shots, minimum_shots)
    uniform_cov = np.diag(per_shot_variances / uniform)
    uniform_combined = combine_observable_impacts(
        raw_batches, mode_covariance=sigma_mode, shot_covariance=uniform_cov
    )
    uniform_model = build_krylov_error_model(uniform_combined, specs, plan.dimension)

    rng = np.random.default_rng(seed)
    uniform_values = sample_pm1_expectations(finite_values, uniform, rng)
    raw_h, raw_s = assemble_krylov_matrices(
        _values_by_name(plan, uniform_values), specs, plan.dimension
    )
    corrected_h, corrected_s = debias_krylov_matrices(raw_h, raw_s, uniform_model)

    policies: list[PolicyResult] = [
        PolicyResult(
            name="ideal_qk",
            energy=plan.ideal_qk_energy,
            ground_energy_error=abs(plan.ideal_qk_energy - plan.exact_ground_energy),
            deviation_from_ideal_qk=0.0,
            retained_rank=plan.dimension,
            overlap_floor=1e-12,
            shots=0,
            weighted_two_qubit_executions=0,
        )
    ]
    try:
        raw_result, raw_assessment = solve_noise_aware_krylov(
            raw_h, raw_s, uniform_model,
            safety_factor=overlap_safety_factor,
            absolute_floor=1e-12,
            include_predicted_bias=True,
            max_condition_number=max_condition_number,
        )
        policies.append(_policy(
            "noise_modewise", raw_result, plan, uniform,
            float(np.max(raw_assessment.thresholds[raw_assessment.keep], initial=1e-12)),
        ))
    except ValueError:
        pass
    try:
        corrected_result, corrected_assessment = solve_noise_aware_krylov(
            corrected_h, corrected_s, uniform_model,
            safety_factor=overlap_safety_factor,
            absolute_floor=1e-12,
            include_predicted_bias=False,
            max_condition_number=max_condition_number,
        )
        policies.append(_policy(
            "debiased_modewise", corrected_result, plan, uniform,
            float(np.max(corrected_assessment.thresholds[corrected_assessment.keep], initial=1e-12)),
        ))
    except ValueError:
        corrected_result = None

    count = len(plan.estimators)
    minimum_pilot = count * minimum_shots
    pilot_total = max(minimum_pilot, int(round(total_shots * pilot_fraction)))
    pilot_total = min(pilot_total, total_shots)
    pilot = _uniform_allocation(count, pilot_total, minimum_shots)
    remaining = total_shots - pilot_total
    pilot_values = sample_pm1_expectations(finite_values, pilot, rng)
    pilot_h, pilot_s = assemble_krylov_matrices(
        _values_by_name(plan, pilot_values), specs, plan.dimension
    )
    pilot_cov = np.diag(per_shot_variances / pilot)
    pilot_combined = combine_observable_impacts(
        raw_batches, mode_covariance=sigma_mode, shot_covariance=pilot_cov
    )
    pilot_model = build_krylov_error_model(pilot_combined, specs, plan.dimension)
    pilot_h, pilot_s = debias_krylov_matrices(pilot_h, pilot_s, pilot_model)

    pilot_is_safe = False
    try:
        pilot_eigen, pilot_assessment = solve_noise_aware_krylov(
            pilot_h, pilot_s, pilot_model,
            safety_factor=overlap_safety_factor,
            absolute_floor=1e-12,
            include_predicted_bias=False,
            max_condition_number=max_condition_number,
        )
        pilot_is_safe = (
            pilot_eigen.retained_overlap_rank == plan.dimension
            and _minimum_retained_overlap_snr(pilot_assessment) >= adaptive_min_overlap_snr
        )
    except ValueError:
        pilot_eigen, pilot_assessment = solve_noise_aware_krylov(
            plan.ideal_h, plan.ideal_s, calibration_model,
            safety_factor=0.0,
            absolute_floor=1e-12,
            include_predicted_bias=False,
            max_condition_number=max_condition_number,
        )

    energy_scores = _energy_observable_sensitivities(pilot_eigen, calibration_model)
    overlap_scores = _overlap_stability_scores(pilot_s, calibration_model)
    additional = guarded_shot_allocation(
        energy_scores,
        overlap_scores,
        per_shot_variances,
        remaining,
        uniform_fraction=(adaptive_uniform_fraction if pilot_is_safe else 1.0),
        overlap_weight=adaptive_overlap_weight,
        max_weight_ratio=adaptive_max_weight_ratio,
    )
    adaptive = pilot + additional
    additional_values = _sample_additional(finite_values, additional, rng)
    final_values = (pilot * pilot_values + additional * additional_values) / adaptive
    adaptive_h, adaptive_s = assemble_krylov_matrices(
        _values_by_name(plan, final_values), specs, plan.dimension
    )
    adaptive_cov = np.diag(per_shot_variances / adaptive)
    adaptive_combined = combine_observable_impacts(
        raw_batches, mode_covariance=sigma_mode, shot_covariance=adaptive_cov
    )
    adaptive_model = build_krylov_error_model(adaptive_combined, specs, plan.dimension)
    adaptive_h, adaptive_s = debias_krylov_matrices(adaptive_h, adaptive_s, adaptive_model)
    try:
        adaptive_result, adaptive_assessment = solve_noise_aware_krylov(
            adaptive_h, adaptive_s, adaptive_model,
            safety_factor=overlap_safety_factor,
            absolute_floor=1e-12,
            include_predicted_bias=False,
            max_condition_number=max_condition_number,
        )
        policies.append(_policy(
            "adaptive_guarded", adaptive_result, plan, adaptive,
            float(np.max(adaptive_assessment.thresholds[adaptive_assessment.keep], initial=1e-12)),
        ))
    except ValueError:
        pass

    return KrylovBenchmarkResult(
        plan=plan,
        noise=noise,
        policies=tuple(policies),
        mode_names=calibration_combined.mode_names,
        observable_first_order_rmse=first_order_rmse,
        observable_first_order_max_error=first_order_max,
        uniform_allocation=uniform,
        adaptive_allocation=adaptive,
    )
