"""Fair pilot-stage adaptive policies for the Quantum Krylov benchmark."""

from __future__ import annotations

import numpy as np

from .impact import combine_observable_impacts
from .krylov import build_krylov_error_model, debias_krylov_matrices, optimal_shot_allocation
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


def _policy(
    name: str,
    result,
    plan,
    allocation: np.ndarray,
    threshold: float,
) -> PolicyResult:
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
            float(
                np.real(
                    vector.conj().T
                    @ (dh - energy * ds)
                    @ vector
                )
            )
            for dh, ds in zip(
                model.observable_weights_h,
                model.observable_weights_s,
                strict=True,
            )
        ],
        dtype=float,
    )


def run_tfim_qkrylov_adaptive_benchmark(
    *,
    n_qubits: int = 2,
    dimension: int = 3,
    time_step: float = 0.35,
    trotter_steps: int = 2,
    total_shots: int = 100_000,
    minimum_shots: int = 100,
    pilot_fraction: float = 0.20,
    seed: int = 7,
    noise: EffectiveNoiseParameters | None = None,
    calibration_relative_sigma: float = 0.10,
    overlap_safety_factor: float = 1.0,
    support_cap: int | None = None,
) -> KrylovBenchmarkResult:
    """Run a mode-resolved, resource-accounted adaptive QK ablation.

    The adaptive policy spends ``pilot_fraction`` of the *same* total shot
    budget uniformly, derives an energy sensitivity from the pilot H/S pair,
    and allocates only the remaining shots adaptively. Pilot counts are reused
    in the final estimator, so no hidden measurement budget is introduced.
    """

    pilot_fraction = float(pilot_fraction)
    if not 0.0 < pilot_fraction < 1.0:
        raise ValueError("pilot_fraction must lie strictly between zero and one")
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
    calibration_combined = combine_observable_impacts(
        raw_batches, mode_covariance=sigma_mode
    )
    calibration_model = build_krylov_error_model(
        calibration_combined,
        tuple(estimator.spec for estimator in plan.estimators),
        plan.dimension,
    )
    specs = tuple(estimator.spec for estimator in plan.estimators)

    first_order_residual = calibration_combined.predicted - finite_values
    first_order_rmse = float(np.sqrt(np.mean(first_order_residual**2)))
    first_order_max = float(np.max(np.abs(first_order_residual), initial=0.0))
    per_shot_variances = np.maximum(
        1.0 - np.clip(finite_values, -1.0, 1.0) ** 2,
        1e-12,
    )
    uniform = _uniform_allocation(
        len(plan.estimators), total_shots, minimum_shots
    )
    uniform_cov = np.diag(per_shot_variances / uniform)
    uniform_combined = combine_observable_impacts(
        raw_batches,
        mode_covariance=sigma_mode,
        shot_covariance=uniform_cov,
    )
    uniform_model = build_krylov_error_model(
        uniform_combined, specs, plan.dimension
    )

    rng = np.random.default_rng(seed)
    uniform_values = sample_pm1_expectations(finite_values, uniform, rng)
    raw_h, raw_s = assemble_krylov_matrices(
        _values_by_name(plan, uniform_values), specs, plan.dimension
    )
    corrected_h, corrected_s = debias_krylov_matrices(
        raw_h, raw_s, uniform_model
    )

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
            raw_h,
            raw_s,
            uniform_model,
            safety_factor=overlap_safety_factor,
            absolute_floor=1e-12,
            include_predicted_bias=True,
        )
        policies.append(
            _policy(
                "noise_modewise",
                raw_result,
                plan,
                uniform,
                float(np.max(raw_assessment.thresholds[raw_assessment.keep], initial=1e-12)),
            )
        )
    except ValueError:
        pass
    try:
        corrected_result, corrected_assessment = solve_noise_aware_krylov(
            corrected_h,
            corrected_s,
            uniform_model,
            safety_factor=overlap_safety_factor,
            absolute_floor=1e-12,
            include_predicted_bias=False,
        )
        policies.append(
            _policy(
                "debiased_modewise",
                corrected_result,
                plan,
                uniform,
                float(
                    np.max(
                        corrected_assessment.thresholds[corrected_assessment.keep],
                        initial=1e-12,
                    )
                ),
            )
        )
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
        raw_batches,
        mode_covariance=sigma_mode,
        shot_covariance=pilot_cov,
    )
    pilot_model = build_krylov_error_model(pilot_combined, specs, plan.dimension)
    pilot_h, pilot_s = debias_krylov_matrices(pilot_h, pilot_s, pilot_model)

    try:
        pilot_eigen, _ = solve_noise_aware_krylov(
            pilot_h,
            pilot_s,
            pilot_model,
            safety_factor=overlap_safety_factor,
            absolute_floor=1e-12,
            include_predicted_bias=False,
        )
        sensitivities = _energy_observable_sensitivities(
            pilot_eigen, calibration_model
        )
    except ValueError:
        # Fallback is deterministic and only controls allocation; no extra
        # measurement data are consumed.
        ideal_eigen, _ = solve_noise_aware_krylov(
            plan.ideal_h,
            plan.ideal_s,
            calibration_model,
            safety_factor=0.0,
            absolute_floor=1e-12,
            include_predicted_bias=False,
        )
        sensitivities = _energy_observable_sensitivities(
            ideal_eigen, calibration_model
        )

    additional = optimal_shot_allocation(
        sensitivities,
        per_shot_variances,
        remaining,
        minimum_shots=0,
    )
    adaptive = pilot + additional
    additional_values = _sample_additional(finite_values, additional, rng)
    final_values = (
        pilot * pilot_values + additional * additional_values
    ) / adaptive
    adaptive_h, adaptive_s = assemble_krylov_matrices(
        _values_by_name(plan, final_values), specs, plan.dimension
    )
    adaptive_cov = np.diag(per_shot_variances / adaptive)
    adaptive_combined = combine_observable_impacts(
        raw_batches,
        mode_covariance=sigma_mode,
        shot_covariance=adaptive_cov,
    )
    adaptive_model = build_krylov_error_model(
        adaptive_combined, specs, plan.dimension
    )
    adaptive_h, adaptive_s = debias_krylov_matrices(
        adaptive_h, adaptive_s, adaptive_model
    )
    try:
        adaptive_result, adaptive_assessment = solve_noise_aware_krylov(
            adaptive_h,
            adaptive_s,
            adaptive_model,
            safety_factor=overlap_safety_factor,
            absolute_floor=1e-12,
            include_predicted_bias=False,
        )
        policies.append(
            _policy(
                "adaptive_modewise",
                adaptive_result,
                plan,
                adaptive,
                float(
                    np.max(
                        adaptive_assessment.thresholds[adaptive_assessment.keep],
                        initial=1e-12,
                    )
                ),
            )
        )
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
