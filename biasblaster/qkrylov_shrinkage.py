"""Fixed-budget uncertainty-shrinkage ablation for Quantum Krylov."""

from __future__ import annotations

import numpy as np

from .impact import combine_observable_impacts
from .krylov import build_krylov_error_model, debias_krylov_matrices
from .krylov_regularization import solve_noise_aware_krylov
from .krylov_robust import BiasShrinkageResult, with_shrunk_observable_bias
from .qkrylov_experiment import (
    EffectiveNoiseParameters,
    PolicyResult,
    _calibration_covariance,
    _raw_impacts_and_finite_values,
    _uniform_allocation,
    _values_by_name,
    assemble_krylov_matrices,
    build_tfim_krylov_plan,
    sample_pm1_expectations,
)


def run_tfim_qkrylov_shrinkage_policy(
    *,
    n_qubits: int = 2,
    dimension: int = 2,
    time_step: float = 0.2,
    trotter_steps: int = 1,
    total_shots: int = 20_000,
    minimum_shots: int = 100,
    seed: int = 7,
    noise: EffectiveNoiseParameters | None = None,
    calibration_relative_sigma: float = 0.10,
    overlap_safety_factor: float = 1.0,
    shrinkage_strength: float = 1.0,
    shrinkage_variance_floor: float = 0.0,
    support_cap: int | None = None,
) -> tuple[PolicyResult, BiasShrinkageResult]:
    """Run shrinkage debiasing with the same uniform budget as the baseline."""

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
    per_shot_variances = np.maximum(
        1.0 - np.clip(finite_values, -1.0, 1.0) ** 2, 1e-12
    )
    uniform = _uniform_allocation(len(plan.estimators), total_shots, minimum_shots)
    shot_covariance = np.diag(per_shot_variances / uniform)
    combined = combine_observable_impacts(
        raw_batches,
        mode_covariance=sigma_mode,
        shot_covariance=shot_covariance,
    )
    shrunk, shrinkage = with_shrunk_observable_bias(
        combined,
        strength=shrinkage_strength,
        variance_floor=shrinkage_variance_floor,
    )
    specs = tuple(estimator.spec for estimator in plan.estimators)
    model = build_krylov_error_model(shrunk, specs, plan.dimension)

    rng = np.random.default_rng(seed)
    sampled = sample_pm1_expectations(finite_values, uniform, rng)
    measured_h, measured_s = assemble_krylov_matrices(
        _values_by_name(plan, sampled), specs, plan.dimension
    )
    corrected_h, corrected_s = debias_krylov_matrices(
        measured_h, measured_s, model
    )
    eigen, assessment = solve_noise_aware_krylov(
        corrected_h,
        corrected_s,
        model,
        safety_factor=overlap_safety_factor,
        absolute_floor=1e-12,
        include_predicted_bias=False,
    )
    weighted = int(
        sum(
            int(uniform[index]) * estimator.two_qubit_gates
            for index, estimator in enumerate(plan.estimators)
        )
    )
    policy = PolicyResult(
        name="shrinkage_modewise",
        energy=eigen.energy,
        ground_energy_error=abs(eigen.energy - plan.exact_ground_energy),
        deviation_from_ideal_qk=abs(eigen.energy - plan.ideal_qk_energy),
        retained_rank=eigen.retained_overlap_rank,
        overlap_floor=float(
            np.max(assessment.thresholds[assessment.keep], initial=1e-12)
        ),
        shots=int(np.sum(uniform)),
        weighted_two_qubit_executions=weighted,
    )
    return policy, shrinkage
