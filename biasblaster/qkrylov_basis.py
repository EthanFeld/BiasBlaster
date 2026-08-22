"""Hardware-aware selection of time-evolution Krylov basis spacing.

The selector deliberately does not use the exact ground-state energy.  It asks
which candidate time grid is expected to produce the most statistically and
numerically resolvable overlap subspace under the calibrated channel model and
a fixed measurement budget.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .impact import combine_observable_impacts
from .krylov import build_krylov_error_model
from .krylov_regularization import assess_overlap_modes
from .qkrylov_experiment import (
    EffectiveNoiseParameters,
    KrylovExperimentPlan,
    _calibration_covariance,
    _raw_impacts_and_finite_values,
    _uniform_allocation,
    _values_by_name,
    assemble_krylov_matrices,
    build_tfim_krylov_plan,
)


@dataclass(frozen=True)
class KrylovTimeCandidateAssessment:
    time_step: float
    retained_rank: int
    dimension: int
    robust_margin: float
    minimum_overlap_eigenvalue: float
    maximum_overlap_eigenvalue: float
    predicted_condition_number: float
    conditioning_floor: float
    first_order_residual_rmse: float
    weighted_two_qubit_executions: int

    @property
    def full_rank(self) -> bool:
        return self.retained_rank == self.dimension


@dataclass(frozen=True)
class KrylovTimeSelection:
    selected_time_step: float
    selected_plan: KrylovExperimentPlan
    assessments: tuple[KrylovTimeCandidateAssessment, ...]


def _predicted_shot_covariance(predicted: np.ndarray, allocation: np.ndarray) -> np.ndarray:
    means = np.clip(np.asarray(predicted, dtype=float), -1.0, 1.0)
    allocation = np.asarray(allocation, dtype=int)
    if allocation.shape != means.shape or np.any(allocation < 1):
        raise ValueError("allocation must be positive and match predicted observables")
    per_shot = np.maximum(1.0 - means * means, 1e-12)
    return np.diag(per_shot / allocation)


def assess_tfim_time_step(
    time_step: float,
    *,
    n_qubits: int = 2,
    dimension: int = 2,
    trotter_steps: int = 1,
    total_shots: int = 20_000,
    minimum_shots: int = 100,
    noise: EffectiveNoiseParameters | None = None,
    calibration_relative_sigma: float = 0.10,
    overlap_safety_factor: float = 1.0,
    max_condition_number: float | None = 25.0,
    support_cap: int | None = None,
) -> tuple[KrylovTimeCandidateAssessment, KrylovExperimentPlan]:
    """Assess one candidate basis using only the propagated error model.

    The score uses the expected noisy overlap matrix, calibration covariance and
    the shot covariance implied by a uniform planning allocation.  The exact
    ground-state energy is never consulted when ranking candidates.
    """

    time_step = float(time_step)
    if time_step <= 0 or not np.isfinite(time_step):
        raise ValueError("time_step must be finite and positive")
    if noise is None:
        noise = EffectiveNoiseParameters()
    plan = build_tfim_krylov_plan(
        n_qubits=n_qubits,
        dimension=dimension,
        time_step=time_step,
        trotter_steps=trotter_steps,
    )
    raw_batches, finite_values = _raw_impacts_and_finite_values(
        plan, noise, support_cap=support_cap
    )
    raw = combine_observable_impacts(raw_batches)
    allocation = _uniform_allocation(len(plan.estimators), total_shots, minimum_shots)
    mode_covariance = _calibration_covariance(raw, calibration_relative_sigma)
    shot_covariance = _predicted_shot_covariance(raw.predicted, allocation)
    combined = combine_observable_impacts(
        raw_batches,
        mode_covariance=mode_covariance,
        shot_covariance=shot_covariance,
    )
    specs = tuple(estimator.spec for estimator in plan.estimators)
    model = build_krylov_error_model(combined, specs, plan.dimension)
    ideal_h, ideal_s = assemble_krylov_matrices(
        _values_by_name(plan, raw.ideal), specs, plan.dimension
    )
    del ideal_h
    expected_s = ideal_s + model.bias_s
    assessment = assess_overlap_modes(
        expected_s,
        model,
        safety_factor=overlap_safety_factor,
        absolute_floor=1e-12,
        include_predicted_bias=False,
        max_condition_number=max_condition_number,
    )
    eigenvalues = np.asarray(assessment.eigenvalues, dtype=float)
    largest = max(float(np.max(eigenvalues, initial=0.0)), 1e-15)
    margins = (eigenvalues - assessment.thresholds) / largest
    retained = int(np.count_nonzero(assessment.keep))
    if retained:
        robust_margin = float(np.min(margins[assessment.keep]))
        smallest_kept = float(np.min(eigenvalues[assessment.keep]))
        condition = largest / max(smallest_kept, 1e-15)
    else:
        robust_margin = float(np.max(margins, initial=-np.inf))
        condition = float("inf")
    residual = combined.predicted - finite_values
    weighted = int(sum(
        int(allocation[index]) * estimator.two_qubit_gates
        for index, estimator in enumerate(plan.estimators)
    ))
    result = KrylovTimeCandidateAssessment(
        time_step=time_step,
        retained_rank=retained,
        dimension=dimension,
        robust_margin=robust_margin,
        minimum_overlap_eigenvalue=float(np.min(eigenvalues, initial=0.0)),
        maximum_overlap_eigenvalue=float(np.max(eigenvalues, initial=0.0)),
        predicted_condition_number=float(condition),
        conditioning_floor=float(assessment.conditioning_floor),
        first_order_residual_rmse=float(np.sqrt(np.mean(residual * residual))),
        weighted_two_qubit_executions=weighted,
    )
    return result, plan


def select_tfim_time_step(
    candidate_time_steps: Sequence[float],
    *,
    n_qubits: int = 2,
    dimension: int = 2,
    trotter_steps: int = 1,
    total_shots: int = 20_000,
    minimum_shots: int = 100,
    noise: EffectiveNoiseParameters | None = None,
    calibration_relative_sigma: float = 0.10,
    overlap_safety_factor: float = 1.0,
    max_condition_number: float | None = 25.0,
    support_cap: int | None = None,
) -> KrylovTimeSelection:
    """Choose the candidate with the strongest predicted resolvable subspace.

    Candidates are ordered lexicographically by retained rank, robust overlap
    margin, lower first-order model residual, and lower two-qubit execution
    proxy.  The energy reference is intentionally absent from this decision.
    """

    candidates = tuple(float(value) for value in candidate_time_steps)
    if not candidates:
        raise ValueError("at least one candidate_time_step is required")
    if len(set(candidates)) != len(candidates):
        raise ValueError("candidate_time_steps must be unique")

    pairs = [
        assess_tfim_time_step(
            value,
            n_qubits=n_qubits,
            dimension=dimension,
            trotter_steps=trotter_steps,
            total_shots=total_shots,
            minimum_shots=minimum_shots,
            noise=noise,
            calibration_relative_sigma=calibration_relative_sigma,
            overlap_safety_factor=overlap_safety_factor,
            max_condition_number=max_condition_number,
            support_cap=support_cap,
        )
        for value in candidates
    ]
    best_index = max(
        range(len(pairs)),
        key=lambda index: (
            pairs[index][0].retained_rank,
            pairs[index][0].robust_margin,
            -pairs[index][0].first_order_residual_rmse,
            -pairs[index][0].weighted_two_qubit_executions,
        ),
    )
    return KrylovTimeSelection(
        selected_time_step=pairs[best_index][0].time_step,
        selected_plan=pairs[best_index][1],
        assessments=tuple(pair[0] for pair in pairs),
    )
