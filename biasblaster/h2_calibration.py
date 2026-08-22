"""Pulse-free System Model H2 calibration layer for Quantum Krylov studies.

This module uses only publicly reported component-level H2 calibration metrics:
1Q/2Q average gate infidelity, asymmetric SPAM error, and their reported
uncertainties. It deliberately does *not* infer or use pulse shapes, pulse
Hamiltonians, GRAPE controls, or pulse-derived Jacobians because those are not
part of the public H2 calibration interface.

The gate infidelities are mapped to an effective uniform Pauli channel for the
computational-space regression model. For dimension d, a Pauli channel with
non-identity probability p has average infidelity r = d p / (d + 1), so
p = r (d + 1) / d. This is an effective reduction, not a claim that H2 noise is
depolarizing.

Public H2 SPAM data combines preparation and measurement. We therefore model
it as one effective asymmetric final-boundary readout channel and do not add a
separate preparation fault, which would double count the published SPAM rate.
Memory error is retained as profile metadata but is not injected without an
H2-native scheduled circuit from which a depth-1 transport time can be defined.
Final-only measurement crosstalk is likewise not injected because it cannot
change the already-completed ancilla measurement used by these estimators.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .channel import (
    ChannelMode,
    NoiseChannelApplication,
    channel_delta,
    computational_readout_ptm,
    depolarizing_ptm,
    estimate_observable_impacts,
    propagate_noisy_observables,
)
from .impact import collapse_shared_modes, combine_observable_impacts
from .krylov import (
    build_krylov_error_model,
    debias_krylov_matrices,
    solve_krylov_generalized_eigenproblem,
)
from .krylov_regularization import assess_overlap_modes, solve_noise_aware_krylov
from .krylov_robust import with_shrunk_observable_bias
from .qkrylov_experiment import (
    KrylovExperimentPlan,
    PolicyResult,
    _uniform_allocation,
    _values_by_name,
    assemble_krylov_matrices,
    build_tfim_krylov_plan,
    sample_pm1_expectations,
    zero_state_pauli_moments,
)


H2_PERFORMANCE_VALIDATION_URL = (
    "https://docs.quantinuum.com/systems/user_guide/hardware_user_guide/"
    "performance_validation.html"
)


@dataclass(frozen=True)
class H2CalibrationProfile:
    """Published component-level H2 calibration values and uncertainties."""

    system_name: str
    one_qubit_infidelity: float
    one_qubit_infidelity_sigma: float
    two_qubit_infidelity: float
    two_qubit_infidelity_sigma: float
    spam_0: float
    spam_0_sigma: float
    spam_1: float
    spam_1_sigma: float
    memory_error_per_depth1: float
    memory_error_sigma: float
    measurement_crosstalk: float
    measurement_crosstalk_sigma: float
    source_url: str = H2_PERFORMANCE_VALIDATION_URL

    def __post_init__(self) -> None:
        if self.system_name not in {"H2-1", "H2-2"}:
            raise ValueError("system_name must be H2-1 or H2-2")
        fields = (
            "one_qubit_infidelity",
            "one_qubit_infidelity_sigma",
            "two_qubit_infidelity",
            "two_qubit_infidelity_sigma",
            "spam_0",
            "spam_0_sigma",
            "spam_1",
            "spam_1_sigma",
            "memory_error_per_depth1",
            "memory_error_sigma",
            "measurement_crosstalk",
            "measurement_crosstalk_sigma",
        )
        for name in fields:
            value = float(getattr(self, name))
            if value < 0.0 or not np.isfinite(value):
                raise ValueError(f"{name} must be finite and nonnegative")
            object.__setattr__(self, name, value)

    @property
    def p1(self) -> float:
        """Effective 1Q non-identity Pauli probability matching the RB infidelity."""

        return 1.5 * self.one_qubit_infidelity

    @property
    def p1_sigma(self) -> float:
        return 1.5 * self.one_qubit_infidelity_sigma

    @property
    def p2(self) -> float:
        """Effective 2Q non-identity Pauli probability matching the RB infidelity."""

        return 1.25 * self.two_qubit_infidelity

    @property
    def p2_sigma(self) -> float:
        return 1.25 * self.two_qubit_infidelity_sigma

    @property
    def modeled_parameter_means(self) -> Mapping[str, float]:
        return {
            "h2_p1": self.p1,
            "h2_p2": self.p2,
            "h2_spam0": self.spam_0,
            "h2_spam1": self.spam_1,
        }

    @property
    def modeled_parameter_sigmas(self) -> Mapping[str, float]:
        return {
            "h2_p1": self.p1_sigma,
            "h2_p2": self.p2_sigma,
            "h2_spam0": self.spam_0_sigma,
            "h2_spam1": self.spam_1_sigma,
        }


H2_1_CALIBRATION = H2CalibrationProfile(
    system_name="H2-1",
    one_qubit_infidelity=1.9e-5,
    one_qubit_infidelity_sigma=4.2e-6,
    two_qubit_infidelity=1.1e-3,
    two_qubit_infidelity_sigma=8.1e-5,
    spam_0=6.0e-4,
    spam_0_sigma=8.2e-5,
    spam_1=1.4e-3,
    spam_1_sigma=1.2e-4,
    memory_error_per_depth1=2.0e-4,
    memory_error_sigma=2.3e-5,
    measurement_crosstalk=6.6e-6,
    measurement_crosstalk_sigma=9.0e-7,
)

H2_2_CALIBRATION = H2CalibrationProfile(
    system_name="H2-2",
    one_qubit_infidelity=2.8e-5,
    one_qubit_infidelity_sigma=3.6e-6,
    two_qubit_infidelity=8.3e-4,
    two_qubit_infidelity_sigma=4.8e-5,
    spam_0=6.7e-4,
    spam_0_sigma=8.7e-5,
    spam_1=1.2e-3,
    spam_1_sigma=1.1e-4,
    memory_error_per_depth1=1.2e-4,
    memory_error_sigma=2.0e-5,
    measurement_crosstalk=2.2e-5,
    measurement_crosstalk_sigma=5.3e-7,
)


def get_h2_calibration(system_name: str) -> H2CalibrationProfile:
    name = str(system_name).upper().replace("_", "-")
    if name in {"H2-1", "H2-1E"}:
        return H2_1_CALIBRATION
    if name in {"H2-2", "H2-2E"}:
        return H2_2_CALIBRATION
    raise ValueError("system_name must identify H2-1 or H2-2")


@dataclass(frozen=True)
class H2PreparedKrylov:
    profile: H2CalibrationProfile
    plan: KrylovExperimentPlan
    raw_batches: tuple
    raw_combined: object
    finite_values: np.ndarray
    per_shot_variances: np.ndarray


@dataclass(frozen=True)
class H2TimeAssessment:
    time_step: float
    retained_rank: int
    dimension: int
    robust_margin: float
    predicted_condition_number: float
    conditioning_floor: float
    weighted_two_qubit_executions: int

    @property
    def full_rank(self) -> bool:
        return self.retained_rank == self.dimension


@dataclass(frozen=True)
class H2TimeSelection:
    selected_time_step: float
    assessments: tuple[H2TimeAssessment, ...]


def _derivative(builder, epsilon: float = 1e-7) -> np.ndarray:
    return channel_delta(builder(epsilon)) / epsilon


def _readout_derivative(which: str, epsilon: float = 1e-7) -> np.ndarray:
    if which == "spam0":
        return channel_delta(computational_readout_ptm(epsilon, 0.0)) / epsilon
    if which == "spam1":
        return channel_delta(computational_readout_ptm(0.0, epsilon)) / epsilon
    raise ValueError("which must be spam0 or spam1")


def _h2_occurrences(operations, profile: H2CalibrationProfile):
    one_derivative = _derivative(lambda p: depolarizing_ptm(p, 1))
    two_derivative = _derivative(lambda p: depolarizing_ptm(p, 2))
    spam0_derivative = _readout_derivative("spam0")
    spam1_derivative = _readout_derivative("spam1")

    modes: list[ChannelMode] = []
    channels: list[NoiseChannelApplication] = []
    for gate_index, operation in enumerate(operations):
        boundary = gate_index + 1
        if len(operation.qubits) == 1:
            modes.append(ChannelMode(
                "h2_p1", boundary, operation.qubits,
                one_derivative, mean=profile.p1,
            ))
            channels.append(NoiseChannelApplication(
                "h2_p1", boundary, operation.qubits,
                depolarizing_ptm(profile.p1, 1),
            ))
        elif len(operation.qubits) == 2:
            modes.append(ChannelMode(
                "h2_p2", boundary, operation.qubits,
                two_derivative, mean=profile.p2,
            ))
            channels.append(NoiseChannelApplication(
                "h2_p2", boundary, operation.qubits,
                depolarizing_ptm(profile.p2, 2),
            ))
        else:
            raise ValueError("H2 reduced model supports only one- and two-qubit gates")

    # The benchmark measures only the ancilla qubit (qubit zero). Public H2
    # calibration exposes combined asymmetric SPAM, so apply it once here as an
    # effective terminal confusion channel; do not add an independent p_init.
    boundary = len(operations)
    modes.extend((
        ChannelMode("h2_spam0", boundary, (0,), spam0_derivative, mean=profile.spam_0),
        ChannelMode("h2_spam1", boundary, (0,), spam1_derivative, mean=profile.spam_1),
    ))
    channels.append(NoiseChannelApplication(
        "h2_spam", boundary, (0,),
        computational_readout_ptm(profile.spam_0, profile.spam_1),
    ))
    return modes, channels


def prepare_h2_krylov(
    profile: H2CalibrationProfile,
    *,
    n_qubits: int = 2,
    dimension: int = 2,
    time_step: float = 0.2,
    trotter_steps: int = 1,
    support_cap: int | None = None,
) -> H2PreparedKrylov:
    """Build and propagate one Krylov plan using only H2 component calibration."""

    plan = build_tfim_krylov_plan(
        n_qubits=n_qubits,
        dimension=dimension,
        time_step=time_step,
        trotter_steps=trotter_steps,
    )
    batches = []
    finite_values = np.empty(len(plan.estimators), dtype=float)
    for index, estimator in enumerate(plan.estimators):
        modes, channels = _h2_occurrences(estimator.operations, profile)
        initial = zero_state_pauli_moments(estimator.n_qubits)
        raw = estimate_observable_impacts(
            estimator.operations,
            initial,
            {estimator.name: estimator.observable},
            modes,
            estimator.n_qubits,
            support_cap=support_cap,
        )
        batches.append(collapse_shared_modes(raw))
        finite = propagate_noisy_observables(
            estimator.operations,
            initial,
            {estimator.name: estimator.observable},
            channels,
            estimator.n_qubits,
            support_cap=support_cap,
        )
        finite_values[index] = finite[estimator.name]
    raw_combined = combine_observable_impacts(batches)
    per_shot = np.maximum(1.0 - np.clip(finite_values, -1.0, 1.0) ** 2, 1e-12)
    return H2PreparedKrylov(
        profile=profile,
        plan=plan,
        raw_batches=tuple(batches),
        raw_combined=raw_combined,
        finite_values=finite_values,
        per_shot_variances=per_shot,
    )


def h2_mode_covariance(prepared: H2PreparedKrylov) -> np.ndarray:
    sigmas = prepared.profile.modeled_parameter_sigmas
    values = []
    for name in prepared.raw_combined.mode_names:
        if name not in sigmas:
            raise ValueError(f"missing H2 calibration uncertainty for mode {name}")
        values.append(float(sigmas[name]) ** 2)
    return np.diag(values)


def _combined_for_budget(prepared: H2PreparedKrylov, allocation: np.ndarray):
    shot_covariance = np.diag(prepared.per_shot_variances / allocation)
    return combine_observable_impacts(
        prepared.raw_batches,
        mode_covariance=h2_mode_covariance(prepared),
        shot_covariance=shot_covariance,
    )


def assess_h2_time_step(
    prepared: H2PreparedKrylov,
    total_shots: int,
    *,
    minimum_shots: int = 100,
    overlap_safety_factor: float = 1.0,
    max_condition_number: float | None = 25.0,
) -> H2TimeAssessment:
    """Assess a basis using predicted H2 calibration/shot uncertainty only."""

    count = len(prepared.plan.estimators)
    allocation = _uniform_allocation(count, total_shots, minimum_shots)
    # Selection must not use finite-channel sampled truth. Use the first-order
    # predicted means for the shot planning covariance instead.
    predicted = np.clip(prepared.raw_combined.predicted, -1.0, 1.0)
    predicted_variances = np.maximum(1.0 - predicted * predicted, 1e-12)
    combined = combine_observable_impacts(
        prepared.raw_batches,
        mode_covariance=h2_mode_covariance(prepared),
        shot_covariance=np.diag(predicted_variances / allocation),
    )
    specs = tuple(estimator.spec for estimator in prepared.plan.estimators)
    model = build_krylov_error_model(combined, specs, prepared.plan.dimension)
    expected_s = prepared.plan.ideal_s + model.bias_s
    assessment = assess_overlap_modes(
        expected_s,
        model,
        safety_factor=overlap_safety_factor,
        absolute_floor=1e-12,
        include_predicted_bias=False,
        max_condition_number=max_condition_number,
    )
    values = np.asarray(assessment.eigenvalues, dtype=float)
    largest = max(float(np.max(values)), 1e-15)
    keep = assessment.keep
    retained = int(np.count_nonzero(keep))
    margins = (values - assessment.thresholds) / largest
    robust_margin = float(np.min(margins[keep])) if retained else float(np.max(margins))
    condition = (
        largest / max(float(np.min(values[keep])), 1e-15)
        if retained else float("inf")
    )
    weighted = int(sum(
        int(allocation[index]) * estimator.two_qubit_gates
        for index, estimator in enumerate(prepared.plan.estimators)
    ))
    return H2TimeAssessment(
        time_step=float(prepared.plan.times[1] - prepared.plan.times[0])
        if prepared.plan.dimension > 1 else 0.0,
        retained_rank=retained,
        dimension=prepared.plan.dimension,
        robust_margin=robust_margin,
        predicted_condition_number=float(condition),
        conditioning_floor=float(assessment.conditioning_floor),
        weighted_two_qubit_executions=weighted,
    )


def select_h2_time_step(
    profile: H2CalibrationProfile,
    candidate_time_steps: Sequence[float],
    total_shots: int,
    *,
    n_qubits: int = 2,
    dimension: int = 2,
    trotter_steps: int = 1,
    minimum_shots: int = 100,
    overlap_safety_factor: float = 1.0,
    max_condition_number: float | None = 25.0,
    minimum_robust_margin: float = 0.0,
    support_cap: int | None = None,
) -> H2TimeSelection:
    """Choose the smallest calibration-resolvable time spacing without an energy oracle."""

    candidates = tuple(sorted({float(value) for value in candidate_time_steps}))
    if not candidates or candidates[0] <= 0:
        raise ValueError("candidate time steps must be positive")
    assessments = []
    for time_step in candidates:
        prepared = prepare_h2_krylov(
            profile,
            n_qubits=n_qubits,
            dimension=dimension,
            time_step=time_step,
            trotter_steps=trotter_steps,
            support_cap=support_cap,
        )
        assessments.append(assess_h2_time_step(
            prepared,
            total_shots,
            minimum_shots=minimum_shots,
            overlap_safety_factor=overlap_safety_factor,
            max_condition_number=max_condition_number,
        ))
    feasible = [
        item for item in assessments
        if item.full_rank and item.robust_margin >= minimum_robust_margin
    ]
    if feasible:
        selected = min(feasible, key=lambda item: item.time_step)
    else:
        selected = max(
            assessments,
            key=lambda item: (item.retained_rank, item.robust_margin, -item.time_step),
        )
    return H2TimeSelection(selected.time_step, tuple(assessments))


def _policy_result(name: str, eigen, assessment, prepared, allocation) -> PolicyResult:
    energy = float(eigen.energy)
    weighted = int(sum(
        int(allocation[index]) * estimator.two_qubit_gates
        for index, estimator in enumerate(prepared.plan.estimators)
    ))
    return PolicyResult(
        name=name,
        energy=energy,
        ground_energy_error=abs(energy - prepared.plan.exact_ground_energy),
        deviation_from_ideal_qk=abs(energy - prepared.plan.ideal_qk_energy),
        retained_rank=eigen.retained_overlap_rank,
        overlap_floor=float(
            np.max(assessment.thresholds[assessment.keep], initial=1e-12)
        ),
        shots=int(np.sum(allocation)),
        weighted_two_qubit_executions=weighted,
    )


def evaluate_h2_krylov(
    prepared: H2PreparedKrylov,
    total_shots: int,
    seed: int,
    *,
    minimum_shots: int = 100,
    overlap_safety_factor: float = 1.0,
    max_condition_number: float | None = 25.0,
    shrinkage_strength: float = 1.0,
) -> tuple[PolicyResult, ...]:
    """Evaluate vanilla, full-debias, and shrinkage policies at one shot budget."""

    allocation = _uniform_allocation(
        len(prepared.plan.estimators), total_shots, minimum_shots
    )
    combined = _combined_for_budget(prepared, allocation)
    specs = tuple(estimator.spec for estimator in prepared.plan.estimators)
    model = build_krylov_error_model(combined, specs, prepared.plan.dimension)
    rng = np.random.default_rng(seed)
    sampled = sample_pm1_expectations(prepared.finite_values, allocation, rng)
    measured_h, measured_s = assemble_krylov_matrices(
        _values_by_name(prepared.plan, sampled), specs, prepared.plan.dimension
    )

    policies: list[PolicyResult] = []
    weighted = int(sum(
        int(allocation[index]) * estimator.two_qubit_gates
        for index, estimator in enumerate(prepared.plan.estimators)
    ))
    try:
        vanilla = solve_krylov_generalized_eigenproblem(
            measured_h, measured_s, overlap_floor=1e-10
        )
        policies.append(PolicyResult(
            name="vanilla",
            energy=vanilla.energy,
            ground_energy_error=abs(vanilla.energy - prepared.plan.exact_ground_energy),
            deviation_from_ideal_qk=abs(vanilla.energy - prepared.plan.ideal_qk_energy),
            retained_rank=vanilla.retained_overlap_rank,
            overlap_floor=1e-10,
            shots=int(np.sum(allocation)),
            weighted_two_qubit_executions=weighted,
        ))
    except ValueError:
        pass

    try:
        raw_eigen, raw_assessment = solve_noise_aware_krylov(
            measured_h,
            measured_s,
            model,
            safety_factor=overlap_safety_factor,
            absolute_floor=1e-12,
            include_predicted_bias=True,
            max_condition_number=max_condition_number,
        )
        policies.append(_policy_result(
            "noise_aware", raw_eigen, raw_assessment, prepared, allocation
        ))
    except ValueError:
        pass

    corrected_h, corrected_s = debias_krylov_matrices(measured_h, measured_s, model)
    try:
        debiased_eigen, debiased_assessment = solve_noise_aware_krylov(
            corrected_h,
            corrected_s,
            model,
            safety_factor=overlap_safety_factor,
            absolute_floor=1e-12,
            include_predicted_bias=False,
            max_condition_number=max_condition_number,
        )
        policies.append(_policy_result(
            "full_debias", debiased_eigen, debiased_assessment, prepared, allocation
        ))
    except ValueError:
        pass

    shrunk, _ = with_shrunk_observable_bias(combined, strength=shrinkage_strength)
    shrinkage_model = build_krylov_error_model(
        shrunk, specs, prepared.plan.dimension
    )
    shrink_h, shrink_s = debias_krylov_matrices(
        measured_h, measured_s, shrinkage_model
    )
    try:
        shrink_eigen, shrink_assessment = solve_noise_aware_krylov(
            shrink_h,
            shrink_s,
            shrinkage_model,
            safety_factor=overlap_safety_factor,
            absolute_floor=1e-12,
            include_predicted_bias=False,
            max_condition_number=max_condition_number,
        )
        policies.append(_policy_result(
            "shrinkage", shrink_eigen, shrink_assessment, prepared, allocation
        ))
    except ValueError:
        pass
    return tuple(policies)
