"""Executable Quantum Krylov benchmark built from Hadamard-test estimators.

The harness deliberately separates three concerns:

1. build a small time-evolution Krylov problem whose H/S matrix elements are
   estimated by real circuits;
2. propagate a calibrated local channel model through each compiled estimator;
3. compare raw, regularized, debiased, and sensitivity-weighted measurement
   policies at a fixed shot budget.

Qiskit is an optional dependency. The core BiasBlaster package remains
NumPy/SciPy-only; install ``biasblaster[qkrylov]`` to build these circuits.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping, Sequence

import numpy as np

from .channel import (
    ChannelMode,
    NoiseChannelApplication,
    ObservableImpactBatch,
    channel_delta,
    computational_readout_ptm,
    dephasing_ptm,
    depolarizing_ptm,
    estimate_observable_impacts,
    pauli_error_ptm,
    propagate_noisy_observables,
)
from .impact import combine_observable_impacts
from .krylov import (
    KrylovObservableSpec,
    build_krylov_error_model,
    debias_krylov_matrices,
    estimate_energy_impact,
    optimal_shot_allocation,
    recommended_overlap_floor,
    solve_krylov_generalized_eigenproblem,
)
from .model import CircuitOperation
from .pauli import embed_pauli_label, pauli_matrix


@dataclass(frozen=True)
class PauliHamiltonianTerm:
    coefficient: float
    label: str

    def __post_init__(self) -> None:
        coefficient = float(self.coefficient)
        if not np.isfinite(coefficient):
            raise ValueError("Hamiltonian coefficient must be finite")
        if not self.label or any(character not in "IXYZ" for character in self.label):
            raise ValueError("invalid Pauli Hamiltonian label")
        object.__setattr__(self, "coefficient", coefficient)


@dataclass(frozen=True)
class KrylovEstimatorCircuit:
    name: str
    spec: KrylovObservableSpec
    operations: tuple[CircuitOperation, ...]
    n_qubits: int
    observable: Mapping[str, float]
    two_qubit_gates: int


@dataclass(frozen=True)
class KrylovExperimentPlan:
    n_data_qubits: int
    dimension: int
    times: np.ndarray
    hamiltonian_terms: tuple[PauliHamiltonianTerm, ...]
    hamiltonian_matrix: np.ndarray
    ideal_h: np.ndarray
    ideal_s: np.ndarray
    ideal_qk_energy: float
    exact_ground_energy: float
    estimators: tuple[KrylovEstimatorCircuit, ...]


@dataclass(frozen=True)
class EffectiveNoiseParameters:
    """Small computational-space noise model used for offline ablations.

    ``p1``, ``p2``, ``p_meas`` and ``p_init`` default to the four directly
    interpretable probabilities in the published Helios-1E table. This is not
    a replica of the full Helios emulator: transport timing, leakage/seepage,
    crosstalk, asymmetric fault weights and runtime scheduling are omitted.
    Use the real Quantinuum emulator for the hardware-faithful validation.
    """

    p1: float = 2.5e-5
    p2: float = 8.0e-4
    p_meas: float = 1.0e-6
    p_init: float = 5.0e-4
    p_dephase_1q: float = 0.0
    p_dephase_2q: float = 0.0
    scale: float = 1.0

    def __post_init__(self) -> None:
        for name in (
            "p1", "p2", "p_meas", "p_init", "p_dephase_1q", "p_dephase_2q"
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")
            object.__setattr__(self, name, value)
        scale = float(self.scale)
        if scale < 0.0 or not np.isfinite(scale):
            raise ValueError("scale must be finite and nonnegative")
        object.__setattr__(self, "scale", scale)

    def probability(self, name: str) -> float:
        return min(1.0, float(getattr(self, name)) * self.scale)


@dataclass(frozen=True)
class PolicyResult:
    name: str
    energy: float
    ground_energy_error: float
    deviation_from_ideal_qk: float
    retained_rank: int
    overlap_floor: float
    shots: int
    weighted_two_qubit_executions: int


@dataclass(frozen=True)
class KrylovBenchmarkResult:
    plan: KrylovExperimentPlan
    noise: EffectiveNoiseParameters
    policies: tuple[PolicyResult, ...]
    mode_names: tuple[str, ...]
    observable_first_order_rmse: float
    observable_first_order_max_error: float
    uniform_allocation: np.ndarray
    adaptive_allocation: np.ndarray


def tfim_hamiltonian_terms(
    n_qubits: int,
    *,
    coupling: float = 1.0,
    transverse_field: float = 0.8,
    longitudinal_field: float = 0.2,
) -> tuple[PauliHamiltonianTerm, ...]:
    """Return an open-chain transverse/longitudinal-field Ising Hamiltonian."""

    if n_qubits < 1:
        raise ValueError("n_qubits must be positive")
    terms: list[PauliHamiltonianTerm] = []
    for qubit in range(n_qubits - 1):
        terms.append(
            PauliHamiltonianTerm(
                -float(coupling), embed_pauli_label("ZZ", (qubit, qubit + 1), n_qubits)
            )
        )
    for qubit in range(n_qubits):
        terms.append(
            PauliHamiltonianTerm(
                -float(transverse_field), embed_pauli_label("X", (qubit,), n_qubits)
            )
        )
        if longitudinal_field:
            terms.append(
                PauliHamiltonianTerm(
                    -float(longitudinal_field), embed_pauli_label("Z", (qubit,), n_qubits)
                )
            )
    return tuple(terms)


def hamiltonian_matrix(terms: Sequence[PauliHamiltonianTerm]) -> np.ndarray:
    terms = tuple(terms)
    if not terms:
        raise ValueError("at least one Hamiltonian term is required")
    width = len(terms[0].label)
    if any(len(term.label) != width for term in terms):
        raise ValueError("all Hamiltonian terms must share a register width")
    result = np.zeros((2**width, 2**width), dtype=complex)
    for term in terms:
        result += term.coefficient * pauli_matrix(term.label)
    return result


def zero_state_pauli_moments(n_qubits: int) -> dict[str, float]:
    """Return the sparse Pauli moment vector of |0...0>."""

    if n_qubits < 1:
        raise ValueError("n_qubits must be positive")
    moments: dict[str, float] = {}
    for mask in range(1 << n_qubits):
        chars = ["I"] * n_qubits
        for qubit in range(n_qubits):
            if mask & (1 << qubit):
                chars[n_qubits - 1 - qubit] = "Z"
        moments["".join(chars)] = 1.0
    return moments


def _require_qiskit():
    try:
        from qiskit import QuantumCircuit, transpile
        from qiskit.quantum_info import Operator, Statevector
    except ImportError as exc:  # pragma: no cover - dependency error path
        raise ImportError(
            "Quantum Krylov circuit construction requires the optional qkrylov extra: "
            "python -m pip install -e '.[qkrylov]'"
        ) from exc
    return QuantumCircuit, transpile, Operator, Statevector


def _reference_circuit(n_qubits: int):
    QuantumCircuit, _, _, _ = _require_qiskit()
    circuit = QuantumCircuit(n_qubits)
    for qubit in range(n_qubits):
        circuit.h(qubit)
    return circuit


def _support(label: str) -> list[tuple[int, str]]:
    width = len(label)
    return [
        (qubit, label[width - 1 - qubit])
        for qubit in range(width)
        if label[width - 1 - qubit] != "I"
    ]


def trotter_evolution_circuit(
    n_qubits: int,
    time: float,
    terms: Sequence[PauliHamiltonianTerm],
    *,
    steps: int = 2,
):
    """Build a first-order product-formula circuit using X, Z and ZZ terms."""

    QuantumCircuit, _, _, _ = _require_qiskit()
    if steps < 1:
        raise ValueError("steps must be positive")
    circuit = QuantumCircuit(n_qubits)
    if abs(float(time)) < 1e-15:
        return circuit
    dt = float(time) / steps
    for _ in range(steps):
        for term in terms:
            support = _support(term.label)
            angle = 2.0 * term.coefficient * dt
            if len(support) == 1 and support[0][1] == "X":
                circuit.rx(angle, support[0][0])
            elif len(support) == 1 and support[0][1] == "Z":
                circuit.rz(angle, support[0][0])
            elif len(support) == 2 and all(character == "Z" for _, character in support):
                circuit.rzz(angle, support[0][0], support[1][0])
            else:
                raise ValueError(
                    "the built-in product formula currently supports only X, Z and ZZ terms"
                )
    return circuit


def _append_pauli(circuit, label: str) -> None:
    for qubit, character in _support(label):
        if character == "X":
            circuit.x(qubit)
        elif character == "Y":
            circuit.y(qubit)
        elif character == "Z":
            circuit.z(qubit)
        else:  # pragma: no cover - _support removes identity
            raise ValueError("unsupported Pauli character")


def _matrix_element_test(
    prep,
    left,
    right,
    *,
    pauli_label: str | None,
    component: str,
):
    """Build a Hadamard test for <psi|U_left^dag P U_right|psi>."""

    QuantumCircuit, _, _, _ = _require_qiskit()
    n_data = prep.num_qubits
    if component not in ("real", "imag"):
        raise ValueError("component must be real or imag")

    target = QuantumCircuit(n_data)
    target.compose(right, inplace=True)
    if pauli_label is not None:
        _append_pauli(target, pauli_label)
    target.compose(left.inverse(), inplace=True)

    circuit = QuantumCircuit(n_data + 1)
    circuit.compose(prep, qubits=range(1, n_data + 1), inplace=True)
    circuit.h(0)
    controlled = target.to_gate(label="matrix_element").control(1)
    circuit.append(controlled, [0, *range(1, n_data + 1)])
    if component == "real":
        circuit.h(0)
    else:
        # Sdg then H rotates a Y measurement into computational-basis Z.
        circuit.sdg(0)
        circuit.h(0)
    return circuit


def _compile_to_operations(circuit) -> tuple[CircuitOperation, ...]:
    _, transpile, Operator, _ = _require_qiskit()
    compiled = transpile(
        circuit,
        basis_gates=["rz", "sx", "x", "cx"],
        optimization_level=1,
    )
    operations: list[CircuitOperation] = []
    for instruction in compiled.data:
        operation = instruction.operation
        if operation.name in {"barrier", "delay"}:
            continue
        qubits = tuple(compiled.find_bit(qubit).index for qubit in instruction.qubits)
        if len(qubits) not in (1, 2):
            raise ValueError(
                f"compiled estimator contains unsupported {len(qubits)}-qubit gate {operation.name}"
            )
        matrix = np.asarray(Operator(operation).data, dtype=complex)
        angle = 0.0
        if operation.params:
            try:
                angle = float(operation.params[0])
            except (TypeError, ValueError):
                angle = 0.0
        operations.append(CircuitOperation(operation.name, qubits, matrix, angle))
    return tuple(operations)


def _make_estimator(
    name: str,
    spec: KrylovObservableSpec,
    circuit,
) -> KrylovEstimatorCircuit:
    operations = _compile_to_operations(circuit)
    n_qubits = circuit.num_qubits
    observable = {embed_pauli_label("Z", (0,), n_qubits): 1.0}
    return KrylovEstimatorCircuit(
        name=name,
        spec=spec,
        operations=operations,
        n_qubits=n_qubits,
        observable=observable,
        two_qubit_gates=sum(len(operation.qubits) == 2 for operation in operations),
    )


def assemble_krylov_matrices(
    values: Mapping[str, float],
    specs: Sequence[KrylovObservableSpec],
    dimension: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Assemble measured scalar estimators into Hermitian H and S matrices."""

    h = np.zeros((dimension, dimension), dtype=complex)
    s = np.eye(dimension, dtype=complex)
    for spec in specs:
        if spec.name not in values:
            raise ValueError(f"missing measured value for {spec.name}")
        value = float(values[spec.name]) * spec.scale
        target = h if spec.target == "H" else s
        if spec.component == "real":
            target[spec.row, spec.col] += value
            if spec.row != spec.col:
                target[spec.col, spec.row] += value
        else:
            if spec.row == spec.col:
                raise ValueError("Hermitian diagonal cannot have imaginary estimator")
            target[spec.row, spec.col] += 1j * value
            target[spec.col, spec.row] -= 1j * value
    return h, s


def build_tfim_krylov_plan(
    n_qubits: int = 2,
    dimension: int = 3,
    *,
    time_step: float = 0.35,
    trotter_steps: int = 2,
    coupling: float = 1.0,
    transverse_field: float = 0.8,
    longitudinal_field: float = 0.2,
) -> KrylovExperimentPlan:
    """Build a time-evolution Krylov problem and its Hadamard-test estimators."""

    _, _, _, Statevector = _require_qiskit()
    if dimension < 1:
        raise ValueError("dimension must be positive")
    terms = tfim_hamiltonian_terms(
        n_qubits,
        coupling=coupling,
        transverse_field=transverse_field,
        longitudinal_field=longitudinal_field,
    )
    physical_h = hamiltonian_matrix(terms)
    prep = _reference_circuit(n_qubits)
    times = np.arange(dimension, dtype=float) * float(time_step)
    evolutions = tuple(
        trotter_evolution_circuit(
            n_qubits, time, terms, steps=trotter_steps
        )
        for time in times
    )

    states: list[np.ndarray] = []
    for evolution in evolutions:
        prepared = prep.compose(evolution)
        states.append(np.asarray(Statevector.from_instruction(prepared).data, dtype=complex))
    ideal_s = np.empty((dimension, dimension), dtype=complex)
    ideal_h = np.empty_like(ideal_s)
    for row in range(dimension):
        for col in range(dimension):
            ideal_s[row, col] = np.vdot(states[row], states[col])
            ideal_h[row, col] = np.vdot(states[row], physical_h @ states[col])

    estimators: list[KrylovEstimatorCircuit] = []
    for row in range(dimension):
        for col in range(row + 1, dimension):
            for component in ("real", "imag"):
                name = f"S_{row}_{col}_{component}"
                spec = KrylovObservableSpec(name, "S", row, col, component)
                test = _matrix_element_test(
                    prep, evolutions[row], evolutions[col],
                    pauli_label=None, component=component,
                )
                estimators.append(_make_estimator(name, spec, test))

    for row in range(dimension):
        for col in range(row, dimension):
            components = ("real",) if row == col else ("real", "imag")
            for term_index, term in enumerate(terms):
                for component in components:
                    name = f"H_{row}_{col}_p{term_index}_{component}"
                    spec = KrylovObservableSpec(
                        name, "H", row, col, component, scale=term.coefficient
                    )
                    test = _matrix_element_test(
                        prep, evolutions[row], evolutions[col],
                        pauli_label=term.label, component=component,
                    )
                    estimators.append(_make_estimator(name, spec, test))

    ideal_result = solve_krylov_generalized_eigenproblem(
        ideal_h, ideal_s, overlap_floor=1e-10
    )
    ground_energy = float(np.min(np.linalg.eigvalsh(physical_h)).real)
    return KrylovExperimentPlan(
        n_data_qubits=n_qubits,
        dimension=dimension,
        times=times,
        hamiltonian_terms=terms,
        hamiltonian_matrix=physical_h,
        ideal_h=ideal_h,
        ideal_s=ideal_s,
        ideal_qk_energy=ideal_result.energy,
        exact_ground_energy=ground_energy,
        estimators=tuple(estimators),
    )


def _linear_derivative(channel_builder, arity: int, epsilon: float = 1e-7) -> np.ndarray:
    matrix = channel_builder(epsilon)
    return channel_delta(matrix) / epsilon


def _effective_noise_occurrences(
    operations: Sequence[CircuitOperation],
    n_qubits: int,
    noise: EffectiveNoiseParameters,
) -> tuple[list[ChannelMode], list[NoiseChannelApplication]]:
    """Translate the offline effective noise model into local channel events."""

    modes: list[ChannelMode] = []
    channels: list[NoiseChannelApplication] = []

    p_init = noise.probability("p_init")
    if p_init > 0:
        init_ptm = pauli_error_ptm({"X": p_init})
        init_derivative = _linear_derivative(lambda p: pauli_error_ptm({"X": p}), 1)
        for qubit in range(n_qubits):
            modes.append(ChannelMode("p_init", 0, (qubit,), init_derivative, mean=p_init))
            channels.append(NoiseChannelApplication("p_init", 0, (qubit,), init_ptm))

    one_derivative = _linear_derivative(lambda p: depolarizing_ptm(p, 1), 1)
    two_derivative = _linear_derivative(lambda p: depolarizing_ptm(p, 2), 2)
    dephase_derivative = _linear_derivative(dephasing_ptm, 1)
    for gate_index, operation in enumerate(operations):
        boundary = gate_index + 1
        if len(operation.qubits) == 1:
            p1 = noise.probability("p1")
            if p1 > 0:
                modes.append(ChannelMode("p1", boundary, operation.qubits, one_derivative, mean=p1))
                channels.append(
                    NoiseChannelApplication(
                        "p1", boundary, operation.qubits, depolarizing_ptm(p1, 1)
                    )
                )
            p_dephase = noise.probability("p_dephase_1q")
            if p_dephase > 0:
                modes.append(
                    ChannelMode(
                        "p_dephase_1q", boundary, operation.qubits,
                        dephase_derivative, mean=p_dephase,
                    )
                )
                channels.append(
                    NoiseChannelApplication(
                        "p_dephase_1q", boundary, operation.qubits,
                        dephasing_ptm(p_dephase),
                    )
                )
        else:
            p2 = noise.probability("p2")
            if p2 > 0:
                modes.append(ChannelMode("p2", boundary, operation.qubits, two_derivative, mean=p2))
                channels.append(
                    NoiseChannelApplication(
                        "p2", boundary, operation.qubits, depolarizing_ptm(p2, 2)
                    )
                )
            p_dephase = noise.probability("p_dephase_2q")
            if p_dephase > 0:
                for qubit in operation.qubits:
                    modes.append(
                        ChannelMode(
                            "p_dephase_2q", boundary, (qubit,),
                            dephase_derivative, mean=p_dephase,
                        )
                    )
                    channels.append(
                        NoiseChannelApplication(
                            "p_dephase_2q", boundary, (qubit,),
                            dephasing_ptm(p_dephase),
                        )
                    )

    p_meas = noise.probability("p_meas")
    if p_meas > 0:
        readout_derivative = _linear_derivative(
            lambda p: computational_readout_ptm(p, p), 1
        )
        boundary = len(operations)
        modes.append(ChannelMode("p_meas", boundary, (0,), readout_derivative, mean=p_meas))
        channels.append(
            NoiseChannelApplication(
                "p_meas", boundary, (0,), computational_readout_ptm(p_meas, p_meas)
            )
        )
    return modes, channels


def _collapse_shared_modes(batch: ObservableImpactBatch) -> ObservableImpactBatch:
    """Sum repeated local occurrences of the same calibrated parameter."""

    if batch.covariance is not None or batch.mode_covariance is not None:
        raise ValueError("collapse raw sensitivity batches before applying covariance")
    ordered: list[str] = []
    means: dict[str, float] = {}
    for index, name in enumerate(batch.mode_names):
        mean = float(batch.mode_means[index])
        if name not in means:
            ordered.append(name)
            means[name] = mean
        elif not np.isclose(means[name], mean, atol=1e-12, rtol=0.0):
            raise ValueError(f"inconsistent repeated mean for {name}")
    jacobian = np.zeros((len(batch.names), len(ordered)), dtype=float)
    index_by_name = {name: index for index, name in enumerate(ordered)}
    for column, name in enumerate(batch.mode_names):
        jacobian[:, index_by_name[name]] += batch.jacobian[:, column]
    mode_means = np.asarray([means[name] for name in ordered], dtype=float)
    bias = jacobian @ mode_means
    return ObservableImpactBatch(
        names=batch.names,
        ideal=batch.ideal.copy(),
        bias=bias,
        predicted=batch.ideal + bias,
        jacobian=jacobian,
        mode_names=tuple(ordered),
        mode_means=mode_means,
        covariance=None,
        mode_covariance=None,
        forward_dropped_l2=batch.forward_dropped_l2,
        backward_dropped_l2=batch.backward_dropped_l2.copy(),
    )


def _raw_impacts_and_finite_values(
    plan: KrylovExperimentPlan,
    noise: EffectiveNoiseParameters,
    *,
    support_cap: int | None = None,
) -> tuple[list[ObservableImpactBatch], np.ndarray]:
    batches: list[ObservableImpactBatch] = []
    finite = np.empty(len(plan.estimators), dtype=float)
    for index, estimator in enumerate(plan.estimators):
        modes, channels = _effective_noise_occurrences(
            estimator.operations, estimator.n_qubits, noise
        )
        initial = zero_state_pauli_moments(estimator.n_qubits)
        raw = estimate_observable_impacts(
            estimator.operations,
            initial,
            {estimator.name: estimator.observable},
            modes,
            estimator.n_qubits,
            support_cap=support_cap,
        )
        batches.append(_collapse_shared_modes(raw))
        finite_result = propagate_noisy_observables(
            estimator.operations,
            initial,
            {estimator.name: estimator.observable},
            channels,
            estimator.n_qubits,
            support_cap=support_cap,
        )
        finite[index] = finite_result[estimator.name]
    return batches, finite


def _calibration_covariance(
    combined: ObservableImpactBatch,
    relative_sigma: float,
) -> np.ndarray:
    relative_sigma = float(relative_sigma)
    if relative_sigma < 0 or not np.isfinite(relative_sigma):
        raise ValueError("calibration_relative_sigma must be finite and nonnegative")
    scales = relative_sigma * np.maximum(np.abs(combined.mode_means), 1e-15)
    return np.diag(scales * scales)


def _uniform_allocation(count: int, total_shots: int, minimum_shots: int) -> np.ndarray:
    if count < 1:
        return np.zeros(0, dtype=int)
    if minimum_shots < 1 or total_shots < count * minimum_shots:
        raise ValueError("shot budget is smaller than minimum allocation")
    result = np.full(count, minimum_shots, dtype=int)
    remaining = total_shots - count * minimum_shots
    result += remaining // count
    result[: remaining % count] += 1
    return result


def sample_pm1_expectations(
    expectations: Sequence[float],
    shots: Sequence[int],
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample independent +/-1 observables from their finite-channel means."""

    means = np.clip(np.asarray(expectations, dtype=float), -1.0, 1.0)
    allocations = np.asarray(shots, dtype=int)
    if allocations.shape != means.shape or np.any(allocations < 1):
        raise ValueError("shots must be positive and match expectations")
    values = np.empty_like(means)
    for index, (mean, count) in enumerate(zip(means, allocations, strict=True)):
        plus = rng.binomial(int(count), 0.5 * (1.0 + float(mean)))
        values[index] = 2.0 * plus / int(count) - 1.0
    return values


def _values_by_name(plan: KrylovExperimentPlan, values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    if array.shape != (len(plan.estimators),):
        raise ValueError("estimator value vector has wrong shape")
    return {estimator.name: float(array[index]) for index, estimator in enumerate(plan.estimators)}


def _solve_policy(
    name: str,
    h: np.ndarray,
    s: np.ndarray,
    *,
    floor: float,
    plan: KrylovExperimentPlan,
    allocation: np.ndarray,
) -> PolicyResult:
    try:
        result = solve_krylov_generalized_eigenproblem(h, s, overlap_floor=floor)
        energy = result.energy
        rank = result.retained_overlap_rank
    except ValueError:
        energy = float("nan")
        rank = 0
    weighted = int(sum(
        int(allocation[index]) * estimator.two_qubit_gates
        for index, estimator in enumerate(plan.estimators)
    ))
    return PolicyResult(
        name=name,
        energy=energy,
        ground_energy_error=abs(energy - plan.exact_ground_energy) if np.isfinite(energy) else float("inf"),
        deviation_from_ideal_qk=abs(energy - plan.ideal_qk_energy) if np.isfinite(energy) else float("inf"),
        retained_rank=rank,
        overlap_floor=float(floor),
        shots=int(np.sum(allocation)),
        weighted_two_qubit_executions=weighted,
    )


def run_tfim_qkrylov_benchmark(
    *,
    n_qubits: int = 2,
    dimension: int = 3,
    time_step: float = 0.35,
    trotter_steps: int = 2,
    total_shots: int = 100_000,
    minimum_shots: int = 100,
    seed: int = 7,
    noise: EffectiveNoiseParameters | None = None,
    calibration_relative_sigma: float = 0.10,
    overlap_safety_factor: float = 1.0,
    support_cap: int | None = None,
) -> KrylovBenchmarkResult:
    """Run the offline ablation at a fixed total measurement-shot budget."""

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

    specs = tuple(estimator.spec for estimator in plan.estimators)
    ideal_values = _values_by_name(plan, raw_combined.ideal)
    assembled_ideal_h, assembled_ideal_s = assemble_krylov_matrices(
        ideal_values, specs, plan.dimension
    )
    if not np.allclose(assembled_ideal_h, plan.ideal_h, atol=2e-7, rtol=0.0):
        raise RuntimeError("Hadamard-test H estimators do not reproduce the ideal Krylov H")
    if not np.allclose(assembled_ideal_s, plan.ideal_s, atol=2e-7, rtol=0.0):
        raise RuntimeError("Hadamard-test S estimators do not reproduce the ideal Krylov S")

    first_order_residual = calibration_combined.predicted - finite_values
    first_order_rmse = float(np.sqrt(np.mean(first_order_residual**2)))
    first_order_max = float(np.max(np.abs(first_order_residual), initial=0.0))

    per_shot_variances = np.maximum(1.0 - np.clip(finite_values, -1.0, 1.0) ** 2, 1e-12)
    uniform = _uniform_allocation(len(plan.estimators), total_shots, minimum_shots)
    uniform_shot_covariance = np.diag(per_shot_variances / uniform)
    uniform_combined = combine_observable_impacts(
        raw_batches,
        mode_covariance=sigma_mode,
        shot_covariance=uniform_shot_covariance,
    )
    uniform_model = build_krylov_error_model(
        uniform_combined, specs, plan.dimension
    )
    noise_floor = recommended_overlap_floor(
        uniform_model,
        safety_factor=overlap_safety_factor,
        absolute_floor=1e-10,
    )

    rng = np.random.default_rng(seed)
    uniform_values = sample_pm1_expectations(finite_values, uniform, rng)
    raw_h, raw_s = assemble_krylov_matrices(
        _values_by_name(plan, uniform_values), specs, plan.dimension
    )
    debiased_h, debiased_s = debias_krylov_matrices(raw_h, raw_s, uniform_model)

    policies: list[PolicyResult] = []
    policies.append(
        PolicyResult(
            name="ideal_qk",
            energy=plan.ideal_qk_energy,
            ground_energy_error=abs(plan.ideal_qk_energy - plan.exact_ground_energy),
            deviation_from_ideal_qk=0.0,
            retained_rank=plan.dimension,
            overlap_floor=1e-10,
            shots=0,
            weighted_two_qubit_executions=0,
        )
    )
    policies.append(
        _solve_policy(
            "noisy_uniform", raw_h, raw_s, floor=1e-10,
            plan=plan, allocation=uniform,
        )
    )
    policies.append(
        _solve_policy(
            "noise_regularized", raw_h, raw_s, floor=noise_floor,
            plan=plan, allocation=uniform,
        )
    )
    policies.append(
        _solve_policy(
            "debiased_regularized", debiased_h, debiased_s, floor=noise_floor,
            plan=plan, allocation=uniform,
        )
    )

    # Shot allocation is computed from the first-pass, debiased matrices. If a
    # pathological sampled S prevents that local sensitivity calculation, fall
    # back to the ideal Krylov matrices only for the allocation diagnostic.
    calibration_model = build_krylov_error_model(
        calibration_combined, specs, plan.dimension
    )
    try:
        energy_impact = estimate_energy_impact(
            debiased_h, debiased_s, calibration_model,
            overlap_floor=max(noise_floor, 1e-10),
        )
    except ValueError:
        energy_impact = estimate_energy_impact(
            plan.ideal_h, plan.ideal_s, calibration_model,
            overlap_floor=1e-10,
        )
    adaptive = optimal_shot_allocation(
        energy_impact.observable_sensitivities,
        per_shot_variances,
        total_shots,
        minimum_shots=minimum_shots,
    )
    adaptive_shot_covariance = np.diag(per_shot_variances / adaptive)
    adaptive_combined = combine_observable_impacts(
        raw_batches,
        mode_covariance=sigma_mode,
        shot_covariance=adaptive_shot_covariance,
    )
    adaptive_model = build_krylov_error_model(
        adaptive_combined, specs, plan.dimension
    )
    adaptive_floor = recommended_overlap_floor(
        adaptive_model,
        safety_factor=overlap_safety_factor,
        absolute_floor=1e-10,
    )
    adaptive_values = sample_pm1_expectations(finite_values, adaptive, rng)
    adaptive_h, adaptive_s = assemble_krylov_matrices(
        _values_by_name(plan, adaptive_values), specs, plan.dimension
    )
    adaptive_h, adaptive_s = debias_krylov_matrices(
        adaptive_h, adaptive_s, adaptive_model
    )
    policies.append(
        _solve_policy(
            "adaptive_full", adaptive_h, adaptive_s, floor=adaptive_floor,
            plan=plan, allocation=adaptive,
        )
    )

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
