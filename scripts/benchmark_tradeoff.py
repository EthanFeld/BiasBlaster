"""Run an ideal-simulator self-consistency benchmark for control estimators."""

from __future__ import annotations

import argparse
import atexit
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import expm, logm

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from biasblaster import (
    CircuitOperation,
    BiasOptimizationResult,
    FPGALink,
    FpgaOptimizerConfig,
    PulseCalibrationResult,
    calibrate_pulse_error_model,
    clear_transport_cache,
    estimate_hybrid_pauli_transport,
    estimate_nearest_clifford_transport,
    nearest_clifford_process_fidelity,
    optimize_bias_controls,
    optimize_bias_controls_fpga_greedy,
    optimize_local_error_controls,
    pauli_labels,
    pauli_matrix,
)

_ERROR_MODELS = (
    "iid-coherent",
    "quasistatic-z-drift",
    "extreme-z-zz-bias",
    "spatially-correlated",
    "sparse-outliers",
)
_DEFAULT_MQT_BENCHMARKS = (
    "ghz",
    "graphstate",
    "qnn",
    "qft",
    "qaoa",
    "grover",
)
_BASIS_GATES = ("rz", "sx", "x", "cx")
_MQT_SEEDED_BENCHMARKS = frozenset(("graphstate", "qaoa"))


def _rz(angle: float) -> np.ndarray:
    return np.diag([np.exp(-0.5j * angle), np.exp(0.5j * angle)])


def _h() -> np.ndarray:
    return np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2.0)


def _cx() -> np.ndarray:
    return np.array(
        # Qiskit ordering: for qargs (control, target), qarg 0 is the
        # least-significant factor. This is CX(control=qarg 0, target=qarg 1).
        [[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]],
        dtype=complex,
    )


def _synthetic_circuit(
    width: int,
    circuit_index: int,
    rng: np.random.Generator,
    layers: int,
) -> tuple[CircuitOperation, ...]:
    """Keep a dependency-free fixture path for unit tests and quick checks."""

    operations: list[CircuitOperation] = []
    for layer in range(layers):
        for qubit in range(width):
            angle = 0.17 * (circuit_index + 1) + 0.11 * (layer + 1) * (qubit + 1)
            angle += float(rng.normal(scale=0.01))
            operations.append(CircuitOperation("rz", (qubit,), _rz(angle), angle))
            operations.append(CircuitOperation("h", (qubit,), _h()))
        for qubit in range(layer % 2, width - 1, 2):
            operations.append(CircuitOperation("cx", (qubit, qubit + 1), _cx()))
    return tuple(operations)


def _local_generators(
    operations: tuple[CircuitOperation, ...],
    n_qubits: int,
    rng: np.random.Generator,
    error_model: str,
) -> np.ndarray:
    """Build deterministic seed amplitudes for synthetic correction pulses.

    These stress different algebraic error structures. They are converted to
    local coordinates and Jacobians by simulated pulse evolution; they are not
    fitted to a physical device or a claim about a hardware noise process.
    """

    scale = 7.5e-4
    gate_count = len(operations)
    if error_model == "iid-coherent":
        return scale * rng.normal(size=(gate_count, 3))

    if error_model == "quasistatic-z-drift":
        generators = 0.15 * scale * rng.normal(size=(gate_count, 3))
        per_qubit_drift = scale * rng.normal(size=n_qubits)
        for gate, operation in enumerate(operations):
            generators[gate, 2] += float(np.mean(per_qubit_drift[list(operation.qubits)]))
        return generators

    if error_model == "extreme-z-zz-bias":
        # Deliberately lopsided coherent stress case: every gate has an
        # aligned Z (or ZZ) term; transverse modes are 50x smaller.
        generators = 0.02 * scale * rng.normal(size=(gate_count, 3))
        generators[:, 2] += scale * (1.0 + 0.10 * rng.normal(size=gate_count))
        return generators

    if error_model == "spatially-correlated":
        generators = 0.20 * scale * rng.normal(size=(gate_count, 3))
        shared_mode = scale * rng.normal(size=3)
        per_qubit_modes = scale * rng.normal(size=(n_qubits, 3))
        for gate, operation in enumerate(operations):
            generators[gate] += 0.55 * shared_mode
            generators[gate] += 0.75 * np.mean(per_qubit_modes[list(operation.qubits)], axis=0)
        return generators

    if error_model == "sparse-outliers":
        generators = 0.20 * scale * rng.normal(size=(gate_count, 3))
        outlier_count = max(1, int(np.ceil(0.12 * gate_count)))
        outlier_gates = rng.choice(gate_count, size=outlier_count, replace=False)
        generators[outlier_gates] += 3.0 * scale * rng.normal(size=(outlier_count, 3))
        return generators

    raise ValueError(f"unsupported error_model: {error_model}")


def _local_basis_matrices(arity: int) -> tuple[np.ndarray, ...]:
    labels = ("X", "Y", "Z") if arity == 1 else ("XX", "YY", "ZZ")
    # Local labels are in Qiskit qarg order, while matrices are big-endian.
    return tuple(pauli_matrix(label[::-1]) for label in labels)


def _special_unitary_generator(unitary: np.ndarray) -> np.ndarray:
    """Return H with exp(-i H) equal to unitary up to global phase."""

    value = np.asarray(unitary, dtype=complex)
    dimension = value.shape[0]
    phase = np.angle(np.linalg.det(value)) / dimension
    special = np.exp(-1j * phase) * value
    generator = 1j * logm(special)
    generator = 0.5 * (generator + generator.conj().T)
    generator -= np.trace(generator) * np.eye(dimension) / dimension
    if not np.allclose(expm(-1j * generator), special, atol=1e-8, rtol=0.0):
        raise ValueError("target unitary is outside the principal pulse chart")
    return generator


class _PulseGateModel:
    """Platform-agnostic one-slice closed-system bilinear pulse model."""

    def __init__(
        self,
        operation: CircuitOperation,
        drift: np.ndarray,
        controls: tuple[np.ndarray, ...],
        base_pulse: np.ndarray,
        coordinate_scales: np.ndarray,
        base_calibration: PulseCalibrationResult | None = None,
    ) -> None:
        self.operation = operation
        self.drift = drift
        self.controls = controls
        self.base_pulse = base_pulse
        self.coordinate_scales = coordinate_scales
        self.base_calibration = base_calibration

    @property
    def gate_time(self) -> float:
        return 1.0

    def calibration(self, normalized_controls: np.ndarray | None = None) -> PulseCalibrationResult:
        values = self._pulse_values(normalized_controls)
        coordinate_labels = (
            ("X", "Y", "Z")
            if len(self.operation.qubits) == 1
            else pauli_labels(2)[1:]
        )
        # Pulse matrices use big-endian tensor order; transport labels use
        # Qiskit qarg order. Keep returned coordinate indices in qarg order.
        matrix_labels = tuple(label[::-1] for label in coordinate_labels)
        return calibrate_pulse_error_model(
            values,
            self.drift,
            self.controls,
            self.gate_time,
            self.operation.unitary,
            matrix_labels,
        )

    def _pulse_values(self, normalized_controls: np.ndarray | None = None) -> np.ndarray:
        """Return segment amplitudes after applying normalized pulse controls."""

        values = self.base_pulse.copy()
        if normalized_controls is not None:
            update = np.asarray(normalized_controls, dtype=float).reshape(-1)
            if update.shape != self.coordinate_scales.shape:
                raise ValueError("pulse controls have wrong shape")
            values[0, : update.size] += update * self.coordinate_scales
        return values

    def unitary(self, normalized_controls: np.ndarray | None = None) -> np.ndarray:
        """Evolve one closed-system piecewise-constant pulse segment."""

        values = self._pulse_values(normalized_controls)
        hamiltonian = self.drift + sum(
            amplitude * control
            for amplitude, control in zip(values[0], self.controls, strict=True)
        )
        return expm(-1j * hamiltonian)


def _pulse_gate_model(operation: CircuitOperation, error: np.ndarray) -> _PulseGateModel:
    """Build a bilinear closed-system pulse around one target operation.

    This uses a one-slice bilinear Hamiltonian,
    ``H = H_target + sum_m u_m H_m``.  ``H_target`` realizes the ideal local
    operation up to global phase at zero control amplitude; the seeded pulse
    amplitudes perturb it through fixed control Hamiltonians.  The resulting
    error coordinates and Jacobian are derived by ``calibrate_pulse_error_model``.
    """

    arity = len(operation.qubits)
    basis = _local_basis_matrices(arity)
    controls = basis
    base_pulse = np.asarray(error, dtype=float).reshape(1, len(basis))
    model = _PulseGateModel(
        operation=operation,
        drift=_special_unitary_generator(operation.unitary),
        controls=controls,
        base_pulse=base_pulse,
        coordinate_scales=np.ones(len(basis), dtype=float),
    )
    raw = model.calibration()
    response = np.linalg.norm(raw.jacobians, axis=0)
    scales = np.divide(1.0, response, out=np.zeros_like(response), where=response > 1.0e-12)
    return _PulseGateModel(
        operation=operation,
        drift=model.drift,
        controls=model.controls,
        base_pulse=model.base_pulse,
        coordinate_scales=scales,
        base_calibration=raw,
    )


def _pulse_calibration(models: tuple[_PulseGateModel, ...]) -> tuple[np.ndarray, np.ndarray]:
    calibrations = tuple(
        model.base_calibration if model.base_calibration is not None else model.calibration()
        for model in models
    )
    mode_count = 15
    control_count = models[0].coordinate_scales.size
    generators = np.zeros((len(models), mode_count), dtype=float)
    jacobians = np.zeros((len(models), mode_count, control_count), dtype=float)
    for gate, (calibration, model) in enumerate(zip(calibrations, models, strict=True)):
        local_mode_count = calibration.generators.size
        generators[gate, :local_mode_count] = calibration.generators
        jacobians[gate, :local_mode_count] = calibration.jacobians * model.coordinate_scales
    return generators, jacobians


def _post_update_generators(
    models: tuple[_PulseGateModel, ...], controls: np.ndarray,
) -> np.ndarray:
    """Recalibrate actual updated pulses for nonlinear transport certificates."""

    values = np.asarray(controls, dtype=float)
    if values.shape != (len(models), models[0].coordinate_scales.size):
        raise ValueError("pulse controls have wrong shape")
    generators = np.zeros((len(models), 15), dtype=float)
    for gate, model in enumerate(models):
        calibration = model.calibration(values[gate])
        generators[gate, :calibration.generators.size] = calibration.generators
    return generators


def _pulse_operations(
    models: tuple[_PulseGateModel, ...], controls: np.ndarray | None = None,
) -> tuple[CircuitOperation, ...]:
    if controls is not None and np.asarray(controls).shape != (len(models), models[0].coordinate_scales.size):
        raise ValueError("pulse controls have wrong shape")
    values = np.zeros((len(models), models[0].coordinate_scales.size)) if controls is None else np.asarray(controls, dtype=float)
    return tuple(
        CircuitOperation(model.operation.name, model.operation.qubits, model.unitary(values[index]), model.operation.angle)
        for index, model in enumerate(models)
    )


def _bind_parameters(circuit: object, seed: int) -> object:
    parameters = tuple(sorted(circuit.parameters, key=lambda parameter: parameter.name))
    if not parameters:
        return circuit
    rng = np.random.default_rng(seed)
    values = {parameter: float(rng.uniform(-np.pi, np.pi)) for parameter in parameters}
    return circuit.assign_parameters(values, inplace=False)


def _mqt_operations(benchmark: str, width: int, seed: int) -> tuple[CircuitOperation, ...]:
    """Generate and normalize an MQT Bench circuit to one- and two-qubit gates."""

    try:
        from mqt.bench import BenchmarkLevel, get_benchmark
        from qiskit import transpile
        from qiskit.quantum_info import Statevector
    except ImportError as error:  # pragma: no cover - exercised in benchmark-only environments
        raise RuntimeError("MQT Bench requires `python -m pip install -e .[benchmark]`.") from error

    factory_options = {"seed": seed} if benchmark in _MQT_SEEDED_BENCHMARKS else {}
    circuit = get_benchmark(
        benchmark,
        BenchmarkLevel.INDEP,
        width,
        random_parameters=False,
        **factory_options,
    )
    if circuit.num_qubits != width:
        raise ValueError(f"MQT Bench returned {circuit.num_qubits} qubits for requested width {width}")
    circuit = _bind_parameters(circuit, seed)
    circuit = circuit.remove_final_measurements(inplace=False)
    probes = _haar_probes(width, 4, int(np.random.SeedSequence((seed, width, 741)).generate_state(1)[0]))
    source_states = _qiskit_evolve_states(circuit, probes, Statevector)
    circuit = transpile(circuit, basis_gates=list(_BASIS_GATES), optimization_level=1, seed_transpiler=seed)
    transpiled_states = _qiskit_evolve_states(circuit, probes, Statevector)
    if not _equivalent_up_to_global_phase(source_states, transpiled_states):
        raise ValueError(f"transpilation changed the tested action for {benchmark}@{width}")

    operations: list[CircuitOperation] = []
    for instruction in circuit.data:
        operation = instruction.operation
        if operation.name in {"barrier", "delay"}:
            continue
        qubits = tuple(circuit.find_bit(qubit).index for qubit in instruction.qubits)
        if len(qubits) not in (1, 2):
            raise ValueError(f"unsupported {len(qubits)}-qubit operation {operation.name}")
        try:
            unitary = np.asarray(operation.to_matrix(), dtype=complex)
        except (AttributeError, TypeError) as error:
            raise ValueError(f"non-unitary operation {operation.name}") from error
        angle = float(operation.params[0]) if len(operation.params) == 1 else 0.0
        operations.append(CircuitOperation(operation.name, qubits, unitary, angle))
    if not operations:
        raise ValueError("MQT Bench circuit did not contain any unitary one- or two-qubit operations")
    values = tuple(operations)
    extracted_states = _evolve_states(values, width, probes)
    if not _equivalent_up_to_global_phase(transpiled_states, extracted_states):
        raise ValueError(f"extracted operations changed the tested action for {benchmark}@{width}")
    return values


def _gate_count_filter(
    candidates: list[tuple[str, tuple[CircuitOperation, ...]]],
) -> tuple[list[tuple[str, tuple[CircuitOperation, ...]]], tuple[str, ...]]:
    """Exclude per-size gate-count outliers with Tukey's 1.5-IQR fence."""

    if len(candidates) < 4:
        return candidates, ()
    counts = np.array([len(operations) for _, operations in candidates], dtype=float)
    lower, upper = np.quantile(counts, (0.25, 0.75))
    fence = upper + 1.5 * (upper - lower)
    retained = [(name, operations) for name, operations in candidates if len(operations) <= fence]
    excluded = tuple(name for name, operations in candidates if len(operations) > fence)
    return retained, excluded


def _haar_probes(n_qubits: int, count: int, seed: int) -> np.ndarray:
    """Return deterministic normalized Haar-random state vectors as columns."""

    rng = np.random.default_rng(seed)
    dimension = 2**n_qubits
    values = rng.normal(size=(dimension, count)) + 1j * rng.normal(size=(dimension, count))
    return values / np.linalg.norm(values, axis=0, keepdims=True)


def _apply_local_unitary(
    states: np.ndarray,
    unitary: np.ndarray,
    qubits: tuple[int, ...],
    n_qubits: int,
) -> np.ndarray:
    """Apply a Qiskit-ordered local matrix to a batch of state vectors."""

    state_count = states.shape[1]
    local_axes = tuple(n_qubits - 1 - qubit for qubit in reversed(qubits))
    remaining_axes = tuple(axis for axis in range(n_qubits) if axis not in local_axes)
    permutation = remaining_axes + local_axes + (n_qubits,)
    inverse_permutation = tuple(np.argsort(permutation))
    tensor = states.reshape((2,) * n_qubits + (state_count,))
    moved = tensor.transpose(permutation).reshape(-1, 2 ** len(qubits), state_count)
    transformed = np.einsum("ab,xbs->xas", unitary, moved, optimize=True)
    restored = transformed.reshape((2,) * len(remaining_axes) + (2,) * len(qubits) + (state_count,))
    return restored.transpose(inverse_permutation).reshape(states.shape)


def _equivalent_up_to_global_phase(left: np.ndarray, right: np.ndarray, *, atol: float = 1e-9) -> bool:
    """Return whether two equally shaped arrays differ only by global phase."""

    if left.shape != right.shape:
        return False
    overlap = np.vdot(right, left)
    if abs(overlap) <= atol:
        return np.allclose(left, right, atol=atol, rtol=0.0)
    return bool(np.allclose(left, overlap / abs(overlap) * right, atol=atol, rtol=0.0))


def _qiskit_evolve_states(circuit: object, probes: np.ndarray, statevector_type: type) -> np.ndarray:
    """Evolve a batch of columns through a Qiskit circuit without dense operators."""

    return np.column_stack(
        [np.asarray(statevector_type(probes[:, index]).evolve(circuit).data, dtype=complex) for index in range(probes.shape[1])]
    )


def _local_error_unitary(coefficients: np.ndarray, arity: int) -> np.ndarray:
    labels = ("X", "Y", "Z") if arity == 1 else ("XX", "YY", "ZZ")
    hamiltonian = sum(float(value) * pauli_matrix(label) for value, label in zip(coefficients, labels, strict=True))
    return expm(-1j * hamiltonian)


def _evolve_states(
    operations: tuple[CircuitOperation, ...],
    n_qubits: int,
    probes: np.ndarray,
    generators: np.ndarray | None = None,
) -> np.ndarray:
    """Evolve ideal or coherently perturbed circuits on the probe ensemble.

    The transport estimator propagates a local generator through its own ideal
    operation and every later operation. The matching finite-error convention
    therefore applies the local error *before* its ideal operation.
    """

    states = probes.copy()
    for gate, operation in enumerate(operations):
        if generators is not None:
            error = _local_error_unitary(generators[gate], len(operation.qubits))
            states = _apply_local_unitary(states, error, operation.qubits, n_qubits)
        states = _apply_local_unitary(states, operation.unitary, operation.qubits, n_qubits)
    return states


def _mean_output_state_fidelity(ideal: np.ndarray, actual: np.ndarray) -> float:
    """Return mean sampled-Haar output-state fidelity, not gate fidelity."""

    overlaps = np.sum(ideal.conj() * actual, axis=0)
    fidelities = np.clip(np.abs(overlaps) ** 2, 0.0, 1.0)
    return float(np.mean(fidelities))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", choices=("mqt-bench", "synthetic"), default="mqt-bench")
    parser.add_argument("--mqt-benchmarks", nargs="+", default=_DEFAULT_MQT_BENCHMARKS)
    parser.add_argument(
        "--estimator",
        choices=("no-bias", "local-error-oracle", "nearest-clifford", "anchored-pauli", "all"),
        default="all",
    )
    parser.add_argument("--support-cap", type=int, default=64)
    parser.add_argument("--support-caps", nargs="+", type=int, help="Exact Pauli-correction caps to compare.")
    parser.add_argument("--error-models", nargs="+", choices=_ERROR_MODELS, default=_ERROR_MODELS)
    parser.add_argument("--widths", nargs="+", type=int, default=(6, 8, 10))
    parser.add_argument("--layers", type=int, default=3, help="Synthetic-corpus layer count only.")
    parser.add_argument("--circuits", type=int, default=3, help="Synthetic-corpus fixtures per width and seed only.")
    parser.add_argument("--effort-fraction", type=float, default=0.25)
    parser.add_argument(
        "--optimization-backend",
        choices=("cpu", "fpga-greedy"),
        default="cpu",
        help="Backend for transported-bias optimization.",
    )
    parser.add_argument("--fpga-port", help="Connected FPGA serial port for fpga-greedy.")
    parser.add_argument(
        "--fpga-baud", type=int, default=1_000_000,
        help="FPGA UART baud; use 115200 only with the legacy firmware image.",
    )
    parser.add_argument(
        "--fpga-fallback",
        choices=("cpu", "error"),
        default="cpu",
        help="Fallback when a greedy FPGA gate record cannot be represented.",
    )
    parser.add_argument("--fpga-gate-kind", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument(
        "--fpga-max-sweeps", type=int, default=2,
        help="Maximum greedy FPGA refinement sweeps per transported case.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--seeds", nargs="+", type=int, help="Independent deterministic seeds; overrides --seed.")
    parser.add_argument("--fidelity-probes", type=int, default=32, help="Haar-random output-state probes per circuit.")
    parser.add_argument("--output", type=Path, help="Write per-case benchmark records as JSON.")
    return parser


def _control_aware_selection_matrix(
    generators: np.ndarray,
    jacobians: np.ndarray,
    effort_fraction: float,
) -> np.ndarray:
    """Score transport rows by current bias and budget-sized control response."""

    budget = effort_fraction * float(np.linalg.norm(generators))
    columns = [generators.reshape(-1)]
    columns.extend(
        budget * jacobians[:, :, control].reshape(-1)
        for control in range(jacobians.shape[2])
    )
    return np.column_stack(columns)


def _fixtures(args: argparse.Namespace, seeds: tuple[int, ...], widths: tuple[int, ...]) -> tuple[
    list[tuple[str, int, int, tuple[CircuitOperation, ...]]], tuple[str, ...]
]:
    fixtures: list[tuple[str, int, int, tuple[CircuitOperation, ...]]] = []
    gate_outliers: set[str] = set()
    for seed in seeds:
        for width in widths:
            if args.corpus == "synthetic":
                for circuit_index in range(args.circuits):
                    rng = np.random.default_rng(np.random.SeedSequence((seed, width, circuit_index)))
                    operations = _synthetic_circuit(width, circuit_index, rng, args.layers)
                    fixtures.append((f"synthetic-{circuit_index}", seed, width, operations))
                continue

            candidates: list[tuple[str, tuple[CircuitOperation, ...]]] = []
            for benchmark_index, benchmark in enumerate(args.mqt_benchmarks):
                try:
                    operations = _mqt_operations(benchmark, width, seed + benchmark_index)
                except (RuntimeError, ValueError) as error:
                    print(f"skipped_benchmark={benchmark}@{width}: {error}", file=sys.stderr)
                    continue
                candidates.append((benchmark, operations))
            retained, excluded = _gate_count_filter(candidates)
            gate_outliers.update(f"{benchmark}@{width}" for benchmark in excluded)
            fixtures.extend((benchmark, seed, width, operations) for benchmark, operations in retained)
    if not fixtures:
        raise SystemExit("no usable benchmark circuits were generated")
    return fixtures, tuple(sorted(gate_outliers))


def main() -> None:
    args = _parser().parse_args()
    support_caps = (
        tuple(args.support_caps)
        if args.support_caps
        else ((64, 2048) if args.estimator == "all" else (args.support_cap,))
    )
    seeds = tuple(args.seeds) if args.seeds else (args.seed,)
    widths = tuple(args.widths)
    error_models = tuple(args.error_models)
    if (
        any(width < 1 for width in widths)
        or args.layers < 1
        or args.circuits < 1
        or args.fidelity_probes < 1
        or any(cap < 0 for cap in support_caps)
        or (args.estimator != "anchored-pauli" and any(cap == 0 for cap in support_caps))
        or not 0 <= args.effort_fraction <= 1
    ):
        raise SystemExit("widths, layers, circuits, support caps, and fidelity probes must be positive; effort-fraction must lie in [0, 1]")
    if args.optimization_backend == "fpga-greedy" and not args.fpga_port:
        raise SystemExit("--fpga-port is required with --optimization-backend fpga-greedy")
    if args.fpga_max_sweeps < 1:
        raise SystemExit("--fpga-max-sweeps must be positive")
    if args.fpga_baud < 1:
        raise SystemExit("--fpga-baud must be positive")
    if any(len(set(values)) != len(values) for values in (support_caps, seeds, widths, error_models)):
        raise SystemExit("support caps, seeds, widths, and error models must not contain duplicates")

    configurations: list[tuple[str, int | None]] = []
    if args.estimator in ("no-bias", "all"):
        configurations.append(("no-bias", None))
    if args.estimator in ("local-error-oracle", "all"):
        configurations.append(("local-error-oracle", None))
    if args.estimator in ("nearest-clifford", "all"):
        configurations.append(("nearest-clifford", None))
    if args.estimator in ("anchored-pauli", "all"):
        anchored_caps = (0,) + support_caps if args.estimator == "all" else support_caps
        configurations.extend(("anchored-pauli", cap) for cap in anchored_caps)

    fixtures, excluded_outliers = _fixtures(args, seeds, widths)
    fpga_link: FPGALink | None = None
    fpga_link_setup_ms: float | None = None
    fpga_config = FpgaOptimizerConfig(max_sweeps=args.fpga_max_sweeps)
    if args.optimization_backend == "fpga-greedy":
        started = time.perf_counter()
        fpga_link = FPGALink(args.fpga_port, baud=args.fpga_baud)
        try:
            if fpga_link.ping() != 0xA5:
                raise RuntimeError("FPGA ping mismatch; expected 0xA5")
        except Exception:
            fpga_link.close()
            raise
        fpga_link_setup_ms = (time.perf_counter() - started) * 1e3
        # Keep one UART session for all cases. The greedy algorithm is reply-
        # dependent, so records cannot be wire-pipelined without new FPGA RTL.
        atexit.register(fpga_link.close)
    rows_by_configuration: dict[tuple[str, str, int | None], list[dict[str, float | None]]] = {
        (error_model, estimator, support_cap): []
        for error_model in error_models
        for estimator, support_cap in configurations
    }
    case_rows: list[dict[str, str | int | float | None]] = []
    for fixture_index, (benchmark, seed, width, operations) in enumerate(fixtures):
        probe_seed = int(np.random.SeedSequence((seed, width, fixture_index, 991)).generate_state(1)[0])
        probes = _haar_probes(width, args.fidelity_probes, probe_seed)
        ideal_states = _evolve_states(operations, width, probes)
        transports: dict[tuple[str, int | None], tuple[object, float, float, float]] = {}
        for estimator, support_cap in configurations:
            if estimator in ("no-bias", "local-error-oracle", "anchored-pauli"):
                continue
            started = time.perf_counter()
            if estimator == "nearest-clifford":
                transport = estimate_nearest_clifford_transport(
                    operations,
                    width,
                    two_qubit_generator_basis="full-su4",
                )
            else:
                raise AssertionError(f"unexpected prebuilt estimator: {estimator}")
            transport_build_ms = (time.perf_counter() - started) * 1e3
            if estimator == "nearest-clifford":
                local_fidelities = np.array(
                    [nearest_clifford_process_fidelity(operation.unitary) for operation in operations]
                )
                nearest_mean_fidelity = float(np.mean(local_fidelities))
                nearest_min_fidelity = float(np.min(local_fidelities))
            else:
                nearest_mean_fidelity = float("nan")
                nearest_min_fidelity = float("nan")
            transports[(estimator, support_cap)] = (
                transport,
                transport_build_ms,
                nearest_mean_fidelity,
                nearest_min_fidelity,
            )

        for model_index, error_model in enumerate(error_models):
            error_seed = int(np.random.SeedSequence((seed, width, fixture_index, model_index)).generate_state(1)[0])
            seeded_errors = _local_generators(operations, width, np.random.default_rng(error_seed), error_model)
            started = time.perf_counter()
            pulse_models = tuple(
                _pulse_gate_model(operation, seeded_errors[gate])
                for gate, operation in enumerate(operations)
            )
            generators, jacobians = _pulse_calibration(pulse_models)
            projection_residual = float(np.mean([
                model.base_calibration.omitted_generator_norm
                for model in pulse_models
                if model.base_calibration is not None
            ]))
            analytic_coordinate_model_ms = (time.perf_counter() - started) * 1e3
            started = time.perf_counter()
            before_fidelity = _mean_output_state_fidelity(
                ideal_states,
                _evolve_states(_pulse_operations(pulse_models), width, probes),
            )
            baseline_rescore_ms = (time.perf_counter() - started) * 1e3
            for estimator, support_cap in configurations:
                if estimator == "no-bias":
                    controls = np.zeros((generators.shape[0], jacobians.shape[2]), dtype=float)
                    result = BiasOptimizationResult(
                        controls=controls,
                        updated_generators=generators.copy(),
                        predicted_before_norm=0.0,
                        predicted_after_norm=0.0,
                        control_norm=0.0,
                        budget=0.0,
                    )
                    optimization_ms = 0.0
                    support = 0.0
                    bound = None
                    transport_cold_build_ms = float("nan")
                    retained_payload_bytes = 0.0
                    nearest_mean_fidelity = float("nan")
                    nearest_min_fidelity = float("nan")
                elif estimator == "local-error-oracle":
                    started = time.perf_counter()
                    result = optimize_local_error_controls(generators, jacobians, args.effort_fraction)
                    optimization_ms = (time.perf_counter() - started) * 1e3
                    support = float("nan")
                    bound = None
                    transport_cold_build_ms = float("nan")
                    retained_payload_bytes = 0.0
                    nearest_mean_fidelity = float("nan")
                    nearest_min_fidelity = float("nan")
                else:
                    if estimator == "anchored-pauli":
                        # Every reported build time starts cold; otherwise the
                        # module cache makes later error-model rows look free.
                        clear_transport_cache()
                        started = time.perf_counter()
                        transport = estimate_hybrid_pauli_transport(
                            operations,
                            width,
                            support_cap=support_cap,
                            two_qubit_generator_basis="full-su4",
                            selection_matrix=_control_aware_selection_matrix(
                                generators, jacobians, args.effort_fraction,
                            ),
                        )
                        transport_cold_build_ms = (time.perf_counter() - started) * 1e3
                        nearest_mean_fidelity = float("nan")
                        nearest_min_fidelity = float("nan")
                    else:
                        transport, transport_cold_build_ms, nearest_mean_fidelity, nearest_min_fidelity = transports[(estimator, support_cap)]
                    started = time.perf_counter()
                    if args.optimization_backend == "fpga-greedy":
                        result = optimize_bias_controls_fpga_greedy(
                            generators,
                            transport,
                            jacobians,
                            args.effort_fraction,
                            link=fpga_link,
                            config=fpga_config,
                            gate_kind=args.fpga_gate_kind,
                            fallback=args.fpga_fallback,
                        )
                    else:
                        result = optimize_bias_controls(
                            generators, transport, jacobians, args.effort_fraction
                        )
                    optimization_ms = (time.perf_counter() - started) * 1e3
                    support = float(transport.support_count)
                    bound = (
                        transport.truncation_bound_for(
                            _post_update_generators(pulse_models, result.controls),
                        )
                        if estimator == "anchored-pauli" and transport.truncation_bound is not None
                        else None
                    )
                    retained_payload_bytes = float(transport.retained_payload_bytes)
                started = time.perf_counter()
                after_fidelity = _mean_output_state_fidelity(
                    ideal_states,
                    _evolve_states(_pulse_operations(pulse_models, result.controls), width, probes),
                )
                post_update_rescore_ms = (time.perf_counter() - started) * 1e3
                row = {
                    "gates": float(len(operations)),
                    "support": support,
                    "bound": bound,
                    "before_fidelity": before_fidelity,
                    "after_fidelity": after_fidelity,
                    "transport_cold_build_ms": transport_cold_build_ms,
                    "optimization_ms": optimization_ms,
                    "payload_bytes": retained_payload_bytes,
                    "controls": float(result.controls.size / result.controls.shape[0]),
                    "projection_residual": projection_residual,
                    "analytic_coordinate_model_ms": analytic_coordinate_model_ms,
                    "baseline_rescore_ms": baseline_rescore_ms,
                    "post_update_rescore_ms": post_update_rescore_ms,
                    "fidelity_gain": after_fidelity - before_fidelity,
                    "nearest_mean_local_process_fidelity": nearest_mean_fidelity,
                    "nearest_min_local_process_fidelity": nearest_min_fidelity,
                }
                rows_by_configuration[(error_model, estimator, support_cap)].append(row)
                case_rows.append(
                    {
                        "benchmark": benchmark,
                        "seed": seed,
                        "width": width,
                        "fixture_index": fixture_index,
                        "error_model": error_model,
                        "estimator": estimator,
                        "support_cap": support_cap,
                        **row,
                    }
                )

    def mean(rows: list[dict[str, float | None]], key: str) -> float:
        return float(np.mean([float(row[key]) for row in rows]))

    def case_sem(rows: list[dict[str, float | None]], key: str) -> float | None:
        values = np.array([float(row[key]) for row in rows], dtype=float)
        if len(values) < 2:
            return None
        return float(np.std(values, ddof=1) / np.sqrt(len(values)))

    def number_or_na(value: float | None, *, precision: str = ".3e") -> str:
        return "not-applicable" if value is None else format(value, precision)

    def optional_mean(rows: list[dict[str, float | None]], key: str) -> float | None:
        values = [float(row[key]) for row in rows if row[key] is not None]
        return None if not values else float(np.mean(values))

    retained_names = tuple(sorted({benchmark for benchmark, _, _, _ in fixtures}))
    gate_counts = np.array([len(operations) for _, _, _, operations in fixtures])
    for error_model in error_models:
        for estimator, support_cap in configurations:
            rows = rows_by_configuration[(error_model, estimator, support_cap)]
            before_fidelity = mean(rows, "before_fidelity")
            after_fidelity = mean(rows, "after_fidelity")
            before_infidelity = 1.0 - before_fidelity
            after_infidelity = 1.0 - after_fidelity
            infidelity_reduction = 0.0 if before_infidelity <= 0.0 else 1.0 - after_infidelity / before_infidelity
            reductions = np.array(
                [
                    1.0 - (1.0 - row["after_fidelity"]) / (1.0 - row["before_fidelity"])
                    for row in rows
                    if row["before_fidelity"] < 1.0
                ]
            )
            print(f"corpus={args.corpus}")
            print(f"error_model={error_model}")
            print(f"estimator={estimator}")
            print(f"sample_count={len(rows)}")
            print(f"seed_count={len(seeds)}")
            print(f"widths={','.join(str(width) for width in widths)}")
            if args.corpus == "synthetic":
                print(f"layers={args.layers}")
                print(f"circuit_count={args.circuits}")
            print(f"fidelity_probe_count={args.fidelity_probes}")
            print(f"optimization_backend={args.optimization_backend}")
            if args.optimization_backend == "fpga-greedy":
                print(f"fpga_port={args.fpga_port}")
                print(f"fpga_baud={args.fpga_baud}")
                print(f"fpga_fallback={args.fpga_fallback}")
                print(f"fpga_max_sweeps={args.fpga_max_sweeps}")
                print("fpga_link_batch=shared-benchmark-session")
                print(f"fpga_link_setup_ms={fpga_link_setup_ms:.3f}")
            print("fidelity_metric=sampled-haar-output-state-fidelity; not exact-average-gate-fidelity")
            print("control_model=one-slice-closed-system-bilinear")
            print("local_coordinate_source=analytic-simulator-oracle; no calibration-estimation noise or model mismatch")
            print("truncation_bound_scope=actual-post-update-local-coordinates; cap0=unavailable")
            print("two_qubit_generator_basis=full-su4")
            print(f"pulse_coordinates_per_gate={rows[0]['controls']:.0f}")
            print(f"mean_local_generator_projection_residual={mean(rows, 'projection_residual'):.6e}")
            print(f"retained_benchmark_count={len(retained_names)}")
            print(f"retained_benchmarks={','.join(retained_names)}")
            print(f"excluded_gate_count_outliers={','.join(excluded_outliers) if excluded_outliers else 'none'}")
            print(f"gate_count_range={gate_counts.min()}-{gate_counts.max()}")
            print(f"mean_gate_count={mean(rows, 'gates'):.2f}")
            cap_label = (
                0 if estimator == "no-bias"
                else "not-used" if estimator in ("local-error-oracle", "nearest-clifford")
                else (support_cap if support_cap is not None else "unbounded")
            )
            print(f"support_cap={cap_label}")
            support_label = "not-applicable" if estimator == "local-error-oracle" else f"{mean(rows, 'support'):.2f}"
            bound = optional_mean(rows, "bound") if estimator == "anchored-pauli" else None
            bound_label = "unavailable" if estimator == "anchored-pauli" and bound is None else number_or_na(bound)
            print(f"mean_retained_pauli_support={support_label}")
            print(f"mean_post_update_truncation_bound={bound_label}")
            print(f"mean_sampled_haar_output_state_fidelity_before={before_fidelity:.9f}")
            print(f"mean_sampled_haar_output_state_fidelity_after={after_fidelity:.9f}")
            print(f"mean_sampled_haar_fidelity_gain={mean(rows, 'fidelity_gain'):.9e}")
            print(f"sampled_haar_fidelity_gain_across_case_sem={number_or_na(case_sem(rows, 'fidelity_gain'))}")
            print(f"sampled_haar_non_improving_case_fraction={np.mean([row['fidelity_gain'] <= 0.0 for row in rows]):.2%}")
            print(f"mean_sampled_haar_output_state_infidelity_reduction={infidelity_reduction:.2%}")
            reduction_p05 = None if not len(reductions) else float(np.quantile(reductions, 0.05))
            reduction_p95 = None if not len(reductions) else float(np.quantile(reductions, 0.95))
            print(f"sampled_haar_output_state_infidelity_reduction_p05={number_or_na(reduction_p05, precision='.2%')}")
            print(f"sampled_haar_output_state_infidelity_reduction_p95={number_or_na(reduction_p95, precision='.2%')}")
            if estimator == "nearest-clifford":
                print(f"mean_nearest_clifford_local_process_fidelity={mean(rows, 'nearest_mean_local_process_fidelity'):.9f}")
                print(f"min_nearest_clifford_local_process_fidelity={min(row['nearest_min_local_process_fidelity'] for row in rows):.9f}")
                print("nearest_clifford_circuit_approximation_bound=not-available")
            print(f"mean_analytic_simulator_coordinate_model_ms={mean(rows, 'analytic_coordinate_model_ms'):.3f}")
            print(f"mean_baseline_rescore_ms={mean(rows, 'baseline_rescore_ms'):.3f}")
            print(f"mean_post_update_rescore_ms={mean(rows, 'post_update_rescore_ms'):.3f}")
            transport_time_label = "not-applicable" if estimator in ("no-bias", "local-error-oracle") else f"{mean(rows, 'transport_cold_build_ms'):.3f}"
            print(f"mean_transport_cold_build_ms={transport_time_label}")
            print(f"mean_optimization_ms={mean(rows, 'optimization_ms'):.3f}")
            payload_bytes_label = "not-applicable" if estimator in ("no-bias", "local-error-oracle") else f"{mean(rows, 'payload_bytes'):.0f}"
            print(f"mean_transport_payload_bytes={payload_bytes_label}")
            print("statement=ideal-circuit coherent-error simulation; not hardware fidelity")

    if args.output is not None:
        def json_value(value: str | int | float | None) -> str | int | float | None:
            return None if isinstance(value, float) and not np.isfinite(value) else value

        payload = {
            "corpus": args.corpus,
            "fidelity_probe_count": args.fidelity_probes,
            "optimization_backend": args.optimization_backend,
            "fpga_port": args.fpga_port if args.optimization_backend == "fpga-greedy" else None,
            "fpga_baud": args.fpga_baud if fpga_link is not None else None,
            "fpga_link_batch": "shared-benchmark-session" if fpga_link is not None else None,
            "fpga_link_setup_ms": fpga_link_setup_ms,
            "fpga_max_sweeps": args.fpga_max_sweeps if fpga_link is not None else None,
            "records": [
                {key: json_value(value) for key, value in row.items()}
                for row in case_rows
            ],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"per_case_output={args.output}")
    if fpga_link is not None:
        fpga_link.close()


if __name__ == "__main__":
    main()
