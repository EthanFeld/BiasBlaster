"""Optional Quantinuum Nexus bridge for Quantum Krylov estimator circuits.

This module is intentionally lazy-imported: BiasBlaster's core package does not
require Nexus credentials or the Quantinuum SDK. Install the ``quantinuum``
extra to upload, compile and execute the same scalar estimator circuits used by
the offline benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .qkrylov_experiment import KrylovEstimatorCircuit, KrylovExperimentPlan


@dataclass(frozen=True)
class NexusExecutionConfig:
    """Configuration for a batched Nexus Quantum Krylov execution."""

    project_name: str = "BiasBlaster Quantum Krylov"
    system_name: str = "Helios-1E-lite"
    optimisation_level: int = 1
    timeout: float = 900.0
    max_cost: float | None = None

    def __post_init__(self) -> None:
        if not self.project_name or not self.system_name:
            raise ValueError("project_name and system_name must be nonempty")
        if self.optimisation_level < 0:
            raise ValueError("optimisation_level must be nonnegative")
        timeout = float(self.timeout)
        if timeout <= 0 or not np.isfinite(timeout):
            raise ValueError("timeout must be finite and positive")
        object.__setattr__(self, "timeout", timeout)
        if self.max_cost is not None:
            max_cost = float(self.max_cost)
            if max_cost <= 0 or not np.isfinite(max_cost):
                raise ValueError("max_cost must be finite and positive")
            object.__setattr__(self, "max_cost", max_cost)


def _require_quantinuum_stack():
    try:
        import qnexus as qnx
        from qiskit import QuantumCircuit
        from pytket.extensions.qiskit import qiskit_to_tk
    except ImportError as exc:  # pragma: no cover - optional dependency path
        raise ImportError(
            "Quantinuum execution requires the optional extra: "
            "python -m pip install -e '.[quantinuum]'"
        ) from exc
    return qnx, QuantumCircuit, qiskit_to_tk


def estimator_to_qiskit(estimator: KrylovEstimatorCircuit):
    """Reconstruct one compiled estimator as a one-bit Qiskit circuit."""

    _, QuantumCircuit, _ = _require_quantinuum_stack()
    circuit = QuantumCircuit(estimator.n_qubits, 1, name=estimator.name)
    for operation in estimator.operations:
        if operation.name == "rz" and len(operation.qubits) == 1:
            circuit.rz(operation.angle, operation.qubits[0])
        elif operation.name == "sx" and len(operation.qubits) == 1:
            circuit.sx(operation.qubits[0])
        elif operation.name == "x" and len(operation.qubits) == 1:
            circuit.x(operation.qubits[0])
        elif operation.name == "cx" and len(operation.qubits) == 2:
            circuit.cx(operation.qubits[0], operation.qubits[1])
        else:
            # The current benchmark compiler emits only rz/sx/x/cx. Keeping a
            # unitary fallback makes the bridge usable with custom estimators.
            circuit.unitary(operation.unitary, list(operation.qubits), label=operation.name)
    circuit.measure(0, 0)
    return circuit


def estimator_to_pytket(estimator: KrylovEstimatorCircuit):
    """Convert one BiasBlaster estimator into a measured pytket Circuit."""

    _, _, qiskit_to_tk = _require_quantinuum_stack()
    return qiskit_to_tk(estimator_to_qiskit(estimator))


def plan_to_pytket(plan: KrylovExperimentPlan):
    """Convert all scalar H/S estimator circuits in plan order."""

    return [estimator_to_pytket(estimator) for estimator in plan.estimators]


def _helios_config(qnx, plan: KrylovExperimentPlan, config: NexusExecutionConfig):
    kwargs = {
        "system_name": config.system_name,
        "emulator_config": qnx.models.HeliosEmulatorConfig(
            n_qubits=max(estimator.n_qubits for estimator in plan.estimators),
        ),
    }
    if config.max_cost is not None:
        # max_cost remains accepted by HeliosConfig although newer Nexus APIs
        # prefer per-job max-cost controls.
        kwargs["max_cost"] = config.max_cost
    return qnx.models.HeliosConfig(**kwargs)


def upload_compile_plan(
    plan: KrylovExperimentPlan,
    *,
    config: NexusExecutionConfig | None = None,
):
    """Upload and compile all estimator circuits; return project/config/refs."""

    qnx, _, _ = _require_quantinuum_stack()
    if config is None:
        config = NexusExecutionConfig()
    project = qnx.projects.get_or_create(name=config.project_name)
    circuits = plan_to_pytket(plan)
    uploaded = [
        qnx.circuits.upload(
            circuit=circuit,
            name=f"BiasBlaster-{estimator.name}",
            project=project,
            properties={
                "biasblaster_estimator": estimator.name,
                "krylov_dimension": plan.dimension,
            },
        )
        for estimator, circuit in zip(plan.estimators, circuits, strict=True)
    ]
    backend = _helios_config(qnx, plan, config)
    compiled = qnx.compile(
        programs=uploaded,
        backend_config=backend,
        name="BiasBlaster-QK-compile",
        project=project,
        optimisation_level=config.optimisation_level,
        timeout=config.timeout,
    )
    return project, backend, compiled


def _bit_from_outcome(outcome) -> int:
    if isinstance(outcome, tuple):
        if len(outcome) != 1:
            raise ValueError("expected exactly one measured classical bit")
        return int(outcome[0])
    if isinstance(outcome, (int, np.integer)):
        return int(outcome) & 1
    text = str(outcome).replace(" ", "")
    if text.endswith("0"):
        return 0
    if text.endswith("1"):
        return 1
    raise ValueError(f"cannot decode Nexus outcome {outcome!r}")


def counts_to_pm1_expectation(counts) -> float:
    """Convert one-bit counts to <Z> = P(0)-P(1)."""

    zero = one = 0
    for outcome, count in counts.items():
        bit = _bit_from_outcome(outcome)
        if bit == 0:
            zero += int(count)
        else:
            one += int(count)
    total = zero + one
    if total <= 0:
        raise ValueError("empty measurement counts")
    return float((zero - one) / total)


def execute_nexus_estimators(
    plan: KrylovExperimentPlan,
    shots: Sequence[int],
    *,
    config: NexusExecutionConfig | None = None,
) -> np.ndarray:
    """Compile/execute estimator circuits and return +/-1 means in plan order.

    The returned vector can be passed directly to
    ``assemble_krylov_matrices`` through the estimator names in ``plan``. Nexus
    credentials/project access are intentionally left to qnexus itself.
    """

    qnx, _, _ = _require_quantinuum_stack()
    allocations = np.asarray(shots, dtype=int)
    if allocations.shape != (len(plan.estimators),) or np.any(allocations < 1):
        raise ValueError("shots must be positive and match the estimator count")
    if len(plan.estimators) > 300:
        raise ValueError(
            "Nexus jobs accept at most 300 programs; split this plan into batches"
        )
    if config is None:
        config = NexusExecutionConfig()
    project, backend, compiled = upload_compile_plan(plan, config=config)
    results = qnx.execute(
        programs=compiled,
        n_shots=[int(value) for value in allocations],
        backend_config=backend,
        name="BiasBlaster-QK-execute",
        project=project,
        timeout=config.timeout,
    )
    if len(results) != len(plan.estimators):
        raise RuntimeError("Nexus returned a different number of results than estimators")
    return np.asarray(
        [counts_to_pm1_expectation(result.get_counts()) for result in results],
        dtype=float,
    )
