"""Calibration-map ingestion for channel-level BiasBlaster propagation.

A calibrated device model is represented as named physical parameters plus
rules that attach each parameter's local PTM derivative to matching circuit
locations. Repeated occurrences of a shared parameter retain the same name, so
calibration covariance can represent quasistatic drift and cross-parameter
correlations across gates and across Quantum Krylov estimator circuits.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Literal, Mapping, Sequence

import numpy as np

from .channel import ChannelMode
from .model import CircuitOperation


CalibrationLocation = Literal[
    "before_gate",
    "after_gate",
    "initial",
    "measurement",
]


@dataclass(frozen=True)
class CalibrationParameter:
    """One scalar calibrated hardware parameter."""

    name: str
    mean: float

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("calibration parameter name must be nonempty")
        mean = float(self.mean)
        if not np.isfinite(mean):
            raise ValueError("calibration parameter mean must be finite")
        object.__setattr__(self, "mean", mean)


@dataclass(frozen=True)
class CalibrationRule:
    """Attach one calibrated PTM derivative to matching circuit locations.

    ``gate_names`` and ``qubits`` are optional filters for gate-local rules.
    ``arity`` is inferred from ``delta_ptm`` and must agree with every match.
    Boundary rules (`initial`/`measurement`) require explicit ``qubits``.
    """

    parameter: str
    location: CalibrationLocation
    delta_ptm: np.ndarray
    gate_names: tuple[str, ...] = ()
    qubits: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if not self.parameter:
            raise ValueError("rule parameter must be nonempty")
        if self.location not in {
            "before_gate", "after_gate", "initial", "measurement"
        }:
            raise ValueError("unsupported calibration rule location")
        gate_names = tuple(str(name) for name in self.gate_names)
        if any(not name for name in gate_names):
            raise ValueError("gate_names cannot contain empty strings")
        qubits = None if self.qubits is None else tuple(int(q) for q in self.qubits)
        if qubits is not None:
            if not qubits or len(set(qubits)) != len(qubits) or min(qubits) < 0:
                raise ValueError("rule qubits must be nonempty, distinct, and nonnegative")
            if len(qubits) not in (1, 2):
                raise ValueError("calibration rules currently support one- or two-qubit locality")
        delta = np.asarray(self.delta_ptm, dtype=float)
        if delta.shape not in ((4, 4), (16, 16)) or not np.all(np.isfinite(delta)):
            raise ValueError("delta_ptm must be a finite 1Q or 2Q Pauli-transfer derivative")
        arity = 1 if delta.shape == (4, 4) else 2
        if qubits is not None and len(qubits) != arity:
            raise ValueError("rule qubits and delta_ptm arity disagree")
        if self.location in {"initial", "measurement"}:
            if qubits is None:
                raise ValueError("boundary calibration rules require explicit qubits")
            if gate_names:
                raise ValueError("boundary calibration rules cannot filter gate names")
        delta = delta.copy()
        delta.setflags(write=False)
        object.__setattr__(self, "gate_names", gate_names)
        object.__setattr__(self, "qubits", qubits)
        object.__setattr__(self, "delta_ptm", delta)

    @property
    def arity(self) -> int:
        return 1 if self.delta_ptm.shape == (4, 4) else 2

    def matches(self, operation: CircuitOperation) -> bool:
        if self.location not in {"before_gate", "after_gate"}:
            return False
        if len(operation.qubits) != self.arity:
            return False
        if self.gate_names and operation.name not in self.gate_names:
            return False
        if self.qubits is not None and tuple(operation.qubits) != self.qubits:
            return False
        return True


@dataclass(frozen=True)
class ExpandedCalibration:
    """Circuit-local channel modes plus covariance in occurrence coordinates."""

    modes: tuple[ChannelMode, ...]
    mode_covariance: np.ndarray | None
    parameter_names: tuple[str, ...]
    occurrence_parameter_indices: np.ndarray


@dataclass(frozen=True)
class CalibratedErrorMap:
    """Named calibrated parameters, local attachment rules, and covariance."""

    parameters: tuple[CalibrationParameter, ...]
    rules: tuple[CalibrationRule, ...]
    covariance: np.ndarray | None = None

    def __post_init__(self) -> None:
        parameters = tuple(self.parameters)
        rules = tuple(self.rules)
        names = tuple(parameter.name for parameter in parameters)
        if len(set(names)) != len(names):
            raise ValueError("calibration parameter names must be unique")
        if not parameters:
            raise ValueError("at least one calibration parameter is required")
        unknown = sorted({rule.parameter for rule in rules} - set(names))
        if unknown:
            raise ValueError(f"rules reference unknown parameters: {unknown}")
        covariance = None
        if self.covariance is not None:
            covariance = np.asarray(self.covariance, dtype=float)
            if covariance.shape != (len(parameters), len(parameters)):
                raise ValueError("calibration covariance has the wrong shape")
            if not np.all(np.isfinite(covariance)):
                raise ValueError("calibration covariance must be finite")
            if not np.allclose(covariance, covariance.T, atol=1e-10, rtol=0.0):
                raise ValueError("calibration covariance must be symmetric")
            if np.min(np.linalg.eigvalsh(covariance), initial=0.0) < -1e-10:
                raise ValueError("calibration covariance must be positive semidefinite")
            covariance = covariance.copy()
            covariance.setflags(write=False)
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "rules", rules)
        object.__setattr__(self, "covariance", covariance)

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(parameter.name for parameter in self.parameters)

    def expand(
        self,
        circuit: Sequence[CircuitOperation],
        n_qubits: int,
    ) -> ExpandedCalibration:
        """Expand calibration rules into channel modes for one compiled circuit."""

        operations = tuple(circuit)
        if n_qubits < 1:
            raise ValueError("n_qubits must be positive")
        if any(q >= n_qubits for op in operations for q in op.qubits):
            raise ValueError("circuit operation lies outside register")
        if any(
            rule.qubits is not None and any(q >= n_qubits for q in rule.qubits)
            for rule in self.rules
        ):
            raise ValueError("calibration rule lies outside register")

        means = {parameter.name: parameter.mean for parameter in self.parameters}
        parameter_index = {
            parameter.name: index for index, parameter in enumerate(self.parameters)
        }
        modes: list[ChannelMode] = []
        indices: list[int] = []

        for rule in self.rules:
            if rule.location == "initial":
                modes.append(ChannelMode(
                    rule.parameter,
                    0,
                    rule.qubits,
                    rule.delta_ptm,
                    mean=means[rule.parameter],
                ))
                indices.append(parameter_index[rule.parameter])
            elif rule.location == "measurement":
                modes.append(ChannelMode(
                    rule.parameter,
                    len(operations),
                    rule.qubits,
                    rule.delta_ptm,
                    mean=means[rule.parameter],
                ))
                indices.append(parameter_index[rule.parameter])

        for gate_index, operation in enumerate(operations):
            for rule in self.rules:
                if not rule.matches(operation):
                    continue
                boundary = gate_index if rule.location == "before_gate" else gate_index + 1
                modes.append(ChannelMode(
                    rule.parameter,
                    boundary,
                    operation.qubits,
                    rule.delta_ptm,
                    mean=means[rule.parameter],
                ))
                indices.append(parameter_index[rule.parameter])

        occurrence_indices = np.asarray(indices, dtype=int)
        expanded_covariance = None
        if self.covariance is not None:
            if len(occurrence_indices):
                expanded_covariance = self.covariance[
                    np.ix_(occurrence_indices, occurrence_indices)
                ].copy()
            else:
                expanded_covariance = np.zeros((0, 0), dtype=float)
        return ExpandedCalibration(
            modes=tuple(modes),
            mode_covariance=expanded_covariance,
            parameter_names=self.parameter_names,
            occurrence_parameter_indices=occurrence_indices,
        )

    def to_dict(self) -> dict:
        return {
            "parameters": [
                {"name": parameter.name, "mean": parameter.mean}
                for parameter in self.parameters
            ],
            "covariance": None if self.covariance is None else self.covariance.tolist(),
            "rules": [
                {
                    "parameter": rule.parameter,
                    "location": rule.location,
                    "delta_ptm": rule.delta_ptm.tolist(),
                    "gate_names": list(rule.gate_names),
                    "qubits": None if rule.qubits is None else list(rule.qubits),
                }
                for rule in self.rules
            ],
        }

    @classmethod
    def from_dict(cls, payload: Mapping) -> "CalibratedErrorMap":
        parameters = tuple(
            CalibrationParameter(str(item["name"]), float(item["mean"]))
            for item in payload.get("parameters", ())
        )
        rules = tuple(
            CalibrationRule(
                parameter=str(item["parameter"]),
                location=str(item["location"]),
                delta_ptm=np.asarray(item["delta_ptm"], dtype=float),
                gate_names=tuple(item.get("gate_names", ())),
                qubits=None if item.get("qubits") is None else tuple(item["qubits"]),
            )
            for item in payload.get("rules", ())
        )
        covariance_payload = payload.get("covariance")
        covariance = (
            None
            if covariance_payload is None
            else np.asarray(covariance_payload, dtype=float)
        )
        return cls(parameters=parameters, rules=rules, covariance=covariance)


def load_calibrated_error_map(path: str | Path) -> CalibratedErrorMap:
    """Load a calibrated error map from JSON."""

    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return CalibratedErrorMap.from_dict(payload)


def save_calibrated_error_map(
    calibration: CalibratedErrorMap,
    path: str | Path,
    *,
    indent: int = 2,
) -> None:
    """Write a calibrated error map to JSON."""

    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(calibration.to_dict(), handle, indent=indent)
        handle.write("\n")
