"""Reproduce the compact Pauli-versus-nearest-Clifford tradeoff."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from biasblaster import (
    CircuitOperation,
    estimate_nearest_clifford_transport,
    estimate_pauli_transport,
    optimize_bias_controls,
)


def _rz(angle: float) -> np.ndarray:
    return np.diag([np.exp(-0.5j * angle), np.exp(0.5j * angle)])


def _h() -> np.ndarray:
    return np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2.0)


def _cx() -> np.ndarray:
    return np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
        dtype=complex,
    )


def _circuit(width: int, circuit_index: int, rng: np.random.Generator) -> tuple[CircuitOperation, ...]:
    operations: list[CircuitOperation] = []
    for layer in range(3):
        for qubit in range(width):
            angle = 0.17 * (circuit_index + 1) + 0.11 * (layer + 1) * (qubit + 1)
            angle += float(rng.normal(scale=0.01))
            operations.append(CircuitOperation("rz", (qubit,), _rz(angle), angle))
            operations.append(CircuitOperation("h", (qubit,), _h()))
        for qubit in range(0, width - 1, 2):
            operations.append(CircuitOperation("cx", (qubit, qubit + 1), _cx()))
    return tuple(operations)


def _infidelity_proxy(norm: float) -> float:
    """A first-order coherent-norm proxy, not a fidelity prediction."""

    return 0.5 * float(norm) ** 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--estimator", choices=("nearest-clifford", "pauli"), default="pauli")
    parser.add_argument("--support-cap", type=int, default=64)
    parser.add_argument("--profile", choices=("default", "rydberg-amplitude-5pct"), default="default")
    parser.add_argument("--width", type=int, default=4)
    parser.add_argument("--circuits", type=int, default=3)
    parser.add_argument("--effort-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=7)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.width < 1 or args.circuits < 1 or args.support_cap < 1 or not 0 <= args.effort_fraction <= 1:
        raise SystemExit("width, circuits, and support-cap must be positive; effort-fraction must lie in [0, 1]")

    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, float]] = []
    for circuit_index in range(args.circuits):
        operations = _circuit(args.width, circuit_index, rng)
        generators = 7.5e-4 * rng.normal(size=(len(operations), 3))
        if args.profile == "rydberg-amplitude-5pct":
            generators *= 1.05
        jacobians = np.zeros((len(operations), 3, 3), dtype=float)
        jacobians[:] = np.eye(3)
        started = time.perf_counter()
        if args.estimator == "pauli":
            transport = estimate_pauli_transport(operations, args.width, support_cap=args.support_cap)
        else:
            transport = estimate_nearest_clifford_transport(operations, args.width)
        elapsed = time.perf_counter() - started
        result = optimize_bias_controls(generators, transport, jacobians, args.effort_fraction)
        rows.append(
            {
                "gates": float(len(operations)),
                "support": float(transport.support_count),
                "bound": float(transport.truncation_bound_for(generators)),
                "before": _infidelity_proxy(result.predicted_before_norm),
                "after": _infidelity_proxy(result.predicted_after_norm),
                "runtime_ms": elapsed * 1e3,
                "bytes": float(transport.retained_bytes),
            }
        )

    def mean(key: str) -> float:
        return float(np.mean([row[key] for row in rows]))

    before = mean("before")
    after = mean("after")
    reduction = 0.0 if before == 0 else 1.0 - after / before
    print(f"estimator={args.estimator}")
    print(f"profile={args.profile}")
    print(f"circuit_count={args.circuits}")
    print(f"mean_gate_count={mean('gates'):.2f}")
    print(f"support_cap={args.support_cap if args.estimator == 'pauli' else 'not-used'}")
    print(f"mean_retained_pauli_support={mean('support'):.2f}")
    print(f"mean_truncation_bound={mean('bound'):.6e}")
    print(f"mean_before_coherent_infidelity_proxy={before:.6e}")
    print(f"mean_after_coherent_infidelity_proxy={after:.6e}")
    print(f"mean_reduction={reduction:.2%}")
    print(f"mean_runtime_ms={mean('runtime_ms'):.3f}")
    print(f"mean_transport_bytes={mean('bytes'):.0f}")
    print("statement=algebraic first-order model estimates only; not hardware fidelity")


if __name__ == "__main__":
    main()
