"""Select a hardware-aware Krylov time step, then evaluate it at fixed shots."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from biasblaster import EffectiveNoiseParameters
from biasblaster.qkrylov_adaptive import run_tfim_qkrylov_adaptive_benchmark
from biasblaster.qkrylov_basis import select_tfim_time_step


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=str, default="0.2,0.4,0.6")
    parser.add_argument("--qubits", type=int, default=2)
    parser.add_argument("--dimension", type=int, default=2)
    parser.add_argument("--trotter-steps", type=int, default=1)
    parser.add_argument("--shots", type=int, default=20_000)
    parser.add_argument("--minimum-shots", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--noise-scale", type=float, default=1.0)
    parser.add_argument("--max-condition-number", type=float, default=25.0)
    parser.add_argument("--json", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    candidates = [float(value) for value in args.candidates.split(",") if value.strip()]
    noise = EffectiveNoiseParameters(scale=args.noise_scale)
    selection = select_tfim_time_step(
        candidates,
        n_qubits=args.qubits,
        dimension=args.dimension,
        trotter_steps=args.trotter_steps,
        total_shots=args.shots,
        minimum_shots=args.minimum_shots,
        noise=noise,
        max_condition_number=args.max_condition_number,
    )
    result = run_tfim_qkrylov_adaptive_benchmark(
        n_qubits=args.qubits,
        dimension=args.dimension,
        time_step=selection.selected_time_step,
        trotter_steps=args.trotter_steps,
        total_shots=args.shots,
        minimum_shots=args.minimum_shots,
        seed=args.seed,
        noise=noise,
        max_condition_number=args.max_condition_number,
    )
    payload = {
        "selected_time_step": selection.selected_time_step,
        "candidates": [
            {
                "time_step": item.time_step,
                "retained_rank": item.retained_rank,
                "robust_margin": item.robust_margin,
                "predicted_condition_number": item.predicted_condition_number,
                "conditioning_floor": item.conditioning_floor,
                "first_order_residual_rmse": item.first_order_residual_rmse,
                "weighted_two_qubit_executions": item.weighted_two_qubit_executions,
            }
            for item in selection.assessments
        ],
        "ideal_qk_energy": result.plan.ideal_qk_energy,
        "exact_ground_energy": result.plan.exact_ground_energy,
        "policies": [
            {
                "name": policy.name,
                "energy": policy.energy,
                "ground_energy_error": policy.ground_energy_error,
                "deviation_from_ideal_qk": policy.deviation_from_ideal_qk,
                "retained_rank": policy.retained_rank,
            }
            for policy in result.policies
        ],
    }
    print(json.dumps(payload, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
