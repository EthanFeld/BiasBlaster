"""Compare fixed and model-selected Krylov spacings across shot-noise seeds."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from biasblaster import EffectiveNoiseParameters
from biasblaster.qkrylov_adaptive import run_tfim_qkrylov_adaptive_benchmark
from biasblaster.qkrylov_basis import select_tfim_time_step
from biasblaster.qkrylov_shrinkage import run_tfim_qkrylov_shrinkage_policy


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=str, default="0.2,0.4,0.6")
    parser.add_argument("--baseline-time-step", type=float, default=0.2)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--qubits", type=int, default=2)
    parser.add_argument("--dimension", type=int, default=2)
    parser.add_argument("--trotter-steps", type=int, default=1)
    parser.add_argument("--shots", type=int, default=20_000)
    parser.add_argument("--minimum-shots", type=int, default=100)
    parser.add_argument("--noise-scale", type=float, default=1.0)
    parser.add_argument("--max-condition-number", type=float, default=25.0)
    parser.add_argument("--shrinkage-strength", type=float, default=1.0)
    parser.add_argument("--json", action="store_true")
    return parser


def _policy(result, name: str):
    by_name = {policy.name: policy for policy in result.policies}
    if name not in by_name:
        raise RuntimeError(f"benchmark did not produce required policy {name}")
    return by_name[name]


def _stats(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "std": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _comparison(candidate: np.ndarray, baseline: np.ndarray) -> dict[str, float]:
    relative = (candidate - baseline) / np.maximum(np.abs(baseline), 1e-15)
    return {
        "beats_fraction": float(np.mean(candidate < baseline)),
        "mean_relative_change": float(np.mean(relative)),
        "median_relative_change": float(np.median(relative)),
    }


def main() -> None:
    args = _parser().parse_args()
    if args.seeds < 1:
        raise ValueError("--seeds must be positive")
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

    rows = []
    for seed in range(args.seed_start, args.seed_start + args.seeds):
        baseline = run_tfim_qkrylov_adaptive_benchmark(
            n_qubits=args.qubits,
            dimension=args.dimension,
            time_step=args.baseline_time_step,
            trotter_steps=args.trotter_steps,
            total_shots=args.shots,
            minimum_shots=args.minimum_shots,
            seed=seed,
            noise=noise,
            max_condition_number=args.max_condition_number,
        )
        selected = run_tfim_qkrylov_adaptive_benchmark(
            n_qubits=args.qubits,
            dimension=args.dimension,
            time_step=selection.selected_time_step,
            trotter_steps=args.trotter_steps,
            total_shots=args.shots,
            minimum_shots=args.minimum_shots,
            seed=seed,
            noise=noise,
            max_condition_number=args.max_condition_number,
        )
        baseline_shrinkage, baseline_shrink = run_tfim_qkrylov_shrinkage_policy(
            n_qubits=args.qubits,
            dimension=args.dimension,
            time_step=args.baseline_time_step,
            trotter_steps=args.trotter_steps,
            total_shots=args.shots,
            minimum_shots=args.minimum_shots,
            seed=seed,
            noise=noise,
            shrinkage_strength=args.shrinkage_strength,
        )
        selected_shrinkage, selected_shrink = run_tfim_qkrylov_shrinkage_policy(
            n_qubits=args.qubits,
            dimension=args.dimension,
            time_step=selection.selected_time_step,
            trotter_steps=args.trotter_steps,
            total_shots=args.shots,
            minimum_shots=args.minimum_shots,
            seed=seed,
            noise=noise,
            shrinkage_strength=args.shrinkage_strength,
        )
        base_policy = _policy(baseline, "debiased_modewise")
        selected_policy = _policy(selected, "debiased_modewise")
        rows.append({
            "seed": seed,
            "baseline_ground_error": base_policy.ground_energy_error,
            "selected_ground_error": selected_policy.ground_energy_error,
            "baseline_shrinkage_ground_error": baseline_shrinkage.ground_energy_error,
            "selected_shrinkage_ground_error": selected_shrinkage.ground_energy_error,
            "baseline_qk_deviation": base_policy.deviation_from_ideal_qk,
            "selected_qk_deviation": selected_policy.deviation_from_ideal_qk,
            "baseline_shrinkage_qk_deviation": baseline_shrinkage.deviation_from_ideal_qk,
            "selected_shrinkage_qk_deviation": selected_shrinkage.deviation_from_ideal_qk,
            "baseline_rank": base_policy.retained_rank,
            "selected_rank": selected_policy.retained_rank,
            "baseline_shrinkage_rank": baseline_shrinkage.retained_rank,
            "selected_shrinkage_rank": selected_shrinkage.retained_rank,
            "baseline_ideal_qk_error": baseline.policies[0].ground_energy_error,
            "selected_ideal_qk_error": selected.policies[0].ground_energy_error,
            "baseline_two_qubit_executions": base_policy.weighted_two_qubit_executions,
            "selected_two_qubit_executions": selected_policy.weighted_two_qubit_executions,
            "baseline_shrinkage_two_qubit_executions": baseline_shrinkage.weighted_two_qubit_executions,
            "selected_shrinkage_two_qubit_executions": selected_shrinkage.weighted_two_qubit_executions,
            "baseline_shrinkage_mean_weight": float(np.mean(baseline_shrink.weights)),
            "selected_shrinkage_mean_weight": float(np.mean(selected_shrink.weights)),
        })

    baseline_errors = np.asarray([row["baseline_ground_error"] for row in rows], dtype=float)
    selected_errors = np.asarray([row["selected_ground_error"] for row in rows], dtype=float)
    baseline_shrinkage_errors = np.asarray(
        [row["baseline_shrinkage_ground_error"] for row in rows], dtype=float
    )
    selected_shrinkage_errors = np.asarray(
        [row["selected_shrinkage_ground_error"] for row in rows], dtype=float
    )
    baseline_qk = np.asarray([row["baseline_qk_deviation"] for row in rows], dtype=float)
    selected_qk = np.asarray([row["selected_qk_deviation"] for row in rows], dtype=float)
    baseline_shrinkage_qk = np.asarray(
        [row["baseline_shrinkage_qk_deviation"] for row in rows], dtype=float
    )
    selected_shrinkage_qk = np.asarray(
        [row["selected_shrinkage_qk_deviation"] for row in rows], dtype=float
    )
    payload = {
        "configuration": {
            "baseline_time_step": args.baseline_time_step,
            "selected_time_step": selection.selected_time_step,
            "candidate_time_steps": candidates,
            "seeds": args.seeds,
            "shots": args.shots,
            "noise_scale": args.noise_scale,
            "shrinkage_strength": args.shrinkage_strength,
        },
        "selection": [
            {
                "time_step": item.time_step,
                "retained_rank": item.retained_rank,
                "robust_margin": item.robust_margin,
                "predicted_condition_number": item.predicted_condition_number,
                "first_order_residual_rmse": item.first_order_residual_rmse,
            }
            for item in selection.assessments
        ],
        "comparison": {
            "baseline_ground_error": _stats(baseline_errors.tolist()),
            "selected_ground_error": _stats(selected_errors.tolist()),
            "baseline_shrinkage_ground_error": _stats(baseline_shrinkage_errors.tolist()),
            "selected_shrinkage_ground_error": _stats(selected_shrinkage_errors.tolist()),
            "baseline_qk_deviation": _stats(baseline_qk.tolist()),
            "selected_qk_deviation": _stats(selected_qk.tolist()),
            "baseline_shrinkage_qk_deviation": _stats(baseline_shrinkage_qk.tolist()),
            "selected_shrinkage_qk_deviation": _stats(selected_shrinkage_qk.tolist()),
            "selected_vs_baseline_ground": _comparison(selected_errors, baseline_errors),
            "selected_shrinkage_vs_baseline_full_debias_ground": _comparison(
                selected_shrinkage_errors, baseline_errors
            ),
            "selected_shrinkage_vs_baseline_shrinkage_ground": _comparison(
                selected_shrinkage_errors, baseline_shrinkage_errors
            ),
            "selected_shrinkage_vs_baseline_shrinkage_qk": _comparison(
                selected_shrinkage_qk, baseline_shrinkage_qk
            ),
            "selected_shrinkage_vs_selected_full_debias_qk": _comparison(
                selected_shrinkage_qk, selected_qk
            ),
            "baseline_full_rank_fraction": float(np.mean([
                row["baseline_rank"] == args.dimension for row in rows
            ])),
            "selected_full_rank_fraction": float(np.mean([
                row["selected_rank"] == args.dimension for row in rows
            ])),
            "baseline_shrinkage_full_rank_fraction": float(np.mean([
                row["baseline_shrinkage_rank"] == args.dimension for row in rows
            ])),
            "selected_shrinkage_full_rank_fraction": float(np.mean([
                row["selected_shrinkage_rank"] == args.dimension for row in rows
            ])),
            "baseline_shrinkage_mean_weight": float(np.mean([
                row["baseline_shrinkage_mean_weight"] for row in rows
            ])),
            "selected_shrinkage_mean_weight": float(np.mean([
                row["selected_shrinkage_mean_weight"] for row in rows
            ])),
        },
        "rows": rows,
    }
    print(json.dumps(payload, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
