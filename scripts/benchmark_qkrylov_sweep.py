"""Evaluate QK mitigation policies over repeated independent shot-noise seeds."""

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
from biasblaster.qkrylov_shrinkage import run_tfim_qkrylov_shrinkage_policy


POLICIES = (
    "noise_modewise",
    "debiased_modewise",
    "shrinkage_modewise",
    "adaptive_guarded",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--qubits", type=int, default=2)
    parser.add_argument("--dimension", type=int, default=2)
    parser.add_argument("--time-step", type=float, default=0.2)
    parser.add_argument("--trotter-steps", type=int, default=1)
    parser.add_argument("--shots", type=int, default=20_000)
    parser.add_argument("--minimum-shots", type=int, default=100)
    parser.add_argument("--pilot-fraction", type=float, default=0.2)
    parser.add_argument("--noise-scale", type=float, default=1.0)
    parser.add_argument("--shrinkage-strength", type=float, default=1.0)
    parser.add_argument("--json", action="store_true")
    return parser


def _finite_stats(values: list[float]) -> dict[str, float | int]:
    finite = [float(value) for value in values if np.isfinite(value)]
    if not finite:
        return {"count": 0, "mean": float("nan"), "median": float("nan"), "std": float("nan")}
    return {
        "count": len(finite),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "std": float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0,
    }


def _comparison(candidate: np.ndarray, baseline: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(candidate) & np.isfinite(baseline)
    if not np.any(valid):
        return {"beats_fraction": float("nan"), "mean_relative_change": float("nan")}
    candidate = candidate[valid]
    baseline = baseline[valid]
    denominator = np.maximum(np.abs(baseline), 1e-15)
    return {
        "beats_fraction": float(np.mean(candidate < baseline)),
        "mean_relative_change": float(np.mean((candidate - baseline) / denominator)),
    }


def main() -> None:
    args = _parser().parse_args()
    if args.seeds < 1:
        raise ValueError("--seeds must be positive")
    noise = EffectiveNoiseParameters(scale=args.noise_scale)
    rows: list[dict[str, object]] = []
    shrinkage_weight_means: list[float] = []
    for seed in range(args.seed_start, args.seed_start + args.seeds):
        result = run_tfim_qkrylov_adaptive_benchmark(
            n_qubits=args.qubits,
            dimension=args.dimension,
            time_step=args.time_step,
            trotter_steps=args.trotter_steps,
            total_shots=args.shots,
            minimum_shots=args.minimum_shots,
            pilot_fraction=args.pilot_fraction,
            seed=seed,
            noise=noise,
        )
        shrinkage_policy, shrinkage = run_tfim_qkrylov_shrinkage_policy(
            n_qubits=args.qubits,
            dimension=args.dimension,
            time_step=args.time_step,
            trotter_steps=args.trotter_steps,
            total_shots=args.shots,
            minimum_shots=args.minimum_shots,
            seed=seed,
            noise=noise,
            shrinkage_strength=args.shrinkage_strength,
        )
        policy_map = {policy.name: policy for policy in result.policies}
        policy_map[shrinkage_policy.name] = shrinkage_policy
        shrinkage_weight_means.append(float(np.mean(shrinkage.weights)))
        row: dict[str, object] = {
            "seed": seed,
            "ideal_qk_ground_error": result.policies[0].ground_energy_error,
            "shrinkage_mean_weight": shrinkage_weight_means[-1],
        }
        for name in POLICIES:
            policy = policy_map.get(name)
            row[f"{name}_ground_error"] = (
                policy.ground_energy_error if policy is not None else float("nan")
            )
            row[f"{name}_qk_deviation"] = (
                policy.deviation_from_ideal_qk if policy is not None else float("nan")
            )
            row[f"{name}_rank"] = policy.retained_rank if policy is not None else 0
        rows.append(row)

    summary: dict[str, object] = {
        "configuration": {
            "seeds": args.seeds,
            "seed_start": args.seed_start,
            "qubits": args.qubits,
            "dimension": args.dimension,
            "time_step": args.time_step,
            "trotter_steps": args.trotter_steps,
            "shots": args.shots,
            "noise_scale": args.noise_scale,
            "shrinkage_strength": args.shrinkage_strength,
        },
        "primary_metric": "absolute deviation from ideal finite-dimensional QK energy",
        "mean_shrinkage_weight": float(np.mean(shrinkage_weight_means)),
        "policies": {},
    }
    for policy in POLICIES:
        qk_values = [float(row[f"{policy}_qk_deviation"]) for row in rows]
        ground_values = [float(row[f"{policy}_ground_error"]) for row in rows]
        summary["policies"][policy] = {
            "qk_deviation": _finite_stats(qk_values),
            "ground_error": _finite_stats(ground_values),
            "full_rank_fraction": float(np.mean([
                int(row[f"{policy}_rank"]) == args.dimension for row in rows
            ])),
        }

    noisy_qk = np.asarray([float(row["noise_modewise_qk_deviation"]) for row in rows])
    debiased_qk = np.asarray([float(row["debiased_modewise_qk_deviation"]) for row in rows])
    shrinkage_qk = np.asarray([float(row["shrinkage_modewise_qk_deviation"]) for row in rows])
    adaptive_qk = np.asarray([float(row["adaptive_guarded_qk_deviation"]) for row in rows])
    summary["comparisons"] = {
        "debiased_vs_noisy_qk_deviation": _comparison(debiased_qk, noisy_qk),
        "shrinkage_vs_noisy_qk_deviation": _comparison(shrinkage_qk, noisy_qk),
        "shrinkage_vs_full_debias_qk_deviation": _comparison(shrinkage_qk, debiased_qk),
        "adaptive_vs_debiased_qk_deviation": _comparison(adaptive_qk, debiased_qk),
    }

    payload = {"summary": summary, "rows": rows}
    if args.json:
        print(json.dumps(payload, indent=2, allow_nan=True))
        return

    print(json.dumps(summary, indent=2, allow_nan=True))
    print("\nper-seed absolute deviation from ideal QK energy")
    print(
        f"{'seed':>5s} {'noisy':>12s} {'debiased':>12s} "
        f"{'shrinkage':>12s} {'adaptive':>12s}"
    )
    for row in rows:
        print(
            f"{int(row['seed']):5d} "
            f"{float(row['noise_modewise_qk_deviation']):12.5g} "
            f"{float(row['debiased_modewise_qk_deviation']):12.5g} "
            f"{float(row['shrinkage_modewise_qk_deviation']):12.5g} "
            f"{float(row['adaptive_guarded_qk_deviation']):12.5g}"
        )


if __name__ == "__main__":
    main()
