"""Evaluate QK mitigation policies over repeated independent shot-noise seeds."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from biasblaster import EffectiveNoiseParameters
from biasblaster.qkrylov_adaptive import run_tfim_qkrylov_adaptive_benchmark


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


def main() -> None:
    args = _parser().parse_args()
    if args.seeds < 1:
        raise ValueError("--seeds must be positive")
    noise = EffectiveNoiseParameters(scale=args.noise_scale)
    rows: list[dict[str, object]] = []
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
        policy_map = {policy.name: policy for policy in result.policies}
        rows.append({
            "seed": seed,
            "ideal_qk_error": result.policies[0].ground_energy_error,
            **{
                f"{name}_error": (
                    policy_map[name].ground_energy_error if name in policy_map else float("nan")
                )
                for name in ("noise_modewise", "debiased_modewise", "adaptive_guarded")
            },
            **{
                f"{name}_rank": (
                    policy_map[name].retained_rank if name in policy_map else 0
                )
                for name in ("noise_modewise", "debiased_modewise", "adaptive_guarded")
            },
        })

    policies = ("noise_modewise", "debiased_modewise", "adaptive_guarded")
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
        },
        "policies": {},
    }
    for policy in policies:
        values = [float(row[f"{policy}_error"]) for row in rows]
        stats = _finite_stats(values)
        stats["full_rank_fraction"] = float(np.mean([
            int(row[f"{policy}_rank"]) == args.dimension for row in rows
        ]))
        summary["policies"][policy] = stats

    raw = np.asarray([float(row["noise_modewise_error"]) for row in rows])
    debiased = np.asarray([float(row["debiased_modewise_error"]) for row in rows])
    adaptive = np.asarray([float(row["adaptive_guarded_error"]) for row in rows])
    valid_debiased = np.isfinite(raw) & np.isfinite(debiased)
    valid_adaptive = np.isfinite(debiased) & np.isfinite(adaptive)
    summary["comparisons"] = {
        "debiased_beats_noisy_fraction": float(np.mean(debiased[valid_debiased] < raw[valid_debiased])) if np.any(valid_debiased) else float("nan"),
        "debiased_mean_relative_error_change": float(np.mean((debiased[valid_debiased] - raw[valid_debiased]) / raw[valid_debiased])) if np.any(valid_debiased) else float("nan"),
        "adaptive_beats_debiased_fraction": float(np.mean(adaptive[valid_adaptive] < debiased[valid_adaptive])) if np.any(valid_adaptive) else float("nan"),
        "adaptive_mean_relative_error_change": float(np.mean((adaptive[valid_adaptive] - debiased[valid_adaptive]) / debiased[valid_adaptive])) if np.any(valid_adaptive) else float("nan"),
    }
    payload = {"summary": summary, "rows": rows}
    if args.json:
        print(json.dumps(payload, indent=2, allow_nan=True))
        return

    print(json.dumps(summary, indent=2, allow_nan=True))
    print("\nper-seed errors")
    print(f"{'seed':>5s} {'noisy':>12s} {'debiased':>12s} {'adaptive':>12s}")
    for row in rows:
        print(
            f"{int(row['seed']):5d} "
            f"{float(row['noise_modewise_error']):12.5g} "
            f"{float(row['debiased_modewise_error']):12.5g} "
            f"{float(row['adaptive_guarded_error']):12.5g}"
        )


if __name__ == "__main__":
    main()
