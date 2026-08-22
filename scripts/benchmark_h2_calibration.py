"""Benchmark calibration-aware Quantum Krylov with published H2 component data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from biasblaster.h2_calibration import (
    assess_h2_time_step,
    evaluate_h2_krylov,
    get_h2_calibration,
    prepare_h2_krylov,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--systems", default="H2-1,H2-2")
    parser.add_argument("--budgets", default="5000,20000,100000")
    parser.add_argument("--candidates", default="0.1,0.2,0.3,0.4,0.5,0.6")
    parser.add_argument("--baseline-time-step", type=float, default=0.2)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--qubits", type=int, default=2)
    parser.add_argument("--dimension", type=int, default=2)
    parser.add_argument("--trotter-steps", type=int, default=1)
    parser.add_argument("--minimum-shots", type=int, default=100)
    parser.add_argument("--max-condition-number", type=float, default=25.0)
    parser.add_argument("--shrinkage-strength", type=float, default=1.0)
    parser.add_argument("--json", action="store_true")
    return parser


def _policy_map(policies):
    return {policy.name: policy for policy in policies}


def _stats(values):
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return {
            "count": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "std": float("nan"),
        }
    return {
        "count": int(finite.size),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "std": float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0,
    }


def _relative_improvement(candidate, baseline):
    candidate = np.asarray(candidate, dtype=float)
    baseline = np.asarray(baseline, dtype=float)
    valid = np.isfinite(candidate) & np.isfinite(baseline) & (baseline > 0)
    if not np.any(valid):
        return {
            "paired_count": 0,
            "beats_fraction": float("nan"),
            "mean_relative_improvement": float("nan"),
            "median_relative_improvement": float("nan"),
        }
    gains = 1.0 - candidate[valid] / baseline[valid]
    return {
        "paired_count": int(np.count_nonzero(valid)),
        "beats_fraction": float(np.mean(candidate[valid] < baseline[valid])),
        "mean_relative_improvement": float(np.mean(gains)),
        "median_relative_improvement": float(np.median(gains)),
    }


def _select(assessments):
    feasible = [item for item in assessments if item.full_rank and item.robust_margin >= 0.0]
    if feasible:
        return min(feasible, key=lambda item: item.time_step)
    return max(
        assessments,
        key=lambda item: (item.retained_rank, item.robust_margin, -item.time_step),
    )


def main() -> None:
    args = _parser().parse_args()
    if args.seeds < 1:
        raise ValueError("--seeds must be positive")
    systems = [item.strip() for item in args.systems.split(",") if item.strip()]
    budgets = sorted({int(item) for item in args.budgets.split(",") if item.strip()})
    candidates = sorted({float(item) for item in args.candidates.split(",") if item.strip()})
    if args.baseline_time_step not in candidates:
        candidates.append(float(args.baseline_time_step))
        candidates.sort()

    payload = {
        "methodology": {
            "pulse_data_used": False,
            "gate_model": "RB infidelity converted to effective uniform Pauli probability",
            "spam_model": "published asymmetric combined SPAM as terminal readout confusion",
            "memory_error_used": False,
            "memory_reason": "requires H2-native scheduled depth/transport timing",
            "measurement_crosstalk_used": False,
            "measurement_crosstalk_reason": "estimators use final-only ancilla measurement",
            "basis_selection_uses_target_energy": False,
        },
        "systems": {},
    }

    for system in systems:
        profile = get_h2_calibration(system)
        prepared = {
            time_step: prepare_h2_krylov(
                profile,
                n_qubits=args.qubits,
                dimension=args.dimension,
                time_step=time_step,
                trotter_steps=args.trotter_steps,
            )
            for time_step in candidates
        }
        baseline = prepared[float(args.baseline_time_step)]
        system_payload = {
            "calibration": {
                "source": profile.source_url,
                "one_qubit_infidelity": profile.one_qubit_infidelity,
                "one_qubit_infidelity_sigma": profile.one_qubit_infidelity_sigma,
                "effective_p1": profile.p1,
                "effective_p1_sigma": profile.p1_sigma,
                "two_qubit_infidelity": profile.two_qubit_infidelity,
                "two_qubit_infidelity_sigma": profile.two_qubit_infidelity_sigma,
                "effective_p2": profile.p2,
                "effective_p2_sigma": profile.p2_sigma,
                "spam_0": profile.spam_0,
                "spam_0_sigma": profile.spam_0_sigma,
                "spam_1": profile.spam_1,
                "spam_1_sigma": profile.spam_1_sigma,
                "memory_error_per_depth1": profile.memory_error_per_depth1,
                "memory_error_sigma": profile.memory_error_sigma,
                "measurement_crosstalk": profile.measurement_crosstalk,
                "measurement_crosstalk_sigma": profile.measurement_crosstalk_sigma,
            },
            "first_order_validation": {
                str(time_step): {
                    "rmse": float(np.sqrt(np.mean(
                        (item.raw_combined.predicted - item.finite_values) ** 2
                    ))),
                    "max_error": float(np.max(np.abs(
                        item.raw_combined.predicted - item.finite_values
                    ))),
                }
                for time_step, item in prepared.items()
            },
            "budgets": {},
        }

        for budget in budgets:
            assessments = [
                assess_h2_time_step(
                    prepared[time_step],
                    budget,
                    minimum_shots=args.minimum_shots,
                    max_condition_number=args.max_condition_number,
                )
                for time_step in candidates
            ]
            selected_assessment = _select(assessments)
            selected = prepared[selected_assessment.time_step]

            rows = []
            for seed in range(args.seed_start, args.seed_start + args.seeds):
                fixed_map = _policy_map(evaluate_h2_krylov(
                    baseline,
                    budget,
                    seed,
                    minimum_shots=args.minimum_shots,
                    max_condition_number=args.max_condition_number,
                    shrinkage_strength=args.shrinkage_strength,
                ))
                selected_map = _policy_map(evaluate_h2_krylov(
                    selected,
                    budget,
                    seed,
                    minimum_shots=args.minimum_shots,
                    max_condition_number=args.max_condition_number,
                    shrinkage_strength=args.shrinkage_strength,
                ))
                row = {"seed": seed}
                for prefix, policy_map in (("fixed", fixed_map), ("selected", selected_map)):
                    for name in ("vanilla", "noise_aware", "full_debias", "shrinkage"):
                        policy = policy_map.get(name)
                        row[f"{prefix}_{name}_ground_error"] = (
                            float(policy.ground_energy_error) if policy is not None else float("nan")
                        )
                        row[f"{prefix}_{name}_rank"] = (
                            int(policy.retained_rank) if policy is not None else 0
                        )
                rows.append(row)

            def values(key):
                return [float(row[key]) for row in rows]

            fixed_vanilla = values("fixed_vanilla_ground_error")
            fixed_shrinkage = values("fixed_shrinkage_ground_error")
            selected_shrinkage = values("selected_shrinkage_ground_error")
            fixed_full = values("fixed_full_debias_ground_error")
            selected_full = values("selected_full_debias_ground_error")
            system_payload["budgets"][str(budget)] = {
                "selected_time_step": selected_assessment.time_step,
                "selection": [
                    {
                        "time_step": item.time_step,
                        "retained_rank": item.retained_rank,
                        "robust_margin": item.robust_margin,
                        "predicted_condition_number": item.predicted_condition_number,
                        "conditioning_floor": item.conditioning_floor,
                    }
                    for item in assessments
                ],
                "ideal": {
                    "fixed_qk_ground_error": abs(
                        baseline.plan.ideal_qk_energy - baseline.plan.exact_ground_energy
                    ),
                    "selected_qk_ground_error": abs(
                        selected.plan.ideal_qk_energy - selected.plan.exact_ground_energy
                    ),
                },
                "ground_error": {
                    "fixed_vanilla": _stats(fixed_vanilla),
                    "fixed_full_debias": _stats(fixed_full),
                    "fixed_shrinkage": _stats(fixed_shrinkage),
                    "selected_full_debias": _stats(selected_full),
                    "selected_shrinkage": _stats(selected_shrinkage),
                },
                "comparisons": {
                    "fixed_shrinkage_vs_fixed_vanilla": _relative_improvement(
                        fixed_shrinkage, fixed_vanilla
                    ),
                    "selected_shrinkage_vs_fixed_vanilla": _relative_improvement(
                        selected_shrinkage, fixed_vanilla
                    ),
                    "selected_shrinkage_vs_fixed_shrinkage": _relative_improvement(
                        selected_shrinkage, fixed_shrinkage
                    ),
                    "selected_full_debias_vs_fixed_full_debias": _relative_improvement(
                        selected_full, fixed_full
                    ),
                },
                "full_rank_fraction": {
                    key: float(np.mean([
                        row[f"{key}_rank"] == args.dimension for row in rows
                    ]))
                    for key in (
                        "fixed_vanilla", "fixed_full_debias", "fixed_shrinkage",
                        "selected_full_debias", "selected_shrinkage",
                    )
                },
            }
        payload["systems"][profile.system_name] = system_payload

    print(json.dumps(payload, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
