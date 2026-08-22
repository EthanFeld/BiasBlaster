"""Run the channel-aware Quantum Krylov ablation benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from biasblaster import EffectiveNoiseParameters
from biasblaster.qkrylov_adaptive import run_tfim_qkrylov_adaptive_benchmark


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qubits", type=int, default=2)
    parser.add_argument("--dimension", type=int, default=3)
    parser.add_argument("--time-step", type=float, default=0.35)
    parser.add_argument("--trotter-steps", type=int, default=2)
    parser.add_argument("--shots", type=int, default=100_000)
    parser.add_argument("--minimum-shots", type=int, default=100)
    parser.add_argument("--pilot-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--noise-scale", type=float, default=1.0)
    parser.add_argument("--p1", type=float, default=2.5e-5)
    parser.add_argument("--p2", type=float, default=8.0e-4)
    parser.add_argument("--p-meas", type=float, default=1.0e-6)
    parser.add_argument("--p-init", type=float, default=5.0e-4)
    parser.add_argument("--p-dephase-1q", type=float, default=0.0)
    parser.add_argument("--p-dephase-2q", type=float, default=0.0)
    parser.add_argument("--calibration-relative-sigma", type=float, default=0.10)
    parser.add_argument("--overlap-safety-factor", type=float, default=1.0)
    parser.add_argument("--support-cap", type=int, default=None)
    parser.add_argument("--json", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    noise = EffectiveNoiseParameters(
        p1=args.p1,
        p2=args.p2,
        p_meas=args.p_meas,
        p_init=args.p_init,
        p_dephase_1q=args.p_dephase_1q,
        p_dephase_2q=args.p_dephase_2q,
        scale=args.noise_scale,
    )
    result = run_tfim_qkrylov_adaptive_benchmark(
        n_qubits=args.qubits,
        dimension=args.dimension,
        time_step=args.time_step,
        trotter_steps=args.trotter_steps,
        total_shots=args.shots,
        minimum_shots=args.minimum_shots,
        pilot_fraction=args.pilot_fraction,
        seed=args.seed,
        noise=noise,
        calibration_relative_sigma=args.calibration_relative_sigma,
        overlap_safety_factor=args.overlap_safety_factor,
        support_cap=args.support_cap,
    )
    payload = {
        "n_data_qubits": result.plan.n_data_qubits,
        "dimension": result.plan.dimension,
        "estimators": len(result.plan.estimators),
        "ideal_qk_energy": result.plan.ideal_qk_energy,
        "exact_ground_energy": result.plan.exact_ground_energy,
        "first_order_observable_rmse": result.observable_first_order_rmse,
        "first_order_observable_max_error": result.observable_first_order_max_error,
        "mode_names": result.mode_names,
        "policies": [
            {
                "name": policy.name,
                "energy": policy.energy,
                "ground_energy_error": policy.ground_energy_error,
                "deviation_from_ideal_qk": policy.deviation_from_ideal_qk,
                "retained_rank": policy.retained_rank,
                "overlap_floor": policy.overlap_floor,
                "shots": policy.shots,
                "weighted_two_qubit_executions": policy.weighted_two_qubit_executions,
            }
            for policy in result.policies
        ],
    }
    if args.json:
        print(json.dumps(payload, indent=2, allow_nan=True))
        return

    print(
        f"TFIM QK benchmark: {payload['n_data_qubits']} data qubits, "
        f"K={payload['dimension']}, {payload['estimators']} scalar estimators"
    )
    print(f"exact ground energy: {payload['exact_ground_energy']:.10f}")
    print(f"ideal QK energy:      {payload['ideal_qk_energy']:.10f}")
    print(
        "first-order observable residual: "
        f"RMSE={payload['first_order_observable_rmse']:.3e}, "
        f"max={payload['first_order_observable_max_error']:.3e}"
    )
    print()
    header = (
        f"{'policy':24s} {'energy':>14s} {'|E-E0|':>14s} "
        f"{'|E-EQK|':>14s} {'rank':>6s} {'shots':>10s} {'2Qxshots':>12s}"
    )
    print(header)
    print("-" * len(header))
    for policy in result.policies:
        print(
            f"{policy.name:24s} {policy.energy:14.8f} "
            f"{policy.ground_energy_error:14.6e} "
            f"{policy.deviation_from_ideal_qk:14.6e} "
            f"{policy.retained_rank:6d} {policy.shots:10d} "
            f"{policy.weighted_two_qubit_executions:12d}"
        )


if __name__ == "__main__":
    main()
