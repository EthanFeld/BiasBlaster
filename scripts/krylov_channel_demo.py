"""Small end-to-end demo of channel-aware Quantum Krylov error analysis."""

import numpy as np

from biasblaster import (
    ChannelMode,
    KrylovObservableSpec,
    ObservableImpactBatch,
    amplitude_damping_ptm,
    build_krylov_error_model,
    channel_delta,
    dephasing_ptm,
    estimate_energy_impact,
    estimate_observable_impacts,
    optimal_shot_allocation,
    recommended_overlap_floor,
)


def main() -> None:
    # A zero-depth one-qubit example keeps the mechanics visible. The measured
    # scalars are then mapped into a 1x1 Krylov H/S generalized eigenproblem.
    initial_state = {"I": 1.0, "Z": -1.0}  # |1>
    observables = {
        "h00": {"Z": 1.0},
        "s00": {"I": 1.0},
    }

    phase_derivative = channel_delta(dephasing_ptm(1e-4)) / 1e-4
    damping_derivative = channel_delta(amplitude_damping_ptm(1e-4)) / 1e-4
    modes = [
        ChannelMode("quasistatic_z", 0, (0,), phase_derivative, mean=2e-3),
        ChannelMode("t1_decay", 0, (0,), damping_derivative, mean=1e-3),
    ]
    mode_covariance = np.array([
        [2e-7, 5e-8],
        [5e-8, 1e-7],
    ])
    shot_covariance = np.diag([2e-4, 1e-6])

    impacts = estimate_observable_impacts(
        [],
        initial_state,
        observables,
        modes,
        1,
        mode_covariance=mode_covariance,
        shot_covariance=shot_covariance,
    )

    specs = [
        KrylovObservableSpec("h00", "H", 0, 0),
        KrylovObservableSpec("s00", "S", 0, 0),
    ]
    model = build_krylov_error_model(impacts, specs, dimension=1)

    hamiltonian = np.array([[-1.0]])
    overlap = np.array([[1.0]])
    energy = estimate_energy_impact(hamiltonian, overlap, model)
    floor = recommended_overlap_floor(model)
    allocation = optimal_shot_allocation(
        energy.observable_sensitivities,
        per_shot_variances=[1.0, 0.01],
        total_shots=10_000,
        minimum_shots=100,
    )

    print("observable bias:", dict(zip(impacts.names, impacts.bias)))
    print("predicted energy bias:", energy.predicted_bias)
    print("predicted energy standard deviation:", energy.standard_deviation)
    print("recommended overlap floor:", floor)
    print("shot allocation:", dict(zip(impacts.names, allocation)))


if __name__ == "__main__":
    main()
