import numpy as np
import pytest

pytest.importorskip("qiskit")

from biasblaster.qkrylov_adaptive import (
    guarded_shot_allocation,
    run_tfim_qkrylov_adaptive_benchmark,
)
from biasblaster.qkrylov_experiment import EffectiveNoiseParameters


def test_guarded_allocation_reserves_uniform_exploration_and_caps_concentration():
    allocation = guarded_shot_allocation(
        energy_sensitivities=np.array([1000.0, 1.0, 0.1, 0.0]),
        overlap_scores=np.array([0.0, 1.0, 10.0, 1.0]),
        per_shot_variances=np.ones(4),
        total_shots=10_000,
        uniform_fraction=0.5,
        overlap_weight=0.5,
        max_weight_ratio=4.0,
    )
    assert allocation.sum() == 10_000
    # Half the budget is guaranteed uniform before targeted allocation.
    assert np.min(allocation) >= 1250
    assert np.max(allocation) / np.min(allocation) < 4.0


def test_adaptive_policy_reuses_pilot_within_fixed_total_budget():
    total_shots = 20_000
    result = run_tfim_qkrylov_adaptive_benchmark(
        n_qubits=2,
        dimension=2,
        time_step=0.2,
        trotter_steps=1,
        total_shots=total_shots,
        minimum_shots=100,
        pilot_fraction=0.2,
        seed=11,
        noise=EffectiveNoiseParameters(),
    )
    assert result.uniform_allocation.sum() == total_shots
    assert result.adaptive_allocation.sum() == total_shots
    assert np.all(result.adaptive_allocation >= 100)
    names = {policy.name for policy in result.policies}
    assert "ideal_qk" in names
    assert "debiased_modewise" in names
    assert "adaptive_guarded" in names
    for policy in result.policies:
        if policy.name != "ideal_qk":
            assert policy.shots == total_shots
