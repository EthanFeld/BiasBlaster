import numpy as np
import pytest

pytest.importorskip("qiskit")

from biasblaster.qkrylov_adaptive import run_tfim_qkrylov_adaptive_benchmark
from biasblaster.qkrylov_experiment import EffectiveNoiseParameters


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
    assert "adaptive_modewise" in names
    for policy in result.policies:
        if policy.name != "ideal_qk":
            assert policy.shots == total_shots
