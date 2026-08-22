# Quantinuum Nexus validation path

The offline Quantum Krylov benchmark in `qkrylov_experiment.py` is a controlled regression environment. It must not be described as a reproduction of the full Helios emulator. Competition-facing evidence should use the official Quantinuum stack and preserve the same estimator ordering and resource accounting used by the offline ablations.

## Installation

```bash
python -m pip install -e '.[quantinuum]'
```

The optional extra installs Qiskit, pytket, the Qiskit/pytket conversion extension, and qnexus. Authentication and project permissions are handled by Nexus rather than BiasBlaster.

## Circuit path

`build_tfim_krylov_plan(...)` produces the same scalar Hadamard-test estimator circuits used by the offline benchmark. The Nexus adapter then performs:

```text
Krylov estimator plan
      |
      v
BiasBlaster CircuitOperation representation
      |
      v
Qiskit measured estimator
      |
      v
pytket Circuit
      |
      v
Nexus upload -> Quantinuum compile -> execute
      |
      v
P(0)-P(1) scalar estimates in original estimator order
      |
      v
assemble H/S -> noise-aware solve
```

The relevant entry points are:

```python
from biasblaster import (
    NexusExecutionConfig,
    execute_nexus_estimators,
    plan_to_pytket,
    upload_compile_plan,
)
```

The default `NexusExecutionConfig` targets `Helios-1E-lite`, the cloud-hosted Helios emulator route documented by Quantinuum for Nexus. Override `system_name` when challenge access exposes a different emulator or hardware target.

## Resource discipline

`execute_nexus_estimators(plan, shots, ...)` accepts one shot count per scalar estimator. This is important for the adaptive policy: its shot allocation can be executed without converting the allocation back into a uniform budget.

The adapter does not execute anything at import time and does not hide cloud cost. A challenge run should explicitly choose the system, shot vector, and any cost guard before submission.

The offline field `weighted_two_qubit_executions = sum_i shots_i * n_2q(i)` is only a local cost proxy. It is not an HQC estimate. Report actual HQCs/cost returned by the Quantinuum service for competition results.

## Recommended validation sequence

1. Build and freeze a QK estimator plan.
2. Run the noise-free or syntax-check emulator path to verify conventions.
3. Run the official Helios emulator at a fixed shot budget.
4. Compare observed scalar estimator shifts with BiasBlaster's first-order predictions.
5. Assemble measured `H` and `S` and record overlap eigenvalues/conditioning.
6. Compare uncorrected, propagated-bias-corrected, mode-regularized, and hardware-aware-basis policies at matched resource cost.
7. Repeat over several shot-noise seeds or independent jobs.
8. Only after freezing the policy, repeat on hardware if challenge access permits.

## Calibration-model boundary

The offline defaults currently map the published simple probabilities `p1`, `p2`, `p_meas`, and `p_init` into reduced local channels, with optional dephasing. The official Helios error model includes additional structure such as asymmetric errors, crosstalk, spontaneous emission, leakage/seepage, transport/idle effects, and gate-dependent behavior. Those mechanisms should be represented from the calibration information actually available to the challenge environment or validated empirically against official emulator outputs.

Full non-Markovian environment memory is outside the present local-channel model. Full leakage/seepage dynamics also require an enlarged local operator basis; BiasBlaster's existing heralded-erasure helper is an effective computational-subspace model, not a full qutrit simulator.

## Current Quantinuum references

- Nexus circuit upload: https://docs.quantinuum.com/nexus/api/api_circuits.html
- Nexus compile/execute APIs: https://docs.quantinuum.com/nexus/api/api_jobs.html
- Nexus backend configuration: https://docs.quantinuum.com/nexus/concepts/backends.html
- Helios emulators: https://docs.quantinuum.com/systems/user_guide/emulator_user_guide/emulators/helios_emulators.html
- Quantinuum emulator noise model: https://docs.quantinuum.com/systems/user_guide/emulator_user_guide/noise_model.html
