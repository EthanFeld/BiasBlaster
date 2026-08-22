# Quantum Krylov benchmark harness

`biasblaster.qkrylov_experiment` turns the channel-propagation primitives into an executable Quantum Krylov ablation.

## What is measured

For time-evolved basis states

\[
|\phi_j\rangle = U(t_j)|\phi_0\rangle,
\]

the benchmark constructs Hadamard-test circuits for the overlap matrix

\[
S_{ij}=\langle\phi_i|\phi_j\rangle
\]

and for every Pauli term `P_a` in the Hamiltonian

\[
H=\sum_a h_a P_a,
\qquad
H_{ij}=\sum_a h_a\langle\phi_i|P_a|\phi_j\rangle.
\]

Each real or imaginary component is therefore tied to an actual compiled estimator circuit. The estimator circuits are compiled to the local `CircuitOperation` representation before channel sensitivity is propagated.

The built-in problem is an open-chain transverse/longitudinal-field Ising model. Time evolution is implemented with a first-order product formula over X, Z and ZZ terms. This is intentionally small enough for exact reference calculations while still generating nontrivial H/S conditioning and two-qubit circuits.

## Ablations

At one fixed total shot budget the CLI reports:

1. `ideal_qk` — noiseless projected problem;
2. `noisy_uniform` — finite local channels + uniform shots;
3. `noise_regularized` — same data with the error-derived overlap floor;
4. `debiased_regularized` — subtract first-order H/S bias and use the error-derived floor;
5. `adaptive_full` — debiasing + regularization + energy-sensitivity-weighted shot allocation.

The main reported quantities are absolute ground-energy error, deviation from the ideal QK solution, retained overlap rank, total shots and a simple `two_qubit_gates * shots` execution-cost proxy.

Run:

```bash
python -m pip install -e '.[qkrylov]'
python scripts/benchmark_qkrylov.py --qubits 2 --dimension 3 --shots 100000
```

Use `--json` for machine-readable output.

## First-order validation

The benchmark computes each estimator twice:

- first-order prediction from the propagated local channel Jacobian;
- sequential finite-channel PTM propagation.

It reports the RMSE and maximum difference between the two. This is important because a bias correction should not be trusted once the first-order approximation is outside its useful regime.

## Offline Helios-like model

`EffectiveNoiseParameters` defaults to the currently published Helios-1E values for the four simple probabilities that map directly into this reduced computational-space model:

- `p1 = 2.5e-5`;
- `p2 = 8e-4`;
- `p_meas = 1e-6`;
- `p_init = 5e-4`.

Source: Quantinuum Helios emulator documentation:
https://docs.quantinuum.com/systems/user_guide/emulator_user_guide/emulators/helios_emulators.html

This reduced model is **not the Helios emulator**. In particular it does not reproduce asymmetric Pauli weights, spontaneous emission, leakage/seepage, crosstalk, transport/idle timing, coherent quadratic dephasing or the angle-dependent RZZ fault model. Quantinuum documents those mechanisms separately:
https://docs.quantinuum.com/systems/user_guide/emulator_user_guide/noise_model.html

The reduced model exists for fast regression tests and controlled ablations only. Any competition claim should be validated with the official emulator or hardware.

## Hardware/emulator validation sequence

For a Grand Challenge run, use the following sequence:

1. Generate the exact same estimator circuits and baseline shot allocation.
2. Compile them through the Quantinuum stack rather than treating the local Qiskit basis as hardware-native.
3. Execute noise-free emulator runs to validate circuit convention and H/S assembly.
4. Execute the default Helios emulator to obtain the observed matrix-element bias and variance.
5. Build the calibrated channel model from the available emulator/device parameters and compare predicted versus observed estimator bias.
6. Run the five ablations at matched HQC/shot budget.
7. Repeat over noise scaling and Krylov dimension.
8. If hardware access is granted, freeze the policy before the hardware run and report the same metrics.

The primary result should be energy error versus HQC (or matched shot cost), with predicted-versus-observed H/S bias as the model-validation figure.

## Scope and interpretation

The benchmark currently treats measurements as independent +/-1 estimators. Shared-shot covariance from commuting-group measurements can be supplied to the underlying channel/Krylov APIs, but is not generated automatically by this harness.

The current leakage helper in BiasBlaster remains an effective computational-subspace erasure model. Full Helios leakage/seepage behavior needs an enlarged local basis or direct emulator data and should not be described as exact Pauli propagation.
