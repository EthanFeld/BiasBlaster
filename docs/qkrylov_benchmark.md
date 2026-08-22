# Quantum Krylov benchmark harness

`biasblaster.qkrylov_experiment` turns the channel-propagation primitives into an executable Quantum Krylov ablation. The purpose is not merely to subtract an output bias: the propagated error model is used to decide which Krylov directions are trustworthy, how strongly predicted bias should be corrected, which time grid should be used, and where a measurement budget should be spent.

## What is measured

For time-evolved basis states

\[
|\phi_j\rangle = U(t_j)|\phi_0\rangle,
\]

the benchmark constructs Hadamard-test circuits for

\[
S_{ij}=\langle\phi_i|\phi_j\rangle
\]

and, for every Pauli term `P_a` in

\[
H=\sum_a h_aP_a,
\qquad
H_{ij}=\sum_a h_a\langle\phi_i|P_a|\phi_j\rangle.
\]

Every real/imaginary scalar estimator is tied to an actual compiled circuit. The circuits are compiled to BiasBlaster's local `CircuitOperation` representation before channel sensitivity is propagated, so different H/S matrix elements can have different hardware-error sensitivities while still sharing correlated calibration parameters.

The built-in regression problem is an open-chain transverse/longitudinal-field Ising model. Time evolution uses a first-order product formula over X, Z and ZZ terms. It is intentionally small enough for exact references while still producing nontrivial overlap conditioning and two-qubit estimator circuits.

## Current policies

The regression suite separates several effects instead of bundling them into one opaque mitigation result:

- `ideal_qk`: noiseless finite-dimensional projected problem.
- `noise_modewise`: measured H/S with overlap directions retained only when their measured overlap eigenvalue exceeds the propagated uncertainty scale.
- `debiased_modewise`: full first-order propagated-bias subtraction followed by the same overlap test.
- `shrinkage_modewise`: uncertainty-weighted bias subtraction. For scalar estimator bias `b_i` and variance `sigma_i^2`, the default correction weight is

  \[
  \alpha_i=\frac{b_i^2}{b_i^2+\sigma_i^2},
  \]

  so poorly resolved corrections are automatically reduced rather than blindly subtracted.
- `adaptive_guarded`: a paid/reused pilot stage estimates energy and overlap sensitivities, then reallocates only when the pilot subspace is sufficiently resolved. Otherwise the remaining budget stays uniform. This policy is retained as an ablation because it is not yet consistently superior in the small regression benchmark.

`krylov_regularization.py` can also impose an explicit maximum overlap condition number. This is a numerical-stability guard, not a claim that one universal condition-number cutoff is optimal.

## Hardware-aware basis selection

`qkrylov_basis.py` evaluates candidate time spacings using the *predicted noisy overlap matrix*, propagated calibration covariance, planned shot covariance, retained overlap rank, and numerical conditioning. It never uses the exact ground-state energy to choose a candidate.

The default selection policy chooses the **smallest candidate time spacing whose full requested Krylov rank is predicted to be resolvable**. This implements the intended tradeoff:

- spacing that is too small creates nearly dependent Krylov states and amplifies H/S noise;
- spacing that is unnecessarily large can worsen the target finite-dimensional Krylov approximation;
- the hardware model determines the minimum separation that clears the uncertainty/conditioning threshold.

Run the selector and a fixed-budget realization with:

```bash
python scripts/benchmark_qkrylov_basis.py \
  --candidates 0.2,0.4,0.6 \
  --qubits 2 --dimension 2 --shots 20000 --json
```

Use `benchmark_qkrylov_basis_sweep.py` to compare the selected grid against a fixed baseline over repeated shot-noise seeds. Cross-basis comparisons should use error relative to the exact problem reference when available; deviation from each grid's own noiseless finite-dimensional QK energy is not a fair cross-basis metric.

## Fixed-budget execution and resource accounting

The primary offline budget is the total number of scalar-estimator shots. The benchmark also reports

```text
sum_i shots_i * two_qubit_gate_count_i
```

as a local execution-complexity proxy. This quantity is **not an HQC estimate**. Competition results should report actual Quantinuum resource/cost information from the service at matched experimental conditions.

Basic run:

```bash
python -m pip install -e '.[qkrylov]'
python scripts/benchmark_qkrylov.py \
  --qubits 2 --dimension 2 --time-step 0.2 \
  --shots 20000 --json
```

Repeated correction-policy study:

```bash
python scripts/benchmark_qkrylov_sweep.py \
  --seeds 10 --qubits 2 --dimension 2 \
  --time-step 0.2 --shots 20000 --json
```

## First-order validation

Each estimator is evaluated in two ways:

1. first-order prediction from the propagated local-channel Jacobian;
2. sequential finite-channel PTM propagation.

The benchmark reports the RMSE and maximum difference. This is a required model-validity diagnostic: predicted-bias correction should not be interpreted as reliable once the linear channel approximation is outside its useful regime.

## Offline Helios-like regression model

`EffectiveNoiseParameters` defaults to the published Helios-1E values for four simple probabilities that map directly into the reduced computational-space model:

- `p1 = 2.5e-5`;
- `p2 = 8e-4`;
- `p_meas = 1e-6`;
- `p_init = 5e-4`.

Source: Quantinuum Helios emulator documentation:
https://docs.quantinuum.com/systems/user_guide/emulator_user_guide/emulators/helios_emulators.html

This reduced model is **not the Helios emulator**. It does not reproduce the full asymmetric fault model, crosstalk, spontaneous emission, leakage/seepage, transport/idle effects, coherent dephasing structure, or angle-dependent RZZ behavior documented by Quantinuum:
https://docs.quantinuum.com/systems/user_guide/emulator_user_guide/noise_model.html

The reduced model exists for deterministic regression tests and controlled ablations only. Offline performance numbers are not hardware-performance claims.

## Quantinuum Nexus path

`quantinuum_nexus.py` converts the same estimator plan to pytket circuits and follows the Nexus path:

```text
BiasBlaster estimator plan
  -> Qiskit measured circuit
  -> pytket Circuit
  -> Nexus upload
  -> Quantinuum compile
  -> Helios emulator / hardware execute
  -> P(0)-P(1) scalar estimator vector
  -> H/S assembly and error-aware solver
```

Install the optional stack with:

```bash
python -m pip install -e '.[quantinuum]'
```

The default execution config uses `Helios-1E-lite`; challenge access can override the system name. See `docs/quantinuum_nexus.md` for the explicit validation sequence and scope boundaries.

## Recommended competition experiment

At matched service cost / HQC, compare a frozen sequence of policies rather than tuning after seeing hardware outcomes:

1. fixed-grid raw QK;
2. mode-wise overlap regularization;
3. full propagated-bias correction;
4. uncertainty-shrinkage correction;
5. model-selected Krylov basis + shrinkage;
6. guarded adaptive shot allocation only if the pilot criterion declares the subspace reliable.

The primary figure should be energy error versus actual Quantinuum resource cost. Supporting figures should include predicted-versus-observed scalar/H/S bias, overlap spectra and retained rank, first-order model residual, correction shrinkage weights, and the fraction of repeated jobs on which each policy improves over the frozen baseline.

## Scope and interpretation

Measurements are currently treated as independent +/-1 scalar estimators in the built-in harness. Shared-shot covariance from grouped measurements can already be supplied to the underlying channel/Krylov APIs, but this harness does not generate grouping covariance automatically.

The current leakage helper remains an effective computational-subspace erasure/survival model. Full Helios leakage/seepage dynamics require an enlarged local operator basis or direct official-emulator data and should not be described as exact Pauli propagation.

General non-Markovian process memory is outside the present boundary-local channel model. Quasistatic/common-mode and finite-range correlated calibrated parameters are supported through the mode covariance matrix.
