# Hardware-aware Quantum Krylov channel propagation

This extension generalizes BiasBlaster from coherent Hamiltonian-generator error to calibrated local quantum-channel error and projects that error directly onto Quantum Krylov observables.

## Core model

For an ideal circuit with Pauli-transfer representation `R_C`, attach a calibrated local error mode `eta_k Delta_k` to a circuit boundary. To first order,

\[
\Delta R_C \approx \sum_k R_{>k}\,\eta_k\Delta_k\,R_{<k}.
\]

For a measured observable `O` and initial state `rho`, the corresponding scalar sensitivity is

\[
J_{O,k}=\langle\!\langle O|R_{>k}\Delta_kR_{<k}|\rho\rangle\!\rangle.
\]

The implementation does not form a global `4^n x 4^n` superoperator. It propagates a sparse forward Pauli state and a sparse backward observable through local one- and two-qubit PTMs, then contracts the local perturbation at the error boundary.

`ChannelMode.mean` is the calibrated error coefficient. For a finite calibrated local channel `R_noise`, a simple first-order representation is

```python
ChannelMode(..., delta_ptm=channel_delta(R_noise), mean=1.0)
```

For a parameterized calibration model, use the actual derivative `dR/deta` as `delta_ptm` and the calibrated parameter as `mean`.

## Error families

The channel layer supports any supplied real one- or two-qubit PTM, plus helpers for:

- coherent unitary errors and Pauli-generator derivatives;
- stochastic Pauli and depolarizing channels;
- dephasing;
- amplitude damping / T1-type decay;
- generalized amplitude damping / finite-temperature relaxation;
- reset or preparation faults;
- effective computational-basis readout confusion;
- arbitrary one- or two-qubit Kraus channels, including correlated/crosstalk models;
- a trace-decreasing heralded erasure model for computational-subspace leakage survival.

Correlated or quasistatic error parameters are represented by non-diagonal `mode_covariance`. This is important: two gates can share a drift parameter or have spatially correlated calibration error without changing the propagation algorithm.

The heralded-erasure helper is not a qutrit leakage simulator. It is an effective computational-subspace survival model. Full leakage/seepage dynamics require an enlarged local operator basis and are intentionally outside the current Pauli-only implementation.

## Bias and covariance

For observable vector `x`, `estimate_observable_impacts` returns

\[
\hat b_x = J\bar\eta,
\qquad
\Sigma_x = J\Sigma_\eta J^T + \Sigma_{\rm shots}.
\]

This propagates both systematic calibration bias and uncertainty. The shot covariance can be supplied independently and is added at the observable level.

Sparse support capping is available for forward/backward propagation. `forward_dropped_l2` and `backward_dropped_l2` are diagnostics only; unlike BiasBlaster's existing coherent transport certificate, they are **not** presented as rigorous expectation-value error bounds.

## Quantum Krylov projection

`KrylovObservableSpec` maps measured real/imaginary scalar components into Hermitian projected Hamiltonian and overlap matrices. The resulting error model contains

\[
\Delta H,\quad \Delta S,
\quad \frac{\partial H}{\partial\eta_k},
\quad \frac{\partial S}{\partial\eta_k}.
\]

For a generalized eigenpair

\[
Hc=ESc,\qquad c^\dagger S c=1,
\]

the first-order energy sensitivity is

\[
\frac{\partial E}{\partial\eta_k}
= c^\dagger\left(
\frac{\partial H}{\partial\eta_k}
-E\frac{\partial S}{\partial\eta_k}
\right)c.
\]

`estimate_energy_impact` returns these mode sensitivities, predicted systematic energy bias, and propagated variance.

## Adaptations implemented

### Matrix debiasing

```python
H_corrected, S_corrected = debias_krylov_matrices(H_measured, S_measured, model)
```

subtracts the first-order predicted systematic bias.

### Noise-aware overlap filtering

`recommended_overlap_floor` combines the predicted overlap-matrix bias scale with a covariance-derived perturbation scale. Use the result as the cutoff passed to `solve_krylov_generalized_eigenproblem` so directions smaller than the hardware/measurement uncertainty are not blindly retained.

The covariance term uses an expected Frobenius perturbation scale and is a heuristic regularization scale, not a statistical confidence bound.

### Sensitivity-weighted shot allocation

`optimal_shot_allocation` uses

\[
N_i \propto \left|\frac{\partial E}{\partial x_i}\right|\sqrt{v_i},
\]

where `v_i` is the per-shot variance of measured scalar `x_i`. This minimizes first-order energy variance under a fixed independent-shot budget.

### Energy-targeted control update

If calibrated controls change error-mode coordinates according to

\[
\eta \mapsto \eta + C\delta,
\]

`optimize_energy_bias_controls` minimizes the first-order absolute energy bias under an L2 control budget. This is the Quantum-Krylov analogue of BiasBlaster's circuit-bias control objective: controls are spent according to predicted impact on the requested eigenvalue rather than raw local gate error.

## Recommended challenge experiment

Compare, at identical HQC/shot budget:

1. vanilla Quantum Krylov;
2. numerical overlap regularization only;
3. coherent-only BiasBlaster information;
4. full channel bias/covariance debiasing;
5. full channel model + noise-aware overlap floor + sensitivity-weighted shots;
6. optionally, the above plus energy-targeted control or compilation choices.

The primary plot should be absolute eigenenergy error versus HQC/shot cost. Secondary plots should report retained Krylov rank, predicted versus observed H/S bias, calibration-model residual, and the fraction of runs for which the adaptive policy improves over the baseline.

## Scope

The implementation is a first-order local-channel model. Arbitrary local Markovian errors and finite-range correlated channels fit naturally. Quasistatic/common-mode correlations fit through parameter covariance. Strong channel composition can be checked with `propagate_noisy_observables`, which applies finite PTMs sequentially.

General non-Markovian environment memory is not represented by independent boundary-local channels. Full qutrit leakage/seepage dynamics are also not yet represented; the current leakage helper is a heralded erasure/survival approximation.
