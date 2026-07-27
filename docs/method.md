# Method

BiasBlaster represents each ideal circuit operation as a bounded local unitary and each local coherent error as three generator-mode coefficients by default. The optional full-SU(4) two-qubit basis has 15 coefficients. Qubit 0 is the rightmost Pauli factor.

Circuit-operation matrices use Qiskit's qarg convention: the first qarg is
the least-significant local tensor factor. Local generator strings are instead
listed in qarg order; the transport converts between these conventions.

## Pauli propagation

For an operation (U_g), the local Pauli-transfer matrix is

\[
T_g[P,Q] = 2^{-k}\operatorname{Tr}(P U_g Q U_g^\dagger),
\]

where (k) is the operation arity. A sparse transport column starts with one local generator mode and applies the exact local transfer matrix through its own operation and every later operation. The end bias is the contraction

\[
b = \sum_g M_g \theta_g.
\]

Only retained end-Pauli labels and their source-mode vectors are stored, so the normal representation does not allocate a (4^n) output array. A nonzero `coefficient_tol` is bounded pruning, not an exact calculation: its dropped coefficient map is included in `truncation_bound` just like support-cap omissions.

The default two-qubit generator basis is `XX`, `YY`, `ZZ`. Pass `two_qubit_generator_basis="full-su4"` to anchored Pauli or nearest-Clifford transport for all 15 nonidentity two-qubit Paulis. Full-SU(4) inputs use `(gate_count, 15)` generator/Jacobian arrays; unused slots for one-qubit gates must be zero. The benchmark uses this full basis. Full-SU(4) problems normally exceed the FPGA adapter's five-row payload and therefore use the exact CPU fallback.

## Nearest-Clifford replacement

The nearest estimator replaces each one- or two-qubit ideal unitary with the same-arity Clifford maximizing phase-invariant process fidelity. Exact Clifford inputs take the fast tableau path. Ties are deterministic because candidate tableaux are canonically serialized before selection. The resulting transport stores one signed final Pauli label per gate and mode.

This replacement has no circuit-level approximation certificate. In particular,
`SparseCliffordTransport.truncation_bound == 0` means that this representation
does not prune Pauli terms; it does **not** bound error from replacing gates by
Cliffords. The benchmark reports mean and minimum *local* nearest-Clifford
process fidelity as diagnostics, and labels a circuit-level approximation bound
as unavailable.

## Truncation bound

When a support cap is reached, the largest coefficient vectors by Euclidean norm are retained. At each pruning event, the Frobenius norm of the dropped coefficient map is added to `truncation_bound`. Future ideal Pauli transfer is orthogonal, so that event norm cannot increase; triangle inequality across pruning events preserves certification. For a local-generator input (\theta), the reported omitted-bias bound is

\[
\|\Delta b\|_2 \leq \texttt{truncation\_bound}\,\|\theta\|_2.
\]

The capped result is therefore an explicitly bounded estimate, never an exact result.

## Anchored Pauli corrections

`estimate_hybrid_pauli_transport` splits each operation's Pauli-transfer map
as \(T_U=T_C+(T_U-T_C)\), using its nearest Clifford \(C\). The Clifford
backbone is always retained; only correction rows consume `support_cap`.
Therefore correction support 0 is exactly nearest-Clifford transport, while
larger caps provide more exact-correction capacity toward full Pauli transport.

Its `truncation_bound` is the standard exact-transport bound for omitted
correction rows. At correction cap 0, the result is exactly nearest-Clifford
transport but its correction map is not built, so this bound is reported as
unavailable rather than zero. No guessed dropped branch is treated as an exact
result.

When local generator and calibrated Jacobian estimates are available, a
`selection_matrix` may rank retained rows by predicted bias and allowed-control
response instead of raw row norm. This uses the same local model supplied to
the optimizer; it does not inspect simulator fidelity after optimization.

The benchmark exposes this estimator as `anchored-pauli`. Its `support_cap`
counts exact correction rows; cap 0 is reported alongside the separately named
nearest-Clifford diagnostic and must produce the same transport. The old
exclusive-Pauli benchmark path is intentionally absent: it did not share this
nearest-Clifford null case and was not a fair support trade-off.

## Optimizer and effort budget

Calibrated local Jacobians (J_{g,m,c}) describe the first-order change in generator mode (m) at gate (g) for control direction (c). The optimizer solves one trust-region least-squares problem for either transport representation:

\[
\min_\delta \|b + A\delta\|_2 \quad \text{subject to} \quad \|\delta\|_2 \leq f\|\theta\|_2,
\]

where (f) is `effort_fraction`. It returns controls, updated local generators, predicted norms, and the enforced budget. This optimizer-level `truncation_bound` is evaluated on the *linearized* updated generators. Benchmark reporting recalibrates the updated simulator pulse and evaluates the transport bound on those actual post-update local coordinates; it reports unavailable when no exact-correction certificate exists.

The CPU implementation uses an SVD-based trust-region solve. The current
Tang Nano FPGA image exposes only a fixed-point damped Gauss--Newton datapath,
limited to five residual rows and six flattened controls; it cannot execute
this CPU solve exactly. The FPGA adapter therefore computes the authoritative
CPU target on the host and uses the board only to apply and verify its Q12
quantization. An exact FPGA implementation would require new RTL/firmware.

The FPGA adapter also exposes a greedy gate path. It scores each gate by its
predicted full-residual decrease under a gate-local L2 budget. For a selected
gate, host compresses all end-bias rows into its local normal equations
`J_g.T @ J_g` and `J_g.T @ b`; resulting eigen rows fit the five-row FPGA
record exactly when local rank is at most five. Board computes a fixed-point
update, host radially projects it into L2 ball, accepts only when full
transported residual falls. Gates re-rank, revisit for up to two sweeps by
default; callers can request more when extra refinement is worth UART time.
This remains coordinate-style heuristic, not global SVD solution.
The FPGA wire image uses a 1,000,000 baud UART (27 clocks/bit) and supports
opcode `0x10` for batches of up to four reply-independent gate records. Greedy
selection remains one gate at a time because its residual rescore depends on
each returned update.

`optimize_local_error_controls` is a no-transport oracle baseline. It uses the same Jacobians and budget but solves (\min_\delta \|\theta + J\delta\|_2), directly against supplied local error coefficients.

## Pulse-derived local coordinates

The benchmark does not use hand-written "calibrated" Jacobians. Each local
operation uses a one-slice, closed-system, piecewise-constant bilinear Hamiltonian
(H(u)=H_\mathrm{target}+\sum_m u_mH_m). The target drift realizes the ideal
unitary up to global phase; fixed local Pauli control Hamiltonians perturb it.
For the simulated unitary (U(u)) and target (U_t), it defines the pre-gate
error (E(u)\sim U_t^\dagger U(u)=\exp(-iH_\mathrm{err}(u)), up to global phase). Exact Frechet
derivatives are converted through the inverse principal-log differential to
derive the local Jacobian. The benchmark projects two-qubit errors onto the
complete 15-element traceless Pauli basis, so its projection residual is only
numerical roundoff.

After optimization, benchmark fidelity is recomputed from the updated pulse
Hamiltonians, not from (\theta + J\delta).  Thus the linear model is used only
for optimization; reported fidelity includes finite, nonlinear pulse evolution.
This is an analytic simulator oracle: the benchmark creates the pulse model,
its seeded error, and its exact Frechet derivatives in the same process. It is
therefore a self-consistency test of the estimator and optimizer under their
assumed model, not evidence about calibration-estimation sample cost,
model-mismatch robustness, or device performance. This is a platform-agnostic
closed-system bilinear pulse model, not a device-calibrated waveform model.

Reported fidelity is the mean overlap on a finite shared Haar-random probe
ensemble, not an exact average gate/process fidelity. The benchmark prints the
probe count, across-case standard error of the paired fidelity gain, and the
fraction of non-improving cases; scientific comparisons should retain
per-instance data and increase probe and seed counts until those uncertainty
estimates are adequate. Pass `--output results.json` to write every case's
fixture, estimator configuration, timings, support, bound, and fidelity data.

Benchmark transport timing is reported as a cold build: its immutable gate-action
cache is cleared before each timed estimator construction. Cached rescore time is
not presented as build time.

`mean_transport_payload_bytes` counts stored numeric/string payload only. It is
not a measurement of Python process memory, which also includes containers,
strings, and allocator overhead.

## Scope

These are algebraic, first-order coherent-model estimates. They are not claims about hardware fidelity. The optional pulse simulator is a closed-system ideal-Hamiltonian check and excludes leakage, decoherence, transfer-function distortion, calibration drift, and hardware validation.
