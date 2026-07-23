# Method

BiasBlaster represents each ideal circuit operation as a bounded local unitary and each local coherent error as three generator-mode coefficients. Qubit 0 is the rightmost Pauli factor.

## Pauli propagation

For an operation (U_g), the local Pauli-transfer matrix is

\[
T_g[P,Q] = 2^{-k}\operatorname{Tr}(P U_g Q U_g^\dagger),
\]

where (k) is the operation arity. A sparse transport column starts with one local generator mode and applies the exact local transfer matrix through every later operation. The end bias is the contraction

\[
b = \sum_g M_g \theta_g.
\]

Only retained end-Pauli labels and their source-mode vectors are stored, so the normal representation does not allocate a (4^n) output array.

## Nearest-Clifford replacement

The nearest estimator replaces each one- or two-qubit ideal unitary with the same-arity Clifford maximizing phase-invariant process fidelity. Exact Clifford inputs take the fast tableau path. Ties are deterministic because candidate tableaux are canonically serialized before selection. The resulting transport stores one signed final Pauli label per gate and mode.

## Truncation bound

When a support cap is reached, the largest coefficient vectors by Euclidean norm are retained. The norms of omitted vectors are summed into `truncation_bound`. For a local-generator input (	heta), the reported omitted-bias bound is

\[
\|\Delta b\|_2 \leq \texttt{truncation\_bound}\,\|\theta\|_2.
\]

The capped result is therefore an explicitly bounded estimate, never an exact result.

## Optimizer and effort budget

Calibrated local Jacobians (J_{g,m,c}) describe the first-order change in generator mode (m) at gate (g) for control direction (c). The optimizer solves one trust-region least-squares problem for either transport representation:

\[
\min_\delta \|b + A\delta\|_2 \quad \text{subject to} \quad \|\delta\|_2 \leq f\|\theta\|_2,
\]

where (f) is `effort_fraction`. It returns controls, updated local generators, predicted norms, and the enforced budget.

## Scope

These are algebraic, first-order coherent-model estimates. They are not claims about hardware fidelity. The optional pulse simulator is a closed-system ideal-Hamiltonian check and excludes leakage, decoherence, transfer-function distortion, calibration drift, and hardware validation.
