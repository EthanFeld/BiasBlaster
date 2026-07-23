# BiasBlaster Cleanup Handoff Guide

## Objective

Convert this repository into a small, recruiter-readable personal project about scalable coherent-bias estimation for pulse-control prioritization.

Preserve these capabilities:

1. Represent local coherent error generators for ideal circuit operations.
2. Estimate end-of-circuit bias with two methods:
   - nearest-Clifford approximation;
   - bounded sparse Pauli propagation with an explicit truncation bound.
3. Optimize calibrated local control/Jacobian directions against either estimate.
4. Retain an optional, closed-system piecewise-constant pulse simulator for small local validation.
5. Reproduce one benchmark showing Pauli-propagation quality versus nearest-Clifford latency and memory.

Do **not** preserve every historical research branch. The final project must not claim hardware-validated pulse fidelity.

## Nonnegotiable Safety Rules

1. Create a cleanup branch before changing code.
2. Do not delete a source path until its retained behavior has migrated and a focused replacement test passes.
3. Do not reintroduce Torch, ML training, VQE-specialized models, or dense full-circuit Pauli vectors into the core package.
4. Do not call a truncation result exact. Return and display its omitted-transport bound.
5. Do not describe the closed-system simulator as a hardware model.
6. Keep commits small and themed: extraction, migration, deletion, documentation, then benchmark cleanup.

## Target Public API

Create a compact package surface. Public names should be descriptive; numbered module names must not appear in documentation or examples.

```text
biasblaster/
  __init__.py
  model.py
  pauli.py
  tableau.py
  local_transfer.py
  transport.py
  nearest_clifford.py
  optimizer.py
  pulse_simulator.py
```

Expose only these concepts from `biasblaster.__init__`:

- `CircuitOperation`
- `SparsePauliTransport`
- `SparseCliffordTransport`
- `estimate_pauli_transport`
- `estimate_nearest_clifford_transport`
- `optimize_bias_controls`
- `simulate_piecewise_constant_pulse`
- Pauli/tableau helpers needed by users

### Required data contracts

`CircuitOperation`

```python
@dataclass(frozen=True)
class CircuitOperation:
    name: str
    qubits: tuple[int, ...]
    unitary: np.ndarray
    angle: float = 0.0
```

`SparsePauliTransport`

- Maps end-Pauli labels to vectors of local generator-mode coefficients.
- Stores `n_qubits`, `gate_count`, active gate indices, support count, and `truncation_bound`.
- Provides a method to contract current local generators into a sparse end bias.
- Provides a method to compute local priorities `M_g.T @ b`.
- Never materializes `4**n` output arrays.

`SparseCliffordTransport`

- Stores one signed final Pauli label for each gate/mode.
- Stores no learned model state.
- Offers compact and bounded-window dense adapters only when explicitly requested.

`optimize_bias_controls`

- Inputs: local generators, transport, local calibrated Jacobians, effort fraction.
- Outputs: controls, updated generators, predicted before/after norm, control norm, budget.
- Must support both retained transport types without caller-side conversion.

## Step 1: Freeze Current Behavior

Before refactoring, add or retain five small golden tests. These tests define the supported project, not the old repository shape.

1. **Tableau test**: Clifford conjugation maps known Pauli strings and signs correctly.
2. **Pauli branch test**: an RZ-like non-Clifford operation splits X/Y components; a support cap retains the largest terms and emits nonzero bound.
3. **Nearest-Clifford test**: a Clifford circuit agrees with exact Pauli transport; RZ near `pi/2` selects the S-like Clifford deterministically.
4. **Optimizer test**: a calibrated local Jacobian reduces predicted bias and never exceeds the global effort budget.
5. **Pulse test**: closed-system pulse derivative agrees with finite differences on a one-qubit Hamiltonian.

Add one integration test:

- Build a two-qubit circuit with a non-Clifford rotation.
- Generate deterministic local coherent errors.
- Optimize controls with Pauli transport and nearest-Clifford transport using equal effort budgets.
- Evaluate both updates with uncapped Pauli propagation.
- Assert Pauli transport is no worse than nearest Clifford for this fixed fixture.

Run these tests before every deletion phase.

## Step 2: Extract Core Types and Math

1. Move `LocalOperation` from the current ML module into `model.py` and rename it `CircuitOperation`.
2. Move pure NumPy local Pauli-transfer helpers from the ML module into `local_transfer.py`:
   - local Pauli labels;
   - local Pauli matrices;
   - local Pauli transfer calculation;
   - embedding/extracting local Pauli labels;
   - local generator-mode labels.
3. Keep all Torch imports out of these files.
4. Move/re-export symplectic tableau code into `tableau.py` without changing its tested conjugation convention.
5. Update all retained consumers to import only from the new files.

Acceptance:

- Core package imports successfully in an environment containing only NumPy and SciPy.
- No retained core file imports Torch, Qiskit, MQT Bench, or a machine-learning module.

## Step 3: Consolidate Pauli Propagation

1. Create `transport.py` around the current sparse propagation behavior.
2. Retain exact local Pauli conjugation through supported ideal operations.
3. Accept `support_cap` and `coefficient_tol` on every sparse propagation entry point.
4. On cap overflow:
   - retain largest coefficient vectors by norm;
   - accumulate dropped-vector norms into a conservative truncation bound;
   - keep the bound attached to the result and never silently discard it.
5. Remove VQE event/motif assumptions from the core propagator. It must accept an ordinary ordered sequence of `CircuitOperation` values.
6. Keep only a small MQT/Qiskit adapter in benchmark code, not in the library.

Acceptance:

- Small uncapped circuits agree with dense local Pauli algebra.
- Capped circuits report nonzero bounds exactly when terms are dropped.
- Memory is proportional to retained support times active source modes, not `4**n`.

## Step 4: Retain and Simplify Nearest Clifford

1. Move current nearest-Clifford logic into `nearest_clifford.py`.
2. Preserve same-arity candidate selection by global-phase-invariant process fidelity.
3. Preserve deterministic tie-breaking using canonical tableau serialization.
4. Preserve one- and two-qubit operation support only. Reject larger local arity with a clear error.
5. Keep exact Clifford fast path before candidate search.
6. Store compact mode labels/signs. Do not make dense maps the normal representation.

Acceptance:

- Exact Clifford input yields exact signed Pauli transport.
- Nearby RZ cases choose the expected nearest Clifford.
- Input operations are never mutated.

## Step 5: Replace Duplicate Optimizers

1. Create `optimizer.py` with one public optimizer: `optimize_bias_controls`.
2. Use calibrated local Jacobians of shape `(gate, mode, control)`.
3. Implement one trust-region least-squares solve for both transport forms.
4. Add a compact operator path for nearest-Clifford transport. Avoid materializing the current gate-by-support dense adapter where possible.
5. Add a sparse-matrix/operator path for Pauli transport. Do not keep separate ML, VQE-priority, calibration-window, and dense optimizers.
6. Return one result dataclass with controls, updated generators, predicted norms, budget, and optional truncation bound.

Acceptance:

- Both transport methods use the same effort-budget semantics.
- Existing optimizer golden tests pass.
- Nearest-Clifford control does not require a dense Pauli output window.

## Step 6: Preserve Only Minimal Pulse Simulation

1. Move the current piecewise-constant unitary propagator and Fréchet derivatives into `pulse_simulator.py`.
2. Rename public function to `simulate_piecewise_constant_pulse`.
3. Keep only closed-system Hamiltonian evolution and exact derivative checks.
4. Add a module-level warning in the docstring:

   > This validates an ideal Hamiltonian model only. It excludes leakage, decoherence, transfer-function distortion, calibration drift, and hardware validation.

5. Remove the larger historical HSCA/BCH pulse-loop stack unless a retained integration test requires a specific helper. If a helper is needed, copy only that helper into the new module with direct tests.

Acceptance:

- One-qubit finite-difference derivative test passes.
- No README language implies real-device fidelity prediction.

## Step 7: Delete Dead Paths

After Steps 1–6 pass, remove these branches and their tests/scripts unless a retained core file still imports them:

- All Torch/ML model modules, training code, and learned-priority pipelines.
- VQE-specialized motif, tensor, kernel, full-ML, and hybrid-stitch models.
- Legacy numbered analysis/study/ceiling/comparison modules that are not required by the new core tests.
- Duplicate dense transport and duplicate optimizer implementations.
- One-off benchmark scripts that do not support the final project narrative.
- Generated caches, `__pycache__`, `.pytest_cache`, benchmark output blobs, and stale specifications.

Before each deletion:

```powershell
rg -n "<symbol-or-module-name>" biasblaster scripts tests
```

Delete only when all retained references have migrated.

## Step 8: Consolidate Benchmarks

Keep one script only:

```text
scripts/benchmark_tradeoff.py
```

Required flags:

- `--estimator nearest-clifford|pauli`
- `--support-cap <int>`
- `--profile default|rydberg-amplitude-5pct`
- `--width`, `--circuits`, `--effort-fraction`, `--seed`

Required output:

- circuit count and mean gate count;
- retained Pauli support and truncation bound;
- before/after coherent-infidelity proxy;
- reduction;
- runtime and transport bytes;
- explicit statement that results are algebraic model estimates, not hardware fidelity.

Fold current nearest-versus-Pauli, truncation sweep, and Rydberg amplitude-profile scripts into this one command. Do not preserve a benchmark merely because it produced an old result.

## Step 9: Simplify Dependencies and Packaging

1. Core `pyproject.toml` dependencies: NumPy and SciPy only.
2. Put Qiskit/MQT Bench in an optional `benchmark` extra.
3. Remove Torch from all default and optional dependencies unless a separate archived experiment package is intentionally retained outside this repository.
4. Rebuild `biasblaster.__init__` manually from the target public API; do not carry forward the current giant export list.
5. Verify:

```powershell
python -c "import biasblaster; print('core import OK')"
python -m pytest -q
```

## Step 10: Write Recruiter-Facing Documentation

Create only these documents:

- `README.md`
- `docs/method.md`
- this cleanup guide may be deleted after the cleanup branch merges if it is no longer useful.

`README.md` must contain:

1. One-sentence project purpose.
2. Tiny architecture diagram: local generators -> estimator -> calibrated-control optimizer.
3. One install command.
4. One benchmark command.
5. One compact results table: nearest Clifford versus Pauli propagation versus support cap.
6. Honest limitations: first-order coherent model; no hardware calibration/leakage/decoherence validation.

`docs/method.md` must define:

- Pauli propagation;
- nearest-Clifford replacement;
- truncation bound;
- optimizer objective and effort budget;
- why results are not hardware-fidelity claims.

## Final Acceptance Checklist

Do not declare cleanup complete until all items pass.

- [ ] Core import works without Torch.
- [ ] Five core unit tests and one integration test pass.
- [ ] One benchmark command runs from a clean environment.
- [ ] Pauli cap is visible in benchmark output and produces a bound when it truncates.
- [ ] Nearest-Clifford path is directly optimizer-compatible.
- [ ] No public module/file name contains historical module numbers.
- [ ] No ML/VQE/training path remains in the repository.
- [ ] README includes one reproducible result and limitations.
- [ ] No claim of real-hardware pulse accuracy remains.
- [ ] `git status` contains only intentional source, test, documentation, and packaging changes.

## Definition of Done

A recruiter should be able to clone the repository, read the README, run one command, see the Pauli-quality versus nearest-Clifford-scale tradeoff, and understand the limitation statement in under five minutes.
