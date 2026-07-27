# BiasBlaster

tldr: Predict how the circuit is gonna mess up -> More efficent reduction of errors

BiasBlaster turns local coherent-error estimates into a sparse end-of-circuit bias model, then uses pulse-derived local control Jacobians to reduce that model within a global effort budget.

```text
local coherent generators + pulse-derived Jacobians
                    |
                    v
Pauli propagation (exact or support-capped) / nearest-Clifford approximation
                    |
                    v
end-bias estimate -> trust-region control update -> direct output-state fidelity
```
## Why

Future quantum data centers will need cheap ways of reducing error. More scenario information can produce better-targeted controls, but most tailored mitigation methods need extra calibration runs. This repo predicts circuit-level coherent bias from local errors, then selects controls to reduce it.


## What is modeled

- **Inputs:** ordered one- and two-qubit ideal unitaries, three one-qubit or full 15-mode two-qubit coherent generators, and local control Jacobians.
- **Output:** a sparse Pauli representation of end bias and a control update constrained by `effort_fraction * ||local generators||_2`.
- **Exact propagation:** uncapped Pauli transport preserves the full sparse Pauli transport for the supplied ideal circuit.
- **Bounded propagation:** a support cap retains the largest transport terms and reports an omitted-transport bound; it is never represented as exact.
- **Fast approximation:** nearest-Clifford transport stores one signed destination Pauli per gate and mode. It has no truncation certificate, so its controls are directly rescored by fidelity.
- **No-transport baseline:** local-error oracle control minimizes supplied local generator error directly under the same Jacobians and effort budget. It is a synthetic-model baseline, not an available hardware control method.

## Optimization methodology

For each gate `g`, pulse-derived local Jacobian `J_g` maps normalized pulse coordinates `delta_g` to a first-order coherent-generator update:

\[
\theta_g \mapsto \theta_g + J_g\delta_g.
\]

Transport maps each gate's generator through that gate and every downstream ideal operation. Stacking all controls gives the final first-order Pauli bias

\[
b(\delta) = b_0 + A\delta.
\]

BiasBlaster solves the convex trust-region least-squares problem

\[
\min_\delta \|b_0 + A\delta\|_2
\quad\text{subject to}\quad
\|\delta\|_2 \leq \texttt{effort_fraction}\,\|\theta\|_2.
\]

The SVD-based solver returns the global optimum of this linear model. It optimizes all gates jointly: each gate's control column already contains only that gate's downstream Pauli transport, while the solve preserves cross-gate cancellation opportunities. A greedy gate-by-gate update can reduce the bias, but cannot improve this model objective beyond the global solve.

Without bias transport, the local-error oracle uses same budget and Jacobians but solves `min ||theta + J delta||_2`. This isolates added value of circuit-level bias estimation.

Let `G` be gate count, `m` error modes per gate, and `c` controls per gate. Local control optimization has residual dimension `N = Gm` and control dimension `P = Gc`; SVD cost is `O(min(NP^2, N^2P))` time and `O(NP)` memory. The benchmark uses up to 15 modes and three controls per gate, so its asymptotic local solve remains `O(G^3)` time and `O(G^2)` memory, with larger full-SU(4) constants. It uses a one-slice, closed-system, piecewise-constant GRAPE-style model, `H = H_target + sum(u_m H_m)`. It derives local error coordinates and their Jacobian once from exact Frechet derivatives, then rescores fidelity from evolved updated pulses rather than `theta + J delta`. This remains a platform-agnostic simulator model, not hardware validation.

## Reproduce the MQT Bench suite

Install the benchmark extra:

```powershell
python -m pip install -e ".[benchmark]"
```

Run the documented suite:

```powershell
python scripts/benchmark_tradeoff.py `
  --corpus mqt-bench --mqt-benchmarks ghz graphstate qnn qft qaoa grover `
  --estimator all --support-caps 64 2048 `
  --widths 6 8 10 --seeds 7 --fidelity-probes 8 `
  --error-models iid-coherent quasistatic-z-drift spatially-correlated sparse-outliers `
  --effort-fraction 0.25
```

MQT Bench supplies circuit structure and the harness compiles it to `rz`, `sx`, `x`, and `cx`. The suite evaluates 6-, 8-, and 10-qubit GHZ, graph-state, QNN, QFT, and QAOA instances. Grover is generated too, then excluded by the deterministic per-size Tukey 1.5-IQR upper gate-count fence. The command prints the retained corpus, exclusions, support, truncation bound, timings, and fidelity metric.

Each synthetic error ensemble seeds three local pulse amplitudes per gate. The harness evolves the closed-system bilinear control Hamiltonian, derives full-SU(4) two-qubit local error coordinates and Jacobians from exact Frechet derivatives, normalizes pulse-coordinate columns by their local response, and recomputes every after-control circuit from its updated pulse unitaries. Its projection residual is numerical roundoff rather than a dropped two-qubit error mode.

`--include-unbounded-pauli` is intentionally excluded from the 6/8/10 suite:
uncapped Pauli support can grow exponentially with circuit width. Use it only
for small-width exact checks.


### Current closed-system GRAPE suite result

Run 2026-07-25 with the documented command: 15 retained fixtures (five
benchmarks at widths 6, 8, and 10); the per-size Tukey filter excluded all
Grover fixtures. Parentheses are mean output-state-infidelity reduction.

| Error model | Before | Oracle | Nearest Clifford | Pauli cap 64 | Pauli cap 2048 |
| --- | ---: | ---: | ---: | ---: | ---: |
| IID coherent | 0.999860111 | 0.999921309 (43.75%) | 0.999959423 (70.99%) | 0.999933500 (52.46%) | **0.999966235 (75.86%)** |
| Quasistatic Z/ZZ drift | 0.999812961 | 0.999894772 (43.74%) | 0.999921224 (57.88%) | 0.999871990 (31.56%) | **0.999925405 (60.12%)** |
| Spatially correlated | 0.999744305 | 0.999856148 (43.74%) | 0.999891869 (57.71%) | 0.999874287 (50.83%) | **0.999903438 (62.24%)** |
| Sparse outliers | 0.999848499 | 0.999914773 (43.74%) | 0.999956895 (71.55%) | 0.999925816 (51.03%) | **0.999964087 (76.30%)** |

Mean post-update omitted-bias certificates for capped Pauli transport:

| Error model | Cap 64 | Cap 2048 |
| --- | ---: | ---: |
| IID coherent | 2.160715e+0 | 2.977391e-1 |
| Quasistatic Z/ZZ drift | 1.091063e+0 | 1.416930e-1 |
| Spatially correlated | 1.640222e+0 | 2.210078e-1 |
| Sparse outliers | 2.335044e+0 | 3.207195e-1 |

These certificates bound omitted Pauli-bias norm, not fidelity error. This is
a closed-system, simulator-only result; it is not process fidelity or hardware
validation.

Two-qubit pulse calibration uses all 15 traceless Pauli modes. Reported
projection residual is only floating-point roundoff; optimizer, transport
certificate, and nonlinear rescore use the same local-error space.

### Why unbounded Pauli matters

The pulse simulator and transport model use the same convention: each local
coherent correction occurs **before** its ideal operation. Pauli transport
propagates that generator through the operation itself and all later
operations. A support cap is still an approximation; its reported omitted-bias
bound must be read beside every capped result. Exact timings and retained
support depend on corpus, pulse-error model, and machine, so the harness emits
them rather than storing stale aggregate numbers here.

### Synthetic coherent-error models

- **IID coherent:** independent seed amplitudes at every gate and correction mode.
- **Quasistatic Z/ZZ drift:** a per-qubit drift shared by every gate touching that qubit, plus a small independent residual.
- **Extreme Z/ZZ bias:** aligned Z/ZZ coherent error at every gate; transverse modes are 50x smaller.
- **Spatially correlated:** shared global and per-qubit mode components, plus a small independent residual.
- **Sparse outliers:** low baseline coherent terms with larger perturbations at 12% of gates.

These models seed simulated correction pulses; they broaden the algebraic stress test but are not fitted device-noise distributions.

### Reading the fidelity metric

For each input state `|psi>`, the harness evolves the ideal circuit and the circuit built from its simulated pulse unitaries. It measures `F = |<psi_ideal|psi_actual>|^2` exactly for that pair of output states. The mean over Haar probes is a sampled output-state-fidelity estimate, not an exact process or average-gate fidelity. The same probe states are used for before/after comparison.

## Tang Nano 20K FPGA optimizer

`biasblaster.fpga_optimizer.optimize_bias_controls_fpga` keeps the same input
and output contract as `optimize_bias_controls`: local generators, transport,
local Jacobians, effort fraction, and `BiasOptimizationResult`. It builds the
same transported end-bias residual and control design matrix. The repository
trust-region solve remains source of truth; its target is quantized and sent to
the deployed BiasSteerer Tang Nano 20K image over UART for bounded application
and reply verification.

### How FPGA mode works

1. Host builds the current transported bias residual and each gate's local
   control Jacobian.
2. Host packs one gate record: up to five residual rows, six controls, bounds,
   damping, and a gate-local effort limit.
3. Tang Nano runs fixed-point damped Gauss--Newton plus bounded retraction,
   then returns six Q12 controls over UART.
4. Host decodes, projects the result into the gate's L2 budget, rescoring the
   full transported residual before accepting it.

There are two modes:

- `optimize_bias_controls_fpga`: CPU computes the authoritative global
  trust-region target; FPGA applies and verifies its Q12 representation. Use
  this for CPU-contract parity.
- `optimize_bias_controls_fpga_greedy`: host chooses one high-value gate at a
  time and FPGA computes that gate's local update. Use this for real on-board
  local optimization; it is heuristic, not the global CPU optimum.

Greedy records cannot be batched: gate `n + 1` needs gate `n`'s reply and the
new full-residual score. The low-level batch command is only for up to four
already-computed, reply-independent records.

The fixed hardware record supports five residual rows and six flattened control
coordinates. The deployed datapath is not the CPU optimizer: it is a small
fixed-point damped Gauss–Newton engine, whereas the CPU uses an SVD-based
global trust-region solve. Exact CPU optimization therefore does not fit on
the current FPGA image. The host computes the CPU target, then the FPGA applies
and echoes that quantized target for parity verification. Larger or
unrepresentable payloads use the exact CPU optimizer by default; pass
`fallback="error"` to reject them. With no `link` or `port`, the module runs
the fixed-point reference for offline protocol/parity tests.

```powershell
python -m pip install -e ".[fpga]"
$env:BIASBLASTER_FPGA_PORT = "COM6"
```

```python
from biasblaster import optimize_bias_controls_fpga

result = optimize_bias_controls_fpga(
    local_generators, transport, local_jacobians,
    effort_fraction=0.25,
    port="COM6",                 # omit for fixed-point reference
    fallback="cpu",              # or "error"
)
```

Protocol matches BiasSteerer: `0x06` ping returns `0xA5`; `0x0F` carries
Q12/Q15 residual/Jacobian data and returns six Q12 controls. FPGA output must
match the host-computed CPU target within one Q12 LSB (`1/4096`); otherwise
the call fails. Global effort-budget and trust-region semantics remain those
of the CPU optimizer. Implementing the CPU solve on the FPGA would require a
new RTL/firmware datapath; no such implementation is present in this repo.
The current image uses 1,000,000 baud (exact 27 MHz divider); pass
`baud=115200` only for the legacy image. Opcode `0x10` batches up to four
precomputed independent `0x0F`-format records. It is exposed as
`FPGALink.pulse_gate_bias_optimize_batch`; greedy updates deliberately do not
use it, because each next record depends on the prior reply and rescore.

For a connected board, `optimize_bias_controls_fpga_greedy` provides a
gate-local alternative. Host ranks gates by predicted full-bias reduction,
compresses every gate's local normal equations into at most five FPGA rows,
and sends one fixed-point update. Returned controls are radially projected to
the gate's L2 effort ball, accepted only when they lower the complete residual,
and revisited for up to two sweeps by default. Pass a larger
`--fpga-max-sweeps` benchmark value or `FpgaOptimizerConfig(max_sweeps=...)`
when extra refinement is worth UART time. This lets later bias changes promote an
earlier gate. Pass `port` for a real UART connection or `link`
for an existing connection. Its FPGA result is a greedy damped Gauss–Newton
update, not the global SVD CPU optimum; oversized/unrepresentable records use
the same greedy ordering on the CPU when `fallback="cpu"` is selected.

```python
from biasblaster import optimize_bias_controls_fpga_greedy

result = optimize_bias_controls_fpga_greedy(
    local_generators, transport, local_jacobians,
    effort_fraction=0.25,
    port="COM6",
    fallback="cpu",
)
```

`benchmark_tradeoff.py --optimization-backend fpga-greedy` opens, pings, and
reuses one UART link for its complete benchmark. Gate updates remain sequential:
each next record depends on prior FPGA reply and residual recomputation.

## Scope and limitations

BiasBlaster models first-order coherent, ideal-circuit transport. The fidelity benchmark does not establish hardware fidelity or validate leakage, decoherence, stochastic noise, transfer-function distortion, crosstalk outside supplied Jacobians, calibration drift, state-preparation/readout error, or device-specific pulse behavior. The optional pulse simulator checks only closed-system, piecewise-constant Hamiltonian evolution.

See [docs/method.md](docs/method.md) for the transport equations, truncation bound, and effort-budget objective.
