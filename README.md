# BiasBlaster

BiasBlaster estimates scalable coherent bias at the end of an ideal circuit and prioritizes calibrated local control directions against that estimate.

```text
local coherent generators  ->  Pauli / nearest-Clifford estimator  ->  calibrated-control optimizer
```

Install the core package with:

```powershell
python -m pip install -e .
```

Run the reproducible tradeoff benchmark with:

```powershell
python scripts/benchmark_tradeoff.py --estimator pauli --support-cap 64 --width 4 --circuits 3 --seed 7
```

The benchmark reports algebraic model estimates, including the explicit omitted-transport bound for capped Pauli propagation. This representative run uses `--width 4 --circuits 3 --effort-fraction 0.25 --seed 7`; exact timings vary by machine.

| estimator | support cap | support | memory | before → after proxy | runtime |
| --- | ---: | ---: | ---: | ---: | ---: |
| nearest Clifford | not used | 28 | 1,080 B | 2.32e-5 → 6.42e-6 | 12.8 ms |
| Pauli propagation | 16 | 16 | 11,520 B | 9.00e-6 → 5.71e-7 | 11.7 ms |
| Pauli propagation | 64 | 30 | 21,600 B | 2.05e-5 → 4.95e-6 | 12.6 ms |

The cap-16 run reported a mean omitted-transport bound of `2.94e-1`; the uncapped/64-support runs reported zero for this fixture. The bound is part of the result and is not silently discarded.

The model is first-order and coherent. The optional pulse simulator validates only closed-system, piecewise-constant Hamiltonian evolution; it does not validate hardware fidelity, leakage, decoherence, transfer-function distortion, or calibration drift.

See [docs/method.md](docs/method.md) for the transport equations, truncation bound, and effort-budget objective.
