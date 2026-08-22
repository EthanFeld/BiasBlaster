# Multi-circuit Quantum Krylov impact assembly

Quantum Krylov matrix elements are often estimated by different circuits. Run
`estimate_observable_impacts` independently for each circuit, using the same
`ChannelMode.name` for calibration parameters that are physically shared, then
combine the raw sensitivity batches globally:

```python
from biasblaster import combine_observable_impacts

combined = combine_observable_impacts(
    [h_batch, s_batch, other_batch],
    mode_covariance=global_calibration_covariance,
    shot_covariance=global_shot_covariance,
)
```

`combine_observable_impacts` aligns Jacobian columns by mode name. A shared
drift mode therefore induces covariance between matrix elements even when those
elements came from different circuits. Modes that are absent from one circuit
receive a zero sensitivity in that circuit.

Apply covariance only after combining raw batches. This avoids double counting
and allows the covariance model to include device-wide/common-mode calibration
parameters. The resulting `ObservableImpactBatch` can be passed directly to
`build_krylov_error_model`.
