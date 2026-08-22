"""Composition helpers for hardware-impact estimates from different circuits."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .channel import ObservableImpactBatch


def combine_observable_impacts(
    batches: Sequence[ObservableImpactBatch],
    *,
    mode_covariance: np.ndarray | None = None,
    shot_covariance: np.ndarray | None = None,
) -> ObservableImpactBatch:
    """Combine estimator batches from different circuits into one global model.

    Error modes are aligned by ``mode_names``. Reusing the same mode name in
    different circuits means those observables depend on the same calibrated
    hardware parameter; a global covariance can therefore create correlations
    between H/S estimates measured by different circuits.

    Constituent batches must be raw sensitivity batches with ``covariance=None``.
    Supply calibration and shot covariance here, after global mode alignment.
    """

    batches = tuple(batches)
    if not batches:
        raise ValueError("at least one observable-impact batch is required")
    if any(batch.covariance is not None for batch in batches):
        raise ValueError(
            "combine raw batches with covariance=None, then supply covariance globally"
        )

    observable_names: list[str] = []
    mode_names: list[str] = []
    mode_means: dict[str, float] = {}
    for batch in batches:
        if len(set(batch.names)) != len(batch.names):
            raise ValueError("observable names must be unique within each batch")
        if len(set(batch.mode_names)) != len(batch.mode_names):
            raise ValueError("mode names must be unique within each batch")
        for name in batch.names:
            if name in observable_names:
                raise ValueError(f"duplicate observable name across batches: {name}")
            observable_names.append(name)
        for index, name in enumerate(batch.mode_names):
            mean = float(batch.mode_means[index])
            if name in mode_means:
                if not np.isclose(mode_means[name], mean, atol=1e-12, rtol=0.0):
                    raise ValueError(f"inconsistent calibrated mean for shared mode {name}")
            else:
                mode_names.append(name)
                mode_means[name] = mean

    total_observables = len(observable_names)
    total_modes = len(mode_names)
    mode_index = {name: index for index, name in enumerate(mode_names)}
    jacobian = np.zeros((total_observables, total_modes), dtype=float)
    ideal = np.empty(total_observables, dtype=float)
    backward_dropped = np.empty(total_observables, dtype=float)

    row = 0
    forward_dropped_sq = 0.0
    for batch in batches:
        count = len(batch.names)
        ideal[row : row + count] = batch.ideal
        backward_dropped[row : row + count] = batch.backward_dropped_l2
        for local_column, name in enumerate(batch.mode_names):
            jacobian[row : row + count, mode_index[name]] = batch.jacobian[:, local_column]
        forward_dropped_sq += float(batch.forward_dropped_l2) ** 2
        row += count

    means = np.asarray([mode_means[name] for name in mode_names], dtype=float)
    bias = jacobian @ means
    covariance = None
    sigma_copy = None
    if mode_covariance is not None:
        sigma = np.asarray(mode_covariance, dtype=float)
        if sigma.shape != (total_modes, total_modes):
            raise ValueError("mode_covariance has the wrong shape")
        if not np.allclose(sigma, sigma.T, atol=1e-10, rtol=0.0):
            raise ValueError("mode_covariance must be symmetric")
        if np.min(np.linalg.eigvalsh(sigma), initial=0.0) < -1e-10:
            raise ValueError("mode_covariance must be positive semidefinite")
        sigma_copy = sigma.copy()
        covariance = jacobian @ sigma @ jacobian.T

    if shot_covariance is not None:
        shot = np.asarray(shot_covariance, dtype=float)
        if shot.shape != (total_observables, total_observables):
            raise ValueError("shot_covariance has the wrong shape")
        if not np.allclose(shot, shot.T, atol=1e-10, rtol=0.0):
            raise ValueError("shot_covariance must be symmetric")
        if np.min(np.linalg.eigvalsh(shot), initial=0.0) < -1e-10:
            raise ValueError("shot_covariance must be positive semidefinite")
        covariance = shot.copy() if covariance is None else covariance + shot

    return ObservableImpactBatch(
        names=tuple(observable_names),
        ideal=ideal,
        bias=bias,
        predicted=ideal + bias,
        jacobian=jacobian,
        mode_names=tuple(mode_names),
        mode_means=means,
        covariance=covariance,
        mode_covariance=sigma_copy,
        forward_dropped_l2=float(np.sqrt(forward_dropped_sq)),
        backward_dropped_l2=backward_dropped,
    )
