"""Uncertainty-aware correction policies for noisy Quantum Krylov observables."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .channel import ObservableImpactBatch


@dataclass(frozen=True)
class BiasShrinkageResult:
    """Observable-level correction recommended from bias-to-uncertainty ratio."""

    raw_bias: np.ndarray
    correction: np.ndarray
    weights: np.ndarray
    variances: np.ndarray


def shrinkage_weights(
    bias: np.ndarray,
    variances: np.ndarray,
    *,
    strength: float = 1.0,
    variance_floor: float = 0.0,
) -> np.ndarray:
    """Return bounded bias-correction weights from local bias SNR.

    The default rule is

    ``alpha_i = b_i^2 / (b_i^2 + strength * sigma_i^2)``.

    It approaches one when the predicted systematic bias is large compared with
    its uncertainty and approaches zero when correction would mostly inject
    model/shot noise. This is a shrinkage heuristic rather than an unbiased
    estimator; its purpose is lower expected squared correction error.
    """

    b = np.asarray(bias, dtype=float)
    v = np.asarray(variances, dtype=float)
    if b.ndim != 1 or v.shape != b.shape:
        raise ValueError("bias and variances must be same-length vectors")
    strength = float(strength)
    variance_floor = float(variance_floor)
    if strength < 0 or not np.isfinite(strength):
        raise ValueError("strength must be finite and nonnegative")
    if variance_floor < 0 or not np.isfinite(variance_floor):
        raise ValueError("variance_floor must be finite and nonnegative")
    if np.any(v < -1e-15) or not np.all(np.isfinite(v)):
        raise ValueError("variances must be finite and nonnegative")
    effective = np.maximum(v, 0.0) + variance_floor
    numerator = b * b
    denominator = numerator + strength * effective
    weights = np.zeros_like(b)
    nonzero = denominator > 0
    weights[nonzero] = numerator[nonzero] / denominator[nonzero]
    return np.clip(weights, 0.0, 1.0)


def shrink_observable_bias(
    impacts: ObservableImpactBatch,
    *,
    strength: float = 1.0,
    variance_floor: float = 0.0,
) -> BiasShrinkageResult:
    """Compute the uncertainty-weighted correction for each observable."""

    if impacts.covariance is None:
        variances = np.zeros(len(impacts.names), dtype=float)
    else:
        variances = np.maximum(np.diag(impacts.covariance), 0.0)
    weights = shrinkage_weights(
        impacts.bias,
        variances,
        strength=strength,
        variance_floor=variance_floor,
    )
    return BiasShrinkageResult(
        raw_bias=impacts.bias.copy(),
        correction=weights * impacts.bias,
        weights=weights,
        variances=variances,
    )


def with_shrunk_observable_bias(
    impacts: ObservableImpactBatch,
    *,
    strength: float = 1.0,
    variance_floor: float = 0.0,
) -> tuple[ObservableImpactBatch, BiasShrinkageResult]:
    """Return an impact batch whose systematic correction is shrinkage-weighted.

    Jacobians and covariance are retained unchanged: only the deterministic
    correction applied downstream is shrunk. This keeps uncertainty propagation
    conservative while avoiding full subtraction of poorly resolved bias.
    """

    result = shrink_observable_bias(
        impacts,
        strength=strength,
        variance_floor=variance_floor,
    )
    adjusted = ObservableImpactBatch(
        names=impacts.names,
        ideal=impacts.ideal.copy(),
        bias=result.correction.copy(),
        predicted=impacts.ideal + result.correction,
        jacobian=impacts.jacobian.copy(),
        mode_names=impacts.mode_names,
        mode_means=impacts.mode_means.copy(),
        covariance=None if impacts.covariance is None else impacts.covariance.copy(),
        mode_covariance=(
            None if impacts.mode_covariance is None else impacts.mode_covariance.copy()
        ),
        forward_dropped_l2=impacts.forward_dropped_l2,
        backward_dropped_l2=impacts.backward_dropped_l2.copy(),
    )
    return adjusted, result
