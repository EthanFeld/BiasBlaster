import numpy as np

from biasblaster.krylov import KrylovErrorModel
from biasblaster.krylov_regularization import assess_overlap_modes, solve_noise_aware_krylov


def _model(covariance):
    weights_h = np.zeros((2, 2, 2), dtype=complex)
    weights_s = np.zeros((2, 2, 2), dtype=complex)
    weights_s[0, 0, 0] = 1.0
    weights_s[1, 1, 1] = 1.0
    return KrylovErrorModel(
        bias_h=np.zeros((2, 2), dtype=complex),
        bias_s=np.zeros((2, 2), dtype=complex),
        jacobian_h=np.zeros((0, 2, 2), dtype=complex),
        jacobian_s=np.zeros((0, 2, 2), dtype=complex),
        mode_names=(),
        mode_means=np.zeros(0),
        mode_covariance=None,
        observable_covariance=np.asarray(covariance, dtype=float),
        observable_weights_h=weights_h,
        observable_weights_s=weights_s,
        observable_names=("s00", "s11"),
    )


def test_modewise_overlap_test_keeps_well_resolved_small_direction():
    overlap = np.diag([1.0, 0.05])
    model = _model(np.diag([0.2**2, 0.005**2]))
    assessment = assess_overlap_modes(overlap, model, safety_factor=2.0)
    # Global Frobenius uncertainty is dominated by the first direction, while
    # the small second direction is itself measured precisely enough to keep.
    assert assessment.keep.tolist() == [True, True]
    assert assessment.standard_deviation[0] == 0.005
    assert assessment.standard_deviation[1] == 0.2


def test_modewise_solver_discards_only_uncertain_overlap_direction():
    # scipy.eigh returns ascending overlap eigenvalues, so the first eigenvector
    # here corresponds to the second computational basis vector.
    overlap = np.diag([1.0, 0.05])
    hamiltonian = np.diag([0.3, 0.8])
    model = _model(np.diag([0.01**2, 0.10**2]))
    result, assessment = solve_noise_aware_krylov(
        hamiltonian,
        overlap,
        model,
        safety_factor=1.0,
    )
    assert result.retained_overlap_rank == 1
    assert np.count_nonzero(assessment.keep) == 1
    assert np.isclose(result.energy, 0.3)


def test_condition_guard_rejects_precise_but_amplifying_overlap_direction():
    # The 0.02 direction is statistically precise, but whitening it would
    # amplify projected-matrix noise by ~50x relative to the dominant mode.
    overlap = np.diag([1.0, 0.02])
    hamiltonian = np.diag([0.4, -0.2])
    model = _model(np.diag([1e-8, 1e-8]))
    assessment = assess_overlap_modes(
        overlap,
        model,
        safety_factor=1.0,
        max_condition_number=25.0,
    )
    assert np.isclose(assessment.conditioning_floor, 0.04)
    assert assessment.keep.tolist() == [False, True]
    result, _ = solve_noise_aware_krylov(
        hamiltonian,
        overlap,
        model,
        max_condition_number=25.0,
    )
    assert result.retained_overlap_rank == 1
    assert np.isclose(result.energy, 0.4)
