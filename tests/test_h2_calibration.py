import numpy as np
import pytest

from biasblaster.h2_calibration import (
    H2_1_CALIBRATION,
    H2_2_CALIBRATION,
    get_h2_calibration,
    prepare_h2_krylov,
)


def test_h2_profiles_match_public_component_calibration_and_infidelity_conversion():
    h21 = get_h2_calibration("H2-1")
    h22 = get_h2_calibration("H2-2E")
    assert h21 is H2_1_CALIBRATION
    assert h22 is H2_2_CALIBRATION
    assert np.isclose(h21.one_qubit_infidelity, 1.9e-5)
    assert np.isclose(h21.p1, 1.5 * 1.9e-5)
    assert np.isclose(h21.p2, 1.25 * 1.1e-3)
    assert np.isclose(h21.spam_0, 6.0e-4)
    assert np.isclose(h21.spam_1, 1.4e-3)
    assert np.isclose(h22.one_qubit_infidelity, 2.8e-5)
    assert np.isclose(h22.p2, 1.25 * 8.3e-4)


def test_h2_calibration_path_builds_without_pulse_data():
    pytest.importorskip("qiskit")
    prepared = prepare_h2_krylov(
        H2_1_CALIBRATION,
        n_qubits=2,
        dimension=2,
        time_step=0.4,
        trotter_steps=1,
    )
    assert prepared.raw_combined.mode_names == (
        "h2_p1", "h2_p2", "h2_spam0", "h2_spam1"
    )
    assert np.all(np.isfinite(prepared.finite_values))
    assert prepared.profile.memory_error_per_depth1 > 0.0
    # Memory is metadata only until an H2-native scheduled circuit is available.
    assert "memory" not in " ".join(prepared.raw_combined.mode_names)
