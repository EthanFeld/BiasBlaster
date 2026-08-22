import numpy as np

from biasblaster.calibration import (
    CalibratedErrorMap,
    CalibrationParameter,
    CalibrationRule,
    combine_calibrated_impacts,
    estimate_calibrated_impacts,
)
from biasblaster.channel import (
    channel_delta,
    dephasing_ptm,
    depolarizing_ptm,
    estimate_observable_impacts,
)
from biasblaster.model import CircuitOperation


def _x_gate(qubit=0):
    return CircuitOperation(
        "x", (qubit,), np.array([[0, 1], [1, 0]], dtype=complex)
    )


def test_calibration_map_expands_shared_gate_parameter_and_covariance():
    derivative = channel_delta(depolarizing_ptm(1e-5, 1)) / 1e-5
    calibration = CalibratedErrorMap(
        parameters=(CalibrationParameter("p1", 2.5e-5),),
        rules=(
            CalibrationRule(
                parameter="p1",
                location="after_gate",
                delta_ptm=derivative,
                gate_names=("x",),
            ),
        ),
        covariance=np.array([[4e-12]]),
    )
    expanded = calibration.expand([_x_gate(), _x_gate()], 1)
    assert [mode.name for mode in expanded.modes] == ["p1", "p1"]
    assert [mode.boundary for mode in expanded.modes] == [1, 2]
    assert np.allclose(expanded.mode_covariance, np.full((2, 2), 4e-12))
    assert expanded.occurrence_parameter_indices.tolist() == [0, 0]


def test_calibration_map_supports_spam_boundaries_and_json_roundtrip():
    derivative = channel_delta(dephasing_ptm(1e-5)) / 1e-5
    calibration = CalibratedErrorMap(
        parameters=(
            CalibrationParameter("prep", 1e-3),
            CalibrationParameter("meas", 2e-3),
        ),
        rules=(
            CalibrationRule("prep", "initial", derivative, qubits=(0,)),
            CalibrationRule("meas", "measurement", derivative, qubits=(0,)),
        ),
        covariance=np.diag([1e-8, 2e-8]),
    )
    expanded = calibration.expand([_x_gate()], 1)
    assert [mode.boundary for mode in expanded.modes] == [0, 1]
    restored = CalibratedErrorMap.from_dict(calibration.to_dict())
    assert restored.parameter_names == ("prep", "meas")
    assert np.allclose(restored.covariance, calibration.covariance)
    assert np.allclose(restored.rules[0].delta_ptm, derivative)


def test_estimate_calibrated_impacts_collapses_repeated_gate_occurrences():
    derivative = channel_delta(dephasing_ptm(1e-6)) / 1e-6
    variance = 9e-8
    calibration = CalibratedErrorMap(
        parameters=(CalibrationParameter("phase", 1e-3),),
        rules=(
            CalibrationRule(
                "phase",
                "after_gate",
                derivative,
                gate_names=("x",),
            ),
        ),
        covariance=np.array([[variance]]),
    )
    impacts = estimate_calibrated_impacts(
        [_x_gate(), _x_gate()],
        {"I": 1.0, "X": 1.0},
        {"x_expectation": {"X": 1.0}},
        calibration,
        1,
    )
    assert impacts.mode_names == ("phase",)
    assert impacts.jacobian.shape == (1, 1)
    assert impacts.covariance is not None
    expected_variance = impacts.jacobian[0, 0] ** 2 * variance
    assert np.allclose(impacts.covariance[0, 0], expected_variance)


def test_combine_calibrated_impacts_induces_cross_circuit_covariance():
    derivative = channel_delta(dephasing_ptm(1e-6)) / 1e-6
    variance = 4e-8
    calibration = CalibratedErrorMap(
        parameters=(CalibrationParameter("shared_phase", 2e-3),),
        rules=(
            CalibrationRule(
                "shared_phase", "measurement", derivative, qubits=(0,)
            ),
        ),
        covariance=np.array([[variance]]),
    )
    initial = {"I": 1.0, "X": 1.0}
    batches = []
    for name in ("left", "right"):
        expanded = calibration.expand([], 1)
        batches.append(
            estimate_observable_impacts(
                [],
                initial,
                {name: {"X": 1.0}},
                expanded.modes,
                1,
            )
        )
    combined = combine_calibrated_impacts(batches, calibration)
    assert combined.mode_names == ("shared_phase",)
    assert combined.covariance is not None
    expected = np.outer(combined.jacobian[:, 0], combined.jacobian[:, 0]) * variance
    assert np.allclose(combined.covariance, expected)


def test_covariance_for_reorders_named_parameters():
    calibration = CalibratedErrorMap(
        parameters=(
            CalibrationParameter("a", 0.1),
            CalibrationParameter("b", 0.2),
        ),
        rules=(),
        covariance=np.array([[1.0, 0.25], [0.25, 4.0]]),
    )
    assert np.allclose(
        calibration.covariance_for(("b", "a")),
        np.array([[4.0, 0.25], [0.25, 1.0]]),
    )
