import numpy as np

from biasblaster.calibration import (
    CalibratedErrorMap,
    CalibrationParameter,
    CalibrationRule,
)
from biasblaster.channel import channel_delta, dephasing_ptm, depolarizing_ptm
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
