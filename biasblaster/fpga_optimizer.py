"""Tang Nano 20K adapter for :func:`optimize_bias_controls`.

The CPU optimizer is an SVD-based trust-region least-squares solve. The
deployed BiasSteerer image instead exposes a fixed-point damped Gauss--Newton
datapath through opcode ``0x0F``. That datapath cannot represent the CPU
optimizer's arbitrary-size SVD and global trust-region solve. Consequently,
the CPU result remains authoritative: the FPGA receives its quantized target
under equal bounds, applies/echoes it, and the host verifies parity.

Problems larger than the fixed hardware record use the exact CPU optimizer by
default. ``fallback="error"`` makes that boundary explicit. With no serial
link, the same fixed-point datapath runs locally for protocol and parity tests.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Iterable, Literal, Protocol, Sequence

import numpy as np

from .optimizer import (
    BiasOptimizationResult,
    _BiasOptimizationProblem,
    _build_bias_optimization_problem,
    _transport_rows,
    _trust_region_least_squares,
    optimize_bias_controls,
)
from .nearest_clifford import SparseCliffordTransport
from .transport import AnchoredPauliTransport, SparsePauliTransport


FPGA_MAX_ROWS = 5
FPGA_MAX_CONTROLS = 6
FPGA_FULL_COUPLED_CONTROLS = 4
FPGA_THETA_SCALE = 1 << 12
FPGA_RATE_SCALE = 1 << 15
FPGA_JACOBIAN_SCALE = 1 << 15
FPGA_GAIN_SCALE = 1 << 12
FPGA_DAMPING_SCALE = 1 << 30
FPGA_REPLY_BYTES = 15
FPGA_BIAS_PAYLOAD_BYTES = 120
FPGA_PING_REPLY = 0xA5
FPGA_BAUD = 1_000_000
FPGA_LEGACY_BAUD = 115200
FPGA_BIAS_BATCH_MAX_RECORDS = 4
FPGA_BIAS_BATCH_OPCODE = 0x10
FPGA_PORT_ENV_VAR = "BIASBLASTER_FPGA_PORT"
BIASSTEERER_PORT_ENV_VAR = "BIASSTEERER_FPGA_PORT"


class FpgaSolverLink(Protocol):
    """Minimum link contract accepted by :func:`optimize_bias_controls_fpga`."""

    def pulse_gate_bias_optimize(
        self,
        *,
        gate_id: int,
        gate_kind: int,
        alpha_q: int,
        gain_q: int,
        max_step_q: int,
        damping_q: int,
        theta_q: Sequence[int],
        lower_q: Sequence[int],
        upper_q: Sequence[int],
        residuals_q: Sequence[int],
        jac_rows_q: Sequence[Sequence[int]],
        full_metric: bool = False,
        diagonal_row_mask: int = 0,
    ) -> tuple[int, int, np.ndarray]: ...


@dataclass(frozen=True)
class FpgaOptimizerConfig:
    """Fixed-point and payload limits matching deployed Tang Nano RTL.

    These limits describe the target-application record, not an exact FPGA
    implementation of BiasBlaster's CPU trust-region solver.
    """

    max_rows: int = FPGA_MAX_ROWS
    max_controls: int = FPGA_MAX_CONTROLS
    gain: float = 1.0
    damping: float = 1.0e-6
    full_metric: bool = True
    max_sweeps: int = 2
    improvement_tolerance: float = 1.0e-12
    compression_headroom: float = 0.95

    def __post_init__(self) -> None:
        if not 1 <= int(self.max_rows) <= FPGA_MAX_ROWS:
            raise ValueError(f"max_rows must be in [1, {FPGA_MAX_ROWS}]")
        if not 1 <= int(self.max_controls) <= FPGA_MAX_CONTROLS:
            raise ValueError(f"max_controls must be in [1, {FPGA_MAX_CONTROLS}]")
        if not np.isfinite(float(self.gain)) or float(self.gain) < 0.0:
            raise ValueError("gain must be finite and nonnegative")
        if not np.isfinite(float(self.damping)) or float(self.damping) < 0.0:
            raise ValueError("damping must be finite and nonnegative")
        if float(self.gain) * FPGA_GAIN_SCALE > 32767:
            raise ValueError("gain does not fit FPGA Q12")
        if float(self.damping) * FPGA_DAMPING_SCALE > 0xFFFFFFFF:
            raise ValueError("damping does not fit FPGA unsigned Q30")
        if int(self.max_sweeps) < 1:
            raise ValueError("max_sweeps must be positive")
        if not np.isfinite(float(self.improvement_tolerance)) or float(self.improvement_tolerance) < 0.0:
            raise ValueError("improvement_tolerance must be finite and nonnegative")
        if not 0.0 < float(self.compression_headroom) <= 1.0:
            raise ValueError("compression_headroom must lie in (0, 1]")


def _load_serial():
    try:
        import serial
        import serial.tools.list_ports  # noqa: F401
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "FPGA serial support needs pyserial; install biasblaster[fpga]"
        ) from exc
    return serial


def resolve_fpga_port(port: str | None = None) -> str:
    """Resolve explicit or environment-configured Tang Nano serial port."""

    selected = port or os.environ.get(FPGA_PORT_ENV_VAR) or os.environ.get(BIASSTEERER_PORT_ENV_VAR)
    if not selected:
        raise ValueError(
            f"FPGA port required: pass port or set {FPGA_PORT_ENV_VAR}"
        )
    return selected


def list_fpga_ports() -> list[str]:
    """Return visible serial devices; importing pyserial stays lazy."""

    serial = _load_serial()
    return [port.device for port in serial.tools.list_ports.comports()]


class FPGALink:
    """1 Mbaud host link compatible with current BiasSteerer Tang Nano firmware."""

    def __init__(
        self,
        port: str | None = None,
        *,
        baud: int = FPGA_BAUD,
        timeout_s: float = 2.0,
        retries: int = 3,
    ) -> None:
        if timeout_s <= 0.0 or retries < 1:
            raise ValueError("timeout_s must be positive and retries must be positive")
        serial = _load_serial()
        selected = resolve_fpga_port(port)
        self.port = selected
        self.baud = int(baud)
        self.timeout_s = float(timeout_s)
        self.retries = int(retries)
        self._serial = serial
        self._ser = serial.Serial(
            port=selected,
            baudrate=self.baud,
            timeout=self.timeout_s,
            write_timeout=self.timeout_s,
        )
        self._ser.reset_input_buffer()
        self._ser.reset_output_buffer()

    def close(self) -> None:
        if self._ser.is_open:
            self._ser.close()

    def __enter__(self) -> "FPGALink":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _request(self, opcode: int, payload: bytes, reply_len: int) -> bytes:
        last_error: Exception | None = None
        for attempt in range(self.retries):
            try:
                # A shared benchmark link has no unread reply after a complete
                # request. Clearing before every record wastes UART work and
                # can discard a delayed valid reply. Resynchronize on retry.
                if attempt:
                    self._ser.reset_input_buffer()
                self._ser.write(bytes([int(opcode) & 0xFF]) + payload)
                self._ser.flush()
                data = bytearray()
                deadline = time.monotonic() + self.timeout_s
                while len(data) < reply_len and time.monotonic() < deadline:
                    data.extend(self._ser.read(reply_len - len(data)))
                if len(data) != reply_len:
                    raise TimeoutError(
                        f"short FPGA read: expected {reply_len}, got {len(data)}"
                    )
                return bytes(data)
            except (self._serial.SerialException, TimeoutError) as exc:
                last_error = exc
                time.sleep(0.05)
        raise last_error or RuntimeError("FPGA request failed")

    def ping(self) -> int:
        return self._request(0x06, b"", 1)[0]

    def _pack_bias_payload(
        self,
        *,
        gate_id: int,
        gate_kind: int,
        alpha_q: int,
        gain_q: int,
        max_step_q: int,
        damping_q: int,
        theta_q: Sequence[int],
        lower_q: Sequence[int],
        upper_q: Sequence[int],
        residuals_q: Sequence[int],
        jac_rows_q: Sequence[Sequence[int]],
        full_metric: bool = False,
        diagonal_row_mask: int = 0,
    ) -> bytes:
        """Encode one fixed-size calibrated bias record."""

        if not 0 <= int(gate_id) <= 0xFFFF:
            raise ValueError("gate_id must fit unsigned 16-bit")
        if int(gate_kind) not in {1, 2, 3}:
            raise ValueError("gate_kind must be 1, 2, or 3")
        if not 0 <= int(damping_q) <= 0xFFFFFFFF:
            raise ValueError("damping_q must fit unsigned 32-bit")
        residuals = _pack_int16(residuals_q, "residuals_q")
        jacobian = np.asarray(jac_rows_q, dtype=np.int64)
        if jacobian.shape != (residuals.size, FPGA_MAX_CONTROLS):
            raise ValueError("jac_rows_q must have shape (rows, 6)")
        jacobian = _pack_int16(jacobian.reshape(-1), "jac_rows_q").reshape(
            residuals.size, FPGA_MAX_CONTROLS
        )
        if not 1 <= residuals.size <= FPGA_MAX_ROWS:
            raise ValueError(f"one to {FPGA_MAX_ROWS} residual rows required")
        valid_mask = (1 << int(residuals.size)) - 1
        if not 0 <= int(diagonal_row_mask) <= valid_mask:
            raise ValueError("diagonal_row_mask has invalid bits")
        vectors = [
            _pack_int16(vector, name)
            for name, vector in (
                ("theta_q", theta_q),
                ("lower_q", lower_q),
                ("upper_q", upper_q),
            )
        ]
        if any(vector.size != FPGA_MAX_CONTROLS for vector in vectors):
            raise ValueError("theta_q, lower_q, upper_q must each have length 6")
        payload = bytearray(int(gate_id).to_bytes(2, "little"))
        payload.extend(
            [
                (int(gate_kind) | (0x80 if full_metric else 0)) & 0xFF,
                (int(residuals.size) & 0x07) | ((int(diagonal_row_mask) & 0x1F) << 3),
            ]
        )
        payload.extend(_pack_int16([alpha_q, gain_q, max_step_q], "step_q").astype("<i2").tobytes())
        payload.extend(int(damping_q).to_bytes(4, "little", signed=False))
        payload.extend(np.concatenate(vectors).astype("<i2").tobytes())
        for row_index in range(FPGA_MAX_ROWS):
            if row_index < residuals.size:
                row = np.concatenate(([residuals[row_index]], jacobian[row_index]))
            else:
                row = np.zeros(1 + FPGA_MAX_CONTROLS, dtype=np.int16)
            payload.extend(np.asarray(row, dtype="<i2").tobytes())
        if len(payload) != FPGA_BIAS_PAYLOAD_BYTES:
            raise AssertionError("FPGA bias payload size drift")

        return bytes(payload)

    @staticmethod
    def _decode_bias_reply(reply: bytes) -> tuple[int, int, np.ndarray]:
        next_q = np.frombuffer(reply[3:], dtype="<i2").copy()
        return int.from_bytes(reply[:2], "little"), reply[2] & 0x7F, next_q

    def pulse_gate_bias_optimize(
        self,
        **record,
    ) -> tuple[int, int, np.ndarray]:
        """Send one deployed 0x0F record and return ``(id, kind, theta_q)``."""

        reply = self._request(0x0F, self._pack_bias_payload(**record), FPGA_REPLY_BYTES)
        return self._decode_bias_reply(reply)

    def pulse_gate_bias_optimize_batch(
        self,
        records: Sequence[dict[str, object]],
    ) -> list[tuple[int, int, np.ndarray]]:
        """Run up to four reply-independent 0x10 records in one UART request."""

        if not 1 <= len(records) <= FPGA_BIAS_BATCH_MAX_RECORDS:
            raise ValueError(
                f"batch requires one to {FPGA_BIAS_BATCH_MAX_RECORDS} records"
            )
        payload = bytes([len(records)]) + b"".join(
            self._pack_bias_payload(**record) for record in records
        )
        reply = self._request(
            FPGA_BIAS_BATCH_OPCODE,
            payload,
            FPGA_REPLY_BYTES * len(records),
        )
        return [
            self._decode_bias_reply(reply[offset : offset + FPGA_REPLY_BYTES])
            for offset in range(0, len(reply), FPGA_REPLY_BYTES)
        ]


def _pack_int16(values: Iterable[int], name: str) -> np.ndarray:
    array = np.asarray(list(values), dtype=np.int64).reshape(-1)
    if np.any(array < -32768) or np.any(array > 32767):
        raise ValueError(f"{name} contains values outside signed int16")
    return array.astype(np.int16)


def quantize_signed(values: Iterable[float], scale: int) -> np.ndarray:
    """Round finite real values into signed fixed point, matching BiasSteerer."""

    array = np.asarray(list(values), dtype=np.float64).reshape(-1)
    if int(scale) <= 0 or not np.all(np.isfinite(array)):
        raise ValueError("values must be finite and scale must be positive")
    rounded = np.rint(array * int(scale))
    if np.any(rounded < -32768) or np.any(rounded > 32768):
        raise ValueError("value does not fit signed FPGA fixed point")
    return np.clip(rounded, -32768, 32767).astype(np.int16)


def _rtl_trunc_division(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    quotient = abs(int(numerator)) // int(denominator)
    return -quotient if numerator < 0 else quotient


def _validate_fixed_rows(residuals_q, jac_rows_q) -> tuple[np.ndarray, np.ndarray]:
    residuals = np.asarray(residuals_q, dtype=np.int64).reshape(-1)
    jacobian = np.asarray(jac_rows_q, dtype=np.int64)
    if jacobian.shape != (residuals.size, FPGA_MAX_CONTROLS):
        raise ValueError("jac_rows_q must have shape (rows, 6)")
    if not 1 <= residuals.size <= FPGA_MAX_ROWS:
        raise ValueError(f"one to {FPGA_MAX_ROWS} matching rows required")
    return residuals, jacobian


def diagonal_gauss_newton_q(
    residuals_q,
    jac_rows_q,
    *,
    gain_q: int,
    damping_q: int,
) -> np.ndarray:
    """CPU reference for deployed diagonal normal-equation datapath."""

    residuals, jacobian = _validate_fixed_rows(residuals_q, jac_rows_q)
    gradient = np.sum(jacobian * residuals[:, None], axis=0, dtype=np.int64)
    metric = np.sum(jacobian * jacobian, axis=0, dtype=np.int64)
    output = np.zeros(FPGA_MAX_CONTROLS, dtype=np.int64)
    for index, (gradient_value, metric_value) in enumerate(zip(gradient, metric)):
        denominator = int(metric_value) + int(damping_q)
        if denominator and gradient_value and gain_q:
            magnitude = abs(int(gradient_value)) * abs(int(gain_q)) // denominator
            magnitude = min(32768, magnitude)
            sign = -1 if (gradient_value > 0) == (gain_q > 0) else 1
            output[index] = sign * magnitude
    return np.clip(output, -32768, 32767).astype(np.int16)


def structured_gauss_newton_q(
    residuals_q,
    jac_rows_q,
    *,
    gain_q: int,
    damping_q: int,
    sweeps: int = 4,
    diagonal_row_mask: int = 0,
) -> np.ndarray:
    """CPU reference for deployed four-coordinate coupled solver."""

    residuals, jacobian = _validate_fixed_rows(residuals_q, jac_rows_q)
    valid_mask = (1 << int(residuals.size)) - 1
    if not 0 <= int(diagonal_row_mask) <= valid_mask:
        raise ValueError("diagonal_row_mask has invalid bits")
    gradient = jacobian.T @ residuals
    metric = np.zeros((FPGA_MAX_CONTROLS, FPGA_MAX_CONTROLS), dtype=np.int64)
    for row_index, row in enumerate(jacobian):
        outer = np.outer(row, row)
        metric += np.diag(np.diag(outer))
        if not (int(diagonal_row_mask) & (1 << row_index)):
            metric[:FPGA_FULL_COUPLED_CONTROLS, :FPGA_FULL_COUPLED_CONTROLS] += (
                outer[:FPGA_FULL_COUPLED_CONTROLS, :FPGA_FULL_COUPLED_CONTROLS]
                - np.diag(np.diag(outer[:FPGA_FULL_COUPLED_CONTROLS, :FPGA_FULL_COUPLED_CONTROLS]))
            )
    velocity = np.zeros(FPGA_MAX_CONTROLS, dtype=np.int64)
    damping = int(damping_q) // 2
    for _ in range(int(sweeps)):
        for index in range(FPGA_MAX_CONTROLS):
            denominator = int(metric[index, index]) + damping
            if denominator <= 0:
                continue
            numerator = -int(gain_q) * int(gradient[index])
            numerator -= sum(
                int(metric[index, other]) * int(velocity[other])
                for other in range(FPGA_MAX_CONTROLS)
                if other != index
            )
            velocity[index] = max(
                -32768,
                min(32767, _rtl_trunc_division(numerator, denominator)),
            )
    return velocity.astype(np.int16)


def blended_structured_gauss_newton_q(
    residuals_q,
    jac_rows_q,
    *,
    gain_q: int,
    damping_q: int,
    sweeps: int = 4,
    diagonal_row_mask: int = 0,
) -> np.ndarray:
    diagonal = diagonal_gauss_newton_q(
        residuals_q, jac_rows_q, gain_q=gain_q, damping_q=damping_q
    ).astype(np.int64)
    structured = structured_gauss_newton_q(
        residuals_q,
        jac_rows_q,
        gain_q=gain_q,
        damping_q=damping_q,
        sweeps=sweeps,
        diagonal_row_mask=diagonal_row_mask,
    ).astype(np.int64)
    # Verilog >>> 1 is arithmetic shift, including negative odd deltas.
    return np.clip(diagonal + ((structured - diagonal) >> 1), -32768, 32767).astype(np.int16)


def bounded_retraction_step_q(
    theta_q: Sequence[int],
    velocity_q: Sequence[int],
    lower_q: Sequence[int],
    upper_q: Sequence[int],
    *,
    alpha_q: int,
    max_step_q: int,
) -> np.ndarray:
    """Match deployed bounded Q12 retraction; acceleration is unused."""

    theta = np.asarray(theta_q, dtype=np.int64).reshape(-1)
    velocity = np.asarray(velocity_q, dtype=np.int64).reshape(-1)
    lower = np.asarray(lower_q, dtype=np.int64).reshape(-1)
    upper = np.asarray(upper_q, dtype=np.int64).reshape(-1)
    if any(vector.shape != theta.shape for vector in (velocity, lower, upper)):
        raise ValueError("retraction vectors must have equal shape")
    cap = abs(int(max_step_q))
    output: list[int] = []
    for current, step, low, high in zip(theta, velocity, lower, upper):
        delta = (int(alpha_q) * int(step)) >> 12
        delta = max(-cap, min(cap, delta))
        output.append(max(int(low), min(int(high), int(current) + delta)))
    return np.clip(output, -32768, 32767).astype(np.int16)


def _fits(values: np.ndarray, scale: int) -> bool:
    return bool(np.all(np.isfinite(values)) and np.all(np.abs(values * scale) <= 32768.5))


def _quantized_target_with_budget(target: np.ndarray, budget: float) -> np.ndarray:
    """Quantize CPU target, correcting rare L2 overflow from coordinate rounding."""

    quantized = quantize_signed(target, FPGA_THETA_SCALE).astype(np.int64)
    for _ in range(FPGA_MAX_CONTROLS * 32768):
        values = quantized.astype(np.float64) / FPGA_THETA_SCALE
        norm = float(np.linalg.norm(values))
        if norm <= budget + 1.0e-15 or norm == 0.0:
            return quantized.astype(np.int16)
        index = int(np.argmax(np.abs(quantized)))
        if quantized[index] > 0:
            quantized[index] -= 1
        elif quantized[index] < 0:
            quantized[index] += 1
        else:
            break
    return quantized.astype(np.int16)


def _hardware_supported(problem: _BiasOptimizationProblem, config: FpgaOptimizerConfig) -> bool:
    controls = problem.design.shape[1]
    rows = problem.residual.size
    if rows < 1 or rows > config.max_rows or controls < 1 or controls > config.max_controls:
        return False
    if problem.budget * FPGA_THETA_SCALE > 32767.5:
        return False
    return (
        _fits(problem.residual, FPGA_RATE_SCALE)
        and _fits(problem.design, FPGA_JACOBIAN_SCALE)
        and config.gain * FPGA_GAIN_SCALE <= 32767.5
        and config.damping * FPGA_DAMPING_SCALE <= 0xFFFFFFFF
    )


def _cpu_result(
    local_generators: np.ndarray,
    transport: SparsePauliTransport | SparseCliffordTransport | AnchoredPauliTransport,
    local_jacobians: np.ndarray,
    effort_fraction: float,
) -> BiasOptimizationResult:
    return optimize_bias_controls(
        local_generators,
        transport,
        local_jacobians,
        effort_fraction=effort_fraction,
    )


def _project_l2_ball(values: np.ndarray, budget: float) -> np.ndarray:
    """Return ``values`` radially projected into one gate's L2 budget."""

    result = np.asarray(values, dtype=float)
    norm = float(np.linalg.norm(result))
    if budget <= 0.0:
        return np.zeros_like(result)
    if norm <= budget:
        return result
    return result * (budget / norm)


def _local_l2_solution(
    hessian: np.ndarray,
    gradient: np.ndarray,
    budget: float,
) -> np.ndarray:
    """Solve a small L2-constrained normal-equation problem stably.

    This is host-side ranking math only. The selected gate still receives its
    local fixed-point update from the FPGA.
    """

    if budget <= 0.0:
        return np.zeros_like(gradient, dtype=float)
    matrix = 0.5 * (np.asarray(hessian, dtype=float) + np.asarray(hessian, dtype=float).T)
    vector = np.asarray(gradient, dtype=float)
    values, vectors = np.linalg.eigh(matrix)
    values = np.maximum(values, 0.0)
    projected = vectors.T @ vector
    scale = max(float(np.max(values, initial=0.0)), 1.0)
    tolerance = np.finfo(float).eps * scale * max(1, values.size)

    def solution(regularization: float) -> np.ndarray:
        return -vectors @ (projected / (values + regularization))

    resolved = values > tolerance
    if np.all(np.abs(projected[~resolved]) <= tolerance):
        unconstrained = -vectors[:, resolved] @ (projected[resolved] / values[resolved])
        if float(np.linalg.norm(unconstrained)) <= budget:
            return unconstrained

    low = 0.0
    high = max(float(np.linalg.norm(vector)) / budget, 1.0)
    while float(np.linalg.norm(solution(high))) > budget:
        high *= 2.0
    for _ in range(48):
        middle = 0.5 * (low + high)
        if float(np.linalg.norm(solution(middle))) > budget:
            low = middle
        else:
            high = middle
    return solution(high)


def _gate_hessians(problem: _BiasOptimizationProblem) -> tuple[np.ndarray, np.ndarray]:
    """Return per-gate normal matrices and conservative local curvatures."""

    gate_count = problem.local.shape[0]
    control_count = problem.jacobians.shape[2]
    blocks = problem.design.reshape(problem.residual.size, gate_count, control_count)
    hessians = np.einsum("rgc,rgd->gcd", blocks, blocks, optimize=True)
    curvatures = np.linalg.eigvalsh(hessians)[:, -1]
    return hessians, np.maximum(curvatures, np.finfo(float).eps)


def _greedy_gate_scores(
    problem: _BiasOptimizationProblem,
    residual: np.ndarray,
    controls: np.ndarray,
    effort_fraction: float,
    hessians: np.ndarray,
    curvatures: np.ndarray,
) -> np.ndarray:
    """Score each gate by its predicted full-objective decrease.

    The score uses the current transported bias plus local control sensitivity,
    rather than a static local-error heuristic. It therefore reprioritizes a
    gate whenever other accepted updates make its error contribution larger.
    """

    gate_count = problem.local.shape[0]
    control_count = problem.jacobians.shape[2]
    gradient = (problem.design.T @ residual).reshape(gate_count, control_count)
    base_gradient = gradient - np.einsum("gij,gj->gi", hessians, controls)
    candidate = controls - base_gradient / curvatures[:, None]
    budgets = effort_fraction * np.linalg.norm(problem.local, axis=1)
    candidate_norms = np.linalg.norm(candidate, axis=1)
    scales = np.ones(gate_count, dtype=float)
    active = candidate_norms > budgets
    scales[active] = budgets[active] / candidate_norms[active]
    candidate *= scales[:, None]
    delta = candidate - controls
    scores = -2.0 * np.einsum("gc,gc->g", delta, gradient)
    scores -= np.einsum("gi,gij,gj->g", delta, hessians, delta)
    return np.maximum(scores, 0.0)


def _max_record_scale(config: FpgaOptimizerConfig) -> float:
    """Largest common record scale that keeps Q30 damping representable."""

    if config.damping == 0.0:
        return float("inf")
    return float(np.sqrt(0xFFFFFFFF / (config.damping * FPGA_DAMPING_SCALE)))


def _compress_gate_rows(
    residual: np.ndarray,
    jacobian_block: np.ndarray,
    config: FpgaOptimizerConfig,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compress every local bias row into at most five equivalent rows.

    A gate update only consumes ``J.T @ J`` and ``J.T @ residual``. Eigen rows
    preserve those normal equations exactly when their rank fits the record;
    a six-control gate retains its five strongest local directions.
    """

    block = np.asarray(jacobian_block, dtype=float)
    current = np.asarray(residual, dtype=float)
    hessian = 0.5 * (block.T @ block + (block.T @ block).T)
    gradient = block.T @ current
    values, vectors = np.linalg.eigh(hessian)
    values = np.maximum(values, 0.0)
    scale = max(float(np.max(values, initial=0.0)), 1.0)
    tolerance = np.finfo(float).eps * scale * max(1, values.size)
    indices = np.flatnonzero(values > tolerance)[::-1][:config.max_rows]
    if indices.size == 0:
        return np.zeros(1, dtype=float), np.zeros((1, block.shape[1]), dtype=float), 1.0

    selected_values = values[indices]
    selected_vectors = vectors[:, indices]
    square_roots = np.sqrt(selected_values)
    compressed_jacobian = square_roots[:, None] * selected_vectors.T
    compressed_residual = (selected_vectors.T @ gradient) / square_roots
    amplitude = max(
        float(np.max(np.abs(compressed_jacobian))),
        float(np.max(np.abs(compressed_residual))),
        np.finfo(float).tiny,
    )
    record_scale = min(
        float(config.compression_headroom) / amplitude,
        _max_record_scale(config),
    )
    return (
        compressed_residual * record_scale,
        compressed_jacobian * record_scale,
        record_scale,
    )


def _greedy_cpu_result(
    problem: _BiasOptimizationProblem,
    transport: SparsePauliTransport | SparseCliffordTransport | AnchoredPauliTransport,
    effort_fraction: float,
) -> BiasOptimizationResult:
    """CPU fallback matching the multi-sweep gate-local optimization policy."""

    local = problem.local.copy()
    updated = local.copy()
    residual = problem.residual.copy()
    control_count = problem.jacobians.shape[2]
    controls = np.zeros((local.shape[0], control_count), dtype=float)
    hessians, curvatures = _gate_hessians(problem)
    for _ in range(FpgaOptimizerConfig().max_sweeps):
        remaining = set(range(local.shape[0]))
        accepted = False
        while remaining:
            scores = _greedy_gate_scores(
                problem, residual, controls, effort_fraction, hessians, curvatures,
            )
            gate = max(remaining, key=lambda index: (float(scores[index]), -index))
            remaining.remove(gate)
            if scores[gate] <= 0.0:
                continue
            block = problem.design[:, gate * control_count : (gate + 1) * control_count]
            gate_budget = effort_fraction * float(np.linalg.norm(problem.local[gate]))
            base = residual - block @ controls[gate]
            next_controls = _local_l2_solution(block.T @ block, block.T @ base, gate_budget)
            candidate = residual + block @ (next_controls - controls[gate])
            if float(np.linalg.norm(candidate)) >= float(np.linalg.norm(residual)) - 1.0e-12:
                continue
            controls[gate] = next_controls
            updated[gate] = problem.local[gate] + problem.jacobians[gate] @ next_controls
            residual = problem.residual + problem.design @ controls.reshape(-1)
            accepted = True
        if not accepted:
            break

    truncation_bound = (
        transport.truncation_bound_for(updated)
        if isinstance(transport, (SparsePauliTransport, AnchoredPauliTransport))
        else 0.0
    )
    return BiasOptimizationResult(
        controls=controls,
        updated_generators=updated,
        predicted_before_norm=float(np.linalg.norm(problem.residual)),
        predicted_after_norm=float(np.linalg.norm(residual)),
        control_norm=float(np.linalg.norm(controls)),
        budget=problem.budget,
        truncation_bound=truncation_bound,
    )


def _greedy_hardware_supported(
    problem: _BiasOptimizationProblem,
    config: FpgaOptimizerConfig,
) -> bool:
    """Return whether every greedy gate record fits the deployed payload."""

    rows = problem.residual.size
    control_count = problem.jacobians.shape[2]
    if rows < 1 or control_count < 1 or control_count > config.max_controls:
        return False
    if problem.local.shape[0] > 0x10000:
        return False
    for gate in range(problem.local.shape[0]):
        block = problem.design[:, gate * control_count : (gate + 1) * control_count]
        compressed_residual, compressed_jacobian, _ = _compress_gate_rows(
            problem.residual, block, config,
        )
        if not _fits(compressed_residual, FPGA_RATE_SCALE):
            return False
        if not _fits(compressed_jacobian, FPGA_JACOBIAN_SCALE):
            return False
        gate_budget = problem.budget * float(np.linalg.norm(problem.local[gate])) / max(
            float(np.linalg.norm(problem.local)), np.finfo(float).eps,
        )
        if gate_budget * FPGA_THETA_SCALE > 32767.5:
            return False
    return True


def _greedy_gate_update_fpga(
    *,
    gate_id: int,
    gate_kind: int,
    current_controls: np.ndarray,
    gate_budget: float,
    residual: np.ndarray,
    jacobian_block: np.ndarray,
    active_link: FpgaSolverLink | None,
    config: FpgaOptimizerConfig,
    metric_mode: bool,
    record_scale: float,
) -> np.ndarray:
    """Run one fixed-point gate update and return its absolute controls."""

    control_count = current_controls.size
    coordinate_bound = gate_budget
    bound_q = min(
        32767,
        max(1, int(np.rint(coordinate_bound * FPGA_THETA_SCALE))),
    )
    theta_q = np.zeros(FPGA_MAX_CONTROLS, dtype=np.int16)
    lower_q = np.zeros(FPGA_MAX_CONTROLS, dtype=np.int16)
    upper_q = np.zeros(FPGA_MAX_CONTROLS, dtype=np.int16)
    theta_q[:control_count] = quantize_signed(current_controls, FPGA_THETA_SCALE)
    lower_q[:control_count] = -bound_q
    upper_q[:control_count] = bound_q

    residuals_q = quantize_signed(residual, FPGA_RATE_SCALE)
    jacobian_q = quantize_signed(
        jacobian_block.reshape(-1), FPGA_JACOBIAN_SCALE
    ).reshape(residual.size, control_count)
    jacobian_pad = np.zeros((residual.size, FPGA_MAX_CONTROLS), dtype=np.int16)
    jacobian_pad[:, :control_count] = jacobian_q
    gain_q = int(np.rint(config.gain * FPGA_GAIN_SCALE))
    damping_q = max(1, int(np.rint(
        config.damping * record_scale * record_scale * FPGA_DAMPING_SCALE,
    )))

    if active_link is None:
        if metric_mode:
            velocity_q = blended_structured_gauss_newton_q(
                residuals_q,
                jacobian_pad,
                gain_q=gain_q,
                damping_q=damping_q,
            )
        else:
            velocity_q = diagonal_gauss_newton_q(
                residuals_q,
                jacobian_pad,
                gain_q=gain_q,
                damping_q=damping_q,
            )
        next_q = bounded_retraction_step_q(
            theta_q,
            velocity_q,
            lower_q,
            upper_q,
            alpha_q=FPGA_THETA_SCALE,
            max_step_q=bound_q,
        )
    else:
        reply_id, reply_kind, next_q = active_link.pulse_gate_bias_optimize(
            gate_id=gate_id,
            gate_kind=gate_kind,
            alpha_q=FPGA_THETA_SCALE,
            gain_q=gain_q,
            max_step_q=bound_q,
            damping_q=damping_q,
            theta_q=theta_q,
            lower_q=lower_q,
            upper_q=upper_q,
            residuals_q=residuals_q,
            jac_rows_q=jacobian_pad,
            full_metric=metric_mode,
            diagonal_row_mask=0,
        )
        if int(reply_id) != gate_id or int(reply_kind) != gate_kind:
            raise RuntimeError("FPGA greedy reply identity mismatch")
        next_q = _pack_int16(next_q, "FPGA reply")
        if next_q.size != FPGA_MAX_CONTROLS:
            raise RuntimeError("FPGA greedy reply must contain six controls")

    return _project_l2_ball(
        np.clip(
            next_q[:control_count].astype(np.float64) / FPGA_THETA_SCALE,
            -coordinate_bound,
            coordinate_bound,
        ),
        gate_budget,
    )


def optimize_bias_controls_fpga_greedy(
    local_generators: np.ndarray,
    transport: SparsePauliTransport | SparseCliffordTransport | AnchoredPauliTransport,
    local_jacobians: np.ndarray,
    effort_fraction: float = 0.25,
    *,
    link: FpgaSolverLink | None = None,
    port: str | None = None,
    baud: int = FPGA_BAUD,
    config: FpgaOptimizerConfig = FpgaOptimizerConfig(),
    full_metric: bool | None = None,
    gate_kind: int = 1,
    fallback: Literal["cpu", "error"] = "cpu",
) -> BiasOptimizationResult:
    """Greedily optimize gates through the connected FPGA.

    Gates are selected repeatedly from the current transported bias estimate.
    The score combines each gate's transported generator contribution with the
    norm of its control gradient, so large error generators with an effective
    local control direction are serviced first. After every FPGA update the
    end-bias residual is recomputed before selecting the next gate.

    Each gate receives a budget of ``effort_fraction * ||theta_gate||``. A
    conservative per-coordinate bound keeps the sum of gate-control norms
    inside the same global CPU effort budget. ``link=None`` runs the deployed
    fixed-point datapath reference; ``port`` keeps one connected link open for
    the whole greedy pass. If the fixed FPGA record cannot represent a gate,
    ``fallback="cpu"`` runs the same greedy ordering with the CPU trust-region
    step, while ``fallback="error"`` raises.
    """

    if fallback not in {"cpu", "error"}:
        raise ValueError("fallback must be 'cpu' or 'error'")
    if link is not None and port is not None:
        raise ValueError("pass link or port, not both")
    if int(gate_kind) not in {1, 2, 3}:
        raise ValueError("gate_kind must be 1, 2, or 3")

    problem = _build_bias_optimization_problem(
        local_generators, transport, local_jacobians, effort_fraction,
    )
    if problem.budget == 0.0:
        return _greedy_cpu_result(problem, transport, effort_fraction)
    if not _greedy_hardware_supported(problem, config):
        if fallback == "error":
            raise ValueError(
                "greedy problem exceeds Tang Nano fixed payload/quantization limits"
            )
        return _greedy_cpu_result(problem, transport, effort_fraction)

    metric_mode = config.full_metric if full_metric is None else bool(full_metric)
    owned_link: FPGALink | None = None
    active_link: FpgaSolverLink | None = link
    if port is not None:
        owned_link = FPGALink(port, baud=baud)
        active_link = owned_link
        try:
            if owned_link.ping() != FPGA_PING_REPLY:
                raise RuntimeError("FPGA ping mismatch; expected 0xA5")
        except Exception:
            owned_link.close()
            raise

    control_count = problem.jacobians.shape[2]
    controls = np.zeros((problem.local.shape[0], control_count), dtype=float)
    updated = problem.local.copy()
    residual = problem.residual.copy()
    hessians, curvatures = _gate_hessians(problem)
    try:
        for _ in range(config.max_sweeps):
            remaining = set(range(problem.local.shape[0]))
            accepted = False
            while remaining:
                scores = _greedy_gate_scores(
                    problem, residual, controls, effort_fraction, hessians, curvatures,
                )
                gate = max(remaining, key=lambda index: (float(scores[index]), -index))
                remaining.remove(gate)
                if scores[gate] <= config.improvement_tolerance:
                    continue
                gate_budget = effort_fraction * float(
                    np.linalg.norm(problem.local[gate])
                )
                block = problem.design[
                    :, gate * control_count : (gate + 1) * control_count
                ]
                if gate_budget == 0.0 or not np.any(block):
                    continue
                compressed_residual, compressed_jacobian, record_scale = _compress_gate_rows(
                    residual, block, config,
                )
                next_controls = _greedy_gate_update_fpga(
                    gate_id=gate,
                    gate_kind=int(gate_kind),
                    current_controls=controls[gate],
                    gate_budget=gate_budget,
                    residual=compressed_residual,
                    jacobian_block=compressed_jacobian,
                    active_link=active_link,
                    config=config,
                    metric_mode=metric_mode,
                    record_scale=record_scale,
                )
                candidate = residual + block @ (next_controls - controls[gate])
                if float(np.linalg.norm(candidate)) >= float(np.linalg.norm(residual)) - config.improvement_tolerance:
                    continue
                controls[gate] = next_controls
                updated[gate] = problem.local[gate] + problem.jacobians[gate] @ next_controls
                residual = problem.residual + problem.design @ controls.reshape(-1)
                accepted = True
            if not accepted:
                break
    finally:
        if owned_link is not None:
            owned_link.close()

    truncation_bound = (
        transport.truncation_bound_for(updated)
        if isinstance(transport, (SparsePauliTransport, AnchoredPauliTransport))
        else 0.0
    )
    return BiasOptimizationResult(
        controls=controls,
        updated_generators=updated,
        predicted_before_norm=float(np.linalg.norm(problem.residual)),
        predicted_after_norm=float(np.linalg.norm(residual)),
        control_norm=float(np.linalg.norm(controls)),
        budget=problem.budget,
        truncation_bound=truncation_bound,
    )


def optimize_bias_controls_fpga(
    local_generators: np.ndarray,
    transport: SparsePauliTransport | SparseCliffordTransport | AnchoredPauliTransport,
    local_jacobians: np.ndarray,
    effort_fraction: float = 0.25,
    *,
    link: FpgaSolverLink | None = None,
    port: str | None = None,
    baud: int = FPGA_BAUD,
    config: FpgaOptimizerConfig = FpgaOptimizerConfig(),
    full_metric: bool | None = None,
    fallback: Literal["cpu", "error"] = "cpu",
) -> BiasOptimizationResult:
    """Optimize through Tang Nano while preserving BiasBlaster's API.

    ``local_generators``, ``transport``, ``local_jacobians``, and
    ``effort_fraction`` have exactly the CPU optimizer contract. FPGA sees the
    resulting end-bias residual and its linearized control design matrix. The
    repository trust-region solve computes the target first. FPGA applies the
    Q12 target under equal lower/upper bounds, returns it, and host verifies
    parity. This avoids silently substituting BiasSteerer's different damped
    Gauss-Newton objective.

    ``link=None`` runs the bit-exact Python reference, useful for tests. Set
    ``port="COM6"`` or ``BIASBLASTER_FPGA_PORT`` for the attached board.
    """

    if fallback not in {"cpu", "error"}:
        raise ValueError("fallback must be 'cpu' or 'error'")
    if link is not None and port is not None:
        raise ValueError("pass link or port, not both")
    problem = _build_bias_optimization_problem(
        local_generators, transport, local_jacobians, effort_fraction,
    )
    cpu_result = _cpu_result(
        local_generators, transport, local_jacobians, effort_fraction,
    )
    if problem.budget == 0.0:
        return cpu_result
    if not _hardware_supported(problem, config):
        if fallback == "error":
            raise ValueError(
                "problem exceeds Tang Nano fixed payload/quantization limits"
            )
        return cpu_result

    target = cpu_result.controls.reshape(-1)
    if not _fits(target, FPGA_THETA_SCALE):
        if fallback == "error":
            raise ValueError("CPU optimizer target does not fit FPGA Q12")
        return cpu_result

    metric_mode = config.full_metric if full_metric is None else bool(full_metric)
    controls = problem.design.shape[1]
    rows = problem.residual.size
    residuals_q = quantize_signed(problem.residual, FPGA_RATE_SCALE)
    jacobian_q = quantize_signed(problem.design.reshape(-1), FPGA_JACOBIAN_SCALE).reshape(
        rows, controls
    )
    residuals_pad = residuals_q
    jacobian_pad = np.zeros((rows, FPGA_MAX_CONTROLS), dtype=np.int16)
    jacobian_pad[:, :controls] = jacobian_q
    target_q = _quantized_target_with_budget(target, problem.budget)
    theta_q = np.zeros(FPGA_MAX_CONTROLS, dtype=np.int16)
    lower_q = np.zeros(FPGA_MAX_CONTROLS, dtype=np.int16)
    upper_q = np.zeros(FPGA_MAX_CONTROLS, dtype=np.int16)
    theta_q[:controls] = target_q
    lower_q[:controls] = target_q
    upper_q[:controls] = target_q
    # ``max_step_q`` is a signed int16 in the deployed record. Keep the
    # rounded budget representable at the positive end of that type.
    budget_q = min(32767, max(1, int(np.rint(problem.budget * FPGA_THETA_SCALE))))
    alpha_q = FPGA_THETA_SCALE
    gain_q = int(np.rint(config.gain * FPGA_GAIN_SCALE))
    damping_q = max(1, int(np.rint(config.damping * FPGA_DAMPING_SCALE)))
    max_step_q = budget_q
    diagonal_row_mask = 0

    owned_link: FPGALink | None = None
    active_link: FpgaSolverLink | None = link
    if port is not None:
        owned_link = FPGALink(port, baud=baud)
        active_link = owned_link
        try:
            if owned_link.ping() != FPGA_PING_REPLY:
                raise RuntimeError("FPGA ping mismatch; expected 0xA5")
        except Exception:
            owned_link.close()
            raise
    try:
        if active_link is None:
            if metric_mode:
                velocity_q = blended_structured_gauss_newton_q(
                    residuals_pad,
                    jacobian_pad,
                    gain_q=gain_q,
                    damping_q=damping_q,
                    diagonal_row_mask=diagonal_row_mask,
                )
            else:
                velocity_q = diagonal_gauss_newton_q(
                    residuals_pad,
                    jacobian_pad,
                    gain_q=gain_q,
                    damping_q=damping_q,
                )
            next_q = bounded_retraction_step_q(
                theta_q,
                velocity_q,
                lower_q,
                upper_q,
                alpha_q=alpha_q,
                max_step_q=max_step_q,
            )
        else:
            reply_id, reply_kind, next_q = active_link.pulse_gate_bias_optimize(
                gate_id=0,
                gate_kind=1,
                alpha_q=alpha_q,
                gain_q=gain_q,
                max_step_q=max_step_q,
                damping_q=damping_q,
                theta_q=theta_q,
                lower_q=lower_q,
                upper_q=upper_q,
                residuals_q=residuals_pad,
                jac_rows_q=jacobian_pad,
                full_metric=metric_mode,
                diagonal_row_mask=diagonal_row_mask,
            )
            if int(reply_id) != 0 or int(reply_kind) != 1:
                raise RuntimeError("FPGA optimizer reply identity mismatch")
            next_q = _pack_int16(next_q, "FPGA reply")
            if next_q.size != FPGA_MAX_CONTROLS:
                raise RuntimeError("FPGA optimizer reply must contain six controls")
        if not np.array_equal(next_q[:controls], target_q):
            raise RuntimeError("FPGA target update mismatch")
    finally:
        if owned_link is not None:
            owned_link.close()

    flat_controls = next_q[:controls].astype(np.float64) / FPGA_THETA_SCALE
    control_norm = float(np.linalg.norm(flat_controls))
    controls_out = flat_controls.reshape(problem.local.shape[0], problem.jacobians.shape[2])
    updated = problem.local + np.einsum(
        "gmc,gc->gm", problem.jacobians, controls_out
    )
    after = problem.residual + problem.design @ flat_controls
    truncation_bound = (
        transport.truncation_bound_for(updated)
        if isinstance(transport, (SparsePauliTransport, AnchoredPauliTransport))
        else 0.0
    )
    return BiasOptimizationResult(
        controls=controls_out,
        updated_generators=updated,
        predicted_before_norm=float(np.linalg.norm(problem.residual)),
        predicted_after_norm=float(np.linalg.norm(after)),
        control_norm=control_norm,
        budget=problem.budget,
        truncation_bound=truncation_bound,
    )


__all__ = [
    "FPGA_MAX_ROWS",
    "FPGA_MAX_CONTROLS",
    "FpgaOptimizerConfig",
    "FPGALink",
    "FpgaSolverLink",
    "resolve_fpga_port",
    "list_fpga_ports",
    "quantize_signed",
    "diagonal_gauss_newton_q",
    "structured_gauss_newton_q",
    "blended_structured_gauss_newton_q",
    "bounded_retraction_step_q",
    "optimize_bias_controls_fpga",
    "optimize_bias_controls_fpga_greedy",
]
