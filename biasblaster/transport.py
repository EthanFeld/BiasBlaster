"""Bounded sparse Pauli transport for ordinary ordered circuits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from scipy.sparse import csr_matrix

from .local_transfer import conjugate_sparse_modes, local_pauli_transfer
from .model import CircuitOperation
from .nearest_clifford import (
    SparseCliffordTransport,
    _embed_tableau,
    closest_clifford,
    estimate_nearest_clifford_transport,
)
from .pauli import embed_pauli_label, extract_pauli_label, local_generator_labels, pauli_labels
from .tableau import CliffordTableau


_GATE_ACTION_CACHE: dict[
    tuple[int, str, tuple[tuple[tuple[int, ...], bytes], ...]],
    tuple[
        tuple[tuple[tuple[str, ...], np.ndarray, CliffordTableau], ...] | None,
        SparseCliffordTransport,
    ],
] = {}


def clear_transport_cache() -> None:
    """Clear cached gate actions, primarily for reproducible benchmarking."""

    _GATE_ACTION_CACHE.clear()


def _gate_action_cache_key(
    operations: tuple[CircuitOperation, ...], n_qubits: int, basis: str,
) -> tuple[int, str, tuple[tuple[tuple[int, ...], bytes], ...]]:
    return (
        n_qubits,
        basis,
        tuple((operation.qubits, operation.unitary.tobytes()) for operation in operations),
    )


def _cached_nearest_backbone(
    operations: tuple[CircuitOperation, ...], n_qubits: int, basis: str,
) -> SparseCliffordTransport:
    key = _gate_action_cache_key(operations, n_qubits, basis)
    cached = _GATE_ACTION_CACHE.get(key)
    if cached is not None:
        return cached[1]
    backbone = estimate_nearest_clifford_transport(
        operations, n_qubits, two_qubit_generator_basis=basis,
    )
    if len(_GATE_ACTION_CACHE) >= 16:
        _GATE_ACTION_CACHE.pop(next(iter(_GATE_ACTION_CACHE)))
    _GATE_ACTION_CACHE[key] = (None, backbone)
    return backbone


def _cached_nearest_actions(
    operations: tuple[CircuitOperation, ...], n_qubits: int, basis: str,
) -> tuple[tuple[tuple[tuple[str, ...], np.ndarray, CliffordTableau], ...], SparseCliffordTransport]:
    """Cache immutable gate transfer/tableau work across error-model rescoring."""

    key = _gate_action_cache_key(operations, n_qubits, basis)
    cached = _GATE_ACTION_CACHE.get(key)
    if cached is not None and cached[0] is not None:
        return cached
    actions = tuple(
        (
            pauli_labels(len(operation.qubits)),
            local_pauli_transfer(operation),
            _embed_tableau(closest_clifford(operation.unitary), operation.qubits, n_qubits),
        )
        for operation in operations
    )
    backbone = cached[1] if cached is not None else estimate_nearest_clifford_transport(
        operations, n_qubits, two_qubit_generator_basis=basis,
    )
    # Benchmark corpora use a handful of stable circuits; keep cache bounded
    # for library callers that stream unrelated circuits.
    if len(_GATE_ACTION_CACHE) >= 16:
        _GATE_ACTION_CACHE.pop(next(iter(_GATE_ACTION_CACHE)))
    _GATE_ACTION_CACHE[key] = (actions, backbone)
    return actions, backbone


def _validate_operations(operations: Sequence[CircuitOperation], n_qubits: int) -> tuple[CircuitOperation, ...]:
    values = tuple(operations)
    if n_qubits < 1 or not values:
        raise ValueError("operations and a positive n_qubits are required")
    for operation in values:
        if not isinstance(operation, CircuitOperation):
            raise TypeError("operations must contain CircuitOperation values")
        if len(operation.qubits) not in (1, 2):
            raise ValueError("only one- and two-qubit operations are supported")
        if any(qubit >= n_qubits for qubit in operation.qubits):
            raise ValueError("operation qubit outside register")
    return values


def _validate_active(active_gate_indices: Sequence[int] | None, gate_count: int) -> tuple[int, ...]:
    active = tuple(range(gate_count)) if active_gate_indices is None else tuple(int(index) for index in active_gate_indices)
    if not active or len(set(active)) != len(active) or any(index < 0 or index >= gate_count for index in active):
        raise ValueError("active_gate_indices must contain unique valid gate indices")
    return active


def _prune_with_dropped(
    coefficients: dict[str, np.ndarray],
    support_cap: int | None,
    *,
    coefficient_tol: float,
    selection_matrix: np.ndarray | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], float]:
    retained: dict[str, np.ndarray] = {}
    dropped: dict[str, np.ndarray] = {}
    # Dropped rows form a linear map from active local-generator coordinates
    # to Pauli coefficients. Its spectral norm is bounded by its Frobenius
    # norm. This is tighter than summing every dropped row norm, while still
    # certifying the omitted bias for every generator vector.
    dropped_squared_norm = 0.0
    for label, values in coefficients.items():
        if np.max(np.abs(values), initial=0.0) > coefficient_tol:
            retained[label] = values
        else:
            dropped[label] = values
            dropped_squared_norm += float(np.vdot(values, values).real)
    if support_cap is None or len(retained) <= support_cap:
        return retained, dropped, float(np.sqrt(dropped_squared_norm))
    ranked = sorted(
        retained,
        key=lambda label: (
            -float(np.linalg.norm(retained[label] @ selection_matrix))
            if selection_matrix is not None
            else -float(np.linalg.norm(retained[label])),
            -float(np.linalg.norm(retained[label])),
            label,
        ),
    )
    keep = set(ranked[:support_cap])
    for label, values in retained.items():
        if label not in keep:
            dropped[label] = values
            dropped_squared_norm += float(np.vdot(values, values).real)
    return (
        {label: values for label, values in retained.items() if label in keep},
        dropped,
        float(np.sqrt(dropped_squared_norm)),
    )


def _prune(
    coefficients: dict[str, np.ndarray],
    support_cap: int | None,
    *,
    coefficient_tol: float,
    selection_matrix: np.ndarray | None = None,
) -> tuple[dict[str, np.ndarray], float]:
    retained, _, dropped = _prune_with_dropped(
        coefficients,
        support_cap,
        coefficient_tol=coefficient_tol,
        selection_matrix=selection_matrix,
    )
    return retained, dropped


def _validate_selection_matrix(
    selection_matrix: np.ndarray | None, width: int,
) -> np.ndarray | None:
    if selection_matrix is None:
        return None
    values = np.asarray(selection_matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] != width or values.shape[1] < 1:
        raise ValueError("selection_matrix must have shape (active_gate_modes, signals)")
    if not np.all(np.isfinite(values)):
        raise ValueError("selection_matrix must be finite")
    return values


def _merge(target: dict[str, np.ndarray], label: str, values: np.ndarray) -> None:
    if label in target:
        target[label] += values
    else:
        target[label] = values.copy()


@dataclass(frozen=True)
class SparsePauliTransport:
    """Sparse end-Pauli transport columns indexed by local generator modes.

    ``coefficients[label]`` is a vector with one entry per active gate/mode.
    The representation never creates a dense ``4**n`` output array.
    """

    coefficients: Mapping[str, np.ndarray]
    n_qubits: int
    gate_count: int
    active_gate_indices: tuple[int, ...]
    support_count: int
    truncation_bound: float = 0.0
    generator_mode_count: int = 3
    mode_counts: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.n_qubits < 1 or self.gate_count < 1:
            raise ValueError("transport dimensions must be positive")
        active = _validate_active(self.active_gate_indices, self.gate_count)
        if self.truncation_bound < 0 or not np.isfinite(self.truncation_bound):
            raise ValueError("truncation_bound must be finite and nonnegative")
        if self.generator_mode_count not in (3, 15):
            raise ValueError("generator_mode_count must be 3 or 15")
        mode_counts = self.mode_counts or (self.generator_mode_count,) * self.gate_count
        if len(mode_counts) != self.gate_count or any(count < 1 or count > self.generator_mode_count for count in mode_counts):
            raise ValueError("mode_counts must match gates and generator_mode_count")
        width = len(active) * self.generator_mode_count
        copied: dict[str, np.ndarray] = {}
        for label, values in self.coefficients.items():
            label = str(label)
            if len(label) != self.n_qubits or any(character not in "IXYZ" for character in label):
                raise ValueError("invalid Pauli label in transport")
            vector = np.asarray(values, dtype=float)
            if vector.shape != (width,) or not np.all(np.isfinite(vector)):
                raise ValueError("transport columns must have shape (active_gate_count * 3,)")
            if np.any(vector):
                copied[label] = vector.copy()
        if self.support_count != len(copied):
            raise ValueError("support_count must equal the retained Pauli support")
        object.__setattr__(self, "active_gate_indices", active)
        object.__setattr__(self, "coefficients", copied)
        object.__setattr__(self, "mode_counts", tuple(mode_counts))

    @property
    def support(self) -> int:
        return self.support_count

    @property
    def retained_payload_bytes(self) -> int:
        """Return numeric-array payload bytes, not Python object memory."""

        return int(sum(values.nbytes for values in self.coefficients.values()))

    @property
    def retained_bytes(self) -> int:
        """Compatibility alias for :attr:`retained_payload_bytes`."""

        return self.retained_payload_bytes

    def _active_generators(self, generators: np.ndarray) -> np.ndarray:
        local = np.asarray(generators, dtype=float)
        if local.shape != (self.gate_count, self.generator_mode_count) or not np.all(np.isfinite(local)):
            raise ValueError("generators have wrong shape or nonfinite values")
        if any(np.any(local[gate, count:]) for gate, count in enumerate(self.mode_counts)):
            raise ValueError("inactive generator modes must be zero")
        return local[list(self.active_gate_indices)]

    def contract(self, generators: np.ndarray) -> dict[str, float]:
        """Contract local generator coefficients into a sparse end bias."""

        local = self._active_generators(generators).reshape(-1)
        return {
            label: float(np.dot(values, local))
            for label, values in self.coefficients.items()
            if abs(float(np.dot(values, local))) > 1e-12
        }

    def local_priorities(self, generators: np.ndarray) -> np.ndarray:
        """Return ``M_g.T @ b`` for the current transported bias ``b``."""

        self._active_generators(generators)
        bias = self.contract(generators)
        priority = np.zeros((self.gate_count, self.generator_mode_count), dtype=float)
        if bias:
            for index, gate in enumerate(self.active_gate_indices):
                for mode in range(self.generator_mode_count):
                    priority[gate, mode] = sum(
                        float(self.coefficients[label].reshape(-1)[index * self.generator_mode_count + mode]) * value
                        for label, value in bias.items()
                    )
        return priority

    def truncation_bound_for(self, generators: np.ndarray) -> float:
        return float(self.truncation_bound * np.linalg.norm(self._active_generators(generators)))

    def compact_transfer(self) -> np.ndarray:
        """Return a bounded ``(active_gate, support, mode)`` adapter."""

        labels = tuple(self.coefficients)
        return np.stack([values.reshape(len(self.active_gate_indices), self.generator_mode_count) for values in self.coefficients.values()], axis=1) if labels else np.empty((len(self.active_gate_indices), 0, self.generator_mode_count))


@dataclass(frozen=True)
class AnchoredPauliTransport:
    """Native nearest-Clifford backbone plus CSR exact-correction rows."""

    backbone: SparseCliffordTransport
    corrections: Mapping[str, csr_matrix]
    active_gate_indices: tuple[int, ...]
    # ``None`` means the exact-correction omission was not materialized, so no
    # exact-transport certificate is available (the cap-zero Clifford case).
    truncation_bound: float | None = 0.0

    def __post_init__(self) -> None:
        active = _validate_active(self.active_gate_indices, self.backbone.gate_count)
        width = len(active) * self.backbone.generator_mode_count
        copied: dict[str, csr_matrix] = {}
        for label, values in self.corrections.items():
            row = values.tocsr().astype(float, copy=True)
            if row.shape != (1, width) or not np.all(np.isfinite(row.data)):
                raise ValueError("correction rows must be finite shape (1, active_gate_modes)")
            row.eliminate_zeros()
            copied[str(label)] = row
        if self.truncation_bound is not None and (
            self.truncation_bound < 0.0 or not np.isfinite(self.truncation_bound)
        ):
            raise ValueError("truncation_bound must be finite and nonnegative")
        object.__setattr__(self, "active_gate_indices", active)
        object.__setattr__(self, "corrections", copied)

    @property
    def n_qubits(self) -> int:
        return self.backbone.n_qubits

    @property
    def gate_count(self) -> int:
        return self.backbone.gate_count

    @property
    def generator_mode_count(self) -> int:
        return self.backbone.generator_mode_count

    @property
    def mode_counts(self) -> tuple[int, ...]:
        return self.backbone.mode_counts

    @property
    def correction_support_count(self) -> int:
        return len(self.corrections)

    @property
    def support_count(self) -> int:
        labels = set(self.corrections)
        for gate in self.active_gate_indices:
            labels.update(self.backbone.mode_labels[gate][:self.backbone.mode_counts[gate]])
        return len(labels)

    @property
    def support(self) -> int:
        return self.support_count

    @property
    def retained_payload_bytes(self) -> int:
        """Return numeric-array payload bytes, not Python object memory."""

        correction_bytes = sum(
            row.data.nbytes + row.indices.nbytes + row.indptr.nbytes
            for row in self.corrections.values()
        )
        return self.backbone.retained_payload_bytes + correction_bytes

    @property
    def retained_bytes(self) -> int:
        """Compatibility alias for :attr:`retained_payload_bytes`."""

        return self.retained_payload_bytes

    def _active_generators(self, generators: np.ndarray) -> np.ndarray:
        local = np.asarray(generators, dtype=float)
        if local.shape != (self.gate_count, self.generator_mode_count) or not np.all(np.isfinite(local)):
            raise ValueError("generators have wrong shape or nonfinite values")
        if any(np.any(local[gate, count:]) for gate, count in enumerate(self.mode_counts)):
            raise ValueError("inactive generator modes must be zero")
        return local[list(self.active_gate_indices)].reshape(-1)

    def contract(self, generators: np.ndarray) -> dict[str, float]:
        local = np.asarray(generators, dtype=float)
        active_values = self._active_generators(local)
        result: dict[str, float] = {}
        for gate in self.active_gate_indices:
            modes = self.backbone.mode_labels[gate]
            count = self.backbone.mode_counts[gate]
            for mode, label in enumerate(modes[:count]):
                result[label] = result.get(label, 0.0) + float(self.backbone.signs[gate, mode] * local[gate, mode])
        for label, values in self.corrections.items():
            result[label] = result.get(label, 0.0) + float((values @ active_values).item())
        return {label: value for label, value in result.items() if abs(value) > 1e-12}

    def truncation_bound_for(self, generators: np.ndarray) -> float | None:
        active = self._active_generators(generators)
        if self.truncation_bound is None:
            return None
        return float(self.truncation_bound * np.linalg.norm(active))


def estimate_pauli_transport(
    operations: Sequence[CircuitOperation],
    n_qubits: int,
    *,
    support_cap: int | None = None,
    coefficient_tol: float = 1e-12,
    active_gate_indices: Sequence[int] | None = None,
    two_qubit_generator_basis: str = "diagonal",
) -> SparsePauliTransport:
    """Estimate bounded end transport with local Pauli propagation.

    ``support_cap`` and nonzero ``coefficient_tol`` prune retained vectors.
    At each pruning event, the Frobenius norm of the dropped coefficient map
    is accumulated in ``truncation_bound``.
    """

    values = _validate_operations(operations, n_qubits)
    if support_cap is not None and (not isinstance(support_cap, int) or support_cap < 1):
        raise ValueError("support_cap must be a positive integer")
    if coefficient_tol < 0:
        raise ValueError("coefficient_tol must be nonnegative")
    if two_qubit_generator_basis not in ("diagonal", "full-su4"):
        raise ValueError("two_qubit_generator_basis must be 'diagonal' or 'full-su4'")
    mode_count = 15 if two_qubit_generator_basis == "full-su4" else 3
    mode_counts = tuple(15 if mode_count == 15 and len(operation.qubits) == 2 else 3 for operation in values)
    active = _validate_active(active_gate_indices, len(values))
    offsets = {gate: index * mode_count for index, gate in enumerate(active)}
    width = len(active) * mode_count
    coefficients: dict[str, np.ndarray] = {}
    truncation_bound = 0.0

    for gate_index, operation in enumerate(values):
        if gate_index in offsets:
            for mode, local_label in enumerate(local_generator_labels(len(operation.qubits), full_two_qubit=mode_count == 15)):
                vector = np.zeros(width, dtype=float)
                vector[offsets[gate_index] + mode] = 1.0
                label = embed_pauli_label(local_label, operation.qubits, n_qubits)
                _merge(coefficients, label, vector)
        # Errors occur before their ideal operation. Propagate all sources
        # introduced so far through this operation in application order.
        coefficients = conjugate_sparse_modes(
            coefficients,
            operation,
            n_qubits,
            # Prune only after the full local transfer. This lets _prune
            # account for every tolerance omission in its returned bound.
            coefficient_tol=0.0,
        )
        coefficients, dropped = _prune(coefficients, support_cap, coefficient_tol=coefficient_tol)
        truncation_bound += dropped

    return SparsePauliTransport(
        coefficients=coefficients,
        n_qubits=n_qubits,
        gate_count=len(values),
        active_gate_indices=active,
        support_count=len(coefficients),
        truncation_bound=truncation_bound,
        generator_mode_count=mode_count,
        mode_counts=mode_counts,
    )


@dataclass(frozen=True)
class _CsrPauliMap:
    """Internal global CSR map: one row per Pauli label."""

    labels: tuple[str, ...]
    values: csr_matrix


def _combine_csr_maps(width: int, *maps: tuple[_CsrPauliMap, float]) -> _CsrPauliMap:
    labels = tuple(sorted({label for value, _ in maps for label in value.labels}))
    if not labels:
        return _CsrPauliMap((), csr_matrix((0, width), dtype=float))
    output = {label: row for row, label in enumerate(labels)}
    data_parts: list[np.ndarray] = []
    row_parts: list[np.ndarray] = []
    column_parts: list[np.ndarray] = []
    for value, sign in maps:
        if not value.values.nnz:
            continue
        source_rows = np.repeat(np.arange(len(value.labels)), np.diff(value.values.indptr))
        rows = np.fromiter((output[value.labels[row]] for row in source_rows), dtype=int, count=source_rows.size)
        data_parts.append(value.values.data * sign)
        row_parts.append(rows)
        column_parts.append(value.values.indices)
    if not data_parts:
        return _CsrPauliMap(labels, csr_matrix((len(labels), width), dtype=float))
    matrix = csr_matrix(
        (np.concatenate(data_parts), (np.concatenate(row_parts), np.concatenate(column_parts))),
        shape=(len(labels), width),
    )
    matrix.eliminate_zeros()
    return _CsrPauliMap(labels, matrix)


def _source_csr_map(width: int, entries: Sequence[tuple[str, int]]) -> _CsrPauliMap:
    labels = tuple(sorted({label for label, _ in entries}))
    index = {label: row for row, label in enumerate(labels)}
    matrix = csr_matrix(
        (
            np.ones(len(entries), dtype=float),
            (np.fromiter((index[label] for label, _ in entries), dtype=int), np.fromiter((column for _, column in entries), dtype=int)),
        ),
        shape=(len(labels), width),
    )
    return _CsrPauliMap(labels, matrix)


def _conjugate_global_csr(
    coefficients: _CsrPauliMap, operation: CircuitOperation, n_qubits: int,
    *, labels: tuple[str, ...], transfer: np.ndarray,
) -> _CsrPauliMap:
    if not coefficients.labels:
        return _CsrPauliMap((), csr_matrix((0, coefficients.values.shape[1]), dtype=float))
    entries: list[tuple[int, str, float]] = []
    for source_row, global_label in enumerate(coefficients.labels):
        source = labels.index(extract_pauli_label(global_label, operation.qubits))
        for target, factor in enumerate(transfer[:, source]):
            if factor == 0.0:
                continue
            characters = list(global_label)
            for qubit, character in zip(operation.qubits, labels[target]):
                characters[n_qubits - 1 - qubit] = character
            entries.append((source_row, "".join(characters), float(factor)))
    output_labels = tuple(sorted({label for _, label, _ in entries}))
    output = {label: row for row, label in enumerate(output_labels)}
    data_parts: list[np.ndarray] = []
    row_parts: list[np.ndarray] = []
    column_parts: list[np.ndarray] = []
    for source_row, label, factor in entries:
        start, end = coefficients.values.indptr[source_row:source_row + 2]
        if start == end:
            continue
        count = end - start
        data_parts.append(coefficients.values.data[start:end] * factor)
        row_parts.append(np.full(count, output[label], dtype=int))
        column_parts.append(coefficients.values.indices[start:end])
    width = coefficients.values.shape[1]
    if not data_parts:
        return _CsrPauliMap(output_labels, csr_matrix((len(output_labels), width), dtype=float))
    matrix = csr_matrix(
        (np.concatenate(data_parts), (np.concatenate(row_parts), np.concatenate(column_parts))),
        shape=(len(output_labels), width),
    )
    matrix.eliminate_zeros()
    return _CsrPauliMap(output_labels, matrix)


def _conjugate_global_tableau(coefficients: _CsrPauliMap, tableau: CliffordTableau) -> _CsrPauliMap:
    if not coefficients.labels:
        return _CsrPauliMap((), csr_matrix((0, coefficients.values.shape[1]), dtype=float))
    images = [tableau.conjugate(label) for label in coefficients.labels]
    output_labels = tuple(sorted({image.label for image in images}))
    output = {label: row for row, label in enumerate(output_labels)}
    source_rows = np.repeat(np.arange(len(coefficients.labels)), np.diff(coefficients.values.indptr))
    rows = np.fromiter((output[images[row].label] for row in source_rows), dtype=int, count=source_rows.size)
    signs = np.fromiter((images[row].sign for row in source_rows), dtype=float, count=source_rows.size)
    matrix = csr_matrix(
        (coefficients.values.data * signs, (rows, coefficients.values.indices)),
        shape=(len(output_labels), coefficients.values.shape[1]),
    )
    matrix.eliminate_zeros()
    return _CsrPauliMap(output_labels, matrix)


def _prune_global_csr(
    coefficients: _CsrPauliMap, support_cap: int,
    *, coefficient_tol: float, selection_matrix: np.ndarray | None,
) -> tuple[_CsrPauliMap, float]:
    matrix = coefficients.values
    count = len(coefficients.labels)
    if not count:
        return coefficients, 0.0
    rows = np.repeat(np.arange(count), np.diff(matrix.indptr))
    maximum = np.zeros(count, dtype=float)
    if matrix.nnz:
        np.maximum.at(maximum, rows, np.abs(matrix.data))
    squared = np.asarray(matrix.multiply(matrix).sum(axis=1)).reshape(-1)
    retained = maximum > coefficient_tol
    if int(np.count_nonzero(retained)) > support_cap:
        projected = np.asarray(matrix @ selection_matrix) if selection_matrix is not None else None
        scores = np.linalg.norm(projected, axis=1) if projected is not None else np.sqrt(squared)
        ranked = sorted(
            np.flatnonzero(retained),
            key=lambda row: (-float(scores[row]), -float(np.sqrt(squared[row])), coefficients.labels[row]),
        )
        retained[:] = False
        retained[np.asarray(ranked[:support_cap], dtype=int)] = True
    dropped = float(np.sqrt(np.sum(squared[~retained])))
    kept_rows = np.flatnonzero(retained)
    return _CsrPauliMap(
        tuple(coefficients.labels[row] for row in kept_rows), matrix[kept_rows].tocsr(),
    ), dropped


def estimate_hybrid_pauli_transport(
    operations: Sequence[CircuitOperation],
    n_qubits: int,
    *,
    support_cap: int,
    coefficient_tol: float = 1e-12,
    active_gate_indices: Sequence[int] | None = None,
    two_qubit_generator_basis: str = "diagonal",
    selection_matrix: np.ndarray | None = None,
) -> AnchoredPauliTransport:
    """Nearest-Clifford backbone plus bounded exact Pauli corrections.

    Each operation's Pauli-transfer map is split exactly as
    ``T_U = T_C + (T_U - T_C)``, where ``C`` is its nearest Clifford. The
    all-Clifford backbone is kept without truncation; only the sparse
    correction map is capped. Therefore ``support_cap=0`` is exactly
    nearest-Clifford transport, and larger caps provide more retained exact
    Pauli-correction capacity toward full Pauli transport. This anchors the
    cap family at nearest Clifford, unlike guessed tail replacement.

    ``truncation_bound`` is the usual exact-transport truncation certificate:
    it bounds only omitted correction rows. At ``support_cap=0`` the correction
    map is intentionally not materialized, so the result is ``None`` rather
    than a false zero certificate. When ``selection_matrix`` is supplied, rows
    are ranked by projected optimizer signal instead of raw transport norm.
    """

    values = _validate_operations(operations, n_qubits)
    if not isinstance(support_cap, int) or support_cap < 0:
        raise ValueError("support_cap must be a nonnegative integer")
    if coefficient_tol < 0:
        raise ValueError("coefficient_tol must be nonnegative")
    if two_qubit_generator_basis not in ("diagonal", "full-su4"):
        raise ValueError("two_qubit_generator_basis must be 'diagonal' or 'full-su4'")
    mode_count = 15 if two_qubit_generator_basis == "full-su4" else 3
    mode_counts = tuple(
        15 if mode_count == 15 and len(operation.qubits) == 2 else 3
        for operation in values
    )
    active = _validate_active(active_gate_indices, len(values))
    offsets = {gate: index * mode_count for index, gate in enumerate(active)}
    width = len(active) * mode_count
    selection = _validate_selection_matrix(selection_matrix, width)
    if support_cap == 0:
        return AnchoredPauliTransport(
            _cached_nearest_backbone(values, n_qubits, two_qubit_generator_basis), {}, active,
            truncation_bound=None,
        )
    gate_actions, native_backbone = _cached_nearest_actions(
        values, n_qubits, two_qubit_generator_basis,
    )

    backbone = _CsrPauliMap((), csr_matrix((0, width), dtype=float))
    correction = _CsrPauliMap((), csr_matrix((0, width), dtype=float))
    truncation_bound = 0.0

    for gate_index, operation in enumerate(values):
        if gate_index in offsets:
            sources = _source_csr_map(
                width,
                tuple(
                    (
                        embed_pauli_label(local_label, operation.qubits, n_qubits),
                        offsets[gate_index] + mode,
                    )
                    for mode, local_label in enumerate(
                        local_generator_labels(
                            len(operation.qubits), full_two_qubit=mode_count == 15,
                        )
                    )
                ),
            )
            backbone = _combine_csr_maps(width, (backbone, 1.0), (sources, 1.0))
        labels, transfer, clifford = gate_actions[gate_index]
        clifford_backbone = _conjugate_global_tableau(backbone, clifford)
        exact_backbone = _conjugate_global_csr(
            backbone, operation, n_qubits,
            labels=labels, transfer=transfer,
        )
        exact_correction = _conjugate_global_csr(
            correction, operation, n_qubits,
            labels=labels, transfer=transfer,
        )
        exact_correction = _combine_csr_maps(
            width,
            (exact_correction, 1.0),
            (exact_backbone, 1.0),
            (clifford_backbone, -1.0),
        )
        correction, dropped = _prune_global_csr(
            exact_correction,
            support_cap,
            coefficient_tol=coefficient_tol,
            selection_matrix=selection,
        )
        truncation_bound += dropped
        backbone = clifford_backbone

    return AnchoredPauliTransport(
        backbone=native_backbone,
        corrections={
            label: correction.values.getrow(row)
            for row, label in enumerate(correction.labels)
        },
        active_gate_indices=active,
        truncation_bound=truncation_bound,
    )


def propagate_pauli_bias(
    operations: Sequence[CircuitOperation],
    generators: np.ndarray,
    n_qubits: int,
    *,
    support_cap: int | None = None,
    coefficient_tol: float = 1e-12,
) -> tuple[dict[str, float], float]:
    """Convenience scalar path returning bias and its omitted-transport bound."""

    transport = estimate_pauli_transport(
        operations,
        n_qubits,
        support_cap=support_cap,
        coefficient_tol=coefficient_tol,
    )
    return transport.contract(generators), transport.truncation_bound_for(generators)
