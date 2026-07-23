"""Scalable coherent-bias estimation for pulse-control prioritization."""

from .local_transfer import conjugate_sparse_pauli, local_pauli_transfer, pauli_transfer
from .model import CircuitOperation
from .nearest_clifford import (
    SparseCliffordTransport,
    closest_clifford,
    estimate_nearest_clifford_transport,
)
from .optimizer import BiasOptimizationResult, OptimizationResult, optimize_bias_controls
from .pauli import (
    embed_pauli_label,
    extract_pauli_label,
    local_generator_labels,
    pauli_labels,
    pauli_matrix,
)
from .pulse_simulator import PulseSimulationResult, simulate_piecewise_constant_pulse
from .tableau import (
    CliffordTableau,
    PauliString,
    SymplecticPauli,
    SymplecticTableau,
    conjugate_pauli,
    named_clifford,
    pauli_to_symplectic,
    symplectic_to_pauli,
    tableau_from_gate_sequence,
    tableau_from_symplectic,
    tableau_from_unitary,
)
from .transport import SparsePauliTransport, estimate_pauli_transport, propagate_pauli_bias

__all__ = [
    "CircuitOperation",
    "SparsePauliTransport",
    "SparseCliffordTransport",
    "estimate_pauli_transport",
    "estimate_nearest_clifford_transport",
    "optimize_bias_controls",
    "simulate_piecewise_constant_pulse",
    "BiasOptimizationResult",
    "OptimizationResult",
    "PulseSimulationResult",
    "propagate_pauli_bias",
    "pauli_labels",
    "pauli_matrix",
    "embed_pauli_label",
    "extract_pauli_label",
    "local_generator_labels",
    "local_pauli_transfer",
    "pauli_transfer",
    "conjugate_sparse_pauli",
    "CliffordTableau",
    "SymplecticTableau",
    "SymplecticPauli",
    "PauliString",
    "conjugate_pauli",
    "named_clifford",
    "tableau_from_gate_sequence",
    "tableau_from_symplectic",
    "tableau_from_unitary",
    "pauli_to_symplectic",
    "symplectic_to_pauli",
    "closest_clifford",
]
