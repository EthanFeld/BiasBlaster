"""Scalable coherent-bias estimation for pulse-control prioritization."""

from .local_transfer import conjugate_sparse_pauli, local_pauli_transfer, pauli_transfer
from .model import CircuitOperation
from .nearest_clifford import (
    SparseCliffordTransport,
    closest_clifford,
    estimate_nearest_clifford_transport,
    nearest_clifford_process_fidelity,
)
from .optimizer import BiasOptimizationResult, OptimizationResult, optimize_bias_controls, optimize_local_error_controls
from .fpga_optimizer import (
    FPGALink,
    FpgaOptimizerConfig,
    list_fpga_ports,
    optimize_bias_controls_fpga,
    optimize_bias_controls_fpga_greedy,
    resolve_fpga_port,
)
from .pauli import (
    embed_pauli_label,
    extract_pauli_label,
    local_generator_labels,
    pauli_labels,
    pauli_matrix,
)
from .pulse_simulator import (
    PulseCalibrationResult,
    PulseSimulationResult,
    calibrate_pulse_error_model,
    simulate_piecewise_constant_pulse,
)
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
from .transport import (
    AnchoredPauliTransport,
    SparsePauliTransport,
    clear_transport_cache,
    estimate_hybrid_pauli_transport,
    estimate_pauli_transport,
    propagate_pauli_bias,
)

__all__ = [
    "CircuitOperation",
    "SparsePauliTransport",
    "AnchoredPauliTransport",
    "clear_transport_cache",
    "SparseCliffordTransport",
    "estimate_pauli_transport",
    "estimate_hybrid_pauli_transport",
    "estimate_nearest_clifford_transport",
    "optimize_bias_controls",
    "optimize_local_error_controls",
    "optimize_bias_controls_fpga",
    "optimize_bias_controls_fpga_greedy",
    "FpgaOptimizerConfig",
    "FPGALink",
    "resolve_fpga_port",
    "list_fpga_ports",
    "simulate_piecewise_constant_pulse",
    "calibrate_pulse_error_model",
    "BiasOptimizationResult",
    "OptimizationResult",
    "PulseSimulationResult",
    "PulseCalibrationResult",
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
    "nearest_clifford_process_fidelity",
]
