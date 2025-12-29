"""
Core data structures and types for the HOT Framework.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Literal, Union, Tuple, Callable, TYPE_CHECKING
from enum import Enum
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import QuantumRegister
from qiskit.transpiler import Layout
import networkx as nx

if TYPE_CHECKING:
    from .variational import CoDesignResult


class AlgorithmClass(Enum):
    """Algorithm classification for HOT optimization."""
    FIXED_UNITARY = "fixed_unitary"
    VARIATIONAL = "variational"


class MeasurementClass(Enum):
    """Measurement classification for hygiene policies."""
    ESSENTIAL = "essential"
    DIAGNOSTIC = "diagnostic"
    EXTRANEOUS = "extraneous"


@dataclass
class IslandMetrics:
    """Quality metrics for a qubit island."""
    mean_readout_error: float
    mean_two_qubit_error: float
    mean_T1: float
    mean_T2: float
    calibration_stability: float
    expected_swaps: int
    coherence_score: float


@dataclass
class Island:
    """A connected subgraph of qubits selected for algorithm execution."""
    qubits: List[int]
    edges: List[Tuple[int, int]]
    score: float
    metrics: IslandMetrics
    size: int = field(init=False)
    
    def __post_init__(self):
        self.size = len(self.qubits)


@dataclass
class Scaffold:
    """Topological embedding of logical qubits into physical island."""
    mapping: Dict[int, int]  # logical -> physical qubit mapping
    swap_schedule: List[Dict[str, Any]]
    estimated_error: float
    topology_pattern: str
    island: Island
    
    def get_layout(self) -> Layout:
        """Create Qiskit Layout from mapping."""
        layout = Layout()
        if not self.mapping:
            return layout

        max_logical = max(self.mapping.keys())
        qreg = QuantumRegister(max_logical + 1, 'q')
        for logical_q, physical_q in self.mapping.items():
            layout[qreg[logical_q]] = physical_q
        return layout


@dataclass
class OrphanPolicy:
    """Policy for handling qubits outside the selected island."""
    mode: Literal['strict', 'relaxed', 'diagnostic']
    allowed_orphans: Optional[List[int]] = None
    distance_threshold: int = 2


@dataclass
class MeasurementPolicy:
    """Configuration for measurement hygiene."""
    allow_mid_circuit: bool = False
    max_concurrent_adjacent: int = 1
    stagger_readout: bool = True
    readout_error_mitigation: bool = True


@dataclass
class JobVariants:
    """Instrumented and sanitized job variants."""
    sanitized: QuantumCircuit
    instrumented: QuantumCircuit
    comparison_metadata: Dict[str, Any]


@dataclass
class AlgorithmSpec:
    """Specification of an algorithm for HOT compilation."""
    name: str
    algorithm_class: AlgorithmClass
    min_qubits: int
    max_qubits: int
    interaction_graph: nx.Graph
    circuit: Optional[QuantumCircuit] = None
    output_qubits: Optional[List[int]] = None
    cost_function: Optional[Callable[[str], float]] = None
    
    # For variational algorithms
    initial_parameters: Optional[np.ndarray] = None
    parameter_bounds: Optional[List[Tuple[float, float]]] = None
    
    def random_initial_parameters(self) -> np.ndarray:
        """Generate random initial parameters for variational algorithms."""
        if self.parameter_bounds:
            return np.array([
                np.random.uniform(low, high) 
                for low, high in self.parameter_bounds
            ])
        else:
            # Default: random in [0, 2π]
            n_params = len(self.initial_parameters) if self.initial_parameters is not None else 4
            return np.random.uniform(0, 2*np.pi, n_params)


@dataclass
class CalibrationData:
    """Backend calibration data for island selection."""
    timestamp: str
    backend_name: str
    topology: nx.Graph
    
    # Qubit properties
    readout_errors: Dict[int, float]
    T1_times: Dict[int, float]  # in microseconds
    T2_times: Dict[int, float]  # in microseconds
    
    # Edge properties
    two_qubit_errors: Dict[Tuple[int, int], float]
    
    # Historical stability data
    historical_snapshots: Optional[List[Dict[str, Any]]] = None
    
    def get_qubit_error(self, qubit: int) -> float:
        """Get aggregate error metric for a qubit."""
        ro_error = self.readout_errors.get(qubit, 0.0)
        t1 = self.T1_times.get(qubit, 100.0)
        t2 = self.T2_times.get(qubit, 100.0)
        
        # Coherence penalty (inverse of min(T1, T2))
        coherence_penalty = 1.0 / min(t1, t2)
        
        return ro_error + 0.1 * coherence_penalty
    
    def get_edge_error(self, edge: Tuple[int, int]) -> float:
        """Get two-qubit gate error for an edge."""
        return self.two_qubit_errors.get(edge, 1.0)
    
    def get_calibration_stability(self, qubits: List[int]) -> float:
        """Compute calibration stability for a set of qubits."""
        if not self.historical_snapshots:
            return 1.0  # Assume stable if no history
        
        # Compute variance of errors over time
        errors_over_time = []
        for snapshot in self.historical_snapshots[-10:]:  # Last 10 snapshots
            snapshot_errors = []
            for q in qubits:
                ro_err = snapshot['readout_errors'].get(q, 0.0)
                snapshot_errors.append(ro_err)
            errors_over_time.append(np.mean(snapshot_errors))
        
        # Lower variance = higher stability
        variance = np.var(errors_over_time) if len(errors_over_time) > 1 else 0.0
        return 1.0 / (1.0 + variance)


@dataclass
class HOTResult:
    """Complete result from HOT compilation."""
    compiled_circuit: QuantumCircuit
    job_variants: JobVariants
    island: Island
    scaffold: Scaffold
    algorithm_class: AlgorithmClass
    codesign_result: Optional['CoDesignResult'] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Backward-compatibility: allow callers to pass strings.
        if isinstance(self.algorithm_class, str):
            self.algorithm_class = AlgorithmClass(self.algorithm_class)
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"HOT Compilation Summary",
            f"=======================",
            f"Backend: {self.metadata.get('backend', 'Unknown')}",
            f"Algorithm class: {self.algorithm_class}",
            f"",
            f"Island Selection:",
            f"  Qubits: {self.island.qubits}",
            f"  Score: {self.island.score:.4f}",
            f"  Size: {self.island.size} qubits",
            f"",
            f"Scaffold:",
            f"  Pattern: {self.scaffold.topology_pattern}",
            f"  Estimated error: {self.scaffold.estimated_error:.4f}",
            f"  SWAPs required: {len(self.scaffold.swap_schedule)}",
        ]
        
        if self.codesign_result:
            lines.extend([
                f"",
                f"Variational Co-Design:",
                f"  Best configuration: {self.codesign_result.best_configuration}",
                f"  Improvement: {self.codesign_result.improvement_over_baseline:.1%}",
                f"",
                f"  Configuration comparison:",
            ])
            for name, result in self.codesign_result.configurations.items():
                lines.append(f"    {name}: expected={result.expected_cost:.2f}")
        
        lines.extend([
            f"",
            f"Measurement Hygiene:",
            f"  Sanitized measurements: {self.job_variants.comparison_metadata.get('sanitized_measurements', 'N/A')}",
            f"  Instrumented measurements: {self.job_variants.comparison_metadata.get('instrumented_measurements', 'N/A')}",
        ])
        
        return "\n".join(lines)


@dataclass
class DistributionMetrics:
    """Metrics for comparing measurement distributions."""
    tvd: Optional[float]  # Total Variation Distance from baseline
    kl_divergence: Optional[float]  # KL divergence from baseline
    entropy: float  # Shannon entropy of distribution
    
    # Cost-weighted metrics
    expected_cost: float
    cost_variance: float
    best_of_sample: float
    
    # Top-K analysis
    top_k_probability: float
    top_k_avg_cost: float
    top_k_best_cost: float
    
    # Interference signature
    constructive_interference_score: float


class HOTError(Exception):
    """Base exception for HOT framework."""
    pass


class NoViableIslandError(HOTError):
    """Raised when no suitable island can be found."""
    pass


class OrphanViolationError(HOTError):
    """Raised when orphan policy is violated."""
    pass


class MeasurementPolicyViolation(HOTError):
    """Raised when measurement policy cannot be satisfied."""
    pass
