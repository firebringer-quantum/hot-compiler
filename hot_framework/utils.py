"""
Utility functions for HOT Framework.
"""

from typing import Dict, List, Any, Optional, Callable, TYPE_CHECKING
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
import networkx as nx

from .core import DistributionMetrics, AlgorithmSpec, AlgorithmClass

if TYPE_CHECKING:
    from .variational import RetuningConfig, CoDesignResult


def compute_distribution_metrics(counts: Dict[str, int], cost_function: Callable[[str], float],
                                baseline_counts: Optional[Dict[str, int]] = None,
                                k: int = 10) -> DistributionMetrics:
    """
    Compute comprehensive distribution metrics.
    
    Args:
        counts: Measurement counts dictionary
        cost_function: Function to compute cost for each bitstring
        baseline_counts: Optional baseline distribution for comparison
        k: Number of top solutions to analyze
        
    Returns:
        DistributionMetrics object
    """
    total = sum(counts.values())
    probs = {bs: c/total for bs, c in counts.items()}
    costs = {bs: cost_function(bs) for bs in counts}
    
    # Basic metrics
    entropy_val = -sum(p * np.log2(p) for p in probs.values() if p > 0)
    expected = sum(probs[bs] * costs[bs] for bs in probs)
    variance = sum(probs[bs] * (costs[bs] - expected)**2 for bs in probs)
    
    # Top-K analysis
    sorted_by_count = sorted(counts.items(), key=lambda x: -x[1])[:k]
    top_k_prob = sum(c for _, c in sorted_by_count) / total
    top_k_costs = [costs[bs] for bs, _ in sorted_by_count]
    
    # Comparison to baseline
    tvd = None
    kl_divergence = None
    
    if baseline_counts:
        baseline_total = sum(baseline_counts.values())
        baseline_probs = {bs: c/baseline_total for bs, c in baseline_counts.items()}
        
        all_bitstrings = set(probs.keys()) | set(baseline_probs.keys())
        tvd = 0.5 * sum(abs(probs.get(bs, 0) - baseline_probs.get(bs, 0)) 
                       for bs in all_bitstrings)
        
        # KL divergence (with smoothing)
        eps = 1e-10
        kl = sum(probs.get(bs, eps) * np.log(probs.get(bs, eps) / baseline_probs.get(bs, eps))
                for bs in all_bitstrings if probs.get(bs, 0) > 0)
        kl_divergence = kl
    
    # Interference score: correlation between probability and solution quality
    # Higher score = layout pushes probability toward better solutions
    sorted_by_cost = sorted(costs.items(), key=lambda x: x[1])
    ranks = {bs: i for i, (bs, _) in enumerate(sorted_by_cost)}
    
    if len(ranks) > 1:
        prob_values = [probs.get(bs, 0) for bs in ranks]
        rank_values = [ranks[bs] for bs in ranks]
        interference_score = -np.corrcoef(prob_values, rank_values)[0, 1]
    else:
        interference_score = 0.0
    
    return DistributionMetrics(
        tvd=tvd,
        kl_divergence=kl_divergence,
        entropy=entropy_val,
        expected_cost=expected,
        cost_variance=variance,
        best_of_sample=min(costs.values()),
        top_k_probability=top_k_prob,
        top_k_avg_cost=np.mean(top_k_costs) if top_k_costs else 0.0,
        top_k_best_cost=min(top_k_costs) if top_k_costs else 0.0,
        constructive_interference_score=interference_score
    )


def run_controlled_comparison(algorithm_spec: AlgorithmSpec, backend: Any,
                             retuning_config: 'RetuningConfig', shots: int = 8192,
                             repetitions: int = 5) -> 'ComparisonResult':
    """
    Run controlled comparison of HOT vs No-HOT configurations.
    
    Args:
        algorithm_spec: Algorithm specification
        backend: Quantum backend
        retuning_config: Retuning configuration
        shots: Number of shots per run
        repetitions: Number of repetitions for statistics
        
    Returns:
        ComparisonResult with statistical analysis
    """
    from .variational import VariationalCoDesigner
    from .core import CalibrationData
    
    # Get calibration data
    calibration_data = _get_calibration_data(backend)
    
    # Run co-design
    codesigner = VariationalCoDesigner(algorithm_spec, backend, calibration_data)
    codesign_result = codesigner.run_codesign(retuning_config)
    
    # Run multiple repetitions for statistics
    all_results = {}
    for config_name in codesign_result.configurations.keys():
        config_results = []
        for rep in range(repetitions):
            # Re-run evaluation with new sampling
            result = _rerun_evaluation(
                algorithm_spec, backend, calibration_data,
                codesign_result.configurations[config_name],
                shots
            )
            config_results.append(result)
        all_results[config_name] = config_results
    
    return ComparisonResult(
        codesign_result=codesign_result,
        repeated_results=all_results,
        shots=shots,
        repetitions=repetitions
    )


def get_calibration_data(backend: Any) -> 'CalibrationData':
    """Get calibration data from backend."""
    from .core import CalibrationData
    import networkx as nx
    
    # This is a simplified implementation
    # In practice, would fetch real calibration data from IBM Quantum
    
    # Create topology
    topology = nx.Graph()

    if hasattr(backend, 'configuration') and callable(getattr(backend, 'configuration')):
        config = backend.configuration()
        coupling_map = config.coupling_map
        n_qubits = config.n_qubits
        topology.add_edges_from(coupling_map)
    else:
        # BackendV2 (e.g., GenericBackendV2)
        n_qubits = int(getattr(backend, 'num_qubits'))
        coupling_map = getattr(backend, 'coupling_map', None)
        if coupling_map is not None:
            # CouplingMap or list
            edges = coupling_map.get_edges() if hasattr(coupling_map, 'get_edges') else coupling_map
            topology.add_edges_from(edges)
        else:
            # Fallback to target if available
            target = getattr(backend, 'target', None)
            if target is not None and hasattr(target, 'build_coupling_map'):
                cm = target.build_coupling_map()
                topology.add_edges_from(cm.get_edges())
    
    # Mock calibration data
    readout_errors = {i: 0.01 + 0.005 * np.random.random() for i in range(n_qubits)}
    T1_times = {i: 100 + 20 * np.random.random() for i in range(n_qubits)}
    T2_times = {i: 80 + 15 * np.random.random() for i in range(n_qubits)}
    
    two_qubit_errors = {}
    for edge in topology.edges():
        two_qubit_errors[tuple(edge)] = 0.02 + 0.01 * np.random.random()
    
    return CalibrationData(
        timestamp="2024-01-01T00:00:00Z",
        backend_name=_backend_name(backend),
        topology=topology,
        readout_errors=readout_errors,
        T1_times=T1_times,
        T2_times=T2_times,
        two_qubit_errors=two_qubit_errors,
        historical_snapshots=None
    )


def _get_calibration_data(backend: Any) -> 'CalibrationData':
    """Backward-compatible alias for get_calibration_data."""
    return get_calibration_data(backend)


def _backend_name(backend: Any) -> str:
    """Return backend name regardless of whether it's a method or attribute."""
    name_attr = getattr(backend, 'name', None)
    if callable(name_attr):
        return name_attr()
    if isinstance(name_attr, str):
        return name_attr
    return backend.__class__.__name__


def _rerun_evaluation(algorithm_spec: AlgorithmSpec, backend: Any,
                      calibration_data: 'CalibrationData', 
                      config_result: 'ConfigurationResult', shots: int) -> Dict[str, float]:
    """Re-run evaluation with fresh sampling."""
    circuit = config_result.circuit
    counts_dict = _statevector_sample_counts(circuit, shots)
    
    # Compute metrics
    if algorithm_spec.cost_function is not None:
        cost_function = algorithm_spec.cost_function
    else:
        cost_function = lambda bitstring: bitstring.count('1')
    
    metrics = compute_distribution_metrics(counts_dict, cost_function)
    
    return {
        'expected_cost': metrics.expected_cost,
        'best_of_sample': metrics.best_of_sample,
        'entropy': metrics.entropy,
        'top_k_probability': metrics.top_k_probability
    }


def _statevector_sample_counts(circuit: QuantumCircuit, shots: int) -> Dict[str, int]:
    """Sample counts from a circuit using Statevector probabilities.

    Used as a Qiskit-2.x-compatible fallback for tests and local comparisons.
    """
    n_qubits = circuit.num_qubits
    try:
        no_meas = circuit.remove_final_measurements(inplace=False)
    except Exception:
        no_meas = circuit.copy()

    state = Statevector.from_instruction(no_meas)
    probs = np.asarray(state.probabilities(), dtype=float)
    probs = np.clip(probs, 0.0, 1.0)
    s = probs.sum()
    if s <= 0:
        return {}
    probs = probs / s

    outcomes = np.random.choice(len(probs), size=shots, p=probs)
    counts: Dict[str, int] = {}
    for outcome in outcomes:
        bitstring = format(int(outcome), f"0{n_qubits}b")
        counts[bitstring] = counts.get(bitstring, 0) + 1
    return counts


class ComparisonResult:
    """Results from controlled comparison experiment."""
    
    def __init__(self, codesign_result: 'CoDesignResult', repeated_results: Dict[str, List[Dict[str, float]]],
                 shots: int, repetitions: int):
        self.codesign_result = codesign_result
        self.repeated_results = repeated_results
        self.shots = shots
        self.repetitions = repetitions
    
    def report(self) -> str:
        """Generate comparison report."""
        lines = [
            "HOT vs No-HOT Controlled Comparison",
            "====================================",
            f"Shots per run: {self.shots}",
            f"Repetitions: {self.repetitions}",
            "",
            "Configuration Summary:",
        ]
        
        for name, result in self.codesign_result.configurations.items():
            lines.append(f"  {name}: expected={result.expected_cost:.4f}, best={result.best_of_sample:.4f}")
        
        lines.extend([
            "",
            "Statistical Analysis:",
        ])
        
        # Compute statistics for each configuration
        for config_name, results in self.repeated_results.items():
            expected_costs = [r['expected_cost'] for r in results]
            best_costs = [r['best_of_sample'] for r in results]
            
            lines.extend([
                f"  {config_name}:",
                f"    Expected: {np.mean(expected_costs):.4f} ± {np.std(expected_costs):.4f}",
                f"    Best: {np.mean(best_costs):.4f} ± {np.std(best_costs):.4f}",
            ])
        
        lines.extend([
            "",
            f"Best configuration: {self.codesign_result.best_configuration}",
            f"Improvement: {self.codesign_result.improvement_over_baseline:.1%}",
            "",
            "Recommendation: Use HOT with retuned parameters"
        ])
        
        return "\n".join(lines)


def create_interaction_graph_from_circuit(circuit: QuantumCircuit) -> nx.Graph:
    """
    Create interaction graph from quantum circuit.
    
    Args:
        circuit: Quantum circuit to analyze
        
    Returns:
        Interaction graph showing qubit connectivity requirements
    """
    graph = nx.Graph()
    qubit_to_index = {q: i for i, q in enumerate(circuit.qubits)}

    def _inst_parts(inst):
        op = getattr(inst, "operation", None)
        qargs = getattr(inst, "qubits", None)
        cargs = getattr(inst, "clbits", None)
        if op is None:
            op, qargs, cargs = inst
        return op, qargs, cargs
    
    # Add all qubits
    for qubit in circuit.qubits:
        graph.add_node(qubit_to_index[qubit])
    
    # Add edges for two-qubit gates
    for inst in circuit.data:
        instruction, qargs, cargs = _inst_parts(inst)
        if len(qargs) == 2:
            q1, q2 = qubit_to_index[qargs[0]], qubit_to_index[qargs[1]]
            graph.add_edge(q1, q2)
    
    return graph


def validate_algorithm_spec(algorithm_spec: AlgorithmSpec) -> List[str]:
    """
    Validate algorithm specification.
    
    Args:
        algorithm_spec: Algorithm specification to validate
        
    Returns:
        List of validation warnings/errors
    """
    warnings = []
    
    # Check basic requirements
    if algorithm_spec.min_qubits <= 0:
        warnings.append("min_qubits must be positive")
    
    if algorithm_spec.max_qubits < algorithm_spec.min_qubits:
        warnings.append("max_qubits must be >= min_qubits")
    
    if not algorithm_spec.interaction_graph.nodes():
        warnings.append("interaction_graph must have nodes")
    
    # Check variational-specific requirements
    if algorithm_spec.algorithm_class == AlgorithmClass.VARIATIONAL:
        if algorithm_spec.initial_parameters is None:
            warnings.append("Variational algorithms should have initial_parameters")
        
        if algorithm_spec.cost_function is None:
            warnings.append("Variational algorithms should have cost_function")
    
    # Check consistency
    n_graph_qubits = algorithm_spec.interaction_graph.number_of_nodes()
    if n_graph_qubits < algorithm_spec.min_qubits or n_graph_qubits > algorithm_spec.max_qubits:
        warnings.append(
            f"interaction_graph size ({n_graph_qubits}) doesn't match qubit range "
            f"({algorithm_spec.min_qubits}-{algorithm_spec.max_qubits})"
        )
    
    return warnings


def estimate_runtime(circuit: QuantumCircuit, backend: Any) -> Dict[str, float]:
    """
    Estimate circuit runtime on backend.
    
    Args:
        circuit: Quantum circuit
        backend: Target backend
        
    Returns:
        Runtime estimates in different units
    """
    # Simplified runtime estimation
    gate_times = {
        'x': 50,      # nanoseconds
        'y': 50,
        'z': 50,
        'rx': 50,
        'ry': 50,
        'rz': 50,
        'cx': 200,    # nanoseconds
        'cz': 200,
        'swap': 400,
        'measure': 1000,
    }
    
    total_time = 0
    for inst in circuit.data:
        instruction = getattr(inst, "operation", None)
        if instruction is None:
            instruction, _, _ = inst
        gate_time = gate_times.get(instruction.name, 100)  # default 100ns
        total_time += gate_time
    
    return {
        'nanoseconds': total_time,
        'microseconds': total_time / 1000,
        'milliseconds': total_time / 1_000_000,
        'seconds': total_time / 1_000_000_000
    }
