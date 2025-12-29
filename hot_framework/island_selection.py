"""
O - Optimized island selection for HOT Framework.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import networkx as nx
from datetime import timedelta

from .core import Island, IslandMetrics, AlgorithmSpec, CalibrationData, NoViableIslandError


def enumerate_connected_subgraphs(topology: nx.Graph, size: int) -> List[nx.Graph]:
    """
    Enumerate all connected subgraphs of given size.
    
    Args:
        topology: Backend connectivity graph
        size: Number of qubits in subgraph
        
    Returns:
        List of connected subgraphs
    """
    subgraphs = []
    nodes = list(topology.nodes())
    
    # Generate all combinations of nodes of given size
    from itertools import combinations
    for node_combo in combinations(nodes, size):
        subgraph = topology.subgraph(node_combo)
        
        # Check if connected
        if nx.is_connected(subgraph):
            subgraphs.append(subgraph.copy())
    
    return subgraphs


def find_embedding(interaction_graph: nx.Graph, subgraph: nx.Graph) -> Optional[Dict[int, int]]:
    """
    Find embedding of interaction graph into subgraph.
    
    Args:
        interaction_graph: Logical connectivity requirements
        subgraph: Physical subgraph to embed into
        
    Returns:
        Mapping from logical to physical qubits, or None if no embedding exists
    """
    from networkx.algorithms import isomorphism

    edge_match = isomorphism.categorical_edge_match('weight', 1.0)

    matcher = isomorphism.GraphMatcher(
        subgraph,
        interaction_graph,
        edge_match=edge_match,
    )

    if matcher.subgraph_is_isomorphic():
        mapping_phys_to_logical = next(matcher.subgraph_isomorphisms_iter())
        return {logical: physical for physical, logical in mapping_phys_to_logical.items()}

    if interaction_graph.number_of_nodes() == subgraph.number_of_nodes():
        exact_matcher = isomorphism.GraphMatcher(
            interaction_graph,
            subgraph,
            edge_match=edge_match,
        )
        if exact_matcher.is_isomorphic():
            return next(exact_matcher.isomorphisms_iter())

    return None


def compute_island_score(subgraph: nx.Graph, calibration_data: CalibrationData,
                         stability_window: timedelta = timedelta(hours=24)) -> float:
    """
    Compute quality score for a candidate island.
    
    Score formula:
    S(I) = α * mean_readout_error + β * mean_2q_error + γ * calibration_stability + δ * expected_swaps
    
    Args:
        subgraph: Candidate subgraph
        calibration_data: Device calibration information
        stability_window: Time window for stability analysis
        
    Returns:
        Quality score (lower is better)
    """
    # Weight parameters
    alpha = 0.3  # Readout error weight
    beta = 0.4   # Two-qubit error weight  
    gamma = 0.2  # Stability weight
    delta = 0.1  # SWAP overhead weight
    
    # Compute mean readout error
    readout_errors = [
        calibration_data.readout_errors.get(q, 0.0) 
        for q in subgraph.nodes
    ]
    mean_readout_error = np.mean(readout_errors) if readout_errors else 1.0
    
    # Compute mean two-qubit error
    two_q_errors = [
        calibration_data.two_qubit_errors.get(edge, 1.0)
        for edge in subgraph.edges
    ]
    mean_two_q_error = np.mean(two_q_errors) if two_q_errors else 1.0
    
    # Compute calibration stability
    stability = calibration_data.get_calibration_stability(list(subgraph.nodes))
    stability_penalty = 1.0 - stability  # Convert to penalty (higher = worse)
    
    # Estimate SWAP overhead (simplified - assumes linear scaling with size)
    expected_swaps = subgraph.number_of_nodes() - 1  # Rough estimate
    
    # Combine scores
    score = (
        alpha * mean_readout_error +
        beta * mean_two_q_error +
        gamma * stability_penalty +
        delta * expected_swaps
    )
    
    return score


def discover_islands(backend: Any, calibration_data: CalibrationData, 
                   min_size: int, max_size: int) -> List[Island]:
    """
    Enumerate and rank candidate islands.
    
    Args:
        backend: Quantum backend
        calibration_data: Device calibration information
        min_size: Minimum island size
        max_size: Maximum island size
        
    Returns:
        List of Island objects sorted by quality score
    """
    topology = calibration_data.topology
    islands = []
    
    for size in range(min_size, max_size + 1):
        subgraphs = enumerate_connected_subgraphs(topology, size)
        
        for subgraph in subgraphs:
            score = compute_island_score(subgraph, calibration_data)
            
            # Compute detailed metrics
            readout_errors = [
                calibration_data.readout_errors.get(q, 0.0) 
                for q in subgraph.nodes
            ]
            two_q_errors = [
                calibration_data.two_qubit_errors.get(edge, 1.0)
                for edge in subgraph.edges
            ]
            t1_times = [
                calibration_data.T1_times.get(q, 100.0)
                for q in subgraph.nodes
            ]
            t2_times = [
                calibration_data.T2_times.get(q, 100.0)
                for q in subgraph.nodes
            ]
            
            metrics = IslandMetrics(
                mean_readout_error=np.mean(readout_errors),
                mean_two_qubit_error=np.mean(two_q_errors),
                mean_T1=np.mean(t1_times),
                mean_T2=np.mean(t2_times),
                calibration_stability=calibration_data.get_calibration_stability(list(subgraph.nodes)),
                expected_swaps=size - 1,  # Simplified estimate
                coherence_score=np.mean(t2_times) / np.mean(t1_times)  # T2/T1 ratio
            )
            
            island = Island(
                qubits=list(subgraph.nodes),
                edges=list(subgraph.edges),
                score=score,
                metrics=metrics
            )
            islands.append(island)
    
    # Sort by score (lower is better)
    return sorted(islands, key=lambda i: i.score)


def can_embed(scaffold_requirements: Any, island: Island) -> bool:
    """
    Check if island can embed the required scaffold.
    
    Args:
        scaffold_requirements: Scaffold embedding requirements
        island: Candidate island
        
    Returns:
        True if embedding is possible
    """
    # Simplified check - just verify size constraints
    if hasattr(scaffold_requirements, 'min_qubits'):
        if island.size < scaffold_requirements.min_qubits:
            return False
    
    if hasattr(scaffold_requirements, 'max_qubits'):
        if island.size > scaffold_requirements.max_qubits:
            return False
    
    # Check connectivity requirements if specified
    if hasattr(scaffold_requirements, 'required_edges'):
        island_edge_set = set(tuple(sorted(edge)) for edge in island.edges)
        for req_edge in scaffold_requirements.required_edges:
            if tuple(sorted(req_edge)) not in island_edge_set:
                return False
    
    return True


def select_island(algorithm_spec: AlgorithmSpec, discovered_islands: List[Island],
                 scaffold_requirements: Any, selection_policy: str = 'minimize_error') -> Island:
    """
    Select best island for the given algorithm.
    
    Args:
        algorithm_spec: Algorithm specification
        discovered_islands: List of candidate islands
        scaffold_requirements: Scaffold embedding requirements
        selection_policy: Selection strategy
        
    Returns:
        Selected island
        
    Raises:
        NoViableIslandError: If no suitable island found
    """
    # Filter to islands that can embed the scaffold
    viable = [
        island for island in discovered_islands
        if can_embed(scaffold_requirements, island)
    ]
    
    if not viable:
        raise NoViableIslandError(
            f"No island of size {scaffold_requirements.min_qubits}-"
            f"{scaffold_requirements.max_qubits} can embed the required topology"
        )
    
    # Apply selection policy
    if selection_policy == 'minimize_error':
        return min(viable, key=lambda i: i.score)
    elif selection_policy == 'minimize_size':
        return min(viable, key=lambda i: i.size)
    elif selection_policy == 'maximize_coherence':
        return max(viable, key=lambda i: i.metrics.mean_T2)
    elif selection_policy == 'balanced':
        # Balanced score combining multiple factors
        def balanced_score(island):
            return (
                island.score * 0.4 +  # Error score
                island.metrics.coherence_score * 0.3 +  # Coherence
                (1.0 / island.size) * 0.3  # Size preference
            )
        return min(viable, key=balanced_score)
    else:
        raise ValueError(f"Unknown selection policy: {selection_policy}")


def apply_orphan_policy(circuit: Any, island: Island, policy: Any, 
                       backend_topology: nx.Graph) -> Any:
    """
    Enforce orphan exclusion according to policy.
    
    Args:
        circuit: Quantum circuit
        island: Selected island
        policy: Orphan exclusion policy
        backend_topology: Full backend topology
        
    Returns:
        Modified circuit
        
    Raises:
        OrphanViolationError: If policy cannot be satisfied
    """
    from .core import OrphanViolationError
    
    all_qubits = set(backend_topology.nodes)
    island_qubits = set(island.qubits)
    orphans = all_qubits - island_qubits
    
    if policy.mode == 'strict':
        from qiskit.circuit import QuantumCircuit

        if not isinstance(circuit, QuantumCircuit):
            return circuit

        def _qubit_index(q):
            try:
                return circuit.find_bit(q).index
            except Exception:
                return circuit.qubits.index(q)

        new_circuit = QuantumCircuit(*circuit.qregs, *circuit.cregs, name=circuit.name)
        for inst in circuit.data:
            instr = getattr(inst, "operation", None)
            qargs = getattr(inst, "qubits", None)
            cargs = getattr(inst, "clbits", None)
            if instr is None:
                instr, qargs, cargs = inst

            q_indices = {_qubit_index(q) for q in qargs}
            if q_indices.issubset(island_qubits):
                new_circuit.append(instr, qargs, cargs)

        return new_circuit
        
    elif policy.mode == 'relaxed':
        # Allow only distant orphans if backend requires
        # Check distance constraints
        for orphan in orphans:
            min_distance = min(
                len(nx.shortest_path(backend_topology, orphan, island_q))
                for island_q in island_qubits
            )
            if min_distance < policy.distance_threshold:
                raise OrphanViolationError(
                    f"Qubit {orphan} too close to island (distance {min_distance})"
                )
                
    elif policy.mode == 'diagnostic':
        # Add measurement on specified orphans for crosstalk analysis
        if policy.allowed_orphans:
            from qiskit.circuit import QuantumCircuit, ClassicalRegister

            if not isinstance(circuit, QuantumCircuit):
                return circuit

            to_measure = [o for o in policy.allowed_orphans if o in orphans]
            if not to_measure:
                return circuit

            to_measure = [o for o in to_measure if isinstance(o, int) and 0 <= o < circuit.num_qubits]
            if not to_measure:
                return circuit

            needed = len(to_measure)
            available = len(circuit.clbits)
            if available < needed:
                circuit.add_register(ClassicalRegister(needed - available, "orphan_diag"))

            for i, orphan in enumerate(to_measure):
                circuit.measure(orphan, circuit.clbits[i])

    return circuit
