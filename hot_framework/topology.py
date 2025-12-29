"""
Topology analysis and embedding utilities for HOT Framework.
"""

from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import networkx as nx

from .core import CalibrationData


def compute_embedding_score(embedding: Dict[int, int], calibration_data: CalibrationData,
                           weights: Dict[str, float], interaction_graph: Optional[nx.Graph] = None) -> float:
    """
    Compute error score for a candidate embedding.
    
    Args:
        embedding: Mapping from logical to physical qubits
        calibration_data: Device calibration information
        weights: Weights for different error contributions
        
    Returns:
        Aggregate error score (lower is better)
    """
    # Initialize score components
    two_qubit_error = 0.0
    readout_error = 0.0
    coherence_penalty = 0.0
    
    # Compute two-qubit error for mapped interaction edges.
    if interaction_graph is not None:
        for logical_u, logical_v in interaction_graph.edges:
            if logical_u in embedding and logical_v in embedding:
                edge = tuple(sorted((embedding[logical_u], embedding[logical_v])))
                if edge in calibration_data.two_qubit_errors:
                    two_qubit_error += calibration_data.get_edge_error(edge)
    else:
        for logical_q1, physical_q1 in embedding.items():
            for logical_q2, physical_q2 in embedding.items():
                if logical_q1 < logical_q2:  # Avoid double counting
                    edge = tuple(sorted((physical_q1, physical_q2)))
                    if edge in calibration_data.two_qubit_errors:
                        two_qubit_error += calibration_data.get_edge_error(edge)
    
    # Compute readout error for mapped qubits
    for physical_q in embedding.values():
        readout_error += calibration_data.get_qubit_error(physical_q)
    
    # Compute coherence penalty
    t1_times = [calibration_data.T1_times.get(q, 100.0) for q in embedding.values()]
    t2_times = [calibration_data.T2_times.get(q, 100.0) for q in embedding.values()]
    
    for t1, t2 in zip(t1_times, t2_times):
        coherence_penalty += 1.0 / min(t1, t2)
    
    # Combine with weights
    score = (
        weights['two_qubit_error'] * two_qubit_error +
        weights['readout_error'] * readout_error +
        weights['T1_T2_ratio'] * coherence_penalty
    )
    
    return score


def compute_swap_schedule(embedding: Dict[int, int], 
                         calibration_data: CalibrationData) -> List[Dict[str, Any]]:
    """
    Compute required SWAP operations for embedding.
    
    Args:
        embedding: Logical to physical qubit mapping
        calibration_data: Device calibration information
        
    Returns:
        List of SWAP operations with timing and error estimates
    """
    swap_schedule = []
    
    # Simplified implementation - in practice would analyze circuit requirements
    # and compute minimal SWAP paths
    
    # For now, assume no SWAPs needed if embedding is direct
    # Real implementation would:
    # 1. Analyze circuit connectivity requirements
    # 2. Find shortest paths between non-adjacent qubits
    # 3. Insert SWAPs along those paths
    # 4. Estimate timing and error for each SWAP
    
    return swap_schedule


def detect_pattern(subgraph: nx.Graph) -> str:
    """
    Detect topological pattern of a subgraph.
    
    Args:
        subgraph: Subgraph to analyze
        
    Returns:
        String describing detected pattern
    """
    n_nodes = subgraph.number_of_nodes()
    n_edges = subgraph.number_of_edges()
    
    # Check for common patterns
    if n_nodes == 2 and n_edges == 1:
        return "pair"
    elif n_nodes == 3:
        if n_edges == 2:
            return "line_3"
        elif n_edges == 3:
            return "triangle"
    elif n_nodes == 4:
        if n_edges == 3:
            return "line_4"
        elif n_edges == 4:
            # Could be square or star
            degrees = [d for _, d in subgraph.degree()]
            if degrees.count(2) == 4:  # All nodes degree 2
                return "square"
            elif degrees.count(3) == 1 and degrees.count(1) == 3:  # One center
                return "star_4"
        elif n_edges == 5:
            return "diamond"
        elif n_edges == 6:
            return "complete_4"
    elif n_nodes == 5:
        if n_edges == 4:
            return "line_5"
        elif n_edges == 5:
            return "ring_5"
        elif n_edges == 8:
            return "cross_5"
    
    # Generic description
    if n_edges == n_nodes - 1:
        return f"tree_{n_nodes}"
    elif n_edges == n_nodes:
        return f"ring_{n_nodes}"
    elif n_edges > n_nodes:
        return f"dense_{n_nodes}"
    else:
        return f"sparse_{n_nodes}"


def distance(qubit1: int, qubit2: int, topology: nx.Graph) -> int:
    """
    Compute shortest path distance between two qubits.
    
    Args:
        qubit1: First qubit
        qubit2: Second qubit
        topology: Backend connectivity graph
        
    Returns:
        Shortest path distance
    """
    try:
        return nx.shortest_path_length(topology, qubit1, qubit2)
    except nx.NetworkXNoPath:
        return float('inf')


def get_neighboring_qubits(qubit: int, topology: nx.Graph) -> List[int]:
    """
    Get direct neighbors of a qubit.
    
    Args:
        qubit: Target qubit
        topology: Backend connectivity graph
        
    Returns:
        List of neighboring qubits
    """
    return list(topology.neighbors(qubit))


def analyze_connectivity_pattern(topology: nx.Graph) -> Dict[str, Any]:
    """
    Analyze overall connectivity pattern of a topology.
    
    Args:
        topology: Backend connectivity graph
        
    Returns:
        Dictionary with connectivity metrics
    """
    analysis = {}
    
    # Basic metrics
    analysis['n_nodes'] = topology.number_of_nodes()
    analysis['n_edges'] = topology.number_of_edges()
    analysis['density'] = nx.density(topology)
    
    # Degree distribution
    degrees = [d for _, d in topology.degree()]
    analysis['min_degree'] = min(degrees)
    analysis['max_degree'] = max(degrees)
    analysis['mean_degree'] = np.mean(degrees)
    analysis['degree_std'] = np.std(degrees)
    
    # Connectivity
    analysis['is_connected'] = nx.is_connected(topology)
    if analysis['is_connected']:
        analysis['diameter'] = nx.diameter(topology)
        analysis['avg_shortest_path'] = nx.average_shortest_path_length(topology)
    
    # Clustering
    analysis['avg_clustering'] = nx.average_clustering(topology)
    
    # Centrality measures
    betweenness = nx.betweenness_centrality(topology)
    analysis['max_betweenness'] = max(betweenness.values())
    analysis['avg_betweenness'] = np.mean(list(betweenness.values()))
    
    return analysis


def find_optimal_paths(topology: nx.Graph, source_qubits: List[int], 
                      target_qubits: List[int]) -> Dict[Tuple[int, int], List[int]]:
    """
    Find optimal paths between source and target qubits.
    
    Args:
        topology: Backend connectivity graph
        source_qubits: List of source qubits
        target_qubits: List of target qubits
        
    Returns:
        Dictionary mapping (source, target) to optimal path
    """
    paths = {}
    
    for source in source_qubits:
        for target in target_qubits:
            if source != target:
                try:
                    path = nx.shortest_path(topology, source, target)
                    paths[(source, target)] = path
                except nx.NetworkXNoPath:
                    paths[(source, target)] = []
    
    return paths


def estimate_crosstalk_potential(topology: nx.Graph, island_qubits: List[int],
                                distance_threshold: int = 2) -> Dict[str, Any]:
    """
    Estimate crosstalk potential for an island.
    
    Args:
        topology: Backend connectivity graph
        island_qubits: Qubits in the selected island
        distance_threshold: Distance threshold for crosstalk consideration
        
    Returns:
        Crosstalk analysis metrics
    """
    island_set = set(island_qubits)
    all_qubits = set(topology.nodes())
    nearby_qubits = set()
    
    # Find qubits within threshold distance
    for q in all_qubits:
        if q not in island_set:
            min_dist = min(
                distance(q, island_q, topology) 
                for island_q in island_qubits
            )
            if min_dist <= distance_threshold:
                nearby_qubits.add(q)
    
    # Compute crosstalk metrics
    n_nearby = len(nearby_qubits)
    n_island = len(island_qubits)
    crosstalk_ratio = n_nearby / n_island if n_island > 0 else 0
    
    # Find most problematic nearby qubits
    problematic_qubits = []
    for nearby in nearby_qubits:
        min_dist = min(
            distance(nearby, island_q, topology) 
            for island_q in island_qubits
        )
        problematic_qubits.append((nearby, min_dist))
    
    problematic_qubits.sort(key=lambda x: x[1])  # Sort by distance
    
    return {
        'island_qubits': island_qubits,
        'nearby_qubits': list(nearby_qubits),
        'n_nearby': n_nearby,
        'crosstalk_ratio': crosstalk_ratio,
        'problematic_qubits': problematic_qubits[:5],  # Top 5 worst
        'distance_threshold': distance_threshold
    }
