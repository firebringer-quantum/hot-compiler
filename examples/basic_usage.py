"""
Basic usage examples for the HOT Framework.
"""

import numpy as np
import networkx as nx
from qiskit import QuantumCircuit
from qiskit.providers.fake_provider import GenericBackendV2

from hot_framework import (
    HOTCompiler, OrphanPolicy, MeasurementPolicy, RetuningConfig,
    AlgorithmSpec, AlgorithmClass, DynamicalDecoupling, ZeroNoiseExtrapolation
)


def example_fixed_unitary():
    """Example: Fixed-unitary algorithm (Grover's search)."""
    print("=== Fixed-Unitary Algorithm Example ===")
    
    # Create a simple Grover circuit
    def create_grover_circuit(n_qubits=3):
        circuit = QuantumCircuit(n_qubits, n_qubits)
        
        # Initialize
        circuit.h(range(n_qubits))
        
        # Oracle (simplified - marks |111> state)
        circuit.x(range(n_qubits))
        circuit.h(n_qubits-1)
        circuit.mcx(list(range(n_qubits-1)), n_qubits-1)
        circuit.h(n_qubits-1)
        circuit.x(range(n_qubits))
        
        # Diffusion operator
        circuit.h(range(n_qubits))
        circuit.x(range(n_qubits))
        circuit.h(n_qubits-1)
        circuit.mcx(list(range(n_qubits-1)), n_qubits-1)
        circuit.h(n_qubits-1)
        circuit.x(range(n_qubits))
        circuit.h(range(n_qubits))
        
        # Measurement
        circuit.measure(range(n_qubits), range(n_qubits))
        
        return circuit
    
    # Create circuit
    grover_circuit = create_grover_circuit(3)
    
    # Use a generic backend for demonstration
    backend = GenericBackendV2(
        num_qubits=5,
        coupling_map=[[0, 1], [1, 2], [2, 3], [3, 4]],
    )
    
    # Initialize HOT compiler
    compiler = HOTCompiler(backend, calibration_source='live')
    
    # Compile with HOT
    result = compiler.compile(
        grover_circuit,
        algorithm_class='fixed_unitary',
        orphan_policy=OrphanPolicy(mode='strict'),
        measurement_policy=MeasurementPolicy(stagger_readout=True),
        mitigation_hooks=[
            DynamicalDecoupling(sequence='XY4', threshold_idle_time=100)
        ]
    )
    
    # Print results
    print(result.summary())
    print(f"Compiled circuit depth: {result.compiled_circuit.depth()}")
    print(f"Original circuit depth: {grover_circuit.depth()}")
    
    return result


def example_variational():
    """Example: Variational algorithm (QAOA)."""
    print("\n=== Variational Algorithm Example ===")
    
    # Create QAOA specification
    def create_qaoa_spec():
        # Simple MaxCut problem on 3 nodes
        n_qubits = 3
        
        # Create interaction graph (triangle)
        interaction_graph = nx.Graph()
        interaction_graph.add_edges_from([(0, 1), (1, 2), (0, 2)])
        
        # Cost function (MaxCut)
        def maxcut_cost(bitstring):
            cut_size = 0
            for i, j in interaction_graph.edges():
                if bitstring[i] != bitstring[j]:
                    cut_size += 1
            return -cut_size  # Negative because we minimize
        
        # Initial parameters
        initial_params = np.array([0.5, 0.5, 0.3, 0.3])  # 2 layers * 2 parameters
        
        return AlgorithmSpec(
            name="maxcut_qaoa",
            algorithm_class=AlgorithmClass.VARIATIONAL,
            min_qubits=n_qubits,
            max_qubits=n_qubits,
            interaction_graph=interaction_graph,
            cost_function=maxcut_cost,
            initial_parameters=initial_params,
            parameter_bounds=[(0, 2*np.pi)] * len(initial_params),
            output_qubits=list(range(n_qubits))
        )
    
    # Create specification
    qaoa_spec = create_qaoa_spec()
    
    # Use a generic backend
    backend = GenericBackendV2(
        num_qubits=5,
        coupling_map=[[0, 1], [1, 2], [2, 3], [3, 4]],
    )
    
    # Initialize HOT compiler
    compiler = HOTCompiler(backend, calibration_source='live')
    
    # Compile with co-design
    result = compiler.compile(
        qaoa_spec,
        algorithm_class='variational',
        retuning_config=RetuningConfig(
            optimizer='COBYLA',
            max_iterations=50,  # Reduced for demo
            objective='expected_cost',
            init_strategy='transfer'
        ),
        orphan_policy=OrphanPolicy(mode='strict'),
        mitigation_hooks=[
            DynamicalDecoupling(sequence='XY4'),
            ZeroNoiseExtrapolation(scale_factors=[1.0, 1.5, 2.0])
        ]
    )
    
    # Print results
    print(result.summary())
    
    if result.codesign_result:
        print("\nCo-design Results:")
        for name, config_result in result.codesign_result.configurations.items():
            print(f"  {name}: expected={config_result.expected_cost:.4f}, "
                  f"best={config_result.best_of_sample:.4f}")
        
        print(f"\nBest configuration: {result.codesign_result.best_configuration}")
        print(f"Improvement: {result.codesign_result.improvement_over_baseline:.1%}")
    
    return result


def example_backend_analysis():
    """Example: Backend analysis and optimization suggestions."""
    print("\n=== Backend Analysis Example ===")
    
    # Use a generic backend
    backend = GenericBackendV2(
        num_qubits=5,
        coupling_map=[[0, 1], [1, 2], [2, 3], [3, 4]],
    )
    
    # Initialize HOT compiler
    compiler = HOTCompiler(backend, calibration_source='live')
    
    # Analyze backend
    analysis = compiler.analyze_backend()
    
    print(f"Backend: {analysis['backend_name']}")
    print(f"Qubits: {analysis['n_qubits']}")
    print(f"Connectivity density: {analysis['connectivity']['density']:.3f}")
    print(f"Mean readout error: {analysis['calibration_quality']['mean_readout_error']:.4f}")
    print(f"Mean 2Q error: {analysis['calibration_quality']['mean_two_q_error']:.4f}")
    
    # Create a test circuit for suggestions
    test_circuit = QuantumCircuit(4, 4)
    test_circuit.h(range(4))
    test_circuit.cx(0, 1)
    test_circuit.cx(1, 2)
    test_circuit.cx(2, 3)
    test_circuit.rz(0.5, 0)
    test_circuit.cx(0, 1)
    test_circuit.measure(range(4), range(4))
    
    # Get optimization suggestions
    suggestions = compiler.suggest_optimizations(test_circuit)
    
    print("\nOptimization Suggestions:")
    for suggestion in suggestions:
        print(f"  - {suggestion}")


def example_measurement_hygiene():
    """Example: Measurement hygiene demonstration."""
    print("\n=== Measurement Hygiene Example ===")
    
    # Create circuit with multiple measurements
    circuit = QuantumCircuit(4, 6)
    
    # Some gates
    circuit.h(range(4))
    circuit.cx(0, 1)
    circuit.cx(2, 3)
    circuit.rz(0.3, 0)
    
    # Mid-circuit measurement (diagnostic)
    circuit.measure(0, 0)
    circuit.measure(1, 1)
    
    # More gates
    circuit.cx(1, 2)
    circuit.h(3)
    
    # Final measurements
    circuit.measure(range(4), range(2, 6))
    
    # Use a generic backend
    backend = GenericBackendV2(
        num_qubits=5,
        coupling_map=[[0, 1], [1, 2], [2, 3], [3, 4]],
    )
    compiler = HOTCompiler(backend)
    
    # Compile with different measurement policies
    strict_policy = MeasurementPolicy(
        allow_mid_circuit=False,
        stagger_readout=True,
        max_concurrent_adjacent=1
    )
    
    relaxed_policy = MeasurementPolicy(
        allow_mid_circuit=True,
        stagger_readout=False,
        max_concurrent_adjacent=2
    )
    
    original_meas = 0
    for inst in circuit.data:
        instr = getattr(inst, "operation", None)
        if instr is None:
            instr, _, _ = inst
        if instr.name == 'measure':
            original_meas += 1
    print(f"Original circuit measurements: {original_meas}")
    
    # Compile with strict policy
    result_strict = compiler.compile(
        circuit,
        measurement_policy=strict_policy,
        orphan_policy=OrphanPolicy(mode='strict')
    )
    
    # Compile with relaxed policy
    result_relaxed = compiler.compile(
        circuit,
        measurement_policy=relaxed_policy,
        orphan_policy=OrphanPolicy(mode='relaxed')
    )
    
    print(f"Strict policy measurements: {result_strict.job_variants.comparison_metadata['sanitized_measurements']}")
    print(f"Relaxed policy measurements: {result_relaxed.job_variants.comparison_metadata['sanitized_measurements']}")
    print(f"Diagnostic measurements available: {result_strict.job_variants.comparison_metadata['diagnostic_measurements']}")


if __name__ == "__main__":
    """Run all examples."""
    print("HOT Framework Examples")
    print("====================")
    
    try:
        # Run examples
        example_fixed_unitary()
        example_variational()
        example_backend_analysis()
        example_measurement_hygiene()
        
        print("\n=== All Examples Completed Successfully ===")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()
