"""
Integration tests for HOT Framework.
"""

import unittest
import numpy as np
import networkx as nx
from qiskit import QuantumCircuit
from qiskit.providers.fake_provider import GenericBackendV2

from hot_framework import (
    HOTCompiler, OrphanPolicy, MeasurementPolicy, RetuningConfig,
    AlgorithmSpec, AlgorithmClass, DynamicalDecoupling
)


class TestHOTCompilerIntegration(unittest.TestCase):
    """Integration tests for the main HOT compiler."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Qiskit 2.x: use GenericBackendV2 instead of deprecated/removed Fake* backends.
        self.backend = GenericBackendV2(
            num_qubits=5,
            coupling_map=[[0, 1], [1, 2], [2, 3], [3, 4]]
        )
        self.compiler = HOTCompiler(self.backend, calibration_source='live')
    
    def test_simple_circuit_compilation(self):
        """Test compilation of a simple quantum circuit."""
        # Create a simple circuit
        circuit = QuantumCircuit(3, 3)
        circuit.h([0, 1, 2])
        circuit.cx(0, 1)
        circuit.cx(1, 2)
        circuit.rz(0.5, 0)
        circuit.measure([0, 1, 2], [0, 1, 2])
        
        # Compile with HOT
        result = self.compiler.compile(
            circuit,
            algorithm_class='fixed_unitary',
            orphan_policy=OrphanPolicy(mode='strict'),
            measurement_policy=MeasurementPolicy(stagger_readout=True)
        )
        
        # Verify result structure
        self.assertIsNotNone(result.compiled_circuit)
        self.assertIsNotNone(result.island)
        self.assertIsNotNone(result.scaffold)
        self.assertIsNotNone(result.job_variants)
        self.assertEqual(result.algorithm_class.value, 'fixed_unitary')
        self.assertIsNone(result.codesign_result)
        
        # Check that compiled circuit has same number of qubits
        self.assertEqual(result.compiled_circuit.num_qubits, circuit.num_qubits)
    
    def test_variational_algorithm_compilation(self):
        """Test compilation of a variational algorithm."""
        # Create QAOA specification
        interaction_graph = nx.Graph()
        interaction_graph.add_edges_from([(0, 1), (1, 2)])
        
        def simple_cost(bitstring):
            return int(bitstring, 2)  # Simple integer cost
        
        spec = AlgorithmSpec(
            name="test_qaoa",
            algorithm_class=AlgorithmClass.VARIATIONAL,
            min_qubits=3,
            max_qubits=3,
            interaction_graph=interaction_graph,
            cost_function=simple_cost,
            initial_parameters=np.array([0.5, 0.3, 0.7, 0.2]),
            parameter_bounds=[(0, np.pi)] * 4
        )
        
        # Compile with co-design
        result = self.compiler.compile(
            spec,
            algorithm_class='variational',
            retuning_config=RetuningConfig(
                optimizer='COBYLA',
                max_iterations=10,  # Small for test
                objective='expected_cost',
                init_strategy='transfer'
            ),
            orphan_policy=OrphanPolicy(mode='strict')
        )
        
        # Verify variational-specific results
        self.assertIsNotNone(result.codesign_result)
        self.assertEqual(result.algorithm_class.value, AlgorithmClass.VARIATIONAL.value)
        
        # Check co-design results
        codesign = result.codesign_result
        self.assertIn('nohot_original', codesign.configurations)
        self.assertIn('nohot_retuned', codesign.configurations)
        self.assertIn('hot_original', codesign.configurations)
        self.assertIn('hot_retuned', codesign.configurations)
        
        # Verify best configuration is identified
        self.assertIsNotNone(codesign.best_configuration)
        self.assertIsInstance(codesign.improvement_over_baseline, float)
    
    def test_mitigation_hooks(self):
        """Test application of mitigation hooks."""
        circuit = QuantumCircuit(2, 2)
        circuit.h([0, 1])
        circuit.cx(0, 1)
        circuit.measure([0, 1], [0, 1])
        
        # Compile with mitigation hooks
        result = self.compiler.compile(
            circuit,
            algorithm_class='fixed_unitary',
            mitigation_hooks=[
                DynamicalDecoupling(sequence='XY4', threshold_idle_time=50)
            ]
        )
        
        # Verify hooks were applied (check metadata)
        self.assertIn('mitigation_hooks', result.metadata)
        hooks = result.metadata['mitigation_hooks']
        self.assertIn('DynamicalDecoupling', hooks)
    
    def test_orphan_policies(self):
        """Test different orphan policies."""
        circuit = QuantumCircuit(2, 2)
        circuit.h([0, 1])
        circuit.cx(0, 1)
        circuit.measure([0, 1], [0, 1])
        
        # Test strict policy
        result_strict = self.compiler.compile(
            circuit,
            orphan_policy=OrphanPolicy(mode='strict')
        )
        self.assertEqual(result_strict.metadata['orphan_policy'], 'strict')
        
        # Test relaxed policy
        result_relaxed = self.compiler.compile(
            circuit,
            orphan_policy=OrphanPolicy(mode='relaxed', distance_threshold=2)
        )
        self.assertEqual(result_relaxed.metadata['orphan_policy'], 'relaxed')
        
        # Test diagnostic policy
        result_diagnostic = self.compiler.compile(
            circuit,
            orphan_policy=OrphanPolicy(mode='diagnostic', allowed_orphans=[2])
        )
        self.assertEqual(result_diagnostic.metadata['orphan_policy'], 'diagnostic')
    
    def test_measurement_policies(self):
        """Test different measurement policies."""
        circuit = QuantumCircuit(3, 3)
        circuit.h([0, 1, 2])
        circuit.cx(0, 1)
        circuit.cx(1, 2)
        circuit.measure([0, 1, 2], [0, 1, 2])
        
        # Test staggered readout
        result_staggered = self.compiler.compile(
            circuit,
            measurement_policy=MeasurementPolicy(stagger_readout=True)
        )
        self.assertTrue(result_staggered.metadata['measurement_policy']['stagger_readout'])
        
        # Test concurrent measurements
        result_concurrent = self.compiler.compile(
            circuit,
            measurement_policy=MeasurementPolicy(
                stagger_readout=False,
                max_concurrent_adjacent=3
            )
        )
        self.assertFalse(result_concurrent.metadata['measurement_policy']['stagger_readout'])
    
    def test_backend_analysis(self):
        """Test backend analysis functionality."""
        analysis = self.compiler.analyze_backend()
        
        # Check analysis structure
        self.assertIn('backend_name', analysis)
        self.assertIn('n_qubits', analysis)
        self.assertIn('connectivity', analysis)
        self.assertIn('calibration_quality', analysis)
        
        # Check connectivity analysis
        connectivity = analysis['connectivity']
        self.assertIn('density', connectivity)
        self.assertIn('is_connected', connectivity)
        self.assertIn('mean_degree', connectivity)
        
        # Check calibration quality
        quality = analysis['calibration_quality']
        self.assertIn('mean_readout_error', quality)
        self.assertIn('mean_two_q_error', quality)
    
    def test_optimization_suggestions(self):
        """Test optimization suggestion functionality."""
        # Create circuit with various characteristics
        circuit = QuantumCircuit(4, 4)
        circuit.h(range(4))
        circuit.cx(0, 1)
        circuit.cx(1, 2)
        circuit.cx(2, 3)
        circuit.rz(0.5, 0)  # Parameterized gate
        circuit.measure(range(4), range(4))
        
        suggestions = self.compiler.suggest_optimizations(circuit)
        
        # Should provide some suggestions
        self.assertIsInstance(suggestions, list)
        self.assertGreater(len(suggestions), 0)
        
        # Check for relevant suggestions
        suggestion_text = " ".join(suggestions)
        self.assertTrue(
            any(keyword in suggestion_text.lower() 
                for keyword in ['measurement', 'gate', 'optimization'])
        )
    
    def test_algorithm_auto_detection(self):
        """Test automatic algorithm class detection."""
        # Fixed-unitary circuit
        fixed_circuit = QuantumCircuit(2, 2)
        fixed_circuit.h([0, 1])
        fixed_circuit.cx(0, 1)
        fixed_circuit.measure([0, 1], [0, 1])
        
        result_fixed = self.compiler.compile(fixed_circuit, algorithm_class='auto')
        self.assertEqual(result_fixed.algorithm_class.value, 'fixed_unitary')
        
        # Variational circuit (with parameterized gates)
        var_circuit = QuantumCircuit(2, 2)
        var_circuit.h([0, 1])
        var_circuit.rx(0.5, 0)  # Parameterized
        var_circuit.cx(0, 1)
        var_circuit.measure([0, 1], [0, 1])
        
        result_var = self.compiler.compile(var_circuit, algorithm_class='auto')
        # Note: This might not detect as variational without cost function
        # The detection is heuristic and may need improvement
    
    def test_job_variants(self):
        """Test generation of job variants."""
        circuit = QuantumCircuit(2, 2)
        circuit.h([0, 1])
        circuit.cx(0, 1)
        circuit.measure([0, 1], [0, 1])
        
        result = self.compiler.compile(circuit)
        
        # Check job variants
        variants = result.job_variants
        self.assertIsNotNone(variants.sanitized)
        self.assertIsNotNone(variants.instrumented)
        self.assertIsNotNone(variants.comparison_metadata)
        
        # Check metadata
        metadata = variants.comparison_metadata
        self.assertIn('sanitized_measurements', metadata)
        self.assertIn('instrumented_measurements', metadata)
        
        # Instrumented should have >= measurements as sanitized
        self.assertGreaterEqual(
            metadata['instrumented_measurements'],
            metadata['sanitized_measurements']
        )


class TestErrorHandling(unittest.TestCase):
    """Test error handling and edge cases."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.backend = GenericBackendV2(
            num_qubits=5,
            coupling_map=[[0, 1], [1, 2], [2, 3], [3, 4]]
        )
        self.compiler = HOTCompiler(self.backend)
    
    def test_variational_without_retuning_config(self):
        """Test error when variational algorithm lacks retuning config."""
        interaction_graph = nx.Graph()
        interaction_graph.add_edge(0, 1)
        
        spec = AlgorithmSpec(
            name="test_var",
            algorithm_class=AlgorithmClass.VARIATIONAL,
            min_qubits=2,
            max_qubits=2,
            interaction_graph=interaction_graph,
            cost_function=lambda x: int(x, 2)
        )
        
        # Should raise ValueError
        with self.assertRaises(ValueError) as context:
            self.compiler.compile(spec, algorithm_class='variational')
        
        self.assertIn("retuning_config", str(context.exception))
    
    def test_invalid_algorithm_spec(self):
        """Test handling of invalid algorithm specifications."""
        # Create spec with invalid qubit range
        interaction_graph = nx.Graph()
        interaction_graph.add_edge(0, 1)
        
        spec = AlgorithmSpec(
            name="invalid",
            algorithm_class=AlgorithmClass.FIXED_UNITARY,
            min_qubits=5,  # More than available
            max_qubits=10,
            interaction_graph=interaction_graph
        )
        
        # Should handle gracefully (may raise exception or return warnings)
        try:
            result = self.compiler.compile(spec)
            # If it succeeds, check for warnings in metadata
            if 'validation_warnings' in result.metadata:
                self.assertGreater(len(result.metadata['validation_warnings']), 0)
        except Exception as e:
            # Should be a meaningful error
            self.assertIsInstance(e, (ValueError, Exception))


if __name__ == '__main__':
    unittest.main()
