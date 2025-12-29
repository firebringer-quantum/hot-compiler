"""
Tests for HOT Framework core components.
"""

import unittest
import numpy as np
import networkx as nx
from qiskit import QuantumCircuit

from hot_framework.core import (
    Island, IslandMetrics, Scaffold, OrphanPolicy, MeasurementPolicy,
    AlgorithmSpec, AlgorithmClass, CalibrationData, HOTResult,
    MeasurementClass, NoViableIslandError
)


class TestCoreComponents(unittest.TestCase):
    """Test core data structures and utilities."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create test topology
        self.topology = nx.Graph()
        self.topology.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0)])
        
        # Create test calibration data
        self.calibration = CalibrationData(
            timestamp="2024-01-01T00:00:00Z",
            backend_name="test_backend",
            topology=self.topology,
            readout_errors={0: 0.01, 1: 0.02, 2: 0.015, 3: 0.025},
            T1_times={0: 100, 1: 90, 2: 110, 3: 85},
            T2_times={0: 80, 1: 75, 2: 85, 3: 70},
            two_qubit_errors={(0, 1): 0.02, (1, 2): 0.025, (2, 3): 0.018, (3, 0): 0.022}
        )
    
    def test_island_creation(self):
        """Test Island creation and properties."""
        metrics = IslandMetrics(
            mean_readout_error=0.02,
            mean_two_qubit_error=0.025,
            mean_T1=100.0,
            mean_T2=80.0,
            calibration_stability=0.9,
            expected_swaps=2,
            coherence_score=0.8
        )
        
        island = Island(
            qubits=[0, 1, 2],
            edges=[(0, 1), (1, 2)],
            score=0.5,
            metrics=metrics
        )
        
        self.assertEqual(island.size, 3)
        self.assertEqual(island.qubits, [0, 1, 2])
        self.assertEqual(island.score, 0.5)
    
    def test_scaffold_creation(self):
        """Test Scaffold creation and layout."""
        metrics = IslandMetrics(
            mean_readout_error=0.02,
            mean_two_qubit_error=0.025,
            mean_T1=100.0,
            mean_T2=80.0,
            calibration_stability=0.9,
            expected_swaps=1,
            coherence_score=0.8
        )
        
        island = Island(
            qubits=[0, 1],
            edges=[(0, 1)],
            score=0.3,
            metrics=metrics
        )
        
        scaffold = Scaffold(
            mapping={0: 0, 1: 1},
            swap_schedule=[],
            estimated_error=0.1,
            topology_pattern="pair",
            island=island
        )
        
        self.assertEqual(scaffold.mapping, {0: 0, 1: 1})
        self.assertEqual(scaffold.topology_pattern, "pair")
        self.assertIsNotNone(scaffold.get_layout())
    
    def test_orphan_policy(self):
        """Test OrphanPolicy creation and validation."""
        strict_policy = OrphanPolicy(mode='strict')
        relaxed_policy = OrphanPolicy(mode='relaxed', distance_threshold=3)
        diagnostic_policy = OrphanPolicy(mode='diagnostic', allowed_orphans=[4, 5])
        
        self.assertEqual(strict_policy.mode, 'strict')
        self.assertEqual(relaxed_policy.distance_threshold, 3)
        self.assertEqual(diagnostic_policy.allowed_orphans, [4, 5])
    
    def test_measurement_policy(self):
        """Test MeasurementPolicy creation."""
        policy = MeasurementPolicy(
            allow_mid_circuit=False,
            stagger_readout=True,
            max_concurrent_adjacent=2
        )
        
        self.assertFalse(policy.allow_mid_circuit)
        self.assertTrue(policy.stagger_readout)
        self.assertEqual(policy.max_concurrent_adjacent, 2)
    
    def test_algorithm_spec(self):
        """Test AlgorithmSpec creation and validation."""
        interaction_graph = nx.Graph()
        interaction_graph.add_edges_from([(0, 1), (1, 2)])
        
        spec = AlgorithmSpec(
            name="test_algorithm",
            algorithm_class=AlgorithmClass.VARIATIONAL,
            min_qubits=3,
            max_qubits=5,
            interaction_graph=interaction_graph,
            initial_parameters=np.array([0.5, 0.3, 0.7, 0.2]),
            cost_function=lambda x: x.count('1')
        )
        
        self.assertEqual(spec.name, "test_algorithm")
        self.assertEqual(spec.algorithm_class, AlgorithmClass.VARIATIONAL)
        self.assertEqual(spec.min_qubits, 3)
        self.assertEqual(spec.max_qubits, 5)
        self.assertIsNotNone(spec.initial_parameters)
        
        # Test random parameter generation
        random_params = spec.random_initial_parameters()
        self.assertEqual(len(random_params), len(spec.initial_parameters))
    
    def test_calibration_data(self):
        """Test CalibrationData methods."""
        # Test qubit error calculation
        qubit_error = self.calibration.get_qubit_error(0)
        expected_error = self.calibration.readout_errors[0] + 0.1 * (1.0 / min(
            self.calibration.T1_times[0], self.calibration.T2_times[0]
        ))
        self.assertAlmostEqual(qubit_error, expected_error, places=6)
        
        # Test edge error calculation
        edge_error = self.calibration.get_edge_error((0, 1))
        self.assertEqual(edge_error, self.calibration.two_qubit_errors[(0, 1)])
        
        # Test calibration stability
        stability = self.calibration.get_calibration_stability([0, 1])
        self.assertEqual(stability, 1.0)  # No historical data
    
    def test_measurement_class(self):
        """Test MeasurementClass enum."""
        self.assertEqual(MeasurementClass.ESSENTIAL.value, "essential")
        self.assertEqual(MeasurementClass.DIAGNOSTIC.value, "diagnostic")
        self.assertEqual(MeasurementClass.EXTRANEOUS.value, "extraneous")


class TestAlgorithmSpecValidation(unittest.TestCase):
    """Test algorithm specification validation."""
    
    def test_valid_spec(self):
        """Test validation of valid specification."""
        interaction_graph = nx.Graph()
        interaction_graph.add_edges_from([(0, 1)])
        
        spec = AlgorithmSpec(
            name="valid_algorithm",
            algorithm_class=AlgorithmClass.FIXED_UNITARY,
            min_qubits=2,
            max_qubits=4,
            interaction_graph=interaction_graph
        )
        
        from hot_framework.utils import validate_algorithm_spec
        warnings = validate_algorithm_spec(spec)
        self.assertEqual(len(warnings), 0)
    
    def test_invalid_spec(self):
        """Test validation of invalid specification."""
        interaction_graph = nx.Graph()  # Empty graph
        
        spec = AlgorithmSpec(
            name="invalid_algorithm",
            algorithm_class=AlgorithmClass.VARIATIONAL,
            min_qubits=0,  # Invalid
            max_qubits=2,
            interaction_graph=interaction_graph,
            initial_parameters=None,  # Missing for variational
            cost_function=None  # Missing for variational
        )
        
        from hot_framework.utils import validate_algorithm_spec
        warnings = validate_algorithm_spec(spec)
        self.assertGreater(len(warnings), 0)
        
        # Check for specific warnings
        warning_text = " ".join(warnings)
        self.assertIn("min_qubits must be positive", warning_text)
        self.assertIn("interaction_graph must have nodes", warning_text)


class TestDistributionMetrics(unittest.TestCase):
    """Test distribution metrics computation."""
    
    def test_compute_distribution_metrics(self):
        """Test distribution metrics calculation."""
        from hot_framework.utils import compute_distribution_metrics
        
        # Create test counts
        counts = {
            '000': 100,
            '001': 50,
            '010': 30,
            '011': 20,
            '100': 80,
            '101': 40,
            '110': 25,
            '111': 15
        }
        
        # Simple cost function (number of 1s)
        cost_function = lambda bitstring: bitstring.count('1')
        
        metrics = compute_distribution_metrics(counts, cost_function)
        
        # Check basic metrics
        self.assertGreater(metrics.entropy, 0)
        self.assertGreater(metrics.expected_cost, 0)
        self.assertGreaterEqual(metrics.best_of_sample, 0)
        self.assertLessEqual(metrics.top_k_probability, 1.0)
        
        # Check with baseline
        baseline_counts = {'000': 200, '111': 200}
        metrics_with_baseline = compute_distribution_metrics(
            counts, cost_function, baseline_counts
        )
        
        self.assertIsNotNone(metrics_with_baseline.tvd)
        self.assertIsNotNone(metrics_with_baseline.kl_divergence)


if __name__ == '__main__':
    unittest.main()
