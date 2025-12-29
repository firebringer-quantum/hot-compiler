"""
Variational algorithm co-design workflow for HOT Framework.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable, Literal, Protocol, Mapping, Union
import numpy as np
from scipy.optimize import minimize
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
import networkx as nx

from .core import (
    AlgorithmSpec, AlgorithmClass, Scaffold, Island, CalibrationData,
    HOTResult, DistributionMetrics
)
from .mitigation import compute_scaffold
from .island_selection import discover_islands, select_island
from .measurement import classify_measurements, apply_measurement_policy, generate_job_variants
from .utils import compute_distribution_metrics


def normalize_counts(quasi_dist: Mapping[Union[int, str], float], n_qubits: int, shots: int) -> Dict[str, int]:
    """Normalize quasi distribution results into bitstring-keyed integer counts."""
    counts: Dict[str, int] = {}
    for key, prob in quasi_dist.items():
        if isinstance(key, int):
            bitstring = format(key, f"0{n_qubits}b")
        else:
            bitstring = str(key)
        counts[bitstring] = counts.get(bitstring, 0) + int(round(float(prob) * shots))
    return counts


class SamplerProvider(Protocol):
    def sample_counts(self, circuit: QuantumCircuit, shots: int) -> Dict[str, int]:
        ...


class LocalStatevectorSampler:
    def sample_counts(self, circuit: QuantumCircuit, shots: int) -> Dict[str, int]:
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


@dataclass
class RetuningConfig:
    """Configuration for variational parameter retuning."""
    
    optimizer: str = 'COBYLA'
    max_iterations: int = 100
    objective: Literal['expected_cost', 'best_of_sample', 'cvar'] = 'expected_cost'
    cvar_alpha: float = 0.1
    
    # Initialization strategy for new layout
    init_strategy: Literal['random', 'transfer', 'perturb'] = 'transfer'
    perturbation_scale: float = 0.1
    
    # Early stopping
    convergence_threshold: float = 1e-4
    patience: int = 10
    
    # Sampling
    shots_per_iteration: int = 8192
    sampler_options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConfigurationResult:
    """Result for a single configuration evaluation."""
    
    name: str
    scaffold: Scaffold
    island: Island
    circuit: QuantumCircuit
    parameters: np.ndarray
    expected_cost: float
    best_of_sample: float
    distribution_metrics: Optional[DistributionMetrics] = None
    optimization_history: List[float] = field(default_factory=list)


@dataclass
class CoDesignResult:
    """Complete co-design comparison results."""
    
    configurations: Dict[str, ConfigurationResult]
    best_configuration: str
    improvement_over_baseline: float
    baseline_configuration: str = 'nohot_original'
    
    def get_best_result(self) -> ConfigurationResult:
        """Get the best configuration result."""
        return self.configurations[self.best_configuration]


class VariationalCoDesigner:
    """
    Manages the co-design process for variational algorithms under HOT.
    """
    
    def __init__(self, algorithm_spec: AlgorithmSpec, backend: Any, 
                 calibration_data: CalibrationData, sampler: Optional[SamplerProvider] = None):
        self.algorithm = algorithm_spec
        self.backend = backend
        self.calibration = calibration_data
        self._default_shots = 8192
        self._sampler: SamplerProvider = sampler or LocalStatevectorSampler()
    
    def run_codesign(self, retuning_config: RetuningConfig) -> CoDesignResult:
        """
        Execute the full co-design workflow.
        
        Args:
            retuning_config: Configuration for parameter retuning
            
        Returns:
            CoDesignResult with all four configurations and comparison metrics
        """
        # 1. Baseline assessment
        baseline = self._run_baseline()
        
        # 2-3. Island selection and scaffolding (H, O)
        scaffold = compute_scaffold(
            self.algorithm.interaction_graph,
            self.calibration.topology,
            self.calibration
        )
        
        islands = discover_islands(
            self.backend,
            self.calibration,
            self.algorithm.min_qubits,
            self.algorithm.max_qubits
        )
        
        island = select_island(
            self.algorithm,
            islands,
            scaffold,
            selection_policy='minimize_error'
        )
        
        # Update scaffold with selected island
        scaffold.island = island
        
        # 4. Parameter retuning
        hot_retuned_params = self._retune_parameters(
            scaffold,
            island,
            retuning_config,
            init_from=baseline.parameters if retuning_config.init_strategy == 'transfer' else None
        )
        
        # Also retune on original layout as control
        nohot_retuned_params = self._retune_parameters(
            baseline.scaffold,
            baseline.island,
            retuning_config,
            init_from=baseline.parameters if retuning_config.init_strategy == 'transfer' else None
        )
        
        # 5. Validation - run all four configurations
        results = {
            'nohot_original': baseline,
            'nohot_retuned': self._evaluate(
                baseline.scaffold, baseline.island, nohot_retuned_params, 'nohot_retuned'
            ),
            'hot_original': self._evaluate(
                scaffold, island, baseline.parameters, 'hot_original'
            ),
            'hot_retuned': self._evaluate(
                scaffold, island, hot_retuned_params, 'hot_retuned'
            )
        }
        
        # 6. Find best configuration
        best_config = min(results.items(), key=lambda x: x[1].expected_cost)
        improvement = (
            baseline.expected_cost - results[best_config[0]].expected_cost
        ) / baseline.expected_cost
        
        return CoDesignResult(
            configurations=results,
            best_configuration=best_config[0],
            improvement_over_baseline=improvement
        )
    
    def _run_baseline(self) -> ConfigurationResult:
        """Run baseline assessment with default layout."""
        # Create baseline scaffold (identity mapping)
        n_qubits = self.algorithm.interaction_graph.number_of_nodes()
        baseline_mapping = {i: i for i in range(n_qubits)}
        
        # Create baseline island (first n qubits)
        baseline_qubits = list(range(n_qubits))
        baseline_edges = [(i, i+1) for i in range(n_qubits-1)]
        
        from .core import Island, IslandMetrics
        baseline_island = Island(
            qubits=baseline_qubits,
            edges=baseline_edges,
            score=1.0,  # Placeholder
            metrics=IslandMetrics(
                mean_readout_error=0.01,
                mean_two_qubit_error=0.02,
                mean_T1=100.0,
                mean_T2=80.0,
                calibration_stability=0.9,
                expected_swaps=0,
                coherence_score=0.8
            )
        )
        
        from .core import Scaffold
        baseline_scaffold = Scaffold(
            mapping=baseline_mapping,
            swap_schedule=[],
            estimated_error=0.1,
            topology_pattern='linear',
            island=baseline_island
        )
        
        # Evaluate with initial parameters
        initial_params = self.algorithm.initial_parameters
        if initial_params is None:
            initial_params = self.algorithm.random_initial_parameters()
        
        return self._evaluate(
            baseline_scaffold, baseline_island, initial_params, 'nohot_original'
        )
    
    def _retune_parameters(self, scaffold: Scaffold, island: Island,
                          config: RetuningConfig, init_from: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Optimize variational parameters for a specific layout.
        
        Args:
            scaffold: Scaffold for the layout
            island: Selected island
            config: Retuning configuration
            init_from: Initial parameters (if any)
            
        Returns:
            Optimized parameters
        """
        # Initialize parameters
        if config.init_strategy == 'random' or init_from is None:
            params = self.algorithm.random_initial_parameters()
        elif config.init_strategy == 'transfer':
            params = init_from.copy()
        elif config.init_strategy == 'perturb':
            params = init_from + np.random.normal(0, config.perturbation_scale, len(init_from))
        
        # Set up objective function
        if config.objective == 'expected_cost':
            objective = lambda p: self._expected_cost(p, scaffold, island)
        elif config.objective == 'best_of_sample':
            objective = lambda p: self._best_of_sample(p, scaffold, island)
        elif config.objective == 'cvar':
            objective = lambda p: self._cvar_cost(p, scaffold, island, config.cvar_alpha)
        
        # Optimize
        result = minimize(
            objective,
            params,
            method=config.optimizer,
            options={
                'maxiter': config.max_iterations,
                'ftol': config.convergence_threshold,
                'disp': True
            }
        )
        
        return result.x
    
    def _evaluate(self, scaffold: Scaffold, island: Island, 
                  parameters: np.ndarray, name: str) -> ConfigurationResult:
        """
        Evaluate a specific configuration.
        
        Args:
            scaffold: Scaffold for the layout
            island: Selected island
            parameters: Variational parameters
            name: Configuration name
            
        Returns:
            Configuration result
        """
        # Build circuit with parameters
        circuit = self._build_circuit(scaffold, parameters)
        
        # Run evaluation
        expected_cost = self._expected_cost(parameters, scaffold, island)
        best_of_sample = self._best_of_sample(parameters, scaffold, island)
        
        # Compute distribution metrics
        distribution_metrics = self._compute_distribution_metrics(
            parameters, scaffold, island
        )
        
        return ConfigurationResult(
            name=name,
            scaffold=scaffold,
            island=island,
            circuit=circuit,
            parameters=parameters,
            expected_cost=expected_cost,
            best_of_sample=best_of_sample,
            distribution_metrics=distribution_metrics
        )
    
    def _build_circuit(self, scaffold: Scaffold, parameters: np.ndarray) -> QuantumCircuit:
        """
        Build quantum circuit with given parameters and scaffold.
        
        Args:
            scaffold: Scaffold with layout information
            parameters: Variational parameters
            
        Returns:
            Parameterized quantum circuit
        """
        # Prefer a provided template circuit from the spec.
        if self.algorithm.circuit is not None:
            base = self.algorithm.circuit.copy()

            # Bind parameters if circuit is parameterized.
            try:
                if base.parameters:
                    ordered = list(base.parameters)
                    if len(ordered) == len(parameters):
                        base = base.assign_parameters(dict(zip(ordered, parameters)), inplace=False)
            except Exception:
                pass

            mapped = self._apply_scaffold_mapping(base, scaffold)
            return self._ensure_measurements(mapped)

        # Fallback placeholder builder
        n_qubits = len(scaffold.mapping)
        circuit = QuantumCircuit(n_qubits, n_qubits)

        param_idx = 0
        for layer in range(2):
            for i in range(n_qubits):
                if param_idx < len(parameters):
                    circuit.rz(parameters[param_idx], i)
                    param_idx += 1
            for i in range(n_qubits):
                if param_idx < len(parameters):
                    circuit.rx(parameters[param_idx], i)
                    param_idx += 1

        for i in range(n_qubits):
            circuit.measure(i, i)

        return circuit

    def _apply_scaffold_mapping(self, circuit: QuantumCircuit, scaffold: Scaffold) -> QuantumCircuit:
        """Apply scaffold logical->physical mapping by compacting physical qubits to a contiguous register."""
        mapping = scaffold.mapping
        logical_n = circuit.num_qubits
        qubit_to_index = {q: i for i, q in enumerate(circuit.qubits)}
        clbit_to_index = {c: i for i, c in enumerate(circuit.clbits)}

        physical_qubits = [mapping.get(i, i) for i in range(logical_n)]
        physical_to_compact = {p: i for i, p in enumerate(physical_qubits)}

        mapped = QuantumCircuit(logical_n, circuit.num_clbits)
        for inst in circuit.data:
            instr = getattr(inst, "operation", None)
            qargs = getattr(inst, "qubits", None)
            cargs = getattr(inst, "clbits", None)
            if instr is None:
                instr, qargs, cargs = inst
            new_qargs = []
            for q in qargs:
                old_index = qubit_to_index[q]
                physical_index = mapping.get(old_index, old_index)
                new_index = physical_to_compact.get(physical_index, old_index)
                new_qargs.append(mapped.qubits[new_index])
            new_cargs = [mapped.clbits[clbit_to_index[c]] for c in cargs]
            mapped.append(instr, new_qargs, new_cargs)

        return mapped

    def _ensure_measurements(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """Ensure the circuit has final measurements on all qubits if it has no measurements."""
        has_meas = False
        for inst in circuit.data:
            instr = getattr(inst, "operation", None)
            if instr is None:
                instr, _, _ = inst
            if instr.name == 'measure':
                has_meas = True
                break
        if has_meas:
            return circuit

        measured = circuit.copy()
        if measured.num_clbits == 0:
            measured.add_register(*QuantumCircuit(measured.num_qubits, measured.num_qubits).cregs)

        n = measured.num_qubits
        m = min(measured.num_clbits, n)
        measured.measure(list(range(m)), list(range(m)))
        return measured
    
    def _expected_cost(self, parameters: np.ndarray, scaffold: Scaffold, 
                       island: Island) -> float:
        """Compute expected cost for given parameters."""
        circuit = self._build_circuit(scaffold, parameters)

        counts = self._run_circuit_counts(circuit, shots=self._default_shots)
        
        # Compute expected cost
        if self.algorithm.cost_function is not None:
            total_shots = sum(counts.values())
            expected_cost = 0.0

            for bitstring, count in counts.items():
                cost = self.algorithm.cost_function(bitstring)
                expected_cost += count * cost

            return expected_cost / total_shots if total_shots > 0 else float('inf')
        else:
            # Default: minimize number of 1s
            expected_ones = 0.0
            total_shots = sum(counts.values())
            for bitstring, count in counts.items():
                ones = bitstring.count('1')
                expected_ones += count * ones
            return expected_ones / total_shots if total_shots > 0 else float('inf')
    
    def _best_of_sample(self, parameters: np.ndarray, scaffold: Scaffold,
                        island: Island) -> float:
        """Compute best-of-sample cost for given parameters."""
        circuit = self._build_circuit(scaffold, parameters)

        counts = self._run_circuit_counts(circuit, shots=self._default_shots)
        
        if self.algorithm.cost_function is not None:
            costs = [self.algorithm.cost_function(bitstring) for bitstring in counts.keys()]
            return min(costs) if costs else float('inf')
        else:
            # Default: minimize number of 1s
            min_ones = min(bitstring.count('1') for bitstring in counts.keys())
            return min_ones if counts else float('inf')
    
    def _cvar_cost(self, parameters: np.ndarray, scaffold: Scaffold,
                   island: Island, alpha: float) -> float:
        """Compute CVaR cost for given parameters."""
        circuit = self._build_circuit(scaffold, parameters)

        counts = self._run_circuit_counts(circuit, shots=self._default_shots)
        
        if self.algorithm.cost_function is not None:
            costs = [(bitstring, self.algorithm.cost_function(bitstring), count)
                     for bitstring, count in counts.items()]
        else:
            costs = [(bitstring, bitstring.count('1'), count)
                     for bitstring, count in counts.items()]
        
        # Sort by cost (best first for minimization)
        costs.sort(key=lambda x: x[1])
        
        # Accumulate until we have alpha fraction
        total_shots = sum(count for _, _, count in costs)
        threshold_count = int(alpha * total_shots)
        accumulated = 0
        cvar_sum = 0.0

        for _, cost, count in costs:
            take = min(count, threshold_count - accumulated)
            cvar_sum += cost * take
            accumulated += take
            if accumulated >= threshold_count:
                break

        return cvar_sum / accumulated if accumulated > 0 else costs[0][1]
    
    def _compute_distribution_metrics(self, parameters: np.ndarray, 
                                      scaffold: Scaffold, island: Island) -> DistributionMetrics:
        """Compute comprehensive distribution metrics."""
        circuit = self._build_circuit(scaffold, parameters)

        counts_dict = self._run_circuit_counts(circuit, shots=self._default_shots)
        
        if self.algorithm.cost_function is not None:
            cost_function = self.algorithm.cost_function
        else:
            cost_function = lambda bitstring: bitstring.count('1')
        
        return compute_distribution_metrics(counts_dict, cost_function)

    def _run_circuit_counts(self, circuit: QuantumCircuit, shots: int) -> Dict[str, int]:
        """Execute a circuit locally via Statevector sampling and return counts.

        This keeps the variational pipeline working across Qiskit versions without
        relying on `qiskit.primitives.Sampler`.
        """
        return self._sampler.sample_counts(circuit, shots)
