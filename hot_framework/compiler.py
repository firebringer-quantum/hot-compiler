"""
Main HOT Compiler class and API.
"""

from typing import Dict, List, Any, Optional, Union, Literal
import numpy as np
import networkx as nx
from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap

from .core import (
    HOTResult, AlgorithmSpec, AlgorithmClass, OrphanPolicy, MeasurementPolicy,
    CalibrationData, HOTError, Scaffold, Island
)
from .mitigation import MitigationHook, compute_scaffold
from .island_selection import discover_islands, select_island, apply_orphan_policy
from .measurement import classify_measurements, apply_measurement_policy, generate_job_variants
from .variational import VariationalCoDesigner, RetuningConfig
from .utils import (
    create_interaction_graph_from_circuit, validate_algorithm_spec,
    get_calibration_data
)


class HOTCompiler:
    """
    Main entry point for HOT framework.
    
    Provides systematic methodology for quantum circuit optimization that operates
    above standard SDK compilation through three interconnected concerns:
    - H: Hardware-aware error mitigation through topological scaffolding
    - O: Optimal island selection via calibration-driven qubit clustering  
    - T: Tidy measurement hygiene to minimize crosstalk and extraneous readout
    """
    
    def __init__(self, backend, calibration_source: Union[str, CalibrationData] = 'live'):
        """
        Initialize HOT compiler.
        
        Args:
            backend: Quantum backend (IBM Quantum backend or simulator)
            calibration_source: 'live' for current calibration, 'cached' for recent,
                              or CalibrationData object
        """
        self.backend = backend
        
        if isinstance(calibration_source, CalibrationData):
            self.calibration = calibration_source
        else:
            self.calibration = self._load_calibration(calibration_source)
    
    def _load_calibration(self, source: str) -> CalibrationData:
        """Load calibration data from specified source."""
        if source == 'live':
            return get_calibration_data(self.backend)
        elif source == 'cached':
            # In practice, would load from cache
            return get_calibration_data(self.backend)
        else:
            raise ValueError(f"Unknown calibration source: {source}")
    
    def compile(
        self,
        circuit_or_spec: Union[QuantumCircuit, AlgorithmSpec],
        algorithm_class: Literal['fixed_unitary', 'variational', 'auto'] = 'auto',
        orphan_policy: OrphanPolicy = OrphanPolicy(mode='strict'),
        measurement_policy: MeasurementPolicy = MeasurementPolicy(),
        mitigation_hooks: Optional[List[MitigationHook]] = None,
        retuning_config: Optional[RetuningConfig] = None,
        **kwargs
    ) -> HOTResult:
        """
        Compile a circuit through the full HOT pipeline.
        
        Args:
            circuit_or_spec: Qiskit circuit or algorithm specification
            algorithm_class: 'fixed_unitary', 'variational', or 'auto' to detect
            orphan_policy: How to handle qubits outside the selected island
            measurement_policy: Measurement hygiene configuration
            mitigation_hooks: Optional error mitigation strategies
            retuning_config: Required if variational
            **kwargs: Additional compilation options
            
        Returns:
            HOTResult containing compiled circuit(s) and metadata
            
        Raises:
            ValueError: If required parameters are missing
            HOTError: If compilation fails
        """
        # Convert to algorithm spec if raw circuit provided
        spec = self._to_algorithm_spec(circuit_or_spec, algorithm_class)
        
        # Validate specification
        validation_warnings = validate_algorithm_spec(spec)
        if validation_warnings:
            print("Validation warnings:")
            for warning in validation_warnings:
                print(f"  - {warning}")
        
        # Auto-detect algorithm class if needed
        if algorithm_class == 'auto':
            algorithm_class = self._detect_algorithm_class(spec)
            spec.algorithm_class = AlgorithmClass(algorithm_class)
        else:
            spec.algorithm_class = AlgorithmClass(algorithm_class)
        
        # Validate inputs
        if spec.algorithm_class == AlgorithmClass.VARIATIONAL and retuning_config is None:
            raise ValueError(
                "Variational algorithms require retuning_config. "
                "HOT layout changes invalidate pre-optimized parameters."
            )
        
        # ===== O: Island Selection =====
        islands = discover_islands(
            self.backend,
            self.calibration,
            spec.min_qubits,
            spec.max_qubits
        )
        
        # ===== H: Scaffolding =====
        scaffold = compute_scaffold(
            spec.interaction_graph,
            self.calibration.topology,
            self.calibration
        )
        
        island = select_island(
            spec,
            islands,
            scaffold,
            selection_policy=kwargs.get('selection_policy', 'minimize_error')
        )
        
        # Update scaffold with selected island
        scaffold.island = island
        
        # ===== Algorithm-class specific handling =====
        if spec.algorithm_class == AlgorithmClass.FIXED_UNITARY:
            # Direct compilation - no parameter retuning needed
            compiled_circuit = self._compile_fixed_unitary(
                spec,
                scaffold,
                island,
                orphan_policy,
                mitigation_hooks
            )
            codesign_result = None
            
        else:  # VARIATIONAL
            # Co-design workflow
            codesigner = VariationalCoDesigner(spec, self.backend, self.calibration)
            codesign_result = codesigner.run_codesign(retuning_config)
            
            compiled_circuit = codesign_result.get_best_result().circuit
        
        # ===== T: Measurement Hygiene =====
        classifications = classify_measurements(compiled_circuit, spec)
        compiled_circuit = apply_measurement_policy(
            compiled_circuit,
            classifications,
            measurement_policy,
            island
        )
        
        # Apply orphan policy
        compiled_circuit = apply_orphan_policy(
            compiled_circuit,
            island,
            orphan_policy,
            self.calibration.topology
        )
        
        # Apply mitigation hooks
        zne_variants_count = None
        mitigation_variants = None
        if mitigation_hooks:
            for hook in mitigation_hooks:
                hook_result = hook.apply(compiled_circuit, scaffold, self.calibration)
                if isinstance(hook_result, list):
                    # Some hooks (e.g., ZNE) may return multiple circuit variants.
                    # Keep the pipeline single-circuit by selecting the first variant,
                    # but record how many variants were produced.
                    zne_variants_count = len(hook_result)
                    mitigation_variants = [
                        {
                            'index': i,
                            'num_qubits': c.num_qubits,
                            'depth': c.depth(),
                            'size': len(c.data),
                        }
                        for i, c in enumerate(hook_result)
                    ]
                    compiled_circuit = hook_result[0] if hook_result else compiled_circuit
                else:
                    compiled_circuit = hook_result
        
        # Generate job variants
        job_variants = generate_job_variants(
            compiled_circuit,
            classifications,
            measurement_policy,
            scaffold
        )
        
        # ===== Build result =====
        backend_name = self._backend_name(self.backend)
        measurement_policy_dict = {
            'allow_mid_circuit': measurement_policy.allow_mid_circuit,
            'max_concurrent_adjacent': measurement_policy.max_concurrent_adjacent,
            'stagger_readout': measurement_policy.stagger_readout,
            'readout_error_mitigation': measurement_policy.readout_error_mitigation,
        }
        return HOTResult(
            compiled_circuit=compiled_circuit,
            job_variants=job_variants,
            island=island,
            scaffold=scaffold,
            algorithm_class=spec.algorithm_class,
            codesign_result=codesign_result,
            metadata={
                'backend': backend_name,
                'algorithm_class': spec.algorithm_class.value,
                'calibration_timestamp': self.calibration.timestamp,
                'estimated_error': scaffold.estimated_error,
                'orphan_policy': orphan_policy.mode,
                'measurement_policy': measurement_policy_dict,
                'mitigation_hooks': [type(h).__name__ for h in mitigation_hooks] if mitigation_hooks else [],
                'zne_variants_count': zne_variants_count,
                'mitigation_variants': mitigation_variants,
                'validation_warnings': validation_warnings
            }
        )

    @staticmethod
    def _backend_name(backend: Any) -> str:
        """Return backend name regardless of whether it's a method or attribute."""
        from .utils import _backend_name as utils_backend_name
        return utils_backend_name(backend)
    
    def _to_algorithm_spec(self, circuit_or_spec: Union[QuantumCircuit, AlgorithmSpec],
                          algorithm_class: str) -> AlgorithmSpec:
        """Convert circuit to algorithm specification if needed."""
        if isinstance(circuit_or_spec, AlgorithmSpec):
            return circuit_or_spec
        
        # Convert circuit to spec
        circuit = circuit_or_spec
        
        # Create interaction graph
        interaction_graph = create_interaction_graph_from_circuit(circuit)
        
        # Determine qubit requirements
        n_qubits = circuit.num_qubits
        
        # Auto-detect algorithm class if not specified
        if algorithm_class == 'auto':
            algorithm_class = self._detect_algorithm_class_from_circuit(circuit)
        
        # Create spec
        spec = AlgorithmSpec(
            name="user_circuit",
            algorithm_class=AlgorithmClass(algorithm_class),
            min_qubits=n_qubits,
            max_qubits=n_qubits,
            interaction_graph=interaction_graph,
            circuit=circuit,
            output_qubits=list(range(n_qubits))  # Assume all qubits are output
        )
        
        return spec
    
    def _detect_algorithm_class(self, spec: AlgorithmSpec) -> str:
        """Auto-detect algorithm class from specification."""
        if spec.algorithm_class != AlgorithmClass.VARIATIONAL and spec.initial_parameters is not None:
            return 'variational'
        elif spec.cost_function is not None:
            return 'variational'
        else:
            return 'fixed_unitary'
    
    def _detect_algorithm_class_from_circuit(self, circuit: QuantumCircuit) -> str:
        """Auto-detect algorithm class from circuit structure."""
        def _inst_parts(inst):
            op = getattr(inst, "operation", None)
            qargs = getattr(inst, "qubits", None)
            cargs = getattr(inst, "clbits", None)
            if op is None:
                op, qargs, cargs = inst
            return op, qargs, cargs

        # Look for parameterized gates
        has_parameters = any(
            hasattr(instruction, 'params') and instruction.params
            for instruction, _, _ in (_inst_parts(inst) for inst in circuit.data)
        )
        
        # Look for classical feedback loops (simplified)
        has_classical_feedback = any(
            instruction.name in ['if_test', 'while_loop']
            for instruction, _, _ in (_inst_parts(inst) for inst in circuit.data)
        )
        
        if has_parameters or has_classical_feedback:
            return 'variational'
        else:
            return 'fixed_unitary'
    
    def _compile_fixed_unitary(self, spec: AlgorithmSpec, scaffold: 'Scaffold',
                              island: 'Island', orphan_policy: OrphanPolicy,
                              mitigation_hooks: Optional[List[MitigationHook]]) -> QuantumCircuit:
        """Compile fixed-unitary algorithm."""
        if spec.circuit is None:
            raise ValueError("Fixed-unitary algorithm requires a circuit")
        
        # Apply layout transformation
        circuit = spec.circuit.copy()
        
        # Map logical qubits to physical qubits according to scaffold
        mapped_circuit = self._apply_layout(circuit, scaffold.mapping)
        
        return mapped_circuit
    
    def _apply_layout(self, circuit: QuantumCircuit, mapping: Dict[int, int]) -> QuantumCircuit:
        """Apply qubit layout mapping to circuit."""
        # Simplified layout application
        # In practice, would use Qiskit's transpiler with layout constraint
        
        qubit_to_index = {q: i for i, q in enumerate(circuit.qubits)}
        clbit_to_index = {c: i for i, c in enumerate(circuit.clbits)}

        # Keep logical circuit width stable; compact selected physical qubits.
        logical_n = circuit.num_qubits
        physical_qubits = [mapping.get(i, i) for i in range(logical_n)]
        physical_to_compact = {p: i for i, p in enumerate(physical_qubits)}

        mapped = QuantumCircuit(logical_n, circuit.num_clbits)
        
        # Create mapping from old to new qubit indices
        for inst in circuit.data:
            instruction = getattr(inst, "operation", None)
            qargs = getattr(inst, "qubits", None)
            cargs = getattr(inst, "clbits", None)
            if instruction is None:
                instruction, qargs, cargs = inst
            new_qargs = []
            for q in qargs:
                old_index = qubit_to_index[q]
                physical_index = mapping.get(old_index, old_index)
                new_index = physical_to_compact.get(physical_index, old_index)
                new_qargs.append(mapped.qubits[new_index])
            new_cargs = [mapped.clbits[clbit_to_index[c]] for c in cargs]
            
            # Add instruction to mapped circuit
            mapped.append(instruction, new_qargs, new_cargs)
        
        return mapped
    
    def analyze_backend(self) -> Dict[str, Any]:
        """Analyze backend characteristics for HOT optimization."""
        from .topology import analyze_connectivity_pattern
        
        topology_analysis = analyze_connectivity_pattern(self.calibration.topology)
        
        # Calibration quality metrics
        readout_errors = list(self.calibration.readout_errors.values())
        two_q_errors = list(self.calibration.two_qubit_errors.values())
        
        return {
            'backend_name': self._backend_name(self.backend),
            'n_qubits': self.calibration.topology.number_of_nodes(),
            'connectivity': topology_analysis,
            'calibration_quality': {
                'mean_readout_error': sum(readout_errors) / len(readout_errors),
                'mean_two_q_error': sum(two_q_errors) / len(two_q_errors),
                'readout_error_std': np.std(readout_errors),
                'two_q_error_std': np.std(two_q_errors)
            },
            'calibration_timestamp': self.calibration.timestamp
        }
    
    def suggest_optimizations(self, circuit: QuantumCircuit) -> List[str]:
        """Suggest specific optimizations for a given circuit."""
        suggestions = []

        def _inst_parts(inst):
            op = getattr(inst, "operation", None)
            qargs = getattr(inst, "qubits", None)
            cargs = getattr(inst, "clbits", None)
            if op is None:
                op, qargs, cargs = inst
            return op, qargs, cargs
        
        # Analyze circuit structure
        interaction_graph = create_interaction_graph_from_circuit(circuit)
        
        # Check connectivity requirements
        required_edges = interaction_graph.number_of_edges()
        available_edges = self.calibration.topology.number_of_edges()
        
        if required_edges > available_edges:
            suggestions.append(
                f"Circuit requires {required_edges} connections but backend only has {available_edges}. "
                "Consider SWAP optimization or different backend."
            )
        
        # Check measurement patterns
        measurements = []
        for inst in circuit.data:
            instr, qargs, cargs = _inst_parts(inst)
            if instr.name == 'measure':
                measurements.append((instr, qargs, cargs))
        
        if len(measurements) > circuit.num_qubits * 0.8:
            suggestions.append(
                "High measurement density detected. Consider measurement hygiene optimization."
            )
        
        # Check for two-qubit gate density
        two_q_gates = []
        for inst in circuit.data:
            instr, qargs, _ = _inst_parts(inst)
            if len(qargs) == 2:
                two_q_gates.append(instr)
        
        if len(two_q_gates) > len(circuit.data) * 0.5:
            suggestions.append(
                "High two-qubit gate density. Consider dynamical decoupling and error mitigation."
            )
        
        # Algorithm-specific suggestions
        algorithm_class = self._detect_algorithm_class_from_circuit(circuit)
        if algorithm_class == 'variational':
            suggestions.append(
                "Variational algorithm detected. HOT co-design with parameter retuning recommended."
            )
        
        return suggestions
