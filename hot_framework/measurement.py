"""
T - Tidy measurement hygiene for HOT Framework.
"""

from typing import Dict, List, Any, Optional, Set, Union, Tuple
from dataclasses import dataclass
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Instruction

from .core import (
    MeasurementClass, MeasurementPolicy, Island, Scaffold, AlgorithmSpec,
    JobVariants, MeasurementPolicyViolation
)


MeasurementKey = Tuple[int, int, int]


def _inst_parts(inst):
    op = getattr(inst, "operation", None)
    qargs = getattr(inst, "qubits", None)
    cargs = getattr(inst, "clbits", None)
    if op is None:
        op, qargs, cargs = inst
    return op, qargs, cargs


def classify_measurements(circuit: QuantumCircuit, algorithm_spec: AlgorithmSpec) -> Dict[MeasurementKey, MeasurementClass]:
    """
    Classify each measurement in the circuit.
    
    Args:
        circuit: Quantum circuit to analyze
        algorithm_spec: Algorithm specification
        
    Returns:
        Dictionary mapping measurement operations to classifications
    """
    classifications: Dict[MeasurementKey, MeasurementClass] = {}
    qubit_to_index = {q: i for i, q in enumerate(circuit.qubits)}
    clbit_to_index = {c: i for i, c in enumerate(circuit.clbits)}
    occurrence: Dict[Tuple[int, int], int] = {}
    
    for data_index, inst in enumerate(circuit.data):
        instruction, qargs, cargs = _inst_parts(inst)
        if not is_measurement(instruction):
            continue
        
        q_i = qubit_to_index[qargs[0]]
        c_i = clbit_to_index[cargs[0]] if cargs else 0
        pair = (q_i, c_i)
        occ = occurrence.get(pair, 0)
        occurrence[pair] = occ + 1
        key: MeasurementKey = (q_i, c_i, occ)
        
        # Check if measurement feeds into classical control
        if has_classical_dependency(circuit, data_index):
            classifications[key] = MeasurementClass.ESSENTIAL
        
        # Check if this is a final readout on an algorithm qubit
        elif (algorithm_spec.output_qubits and 
              q_i in algorithm_spec.output_qubits and 
              is_final_measurement(circuit, data_index)):
            classifications[key] = MeasurementClass.ESSENTIAL
        
        # Check if explicitly marked diagnostic
        elif hasattr(instruction, 'metadata') and instruction.metadata.get('diagnostic', False):
            classifications[key] = MeasurementClass.DIAGNOSTIC
        
        else:
            classifications[key] = MeasurementClass.EXTRANEOUS
    
    return classifications


def _measurement_key_map(circuit: QuantumCircuit) -> Dict[int, MeasurementKey]:
    """Map circuit.data index -> stable measurement key."""
    qubit_to_index = {q: i for i, q in enumerate(circuit.qubits)}
    clbit_to_index = {c: i for i, c in enumerate(circuit.clbits)}
    occurrence: Dict[Tuple[int, int], int] = {}
    index_to_key: Dict[int, MeasurementKey] = {}

    for data_index, inst in enumerate(circuit.data):
        instr, qargs, cargs = _inst_parts(inst)
        if not is_measurement(instr):
            continue
        q_i = qubit_to_index[qargs[0]]
        c_i = clbit_to_index[cargs[0]] if cargs else 0
        pair = (q_i, c_i)
        occ = occurrence.get(pair, 0)
        occurrence[pair] = occ + 1
        index_to_key[data_index] = (q_i, c_i, occ)

    return index_to_key


def _classification_for(classifications: Dict[MeasurementKey, MeasurementClass],
                        measurement_key: MeasurementKey) -> Optional[MeasurementClass]:
    return classifications.get(measurement_key)


def is_measurement(instruction: Instruction) -> bool:
    """Check if instruction is a measurement."""
    return instruction.name == 'measure'


def has_classical_dependency(circuit: QuantumCircuit, measurement_index: int) -> bool:
    """
    Check if measurement result is used for classical control.
    
    Args:
        circuit: Quantum circuit
        measurement: Measurement instruction to check
        
    Returns:
        True if measurement feeds into classical control
    """
    # Simplified implementation - in practice would analyze classical register usage
    # For now, assume mid-circuit measurements are for control
    return not is_final_measurement(circuit, measurement_index)


def is_final_measurement(circuit: QuantumCircuit, measurement_index: int) -> bool:
    """
    Check if measurement is in the final layer of the circuit.
    
    Args:
        circuit: Quantum circuit
        measurement: Measurement instruction to check
        
    Returns:
        True if measurement is in final layer
    """
    # Check if there are any non-measurement operations after this
    for i in range(measurement_index + 1, len(circuit.data)):
        instr, _, _ = _inst_parts(circuit.data[i])
        if not is_measurement(instr):
            return False
    
    return True


def apply_measurement_policy(circuit: QuantumCircuit, classifications: Dict[MeasurementKey, MeasurementClass],
                            policy: MeasurementPolicy, island: Island) -> QuantumCircuit:
    """
    Enforce measurement hygiene according to policy.
    
    Args:
        circuit: Quantum circuit
        classifications: Measurement classifications
        policy: Measurement policy to apply
        island: Selected island
        
    Returns:
        Modified circuit
        
    Raises:
        MeasurementPolicyViolation: If policy cannot be satisfied
    """
    modified_circuit = circuit.copy()
    index_to_key = _measurement_key_map(modified_circuit)
    
    # Remove extraneous measurements
    extraneous_indices = [
        i for i, inst in enumerate(modified_circuit.data)
        if is_measurement(_inst_parts(inst)[0])
        and _classification_for(classifications, index_to_key.get(i, (-1, -1, -1))) == MeasurementClass.EXTRANEOUS
    ]
    if extraneous_indices:
        modified_circuit = remove_measurements(modified_circuit, extraneous_indices)
        index_to_key = _measurement_key_map(modified_circuit)
    
    # Handle mid-circuit measurements
    if not policy.allow_mid_circuit:
        mid_circuit_indices = [
            i for i, inst in enumerate(modified_circuit.data)
            if is_measurement(_inst_parts(inst)[0])
            and not is_final_measurement(modified_circuit, i)
        ]
        
        if mid_circuit_indices:
            # Check if any are essential
            essential_mid = [
                i for i in mid_circuit_indices
                if _classification_for(classifications, index_to_key.get(i, (-1, -1, -1))) == MeasurementClass.ESSENTIAL
            ]
            
            if essential_mid and not policy.allow_mid_circuit:
                raise MeasurementPolicyViolation(
                    "Essential mid-circuit measurements present but policy forbids them"
                )
            
            # Remove non-essential mid-circuit measurements
            non_essential_mid = [
                i for i in mid_circuit_indices
                if _classification_for(classifications, index_to_key.get(i, (-1, -1, -1))) != MeasurementClass.ESSENTIAL
            ]
            if non_essential_mid:
                modified_circuit = remove_measurements(modified_circuit, non_essential_mid)
                index_to_key = _measurement_key_map(modified_circuit)
    
    # Stagger readout to reduce crosstalk
    if policy.stagger_readout:
        modified_circuit = stagger_final_measurements(modified_circuit, island)
    
    return modified_circuit


def remove_measurements(circuit: QuantumCircuit, measurements: Union[List[Instruction], List[int]]) -> QuantumCircuit:
    """
    Remove specified measurements from circuit.
    
    Args:
        circuit: Quantum circuit
        measurements: List of measurement instructions to remove
        
    Returns:
        Circuit with measurements removed
    """
    modified = circuit.copy()
    
    # Backward-compatible wrapper: accept either Instruction objects or integer indices.
    if measurements and isinstance(measurements[0], int):
        return remove_measurements_by_index(modified, measurements)  # type: ignore[arg-type]

    # Filter out by instruction identity (legacy behavior)
    new_data = []
    measurements_set = set(id(m) for m in measurements)
    for inst in modified.data:
        instr, qargs, cargs = _inst_parts(inst)
        if is_measurement(instr) and id(instr) in measurements_set:
            continue
        new_data.append((instr, qargs, cargs))
    modified.data = new_data
    return modified


def remove_measurements_by_index(circuit: QuantumCircuit, measurement_indices: List[int]) -> QuantumCircuit:
    """Remove measurements by their circuit.data indices."""
    modified = circuit.copy()
    indices_set = set(measurement_indices)
    new_data = []
    for i, inst in enumerate(modified.data):
        instr, qargs, cargs = _inst_parts(inst)
        if i in indices_set and is_measurement(instr):
            continue
        new_data.append((instr, qargs, cargs))
    modified.data = new_data
    return modified


def stagger_final_measurements(circuit: QuantumCircuit, island: Island) -> QuantumCircuit:
    """
    Stagger final measurements to reduce crosstalk.
    
    Args:
        circuit: Quantum circuit
        island: Selected island with qubit layout
        
    Returns:
        Circuit with staggered measurements
    """
    # Group measurements by adjacency to avoid simultaneous measurement of neighbors
    staggered = circuit.copy()
    
    # Find final measurements
    final_measurements = []
    final_measurement_indices = set()
    for i, inst in enumerate(staggered.data):
        instr, qargs, cargs = _inst_parts(inst)
        if is_measurement(instr) and is_final_measurement(staggered, i):
            final_measurements.append((instr, qargs, cargs))
            final_measurement_indices.add(i)
    
    if len(final_measurements) <= 1:
        return staggered  # No need to stagger
    
    # Create measurement groups to avoid adjacent qubits
    island_topology = island.metrics  # Use island topology information
    measurement_groups = create_measurement_groups(final_measurements, island)

    # Rebuild circuit with staggered measurements using Qiskit APIs (no raw tuple injection).
    rebuilt = QuantumCircuit(*staggered.qregs, *staggered.cregs)

    # Add non-measurement operations (and non-final measurements) first
    for i, inst in enumerate(staggered.data):
        instr, qargs, cargs = _inst_parts(inst)
        if i in final_measurement_indices:
            continue
        rebuilt.append(instr, qargs, cargs)

    # Add staggered final measurements group by group, inserting barriers between groups
    for group in measurement_groups:
        for instr, qargs, cargs in group:
            rebuilt.append(instr, qargs, cargs)

        if group != measurement_groups[-1]:
            rebuilt.barrier(*rebuilt.qubits)

    return rebuilt


def create_measurement_groups(measurements: List[tuple], island: Island) -> List[List[tuple]]:
    """
    Create groups of measurements that can be performed simultaneously.
    
    Args:
        measurements: List of (instruction, qargs, cargs) tuples
        island: Island with qubit layout information
        
    Returns:
        List of measurement groups
    """
    groups = []
    used_qubits = set()
    island_edges = set(tuple(sorted(edge)) for edge in island.edges)
    
    # Simple greedy algorithm
    remaining_measurements = measurements.copy()
    
    while remaining_measurements:
        current_group = []
        current_qubits = set()
        
        # Add measurements to current group if they don't conflict
        i = 0
        while i < len(remaining_measurements):
            instr, qargs, cargs = remaining_measurements[i]
            qubit = qargs[0]
            
            # Check if qubit conflicts with current group
            conflict = False
            for group_qubit in current_qubits:
                if (qubit, group_qubit) in island_edges or (group_qubit, qubit) in island_edges:
                    conflict = True
                    break
            
            if not conflict and qubit not in used_qubits:
                current_group.append(remaining_measurements.pop(i))
                current_qubits.add(qubit)
                used_qubits.add(qubit)
            else:
                i += 1
        
        if current_group:
            groups.append(current_group)
        else:
            # Safety check - if we can't add any measurements, take one anyway
            groups.append([remaining_measurements.pop(0)])
            used_qubits.add(groups[-1][0][1][0])
    
    return groups


def wrap_mid_circuit_measurement(circuit: QuantumCircuit, measurement_op: Instruction,
                                calibration_data: Any) -> QuantumCircuit:
    """
    Apply protective measures around mid-circuit measurement.
    
    Args:
        circuit: Quantum circuit
        measurement_op: Measurement operation to wrap
        calibration_data: Device calibration information
        
    Returns:
        Circuit with protective measures applied
    """
    wrapped = circuit.copy()
    
    # This is a simplified implementation
    # In practice would:
    # 1. Apply dynamical decoupling on idle neighbors
    # 2. Add readout error mitigation tags
    # 3. Ensure adequate delay after reset
    
    return wrapped


def generate_job_variants(circuit: QuantumCircuit, classifications: Dict[MeasurementKey, MeasurementClass],
                          policy: MeasurementPolicy, scaffold: Scaffold) -> JobVariants:
    """
    Generate instrumented and sanitized job variants.
    
    Args:
        circuit: Compiled quantum circuit
        classifications: Measurement classifications
        policy: Measurement policy
        scaffold: Scaffold information
        
    Returns:
        Job variants for different purposes
    """
    # Sanitized: essential measurements only
    sanitized_policy = MeasurementPolicy(
        allow_mid_circuit=policy.allow_mid_circuit,
        max_concurrent_adjacent=policy.max_concurrent_adjacent,
        stagger_readout=policy.stagger_readout,
        readout_error_mitigation=policy.readout_error_mitigation
    )
    
    sanitized = apply_measurement_policy(
        circuit.copy(),
        classifications,
        sanitized_policy,
        scaffold.island
    )
    
    # Instrumented: essential + diagnostic
    instrumented_policy = MeasurementPolicy(
        allow_mid_circuit=True,  # Allow all measurements for diagnostics
        max_concurrent_adjacent=policy.max_concurrent_adjacent,
        stagger_readout=False,  # Don't stagger for full diagnostics
        readout_error_mitigation=True
    )
    
    # For instrumented, keep all measurements
    instrumented_classifications: Dict[MeasurementKey, MeasurementClass] = {}
    for key in _measurement_key_map(circuit).values():
        instrumented_classifications[key] = MeasurementClass.ESSENTIAL
    
    instrumented = apply_measurement_policy(
        circuit.copy(),
        instrumented_classifications,
        instrumented_policy,
        scaffold.island
    )
    
    # Add process tomography circuits if requested
    # This would be implemented in a full version
    
    return JobVariants(
        sanitized=sanitized,
        instrumented=instrumented,
        comparison_metadata={
            'sanitized_measurements': count_measurements(sanitized),
            'instrumented_measurements': count_measurements(instrumented),
            'essential_measurements': len([
                cls for cls in classifications.values()
                if cls == MeasurementClass.ESSENTIAL
            ]),
            'diagnostic_measurements': len([
                cls for cls in classifications.values()
                if cls == MeasurementClass.DIAGNOSTIC
            ]),
            'extraneous_measurements': len([
                cls for cls in classifications.values()
                if cls == MeasurementClass.EXTRANEOUS
            ])
        }
    )


def count_measurements(circuit: QuantumCircuit) -> int:
    """Count the number of measurement operations in a circuit."""
    total = 0
    for inst in circuit.data:
        instr, _, _ = _inst_parts(inst)
        if is_measurement(instr):
            total += 1
    return total


def add_process_tomography_circuits(circuit: QuantumCircuit, scaffold: Scaffold,
                                   target_layers: List[str]) -> QuantumCircuit:
    """
    Add process tomography circuits for characterization.
    
    Args:
        circuit: Base circuit
        scaffold: Scaffold information
        target_layers: Types of layers to characterize
        
    Returns:
        Circuit with tomography additions
    """
    # Simplified implementation - would add full tomography circuits
    return circuit
