# HOT Compiler Usage Guide

This guide shows how to use the HOT compiler (`hot_framework.HOTCompiler`) to compile Qiskit circuits for improved execution robustness.

## 1) Installation

From the repository root:

```bash
python -m venv quantum_venv
# Windows
.\\quantum_venv\\Scripts\\activate

pip install -U pip
pip install -r requirements.txt
pip install -e .
```

## 2) Core concepts

HOT applies three coordinated techniques:

- **H (Hardware-aware mitigation)**
  - Computes a scaffold mapping logical→physical qubits.
  - Optional mitigation hooks can modify the circuit.

- **O (Optimal island selection)**
  - Selects a connected subset (“island”) of qubits with better calibration metrics.

- **T (Tidy measurement hygiene)**
  - Classifies measurements and removes extraneous ones.
  - Can stagger final readout via barriers to reduce adjacent readout crosstalk.

## 3) The main API: `HOTCompiler.compile`

### Compile a fixed-unitary circuit

```python
from qiskit import QuantumCircuit
from qiskit.providers.fake_provider import GenericBackendV2

from hot_framework import HOTCompiler, OrphanPolicy, MeasurementPolicy

backend = GenericBackendV2(
    num_qubits=5,
    coupling_map=[[0, 1], [1, 2], [2, 3], [3, 4]],
)

qc = QuantumCircuit(2, 2)
qc.h(0)
qc.cx(0, 1)
qc.measure([0, 1], [0, 1])

compiler = HOTCompiler(backend)
result = compiler.compile(
    qc,
    algorithm_class="fixed_unitary",
    orphan_policy=OrphanPolicy(mode="strict"),
    measurement_policy=MeasurementPolicy(
        allow_mid_circuit=False,
        stagger_readout=True,
        max_concurrent_adjacent=1,
    ),
)

print(result.summary())
compiled = result.compiled_circuit
```

### Compile a variational algorithm (co-design)

Variational compilation takes an `AlgorithmSpec` and a `RetuningConfig`.

```python
import numpy as np
import networkx as nx

from qiskit.providers.fake_provider import GenericBackendV2

from hot_framework import (
    HOTCompiler,
    AlgorithmSpec,
    AlgorithmClass,
    RetuningConfig,
    OrphanPolicy,
)

interaction_graph = nx.Graph()
interaction_graph.add_edges_from([(0, 1), (1, 2)])

def cost_fn(bitstring: str) -> float:
    return float(bitstring.count("1"))

spec = AlgorithmSpec(
    name="toy_variational",
    algorithm_class=AlgorithmClass.VARIATIONAL,
    min_qubits=3,
    max_qubits=3,
    interaction_graph=interaction_graph,
    cost_function=cost_fn,
    initial_parameters=np.array([0.1, 0.2, 0.3, 0.4]),
    parameter_bounds=[(0.0, 6.283185307)] * 4,
    output_qubits=[0, 1, 2],
)

backend = GenericBackendV2(
    num_qubits=5,
    coupling_map=[[0, 1], [1, 2], [2, 3], [3, 4]],
)

compiler = HOTCompiler(backend)
result = compiler.compile(
    spec,
    algorithm_class="variational",
    retuning_config=RetuningConfig(
        optimizer="COBYLA",
        max_iterations=10,
        objective="expected_cost",
        init_strategy="transfer",
    ),
    orphan_policy=OrphanPolicy(mode="strict"),
)

if result.codesign_result:
    print(result.codesign_result.best_configuration)
```

## 4) Policies

### Orphan policy

- `strict`: drops circuit instructions that touch qubits outside the island
- `relaxed`: permits nearby qubits only if they meet a distance threshold
- `diagnostic`: optionally measures specified orphans (best-effort)

### Measurement policy

Controls:

- mid-circuit measurement allowance
- max adjacent concurrent measurements
- optional final readout staggering

## 5) Mitigation hooks

Mitigation hooks are optional transformations applied during compilation.

```python
from hot_framework import DynamicalDecoupling

result = compiler.compile(
    qc,
    mitigation_hooks=[DynamicalDecoupling(sequence="XY4", threshold_idle_time=100)],
)
```

Note: Some hooks may generate multiple circuit variants (e.g. ZNE). The current pipeline compiles the first and records variant metadata.

## 6) IBM Quantum / Runtime notes

This package includes `qiskit-ibm-runtime`, but the compiler itself does not require IBM Runtime to run locally.

- Store credentials in environment variables or `.env`.
- Avoid hardcoding API keys.

If you want a dedicated Runtime how-to, see `docs/IBM_RUNTIME.md`.
