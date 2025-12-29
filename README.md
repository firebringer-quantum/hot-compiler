# HOT Framework (Qiskit Compiler)

HOT is a *routine compiler* for Qiskit `QuantumCircuit`s that applies three coordinated techniques:

- **H**: Hardware-aware mitigation scaffolding and hook-based circuit transforms
- **O**: Calibration-driven *island selection* (choose the best-connected/lowest-error subset of qubits)
- **T**: Measurement hygiene (remove extraneous measurements, optional staggering to reduce crosstalk)

This repository contains a Qiskit 2.x–compatible implementation with a small, Python-first API.

## Installation

### Option A: install from source (recommended for development)

```bash
python -m venv quantum_venv
# Windows
.\\quantum_venv\\Scripts\\activate

pip install -U pip
pip install -r requirements.txt
pip install -e .
```

### Option B: pip install (if you publish a wheel)

```bash
pip install hot-framework
```

## Quick start

### 1) Compile a circuit

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
    measurement_policy=MeasurementPolicy(stagger_readout=True),
)

compiled = result.compiled_circuit
print(result.algorithm_class.value)
print(result.metadata)
```

### 2) Run the example scripts

```bash
python examples/basic_usage.py
```

## API overview

### `HOTCompiler`

- **Constructor**: `HOTCompiler(backend)`
- **Main entry point**: `HOTCompiler.compile(circuit_or_spec, ...) -> HOTResult`
  - `circuit_or_spec` may be:
    - a Qiskit `QuantumCircuit` (fixed-unitary path), or
    - an `AlgorithmSpec` (variational co-design path)

### Core inputs

- **Algorithm class**
  - Pass `algorithm_class="fixed_unitary"`, `"variational"`, or `"auto"`.

- **Orphan policy** (`OrphanPolicy`)
  - `mode="strict"`: removes operations that touch qubits outside the selected island.
  - `mode="relaxed"`: allows orphans if they are far enough from the island.
  - `mode="diagnostic"`: optionally measures allowed orphans (best-effort; safe no-op if the circuit doesn’t contain those indices).

- **Measurement policy** (`MeasurementPolicy`)
  - Removes extraneous measurements, can forbid mid-circuit measurements, and can stagger final readout.

- **Mitigation hooks** (`MitigationHook`)
  - Optional list of transforms applied during compilation.
  - Built-ins:
    - `DynamicalDecoupling(...)`
    - `ZeroNoiseExtrapolation(...)` (may produce circuit variants; the pipeline keeps the first variant and records metadata)
    - `PauliTwirl(...)` (paired Paulis around 2Q gates)

### Outputs

Compilation returns `HOTResult`:

- `compiled_circuit`: the compiled/stabilized circuit ready for execution
- `job_variants`: `JobVariants(sanitized, instrumented, comparison_metadata)`
- `island`, `scaffold`: selected topology and mapping
- `algorithm_class`: `AlgorithmClass` enum
- `metadata`: JSON-friendly details (backend name, policies, mitigation variants, etc.)

## IBM Quantum / Runtime configuration (optional)

This repo includes `qiskit-ibm-runtime` as a dependency. To execute compiled circuits on IBM backends you typically need IBM Quantum credentials.

- **Do not hardcode keys.** Use environment variables or a local `.env` file.
- This project includes a `.env` file placeholder at repo root.

See `docs/USAGE.md` for more details.

## Documentation

- `docs/USAGE.md` — detailed usage and recipes
- `HOT_Framework_v2.md` — framework/spec reference
- `HOT_Fix_Plan.md` — historical consistency fix plan

## Development

- Run tests:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

## License

MIT (see repository license file if present).
