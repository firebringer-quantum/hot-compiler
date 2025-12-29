# HOT Framework Compiler – Consistency Fix Plan (Post-Scan)

This document consolidates **all consistency fixes** identified during the repo-wide scan of the HOT Framework implementation.

Scope:
- **No code changes in this document**; this is the authoritative checklist.
- Focus is on **consistency, correctness, and minimizing breakage**.
- Preserve functionality: avoid removing features; prefer deprecation/aliases.

---

## 0) Target Consistency Principles (Authoritative)

### 0.1 Canonical Types
- **Algorithm class** is represented internally as `AlgorithmClass` (enum).
- External/serialized representations may use `.value` strings (`"fixed_unitary"`, `"variational"`).
- **Measurement class** is represented as `MeasurementClass` (enum).

### 0.2 Circuit Variant Handling
- The pipeline must support mitigation hooks that return:
  - a **single** `QuantumCircuit`, or
  - a **list** of circuits (e.g., ZNE).
- The compiler should decide:
  - either “pipeline stays single-circuit” (select one + store variants separately), or
  - “pipeline becomes multi-variant aware” (propagate `JobVariants` + mitigation variants).

### 0.3 Metadata Serialization
- `HOTResult.metadata` must contain JSON-serializable primitives where practical.
- Any dataclasses stored in metadata must be converted to `dict`.

---

## 1) Package Wiring / Imports

### 1.1 `hot_framework/__init__.py`
**Status:** already corrected.

**Keep consistent exports:**
- Export `HOTCompiler` from `hot_framework/compiler.py`.
- Export dataclasses/enums from `hot_framework/core.py`.

---

## 2) `hot_framework/core.py` (Types & Forward References)

### 2.1 Decide Canonical Storage for `HOTResult.algorithm_class`
**Current:** `Union[AlgorithmClass, str]`.

**Fix recommendation:**
- Prefer storing `AlgorithmClass` in `HOTResult.algorithm_class`.
- If a string is required for compatibility, store it in `metadata['algorithm_class']`.

### 2.2 Forward Reference / Circular Import
**Current:** `codesign_result: Optional['CoDesignResult']` with `TYPE_CHECKING` import.

**Fix recommendation:**
- Keep current approach.
- Ensure `variational.py` does not import `HOTResult` at module import time if it creates cycles.

---

## 3) `hot_framework/compiler.py` (Main Pipeline Correctness)

### 3.1 Metadata Consistency
**Current:** `metadata['measurement_policy']` is dict-like (good).

**Fix recommendation:**
- Ensure all metadata entries remain serializable:
  - `orphan_policy` should be a string mode.
  - `mitigation_hooks` should be list of strings.

### 3.2 Backend Name Retrieval
**Fix recommendation:**
- Standardize backend naming helper and use everywhere (`compiler.py`, `utils.py`, etc.).

### 3.3 Mitigation Hook Variant Propagation
**Current:** if hook returns list, compiler selects first and records count.

**Fix recommendation (choose one):**
- **Option A (minimal):**
  - Keep selecting first circuit for compilation.
  - Store all variants in `HOTResult.metadata['mitigation_variants']` (or a new field).
- **Option B (correct-by-design):**
  - Extend `HOTResult` and `JobVariants` to include mitigation variant sets explicitly.

---

## 4) `hot_framework/utils.py` (Calibration + Imports)

### 4.1 Public API for Calibration Fetch
**Current:** `get_calibration_data()` exists and `_get_calibration_data()` is alias (good).

**Fix recommendation:**
- Update all internal imports to use `get_calibration_data`.
- Keep `_get_calibration_data` indefinitely or deprecate with warnings.

### 4.2 Ensure CalibrationData Matches Real IBM Runtime Data (Later)
**Current:** calibration data is mocked via random numbers.

**Fix recommendation:**
- Introduce an adapter for IBM Runtime backends:
  - read coupling map
  - read qubit properties (T1/T2/readout error)
  - read gate errors

---

## 5) `hot_framework/measurement.py` (Highest Priority Correctness Fix)

### 5.1 Invalid Barrier Insertion
**Issue:** The code appends a tuple like `('barrier', ...)` into `QuantumCircuit.data`.

**Fix recommendation:**
- Replace tuple insertion with real Qiskit API:
  - `circuit.barrier(*qubits)`
  - or append a `Barrier` `Instruction` object.

### 5.2 Measurement Classification Key Stability
**Issue:** measurement classification maps `Instruction` object identity to class; copying/modifying circuits invalidates references.

**Fix recommendation:**
- Use a stable key:
  - index in circuit data, or
  - `(qubit_index, clbit_index, is_mid_circuit)` tuple, or
  - embed classification into `instruction.metadata` and re-read.

### 5.3 `has_classical_dependency()` Heuristic
**Issue:** assumes “mid-circuit measurement => classical dependency”.

**Fix recommendation:**
- Implement minimal real detection:
  - search for `if_test` / `c_if` usage referencing the same clbit.

---

## 6) `hot_framework/variational.py` (Second Highest Priority Correctness Fix)

### 6.1 Sampler Output Format Assumption
**Issue:** `Sampler` quasi distributions are not guaranteed to be bitstring-keyed; often int keys are returned.

**Fix recommendation:**
- Normalize sampler results to bitstrings:
  - convert integer outcome keys to bitstrings of length `n_qubits`.
- Use a single helper `normalize_counts(quasi_dist, n_qubits, shots)`.

### 6.2 Placeholder Circuit Builder
**Issue:** `_build_circuit()` constructs a generic circuit ignoring `AlgorithmSpec.circuit`.

**Fix recommendation:**
- If `AlgorithmSpec.circuit` exists:
  - parameter-bind (if any)
  - apply layout/scaffold mapping
  - measure outputs per spec
- Else:
  - keep placeholder builder but mark as fallback.

### 6.3 Runtime vs Local Primitives
**Issue:** current `Sampler` usage is local primitive, not IBM Runtime.

**Fix recommendation:**
- Define an abstraction:
  - `SamplerProvider` or `ExecutionBackend`
  - support local sampler AND IBM Runtime sampler.

---

## 7) `hot_framework/island_selection.py` (Third Highest Priority Correctness Fix)

### 7.1 `apply_orphan_policy()` Is No-Op
**Issue:** strict/diagnostic policies contain `pass` and do nothing.

**Fix recommendation:**
- Implement strict mode by filtering circuit operations to allowed qubits.
- Implement diagnostic mode by adding measurements on specified orphans.

### 7.2 `find_embedding()` Uses Exact Isomorphism
**Issue:** exact isomorphism is too strict; embeddable graphs may not be isomorphic.

**Fix recommendation:**
- Switch to subgraph monomorphism / VF2 `GraphMatcher(...).subgraph_is_isomorphic()`.
- If not available/too slow, implement heuristic mapping:
  - start from best qubit
  - grow mapping via BFS while minimizing edge error.

---

## 8) `hot_framework/mitigation.py` (Medium Priority)

### 8.1 Abstract Base Method Body
**Issue:** abstract `apply()` has `pass`.

**Fix recommendation:**
- Replace with `raise NotImplementedError`.

### 8.2 Hook Semantics
**Issue:** PauliTwirl currently mutates by randomly inserting gates, but it doesn’t properly wrap two-qubit gates with inverse Paulis tied to the initial selection.

**Fix recommendation:**
- Use a consistent twirl pair: sample Paulis, apply before, apply corresponding inverse after.

---

## 9) `hot_framework/topology.py` (Low/Medium Priority)

### 9.1 Two-Qubit Error Aggregation
**Issue:** `compute_embedding_score()` sums over all pairs, not actual interaction graph edges.

**Fix recommendation:**
- Accept `interaction_graph` edges and score only those mapped edges.

---

## 10) Tests / CI Alignment

### 10.1 Current Test Fragility
- Tests may fail due to:
  - invalid barrier insertion
  - sampler output bitstring assumption
  - orphan policy being a no-op

**Fix recommendation:**
- After implementing fixes in sections 5–7, run unit tests and update tests only if behavior intentionally changes.

---

## 11) Suggested Implementation Order (Best ROI)

1. **`measurement.py`**
   - barrier insertion correctness
   - stable classification keys

2. **`variational.py`**
   - sampler output normalization
   - respect `AlgorithmSpec.circuit`

3. **`island_selection.py`**
   - implement orphan policy
   - relax embedding search

4. **`topology.py` / `mitigation.py`**
   - scoring semantics
   - mitigation hook correctness

---

## 12) Definition of “Consistent Enough to Extend Into HOT Runtime Compiler”

The codebase is ready for IBM Runtime code generation when:
- compilation produces a circuit that is valid Qiskit IR (no invalid `data` tuples)
- variational evaluation uses consistent outcome key formatting
- orphan policy is enforced as specified
- island selection/scaffolding are deterministic given calibration inputs

---

## Appendix: Glossary
- **Island:** chosen subgraph of qubits used for computation.
- **Scaffold:** logical-to-physical embedding and any SWAP schedule.
- **H/O/T:** Hardware mitigation / Optimized island / Tidy measurement hygiene.
