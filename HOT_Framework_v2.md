# HOT Framework v2.0 Specification

## Hardware-Optimized Techniques for Quantum Circuit Execution

**Version:** 2.0  
**Authors:** Justin Hughes / Firebringer AI  
**Status:** Draft for Review

---

## 1. Executive Summary

The HOT Framework provides a systematic methodology for quantum circuit optimization that operates above standard SDK compilation. HOT addresses three interconnected concerns:

- **H**ardware-aware error mitigation through topological scaffolding
- **O**ptimal island selection via calibration-driven qubit clustering  
- **T**idy measurement hygiene to minimize crosstalk and extraneous readout

**Critical Finding (v2.0):** For variational algorithms, HOT must be applied as a **co-design process** where layout optimization and parameter tuning occur jointly. Layout changes alone can degrade performance; layout + retuning consistently yields the best results.

---

## 2. Framework Architecture

### 2.1 Inputs

| Input | Description | Source |
|-------|-------------|--------|
| Algorithm specification | High-level description (Grover, QAOA, HHL, custom) | User/AI |
| Algorithm class | `fixed_unitary` or `variational` | Auto-detected or specified |
| Backend identifier | Target quantum device | User |
| Calibration data | Live or recent calibration snapshot | Backend API |
| Cost function (variational only) | Objective being optimized | User |

### 2.2 Outputs

| Output | Description |
|--------|-------------|
| Compiled circuit | Mapped to selected island with scaffolding applied |
| Island report | Selected qubits, error metrics, rationale |
| Hygiene manifest | Measurement classification and policy decisions |
| Retuning recommendation | For variational: whether angles need re-optimization |
| Job variants | Instrumented (diagnostic) and sanitized (production) versions |

### 2.3 Algorithm Classification

```
┌─────────────────────────────────────────────────────────────────┐
│                    Algorithm Classification                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  FIXED_UNITARY                    VARIATIONAL                   │
│  ─────────────                    ───────────                   │
│  • Shor's algorithm               • QAOA                        │
│  • HHL                            • VQE                         │
│  • Grover's search                • QSVM                        │
│  • QFT (standalone)               • Variational eigensolvers    │
│  • Phase estimation               • Quantum autoencoders        │
│                                                                  │
│  HOT Application:                 HOT Application:              │
│  Direct benefit from              Requires co-design:           │
│  layout optimization              layout + parameter retuning   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. H — Hardware/Topological Gate-Based Mitigation

### 3.1 Topological Scaffolding Layer

The scaffolding layer embeds the logical interaction graph into a connectivity pattern that minimizes error accumulation.

#### 3.1.1 Scaffolding Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| **Connectivity optimization** | Select subgraph matching algorithm's CNOT pattern with minimal aggregate error | IWSA, dense interaction graphs |
| **SWAP-bolstering** | Insert structured SWAP sequences along pre-computed low-error paths | Algorithms requiring long-range connectivity |
| **Topological embedding** | Map to specific patterns (Möbius ring, ladder, star) | Symmetry-sensitive algorithms |

#### 3.1.2 Scaffolding Algorithm

```python
def compute_scaffold(interaction_graph, backend_topology, calibration_data):
    """
    Embed logical interaction graph into hardware topology.
    
    Returns:
        scaffold: Mapping from logical to physical qubits
        swap_schedule: Required SWAP insertions (if any)
        estimated_error: Aggregate error estimate for this embedding
    """
    
    # 1. Extract candidate subgraphs of required size
    candidates = enumerate_connected_subgraphs(
        backend_topology, 
        size=interaction_graph.num_nodes
    )
    
    # 2. Score each candidate by error metrics
    scored = []
    for subgraph in candidates:
        # Compute embedding cost
        embedding = find_embedding(interaction_graph, subgraph)
        if embedding is None:
            continue
            
        score = compute_embedding_score(
            embedding,
            calibration_data,
            weights={
                'two_qubit_error': 0.6,
                'readout_error': 0.25,
                'T1_T2_ratio': 0.15
            }
        )
        scored.append((subgraph, embedding, score))
    
    # 3. Select best embedding
    best = min(scored, key=lambda x: x[2])
    
    # 4. Compute any required SWAPs
    swap_schedule = compute_swap_schedule(best[1], calibration_data)
    
    return Scaffold(
        mapping=best[1],
        swap_schedule=swap_schedule,
        estimated_error=best[2],
        topology_pattern=detect_pattern(best[0])
    )
```

#### 3.1.3 Error Metric Aggregation

For a candidate embedding, the aggregate error score is:

$$
E_{agg} = \sum_{(i,j) \in \text{edges}} w_{2Q} \cdot \epsilon_{2Q}(i,j) + \sum_{q \in \text{qubits}} \left( w_{RO} \cdot \epsilon_{RO}(q) + w_T \cdot f(T_1(q), T_2(q)) \right)
$$

Where:
- $\epsilon_{2Q}(i,j)$ = two-qubit gate error on edge $(i,j)$
- $\epsilon_{RO}(q)$ = readout error on qubit $q$
- $f(T_1, T_2)$ = coherence penalty function (e.g., $1/\min(T_1, T_2)$)
- $w_{2Q}, w_{RO}, w_T$ = configurable weights

### 3.2 Gate-Oriented Mitigation Hooks

Mitigation strategies attach to the scaffold, not raw physical indices:

```python
class MitigationHook:
    """Base class for scaffold-anchored mitigation."""
    
    def apply(self, circuit, scaffold, calibration_data):
        raise NotImplementedError

class DynamicalDecoupling(MitigationHook):
    """Insert DD sequences on idle qubits within the scaffold."""
    
    def __init__(self, sequence='XY4', threshold_idle_time=100):
        self.sequence = sequence
        self.threshold = threshold_idle_time
    
    def apply(self, circuit, scaffold, calibration_data):
        idle_periods = identify_idle_periods(circuit, scaffold)
        for qubit, (start, end) in idle_periods:
            if (end - start) > self.threshold:
                insert_dd_sequence(circuit, qubit, start, end, self.sequence)

class ZeroNoiseExtrapolation(MitigationHook):
    """Mark layers for ZNE scaling."""
    
    def __init__(self, target_layers='two_qubit', scale_factors=[1, 1.5, 2]):
        self.target_layers = target_layers
        self.scale_factors = scale_factors
    
    def apply(self, circuit, scaffold, calibration_data):
        # Returns multiple circuit variants for ZNE
        return [scale_circuit(circuit, scaffold, sf) for sf in self.scale_factors]
```

---

## 4. O — Optimized Island Selection

### 4.1 Qubit Island Discovery

Islands are connected or near-connected subgraphs ranked by aggregate quality metrics.

#### 4.1.1 Island Scoring Function

For a candidate island $I$:

$$
S(I) = \alpha \cdot \bar{\epsilon}_{RO}(I) + \beta \cdot \bar{\epsilon}_{2Q}(I) + \gamma \cdot \sigma_{calib}(I) + \delta \cdot \text{SWAP}_{expected}(I)
$$

Where:
- $\bar{\epsilon}_{RO}(I)$ = mean readout error across island qubits
- $\bar{\epsilon}_{2Q}(I)$ = mean 2Q error across island edges
- $\sigma_{calib}(I)$ = calibration stability (variance over recent snapshots)
- $\text{SWAP}_{expected}(I)$ = expected SWAP count for target algorithm

#### 4.1.2 Island Discovery Algorithm

```python
def discover_islands(backend, calibration_data, min_size, max_size):
    """
    Enumerate and rank candidate islands.
    
    Returns:
        List of Island objects sorted by quality score
    """
    
    topology = get_backend_topology(backend)
    
    islands = []
    for size in range(min_size, max_size + 1):
        for subgraph in enumerate_connected_subgraphs(topology, size):
            score = compute_island_score(
                subgraph, 
                calibration_data,
                stability_window=timedelta(hours=24)
            )
            islands.append(Island(
                qubits=subgraph.nodes,
                edges=subgraph.edges,
                score=score,
                metrics=extract_island_metrics(subgraph, calibration_data)
            ))
    
    return sorted(islands, key=lambda i: i.score)
```

### 4.2 Island Selection Policy

Given an algorithm's requirements, select the optimal island:

```python
def select_island(
    algorithm_spec,
    discovered_islands,
    scaffold_requirements,
    selection_policy='minimize_error'
):
    """
    Select best island for the given algorithm.
    
    Policies:
        'minimize_error': Lowest aggregate error that fits
        'minimize_size': Smallest island that fits (reduces crosstalk surface)
        'maximize_coherence': Best T1/T2 ratios
        'balanced': Weighted combination
    """
    
    # Filter to islands that can embed the scaffold
    viable = [
        island for island in discovered_islands
        if can_embed(scaffold_requirements, island)
    ]
    
    if not viable:
        raise NoViableIslandError(
            f"No island of size {scaffold_requirements.min_qubits}-"
            f"{scaffold_requirements.max_qubits} can embed the required topology"
        )
    
    # Apply selection policy
    if selection_policy == 'minimize_error':
        return min(viable, key=lambda i: i.score)
    elif selection_policy == 'minimize_size':
        return min(viable, key=lambda i: len(i.qubits))
    elif selection_policy == 'maximize_coherence':
        return max(viable, key=lambda i: i.metrics.mean_T2)
    elif selection_policy == 'balanced':
        return min(viable, key=lambda i: balanced_score(i))
```

### 4.3 Orphan-Exclusion Controls

Orphan qubits (qubits outside the active island) can introduce noise through crosstalk. HOT provides explicit control:

| Mode | Behavior | Use Case |
|------|----------|----------|
| **strict** | No initialization or measurement of qubits outside island | Production runs, noise-sensitive algorithms |
| **relaxed** | Allow spectator qubits if backend requires, placed maximally distant | Backends with measurement constraints |
| **diagnostic** | Include specific orphan measurements for crosstalk characterization | Calibration and debugging |

```python
@dataclass
class OrphanPolicy:
    mode: Literal['strict', 'relaxed', 'diagnostic']
    allowed_orphans: Optional[List[int]] = None  # For diagnostic mode
    distance_threshold: int = 2  # Minimum distance from island in relaxed mode
    
def apply_orphan_policy(circuit, island, policy, backend_topology):
    """
    Enforce orphan exclusion according to policy.
    """
    all_qubits = set(backend_topology.nodes)
    island_qubits = set(island.qubits)
    orphans = all_qubits - island_qubits
    
    if policy.mode == 'strict':
        # Remove all operations on orphan qubits
        circuit = filter_operations(circuit, allowed_qubits=island_qubits)
        
    elif policy.mode == 'relaxed':
        # Allow only distant orphans if backend requires
        required_orphans = get_backend_required_qubits(circuit, backend)
        for orphan in required_orphans:
            if distance(orphan, island, backend_topology) < policy.distance_threshold:
                raise OrphanViolationError(
                    f"Qubit {orphan} required by backend but too close to island"
                )
                
    elif policy.mode == 'diagnostic':
        # Add measurement on specified orphans for crosstalk analysis
        for orphan in policy.allowed_orphans:
            circuit.measure(orphan, classical_bit=f'orphan_{orphan}')
    
    return circuit
```

---

## 5. T — Tidy Measurement (Measurement Hygiene)

### 5.1 Measurement Classification

Every measurement in a circuit is classified:

| Class | Definition | Handling |
|-------|------------|----------|
| **Essential** | Algorithm semantics depend on result | Always include |
| **Diagnostic** | For debugging, tomography, or characterization | Include in instrumented runs only |
| **Extraneous** | Safe to remove without affecting algorithm | Remove in sanitized runs |

```python
def classify_measurements(circuit, algorithm_spec):
    """
    Classify each measurement in the circuit.
    
    Returns:
        Dict mapping measurement operations to classifications
    """
    
    classifications = {}
    
    for op in circuit.operations:
        if not is_measurement(op):
            continue
            
        qubit = op.qubits[0]
        
        # Check if measurement feeds into classical control
        if has_classical_dependency(circuit, op):
            classifications[op] = MeasurementClass.ESSENTIAL
            
        # Check if this is a final readout on an algorithm qubit
        elif qubit in algorithm_spec.output_qubits and is_final_measurement(circuit, op):
            classifications[op] = MeasurementClass.ESSENTIAL
            
        # Check if explicitly marked diagnostic
        elif op.metadata.get('diagnostic', False):
            classifications[op] = MeasurementClass.DIAGNOSTIC
            
        else:
            classifications[op] = MeasurementClass.EXTRANEOUS
    
    return classifications
```

### 5.2 Measurement Policy Rules

1. **No mid-circuit measurement on non-essential qubits** unless explicitly requested
2. **Minimize concurrent measurement on adjacent qubits** to reduce crosstalk
3. **Stagger readout** when possible on backends that support it

```python
@dataclass
class MeasurementPolicy:
    allow_mid_circuit: bool = False
    max_concurrent_adjacent: int = 1
    stagger_readout: bool = True
    readout_error_mitigation: bool = True

def apply_measurement_policy(circuit, classifications, policy, island):
    """
    Enforce measurement hygiene according to policy.
    """
    
    # Remove extraneous measurements
    circuit = remove_measurements(
        circuit, 
        [op for op, cls in classifications.items() 
         if cls == MeasurementClass.EXTRANEOUS]
    )
    
    # Handle mid-circuit measurements
    if not policy.allow_mid_circuit:
        mid_circuit = [
            op for op in circuit.operations 
            if is_measurement(op) and not is_final_measurement(circuit, op)
        ]
        if mid_circuit:
            # Check if any are essential
            essential_mid = [op for op in mid_circuit 
                           if classifications.get(op) == MeasurementClass.ESSENTIAL]
            if essential_mid and not policy.allow_mid_circuit:
                raise MeasurementPolicyViolation(
                    "Essential mid-circuit measurements present but policy forbids them"
                )
            # Remove non-essential mid-circuit measurements
            circuit = remove_measurements(
                circuit,
                [op for op in mid_circuit if op not in essential_mid]
            )
    
    # Stagger readout to reduce crosstalk
    if policy.stagger_readout:
        circuit = stagger_final_measurements(circuit, island)
    
    return circuit
```

### 5.3 Mid-Circuit Measurement & Reset Handling

When mid-circuit measurement/reset is required (e.g., ancilla reuse):

```python
def wrap_mid_circuit_measurement(circuit, measurement_op, calibration_data):
    """
    Apply protective measures around mid-circuit measurement.
    """
    
    qubit = measurement_op.qubits[0]
    neighbors = get_neighboring_qubits(qubit, circuit.topology)
    
    # 1. Apply dynamical decoupling on idle neighbors during measurement
    for neighbor in neighbors:
        if is_idle_during(neighbor, measurement_op, circuit):
            insert_dd_sequence(circuit, neighbor, 
                             start=measurement_op.start_time,
                             duration=measurement_op.duration)
    
    # 2. Add readout error mitigation tag
    measurement_op.metadata['apply_rem'] = True
    
    # 3. If reset follows, ensure adequate delay
    if has_reset_after(circuit, measurement_op):
        insert_delay(circuit, qubit, 
                    after=measurement_op,
                    duration=calibration_data.recommended_reset_delay(qubit))
    
    return circuit
```

### 5.4 Job-Level Hygiene Variants

HOT generates two standard job variants:

| Variant | Purpose | Measurements | Mitigation |
|---------|---------|--------------|------------|
| **Instrumented** | Diagnostics, characterization | Essential + Diagnostic | Full telemetry |
| **Sanitized** | Production execution | Essential only | Minimal overhead |

```python
def generate_job_variants(circuit, classifications, policy, scaffold):
    """
    Generate instrumented and sanitized job variants.
    """
    
    # Sanitized: essential measurements only
    sanitized = apply_measurement_policy(
        circuit.copy(),
        classifications,
        policy,
        scaffold.island
    )
    
    # Instrumented: essential + diagnostic
    instrumented_policy = dataclasses.replace(policy, 
        allow_diagnostic=True,
        add_tomography=True
    )
    instrumented = apply_measurement_policy(
        circuit.copy(),
        classifications,
        instrumented_policy,
        scaffold.island
    )
    
    # Add process tomography circuits if requested
    if instrumented_policy.add_tomography:
        instrumented = add_process_tomography_circuits(
            instrumented, 
            scaffold,
            target_layers=['two_qubit']
        )
    
    return JobVariants(
        sanitized=sanitized,
        instrumented=instrumented,
        comparison_metadata={
            'sanitized_measurements': count_measurements(sanitized),
            'instrumented_measurements': count_measurements(instrumented),
            'tomography_circuits': len(instrumented.tomography_circuits)
        }
    )
```

---

## 6. Variational Algorithm Co-Design (v2.0 Addition)

### 6.1 The Retuning Requirement

**Key Finding:** For variational algorithms, changing the qubit layout invalidates previously optimized parameters. The distribution of measurement outcomes shifts, and parameters optimized for one layout are suboptimal for another.

**Empirical Evidence (Market Split QAOA):**

| Configuration | Expected Violation | Best-of-Sample |
|---------------|-------------------|----------------|
| HOT + Retuned | 4,372,827 | 3,050,895 |
| No-HOT + Retuned | 4,428,799 | 3,050,895 |
| HOT + Original | 4,453,395 | 3,050,895 |
| No-HOT + Original | 4,536,033 | 3,221,358 |

**Interpretation:**
- HOT alone (without retuning) performs *worse* than retuned No-HOT
- HOT + retuning achieves the best results
- Retuning is mandatory for variational algorithms when layout changes

### 6.2 Co-Design Workflow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Variational Algorithm Co-Design Workflow                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  1. BASELINE ASSESSMENT                                                      │
│     ├── Run algorithm with default layout and initial parameters             │
│     ├── Record: expected cost, best-of-sample, distribution metrics          │
│     └── This is the "No-HOT Original" baseline                               │
│                                                                              │
│  2. ISLAND SELECTION (O)                                                     │
│     ├── Discover candidate islands from calibration data                     │
│     ├── Select optimal island for algorithm topology                         │
│     └── This changes the physical qubit mapping                              │
│                                                                              │
│  3. SCAFFOLDING (H)                                                          │
│     ├── Embed logical qubits into selected island                            │
│     ├── Apply topological pattern (Möbius, etc.)                             │
│     └── Determine SWAP schedule if needed                                    │
│                                                                              │
│  4. PARAMETER RETUNING (Critical for variational)                            │
│     ├── Initialize: either random, or transfer from baseline                 │
│     ├── Optimize parameters for NEW layout using:                            │
│     │   ├── Same optimizer as original (COBYLA, SPSA, etc.)                  │
│     │   ├── Same objective function                                          │
│     │   └── Same or comparable iteration budget                              │
│     └── This is the "HOT Retuned" configuration                              │
│                                                                              │
│  5. VALIDATION                                                               │
│     ├── Compare all four configurations:                                     │
│     │   ├── No-HOT Original                                                  │
│     │   ├── No-HOT Retuned (retune on original layout as control)            │
│     │   ├── HOT Original (HOT layout, original parameters)                   │
│     │   └── HOT Retuned (HOT layout, retuned parameters)                     │
│     └── Report both expected and best-of-sample metrics                      │
│                                                                              │
│  6. MEASUREMENT HYGIENE (T)                                                  │
│     ├── Classify measurements for final validated circuit                    │
│     └── Generate job variants                                                │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6.3 Retuning API

```python
@dataclass
class RetuningConfig:
    """Configuration for variational parameter retuning."""
    
    optimizer: str = 'COBYLA'  # Or 'SPSA', 'NFT', 'BFGS', etc.
    max_iterations: int = 100
    objective: Literal['expected_cost', 'best_of_sample', 'cvar'] = 'expected_cost'
    cvar_alpha: float = 0.1  # For CVaR objective
    
    # Initialization strategy for new layout
    init_strategy: Literal['random', 'transfer', 'perturb'] = 'transfer'
    perturbation_scale: float = 0.1  # For 'perturb' strategy
    
    # Early stopping
    convergence_threshold: float = 1e-4
    patience: int = 10

class VariationalCoDesigner:
    """
    Manages the co-design process for variational algorithms under HOT.
    """
    
    def __init__(self, algorithm_spec, backend, calibration_data):
        self.algorithm = algorithm_spec
        self.backend = backend
        self.calibration = calibration_data
        
    def run_codesign(self, retuning_config: RetuningConfig) -> CoDesignResult:
        """
        Execute the full co-design workflow.
        
        Returns:
            CoDesignResult with all four configurations and comparison metrics
        """
        
        # 1. Baseline assessment
        baseline = self._run_baseline()
        
        # 2-3. Island selection and scaffolding (H, O)
        scaffold = compute_scaffold(
            self.algorithm.interaction_graph,
            self.backend.topology,
            self.calibration
        )
        island = select_island(
            self.algorithm,
            discover_islands(self.backend, self.calibration, 
                           self.algorithm.min_qubits, 
                           self.algorithm.max_qubits),
            scaffold
        )
        
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
            'nohot_retuned': self._evaluate(baseline.scaffold, baseline.island, nohot_retuned_params),
            'hot_original': self._evaluate(scaffold, island, baseline.parameters),
            'hot_retuned': self._evaluate(scaffold, island, hot_retuned_params)
        }
        
        # 6. Measurement hygiene on best configuration
        best_config = min(results.items(), key=lambda x: x[1].expected_cost)
        final_circuit = self._apply_measurement_hygiene(
            best_config[1].circuit,
            scaffold if 'hot' in best_config[0] else baseline.scaffold
        )
        
        return CoDesignResult(
            configurations=results,
            best_configuration=best_config[0],
            final_circuit=final_circuit,
            improvement_over_baseline=(
                baseline.expected_cost - results[best_config[0]].expected_cost
            ) / baseline.expected_cost
        )
    
    def _retune_parameters(self, scaffold, island, config, init_from=None):
        """
        Optimize variational parameters for a specific layout.
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
                'ftol': config.convergence_threshold
            }
        )
        
        return result.x
```

### 6.4 Objective Function Selection

**Recommendation:** Use expected cost as the primary optimization objective, track best-of-sample as secondary metric.

| Objective | Definition | Pros | Cons |
|-----------|------------|------|------|
| **Expected cost** | $\mathbb{E}[C(x)]$ over measured distribution | Stable, matches QAOA spirit, exploits distribution shaping | May not find single best string |
| **Best-of-sample** | $\min_{x \in \text{samples}} C(x)$ | Direct "winner-take-all" | Noisy, sample-dependent |
| **CVaR-α** | Expected cost over worst α-fraction | Risk-aware, finds reliable good solutions | Requires more samples |

```python
def compute_objective(counts, cost_function, objective_type, alpha=0.1):
    """
    Compute objective value from measurement counts.
    """
    
    total_shots = sum(counts.values())
    costs = [(bitstring, cost_function(bitstring), count) 
             for bitstring, count in counts.items()]
    
    if objective_type == 'expected':
        return sum(cost * count for _, cost, count in costs) / total_shots
        
    elif objective_type == 'best_of_sample':
        return min(cost for _, cost, _ in costs)
        
    elif objective_type == 'cvar':
        # Sort by cost (best first for minimization)
        costs.sort(key=lambda x: x[1])
        
        # Accumulate until we have alpha fraction
        threshold_count = int(alpha * total_shots)
        accumulated = 0
        cvar_sum = 0
        
        for _, cost, count in costs:
            take = min(count, threshold_count - accumulated)
            cvar_sum += cost * take
            accumulated += take
            if accumulated >= threshold_count:
                break
                
        return cvar_sum / accumulated if accumulated > 0 else costs[0][1]
```

### 6.5 Distribution Analysis Metrics

When comparing HOT vs No-HOT configurations, track these distribution metrics:

```python
@dataclass
class DistributionMetrics:
    """Metrics for comparing measurement distributions."""
    
    tvd: float  # Total Variation Distance from baseline
    kl_divergence: float  # KL divergence from baseline
    entropy: float  # Shannon entropy of distribution
    
    # Cost-weighted metrics
    expected_cost: float
    cost_variance: float
    best_of_sample: float
    
    # Top-K analysis
    top_k_probability: float  # Probability mass in top K bitstrings
    top_k_avg_cost: float  # Average cost among top K
    top_k_best_cost: float  # Best cost among top K
    
    # Interference signature (for HOT analysis)
    constructive_interference_score: float  # Higher = more probability on good solutions

def compute_distribution_metrics(counts, cost_function, baseline_counts=None, k=10):
    """
    Compute comprehensive distribution metrics.
    """
    
    total = sum(counts.values())
    probs = {bs: c/total for bs, c in counts.items()}
    costs = {bs: cost_function(bs) for bs in counts}
    
    # Basic metrics
    entropy = -sum(p * np.log2(p) for p in probs.values() if p > 0)
    expected = sum(probs[bs] * costs[bs] for bs in probs)
    variance = sum(probs[bs] * (costs[bs] - expected)**2 for bs in probs)
    
    # Top-K analysis
    sorted_by_count = sorted(counts.items(), key=lambda x: -x[1])[:k]
    top_k_prob = sum(c for _, c in sorted_by_count) / total
    top_k_costs = [costs[bs] for bs, _ in sorted_by_count]
    
    # Comparison to baseline
    if baseline_counts:
        baseline_total = sum(baseline_counts.values())
        baseline_probs = {bs: c/baseline_total for bs, c in baseline_counts.items()}
        
        all_bitstrings = set(probs.keys()) | set(baseline_probs.keys())
        tvd = 0.5 * sum(abs(probs.get(bs, 0) - baseline_probs.get(bs, 0)) 
                       for bs in all_bitstrings)
        
        # KL divergence (with smoothing)
        eps = 1e-10
        kl = sum(probs.get(bs, eps) * np.log(probs.get(bs, eps) / baseline_probs.get(bs, eps))
                for bs in all_bitstrings if probs.get(bs, 0) > 0)
    else:
        tvd = None
        kl = None
    
    # Interference score: correlation between probability and solution quality
    # Higher score = layout pushes probability toward better solutions
    sorted_by_cost = sorted(costs.items(), key=lambda x: x[1])
    ranks = {bs: i for i, (bs, _) in enumerate(sorted_by_cost)}
    interference_score = -np.corrcoef(
        [probs.get(bs, 0) for bs in ranks],
        [ranks[bs] for bs in ranks]
    )[0, 1]  # Negative because lower cost = better
    
    return DistributionMetrics(
        tvd=tvd,
        kl_divergence=kl,
        entropy=entropy,
        expected_cost=expected,
        cost_variance=variance,
        best_of_sample=min(costs.values()),
        top_k_probability=top_k_prob,
        top_k_avg_cost=np.mean(top_k_costs),
        top_k_best_cost=min(top_k_costs),
        constructive_interference_score=interference_score
    )
```

---

## 7. Complete HOT Pipeline

### 7.1 Unified Entry Point

```python
class HOTCompiler:
    """
    Main entry point for HOT framework.
    """
    
    def __init__(self, backend, calibration_source='live'):
        self.backend = backend
        self.calibration = self._load_calibration(calibration_source)
        
    def compile(
        self,
        circuit_or_spec,
        algorithm_class: Literal['fixed_unitary', 'variational'] = 'auto',
        orphan_policy: OrphanPolicy = OrphanPolicy(mode='strict'),
        measurement_policy: MeasurementPolicy = MeasurementPolicy(),
        mitigation_hooks: List[MitigationHook] = None,
        retuning_config: RetuningConfig = None  # Required if variational
    ) -> HOTResult:
        """
        Compile a circuit through the full HOT pipeline.
        
        Args:
            circuit_or_spec: Qiskit circuit or algorithm specification
            algorithm_class: 'fixed_unitary', 'variational', or 'auto' to detect
            orphan_policy: How to handle qubits outside the selected island
            measurement_policy: Measurement hygiene configuration
            mitigation_hooks: Optional error mitigation strategies
            retuning_config: Required for variational algorithms
            
        Returns:
            HOTResult containing compiled circuit(s) and metadata
        """
        
        # Auto-detect algorithm class if needed
        if algorithm_class == 'auto':
            algorithm_class = self._detect_algorithm_class(circuit_or_spec)
        
        # Validate inputs
        if algorithm_class == 'variational' and retuning_config is None:
            raise ValueError(
                "Variational algorithms require retuning_config. "
                "HOT layout changes invalidate pre-optimized parameters."
            )
        
        # Convert to algorithm spec if raw circuit provided
        spec = self._to_algorithm_spec(circuit_or_spec)
        
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
            self.backend.topology,
            self.calibration
        )
        
        island = select_island(spec, islands, scaffold)
        
        # ===== Algorithm-class specific handling =====
        if algorithm_class == 'fixed_unitary':
            # Direct compilation - no parameter retuning needed
            compiled_circuit = self._compile_fixed_unitary(
                spec.circuit,
                scaffold,
                island,
                orphan_policy,
                mitigation_hooks
            )
            result_type = 'single'
            
        else:  # variational
            # Co-design workflow
            codesign_result = VariationalCoDesigner(
                spec, self.backend, self.calibration
            ).run_codesign(retuning_config)
            
            compiled_circuit = codesign_result.final_circuit
            result_type = 'codesign'
        
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
            self.backend.topology
        )
        
        # Generate job variants
        job_variants = generate_job_variants(
            compiled_circuit,
            classifications,
            measurement_policy,
            scaffold
        )
        
        # ===== Build result =====
        return HOTResult(
            compiled_circuit=compiled_circuit,
            job_variants=job_variants,
            island=island,
            scaffold=scaffold,
            algorithm_class=algorithm_class,
            codesign_result=codesign_result if algorithm_class == 'variational' else None,
            metadata={
                'backend': self.backend.name,
                'calibration_timestamp': self.calibration.timestamp,
                'estimated_error': scaffold.estimated_error,
                'orphan_policy': orphan_policy,
                'measurement_policy': measurement_policy
            }
        )
```

### 7.2 Result Structure

```python
@dataclass
class HOTResult:
    """Complete result from HOT compilation."""
    
    # Primary outputs
    compiled_circuit: QuantumCircuit
    job_variants: JobVariants
    
    # Selection metadata
    island: Island
    scaffold: Scaffold
    
    # Algorithm handling
    algorithm_class: str
    codesign_result: Optional[CoDesignResult]  # For variational
    
    # Execution metadata
    metadata: Dict
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"HOT Compilation Summary",
            f"=======================",
            f"Backend: {self.metadata['backend']}",
            f"Algorithm class: {self.algorithm_class}",
            f"",
            f"Island Selection:",
            f"  Qubits: {self.island.qubits}",
            f"  Score: {self.island.score:.4f}",
            f"",
            f"Scaffold:",
            f"  Pattern: {self.scaffold.topology_pattern}",
            f"  Estimated error: {self.scaffold.estimated_error:.4f}",
            f"  SWAPs required: {len(self.scaffold.swap_schedule)}",
        ]
        
        if self.codesign_result:
            lines.extend([
                f"",
                f"Variational Co-Design:",
                f"  Best configuration: {self.codesign_result.best_configuration}",
                f"  Improvement over baseline: {self.codesign_result.improvement_over_baseline:.1%}",
                f"",
                f"  Configuration comparison:",
            ])
            for name, result in self.codesign_result.configurations.items():
                lines.append(f"    {name}: expected={result.expected_cost:.2f}")
        
        lines.extend([
            f"",
            f"Measurement Hygiene:",
            f"  Sanitized measurements: {self.job_variants.comparison_metadata['sanitized_measurements']}",
            f"  Instrumented measurements: {self.job_variants.comparison_metadata['instrumented_measurements']}",
        ])
        
        return "\n".join(lines)
```

---

## 8. Usage Examples

### 8.1 Fixed-Unitary Algorithm (HHL)

```python
from hot_framework import HOTCompiler, OrphanPolicy, MeasurementPolicy

# Initialize compiler
compiler = HOTCompiler(backend=ibm_fez, calibration_source='live')

# Define HHL circuit
hhl_circuit = build_hhl_circuit(matrix_A, vector_b)

# Compile with HOT
result = compiler.compile(
    hhl_circuit,
    algorithm_class='fixed_unitary',
    orphan_policy=OrphanPolicy(mode='strict'),
    measurement_policy=MeasurementPolicy(stagger_readout=True)
)

# Execute
job = backend.run(result.job_variants.sanitized, shots=8192)
```

### 8.2 Variational Algorithm (QAOA)

```python
from hot_framework import HOTCompiler, RetuningConfig

# Initialize compiler
compiler = HOTCompiler(backend=ibm_fez)

# Define QAOA for Market Split
qaoa_spec = QAOASpec(
    cost_hamiltonian=market_split_hamiltonian,
    mixer_hamiltonian=standard_mixer,
    p_layers=2,
    initial_parameters=[0.5, 0.5, 0.3, 0.3]  # From prior optimization
)

# Compile with co-design
result = compiler.compile(
    qaoa_spec,
    algorithm_class='variational',
    retuning_config=RetuningConfig(
        optimizer='COBYLA',
        max_iterations=100,
        objective='expected_cost',
        init_strategy='transfer'
    ),
    orphan_policy=OrphanPolicy(mode='strict')
)

# Review co-design results
print(result.summary())
# Shows: HOT + Retuned beats all other configurations

# Execute best configuration
job = backend.run(result.job_variants.sanitized, shots=8192)
```

### 8.3 Comparing HOT vs No-HOT

```python
# Run controlled comparison
from hot_framework import run_controlled_comparison

comparison = run_controlled_comparison(
    algorithm_spec=qaoa_spec,
    backend=ibm_fez,
    retuning_config=RetuningConfig(
        optimizer='COBYLA',
        max_iterations=100,
        objective='expected_cost'
    ),
    shots=8192,
    repetitions=5  # Statistical significance
)

# Generate report
print(comparison.report())
# Outputs:
# - Expected violation: HOT+Retuned < No-HOT+Retuned < HOT+Orig < No-HOT+Orig
# - Best-of-sample: Multiple configs tied
# - Distribution metrics: TVD, interference scores
# - Recommendation: Use HOT with retuned parameters
```

---

## 9. Appendix: Theoretical Foundations

### 9.1 Why Layout Affects Variational Parameters

Variational algorithms optimize parameters $\vec{\theta}$ to minimize:

$$
\langle C \rangle = \text{Tr}\left[ C \cdot \rho(\vec{\theta}) \right]
$$

where $\rho(\vec{\theta})$ is the prepared quantum state. On real hardware, this becomes:

$$
\langle C \rangle_{hardware} = \text{Tr}\left[ C \cdot \mathcal{E}_{layout}(\rho(\vec{\theta})) \right]
$$

where $\mathcal{E}_{layout}$ is a layout-dependent noise channel incorporating:
- Gate errors (layout-dependent via calibration)
- Crosstalk (layout-dependent via qubit proximity)
- Measurement errors (qubit-dependent)

When layout changes, $\mathcal{E}_{layout}$ changes, so the optimal $\vec{\theta}^*$ changes.

### 9.2 HOT's Distribution Reshaping Effect

HOT selects layouts that:
1. Minimize aggregate error ($\mathcal{E}_{layout}$ closer to identity)
2. Create favorable interference patterns in the measurement distribution

The TVD between HOT and No-HOT distributions reflects both effects. When TVD is high (~0.41 in your experiments), HOT is significantly reshaping the distribution—potentially beneficially, but requiring parameter retuning to exploit.

### 9.3 Expected vs Best-of-Sample Objectives

**Expected value** captures the full distribution:
$$
\mathbb{E}[C] = \sum_x p(x) \cdot C(x)
$$

**Best-of-sample** captures only the tail:
$$
\min_{x \in \text{support}(p)} C(x)
$$

When multiple layouts achieve the same best-of-sample (as in your experiment), expected value discriminates by favoring layouts that concentrate probability on good solutions, not just those that occasionally sample them.

---

## 10. Changelog

### v2.0 (Current)
- Added variational algorithm co-design workflow
- Introduced `RetuningConfig` and `VariationalCoDesigner`
- Added distribution metrics and comparison tools
- Documented the "retuning requirement" with empirical evidence
- Clarified algorithm classification (fixed_unitary vs variational)

### v1.0 (Initial)
- Core H-O-T framework
- Island selection and scaffolding
- Measurement hygiene
- Orphan exclusion policies
