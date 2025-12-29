import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
from qiskit import QuantumCircuit, transpile
from qiskit.providers.fake_provider import GenericBackendV2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hot_framework import HOTCompiler, OrphanPolicy, MeasurementPolicy, AlgorithmSpec, AlgorithmClass


@dataclass
class CircuitMetrics:
    depth: int
    size: int
    width: int
    op_counts: Dict[str, int]
    n_2q: int
    n_native: int
    n_non_native: int


def _inst_parts(inst):
    op = getattr(inst, "operation", None)
    qargs = getattr(inst, "qubits", None)
    cargs = getattr(inst, "clbits", None)
    if op is None:
        op, qargs, cargs = inst
    return op, qargs, cargs


def _basis_names(backend: Any) -> List[str]:
    # BackendV2/Target path
    target = getattr(backend, "target", None)
    if target is not None:
        try:
            return sorted(list(target.operation_names))
        except Exception:
            pass

    # BackendV1 fallback
    cfg = getattr(backend, "configuration", None)
    if callable(cfg):
        cfg = cfg()
    basis = getattr(cfg, "basis_gates", None)
    return sorted(list(basis)) if basis else []


def compute_metrics(circuit: QuantumCircuit, *, basis_names: List[str]) -> CircuitMetrics:
    op_counts = {str(k): int(v) for k, v in circuit.count_ops().items()}

    n_2q = 0
    n_native = 0
    n_non_native = 0

    basis = set(basis_names)

    for inst in circuit.data:
        op, qargs, _cargs = _inst_parts(inst)
        if len(qargs) == 2:
            n_2q += 1
        if op.name in basis:
            n_native += 1
        else:
            n_non_native += 1

    return CircuitMetrics(
        depth=int(circuit.depth()),
        size=int(len(circuit.data)),
        width=int(circuit.num_qubits),
        op_counts=op_counts,
        n_2q=n_2q,
        n_native=n_native,
        n_non_native=n_non_native,
    )


def dump_qasm(circuit: QuantumCircuit) -> str:
    qasm = getattr(circuit, "qasm", None)
    if callable(qasm):
        try:
            return str(qasm())
        except Exception:
            pass

    try:
        from qiskit import qasm2
    except Exception as e:
        raise RuntimeError("Unable to export QASM for artifacts") from e

    try:
        return str(qasm2.dumps(circuit))
    except Exception:
        return str(circuit.draw(output="text"))


def build_backend(num_qubits: int, coupling: List[List[int]]) -> Any:
    return GenericBackendV2(num_qubits=num_qubits, coupling_map=coupling)


def coupling_line(num_qubits: int) -> List[List[int]]:
    return [[i, i + 1] for i in range(max(0, num_qubits - 1))]


def coupling_all_to_all(num_qubits: int) -> List[List[int]]:
    edges: List[List[int]] = []
    for i in range(num_qubits):
        for j in range(i + 1, num_qubits):
            edges.append([i, j])
    return edges


def build_hhl_circuit_from_instance(instance_json: Path, *, precision_bits: int, mode: str) -> Tuple[QuantumCircuit, Dict[str, Any]]:
    from HHL_Algorithm.qiskit_bsd_prototype import QiskitBSDPrototype

    prototype = QiskitBSDPrototype(use_ibm_runtime=False)
    instance = prototype.load_spectral_instance(str(instance_json))
    H = instance["H"].real

    if mode == "qpe":
        qc, meta = prototype.build_phase_estimation_circuit(H, precision_bits=precision_bits)
    elif mode == "hhl":
        b = prototype.define_bsd_rhs_vector(instance, vector_type="uniform")
        qc, meta = prototype.build_hhl_circuit(H, b, precision_bits=precision_bits)
    else:
        raise ValueError(f"Unknown HHL mode: {mode}")

    meta = dict(meta)
    meta["instance_curve_id"] = instance.get("curve_id")
    meta["instance_dimension"] = instance.get("dimension")
    return qc, meta


def build_interaction_graph_any_arity(circuit: QuantumCircuit) -> nx.Graph:
    g = nx.Graph()
    g.add_nodes_from(range(circuit.num_qubits))

    def _q_index(q):
        try:
            return circuit.find_bit(q).index
        except Exception:
            return circuit.qubits.index(q)

    for inst in circuit.data:
        _op, qargs, _cargs = _inst_parts(inst)
        q_idx = [_q_index(q) for q in qargs]
        for i in range(len(q_idx)):
            for j in range(i + 1, len(q_idx)):
                g.add_edge(int(q_idx[i]), int(q_idx[j]))

    return g


def compile_with_sabre(
    circuit: QuantumCircuit,
    backend: Any,
    *,
    seed: int,
    optimization_level: int,
) -> QuantumCircuit:
    return transpile(
        circuit,
        backend=backend,
        optimization_level=optimization_level,
        layout_method="sabre",
        routing_method="sabre",
        seed_transpiler=seed,
    )


def finalize_to_backend_native(
    circuit: QuantumCircuit,
    backend: Any,
    *,
    seed: int,
    optimization_level: int,
) -> QuantumCircuit:
    return transpile(
        circuit,
        backend=backend,
        optimization_level=optimization_level,
        seed_transpiler=seed,
    )


def compile_with_hot(
    circuit: QuantumCircuit,
    backend: Any,
    *,
    seed: int,
) -> QuantumCircuit:
    # HOT currently doesn’t take a random seed param; keep `seed` in report for reproducibility.
    interaction_graph = build_interaction_graph_any_arity(circuit)
    spec = AlgorithmSpec(
        name=circuit.name or "benchmark_case",
        algorithm_class=AlgorithmClass.FIXED_UNITARY,
        min_qubits=circuit.num_qubits,
        max_qubits=circuit.num_qubits,
        interaction_graph=interaction_graph,
        circuit=circuit,
        output_qubits=list(range(circuit.num_qubits)),
    )

    compiler = HOTCompiler(backend, calibration_source="live")
    result = compiler.compile(
        spec,
        algorithm_class="fixed_unitary",
        orphan_policy=OrphanPolicy(mode="strict"),
        measurement_policy=MeasurementPolicy(stagger_readout=True),
    )
    return result.compiled_circuit


def run_case(
    *,
    case_name: str,
    circuit: QuantumCircuit,
    backend: Any,
    seed: int,
    optimization_level: int,
    artifacts_dir: Optional[Path],
) -> Dict[str, Any]:
    basis_names = _basis_names(backend)

    baseline_metrics = compute_metrics(circuit, basis_names=basis_names)

    t0b = time.perf_counter()
    baseline_native = finalize_to_backend_native(
        circuit, backend, seed=seed, optimization_level=optimization_level
    )
    t1b = time.perf_counter()
    baseline_native_metrics = compute_metrics(baseline_native, basis_names=basis_names)

    t0 = time.perf_counter()
    hot_circuit = compile_with_hot(circuit, backend, seed=seed)
    t1 = time.perf_counter()
    hot_metrics = compute_metrics(hot_circuit, basis_names=basis_names)

    t0h2 = time.perf_counter()
    hot_native = finalize_to_backend_native(
        hot_circuit, backend, seed=seed, optimization_level=optimization_level
    )
    t1h2 = time.perf_counter()
    hot_native_metrics = compute_metrics(hot_native, basis_names=basis_names)

    t2 = time.perf_counter()
    sabre_circuit = compile_with_sabre(circuit, backend, seed=seed, optimization_level=optimization_level)
    t3 = time.perf_counter()
    sabre_metrics = compute_metrics(sabre_circuit, basis_names=basis_names)

    t0s2 = time.perf_counter()
    sabre_native = finalize_to_backend_native(
        sabre_circuit, backend, seed=seed, optimization_level=optimization_level
    )
    t1s2 = time.perf_counter()
    sabre_native_metrics = compute_metrics(sabre_native, basis_names=basis_names)

    if artifacts_dir is not None:
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        (artifacts_dir / f"{case_name}_baseline.qasm").write_text(
            dump_qasm(circuit), encoding="utf-8", errors="replace"
        )
        (artifacts_dir / f"{case_name}_baseline_native.qasm").write_text(
            dump_qasm(baseline_native), encoding="utf-8", errors="replace"
        )
        (artifacts_dir / f"{case_name}_hot.qasm").write_text(
            dump_qasm(hot_circuit), encoding="utf-8", errors="replace"
        )
        (artifacts_dir / f"{case_name}_hot_native.qasm").write_text(
            dump_qasm(hot_native), encoding="utf-8", errors="replace"
        )
        (artifacts_dir / f"{case_name}_sabre.qasm").write_text(
            dump_qasm(sabre_circuit), encoding="utf-8", errors="replace"
        )
        (artifacts_dir / f"{case_name}_sabre_native.qasm").write_text(
            dump_qasm(sabre_native), encoding="utf-8", errors="replace"
        )

    return {
        "case": case_name,
        "seed": seed,
        "optimization_level": optimization_level,
        "backend": {
            "name": getattr(backend, "name", None)() if callable(getattr(backend, "name", None)) else getattr(backend, "name", None),
            "num_qubits": int(getattr(backend, "num_qubits", circuit.num_qubits)),
            "basis_gates": basis_names,
        },
        "baseline": {
            "metrics": asdict(baseline_metrics),
            "native_metrics": asdict(baseline_native_metrics),
            "finalize_seconds": float(t1b - t0b),
        },
        "hot": {
            "metrics": asdict(hot_metrics),
            "compile_seconds": float(t1 - t0),
            "native_metrics": asdict(hot_native_metrics),
            "finalize_seconds": float(t1h2 - t0h2),
        },
        "sabre": {
            "metrics": asdict(sabre_metrics),
            "compile_seconds": float(t3 - t2),
            "native_metrics": asdict(sabre_native_metrics),
            "finalize_seconds": float(t1s2 - t0s2),
        },
    }


def build_regev_circuit(*, n_solution_qubits: int, apply_mobius: bool, blind: bool) -> QuantumCircuit:
    repo_root = Path(__file__).resolve().parents[1]
    regev_dir = repo_root / "Regev_Algorithm"
    sys.path.insert(0, str(regev_dir))
    if blind:
        import regev_v5_harness_blind as harness
    else:
        import regev_v5_harness as harness
    qc = harness.build_regev_style_toy_circuit(
        n_solution_qubits=int(n_solution_qubits),
        apply_mobius=bool(apply_mobius),
    )
    return qc


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(Path("benchmarks") / "results.json"))
    parser.add_argument("--artifacts", default="")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--optimization-level", type=int, default=3)
    parser.add_argument("--backend-qubits", type=int, default=27)
    parser.add_argument("--topology", choices=["line", "all_to_all"], default="line")
    parser.add_argument("--precision-bits", type=int, default=4)
    parser.add_argument("--hhl-mode", choices=["qpe", "hhl"], default="qpe")
    parser.add_argument("--hhl-instance", default=str(Path("HHL_Algorithm") / "instances" / "11a1.json"))
    parser.add_argument("--include-regev", action="store_true")
    parser.add_argument("--regev-n", type=int, default=8)
    parser.add_argument("--regev-mobius", action="store_true")
    parser.add_argument("--regev-blind", action="store_true")

    args = parser.parse_args()

    if args.topology == "line":
        coupling = coupling_line(args.backend_qubits)
    else:
        coupling = coupling_all_to_all(args.backend_qubits)
    backend = build_backend(args.backend_qubits, coupling)

    results: List[Dict[str, Any]] = []

    instance_path = Path(args.hhl_instance)
    hhl_circuit, hhl_meta = build_hhl_circuit_from_instance(
        instance_path, precision_bits=args.precision_bits, mode=args.hhl_mode
    )
    hhl_circuit.name = f"bsd_{args.hhl_mode}_{instance_path.stem}"

    artifacts_dir = Path(args.artifacts) if args.artifacts else None

    results.append(
        {
            **run_case(
                case_name=hhl_circuit.name,
                circuit=hhl_circuit,
                backend=backend,
                seed=args.seed,
                optimization_level=args.optimization_level,
                artifacts_dir=artifacts_dir,
            ),
            "case_metadata": hhl_meta,
        }
    )

    if args.include_regev:
        regev_circuit = build_regev_circuit(
            n_solution_qubits=args.regev_n,
            apply_mobius=args.regev_mobius,
            blind=args.regev_blind,
        )
        regev_circuit.name = (
            f"regev_v5_n{args.regev_n}" + ("_mobius" if args.regev_mobius else "") + ("_blind" if args.regev_blind else "")
        )
        results.append(
            run_case(
                case_name=regev_circuit.name,
                circuit=regev_circuit,
                backend=backend,
                seed=args.seed,
                optimization_level=args.optimization_level,
                artifacts_dir=artifacts_dir,
            )
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "env": {
            "python": os.sys.version,
            "qiskit": __import__("qiskit").__version__,
        },
        "results": results,
    }

    out_path.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
