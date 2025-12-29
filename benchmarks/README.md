# Benchmarks

This folder contains a small benchmarking harness to compare HOT compilation against Qiskit SABRE transpilation.

## Run

```bash
python benchmarks/run_hot_vs_sabre.py --artifacts benchmarks/artifacts
```

## Output

- `benchmarks/results.json`
- Optional QASM artifacts in `benchmarks/artifacts/`
