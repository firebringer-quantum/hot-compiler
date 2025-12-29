# IBM Runtime / IBM Quantum Notes

This project depends on `qiskit-ibm-runtime` so you can execute compiled circuits on IBM backends.

## Credentials

Use environment variables or a local `.env` file.

Common patterns:

- `QISKIT_IBM_TOKEN`
- `QISKIT_IBM_INSTANCE`

Do not commit credentials to git.

## Typical execution flow

1. Build or load a `QuantumCircuit`.
2. Compile with `HOTCompiler.compile(...)`.
3. Submit `result.compiled_circuit` (or `result.job_variants.sanitized`) using your chosen Qiskit Runtime primitive / job flow.

This repository intentionally keeps Runtime submission outside the compiler so you can integrate it into:

- Qiskit Runtime `Sampler`/`Estimator` flows
- Batch jobs
- Custom transpilation/execution pipelines

## Backend selection

During development, the tests/examples use `GenericBackendV2`.

For real hardware, use your IBM runtime backend object.
