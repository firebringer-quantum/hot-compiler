from qiskit import QuantumCircuit
from qiskit.providers.fake_provider import GenericBackendV2

from hot_framework import HOTCompiler, OrphanPolicy, MeasurementPolicy


def main() -> None:
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

    print(result.summary())
    print("Compiled depth:", result.compiled_circuit.depth())


if __name__ == "__main__":
    main()
