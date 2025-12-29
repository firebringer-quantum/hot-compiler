import numpy as np
import networkx as nx

from qiskit.providers.fake_provider import GenericBackendV2

from hot_framework import HOTCompiler, AlgorithmSpec, AlgorithmClass, RetuningConfig, OrphanPolicy


def main() -> None:
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
        parameter_bounds=[(0.0, 2 * np.pi)] * 4,
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

    print(result.summary())
    if result.codesign_result:
        print("Best configuration:", result.codesign_result.best_configuration)


if __name__ == "__main__":
    main()
