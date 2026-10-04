from typing import Set
import networkx as nx

class ModularityEngine:
    @staticmethod
    def compute_delta_q(
        graph: nx.Graph,
        node: str,
        community_members: Set[str]
    ) -> float:
        m = graph.number_of_edges()
        if m == 0:
            return 0.0

        k_v = graph.degree(node)
        k_v_in = sum(1 for nbr in graph.neighbors(node) if nbr in community_members)
        sigma_tot = sum(graph.degree(u) for u in community_members if graph.has_node(u))

        return (k_v_in / (2.0 * m)) - ((sigma_tot * k_v) / (2.0 * (m ** 2)))

    @staticmethod
    def calculate_drift(mutations: int, initial_size: int) -> float:
        return mutations / float(max(initial_size, 1))
