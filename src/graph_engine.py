from typing import Set, Optional
import networkx as nx

class ModularityEngine:
    @staticmethod
    def compute_delta_q(
        graph: nx.Graph,
        node: str,
        community_members: Set[str],
        weight: Optional[str] = "weight"
    ) -> float:
        """
        Computes the modularity gain (Delta Q) for placing a node into a target community.
        Supports both weighted and unweighted graphs.
        """
        if graph.is_directed():
            graph = graph.to_undirected()

        total_weight = graph.size(weight=weight)
        if total_weight == 0:
            return 0.0

        # Degree of candidate node (k_v)
        k_v = graph.degree(node, weight=weight) if graph.has_node(node) else 0.0

        # Sum of edge weights connecting node to target community (k_v_in)
        k_v_in = 0.0
        if graph.has_node(node):
            for nbr in graph.neighbors(node):
                if nbr in community_members:
                    edge_w = graph[node][nbr].get(weight, 1.0) if weight else 1.0
                    k_v_in += edge_w

        # Total degree of community members (Sigma_tot)
        sigma_tot = sum(
            graph.degree(u, weight=weight) 
            for u in community_members 
            if graph.has_node(u)
        )

        m = total_weight
        in_term = k_v_in / (2.0 * m)
        tot_term = (sigma_tot * k_v) / (2.0 * (m ** 2))

        return in_term - tot_term

    @staticmethod
    def calculate_drift(
        mutations: float,
        initial_size: int,
        baseline_edges: int = 0
    ) -> float:
        """
        Computes structural community drift relative to baseline capacity.
        
        Guards:
        1. Zero-baseline fast-path: If a community was created during streaming
           (initial_size == 0), it returns 0.0 drift to allow in-memory absorption.
        2. Capacity scaling: Accounts for both initial vertex size (|V_C|) and
           intra-community edge baseline (|E_C|).
        """
        if initial_size == 0:
            return 0.0

        baseline_capacity = float(initial_size + baseline_edges)
        if baseline_capacity <= 0.0:
            return 0.0

        return float(mutations) / baseline_capacity