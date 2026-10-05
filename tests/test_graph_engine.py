import os
import sys
import unittest
import networkx as nx

# Add project root to path for src imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.graph_engine import ModularityEngine


class TestModularityEngine(unittest.TestCase):
    def setUp(self):
        self.engine = ModularityEngine()

    # -------------------------------------------------------------
    # 1. Structural Drift Calculation Tests
    # -------------------------------------------------------------
    def test_calculate_drift_zero_baseline_fast_path(self):
        """Newly created streaming communities (initial_size == 0) must return 0.0 drift."""
        drift = self.engine.calculate_drift(mutations=5, initial_size=0, baseline_edges=0)
        self.assertEqual(drift, 0.0)

    def test_calculate_drift_zero_capacity_guard(self):
        """Zero denominator capacity must return 0.0 instead of dividing by zero."""
        drift = self.engine.calculate_drift(mutations=2, initial_size=0, baseline_edges=0)
        self.assertEqual(drift, 0.0)

    def test_calculate_drift_vertex_only(self):
        """Standard vertex-only drift calculation."""
        # 3 mutations on 20 initial nodes = 3 / 20 = 0.15
        drift = self.engine.calculate_drift(mutations=3, initial_size=20)
        self.assertAlmostEqual(drift, 0.15, places=5)

    def test_calculate_drift_with_baseline_edges(self):
        """Drift calculation scaled across both initial vertices and intra-community edges."""
        # 3 mutations on 10 nodes + 10 baseline edges = 3 / (10 + 10) = 0.15
        drift = self.engine.calculate_drift(mutations=3, initial_size=10, baseline_edges=10)
        self.assertAlmostEqual(drift, 0.15, places=5)

    def test_calculate_drift_below_threshold(self):
        """Mutations under the 15% threshold should evaluate correctly."""
        # 1 mutation on 10 initial nodes = 0.10 (< 0.15)
        drift = self.engine.calculate_drift(mutations=1, initial_size=10)
        self.assertLess(drift, 0.15)

    # -------------------------------------------------------------
    # 2. Modularity Gain (Delta Q) Calculation Tests
    # -------------------------------------------------------------
    def test_compute_delta_q_empty_graph(self):
        """An empty graph with zero edges must return 0.0 without ZeroDivisionError."""
        G = nx.Graph()
        G.add_node("A")
        delta_q = self.engine.compute_delta_q(G, "A", community_members={"B"})
        self.assertEqual(delta_q, 0.0)

    def test_compute_delta_q_missing_candidate_node(self):
        """Candidate node not yet present in graph should not throw NetworkXError."""
        G = nx.Graph()
        G.add_edge("B", "C", weight=1.0)
        # "A" is not yet in G
        delta_q = self.engine.compute_delta_q(G, "A", community_members={"B", "C"})
        self.assertEqual(delta_q, 0.0)

    def test_compute_delta_q_positive_gain_for_connected_community(self):
        """Placing a node into a well-connected community should yield higher Delta Q."""
        G = nx.Graph()
        # Community C1: {B, C}
        G.add_edge("B", "C", weight=1.0)
        # Community C2: {D, E}
        G.add_edge("D", "E", weight=1.0)
        # Candidate node A connects strongly to C1 (B and C)
        G.add_edge("A", "B", weight=2.0)
        G.add_edge("A", "C", weight=2.0)

        delta_q_c1 = self.engine.compute_delta_q(G, "A", community_members={"B", "C"})
        delta_q_c2 = self.engine.compute_delta_q(G, "A", community_members={"D", "E"})

        self.assertGreater(delta_q_c1, delta_q_c2)

    def test_compute_delta_q_directed_graph_conversion(self):
        """Passing a DiGraph should be safely converted to undirected representation."""
        G = nx.DiGraph()
        G.add_edge("A", "B", weight=1.0)
        delta_q = self.engine.compute_delta_q(G, "A", community_members={"B"})
        self.assertIsInstance(delta_q, float)


if __name__ == "__main__":
    unittest.main()