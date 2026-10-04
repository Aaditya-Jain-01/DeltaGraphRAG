import os
import json
import networkx as nx
import matplotlib

# Force headless rendering to eliminate Tkinter/Tcl dependencies
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WORKSPACE_DIR = "data"

# Load graph topology and communities
with open(os.path.join(WORKSPACE_DIR, "base_graph.json"), "r", encoding="utf-8") as f:
    graph_data = json.load(f)

with open(os.path.join(WORKSPACE_DIR, "base_communities.json"), "r", encoding="utf-8") as f:
    node_to_comm = json.load(f)

G = nx.node_link_graph(graph_data)

# Filter out isolated singletons for clean visualization
connected_nodes = [n for n in G.nodes() if G.degree(n) > 0]
subgraph = G.subgraph(connected_nodes)

print(f"[STATUS] Plotting graph with {subgraph.number_of_nodes()} connected nodes...")

fig, ax = plt.subplots(figsize=(14, 10), dpi=300)
pos = nx.spring_layout(subgraph, k=0.15, seed=42)

# Color nodes by community ID
node_colors = [node_to_comm.get(n, 0) for n in subgraph.nodes()]

nx.draw_networkx_nodes(
    subgraph,
    pos,
    node_size=35,
    node_color=node_colors,
    cmap=plt.cm.tab20,
    alpha=0.85,
    ax=ax,
)

nx.draw_networkx_edges(
    subgraph,
    pos,
    alpha=0.18,
    edge_color="#888888",
    width=0.8,
    ax=ax,
)

ax.set_title(
    "DeltaGraphRAG: Louvain Community Topology (Q ≈ 0.94)",
    fontsize=16,
    fontweight="bold",
    pad=20,
)
ax.axis("off")
fig.tight_layout()

out_path = os.path.join(WORKSPACE_DIR, "graph_topology.png")
fig.savefig(out_path)
plt.close(fig)

print(f"[STATUS] Visualization successfully saved to: {out_path}")