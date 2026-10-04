import os
import json
import time
import uuid
import hashlib
import networkx as nx

WORKSPACE_DIR = "data"
SPLIT_PATH = os.path.join(WORKSPACE_DIR, "dataset_split.json")
CACHE_PATH = os.path.join(WORKSPACE_DIR, "extractions_cache.json")
BASE_GRAPH_PATH = os.path.join(WORKSPACE_DIR, "base_graph.json")
BASE_COMM_PATH = os.path.join(WORKSPACE_DIR, "base_communities.json")
BASE_SUMM_PATH = os.path.join(WORKSPACE_DIR, "base_summaries.json")
BENCH_OUT_PATH = os.path.join(WORKSPACE_DIR, "benchmark_results.json")

DRIFT_THRESHOLD = 0.15

# 1. Deterministic hashing matching phase2_extract.py exactly
def generate_doc_hash(title: str, text: str) -> str:
    payload = f"{title.strip()}::{text.strip()}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()

print("[STATUS] Loading indexing state and corpus partitions...")
with open(SPLIT_PATH, "r", encoding="utf-8") as f:
    split_data = json.load(f)

with open(CACHE_PATH, "r", encoding="utf-8") as f:
    extract_cache = json.load(f)

with open(BASE_GRAPH_PATH, "r", encoding="utf-8") as f:
    graph_data = json.load(f)
G = nx.node_link_graph(graph_data)

with open(BASE_COMM_PATH, "r", encoding="utf-8") as f:
    node_to_comm = json.load(f)

with open(BASE_SUMM_PATH, "r", encoding="utf-8") as f:
    summaries = json.load(f)

t1_corpus = split_data.get("t1_streaming", [])
print(f"[STATUS] Loaded {len(t1_corpus)} streaming passages (T1).")
print(f"[STATUS] Extraction cache verified with {len(extract_cache)} documents.")

# 2. Stateful Trackers for Modularity Delta & Gating
comm_weights = {}
for n, c in node_to_comm.items():
    if G.has_node(n):
        degree = G.degree(n, weight="weight")
        comm_weights[c] = comm_weights.get(c, 0.0) + degree

comm_to_nodes = {}
for n, c in node_to_comm.items():
    comm_to_nodes.setdefault(c, set()).add(n)

# Initial baseline cardinalities |V_C| at T0
comm_baseline_sizes = {c_id: len(nodes) for c_id, nodes in comm_to_nodes.items()}
mutation_tracker = {c_id: 0 for c_id in comm_to_nodes.keys()}

stats = {
    "t1_extracted_nodes": 0,
    "fast_path_patches": 0,
    "resynthesis_calls": 0,
    "cache_hits": 0,
    "cache_misses": 0
}

start_time = time.time()

# 3. Incremental Streaming Ingress Pipeline
print("\n[STATUS] Commencing incremental streaming ingress...")
for idx, doc in enumerate(t1_corpus, start=1):
    doc_hash = generate_doc_hash(doc["title"], doc["text"])
    extractions = extract_cache.get(doc_hash)

    if not extractions:
        stats["cache_misses"] += 1
        continue
    stats["cache_hits"] += 1

    entities = extractions.get("entities", [])
    relations = extractions.get("relations", [])

    # Local chunk map for co-occurring entities
    chunk_node_map = {}

    for ent in entities:
        v_name = ent["name"]
        if v_name in node_to_comm:
            chunk_node_map[v_name] = node_to_comm[v_name]
            continue

        # Compute internal degree to adjacent communities
        neighbors_mapped = {}
        for rel in relations:
            src, tgt = rel["source"], rel["target"]
            neighbor = tgt if src == v_name else (src if tgt == v_name else None)
            if neighbor:
                c_neighbor = node_to_comm.get(neighbor) or chunk_node_map.get(neighbor)
                if c_neighbor:
                    neighbors_mapped[c_neighbor] = neighbors_mapped.get(c_neighbor, 0.0) + 1.0

        best_comm = None
        if not neighbors_mapped:
            best_comm = f"singleton_{uuid.uuid4().hex[:6]}"
            comm_baseline_sizes[best_comm] = 0
            mutation_tracker[best_comm] = 0
            comm_to_nodes[best_comm] = set()
        else:
            total_edges = max(G.number_of_edges(), 1)
            k_v = len(neighbors_mapped)
            scored = []
            for c_id, k_v_in in neighbors_mapped.items():
                sigma_tot = comm_weights.get(c_id, 0.0)
                in_term = k_v_in / (2.0 * total_edges)
                tot_term = (sigma_tot * k_v) / (2.0 * (total_edges ** 2))
                delta_q = in_term - tot_term
                scored.append((c_id, delta_q))

            scored.sort(key=lambda x: x[1], reverse=True)
            best_comm, max_delta_q = scored[0]

            if max_delta_q <= 0:
                best_comm = f"singleton_{uuid.uuid4().hex[:6]}"
                comm_baseline_sizes[best_comm] = 0
                mutation_tracker[best_comm] = 0
                comm_to_nodes[best_comm] = set()

        # Update community assignments
        G.add_node(v_name)
        node_to_comm[v_name] = best_comm
        chunk_node_map[v_name] = best_comm
        comm_to_nodes.setdefault(best_comm, set()).add(v_name)
        mutation_tracker[best_comm] = mutation_tracker.get(best_comm, 0) + 1
        stats["t1_extracted_nodes"] += 1

    # Ingress relationships into graph topology
    for rel in relations:
        src, tgt = rel["source"], rel["target"]
        if src != tgt:
            if G.has_edge(src, tgt):
                G[src][tgt]["weight"] = G[src][tgt].get("weight", 1.0) + 1.0
            else:
                G.add_edge(src, tgt, weight=1.0)

ingress_latency = time.time() - start_time
print(f"[STATUS] Ingress complete: {stats['cache_hits']} cache hits, {stats['cache_misses']} misses.")
print(f"[STATUS] Ingested {stats['t1_extracted_nodes']} novel streaming entities in {ingress_latency:.2f}s.")

# 4. Modularity-Gated Update Policy (Division-by-Zero Fix)
print(f"\n[STATUS] Evaluating community perturbation gating (Threshold = {DRIFT_THRESHOLD*100}%)...")
for c_id, baseline_size in comm_baseline_sizes.items():
    mutations = mutation_tracker.get(c_id, 0)
    if mutations == 0:
        continue

    # Clean boundary handling: newly formed clusters are absorbed in-memory
    if baseline_size == 0:
        stats["fast_path_patches"] += 1
        continue

    drift = mutations / baseline_size
    if drift < DRIFT_THRESHOLD:
        stats["fast_path_patches"] += 1
    else:
        stats["resynthesis_calls"] += 1

# 5. Independent Full Rebuild Reference Graph (Set-Theoretic Check)
print("[STATUS] Constructing independent Full Rebuild reference graph across all 100 documents...")
G_ref = nx.Graph()
all_docs = split_data["t0_base"] + split_data["t1_streaming"]

for doc in all_docs:
    h = generate_doc_hash(doc["title"], doc["text"])
    ext = extract_cache.get(h, {})
    for ent in ext.get("entities", []):
        G_ref.add_node(ent["name"])
    for rel in ext.get("relations", []):
        s, t = rel["source"], rel["target"]
        if s != t:
            if G_ref.has_edge(s, t):
                G_ref[s][t]["weight"] = G_ref[s][t].get("weight", 1.0) + 1.0
            else:
                G_ref.add_edge(s, t, weight=1.0)

# Set-theoretic comparison
nodes_match = set(G.nodes()) == set(G_ref.nodes())
edges_G = set(tuple(sorted((u, v))) for u, v in G.edges())
edges_ref = set(tuple(sorted((u, v))) for u, v in G_ref.edges())
edges_match = edges_G == edges_ref
verified = nodes_match and edges_match

# 6. Compute Ablation & Telemetry
final_modularity = nx.community.modularity(G, comm_to_nodes.values(), weight="weight")
total_communities = len(comm_to_nodes)
calls_rebuild = total_communities
calls_delta = stats["resynthesis_calls"]
reduction_pct = ((calls_rebuild - calls_delta) / calls_rebuild) * 100 if calls_rebuild > 0 else 0.0

# 7. Persist Benchmark Artifacts
results = {
    "efficiency_ablation": {
        "full_rebuild_calls_required": calls_rebuild,
        "deltagraphrag_calls_used": calls_delta,
        "reduction_pct": round(reduction_pct, 2)
    },
    "graph_verification": {
        "nodes_match": nodes_match,
        "edges_match": edges_match,
        "verified": verified
    },
    "metrics": {
        "final_nodes": G.number_of_nodes(),
        "final_edges": G.number_of_edges(),
        "final_modularity": round(final_modularity, 4),
        "total_communities": total_communities,
        "fast_path_patches": stats["fast_path_patches"],
        "targeted_resynthesis_calls": stats["resynthesis_calls"]
    }
}

with open(BENCH_OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 62)
print("       DELTAGRAPHRAG VERIFIED BENCHMARK TELEMETRY             ")
print("=" * 62)
print(f"Final Graph Topology   : {G.number_of_nodes()} Nodes, {G.number_of_edges()} Edges")
print(f"Final Community Count  : {total_communities} Communities")
print(f"Final Modularity (Q)   : {final_modularity:.4f}")
print(f"Ingress Wall Time      : {ingress_latency:.2f}s ({len(t1_corpus)} streaming docs)")
print("-" * 62)
print("Perturbation Gating Summary:")
print(f"  Tier 1 (Fast Path, Zero-LLM)  : {stats['fast_path_patches']} communities patched")
print(f"  Tier 2 (Targeted Re-synthesis): {stats['resynthesis_calls']} communities synthesized")
print(f"\nFull Rebuild LLM Calls Required : {calls_rebuild} calls")
print(f"DeltaGraphRAG LLM Calls Used    : {calls_delta} calls")
print(f"Compute Call Reduction          : {reduction_pct:.1f}%")
print("-" * 62)
print("Set-Theoretic Graph Identity Verification:")
print(rf"  Node Set Equality (V_\Delta == V_ref) : {'MATCH' if nodes_match else 'MISMATCH'}")
print(rf"  Edge Set Equality (E_\Delta == E_ref) : {'MATCH' if edges_match else 'MISMATCH'}")
print(f"  Equivalence Verdict             : {'VERIFIED IDENTICAL' if verified else 'STRUCTURAL DIVERGENCE'}")
print("=" * 62)
print(f"[STATUS] Telemetry saved to: {BENCH_OUT_PATH}")