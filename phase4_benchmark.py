import os
import json
import time
import uuid
import math
import hashlib
from collections import Counter, defaultdict
import networkx as nx

WORKSPACE_DIR = "data"
SPLIT_PATH = os.path.join(WORKSPACE_DIR, "dataset_split.json")
CACHE_PATH = os.path.join(WORKSPACE_DIR, "extractions_cache.json")
BASE_GRAPH_PATH = os.path.join(WORKSPACE_DIR, "base_graph.json")
BASE_COMM_PATH = os.path.join(WORKSPACE_DIR, "base_communities.json")
BASE_SUMM_PATH = os.path.join(WORKSPACE_DIR, "base_summaries.json")
BENCH_OUT_PATH = os.path.join(WORKSPACE_DIR, "benchmark_results.json")

DRIFT_THRESHOLD = 0.15

def compute_nmi(labels_true, labels_pred):
    """Computes Normalized Mutual Information (arithmetic mean) in pure Python."""
    if not labels_true or not labels_pred:
        return 0.0
    n = len(labels_true)
    
    # Contingency matrix
    contingency = defaultdict(lambda: defaultdict(int))
    count_t = Counter(labels_true)
    count_p = Counter(labels_pred)
    
    for t, p in zip(labels_true, labels_pred):
        contingency[t][p] += 1
        
    # Entropies
    h_t = -sum((cnt / n) * math.log(cnt / n) for cnt in count_t.values())
    h_p = -sum((cnt / n) * math.log(cnt / n) for cnt in count_p.values())
    
    if h_t + h_p == 0:
        return 1.0
        
    # Mutual Information
    mi = 0.0
    for t, row in contingency.items():
        for p, count_tp in row.items():
            if count_tp > 0:
                p_tp = count_tp / n
                p_t = count_t[t] / n
                p_p = count_p[p] / n
                mi += p_tp * math.log(p_tp / (p_t * p_p))
                
    return (2.0 * mi) / (h_t + h_p)

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

# 1. Establish initial structural baselines (|V_C| and internal edges |E_C|)
comm_to_nodes = {}
for n, c in node_to_comm.items():
    comm_to_nodes.setdefault(c, set()).add(n)

comm_baseline_v = {c_id: len(nodes) for c_id, nodes in comm_to_nodes.items()}
comm_baseline_e = {}
for c_id, nodes in comm_to_nodes.items():
    sub = G.subgraph(nodes)
    comm_baseline_e[c_id] = sub.number_of_edges()

mutation_v = {c_id: 0 for c_id in comm_to_nodes.keys()}
mutation_e = {c_id: 0 for c_id in comm_to_nodes.keys()}

comm_weights = {}
for n, c in node_to_comm.items():
    if G.has_node(n):
        degree = G.degree(n, weight="weight")
        comm_weights[c] = comm_weights.get(c, 0.0) + degree

stats = {
    "t1_extracted_nodes": 0,
    "fast_path_patches": 0,
    "resynthesis_calls": 0,
    "cache_hits": 0,
    "cache_misses": 0
}

start_time = time.time()

# 2. Incremental Streaming Ingress Pipeline
print("\n[STATUS] Ingesting streaming entities and relationships...")
for doc in t1_corpus:
    doc_hash = generate_doc_hash(doc["title"], doc["text"])
    extractions = extract_cache.get(doc_hash)

    if not extractions:
        stats["cache_misses"] += 1
        continue
    stats["cache_hits"] += 1

    entities = extractions.get("entities", [])
    relations = extractions.get("relations", [])
    chunk_node_map = {}

    for ent in entities:
        v_name = ent["name"]
        if v_name in node_to_comm:
            chunk_node_map[v_name] = node_to_comm[v_name]
            continue

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
            comm_baseline_v[best_comm] = 0
            comm_baseline_e[best_comm] = 0
            mutation_v[best_comm] = 0
            mutation_e[best_comm] = 0
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
                comm_baseline_v[best_comm] = 0
                comm_baseline_e[best_comm] = 0
                mutation_v[best_comm] = 0
                mutation_e[best_comm] = 0
                comm_to_nodes[best_comm] = set()

        G.add_node(v_name)
        node_to_comm[v_name] = best_comm
        chunk_node_map[v_name] = best_comm
        comm_to_nodes.setdefault(best_comm, set()).add(v_name)
        mutation_v[best_comm] = mutation_v.get(best_comm, 0) + 1
        stats["t1_extracted_nodes"] += 1

    # Ingress relationships & track intra-community edge mutations
    for rel in relations:
        src, tgt = rel["source"], rel["target"]
        if src != tgt:
            if G.has_edge(src, tgt):
                G[src][tgt]["weight"] = G[src][tgt].get("weight", 1.0) + 1.0
            else:
                G.add_edge(src, tgt, weight=1.0)

            c_src = node_to_comm.get(src)
            c_tgt = node_to_comm.get(tgt)
            if c_src and c_src == c_tgt:
                mutation_e[c_src] = mutation_e.get(c_src, 0) + 1

ingress_latency = time.time() - start_time

# 3. Comprehensive Drift Gating (Entity + Edge Perturbations)
print(f"[STATUS] Evaluating structural drift (Threshold = {DRIFT_THRESHOLD*100}%)...")
for c_id, base_v in comm_baseline_v.items():
    d_v = mutation_v.get(c_id, 0)
    d_e = mutation_e.get(c_id, 0)
    total_mutations = d_v + d_e

    if total_mutations == 0:
        continue

    # Boundary handling for novel clusters
    if base_v == 0:
        stats["fast_path_patches"] += 1
        continue

    base_denom = base_v + comm_baseline_e.get(c_id, 0)
    drift = total_mutations / base_denom if base_denom > 0 else 1.0

    if drift < DRIFT_THRESHOLD:
        stats["fast_path_patches"] += 1
    else:
        stats["resynthesis_calls"] += 1

# 4. Construct Full Rebuild Reference Graph & Baseline Partition
print("[STATUS] Running reference Full Rebuild & Global Louvain clustering on 100 documents...")
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

# Full Louvain rebuild communities on G_ref
ref_communities = nx.community.louvain_communities(G_ref, weight="weight", seed=42)
ref_node_to_comm = {}
for idx, cset in enumerate(ref_communities):
    for node in cset:
        ref_node_to_comm[node] = idx

# 5. Rigorous Equivalence & Alignment Checks
nodes_match = set(G.nodes()) == set(G_ref.nodes())

edges_G = set(tuple(sorted((u, v))) for u, v in G.edges())
edges_ref = set(tuple(sorted((u, v))) for u, v in G_ref.edges())
edges_match = edges_G == edges_ref

# Edge weights equality check
weights_match = True
for u, v in edges_G:
    if G[u][v].get("weight", 1.0) != G_ref[u][v].get("weight", 1.0):
        weights_match = False
        break

# Normalized Mutual Information (Partition Alignment)
common_nodes = sorted(list(set(G.nodes()).intersection(set(G_ref.nodes()))))
labels_incremental = [node_to_comm[n] for n in common_nodes]
labels_rebuild = [ref_node_to_comm[n] for n in common_nodes]
nmi_score = compute_nmi(labels_rebuild, labels_incremental)

# Real baseline rebuild calls = exact count of communities in full Louvain rebuild
rebuild_calls_baseline = len(ref_communities)
incremental_calls_used = stats["resynthesis_calls"]
reduction_pct = ((rebuild_calls_baseline - incremental_calls_used) / rebuild_calls_baseline) * 100

final_modularity = nx.community.modularity(G, comm_to_nodes.values(), weight="weight")

# 6. Save verified results
results = {
    "efficiency_ablation": {
        "full_rebuild_calls_required": rebuild_calls_baseline,
        "deltagraphrag_calls_used": incremental_calls_used,
        "reduction_pct": round(reduction_pct, 2)
    },
    "graph_verification": {
        "nodes_match": nodes_match,
        "edges_match": edges_match,
        "weights_match": weights_match,
        "partition_nmi": round(nmi_score, 4)
    },
    "metrics": {
        "final_nodes": G.number_of_nodes(),
        "final_edges": G.number_of_edges(),
        "final_modularity": round(final_modularity, 4),
        "total_incremental_communities": len(comm_to_nodes),
        "rebuild_baseline_communities": len(ref_communities),
        "fast_path_patches": stats["fast_path_patches"],
        "targeted_resynthesis_calls": stats["resynthesis_calls"]
    }
}

with open(BENCH_OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 62)
print("       DELTAGRAPHRAG RIGOROUS EMPIRICAL BENCHMARK             ")
print("=" * 62)
print(f"Topology Verification (Nodes, Edges, Weights):")
print(f"  Nodes Match (V_delta == V_ref)       : {'MATCH' if nodes_match else 'MISMATCH'}")
print(f"  Edges Match (E_delta == E_ref)       : {'MATCH' if edges_match else 'MISMATCH'}")
print(f"  Weights Match (W_delta == W_ref)     : {'MATCH' if weights_match else 'MISMATCH'}")
print(f"  Partition Alignment (Louvain NMI)    : {nmi_score:.4f}")
print("-" * 62)
print(f"Synthesis Call Ablation:")
print(f"  Full Rebuild Community Count (Calls) : {rebuild_calls_baseline}")
print(f"  DeltaGraphRAG Re-synthesis Calls     : {incremental_calls_used}")
print(f"  Tier 1 In-Memory Fast-Path Patches   : {stats['fast_path_patches']}")
print(f"  Modeled Synthesis Call Reduction     : {reduction_pct:.1f}%")
print("=" * 62)
print(f"[STATUS] Telemetry written to: {BENCH_OUT_PATH}")