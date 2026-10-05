import os
import sys
import json
import time
import uuid
import math
import hashlib
from collections import Counter, defaultdict
import networkx as nx
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

WORKSPACE_DIR = "data"
SPLIT_PATH = os.path.join(WORKSPACE_DIR, "dataset_split.json")
CACHE_PATH = os.path.join(WORKSPACE_DIR, "extractions_cache.json")
BASE_GRAPH_PATH = os.path.join(WORKSPACE_DIR, "base_graph.json")
BASE_COMM_PATH = os.path.join(WORKSPACE_DIR, "base_communities.json")
BASE_SUMM_PATH = os.path.join(WORKSPACE_DIR, "base_summaries.json")
BENCH_OUT_PATH = os.path.join(WORKSPACE_DIR, "benchmark_results.json")
CONFIG_PATH = os.path.join(WORKSPACE_DIR, "model_config.json")

DRIFT_THRESHOLD = 0.15

# Initialize Groq client for real LLM synthesis
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("Missing GROQ_API_KEY in .env file.")

client = Groq(api_key=api_key)

# Load model configuration
model_id = "llama-3.3-70b-versatile"
if os.path.exists(CONFIG_PATH):
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)
            model_id = cfg.get("fast_model", model_id)
    except Exception:
        pass

def compute_nmi(labels_true, labels_pred):
    """Computes Normalized Mutual Information in pure Python."""
    if not labels_true or not labels_pred:
        return 0.0
    n = len(labels_true)
    contingency = defaultdict(lambda: defaultdict(int))
    count_t = Counter(labels_true)
    count_p = Counter(labels_pred)
    for t, p in zip(labels_true, labels_pred):
        contingency[t][p] += 1
    h_t = -sum((cnt / n) * math.log(cnt / n) for cnt in count_t.values())
    h_p = -sum((cnt / n) * math.log(cnt / n) for cnt in count_p.values())
    if h_t + h_p == 0:
        return 1.0
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

def synthesize_community_summary(comm_id: str, member_nodes: list, graph: nx.Graph) -> dict:
    """Live API synthesis: Calls Groq to generate a holistic community description."""
    subgraph = graph.subgraph(member_nodes)
    edges_desc = []
    for u, v, data in subgraph.edges(data=True):
        edges_desc.append(f"{u} connected to {v} (weight {data.get('weight', 1.0)})")
    
    context_str = f"Community Members: {', '.join(member_nodes[:20])}\nKey Relationships:\n" + "\n".join(edges_desc[:25])
    prompt = (
        f"You are a knowledge graph synthesizer. Provide a concise, factual summary (2-3 sentences) "
        f"characterizing the core theme, entities, and primary connections in this community cluster:\n\n{context_str}"
    )

    t0 = time.time()
    response = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=256
    )
    latency = time.time() - t0
    
    return {
        "summary": response.choices[0].message.content.strip(),
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens,
        "latency_sec": latency
    }

print(f"[STATUS] Initializing benchmark pipeline (Synthesis Engine: {model_id})...")
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

# 1. Structural Baselines (|V_C| and intra-community edges |E_C|)
comm_to_nodes = {}
for n, c in node_to_comm.items():
    comm_to_nodes.setdefault(c, set()).add(n)

comm_baseline_v = {c: len(nodes) for c, nodes in comm_to_nodes.items()}
comm_baseline_e = {}
for c, nodes in comm_to_nodes.items():
    sub = G.subgraph(nodes)
    comm_baseline_e[c] = sub.number_of_edges()

mutation_v = {c: 0 for c in comm_to_nodes.keys()}
mutation_e_intra = {c: 0 for c in comm_to_nodes.keys()}
mutation_e_inter = {c: 0 for c in comm_to_nodes.keys()}

comm_weights = {}
for n, c in node_to_comm.items():
    if G.has_node(n):
        degree = G.degree(n, weight="weight")
        comm_weights[c] = comm_weights.get(c, 0.0) + degree

stats = {
    "t1_extracted_nodes": 0,
    "fast_path_patches": 0,
    "resynthesis_calls": 0,
    "actual_api_calls_made": 0,
    "actual_tokens_consumed": 0,
    "actual_synthesis_time": 0.0,
    "cache_hits": 0,
    "cache_misses": 0
}

start_ingress = time.time()

# 2. Incremental Streaming Ingress
print("\n[STATUS] Streaming ingress underway...")
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
            mutation_e_intra[best_comm] = 0
            mutation_e_inter[best_comm] = 0
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
                mutation_e_intra[best_comm] = 0
                mutation_e_inter[best_comm] = 0
                comm_to_nodes[best_comm] = set()

        G.add_node(v_name)
        node_to_comm[v_name] = best_comm
        chunk_node_map[v_name] = best_comm
        comm_to_nodes.setdefault(best_comm, set()).add(v_name)
        mutation_v[best_comm] = mutation_v.get(best_comm, 0) + 1
        stats["t1_extracted_nodes"] += 1

    # Ingress relationships & track edge mutations
    for rel in relations:
        src, tgt = rel["source"], rel["target"]
        if src != tgt:
            if G.has_edge(src, tgt):
                G[src][tgt]["weight"] = G[src][tgt].get("weight", 1.0) + 1.0
            else:
                G.add_edge(src, tgt, weight=1.0)

            c_src = node_to_comm.get(src)
            c_tgt = node_to_comm.get(tgt)
            if c_src and c_tgt:
                if c_src == c_tgt:
                    mutation_e_intra[c_src] = mutation_e_intra.get(c_src, 0) + 1
                else:
                    mutation_e_inter[c_src] = mutation_e_inter.get(c_src, 0) + 1
                    mutation_e_inter[c_tgt] = mutation_e_inter.get(c_tgt, 0) + 1

ingress_latency = time.time() - start_ingress

# 3. Two-Tier Perturbation Gating & Live Summarization
print(f"[STATUS] Evaluating structural drift (Threshold = {DRIFT_THRESHOLD*100}%)...")
re_synthesize_queue = []

for c_id, base_v in comm_baseline_v.items():
    d_v = mutation_v.get(c_id, 0)
    d_intra = mutation_e_intra.get(c_id, 0)
    d_inter = mutation_e_inter.get(c_id, 0)
    total_mutations = d_v + d_intra + (0.5 * d_inter)

    if total_mutations == 0:
        continue

    # Singleton / novel clusters absorbed into memory without LLM overhead
    if base_v == 0:
        stats["fast_path_patches"] += 1
        continue

    base_denom = base_v + comm_baseline_e.get(c_id, 0)
    drift = total_mutations / base_denom if base_denom > 0 else 1.0

    if drift < DRIFT_THRESHOLD:
        stats["fast_path_patches"] += 1
    else:
        stats["resynthesis_calls"] += 1
        re_synthesize_queue.append(c_id)

# Execute live LLM calls for any triggered communities
if re_synthesize_queue:
    print(f"[STATUS] Executing live Groq synthesis for {len(re_synthesize_queue)} breached communities...")
    for c_id in re_synthesize_queue:
        mem = list(comm_to_nodes[c_id])
        res = synthesize_community_summary(c_id, mem, G)
        summaries[c_id] = res["summary"]
        stats["actual_api_calls_made"] += 1
        stats["actual_tokens_consumed"] += res["total_tokens"]
        stats["actual_synthesis_time"] += res["latency_sec"]
        print(f"  [SYNTHESIS COMPLETE] Community {c_id}: {res['total_tokens']} tokens, {res['latency_sec']:.2f}s")
else:
    print("[STATUS] Fast-path absorbed all streaming mutations (0 communities breached drift threshold).")

# Save updated community summaries
with open(BASE_SUMM_PATH, "w", encoding="utf-8") as f:
    json.dump(summaries, f, indent=2)

# 4. Independent Reference Graph & Measured Full-Rebuild Baseline
print("[STATUS] Constructing Full Rebuild reference graph & global Louvain partitioning...")
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

ref_communities = nx.community.louvain_communities(G_ref, weight="weight", seed=42)
ref_node_to_comm = {}
for idx, cset in enumerate(ref_communities):
    for node in cset:
        ref_node_to_comm[node] = idx

# 5. Measure Empirical Full Rebuild Baseline via Live Sample Calls
SAMPLE_SIZE = min(3, len(ref_communities))
print(f"[STATUS] Measuring empirical Full Rebuild token cost across {SAMPLE_SIZE} sample communities...")
sample_tokens = []
sample_latencies = []

for cset in list(ref_communities)[:SAMPLE_SIZE]:
    res = synthesize_community_summary("sample_ref", list(cset), G_ref)
    sample_tokens.append(res["total_tokens"])
    sample_latencies.append(res["latency_sec"])
    time.sleep(0.5)

avg_tokens_per_comm = sum(sample_tokens) / len(sample_tokens)
avg_latency_per_comm = sum(sample_latencies) / len(sample_latencies)

measured_rebuild_calls = len(ref_communities)
measured_rebuild_tokens = int(measured_rebuild_calls * avg_tokens_per_comm)
measured_rebuild_wall_time = measured_rebuild_calls * avg_latency_per_comm

delta_calls = stats["actual_api_calls_made"]
delta_tokens = stats["actual_tokens_consumed"]

token_savings_pct = (
    ((measured_rebuild_tokens - delta_tokens) / measured_rebuild_tokens) * 100.0
    if measured_rebuild_tokens > 0 else 0.0
)
call_reduction_pct = (
    ((measured_rebuild_calls - delta_calls) / measured_rebuild_calls) * 100.0
    if measured_rebuild_calls > 0 else 0.0
)

# 6. Set-Theoretic & Alignment Verification
nodes_match = set(G.nodes()) == set(G_ref.nodes())
edges_G = set(tuple(sorted((u, v))) for u, v in G.edges())
edges_ref = set(tuple(sorted((u, v))) for u, v in G_ref.edges())
edges_match = edges_G == edges_ref

weights_match = True
for u, v in edges_G:
    if G[u][v].get("weight", 1.0) != G_ref[u][v].get("weight", 1.0):
        weights_match = False
        break

common_nodes = sorted(list(set(G.nodes()).intersection(set(G_ref.nodes()))))
labels_incremental = [node_to_comm[n] for n in common_nodes]
labels_rebuild = [ref_node_to_comm[n] for n in common_nodes]
nmi_score = compute_nmi(labels_rebuild, labels_incremental)

final_modularity = nx.community.modularity(G, comm_to_nodes.values(), weight="weight")

# 7. Write Verified Telemetry Artifact
results = {
    "empirical_synthesis_ablation": {
        "full_rebuild_required_calls": measured_rebuild_calls,
        "deltagraphrag_actual_calls": delta_calls,
        "api_call_reduction_pct": round(call_reduction_pct, 2),
        "full_rebuild_modeled_tokens": measured_rebuild_tokens,
        "deltagraphrag_measured_tokens": delta_tokens,
        "token_reduction_pct": round(token_savings_pct, 2),
        "rebuild_modeled_latency_sec": round(measured_rebuild_wall_time, 2),
        "deltagraphrag_ingress_wall_sec": round(ingress_latency, 2)
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
        "rebuild_baseline_communities": measured_rebuild_calls,
        "fast_path_patches": stats["fast_path_patches"],
        "targeted_resynthesis_calls": delta_calls
    }
}

with open(BENCH_OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 65)
print("     DELTAGRAPHRAG VERIFIED EMPIRICAL BENCHMARK (LIVE LLM)    ")
print("=" * 65)
print("1. Topological & Structural Equivalence:")
print(f"  Nodes Match (V_delta == V_ref)       : {'MATCH' if nodes_match else 'MISMATCH'}")
print(f"  Edges Match (E_delta == E_ref)       : {'MATCH' if edges_match else 'MISMATCH'}")
print(f"  Weights Match (W_delta == W_ref)     : {'MATCH' if weights_match else 'MISMATCH'}")
print(f"  Louvain Alignment Score (NMI)        : {nmi_score:.4f}")
print("-" * 65)
print("2. Live Measured Synthesis Invocations & Cost:")
print(f"  Full Rebuild Communities to Index    : {measured_rebuild_calls} communities")
print(f"  Empirical Mean Tokens / Community    : {avg_tokens_per_comm:.1f} tokens")
print(f"  Projected Full Rebuild Total Tokens  : {measured_rebuild_tokens:,} tokens")
print(f"  DeltaGraphRAG Live Re-synthesis Calls: {delta_calls} calls")
print(f"  DeltaGraphRAG Measured Live Tokens   : {delta_tokens:,} tokens")
print(f"  Empirical Synthesis Call Reduction   : {call_reduction_pct:.1f}%")
print(f"  Empirical Token Consumption Savings  : {token_savings_pct:.1f}%")
print("=" * 65)
print(f"[STATUS] Telemetry saved to: {BENCH_OUT_PATH}")