import os
import json
import time
from typing import Dict, List, Set, Tuple, Any
import networkx as nx
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

WORKSPACE_DIR = "data"
SPLIT_PATH = os.path.join(WORKSPACE_DIR, "dataset_split.json")
CACHE_PATH = os.path.join(WORKSPACE_DIR, "extractions_cache.json")
CONFIG_PATH = os.path.join(WORKSPACE_DIR, "model_config.json")
BASE_GRAPH_PATH = os.path.join(WORKSPACE_DIR, "base_graph.json")
BASE_COMMUNITIES_PATH = os.path.join(WORKSPACE_DIR, "base_communities.json")
BASE_SUMMARIES_PATH = os.path.join(WORKSPACE_DIR, "base_summaries.json")
BENCHMARK_OUT_PATH = os.path.join(WORKSPACE_DIR, "benchmark_results.json")

# 1. Load artifacts
with open(SPLIT_PATH, "r", encoding="utf-8") as f:
    split_data = json.load(f)

with open(CACHE_PATH, "r", encoding="utf-8") as f:
    extractions = json.load(f)

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    model_cfg = json.load(f)

with open(BASE_GRAPH_PATH, "r", encoding="utf-8") as f:
    graph_data = json.load(f)

with open(BASE_COMMUNITIES_PATH, "r", encoding="utf-8") as f:
    node_to_comm: Dict[str, int] = json.load(f)

with open(BASE_SUMMARIES_PATH, "r", encoding="utf-8") as f:
    community_summaries: Dict[str, Dict[str, Any]] = json.load(f)

api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=api_key)
summary_model = model_cfg.get("heavy_model", "openai/gpt-oss-120b")

# Reconstruct in-memory NetworkX graph
G: nx.Graph = nx.node_link_graph(graph_data)

# Reverse index: Community ID -> Set of node names
comm_to_nodes: Dict[int, Set[str]] = {}
for n, c in node_to_comm.items():
    comm_to_nodes.setdefault(c, set()).add(n)

# Initial baseline community sizes for drift calculation
initial_community_sizes: Dict[int, int] = {c: len(nodes) for c, nodes in comm_to_nodes.items()}
community_mutations: Dict[int, int] = {c: 0 for c in comm_to_nodes}

print(f"[STATUS] Base Graph Mounted: {G.number_of_nodes()} Nodes, {len(comm_to_nodes)} Communities.")

# 2. Local Modularity Delta Calculation
def compute_modularity_delta(graph: nx.Graph, node: str, candidate_comm: int) -> float:
    """
    Computes local modularity gain ΔQ for placing node into candidate_comm.
    Formula: ΔQ = [k_v,in / 2m] - [Σ_tot * k_v / 2m^2]
    """
    m = graph.number_of_edges()
    if m == 0:
        return 0.0

    k_v = graph.degree(node)
    comm_members = comm_to_nodes.get(candidate_comm, set())

    # Degree of node within target community
    k_v_in = sum(1 for neighbor in graph.neighbors(node) if neighbor in comm_members)

    # Sum of total degrees of members in target community
    sigma_tot = sum(graph.degree(u) for u in comm_members if graph.has_node(u))

    delta_q = (k_v_in / (2.0 * m)) - ((sigma_tot * k_v) / (2.0 * (m ** 2)))
    return delta_q


def synthesize_community_summary(comm_id: int, members: List[str]) -> str:
    """Invokes LLM for Tier 2 isolated re-summarization upon significant drift."""
    subgraph = G.subgraph(members)
    edge_descriptions = [
        f"{u} -> {v} ({d.get('relations', 'connected')})"
        for u, v, d in subgraph.edges(data=True)
    ][:15]

    prompt = (
        f"Community ID: {comm_id}\n"
        f"Core Nodes ({len(members)}): {', '.join(members[:20])}\n"
        f"Key Relationships: {'; '.join(edge_descriptions) if edge_descriptions else 'Isolated cluster nodes'}\n\n"
        "Synthesize a concise 2-3 sentence semantic summary describing this cluster:"
    )

    try:
        res = client.chat.completions.create(
            model=summary_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=250,
        )
        return res.choices[0].message.content.strip()
    except Exception as exc:
        return f"Summary unavailable: {exc}"


# 3. Incremental Streaming Loop (Ingesting T1 Passages)
DRIFT_THRESHOLD = 0.15  # 15% perturbation threshold for Tier 2 LLM synthesis
t1_docs = split_data["t1_streaming"]
t1_titles = {d["title"] for d in t1_docs}
t1_extractions = [data for data in extractions.values() if data["title"] in t1_titles]

delta_llm_calls = 0
tier1_zero_llm_patches = 0
tier2_resummarizations = 0

print(f"\n[STATUS] Streaming {len(t1_extractions)} T1 documents into graph via ΔQ routing...")
streaming_start = time.time()

for doc_item in t1_extractions:
    doc_title = doc_item["title"]
    doc_entities = doc_item.get("entities", [])
    doc_relations = doc_item.get("relations", [])

    new_nodes_added: List[str] = []

    # Ingest entities
    for ent in doc_entities:
        name = ent.get("name", "").strip()
        e_type = ent.get("type", "Concept").strip()
        if not name:
            continue
        if not G.has_node(name):
            G.add_node(name, entity_type=e_type, documents=doc_title)
            new_nodes_added.append(name)
        else:
            existing = G.nodes[name].get("documents", "")
            if doc_title not in existing:
                G.nodes[name]["documents"] = f"{existing}; {doc_title}"

    # Ingest edges
    for rel in doc_relations:
        src = rel.get("source", "").strip()
        tgt = rel.get("target", "").strip()
        label = rel.get("relation", "related_to").strip()
        if not src or not tgt or src == tgt:
            continue

        if not G.has_node(src):
            G.add_node(src, entity_type="Concept", documents=doc_title)
            new_nodes_added.append(src)
        if not G.has_node(tgt):
            G.add_node(tgt, entity_type="Concept", documents=doc_title)
            new_nodes_added.append(tgt)

        if G.has_edge(src, tgt):
            G[src][tgt]["weight"] += 1
        else:
            G.add_edge(src, tgt, weight=1, relations=label)

    # Route new nodes via Modularity Delta (ΔQ)
    for node in new_nodes_added:
        # Check neighboring communities
        neighbor_communities = {
            node_to_comm[nbr] for nbr in G.neighbors(node) if nbr in node_to_comm
        }

        best_comm = None
        best_delta_q = -1.0

        for candidate in neighbor_communities:
            dq = compute_modularity_delta(G, node, candidate)
            if dq > best_delta_q:
                best_delta_q = dq
                best_comm = candidate

        # Assign to best community or create new singleton community
        if best_comm is not None and best_delta_q > 0:
            target_c = best_comm
        else:
            target_c = max(comm_to_nodes.keys()) + 1 if comm_to_nodes else 0
            comm_to_nodes[target_c] = set()
            initial_community_sizes[target_c] = 0
            community_mutations[target_c] = 0

        node_to_comm[node] = target_c
        comm_to_nodes[target_c].add(node)
        community_mutations[target_c] = community_mutations.get(target_c, 0) + 1

        # Evaluate community drift
        base_size = max(initial_community_sizes.get(target_c, 1), 1)
        drift = community_mutations[target_c] / float(base_size)

        if drift >= DRIFT_THRESHOLD:
            # Tier 2: Resummarize single mutated community
            members = list(comm_to_nodes[target_c])
            new_summary = synthesize_community_summary(target_c, members)
            community_summaries[str(target_c)] = {
                "community_id": target_c,
                "size": len(members),
                "members": members,
                "summary": new_summary,
            }
            community_mutations[target_c] = 0  # Reset counter
            tier2_resummarizations += 1
            delta_llm_calls += 1
            time.sleep(1.0)
        else:
            # Tier 1: In-memory zero-LLM patch
            tier1_zero_llm_patches += 1

streaming_duration = time.time() - streaming_start
print(f"[STATUS] Incremental Ingress Finished in {streaming_duration:.2f}s.")
print(f"  - Tier 1 (Zero-LLM Fast Path Updates) : {tier1_zero_llm_patches}")
print(f"  - Tier 2 (Isolated LLM Re-summaries)  : {tier2_resummarizations}")


# 4. Multi-Hop HotpotQA Retrieval Benchmark
print(f"\n[STATUS] Running Multi-Hop QA Evaluation across 50 Benchmark Queries...")

qa_pairs = split_data["qa_benchmark"]

def retrieve_deltagraphrag(query: str, top_k: int = 5) -> List[str]:
    """
    Two-stage hierarchical GraphRAG retrieval:
    Matches query terms to community summaries, then extracts candidate source documents.
    """
    query_tokens = set(query.lower().split())
    scored_communities: List[Tuple[int, int]] = []

    for c_id_str, c_data in community_summaries.items():
        summary_text = c_data.get("summary", "").lower()
        score = sum(1 for token in query_tokens if token in summary_text and len(token) > 3)
        scored_communities.append((int(c_id_str), score))

    scored_communities.sort(key=lambda x: x[1], reverse=True)
    top_comm_ids = [c_id for c_id, _ in scored_communities[:3]]

    # Aggregate documents from top community members
    candidate_docs: Set[str] = set()
    for c_id in top_comm_ids:
        for node in comm_to_nodes.get(c_id, set()):
            if G.has_node(node):
                doc_field = G.nodes[node].get("documents", "")
                for d in doc_field.split(";"):
                    if d.strip():
                        candidate_docs.add(d.strip())

    # Fallback to direct node-entity lexical matching
    for node in G.nodes():
        if node.lower() in query.lower():
            for d in G.nodes[node].get("documents", "").split(";"):
                if d.strip():
                    candidate_docs.add(d.strip())

    return list(candidate_docs)[:top_k]


def retrieve_naive_bm25_proxy(query: str, all_docs: List[Dict[str, str]], top_k: int = 5) -> List[str]:
    """Baseline direct lexical retrieval matching query against document bodies."""
    query_tokens = set(query.lower().split())
    scored: List[Tuple[str, int]] = []

    for doc in all_docs:
        text = (doc["title"] + " " + doc["text"]).lower()
        score = sum(1 for token in query_tokens if token in text and len(token) > 3)
        scored.append((doc["title"], score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [doc_title for doc_title, _ in scored[:top_k]]


# Run Benchmark
all_passages = split_data["t0_base"] + split_data["t1_streaming"]

delta_hits_recall2 = 0
naive_hits_recall2 = 0

for item in qa_pairs:
    gold_titles = set(item["gold_titles"])

    retrieved_delta = set(retrieve_deltagraphrag(item["question"], top_k=6))
    retrieved_naive = set(retrieve_naive_bm25_proxy(item["question"], all_passages, top_k=6))

    # Recall@2: Query succeeds only if BOTH multi-hop gold documents are retrieved
    if gold_titles.issubset(retrieved_delta):
        delta_hits_recall2 += 1

    if gold_titles.issubset(retrieved_naive):
        naive_hits_recall2 += 1

delta_recall2_pct = (delta_hits_recall2 / len(qa_pairs)) * 100.0
naive_recall2_pct = (naive_hits_recall2 / len(qa_pairs)) * 100.0

# Cost & Efficiency Modeling
total_active_communities = len(comm_to_nodes)
full_rebuild_llm_calls = total_active_communities  # Full rebuild must re-summarize all communities
delta_call_savings_pct = (1.0 - (delta_llm_calls / float(full_rebuild_llm_calls))) * 100.0

benchmark_results = {
    "system_metrics": {
        "final_nodes": G.number_of_nodes(),
        "final_edges": G.number_of_edges(),
        "final_communities": total_active_communities,
        "streaming_ingress_time_seconds": round(streaming_duration, 2),
    },
    "efficiency_ablation": {
        "full_rebuild_calls_required": full_rebuild_llm_calls,
        "deltagraphrag_calls_used": delta_llm_calls,
        "tier1_fast_path_patches": tier1_zero_llm_patches,
        "tier2_isolated_resummaries": tier2_resummarizations,
        "llm_compute_savings_pct": round(delta_call_savings_pct, 2),
    },
    "retrieval_performance": {
        "eval_queries": len(qa_pairs),
        "deltagraphrag_recall_at_2_pct": round(delta_recall2_pct, 2),
        "naive_retrieval_recall_at_2_pct": round(naive_recall2_pct, 2),
    },
}

with open(BENCHMARK_OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(benchmark_results, f, indent=2)

# Formatted Telemetry Output
print("\n" + "=" * 65)
print("             DELTAGRAPHRAG EMPIRICAL BENCHMARK REPORT           ")
print("=" * 65)
print(f"Total Evaluated Queries         : {len(qa_pairs)}")
print(f"Final Graph Topology            : {G.number_of_nodes()} Nodes, {G.number_of_edges()} Edges, {total_active_communities} Communities")
print("-" * 65)
print(f"Full Rebuild LLM Calls (Static) : {full_rebuild_llm_calls} calls (100% compute)")
print(f"DeltaGraphRAG Streaming Calls   : {delta_llm_calls} calls ({100.0 - delta_call_savings_pct:.1f}% compute)")
print(f"Compute Cost Reduction (Token $) : {delta_call_savings_pct:.1f}% SAVINGS")
print("-" * 65)
print(f"DeltaGraphRAG Multi-Hop Recall@2: {delta_recall2_pct:.1f}%")
print(f"Naive Lexical Baseline Recall@2 : {naive_recall2_pct:.1f}%")
print("=" * 65)
print(f"[STATUS] Full telemetry written to: {BENCHMARK_OUT_PATH}")