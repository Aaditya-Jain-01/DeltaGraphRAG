import os
import json
import time
from typing import Dict, List, Any
import networkx as nx
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

WORKSPACE_DIR = "data"
SPLIT_PATH = os.path.join(WORKSPACE_DIR, "dataset_split.json")
CACHE_PATH = os.path.join(WORKSPACE_DIR, "extractions_cache.json")
CONFIG_PATH = os.path.join(WORKSPACE_DIR, "model_config.json")

GRAPH_OUT_PATH = os.path.join(WORKSPACE_DIR, "base_graph.json")
GRAPHML_OUT_PATH = os.path.join(WORKSPACE_DIR, "base_graph.graphml")
SUMMARIES_OUT_PATH = os.path.join(WORKSPACE_DIR, "base_summaries.json")
COMMUNITIES_OUT_PATH = os.path.join(WORKSPACE_DIR, "base_communities.json")

# 1. Load artifacts and models
with open(SPLIT_PATH, "r", encoding="utf-8") as f:
    split_data = json.load(f)

with open(CACHE_PATH, "r", encoding="utf-8") as f:
    extractions = json.load(f)

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    model_cfg = json.load(f)

api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=api_key)

summary_model = model_cfg.get("heavy_model", "openai/gpt-oss-120b")
print(f"[STATUS] Summarization Model: {summary_model}")

# 2. Filter extractions to T0 base corpus (80 documents)
t0_titles = {doc["title"] for doc in split_data["t0_base"]}
t0_extractions = [data for data in extractions.values() if data["title"] in t0_titles]
print(f"[STATUS] Isolated {len(t0_extractions)} T0 extractions for base graph construction.")

# 3. Construct base knowledge graph
G = nx.Graph()

for item in t0_extractions:
    doc_title = item["title"]
    
    # Ingest entities (nodes)
    for entity in item.get("entities", []):
        name = entity.get("name", "").strip()
        e_type = entity.get("type", "Concept").strip()
        if not name:
            continue
        if not G.has_node(name):
            G.add_node(name, entity_type=e_type, documents=doc_title)
        else:
            existing_docs = G.nodes[name].get("documents", "")
            if doc_title not in existing_docs:
                G.nodes[name]["documents"] = f"{existing_docs}; {doc_title}"

    # Ingest relations (edges)
    for rel in item.get("relations", []):
        src = rel.get("source", "").strip()
        tgt = rel.get("target", "").strip()
        label = rel.get("relation", "related_to").strip()
        
        if not src or not tgt or src == tgt:
            continue
        
        if not G.has_node(src):
            G.add_node(src, entity_type="Concept", documents=doc_title)
        if not G.has_node(tgt):
            G.add_node(tgt, entity_type="Concept", documents=doc_title)
            
        if G.has_edge(src, tgt):
            G[src][tgt]["weight"] += 1
            existing_rels = G[src][tgt].get("relations", "")
            if label not in existing_rels:
                G[src][tgt]["relations"] = f"{existing_rels}, {label}"
        else:
            G.add_edge(src, tgt, weight=1, relations=label)

print(f"[STATUS] Base Graph Topology Built: {G.number_of_nodes()} Nodes, {G.number_of_edges()} Edges")

# 4. Partition graph using Louvain community detection
raw_communities = nx.community.louvain_communities(G, seed=42)
communities: List[List[str]] = [sorted(list(c)) for c in raw_communities]

node_to_comm: Dict[str, int] = {}
for comm_id, comm_nodes in enumerate(communities):
    for node in comm_nodes:
        node_to_comm[node] = comm_id
        G.nodes[node]["community"] = comm_id

print(f"[STATUS] Detected {len(communities)} Louvain communities (Modularity Q: {nx.community.modularity(G, raw_communities):.4f})")

# 5. Generate hierarchical community summaries
SUMMARY_PROMPT = """You are a knowledge graph synthesizer.
Below is an entity cluster representing an interconnected community in the knowledge graph.
Synthesize a concise 2-3 sentence semantic summary describing the core theme, key entities, and significant relationships governing this cluster.

Entity Cluster Information:
{cluster_details}

Summary:"""


def summarize_cluster(cluster_info: str, max_retries: int = 5) -> str:
    delay = 2.0
    for attempt in range(max_retries):
        try:
            res = client.chat.completions.create(
                model=summary_model,
                messages=[{"role": "user", "content": SUMMARY_PROMPT.format(cluster_details=cluster_info)}],
                temperature=0.1,
                max_tokens=300,
            )
            return res.choices[0].message.content.strip()
        except Exception as exc:
            err = str(exc).lower()
            if "429" in err or "rate limit" in err:
                print(f"  [RATE-LIMIT] Backing off for {delay:.1f}s...")
                time.sleep(delay)
                delay *= 2.0
            else:
                time.sleep(1.0)
    return "Summary generation unavailable."


community_summaries: Dict[str, Dict[str, Any]] = {}
print(f"\n[STATUS] Synthesizing summaries across {len(communities)} communities...")

for comm_id, members in enumerate(communities):
    subgraph = G.subgraph(members)
    edge_descriptions = [
        f"{u} -> {v} ({d.get('relations', 'connected')})"
        for u, v, d in subgraph.edges(data=True)
    ][:15]
    
    cluster_payload = f"Community ID: {comm_id}\n"
    cluster_payload += f"Core Nodes ({len(members)}): {', '.join(members[:20])}\n"
    cluster_payload += f"Key Relationships: {'; '.join(edge_descriptions) if edge_descriptions else 'Isolated cluster nodes'}"
    
    summary_text = summarize_cluster(cluster_payload)
    community_summaries[str(comm_id)] = {
        "community_id": comm_id,
        "size": len(members),
        "members": members,
        "summary": summary_text
    }
    
    print(f"  ✓ Community {comm_id:02d} ({len(members):2d} entities) synthesized.")
    time.sleep(1.2)

# 6. Checkpoint persistence
graph_json = nx.node_link_data(G)
with open(GRAPH_OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(graph_json, f, indent=2)

nx.write_graphml(G, GRAPHML_OUT_PATH)

with open(COMMUNITIES_OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(node_to_comm, f, indent=2)

with open(SUMMARIES_OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(community_summaries, f, indent=2)

print("\n" + "-" * 55)
print(f"Base Graph Saved      : {GRAPH_OUT_PATH}")
print(f"GraphML Export Saved  : {GRAPHML_OUT_PATH}")
print(f"Community Map Saved   : {COMMUNITIES_OUT_PATH}")
print(f"Summaries Saved       : {SUMMARIES_OUT_PATH}")
print("-" * 55)