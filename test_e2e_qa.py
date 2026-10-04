import os
import json
import networkx as nx
from dotenv import load_dotenv
from groq import Groq
from src.graph_engine import ModularityEngine

load_dotenv()

# Verify src package integrity
assert hasattr(ModularityEngine, "compute_delta_q"), "Failed to import src.graph_engine!"
print("[STATUS] src/ modules verified successfully.")

WORKSPACE_DIR = "data"
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# 1. Load runtime assets
with open(os.path.join(WORKSPACE_DIR, "dataset_split.json"), "r", encoding="utf-8") as f:
    split_data = json.load(f)

with open(os.path.join(WORKSPACE_DIR, "base_graph.json"), "r", encoding="utf-8") as f:
    graph_data = json.load(f)

with open(os.path.join(WORKSPACE_DIR, "base_communities.json"), "r", encoding="utf-8") as f:
    node_to_comm = json.load(f)

with open(os.path.join(WORKSPACE_DIR, "base_summaries.json"), "r", encoding="utf-8") as f:
    summaries = json.load(f)

# Choose verified dense model for direct Q&A
available_models = [m.id for m in client.models.list().data]
if "qwen/qwen3.8-27b" in available_models:
    model_id = "qwen/qwen3.8-27b"
else:
    model_id = "openai/gpt-oss-120b"

print(f"[STATUS] Inference Engine Active: {model_id}")

G = nx.node_link_graph(graph_data)

# Reverse index: Community ID -> Set of node names
comm_to_nodes = {}
for n, c in node_to_comm.items():
    comm_to_nodes.setdefault(c, set()).add(n)


def retrieve_graph_context(query: str, top_k_comm: int = 3) -> str:
    """Extracts top matching community summaries and connected node relationships."""
    tokens = set(query.lower().split())

    scored = []
    for c_id, data in summaries.items():
        summary_text = data.get("summary", "").lower()
        score = sum(1 for t in tokens if t in summary_text and len(t) > 3)
        scored.append((c_id, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    top_comms = [c_id for c_id, _ in scored[:top_k_comm]]

    context_chunks = []
    for c_id in top_comms:
        c_summary = summaries[c_id]["summary"]
        context_chunks.append(f"[Community {c_id}] {c_summary}")

        members = comm_to_nodes.get(int(c_id), set())
        subgraph = G.subgraph(members)
        triples = [
            f"{u} -> {v} ({d.get('relations', 'related')})"
            for u, v, d in list(subgraph.edges(data=True))[:6]
        ]
        if triples:
            context_chunks.append(f"  Facts: {'; '.join(triples)}")

    return "\n\n".join(context_chunks)


def answer_query(query: str, context: str) -> str:
    """Generates precise answer grounded on the extracted graph sub-network."""
    prompt = f"""You are a knowledge graph question answering engine.
Answer the question concisely using the provided graph context. If the answer is directly supported, state it in one short sentence.

Context:
{context}

Question: {query}
Answer:"""

    response = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=1024,
    )
    raw_content = response.choices[0].message.content or ""
    return raw_content.strip()


# 2. Run inference on benchmark multi-hop questions
sample_queries = split_data["qa_benchmark"][:3]

print(f"\n[STATUS] Running End-to-End QA Inference...\n")
print("=" * 70)

for idx, item in enumerate(sample_queries, start=1):
    q = item["question"]
    gold = item["answer"]

    context = retrieve_graph_context(q)
    predicted = answer_query(q, context)

    print(f"Test Query {idx}:")
    print(f"  Question : {q}")
    print(f"  Gold     : {gold}")
    print(f"  Predicted: {predicted}")
    print("-" * 70)