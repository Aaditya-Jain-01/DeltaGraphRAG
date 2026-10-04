import os
import sys
import json
import argparse
import networkx as nx
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

WORKSPACE_DIR = "data"
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("Missing GROQ_API_KEY in .env")

client = Groq(api_key=api_key)

# Load graph topology and community structures
with open(os.path.join(WORKSPACE_DIR, "base_graph.json"), "r", encoding="utf-8") as f:
    graph_data = json.load(f)
G = nx.node_link_graph(graph_data)

with open(os.path.join(WORKSPACE_DIR, "base_communities.json"), "r", encoding="utf-8") as f:
    node_to_comm = json.load(f)

with open(os.path.join(WORKSPACE_DIR, "base_summaries.json"), "r", encoding="utf-8") as f:
    summaries = json.load(f)

comm_to_nodes = {}
for n, c in node_to_comm.items():
    comm_to_nodes.setdefault(c, set()).add(n)


def query_graph(question: str, top_k: int = 3) -> str:
    tokens = set(question.lower().split())
    
    # 1. Rank communities by keyword overlap
    scored = []
    for c_id, data in summaries.items():
        summary_text = data.get("summary", "").lower()
        score = sum(1 for t in tokens if t in summary_text and len(t) > 3)
        scored.append((c_id, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    top_comms = [c_id for c_id, _ in scored[:top_k]]

    # 2. Extract facts and summaries
    context_chunks = []
    for c_id in top_comms:
        c_summary = summaries[c_id].get("summary", "")
        context_chunks.append(f"[Community {c_id}] {c_summary}")
        members = comm_to_nodes.get(int(c_id), set())
        subgraph = G.subgraph(members)
        triples = [
            f"{u} -> {v} ({d.get('relations', 'related')})"
            for u, v, d in list(subgraph.edges(data=True))[:5]
        ]
        if triples:
            context_chunks.append(f"  Facts: {'; '.join(triples)}")

    context = "\n\n".join(context_chunks)

    # 3. Direct synthesis
    prompt = f"""You are a multi-hop knowledge graph reasoning assistant.
Answer the user's question concisely using ONLY the provided Knowledge Graph context.

Context:
{context}

Question: {question}
Answer:"""

    response = client.chat.completions.create(
        model="qwen/qwen3.8-27b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=256,
    )
    return response.choices[0].message.content.strip()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query DeltaGraphRAG Knowledge Graph")
    parser.add_argument("--question", type=str, help="Question to ask the graph")
    args = parser.parse_args()

    q = args.question or input("\nEnter question: ")
    print(f"\n[QUERY] {q}")
    ans = query_graph(q)
    print(f"[ANSWER] {ans}\n")