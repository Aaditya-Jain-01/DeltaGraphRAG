import os
import re
import json
import time
import string
from collections import Counter
from typing import Dict, List, Set, Tuple
import networkx as nx
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

WORKSPACE_DIR = "data"
METRICS_OUT_PATH = os.path.join(WORKSPACE_DIR, "generation_metrics.json")

# 1. Load data artifacts
with open(os.path.join(WORKSPACE_DIR, "dataset_split.json"), "r", encoding="utf-8") as f:
    split_data = json.load(f)

with open(os.path.join(WORKSPACE_DIR, "base_graph.json"), "r", encoding="utf-8") as f:
    graph_data = json.load(f)

with open(os.path.join(WORKSPACE_DIR, "base_communities.json"), "r", encoding="utf-8") as f:
    node_to_comm: Dict[str, int] = json.load(f)

with open(os.path.join(WORKSPACE_DIR, "base_summaries.json"), "r", encoding="utf-8") as f:
    summaries: Dict[str, Dict] = json.load(f)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Select fast, non-truncating dense model for direct Q&A
available_models = [m.id for m in client.models.list().data]
model_id = "qwen/qwen3.8-27b" if "qwen/qwen3.8-27b" in available_models else "openai/gpt-oss-120b"
print(f"[STATUS] Initializing evaluation engine on {model_id}...")

G: nx.Graph = nx.node_link_graph(graph_data)

comm_to_nodes: Dict[int, Set[str]] = {}
for n, c in node_to_comm.items():
    comm_to_nodes.setdefault(c, set()).add(n)

# 2. Standard HotpotQA Metric Helpers
def normalize_answer(s: str) -> str:
    """Lowercases, removes punctuation, articles, and extra whitespace."""
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def compute_exact_match(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def compute_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()

    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(gold_tokens)
    return (2 * precision * recall) / (precision + recall)


# 3. Context Retrieval & Inference Engine
def retrieve_graph_context(query: str, top_k_comm: int = 3) -> str:
    query_tokens = set(query.lower().split())

    scored = []
    for c_id, data in summaries.items():
        summary_text = data.get("summary", "").lower()
        score = sum(1 for t in query_tokens if t in summary_text and len(t) > 3)
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


def answer_query(query: str, context: str, max_retries: int = 3) -> str:
    prompt = f"""You are an exact multi-hop question answering engine.
Answer the question directly, factually, and as briefly as possible using ONLY the context provided.
Do not provide introductory filler or conversational preamble.

Context:
{context}

Question: {query}
Direct Answer:"""

    for attempt in range(max_retries):
        try:
            res = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=256,
            )
            raw = res.choices[0].message.content or ""
            return raw.strip()
        except Exception:
            time.sleep(2.0)
    return ""


# 4. Execute Benchmark across all 50 Queries
qa_items = split_data["qa_benchmark"]
results_log = []

total_em = 0.0
total_f1 = 0.0
start_time = time.time()

print(f"\n[STATUS] Starting evaluation across all {len(qa_items)} queries...\n")

for idx, item in enumerate(qa_items, start=1):
    q = item["question"]
    gold = item["answer"]

    context = retrieve_graph_context(q)
    prediction = answer_query(q, context)

    em = compute_exact_match(prediction, gold)
    f1 = compute_f1(prediction, gold)

    total_em += em
    total_f1 += f1

    results_log.append({
        "id": item["id"],
        "question": q,
        "gold": gold,
        "prediction": prediction,
        "exact_match": em,
        "f1": f1
    })

    print(f"[{idx:02d}/{len(qa_items)}] EM: {em:.0f} | F1: {f1:.2f} | Gold: '{gold}' | Pred: '{prediction}'")
    time.sleep(0.5)

mean_em = (total_em / len(qa_items)) * 100.0
mean_f1 = (total_f1 / len(qa_items)) * 100.0
elapsed = time.time() - start_time

payload = {
    "model_evaluated": model_id,
    "total_queries": len(qa_items),
    "exact_match_pct": round(mean_em, 2),
    "f1_score_pct": round(mean_f1, 2),
    "runtime_seconds": round(elapsed, 2),
    "predictions": results_log
}

with open(METRICS_OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2)

print("\n" + "=" * 60)
print("       DELTAGRAPHRAG GENERATION EVALUATION REPORT       ")
print("=" * 60)
print(f"Total Evaluated Queries : {len(qa_items)}")
print(f"Model Evaluated         : {model_id}")
print(f"Evaluation Runtime      : {elapsed:.2f}s")
print("-" * 60)
print(f"Exact Match (EM)        : {mean_em:.2f}%")
print(f"Mean Token F1           : {mean_f1:.2f}%")
print("=" * 60)
print(f"[STATUS] Full evaluation log written to: {METRICS_OUT_PATH}")