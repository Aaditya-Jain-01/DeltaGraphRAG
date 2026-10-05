import os
import re
import sys
import json
import time
import string
from collections import Counter, defaultdict
import networkx as nx
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

WORKSPACE_DIR = "data"
SPLIT_PATH = os.path.join(WORKSPACE_DIR, "dataset_split.json")
INCR_GRAPH_PATH = os.path.join(WORKSPACE_DIR, "incremental_graph.json")
INCR_COMM_PATH = os.path.join(WORKSPACE_DIR, "incremental_communities.json")
INCR_SUMM_PATH = os.path.join(WORKSPACE_DIR, "incremental_summaries.json")
REBUILD_GRAPH_PATH = os.path.join(WORKSPACE_DIR, "rebuild_graph.json")
REBUILD_COMM_PATH = os.path.join(WORKSPACE_DIR, "rebuild_communities.json")
REBUILD_SUMM_PATH = os.path.join(WORKSPACE_DIR, "rebuild_summaries.json")
OUTPUT_PATH = os.path.join(WORKSPACE_DIR, "generation_metrics.json")
CONFIG_PATH = os.path.join(WORKSPACE_DIR, "model_config.json")

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("Missing GROQ_API_KEY in .env file.")

client = Groq(api_key=api_key)

eval_model_id = "qwen/qwen3.8-27b"
synth_model_id = "openai/gpt-oss-120b"
if os.path.exists(CONFIG_PATH):
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)
            eval_model_id = cfg.get("eval_model") or cfg.get("query_model") or eval_model_id
            synth_model_id = cfg.get("fast_model") or synth_model_id
    except Exception:
        pass

def normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    return " ".join(s.split())

def compute_f1(gold: str, pred: str) -> float:
    gold_toks = normalize_text(gold).split()
    pred_toks = normalize_text(pred).split()
    if not gold_toks or not pred_toks:
        return 1.0 if gold_toks == pred_toks else 0.0
    common = set(gold_toks) & set(pred_toks)
    if not common:
        return 0.0
    prec = len(common) / len(pred_toks)
    rec = len(common) / len(gold_toks)
    return (2 * prec * rec) / (prec + rec)

def compute_em(gold: str, pred: str) -> float:
    return 1.0 if normalize_text(gold) == normalize_text(pred) else 0.0

def extract_summary_text(val) -> str:
    if isinstance(val, str):
        return val.strip()
    if isinstance(val, dict):
        for k in ["summary", "description", "content", "text", "body"]:
            if k in val and isinstance(val[k], str):
                return val[k].strip()
        return " ".join(str(v).strip() for v in val.values() if isinstance(v, (str, int, float))).strip()
    return str(val).strip()

def synthesize_rebuild_community(member_nodes: list, graph: nx.Graph) -> str:
    """Synthesizes an independent summary directly from the full-rebuild graph."""
    subgraph = graph.subgraph(member_nodes)
    edges_desc = [f"{u} connected to {v}" for u, v in list(subgraph.edges())[:20]]
    context_str = f"Members: {', '.join(member_nodes[:20])}\nConnections:\n" + "\n".join(edges_desc)
    prompt = (
        f"You are a knowledge graph synthesizer. Provide a concise, factual summary (2-3 sentences) "
        f"characterizing the core theme, entities, and primary connections in this community cluster:\n\n{context_str}"
    )
    try:
        resp = client.chat.completions.create(
            model=synth_model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=256
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return f"Community containing {', '.join(member_nodes[:10])}."

def retrieve_context_and_patch(query: str, graph: nx.Graph, node_to_comm: dict, summaries: dict, 
                               comm_to_nodes: dict, is_rebuild: bool = False, top_k: int = 2) -> tuple:
    query_tokens = set(normalize_text(query).split())
    scored_nodes = []
    for node in graph.nodes():
        node_toks = set(normalize_text(str(node)).split())
        overlap = len(query_tokens & node_toks)
        if overlap > 0:
            scored_nodes.append((node, overlap))
    
    scored_nodes.sort(key=lambda x: x[1], reverse=True)
    top_nodes = [n for n, _ in scored_nodes[:10]]
    
    target_comms = Counter()
    for n in top_nodes:
        c = node_to_comm.get(n)
        if c is not None:
            target_comms[str(c)] += 1
            
    top_comm_ids = [c for c, _ in target_comms.most_common(top_k)]
    
    contexts = []
    for cid in top_comm_ids:
        c_str = str(cid)
        # Synthesize independent summary for full rebuild if not already cached
        if is_rebuild and c_str not in summaries and cid not in summaries:
            members = comm_to_nodes.get(c_str, []) or comm_to_nodes.get(cid, [])
            if members:
                summ_text = synthesize_rebuild_community(members, graph)
                summaries[c_str] = summ_text
                time.sleep(0.3)
        
        raw = summaries.get(c_str) or summaries.get(cid)
        if raw is not None:
            text = extract_summary_text(raw)
            if text:
                contexts.append(text)
            
    combined = "\n\n".join(contexts) if contexts else "No relevant graph context found."
    return combined, set(top_comm_ids)

def call_eval_llm(prompt: str) -> str:
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=eval_model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=64
            )
            msg = resp.choices[0].message
            content = msg.content or getattr(msg, "reasoning", "") or ""
            if content.strip():
                return content.strip()
            time.sleep(1.0)
        except Exception as e:
            if "429" in str(e) or "rate_limit" in str(e).lower():
                time.sleep(2.5 * (attempt + 1))
                continue
            time.sleep(1.0)
    return ""

def load_evaluation_queries() -> list:
    queries = []
    seen = set()
    def add_pair(q, a):
        if q and a and isinstance(q, str) and isinstance(a, str):
            norm_q = q.strip().lower()
            if norm_q not in seen:
                seen.add(norm_q)
                queries.append({"question": q.strip(), "answer": a.strip()})

    if os.path.exists(SPLIT_PATH):
        try:
            with open(SPLIT_PATH, "r", encoding="utf-8") as f:
                s_data = json.load(f)
            if isinstance(s_data, dict):
                for k, v in s_data.items():
                    if isinstance(v, list):
                        for item in v:
                            if isinstance(item, dict):
                                add_pair(item.get("question"), item.get("answer"))
                                for sub_key in ["questions", "qa", "qa_pairs"]:
                                    if sub_key in item and isinstance(item[sub_key], list):
                                        for sub in item[sub_key]:
                                            if isinstance(sub, dict):
                                                add_pair(sub.get("question"), sub.get("answer"))
        except Exception:
            pass

    if len(queries) < 50 and os.path.exists(OUTPUT_PATH):
        try:
            with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
                g_data = json.load(f)
            for p in g_data.get("predictions", []):
                if isinstance(p, dict):
                    add_pair(p.get("query"), p.get("gold"))
        except Exception:
            pass

    return queries[:50]

print(f"[STATUS] Initializing A/B Evaluation Suite (Eval Engine: {eval_model_id})...")

# Load Pipeline A: Incremental Graph
with open(INCR_GRAPH_PATH, "r", encoding="utf-8") as f:
    G_incr = nx.node_link_graph(json.load(f))
with open(INCR_COMM_PATH, "r", encoding="utf-8") as f:
    comm_incr = json.load(f)
with open(INCR_SUMM_PATH, "r", encoding="utf-8") as f:
    summ_incr = json.load(f)

comm_to_nodes_incr = defaultdict(list)
for n, c in comm_incr.items():
    comm_to_nodes_incr[str(c)].append(n)

# Load Pipeline B: Full Rebuild Graph
with open(REBUILD_GRAPH_PATH, "r", encoding="utf-8") as f:
    G_rebuild = nx.node_link_graph(json.load(f))
with open(REBUILD_COMM_PATH, "r", encoding="utf-8") as f:
    comm_rebuild = json.load(f)

# Load existing rebuild summaries or start fresh
summ_rebuild = {}
if os.path.exists(REBUILD_SUMM_PATH):
    try:
        with open(REBUILD_SUMM_PATH, "r", encoding="utf-8") as f:
            summ_rebuild = json.load(f)
    except Exception:
        summ_rebuild = {}

comm_to_nodes_rebuild = defaultdict(list)
for n, c in comm_rebuild.items():
    comm_to_nodes_rebuild[str(c)].append(n)

eval_queries = load_evaluation_queries()
total_queries = len(eval_queries)
print(f"[STATUS] Benchmarking {total_queries} queries across Incremental vs. Full Rebuild pipelines...\n")

results = []
jaccard_scores = []
identical_predictions = 0
t_start = time.time()

for idx, q_item in enumerate(eval_queries, 1):
    q_text = q_item["question"]
    gold = q_item["answer"]
    
    # 1. Retrieve under Pipeline A (Incremental: base summaries with Tier 1 in-memory patches)
    ctx_incr, _ = retrieve_context_and_patch(
        q_text, G_incr, comm_incr, summ_incr, comm_to_nodes_incr, is_rebuild=False
    )
    
    # 2. Retrieve under Pipeline B (Full Rebuild: independent summaries synthesized from G_ref)
    ctx_rebuild, _ = retrieve_context_and_patch(
        q_text, G_rebuild, comm_rebuild, summ_rebuild, comm_to_nodes_rebuild, is_rebuild=True
    )
    
    # Context Text Jaccard
    t_inc = set(normalize_text(ctx_incr).split())
    t_reb = set(normalize_text(ctx_rebuild).split())
    union_t = len(t_inc | t_reb)
    jaccard = (len(t_inc & t_reb) / union_t) if union_t > 0 else 1.0
    jaccard_scores.append(jaccard)
    
    # Inference for Incremental
    prompt_incr = f"Answer the following question based ONLY on the provided graph context. Keep the answer concise.\n\nContext:\n{ctx_incr}\n\nQuestion: {q_text}\nAnswer:"
    pred_incr = call_eval_llm(prompt_incr)
    time.sleep(0.1)
    
    # Inference for Full Rebuild
    if ctx_incr == ctx_rebuild:
        pred_rebuild = pred_incr
    else:
        prompt_reb = f"Answer the following question based ONLY on the provided graph context. Keep the answer concise.\n\nContext:\n{ctx_rebuild}\n\nQuestion: {q_text}\nAnswer:"
        pred_rebuild = call_eval_llm(prompt_reb)
        time.sleep(0.1)
        
    if pred_incr == pred_rebuild:
        identical_predictions += 1
        
    em_incr = compute_em(gold, pred_incr)
    f1_incr = compute_f1(gold, pred_incr)
    em_rebuild = compute_em(gold, pred_rebuild)
    f1_rebuild = compute_f1(gold, pred_rebuild)
    
    results.append({
        "query": q_text,
        "gold": gold,
        "incremental": {"prediction": pred_incr, "em": em_incr, "f1": f1_incr},
        "full_rebuild": {"prediction": pred_rebuild, "em": em_rebuild, "f1": f1_rebuild},
        "context_jaccard": jaccard
    })
    
    print(f"[{idx:02d}/{total_queries}] EM_inc: {int(em_incr)} | EM_reb: {int(em_rebuild)} | F1_inc: {f1_incr:.2f} | F1_reb: {f1_rebuild:.2f}")

# Persist synthesized rebuild summaries for reproducibility
with open(REBUILD_SUMM_PATH, "w", encoding="utf-8") as f:
    json.dump(summ_rebuild, f, indent=2)

total_eval = len(results)
mean_em_incr = sum(r["incremental"]["em"] for r in results) / total_eval
mean_f1_incr = sum(r["incremental"]["f1"] for r in results) / total_eval
mean_em_rebuild = sum(r["full_rebuild"]["em"] for r in results) / total_eval
mean_f1_rebuild = sum(r["full_rebuild"]["f1"] for r in results) / total_eval
mean_jaccard = sum(jaccard_scores) / total_eval
agreement_pct = (identical_predictions / total_eval) * 100.0

out_payload = {
    "total_queries": total_eval,
    "model_evaluated": eval_model_id,
    "evaluation_runtime_sec": round(time.time() - t_start, 2),
    "exact_match_pct": round(mean_em_incr * 100, 2),
    "f1_score_pct": round(mean_f1_incr * 100, 2),
    "ab_comparison": {
        "context_jaccard_parity_pct": round(mean_jaccard * 100, 2),
        "prediction_agreement_pct": round(agreement_pct, 2),
        "incremental_em_pct": round(mean_em_incr * 100, 2),
        "full_rebuild_em_pct": round(mean_em_rebuild * 100, 2),
        "incremental_f1_pct": round(mean_f1_incr * 100, 2),
        "full_rebuild_f1_pct": round(mean_f1_rebuild * 100, 2),
        "quality_degradation_pct": round(abs(mean_f1_rebuild - mean_f1_incr) * 100, 2)
    },
    "predictions": [
        {
            "query": r["query"],
            "gold": r["gold"],
            "prediction": r["incremental"]["prediction"],
            "exact_match": r["incremental"]["em"],
            "f1": r["incremental"]["f1"]
        }
        for r in results
    ]
}

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(out_payload, f, indent=2)

print("\n" + "=" * 65)
print("     DELTAGRAPHRAG vs. FULL REBUILD A/B EVALUATION REPORT     ")
print("=" * 65)
print(f"Total Evaluated Queries         : {total_eval}")
print(f"Model Evaluated                 : {eval_model_id}")
print(f"Retrieval Context Jaccard Parity: {mean_jaccard * 100:.2f}%")
print(f"Answer Agreement Rate           : {agreement_pct:.2f}% ({identical_predictions}/{total_eval} identical)")
print("-" * 65)
print(f"Full Rebuild Reference Exact Match : {mean_em_rebuild * 100:.2f}%")
print(f"DeltaGraphRAG Exact Match          : {mean_em_incr * 100:.2f}% (Delta: {(mean_em_incr - mean_em_rebuild) * 100:+.2f}%)")
print(f"Full Rebuild Reference Mean F1     : {mean_f1_rebuild * 100:.2f}%")
print(f"DeltaGraphRAG Mean Token F1        : {mean_f1_incr * 100:.2f}% (Delta: {(mean_f1_incr - mean_f1_rebuild) * 100:+.2f}%)")
print(f"Relative Quality Degradation       : {abs(mean_f1_rebuild - mean_f1_incr) * 100:.2f}%")
print("=" * 65)
print(f"[STATUS] Live A/B parity report written to: {OUTPUT_PATH}")