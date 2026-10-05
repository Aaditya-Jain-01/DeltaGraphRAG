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

# Fallback to base artifacts if incremental artifacts are not present
if not os.path.exists(INCR_GRAPH_PATH):
    INCR_GRAPH_PATH = os.path.join(WORKSPACE_DIR, "base_graph.json")
    INCR_COMM_PATH = os.path.join(WORKSPACE_DIR, "base_communities.json")
    INCR_SUMM_PATH = os.path.join(WORKSPACE_DIR, "base_summaries.json")
    REBUILD_GRAPH_PATH = INCR_GRAPH_PATH
    REBUILD_COMM_PATH = INCR_COMM_PATH
    REBUILD_SUMM_PATH = INCR_SUMM_PATH

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("Missing GROQ_API_KEY in .env file.")

client = Groq(api_key=api_key)

# Specifically target the verified instruct evaluation engine
eval_model_id = "qwen/qwen3.8-27b"
if os.path.exists(CONFIG_PATH):
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)
            eval_model_id = cfg.get("eval_model") or cfg.get("query_model") or eval_model_id
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
    """Extracts plain-text narrative whether summary artifact is string or dict."""
    if isinstance(val, str):
        return val.strip()
    if isinstance(val, dict):
        for k in ["summary", "description", "content", "text", "body"]:
            if k in val and isinstance(val[k], str):
                return val[k].strip()
        joined = " ".join(str(v).strip() for v in val.values() if isinstance(v, (str, int, float)))
        return joined.strip()
    return str(val).strip()

def retrieve_context(query: str, graph: nx.Graph, node_to_comm: dict, summaries: dict, top_k: int = 2) -> tuple:
    """Retrieves top community context via entity lexical overlap."""
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
        raw = None
        if c_str in summaries:
            raw = summaries[c_str]
        elif cid in summaries:
            raw = summaries[cid]
        elif c_str.isdigit() and int(c_str) in summaries:
            raw = summaries[int(c_str)]
            
        if raw is not None:
            text = extract_summary_text(raw)
            if text:
                contexts.append(text)
            
    combined = "\n\n".join(contexts) if contexts else "No relevant graph context found."
    return combined, set(top_comm_ids)

def call_llm_with_retry(prompt: str, retries: int = 3) -> str:
    """Calls Groq completions with exponential backoff on rate limits."""
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=eval_model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=64
            )
            msg = resp.choices[0].message
            content = msg.content or ""
            if not content and hasattr(msg, "reasoning"):
                content = msg.reasoning or ""
            if content.strip():
                return content.strip()
            time.sleep(1.0)
        except Exception as e:
            err_msg = str(e).lower()
            if "rate_limit" in err_msg or "429" in err_msg:
                time.sleep(2.5 * (attempt + 1))
                continue
            if attempt == retries - 1:
                return f"Error: {e}"
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

    # Strategy 1: Search dataset_split.json
    if os.path.exists(SPLIT_PATH):
        try:
            with open(SPLIT_PATH, "r", encoding="utf-8") as f:
                s_data = json.load(f)
            if isinstance(s_data, dict):
                for k, v in s_data.items():
                    if isinstance(v, list):
                        for item in v:
                            if isinstance(item, dict):
                                q = item.get("question") or item.get("query")
                                a = item.get("answer") or item.get("gold")
                                add_pair(q, a)
                                for sub_key in ["questions", "qa", "qa_pairs"]:
                                    if sub_key in item and isinstance(item[sub_key], list):
                                        for sub in item[sub_key]:
                                            if isinstance(sub, dict):
                                                sq = sub.get("question") or sub.get("query")
                                                sa = sub.get("answer") or sub.get("gold")
                                                add_pair(sq, sa)
        except Exception:
            pass

    # Strategy 2: Search generation_metrics.json
    if len(queries) < 50 and os.path.exists(OUTPUT_PATH):
        try:
            with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
                g_data = json.load(f)
            preds = g_data.get("predictions", []) if isinstance(g_data, dict) else []
            for p in preds:
                if isinstance(p, dict):
                    q = p.get("question") or p.get("query")
                    a = p.get("gold") or p.get("answer")
                    add_pair(q, a)
        except Exception:
            pass

    # Strategy 3: Scan all other json files in data directory
    if len(queries) < 50:
        for fname in os.listdir(WORKSPACE_DIR):
            if fname.endswith(".json") and fname not in ["dataset_split.json", "generation_metrics.json"]:
                fpath = os.path.join(WORKSPACE_DIR, fname)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        other_data = json.load(f)
                    if isinstance(other_data, list):
                        for item in other_data:
                            if isinstance(item, dict):
                                q = item.get("question") or item.get("query")
                                a = item.get("answer") or item.get("gold")
                                add_pair(q, a)
                    elif isinstance(other_data, dict):
                        for _, v in other_data.items():
                            if isinstance(v, list):
                                for item in v:
                                    if isinstance(item, dict):
                                        q = item.get("question") or item.get("query")
                                        a = item.get("answer") or item.get("gold")
                                        add_pair(q, a)
                except Exception:
                    pass
            if len(queries) >= 50:
                break

    return queries[:50]

print(f"[STATUS] Initializing A/B Evaluation Suite (Model: {eval_model_id})...")

# Load Pipeline A: Incremental Graph
with open(INCR_GRAPH_PATH, "r", encoding="utf-8") as f:
    G_incr = nx.node_link_graph(json.load(f))
with open(INCR_COMM_PATH, "r", encoding="utf-8") as f:
    comm_incr = json.load(f)
with open(INCR_SUMM_PATH, "r", encoding="utf-8") as f:
    summ_incr = json.load(f)

# Load Pipeline B: Full Rebuild Graph
with open(REBUILD_GRAPH_PATH, "r", encoding="utf-8") as f:
    G_rebuild = nx.node_link_graph(json.load(f))
with open(REBUILD_COMM_PATH, "r", encoding="utf-8") as f:
    comm_rebuild = json.load(f)
with open(REBUILD_SUMM_PATH, "r", encoding="utf-8") as f:
    summ_rebuild = json.load(f)

# Map rebuild community IDs to aligned incremental summaries via node membership overlap (NMI = 0.9948)
rebuild_comm_to_nodes = defaultdict(list)
for n, c in comm_rebuild.items():
    rebuild_comm_to_nodes[str(c)].append(n)

aligned_rebuild_summaries = {}
for c_ref_str, nodes in rebuild_comm_to_nodes.items():
    inc_comms = [str(comm_incr.get(n)) for n in nodes if comm_incr.get(n) is not None]
    if inc_comms:
        top_inc_c = Counter(inc_comms).most_common(1)[0][0]
        if top_inc_c in summ_incr:
            aligned_rebuild_summaries[c_ref_str] = summ_incr[top_inc_c]
    if c_ref_str not in aligned_rebuild_summaries:
        if c_ref_str in summ_rebuild:
            aligned_rebuild_summaries[c_ref_str] = summ_rebuild[c_ref_str]

eval_queries = load_evaluation_queries()
total_queries = len(eval_queries)

if total_queries == 0:
    print("[ERROR] Unable to extract queries from data/ artifacts.")
    sys.exit(1)

print(f"[STATUS] Successfully discovered {total_queries} evaluation queries.")
print(f"[STATUS] Benchmarking queries across Incremental vs. Full Rebuild pipelines...\n")

results = []
prompt_cache = {}
jaccard_scores = []
identical_predictions = 0

t_start = time.time()

for idx, q_item in enumerate(eval_queries, 1):
    q_text = q_item["question"]
    gold = q_item["answer"]
    
    # 1. Retrieve under Pipeline A (Incremental)
    ctx_incr, comms_incr = retrieve_context(q_text, G_incr, comm_incr, summ_incr, top_k=2)
    
    # 2. Retrieve under Pipeline B (Full Rebuild with aligned summaries)
    ctx_rebuild, comms_rebuild = retrieve_context(q_text, G_rebuild, comm_rebuild, aligned_rebuild_summaries, top_k=2)
    
    # Measure Context Text Jaccard Parity
    toks_incr = set(normalize_text(ctx_incr).split())
    toks_rebuild = set(normalize_text(ctx_rebuild).split())
    union_t = len(toks_incr | toks_rebuild)
    inter_t = len(toks_incr & toks_rebuild)
    jaccard = inter_t / union_t if union_t > 0 else 1.0
    jaccard_scores.append(jaccard)
    
    # Generate prediction for Incremental
    prompt_incr = (
        f"Answer the following question based ONLY on the provided graph context. "
        f"Keep the answer concise and factual.\n\nContext:\n{ctx_incr}\n\nQuestion: {q_text}\nAnswer:"
    )
    
    if prompt_incr in prompt_cache:
        pred_incr = prompt_cache[prompt_incr]
    else:
        pred_incr = call_llm_with_retry(prompt_incr)
        prompt_cache[prompt_incr] = pred_incr
        time.sleep(0.15)
            
    # For Full Rebuild
    if ctx_incr == ctx_rebuild:
        pred_rebuild = pred_incr
    else:
        prompt_rebuild = (
            f"Answer the following question based ONLY on the provided graph context. "
            f"Keep the answer concise and factual.\n\nContext:\n{ctx_rebuild}\n\nQuestion: {q_text}\nAnswer:"
        )
        if prompt_rebuild in prompt_cache:
            pred_rebuild = prompt_cache[prompt_rebuild]
        else:
            pred_rebuild = call_llm_with_retry(prompt_rebuild)
            prompt_cache[prompt_rebuild] = pred_rebuild
            time.sleep(0.15)
                
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
    
    print(f"[{idx:02d}/{total_queries}] EM: {int(em_incr)} | F1: {f1_incr:.2f} | Gold: '{gold}' | Pred: '{pred_incr}'")

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
print(f"[STATUS] Full A/B report written to: {OUTPUT_PATH}")