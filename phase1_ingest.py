import os
import json
from typing import Dict, List, Any
from dotenv import load_dotenv
from datasets import load_dataset
from groq import Groq

load_dotenv()

WORKSPACE_DIR = "data"
os.makedirs(WORKSPACE_DIR, exist_ok=True)

# 1. Groq Client Authentication & Active Model Discovery
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("Missing environment variable: Set 'GROQ_API_KEY' in your .env file.")

client = Groq(api_key=api_key)
all_model_ids = [m.id for m in client.models.list().data]

print(f"[STATUS] Probing {len(all_model_ids)} account models for active inference...")

verified_models: List[str] = []
for mid in all_model_ids:
    if any(k in mid.lower() for k in ["whisper", "guard", "vision", "embed", "safeguard", "tts"]):
        continue
    try:
        client.chat.completions.create(
            model=mid,
            messages=[{"role": "user", "content": "PING"}],
            max_tokens=2,
            temperature=0.0
        )
        verified_models.append(mid)
        print(f"  ✓ Active model verified: {mid}")
    except Exception:
        continue

if not verified_models:
    raise RuntimeError("No text inference models could be verified on this Groq account.")

fast_model = next(
    (m for m in verified_models if any(k in m.lower() for k in ["8b", "instant", "mini", "small"])),
    verified_models[0]
)
heavy_model = next(
    (m for m in verified_models if any(k in m.lower() for k in ["70b", "versatile", "120b", "large"])),
    verified_models[-1]
)

print(f"\n[STATUS] Production Fast Extractor : {fast_model}")
print(f"[STATUS] Production Heavy Summarizer: {heavy_model}")

with open(os.path.join(WORKSPACE_DIR, "model_config.json"), "w") as f:
    json.dump({"fast_model": fast_model, "heavy_model": heavy_model}, f, indent=2)


# 2. HotpotQA Benchmark Ingestion & Partitioning
def load_and_partition_benchmark(
    split_size: int = 50, base_ratio: float = 0.80
) -> Dict[str, Any]:
    print(f"\n[STATUS] Loading hotpotqa/hotpot_qa benchmark (validation[:{split_size}])...")
    
    try:
        dataset = load_dataset(
            "hotpotqa/hotpot_qa", "distractor", split=f"validation[:{split_size}]"
        )
    except Exception:
        dataset = load_dataset(
            "hotpotqa/hotpot_qa", split=f"validation[:{split_size}]"
        )

    corpus_map: Dict[str, str] = {}
    qa_benchmarks: List[Dict[str, Any]] = []

    for item in dataset:
        q_id = item["id"]
        question = item["question"]
        gold_answer = item["answer"]
        supporting_titles = set(item["supporting_facts"]["title"])

        matched_titles = []
        for title, sentences in zip(
            item["context"]["title"], item["context"]["sentences"]
        ):
            if title in supporting_titles:
                if title not in corpus_map:
                    corpus_map[title] = " ".join(sentences)
                matched_titles.append(title)

        if len(matched_titles) >= 2:
            qa_benchmarks.append(
                {
                    "id": q_id,
                    "question": question,
                    "answer": gold_answer,
                    "gold_titles": list(supporting_titles),
                }
            )

    corpus_records = [
        {"title": title, "text": text} for title, text in corpus_map.items()
    ]

    split_idx = int(len(corpus_records) * base_ratio)
    t0_base = corpus_records[:split_idx]
    t1_streaming = corpus_records[split_idx:]

    partition_artifact = {
        "metadata": {
            "total_documents": len(corpus_records),
            "base_documents_t0": len(t0_base),
            "streaming_documents_t1": len(t1_streaming),
            "eval_queries": len(qa_benchmarks),
        },
        "t0_base": t0_base,
        "t1_streaming": t1_streaming,
        "qa_benchmark": qa_benchmarks,
    }

    artifact_path = os.path.join(WORKSPACE_DIR, "dataset_split.json")
    with open(artifact_path, "w", encoding="utf-8") as f:
        json.dump(partition_artifact, f, indent=2)

    return partition_artifact


benchmark_data = load_and_partition_benchmark()
meta = benchmark_data["metadata"]

print("-" * 55)
print(f"Total Unique Passages Ingested : {meta['total_documents']}")
print(f"T0 Base Corpus (Graph Seed)    : {meta['base_documents_t0']}")
print(f"T1 Streaming Corpus (Update)   : {meta['streaming_documents_t1']}")
print(f"Verified Multi-Hop Queries     : {meta['eval_queries']}")
print("Checkpoint Saved               : data/dataset_split.json")
print("-" * 55)