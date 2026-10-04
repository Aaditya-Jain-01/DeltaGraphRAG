import os
import re
import json
import time
import hashlib
from typing import Dict, List, Any
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

WORKSPACE_DIR = "data"
SPLIT_PATH = os.path.join(WORKSPACE_DIR, "dataset_split.json")
CONFIG_PATH = os.path.join(WORKSPACE_DIR, "model_config.json")
CACHE_PATH = os.path.join(WORKSPACE_DIR, "extractions_cache.json")

if not os.path.exists(SPLIT_PATH):
    raise FileNotFoundError("Missing Phase 1 artifact: Run phase1_ingest.py first.")

with open(SPLIT_PATH, "r", encoding="utf-8") as f:
    split_data = json.load(f)

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    model_cfg = json.load(f)

corpus: List[Dict[str, str]] = split_data["t0_base"] + split_data["t1_streaming"]
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("Missing GROQ_API_KEY in .env file.")

client = Groq(api_key=api_key)

# Select extractor model from verified configuration
extractor_model = model_cfg.get("fast_model", "openai/gpt-oss-20b")
print(f"[STATUS] Extractor Model Configured : {extractor_model}")
print(f"[STATUS] Total Passages to Ingest   : {len(corpus)}")

# Load existing cache and purge zero-extraction failures
cache: Dict[str, Dict[str, Any]] = {}
if os.path.exists(CACHE_PATH):
    try:
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            raw_cache = json.load(f)
        cache = {
            k: v for k, v in raw_cache.items()
            if len(v.get("entities", [])) > 0
        }
        print(f"[STATUS] Retained {len(cache)} valid cached extractions from disk.")
    except Exception:
        cache = {}


def generate_doc_hash(title: str, text: str) -> str:
    """Computes deterministic SHA-256 hash for passage idempotency."""
    payload = f"{title.strip()}::{text.strip()}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


EXTRACTION_SYSTEM_PROMPT = """You are a knowledge graph extraction engine.
Extract core named entities and factual relationships from the passage.
You must respond ONLY with a valid JSON object matching this schema:
{
  "entities": [
    {"name": "Entity Name", "type": "Person | Organization | Location | Work | Concept"}
  ],
  "relations": [
    {"source": "Entity Name", "target": "Entity Name", "relation": "relationship_label"}
  ]
}"""


def call_llm_json(prompt_text: str, model_id: str, max_retries: int = 5) -> Dict[str, Any]:
    """Queries Groq with native JSON mode enforcement and exponential backoff."""
    delay = 2.0
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt_text},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=1024,
            )
            raw_text = response.choices[0].message.content.strip()
            data = json.loads(raw_text)
            return {
                "entities": data.get("entities", []),
                "relations": data.get("relations", [])
            }
        except Exception as exc:
            err = str(exc).lower()
            if "429" in err or "rate limit" in err:
                print(f"  [RATE-LIMIT] Backing off for {delay:.1f}s (retry {attempt + 1}/{max_retries})...")
                time.sleep(delay)
                delay *= 2.0
            else:
                time.sleep(1.0)
    return {"entities": [], "relations": []}


# Incremental extraction loop
extracted_new = 0
start_time = time.time()

for idx, doc in enumerate(corpus, start=1):
    doc_hash = generate_doc_hash(doc["title"], doc["text"])

    if doc_hash in cache:
        continue

    prompt = f"Title: {doc['title']}\nPassage:\n{doc['text']}"
    parsed = call_llm_json(prompt, extractor_model)

    # Retry once if zero entities returned
    if len(parsed.get("entities", [])) == 0:
        time.sleep(1.0)
        parsed = call_llm_json(prompt, extractor_model)

    cache[doc_hash] = {
        "title": doc["title"],
        "entities": parsed.get("entities", []),
        "relations": parsed.get("relations", []),
    }
    extracted_new += 1

    # Atomically persist to disk after each item
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)

    n_e = len(parsed.get("entities", []))
    n_r = len(parsed.get("relations", []))
    print(f"[{idx:03d}/{len(corpus)}] Ingested '{doc['title'][:25]:<25}' -> {n_e:2d} entities, {n_r:2d} relations")

    time.sleep(1.2)

elapsed = time.time() - start_time

print("-" * 55)
print(f"Total Passages Cached       : {len(cache)} / {len(corpus)}")
print(f"New Extractions Executed    : {extracted_new}")
print(f"Execution Wall Time         : {elapsed:.1f} seconds")
print(f"Checkpoint Verified         : data/extractions_cache.json")
print("-" * 55)