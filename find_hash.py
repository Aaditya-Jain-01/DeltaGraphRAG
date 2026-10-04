import os
import json
import hashlib

# 1. Inspect existing Python files for the cache key logic
py_files = [f for f in os.listdir(".") if f.endswith(".py")]
print(f"[STATUS] Python files detected: {py_files}")

print("\n[STATUS] Scanning Python scripts for cache lookup logic...")
for py_file in py_files:
    try:
        with open(py_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                l_lower = line.lower()
                if any(k in l_lower for k in ["extractions_cache", "extract_cache", "sha256", "cache.get", "cache["]):
                    if not line.strip().startswith("#"):
                        print(f"  [{py_file}] {line.strip()}")
    except Exception:
        pass

# 2. Test candidate serializations against all cache keys
with open("data/extractions_cache.json", "r", encoding="utf-8") as f:
    cache = json.load(f)
cache_keys = set(cache.keys())

with open("data/dataset_split.json", "r", encoding="utf-8") as f:
    splits = json.load(f)

all_docs = splits.get("t1_streaming", []) + splits.get("t0_base", [])
print(f"\n[STATUS] Total cache entries: {len(cache_keys)}")
print(f"[STATUS] Total documents in split: {len(all_docs)}")

found = False
for doc in all_docs[:10]:
    if not isinstance(doc, dict):
        continue
    
    t = doc.get("title", "")
    x = doc.get("text", "")

    candidates = [
        ("doc['text']", x),
        ("doc['text'].strip()", x.strip()),
        ("doc['title'] + ' ' + doc['text']", f"{t} {x}"),
        ("doc['title'] + '\\n' + doc['text']", f"{t}\n{x}"),
        ("json.dumps(doc)", json.dumps(doc)),
        ("json.dumps(doc, sort_keys=True)", json.dumps(doc, sort_keys=True)),
        ("json.dumps(doc, separators=(',', ':'))", json.dumps(doc, separators=(',', ':'))),
        ("str(doc)", str(doc)),
    ]

    for label, payload in candidates:
        h = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        if h in cache_keys:
            print(f"\n[FOUND MATCH] Key pattern: {label}")
            print(f"Generated Hash : {h}")
            found = True
            break
    if found:
        break

if not found:
    # Inspect actual content of first cache value to trace fields
    first_k = next(iter(cache_keys))
    print(f"\n[INFO] Sample cache value keys for '{first_k[:16]}...': {list(cache[first_k].keys())}")