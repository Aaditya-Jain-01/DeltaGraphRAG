import json

# 1. Print the exact hash function from phase2_extract.py
print("=== HASH FUNCTION IN phase2_extract.py ===")
with open("phase2_extract.py", "r", encoding="utf-8") as f:
    lines = f.readlines()
    for idx, line in enumerate(lines):
        if "hashlib" in line or "doc_hash" in line:
            start = max(0, idx - 4)
            end = min(len(lines), idx + 8)
            for j in range(start, end):
                print(f"Line {j+1}: {lines[j].rstrip()}")
            break

# 2. Print sample structure from extractions_cache.json
print("\n=== SAMPLE CACHE ENTRY ===")
with open("data/extractions_cache.json", "r", encoding="utf-8") as f:
    cache = json.load(f)
    first_k = next(iter(cache))
    item = cache[first_k]
    print(f"Key: {first_k}")
    print(f"Title: {item.get('title')}")
    print(f"Entities sample: {item.get('entities', [])[:2]}")
    print(f"Relations sample: {item.get('relations', [])[:2]}")