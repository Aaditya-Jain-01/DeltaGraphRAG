import json

with open("data/extractions_cache.json", "r", encoding="utf-8") as f:
    cache = json.load(f)

with open("data/dataset_split.json", "r", encoding="utf-8") as f:
    splits = json.load(f)

print(f"Total items in extraction cache: {len(cache)}")
print(f"First 3 cache keys: {list(cache.keys())[:3]}")

t1_list = splits.get("t1_streaming", [])
print(f"Total T1 items: {len(t1_list)}")
if t1_list:
    sample = t1_list[0]
    print(f"T1 item type: {type(sample)}")
    print(f"Sample T1 content preview: {repr(sample)[:200]}")