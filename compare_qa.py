import json
import os
import sys

GEN_PATH = "data/generation_metrics.json"

if not os.path.exists(GEN_PATH):
    print("[ERROR] Please run 'python evaluate_e2e.py' first to generate empirical A/B evaluation data.")
    sys.exit(1)

with open(GEN_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

ab = data.get("ab_comparison", {})
total = data.get("total_queries", 50)
model = data.get("model_evaluated", "Unknown")

print("=" * 65)
print("     DELTAGRAPHRAG vs. FULL REBUILD EMPIRICAL A/B REPORT     ")
print("=" * 65)
print(f"Evaluated Test Queries           : {total}")
print(f"Inference Model                  : {model}")
print(f"Retrieval Context Jaccard Parity : {ab.get('context_jaccard_parity_pct', 0.0):.2f}%")
print(f"Direct Prediction Agreement Rate : {ab.get('prediction_agreement_pct', 0.0):.2f}%")
print("-" * 65)
print("Empirical Quality Comparison:")
print(f"  Full Rebuild Exact Match       : {ab.get('full_rebuild_em_pct', 0.0):.2f}%")
print(f"  DeltaGraphRAG Exact Match      : {ab.get('incremental_em_pct', 0.0):.2f}%")
print(f"  Full Rebuild Mean Token F1     : {ab.get('full_rebuild_f1_pct', 0.0):.2f}%")
print(f"  DeltaGraphRAG Mean Token F1    : {ab.get('incremental_f1_pct', 0.0):.2f}%")
print(f"  Measured Quality Degradation   : {ab.get('quality_degradation_pct', 0.0):.2f}%")
print("=" * 65)