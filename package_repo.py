import os

# 1. Ensure directories exist
os.makedirs("src", exist_ok=True)

# 2. Write requirements.txt
requirements = """groq>=0.11.0
datasets>=3.0.0
networkx>=3.2.0
matplotlib>=3.8.0
pydantic>=2.0.0
python-dotenv>=1.0.0
"""
with open("requirements.txt", "w", encoding="utf-8") as f:
    f.write(requirements)

# 3. Write src/__init__.py
with open("src/__init__.py", "w", encoding="utf-8") as f:
    f.write('"""DeltaGraphRAG core package."""\n')

# 4. Write src/extractor.py
extractor_code = '''import json
import time
import hashlib
from typing import Dict, Any
from groq import Groq

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

class GraphExtractor:
    def __init__(self, client: Groq, model: str = "openai/gpt-oss-120b"):
        self.client = client
        self.model = model

    @staticmethod
    def hash_passage(title: str, text: str) -> str:
        payload = f"{title.strip()}::{text.strip()}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def extract(self, title: str, text: str, max_retries: int = 5) -> Dict[str, Any]:
        prompt = f"Title: {title}\\nPassage:\\n{text}"
        delay = 2.0
        for attempt in range(max_retries):
            try:
                res = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.0,
                    max_tokens=1024,
                )
                data = json.loads(res.choices[0].message.content.strip())
                return {
                    "entities": data.get("entities", []),
                    "relations": data.get("relations", [])
                }
            except Exception as exc:
                err = str(exc).lower()
                if "429" in err or "rate limit" in err:
                    time.sleep(delay)
                    delay *= 2.0
                else:
                    time.sleep(1.0)
        return {"entities": [], "relations": []}
'''
with open("src/extractor.py", "w", encoding="utf-8") as f:
    f.write(extractor_code)

# 5. Write src/graph_engine.py
engine_code = '''from typing import Set
import networkx as nx

class ModularityEngine:
    @staticmethod
    def compute_delta_q(
        graph: nx.Graph,
        node: str,
        community_members: Set[str]
    ) -> float:
        m = graph.number_of_edges()
        if m == 0:
            return 0.0

        k_v = graph.degree(node)
        k_v_in = sum(1 for nbr in graph.neighbors(node) if nbr in community_members)
        sigma_tot = sum(graph.degree(u) for u in community_members if graph.has_node(u))

        return (k_v_in / (2.0 * m)) - ((sigma_tot * k_v) / (2.0 * (m ** 2)))

    @staticmethod
    def calculate_drift(mutations: int, initial_size: int) -> float:
        return mutations / float(max(initial_size, 1))
'''
with open("src/graph_engine.py", "w", encoding="utf-8") as f:
    f.write(engine_code)

print("[STATUS] Requirements and src/ modules generated successfully.")