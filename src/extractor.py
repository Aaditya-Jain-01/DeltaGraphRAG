import json
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
        prompt = f"Title: {title}\nPassage:\n{text}"
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
