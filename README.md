```markdown
# DeltaGraphRAG: Modularity-Gated Incremental Graph Updates

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Groq](https://img.shields.io/badge/Groq-Cloud_Inference-orange)
![NetworkX](https://img.shields.io/badge/NetworkX-Graph_Topology-lightgrey)
![HotpotQA](https://img.shields.io/badge/Benchmark-HotpotQA-purple)
![GitHub stars](https://img.shields.io/github/stars/Aaditya-Jain-01/DeltaGraphRAG?style=social)
![GitHub forks](https://img.shields.io/github/forks/Aaditya-Jain-01/DeltaGraphRAG?style=social)

## 🧠 Overview
**DeltaGraphRAG** is an incremental indexing framework designed for Graph Retrieval-Augmented Generation (GraphRAG). Traditional GraphRAG architectures require global re-clustering and full-graph re-summarization whenever new documents arrive, resulting in linear compute escalation.

DeltaGraphRAG replaces global re-indexing with a **two-tier modularity-gated (\(\Delta Q\)) update policy**. By evaluating local Newman-Girvan modularity variations during document ingress, newly observed entities are deterministically routed to adjacent semantic clusters or assigned to singleton components, reducing LLM synthesis compute by **99.6%** without sacrificing cross-document reasoning.

```

## ⚙️ Features

* **Local Modularity Delta ($\Delta Q$) Routing**: Deterministically routes streaming entities to optimal clusters using localized edge-degree variation.
* **Two-Tier Perturbation Gating**:
* **Tier 1 (Fast Path)**: Absorbs incremental updates in memory with zero LLM overhead for communities with perturbation below 15%.
* **Tier 2 (Targeted Re-index)**: Selectively triggers single-community LLM re-synthesis only when structural drift exceeds threshold bounds.


* **High-Modularity Louvain Partitioning**: Yields clean structural partitioning ($Q \approx 0.9430$) on multi-hop corpus data.
* **Idempotent Extraction Engine**: Caches schema-enforced entity-relation extractions backed by deterministic SHA-256 chunk hashing.
* **Interactive Multi-Hop CLI**: Standalone command-line inference engine (`query.py`) for live querying across multi-hop reasoning chains.

---

## 📸 Visualizations & Topology

### Empirical Evaluation Dashboard

*Figure 1: Empirical evaluation across 50 multi-hop HotpotQA queries showing compute reduction (-99.6%), Token F1 density, error taxonomy breakdown, and scale-free Louvain cluster cardinality.*

### Community Topology Distribution

*Figure 2: Knowledge graph topology across 731 entity nodes partitioned via Louvain modularity optimization ($Q \approx 0.9430$). Nodes are colored by detected semantic community.*

---

## 📊 Empirical Evaluation & Benchmarks

DeltaGraphRAG was evaluated on a 100-document multi-hop partition from **HotpotQA**, split into a baseline index ($T_0 = 80$ passages) and a streaming ingestion batch ($T_1 = 20$ passages).

### 1. Ingress Efficiency & Compute Ablation

| Metric | Full Rebuild Baseline | DeltaGraphRAG (Ours) | Delta / Savings |
| --- | --- | --- | --- |
| **Synthesis Invocations** | 257 LLM calls | **1 LLM call** | **-99.6% API calls** |
| **In-Memory Patches** | 0 communities | **72 communities** | Absorbed via Tier 1 |
| **Ingress Wall Latency** | Full re-clustering overhead | **0.01s** (streaming step) | Near-instant routing |
| **Topology Equivalence** | 731 nodes, 516 edges | 731 nodes, 516 edges | **Verified Identical ($V_\Delta = V_{ref}, E_\Delta = E_{ref}$)** |
| **Modularity ($Q$)** | 0.9430 | 0.9430 | Optimal partition preserved |

*Boundary Condition Optimization: By absorbing newly formed singleton clusters into memory during ingress, DeltaGraphRAG avoids re-synthesis thrashing on zero-baseline communities, triggering targeted synthesis only when existing clusters breach the $\tau = 0.15$ perturbation threshold.*

### 2. End-to-End Multi-Hop QA Performance

Evaluated across 50 multi-hop reasoning questions using `qwen/qwen3.8-27b`:

| Evaluation Metric | Score | Context / Methodology |
| --- | --- | --- |
| **Exact Match (EM)** | **32.00%** | Strict string match against gold HotpotQA ground truth |
| **Mean Token F1** | **41.09%** | Precision/Recall overlap across predicted answer tokens |
| **Graph Recall@2** | **60.00%** | Macro-thematic community retrieval ($top\_k=2$) |
| **Lexical BM25 Recall** | 80.00% | Inverted raw-passage token matching baseline |

*Trade-Off Analysis: Pure community summary retrieval excels at holistic, high-level theme synthesis across disparate nodes. Where it faces challenges is micro-entity needle-in-a-haystack lookups, motivating a hybrid Graph + BM25 approach for production workloads.*

---

## 📐 Mathematical Formulation

### 1. Local Modularity Delta ($\Delta Q$)

When an incoming entity $v$ is introduced into graph $G = (V, E)$ with total edge count $m = \vert{}E\vert{}$, its placement into an adjacent community $C$ is governed by the localized Newman-Girvan modularity gain:

$$\Delta Q(v \to C) = \left[ \frac{k_{v, \text{in}}}{2m} \right] - \left[ \frac{\Sigma_{\text{tot}} \cdot k_v}{2m^2} \right]$$

Where:

* $v$: Streaming entity vertex to be ingested.
* $C$: Candidate adjacent community.
* $k_{v, \text{in}}$: Total internal degree connecting vertex $v$ to vertices inside community $C$.
* $k_v$: Total degree of vertex $v$ in graph $G$.
* $m$: Total edge count of graph $G$ ($m = \vert{}E\vert{}$).
* $\Sigma_{\text{tot}}$: Sum of all vertex degrees for members belonging to community $C$.

Node $v$ is assigned to community $C^*$ that maximizes modularity gain:

$$C^* = \arg\max_{C} \, \Delta Q(v \to C)$$

If $\max \Delta Q \le 0$, node $v$ initializes a new singleton community cluster.

### 2. Community Perturbation & Drift Gating

Each community tracks accumulated vertex mutation volume $\vert{}\Delta V_C\vert{}$ relative to its baseline cardinality $\vert{}V_C\vert{}$:

$$\text{Drift}(C) = \frac{\vert{}\Delta V_C\vert{}}{\vert{}V_C\vert{}}$$

The re-indexing policy routes updates based on a 15% perturbation threshold with boundary protection for newly initialized clusters ($\vert{}V_C\vert{} = 0$):

$$\text{Action}(C) = \begin{cases} \text{Tier 1: In-Memory Patch}, & \vert{}V_C\vert{} = 0 \text{ or } \text{Drift}(C) < 0.15 \\ \text{Tier 2: Targeted Re-synthesis}, & \vert{}V_C\vert{} > 0 \text{ and } \text{Drift}(C) \ge 0.15 \end{cases}$$

* **Tier 1 (Zero-LLM Fast Path)**: When $\text{Drift}(C) < 0.15$ or $\vert{}V_C\vert{} = 0$, new nodes are absorbed directly in memory without invoking external models.
* **Tier 2 (Targeted Re-synthesis)**: When $\text{Drift}(C) \ge 0.15$, an isolated LLM re-summarization is triggered solely for community $C$.

---

## 🛠 Installation

Requires **Python 3.10+**.

1. Clone the repository:

```bash
git clone [https://github.com/Aaditya-Jain-01/DeltaGraphRAG.git](https://github.com/Aaditya-Jain-01/DeltaGraphRAG.git)
cd DeltaGraphRAG

```

2. Create and activate a virtual environment:

```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

```

3. Install project dependencies:

```bash
pip install -r requirements.txt

```

4. Configure your Groq API credentials:
Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_groq_api_key_here

```

---

## ▶️ Usage

### Reproducing the Pipeline End-to-End

Execute the phased pipeline sequentially:

```bash
# 1. Ingest HotpotQA benchmark and generate T0/T1 corpus partitions
python phase1_ingest.py

# 2. Extract entities and relationships with SHA-256 caching
python phase2_extract.py

# 3. Construct base graph and synthesize initial Louvain communities
python phase3_base_graph.py

# 4. Run streaming ingress benchmark and modularity evaluation
python phase4_benchmark.py

```

### Running the End-to-End Generation Evaluation

To evaluate generation accuracy across all 50 HotpotQA multi-hop benchmark questions:

```bash
python evaluate_e2e.py

```

### Interactive Multi-Hop Query CLI

Ask arbitrary questions directly against the compiled knowledge graph:

```bash
python query.py --question "Were Scott Derrickson and Ed Wood of the same nationality?"

```

Expected terminal output:

```text
[QUERY] Were Scott Derrickson and Ed Wood of the same nationality?
[ANSWER] Yes, both Scott Derrickson and Edward Davis Wood Jr. are American.

```

### Generating Visualizations

```bash
python visualize_graph.py
python generate_dashboard.py

```

---

## 📁 Repository Structure

```text
DeltaGraphRAG/
├── data/
│   ├── dataset_split.json          # HotpotQA T0/T1 split partitions
│   ├── extractions_cache.json      # SHA-256 cached entity-relation extractions
│   ├── base_graph.json             # Serialized NetworkX base graph
│   ├── base_graph.graphml          # Standard GraphML topological export
│   ├── base_communities.json      # Community membership index
│   ├── base_summaries.json        # Synthesized community descriptions
│   ├── benchmark_results.json      # Phase 4 retrieval benchmark telemetry
│   ├── generation_metrics.json     # 50-query end-to-end evaluation log
│   ├── graph_topology.png          # High-resolution Louvain community plot
│   └── benchmark_dashboard.png     # 4-panel evaluation analytics dashboard
├── src/
│   ├── __init__.py
│   ├── extractor.py                # Schema-enforced extraction engine
│   └── graph_engine.py             # Modularity delta & drift calculation
├── phase1_ingest.py                # Benchmark dataset partitioning
├── phase2_extract.py                # Rate-paced knowledge extraction
├── phase3_base_graph.py            # Louvain community synthesis
├── phase4_benchmark.py             # Streaming ingress & modularity benchmark
├── evaluate_e2e.py                 # Full 50-query EM and F1 evaluation suite
├── query.py                        # Standalone interactive query CLI
├── visualize_graph.py              # Headless network visualization script
├── generate_dashboard.py           # Evaluation analytics generator
├── requirements.txt                # Pinned production dependencies
└── README.md

```

---

## 🪪 License

Licensed under the **MIT License**.

```

```
