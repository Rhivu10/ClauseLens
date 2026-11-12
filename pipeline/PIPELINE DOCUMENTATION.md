# ClauseLens — Pipeline (LangGraph mapping of Nodes 1–6)

## Overview

The `pipeline` package connects all six nodes in one **LangGraph** graph, so a full comparison is a single call
instead of wiring the nodes by hand in every notebook:

```python
from pipeline import PipelineConfig, run_pipeline

cfg = PipelineConfig(
    checkpoint_path="/kaggle/input/checkpoints/checkpoint-2091",   # CUAD LoRA adapter
    document_names=("License Agreement (2009)", "Amended and Restated (2012)"),
    output_dir="/kaggle/working/clauselens_outputs",
)
state = run_pipeline("old.pdf", "new.pdf", cfg)

state["report"]          # Node 6 report (overview, stakeholders, sections, changes, evidence)
state["output_files"]    # report.json, report.md, ui_data.json
state["timings"]         # seconds per node
```

The node code is unchanged; the pipeline only calls the nodes' public functions in order.

---

## The graph

```
START
  │
node_1_parse      Node 1: PDF → hierarchy-aware chunks, then the clean-up (pipeline/cleanup.py)
  │
node_2_index      Node 2: canonical chunks + FAISS / BM25 / graph index
  │
node_3_align      Node 3: clause alignment (Gemma E2B + CUAD adapter)
  │
node_4_deontic    Node 4: changed pairs + duty cues (shall / may / shall not)
  │
  ├── no changed pairs? ──────────────┐   (conditional edge: identical documents
  │                                   │    skip Node 5 and its model loading)
node_5_impact     Node 5: Impact      │
  │               + Critic agents     │
  │                                   │
node_6_report  ◄──────────────────────┘
  │               Node 6: layered report + Markdown + UI data
 END
```

| Graph node | Reads from the state | Writes to the state |
|---|---|---|
| `node_1_parse` | `pdf_a`, `pdf_b` | `node1_result`, `cleanup_stats` |
| `node_2_index` | `node1_result` | `all_chunks`, `index_bundle`, `embedding_model` |
| `node_3_align` | `all_chunks`, `index_bundle`, `embedding_model` | `node3_report` |
| `node_4_deontic` | `all_chunks`, `node3_report` | `node4_output` |
| `node_5_impact` | `all_chunks`, `node4_output` | `node5_output` |
| `node_6_report` | `all_chunks`, `node3_report`, `node5_output` | `report`, `output_files` |

Every node also adds its run time to `timings`.

**Node 5's own loop** (write → check → retry with more context) stays inside `Node5ImpactCore`. It runs the Impact
Agent and the Critic on two GPUs at the same time; one graph step per attempt would make them wait for each other.

---

## Models (Kaggle, GPU T4 x2)

`ModelHub` (pipeline/models.py) loads and hands over the models — the same steps as the real-contract notebook:

```
GPU 0: Gemma E2B + CUAD adapter  → Node 3, Node 4
       freed, then Gemma E4B      → Node 5 Impact Agent
GPU 1: Gemma E4B (or E2B)         → Node 5 Critic
```

On a T4, Gemma 4 runs in float32, so E4B does not fit next to the Node 3 model; Nodes 3 and 4 are finished when
Node 5 starts, so the Node 3 model is freed first. Objects that hold a model (aligner, extractor, agents) are created
inside the graph nodes and dropped when the node returns. With one GPU, or if a model cannot be loaded, both agents
share one model and Node 5 runs sequentially.

**Node 5 warm-up.** The first generation compiles shared torch code. When the two Node 5 threads did that at the
same moment, Kaggle stopped with *"Detected that you are using FX to symbolically trace a dynamo-optimized
function"*. The pipeline first runs two pairs one at a time (like Test A in the notebook). If the clash still
happens, Node 5 continues sequentially from its checkpoint; finished pairs are not redone.

---

## Progress

`run_pipeline(..., on_stage=callback)` calls `callback(name)` when a node starts and
`callback(name, seconds=..., ...)` when it ends (plus `node_5_warmup`). The ClauseLens UI's Kaggle runner uses this
for its progress bar.

---

## Configuration

`PipelineConfig` fields:

| Field | Default | Meaning |
|---|---|---|
| `checkpoint_path` | — | CUAD LoRA adapter folder (`checkpoint-2091`) |
| `hf_token` | `None` | Hugging Face token (the `unsloth/*` models are public) |
| `document_names` | `("Document A", "Document B")` | Names shown in the report |
| `start_new_chunk_on_level` | `2` | Node 1 setting |
| `chunk_max_chars` | `1500` | Node 1 + clean-up chunk size |
| `clean_up` | `True` | Run the clean-up between Node 1 and Node 2 |
| `critic_base_models` | `("unsloth/gemma-4-E4B-it", "unsloth/gemma-4-E2B-it")` | Critic models, tried in order on GPU 1 |
| `critic_adapter` | `None` | Optional adapter for the Critic |
| `impact_base_model` | `"unsloth/gemma-4-E4B-it"` | Impact Agent model; `None` keeps the Node 3 model |
| `warm_up_pairs` | `2` | Node 5 warm-up pairs (0 = off) |
| `output_dir` | `"clauselens_outputs"` | Node 4/5 checkpoints and the report files |
| `node3`, `node4`, `node5` | node defaults | `Node3Config`, `Node4Config`, `Node5Config` |
| `verbose` | `True` | Progress output |

---

## Installation

LangGraph is the only new dependency (on top of the node requirements):

```
pip install langgraph
```

On Kaggle, see `notebooks/ClauseLens_LangGraph_Pipeline.py`.

---

## Testing

Tested on CPU with stand-in models that use the same interface as the Gemma models (`generate`,
`apply_chat_template`, `decode`) on the two-version test contract: all six nodes run in order and build the report;
identical PDFs skip Node 5; the Node 3 model is freed before the Node 5 Impact model is loaded; a simulated compile
clash in Node 5 finishes sequentially with the same report.
