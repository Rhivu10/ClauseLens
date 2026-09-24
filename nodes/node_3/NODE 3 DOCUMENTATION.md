# Node 3 Documentation

## Overview

**Node 3** is the clause alignment layer of ClauseLens.

For every clause (chunk) in a source document, Node 3 finds the matching clause in the target document and decides whether it is **EQUIVALENT**, **MODIFIED**, or has **NO_MATCH**. At document level it also reports clauses that were **REMOVED** (only in the source) and **ADDED** (only in the target).

Node 3 currently:

- Retrieves target-document candidates from the Node 2 FAISS index.
- Applies a similarity gate, so clearly unrelated clauses never reach the model.
- Skips the model when both texts are identical (`exact_text`).
- Verifies remaining pairs with **Gemma 4 E2B fine-tuned on CUAD** (LoRA adapter).
- Parses Gemma's JSON output tolerantly (code fences, extra prose, bad JSON → `UNKNOWN`).
- Runs a deterministic **numeric diff** (amounts, dates, counts) on the full text as an extra flag.
- Shrinks long prompts evenly on both sides to fit the token budget and flags them as `truncated`.
- Recovers from CUDA out-of-memory errors (`UNKNOWN`, `decided_by: "error"`).
- Provides a threshold calibration helper.

Node 3 does **not** extract PDFs or build indexes. Those are Node 1 and Node 2.

---

# Architecture

```text
                    NODE 2
        Canonical chunks + IndexBundle
                       |
                       v
              +------------------+
              |   aligner.py     |
              |  get_candidates  |  semantic_search, target document only
              +--------+---------+
                       |
                       v
               similarity gate  ---- below threshold ---->  NO_MATCH  (no Gemma call)
                       |
                       v
               exact-text check ---- identical ---------->  EQUIVALENT
                       |
                       v
              +------------------+
              |   verifier.py    |
              |  Gemma + LoRA    |  ---->  EQUIVALENT | MODIFIED | NO_MATCH | UNKNOWN
              +--------+---------+
                       |
                       v
              +------------------+
              |   helpers.py     |  parse_verdict, numeric_diff
              +------------------+
```

---

## Files

| File | Purpose |
|---|---|
| `config.py` | `Node3Config` (all tunable values), verdict constants |
| `helpers.py` | Text normalization, numeric diff, verdict parser (no GPU needed) |
| `verifier.py` | Loads Gemma + CUAD LoRA adapter, builds the prompt, runs generation |
| `aligner.py` | `Node3Aligner`: retrieval, gating, clause and document alignment, calibration |

---

## Public Interface

```python
from nodes.node_3 import (
    Node3Config,
    load_gemma_with_adapter,
    GemmaVerifier,
    Node3Aligner,
    show,
)
```

---

## Usage

Node 3 needs the outputs of Node 1 and Node 2 plus the fine-tuned model.

```python
import unsloth  # must be imported before transformers / peft

from nodes.node_1 import process_documents
from nodes.node_2 import prepare_chunks, Node2Indexer
from nodes.node_3 import (
    Node3Config, load_gemma_with_adapter, GemmaVerifier, Node3Aligner, show,
)

# Node 1 + Node 2
result = process_documents(Doc1, Doc2, start_new_chunk_on_level=2, max_chars=1500)
all_chunks = prepare_chunks(
    result, version_ids={"document_a": "version_a", "document_b": "version_b"},
)
indexer = Node2Indexer()
index_bundle = indexer.build(all_chunks)

# Node 3
config = Node3Config()
model, tokenizer = load_gemma_with_adapter(
    CHECKPOINT_PATH, max_seq_length=config.max_seq_length, hf_token=HF_TOKEN,
)
verifier = GemmaVerifier(model, tokenizer, config)
aligner = Node3Aligner(all_chunks, index_bundle, indexer.embedding_model, verifier)

# One clause
r = aligner.align_source_clause(source_chunk, "document_b")
show(r, source_chunk)

# Whole document
report = aligner.align_documents("document_a", "document_b")
print(report["counts"], report["removed"], report["added"])
```

---

## Configuration

`Node3Config` fields:

| Field | Default | Meaning |
|---|---|---|
| `candidate_k` | `3` | Target candidates verified per source chunk |
| `similarity_threshold` | `0.55` | Minimum cosine score to call Gemma. **Tentative — calibrate** |
| `text_chars` | `1200` | Character limit per side (same for source and target) |
| `max_seq_length` | `1024` | Must match the value the model was loaded with |
| `max_new_tokens` | `80` | Gemma output length |
| `cleanup_every` | `10` | Empty the CUDA cache every N Gemma calls |

`max_input_tokens` is derived: `max_seq_length - max_new_tokens - 16`.

---

## Output Format

### `align_source_clause(source_chunk, target_document_id, threshold=None)`

```python
{
    "source_chunk_id": "...",
    "status": "COMPARED",          # NO_CANDIDATES | BELOW_THRESHOLD | COMPARED
    "verdict": "MODIFIED",         # EQUIVALENT | MODIFIED | NO_MATCH | UNKNOWN
    "best_score": 0.7123,
    "matched_chunk_id": "...",     # None unless EQUIVALENT or MODIFIED
    "matches": [
        {
            "candidate_chunk_id": "...",
            "similarity_score": 0.7123,
            "numeric_diff": {"only_in_source": ["$5,000"], "only_in_target": ["$7,500"]},
            "result": {"alignment_result": "MODIFIED", "change_type": "payment amount"},
            "decided_by": "gemma",   # gemma | exact_text | error
            "truncated": False,
        },
    ],
}
```

When several candidates are compared, the best verdict wins in the order
`EQUIVALENT > MODIFIED > NO_MATCH > UNKNOWN`, even if it is not the top-scored candidate.

A source chunk from the same document as the target raises `ValueError` (self-match guard).

### `align_documents(source_doc_id, target_doc_id, max_source_chunks=None, verbose=True)`

```python
{
    "results": [...],                  # one align_source_clause result per source chunk
    "removed": ["..."],                # source chunks with NO_MATCH
    "added":   ["..."],                # target chunks never matched
    "counts":  {"EQUIVALENT": 12, "MODIFIED": 5, "NO_MATCH": 3, "UNKNOWN": 0},
}
```

Matching is greedy: one target chunk may be matched by several source chunks.
For a symmetric view, also run `align_documents("document_b", "document_a")`.

---

## Threshold Calibration

`similarity_threshold = 0.55` is a starting guess. To calibrate:

```python
import numpy as np

rows = aligner.best_scores("document_a", "document_b")
vals = np.array([s for _, s in rows])
print(np.round(np.percentile(vals, [0, 10, 25, 50, 75, 90, 100]), 3))
```

Pick a value between the scores of known-unrelated and known-related clauses, then set
`config.similarity_threshold`. The lowest-scoring chunk should come back as
`BELOW_THRESHOLD` from `align_source_clause`.

---

## Status and Known Limitations

See `NODE 3 REVIEW.md` for the full review. Its cell numbers refer to the original
single-file notebook that Node 3 was packaged from. Current status:

| Test | Result |
|---|---|
| Self-match guard | Rejects a same-document source and target |
| Unmatched clause | `BELOW_THRESHOLD`, score 0.15, no Gemma call (correct) |
| Indemnification / governing law | `BELOW_THRESHOLD` (0.50 / 0.45); source chunks were mis-scoped or too long |
| Full run (2 unrelated contracts) | 29 `NO_MATCH`, 1 `EQUIVALENT`, 86 s |

Still unverified:

- **Chunking quality comes from Node 1.** Node 3 uses the current Node 1 in this repo. The last run
  produced some very long chunks (4,710 chars) and page footers in the text, which lower retrieval
  scores. Check the Node 1 quality cell in the notebook before trusting Node 3 results.
- **`similarity_threshold = 0.55`** is a placeholder until calibrated on labeled related and unrelated pairs.
- **The `MODIFIED` path** has not been exercised. Suggested test: align two identical copies of one
  PDF (all `EQUIVALENT`), then edit amounts, dates and party names in one copy (should be `MODIFIED`).
- **The single `EQUIVALENT`** in the full run (`document_a_0012` → `document_b_0005`) needs a manual
  check; it may be a shared heading or boilerplate matched by the exact-text shortcut.
- **Adapter accuracy.** The LoRA adapter was trained on CUAD extraction QA, not on this comparison
  prompt, so `MODIFIED` vs `EQUIVALENT` accuracy needs measuring.
- **Speed.** Each Gemma call takes about 4–5 s; document pairs with many real matches will be slow.

---

## Requirements

- GPU (tested on Kaggle)
- `unsloth`, `peft`, `transformers==5.5.0`, `torch`
- Node 1 and Node 2 dependencies
- CUAD LoRA checkpoint (`adapter_config.json`, `adapter_model.safetensors`)
