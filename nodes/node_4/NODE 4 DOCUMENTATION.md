# Node 4 Documentation

## Overview

**Node 4** is the Deontic Cue Extractor of ClauseLens.

"Deontic" means related to duty or permission. Node 4 takes the changed clause pairs from Node 3 and labels every duty-bearing sentence as an **OBLIGATION**, **PERMISSION** or **PROHIBITION**. It then reports how the duty changed between versions (for example `PERMISSION -> PROHIBITION` when "may sublicense" becomes "shall not sublicense") and sends the enriched pairs to Node 5.

Node 4 currently:

- Reads the Node 3 `align_documents()` report directly (no manual conversion).
- Keeps only changed pairs: **MODIFIED**, **ADDED** and **DELETED**. EQUIVALENT pairs are skipped.
- Splits each clause into sentences, keeping character offsets into the original chunk text.
- Finds trigger words with ordered regex rules (shall, must, may, shall not, prohibited, entitled to...).
- Ignores definitions such as "shall mean" and "shall be deemed".
- Asks Gemma for a quick check only on ambiguous sentences (weak cues such as "will", or mixed labels).
- Reuses the Gemma model already loaded for Node 3, so it needs no extra GPU memory.
- Processes pairs one at a time and saves each result to disk immediately, so a crashed session resumes where it stopped.

> **Research pivot:** instead of forcing a small model to do formal deontic logic, Node 4 uses simple text patterns and calls the model only when the patterns are unsure.

---

# Architecture

```text
                        NODE 3
             align_documents() report
                          |
                          v
                 +------------------+
                 |  extractor.py    |
                 | pairs_from_node3 |  MODIFIED | ADDED | DELETED
                 +--------+---------+
                          |
                          v
                 +------------------+
                 |   patterns.py    |
                 | split_sentences  |
                 | find_cues        |  ordered regex, masked spans
                 | label_sentence   |
                 +--------+---------+
                          |
               ambiguous? |
                          v
                 +------------------+
                 |   checker.py     |
                 | DeonticChecker   |  Gemma (shared with Node 3), cached
                 +--------+---------+
                          |
                          v
                 enriched pair ----> JSONL checkpoint ----> NODE 5
```

---

## Files

| File | Purpose |
|---|---|
| `config.py` | `Node4Config`, label and change-status constants |
| `patterns.py` | Sentence splitter and regex cue rules (no GPU needed) |
| `checker.py` | `DeonticChecker`: quick Gemma classification of one sentence |
| `extractor.py` | `Node4Extractor`: Node 3 → pairs → labels → checkpoint → Node 5 |

---

## Public Interface

```python
from nodes.node_4 import (
    Node4Config,
    DeonticChecker,
    Node4Extractor,
    read_node4_output,
    tag_text,
)
```

---

## Usage

Node 4 runs straight after Node 3 in the same session.

```python
from nodes.node_4 import Node4Config, DeonticChecker, Node4Extractor

# report = aligner.align_documents("document_a", "document_b")   # Node 3

node4_config = Node4Config(model_check="ambiguous")
checker = DeonticChecker(model, tokenizer, node4_config)   # same Gemma as Node 3
extractor = Node4Extractor(all_chunks, checker, node4_config)

node4_output = extractor.run(
    report,
    checkpoint_path="/kaggle/working/clauselens_outputs/node4_pairs.jsonl",
)
```

### Continuous (streaming) use

`iter_run()` yields each enriched pair as soon as it is ready, so Node 5 can start on the first pair while Node 4 is still working on the rest:

```python
for pair in extractor.iter_run(report, checkpoint_path=NODE4_CHECKPOINT):
    node5_process(pair)
```

### Regex only (no GPU)

```python
extractor = Node4Extractor(all_chunks, config=Node4Config(model_check="off"))
```

### Loading saved output

```python
from nodes.node_4 import read_node4_output
pairs = read_node4_output("/kaggle/working/clauselens_outputs/node4_pairs.jsonl")
```

---

## Cue Rules

Rules run in this order. Each match masks its words, so "shall not" is counted once as a prohibition and never again as "shall".

| Order | Label | Examples | Strength |
|---|---|---|---|
| 1 | *(ignored)* | shall mean, shall have the meaning, shall include, shall be deemed | — |
| 2 | PERMISSION (exemption) | shall not be required to, is not obligated to | strong |
| 3 | PROHIBITION | shall not, must not, may not, in no event shall, neither party shall, is prohibited, prohibits, is not permitted to | strong |
| 3 | PROHIBITION | cannot | weak |
| 4 | PERMISSION | shall have the right to, has the right to, is entitled to, at its sole discretion | strong |
| 5 | OBLIGATION | shall, must, is required to, undertakes to, covenants to | strong |
| 5 | OBLIGATION | agrees to, is responsible for, will | weak |
| 6 | PERMISSION | may | strong |
| 6 | PERMISSION | can | weak |

A sentence is **ambiguous** when it has cues for more than one label, or only weak cues. With several labels, the sentence takes the highest priority: `PROHIBITION > OBLIGATION > PERMISSION`.

---

## Configuration

`Node4Config` fields:

| Field | Default | Meaning |
|---|---|---|
| `model_check` | `"ambiguous"` | `"ambiguous"`: Gemma checks unclear sentences only. `"all"`: every sentence with a cue. `"off"`: regex only |
| `sentence_chars` | `600` | Character limit of a sentence sent to Gemma |
| `max_new_tokens` | `24` | Gemma output length |
| `cleanup_every` | `10` | Empty the CUDA cache every N Gemma calls |

---

## Output Format

### One enriched pair (sent to Node 5)

```python
{
    "pair_id": "MODIFIED:document_a_0007->document_b_0009",
    "change_status": "MODIFIED",               # MODIFIED | ADDED | DELETED
    "source_chunk_id": "document_a_0007",      # None for ADDED
    "target_chunk_id": "document_b_0009",      # None for DELETED
    "alignment": {                             # from Node 3
        "similarity_score": 0.83,
        "change_type": "permission revoked",
        "numeric_diff": {"only_in_source": [], "only_in_target": []},
        "decided_by": "gemma",
        "truncated": False,
    },
    "source": {"chunk_id", "document_id", "version_id", "heading_path",
               "page_start", "page_end", "text"},
    "target": {...},                           # same fields, None for DELETED
    "deontic": {
        "primary_label": "PROHIBITION",        # target side, or source for DELETED
        "shift": "PERMISSION -> PROHIBITION",  # MODIFIED only, None if unchanged
        "labels_added": ["PROHIBITION"],
        "labels_removed": ["PERMISSION"],
        "has_deontic_change": True,
        "source": {
            "primary_label": "PERMISSION",
            "labels": ["PERMISSION"],
            "counts": {"PERMISSION": 1},
            "sentences": [
                {
                    "text": "The Licensee may sublicense the Software.",
                    "start": 0, "end": 41,     # offsets into source.text (for Node 6 links)
                    "cues": [{"cue": "may", "label": "PERMISSION", "strength": "strong"}],
                    "label": "PERMISSION",
                    "ambiguous": False,
                    "decided_by": "regex",     # regex | gemma
                },
            ],
        },
        "target": {...},
    },
}
```

When Gemma overrides a label, the sentence also keeps `"regex_label"` with the original regex answer.

### `run()` summary

```python
{
    "pairs": [...],                            # enriched pairs, in order
    "counts": {"MODIFIED": 4, "DELETED": 25, "ADDED": 31},
    "label_counts": {"OBLIGATION": 30, "PERMISSION": 12, "NONE": 18},
    "deontic_changes": ["MODIFIED:...", ...],  # pair_ids with has_deontic_change
    "skipped": {"EQUIVALENT": 1},              # Node 3 verdicts not sent on
}
```

---

## Checkpoint and Resume

With `checkpoint_path`, Node 4 writes a JSONL file:

- Line 1 is a header with a fingerprint of the input (pair ids, chunk text and `model_check`).
- Every finished pair is appended and flushed immediately.

If the session crashes, run the same cell again: pairs already on disk are loaded instead of processed again. A half-written last line is skipped and redone. If the checkpoint was made from different documents or settings, Node 4 raises `ValueError` instead of mixing results. Delete the file or use a new path.

---

## Known Limitations

- **Regex coverage.** Unusual wording ("it is incumbent upon", "is hereby granted") is not in the rules. Add patterns to `_RULES` in `patterns.py` as they come up.
- **"will" and "may" are broad.** "will" is treated as a weak obligation and checked by Gemma. "may" in a sentence like "this may result in" is tagged PERMISSION by regex; use `model_check="all"` if this matters.
- **Chunk-level labels.** `primary_label` is the most frequent sentence label. A long chunk with many duties is better read through its `sentences` list.
- **Model check accuracy.** The CUAD adapter was not trained on this classification prompt. Gemma's labels have not been measured against labeled sentences yet.
- **Sentence splitting** is rule-based. Uncommon abbreviations can still split a sentence early.
- **Only as good as Node 3.** Node 4 labels whichever pairs Node 3 marks as changed. With an uncalibrated Node 3 threshold, many pairs may be DELETED/ADDED rather than MODIFIED.

---

## Requirements

- Node 3 output (`align_documents()` report) and Node 2 canonical chunks
- For the model check: the Gemma model and tokenizer loaded for Node 3 (GPU)
- For `model_check="off"`: Python standard library only
