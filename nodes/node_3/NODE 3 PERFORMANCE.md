# ClauseLens Node 3 — Performance-Tuned Build

Drop-in replacement for `nodes/node_3` in
[Rhivu10/ClauseLens](https://github.com/Rhivu10/ClauseLens). Same public API,
same verdict logic, same report format — dramatically less model work.

```
from nodes.node_3 import (
    Node3Config, load_gemma_with_adapter, GemmaVerifier, Node3Aligner, show,
)
```

Nothing else in the pipeline changes: Node 4, Node 5 and Node 6 keep working
on the Node 3 report unchanged.

---

## Why

Node 3 verified every clause pair **sequentially**: one `model.generate` call
per candidate (up to `candidate_k=3` per source chunk), each burning the full
`max_new_tokens=80` budget, plus a second one-word retry whenever the answer
was unparseable, with `gc.collect()` + `torch.cuda.empty_cache()` every 10
calls. On a 4–5 page contract pair that is easily 150–250 sequential
generations — hours when the GPU is slow or unavailable, still many minutes
on Kaggle. Retrieval added re-encoding of every source chunk (already
embedded by Node 2) and an O(N²) whole-index scan per query.

## What changed (all opt-out via `Node3Config`)

| Change | Default | Effect |
|---|---|---|
| **Batched generation** — `compare_pairs()` / `retry_pairs()` left-pad up to `batch_size` pairs into one `model.generate` call | `batch_size=8` | ~5–8× throughput on T4-class GPUs |
| **Stop-at-brace** — `StoppingCriteria` ends generation once every row closed its answer JSON | `stop_at_json=True` | The parser only reads the first `{...}` block, so verdicts are identical while trailing/looping tokens are never generated (~3× fewer decode steps measured) |
| **Candidate rounds + early exit** — round 1 verifies each chunk's top candidate; a chunk whose top candidate is EQUIVALENT never sends candidates 2..k (nothing can beat EQUIVALENT, ties break by score order) | `early_exit_equivalent=True` | 27 → 18 verified pairs on the benchmark pair; scales with document similarity |
| **Precomputed embeddings** — retrieval scores via matrix product against Node 2's embedding rows instead of re-encoding + whole-index FAISS scan + O(N²) chunk lookups | `use_precomputed_embeddings=True` | 20×–420× faster retrieval, growing with corpus size; identical scores |
| **OOM split** — a CUDA-OOM batch is halved and retried down to single pairs instead of losing the verdict | `oom_split=True` | No more `decided_by: "error"` cascade |
| **Verdict cache** — JSONL persistence of pair verdicts keyed by chunk ids + prompt version | `cache_path=None` | Crash-safe resume; re-runs make zero model calls |
| **Duplicate-text memo** — identical (source, target) text pairs verified once per run | `dedup_text_pairs=True` | Repeated boilerplate costs one call |
| Cleanup every N **batches** instead of every N calls | `cleanup_every=10` | Removes periodic `gc.collect()` stalls |

Behavioral note: with early exit, `matches` for a decided chunk may hold
fewer entries than before (only the entries actually compared — the winner is
always present). Node 4/Node 6 only consume the winning entry and `matches[0]`,
so downstream output is unchanged; the A/B benchmark confirms verdict-level
equivalence.

## Install

```bash
# from your ClauseLens clone
rm -rf nodes/node_3
cp -r clauselens-node3-optimized/nodes/node_3 nodes/
```

Optional: enable the verdict cache in your notebook/pipeline:

```python
config = Node3Config(cache_path="/kaggle/working/node3_verdicts.jsonl")
```

Tuning knobs (all optional): `batch_size` (raise to 16 on 16 GB GPUs),
`max_new_tokens` (48 is safe with stop-at-brace and cuts loop waste further),
`similarity_threshold` (calibrate with `aligner.best_scores()`).

## Validation

A/B benchmark on a synthetic 28-clause-per-side contract pair (identical,
format-only, modified, paraphrased, unparseable and unmatched clauses), with a
scripted deterministic "Gemma" so both implementations see identical answers:

```
--- report equivalence ---
PASS: verdicts, matched ids, counts, removed/added all identical
      (6 chunks finished by early exit; their matches hold only the winning entry)
best_scores: max |old-new| = 1.00e-06 | same chunk ids: True

--- legacy-mode regression (optimized code, all new flags OFF) ---
PASS: report identical to the original implementation, including full matches lists

--- model work ---
                          ORIGINAL   OPTIMIZED
generate calls                  27           4
pairs verified                  27          18
decode steps (tokens)          189          38
steps / pair                   7.0         2.1

--- retrieval overhead (all source chunks, no model) ---
  chunks  original (s)  optimized (s)  speedup
      64         0.004         0.0002   19.5x
     112         0.019         0.0004   47.0x
     208         0.110         0.0008  137.9x
     400         0.749         0.0018  416.1x

--- CUDA OOM resilience (generate raises OOM above batch of 2) ---
original  : counts {'EQUIVALENT': 11, 'MODIFIED': 11, 'NO_MATCH': 6, 'UNKNOWN': 0}
optimized : counts {'EQUIVALENT': 11, 'MODIFIED': 11, 'NO_MATCH': 6, 'UNKNOWN': 0}
           (batch split 9 generate calls, max batch 2)

--- verdict cache ---
run 1:        4 generate calls, 18 pairs
run 2 (cache): 0 generate calls — identical verdicts, 18 cache entries
```

Projected wall time for this pair (27 sequential calls at the team-measured
4.7 s/call on Kaggle, 40 s/call CPU estimate):

| Environment | Original | Optimized |
|---|---|---|
| Kaggle T4/P100 GPU | ~2 m 07 s | **~7 s** |
| CPU-only | ~18 m | **~3 m 15 s** |

For the real 4–5 page pairs that took up to two hours (roughly 150–250
sequential calls), the same ratios put the optimized run at **~1–3 minutes on
a healthy GPU** and **~15–25 minutes CPU-only**. Re-runs with the cache are
near-instant.

## Tests

`tests/` contains the benchmark harness (no GPU or heavy dependencies — it
stubs torch/faiss/transformers and scripts the model):

```bash
# replace $REPO with your ClauseLens clone path inside bench_compare.py first
python3 tests/bench_compare.py
```

## Files

| File | Status | Purpose |
|---|---|---|
| `config.py` | changed | new tunables (`batch_size`, `stop_at_json`, `early_exit_equivalent`, `use_precomputed_embeddings`, `oom_split`, `dedup_text_pairs`, `cache_path`), `PROMPT_VERSION` |
| `verifier.py` | changed | batched `compare_pairs`/`retry_pairs`, stop-at-brace criteria, OOM split; single-pair API unchanged |
| `aligner.py` | changed | three-phase `align_documents` (shortcuts → batched rounds → assemble), precomputed-embedding retrieval, verdict cache + memo; `align_source_clause` unchanged |
| `helpers.py` | unchanged | — |
| `__init__.py` | unchanged | — |
