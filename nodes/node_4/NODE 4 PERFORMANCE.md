# ClauseLens — Node 4 optimized (drop-in replacement)

Same optimization recipe as the Node 3 package, applied to the deontic
checker: **the model is asked fewer questions, in batches, and generation
stops as soon as the answer JSON is complete.**

- **What changed:** `config.py`, `checker.py`, `extractor.py` inside `nodes/node_4`
- **What did NOT change:** prompts, regex cue rules, verdict parsing, output
  schema, checkpoint format (`NODE4_VERSION` untouched), public API
  (`Node4Config`, `DeonticChecker`, `Node4Extractor`, `read_node4_output`,
  `classify`, `profile`, `enrich_pair`, `iter_run`, `run`)
- **Files copied untouched:** `patterns.py`, `__init__.py`
- Nodes 1, 2, 3, 5, 6 are not involved.

## Install

Copy over your repo's `nodes/node_4` directory (keep your own
`nodes/__init__.py`):

```
cp -r nodes/node_4 /path/to/clauselens/nodes/node_4_new
rm -rf /path/to/clauselens/nodes/node_4
mv /path/to/clauselens/nodes/node_4_new /path/to/clauselens/nodes/node_4
```

The notebook needs no edits:

```python
node4_config = Node4Config(model_check="ambiguous")   # or "all"
checker     = DeonticChecker(model, tokenizer, node4_config)   # reuses Node 3's model
extractor   = Node4Extractor(all_chunks, checker, node4_config)
node4_output = extractor.run(report, checkpoint_path=NODE4_CHECKPOINT)
```

Existing Node 4 checkpoint files remain valid (fingerprint logic unchanged).

## Why it was slow

`DeonticChecker.classify()` called `model.generate()` **once per sentence**.
Node 4 profiles both sides of every changed pair; with `model_check="all"`
that is one serial model call per cued sentence — on a 4–5 page contract
easily 50–150 calls at ~30–40 s each, i.e. the same disease that made
Node 3 take two hours.

## What was changed (all flags in `Node4Config`, all default ON, all reversible)

| Change | Flag | Effect |
|---|---|---|
| Batched classification: up to N sentences per `model.generate` call, left-padded | `batch_size = 16` | generate calls drop ~8x; deontic answers are tiny JSON, so a batch fits easily |
| Window prefetch: `iter_run()` classifies all sentences a window of pairs needs, up front, in batches | `prefetch_pairs = 8` | the per-pair path (`enrich_pair`) runs entirely on warm cache entries — zero generate calls of its own |
| Stop generation at the first `}` of the answer JSON | `stop_at_json = True` | the parser only reads the first `{...}` block, so the parsed label is identical while tokens drop from up to 40 to ~10–15 per answer |
| OOM resilience: a failing batch is halved and retried down to single sentences | `oom_split = True` | no more lost confirmations under memory pressure (a sentence that fails alone still returns `(None, "CUDA OOM")` exactly like before) |
| Persistent verdict cache (JSONL), keyed by normalized sentence + allowed labels + prompt version | `cache_path = None` | crash-safe resume and instant re-runs (0 model calls); set a path to enable |
| Cleanup every N batches instead of every N calls | `cleanup_every = 10` | `gc.collect()` over the Node 2 graph is not free |

Exact rollback to the original behaviour:

```python
config = Node4Config(model_check="ambiguous",
                     batch_size=1, prefetch_pairs=1,
                     stop_at_json=False, oom_split=False, cache_path=None)
```

## Benchmark (synthetic 22-chunk contract pair, stub Gemma — see `tests/`)

Report equivalence, both `model_check` modes: **PASS — byte-identical
output** (pairs, labels, `decided_by`, `gemma_raw`, counts).
Legacy-mode regression (all new flags OFF): **PASS — output AND model
workload identical to the original** (26 calls / 140 steps on both sides).

Model work, `model_check="all"`:

|                      | ORIGINAL | OPTIMIZED |
|----------------------|----------|-----------|
| generate calls       | 26       | 3         |
| sentences verified   | 26       | 26        |
| decode steps         | 140      | 24        |
| steps / sentence     | 5.4      | 0.9       |

Resilience: batch splitting under OOM keeps output identical (14 calls,
max batch 2, 0 error rows); total OOM falls back to regex labels exactly
like the original. Checkpoint resume: re-run with a finished checkpoint
makes 0 model calls. Verdict cache: warm re-run makes 0 model calls,
partial cache lines and stale prompt versions are handled safely, OOM
answers are never persisted.

Projected wall time for the benchmark pair (Gemma work only, using the
team's measured 4.7 s/call on Kaggle GPU, 40 s/call CPU estimate):
**Kaggle GPU 2m02s → 0m14s; CPU 17m20s → 2m00s.**
For real 4–5 page contracts in `model_check="all"` (dozens to ~150 cued
sentences) that is roughly tens of minutes → 2–5 minutes on GPU.

## Reproduce the benchmark

```
python3 tests/bench_run.py     # A/B equivalence, model work, OOM, resume
python3 tests/bench_cache.py   # persistent verdict cache round-trip
```

The tests stub `torch`/`transformers` and use a deterministic scripted
model — no GPU needed. Full transcript: `tests/bench_output.txt`.

## Notes

- `classify()` is now a thin wrapper over `classify_many()` (batch of one);
  its behaviour, including the `(None, "CUDA OOM")` fallback and the per-run
  sentence cache, is unchanged.
- The prefetch uses the *same* `tag_text()` + `_needs_model()` decisions and
  the *same* cache keys as `profile()`, so warmed entries are exactly the
  ones the per-pair path looks up — no behavioural drift possible.
- Recommended companion settings from the Node 3 report still apply: load
  unsloth before transformers, keep the Node 3 verdict cache enabled, and
  consider a shared vLLM server for Nodes 3/4/5 next.
