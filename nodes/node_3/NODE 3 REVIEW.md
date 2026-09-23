# ClauseLens Notebook: Old vs New

Scope: this covers the notebook only (`clauselens_notebook.py`): the Node 3 clause-alignment code, its tests, and the cells around them. Node 1 (`node_1.py`) was patched separately and is mentioned only where it affects the notebook's results.

---

## 1. Summary

| Area | Old notebook | New notebook |
|---|---|---|
| Tests 2 and 3 | Used `document_b` chunks against `document_b`, which is a self-match | Source is always `document_a`, target `document_b`; a guard raises an error on self-match |
| Empty result `[]` | Meant three different things | Explicit `status` field: `NO_CANDIDATES`, `BELOW_THRESHOLD` or `COMPARED` |
| Retrieval | `top_k=20`, then filtered to the target document | Searches the whole index, then filters |
| Text given to Gemma | 350 chars (source) vs 500 (target), cut blindly | Same limit both sides (1,200 chars), shrunk to the token budget, `truncated` flag |
| Threshold | 0.45, uncalibrated | 0.55 (tentative) plus a calibration cell that prints score percentiles |
| Identical clauses | Sent to Gemma | Exact-text shortcut, no model call |
| Output parsing | Strict `json.loads`; anything else gave `UNKNOWN` | Tolerant parser; contradictions (`NO_MATCH` plus a change type) fixed |
| `confidence` | Emitted by the model, e.g. `1.0` | Removed |
| GPU cleanup | `gc.collect()` + `empty_cache()` after every call | `try/finally`, cache cleared every 10 calls, OOM handled |
| Whole-document run | None (one clause at a time) | `align_documents()` with ADDED and REMOVED lists and progress/ETA |
| Environment | `PYTORCH_CUDA_ALLOC_CONF` set after CUDA was initialized | Set at the top of cell 1, before torch loads |
| Repo clone | Never updated an existing clone | Runs `git pull` if the folder exists |

---

## 2. Drawbacks of the old code

### 2.1 Correctness problems

**Tests 2 and 3 compared a document to itself.**
The tests took chunks `document_b_0002` and `document_b_0003` and aligned them against `target_document_id="document_b"`. Those source chunks are in the index, so each is retrieved as its own top match with a score near 1.0. A self-match tells you nothing about how the pipeline handles two different clauses. The labels ("GOVERNING LAW", "UNMATCHED CLAUSE") were also assumed from chunk numbers, never checked against the text.

**Test 1's label was unverified.**
`document_a_0001` was labeled "INDEMNIFICATION", but chunk 0001 is the first chunk of the document, which is normally a title or preamble.

**The Test 3 comment was false.**
The print statement said "no Gemma call was made", but the run time (about 4.7 s, the same as a single generation) showed Gemma did run.

**`[]` was ambiguous.**
`align_source_clause` returned `[]` for "nothing retrieved", "everything below threshold" and, implicitly, "real no-match". The caller couldn't tell these apart.

**Retrieval cap caused false NO_MATCH risk.**
`semantic_search(..., top_k=20)` searches both documents together, and the code then kept only the target's chunks. In a larger contract, the top 20 can be mostly the source document's own neighbors, leaving zero or one target chunk. That returns `[]`, which reads as "no match" even when the clause exists.

**Blind, asymmetric truncation.**
Source text was cut at 350 characters and target text at 500. A change in an amount, date or condition after the cutoff was invisible, so real modifications could come back `EQUIVALENT`. Comparing 350 characters against 500 is also not a like-for-like comparison.

**Uncalibrated similarity gate.**
The 0.45 threshold was picked without looking at scores. The old Test 2/3 output showed a candidate scoring 0.4507, which cleared the gate by 0.0007. A gate that a near-tie can pass isn't separating related from unrelated clauses.

**Contradictory, unreliable model output.**
The old output was `{'alignment_result': 'NO_MATCH', 'change_type': 'REPLACE', 'confidence': 1.0}`.
- `NO_MATCH` with a `change_type` is contradictory, since nothing was matched, so nothing was replaced.
- `REPLACE` was not in the prompt's schema.
- `confidence: 1.0` from a 2B model under greedy decoding is not a real confidence.

**Brittle JSON parsing.**
`json.loads(raw)` fails if the model wraps the answer in a code fence, adds a sentence, echoes the schema, or is cut off at the 64-token limit. Every one of those became `UNKNOWN`.

### 2.2 Inefficiency

- **Cleanup after every call.** `gc.collect()` plus `torch.cuda.empty_cache()` after each generation forces the CUDA allocator to release and re-request memory every time. That slows generation and doesn't reduce peak memory, because the peak happens during generation.
- **The "memory optimizations" mostly targeted the wrong thing.** Capping `top_k` at 20 saved almost nothing, because a flat FAISS search over a few thousand vectors is cheap. The real memory cost was Gemma, and the fix for that was the shorter input (which then made accuracy worse through truncation).
- **Gemma was used where a string comparison would do.** Identical clauses went to the model.
- **Two candidates per source clause meant up to two 4-5 s Gemma calls per clause.** From the notebook timestamps, one call took about 4.7 s and Test 2 (two candidates) about 9 s. Most of that was spent on candidates that weren't good matches.
- **No driver.** Only single-clause calls existed. There was no loop over a document, no progress reporting, and no ADDED or REMOVED detection.
- **Environment mistakes.** `PYTORCH_CUDA_ALLOC_CONF` was set after CUDA was already initialized, so it did nothing. The clone cell never pulled, so a stale copy of the repo could run silently.

---

## 3. Why the old outputs were wrong (or meaningless)

1. **Tests 2 and 3 were self-matches.** For an identical chunk the correct verdict is `EQUIVALENT` with a score near 1.0. That result would look like success while testing nothing.
2. **The pasted old output didn't fit either test.** Both tests printed the same candidate (`document_b_0004`) with the identical score to 16 digits (`0.4506896138191223`). Two different source chunks cannot do that, so the same source variable or the same cell output was reused. Those results can't be trusted as evidence for either test.
3. **The gate passed by accident.** 0.4507 against 0.45 means Gemma ran on a candidate the embeddings barely favored. Even if the final `NO_MATCH` was correct, it came from the model, not from the retrieval gate the code was designed around.
4. **The verdict was internally inconsistent.** A `NO_MATCH` carrying `REPLACE` and a confidence of `1.0` is not a usable result.
5. **Labels didn't match content.** "Indemnification" was chunk 0001, "governing law" was chunk 0002, and "unmatched clause" was chunk 0003. None of these was verified against the text.

---

## 4. Changes made, cell by cell

**Cell 1 (install).** Kept your original install logic. Added `PYTORCH_CUDA_ALLOC_CONF` before torch loads.

**Cell 5.** `MAX_SEQ_LENGTH = 1024` is now a variable, so Node 3 can compute its token budget from it.

**Cell 9 (repo).** Pulls the latest code when the clone exists.

**Cell 13 (Node 1 call).** Passes `start_new_chunk_on_level=2` and `max_chars=1500` explicitly.

**Cell 14 (new): Node 1 quality check.** Prints chunk-length percentiles, counts of tiny and over-limit chunks, and the first chunks with their headings. Run this before trusting anything downstream.

**Cell 17 (new): score sanity check.** Confirms results have a `score` key, that scores are descending, and that they behave like cosine similarity (at most 1.0). A warning here means the gate is invalid.

**Cells 18-19: config and helpers.**
- `NODE3_CANDIDATE_K` 2 to 3, threshold 0.45 to 0.55 (tentative), one shared text limit of 1,200 characters.
- `normalize_text` collapses PDF line breaks.
- `numeric_diff` compares amounts, dates and counts on the full text and attaches the difference as a flag.
- `parse_verdict` handles code fences and extra text, and clears `change_type` unless the verdict is `MODIFIED`.

**Cell 20: retrieval.** Raises `ValueError` on a same-document source and target, searches the whole index, then filters. Raises `KeyError` if the score key is missing.

**Cell 21: Gemma call.**
- Prompt says to use `NO_MATCH` only if the clauses cover different subjects, and drops `confidence`.
- `build_inputs` shrinks both sides equally until the prompt fits the token budget.
- Cleanup runs in a `finally` block, with `empty_cache()` every 10 calls.
- `compare_pair` skips Gemma for identical text and catches CUDA out-of-memory errors.

**Cell 22: pipeline.** Returns a dict with `status`, `verdict`, `best_score`, `matched_chunk_id` and `matches`. A real match (`EQUIVALENT` or `MODIFIED`) is preferred over `NO_MATCH` even when it isn't the top-scored candidate. A `show()` helper prints the result.

**Cell 23 (new): `align_documents()`.** Loops over all source chunks with a progress and ETA printout, and reports `removed` (source clauses with no match) and `added` (target clauses no source matched). It is greedy: several source clauses can map to one target clause.

**Cell 24 (new): calibration.** Prints the best target score for every source chunk with percentiles and the lowest and highest chunks, so the threshold can be chosen from data.

**Cells 25-29: tests.**
- Test 0 checks that the self-match guard fires.
- Test 1 finds an "indemnif..." chunk in `document_a` by keyword.
- Test 2 finds a governing-law chunk in `document_a` by keyword.
- Test 3 automatically picks the lowest-scoring `document_a` chunk. Each test prints the source ID, length and text so you can check what was compared.

**Cell 30 (new).** Optional full-document run behind a `RUN_FULL` flag.

---

## 5. What your latest run shows

| Test | Result | Reading |
|---|---|---|
| 1 | `BELOW_THRESHOLD`, score 0.4994 | Gate worked, but the source was a limitation-of-liability clause that only mentions indemnification, so the test target was mis-scoped |
| 2 | `BELOW_THRESHOLD`, score 0.4504 | Source chunk was 4,710 chars (a whole MISCELLANEOUS section), so its embedding is diluted; not a real governing-law comparison |
| 3 | `BELOW_THRESHOLD`, score 0.1506 | Correct: an unmatched clause is rejected with no Gemma call |
| Full run | 29 `NO_MATCH`, 1 `EQUIVALENT` in 86 s | Plausible for two unrelated contracts, but it doesn't test the `MODIFIED` path |

The Test 2 chunk size and the footer text on Test 3's source (`9 Source: ALLIED ESPORTS ENTERTAINMENT, INC., 8-K, 8/15/2019`) both indicate the run used the old Node 1, not the patched one. The notebook fixes work, but chunking quality still depends on Node 1.

The one `EQUIVALENT` (`document_a_0012` to `document_b_0005`) needs a manual check. Two unrelated contracts should not have an equivalent clause, so it may be a shared heading or boilerplate line matched by the exact-text shortcut.

---

## 6. Still unverified

- **Node 2 internals.** Score semantics are checked at runtime by cell 17, but the embedding model's maximum sequence length and the graph build cost are not visible.
- **The 0.55 threshold** is a placeholder until cell 24 is run on labeled related and unrelated pairs.
- **The `MODIFIED` path** has not been exercised. The best test is to run two identical copies of the same PDF (everything should be `EQUIVALENT`), then edit a few amounts, dates and party names in one copy and confirm those come back `MODIFIED`.
- **Speed on large documents.** The full run took 86 s for 30 chunks, mostly because few chunks passed the gate. A document pair with many real matches will spend about 4-5 s per Gemma call.
- **Gemma adapter behavior.** The LoRA adapter was trained on CUAD extraction QA, not on this comparison prompt, so its accuracy on `MODIFIED` versus `EQUIVALENT` needs measuring on labeled pairs.
