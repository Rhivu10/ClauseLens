"""
ClauseLens — Node 3 (performance-tuned)
Configuration.

All tunable values for clause alignment live here.

New in this version (all have safe defaults; set them to recover the old
behaviour):

  batch_size              pairs verified per model.generate call (1 = old
                          sequential behaviour)
  stop_at_json            stop generation at the first '}' — the answer JSON
                          is complete there, so trailing tokens are never
                          generated (same parsed verdict, ~3x fewer tokens)
  early_exit_equivalent   when the top-scored candidate is EQUIVALENT, the
                          remaining candidates cannot beat it, so they are
                          never sent to the model
  use_precomputed_embeddings
                          reuse the Node 2 embedding matrix instead of
                          re-encoding every source chunk and scanning the
                          whole index per query
  oom_split               on CUDA out-of-memory, split the batch in half and
                          retry (down to single pairs) instead of dropping
                          the verdict
  dedup_text_pairs        identical (source, target) text pairs are verified
                          once per run (repeated boilerplate clauses)
  cache_path              optional JSONL file with persisted verdicts, so a
                          crashed or repeated run resumes instead of
                          re-verifying every pair
"""

from __future__ import annotations

from dataclasses import dataclass


# ============================================================
# VERDICTS
# ============================================================

VALID_VERDICTS = {"EQUIVALENT", "MODIFIED", "NO_MATCH"}

# Order used to pick the best match among several candidates
VERDICT_PRIORITY = ["EQUIVALENT", "MODIFIED", "NO_MATCH", "UNKNOWN"]

# Bumped whenever the prompt or parsing changes, so a stale cache file
# cannot silently return verdicts made by an older prompt.
PROMPT_VERSION = "v3.1"


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class Node3Config:
    # Number of target candidates verified by Gemma per source chunk
    candidate_k: int = 3

    # TENTATIVE — calibrate with Node3Aligner.best_scores()
    similarity_threshold: float = 0.55

    # At or above this score (or with the same heading) two chunks are treated
    # as the same clause: Gemma may answer EQUIVALENT or MODIFIED, not NO_MATCH
    structural_score: float = 0.80

    # SAME character limit for source and target text
    text_chars: int = 1200

    # Gemma generation
    max_seq_length: int = 1024
    max_new_tokens: int = 80

    # Empty CUDA cache every N Gemma BATCHES (was: every N calls)
    cleanup_every: int = 10

    # -------------------- performance tuning --------------------

    # Pairs verified per model.generate call. 1 disables batching.
    batch_size: int = 8

    # Stop generation once every row has closed its answer JSON ('}').
    # The parser only reads the first {...} block, so the parsed verdict is
    # identical while generated tokens drop from up to 80 to ~20-25.
    stop_at_json: bool = True

    # Skip candidates 2..k when the top-scored candidate is EQUIVALENT.
    # Nothing can beat EQUIVALENT and ties break by score order, so the
    # final verdict and matched_chunk_id are unchanged.
    early_exit_equivalent: bool = True

    # Reuse Node 2 embeddings for retrieval instead of re-encoding each
    # source chunk and scanning the whole index. Scores are the same
    # cosine similarities the FAISS flat search would return.
    use_precomputed_embeddings: bool = True

    # On CUDA out-of-memory during a batch, split it in half and retry.
    oom_split: bool = True

    # Verify identical (source, target) text pairs once per run.
    dedup_text_pairs: bool = True

    # Optional JSONL cache of (source_chunk_id, candidate_chunk_id) verdicts
    # for crash-safe resume and instant re-runs. None disables it.
    cache_path: str | None = None

    @property
    def max_input_tokens(self) -> int:
        return self.max_seq_length - self.max_new_tokens - 16
