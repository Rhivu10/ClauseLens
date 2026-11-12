"""
ClauseLens — Node 4 (performance-tuned)
Configuration.

All tunable values for deontic cue extraction live here.

New in this version (all have safe defaults; turn them off to recover the old
behaviour exactly):

  batch_size            sentences verified per model.generate call
                        (1 = old one-sentence-at-a-time behaviour)
  prefetch_pairs        pairs per prefetch window in iter_run(): all sentences
                        a window needs are batch-classified up front, so
                        enrich_pair() finds warm cache entries and makes no
                        model calls (1 = old behaviour)
  stop_at_json          stop generation at the first '}' — the answer JSON is
                        complete there, so trailing tokens are never generated
                        (same parsed label, ~3x fewer tokens)
  oom_split             on CUDA out-of-memory, split the batch in half and
                        retry (down to single sentences) instead of losing
                        the confirmation
  cache_path            optional JSONL file with persisted verdicts, so a
                        crashed or repeated run resumes instead of
                        re-verifying every sentence

The deontic prompt, the regex cue rules and every output field are unchanged;
NODE4_VERSION is untouched, so existing Node 4 checkpoint files stay valid.
"""

from __future__ import annotations

from dataclasses import dataclass


# ============================================================
# LABELS
# ============================================================

OBLIGATION = "OBLIGATION"
PERMISSION = "PERMISSION"
PROHIBITION = "PROHIBITION"
NONE = "NONE"

DEONTIC_LABELS = {OBLIGATION, PERMISSION, PROHIBITION}
VALID_MODEL_LABELS = DEONTIC_LABELS | {NONE}

# Used when one sentence carries cues for several labels
LABEL_PRIORITY = [PROHIBITION, OBLIGATION, PERMISSION, NONE]

# Bump when Node 4 logic changes; checkpoints from older versions are redone
NODE4_VERSION = "5"

MODIFIED = "MODIFIED"
ADDED = "ADDED"
DELETED = "DELETED"
# Node 3 could not decide (verdict UNKNOWN). Passed on, not dropped, so
# Node 5's critic can check it.
UNVERIFIED = "UNVERIFIED"

# Bumped whenever the prompt or parsing changes, so a stale verdict cache
# cannot silently return labels made by an older prompt.
PROMPT_VERSION = "v4.1"


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class Node4Config:
    # When to ask Gemma to confirm a sentence label:
    #   "ambiguous" - only weak cues or mixed labels (default, fast)
    #   "all"       - every sentence that has a cue
    #   "off"       - regex only, no model needed
    model_check: str = "ambiguous"

    # Sentence character limit sent to Gemma
    sentence_chars: int = 600

    # Gemma generation
    max_new_tokens: int = 40

    # Empty CUDA cache every N Gemma BATCHES (was: every N calls)
    cleanup_every: int = 10

    # -------------------- performance tuning --------------------

    # Sentences verified per model.generate call. 1 disables batching.
    # Deontic answers are tiny JSON ({"label": "..."}), so a batch can be
    # larger than Node 3's clause comparisons.
    batch_size: int = 16

    # Pairs per prefetch window in iter_run(). All model-needed sentences of
    # the window are classified in batches BEFORE the pairs are enriched, so
    # the per-pair path runs on warm cache entries only.
    prefetch_pairs: int = 8

    # Stop generation once every row has closed its answer JSON ('}').
    # The parser only reads the first {...} block, so the parsed label is
    # identical while generated tokens drop from up to 40 to ~10-15.
    stop_at_json: bool = True

    # On CUDA out-of-memory during a batch, split it in half and retry.
    oom_split: bool = True

    # Optional JSONL cache of sentence verdicts for crash-safe resume and
    # instant re-runs. None disables it.
    cache_path: str | None = None

    def __post_init__(self):
        if self.model_check not in ("ambiguous", "all", "off"):
            raise ValueError(
                f"model_check must be 'ambiguous', 'all' or 'off', got '{self.model_check}'."
            )
