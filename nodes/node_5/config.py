"""
ClauseLens — Node 5
Configuration.

All tunable values for the Impact Agent, Critic Agent and routing live here.
"""

from __future__ import annotations

from dataclasses import dataclass


# ============================================================
# CRITIC VERDICTS
# ============================================================

VERIFIED = "VERIFIED"
HALLUCINATION = "HALLUCINATION"
MISSING_CONTEXT = "MISSING_CONTEXT"
UNKNOWN = "UNKNOWN"

CRITIC_VERDICTS = {VERIFIED, HALLUCINATION, MISSING_CONTEXT}


# ============================================================
# FINAL STATUS (sent to Node 6)
# ============================================================

STATUS_VERIFIED = "VERIFIED"              # Critic approved the summary
STATUS_LOW_CONFIDENCE = "LOW_CONFIDENCE"  # rejected until max_attempts, or Critic unreadable
STATUS_SKIPPED = "SKIPPED"                # too little text to analyse (e.g. a heading only)


# Bump when Node 5 logic or prompts change; checkpoints from older versions are redone
NODE5_VERSION = "5"


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class Node5Config:
    # Impact attempts per pair. Workflow: "max 2 tries", then LOW_CONFIDENCE
    max_attempts: int = 2

    # Characters of clause text per side in a prompt (shrunk further to fit the token budget)
    text_chars: int = 900

    # Prompt + answer must fit in this many tokens
    max_seq_length: int = 2048

    # Generation
    impact_max_new_tokens: int = 200
    critic_max_new_tokens: int = 60
    repetition_penalty: float = 1.15

    # MISSING_CONTEXT: neighbouring chunks added on each side, per level
    context_neighbors: int = 1
    max_context_level: int = 1

    # Word check, two levels (tuned on five Kaggle runs, E2B and E4B summaries):
    #   more than max_unsupported_words    -> HALLUCINATION, Critic not asked
    #                                         (invented passages had 6 and 10 such words)
    #   more than review_unsupported_words -> the Critic is asked to verify those words
    #                                         (every good summary had 0-4)
    max_unsupported_words: int = 5
    review_unsupported_words: int = 2

    # Changed sentences shown to both agents (per side)
    max_changed_sentences: int = 3

    # Pairs whose text (without the heading line) is shorter than this on both
    # sides are not analysed, e.g. a heading-only chunk like "7. DATA"
    min_chars: int = 40

    # True: Impact Agent and Critic Agent run on separate threads (and GPUs)
    # at the same time. Set False when both agents share one model.
    parallel: bool = True

    # Empty the CUDA cache every N generations per agent
    cleanup_every: int = 10
