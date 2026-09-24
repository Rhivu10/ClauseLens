"""
ClauseLens — Node 4
Configuration.

All tunable values for deontic cue extraction live here.
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


# ============================================================
# CHANGE STATUS (from Node 3)
# ============================================================

# Bump when Node 4 logic changes; checkpoints from older versions are redone
NODE4_VERSION = "2"


MODIFIED = "MODIFIED"
ADDED = "ADDED"
DELETED = "DELETED"


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

    # Empty CUDA cache every N Gemma calls
    cleanup_every: int = 10

    def __post_init__(self):
        if self.model_check not in ("ambiguous", "all", "off"):
            raise ValueError(
                f"model_check must be 'ambiguous', 'all' or 'off', got '{self.model_check}'."
            )
