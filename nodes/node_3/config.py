"""
ClauseLens — Node 3
Configuration.

All tunable values for clause alignment live here.
"""

from __future__ import annotations

from dataclasses import dataclass


# ============================================================
# VERDICTS
# ============================================================

VALID_VERDICTS = {"EQUIVALENT", "MODIFIED", "NO_MATCH"}

# Order used to pick the best match among several candidates
VERDICT_PRIORITY = ["EQUIVALENT", "MODIFIED", "NO_MATCH", "UNKNOWN"]


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

    # Empty CUDA cache every N Gemma calls
    cleanup_every: int = 10

    @property
    def max_input_tokens(self) -> int:
        return self.max_seq_length - self.max_new_tokens - 16
