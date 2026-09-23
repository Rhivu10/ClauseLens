"""
ClauseLens — Node 3

Clause alignment: Node 2 retrieval + similarity gate +
fine-tuned Gemma verification.
"""

# ============================================================
# CONFIGURATION
# ============================================================

from .config import (
    Node3Config,
    VALID_VERDICTS,
    VERDICT_PRIORITY,
)


# ============================================================
# HELPERS
# ============================================================

from .helpers import (
    normalize_text,
    extract_numbers,
    numeric_diff,
    parse_verdict,
)


# ============================================================
# GEMMA VERIFIER
# ============================================================

from .verifier import (
    load_gemma_with_adapter,
    build_compact_prompt,
    GemmaVerifier,
)


# ============================================================
# ALIGNMENT
# ============================================================

from .aligner import (
    Node3Aligner,
    show,
)


# ============================================================
# PUBLIC API
# ============================================================

__all__ = [
    # Configuration
    "Node3Config",
    "VALID_VERDICTS",
    "VERDICT_PRIORITY",

    # Helpers
    "normalize_text",
    "extract_numbers",
    "numeric_diff",
    "parse_verdict",

    # Gemma verifier
    "load_gemma_with_adapter",
    "build_compact_prompt",
    "GemmaVerifier",

    # Alignment
    "Node3Aligner",
    "show",
]
