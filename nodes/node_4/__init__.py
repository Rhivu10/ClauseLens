"""
ClauseLens — Node 4

Deontic cue extractor: regex trigger words + quick Gemma check.
Tags Node 3's changed pairs as Obligation, Permission or Prohibition
and sends them to Node 5.
"""

# ============================================================
# CONFIGURATION
# ============================================================

from .config import (
    Node4Config,
    OBLIGATION,
    PERMISSION,
    PROHIBITION,
    NONE,
    DEONTIC_LABELS,
    LABEL_PRIORITY,
)


# ============================================================
# PATTERNS
# ============================================================

from .patterns import (
    split_sentences,
    find_cues,
    label_sentence,
    tag_text,
)


# ============================================================
# GEMMA CHECK
# ============================================================

from .checker import (
    build_deontic_prompt,
    parse_label,
    DeonticChecker,
)


# ============================================================
# EXTRACTION
# ============================================================

from .extractor import (
    Node4Extractor,
    read_node4_output,
)


# ============================================================
# PUBLIC API
# ============================================================

__all__ = [
    # Configuration
    "Node4Config",
    "OBLIGATION",
    "PERMISSION",
    "PROHIBITION",
    "NONE",
    "DEONTIC_LABELS",
    "LABEL_PRIORITY",

    # Patterns
    "split_sentences",
    "find_cues",
    "label_sentence",
    "tag_text",

    # Gemma check
    "build_deontic_prompt",
    "parse_label",
    "DeonticChecker",

    # Extraction
    "Node4Extractor",
    "read_node4_output",
]
