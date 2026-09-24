"""
ClauseLens — Node 4
Pattern-based deontic cue extraction.

    Chunk text
        |
        +----> sentences (with character offsets into the chunk text)
                   |
                   +----> ordered regex cues
                              |
                              +----> sentence label: OBLIGATION | PERMISSION | PROHIBITION

Cue rules run in order. Each match masks its span, so "shall not"
is counted once as a prohibition and never again as "shall".

No model or GPU dependency.
"""

from __future__ import annotations

import re
from typing import Any

from .config import (
    OBLIGATION,
    PERMISSION,
    PROHIBITION,
    LABEL_PRIORITY,
)


# ============================================================
# CUE RULES
# ============================================================

# (label, strength, pattern). label None = masked, not a duty
# (definitions such as "shall mean"). Order matters.
_RULES: list[tuple[str | None, str, str]] = [
    # Definitions and deeming — not duties
    (None, "strong", r"\bshall\s+(?:mean|have\s+the\s+meanings?|include|be\s+deemed|refer\s+to)\b"),

    # Exemptions: "shall not be required to" grants freedom, not a prohibition
    (PERMISSION, "strong", r"\b(?:shall|will|is|are)\s+not\s+(?:be\s+)?(?:required|obligated|obliged)\s+to\b"),

    # Prohibitions
    (PROHIBITION, "strong", r"\b(?:shall|must|will)\s+not\b"),
    (PROHIBITION, "strong", r"\bmay\s+not\b"),
    (PROHIBITION, "strong", r"\bin\s+no\s+event\s+(?:shall|will|may)\b"),
    (PROHIBITION, "strong", r"\b(?:neither|no)\s+party\s+(?:shall|will|may)\b"),
    (PROHIBITION, "strong", r"\bshall\s+(?:never|refrain\s+from)\b"),
    (PROHIBITION, "strong", r"\b(?:is|are)\s+(?:strictly\s+)?prohibited\b"),
    (PROHIBITION, "strong", r"\bprohibit(?:s|ed|ing)?\b"),
    (PROHIBITION, "strong", r"\b(?:is|are)\s+not\s+(?:permitted|allowed|entitled|authorized)\s+to\b"),
    (PROHIBITION, "weak", r"\bcannot\b"),

    # Permissions (phrases that contain "shall"/"will" come before obligations)
    (PERMISSION, "strong", r"\b(?:shall|will)\s+(?:have|retain)\s+the\s+right\s+to\b"),
    (PERMISSION, "strong", r"\b(?:has|have|retains?)\s+the\s+right\s+to\b"),
    (PERMISSION, "strong", r"\b(?:shall\s+be|is|are)\s+(?:entitled|permitted|authorized|allowed)\s+to\b"),
    (PERMISSION, "strong", r"\bat\s+(?:its|their|his|her)\s+(?:sole\s+)?(?:option|discretion)\b"),

    # Obligations
    (OBLIGATION, "strong", r"\bshall\b"),
    (OBLIGATION, "strong", r"\bmust\b"),
    (OBLIGATION, "strong", r"\b(?:is|are)\s+(?:required|obligated|obliged)\s+to\b"),
    (OBLIGATION, "strong", r"\bundertakes?\s+to\b"),
    (OBLIGATION, "strong", r"\bcovenants?\s+(?:to|that)\b"),
    (OBLIGATION, "weak", r"\bagrees?\s+(?:to|that)\b"),
    (OBLIGATION, "weak", r"\b(?:is|are)\s+responsible\s+for\b"),
    (OBLIGATION, "weak", r"\bwill\b"),

    # Weak permissions
    (PERMISSION, "strong", r"\bmay\b"),
    (PERMISSION, "weak", r"\bcan\b"),
]

_COMPILED = [(label, strength, re.compile(p, re.I)) for label, strength, p in _RULES]


# ============================================================
# SENTENCES
# ============================================================

# Boundary after . ; or : followed by whitespace and a capital, quote or bracket
_BOUNDARY_RE = re.compile(r"(?<=[.;:])\s+(?=[A-Z(\"'])")

# A period after these is not a sentence end
_ABBREVIATIONS = {
    "inc", "co", "corp", "ltd", "llc", "no", "nos", "sec", "secs", "art", "para",
    "mr", "mrs", "ms", "dr", "st", "vs", "etc", "e.g", "i.e", "u.s", "u.k", "a.m", "p.m",
}


def _is_abbreviation(text: str, boundary: int) -> bool:
    word = text[:boundary].rsplit(None, 1)[-1] if text[:boundary].strip() else ""
    return word.endswith(".") and (word[:-1].lower() in _ABBREVIATIONS or len(word) == 2)


def split_sentences(text: str) -> list[dict[str, Any]]:
    """Split chunk text into sentences with start/end offsets into the original text."""
    spans, start = [], 0
    for m in _BOUNDARY_RE.finditer(text):
        if _is_abbreviation(text, m.start()):
            continue
        spans.append((start, m.start()))
        start = m.end()
    spans.append((start, len(text)))

    sentences = []
    for s, e in spans:
        raw = text[s:e]
        lead = len(raw) - len(raw.lstrip())
        stripped = raw.strip()
        if stripped:
            sentences.append({"text": stripped, "start": s + lead, "end": s + lead + len(stripped)})
    return sentences


# ============================================================
# CUES
# ============================================================

def find_cues(sentence: str) -> list[dict[str, Any]]:
    """Ordered, non-overlapping deontic cues in one sentence."""
    work = sentence
    cues = []
    for label, strength, rx in _COMPILED:
        for m in rx.finditer(work):
            if label is not None:
                cues.append({
                    "cue": " ".join(sentence[m.start():m.end()].lower().split()),
                    "label": label,
                    "strength": strength,
                    "start": m.start(),
                })
            # Mask the span so later rules can't reuse it
            work = work[:m.start()] + " " * (m.end() - m.start()) + work[m.end():]
    cues.sort(key=lambda c: c["start"])
    return cues


def label_sentence(cues: list[dict[str, Any]]) -> tuple[str, bool]:
    """
    Returns (label, ambiguous).
    Ambiguous = cues for several labels, or only weak cues.
    """
    labels = {c["label"] for c in cues}
    label = min(labels, key=LABEL_PRIORITY.index)
    ambiguous = len(labels) > 1 or all(c["strength"] == "weak" for c in cues)
    return label, ambiguous


def tag_text(text: str) -> list[dict[str, Any]]:
    """
    Regex pass over one chunk. Returns only sentences that carry a cue:

        {"text", "start", "end", "cues", "label", "ambiguous", "decided_by": "regex"}
    """
    tagged = []
    for sent in split_sentences(text):
        cues = find_cues(sent["text"])
        if not cues:
            continue
        label, ambiguous = label_sentence(cues)
        tagged.append({
            **sent,
            "cues": [{k: v for k, v in c.items() if k != "start"} for c in cues],
            "label": label,
            "ambiguous": ambiguous,
            "decided_by": "regex",
        })
    return tagged
