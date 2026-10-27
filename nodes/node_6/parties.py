"""
ClauseLens — Node 6
Stakeholder names: one clean name per party, so the report can group changes
by who they affect.

Node 5's AFFECTED lists vary ("licensor", "Licensors", "Licensee / buyer",
"both parties", "INC"). normalize_party() turns them into one form:

    "the Licensees"     -> "Licensee"
    "Licensee / buyer"  -> "Licensee"
    "either party"      -> "Both parties"
    "INC", "LLC"        -> dropped (company-name fragments)

No model or GPU dependency.
"""

from __future__ import annotations

import re

BOTH = "Both parties"

_BOTH_RE = re.compile(r"^(?:both|each|either|neither|all|the)?\s*part(?:y|ies)(?:\s+involved)?$", re.I)
_FRAGMENTS = {"inc", "llc", "ltd", "corp", "co", "plc", "gmbh", "sa", "nv", "bv", "pvt"}


def _singular(word: str) -> str:
    w = word.strip()
    low = w.lower()
    if low.endswith("ies") and len(low) > 4:
        return w[:-3] + "y"
    if low.endswith("s") and not low.endswith("ss") and len(low) > 3:
        return w[:-1]
    return w


def normalize_party(name: str) -> str | None:
    """A clean party name, or None for fragments that name nobody."""
    n = re.sub(r"[\[\]<>{}*\"“”]", "", name or "").strip(" .,;:-")
    n = re.sub(r"^(?:the|all|any|its|their)\s+", "", n, flags=re.I)
    if not n:
        return None
    if _BOTH_RE.match(n):
        return BOTH
    n = re.split(r"\s*/\s*", n)[0]                       # "Licensee / buyer" -> "Licensee"
    if n.lower().strip(".") in _FRAGMENTS:
        return None
    words = n.split()
    if len(words) == 1:
        n = _singular(n)                                 # "Licensees" -> "Licensee"
    # Multi-word names are company names: "Echo Therapeutics" stays as it is
    # Title-case lowercase names ("licensor" -> "Licensor"); keep names already capitalised
    return n[:1].upper() + n[1:] if n.islower() else n


def normalize_parties(names: list[str]) -> list[str]:
    out: list[str] = []
    for name in names or []:
        n = normalize_party(name)
        if n and n not in out:
            out.append(n)
    return out
