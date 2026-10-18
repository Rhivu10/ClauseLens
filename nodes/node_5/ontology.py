"""
ClauseLens — Node 5
CUAD ontology and party detection.

    clause text
        |
        +----> detect_categories()  CUAD category hints (keyword match, cuad_ontology.json)
        |           |
        |           +----> stakeholders_for()  who these categories usually affect
        |
        +----> find_parties()  party roles actually named in the text (Licensee, Vendor, ...)

These are HINTS for the Impact Agent and the grounding checks, not facts.
No model or GPU dependency.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

ONTOLOGY_PATH = Path(__file__).with_name("cuad_ontology.json")


# ============================================================
# CUAD CATEGORIES
# ============================================================

@lru_cache(maxsize=4)
def load_ontology(path: str | None = None) -> dict[str, dict[str, Any]]:
    """{category: {"keywords": [...], "stakeholders": [...]}}"""
    with open(path or ONTOLOGY_PATH, encoding="utf-8") as fh:
        return json.load(fh)["categories"]


def detect_categories(text: str, top_k: int = 3, path: str | None = None) -> list[dict[str, Any]]:
    """Best-matching CUAD categories by keyword hits: [{"category", "matched"}], most hits first."""
    low = " ".join(text.lower().split())
    hits = []
    for name, info in load_ontology(path).items():
        matched = [k for k in info["keywords"] if k in low]
        if matched:
            hits.append({"category": name, "matched": matched})
    hits.sort(key=lambda h: -len(h["matched"]))
    return hits[:top_k]


def stakeholders_for(categories: list[dict[str, Any]], path: str | None = None) -> list[str]:
    """Stakeholders linked to the detected categories, in order, without duplicates."""
    onto = load_ontology(path)
    out: list[str] = []
    for c in categories:
        for s in onto[c["category"]]["stakeholders"]:
            if s not in out:
                out.append(s)
    return out


# ============================================================
# PARTIES NAMED IN THE TEXT
# ============================================================

# Role words that name a party in contracts and policies (singular form)
PARTY_ROLES = [
    "licensor", "licensee", "sublicensee", "company", "employer", "employee", "customer", "client",
    "vendor", "supplier", "distributor", "reseller", "contractor", "subcontractor", "consultant",
    "buyer", "seller", "purchaser", "provider", "service provider", "user", "member", "partner",
    "lender", "borrower", "landlord", "tenant", "franchisor", "franchisee", "agent", "manufacturer",
    "affiliate", "subsidiary", "subsidiaries", "third party", "third parties", "shareholder", "investor",
]

# "each party", "either party", "neither party", "the parties" all mean both parties
_BOTH_RE = re.compile(r"\b(?:each|either|neither|both)\s+part(?:y|ies)\b|\bthe\s+parties\b", re.I)


def _singular(word: str) -> str:
    w = word.lower().strip()
    if w.endswith("ies") and w not in ("third parties",):
        return w[:-3] + "y"
    if w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def find_parties(text: str) -> list[str]:
    """Party roles named in the text, e.g. ["Licensee", "Licensor", "Both parties"]."""
    low = text.lower()
    found = []
    for role in PARTY_ROLES:
        if re.search(rf"\b{re.escape(role)}(?:s|es)?\b", low):
            name = role.title()
            if name not in found:
                found.append(name)
    if _BOTH_RE.search(text) and "Both parties" not in found:
        found.append("Both parties")
    return found


def party_is_grounded(party: str, text: str, allowed: list[str]) -> bool:
    """
    True when a party named by the Impact Agent appears in the text, or is one of
    the allowed stakeholder hints. Handles plurals and "the" ("the Licensees").
    """
    p = re.sub(r"^(?:the|all|any|its)\s+", "", party.strip().lower())
    if not p:
        return False
    generic = {"both parties", "either party", "each party", "the parties", "parties", "neither party"}
    if p in generic:
        return True
    # "Licensor/provider" (no spaces) and "Licensor / provider" both split into their parts
    split = lambda s: [x.strip() for x in [s] + re.split(r"\s*/\s*", s) if x.strip()]
    forms = {f for piece in split(p) for f in (piece, _singular(piece))}
    # Allowed names and their singular forms: "Licensee / buyer" -> licensee, buyer.
    # (Kaggle run 3: "all licensees" was rejected against the allowed "Licensee".)
    allowed_forms = set()
    for a in allowed:
        for piece in split(a.lower()):
            allowed_forms |= {piece, _singular(piece)}
    if forms & allowed_forms:
        return True
    low = text.lower()
    for form in forms:
        if re.search(rf"\b{re.escape(form)}(?:s|es)?\b", low):
            return True
    return False
