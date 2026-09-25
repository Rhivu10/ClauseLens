"""
ClauseLens — Node 3
Helpers: text normalization, numeric diff, verdict parser.

No model or GPU dependency.
"""

from __future__ import annotations

import json
import re
from typing import Any

from .config import VALID_VERDICTS


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text: str) -> str:
    """Collapse PDF hard line breaks and repeated whitespace."""
    return re.sub(r"\s+", " ", text).strip()


# ============================================================
# NUMERIC DIFF
# ============================================================

_NUM_RE = re.compile(r"\$?\d[\d,]*(?:\.\d+)?%?")


def extract_numbers(text: str) -> set[str]:
    return {m.group(0).rstrip(",.") for m in _NUM_RE.finditer(text)}


def numeric_diff(source_text: str, target_text: str) -> dict[str, list[str]]:
    """
    Deterministic check on the FULL text (not the truncated prompt): amounts,
    dates and counts present on only one side. Not shown to Gemma; it is
    attached to the result as a flag.
    """
    s, t = extract_numbers(source_text), extract_numbers(target_text)
    return {"only_in_source": sorted(s - t), "only_in_target": sorted(t - s)}


# ============================================================
# VERDICT PARSER
# ============================================================

def parse_verdict(raw: str) -> dict[str, Any]:
    """Tolerant parser: code fences, extra prose, echoed schema, contradictions."""
    m = re.search(r"\{.*?\}", raw, re.S)
    if m:
        try:
            d = json.loads(m.group(0))
            verdict = str(d.get("alignment_result", "")).strip().upper()
            if verdict in VALID_VERDICTS:
                change_type = str(d.get("change_type", "") or "").strip()
                if change_type.upper() == "<CHANGE>":
                    change_type = ""     # placeholder echoed from the prompt
                if verdict != "MODIFIED":
                    change_type = ""     # NO_MATCH/EQUIVALENT cannot have a change type
                return {"alignment_result": verdict, "change_type": change_type}
        except json.JSONDecodeError:
            pass
    return {"alignment_result": "UNKNOWN", "change_type": "", "raw": raw}
