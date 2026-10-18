"""
ClauseLens — Node 5
Prompts and parsers for the Impact Agent and the Critic Agent.

Answers are plain text, not JSON: a small model follows "2-3 sentences, then
one AFFECTED: line" and "one word on the first line" far more reliably than a
JSON schema (Nodes 3 and 4 saw Gemma copy schemas and loop inside JSON).

Every prompt builder takes `limit`, the characters of clause text per side,
so the agent can shrink it until the prompt fits the token budget.
No model or GPU dependency.
"""

from __future__ import annotations

import re
from typing import Any

from .config import CRITIC_VERDICTS, UNKNOWN


_STATUS_TEXT = {
    "MODIFIED": "MODIFIED - the clause exists in both versions and was changed",
    "ADDED": "ADDED - the clause is new in the new version",
    "DELETED": "DELETED - the clause was removed in the new version",
    "UNVERIFIED": "UNCLEAR - this old clause may have been removed or changed; no match was confirmed",
}


def _cut(text: str | None, limit: int) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + " ..."


def _changed_block(view: dict[str, Any], limit: int) -> str:
    """The exact sentences that differ, OLD first, so the change itself is unmissable."""
    old, new = view.get("changed_old") or [], view.get("changed_new") or []
    if not old and not new:
        return ""
    lines = ["WHAT CHANGED (only these sentences differ):"]
    lines += [f"- OLD: {_cut(s, limit // 2)}" for s in old] or ["- OLD: (nothing)"]
    lines += [f"- NEW: {_cut(s, limit // 2)}" for s in new] or ["- NEW: (removed)"]
    return "\n".join(lines) + "\n\n"


def _texts_block(view: dict[str, Any], limit: int) -> str:
    parts = []
    if view["source_text"]:
        label = "OLD VERSION" if view["target_text"] else "OLD VERSION (not in the new version)"
        parts.append(f"{label}:\n{_cut(view['source_text'], limit)}")
    if view["target_text"]:
        label = "NEW VERSION" if view["source_text"] else "NEW VERSION (not in the old version)"
        parts.append(f"{label}:\n{_cut(view['target_text'], limit)}")
    if view.get("context_text"):
        parts.append(f"SURROUNDING TEXT (context only):\n{_cut(view['context_text'], limit)}")
    return "\n\n".join(parts)


# ============================================================
# IMPACT AGENT
# ============================================================

def build_impact_prompt(view: dict[str, Any], limit: int, warning: str | None = None) -> str:
    facts = [f"CHANGE: {_STATUS_TEXT.get(view['status'], view['status'])}"]
    if view.get("heading"):
        facts.append(f"SECTION: {view['heading']}")
    if view.get("shift"):
        facts.append(f"DUTY CHANGE: {view['shift']}")
    if view.get("labels_added"):
        facts.append(f"DUTIES ADDED: {', '.join(view['labels_added'])}")
    if view.get("labels_removed"):
        facts.append(f"DUTIES REMOVED: {', '.join(view['labels_removed'])}")
    if view.get("numbers_changed"):
        facts.append(f"NUMBERS CHANGED: {view['numbers_changed']}")
    if view.get("contract_parties"):
        facts.append(f"PARTIES TO THIS CONTRACT: {', '.join(view['contract_parties'])}")
    if view.get("parties"):
        facts.append(f"PARTIES NAMED IN THE TEXT: {', '.join(view['parties'])}")
    if view.get("stakeholders"):
        facts.append(f"LIKELY STAKEHOLDERS: {', '.join(view['stakeholders'])}")

    warn = ""
    if warning:
        warn = (f"\n\nA REVIEWER REJECTED YOUR LAST ANSWER: {warning}\n"
                f"Fix this. Say only what the text states.")

    numbers = ("\nIf numbers changed, give the old and the new value, written as digits exactly as in the text"
               " (for example: from 60 to 90 days, from $50,000 to $65,000)."
               if view.get("numbers_changed") else "")
    return f"""You explain contract changes in plain English to non-lawyers.

{chr(10).join(facts)}

{_changed_block(view, limit)}{_texts_block(view, limit)}{warn}

Write 2 or 3 short sentences: what is true in the NEW version compared to the OLD one, who is affected, and how.
Use only facts stated in the text. Do not add numbers, dates or parties that are not in the text.{numbers}
Then write one last line in this form, naming the affected parties:
AFFECTED: party one; party two"""


# "AFFECTED:" on its own line or, as seen on Kaggle, at the end of the last sentence
_AFFECTED_RE = re.compile(r"\baffected(?:\s+parties)?\s*:\s*([^\n]*)", re.I)

# Entries that name nobody ("no party concerned" was seen on Kaggle)
_NOBODY_RE = re.compile(r"^(?:no|none|nobody|n/?a|not applicable|unknown)\b|\bno part(?:y|ies)\b", re.I)


def parse_impact(raw: str) -> dict[str, Any]:
    """{"summary": str, "affected": [str]} from the Impact Agent's answer."""
    text = raw.replace("**", "").strip()
    matches = list(_AFFECTED_RE.finditer(text))
    affected: list[str] = []
    if matches:
        # Gemma sometimes writes it twice (inline, then on its own line): use all of them
        for m in matches:
            affected += [a.strip(" .*-") for a in re.split(r"[;,]|\band\b", m.group(1)) if a.strip(" .*-")]
        text = text[:matches[0].start()].strip()
    text = re.sub(r"^\s*(?:summary|impact)\s*:\s*", "", text, flags=re.I)
    summary = " ".join(text.split())

    seen, clean = set(), []
    for a in affected:
        key = a.lower()
        if key in ("party one", "party two", "<party>") or _NOBODY_RE.search(a) or key in seen:
            continue                                  # placeholder, nobody, or duplicate
        seen.add(key)
        clean.append(a)
    return {"summary": summary, "affected": clean}


# ============================================================
# CRITIC AGENT
# ============================================================

def build_critic_prompt(
    view: dict[str, Any],
    summary: str,
    limit: int,
    double_check: list[str] | None = None,
) -> str:
    """The question comes last, so the first word the Critic writes is the verdict."""
    # The change type is a known fact: on Kaggle the Critic rejected a correct
    # "this clause was removed" summary when it only saw the old text.
    words = ""
    if double_check:
        words = ("\n\nThese words in the SUMMARY do not appear in the contract text: "
                 + ", ".join(double_check[:8])
                 + ".\nIf any of them adds a fact the text does not state, answer HALLUCINATION. "
                   "If they only rephrase the text, they are fine.")
    return f"""You are a strict reviewer of contract summaries.

FACT (already verified): CHANGE: {_STATUS_TEXT.get(view['status'], view['status'])}

{_changed_block(view, limit)}{_texts_block(view, limit)}

SUMMARY TO CHECK:
{summary}{words}

Check every claim in the SUMMARY against the contract text above.
Reply with ONE word on the first line:
VERIFIED - every claim in the summary is supported by the text
HALLUCINATION - the summary says something the text does not say
MISSING_CONTEXT - the text depends on other sections, so the summary cannot be checked
Then write one short line explaining why."""


def build_critic_retry_prompt(view: dict[str, Any], summary: str, limit: int) -> str:
    return f"""{_texts_block(view, limit)}

SUMMARY:
{summary}

Is the SUMMARY fully supported by the text? Reply with exactly ONE word:
VERIFIED, HALLUCINATION or MISSING_CONTEXT"""


_VERDICT_RE = re.compile(r"VERIFIED|HALLUCINATION|MISSING[_ ]CONTEXT", re.I)


def parse_critic(raw: str) -> dict[str, str]:
    """
    {"verdict": VERIFIED | HALLUCINATION | MISSING_CONTEXT | UNKNOWN, "reason": str}

    The verdict must be unambiguous: one verdict on the first line, or, if the
    first line has none, exactly one distinct verdict in the whole answer.
    An echoed option list ("VERIFIED - ... HALLUCINATION - ...") is UNKNOWN.
    """
    text = raw.replace("**", "").strip()
    lines = [l for l in text.splitlines() if l.strip()]
    if not lines:
        return {"verdict": UNKNOWN, "reason": "", "raw": raw[:200]}

    def verdicts(s):
        return {m.group(0).upper().replace(" ", "_") for m in _VERDICT_RE.finditer(s)}

    # Several lines each starting with a verdict = the option list was copied
    line_starts = {m.group(0).upper().replace(" ", "_") for l in lines
                   for m in [_VERDICT_RE.match(l.strip(" -*#>"))] if m}
    if len(line_starts) > 1:
        return {"verdict": UNKNOWN, "reason": "", "raw": raw[:200]}

    first = verdicts(lines[0])
    found = first if first else verdicts(text)
    if len(found) != 1:
        return {"verdict": UNKNOWN, "reason": "", "raw": raw[:200]}
    verdict = found.pop()
    m = _VERDICT_RE.search(text)
    reason = " ".join(text[m.end():].strip(" .:-\n").split())[:300]
    return {"verdict": verdict, "reason": reason}
