"""
ClauseLens — Node 6
Evidence: links every change back to the exact sentences in the documents.

For each side of a change (old / new clause) it returns the clause text, its
pages and heading, and every sentence with character offsets into that text and
a `changed` flag, so a UI can highlight exactly what changed.

    old clause text ──split_sentences──► sentences + offsets ─┐
                                                              ├─► changed = not on the other side
    new clause text ──split_sentences──► sentences + offsets ─┘

No model or GPU dependency.
"""

from __future__ import annotations

from typing import Any

from nodes.node_4.patterns import split_sentences


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def mark_sentences(old_text: str | None, new_text: str | None) -> tuple[list[dict], list[dict]]:
    """
    Sentences of both sides with offsets and a `changed` flag. A sentence is
    unchanged when the same sentence (ignoring case and spacing) is on the other
    side. For an ADDED or DELETED clause every sentence is changed.
    """
    old = split_sentences(old_text) if old_text else []
    new = split_sentences(new_text) if new_text else []
    old_set, new_set = {_norm(s["text"]) for s in old}, {_norm(s["text"]) for s in new}
    for s in old:
        s["changed"] = _norm(s["text"]) not in new_set
    for s in new:
        s["changed"] = _norm(s["text"]) not in old_set
    return old, new


def evidence_side(chunk: dict[str, Any] | None, sentences: list[dict]) -> dict[str, Any] | None:
    """One side of the evidence: where the clause is and what in it changed."""
    if not chunk:
        return None
    return {
        "chunk_id": chunk.get("chunk_id"),
        "document_id": chunk.get("document_id"),
        "pages": [chunk.get("page_start"), chunk.get("page_end")],
        "heading_path": chunk.get("heading_path") or [],
        "text": chunk.get("text", ""),
        "sentences": sentences,
        # The spans a UI should highlight (offsets into `text`)
        "highlights": [{"start": s["start"], "end": s["end"]} for s in sentences if s["changed"]],
        "changed_sentences": [s["text"] for s in sentences if s["changed"]],
    }


def build_evidence(result: dict[str, Any]) -> dict[str, Any]:
    """Evidence for one Node 5 result: {"old": side or None, "new": side or None}."""
    src, tgt = result.get("source"), result.get("target")
    old_s, new_s = mark_sentences(src.get("text") if src else None, tgt.get("text") if tgt else None)
    return {"old": evidence_side(src, old_s), "new": evidence_side(tgt, new_s)}
