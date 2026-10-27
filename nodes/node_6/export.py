"""
ClauseLens — Node 6
Saving the report: the JSON for the UI (and as a backup on disk), and a
readable Markdown version for people.
"""

from __future__ import annotations

import json
import os
from typing import Any


def save_report(report: dict[str, Any], path: str) -> str:
    """
    Write the report as JSON. Written to a temporary file first and then renamed,
    so a crash mid-write never leaves a half-written report behind.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    os.replace(tmp, path)
    return path


def load_report(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _quote(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + " …"


def _pages(e: dict[str, Any]) -> str:
    start, end = e["pages"]
    if start is None:
        return ""
    return f"p. {start}" if end in (None, start) else f"pp. {start}–{end}"


def to_markdown(report: dict[str, Any], max_quote: int = 300) -> str:
    """A readable version of the report: overview, review list, stakeholders, then every change with its evidence."""
    ov, docs = report["overview"], report["documents"]
    a = docs.get("document_a", {}).get("name", "Previous version")
    b = docs.get("document_b", {}).get("name", "New version")
    lines = [f"# ClauseLens report", "", f"**{a}** → **{b}**", "", f"> {ov['headline']}", ""]

    if ov.get("unchanged_clauses") is not None:
        lines += [f"Unchanged clauses: {ov['unchanged_clauses']}. Skipped (too little text): {ov['skipped']}.", ""]

    if ov["highlights"]:
        lines += ["## Most important changes", ""]
        for h in ov["highlights"]:
            lines.append(f"- **{h['id']} {h['title']}**: {h['summary']}" + (f" *({h['why']})*" if h["why"] else ""))
        lines.append("")

    if ov["needs_review"]:
        lines += ["## Needs human review", "",
                  "These summaries could not be confirmed (LOW confidence). Check them against the quoted text.", ""]
        by_id = {c["id"]: c for c in report["changes"]}
        for cid in ov["needs_review"]:
            c = by_id[cid]
            lines.append(f"- **{cid} {c['title']}**: {c['review_reason'][:200]}")
        lines.append("")

    if report["stakeholders"]:
        lines += ["## By stakeholder", "", "| Party | Changes |", "|---|---|"]
        for s in report["stakeholders"]:
            lines.append(f"| {s['party']} | {s['change_count']}: {', '.join(s['changes'][:12])}"
                         f"{' …' if len(s['changes']) > 12 else ''} |")
        lines.append("")

    lines += ["## All changes", ""]
    section = None
    for c in report["changes"]:
        if c["section"] != section:
            section = c["section"]
            lines += [f"### {section}", ""]
        flag = "✅" if c["confidence"] == "HIGH" else "⚠️ needs review"
        lines.append(f"#### {c['id']} · {c['type']} · {c['title']}  ({flag})")
        lines.append("")
        lines.append(c["summary"] or "*(no summary)*")
        lines.append("")
        facts = []
        if c["affected"]:
            facts.append("**Affected:** " + ", ".join(c["affected"]))
        if c["duty"]["shift"]:
            facts.append(f"**Duty shift:** {c['duty']['shift']}")
        nums = c["numbers"]
        if nums.get("only_in_source") or nums.get("only_in_target"):
            facts.append(f"**Numbers:** {', '.join(nums.get('only_in_source') or ['-'])} → "
                         f"{', '.join(nums.get('only_in_target') or ['-'])}")
        if facts:
            lines += [" · ".join(facts), ""]
        for side, label in (("old", a), ("new", b)):
            e = c["evidence"][side]
            if e and e["changed_sentences"]:
                quote = " ".join(e["changed_sentences"])
                lines.append(f"> *{label}, {_pages(e)}:* {_quote(quote, max_quote)}")
                lines.append(">")
        if lines[-1] == ">":
            lines[-1] = ""
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def save_markdown(report: dict[str, Any], path: str, max_quote: int = 300) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(to_markdown(report, max_quote))
    return path
