"""
ClauseLens — Node 6
Report Assembler: turns Node 5's results into one layered JSON report.

    Node 5 results (one per changed pair, any order, as they arrive)
          |
          v
    ReportAssembler.add()  ... add() ...  build()
          |
          v
    report
      ├─ overview          headline, counts, highlights, what needs review     (layer 1)
      ├─ stakeholders      party -> its changes                                 (layer 2)
      ├─ sections          section -> its changes                               (layer 2)
      ├─ changes           one entry per change: summary, parties, duty shift,  (layer 3)
      │    └─ evidence     old/new clause, pages, changed sentences + offsets   (layer 4)
      ├─ index             chunk -> changes, page -> changes (document -> report)
      └─ skipped           pairs Node 5 did not analyse, with the reason

Navigation works both ways: from a change to the exact sentences (evidence
offsets), and from a place in a document to its changes (index).
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from .evidence import build_evidence
from .parties import BOTH, normalize_parties

REPORT_VERSION = "1"

_TYPE_ORDER = ["MODIFIED", "ADDED", "DELETED", "UNVERIFIED"]


@dataclass
class Node6Config:
    # How many of the most important changes the overview lists
    max_highlights: int = 5

    # Keep Node 5's attempt-by-attempt history in each change (larger file; for debugging)
    include_history: bool = False

    # Keep the full clause text in the evidence (needed for highlighting in a UI)
    include_text: bool = True


class ReportAssembler:
    """
    Parameters
    ----------
    documents:
        Optional names for the two documents, e.g.
        {"document_a": "License Agreement (2009)", "document_b": "Amended and Restated (2012)"}.

    all_chunks:
        Optional Node 2 canonical chunks: gives document order, the contract's
        parties (for "Both parties"), and chunk counts.

    node3_report:
        Optional Node 3 align_documents() report: gives the number of unchanged clauses.
    """

    def __init__(
        self,
        documents: dict[str, str] | None = None,
        all_chunks: list[dict[str, Any]] | None = None,
        node3_report: dict[str, Any] | None = None,
        config: Node6Config | None = None,
    ):
        self.config = config or Node6Config()
        self.documents = documents or {}
        self.node3_report = node3_report
        self.results: dict[str, dict[str, Any]] = {}

        self._position: dict[str, tuple[str, float]] = {}
        self.chunk_counts: Counter = Counter()
        self.page_counts: dict[str, int] = {}
        self.contract_parties: list[str] = []
        self._section_of = {}
        if all_chunks:
            by_doc: dict[str, list[dict]] = defaultdict(list)
            for c in all_chunks:
                by_doc[c["document_id"]].append(c)
            for doc, chunks in by_doc.items():
                self.chunk_counts[doc] = len(chunks)
                pages = [c.get("page_end") or c.get("page_start") for c in chunks]
                pages = [p for p in pages if isinstance(p, int)]
                if pages:
                    self.page_counts[doc] = max(pages)
                for i, c in enumerate(chunks):
                    self._position[c["chunk_id"]] = (doc, i / max(1, len(chunks)))
            # Section of every chunk, from document order (see _sections_in_order)
            self._section_of: dict[str, str] = {}
            for chunks in by_doc.values():
                self._section_of.update(_sections_in_order(chunks))
            # The contract's parties: named in each document's opening chunk
            from nodes.node_5.ontology import find_parties
            for chunks in by_doc.values():
                for p in find_parties(chunks[0]["text"]):
                    if p != BOTH and p not in self.contract_parties:
                        self.contract_parties.append(p)

    # ============================================================
    # INPUT
    # ============================================================

    def add(self, result: dict[str, Any]) -> None:
        """Add one Node 5 result (call as results arrive; order does not matter)."""
        self.results[result["pair_id"]] = result

    def add_many(self, results) -> "ReportAssembler":
        """Node 5's run() output, its 'results' list, or any iterable of results."""
        if isinstance(results, dict):
            results = results.get("results", [])
        for r in results:
            self.add(r)
        return self

    # ============================================================
    # OUTPUT
    # ============================================================

    def build(self) -> dict[str, Any]:
        """The full layered report, from everything added so far."""
        # Sorted by document order, so the report doesn't depend on the order results arrived in
        ordered = sorted(self.results.values(), key=self._order_key)
        analysed = [r for r in ordered if r.get("impact", {}).get("status") != "SKIPPED"]
        skipped = [r for r in ordered if r.get("impact", {}).get("status") == "SKIPPED"]

        changes = [self._change(i, r) for i, r in enumerate(analysed, 1)]
        return {
            "report_version": REPORT_VERSION,
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "documents": self._documents(),
            "overview": self._overview(changes, skipped),
            "stakeholders": self._by_stakeholder(changes),
            "sections": self._by_section(changes),
            "changes": changes,
            "index": self._index(changes),
            "skipped": [{"pair_id": r["pair_id"], "reason": r["impact"].get("reason", "")} for r in skipped],
        }

    # ------------------------------------------------------------

    def _order_key(self, r: dict[str, Any]) -> tuple:
        """Document order: position in the new version, else in the old one."""
        for cid in (r.get("target_chunk_id"), r.get("source_chunk_id")):
            if cid in self._position:
                return (self._position[cid][1], r["pair_id"])
        return (2.0, r["pair_id"])

    def _change(self, i: int, r: dict[str, Any]) -> dict[str, Any]:
        im, d, al = r.get("impact", {}), r.get("deontic", {}), r.get("alignment", {})
        side = r.get("target") or r.get("source") or {}
        heading_path = side.get("heading_path") or []
        affected = normalize_parties(im.get("affected", []))
        evidence = build_evidence(r)
        if not self.config.include_text:
            for e in evidence.values():
                if e:
                    e.pop("text", None)
        change = {
            "id": f"C{i:03d}",
            "pair_id": r["pair_id"],
            "type": r.get("change_status"),
            "section": self._section_for(r, heading_path),
            "title": _title(heading_path, evidence),
            "summary": im.get("summary", ""),
            "affected": affected,
            "status": im.get("status"),
            "confidence": im.get("confidence"),
            "needs_review": im.get("status") != "VERIFIED",
            "review_reason": im.get("reason", "") if im.get("status") != "VERIFIED" else "",
            "duty": {
                "shift": d.get("shift"),
                "labels_added": d.get("labels_added", []),
                "labels_removed": d.get("labels_removed", []),
            },
            "numbers": al.get("numeric_diff") or {},
            "similarity": al.get("similarity_score"),          # Node 3 score; None for ADDED/DELETED
            "categories": im.get("categories", []),
            "attempts": im.get("attempts"),
            "evidence": evidence,
        }
        change["key_numbers"] = _key_numbers(change)
        change["importance"] = _importance(change)
        if self.config.include_history:
            change["history"] = im.get("history", [])
        return change

    def _section_for(self, r: dict[str, Any], heading_path: list[str]) -> str:
        """Section from document order (new version first), else from the heading path."""
        for cid in (r.get("target_chunk_id"), r.get("source_chunk_id")):
            if cid in self._section_of:
                return self._section_of[cid]
        return _section(heading_path)

    def _documents(self) -> dict[str, Any]:
        out = {}
        for doc in sorted(set(self.chunk_counts) | set(self.documents) | {"document_a", "document_b"}):
            out[doc] = {
                "name": self.documents.get(doc, "Previous version" if doc == "document_a" else "New version"),
                "chunks": self.chunk_counts.get(doc),
                "pages": self.page_counts.get(doc),
            }
        return out

    def _overview(self, changes: list[dict], skipped: list[dict]) -> dict[str, Any]:
        by_type = Counter(c["type"] for c in changes)
        shifts = [c for c in changes if c["duty"]["shift"]]
        review = [c for c in changes if c["needs_review"]]
        unchanged = None
        if self.node3_report:
            unchanged = self.node3_report.get("counts", {}).get("EQUIVALENT")

        parts = [f"{by_type.get(t, 0)} {t.lower()}" for t in _TYPE_ORDER if by_type.get(t)]
        headline = f"{len(changes)} change{'s' * (len(changes) != 1)}"
        if parts:
            headline += ": " + ", ".join(parts)
        if shifts:
            headline += f"; {len(shifts)} duty shift{'s' * (len(shifts) != 1)}"
        headline += f"; {len(review)} need{'s' * (len(review) == 1)} human review"

        # Only confirmed changes are headlined; unconfirmed ones are in needs_review
        top = sorted((c for c in changes if c["importance"] > 0 and not c["needs_review"]),
                     key=lambda c: -c["importance"])
        return {
            "headline": headline,
            "total_changes": len(changes),
            "by_type": {t: by_type.get(t, 0) for t in _TYPE_ORDER},
            "by_confidence": dict(Counter(c["confidence"] or "NONE" for c in changes)),
            "duty_shifts": len(shifts),
            "needs_review": [c["id"] for c in review],
            "unchanged_clauses": unchanged,
            "skipped": len(skipped),
            "highlights": [{"id": c["id"], "title": c["title"], "summary": c["summary"],
                            "why": _why(c)} for c in top[: self.config.max_highlights]],
        }

    def _by_stakeholder(self, changes: list[dict]) -> list[dict[str, Any]]:
        """Party -> its changes. "Both parties" also counts for each contract party."""
        groups: dict[str, list[str]] = defaultdict(list)
        for c in changes:
            parties = list(c["affected"])
            if BOTH in parties and self.contract_parties:
                parties = [p for p in parties if p != BOTH] + self.contract_parties
            for p in dict.fromkeys(parties):
                groups[p].append(c["id"])
        return [{"party": p, "change_count": len(ids), "changes": ids}
                for p, ids in sorted(groups.items(), key=lambda kv: -len(kv[1]))]

    def _by_section(self, changes: list[dict]) -> list[dict[str, Any]]:
        groups: dict[str, list[str]] = {}
        for c in changes:                                # changes are in document order already
            groups.setdefault(c["section"], []).append(c["id"])
        return [{"section": s, "change_count": len(ids), "changes": ids} for s, ids in groups.items()]

    def _index(self, changes: list[dict]) -> dict[str, Any]:
        """Document -> report: which changes touch a chunk or a page."""
        by_chunk: dict[str, list[str]] = defaultdict(list)
        by_page: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
        for c in changes:
            for side in ("old", "new"):
                e = c["evidence"][side]
                if not e:
                    continue
                by_chunk[e["chunk_id"]].append(c["id"])
                start, end = e["pages"]
                if start is not None:
                    for page in range(start, (end or start) + 1):
                        by_page[e["document_id"]][str(page)].append(c["id"])
        return {"by_chunk": dict(by_chunk),
                "by_page": {doc: dict(pages) for doc, pages in by_page.items()}}


# ============================================================
# HELPERS
# ============================================================

_NUMBERED = re.compile(r"^\s*\d{1,2}\.(?!\d)")

# "1. Definitions.", "17. Complete Agreement.", "26.Use of Name.Licensor ..."
_TOP_WITH_TITLE = re.compile(r"^(\d{1,2})\.\s*([A-Z][^.\n]{1,160}?)\.(?:\s|$|(?=[A-Z]))")
# "2. LICENSE", "2. Grant of License; Sublicenses; ... Supply and Sale" (a heading line
# without a full stop, possibly continued on the next line)
_TOP_LINE = re.compile(r"^(\d{1,2})\.\s*([A-Z][A-Za-z ,;&'’\-]{1,160})$")
# "2.4 For a period ...", "10.3 ..." (numbered subsection)
_SUB = re.compile(r"^(\d{1,2})\.\d{1,2}(?!\d)")
# "SCHEDULE A"
_SCHEDULE = re.compile(r"^SCHEDULE\s+([A-Z])\b", re.I)


def _clean(text: str | None) -> str:
    return " ".join((text or "").split())


def _sections_in_order(chunks: list[dict[str, Any]]) -> dict[str, str]:
    """
    Section of every chunk, by walking the document in order: a chunk belongs to
    the last top-level section ("2. Grant of License", "Schedule A") seen so far.
    Node 1's heading tree can't be used for this: on a real EDGAR contract it
    nested most clauses under recital "D." of the preamble.
    """
    names: dict[str, str] = {}
    current = "Preamble"
    current_num = 0          # section numbers only move forward
    in_schedule = False      # numbered items inside a Schedule ("1. Licensor's ...") are not sections
    out: dict[str, str] = {}
    for c in chunks:
        lines = [_clean(l) for l in (c.get("text") or "").splitlines()[:2]]
        # The chunk's own first lines come before its heading: a merged heading line
        # ("4.The Patents and Inventions.") sits above a clause headed "4.1 ..."
        candidates = list(lines) + [_clean(h) for h in (c.get("heading_path") or [])[-1:]]
        if len(lines) == 2 and re.fullmatch(r"\d{1,2}\.", lines[0]):   # "1." / "Definitions."
            candidates.insert(0, f"{lines[0]} {lines[1]}")
        for cand in candidates:
            m = _SCHEDULE.match(cand)
            if m:
                current, in_schedule = f"Schedule {m.group(1).upper()}", True
                break
            if in_schedule:
                continue
            m = _TOP_WITH_TITLE.match(cand) or _TOP_LINE.match(cand)
            if m and int(m.group(1)) >= current_num:
                num, title = m.group(1), m.group(2).strip()
                if len(title) > 60:                       # "Grant of License; Sublicenses; ..."
                    title = title[:60].rsplit(" ", 1)[0].rstrip(";,") + " …"
                names.setdefault(num, f"{num}. {title}")
                current, current_num = names[num], int(num)
                break
            m = _SUB.match(cand)
            if m and int(m.group(1)) >= current_num:
                num = m.group(1)
                if not current.startswith(f"{num}. ") and current != f"Section {num}":
                    current = names.get(num, f"Section {num}")
                current_num = int(num)
                break
        out[c["chunk_id"]] = current
    return out


def _section(heading_path: list[str]) -> str:
    """Fallback without document order: the first numbered heading, else "Other"."""
    for h in heading_path:
        h = _clean(h)
        m = _SUB.match(h)
        if m:
            return f"Section {m.group(1)}"
        if _NUMBERED.match(h):
            return h[:70]
    return "Other"


def _title(heading_path: list[str], evidence: dict[str, Any]) -> str:
    """
    Short readable title: the clause heading, cleaned and cut at ~70 characters.
    When the heading is not a real title (a table row like "5 – 10", a bullet,
    empty), the first words of the changed sentence are used instead.
    """
    title = _clean(heading_path[-1]) if heading_path else ""
    if sum(ch.isalpha() for ch in title) < 4:
        side = evidence.get("new") or evidence.get("old") or {}
        sentences = side.get("changed_sentences") or [s["text"] for s in side.get("sentences", [])]
        title = _clean(sentences[0]) if sentences else title
        title = re.sub(r"^[^A-Za-z]+", "", title)          # table text: "5 – 10 $[**] Licensee ..." -> "Licensee ..."
    if len(title) > 70:
        title = title[:70].rsplit(" ", 1)[0] + " …"
    return title


_UNIT_AFTER = r"\)?\s*(?:\(\w+\)\s*)?(?:calendar\s+|business\s+|successive\s+)?(?:days?|weeks?|months?|years?)\b"


def _key_numbers(change: dict[str, Any]) -> dict[str, list[str]]:
    """
    The numbers in a change that matter: amounts ($), percentages (%) and
    periods (days, months, years). Section references ("3.8" -> "3.7"),
    addresses and zip codes are left out: on a real contract they otherwise
    dominated the highlights.
    """
    texts = " ".join(e["text"] for e in change["evidence"].values() if e and e.get("text"))
    keep = {}
    for side, key in (("old", "only_in_source"), ("new", "only_in_target")):
        out = []
        for tok in change["numbers"].get(key) or []:
            if "$" in tok or "%" in tok:
                out.append(tok)
            elif re.fullmatch(r"\d+\.\d+", tok):
                continue                                  # section reference
            elif re.search(rf"(?<![\d.,]){re.escape(tok)}{_UNIT_AFTER}", texts, re.I):
                out.append(tok)
        keep[side] = out
    return keep


def _importance(c: dict[str, Any]) -> int:
    """Rough ranking for the overview's highlights."""
    score = 0
    if c["duty"]["shift"]:
        score += 3
    if set(c["duty"]["labels_removed"]) & {"OBLIGATION", "PROHIBITION"}:
        score += 2
    if c["duty"]["labels_added"]:
        score += 1
    if c["key_numbers"]["old"] or c["key_numbers"]["new"]:
        score += 3                                       # amounts, percentages, deadlines
    if c["type"] in ("ADDED", "DELETED") and (c["duty"]["labels_added"] or c["duty"]["labels_removed"]):
        score += 1
    return score


def _why(c: dict[str, Any]) -> str:
    reasons = []
    if c["duty"]["shift"]:
        reasons.append(f"duty shift {c['duty']['shift']}")
    if c["key_numbers"]["old"] or c["key_numbers"]["new"]:
        old = ", ".join(c["key_numbers"]["old"]) or "-"
        new = ", ".join(c["key_numbers"]["new"]) or "-"
        reasons.append(f"numbers {old} -> {new}")
    if c["duty"]["labels_removed"] and not c["duty"]["shift"]:
        reasons.append("removes " + "/".join(c["duty"]["labels_removed"]).lower())
    if c["duty"]["labels_added"] and not c["duty"]["shift"]:
        reasons.append("adds " + "/".join(c["duty"]["labels_added"]).lower())
    return "; ".join(reasons)
