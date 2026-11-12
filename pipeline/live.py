"""
ClauseLens — Pipeline
The report as it fills in, for showing results while a run goes on.

Right after Node 4 every changed clause is known; the impact summaries follow
one by one during Node 5. LiveReport builds the Node 6 report over the Node 5
results so far, with "PENDING" placeholders for the rest, so a UI can show the
changes at once and fill in the summaries as they arrive.

    preview()       {"ui": <Node 6 UI data>, "done": 0, "total": n}
    update(result)  {"pair_id", "alignment", "overview", "done", "total"}

Each alignment in the UI data carries "pair_id" and "pending". Pairs Node 5 will
skip (too little text) are skipped here too, so change ids (C001, ...) match the
final report.
"""

from __future__ import annotations

from typing import Any

PENDING = {"status": "PENDING", "confidence": None, "summary": "", "affected": [], "reason": "",
           "categories": [], "attempts": 0}


class LiveReport:
    def __init__(self, documents: dict[str, str], all_chunks: list[dict[str, Any]], node3_report: dict[str, Any],
                 pairs: list[dict[str, Any]], skipped: set[str]):
        self.documents, self.all_chunks, self.node3_report, self.pairs = documents, all_chunks, node3_report, pairs
        self.results: dict[str, dict[str, Any]] = {}
        for p in pairs:
            im = dict(PENDING, status="SKIPPED") if p["pair_id"] in skipped else dict(PENDING)
            self.results[p["pair_id"]] = {**p, "impact": im}
        self.total = len(pairs) - len(skipped)

    def ui(self) -> dict[str, Any]:
        from nodes.node_6 import ReportAssembler, to_ui_data
        rep = ReportAssembler(documents=self.documents, all_chunks=self.all_chunks,
                              node3_report=self.node3_report).add_many(
            [self.results[p["pair_id"]] for p in self.pairs]).build()
        ui = to_ui_data(rep)
        for al, change in zip(ui["alignments"], rep["changes"]):
            al["pair_id"] = change["pair_id"]
            al["pending"] = change["status"] == "PENDING"
        return ui

    def done(self) -> int:
        return sum(1 for r in self.results.values() if r["impact"]["status"] not in ("PENDING", "SKIPPED"))

    def preview(self) -> dict[str, Any]:
        return {"ui": self.ui(), "done": 0, "total": self.total}

    def update(self, result: dict[str, Any]) -> dict[str, Any] | None:
        """Record one Node 5 result; returns the changed clause's new UI entry (None for skipped pairs)."""
        self.results[result["pair_id"]] = result
        if result["impact"]["status"] == "SKIPPED":
            return None
        ui = self.ui()
        al = next((a for a in ui["alignments"] if a["pair_id"] == result["pair_id"]), None)
        return {"pair_id": result["pair_id"], "alignment": al, "overview": ui["overview"],
                "done": self.done(), "total": self.total}
