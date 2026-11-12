"""
ClauseLens — Node 5
Parallel multi-agent core and routing.

    NODE 4 enriched pairs
          |
          v
    +-------------------+        +-------------------+
    | GPU 0             |        | GPU 1             |
    | ImpactAgent.write | -----> | grounding checks  |  (rules: numbers, parties)
    |                   |        | CriticAgent.check |
    +-------------------+        +---------+---------+
          ^                                |
          |        ROUTING                 v
          +---- HALLUCINATION ----  retry with the Critic's warning
          +---- MISSING_CONTEXT --  add neighbouring chunks, retry
                VERIFIED ---------> done (HIGH confidence)
                max_attempts -----> done (LOW_CONFIDENCE)
                                         |
                                         v
                              checkpoint (JSONL) ----> NODE 6

The two agents run on separate threads: while the Critic checks pair 1 on
GPU 1, the Impact Agent already writes pair 2 on GPU 0.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections import Counter, deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import Any, Iterator

from .agents import CriticAgent, ImpactAgent
from .config import (
    Node5Config,
    NODE5_VERSION,
    VERIFIED,
    HALLUCINATION,
    MISSING_CONTEXT,
    UNKNOWN,
    STATUS_VERIFIED,
    STATUS_LOW_CONFIDENCE,
    STATUS_SKIPPED,
)
from .grounding import check_summary, unsupported_words
from .ontology import detect_categories, find_parties, stakeholders_for
from nodes.node_4.patterns import split_sentences


class Node5ImpactCore:
    """
    Parameters
    ----------
    all_chunks:
        Canonical chunks from Node 2 prepare_chunks() (used for MISSING_CONTEXT).

    impact_agent / critic_agent:
        The two agents. They should wrap different models on different GPUs;
        if they share one model, parallel mode is switched off automatically.
    """

    def __init__(
        self,
        all_chunks: list[dict[str, Any]],
        impact_agent: ImpactAgent,
        critic_agent: CriticAgent,
        config: Node5Config | None = None,
    ):
        self.config = config or impact_agent.config
        self.impact_agent = impact_agent
        self.critic_agent = critic_agent
        self.parallel = self.config.parallel and impact_agent.model is not critic_agent.model

        self.chunk_by_id = {c["chunk_id"]: c for c in all_chunks}
        self._doc_order: dict[str, list[str]] = {}
        for c in all_chunks:
            self._doc_order.setdefault(c["document_id"], []).append(c["chunk_id"])
        self._position = {cid: (doc, i) for doc, ids in self._doc_order.items() for i, cid in enumerate(ids)}

        # The contract's own parties, from each document's opening chunk (title +
        # preamble). Always allowed as AFFECTED parties: on Kaggle, a correct
        # summary of "Licensor shall provide updates" was rejected for naming the
        # Licensee, who receives them but isn't named in that clause.
        self.contract_parties: list[str] = []
        for ids in self._doc_order.values():
            for p in find_parties(self.chunk_by_id[ids[0]]["text"]):
                if p not in self.contract_parties and p != "Both parties":
                    self.contract_parties.append(p)

        self.calls = Counter()

    # ============================================================
    # PAIR VIEW
    # ============================================================

    def build_view(self, pair: dict[str, Any], context_level: int = 0) -> dict[str, Any]:
        """Everything the agents see about one pair."""
        src, tgt = pair.get("source"), pair.get("target")
        side = tgt or src
        heading = (side.get("heading_path") or [""])[-1] if side else ""
        d = pair.get("deontic", {})
        source_text = src["text"] if src else None
        target_text = tgt["text"] if tgt else None
        all_text = " ".join(t for t in (source_text, target_text) if t)

        categories = detect_categories(all_text)
        return {
            "pair_id": pair["pair_id"],
            "status": pair["change_status"],
            "heading": heading,
            "source_text": source_text,
            "target_text": target_text,
            "context_text": self._context(pair, context_level) if context_level else "",
            "shift": d.get("shift"),
            "labels_added": d.get("labels_added", []),
            "labels_removed": d.get("labels_removed", []),
            "numbers_changed": _numbers_changed(pair.get("alignment", {})),
            "categories": [c["category"] for c in categories],
            "stakeholders": stakeholders_for(categories),
            "parties": find_parties(all_text),
            "contract_parties": self.contract_parties,
            **self._changed_sentences(source_text, target_text),
        }

    def _changed_sentences(self, source_text: str | None, target_text: str | None) -> dict[str, list[str]]:
        """
        Sentences that differ between the versions (MODIFIED pairs only). On Kaggle
        the Impact Agent described the OLD clause of 2.1 as if it were current;
        pointing at the exact changed sentences keeps both agents on the change.
        """
        if not (source_text and target_text):
            return {"changed_old": [], "changed_new": []}
        old = [s["text"] for s in split_sentences(source_text)]
        new = [s["text"] for s in split_sentences(target_text)]
        norm = lambda s: " ".join(s.lower().split())
        new_set, old_set = {norm(s) for s in new}, {norm(s) for s in old}
        k = self.config.max_changed_sentences
        return {
            "changed_old": [" ".join(s.split()) for s in old if norm(s) not in new_set][:k],
            "changed_new": [" ".join(s.split()) for s in new if norm(s) not in old_set][:k],
        }

    def _context(self, pair: dict[str, Any], level: int) -> str:
        """Neighbouring chunks around each side of the pair (Node 2 chunk order)."""
        k = self.config.context_neighbors * level
        own = {pair.get("source_chunk_id"), pair.get("target_chunk_id")}
        picked: list[str] = []
        for cid in (pair.get("source_chunk_id"), pair.get("target_chunk_id")):
            if cid not in self._position:
                continue
            doc, i = self._position[cid]
            ids = self._doc_order[doc]
            for j in list(range(max(0, i - k), i)) + list(range(i + 1, min(len(ids), i + k + 1))):
                if ids[j] not in own and ids[j] not in picked:
                    picked.append(ids[j])
        return " [...] ".join(self.chunk_by_id[c]["text"] for c in picked)

    def _is_trivial(self, view: dict[str, Any]) -> bool:
        """Too little text to analyse, e.g. a heading-only chunk."""
        for t in (view["source_text"], view["target_text"]):
            if t:
                body = t.strip()
                if view["heading"] and body.startswith(view["heading"]):
                    body = body[len(view["heading"]):]
                if len(body.strip()) >= self.config.min_chars:
                    return False
        return True

    # ============================================================
    # AGENT STEPS (run on worker threads)
    # ============================================================

    def _impact_step(self, job: dict[str, Any]) -> dict[str, Any]:
        view = self.build_view(job["pair"], job["context_level"])
        out = self.impact_agent.write(view, job.get("warning"))
        self.calls["impact"] += 1
        job["view"] = view
        job["draft"] = out
        job["issues"] = check_summary(out["summary"], out["affected"], view,
                                      max_unsupported_words=self.config.max_unsupported_words)
        # Borderline: a few words not in the text go to the Critic to verify
        job["double_check"] = []
        if not job["issues"]:
            words = unsupported_words(out["summary"], _grounding_text(view))
            if len(words) > self.config.review_unsupported_words:
                job["double_check"] = words
        return job

    def _critic_step(self, job: dict[str, Any]) -> dict[str, Any]:
        job["critic"] = self.critic_agent.check(job["view"], job["draft"]["summary"], job.get("double_check"))
        self.calls["critic"] += 1
        return job

    # ============================================================
    # ROUTING
    # ============================================================

    def _route(self, job: dict[str, Any]) -> str:
        """Record this attempt and decide: "done" or "retry"."""
        if job["issues"]:
            verdict, reason, decided_by = HALLUCINATION, "; ".join(job["issues"]), "checks"
        else:
            c = job["critic"]
            verdict, reason, decided_by = c["verdict"], c.get("reason", ""), "critic"

        job["history"].append({
            "attempt": job["attempt"],
            "context_level": job["context_level"],
            "summary": job["draft"]["summary"],
            "affected": job["draft"]["affected"],
            "issues": job["issues"],
            "double_check": job.get("double_check", []),
            "verdict": verdict,
            "reason": reason,
            "decided_by": decided_by,
            "critic_raw": job.get("critic", {}).get("raw") if decided_by == "critic" else None,
        })
        job.pop("critic", None)

        if verdict == VERIFIED:
            job["final"] = STATUS_VERIFIED
            return "done"
        if verdict == UNKNOWN:
            # The Critic's answer was unreadable even after its retry: don't burn
            # an Impact retry on it, forward with a warning
            job["final"] = STATUS_LOW_CONFIDENCE
            job["final_reason"] = "the Critic's answer was unreadable"
            return "done"
        if job["attempt"] >= self.config.max_attempts:
            job["final"] = STATUS_LOW_CONFIDENCE
            job["final_reason"] = f"rejected {job['attempt']} times ({verdict}): {reason}"
            return "done"

        job["attempt"] += 1
        if verdict == MISSING_CONTEXT and job["context_level"] < self.config.max_context_level:
            job["context_level"] += 1
            job["warning"] = None
        else:
            job["warning"] = reason or "the summary was not supported by the text"
        return "retry"

    def _result(self, job: dict[str, Any]) -> dict[str, Any]:
        pair, hist = job["pair"], job["history"]
        last = hist[-1] if hist else {}
        view = job.get("view") or self.build_view(pair)
        return {
            **pair,
            "impact": {
                "status": job["final"],
                "confidence": {STATUS_VERIFIED: "HIGH", STATUS_LOW_CONFIDENCE: "LOW"}.get(job["final"]),
                "summary": last.get("summary", ""),
                "affected": last.get("affected", []),
                "final_verdict": last.get("verdict"),
                "reason": job.get("final_reason") or last.get("reason", ""),
                "attempts": len(hist),
                "context_level": job["context_level"],
                "categories": view["categories"],
                "stakeholder_hints": view["stakeholders"],
                "parties_in_text": view["parties"],
                "history": hist,
            },
        }

    # ============================================================
    # CONTINUOUS / PARALLEL PROCESSING
    # ============================================================

    def iter_run(
        self,
        pairs: list[dict[str, Any]],
        checkpoint_path: str | None = None,
    ) -> Iterator[dict[str, Any]]:
        """
        Yield one finished result per pair, as soon as it is final (completion
        order, not input order). With checkpoint_path, each result is appended to
        a JSONL file immediately; on restart, finished pairs are read from disk.
        """
        fingerprint = self._fingerprint(pairs)
        done = _load_checkpoint(checkpoint_path, fingerprint) if checkpoint_path else {}
        fh = _open_checkpoint(checkpoint_path, fingerprint, bool(done)) if checkpoint_path else None

        def emit(result):
            if fh:
                fh.write(json.dumps(result, ensure_ascii=False) + "\n")
                fh.flush()
            return result

        impact_ex = ThreadPoolExecutor(max_workers=1, thread_name_prefix="impact")
        critic_ex = ThreadPoolExecutor(max_workers=1, thread_name_prefix="critic") if self.parallel else impact_ex
        inflight: dict[Any, tuple[str, dict]] = {}
        queue: deque = deque()

        try:
            for pair in pairs:
                if pair["pair_id"] in done:
                    yield done[pair["pair_id"]]
                    continue
                job = {"pair": pair, "attempt": 1, "context_level": 0, "warning": None, "history": []}
                if self._is_trivial(self.build_view(pair)):
                    job["final"], job["final_reason"] = STATUS_SKIPPED, "too little text to analyse"
                    yield emit(self._result(job))
                    continue
                queue.append(job)

            # Keep the Impact GPU busy with one job queued ahead; the Critic works in parallel
            def feed():
                while queue and sum(1 for s, _ in inflight.values() if s == "impact") < 2:
                    j = queue.popleft()
                    inflight[impact_ex.submit(self._impact_step, j)] = ("impact", j)

            feed()
            while inflight:
                finished, _ = wait(list(inflight), return_when=FIRST_COMPLETED)
                for fut in finished:
                    stage, job = inflight.pop(fut)
                    fut.result()                          # re-raise worker errors
                    if stage == "impact" and not job["issues"]:
                        inflight[critic_ex.submit(self._critic_step, job)] = ("critic", job)
                        continue
                    if self._route(job) == "retry":
                        queue.appendleft(job)             # retries go first
                    else:
                        yield emit(self._result(job))
                feed()
        finally:
            for f in inflight:
                f.cancel()
            impact_ex.shutdown(wait=True, cancel_futures=True)
            if critic_ex is not impact_ex:
                critic_ex.shutdown(wait=True, cancel_futures=True)
            if fh:
                fh.close()

    def run(
        self,
        pairs: list[dict[str, Any]],
        checkpoint_path: str | None = None,
        verbose: bool = True,
    ) -> dict[str, Any]:
        """Process every pair and return the Node 5 output for Node 6 (input order)."""
        if isinstance(pairs, dict) and "pairs" in pairs:
            pairs = pairs["pairs"]                        # accept Node 4's run() output directly
        total, t0, results = len(pairs), time.time(), {}
        for i, r in enumerate(self.iter_run(pairs, checkpoint_path), 1):
            results[r["pair_id"]] = r
            if verbose:
                elapsed = time.time() - t0
                print(f"[{i}/{total}] {r['impact']['status']:14} {r['pair_id']} "
                      f"| attempts {r['impact']['attempts']} | elapsed {elapsed:.0f}s")
        ordered = [results[p["pair_id"]] for p in pairs]
        return {
            "results": ordered,
            "counts": dict(Counter(r["impact"]["status"] for r in ordered)),
            "low_confidence": [r["pair_id"] for r in ordered if r["impact"]["status"] == STATUS_LOW_CONFIDENCE],
            "model_calls": dict(self.calls),
            "parallel": self.parallel,
            "seconds": round(time.time() - t0, 1),
        }

    def _fingerprint(self, pairs: list[dict[str, Any]]) -> str:
        h = hashlib.sha256()
        for p in pairs:
            h.update(p["pair_id"].encode())
            for side in ("source", "target"):
                if p.get(side):
                    h.update(p[side]["text"].encode())
        h.update(f"{NODE5_VERSION}|{self.config.max_attempts}|{self.config.max_context_level}".encode())
        return h.hexdigest()[:16]


# ============================================================
# HELPERS
# ============================================================

def _grounding_text(view: dict[str, Any]) -> str:
    """The same text the grounding checks compare against."""
    return " ".join(t for t in (view.get("source_text"), view.get("target_text"), view.get("context_text"),
                                view.get("heading"), *view.get("contract_parties", [])) if t)


def _numbers_changed(alignment: dict[str, Any]) -> str:
    nd = alignment.get("numeric_diff") or {}
    old, new = nd.get("only_in_source") or [], nd.get("only_in_target") or []
    if not old and not new:
        return ""
    return f"old version only: {', '.join(old) or '-'} | new version only: {', '.join(new) or '-'}"


def _open_checkpoint(path: str, fingerprint: str, resume: bool):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not resume or not os.path.exists(path):
        fh = open(path, "w", encoding="utf-8")
        fh.write(json.dumps({"_meta": {"node": 5, "fingerprint": fingerprint}}) + "\n")
        fh.flush()
        return fh
    needs_newline = False
    with open(path, "rb") as r:
        r.seek(0, os.SEEK_END)
        if r.tell() > 0:
            r.seek(-1, os.SEEK_END)
            needs_newline = r.read(1) != b"\n"
    fh = open(path, "a", encoding="utf-8")
    if needs_newline:
        fh.write("\n")                                  # close a partial line left by a crash
    return fh


def _load_checkpoint(path: str, fingerprint: str) -> dict[str, dict[str, Any]]:
    if not os.path.exists(path):
        return {}
    done = {}
    with open(path, encoding="utf-8") as fh:
        try:
            meta = json.loads(fh.readline()).get("_meta", {})
        except json.JSONDecodeError:
            return {}
        if meta.get("fingerprint") != fingerprint:
            print(f"Checkpoint {path} is from different input or an older Node 5 — starting fresh.")
            return {}
        for line in fh:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            done[row["pair_id"]] = row
    return done


def read_node5_output(path: str) -> list[dict[str, Any]]:
    """Load results from a Node 5 checkpoint file (for Node 6)."""
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "_meta" not in row:
                rows.append(row)
    return rows
