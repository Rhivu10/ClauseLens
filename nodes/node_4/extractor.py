"""
ClauseLens — Node 4
Deontic cue extractor.

    NODE 3 report (align_documents)
          |
          +----> changed pairs: MODIFIED | ADDED | DELETED   (EQUIVALENT skipped)
                      |
                      +----> regex cues per sentence   (patterns.py)
                      |
                      +----> Gemma check, ambiguous only (checker.py)
                      |
                      +----> enriched pair ----> checkpoint (JSONL) ----> NODE 5

Pairs are processed one at a time and each result is appended to the
checkpoint file immediately, so a crashed session resumes where it stopped.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections import Counter
from typing import Any, Iterator

from .checker import DeonticChecker
from .config import (
    Node4Config,
    NODE4_VERSION,
    LABEL_PRIORITY,
    NONE,
    MODIFIED,
    ADDED,
    DELETED,
    UNVERIFIED,
)
from .patterns import tag_text


class Node4Extractor:
    """
    Parameters
    ----------
    all_chunks:
        Canonical chunks from Node 2 prepare_chunks().

    checker:
        DeonticChecker for the Gemma check. May be None only when
        config.model_check is "off".
    """

    def __init__(
        self,
        all_chunks: list[dict[str, Any]],
        checker: DeonticChecker | None = None,
        config: Node4Config | None = None,
    ):
        self.config = config or (checker.config if checker else Node4Config())
        if checker is None and self.config.model_check != "off":
            raise ValueError(
                "Node 4 needs a DeonticChecker unless Node4Config(model_check='off')."
            )
        self.checker = checker
        self.chunk_by_id = {c["chunk_id"]: c for c in all_chunks}

    # ============================================================
    # NODE 3 -> PAIRS
    # ============================================================

    def pairs_from_node3(self, report: dict[str, Any]) -> list[dict[str, Any]]:
        """
        Turn a Node 3 align_documents() report into changed pairs.

        MODIFIED : source chunk + matched target chunk
        DELETED    : source chunk with NO_MATCH (Node 3 "removed")
        ADDED      : target chunk never matched (Node 3 "added")
        UNVERIFIED : source chunk Node 3 could not decide (verdict UNKNOWN)
        """
        pairs = []
        for r in report["results"]:
            if r["verdict"] == "UNKNOWN":
                best = r["matches"][0] if r["matches"] else {}
                pairs.append(_pair(UNVERIFIED, r["source_chunk_id"], None, {
                    "node3_status": r["status"],
                    "best_score": r["best_score"],
                    "best_candidate_chunk_id": best.get("candidate_chunk_id"),
                    "decided_by": best.get("decided_by"),
                    "gemma_raw": str(best.get("result", {}).get("raw", ""))[:200],
                }))
                continue
            if r["verdict"] == MODIFIED:
                match = next(m for m in r["matches"] if m["candidate_chunk_id"] == r["matched_chunk_id"])
                pairs.append(_pair(MODIFIED, r["source_chunk_id"], r["matched_chunk_id"], {
                    "similarity_score": match["similarity_score"],
                    "change_type": match["result"].get("change_type", ""),
                    "numeric_diff": match["numeric_diff"],
                    "decided_by": match["decided_by"],
                    "truncated": match["truncated"],
                }))
            elif r["verdict"] == "NO_MATCH":
                pairs.append(_pair(DELETED, r["source_chunk_id"], None, {
                    "node3_status": r["status"],
                    "best_score": r["best_score"],
                }))

        for chunk_id in report["added"]:
            pairs.append(_pair(ADDED, None, chunk_id, {}))

        return pairs

    # ============================================================
    # TAGGING
    # ============================================================

    def _needs_model(self, sentence: dict[str, Any]) -> bool:
        mode = self.config.model_check
        return mode == "all" or (mode == "ambiguous" and sentence["ambiguous"])

    def profile(self, text: str) -> dict[str, Any]:
        """Deontic profile of one chunk: tagged sentences, counts, primary label."""
        sentences = tag_text(text)

        for s in sentences:
            if self.checker is not None and self._needs_model(s):
                label, raw = self.checker.classify(s["text"], {c["label"] for c in s["cues"]})
                if label is not None:
                    s["regex_label"] = s["label"]
                    s["label"] = label
                    s["decided_by"] = "gemma"
                else:
                    s["gemma_raw"] = raw      # unusable answer; regex label kept

        counts = Counter(s["label"] for s in sentences if s["label"] != NONE)

        return {
            "primary_label": primary_label(sentences),
            "labels": sorted(counts, key=LABEL_PRIORITY.index),
            "counts": dict(counts),
            "sentences": sentences,
        }

    def enrich_pair(self, pair: dict[str, Any]) -> dict[str, Any]:
        """Attach chunk details and deontic labels to one changed pair."""
        src = self.chunk_by_id[pair["source_chunk_id"]] if pair["source_chunk_id"] else None
        tgt = self.chunk_by_id[pair["target_chunk_id"]] if pair["target_chunk_id"] else None

        src_profile = self.profile(src["text"]) if src else None
        tgt_profile = self.profile(tgt["text"]) if tgt else None

        src_labels = set(src_profile["labels"]) if src_profile else set()
        tgt_labels = set(tgt_profile["labels"]) if tgt_profile else set()

        # Shift is judged on the CHANGED sentences only, so unchanged duties in the
        # same clause don't hide the real change. It needs a changed duty sentence
        # on BOTH sides; a duty that was only added or removed shows up in
        # labels_added / labels_removed instead.
        src_changed, tgt_changed = mark_changed(src_profile, tgt_profile)
        shift = None
        if src_changed not in (None, NONE) and tgt_changed not in (None, NONE) and src_changed != tgt_changed:
            shift = f"{src_changed} -> {tgt_changed}"

        labels_added = sorted(tgt_labels - src_labels, key=LABEL_PRIORITY.index)
        labels_removed = sorted(src_labels - tgt_labels, key=LABEL_PRIORITY.index)

        return {
            **pair,
            "source": _chunk_view(src),
            "target": _chunk_view(tgt),
            "deontic": {
                "primary_label": (tgt_profile or src_profile)["primary_label"],
                "shift": shift,
                "labels_added": labels_added,
                "labels_removed": labels_removed,
                "has_deontic_change": bool(shift or labels_added or labels_removed),
                "source": src_profile,
                "target": tgt_profile,
            },
        }

    # ============================================================
    # CONTINUOUS PROCESSING
    # ============================================================

    def iter_run(
        self,
        report: dict[str, Any],
        checkpoint_path: str | None = None,
    ) -> Iterator[dict[str, Any]]:
        """
        Yield enriched pairs one at a time, straight from a Node 3 report.

        With checkpoint_path, every finished pair is appended to a JSONL file.
        On restart, pairs already in the file are yielded from disk instead of
        being processed again.
        """
        pairs = self.pairs_from_node3(report)
        fingerprint = self._fingerprint(pairs)
        done = _load_checkpoint(checkpoint_path, fingerprint) if checkpoint_path else {}

        fh = None
        if checkpoint_path:
            os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
            new_file = not os.path.exists(checkpoint_path) or not done
            fh = open(checkpoint_path, "w" if new_file else "a", encoding="utf-8")
            if new_file:
                fh.write(json.dumps({"_meta": {"node": 4, "fingerprint": fingerprint}}) + "\n")
                fh.flush()
            elif not _ends_with_newline(checkpoint_path):
                fh.write("\n")      # close a partial line left by a crash

        try:
            for pair in pairs:
                if pair["pair_id"] in done:
                    yield done[pair["pair_id"]]
                    continue
                enriched = self.enrich_pair(pair)
                if fh:
                    fh.write(json.dumps(enriched, ensure_ascii=False) + "\n")
                    fh.flush()
                yield enriched
        finally:
            if fh:
                fh.close()

    def run(
        self,
        report: dict[str, Any],
        checkpoint_path: str | None = None,
        verbose: bool = True,
    ) -> dict[str, Any]:
        """Process every changed pair and return the Node 4 output for Node 5."""
        total = len(self.pairs_from_node3(report))
        pairs, t0 = [], time.time()

        for i, p in enumerate(self.iter_run(report, checkpoint_path), 1):
            pairs.append(p)
            if verbose and (i % 10 == 0 or i == total):
                elapsed = time.time() - t0
                eta = elapsed / i * (total - i)
                print(f"[{i}/{total}] elapsed {elapsed:.0f}s | eta {eta:.0f}s")

        skipped = Counter(r["verdict"] for r in report["results"]
                          if r["verdict"] not in (MODIFIED, "NO_MATCH", "UNKNOWN"))
        return {
            "pairs": pairs,
            "counts": dict(Counter(p["change_status"] for p in pairs)),
            "label_counts": dict(Counter(p["deontic"]["primary_label"] for p in pairs)),
            "deontic_changes": [p["pair_id"] for p in pairs if p["deontic"]["has_deontic_change"]],
            "skipped": dict(skipped),
        }

    def _fingerprint(self, pairs: list[dict[str, Any]]) -> str:
        """Identifies the exact input, so a checkpoint from other documents is never reused."""
        h = hashlib.sha256()
        for p in pairs:
            h.update(p["pair_id"].encode())
            for cid in (p["source_chunk_id"], p["target_chunk_id"]):
                if cid:
                    h.update(self.chunk_by_id[cid]["text"].encode())
        h.update(self.config.model_check.encode())
        h.update(NODE4_VERSION.encode())
        return h.hexdigest()[:16]


# ============================================================
# HELPERS
# ============================================================

def primary_label(sentences: list[dict[str, Any]]) -> str:
    """Most frequent label; ties broken by PROHIBITION > OBLIGATION > PERMISSION."""
    counts = Counter(s["label"] for s in sentences if s["label"] != NONE)
    if not counts:
        return NONE
    top = max(counts.values())
    return min((l for l, n in counts.items() if n == top), key=LABEL_PRIORITY.index)


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def mark_changed(src_profile, tgt_profile) -> list[str | None]:
    """
    Flag each sentence as changed or not. In a MODIFIED pair a sentence is
    unchanged if the exact same sentence exists on the other side.
    Returns the primary label of the CHANGED sentences on each side.
    """
    src_texts = {_norm(s["text"]) for s in src_profile["sentences"]} if src_profile else set()
    tgt_texts = {_norm(s["text"]) for s in tgt_profile["sentences"]} if tgt_profile else set()
    for p, other in ((src_profile, tgt_texts), (tgt_profile, src_texts)):
        if p:
            for s in p["sentences"]:
                s["changed"] = _norm(s["text"]) not in other
    return [
        primary_label([s for s in p["sentences"] if s["changed"]]) if p else None
        for p in (src_profile, tgt_profile)
    ]


def _pair(status, source_id, target_id, alignment):
    return {
        "pair_id": f"{status}:{source_id or '-'}->{target_id or '-'}",
        "change_status": status,
        "source_chunk_id": source_id,
        "target_chunk_id": target_id,
        "alignment": alignment,
    }


def _chunk_view(chunk):
    if chunk is None:
        return None
    return {k: chunk.get(k) for k in
            ("chunk_id", "document_id", "version_id", "heading_path", "page_start", "page_end", "text")}


def _load_checkpoint(path: str, fingerprint: str) -> dict[str, dict[str, Any]]:
    if not os.path.exists(path):
        return {}
    done = {}
    with open(path, encoding="utf-8") as fh:
        try:
            meta = json.loads(fh.readline()).get("_meta", {})
        except json.JSONDecodeError:
            return {}               # header never finished writing; start fresh
        if meta.get("fingerprint") != fingerprint:
            print(f"Checkpoint {path} is from different input or an older Node 4 — starting fresh.")
            return {}
        for line in fh:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue            # partial last line from a crash
            done[row["pair_id"]] = row
    return done


def _ends_with_newline(path: str) -> bool:
    with open(path, "rb") as fh:
        fh.seek(0, os.SEEK_END)
        if fh.tell() == 0:
            return True
        fh.seek(-1, os.SEEK_END)
        return fh.read(1) == b"\n"


def read_node4_output(path: str) -> list[dict[str, Any]]:
    """Load enriched pairs from a Node 4 checkpoint file (for Node 5)."""
    with open(path, encoding="utf-8") as fh:
        rows = []
        for line in fh:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "_meta" not in row:
                rows.append(row)
    return rows
