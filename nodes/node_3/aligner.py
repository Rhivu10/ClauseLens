"""
ClauseLens — Node 3
Clause aligner.

For each source chunk:

    Node 2 semantic search (target document only)
          |
          +----> similarity gate  --(below)-->  NO_MATCH (no Gemma call)
          |
          +----> exact-text check --(same)--->  EQUIVALENT
          |
          +----> structural match? (score >= structural_score, or same heading)
          |          yes: Gemma decides EQUIVALENT | MODIFIED only
          |          no : Gemma decides EQUIVALENT | MODIFIED | NO_MATCH
          |
          +----> Gemma verifier  ------------>  EQUIVALENT | MODIFIED | NO_MATCH

Document level: unmatched source chunks are REMOVED,
unmatched target chunks are ADDED.
"""

from __future__ import annotations

import gc
import time
from typing import Any

import torch

from nodes.node_2 import semantic_search

from .config import Node3Config, VERDICT_PRIORITY
from .helpers import normalize_text, numeric_diff, parse_verdict
from .verifier import GemmaVerifier


class Node3Aligner:
    """
    Parameters
    ----------
    all_chunks:
        Canonical chunks from Node 2 prepare_chunks().

    index_bundle:
        IndexBundle from Node2Indexer.build().

    embedding_model:
        The embedding model used by Node 2 (indexer.embedding_model).

    verifier:
        GemmaVerifier wrapping the fine-tuned model.
    """

    def __init__(
        self,
        all_chunks: list[dict[str, Any]],
        index_bundle,
        embedding_model,
        verifier: GemmaVerifier,
        config: Node3Config | None = None,
    ):
        self.all_chunks = all_chunks
        self.index_bundle = index_bundle
        self.embedding_model = embedding_model
        self.verifier = verifier
        self.config = config or verifier.config
        self.chunk_by_id = {c["chunk_id"]: c for c in all_chunks}

    # ============================================================
    # CANDIDATE RETRIEVAL
    # ============================================================

    def get_candidates(
        self,
        source_chunk: dict[str, Any],
        target_document_id: str,
        candidate_k: int | None = None,
        retrieval_top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Retrieve target-document candidates from the existing Node 2 index.
        Searches the whole index (FAISS flat search is cheap) and then filters,
        so a big source document can't crowd the target out of the top-k.
        """
        candidate_k = self.config.candidate_k if candidate_k is None else candidate_k

        if source_chunk["document_id"] == target_document_id:
            raise ValueError(
                f"Source chunk {source_chunk['chunk_id']} is from '{target_document_id}' "
                f"— source and target must be different documents (self-match)."
            )

        top_k = retrieval_top_k or len(self.all_chunks)
        results = semantic_search(
            source_chunk["text"],
            self.index_bundle,
            self.embedding_model,
            top_k=top_k,
        )
        candidates = [r for r in results if r["document_id"] == target_document_id]
        if candidates and "score" not in candidates[0]:
            raise KeyError("semantic_search results have no 'score' key — check Node 2.")
        return candidates[:candidate_k]

    def passes_similarity_gate(self, candidate: dict[str, Any], threshold: float | None = None) -> bool:
        threshold = self.config.similarity_threshold if threshold is None else threshold
        return candidate["score"] >= threshold

    def structural_match(
        self,
        source_chunk: dict[str, Any],
        cand_chunk: dict[str, Any],
        score: float,
    ) -> str | None:
        """
        Returns "score" or "heading" when the two chunks are clearly the same
        clause, else None. A structural match can't be NO_MATCH: on Kaggle,
        Gemma answered NO_MATCH for clauses scoring 0.99 whose duty was reversed.
        """
        if score >= self.config.structural_score:
            return "score"
        hs, ht = _heading(source_chunk), _heading(cand_chunk)
        if hs and hs == ht:
            return "heading"
        return None

    # ============================================================
    # PAIR COMPARISON
    # ============================================================

    def compare_pair(self, source_chunk: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
        """Compare one source chunk to one retrieved candidate."""
        cand_chunk = self.chunk_by_id[candidate["chunk_id"]]
        s = normalize_text(source_chunk["text"])
        t = normalize_text(cand_chunk["text"])
        diff = numeric_diff(s, t)
        structural = self.structural_match(source_chunk, cand_chunk, float(candidate["score"]))

        entry = {
            "candidate_chunk_id": candidate["chunk_id"],
            "similarity_score": round(float(candidate["score"]), 4),
            "numeric_diff": diff,
            "structural_match": structural,
        }

        # Deterministic shortcut: identical text needs no model call
        if s.lower() == t.lower():
            entry.update({
                "result": {"alignment_result": "EQUIVALENT", "change_type": ""},
                "decided_by": "exact_text",
                "truncated": False,
            })
            return entry

        try:
            raw, truncated = self.verifier.compare_clauses(s, t, allow_no_match=structural is None)
            result = parse_verdict(raw)
            entry["decided_by"] = "gemma"
            if structural and result["alignment_result"] in ("NO_MATCH", "UNKNOWN"):
                # Same clause but the text differs: MODIFIED, whatever Gemma said
                entry["gemma_verdict"] = result["alignment_result"]
                result = {"alignment_result": "MODIFIED", "change_type": result.get("change_type", "")}
                entry["decided_by"] = "structure"
            entry["result"] = result
            entry["truncated"] = truncated
        except torch.cuda.OutOfMemoryError:
            gc.collect()
            torch.cuda.empty_cache()
            entry["result"] = {"alignment_result": "UNKNOWN", "change_type": "", "raw": "CUDA OOM"}
            entry["decided_by"] = "error"
            entry["truncated"] = None

        return entry

    # ============================================================
    # SINGLE CLAUSE
    # ============================================================

    def align_source_clause(
        self,
        source_chunk: dict[str, Any],
        target_document_id: str,
        threshold: float | None = None,
    ) -> dict[str, Any]:
        """
        Returns an explicit result dict (never a bare []):

          status:  NO_CANDIDATES  - nothing retrieved from the target document
                   BELOW_THRESHOLD- best embedding score under the gate (no Gemma call)
                   COMPARED       - Gemma / exact-text comparison ran
          verdict: EQUIVALENT | MODIFIED | NO_MATCH | UNKNOWN
        """
        base = {"source_chunk_id": source_chunk["chunk_id"]}
        candidates = self.get_candidates(source_chunk, target_document_id)

        if not candidates:
            return {**base, "status": "NO_CANDIDATES", "verdict": "NO_MATCH",
                    "best_score": None, "matched_chunk_id": None, "matches": []}

        best_score = round(float(candidates[0]["score"]), 4)
        gated = [c for c in candidates if self.passes_similarity_gate(c, threshold)]

        if not gated:
            return {**base, "status": "BELOW_THRESHOLD", "verdict": "NO_MATCH",
                    "best_score": best_score, "matched_chunk_id": None, "matches": []}

        matches = [self.compare_pair(source_chunk, c) for c in gated]

        # Prefer a real match over NO_MATCH even if it is not the top-scored candidate
        ranked = sorted(
            range(len(matches)),
            key=lambda i: (VERDICT_PRIORITY.index(matches[i]["result"]["alignment_result"]), i),
        )
        best = matches[ranked[0]]
        verdict = best["result"]["alignment_result"]

        return {
            **base,
            "status": "COMPARED",
            "verdict": verdict,
            "best_score": best_score,
            "matched_chunk_id": best["candidate_chunk_id"] if verdict in ("EQUIVALENT", "MODIFIED") else None,
            "matches": matches,
        }

    # ============================================================
    # DOCUMENT LEVEL
    # ============================================================

    def align_documents(
        self,
        source_doc_id: str,
        target_doc_id: str,
        max_source_chunks: int | None = None,
        verbose: bool = True,
    ) -> dict[str, Any]:
        """
        Align every source chunk against the target document.
        Greedy (a target chunk may be matched by several sources). Target chunks
        never matched are reported as ADDED; source chunks with NO_MATCH as REMOVED.
        """
        source_chunks = [c for c in self.all_chunks if c["document_id"] == source_doc_id]
        target_chunks = [c for c in self.all_chunks if c["document_id"] == target_doc_id]
        if max_source_chunks:
            source_chunks = source_chunks[:max_source_chunks]

        results, matched_targets = [], set()
        t0 = time.time()

        for i, sc in enumerate(source_chunks, 1):
            r = self.align_source_clause(sc, target_doc_id)
            results.append(r)
            if r["matched_chunk_id"]:
                matched_targets.add(r["matched_chunk_id"])
            if verbose and (i % 10 == 0 or i == len(source_chunks)):
                elapsed = time.time() - t0
                eta = elapsed / i * (len(source_chunks) - i)
                print(f"[{i}/{len(source_chunks)}] elapsed {elapsed:.0f}s | eta {eta:.0f}s")

        return {
            "results": results,
            "removed": [r["source_chunk_id"] for r in results if r["verdict"] == "NO_MATCH"],
            "added": [c["chunk_id"] for c in target_chunks if c["chunk_id"] not in matched_targets],
            "counts": {v: sum(r["verdict"] == v for r in results) for v in VERDICT_PRIORITY},
        }

    # ============================================================
    # THRESHOLD CALIBRATION
    # ============================================================

    def best_scores(self, source_doc_id: str, target_doc_id: str) -> list[tuple[str, float]]:
        """Best target similarity score for every source chunk (no Gemma calls)."""
        rows = []
        for c in self.all_chunks:
            if c["document_id"] != source_doc_id:
                continue
            cand = self.get_candidates(c, target_doc_id, candidate_k=1)
            rows.append((c["chunk_id"], float(cand[0]["score"]) if cand else 0.0))
        return rows


def _heading(chunk: dict[str, Any]) -> str:
    """Last heading of a chunk, normalized ("2.1 Grant of License" -> "2.1 grant of license")."""
    path = chunk.get("heading_path") or []
    return " ".join(str(path[-1]).lower().split()) if path else ""


# ============================================================
# DISPLAY
# ============================================================

def show(result: dict[str, Any], source_chunk: dict[str, Any] | None = None) -> None:
    if source_chunk is not None:
        print("SOURCE:", source_chunk["chunk_id"], "| len:", len(source_chunk["text"]),
              "| heading:", source_chunk["heading_path"][-1:])
        print(source_chunk["text"][:300].replace("\n", " "))
        print()
    print("status :", result["status"])
    print("verdict:", result["verdict"], "| best_score:", result["best_score"],
          "| matched:", result["matched_chunk_id"])
    for m in result["matches"]:
        print("  ->", m["candidate_chunk_id"], "| score", m["similarity_score"],
              "|", m["result"], "| by", m["decided_by"], "| truncated:", m["truncated"],
              "| structural:", m.get("structural_match"))
        if "gemma_verdict" in m:
            print("     Gemma said", m["gemma_verdict"], "— overridden: same clause, text differs")
        if m["numeric_diff"]["only_in_source"] or m["numeric_diff"]["only_in_target"]:
            print("     numeric_diff:", m["numeric_diff"])
