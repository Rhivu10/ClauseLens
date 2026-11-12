"""
ClauseLens — Node 3 (performance-tuned)
Clause aligner.

Per source chunk:

    Node 2 retrieval (target document only)
          |
          +----> similarity gate  --(below)-->  NO_MATCH (no Gemma call)
          |
          +----> exact-text / format-only check --(same)-->  EQUIVALENT
          |
          +----> verdict cache / duplicate-text memo --(hit)-->  stored verdict
          |
          +----> structural match? (score >= structural_score, or same heading)
          |          yes: Gemma answers EQUIVALENT | MODIFIED only
          |          no : Gemma answers EQUIVALENT | MODIFIED | NO_MATCH
          |
          +----> Gemma verifier  ------------>  EQUIVALENT | MODIFIED | NO_MATCH

Document level: unmatched source chunks are REMOVED, unmatched target chunks
are ADDED.

Performance-tuned align_documents() runs in three phases instead of one
sequential loop:

  Phase A (no model): retrieve candidates for every source chunk with one
      matrix product against the precomputed Node 2 embeddings (no
      re-encoding per query, no whole-index scans, no O(N^2) chunk lookups),
      apply the similarity gate, and resolve everything that needs no model:
      exact-text / format-only shortcuts, cached verdicts and duplicate text
      pairs.

  Phase B (model, batched): verify the remaining pairs in "candidate
      rounds". Round 1 verifies the top-scored candidate of every chunk in
      batches of config.batch_size; a chunk whose top candidate is EQUIVALENT
      is finished (nothing can beat it — verdict priority plus score-order
      tie-breaking), so its remaining candidates are never sent. Rounds 2..k
      handle only the chunks that still need them. Unreadable answers are
      retried in one batched one-word pass. CUDA OOM splits the batch.

  Phase C (no model): assemble the same report format as before.

The single-clause API (align_source_clause / compare_pair) is unchanged.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import time
from typing import Any

import numpy as np
import torch

from nodes.node_2 import semantic_search

from .config import Node3Config, PROMPT_VERSION, VERDICT_PRIORITY
from .helpers import formatting_key, normalize_text, numeric_diff, parse_verdict
from .verifier import GemmaVerifier, OOM_RAW


def _decide(
    raw_first: str,
    truncated_first: bool,
    raw_second: str | None,
    truncated_second: bool | None,
    structural,
) -> dict[str, Any]:
    """
    Apply the verdict rules of the original compare_pair:

      parse -> (UNKNOWN and non-structural? retry parse) -> structural override.

    raw_second is the one-word retry answer, when a retry happened.
    """
    def oom(first_raw=None):
        rec = {"result": {"alignment_result": "UNKNOWN", "change_type": "", "raw": "CUDA OOM"},
               "decided_by": "error", "truncated": None}
        if first_raw is not None:
            rec["first_raw"] = first_raw
        return rec

    if raw_first == OOM_RAW and raw_second is None:
        return oom()

    result = parse_verdict(raw_first)
    decided_by = "gemma"
    first_raw = None
    truncated = truncated_first

    if raw_second is not None:
        if raw_second == OOM_RAW:
            return oom(raw_first[:200])
        first_raw = raw_first[:200]
        result = parse_verdict(raw_second)
        decided_by = "gemma_retry"
        truncated = truncated_second

    gemma_verdict = None
    if structural and result["alignment_result"] in ("NO_MATCH", "UNKNOWN"):
        gemma_verdict = result["alignment_result"]
        result = {"alignment_result": "MODIFIED", "change_type": result.get("change_type", "")}
        decided_by = "structure"

    return {"result": result, "decided_by": decided_by, "truncated": truncated,
            "first_raw": first_raw, "gemma_verdict": gemma_verdict}


def _apply_decision(entry: dict[str, Any], rec: dict[str, Any]) -> None:
    """Write a _decide() record into a match entry (same fields as before)."""
    entry.pop("_pending", None)
    entry["result"] = rec["result"]
    entry["decided_by"] = rec["decided_by"]
    entry["truncated"] = rec["truncated"]
    if rec.get("first_raw"):
        entry["first_raw"] = rec["first_raw"]
    if rec.get("gemma_verdict") is not None:
        entry["gemma_verdict"] = rec["gemma_verdict"]


def _rank_entries(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Best entry wins: EQUIVALENT > MODIFIED > NO_MATCH > UNKNOWN, ties by order."""
    ranked = sorted(
        range(len(entries)),
        key=lambda i: (VERDICT_PRIORITY.index(entries[i]["result"]["alignment_result"]), i),
    )
    return entries[ranked[0]]


def _heading(chunk: dict[str, Any]) -> str:
    """Last heading of a chunk, normalized ("2.1 Grant of License" -> "2.1 grant of license")."""
    path = chunk.get("heading_path") or []
    return " ".join(str(path[-1]).lower().split()) if path else ""


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
        Only used when the precomputed-embedding fast path is unavailable.

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

        # Precomputed-embedding state (False = unavailable)
        self._matrix = None
        self._row_by_chunk: dict[str, int] | None = None
        self._doc_cache: dict[str, tuple[list[str], Any]] = {}

        # Verdict cache (disk, optional) and duplicate-text memo (per run)
        self._cache: dict[str, dict[str, Any]] = {}
        self._cache_file: str | None = None
        self._text_memo: dict[tuple[str, str, bool], dict[str, Any]] = {}
        if self.config.cache_path:
            self._load_cache(self.config.cache_path)

    # ============================================================
    # PRECOMPUTED EMBEDDINGS
    # ============================================================

    def _ensure_matrix(self) -> bool:
        """True when the Node 2 embedding matrix matches the chunk list."""
        if self._matrix is not None:
            return self._matrix is not False
        emb = getattr(self.index_bundle, "embeddings", None)
        mapping = getattr(self.index_bundle, "vector_id_to_chunk_id", None)
        if emb is None or mapping is None or not self.config.use_precomputed_embeddings:
            self._matrix = False
            return False
        emb = np.asarray(emb)
        if emb.ndim != 2 or emb.shape[0] != len(self.all_chunks):
            self._matrix = False
            return False
        rows = {}
        for i, chunk in enumerate(self.all_chunks):
            if mapping.get(i) != chunk["chunk_id"]:
                self._matrix = False
                return False
            rows[chunk["chunk_id"]] = i
        self._matrix = emb
        self._row_by_chunk = rows
        return True

    def _doc_rows(self, target_document_id: str):
        """(chunk_ids, embedding rows) for one document, cached."""
        if target_document_id not in self._doc_cache:
            ids = [c["chunk_id"] for c in self.all_chunks if c["document_id"] == target_document_id]
            if ids:
                rows = self._matrix[[self._row_by_chunk[cid] for cid in ids]]
            else:
                rows = np.zeros((0, self._matrix.shape[1]), dtype=self._matrix.dtype)
            self._doc_cache[target_document_id] = (ids, rows)
        return self._doc_cache[target_document_id]

    def _format_candidate(self, cand_chunk: dict[str, Any], score: float) -> dict[str, Any]:
        """Same result shape as Node 2 _format_result."""
        return {
            "chunk_id": cand_chunk["chunk_id"],
            "document_id": cand_chunk["document_id"],
            "version_id": cand_chunk["version_id"],
            "page_start": cand_chunk["page_start"],
            "page_end": cand_chunk["page_end"],
            "heading_path": cand_chunk["heading_path"],
            "text": cand_chunk["text"],
            "score": float(score),
            "retrieval_method": "semantic",
        }

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
        Retrieve target-document candidates.

        Fast path: cosine scores against the target document's rows of the
        Node 2 embedding matrix — the same scores the FAISS flat search
        returns, without re-encoding the query and scanning the whole index.

        Fallback: the original Node 2 semantic_search over the whole index.
        """
        candidate_k = self.config.candidate_k if candidate_k is None else candidate_k

        if source_chunk["document_id"] == target_document_id:
            raise ValueError(
                f"Source chunk {source_chunk['chunk_id']} is from '{target_document_id}' "
                f"— source and target must be different documents (self-match)."
            )

        if self._ensure_matrix() and source_chunk["chunk_id"] in self._row_by_chunk:
            ids, rows = self._doc_rows(target_document_id)
            if not ids:
                return []
            query = self._matrix[self._row_by_chunk[source_chunk["chunk_id"]]]
            sims = rows @ query
            k = min(candidate_k, len(ids))
            order = np.argsort(-sims)[:k]
            return [self._format_candidate(self.chunk_by_id[ids[int(i)]], float(sims[int(i)]))
                    for i in order]

        # Original path (kept for bundles the matrix cannot describe)
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
    # VERDICT CACHE / MEMO
    # ============================================================

    def _cache_key(self, source_id: str, cand_id: str, allow: bool) -> str:
        raw = f"{PROMPT_VERSION}|{source_id}|{cand_id}|{int(allow)}"
        return hashlib.sha1(raw.encode()).hexdigest()

    def _load_cache(self, path: str) -> None:
        self._cache_file = path
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    self._cache[rec["key"]] = rec
                except (json.JSONDecodeError, KeyError):
                    continue

    def _cache_write(self, key: str, record: dict[str, Any]) -> None:
        self._cache[key] = record
        if not self._cache_file:
            return
        try:
            with open(self._cache_file, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"key": key, **record}, ensure_ascii=False) + "\n")
        except OSError:
            self._cache_file = None

    # ============================================================
    # PAIR COMPARISON (single-pair API, unchanged behaviour + caching)
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

        # Deterministic shortcuts: identical text, or text that differs only
        # in formatting, needs no model call
        if s.lower() == t.lower() or formatting_key(s) == formatting_key(t):
            entry.update({
                "result": {"alignment_result": "EQUIVALENT", "change_type": ""},
                "decided_by": "exact_text" if s.lower() == t.lower() else "format_only",
                "truncated": False,
            })
            return entry

        allow = structural is None
        key = self._cache_key(source_chunk["chunk_id"], candidate["chunk_id"], allow)
        if key in self._cache:
            _apply_decision(entry, self._cache[key])
            return entry

        try:
            raw_first, trunc_first = self.verifier.compare_clauses(s, t, allow_no_match=allow)
            raw_second, trunc_second = None, None
            if parse_verdict(raw_first)["alignment_result"] == "UNKNOWN" and not structural:
                raw_second, trunc_second = self.verifier.retry_clauses(s, t, allow_no_match=True)
            rec = _decide(raw_first, trunc_first, raw_second, trunc_second, structural)
            _apply_decision(entry, rec)
            if rec["decided_by"] in ("gemma", "gemma_retry", "structure"):
                self._cache_write(key, rec)
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
        best = _rank_entries(matches)
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
    # DOCUMENT LEVEL (batched, round-based)
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

        Same report format as the original implementation; see the module
        docstring for the three-phase, batched execution.
        """
        t0 = time.time()
        source_chunks = [c for c in self.all_chunks if c["document_id"] == source_doc_id]
        target_chunks = [c for c in self.all_chunks if c["document_id"] == target_doc_id]
        if max_source_chunks:
            source_chunks = source_chunks[:max_source_chunks]

        # ---------------- Phase A: candidates + shortcuts (no model) -------
        plans: list[dict[str, Any]] = []
        model_pairs = 0
        for sc in source_chunks:
            plan = {"chunk": sc, "status": "COMPARED", "best_score": None, "entries": []}
            plans.append(plan)
            candidates = self.get_candidates(sc, target_doc_id)
            if not candidates:
                plan["status"] = "NO_CANDIDATES"
                continue
            plan["best_score"] = round(float(candidates[0]["score"]), 4)
            gated = [c for c in candidates if self.passes_similarity_gate(c)]
            if not gated:
                plan["status"] = "BELOW_THRESHOLD"
                continue
            for cand in gated:
                cand_chunk = self.chunk_by_id[cand["chunk_id"]]
                s = normalize_text(sc["text"])
                t = normalize_text(cand_chunk["text"])
                structural = self.structural_match(sc, cand_chunk, float(cand["score"]))
                entry = {
                    "candidate_chunk_id": cand["chunk_id"],
                    "similarity_score": round(float(cand["score"]), 4),
                    "numeric_diff": numeric_diff(s, t),
                    "structural_match": structural,
                }
                if s.lower() == t.lower() or formatting_key(s) == formatting_key(t):
                    entry["result"] = {"alignment_result": "EQUIVALENT", "change_type": ""}
                    entry["decided_by"] = "exact_text" if s.lower() == t.lower() else "format_only"
                    entry["truncated"] = False
                else:
                    allow = structural is None
                    key = self._cache_key(sc["chunk_id"], cand["chunk_id"], allow)
                    if key in self._cache:
                        _apply_decision(entry, self._cache[key])
                    else:
                        entry["_pending"] = (s, t, structural)
                        model_pairs += 1
                plan["entries"].append(entry)

        if verbose:
            print(f"[Node 3] {len(source_chunks)} source chunks | {len(target_chunks)} target chunks "
                  f"| {model_pairs} pairs need Gemma | phase A {time.time() - t0:.1f}s")

        # ---------------- Phase B: batched verification rounds -------------
        def top_is_equivalent(plan) -> bool:
            return (
                self.config.early_exit_equivalent
                and plan["entries"]
                and "result" in plan["entries"][0]
                and plan["entries"][0]["result"]["alignment_result"] == "EQUIVALENT"
            )

        def has_pending(plan) -> bool:
            return any("_pending" in e for e in plan["entries"])

        # Chunks already decided at rank 0 (shortcut / cache / memo hits):
        # lower candidates are skipped — they can never win, and phase C
        # keeps only entries that carry a final result.
        early_done: set[int] = set()
        for idx, plan in enumerate(plans):
            if plan["status"] == "COMPARED" and top_is_equivalent(plan):
                early_done.add(idx)
                for e in plan["entries"]:
                    e.pop("_pending", None)   # lower candidates can never win

        round_no = 0
        while True:
            active = [i for i, p in enumerate(plans)
                      if p["status"] == "COMPARED" and i not in early_done and has_pending(p)]
            if not active:
                break
            round_no += 1
            if round_no > self.config.candidate_k:
                break
            pos = round_no - 1

            # Collect this round's pairs; resolve memo hits without the model
            specs: list[tuple[int, dict[str, Any]]] = []
            for i in active:
                plan = plans[i]
                if pos < len(plan["entries"]) and "_pending" in plan["entries"][pos]:
                    entry = plan["entries"][pos]
                    s, t, structural = entry["_pending"]
                    memo_key = (formatting_key(s), formatting_key(t), structural is None)
                    if self.config.dedup_text_pairs and memo_key in self._text_memo:
                        rec = self._text_memo[memo_key]
                        _apply_decision(entry, rec)
                    else:
                        specs.append((i, entry, memo_key))

            if specs:
                self._run_round(plans, specs, verbose, t0, round_no, len(plans))

            # Early exit: top candidate EQUIVALENT -> the chunk is done
            for i in active:
                if top_is_equivalent(plans[i]):
                    early_done.add(i)
                    for e in plans[i]["entries"]:
                        e.pop("_pending", None)

        # ---------------- Phase C: assemble (same format as before) --------
        results, matched_targets = [], set()
        for plan in plans:
            base = {"source_chunk_id": plan["chunk"]["chunk_id"]}
            if plan["status"] != "COMPARED":
                results.append({**base, "status": plan["status"], "verdict": "NO_MATCH",
                                "best_score": plan["best_score"], "matched_chunk_id": None,
                                "matches": []})
                continue
            matches = [{k: v for k, v in e.items() if not k.startswith("_")}
                       for e in plan["entries"] if "result" in e]
            if not matches:
                results.append({**base, "status": "BELOW_THRESHOLD", "verdict": "NO_MATCH",
                                "best_score": plan["best_score"], "matched_chunk_id": None,
                                "matches": []})
                continue
            best = _rank_entries(matches)
            verdict = best["result"]["alignment_result"]
            results.append({
                **base,
                "status": "COMPARED",
                "verdict": verdict,
                "best_score": plan["best_score"],
                "matched_chunk_id": best["candidate_chunk_id"] if verdict in ("EQUIVALENT", "MODIFIED") else None,
                "matches": matches,
            })
            if verdict in ("EQUIVALENT", "MODIFIED"):
                matched_targets.add(best["candidate_chunk_id"])

        if verbose:
            counts = {v: sum(r["verdict"] == v for r in results) for v in VERDICT_PRIORITY}
            print(f"[Node 3] done in {time.time() - t0:.0f}s | counts {counts}")

        return {
            "results": results,
            "removed": [r["source_chunk_id"] for r in results if r["verdict"] == "NO_MATCH"],
            "added": [c["chunk_id"] for c in target_chunks if c["chunk_id"] not in matched_targets],
            "counts": {v: sum(r["verdict"] == v for r in results) for v in VERDICT_PRIORITY},
        }

    # ------------------------------------------------------------
    # ONE VERIFICATION ROUND
    # ------------------------------------------------------------

    def _run_round(self, plans, specs, verbose, t0, round_no, total_plans) -> None:
        """
        Verify this round's pairs: group duplicates, one batched first pass,
        one batched retry pass for unreadable answers, then finalize entries,
        memo and cache.
        """
        # Group duplicate text pairs; the model sees one pair per group
        groups: dict[tuple, list[tuple[int, dict[str, Any]]]] = {}
        order: list[tuple] = []
        for i, entry, memo_key in specs:
            if memo_key not in groups:
                groups[memo_key] = []
                order.append(memo_key)
            groups[memo_key].append((i, entry))

        pairs = []
        for memo_key in order:
            _i, entry = groups[memo_key][0]
            s, t, structural = entry["_pending"]
            pairs.append((s, t, structural is None))

        raws = self.verifier.compare_pairs(pairs)

        # Batched one-word retry for unreadable answers (non-structural only)
        retry_pos: dict[int, int] = {}      # group index j -> position in retry batch
        retry_pairs = []
        for j, memo_key in enumerate(order):
            _i, entry = groups[memo_key][0]
            s, t, structural = entry["_pending"]
            raw, _trunc = raws[j]
            if raw != OOM_RAW and structural is None and \
                    parse_verdict(raw)["alignment_result"] == "UNKNOWN":
                retry_pos[j] = len(retry_pairs)
                retry_pairs.append((s, t, True))
        retry_raws = self.verifier.retry_pairs(retry_pairs) if retry_pairs else []

        # Finalize every group (all members share the decision)
        for j, memo_key in enumerate(order):
            members = groups[memo_key]
            _i, first_entry = members[0]
            raw_first, trunc_first = raws[j]
            raw_second, trunc_second = (None, None)
            if j in retry_pos:
                raw_second, trunc_second = retry_raws[retry_pos[j]]
            structural = first_entry["_pending"][2]
            rec = _decide(raw_first, trunc_first, raw_second, trunc_second, structural)

            for i, entry in members:
                _apply_decision(entry, rec)

            if rec["decided_by"] in ("gemma", "gemma_retry", "structure"):
                if self.config.dedup_text_pairs:
                    self._text_memo[memo_key] = rec
                for i, entry in members:
                    plan = plans[i]
                    key = self._cache_key(plan["chunk"]["chunk_id"], entry["candidate_chunk_id"],
                                          structural is None)
                    self._cache_write(key, rec)

        if verbose:
            decided = sum(1 for p in plans if p["status"] != "COMPARED"
                          or not has_pending_entry(p))
            print(f"[Node 3] round {round_no}: {len(pairs)} pairs verified in "
                  f"{self._batch_count(len(pairs))} batches | {decided}/{total_plans} chunks decided "
                  f"| elapsed {time.time() - t0:.0f}s")

    def _batch_count(self, pairs: int) -> int:
        size = max(1, self.config.batch_size)
        return (pairs + size - 1) // size

    # ============================================================
    # THRESHOLD CALIBRATION
    # ============================================================

    def best_scores(self, source_doc_id: str, target_doc_id: str) -> list[tuple[str, float]]:
        """Best target similarity score for every source chunk (no Gemma calls)."""
        if self._ensure_matrix():
            src_ids = [c["chunk_id"] for c in self.all_chunks if c["document_id"] == source_doc_id]
            tgt_ids, tgt_rows = self._doc_rows(target_doc_id)
            if not src_ids:
                return []
            if not tgt_ids:
                return [(cid, 0.0) for cid in src_ids]
            src_rows = self._matrix[[self._row_by_chunk[cid] for cid in src_ids]]
            sims = src_rows @ tgt_rows.T
            best = sims.max(axis=1)
            return [(cid, float(x)) for cid, x in zip(src_ids, best)]

        rows = []
        for c in self.all_chunks:
            if c["document_id"] != source_doc_id:
                continue
            cand = self.get_candidates(c, target_doc_id, candidate_k=1)
            rows.append((c["chunk_id"], float(cand[0]["score"]) if cand else 0.0))
        return rows


def has_pending_entry(plan) -> bool:
    return any("_pending" in e for e in plan["entries"])


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
