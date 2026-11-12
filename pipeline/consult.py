"""
ClauseLens — Consult: answer questions about a finished report with Gemma.

Answers are grounded in the report: for each question the most relevant changes
are found (embedding similarity, plus all changes needing review / with a duty
shift when the question asks for those), and Gemma may only use those changes.
It cites them by id ([C012]); citations to changes it was not given are removed.
An answer without citations is fine when it says the report has no such change
("nothing"); otherwise it is marked as unsupported.

    from pipeline.consult import Consultant
    c = Consultant(report, model, tokenizer)          # report = Node 6 report (report.json)
    c.answer("What changed about the advertising clause?")
    -> {"question", "answer", "citations": ["C002", ...], "used": [...], "supported": bool,
        "kind": "cited" | "nothing" | "unsupported"}

From a shell (GPU machine, after python pipeline/install.py):

    python -m pipeline.consult results/report.json "Which duties changed?" "What was removed?"
"""

from __future__ import annotations

import re
from typing import Any

CONSULT_MODEL = "unsloth/gemma-4-E4B-it"      # the same model as the Node 5 Impact Agent
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 8                                       # changes shown to the model per question
MAX_NEW_TOKENS = 320

_REVIEW_Q = re.compile(r"review|unsure|uncertain|low confidence|not sure", re.I)
_DUTY_Q = re.compile(r"\bdut(y|ies)\b|obligation|shall|must|permission|prohibit|allowed to", re.I)
_CITE = re.compile(r"\[(C\d{3,})\]")
_OTHER_BRACKET = re.compile(r"\s*\[(?!C\d{3,}\])[^\]\n]{1,40}\]")       # "[Report overview]" etc.
_NOTHING = re.compile(r"does ?n[o']t show|not (shown|mentioned|listed|covered|in the report)|no (listed )?changes?\b"
                      r"|none of the (listed )?changes|zero changes|there (are|were) no", re.I)


def _change_text(c: dict[str, Any], limit: int) -> str:
    """One change as the model sees it."""
    old = " ".join(((c["evidence"]["old"] or {}).get("text") or "").split())
    new = " ".join(((c["evidence"]["new"] or {}).get("text") or "").split())
    lines = [f"[{c['id']}] {c['type']} | section: {c['section']} | title: {c['title']}"]
    if c.get("duty", {}).get("shift"):
        lines.append(f"duty shift: {c['duty']['shift']}")
    if c.get("affected"):
        lines.append("affected: " + ", ".join(c["affected"]))
    lines.append(f"summary: {c['summary']}")
    if old:
        lines.append(f"old text: {old[:limit]}")
    if new:
        lines.append(f"new text: {new[:limit]}")
    if c.get("needs_review"):
        lines.append("note: this change needs human review")
    return "\n".join(lines)


def build_prompt(question: str, report: dict[str, Any], changes: list[dict[str, Any]], limit: int = 500) -> str:
    docs = report["documents"]
    names = f"{docs['document_a']['name']} (old) and {docs['document_b']['name']} (new)"
    listed = "\n\n".join(_change_text(c, limit) for c in changes) or "(no matching changes)"
    return f"""You answer questions about the differences between two versions of a contract: {names}.
Report overview: {report['overview']['headline']}; {report['overview']['unchanged_clauses']} clauses are unchanged.

Use ONLY the changes listed below. Do not add facts, numbers, dates or parties that are not in them.
Cite every change you use by its id in square brackets, for example [C003].
If the listed changes do not answer the question, say that the report does not show it.
Answer in plain English, in at most 5 sentences.

CHANGES:
{listed}

QUESTION: {question}

ANSWER:"""


def clean_answer(raw: str, allowed_ids: set[str]) -> dict[str, Any]:
    """Keep citations to changes the model was given; drop the others."""
    text = raw.strip()
    bad = [i for i in _CITE.findall(text) if i not in allowed_ids]
    for i in set(bad):
        text = text.replace(f"[{i}]", "")
    text = _OTHER_BRACKET.sub("", text)
    text = re.sub(r"\s+([.,;:])", r"\1", re.sub(r"[ \t]{2,}", " ", text)).strip()
    cites = list(dict.fromkeys(i for i in _CITE.findall(text) if i in allowed_ids))
    # kind: "cited" (backed by changes), "nothing" (says the report has no such change:
    # a valid answer that needs no citation), "unsupported" (claims without a source)
    kind = "cited" if cites else "nothing" if _NOTHING.search(text) else "unsupported"
    return {"answer": text, "citations": cites, "removed_citations": sorted(set(bad)),
            "supported": kind != "unsupported", "kind": kind}


class Consultant:
    """Answers questions about one report. model/tokenizer: a Gemma (e.g. load_agent_model(CONSULT_MODEL))."""

    def __init__(self, report: dict[str, Any], model, tokenizer, top_k: int = TOP_K, embedder=None):
        from nodes.node_5 import Node5Config
        from nodes.node_5.agents import GemmaAgent

        self.report = report
        self.changes = report["changes"]
        self.top_k = top_k
        cfg = Node5Config()
        cfg.repetition_penalty = 1.1
        cfg.max_seq_length = 4096           # room for TOP_K changes with their text
        self.agent = GemmaAgent(model, tokenizer, cfg)
        if embedder is None:
            from sentence_transformers import SentenceTransformer
            embedder = SentenceTransformer(EMBEDDING_MODEL)
        self.embedder = embedder
        texts = [f"{c['section']} {c['title']} {c['summary']} " + " ".join(c.get("affected") or [])
                 for c in self.changes]
        self._vectors = self.embedder.encode(texts, normalize_embeddings=True) if texts else None

    def relevant(self, question: str) -> list[dict[str, Any]]:
        """The changes the model may use for this question, most relevant first."""
        picked: list[dict[str, Any]] = []
        if _REVIEW_Q.search(question):
            picked += [c for c in self.changes if c.get("needs_review")]
        if _DUTY_Q.search(question):
            picked += [c for c in self.changes if c.get("duty", {}).get("shift")]
        if self._vectors is not None:
            q = self.embedder.encode([question], normalize_embeddings=True)[0]
            scores = self._vectors @ q
            for i in sorted(range(len(self.changes)), key=lambda i: -float(scores[i])):
                picked.append(self.changes[i])
        out, seen = [], set()
        for c in picked:
            if c["id"] not in seen:
                seen.add(c["id"])
                out.append(c)
            if len(out) == self.top_k:
                break
        return out

    def answer(self, question: str) -> dict[str, Any]:
        changes = self.relevant(question)
        raw = self.agent.generate(lambda limit: build_prompt(question, self.report, changes, limit // 2),
                                  MAX_NEW_TOKENS)
        out = clean_answer(raw, {c["id"] for c in changes})
        return {"question": question, **out, "used": [c["id"] for c in changes], "raw": raw[:2000]}


def main(argv: list[str] | None = None) -> int:
    import argparse
    import json
    import sys

    ap = argparse.ArgumentParser(prog="python -m pipeline.consult",
                                 description="Answer questions about a ClauseLens report with Gemma.")
    ap.add_argument("report", help="report.json from a ClauseLens run")
    ap.add_argument("questions", nargs="+")
    args = ap.parse_args(argv)

    import pipeline  # noqa: F401  (imports Unsloth first)
    import torch
    if not torch.cuda.is_available():
        print("No GPU found. Consult needs a CUDA GPU (e.g. Kaggle 'GPU T4 x2').", file=sys.stderr)
        return 1
    from nodes.node_5 import load_agent_model
    model, tokenizer = load_agent_model(base_model=CONSULT_MODEL, device_index=0)
    report = json.load(open(args.report, encoding="utf-8"))
    consultant = Consultant(report, model, tokenizer)
    for q in args.questions:
        a = consultant.answer(q)
        print(f"\nQ: {q}\nA: {a['answer']}\n   sources: {', '.join(a['citations']) or '(none — not supported by the report)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
