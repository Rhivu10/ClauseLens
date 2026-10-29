"""
ClauseLens — Pipeline
Clean-up between Node 1 and Node 2 (Node 1 and Node 2 code are unchanged).

The same clean-up the real-contract notebook runs (CELL 8):
  1. removes browser print / EDGAR noise lines (date, file name, URL, "2/23", "EX-10.1 ... .htm")
  2. splits long chunks: one per definition ("Territory" means ...), one per numbered
     section ("11. Insurance."), then at line boundaries. Node 1 doesn't enforce
     max_chars, and Node 3 only reads the first 1,200 characters of a chunk.
  3. merges tiny fragments (table headings like "Date Filed", "SCHEDULE A", a lone "1.")
     into the next chunk
"""

from __future__ import annotations

import re
from typing import Any

_NOISE_LINE = re.compile(
    r"^\s*(?:"
    r"\d{1,2}/\d{1,2}/\d{2,4},?\s+\d{1,2}:\d{2}\s*(?:AM|PM)?"   # 10/18/26, 1:14 PM
    r"|https?://\S+"                                             # page URL
    r"|ex[\w.\-]*?(?:\.htm)?"                                    # exv10w1, ex10-2.htm
    r"|\d{1,3}\s*/\s*\d{1,3}"                                    # 2/23
    r"|-?\s*\d{1,3}\s*-?"                                        # 4, -1-
    r"|EX-\d+(?:\.\d+)?\s+\d+\s+\S+\.htm\b.*"                    # EX-10.1 2 w74314exv10w1.htm EXHIBIT 10.1
    r")\s*$",
    re.I,
)

# “Territory” means ... / "Term" shall have the meaning ...
_DEF_START = re.compile(r"^\s*[“\"]([^”\"]{1,80})[”\"]\s+(?:means|has the meaning|shall mean|shall have)", re.I)

# "11.Insurance.", "17. Complete Agreement.", "26.Use of Name.Licensor ..." (top-level
# section with a short title; the text may follow the title without a space)
_SECTION_START = re.compile(r"^\s*(\d{1,2})\.\s*([A-Z][A-Za-z;,’'&\- ]{2,60}?)\.(?:\s|$|(?=[A-Z]))")

# "2.1 ", "10.3\xa0" (numbered subsection)
_SUBSECTION_START = re.compile(r"^\s*\d{1,2}\.\d{1,2}[\s\xa0]")


def _clean_text(text: str) -> str:
    return "\n".join(l for l in text.splitlines() if not _NOISE_LINE.match(l))


def _split_long(text: str, max_chars: int) -> list[tuple[str, str | None]]:
    """[(piece_text, heading_or_None)] for one long chunk."""
    lines = text.splitlines()
    pieces: list[tuple[list[str], str | None]] = []

    if sum(1 for l in lines if _DEF_START.match(l)) >= 3:
        # Definitions: one piece per definition, headed by the defined term
        for l in lines:
            m = _DEF_START.match(l)
            if m or not pieces:
                pieces.append(([], f"“{m.group(1)}”" if m else None))
            pieces[-1][0].append(l)
    else:
        # Numbered sections and subsections start a new piece
        for l in lines:
            sec = _SECTION_START.match(l)
            if pieces and (sec or _SUBSECTION_START.match(l)):
                pieces.append(([], f"{sec.group(1)}. {sec.group(2).strip()}." if sec else None))
            elif not pieces:
                pieces.append(([], f"{sec.group(1)}. {sec.group(2).strip()}." if sec else None))
            pieces[-1][0].append(l)

    # Anything still too long: split at line boundaries
    out: list[tuple[str, str | None]] = []
    for plines, head in pieces:
        cur, n = [], 0
        for l in plines:
            if cur and n + len(l) > max_chars:
                out.append(("\n".join(cur), head))
                cur, n = [], 0
            cur.append(l)
            n += len(l) + 1
        if cur:
            out.append(("\n".join(cur), head))
    return [(t, h) for t, h in out if t.strip()]


def clean_chunks(
    result: dict[str, Any],
    max_chars: int = 1500,
    min_chars: int = 40,
) -> dict[str, dict[str, int]]:
    """
    Clean the chunks of a process_documents() result in place and return stats.

    Chunk ids stay unique: a split chunk keeps its id with a suffix
    (document_a_0009_32); a merged fragment's text joins the next chunk's id.
    """
    stats = {}
    for doc_id, d in result.items():
        removed = split = merged = 0
        pieces: list[dict[str, Any]] = []

        # 1 + 2: drop noise lines, split long chunks
        for c in d["chunks"]:
            text = _clean_text(c["text"])
            if not text.strip():
                removed += 1
                continue
            # Long chunks are split; so is any chunk holding more than one numbered
            # section ("10. Indemnity. ... 11.Insurance. ..."), whatever its length,
            # because that means Node 1 missed a heading
            extra_sections = sum(1 for l in text.splitlines()[1:] if _SECTION_START.match(l))
            if len(text) > max_chars or extra_sections:
                parts = _split_long(text, max_chars)
            else:
                parts = [(text, None)]
            split += len(parts) > 1
            for k, (part, head) in enumerate(parts):
                path = list(c.get("heading_path") or [])
                if head:
                    path = (path[:-1] if len(parts) > 1 and path else path) + [head]
                pieces.append({
                    **c,
                    "chunk_id": c["chunk_id"] if len(parts) == 1 else f"{c['chunk_id']}_{k + 1:02d}",
                    "text": part,
                    "heading_path": path,
                })

        # 3: merge tiny fragments into the following chunk
        out: list[dict[str, Any]] = []
        carry: dict[str, Any] | None = None
        for c in pieces:
            if carry is not None:
                c = {**c, "text": carry["text"] + "\n" + c["text"],
                     "page_start": min(carry.get("page_start", c.get("page_start")), c.get("page_start"))}
                carry = None
            if len(c["text"].strip()) < min_chars:
                carry = c
                merged += 1
                continue
            out.append(c)
        if carry is not None:                       # a fragment at the very end joins the last chunk
            if out:
                out[-1] = {**out[-1], "text": out[-1]["text"] + "\n" + carry["text"]}
            else:
                out.append(carry)

        d["chunks"], d["chunk_count"] = out, len(out)
        stats[doc_id] = {"chunks": len(out), "noise_only_removed": removed,
                         "long_chunks_split": split, "tiny_chunks_merged": merged}
    return stats
