# Node 6 Documentation

## Overview

**Node 6** is the Report Assembler, the last node of ClauseLens.

It takes Node 5's results (one per changed clause, with its impact summary and confidence) and builds **one layered
JSON report**: a short overview on top, then the changes grouped by stakeholder and by section, then every change
in detail, down to the **exact sentences** in both documents that caused it. It also saves the report to disk
(backup) and can write a readable Markdown version.

Node 6 currently:

- Builds the report in four layers: overview → stakeholders / sections → changes → evidence.
- Links every change to its evidence: old and new clause text, pages, heading, and each sentence with **character
  offsets** and a `changed` flag, so a UI can highlight exactly what changed.
- Links the other way too: an **index** from each chunk and each page to the changes that touch it.
- Cleans up party names from Node 5 ("licensees", "Licensee / buyer", "either party", "INC") into one name per party,
  and counts "Both parties" changes for each of the contract's parties.
- Ranks the most important changes for the overview (duty shifts, number changes, removed obligations/prohibitions),
  and keeps unconfirmed (LOW_CONFIDENCE) changes out of the highlights but in a **needs review** list.
- Accepts results **one at a time**, in any order, as Node 5 finishes them; `build()` can be called at any point.
- Saves the JSON safely (temporary file, then rename) and writes a Markdown report.
- Converts the report into the data format of the **ClauseLens UI** (`clauselens-ui`, React + Vite, run with Node.js), so real results replace the UI's mock data without changing the UI.

Node 6 uses no model and no GPU.

> **Research pivot (workflow):** multi-granularity bidirectional navigation — read a high-level summary, click a
> point, and see the exact sentence in the document that caused it; or click a place in the document and see its
> changes.

---

# Architecture

```text
                Node 5 results (any order, as they arrive)
                               |
                               v
                     ReportAssembler.add()
                               |
                               v
                           build()
                               |
     +-----------+-------------+-------------+-------------+-----------+
     |           |             |             |             |           |
  overview  stakeholders    sections      changes        index      skipped
  (layer 1)  (layer 2)     (layer 2)     (layer 3)   chunk/page ->   (reason)
                                             |         changes
                                         evidence
                                         (layer 4)
                               |
                               v
               save_report()  (JSON: UI + backup on disk)
               to_markdown()  (readable report)
```

---

## Files

| File | Purpose |
|---|---|
| `assembler.py` | `ReportAssembler`, `Node6Config`: builds the layered report |
| `evidence.py` | changed sentences with offsets for both sides of a change (uses Node 4's sentence splitter) |
| `parties.py` | one clean name per stakeholder |
| `export.py` | `save_report` / `load_report` (JSON), `to_markdown` / `save_markdown` |
| `ui.py` | `to_ui_data`, `save_ui_data`, `save_ui_module`: the report in the ClauseLens UI's data format |

---

## Public Interface

```python
from nodes.node_6 import (
    Node6Config,
    ReportAssembler,
    save_report,
    load_report,
    to_markdown,
    save_markdown,
    to_ui_data,
    save_ui_data,
    save_ui_module,
)
```

---

## Usage

```python
from nodes.node_6 import ReportAssembler, save_report, save_markdown

assembler = ReportAssembler(
    documents={"document_a": "License Agreement (2009)", "document_b": "Amended and Restated (2012)"},
    all_chunks=all_chunks,        # optional: document order, contract parties, chunk counts
    node3_report=report,          # optional: number of unchanged clauses
)
report_json = assembler.add_many(node5_output).build()

save_report(report_json, "/kaggle/working/clauselens_outputs/report.json")     # backup on disk / for the UI
save_markdown(report_json, "/kaggle/working/clauselens_outputs/report.md")     # readable version
print(report_json["overview"]["headline"])
```

### For the ClauseLens UI (React + Vite, Node.js)

The front end is **`clauselens-ui`**, a React + Vite app (run with Node.js; this replaces the Streamlit UI in the
original workflow). Its Compare page reads `documents`, `chunks` and `alignments` from `src/data/mockData.js`.
`to_ui_data()` produces exactly that shape:

```python
from nodes.node_6 import save_ui_data, save_ui_module

save_ui_data(report_json, f"{OUTPUT_DIR}/ui_data.json")       # for a backend endpoint the UI fetches
save_ui_module(report_json, f"{OUTPUT_DIR}/mockData.js")      # drop-in replacement for src/data/mockData.js
```

| UI field | From Node 6 |
|---|---|
| `documents.<doc>.label` | the document names given to `ReportAssembler` |
| `documents.<doc>.chunk_count`, `page_count` | from `all_chunks` |
| `chunks.<doc>[]` | the clauses that have a change: `chunk_id`, `text`, `page_start`, `page_end`, `heading_path` (+ `highlights`) |
| `alignments[].status` | MODIFIED → `diverge`, ADDED → `missing_in_a`, DELETED / UNVERIFIED → `missing_in_b` |
| `alignments[].a`, `.b` | old / new chunk id (`null` when the clause is only in one version) |
| `alignments[].score` | Node 3 similarity (`null` for ADDED / DELETED) |
| `alignments[].note` | the Node 5 summary |

Each alignment also carries `confidence`, `needs_review`, `affected`, `duty_shift`, `numbers`, `title` and
`section`; each chunk carries `highlights` (offsets of the changed sentences). The current UI ignores these, so a
later version can show confidence badges, parties and sentence highlighting without a new backend format.
Unchanged clauses (`match`) are not in the report, so the UI shows only changes.

Checked against the UI: its own `rows.js` logic resolves every alignment to its clauses, and a production
`vite build` with the generated `mockData.js` succeeds; the Compare page renders the real summaries.

### Continuous (as Node 5 finishes)

```python
assembler = ReportAssembler(documents, all_chunks, report)
for result in core.iter_run(node4_output["pairs"], checkpoint_path=NODE5_CHECKPOINT):
    assembler.add(result)
    partial = assembler.build()        # a complete report of everything finished so far
```

Adding results one at a time, in any order, gives the same report as `add_many()`.

---

## Output Format

```python
{
  "report_version": "1",
  "generated_at": "2026-10-27T12:00:00+00:00",
  "documents": {"document_a": {"name": "...", "chunks": 127}, "document_b": {...}},

  "overview": {                                   # layer 1
    "headline": "23 changes: 12 modified, 6 added, 5 deleted; 4 duty shifts; 3 need human review",
    "total_changes": 23,
    "by_type": {"MODIFIED": 12, "ADDED": 6, "DELETED": 5, "UNVERIFIED": 0},
    "by_confidence": {"HIGH": 20, "LOW": 3},
    "duty_shifts": 4,
    "needs_review": ["C007", "C015", "C019"],
    "unchanged_clauses": 65,                      # from Node 3 (EQUIVALENT)
    "skipped": 2,
    "highlights": [{"id": "C003", "title": "2.1 Grant of License", "summary": "...",
                    "why": "duty shift PERMISSION -> PROHIBITION"}],
  },

  "stakeholders": [{"party": "Licensee", "change_count": 18, "changes": ["C001", "C003", ...]}],   # layer 2
  "sections":     [{"section": "2. LICENSE", "change_count": 2, "changes": ["C003", "C004"]}],

  "changes": [{                                   # layer 3, in document order
    "id": "C003",
    "pair_id": "MODIFIED:document_a_0006->document_b_0006",
    "type": "MODIFIED",                           # MODIFIED | ADDED | DELETED | UNVERIFIED
    "section": "2. LICENSE",
    "title": "2.1 Grant of License",
    "summary": "The new version states that the Licensee shall not sublicense ...",
    "affected": ["Licensee", "Licensor"],
    "status": "VERIFIED", "confidence": "HIGH",
    "needs_review": false, "review_reason": "",
    "duty": {"shift": "PERMISSION -> PROHIBITION", "labels_added": ["PROHIBITION"], "labels_removed": []},
    "numbers": {"only_in_source": [], "only_in_target": []},
    "categories": ["License Grant"],
    "importance": 4,
    "evidence": {                                 # layer 4
      "old": {
        "chunk_id": "document_a_0006", "document_id": "document_a", "pages": [1, 1],
        "heading_path": ["2. LICENSE", "2.1 Grant of License"],
        "text": "2.1 Grant of License\nLicensor grants ...",
        "sentences": [{"text": "...", "start": 0, "end": 118, "changed": false}, ...],
        "highlights": [{"start": 181, "end": 251}],          # offsets into `text`
        "changed_sentences": ["Licensee may sublicense the Software to its wholly owned subsidiaries."]
      },
      "new": {...}                                # None for DELETED; "old" is None for ADDED
    }
  }],

  "index": {                                      # document -> report
    "by_chunk": {"document_a_0006": ["C003"], "document_b_0006": ["C003"]},
    "by_page": {"document_a": {"1": ["C001", "C003"]}, "document_b": {...}}
  },

  "skipped": [{"pair_id": "DELETED:document_a_0018->-", "reason": "too little text to analyse"}]
}
```

**Highlights** are the confirmed (VERIFIED) changes ranked by: duty shift (+3), changed numbers such as amounts,
deadlines and notice periods (+3), a removed obligation/prohibition (+2), a duty added (+1), an added/deleted clause
with duties (+1). Unconfirmed changes are never highlighted; they are listed under `needs_review`.

**Stakeholders:** Node 5's names are cleaned ("the Licensees" → Licensee, "Licensee / buyer" → Licensee,
"either party" → Both parties, company-name fragments like "INC" dropped). A "Both parties" change is listed under
each of the contract's parties (found in the documents' opening clauses).

---

## Configuration

`Node6Config` fields:

| Field | Default | Meaning |
|---|---|---|
| `max_highlights` | `5` | How many top changes the overview lists |
| `include_history` | `False` | Keep Node 5's attempt-by-attempt history per change (debugging; larger file) |
| `include_text` | `True` | Keep the clause text in the evidence (needed for highlighting; set False for a small file) |

---

## Known Limitations

- **The report is only as good as Nodes 3–5.** A LOW_CONFIDENCE summary is included but flagged; a wrong pair from
  Node 3 shows up as a wrong change.
- **Sections** come from the first numbered heading in a clause's heading path; clauses without numbered headings
  are grouped under their first heading or "Other".
- **Importance** is a simple rule-based score, not a legal judgement.
- **Sending the data to the laptop UI** is not part of Node 6. The workflow's Streamlit UI was replaced by
  `clauselens-ui` (React + Vite, Node.js). Node 6 produces the JSON / `mockData.js` the UI reads; serving it from
  Kaggle to the laptop (e.g. through the zrok tunnel) is the next integration step.
