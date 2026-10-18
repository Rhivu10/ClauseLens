# Node 5 Documentation

## Overview

**Node 5** is the Parallel Multi-Agent Core of ClauseLens ("the brain").

For every changed clause from Node 4, it writes a short, plain-English **impact summary** (what changed, who is
affected, how), checks that summary against the contract text, and only then sends it to Node 6.

Node 5 currently:

- Runs two agents: the **Impact Agent** (GPU 0) writes the summary; the **Critic Agent** (GPU 1, a separate Gemma)
  checks it.
- Runs them **at the same time**: while the Critic checks pair 1 on GPU 1, the Impact Agent writes pair 2 on GPU 0.
- Runs **rule-based grounding checks** before the Critic: every number in the summary must be in the text, every
  affected party must be named in the text (or be one of the contract's parties), and a summary with 6+ content words
  missing from the text is rejected (3–5 are sent to the Critic to verify). These catch invented facts without relying
  on a model.
- Shows both agents **exactly which sentences changed** (OLD → NEW), so they describe the change, not the old clause.
- Routes each result like the workflow describes:
  - **VERIFIED** → done, HIGH confidence
  - **HALLUCINATION** → the Impact Agent retries with the Critic's warning
  - **MISSING_CONTEXT** → neighbouring clauses are added, then the Impact Agent retries
  - **max 2 attempts** → sent on as **LOW_CONFIDENCE** so a human checks it
- Gives the Impact Agent **CUAD category** and **stakeholder** hints from a draft ontology (`cuad_ontology.json`).
- Skips heading-only chunks (`SKIPPED`), so no model time is wasted on them.
- Saves every result immediately (JSONL). A crashed session resumes where it stopped.

> **Research pivot:** two separate model instances instead of one model switching prompts, so the Critic judges the
> Impact Agent's work without sharing its fine-tuning or its context.

---

# Architecture

```text
                       NODE 4 enriched pairs
                                |
                                v
        +--------------------------+        +--------------------------+
        | GPU 0  ImpactAgent       |        | GPU 1  CriticAgent       |
        | Gemma + CUAD adapter     | -----> | separate Gemma           |
        | (same model as Node 3)   |        | VERIFIED | HALLUCINATION |
        +--------------------------+        | MISSING_CONTEXT          |
                ^         ^                 +------------+-------------+
                |         |   grounding.py               |
                |         +-- numbers / parties fail ----+  (HALLUCINATION, Critic not called)
                |                                        |
                |        ROUTING (core.py)               v
                +----- HALLUCINATION: retry with warning
                +----- MISSING_CONTEXT: add neighbouring chunks, retry
                       VERIFIED --------------------> HIGH confidence
                       2 attempts used -------------> LOW_CONFIDENCE
                                                         |
                                                         v
                                     JSONL checkpoint ---> NODE 6
```

---

## Files

| File | Purpose |
|---|---|
| `config.py` | `Node5Config`, verdicts, final statuses, `NODE5_VERSION` |
| `cuad_ontology.json` | **Draft** 41 CUAD categories → keywords + stakeholders (for the team to review) |
| `ontology.py` | Detects CUAD categories and the parties named in a clause |
| `prompts.py` | Impact and Critic prompts and their parsers (plain text, no JSON) |
| `grounding.py` | Rule checks: numbers and parties must be in the text |
| `agents.py` | `ImpactAgent`, `CriticAgent`, `load_critic_model` |
| `core.py` | `Node5ImpactCore`: parallel loop, routing, context expansion, checkpoint |

---

## Public Interface

```python
from nodes.node_5 import (
    Node5Config,
    load_critic_model,
    ImpactAgent,
    CriticAgent,
    Node5ImpactCore,
    read_node5_output,
)
```

---

## Usage

Node 5 runs straight after Node 4 in the same session. It needs **two GPUs** (Kaggle: *GPU T4 x2*) for parallel
mode; with one GPU it still works, sequentially.

```python
import unsloth  # before transformers / peft
from nodes.node_5 import Node5Config, load_critic_model, ImpactAgent, CriticAgent, Node5ImpactCore

# model, tokenizer: loaded for Node 3 with load_gemma_with_adapter (GPU 0)
node5_config = Node5Config()
critic_model, critic_tokenizer = load_critic_model(device_index=1, hf_token=HF_TOKEN)   # Gemma E4B on GPU 1

impact = ImpactAgent(model, tokenizer, node5_config)
critic = CriticAgent(critic_model, critic_tokenizer or tokenizer, node5_config)
core = Node5ImpactCore(all_chunks, impact, critic, node5_config)

node5_output = core.run(
    node4_output["pairs"],
    checkpoint_path="/kaggle/working/clauselens_outputs/node5_results.jsonl",
)
```

### One GPU / shared model

```python
critic = CriticAgent(model, tokenizer, node5_config)   # same model as the Impact Agent
core = Node5ImpactCore(all_chunks, impact, critic)     # parallel is switched off automatically
```

### Continuous (streaming) use

```python
for result in core.iter_run(node4_output["pairs"], checkpoint_path=NODE5_CHECKPOINT):
    node6_add(result)          # results arrive as soon as each pair is final
```

---

## The two agents

### Impact Agent (GPU 0)

Two options, set with `IMPACT_BASE_MODEL` in the notebook:

| Option | Model | Notes |
|---|---|---|
| `"unsloth/gemma-4-E4B-it"` (default) | plain Gemma **E4B**, loaded with `load_agent_model(device_index=0)` | better at writing summaries; needs ~10 GB, so the notebook **frees the Node 3 model first** (Nodes 3 and 4 are finished by then) |
| `None` | the Gemma E2B + CUAD adapter loaded for Node 3 | no extra memory; the adapter was trained for clause extraction, not writing, and in three Kaggle runs it confused old and new text, ignored "give old and new values", and invented details |

**GPU memory on a T4.** Gemma 4 runs in **float32** on a T4 (no bfloat16, and Unsloth refuses float16 for gemma4),
so models are about twice the expected size. Measured on Kaggle: Node 3 model (E2B + adapter) 7.8 GB, Gemma E4B 10.2 GB,
on 14.6 GB GPUs. E4B therefore does not fit next to the Node 3 model: the first try ran out of memory. The notebook
frees the Node 3 model before loading the E4B Impact Agent, giving one E4B per GPU. To re-run Node 3 or 4 afterwards,
reload the Node 3 model. If E4B still can't be loaded, the Impact Agent shares the Critic's model (sequential).

Its prompt contains:

- the change type (MODIFIED / ADDED / DELETED / UNCLEAR), the section heading
- **WHAT CHANGED**: only the sentences that differ, old then new (on Kaggle the agent once described the old 2.1
  clause as if it were current; this keeps it on the change)
- Node 4's duty change (e.g. `PERMISSION -> PROHIBITION`), duties added or removed
- Node 3's changed numbers (e.g. `$50,000` → `$65,000`)
- the parties named in the text and the stakeholder hints from the ontology
- the old and new clause text (and neighbouring text after MISSING_CONTEXT)
- the Critic's warning, on a retry

It answers with 2–3 sentences about what is true in the NEW version compared to the OLD one (with the old and new
values when numbers changed), then one line `AFFECTED: party; party`.

### Critic Agent (GPU 1)

A **separate** Gemma: by default the larger **Gemma E4B**, plain instruction-tuned, without the CUAD adapter
(`load_critic_model(checkpoint_path=None)`), so it shares no fine-tuning with the Impact Agent. The notebooks try
`CRITIC_BASE_MODELS` in order (E4B, then E2B) and fall back to the shared model. It uses its own tokenizer.

E4B is the default because, in the first Kaggle run, an **E2B** Critic approved a summary with an invented detail
("*exception if the sharing includes personally identifiable information*") and rejected a correct one.
The Critic is also told the change type as a known fact (e.g. "DELETED"), because the E2B Critic rejected a correct
"this clause was removed" summary when it only saw the old text. It sees the
clause text and the summary and answers one word on the first line, then a short reason:

| Verdict | Meaning |
|---|---|
| `VERIFIED` | every claim is supported by the text |
| `HALLUCINATION` | the summary says something the text does not |
| `MISSING_CONTEXT` | the clause depends on other sections, so it can't be checked |

An unreadable answer, or an answer that just repeats the option list, is retried once with a one-word prompt. If
it is still unreadable, the pair goes to Node 6 as LOW_CONFIDENCE (no Impact retry is wasted on it).

### Grounding checks (rules, before the Critic)

| Check | Example caught |
|---|---|
| every number in the summary is in the text | summary says "$70,000", text says "$65,000" |
| every AFFECTED party is named in the text, is a contract party, or is a stakeholder hint | summary says "Employees", the clause only mentions the Licensee |
| 6+ content words not in the text (`max_unsupported_words`) → rejected; 3–5 (`review_unsupported_words`) → the Critic is told to verify those words | "personally identifiable information"; "pay a **fee** each **year**" goes to the Critic |
| the summary is not empty | the model returned nothing |

**Contract parties** are the parties named in each document's opening chunk (title + preamble), e.g. Licensor and
Licensee. Plural and "all"/"the" forms count too ("all licensees" = Licensee). They are always accepted: on Kaggle, a mostly correct 4.2 summary was rejected for naming the Licensee,
who receives the updates but isn't named in that clause.

**The word check** compares word families ("paying" = "pay", "weekly" = "week") and ignores words that describe a
change ("removed", "prohibiting", "increased"...). It was tuned on the first Kaggle run's real summaries:

| Summary | Words not in the text | Result |
|---|---|---|
| 2.1, 4.1, 6.1, 7.1 Employees (good) | 0 | pass |
| 3.1, 8.2 (good) | 1–2 | pass |
| 5.1 run 2 (good paraphrase: "recipients, companies, subject") | 3 | Critic verifies |
| 4.2 ("pay a fee … each year"), title ("all users of the product") | 3 | Critic verifies |
| 7.1 Model Training (invented passages, runs 1 and 2) | 6–10 | rejected |

The first version rejected at 3 words; run 2 showed that also rejects good paraphrases (5.1), hence the two levels.
Run 5 (E4B Impact Agent, which paraphrases more freely) then had correct summaries rejected at 5–6 words, including
"old" and "wording", and the retry warning made the agent drop specific facts ("from 60 to 90 days" became "a certain
number of days"). So: words like old/wording/duty and number words are always allowed, rejection starts at 6, and the
warning now says to keep numbers, deadlines and who must do what. Across all five runs every good summary had 0–4
such words and the two invented passages had 6 and 10.

A failed check counts as HALLUCINATION, with the problem as the warning for the retry. The Critic is not called.

---

## Configuration

`Node5Config` fields:

| Field | Default | Meaning |
|---|---|---|
| `max_attempts` | `2` | Impact attempts per pair (workflow: max 2 tries) |
| `text_chars` | `900` | Clause characters per side in a prompt (shrunk to fit the token budget) |
| `max_seq_length` | `2048` | Prompt + answer token budget |
| `impact_max_new_tokens` | `200` | Impact answer length |
| `critic_max_new_tokens` | `60` | Critic answer length |
| `repetition_penalty` | `1.15` | Discourages the looping seen in Node 3 |
| `max_unsupported_words` | `5` | More content words missing from the text than this → HALLUCINATION, Critic not asked |
| `review_unsupported_words` | `2` | More than this (up to the max) → the Critic is asked to verify those words |
| `max_changed_sentences` | `3` | Changed sentences per side shown to both agents |
| `context_neighbors` | `1` | Chunks added on each side per MISSING_CONTEXT level |
| `max_context_level` | `1` | How many times context can be widened |
| `min_chars` | `40` | Below this (without the heading line) a pair is SKIPPED |
| `parallel` | `True` | Agents on separate threads/GPUs; off automatically if they share a model |
| `cleanup_every` | `10` | Empty the CUDA cache every N generations per agent |

---

## Output Format

### One result (sent to Node 6)

Everything from Node 4's pair, plus `impact`:

```python
{
    "pair_id": "MODIFIED:document_a_0006->document_b_0006",
    "change_status": "MODIFIED",
    "source": {...}, "target": {...}, "alignment": {...}, "deontic": {...},   # from Node 4
    "impact": {
        "status": "VERIFIED",              # VERIFIED | LOW_CONFIDENCE | SKIPPED
        "confidence": "HIGH",              # HIGH | LOW | None
        "summary": "The Licensee can no longer sublicense the Software, not even to its subsidiaries.",
        "affected": ["Licensee", "subsidiaries"],
        "final_verdict": "VERIFIED",
        "reason": "Matches the new clause.",
        "attempts": 1,
        "context_level": 0,
        "categories": ["Affiliate License-Licensee", "License Grant"],
        "stakeholder_hints": ["Licensee / buyer", "Affiliates / subsidiaries"],
        "parties_in_text": ["Licensee", "Subsidiary"],
        "history": [                       # every attempt, for debugging and for Node 6
            {"attempt": 1, "context_level": 0, "summary": "...", "affected": [...],
             "issues": [], "verdict": "VERIFIED", "reason": "...", "decided_by": "critic"},
        ],
    },
}
```

`decided_by` is `"critic"` or `"checks"` (a grounding rule failed).

### `run()` summary

```python
{
    "results": [...],                      # one per input pair, in input order
    "counts": {"VERIFIED": 9, "LOW_CONFIDENCE": 1, "SKIPPED": 3},
    "low_confidence": ["MODIFIED:..."],    # pairs a human should double-check
    "model_calls": {"impact": 12, "critic": 11},
    "parallel": True,
    "seconds": 95.3,
}
```

---

## The CUAD ontology (`cuad_ontology.json`)

The workflow says the team maps CUAD's 41 clause categories to the stakeholders they affect. This file is a
**first draft** of that mapping:

```json
"Affiliate License-Licensee": {
    "keywords": ["sublicense", "subsidiaries", "licensee's affiliates", "its affiliates"],
    "stakeholders": ["Licensee / buyer", "Affiliates / subsidiaries"]
}
```

- Keywords are matched in the clause text; the top 3 categories become hints.
- Their stakeholders are listed in the Impact Agent's prompt and are accepted by the party check.
- Categories with no keywords (e.g. "Document Name") are never detected.

**Please review it.** Edit keywords and stakeholders freely; no code change is needed.

---

## Checkpoint and Resume

Same as Node 4: the first line of the JSONL file is a fingerprint of the input (pair ids, texts, `NODE5_VERSION`,
`max_attempts`, `max_context_level`); every finished result is appended and flushed immediately. Re-running the
cell after a crash reads finished pairs from disk. A checkpoint from other input or an older Node 5 is replaced.

**If you change Node 5's prompts or logic, bump `NODE5_VERSION` in `config.py`.**

---

## Tests (no GPU, scripted agents)

| Case | Result |
|---|---|
| Summary verified on the first try | VERIFIED, 1 attempt |
| Summary invents "$70,000" | caught by the number check, Critic not called, retry with the warning, VERIFIED |
| Summary names "Employees" not in the clause | caught by the party check, retry, VERIFIED |
| Critic says MISSING_CONTEXT | neighbouring chunks added (incl. the referenced section), retry, VERIFIED |
| Critic rejects twice | LOW_CONFIDENCE after 2 attempts |
| Critic copies the option list, then answers garbage | LOW_CONFIDENCE, no Impact retry wasted |
| Heading-only chunk | SKIPPED, no model call |
| Parallel vs sequential | same results; parallel ~40% faster |
| 4 threads using one tokenizer | 0 overlapping calls (lock works) |
| Kaggle summaries: "Affected:" written inline and twice; "no party concerned" | parsed once / dropped |
| Kaggle summaries: invented "personally identifiable information", "pay a fee each year" | flagged by the word check |
| Kaggle summary: good 2.1 paraphrase | passes the word check |
| Licensee affected but not named in the clause | accepted as a contract party; "Employees" still rejected |
| 2.1 changed sentences | only "may sublicense" → "shall not sublicense", not the unchanged "may install" |
| Good 5.1 paraphrase (run 2) | not rejected; sent to the Critic with the 3 words to verify |
| Invented 7.1 passage (run 2) | still rejected outright |
| "all licensees" as AFFECTED (run 3) | accepted against the contract party "Licensee"; "users" still rejected |
| Crash after 2 results + half-written line | resume finishes all pairs, no duplicates; re-run makes 0 model calls |

---

## Kaggle results so far (test contract, 11 analysed pairs)

| Run | Change | Correct VERIFIED | Correctly flagged LOW | Wrong |
|---|---|---|---|---|
| 1 | first version | 7 | 0 | 4 (1 invented detail VERIFIED, 2 correct summaries rejected, 1 vague) |
| 2 | E4B Critic, word check, contract parties | 8 | 1 | 2 (good 5.1 rejected by word check, 2.1 described the old clause) |
| 3 | changed sentences, two-level word check | 8 | 2 (Impact errors caught by the Critic) | 1 (3.2 rejected: plural "licensees" bug, fixed after run 3) |
| 4 | plural fix (Impact Agent still E2B: E4B ran out of GPU memory) | 9 | 2 | 0 |
| 5 | **Impact Agent on E4B** (Node 3 model freed first) | 9, much clearer writing | 0 | 2 correct summaries rejected by the word check; 3 retries became vague (fixed after run 5) |

Since run 2, **no wrong summary has been marked VERIFIED**. The remaining weakness is the Impact Agent's writing
quality, hence the E4B option above.

## Real contract (SEC EDGAR pair)

First run on a real before/after pair (Echo Therapeutics – Ferndale License Agreement 2009 vs its 2012 Amended and
Restated version). The important legal changes were found and explained correctly (Territory expanded to Europe,
Australia, South America; right of first refusal → right of first negotiation; new Minimum Annual Royalty; patent
abandonment now needs consultation instead of consent; new insurance-certificate duty). Fixes made after that run:

| Seen on the real pair | Fix |
|---|---|
| Correct summaries rejected for describing words ("said", "sentence", "additions", "language") | those words are always allowed in the word check |
| Critic rejected correct summaries for wording ("'updated' is interpretive") | the Critic is told to judge facts, not wording, including for the double-check words |
| `[Other Party]` placeholder as an affected party | placeholders in brackets are dropped |
| "Here is the revised explanation based strictly on your constraints: …" | prompt echoes are stripped from the start of a summary |
| "Licensor/provider" rejected (hint is "Licensor / provider") | slash with or without spaces |

Also related: Node 3's formatting-only check (see its docs) and the clean-up step in the real-contract test
notebook (between Node 1 and Node 2; Node 1 itself is unchanged). Together they remove most of the noise pairs
(spacing, ™/®, table headings, sections 11–27 lumped together) before Node 5.

Known PDF limitation: the 2009 PDF's text layer turns "1½%" into "11/2%", so a summary claimed the late fee
"increased from 11/2% to 1½%". Neither the rules nor the Critic can know the two are the same.

## Known Limitations

- **Tested on one fictional contract.** Three Kaggle runs on the same test contract; prompts and the word check were
  tuned on its output, so they may fit it too closely. Test on a real CUAD pair and a second contract next.
- **The Impact Agent on E4B** has not been run on Kaggle yet.
- **Loading the Critic on GPU 1** uses Unsloth first and plain transformers as a fallback. Gemma **E4B** on GPU 1 has
  not been tested on Kaggle yet; the notebooks fall back to E2B, then to the shared model.
- **The word check is a heuristic.** A correct summary that paraphrases heavily can be flagged (it is then retried);
  a small invented detail of 1–2 words can slip through. Tune `max_unsupported_words` on more real output.
- **Grounding checks are strict.** A party that is clearly affected but not named in the clause (and not a stakeholder
  hint) is flagged. This favours precision; widen `cuad_ontology.json` stakeholders if it rejects too much.
- **Number check works on digits only.** "four hours" vs "4 hours" are not compared.
- **The ontology is a draft** and keyword matching is simple; categories are hints, not classifications.
- **A 2B model as Critic** can miss subtle errors. LOW_CONFIDENCE is the safety net for a human reviewer, not proof
  that a VERIFIED summary is perfect.

---

## Requirements

- Two GPUs for parallel mode (Kaggle: GPU T4 x2); one GPU works sequentially
- `unsloth`, `peft`, `transformers==5.5.0`, `bitsandbytes`, `torch`
- Node 4 output (`node4_output["pairs"]`) and Node 2 canonical chunks
- The Gemma model and tokenizer loaded for Node 3
