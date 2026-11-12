"""
ClauseLens — Node 4 (performance-tuned)
Gemma deontic check.

Quick model confirmation of a sentence label. Reuses the Gemma model
already loaded for Node 3, so no extra GPU memory is needed.

Two execution modes with identical prompts and greedy decoding, so labels
are unchanged:

  Batched (new, used by Node4Extractor.iter_run prefetching):
      classify_many() — one model.generate call for up to
      config.batch_size sentences, left-padded.

  Single sentence (original API, kept for profile()/enrich_pair() and
      notebooks): classify() — thin wrapper over the batched path with a
      batch of one.

Speed measures (same recipe as the Node 3 optimization):
  * Batching: batch_size sentences per generate call (near-linear
    throughput gain on GPU; deontic answers are tiny, so 16 fits easily).
  * stop_at_json: a StoppingCriteria that ends generation once every row
    has produced the closing '}' of its answer JSON. The tolerant parser
    only reads the first {...} block, so the parsed label is identical
    while generated tokens drop from up to 40 to ~10-15 per answer.
  * OOM resilience: a batch that triggers CUDA out-of-memory is split in
    half and retried, down to single sentences; a sentence that fails
    alone returns (None, "CUDA OOM") exactly like the original code.
  * Persistent verdict cache (optional, config.cache_path): a JSONL file
    of sentence verdicts, so a crashed or repeated run resumes instead of
    re-verifying every sentence.
  * Cleanup every cleanup_every BATCHES instead of every N calls
    (gc.collect() over a heap holding the Node 2 graph is not free).
"""

from __future__ import annotations

import gc
import json
import os
import re
from typing import Any

import torch

from .config import (
    Node4Config,
    PROMPT_VERSION,
    VALID_MODEL_LABELS,
    LABEL_PRIORITY,
    OBLIGATION,
    PERMISSION,
    PROHIBITION,
    NONE,
)


# ============================================================
# PROMPT
# ============================================================

_LABEL_DEFINITIONS = {
    OBLIGATION:  "a party must do something",
    PERMISSION:  "a party may do something, or is excused from doing it",
    PROHIBITION: "a party must not do something",
    NONE:        "no duty (definition, fact, or description)",
}


def build_deontic_prompt(sentence: str, allowed=VALID_MODEL_LABELS) -> str:
    """Gemma only chooses among the labels the trigger words found, plus NONE."""
    options = [l for l in LABEL_PRIORITY if l in allowed]
    lines = "\n".join(f"{l:11} - {_LABEL_DEFINITIONS[l]}" for l in options)
    # Show a placeholder, not "A|B": Gemma copied "OBLIGATION|NONE" verbatim on Kaggle
    return f"""Classify the legal duty expressed in this contract sentence.

Choose exactly ONE label from this list:
{lines}

Return ONLY this JSON, with <LABEL> replaced by your one chosen label.
No markdown, no extra text:
{{"label": "<LABEL>"}}

SENTENCE:
{sentence}"""


def parse_label(raw: str, allowed=VALID_MODEL_LABELS) -> str | None:
    """Tolerant parser: JSON first, then a bare label word. Returns an allowed label or None."""
    m = re.search(r"\{.*?\}", raw, re.S)
    if m:
        try:
            label = str(json.loads(m.group(0)).get("label", "")).strip().upper()
            return label if label in allowed else None
        except json.JSONDecodeError:
            pass
    words = [w for w in re.findall(r"[A-Z]+", raw.upper()) if w in allowed]
    return words[0] if len(set(words)) == 1 else None


# Raw response returned for a sentence that failed even as a singleton batch.
# The regex label is kept in that case, matching the old CUDA-OOM handling.
OOM_RAW = "CUDA OOM"


# ============================================================
# CACHE KEY
# ============================================================

def deontic_cache_key(sentence: str, allowed) -> tuple[str, tuple[str, ...]]:
    """
    The exact key form the original classify() used: normalized sentence plus
    the allowed set with NONE added, sorted. Shared by classify()/classify_many()
    so prefetch warming and per-pair lookups hit the same entries.
    """
    return (
        " ".join(sentence.lower().split()),
        tuple(sorted(set(allowed) | {NONE})),
    )


# ============================================================
# JSON STOP CRITERIA
# ============================================================

def _text_tokenizer(tokenizer):
    """
    The plain text tokenizer. For Gemma 4, Unsloth returns a Gemma4Processor,
    which wraps the tokenizer and has no pad(), get_vocab() or eos_token_id.
    """
    return getattr(tokenizer, "tokenizer", None) or tokenizer


def _brace_token_ids(tokenizer) -> set[int]:
    """Token ids whose surface form contains '}' (covers '}', '{}', '}}', ...)."""
    try:
        vocab = _text_tokenizer(tokenizer).get_vocab()
    except Exception:
        return set()
    return {i for tok, i in vocab.items() if "}" in tok}


def _make_json_stop_criteria(tokenizer, prompt_len: int):
    """
    Build a transformers StoppingCriteria that stops the whole batch once
    every row has either closed its answer JSON ('}') or finished (EOS).
    Imported lazily so this module stays importable without transformers.
    """
    from transformers import StoppingCriteria

    brace_ids = _brace_token_ids(tokenizer)
    eos = getattr(_text_tokenizer(tokenizer), "eos_token_id", None)
    eos_ids = {eos} if isinstance(eos, int) else set(eos or [])

    class StopAtFirstBrace(StoppingCriteria):
        def __init__(self):
            super().__init__()
            self.prompt_len = prompt_len
            self.done: list[bool] | None = None

        def __call__(self, input_ids, scores, **kwargs) -> bool:
            rows = input_ids.tolist() if hasattr(input_ids, "tolist") else input_ids
            if self.done is None:
                self.done = [False] * len(rows)
            for i, row in enumerate(rows):
                if self.done[i]:
                    continue
                for tok in row[self.prompt_len:]:
                    if tok in brace_ids or tok in eos_ids:
                        self.done[i] = True
                        break
            return all(self.done)

    return StopAtFirstBrace()


# ============================================================
# CHECKER
# ============================================================

class DeonticChecker:
    """Wraps the Gemma model for one-sentence deontic classification."""

    def __init__(self, model, tokenizer, config: Node4Config | None = None):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or Node4Config()
        # Model-verified sentences (cache misses) — same counting spirit as before
        self._calls = 0
        self._batches = 0          # generate calls issued (batched or not)
        # The same sentence often appears on both sides of a MODIFIED pair
        self._cache: dict[tuple, tuple[str | None, str]] = {}
        # Persistent verdict cache (optional)
        self._disk_cache: dict[tuple, tuple[str | None, str]] = {}
        self._persisted: set[tuple] = set()
        if self.config.cache_path:
            self._load_cache(self.config.cache_path)

    # ------------------------------------------------------------
    # PERSISTENT CACHE (JSONL, crash-safe)
    # ------------------------------------------------------------

    def _load_cache(self, path: str) -> None:
        if not os.path.exists(path):
            return
        try:
            with open(path, encoding="utf-8") as fh:
                header = fh.readline()
                meta = json.loads(header).get("_meta", {}) if header.strip() else {}
                if meta.get("prompt_version") != PROMPT_VERSION:
                    print(f"Verdict cache {path} is from another prompt version — starting fresh.")
                    return
                for line in fh:
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue        # partial last line from a crash
                    # JSON turns the (sentence, allowed-tuple) key into lists;
                    # rebuild the exact tuple form deontic_cache_key() produces
                    key = (row["k"][0], tuple(row["k"][1]))
                    self._disk_cache[key] = (row["label"], row["raw"])
        except (OSError, ValueError, KeyError, IndexError, TypeError):
            return

    def _append_cache_rows(self, rows: list[tuple]) -> None:
        """Append freshly computed verdicts to the JSONL cache file."""
        path = self.config.cache_path
        new = [(k, v) for k, v in rows if k not in self._persisted]
        if not new:
            return
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fresh_file = not os.path.exists(path)
        with open(path, "a", encoding="utf-8") as fh:
            if fresh_file:
                fh.write(json.dumps({"_meta": {"node": 4, "prompt_version": PROMPT_VERSION}}) + "\n")
            for key, value in new:
                label, raw = value
                fh.write(json.dumps({"k": list(key), "label": label, "raw": raw},
                                    ensure_ascii=False) + "\n")
                self._persisted.add(key)
            fh.flush()

    # ------------------------------------------------------------
    # ENCODING
    # ------------------------------------------------------------

    def _encode_ids(self, prompt: str) -> dict[str, list]:
        """Chat-template encoding as plain lists (for padding into a batch)."""
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        enc = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
        )
        # Normalize {"input_ids": [[...]], ...} / tensor forms to flat lists.
        out = {}
        for key, value in enc.items():
            if hasattr(value, "tolist"):
                value = value.tolist()
            if value and isinstance(value[0], list):
                value = value[0]
            out[key] = value
        return out

    # ------------------------------------------------------------
    # BATCHED GENERATION
    # ------------------------------------------------------------

    def _cleanup(self):
        if self._batches and self._batches % self.config.cleanup_every == 0:
            gc.collect()
            torch.cuda.empty_cache()

    def _generate_batch(self, prompts: list[str]) -> list[str]:
        """One model.generate call for all prompts. Returns raw strings, input order."""
        encoded = [self._encode_ids(p) for p in prompts]

        # Left padding: all rows share one prompt length, so slicing the
        # generated part per row is a simple [prompt_len:].
        tok = _text_tokenizer(self.tokenizer)
        old_side = getattr(tok, "padding_side", "right")
        old_pad = getattr(tok, "pad_token", None)
        try:
            tok.padding_side = "left"
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            batch = tok.pad(
                {"input_ids": [e["input_ids"] for e in encoded],
                 "attention_mask": [e.get("attention_mask", [1] * len(e["input_ids"])) for e in encoded]},
                padding=True,
                return_tensors="pt",
            )
            prompt_len = int(batch["input_ids"].shape[-1])

            stop = None
            if self.config.stop_at_json:
                stop = _make_json_stop_criteria(self.tokenizer, prompt_len)

            inputs = {k: v.to(self.model.device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    do_sample=False,
                    **({"stopping_criteria": [stop]} if stop is not None else {}),
                )
        finally:
            tok.padding_side = old_side
            if old_pad is None and tok.pad_token is not None:
                tok.pad_token = None
            self._batches += 1
            self._cleanup()

        rows = []
        for i in range(len(prompts)):
            generated = outputs[i][prompt_len:]
            rows.append(self.tokenizer.decode(generated, skip_special_tokens=True).strip())
        return rows

    def _classify_chunk(self, sentences: list[str], allowed_sets: list[set]) -> list[tuple[str | None, str]]:
        """
        Verify one batch-sized chunk. CUDA OOM splits it in half; a sentence
        that fails alone returns (None, OOM_RAW) — the regex label is kept,
        exactly like the original code.
        """
        prompts = [
            build_deontic_prompt(s[:self.config.sentence_chars], a)
            for s, a in zip(sentences, allowed_sets)
        ]
        try:
            raws = self._generate_batch(prompts)
        except torch.cuda.OutOfMemoryError:
            gc.collect()
            torch.cuda.empty_cache()
            if len(prompts) > 1 and self.config.oom_split:
                mid = len(prompts) // 2
                return (self._classify_chunk(sentences[:mid], allowed_sets[:mid])
                        + self._classify_chunk(sentences[mid:], allowed_sets[mid:]))
            return [(None, OOM_RAW)] * len(prompts)
        return [(_parse_or_none(raw, a), raw) for raw, a in zip(raws, allowed_sets)]

    def classify_many(self, items: list[tuple[str, Any]]) -> list[tuple[str | None, str]]:
        """
        Verify a list of (sentence, allowed) items in batches of
        config.batch_size. Returns (label, raw) per item, in the same order.

        Cache hits (this run or the persistent JSONL cache) never touch the
        model; duplicate sentences within one call are verified once.
        """
        keys = [deontic_cache_key(s, a) for s, a in items]
        results: list[tuple[str | None, str] | None] = [None] * len(items)

        # Resolve cache hits and collect the misses (first occurrence only)
        misses: dict[tuple, int] = {}
        for i, key in enumerate(keys):
            if key in self._cache:
                results[i] = self._cache[key]
            elif key in self._disk_cache:
                results[i] = self._disk_cache[key]
                self._cache[key] = self._disk_cache[key]      # promote for next time
            elif key not in misses:
                misses[key] = i

        if misses:
            pending_keys = list(misses)
            sentences = [items[i][0] for i in misses.values()]
            allowed_sets = [set(items[i][1]) | {NONE} for i in misses.values()]

            verdicts: dict[tuple, tuple[str | None, str]] = {}
            size = max(1, self.config.batch_size)
            for start in range(0, len(pending_keys), size):
                chunk_keys = pending_keys[start:start + size]
                chunk_s = sentences[start:start + size]
                chunk_a = allowed_sets[start:start + size]
                chunk_v = self._classify_chunk(chunk_s, chunk_a)
                for key, value in zip(chunk_keys, chunk_v):
                    verdicts[key] = value

            # Store (per-run cache, like the original — including OOM rows)
            fresh_rows = []
            for key, value in verdicts.items():
                self._cache[key] = value
                self._calls += 1
                if value[1] != OOM_RAW:                      # never persist OOM
                    fresh_rows.append((key, value))
            if self.config.cache_path:
                self._append_cache_rows(fresh_rows)

            for i, key in enumerate(keys):
                if results[i] is None and key in verdicts:
                    results[i] = verdicts[key]

        return [r if r is not None else (None, OOM_RAW) for r in results]

    # ------------------------------------------------------------
    # SINGLE-SENTENCE API (unchanged behaviour)
    # ------------------------------------------------------------

    def classify(self, sentence: str, allowed=VALID_MODEL_LABELS) -> tuple[str | None, str]:
        """
        Returns (label, raw). Gemma may only pick one of `allowed` (plus NONE),
        so it can confirm or cancel a regex label but never invent one.
        label is None when the answer is unparseable or not allowed.
        """
        return self.classify_many([(sentence, allowed)])[0]


# ============================================================
# HELPERS
# ============================================================

def _parse_or_none(raw: str, allowed: set) -> str | None:
    """parse_label against the sentence's own allowed set (with NONE)."""
    return parse_label(raw, allowed | {NONE})
