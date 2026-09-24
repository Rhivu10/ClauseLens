"""
ClauseLens — Node 4
Gemma deontic check.

Quick model confirmation of a sentence label. Reuses the Gemma model
already loaded for Node 3, so no extra GPU memory is needed.
"""

from __future__ import annotations

import gc
import json
import re

import torch

from .config import (
    Node4Config,
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
    schema = '{"label":"' + "|".join(options) + '"}'
    return f"""Classify the legal duty expressed in this contract sentence.

{lines}

Return ONLY this JSON, no markdown, no extra text:
{schema}

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


# ============================================================
# CHECKER
# ============================================================

class DeonticChecker:
    """Wraps the Gemma model for one-sentence deontic classification."""

    def __init__(self, model, tokenizer, config: Node4Config | None = None):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or Node4Config()
        self._calls = 0
        # The same sentence often appears on both sides of a MODIFIED pair
        self._cache: dict[tuple, tuple[str | None, str]] = {}

    def _encode(self, prompt: str):
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )

    def classify(self, sentence: str, allowed=VALID_MODEL_LABELS) -> tuple[str | None, str]:
        """
        Returns (label, raw). Gemma may only pick one of `allowed` (plus NONE),
        so it can confirm or cancel a regex label but never invent one.
        label is None when the answer is unparseable or not allowed.
        """
        allowed = set(allowed) | {NONE}
        key = (" ".join(sentence.lower().split()), tuple(sorted(allowed)))
        if key in self._cache:
            return self._cache[key]

        inputs = outputs = None
        try:
            inputs = self._encode(build_deontic_prompt(sentence[:self.config.sentence_chars], allowed))
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    do_sample=False,
                )
            generated = outputs[0][inputs["input_ids"].shape[-1]:]
            raw = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
            label = parse_label(raw, allowed)
        except torch.cuda.OutOfMemoryError:
            gc.collect()
            torch.cuda.empty_cache()
            label, raw = None, "CUDA OOM"
        finally:
            del inputs, outputs
            self._calls += 1
            if self._calls % self.config.cleanup_every == 0:
                gc.collect()
                torch.cuda.empty_cache()

        self._cache[key] = (label, raw)
        return label, raw
