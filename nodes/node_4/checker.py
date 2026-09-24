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

from .config import Node4Config, VALID_MODEL_LABELS


# ============================================================
# PROMPT
# ============================================================

def build_deontic_prompt(sentence: str) -> str:
    return f"""Classify the legal duty expressed in this contract sentence.

OBLIGATION  - a party must do something
PERMISSION  - a party may do something, or is excused from doing it
PROHIBITION - a party must not do something
NONE        - no duty (definition, fact, or description)

Return ONLY this JSON, no markdown, no extra text:
{{"label":"OBLIGATION|PERMISSION|PROHIBITION|NONE"}}

SENTENCE:
{sentence}"""


def parse_label(raw: str) -> str | None:
    """Tolerant parser. Returns a valid label or None."""
    m = re.search(r"\{.*?\}", raw, re.S)
    if m:
        try:
            label = str(json.loads(m.group(0)).get("label", "")).strip().upper()
            if label in VALID_MODEL_LABELS:
                return label
        except json.JSONDecodeError:
            pass
    return None


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
        self._cache: dict[str, str | None] = {}

    def _encode(self, prompt: str):
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )

    def classify(self, sentence: str) -> str | None:
        """Returns OBLIGATION | PERMISSION | PROHIBITION | NONE, or None if unparseable."""
        key = " ".join(sentence.lower().split())
        if key in self._cache:
            return self._cache[key]

        inputs = outputs = None
        try:
            inputs = self._encode(build_deontic_prompt(sentence[:self.config.sentence_chars]))
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    do_sample=False,
                )
            generated = outputs[0][inputs["input_ids"].shape[-1]:]
            label = parse_label(self.tokenizer.decode(generated, skip_special_tokens=True))
        except torch.cuda.OutOfMemoryError:
            gc.collect()
            torch.cuda.empty_cache()
            label = None
        finally:
            del inputs, outputs
            self._calls += 1
            if self._calls % self.config.cleanup_every == 0:
                gc.collect()
                torch.cuda.empty_cache()

        self._cache[key] = label
        return label
