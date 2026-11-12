"""
ClauseLens — Node 3 (performance-tuned)
Gemma verifier.

Loads the base Gemma model with the CUAD-trained LoRA adapter and
compares clauses. Two execution modes with identical prompts and greedy
decoding, so verdicts are unchanged:

  Batched (new, used by align_documents):
      compare_pairs()  / retry_pairs()   — one model.generate call for up to
      config.batch_size pairs, left-padded.

  Single pair (original API, kept for align_source_clause and notebooks):
      compare_clauses() / retry_clauses() — thin wrappers over the batched
      path with a batch of one.

Speed measures:
  * Batching: batch_size pairs per generate call (near-linear throughput
    gain on GPU, ~5-8x at batch 8 on a T4-class card).
  * stop_at_json: a StoppingCriteria that ends generation once every row
    has produced the closing '}' of its answer JSON. The tolerant parser
    only reads the first {...} block, so the parsed verdict is identical
    while generated tokens drop from up to 80 to ~20-25 per answer.
  * OOM resilience: a batch that triggers CUDA out-of-memory is split in
    half and retried, down to single pairs, instead of losing the verdict.
  * Cleanup every cleanup_every BATCHES instead of every N calls
    (gc.collect() over a heap holding the Node 2 graph is not free).
"""

from __future__ import annotations

import gc

import torch

from .config import Node3Config

# Raw response returned for a pair that failed even as a singleton batch.
# parse_verdict() maps it to UNKNOWN, matching the old CUDA-OOM handling.
OOM_RAW = "CUDA OOM"


# ============================================================
# MODEL LOADING
# ============================================================

def load_gemma_with_adapter(
    checkpoint_path: str,
    base_model: str = "unsloth/gemma-4-E2B-it",
    max_seq_length: int = 1024,
    hf_token: str | None = None,
):
    """
    Load base Gemma in 4-bit, attach the CUAD LoRA adapter and switch
    to inference mode. Returns (model, tokenizer).

    Import unsloth at the top of the notebook BEFORE calling this.
    """
    from unsloth import FastLanguageModel
    from peft import PeftModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        dtype=None,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        full_finetuning=False,
        device_map={"": 0},
        token=hf_token,
    )
    model = PeftModel.from_pretrained(model, checkpoint_path)
    FastLanguageModel.for_inference(model)
    return model, tokenizer


# ============================================================
# PROMPT
# ============================================================

def build_compact_prompt(source_text: str, candidate_text: str, allow_no_match: bool = True) -> str:
    """
    allow_no_match=False is used when the aligner already knows both clauses are
    the same clause (very high similarity or same heading): Gemma then only
    decides EQUIVALENT vs MODIFIED.
    """
    if allow_no_match:
        options = """EQUIVALENT - same subject, same meaning
MODIFIED   - same subject, but something changed
NO_MATCH   - the clauses cover different subjects"""
    else:
        options = """These are the same clause taken from two versions of one contract.
EQUIVALENT - the meaning is unchanged
MODIFIED   - something changed"""
    return f"""Compare SOURCE and TARGET legal clauses.

{options}

Differences in parties, dates, amounts, rights, or conditions count as
material differences. A reversed or changed duty is MODIFIED, not NO_MATCH:
for example "may" becoming "shall not", or "is required to" becoming
"is not required to". Do not invent facts not present in the text.

Return ONLY this JSON, with <RESULT> replaced by your one chosen answer
and <CHANGE> by a short label of what changed (empty if nothing changed).
No markdown, no extra text:
{{"alignment_result": "<RESULT>", "change_type": "<CHANGE>"}}

SOURCE:
{source_text}

TARGET:
{candidate_text}"""


def build_one_word_prompt(source_text: str, candidate_text: str, allow_no_match: bool = True) -> str:
    """Retry prompt: the question comes last, so the first word Gemma writes is the answer."""
    options = "EQUIVALENT - same meaning\nMODIFIED - same subject, something changed"
    if allow_no_match:
        options += "\nNO_MATCH - different subjects"
    return f"""SOURCE clause:
{source_text}

TARGET clause:
{candidate_text}

Compare the two clauses. Reply with exactly ONE word:
{options}"""


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
# VERIFIER
# ============================================================

class GemmaVerifier:
    """Wraps the fine-tuned Gemma model for clause comparison."""

    def __init__(self, model, tokenizer, config: Node3Config | None = None):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or Node3Config()
        self._calls = 0       # single-pair generate calls (API compat counter)
        self._batches = 0     # generate calls issued (batched or not)

    # ------------------------------------------------------------
    # ENCODING
    # ------------------------------------------------------------

    def _encode(self, prompt: str):
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )

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

    def build_inputs(
        self,
        source_text: str,
        candidate_text: str,
        allow_no_match: bool = True,
        prompt_builder=None,
    ):
        """Shrink both sides equally until the prompt fits the token budget."""
        prompt_builder = prompt_builder or build_compact_prompt
        limit = self.config.text_chars
        for _ in range(5):
            s, t = source_text[:limit], candidate_text[:limit]
            inputs = self._encode(prompt_builder(s, t, allow_no_match))
            if inputs["input_ids"].shape[-1] <= self.config.max_input_tokens:
                break
            limit = int(limit * 0.75)
        truncated = len(source_text) > limit or len(candidate_text) > limit
        return inputs, truncated

    def _fit_prompt_ids(self, source_text: str, candidate_text: str, allow_no_match: bool, prompt_builder):
        """Same shrinking logic as build_inputs, but returns list-form encoding."""
        limit = self.config.text_chars
        enc, truncated = None, False
        for _ in range(5):
            s, t = source_text[:limit], candidate_text[:limit]
            enc = self._encode_ids(prompt_builder(s, t, allow_no_match))
            if len(enc["input_ids"]) <= self.config.max_input_tokens:
                break
            limit = int(limit * 0.75)
        truncated = len(source_text) > limit or len(candidate_text) > limit
        return enc, truncated

    # ------------------------------------------------------------
    # BATCHED GENERATION
    # ------------------------------------------------------------

    def _cleanup(self):
        if self._batches and self._batches % self.config.cleanup_every == 0:
            gc.collect()
            torch.cuda.empty_cache()

    def _generate_batch(self, pairs, prompt_builder, max_new_tokens, stop_at_brace: bool, **generate_kwargs):
        """
        One model.generate call for all pairs. Returns list of
        (raw_response, truncated) in input order.
        """
        encoded, flags = [], []
        for source_text, candidate_text, _allow in pairs:
            enc, truncated = self._fit_prompt_ids(
                source_text, candidate_text, _allow, prompt_builder
            )
            encoded.append(enc)
            flags.append(truncated)

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
            if stop_at_brace and self.config.stop_at_json:
                stop = _make_json_stop_criteria(self.tokenizer, prompt_len)

            inputs = {k: v.to(self.model.device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                    **({"stopping_criteria": [stop]} if stop is not None else {}),
                    **generate_kwargs,
                )
        finally:
            tok.padding_side = old_side
            if old_pad is None and tok.pad_token is not None:
                tok.pad_token = None
            self._batches += 1
            self._cleanup()

        rows = []
        for i in range(len(pairs)):
            generated = outputs[i][prompt_len:]
            rows.append((self.tokenizer.decode(generated, skip_special_tokens=True).strip(), flags[i]))
        return rows

    def compare_pairs(self, pairs: list[tuple[str, str, bool]]) -> list[tuple[str, bool]]:
        """
        Verify a list of (source_text, candidate_text, allow_no_match) pairs in
        batches of config.batch_size. Returns (raw_response, truncated) per
        pair, in the same order. CUDA OOM splits the batch; a pair that fails
        alone returns (OOM_RAW, False) — parse_verdict maps it to UNKNOWN.
        """
        results: list[tuple[str, bool]] = []
        for start in range(0, len(pairs), max(1, self.config.batch_size)):
            chunk = pairs[start:start + max(1, self.config.batch_size)]
            results.extend(self._compare_chunk(chunk))
        return results

    def _compare_chunk(self, chunk):
        try:
            return self._generate_batch(
                chunk, build_compact_prompt, self.config.max_new_tokens, stop_at_brace=True
            )
        except torch.cuda.OutOfMemoryError:
            gc.collect()
            torch.cuda.empty_cache()
            if len(chunk) > 1 and self.config.oom_split:
                mid = len(chunk) // 2
                return self._compare_chunk(chunk[:mid]) + self._compare_chunk(chunk[mid:])
            return [(OOM_RAW, False)] * len(chunk)

    def retry_pairs(self, pairs: list[tuple[str, str, bool]]) -> list[tuple[str, bool]]:
        """Batched one-word retry for unreadable answers (same prompt as before)."""
        results: list[tuple[str, bool]] = []
        for start in range(0, len(pairs), max(1, self.config.batch_size)):
            chunk = pairs[start:start + max(1, self.config.batch_size)]
            try:
                results.extend(self._generate_batch(
                    chunk, build_one_word_prompt, 8, stop_at_brace=False, repetition_penalty=1.3
                ))
            except torch.cuda.OutOfMemoryError:
                gc.collect()
                torch.cuda.empty_cache()
                if len(chunk) > 1 and self.config.oom_split:
                    mid = len(chunk) // 2
                    results.extend(self.retry_pairs(chunk[:mid]))
                    results.extend(self.retry_pairs(chunk[mid:]))
                else:
                    results.extend([(OOM_RAW, False)] * len(chunk))
        return results

    # ------------------------------------------------------------
    # SINGLE-PAIR API (unchanged behaviour)
    # ------------------------------------------------------------

    def compare_clauses(self, source_text: str, candidate_text: str, allow_no_match: bool = True):
        """Returns (raw_response, truncated) for one pair."""
        self._calls += 1
        return self.compare_pairs([(source_text, candidate_text, allow_no_match)])[0]

    def retry_clauses(self, source_text: str, candidate_text: str, allow_no_match: bool = True):
        """
        Second attempt when the first answer was unreadable. One-word prompt,
        few tokens and a repetition penalty: on Kaggle, Gemma once looped
        ('"1.1 . . . . .') until it ran out of tokens.
        """
        self._calls += 1
        return self.retry_pairs([(source_text, candidate_text, allow_no_match)])[0]
