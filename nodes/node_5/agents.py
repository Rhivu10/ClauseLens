"""
ClauseLens — Node 5
Impact Agent and Critic Agent.

    GPU 0: ImpactAgent  (the Gemma + CUAD adapter loaded for Node 3, or a plain
                         Gemma E4B loaded with load_agent_model)
    GPU 1: CriticAgent  (a separate Gemma instance, see load_agent_model)

Agents may share one tokenizer. Hugging Face fast tokenizers are not safe
to call from two threads at once ("Already borrowed"), so every tokenizer
call goes through one lock. Generation itself runs in parallel.

Batching: generate_many() / write_many() / check_many() put several pairs into
one model.generate call (left padding). Each row gets exactly the prompt the
one-pair methods would build. A batch that runs out of GPU memory is split in
half and retried, down to single pairs.
"""

from __future__ import annotations

import gc
import threading
from typing import Any, Callable

import torch

from .config import Node5Config
from .prompts import (
    build_impact_prompt,
    build_critic_prompt,
    build_critic_retry_prompt,
    parse_impact,
    parse_critic,
)
from .config import UNKNOWN

_TOKENIZER_LOCK = threading.Lock()


def _text_tokenizer(tokenizer):
    """
    The plain text tokenizer. For Gemma 4, Unsloth returns a Gemma4Processor,
    which wraps the tokenizer and has no pad() (same as Node 3 / Node 4).
    """
    return getattr(tokenizer, "tokenizer", None) or tokenizer


# ============================================================
# MODEL LOADING
# ============================================================

CRITIC_BASE_MODELS = ("unsloth/gemma-4-E4B-it", "unsloth/gemma-4-E2B-it")


def load_agent_model(
    base_model: str = CRITIC_BASE_MODELS[0],
    device_index: int = 1,
    checkpoint_path: str | None = None,
    max_seq_length: int = 2048,
    hf_token: str | None = None,
):
    """
    Load an independent Gemma for an agent (the Critic on GPU 1, or optionally
    the Impact Agent on GPU 0). Returns (model, tokenizer).

    Default is the larger Gemma E4B (the workflow allows E2B or E4B): on Kaggle,
    an E2B Critic approved an invented detail and rejected a correct summary.
    checkpoint_path=None loads the plain instruction-tuned model, so the Critic
    shares no fine-tuning with the Impact Agent.

    Tries Unsloth first, then plain transformers.
    Import unsloth at the top of the notebook BEFORE calling this.
    """
    errors = []
    model = None
    try:
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_model,
            dtype=None,
            max_seq_length=max_seq_length,
            load_in_4bit=True,
            full_finetuning=False,
            device_map={"": device_index},
            token=hf_token,
        )
        if checkpoint_path:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, checkpoint_path)
        FastLanguageModel.for_inference(model)
        return model, tokenizer
    except Exception as e:      # noqa: BLE001 — fall back to transformers below
        errors.append(f"unsloth: {e}")
        model = None
        gc.collect()

    import transformers
    from transformers import BitsAndBytesConfig

    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    for cls_name in ("AutoModelForCausalLM", "AutoModelForImageTextToText"):
        cls = getattr(transformers, cls_name, None)
        if cls is None:
            continue
        try:
            model = cls.from_pretrained(base_model, quantization_config=quant,
                                        device_map={"": device_index}, token=hf_token)
            break
        except Exception as e:  # noqa: BLE001
            errors.append(f"{cls_name}: {e}")
    if model is None:
        raise RuntimeError("Could not load the Critic model:\n  " + "\n  ".join(errors))
    if checkpoint_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, checkpoint_path)
    model.eval()

    tokenizer = None
    for cls_name in ("AutoProcessor", "AutoTokenizer"):
        try:
            tokenizer = getattr(transformers, cls_name).from_pretrained(base_model, token=hf_token)
            break
        except Exception as e:  # noqa: BLE001
            errors.append(f"{cls_name}: {e}")
    return model, tokenizer    # tokenizer may be None: then reuse the Impact Agent's


# Kept for the name used in earlier notebooks
load_critic_model = load_agent_model


# ============================================================
# BASE AGENT
# ============================================================

class GemmaAgent:
    """Generation with a token budget, tokenizer lock and periodic CUDA cleanup."""

    name = "agent"

    def __init__(self, model, tokenizer, config: Node5Config | None = None):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or Node5Config()
        self._calls = 0

    def _encode(self, prompt: str):
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        with _TOKENIZER_LOCK:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )

    def generate(self, prompt_for_limit: Callable[[int], str], max_new_tokens: int, **kwargs) -> str:
        """
        prompt_for_limit(limit) builds the prompt with `limit` characters of clause
        text per side. The limit shrinks until the prompt fits the token budget.
        """
        budget = self.config.max_seq_length - max_new_tokens - 16
        limit = self.config.text_chars
        inputs = outputs = None
        try:
            for _ in range(6):
                inputs = self._encode(prompt_for_limit(limit))
                if inputs["input_ids"].shape[-1] <= budget:
                    break
                limit = int(limit * 0.75)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    repetition_penalty=kwargs.get("repetition_penalty", self.config.repetition_penalty),
                )
            generated = outputs[0][inputs["input_ids"].shape[-1]:]
            with _TOKENIZER_LOCK:
                return self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        except torch.cuda.OutOfMemoryError:
            self._empty_cache()
            return ""
        finally:
            del inputs, outputs
            self._calls += 1
            if self._calls % self.config.cleanup_every == 0:
                self._empty_cache()

    # ------------------------------------------------------------
    # BATCHED GENERATION
    # ------------------------------------------------------------

    def _encode_ids(self, prompt: str) -> list[int]:
        """Chat-template token ids as a plain list (for padding into a batch)."""
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        with _TOKENIZER_LOCK:
            enc = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
            )
        ids = enc["input_ids"]
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        return ids

    def generate_many(
        self,
        prompts_for_limit: list[Callable[[int], str]],
        max_new_tokens: int,
        **kwargs,
    ) -> list[str]:
        """
        generate() for several prompts in one model.generate call. Each prompt
        shrinks to the token budget exactly as in generate(). Returns the raw
        answers in input order.
        """
        if not prompts_for_limit:
            return []
        budget = self.config.max_seq_length - max_new_tokens - 16
        encoded = []
        for prompt_for_limit in prompts_for_limit:
            limit = self.config.text_chars
            for _ in range(6):
                ids = self._encode_ids(prompt_for_limit(limit))
                if len(ids) <= budget:
                    break
                limit = int(limit * 0.75)
            encoded.append(ids)
        return self._generate_encoded(encoded, max_new_tokens, **kwargs)

    def _generate_encoded(self, encoded: list[list[int]], max_new_tokens: int, **kwargs) -> list[str]:
        inputs = outputs = None
        try:
            with _TOKENIZER_LOCK:
                tok = _text_tokenizer(self.tokenizer)
                old_side, old_pad = getattr(tok, "padding_side", "right"), getattr(tok, "pad_token", None)
                tok.padding_side = "left"
                if tok.pad_token is None:
                    tok.pad_token = tok.eos_token
                try:
                    batch = tok.pad({"input_ids": encoded, "attention_mask": [[1] * len(e) for e in encoded]},
                                    padding=True, return_tensors="pt")
                finally:
                    tok.padding_side = old_side
                    if old_pad is None:
                        tok.pad_token = None
            prompt_len = int(batch["input_ids"].shape[-1])
            inputs = {k: v.to(self.model.device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    repetition_penalty=kwargs.get("repetition_penalty", self.config.repetition_penalty),
                )
            with _TOKENIZER_LOCK:
                return [self.tokenizer.decode(outputs[i][prompt_len:], skip_special_tokens=True).strip()
                        for i in range(len(encoded))]
        except torch.cuda.OutOfMemoryError:
            inputs = outputs = None
            self._empty_cache()
            if len(encoded) > 1 and self.config.oom_split:
                mid = len(encoded) // 2
                return (self._generate_encoded(encoded[:mid], max_new_tokens, **kwargs)
                        + self._generate_encoded(encoded[mid:], max_new_tokens, **kwargs))
            return [""] * len(encoded)          # like generate(): an empty answer on OOM
        finally:
            del inputs, outputs
            self._calls += 1
            if self._calls % self.config.cleanup_every == 0:
                self._empty_cache()

    def _empty_cache(self):
        gc.collect()
        device = getattr(self.model, "device", None)
        if device is not None and getattr(device, "type", "") == "cuda":
            with torch.cuda.device(device):
                torch.cuda.empty_cache()


# ============================================================
# IMPACT AGENT (GPU 0)
# ============================================================

class ImpactAgent(GemmaAgent):
    """Writes a short impact summary for one changed pair."""

    name = "impact"

    def write(self, view: dict[str, Any], warning: str | None = None) -> dict[str, Any]:
        raw = self.generate(lambda limit: build_impact_prompt(view, limit, warning),
                            self.config.impact_max_new_tokens)
        return {**parse_impact(raw), "raw": raw}

    def write_many(self, views: list[dict[str, Any]], warnings: list[str | None]) -> list[dict[str, Any]]:
        """write() for several pairs in one batch."""
        raws = self.generate_many(
            [(lambda limit, v=v, w=w: build_impact_prompt(v, limit, w)) for v, w in zip(views, warnings)],
            self.config.impact_max_new_tokens,
        )
        return [{**parse_impact(raw), "raw": raw} for raw in raws]


# ============================================================
# CRITIC AGENT (GPU 1)
# ============================================================

class CriticAgent(GemmaAgent):
    """Checks an impact summary against the clause text."""

    name = "critic"

    def check(self, view: dict[str, Any], summary: str, double_check: list[str] | None = None) -> dict[str, str]:
        """double_check: summary words not found in the text, for the Critic to verify."""
        raw = self.generate(lambda limit: build_critic_prompt(view, summary, limit, double_check),
                            self.config.critic_max_new_tokens)
        result = parse_critic(raw)
        if result["verdict"] == UNKNOWN:
            # Unreadable answer: one retry with a one-word prompt
            raw2 = self.generate(lambda limit: build_critic_retry_prompt(view, summary, limit),
                                 8, repetition_penalty=1.3)
            result = parse_critic(raw2)
            result["retried"] = True
            result.setdefault("raw", raw2[:200])
        return result

    def check_many(
        self,
        views: list[dict[str, Any]],
        summaries: list[str],
        double_checks: list[list[str] | None],
    ) -> list[dict[str, str]]:
        """check() for several pairs in one batch; unreadable answers are retried together."""
        raws = self.generate_many(
            [(lambda limit, v=v, s=s, d=d: build_critic_prompt(v, s, limit, d))
             for v, s, d in zip(views, summaries, double_checks)],
            self.config.critic_max_new_tokens,
        )
        results = [parse_critic(raw) for raw in raws]
        again = [i for i, r in enumerate(results) if r["verdict"] == UNKNOWN]
        if again:
            raws2 = self.generate_many(
                [(lambda limit, v=views[i], s=summaries[i]: build_critic_retry_prompt(v, s, limit)) for i in again],
                8, repetition_penalty=1.3,
            )
            for i, raw2 in zip(again, raws2):
                result = parse_critic(raw2)
                result["retried"] = True
                result.setdefault("raw", raw2[:200])
                results[i] = result
        return results
