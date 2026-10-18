"""
ClauseLens — Node 5
Impact Agent and Critic Agent.

    GPU 0: ImpactAgent  (the Gemma + CUAD adapter loaded for Node 3, or a plain
                         Gemma E4B loaded with load_agent_model)
    GPU 1: CriticAgent  (a separate Gemma instance, see load_agent_model)

Agents may share one tokenizer. Hugging Face fast tokenizers are not safe
to call from two threads at once ("Already borrowed"), so every tokenizer
call goes through one lock. Generation itself runs in parallel.
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
