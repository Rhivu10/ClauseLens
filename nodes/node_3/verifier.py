"""
ClauseLens — Node 3
Gemma verifier.

Loads the base Gemma model with the CUAD-trained LoRA adapter and
compares two clauses, returning Gemma's raw JSON response.
"""

from __future__ import annotations

import gc

import torch

from .config import Node3Config


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

def build_compact_prompt(source_text: str, candidate_text: str) -> str:
    return f"""Compare SOURCE and TARGET legal clauses.

Differences in parties, dates, amounts, rights, or conditions count as
material differences. Use NO_MATCH only if the clauses cover different subjects.
Do not invent facts not present in the text.

Return ONLY this JSON, no markdown, no extra text:
{{"alignment_result":"EQUIVALENT|MODIFIED|NO_MATCH","change_type":"short label, or empty"}}

SOURCE:
{source_text}

TARGET:
{candidate_text}"""


# ============================================================
# VERIFIER
# ============================================================

class GemmaVerifier:
    """Wraps the fine-tuned Gemma model for clause comparison."""

    def __init__(self, model, tokenizer, config: Node3Config | None = None):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or Node3Config()
        self._calls = 0

    def _encode(self, prompt: str):
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )

    def build_inputs(self, source_text: str, candidate_text: str):
        """Shrink both sides equally until the prompt fits the token budget."""
        limit = self.config.text_chars
        for _ in range(5):
            s, t = source_text[:limit], candidate_text[:limit]
            inputs = self._encode(build_compact_prompt(s, t))
            if inputs["input_ids"].shape[-1] <= self.config.max_input_tokens:
                break
            limit = int(limit * 0.75)
        truncated = len(source_text) > limit or len(candidate_text) > limit
        return inputs, truncated

    def compare_clauses(self, source_text: str, candidate_text: str):
        """Returns (raw_response, truncated). Cleanup always runs (try/finally)."""
        inputs = outputs = None
        try:
            inputs, truncated = self.build_inputs(source_text, candidate_text)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    do_sample=False,
                )
            generated = outputs[0][inputs["input_ids"].shape[-1]:]
            response = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
            return response, truncated
        finally:
            del inputs, outputs
            self._calls += 1
            if self._calls % self.config.cleanup_every == 0:
                gc.collect()
                torch.cuda.empty_cache()
