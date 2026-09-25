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

    def compare_clauses(self, source_text: str, candidate_text: str, allow_no_match: bool = True):
        """Returns (raw_response, truncated). Cleanup always runs (try/finally)."""
        return self._run(source_text, candidate_text, allow_no_match, build_compact_prompt,
                         max_new_tokens=self.config.max_new_tokens)

    def retry_clauses(self, source_text: str, candidate_text: str, allow_no_match: bool = True):
        """
        Second attempt when the first answer was unreadable. One-word prompt,
        few tokens and a repetition penalty: on Kaggle, Gemma once looped
        ('"1.1 . . . . .') until it ran out of tokens.
        """
        return self._run(source_text, candidate_text, allow_no_match, build_one_word_prompt,
                         max_new_tokens=8, repetition_penalty=1.3)

    def _run(self, source_text, candidate_text, allow_no_match, prompt_builder, **generate_kwargs):
        inputs = outputs = None
        try:
            inputs, truncated = self.build_inputs(source_text, candidate_text, allow_no_match, prompt_builder)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model.generate(**inputs, do_sample=False, **generate_kwargs)
            generated = outputs[0][inputs["input_ids"].shape[-1]:]
            response = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
            return response, truncated
        finally:
            del inputs, outputs
            self._calls += 1
            if self._calls % self.config.cleanup_every == 0:
                gc.collect()
                torch.cuda.empty_cache()
