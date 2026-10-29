"""
ClauseLens — Pipeline
Configuration for one run of Nodes 1-6.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from nodes.node_3 import Node3Config
from nodes.node_4 import Node4Config
from nodes.node_5 import Node5Config


@dataclass
class PipelineConfig:
    # CUAD LoRA adapter folder (checkpoint-2091) for the Node 3 / Node 4 model
    checkpoint_path: str

    # Hugging Face token; None works for the public unsloth/* models
    hf_token: str | None = None

    # Display names of the two documents in the report
    document_names: tuple[str, str] = ("Document A", "Document B")

    # Node 1 + clean-up
    start_new_chunk_on_level: int = 2
    chunk_max_chars: int = 1500
    clean_up: bool = True

    # Node 5 models. The Critic tries these in order on GPU 1 (plain models, no
    # CUAD adapter). The Impact Agent uses IMPACT_BASE_MODEL on GPU 0 after the
    # Node 3 model is freed; None keeps the Node 3 model for the Impact Agent.
    critic_base_models: tuple[str, ...] = ("unsloth/gemma-4-E4B-it", "unsloth/gemma-4-E2B-it")
    critic_adapter: str | None = None
    impact_base_model: str | None = "unsloth/gemma-4-E4B-it"

    # Node 5: run two pairs one at a time before the parallel run, so the two
    # GPUs don't compile shared torch code at the same moment
    warm_up_pairs: int = 2

    # Where Node 4 / Node 5 write their JSONL checkpoints (resume after a crash)
    output_dir: str = "clauselens_outputs"

    # Node settings
    node3: Node3Config = field(default_factory=Node3Config)
    node4: Node4Config = field(default_factory=Node4Config)
    node5: Node5Config = field(default_factory=Node5Config)

    verbose: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in ("checkpoint_path", "document_names", "chunk_max_chars",
                                              "clean_up", "critic_base_models", "impact_base_model")}
