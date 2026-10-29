"""
ClauseLens — Pipeline
Model handling across the nodes (Kaggle GPU T4 x2).

    GPU 0: Gemma E2B + CUAD adapter  -> Node 3, Node 4
           freed, then Gemma E4B      -> Node 5 Impact Agent
    GPU 1: Gemma E4B (or E2B)         -> Node 5 Critic

On a T4, Gemma 4 runs in float32 (Unsloth refuses float16 for gemma4), so E4B
(~10 GB) does not fit next to the Node 3 model (~7.8 GB); Nodes 3 and 4 are done
by the time Node 5 starts, so the Node 3 model is freed first. With one GPU, or
if a model can't be loaded, both agents share one model and Node 5 runs
sequentially.

The loaders are attributes, so tests can replace them with small fakes.
"""

from __future__ import annotations

import gc
from typing import Any, Callable

from .config import PipelineConfig


def _default_node3_loader(cfg: PipelineConfig):
    from nodes.node_3 import load_gemma_with_adapter
    return load_gemma_with_adapter(cfg.checkpoint_path, max_seq_length=cfg.node5.max_seq_length,
                                   hf_token=cfg.hf_token)


def _default_agent_loader(cfg: PipelineConfig, base_model: str, device_index: int,
                          checkpoint_path: str | None = None):
    from nodes.node_5 import load_agent_model
    return load_agent_model(base_model=base_model, device_index=device_index,
                            checkpoint_path=checkpoint_path,
                            max_seq_length=cfg.node5.max_seq_length, hf_token=cfg.hf_token)


def _gpu_count() -> int:
    import torch
    return torch.cuda.device_count()


def _free_gpu_memory() -> None:
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


class ModelHub:
    """Loads, hands over and frees the models as the pipeline moves through the nodes."""

    def __init__(
        self,
        config: PipelineConfig,
        node3_loader: Callable[..., Any] = _default_node3_loader,
        agent_loader: Callable[..., Any] = _default_agent_loader,
        gpu_count: Callable[[], int] = _gpu_count,
    ):
        self.config = config
        self.node3_loader = node3_loader
        self.agent_loader = agent_loader
        self.gpu_count = gpu_count
        self.node3_model = self.node3_tokenizer = None
        self.notes: list[str] = []

    def _note(self, msg: str) -> None:
        self.notes.append(msg)
        if self.config.verbose:
            print(msg)

    # Node 3 + Node 4
    def node3(self):
        if self.node3_model is None:
            self.node3_model, self.node3_tokenizer = self.node3_loader(self.config)
            self._note("Node 3 model: Gemma E2B + CUAD adapter (GPU 0)")
        return self.node3_model, self.node3_tokenizer

    # Node 5
    def node5_agents(self) -> dict[str, Any]:
        """
        Returns {impact_model, impact_tokenizer, critic_model, critic_tokenizer}.
        Frees the Node 3 model when a separate Impact model is loaded.
        """
        cfg = self.config
        model, tokenizer = self.node3()

        critic_model = critic_tokenizer = None
        if self.gpu_count() >= 2:
            for name in cfg.critic_base_models:
                try:
                    critic_model, critic_tokenizer = self.agent_loader(cfg, name, 1, cfg.critic_adapter)
                    self._note(f"Critic: {name} (GPU 1)")
                    break
                except Exception as e:      # noqa: BLE001 — try the next model
                    self._note(f"Critic {name} could not be loaded on GPU 1: {str(e)[:300]}")
                    critic_model = None
                    _free_gpu_memory()
        if critic_model is None:
            critic_model = model
            self._note("Critic shares the Node 3 model (sequential mode).")
        critic_tokenizer = critic_tokenizer or tokenizer

        impact_model, impact_tokenizer = model, tokenizer
        if cfg.impact_base_model and critic_model is not model:
            # Nodes 3 and 4 are finished: free their model to make room for E4B
            self.node3_model = self.node3_tokenizer = None
            model = impact_model = None
            _free_gpu_memory()
            try:
                impact_model, tok = self.agent_loader(cfg, cfg.impact_base_model, 0)
                impact_tokenizer = tok or tokenizer
                self._note(f"Impact Agent: {cfg.impact_base_model} (GPU 0)")
            except Exception as e:          # noqa: BLE001
                self._note(f"Impact model {cfg.impact_base_model} could not be loaded: {str(e)[:300]}")
                self._note("Impact Agent shares the Critic's model (sequential mode).")
                _free_gpu_memory()
                impact_model, impact_tokenizer = critic_model, critic_tokenizer
        elif cfg.impact_base_model:
            self._note("No separate Critic model, so the Node 3 model is kept for both agents.")
        else:
            self._note("Impact Agent: the Node 3 model (Gemma E2B + CUAD adapter)")

        return {"impact_model": impact_model, "impact_tokenizer": impact_tokenizer,
                "critic_model": critic_model, "critic_tokenizer": critic_tokenizer}
