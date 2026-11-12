"""
ClauseLens — Pipeline

LangGraph mapping of Nodes 1-6, so a whole comparison is one call:

    from pipeline import PipelineConfig, run_pipeline
    state = run_pipeline("old.pdf", "new.pdf", PipelineConfig(checkpoint_path=...))
    state["report"]
"""

from .config import PipelineConfig
from .cleanup import clean_chunks
from .models import ModelHub
from .graph import STAGES, PipelineState, build_graph, run_pipeline

__all__ = [
    "PipelineConfig",
    "ModelHub",
    "PipelineState",
    "STAGES",
    "build_graph",
    "run_pipeline",
    "clean_chunks",
]
