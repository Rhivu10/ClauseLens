"""
ClauseLens — Pipeline

LangGraph mapping of Nodes 1-6, so a whole comparison is one call:

    from pipeline import PipelineConfig, run_pipeline
    state = run_pipeline("old.pdf", "new.pdf", PipelineConfig())
    state["report"]

or from a shell (after python pipeline/install.py):

    python -m pipeline old.pdf new.pdf --out results
"""

# Unsloth must be imported before transformers / peft (which the nodes import).
# Skipped where it isn't installed or can't load (e.g. no GPU, CPU tests).
try:
    import unsloth  # noqa: F401
except Exception:  # noqa: BLE001
    pass

from .config import PipelineConfig
from .cleanup import clean_chunks
from .models import ModelHub
from .graph import STAGES, PipelineState, build_graph, run_pipeline
from .live import LiveReport

__all__ = [
    "PipelineConfig",
    "ModelHub",
    "PipelineState",
    "STAGES",
    "build_graph",
    "run_pipeline",
    "clean_chunks",
    "LiveReport",
]
