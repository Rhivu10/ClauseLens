"""
ClauseLens — Pipeline
LangGraph mapping of Nodes 1-6: one call runs the whole comparison.

    START
      |
    node_1_parse      Node 1: PDF -> hierarchy-aware chunks (+ clean-up)
      |
    node_2_index      Node 2: canonical chunks, FAISS / BM25 / graph index
      |
    node_3_align      Node 3: align clauses (Gemma E2B + CUAD adapter)
      |
    node_4_deontic    Node 4: changed pairs + duty cues
      |
      +-- no changed pairs? ------------------+
      |                                       |
    node_5_impact     Node 5: Impact Agent    |
      |               + Critic Agent          |
      |                                       |
    node_6_report  <--------------------------+
      |               Node 6: layered report
     END

Node 5's own loop (write -> check -> retry with more context) stays inside
Node5ImpactCore: it runs the Impact Agent and the Critic on two GPUs at the
same time, which a graph step per attempt would serialise.

Heavy objects that hold the GPU models (aligner, extractor, agents) are created
inside the node functions and dropped when the node returns, so the Node 3
model can be freed before the Node 5 models are loaded.

Usage:

    from pipeline import PipelineConfig, run_pipeline
    cfg = PipelineConfig(checkpoint_path=".../checkpoint-2091",
                         document_names=("2009 agreement", "2012 agreement"))
    state = run_pipeline("old.pdf", "new.pdf", cfg)
    state["report"]            # the Node 6 report
"""

from __future__ import annotations

import os
import re
import time
from typing import Any, Callable, TypedDict

from .cleanup import clean_chunks
from .config import PipelineConfig
from .models import ModelHub

StageCallback = Callable[..., None]

STAGES = [
    "node_1_parse",
    "node_2_index",
    "node_3_align",
    "node_4_deontic",
    "node_5_impact",
    "node_6_report",
]

# torch.compile clash between the two Node 5 threads
_COMPILE_CLASH = re.compile(r"dynamo|FX|symbolic", re.I)


class PipelineState(TypedDict, total=False):
    pdf_a: str
    pdf_b: str
    node1_result: dict[str, Any]
    cleanup_stats: dict[str, Any]
    all_chunks: list[dict[str, Any]]
    index_bundle: Any
    embedding_model: Any
    node3_report: dict[str, Any]
    node4_output: dict[str, Any]
    node5_output: dict[str, Any]
    report: dict[str, Any]
    output_files: dict[str, str]
    timings: dict[str, float]


def build_graph(
    config: PipelineConfig,
    models: ModelHub | None = None,
    on_stage: StageCallback | None = None,
):
    """
    The compiled LangGraph for Nodes 1-6.

    on_stage(name, **info) is called when a node starts (info: {}) and ends
    (info: {"seconds": ..., plus a short summary}), e.g. for a progress bar.
    """
    from langgraph.graph import END, START, StateGraph

    models = models or ModelHub(config)
    say = on_stage or (lambda name, **info: None)
    os.makedirs(config.output_dir, exist_ok=True)

    def timed(name: str, fn: Callable[[PipelineState], dict[str, Any]]):
        def run(state: PipelineState) -> dict[str, Any]:
            say(name)
            t0 = time.time()
            update = fn(state)
            seconds = round(time.time() - t0, 1)
            info = update.pop("_info", {})
            say(name, seconds=seconds, **info)
            if config.verbose:
                print(f"[pipeline] {name} done in {seconds:.0f}s {info or ''}")
            return {**update, "timings": {**state.get("timings", {}), name: seconds}}
        return run

    # ------------------------------------------------------------------ Node 1
    def node_1_parse(state: PipelineState) -> dict[str, Any]:
        from nodes.node_1 import process_documents
        result = process_documents(state["pdf_a"], state["pdf_b"],
                                   start_new_chunk_on_level=config.start_new_chunk_on_level,
                                   max_chars=config.chunk_max_chars)
        stats = clean_chunks(result, config.chunk_max_chars) if config.clean_up else {}
        return {"node1_result": result, "cleanup_stats": stats,
                "_info": {"chunks": {d: r["chunk_count"] for d, r in result.items()}}}

    # ------------------------------------------------------------------ Node 2
    def node_2_index(state: PipelineState) -> dict[str, Any]:
        from nodes.node_2 import Node2Indexer, prepare_chunks
        all_chunks = prepare_chunks(state["node1_result"],
                                    version_ids={"document_a": "version_a", "document_b": "version_b"})
        indexer = Node2Indexer()
        bundle = indexer.build(all_chunks)
        return {"all_chunks": all_chunks, "index_bundle": bundle,
                "embedding_model": indexer.embedding_model,
                "_info": {"total_chunks": len(all_chunks)}}

    # ------------------------------------------------------------------ Node 3
    def node_3_align(state: PipelineState) -> dict[str, Any]:
        from nodes.node_3 import GemmaVerifier, Node3Aligner
        model, tokenizer = models.node3()
        aligner = Node3Aligner(state["all_chunks"], state["index_bundle"], state["embedding_model"],
                               GemmaVerifier(model, tokenizer, config.node3), config.node3)
        report = aligner.align_documents("document_a", "document_b", verbose=config.verbose)
        return {"node3_report": report, "_info": {"counts": report["counts"]}}

    # ------------------------------------------------------------------ Node 4
    def node_4_deontic(state: PipelineState) -> dict[str, Any]:
        from nodes.node_4 import DeonticChecker, Node4Extractor
        model, tokenizer = models.node3()
        extractor = Node4Extractor(state["all_chunks"], DeonticChecker(model, tokenizer, config.node4))
        out = extractor.run(state["node3_report"],
                            checkpoint_path=os.path.join(config.output_dir, "node4_pairs.jsonl"),
                            verbose=config.verbose)
        return {"node4_output": out, "_info": {"counts": out["counts"]}}

    def after_node_4(state: PipelineState) -> str:
        return "node_5_impact" if state["node4_output"]["pairs"] else "node_6_report"

    # ------------------------------------------------------------------ Node 5
    def node_5_impact(state: PipelineState) -> dict[str, Any]:
        from nodes.node_5 import CriticAgent, ImpactAgent, Node5ImpactCore
        m = models.node5_agents()
        impact = ImpactAgent(m["impact_model"], m["impact_tokenizer"], config.node5)
        critic = CriticAgent(m["critic_model"], m["critic_tokenizer"], config.node5)
        core = Node5ImpactCore(state["all_chunks"], impact, critic, config.node5)
        pairs = state["node4_output"]["pairs"]

        # Warm-up (like Test A in the notebook): the first generation compiles
        # shared torch code; two threads doing that at once fail. Two pairs of
        # different lengths, one at a time, so later lengths need no recompile.
        if core.parallel and config.warm_up_pairs:
            say("node_5_warmup")
            both = [p for p in pairs if p.get("source") and p.get("target")]
            for p in (both or pairs)[:config.warm_up_pairs]:
                try:
                    view = core.build_view(p)
                    critic.check(view, impact.write(view)["summary"])
                except Exception as e:      # noqa: BLE001 — the warm-up only has to compile
                    print("warm-up:", type(e).__name__, str(e)[:200])

        checkpoint = os.path.join(config.output_dir, "node5_results.jsonl")
        try:
            out = core.run(pairs, checkpoint_path=checkpoint, verbose=config.verbose)
        except RuntimeError as e:
            if not core.parallel or not _COMPILE_CLASH.search(str(e)):
                raise
            # Same clash despite the warm-up: continue one at a time; finished
            # pairs are read back from the checkpoint, not redone
            print("Node 5: parallel run stopped (torch compile clash); continuing sequentially:", str(e)[:200])
            core.parallel = False
            out = core.run(pairs, checkpoint_path=checkpoint, verbose=config.verbose)
        return {"node5_output": out,
                "_info": {"counts": out["counts"], "parallel": out.get("parallel")}}

    # ------------------------------------------------------------------ Node 6
    def node_6_report(state: PipelineState) -> dict[str, Any]:
        from nodes.node_6 import ReportAssembler, save_markdown, save_report, save_ui_data
        results = (state.get("node5_output") or {}).get("results", [])
        report = ReportAssembler(
            documents={"document_a": config.document_names[0], "document_b": config.document_names[1]},
            all_chunks=state["all_chunks"],
            node3_report=state["node3_report"],
        ).add_many(results).build()
        report["pipeline"] = {"timings": dict(state.get("timings", {})), "config": config.as_dict()}
        files = {
            "report_json": save_report(report, os.path.join(config.output_dir, "report.json")),
            "report_md": save_markdown(report, os.path.join(config.output_dir, "report.md")),
            "ui_data": save_ui_data(report, os.path.join(config.output_dir, "ui_data.json")),
        }
        return {"report": report, "output_files": files,
                "_info": {"headline": report["overview"]["headline"]}}

    g = StateGraph(PipelineState)
    for name, fn in [("node_1_parse", node_1_parse), ("node_2_index", node_2_index),
                     ("node_3_align", node_3_align), ("node_4_deontic", node_4_deontic),
                     ("node_5_impact", node_5_impact), ("node_6_report", node_6_report)]:
        g.add_node(name, timed(name, fn))
    g.add_edge(START, "node_1_parse")
    g.add_edge("node_1_parse", "node_2_index")
    g.add_edge("node_2_index", "node_3_align")
    g.add_edge("node_3_align", "node_4_deontic")
    g.add_conditional_edges("node_4_deontic", after_node_4, ["node_5_impact", "node_6_report"])
    g.add_edge("node_5_impact", "node_6_report")
    g.add_edge("node_6_report", END)
    return g.compile()


def run_pipeline(
    pdf_a: str,
    pdf_b: str,
    config: PipelineConfig,
    on_stage: StageCallback | None = None,
    models: ModelHub | None = None,
) -> PipelineState:
    """Run Nodes 1-6 on two PDFs (old version, new version). Returns the final state."""
    graph = build_graph(config, models=models, on_stage=on_stage)
    return graph.invoke({"pdf_a": pdf_a, "pdf_b": pdf_b, "timings": {}})
