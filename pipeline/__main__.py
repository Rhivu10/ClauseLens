"""
ClauseLens — run Nodes 1-6 from the command line (GPU machine, e.g. Kaggle "GPU T4 x2").

    python pipeline/install.py                       # once: the libraries
    python -m pipeline old.pdf new.pdf --out results

Writes report.json, report.md and ui_data.json to --out and prints the headline.
"""

from __future__ import annotations

import argparse
import os
import sys


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m pipeline",
                                 description="Compare two versions of a contract (ClauseLens Nodes 1-6).")
    ap.add_argument("old_pdf", help="previous version")
    ap.add_argument("new_pdf", help="new version")
    ap.add_argument("--out", default="clauselens_outputs", help="output folder (default: clauselens_outputs)")
    ap.add_argument("--names", nargs=2, metavar=("OLD_NAME", "NEW_NAME"),
                    help="document names in the report (default: the file names)")
    ap.add_argument("--checkpoint", help="CUAD adapter folder (default: the repo's checkpoint-2091)")
    ap.add_argument("--quiet", action="store_true", help="less output")
    args = ap.parse_args(argv)

    for path in (args.old_pdf, args.new_pdf):
        if not os.path.isfile(path):
            ap.error(f"file not found: {path}")

    from pipeline import PipelineConfig, run_pipeline           # imports Unsloth first (see __init__)
    import torch
    torch._dynamo.config.recompile_limit = 64
    if not torch.cuda.is_available():
        print("No GPU found. ClauseLens needs a CUDA GPU (e.g. Kaggle 'GPU T4 x2').", file=sys.stderr)
        return 1

    name = lambda p: os.path.splitext(os.path.basename(p))[0]
    cfg = PipelineConfig(
        document_names=tuple(args.names) if args.names else (name(args.old_pdf), name(args.new_pdf)),
        output_dir=args.out,
        hf_token=os.environ.get("HF_TOKEN"),
        verbose=not args.quiet,
        **({"checkpoint_path": args.checkpoint} if args.checkpoint else {}),
    )

    def on_stage(stage: str, **info) -> None:
        if "seconds" in info:
            print(f"[ClauseLens] {stage} done in {info['seconds']:.0f}s", flush=True)
        else:
            print(f"[ClauseLens] {stage}", flush=True)

    state = run_pipeline(args.old_pdf, args.new_pdf, cfg, on_stage=on_stage)
    print("\n" + state["report"]["overview"]["headline"])
    for kind, path in state["output_files"].items():
        print(f"  {kind:12} {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
