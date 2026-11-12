# ============================================================================
# ClauseLens — full pipeline (Nodes 1-6) through the LangGraph mapping
# Kaggle, accelerator: GPU T4 x2, internet on.
# Add the CUAD adapter dataset (checkpoint-2091) and a dataset with your two PDFs.
# Each "# %%" marker is one Kaggle cell. Run them in order.
# ============================================================================

# %% [code] CELL 1 — INSTALL
%%capture
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"   # before torch starts CUDA

!pip install unsloth
!pip install --no-deps transformers==5.5.0 "tokenizers>=0.22.0,<=0.23.0"
!pip install "huggingface_hub>=1.5.0,<2.0"
!pip install torchcodec
!pip install -q pymupdf numpy sentence-transformers faiss-cpu rank-bm25 networkx langgraph

# %% [code] CELL 2 — CLAUSELENS REPOSITORY
import sys
REPO_PATH = "/kaggle/working/ClauseLens"
if not os.path.exists(REPO_PATH):
    !git clone -b main https://github.com/Rhivu10/ClauseLens.git {REPO_PATH}
else:
    !git -C {REPO_PATH} pull origin main
sys.path.insert(0, REPO_PATH)

import unsloth  # must be imported before transformers / peft
import torch
torch._dynamo.config.recompile_limit = 64
print("GPUs:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])

# %% [code] CELL 3 — SETTINGS
import glob


def find(name):
    hits = glob.glob(f"/kaggle/input/**/{name}", recursive=True)
    if not hits:
        raise FileNotFoundError(f"{name} not found under /kaggle/input — add the dataset that contains it.")
    return hits[0]


from pipeline import PipelineConfig, run_pipeline

PDF_A = find("Echo_Ferndale_2009_original.pdf")    # previous version
PDF_B = find("Echo_Ferndale_2012_restated.pdf")    # new version

cfg = PipelineConfig(
    checkpoint_path=os.path.dirname(find("checkpoint-2091/adapter_config.json")),
    document_names=("License Agreement (2009)", "Amended and Restated License Agreement (2012)"),
    output_dir="/kaggle/working/clauselens_outputs",
)

# %% [code] CELL 4 — RUN NODES 1-6
state = run_pipeline(PDF_A, PDF_B, cfg)

print(state["report"]["overview"]["headline"])
print("seconds per node:", state["timings"])
print("files:", state["output_files"])

# %% [code] CELL 5 — SHOW THE REPORT
from IPython.display import Markdown, display
from nodes.node_6 import to_markdown
display(Markdown(to_markdown(state["report"])))
