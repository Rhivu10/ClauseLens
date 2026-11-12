"""
ClauseLens — Pipeline
Installs the libraries Nodes 1-6 need (GPU machine, e.g. Kaggle "GPU T4 x2").

    python pipeline/install.py

Run it as a script (not "python -m pipeline.install"): importing the pipeline
package already needs these libraries. The Kaggle runner of the web app calls
this file too, so the library list exists only here.
"""

import subprocess
import sys

# In install order. transformers is pinned without its dependencies on purpose:
# Unsloth would otherwise pull a version Gemma 4 + the CUAD adapter weren't tested with.
PIP_STEPS = [
    ["unsloth"],
    ["--no-deps", "transformers==5.5.0", "tokenizers>=0.22.0,<=0.23.0"],
    ["huggingface_hub>=1.5.0,<2.0"],
    ["torchcodec"],
    ["pymupdf", "numpy", "sentence-transformers", "faiss-cpu", "rank-bm25", "networkx", "langgraph"],
]


def main() -> None:
    for step in PIP_STEPS:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *step])
    print("ClauseLens libraries installed.")


if __name__ == "__main__":
    main()
