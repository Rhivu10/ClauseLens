# Node 1 Documentation

## Overview

**Node 1** is responsible for PDF document ingestion and hierarchy-aware chunking.

> **Notice:** Although this is considered a node, it is **not** part of the LangGraph architecture.

Node 1 currently:

- Accepts two PDF paths.
- Extracts text from the PDFs using **PyMuPDF**.
- Preserves PDF layout information:
  - Page number
  - Position
  - Font size
  - Font name
  - Boldness
- Detects potential headings using:
  - Numbering patterns
  - Font size
  - Boldness
  - Line length
- Maintains a hierarchy path.
- Creates hierarchy-aware chunks.
- Returns structured information for both documents.

There is **no ML or embedding model in Node 1**. Embeddings and vector storage belong to **Node 2**.

---

## Public Interface

The current contents of Node 1 are:

```python
from .pdf_parser import (
    extract_lines,
    process_document,
    process_documents,
)

__all__ = [
    "extract_lines",
    "process_document",
    "process_documents",
]
```

This defines the public interface of Node 1.

### Importing Node 1

Use:

```python
from nodes.node_1 import process_documents
```

The primary function is:

```python
process_documents(doc1, doc2)
```

It accepts the paths for **Doc1** and **Doc2** and processes both documents.

---

## Document Paths

Node 1 does **not** contain the document paths internally.

Both **Doc1** and **Doc2** paths are provided through the UI element and passed to:

```python
process_documents(doc1, doc2)
```

---

# Kaggle Setup

## 1. Clone the Repository

Make sure **Internet access is enabled** in Kaggle if you need to clone the GitHub repository.

First, clone the repository:

```bash
!git clone https://github.com/Rhivu10/ClauseLens.git /kaggle/working/ClauseLens
```

If the repository has already been cloned, use:

```bash
%cd /kaggle/working/ClauseLens
!git pull
```

The repository should be available at:

```text
/kaggle/working/ClauseLens
```

---

## 2. Make the Repository Importable

Kaggle needs to know where the Python package is located.

Add the repository path to `sys.path`:

```python
import sys

REPO_PATH = "/kaggle/working/ClauseLens"

if REPO_PATH not in sys.path:
    sys.path.insert(0, REPO_PATH)
```

After this, the `nodes` package should be importable.

---

# Dependencies

Although dependencies are installed during compilation of the entire codebase, for an isolated Node 1 use case it is preferable to install the required dependency separately.

Node 1 uses:

```python
import fitz
```

`fitz` is provided by **PyMuPDF**.

Install it with:

```bash
!pip install -q PyMuPDF
```

---

# Testing Node 1

## 1. Test the Import

To verify that Node 1 can be imported successfully:

```python
from nodes.node_1 import (
    extract_lines,
    process_document,
    process_documents,
)

print("Node 1 import: SUCCESS")
```

Expected output:

```text
Node 1 import: SUCCESS
```

---

## 2. Test Node 1 Execution

For testing purposes, run:

```python
result = process_documents(Doc1, Doc2)

print("Node 1 execution: SUCCESS")

for name, document in result.items():
    print("=" * 70)
    print(name)
    print("Lines:", document["element_count"])
    print("Chunks:", document["chunk_count"])
```

This verifies that:

1. Both documents are processed.
2. Structured document information is returned.
3. The number of extracted elements is available through `element_count`.
4. The number of generated hierarchy-aware chunks is available through `chunk_count`.

---

# Node 1 Responsibilities

The overall responsibility of Node 1 can be summarized as:

```text
PDF Paths
   |
   v
PDF Text Extraction
   |
   v
Layout Information Preservation
   |
   v
Heading Detection
   |
   v
Hierarchy Construction
   |
   v
Hierarchy-Aware Chunking
   |
   v
Structured Document Output
```

Node 1 focuses exclusively on **PDF ingestion, structural analysis, and hierarchy-aware chunking**.

Embedding generation and vector storage are handled by **Node 2**.