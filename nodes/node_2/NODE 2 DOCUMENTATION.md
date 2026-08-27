# Node 2 Documentation

## Overview

**Node 2** is responsible for indexing and retrieving the structured chunks produced by Node 1.

> **Notice:** Although this is considered a node, it is **not** part of the LangGraph architecture.

Node 2 currently:

- Accepts the structured chunk output from Node 1.
- Validates and normalizes the chunk structure.
- Maintains document and version metadata.
- Generates numerical embeddings using a lightweight Sentence Transformer model.
- Stores embeddings in a **FAISS** vector index.
- Builds a **BM25** keyword-search index.
- Builds an in-memory **NetworkX knowledge graph**.
- Extracts relationships involving:
  - Stakeholders
  - Legal terms
  - Dates
- Provides semantic retrieval.
- Provides keyword retrieval.
- Provides graph-based retrieval.
- Provides hybrid semantic + keyword retrieval using Reciprocal Rank Fusion (RRF).
- Maintains mappings between search-index IDs and canonical chunk IDs.

Node 2 does **not** ingest PDFs directly. PDF ingestion and hierarchy-aware chunking belong to **Node 1**.

The current Node 2 implementation is designed to process the **complete chunk collection supplied by Node 1**, rather than a fixed portion of a document.

---

# Node 2 Architecture

The current Node 2 structure is:

```text
node_2/
├── __init__.py
├── chunks.py
├── indexer.py
└── retrieval.py
```

### File Responsibilities

| File | Responsibility |
|---|---|
| `__init__.py` | Defines the public Node 2 interface |
| `chunks.py` | Validates and prepares Node 1 chunks |
| `indexer.py` | Builds embeddings, FAISS, BM25 and NetworkX indexes |
| `retrieval.py` | Performs semantic, keyword, graph and hybrid retrieval |

---
# Public Interface

The public interface of Node 2 is defined in:

```python
nodes/node_2/__init__.py
```

Current contents:

```python
from .chunks import (
    prepare_chunks,
)

from .indexer import (
    Node2Indexer,
    IndexBundle,
)

from .retrieval import (
    semantic_search,
    keyword_search,
    graph_entity_search,
    graph_multi_entity_search,
    hybrid_search,
)

__all__ = [
    "prepare_chunks",
    "Node2Indexer",
    "IndexBundle",
    "semantic_search",
    "keyword_search",
    "graph_entity_search",
    "graph_multi_entity_search",
    "hybrid_search",
]
```

This allows other nodes to import Node 2 through its public interface rather than importing its internal implementation directly.

For example:

```python
from nodes.node_2 import (
    prepare_chunks,
    Node2Indexer,
    semantic_search,
    keyword_search,
    graph_entity_search,
    graph_multi_entity_search,
    hybrid_search,
)
```

---

# Kaggle Setup

## 1. Clone the Repository

Make sure **Internet access is enabled** in Kaggle if the repository needs to be cloned.

```bash
!git clone https://github.com/Rhivu10/ClauseLens.git /kaggle/working/ClauseLens
```

If the repository has already been cloned:

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

The ClauseLens repository contains the nodes under:

```text
ClauseLens/
└── nodes/
    ├── node_1/
    └── node_2/
```

Kaggle therefore needs the repository root available on `sys.path`.

```python
import sys

REPO_PATH = "/kaggle/working/ClauseLens"

if REPO_PATH not in sys.path:
    sys.path.insert(0, REPO_PATH)
```

After this, the following import should work:

```python
from nodes.node_2 import (
    prepare_chunks,
    Node2Indexer,
    IndexBundle,
    semantic_search,
    keyword_search,
    graph_entity_search,
    graph_multi_entity_search,
    hybrid_search,
)

print("Node 2 public API import: SUCCESS")
```

Expected output:

```text
Node 2 public API import: SUCCESS
```

---

# Dependencies

Node 2 uses the following direct dependencies:

```text
NumPy
Sentence Transformers
FAISS
rank-bm25
NetworkX
```

The embedding model used by the current implementation is:

```text
sentence-transformers/all-MiniLM-L6-v2
```

For the complete Node 1 → Node 2 test, PyMuPDF is also required because Node 1 is responsible for PDF ingestion.

### Install Node 2 Dependencies

```python
import sys
import subprocess

packages = [
    "numpy",
    "sentence-transformers",
    "faiss-cpu",
    "rank-bm25",
    "networkx",
]

subprocess.check_call([
    sys.executable,
    "-m",
    "pip",
    "install",
    "-q",
    *packages,
])

print("Node 2 dependencies installed successfully.")
```

### Install PyMuPDF for the Complete Pipeline

PyMuPDF is a **Node 1 dependency**, not a Node 2 dependency.

If testing Node 1 and Node 2 together:

```python
import sys
import subprocess

subprocess.check_call([
    sys.executable,
    "-m",
    "pip",
    "install",
    "-q",
    "pymupdf",
])

print("PyMuPDF installed successfully.")
```

---

# 1. `chunks.py`

## Purpose

`chunks.py` converts the output of Node 1 into the canonical chunk representation expected by Node 2.

It performs:

- Chunk validation.
- Required-field validation.
- Empty-text validation.
- Chunk ID uniqueness validation.
- Document ID assignment.
- Version ID assignment.
- Metadata normalization.

It does **not**:

- Generate embeddings.
- Build FAISS.
- Build BM25.
- Build the knowledge graph.
- Perform retrieval.

---

## `prepare_chunks()`

### Function

```python
prepare_chunks(
    documents,
    version_ids=None
)
```

### Input

#### `documents`

A dictionary containing the structured output generated by Node 1.

The expected structure is:

```python
{
    "document_a": {
        "document_id": "document_a",
        "chunks": [
            {
                "chunk_id": "document_a_0001",
                "text": "...",
                "page_start": 1,
                "page_end": 1,
                "heading_path": [...]
            }
        ]
    },

    "document_b": {
        "document_id": "document_b",
        "chunks": [
            {
                "chunk_id": "document_b_0001",
                "text": "...",
                "page_start": 1,
                "page_end": 1,
                "heading_path": [...]
            }
        ]
    }
}
```

Node 1 produces hierarchy-aware chunks containing the chunk text, page information and heading hierarchy. :contentReference[oaicite:2]{index=2}

#### `version_ids`

Optional dictionary mapping each document ID to a version ID.

Example:

```python
{
    "document_a": "version_a",
    "document_b": "version_b"
}
```

If no version IDs are provided:

```python
version_ids=None
```

the resulting chunks contain:

```python
"version_id": None
```

---

### Output

Returns:

```python
list[dict]
```

Each chunk is converted into the canonical Node 2 representation:

```python
{
    "chunk_id": "...",
    "document_id": "...",
    "version_id": "...",
    "text": "...",
    "page_start": ...,
    "page_end": ...,
    "heading_path": [...]
}
```

### Example

```python
all_chunks = prepare_chunks(
    result,
    version_ids={
        "document_a": "version_a",
        "document_b": "version_b",
    },
)
```

For the current test documents:

```text
document_a → 33 chunks
document_b → 23 chunks
```

Therefore:

```text
Total chunks: 56
```

---

## Validation Performed

`prepare_chunks()` verifies that every chunk contains:

```text
chunk_id
text
page_start
page_end
heading_path
```

It also verifies:

- Text is a string.
- Text is not empty.
- Chunk IDs are unique.

If a required field is missing, a `ValueError` is raised.

If the text is not a string, a `TypeError` is raised.

If duplicate chunk IDs are detected, a `ValueError` is raised.

---

# 2. `indexer.py`

## Purpose

`indexer.py` is the main construction layer of Node 2.

It takes the canonical chunks produced by `prepare_chunks()` and creates:

```text
Chunks
   |
   +---- Embeddings ----> FAISS
   |
   +---- Tokens --------> BM25
   |
   +---- Entities ------> NetworkX
```

The indexer does not perform user queries. Query execution belongs to `retrieval.py`.

---

# `Node2Indexer`

## Initialization

```python
indexer = Node2Indexer()
```

The default embedding model is:

```text
sentence-transformers/all-MiniLM-L6-v2
```

The class can also be initialized with custom configuration:

```python
indexer = Node2Indexer(
    embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
    legal_terms=(
        "payment",
        "expenses",
        "termination",
        "representations and warranties",
        "indemnification",
        "confidentiality",
    ),
    stakeholder_roles=(
        "Company",
        "Bank",
        "Agent",
        "Customer",
        "Affiliate",
    ),
)
```

### Parameters

#### `embedding_model_name`

Name of the Sentence Transformer model used to create embeddings.

Default:

```python
"sentence-transformers/all-MiniLM-L6-v2"
```

#### `legal_terms`

Tuple of legal terms to detect during graph construction.

Default terms:

```python
(
    "payment",
    "expenses",
    "indemnification",
    "confidentiality",
    "representations and warranties",
    "termination",
)
```

#### `stakeholder_roles`

Tuple of stakeholder names/roles used by the current lightweight stakeholder detection mechanism.

Default:

```python
(
    "Company",
    "Bank",
    "Agent",
    "Customer",
    "Affiliate",
)
```

The current implementation intentionally keeps this lightweight and isolated so that a more advanced entity-resolution system can replace it later without changing the graph architecture.

---

# `Node2Indexer.build()`

## Function

```python
index_bundle = indexer.build(
    chunks
)
```

### Input

```python
chunks: list[dict]
```

The canonical chunk list returned by:

```python
prepare_chunks()
```

### Output

Returns:

```python
IndexBundle
```

The `IndexBundle` contains all derived Node 2 indexes and mappings.

---

# `IndexBundle`

The returned object contains:

```python
IndexBundle(
    chunks=...,
    embeddings=...,
    vector_index=...,
    bm25=...,
    tokenized_chunks=...,
    knowledge_graph=...,
    vector_id_to_chunk_id=...,
    bm25_id_to_chunk_id=...,
    embedding_model_name=...,
)
```

### Contents

| Field | Description |
|---|---|
| `chunks` | Canonical Node 2 chunks |
| `embeddings` | Normalized embedding matrix |
| `vector_index` | FAISS vector index |
| `bm25` | BM25 keyword index |
| `tokenized_chunks` | Tokenized text used by BM25 |
| `knowledge_graph` | NetworkX knowledge graph |
| `vector_id_to_chunk_id` | FAISS ID → chunk ID mapping |
| `bm25_id_to_chunk_id` | BM25 ID → chunk ID mapping |
| `embedding_model_name` | Embedding model identifier |

---

# Embedding Generation

For every chunk:

```python
chunk["text"]
```

is passed to the Sentence Transformer model.

The resulting embeddings are stored as a NumPy matrix.

For the current model:

```text
Embedding dimensions: 384
```

For the current 56-chunk test:

```text
Embeddings: (56, 384)
```

The embeddings are L2-normalized before being added to FAISS.

---

# FAISS Vector Store

FAISS is used for semantic similarity search.

The current implementation uses:

```python
faiss.IndexFlatIP(dimension)
```

Because the embeddings are normalized, inner product corresponds to cosine similarity.

For 56 chunks:

```text
FAISS vectors: 56
```

The FAISS ID corresponds to the position of the chunk in the canonical chunk list.

Example:

```text
FAISS ID 0
    ↓
document_a_0001
```

This relationship is maintained through:

```python
vector_id_to_chunk_id
```

Example:

```python
{
    0: "document_a_0001",
    1: "document_a_0002",
    ...
}
```

---

# BM25 Index

BM25 provides keyword-based retrieval.

Each chunk's text is converted into lowercase tokens:

```python
chunk["text"].lower().split()
```

These tokenized documents are passed to:

```python
BM25Okapi
```

For 56 chunks:

```text
BM25 documents: 56
```

The mapping between BM25 result IDs and canonical chunk IDs is maintained through:

```python
bm25_id_to_chunk_id
```

---

# NetworkX Knowledge Graph

The knowledge graph is implemented using:

```python
networkx.MultiDiGraph
```

The graph contains:

- Document nodes.
- Version nodes.
- Chunk nodes.
- Stakeholder nodes.
- Legal-term nodes.
- Date nodes.

---

## Document → Version Relationship

The graph contains:

```text
document
    |
    +---- has_version ----> version
```

Example:

```text
document_a
    |
    +---- has_version ----> document_a:version_a
```

---

## Version → Chunk Relationship

The graph contains:

```text
version
    |
    +---- contains ----> chunk
```

Example:

```text
document_a:version_a
    |
    +---- contains ----> document_a_0018
```

---

## Chunk → Stakeholder Relationship

The current graph uses:

```text
chunk
    |
    +---- mentions ----> stakeholder
```

Example:

```text
document_a_0018
    |
    +---- mentions ----> stakeholder:company
```

Stakeholders are represented as:

```python
{
    "node_type": "stakeholder",
    "name": "Company"
}
```

---

## Chunk → Legal Term Relationship

Legal terms use:

```text
chunk
    |
    +---- mentions ----> legal_term
```

Example:

```text
document_a_0018
    |
    +---- mentions ----> legal_term:expenses
```

Current default legal terms include:

```text
payment
expenses
termination
representations and warranties
indemnification
confidentiality
```

---

## Chunk → Date Relationship

Dates are extracted using deterministic regular expressions.

Supported patterns include:

```text
Month DD, YYYY
Mon. DD, YYYY
MM/DD/YYYY
MM-DD-YYYY
```

Example:

```text
document_a_0018
    |
    +---- mentions ----> date:april 6, 2007
```

The date extractor also normalizes whitespace so that dates split across PDF whitespace can be represented consistently.

---

# 3. `retrieval.py`

## Purpose

`retrieval.py` consumes the `IndexBundle` created by `Node2Indexer`.

It provides four retrieval modes:

```text
Semantic
Keyword
Graph
Hybrid
```

It does not build or modify the indexes.

---

# `semantic_search()`

## Function

```python
semantic_search(
    query,
    index_bundle,
    embedding_model,
    top_k=5,
)
```

### Inputs

#### `query`

Natural-language query.

Example:

```python
"What are the confidentiality obligations?"
```

#### `index_bundle`

The `IndexBundle` returned by:

```python
Node2Indexer.build()
```

#### `embedding_model`

The same Sentence Transformer model used to create the FAISS embeddings.

For example:

```python
indexer.embedding_model
```

#### `top_k`

Maximum number of results to return.

Default:

```python
5
```

### Output

Returns:

```python
list[dict]
```

Each result contains:

```python
{
    "chunk_id": "...",
    "document_id": "...",
    "version_id": "...",
    "page_start": ...,
    "page_end": ...,
    "heading_path": [...],
    "text": "...",
    "score": ...,
    "retrieval_method": "semantic"
}
```

### Example

```python
results = semantic_search(
    "What are the confidentiality obligations?",
    index_bundle,
    indexer.embedding_model,
    top_k=5,
)
```

Expected behavior for the current test documents:

```text
1. document_b_0018
2. document_a_0021
3. document_a_0027
4. document_a_0014
5. document_a_0024
```

The strongest result is:

```text
document_b_0018
```

with the current test score:

```text
0.6527
```

---

# `keyword_search()`

## Function

```python
keyword_search(
    query,
    index_bundle,
    top_k=5,
)
```

### Inputs

#### `query`

Natural-language keyword query.

#### `index_bundle`

Node 2 `IndexBundle`.

#### `top_k`

Maximum number of results.

Default:

```python
5
```

### Output

Returns a list of dictionaries using the same standard result structure.

The retrieval method is:

```python
"bm25"
```

### Example

```python
results = keyword_search(
    "What are the confidentiality obligations?",
    index_bundle,
    top_k=5,
)
```

Expected current top result:

```text
document_b_0018
```

Current test score:

```text
6.8907
```

---

# `get_chunks_for_entity()`

## Function

```python
get_chunks_for_entity(
    entity_node_id,
    index_bundle,
)
```

### Purpose

Returns the chunk IDs directly connected to a graph entity.

The current graph direction is:

```text
chunk
  |
  +---- mentions ----> entity
```

Therefore the function uses the entity's predecessors to find associated chunks.

### Inputs

#### `entity_node_id`

Full NetworkX entity node ID.

Examples:

```python
"stakeholder:company"
```

or:

```python
"legal_term:expenses"
```

#### `index_bundle`

Node 2 `IndexBundle`.

### Output

Returns:

```python
set[str]
```

containing matching chunk IDs.

Example:

```python
get_chunks_for_entity(
    "legal_term:confidentiality",
    index_bundle,
)
```

returns:

```python
{
    "document_b_0018"
}
```

---

# `graph_entity_search()`

## Function

```python
graph_entity_search(
    entity_name,
    entity_type,
    index_bundle,
    top_k=5,
)
```

### Inputs

#### `entity_name`

Entity name.

Example:

```text
Company
```

#### `entity_type`

Graph entity type.

Examples:

```text
stakeholder
legal_term
```

#### `index_bundle`

Node 2 `IndexBundle`.

#### `top_k`

Maximum number of results.

Default:

```python
5
```

### Output

Returns chunks connected to the specified graph entity.

Example:

```python
results = graph_entity_search(
    "Company",
    "stakeholder",
    index_bundle,
    top_k=5,
)
```

The result contains the canonical chunk metadata and:

```python
"retrieval_method": "graph"
```

---

# `graph_multi_entity_search()`

## Function

```python
graph_multi_entity_search(
    stakeholder_name,
    legal_term,
    index_bundle,
    top_k=10,
)
```

### Purpose

Finds chunks that are connected to **both** a stakeholder and a legal term.

It performs:

```text
Stakeholder chunks
        INTERSECTION
Legal-term chunks
        =
Matching chunks
```

### Inputs

#### `stakeholder_name`

Example:

```text
Company
```

#### `legal_term`

Example:

```text
expenses
```

#### `index_bundle`

Node 2 `IndexBundle`.

#### `top_k`

Maximum number of results.

Default:

```python
10
```

### Example

```python
results = graph_multi_entity_search(
    "Company",
    "expenses",
    index_bundle,
)
```

### Current Expected Result

The current test returns:

```text
Matching chunks: 6
```

```text
document_a_0004
document_a_0018
document_a_0019
document_a_0021
document_a_0023
document_a_0028
```

This confirms that the graph can combine multiple entity constraints.

---

# `hybrid_search()`

## Function

```python
hybrid_search(
    query,
    index_bundle,
    embedding_model,
    top_k=5,
    candidate_k=10,
    rrf_k=60,
)
```

### Purpose

Combines:

```text
Semantic retrieval
        +
BM25 keyword retrieval
        ↓
Reciprocal Rank Fusion
        ↓
Hybrid ranking
```

The implementation does **not** directly add FAISS scores to BM25 scores because they operate on different numerical scales.

Instead, it uses Reciprocal Rank Fusion:

```text
RRF contribution = 1 / (rrf_k + rank)
```

---

## Inputs

### `query`

Natural-language query.

Example:

```text
What expenses must the company reimburse?
```

### `index_bundle`

Node 2 `IndexBundle`.

### `embedding_model`

The Sentence Transformer model used by the FAISS index.

Example:

```python
indexer.embedding_model
```

### `top_k`

Number of final results.

Default:

```python
5
```

### `candidate_k`

Number of candidates retrieved from each individual retrieval method before fusion.

Default:

```python
10
```

### `rrf_k`

RRF smoothing constant.

Default:

```python
60
```

---

## Output

Returns a ranked list of chunks.

Each result contains:

```python
"retrieval_method": "hybrid"
```

and the final RRF score.

Example:

```python
results = hybrid_search(
    "What expenses must the company reimburse?",
    index_bundle,
    indexer.embedding_model,
    top_k=5,
)
```

Current test output:

```text
1. document_a_0019
2. document_a_0023
3. document_a_0021
4. document_a_0018
5. document_a_0004
```

---

# Complete Node 2 Initialization

The normal Node 2 initialization sequence is:

```python
from nodes.node_2 import (
    prepare_chunks,
    Node2Indexer,
)

# ------------------------------------------------------------
# 1. Convert Node 1 output
# ------------------------------------------------------------

all_chunks = prepare_chunks(
    result,
    version_ids={
        "document_a": "version_a",
        "document_b": "version_b",
    },
)

# ------------------------------------------------------------
# 2. Build Node 2
# ------------------------------------------------------------

indexer = Node2Indexer()

index_bundle = indexer.build(
    all_chunks
)
```

At this point Node 2 has constructed:

```text
Canonical chunks
        |
        +---- Embeddings
        |
        +---- FAISS
        |
        +---- BM25
        |
        +---- NetworkX graph
```

---

# Complete Retrieval Initialization

After building the index:

```python
from nodes.node_2 import (
    semantic_search,
    keyword_search,
    graph_entity_search,
    graph_multi_entity_search,
    hybrid_search,
)
```

Semantic retrieval:

```python
results = semantic_search(
    "What are the confidentiality obligations?",
    index_bundle,
    indexer.embedding_model,
    top_k=5,
)
```

Keyword retrieval:

```python
results = keyword_search(
    "What are the confidentiality obligations?",
    index_bundle,
    top_k=5,
)
```

Graph retrieval:

```python
results = graph_multi_entity_search(
    "Company",
    "expenses",
    index_bundle,
)
```

Hybrid retrieval:

```python
results = hybrid_search(
    "What expenses must the company reimburse?",
    index_bundle,
    indexer.embedding_model,
    top_k=5,
)
```

---

# Full Node 2 Test

The complete Node 2 test verifies:

1. All chunks from Node 1 are processed.
2. Embedding count matches the chunk count.
3. FAISS vector count matches the chunk count.
4. BM25 document count matches the chunk count.
5. FAISS mappings match the chunk count.
6. BM25 mappings match the chunk count.
7. Semantic retrieval works.
8. BM25 retrieval works.
9. Graph retrieval works.
10. Hybrid retrieval works.
11. Retrieved chunk IDs correspond to canonical chunks.
12. Both documents are represented in the indexes.

For the current two-document test:

```text
document_a → 33 chunks
document_b → 23 chunks
-----------------------
Total      → 56 chunks
```

Expected indexing output:

```text
Embeddings: (56, 384)
FAISS vectors: 56
BM25 documents: 56
Graph nodes: 78
Graph edges: approximately 207
```

The graph edge count can vary depending on the extracted relationships, while the chunk, embedding, FAISS and BM25 counts should remain aligned.

---

# Complete Node 2 Smoke Test

A complete test can be performed using:

```python
from nodes.node_2 import (
    prepare_chunks,
    Node2Indexer,
    semantic_search,
    keyword_search,
    graph_multi_entity_search,
    hybrid_search,
)


# ============================================================
# 1. PREPARE CHUNKS
# ============================================================

all_chunks = prepare_chunks(
    result,
    version_ids={
        "document_a": "version_a",
        "document_b": "version_b",
    },
)

print("Total chunks:", len(all_chunks))


# ============================================================
# 2. BUILD INDEX
# ============================================================

indexer = Node2Indexer()

index_bundle = indexer.build(
    all_chunks
)

print("Embeddings:", index_bundle.embeddings.shape)
print(
    "FAISS vectors:",
    index_bundle.vector_index.ntotal
)
print(
    "BM25 documents:",
    len(index_bundle.tokenized_chunks)
)
print(
    "Graph nodes:",
    index_bundle.knowledge_graph.number_of_nodes()
)
print(
    "Graph edges:",
    index_bundle.knowledge_graph.number_of_edges()
)


# ============================================================
# 3. SEMANTIC SEARCH
# ============================================================

semantic_results = semantic_search(
    "What are the confidentiality obligations?",
    index_bundle,
    indexer.embedding_model,
    top_k=5,
)


# ============================================================
# 4. BM25 SEARCH
# ============================================================

keyword_results = keyword_search(
    "What are the confidentiality obligations?",
    index_bundle,
    top_k=5,
)


# ============================================================
# 5. GRAPH SEARCH
# ============================================================

graph_results = graph_multi_entity_search(
    "Company",
    "expenses",
    index_bundle,
    top_k=10,
)


# ============================================================
# 6. HYBRID SEARCH
# ============================================================

hybrid_results = hybrid_search(
    "What expenses must the company reimburse?",
    index_bundle,
    indexer.embedding_model,
    top_k=5,
    candidate_k=10,
)


# ============================================================
# 7. VALIDATION
# ============================================================

assert len(all_chunks) == 56

assert (
    index_bundle.embeddings.shape[0]
    == len(all_chunks)
)

assert (
    index_bundle.vector_index.ntotal
    == len(all_chunks)
)

assert (
    len(index_bundle.tokenized_chunks)
    == len(all_chunks)
)

assert len(semantic_results) > 0
assert len(keyword_results) > 0
assert len(graph_results) > 0
assert len(hybrid_results) > 0

print()
print("=" * 70)
print("NODE 2 TEST PASSED")
print("=" * 70)
```

---

# Expected Full Test Output

The current two-document test produced:

```text
Index construction complete.
Embeddings: (56, 384)
FAISS vectors: 56
BM25 documents: 56
Graph nodes: 78
Graph edges: 207
```

Index alignment:

```text
Chunk count          : 56
Embedding count      : 56
FAISS vector count   : 56
BM25 document count  : 56
FAISS mapping count  : 56
BM25 mapping count   : 56

Index alignment: PASS
```

Semantic retrieval:

```text
1. document_b_0018 | document_b | score=0.6527
2. document_a_0021 | document_a | score=0.4777
3. document_a_0027 | document_a | score=0.4776
4. document_a_0014 | document_a | score=0.4480
5. document_a_0024 | document_a | score=0.4346
```

BM25 retrieval:

```text
1. document_b_0018 | document_b | score=6.8907
2. document_a_0012 | document_a | score=2.5823
3. document_a_0013 | document_a | score=2.5783
4. document_a_0020 | document_a | score=2.5146
5. document_a_0021 | document_a | score=2.4895
```

Graph retrieval:

```text
Stakeholder: Company
Legal term : expenses
Matching chunks: 6

document_a_0004
document_a_0018
document_a_0019
document_a_0021
document_a_0023
document_a_0028
```

Hybrid retrieval:

```text
1. document_a_0019 | document_a | RRF=0.032266
2. document_a_0023 | document_a | RRF=0.031754
3. document_a_0021 | document_a | RRF=0.031010
4. document_a_0018 | document_a | RRF=0.016393
5. document_a_0004 | document_a | RRF=0.016129
```

Final validation:

```text
======================================================================
NODE 2 FULL DOCUMENT TEST PASSED
======================================================================

Documents processed : 2
Total chunks        : 56
Embeddings          : (56, 384)
FAISS vectors       : 56
BM25 documents      : 56
Graph nodes         : 78
Graph edges         : 207

Retrieval tests:
  Semantic : PASS
  BM25     : PASS
  Graph    : PASS
  Hybrid   : PASS

======================================================================
ALL NODE 2 TESTS PASSED
======================================================================
```

---

# Node 2 Responsibilities

The overall responsibility of Node 2 can be summarized as:

```text
Structured Chunks
        |
        v
Chunk Validation
        |
        v
Embedding Generation
        |
        +-------------------+
        |                   |
        v                   v
      FAISS               BM25
        |                   |
        +---------+---------+
                  |
                  v
             NetworkX
             Knowledge
               Graph
                  |
                  v
             Node 2 Search
                  |
       +----------+----------+
       |          |          |
       v          v          v
    Semantic   Keyword     Graph
       |          |          |
       +----------+----------+
                  |
                  v
               Hybrid
                  |
                  v
                NODE 3
```

Node 2 therefore acts as the **search and indexing layer between document ingestion and document alignment**.

Node 1 is responsible for extracting and structurally chunking the documents. Node 2 transforms those chunks into searchable representations. Node 3 subsequently uses Node 2 to retrieve relevant information for document alignment. This matches the overall project workflow, where Node 3 pulls data from Node 2 and compares corresponding document chunks. :contentReference[oaicite:3]{index=3}

---
## Versioning Status

The current implementation carries:

```python
version_id
```

inside the canonical chunk representation and the knowledge graph.

However, **persistent version-specific storage and save/load functionality are not currently implemented in the Node 2 code**.

Therefore, the current implementation should be considered:

```text
Version-aware metadata
        +
In-memory indexes
```

rather than a fully persistent versioned vector database.

Persistent versioned storage can be added later without changing the fundamental retrieval interface.
