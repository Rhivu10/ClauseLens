# Node 2 Documentation

## Overview

**Node 2** is the indexing and retrieval layer of ClauseLens.

> **Notice:** Although this is considered a node, it is **not** part of the LangGraph architecture.

Node 2 receives the structured, hierarchy-aware chunks produced by Node 1 and creates searchable representations for downstream nodes.

Node 2 currently:

- Validates and normalizes Node 1 chunks.
- Maintains document and version metadata.
- Generates semantic embeddings.
- Builds a FAISS vector index.
- Builds a BM25 keyword index.
- Builds an in-memory NetworkX knowledge graph.
- Creates relationships between chunks, stakeholders, legal terms, and dates.
- Provides semantic retrieval.
- Provides keyword retrieval.
- Provides graph retrieval.
- Provides multi-entity graph retrieval.
- Provides hybrid semantic + keyword retrieval using Reciprocal Rank Fusion.
- Maintains mappings between search indexes and canonical chunk IDs.

Node 2 does **not** ingest PDFs. PDF extraction and hierarchy-aware chunking are handled by Node 1.

---

# Architecture

```text
                         NODE 1
              Structured / Hierarchy-Aware Chunks
                              |
                              v
                       +--------------+
                       |  chunks.py   |
                       |              |
                       | Validate     |
                       | Normalize    |
                       | Prepare      |
                       +------+-------+
                              |
                              v
                      Canonical Chunks
                              |
                              v
                       +--------------+
                       |  indexer.py  |
                       |              |
                       | Build Indexes|
                       +------+-------+
                              |
             +----------------+----------------+
             |                |                |
             v                v                v
        Embeddings          BM25           NetworkX
             |                |                |
             v                |                v
           FAISS              |        Knowledge Graph
             |                |                |
             +----------------+----------------+
                              |
                              v
                       +--------------+
                       | retrieval.py |
                       |              |
                       | Retrieval    |
                       +------+-------+
                              |
             +----------------+----------------+
             |                |                |
             v                v                v
         Semantic          Keyword          Graph
             |                |                |
             +----------------+----------------+
                              |
                              v
                           Hybrid
                              |
                              v
                           NODE 3
```

### Retrieval Responsibilities

| Component | Purpose |
|---|---|
| FAISS | Semantic similarity |
| BM25 | Keyword / lexical matching |
| NetworkX | Entity and relationship retrieval |
| Hybrid Search | Combines semantic and keyword retrieval |

---

# Directory Structure

```text
node_2/
├── __init__.py
├── chunks.py
├── indexer.py
└── retrieval.py
```

| File | Responsibility |
|---|---|
| `__init__.py` | Public Node 2 interface |
| `chunks.py` | Canonical chunk preparation |
| `indexer.py` | Embeddings, FAISS, BM25, NetworkX |
| `retrieval.py` | Search and retrieval functions |

---

# Public Interface

The Node 2 public interface is defined in:

```text
nodes/node_2/__init__.py
```

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

Other nodes should import Node 2 through this interface:

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
```

---

# Kaggle Setup

## 1. Clone the Repository

```bash
!git clone https://github.com/Rhivu10/ClauseLens.git /kaggle/working/ClauseLens
```

If the repository already exists:

```bash
%cd /kaggle/working/ClauseLens
!git pull
```

---

## 2. Add Repository to Python Path

```python
import sys

REPO_PATH = "/kaggle/working/ClauseLens"

if REPO_PATH not in sys.path:
    sys.path.insert(0, REPO_PATH)
```

Test the public interface:

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

Expected:

```text
Node 2 public API import: SUCCESS
```

---

# Dependencies

Node 2 uses:

```text
NumPy
Sentence Transformers
FAISS
rank-bm25
NetworkX
```

The current embedding model is:

```text
sentence-transformers/all-MiniLM-L6-v2
```

## Install Node 2 Dependencies

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

## Node 1 Dependency

PyMuPDF is required when running Node 1 together with Node 2.

It is a **Node 1 dependency**, not a Node 2 dependency.

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
```

---

# 1. `chunks.py`

## Purpose

`chunks.py` converts Node 1's structured document output into the canonical chunk representation used by Node 2.

Responsibilities:

- Validate required fields.
- Validate chunk text.
- Reject empty text.
- Validate unique chunk IDs.
- Assign document IDs.
- Assign version IDs.
- Normalize metadata.

It does not:

- Generate embeddings.
- Build FAISS.
- Build BM25.
- Build the knowledge graph.
- Perform retrieval.

---

# `prepare_chunks()`

## Function

```python
prepare_chunks(
    documents,
    version_ids=None
)
```

## Input

### `documents`

Dictionary containing the structured output of Node 1.

Expected structure:

```python
{
    "document_id": {
        "document_id": "...",
        "source_path": "...",
        "chunks": [
            {
                "chunk_id": "...",
                "text": "...",
                "page_start": ...,
                "page_end": ...,
                "heading_path": [...]
            }
        ]
    }
}
```

The number of documents and chunks is determined by the supplied input.

### `version_ids`

Optional dictionary mapping document IDs to version IDs.

```python
{
    "document_a": "version_a",
    "document_b": "version_b",
}
```

If omitted:

```python
version_ids=None
```

the resulting chunks contain:

```python
"version_id": None
```

---

## Output

Returns:

```python
list[dict]
```

Each chunk has the canonical Node 2 structure:

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

---

## Example

```python
all_chunks = prepare_chunks(
    result,
    version_ids={
        "document_a": "version_a",
        "document_b": "version_b",
    },
)
```

The resulting list contains all chunks supplied by Node 1.

---

## Validation

The function validates:

```text
chunk_id
text
page_start
page_end
heading_path
```

It also checks:

- Text is a string.
- Text is not empty.
- Chunk IDs are unique.

Invalid input raises an appropriate `ValueError` or `TypeError`.

---

# Canonical Chunk

The canonical chunk is the shared reference used by every Node 2 index.

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

The same chunk is represented independently in:

```text
Canonical Chunk
      |
      +---- FAISS
      |
      +---- BM25
      |
      +---- NetworkX
```

All retrieval systems must map their results back to the canonical `chunk_id`.

---

# 2. `indexer.py`

## Purpose

`indexer.py` builds the searchable representations of the canonical chunks.

```text
Canonical Chunks
       |
       +---- Embeddings ----> FAISS
       |
       +---- Tokens --------> BM25
       |
       +---- Entities ------> NetworkX
```

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

---

## Custom Initialization

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

### `embedding_model_name`

Sentence Transformer model used to generate embeddings.

Default:

```python
"sentence-transformers/all-MiniLM-L6-v2"
```

### `legal_terms`

Terms used by the current lightweight graph extraction process.

### `stakeholder_roles`

Stakeholder names or roles used by the current lightweight graph extraction process.

These lists are configuration for the current implementation and are not a universal legal ontology.

---

# `Node2Indexer.build()`

## Function

```python
index_bundle = indexer.build(
    chunks
)
```

## Input

```python
chunks: list[dict]
```

Canonical chunks returned by:

```python
prepare_chunks()
```

## Output

Returns:

```python
IndexBundle
```

containing all Node 2 indexes and mappings.

---

# `IndexBundle`

`IndexBundle` stores the complete in-memory state required by Node 2 retrieval.

Fields:

| Field | Description |
|---|---|
| `chunks` | Canonical chunks |
| `embeddings` | Normalized embedding matrix |
| `vector_index` | FAISS vector index |
| `bm25` | BM25 index |
| `tokenized_chunks` | BM25 tokenized documents |
| `knowledge_graph` | NetworkX knowledge graph |
| `vector_id_to_chunk_id` | FAISS ID → chunk ID |
| `bm25_id_to_chunk_id` | BM25 ID → chunk ID |
| `embedding_model_name` | Embedding model identifier |

---

# Embedding Generation

Each chunk's:

```python
chunk["text"]
```

is passed to the Sentence Transformer.

The current model:

```text
sentence-transformers/all-MiniLM-L6-v2
```

produces:

```text
384-dimensional embeddings
```

Therefore:

```text
Embedding shape = (number_of_chunks, 384)
```

The embeddings are normalized before being added to FAISS.

---

# FAISS Vector Index

FAISS provides semantic similarity retrieval.

The current implementation uses:

```python
faiss.IndexFlatIP(dimension)
```

Normalized embeddings allow inner product to represent cosine similarity.

The required relationship is:

```text
FAISS vector count
        =
canonical chunk count
```

FAISS IDs are mapped to canonical chunk IDs through:

```python
vector_id_to_chunk_id
```

---

# BM25 Index

BM25 provides lexical keyword retrieval.

Chunk text is tokenized using lowercase whitespace splitting:

```python
chunk["text"].lower().split()
```

The resulting token lists are indexed using:

```python
BM25Okapi
```

The required relationship is:

```text
BM25 document count
        =
canonical chunk count
```

BM25 IDs are mapped to canonical chunk IDs through:

```python
bm25_id_to_chunk_id
```

---

# NetworkX Knowledge Graph

The current graph uses:

```python
networkx.MultiDiGraph
```

The graph represents relationships between:

```text
Documents
Versions
Chunks
Stakeholders
Legal Terms
Dates
```

---

# Document and Version Relationships

General structure:

```text
Document
    |
    +---- has_version ----> Version
                              |
                              +---- contains ----> Chunk
```

Version information is retained through:

```python
version_id
```

---

# Stakeholder Relationships

Current graph structure:

```text
Chunk
  |
  +---- mentions ----> Stakeholder
```

Stakeholder nodes contain:

```python
{
    "node_type": "stakeholder",
    "name": "..."
}
```

The stakeholder extraction mechanism is intentionally lightweight.

---

# Legal-Term Relationships

Current graph structure:

```text
Chunk
  |
  +---- mentions ----> Legal Term
```

Legal-term nodes contain:

```python
{
    "node_type": "legal_term",
    "name": "..."
}
```

The available legal terms depend on the configured term list.

---

# Date Relationships

Dates are extracted using deterministic regular expressions.

Supported formats include:

```text
Month DD, YYYY
Mon. DD, YYYY
MM/DD/YYYY
MM-DD-YYYY
```

The extracted date is represented as a graph node:

```text
Chunk
  |
  +---- mentions ----> Date
```

Whitespace normalization is applied to handle PDF extraction artifacts.

---

# 3. `retrieval.py`

## Purpose

`retrieval.py` provides the Node 2 search interface.

It consumes:

```python
IndexBundle
```

and provides:

```text
Semantic Search
Keyword Search
Graph Search
Multi-Entity Graph Search
Hybrid Search
```

Every retrieval method maps its results back to canonical chunks.

---

# Standard Retrieval Result

A retrieval result contains canonical chunk information such as:

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
    "retrieval_method": "..."
}
```

The exact score depends on the retrieval method.

The primary identifier for downstream processing is:

```python
"chunk_id"
```

---

# `semantic_search()`

## Purpose

Finds chunks with similar semantic meaning.

```text
Query
  |
  v
Sentence Transformer
  |
  v
Query Embedding
  |
  v
FAISS
  |
  v
Top-k Chunks
```

## Function

```python
semantic_search(
    query,
    index_bundle,
    embedding_model,
    top_k=5,
)
```

## Inputs

### `query`

Natural-language query.

```python
"What are the confidentiality obligations?"
```

### `index_bundle`

Node 2 `IndexBundle`.

### `embedding_model`

Sentence Transformer used by the FAISS index.

```python
indexer.embedding_model
```

### `top_k`

Maximum number of results.

Default:

```python
5
```

## Output

```python
list[dict]
```

Each result contains canonical chunk metadata and:

```python
"retrieval_method": "semantic"
```

---

## Example

```python
results = semantic_search(
    "What are the confidentiality obligations?",
    index_bundle,
    indexer.embedding_model,
    top_k=5,
)
```

The returned chunks depend on the indexed documents.

---

# `keyword_search()`

## Purpose

Performs lexical retrieval using BM25.

```text
Query
  |
  v
Tokenization
  |
  v
BM25
  |
  v
Top-k Chunks
```

## Function

```python
keyword_search(
    query,
    index_bundle,
    top_k=5,
)
```

## Inputs

### `query`

Natural-language or keyword query.

### `index_bundle`

Node 2 `IndexBundle`.

### `top_k`

Maximum number of results.

Default:

```python
5
```

## Output

```python
list[dict]
```

Each result contains:

```python
"retrieval_method": "bm25"
```

The BM25 score depends on the indexed corpus.

---

## Example

```python
results = keyword_search(
    "confidentiality obligations",
    index_bundle,
    top_k=5,
)
```

---

# `get_chunks_for_entity()`

## Purpose

Retrieves canonical chunks connected to a graph entity.

Graph direction:

```text
Chunk
  |
  +---- mentions ----> Entity
```

Therefore, chunks are obtained from the entity's predecessors.

## Function

```python
get_chunks_for_entity(
    entity_node_id,
    index_bundle,
)
```

## Inputs

### `entity_node_id`

Full graph node ID.

Examples:

```python
"stakeholder:company"
```

```python
"legal_term:confidentiality"
```

### `index_bundle`

Node 2 `IndexBundle`.

## Output

```python
set[str]
```

containing canonical chunk IDs.

---

# `graph_entity_search()`

## Purpose

Retrieves chunks associated with a single graph entity.

## Function

```python
graph_entity_search(
    entity_name,
    entity_type,
    index_bundle,
    top_k=5,
)
```

## Inputs

### `entity_name`

Entity name.

### `entity_type`

Graph entity type.

Supported types include:

```text
stakeholder
legal_term
```

### `index_bundle`

Node 2 `IndexBundle`.

### `top_k`

Maximum number of results.

Default:

```python
5
```

## Output

Returns canonical chunk results with:

```python
"retrieval_method": "graph"
```

---

## Example

```python
results = graph_entity_search(
    "Company",
    "stakeholder",
    index_bundle,
    top_k=5,
)
```

The entity must exist in the graph.

---

# `graph_multi_entity_search()`

## Purpose

Finds chunks satisfying multiple graph constraints.

The current implementation intersects:

```text
Stakeholder chunks
        INTERSECTION
Legal-term chunks
        =
Matching chunks
```

## Function

```python
graph_multi_entity_search(
    stakeholder_name,
    legal_term,
    index_bundle,
    top_k=10,
)
```

## Inputs

### `stakeholder_name`

Stakeholder entity.

### `legal_term`

Legal-term entity.

### `index_bundle`

Node 2 `IndexBundle`.

### `top_k`

Maximum number of results.

Default:

```python
10
```

## Output

Returns canonical chunks matching both graph constraints.

The result count depends on the indexed documents.

---

## Example

```python
results = graph_multi_entity_search(
    "Company",
    "expenses",
    index_bundle,
    top_k=10,
)
```

---

# `hybrid_search()`

## Purpose

Combines semantic retrieval and BM25 retrieval.

```text
                    Query
                      |
             +--------+--------+
             |                 |
             v                 v
        Semantic             BM25
         Search             Search
             |                 |
             +--------+--------+
                      |
                      v
            Reciprocal Rank
                 Fusion
                      |
                      v
               Hybrid Ranking
```

FAISS and BM25 scores are not directly added because they use different scoring scales.

Reciprocal Rank Fusion is used instead:

```text
RRF contribution = 1 / (rrf_k + rank)
```

---

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

## Inputs

### `query`

Natural-language query.

### `index_bundle`

Node 2 `IndexBundle`.

### `embedding_model`

Sentence Transformer used by FAISS.

### `top_k`

Number of final results.

Default:

```python
5
```

### `candidate_k`

Number of candidates retrieved from each underlying search.

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

## Output

Returns ranked canonical chunks containing:

```python
"retrieval_method": "hybrid"
```

and the calculated RRF score.

---

# Complete Node 2 Initialization

```python
from nodes.node_2 import (
    prepare_chunks,
    Node2Indexer,
)


# ============================================================
# 1. PREPARE CANONICAL CHUNKS
# ============================================================

all_chunks = prepare_chunks(
    result,
    version_ids={
        document_id: f"{document_id}_version"
        for document_id in result
    },
)


# ============================================================
# 2. BUILD NODE 2
# ============================================================

indexer = Node2Indexer()

index_bundle = indexer.build(
    all_chunks
)
```

After this:

```text
Canonical Chunks
      |
      +---- Embeddings
      |        |
      |        +---- FAISS
      |
      +---- Tokens
      |        |
      |        +---- BM25
      |
      +---- Entities
               |
               +---- NetworkX
```

Node 2 is ready for retrieval.

---

# Retrieval Usage

## Semantic

```python
results = semantic_search(
    "What are the confidentiality obligations?",
    index_bundle,
    indexer.embedding_model,
    top_k=5,
)
```

## Keyword

```python
results = keyword_search(
    "What are the confidentiality obligations?",
    index_bundle,
    top_k=5,
)
```

## Graph Entity

```python
results = graph_entity_search(
    "Company",
    "stakeholder",
    index_bundle,
    top_k=5,
)
```

## Multi-Entity Graph

```python
results = graph_multi_entity_search(
    "Company",
    "expenses",
    index_bundle,
    top_k=10,
)
```

## Hybrid

```python
results = hybrid_search(
    "What obligations are imposed by the agreement?",
    index_bundle,
    indexer.embedding_model,
    top_k=5,
)
```

---

# Full Node 2 Test

The complete test is dataset-independent.

It verifies the integrity of the indexing and retrieval pipeline rather than expecting specific documents, scores, or entity names.

## Test Requirements

```text
[PASS] Chunks prepared
[PASS] Index construction succeeds
[PASS] Embedding count matches chunk count
[PASS] FAISS vector count matches chunk count
[PASS] BM25 document count matches chunk count
[PASS] FAISS mapping count matches chunk count
[PASS] BM25 mapping count matches chunk count
[PASS] FAISS mappings resolve to canonical chunks
[PASS] BM25 mappings resolve to canonical chunks
[PASS] Semantic search returns canonical chunks
[PASS] BM25 search returns canonical chunks
[PASS] Graph construction succeeds
[PASS] Graph search works when compatible entities exist
[PASS] Hybrid search returns canonical chunks
[PASS] Document coverage is preserved
```

---

# Dataset-Independent Test

```python
from nodes.node_2 import (
    prepare_chunks,
    Node2Indexer,
    semantic_search,
    keyword_search,
    graph_multi_entity_search,
    hybrid_search,
)


print("=" * 70)
print("NODE 2 END-TO-END TEST")
print("=" * 70)


# ============================================================
# 1. CHUNK PREPARATION
# ============================================================

print()
print("[1] CHUNK PREPARATION")
print("-" * 70)

all_chunks = prepare_chunks(
    result,
    version_ids={
        document_id: f"{document_id}_version"
        for document_id in result
    },
)

assert len(all_chunks) > 0

chunk_ids = {
    chunk["chunk_id"]
    for chunk in all_chunks
}

assert len(chunk_ids) == len(all_chunks)

print("Total chunks:", len(all_chunks))
print("Chunk preparation: PASS")


# ============================================================
# 2. INDEX CONSTRUCTION
# ============================================================

print()
print("[2] INDEX CONSTRUCTION")
print("-" * 70)

indexer = Node2Indexer()

index_bundle = indexer.build(
    all_chunks
)

print(
    "Embeddings:",
    index_bundle.embeddings.shape
)

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

print("Index construction: PASS")


# ============================================================
# 3. INDEX ALIGNMENT
# ============================================================

print()
print("[3] VERIFYING INDEX ALIGNMENT")
print("-" * 70)

chunk_count = len(all_chunks)

embedding_count = (
    index_bundle.embeddings.shape[0]
)

faiss_count = (
    index_bundle.vector_index.ntotal
)

bm25_count = (
    len(index_bundle.tokenized_chunks)
)

faiss_mapping_count = (
    len(index_bundle.vector_id_to_chunk_id)
)

bm25_mapping_count = (
    len(index_bundle.bm25_id_to_chunk_id)
)

print("Chunk count          :", chunk_count)
print("Embedding count      :", embedding_count)
print("FAISS vector count   :", faiss_count)
print("BM25 document count  :", bm25_count)
print("FAISS mapping count  :", faiss_mapping_count)
print("BM25 mapping count   :", bm25_mapping_count)

assert embedding_count == chunk_count
assert faiss_count == chunk_count
assert bm25_count == chunk_count
assert faiss_mapping_count == chunk_count
assert bm25_mapping_count == chunk_count

print()
print("Index alignment: PASS")


# ============================================================
# 4. FAISS MAPPINGS
# ============================================================

print()
print("[4] VERIFYING FAISS MAPPINGS")
print("-" * 70)

for faiss_id, chunk_id in (
    index_bundle.vector_id_to_chunk_id.items()
):

    assert chunk_id in chunk_ids

print("FAISS mappings: PASS")


# ============================================================
# 5. BM25 MAPPINGS
# ============================================================

print()
print("[5] VERIFYING BM25 MAPPINGS")
print("-" * 70)

for bm25_id, chunk_id in (
    index_bundle.bm25_id_to_chunk_id.items()
):

    assert chunk_id in chunk_ids

print("BM25 mappings: PASS")


# ============================================================
# 6. SEMANTIC SEARCH
# ============================================================

print()
print("[6] SEMANTIC SEARCH")
print("-" * 70)

semantic_query = (
    "What are the confidentiality obligations?"
)

semantic_results = semantic_search(
    semantic_query,
    index_bundle,
    indexer.embedding_model,
    top_k=5,
)

print("Query:", semantic_query)
print(
    "Results returned:",
    len(semantic_results)
)

assert len(semantic_results) > 0

for item in semantic_results:

    assert (
        item["chunk_id"]
        in chunk_ids
    )

print("Semantic search: PASS")


# ============================================================
# 7. BM25 SEARCH
# ============================================================

print()
print("[7] BM25 KEYWORD SEARCH")
print("-" * 70)

keyword_query = (
    "What are the confidentiality obligations?"
)

keyword_results = keyword_search(
    keyword_query,
    index_bundle,
    top_k=5,
)

print("Query:", keyword_query)
print(
    "Results returned:",
    len(keyword_results)
)

assert len(keyword_results) > 0

for item in keyword_results:

    assert (
        item["chunk_id"]
        in chunk_ids
    )

print("BM25 search: PASS")


# ============================================================
# 8. GRAPH SEARCH
# ============================================================

print()
print("[8] GRAPH SEARCH")
print("-" * 70)

graph = index_bundle.knowledge_graph

stakeholder_nodes = [
    node
    for node, data in graph.nodes(data=True)
    if data.get("node_type") == "stakeholder"
]

legal_term_nodes = [
    node
    for node, data in graph.nodes(data=True)
    if data.get("node_type") == "legal_term"
]

print(
    "Stakeholder nodes:",
    len(stakeholder_nodes)
)

print(
    "Legal-term nodes:",
    len(legal_term_nodes)
)


if stakeholder_nodes and legal_term_nodes:

    stakeholder_node = stakeholder_nodes[0]
    legal_term_node = legal_term_nodes[0]

    stakeholder_name = graph.nodes[
        stakeholder_node
    ].get("name")

    legal_term_name = graph.nodes[
        legal_term_node
    ].get("name")

    print(
        "Testing:",
        stakeholder_name,
        "+",
        legal_term_name
    )

    graph_results = graph_multi_entity_search(
        stakeholder_name,
        legal_term_name,
        index_bundle,
        top_k=10,
    )

    for item in graph_results:

        assert (
            item["chunk_id"]
            in chunk_ids
        )

    print(
        "Graph results:",
        len(graph_results)
    )

    print("Graph search: PASS")

else:

    print(
        "No compatible stakeholder/legal-term "
        "pair found."
    )

    print("Graph construction: PASS")


# ============================================================
# 9. HYBRID SEARCH
# ============================================================

print()
print("[9] HYBRID SEARCH")
print("-" * 70)

hybrid_query = (
    "What obligations are imposed by the agreement?"
)

hybrid_results = hybrid_search(
    hybrid_query,
    index_bundle,
    indexer.embedding_model,
    top_k=5,
    candidate_k=10,
)

print("Query:", hybrid_query)
print(
    "Results returned:",
    len(hybrid_results)
)

assert len(hybrid_results) > 0

for item in hybrid_results:

    assert (
        item["chunk_id"]
        in chunk_ids
    )

print("Hybrid search: PASS")


# ============================================================
# 10. DOCUMENT COVERAGE
# ============================================================

print()
print("[10] VERIFYING DOCUMENT COVERAGE")
print("-" * 70)

input_documents = set(
    result.keys()
)

indexed_documents = {
    chunk["document_id"]
    for chunk in all_chunks
}

print(
    "Input documents:",
    sorted(input_documents)
)

print(
    "Indexed documents:",
    sorted(indexed_documents)
)

assert (
    input_documents
    == indexed_documents
)

print("Document coverage: PASS")


# ============================================================
# FINAL RESULT
# ============================================================

print()
print("=" * 70)
print("NODE 2 END-TO-END TEST PASSED")
print("=" * 70)

print()
print(
    "Documents processed :",
    len(input_documents)
)

print(
    "Total chunks        :",
    chunk_count
)

print(
    "Embeddings          :",
    index_bundle.embeddings.shape
)

print(
    "FAISS vectors       :",
    index_bundle.vector_index.ntotal
)

print(
    "BM25 documents      :",
    len(index_bundle.tokenized_chunks)
)

print(
    "Graph nodes         :",
    graph.number_of_nodes()
)

print(
    "Graph edges         :",
    graph.number_of_edges()
)

print()
print("Retrieval tests:")
print("  Semantic : PASS")
print("  BM25     : PASS")
print("  Graph    : PASS")
print("  Hybrid   : PASS")

print()
print("=" * 70)
print("ALL NODE 2 TESTS PASSED")
print("=" * 70)
```

---

# Expected Test Behavior

The exact values produced by the test depend on the documents supplied.

The following values are **not fixed**:

```text
Number of documents
Number of chunks
Embedding matrix size
FAISS vector count
BM25 document count
Graph node count
Graph edge count
Retrieval scores
Retrieved chunk IDs
Graph entity names
```

The following invariant must always hold:

```text
Canonical chunk count
        =
Embedding count
        =
FAISS vector count
        =
BM25 document count
        =
FAISS mapping count
        =
BM25 mapping count
```

If:

```text
Total chunks = N
```

then:

```text
Embedding shape = (N, 384)
```

because the current embedding model produces 384-dimensional vectors.

---

# Retrieval Expectations

Retrieval tests verify structural correctness rather than specific ranking.

A successful semantic search means:

```text
Query
  |
  v
FAISS
  |
  v
Results
  |
  v
Canonical chunk IDs
```

A successful BM25 search means:

```text
Query
  |
  v
BM25
  |
  v
Results
  |
  v
Canonical chunk IDs
```

A successful hybrid search means:

```text
Semantic Results
       +
BM25 Results
       |
       v
RRF
       |
       v
Ranked Canonical Chunks
```

The test does not require a particular chunk to appear at a particular rank.

Retrieval scores are corpus-dependent and should not be treated as fixed expected values.

---

# Graph Expectations

The graph is also dataset-dependent.

Different documents may contain different:

- Stakeholders.
- Legal terms.
- Dates.
- Relationships.

The test therefore checks whether compatible graph entities exist before attempting a multi-entity query.

If compatible entities exist:

```text
Stakeholder
      +
Legal Term
      |
      v
Graph Search
      |
      v
Canonical Chunks
```

If no compatible pair exists, graph construction is still considered successful as long as the graph itself was constructed correctly.

The absence of a specific stakeholder or legal term is not a Node 2 failure.

---

# Index Alignment

Index alignment is a core Node 2 requirement.

Each search system has its own internal identifier:

```text
FAISS ID
BM25 ID
Graph Node ID
```

These identifiers must ultimately resolve to the canonical chunk:

```text
                 Canonical Chunk
                       |
          +------------+------------+
          |            |            |
          v            v            v
      FAISS ID      BM25 ID     Graph Node
          |            |            |
          +------------+------------+
                       |
                       v
                  chunk_id
```

This ensures that retrieval results retain:

- Original text.
- Document identity.
- Version identity.
- Page information.
- Heading hierarchy.

---

# Node 2 Output

Node 2 does not produce the final document comparison.

Its output is the searchable representation required by downstream nodes:

```text
Node 1
  |
  v
Structured Chunks
  |
  v
Node 2
  |
  +---- FAISS
  +---- BM25
  +---- NetworkX
  |
  v
Retrieval Results
  |
  v
Node 3
```

Node 3 uses these results to identify corresponding clauses between documents and determine whether they are equivalent, modified, added, or deleted.

---

# Versioning Status

The current implementation supports:

```python
version_id
```

as metadata attached to canonical chunks and represented in the knowledge graph.

The current implementation uses in-memory indexes.

Persistent version-specific index storage and save/load functionality are **not currently implemented**.

Therefore the current status is:

```text
Version-aware metadata
        +
In-memory indexes
```

rather than:

```text
Persistent versioned vector database
```

Persistent storage can be added later without changing the fundamental retrieval interface.

---

# Node 2 Responsibilities

```text
Node 1
  |
  | Structured Chunks
  v
Node 2
  |
  +-- Validate / Normalize
  |
  +-- Generate Embeddings
  |
  +-- Build FAISS
  |
  +-- Build BM25
  |
  +-- Build Knowledge Graph
  |
  +-- Semantic Retrieval
  |
  +-- Keyword Retrieval
  |
  +-- Graph Retrieval
  |
  +-- Hybrid Retrieval
  |
  v
Node 3
```

### Summary

| Node | Responsibility |
|---|---|
| Node 1 | PDF ingestion and hierarchy-aware chunking |
| **Node 2** | Indexing and retrieval |
| Node 3 | Clause alignment and change detection |
| Node 4 | Deontic/legal-action extraction |
| Node 5 | Impact analysis |
| Node 6 | Final reporting |

Node 2 therefore serves as the **retrieval and indexing layer between structured document ingestion and clause-level comparison**.
