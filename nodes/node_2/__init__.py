"""
ClauseLens — Node 2

Indexer, versioned vector store, knowledge graph,
and hybrid retrieval interface.
"""

# ============================================================
# STRUCTURED CHUNKS
# ============================================================

from .chunks import (
    prepare_chunks,
)


# ============================================================
# INDEXING
# ============================================================

from .indexer import (
    Node2Indexer,
    IndexBundle,
)


# ============================================================
# RETRIEVAL
# ============================================================

from .retrieval import (
    semantic_search,
    keyword_search,
    graph_entity_search,
    graph_multi_entity_search,
    hybrid_search,
)


# ============================================================
# PUBLIC API
# ============================================================

__all__ = [
    # Chunk preparation
    "prepare_chunks",

    # Indexing
    "Node2Indexer",
    "IndexBundle",

    # Retrieval
    "semantic_search",
    "keyword_search",
    "graph_entity_search",
    "graph_multi_entity_search",
    "hybrid_search",
]