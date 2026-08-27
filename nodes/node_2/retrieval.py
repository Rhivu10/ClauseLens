"""
ClauseLens — Node 2
Retrieval layer.

This module consumes the IndexBundle produced by indexer.py.

Supported retrieval methods:

    1. Semantic retrieval using FAISS
    2. Keyword retrieval using BM25
    3. Graph entity retrieval using NetworkX
    4. Hybrid retrieval using Reciprocal Rank Fusion (RRF)

This module does not build or modify indexes.
"""

from __future__ import annotations

from typing import Any

import faiss
import numpy as np

from .indexer import IndexBundle


# ============================================================
# INTERNAL HELPERS
# ============================================================

def _get_chunk(
    index_bundle: IndexBundle,
    chunk_id: str,
) -> dict[str, Any]:
    """
    Retrieve a canonical chunk using its chunk ID.
    """

    for chunk in index_bundle.chunks:

        if chunk["chunk_id"] == chunk_id:
            return chunk

    raise KeyError(
        f"Chunk not found: {chunk_id}"
    )


def _format_result(
    index_bundle: IndexBundle,
    chunk_id: str,
    score: float,
    retrieval_method: str,
) -> dict[str, Any]:
    """
    Convert a chunk ID into the standard Node 2
    retrieval result format.
    """

    chunk = _get_chunk(
        index_bundle,
        chunk_id,
    )

    return {
        "chunk_id": chunk["chunk_id"],
        "document_id": chunk["document_id"],
        "version_id": chunk["version_id"],
        "page_start": chunk["page_start"],
        "page_end": chunk["page_end"],
        "heading_path": chunk["heading_path"],
        "text": chunk["text"],
        "score": float(score),
        "retrieval_method": retrieval_method,
    }


# ============================================================
# SEMANTIC RETRIEVAL
# ============================================================

def semantic_search(
    query: str,
    index_bundle: IndexBundle,
    embedding_model,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """
    Retrieve semantically similar chunks using FAISS.

    Parameters
    ----------
    query:
        Natural-language search query.

    index_bundle:
        Node 2 IndexBundle produced by Node2Indexer.

    embedding_model:
        The same SentenceTransformer model used to
        create the FAISS embeddings.

    top_k:
        Maximum number of results to return.

    Returns
    -------
    list[dict[str, Any]]
        Ranked semantic retrieval results.

    Notes
    -----
    The query embedding is L2-normalized before searching.
    Since the stored chunk embeddings are also normalized
    and the FAISS index uses inner product, the resulting
    score is cosine similarity.
    """

    if not query or not query.strip():
        return []

    total_vectors = (
        index_bundle.vector_index.ntotal
    )

    if total_vectors == 0:
        return []

    top_k = min(
        max(1, top_k),
        total_vectors,
    )

    query_embedding = (
        embedding_model.encode(
            [query],
            convert_to_numpy=True,
        )
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype=np.float32,
    )

    faiss.normalize_L2(
        query_embedding
    )

    scores, indices = (
        index_bundle.vector_index.search(
            query_embedding,
            top_k,
        )
    )

    results = []

    for score, vector_id in zip(
        scores[0],
        indices[0],
    ):

        if vector_id < 0:
            continue

        chunk_id = (
            index_bundle.vector_id_to_chunk_id[
                int(vector_id)
            ]
        )

        results.append(
            _format_result(
                index_bundle,
                chunk_id,
                score,
                "semantic",
            )
        )

    return results


# ============================================================
# BM25 KEYWORD RETRIEVAL
# ============================================================

def keyword_search(
    query: str,
    index_bundle: IndexBundle,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """
    Retrieve chunks using BM25 keyword matching.

    Parameters
    ----------
    query:
        Natural-language search query.

    index_bundle:
        Node 2 IndexBundle produced by Node2Indexer.

    top_k:
        Maximum number of results to return.

    Returns
    -------
    list[dict[str, Any]]
        Ranked BM25 retrieval results.
    """

    if not query or not query.strip():
        return []

    total_documents = len(
        index_bundle.tokenized_chunks
    )

    if total_documents == 0:
        return []

    top_k = min(
        max(1, top_k),
        total_documents,
    )

    query_tokens = (
        query.lower().split()
    )

    scores = (
        index_bundle.bm25.get_scores(
            query_tokens
        )
    )

    top_indices = np.argsort(
        scores
    )[::-1][:top_k]

    results = []

    for bm25_id in top_indices:

        chunk_id = (
            index_bundle.bm25_id_to_chunk_id[
                int(bm25_id)
            ]
        )

        results.append(
            _format_result(
                index_bundle,
                chunk_id,
                scores[bm25_id],
                "bm25",
            )
        )

    return results


# ============================================================
# GRAPH HELPERS
# ============================================================

def get_chunks_for_entity(
    entity_node_id: str,
    index_bundle: IndexBundle,
) -> set[str]:
    """
    Return chunks directly connected to an entity.

    Current graph direction:

        chunk
          |
          +---- mentions ----> entity

    Therefore, entity predecessors are the associated
    chunk nodes.
    """

    graph = (
        index_bundle.knowledge_graph
    )

    if entity_node_id not in graph:
        return set()

    return {
        node
        for node in graph.predecessors(
            entity_node_id
        )
        if (
            graph.nodes[node].get(
                "node_type"
            )
            == "chunk"
        )
    }


# ============================================================
# SINGLE ENTITY GRAPH RETRIEVAL
# ============================================================

def graph_entity_search(
    entity_name: str,
    entity_type: str,
    index_bundle: IndexBundle,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """
    Retrieve chunks connected to a single graph entity.

    Example
    -------
    graph_entity_search(
        "Company",
        "stakeholder",
        index_bundle,
    )
    """

    if not entity_name or not entity_name.strip():
        return []

    if not entity_type or not entity_type.strip():
        return []

    entity_node_id = (
        f"{entity_type}:"
        f"{entity_name.lower()}"
    )

    chunk_ids = get_chunks_for_entity(
        entity_node_id,
        index_bundle,
    )

    results = []

    for chunk_id in sorted(chunk_ids):

        results.append(
            _format_result(
                index_bundle,
                chunk_id,
                1.0,
                "graph",
            )
        )

    return results[:max(1, top_k)]


# ============================================================
# MULTI-ENTITY GRAPH RETRIEVAL
# ============================================================

def graph_multi_entity_search(
    stakeholder_name: str,
    legal_term: str,
    index_bundle: IndexBundle,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """
    Retrieve chunks connected to both a stakeholder
    and a legal term.

    Example
    -------
    graph_multi_entity_search(
        "Company",
        "expenses",
        index_bundle,
    )

    This performs:

        Company chunks
              INTERSECTION
        Expenses chunks
              =
        matching chunks
    """

    if not stakeholder_name.strip():
        return []

    if not legal_term.strip():
        return []

    stakeholder_node = (
        f"stakeholder:"
        f"{stakeholder_name.lower()}"
    )

    legal_term_node = (
        f"legal_term:"
        f"{legal_term.lower()}"
    )

    stakeholder_chunks = (
        get_chunks_for_entity(
            stakeholder_node,
            index_bundle,
        )
    )

    legal_term_chunks = (
        get_chunks_for_entity(
            legal_term_node,
            index_bundle,
        )
    )

    matching_chunks = (
        stakeholder_chunks
        & legal_term_chunks
    )

    results = []

    for chunk_id in sorted(
        matching_chunks
    ):

        results.append(
            _format_result(
                index_bundle,
                chunk_id,
                1.0,
                "graph",
            )
        )

    return results[:max(1, top_k)]


# ============================================================
# RECIPROCAL RANK FUSION
# ============================================================

def _rrf_score(
    rank: int,
    rrf_k: int,
) -> float:
    """
    Calculate a Reciprocal Rank Fusion contribution.

    RRF contribution:

        1 / (rrf_k + rank)
    """

    return 1.0 / (
        rrf_k + rank
    )


# ============================================================
# HYBRID RETRIEVAL
# ============================================================

def hybrid_search(
    query: str,
    index_bundle: IndexBundle,
    embedding_model,
    top_k: int = 5,
    candidate_k: int = 10,
    rrf_k: int = 60,
) -> list[dict[str, Any]]:
    """
    Combine semantic and BM25 retrieval using
    Reciprocal Rank Fusion (RRF).

    Semantic and BM25 scores are not directly combined.
    Instead, their rankings are fused.

    Parameters
    ----------
    query:
        Natural-language search query.

    index_bundle:
        Node 2 IndexBundle.

    embedding_model:
        Same SentenceTransformer model used by FAISS.

    top_k:
        Number of final hybrid results.

    candidate_k:
        Number of candidates retrieved from each
        individual retrieval method.

    rrf_k:
        RRF smoothing constant.

    Returns
    -------
    list[dict[str, Any]]
        Ranked hybrid retrieval results.
    """

    if not query or not query.strip():
        return []

    candidate_k = max(
        1,
        candidate_k,
    )

    top_k = max(
        1,
        top_k,
    )

    rrf_k = max(
        1,
        rrf_k,
    )

    semantic_results = (
        semantic_search(
            query,
            index_bundle,
            embedding_model,
            candidate_k,
        )
    )

    keyword_results = (
        keyword_search(
            query,
            index_bundle,
            candidate_k,
        )
    )

    fused_scores: dict[
        str,
        float,
    ] = {}

    result_lookup: dict[
        str,
        dict[str, Any],
    ] = {}

    # --------------------------------------------------------
    # Semantic ranking
    # --------------------------------------------------------

    for rank, result in enumerate(
        semantic_results,
        start=1,
    ):

        chunk_id = result[
            "chunk_id"
        ]

        fused_scores[chunk_id] = (
            fused_scores.get(
                chunk_id,
                0.0,
            )
            + _rrf_score(
                rank,
                rrf_k,
            )
        )

        result_lookup[
            chunk_id
        ] = result

    # --------------------------------------------------------
    # BM25 ranking
    # --------------------------------------------------------

    for rank, result in enumerate(
        keyword_results,
        start=1,
    ):

        chunk_id = result[
            "chunk_id"
        ]

        fused_scores[chunk_id] = (
            fused_scores.get(
                chunk_id,
                0.0,
            )
            + _rrf_score(
                rank,
                rrf_k,
            )
        )

        result_lookup[
            chunk_id
        ] = result

    # --------------------------------------------------------
    # Final ranking
    # --------------------------------------------------------

    ranked_chunk_ids = sorted(
        fused_scores,
        key=fused_scores.get,
        reverse=True,
    )[:top_k]

    results = []

    for chunk_id in ranked_chunk_ids:

        result = dict(
            result_lookup[chunk_id]
        )

        result["score"] = (
            fused_scores[chunk_id]
        )

        result["retrieval_method"] = (
            "hybrid"
        )

        results.append(
            result
        )

    return results