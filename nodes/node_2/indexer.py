"""
ClauseLens — Node 2
Indexer and Versioned Vector Store.

Builds the derived indexes used by Node 3:

    Structured chunks
          |
          +----> SentenceTransformer embeddings
          |             |
          |             +----> FAISS vector index
          |
          +----> BM25 keyword index
          |
          +----> NetworkX knowledge graph
                         |
                         +----> stakeholders
                         +----> legal terms
                         +----> dates

The original chunks remain the canonical source data.
Indexes are derived structures.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import faiss
import networkx as nx
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

DEFAULT_LEGAL_TERMS = (
    "payment",
    "expenses",
    "indemnification",
    "confidentiality",
    "representations and warranties",
    "termination",
)

# This is deliberately a small fallback vocabulary.
# It is NOT intended to be the final entity-resolution system.
DEFAULT_STAKEHOLDER_ROLES = (
    "Company",
    "Bank",
    "Agent",
    "Customer",
    "Affiliate",
)


# ============================================================
# INDEX BUNDLE
# ============================================================

@dataclass
class IndexBundle:
    """
    All derived Node 2 indexes.

    Attributes
    ----------
    chunks:
        Canonical structured chunks.

    embeddings:
        Normalized embedding matrix. Row i corresponds to
        chunks[i].

    vector_index:
        FAISS inner-product index over normalized embeddings.

    bm25:
        BM25 keyword-search index.

    tokenized_chunks:
        Tokenized chunk text used by BM25.

    knowledge_graph:
        NetworkX graph containing documents, versions,
        chunks, and extracted entities.

    vector_id_to_chunk_id:
        Mapping from FAISS row/index to chunk ID.

    bm25_id_to_chunk_id:
        Mapping from BM25 row/index to chunk ID.

    embedding_model_name:
        Name of the embedding model used.
    """

    chunks: list[dict[str, Any]]
    embeddings: np.ndarray
    vector_index: faiss.Index
    bm25: BM25Okapi
    tokenized_chunks: list[list[str]]
    knowledge_graph: nx.MultiDiGraph
    vector_id_to_chunk_id: dict[int, str]
    bm25_id_to_chunk_id: dict[int, str]
    embedding_model_name: str


# ============================================================
# INDEXER
# ============================================================

class Node2Indexer:
    """
    Builds all Node 2 indexes from structured chunks.
    """

    def __init__(
        self,
        embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
        legal_terms: tuple[str, ...] = DEFAULT_LEGAL_TERMS,
        stakeholder_roles: tuple[str, ...] = DEFAULT_STAKEHOLDER_ROLES,
    ) -> None:

        self.embedding_model_name = (
            embedding_model_name
        )

        self.legal_terms = legal_terms

        self.stakeholder_roles = stakeholder_roles

        self.embedding_model = (
            SentenceTransformer(
                self.embedding_model_name
            )
        )


    # ========================================================
    # PUBLIC BUILD METHOD
    # ========================================================

    def build(
        self,
        chunks: list[dict[str, Any]],
    ) -> IndexBundle:
        """
        Build the complete Node 2 index bundle.

        Parameters
        ----------
        chunks:
            Canonical chunks produced by prepare_chunks().

        Returns
        -------
        IndexBundle
            FAISS, BM25, NetworkX and their mappings.
        """

        self._validate_chunks(chunks)

        embeddings = self._build_embeddings(
            chunks
        )

        vector_index = self._build_vector_index(
            embeddings
        )

        tokenized_chunks = (
            self._tokenize_chunks(chunks)
        )

        bm25 = BM25Okapi(
            tokenized_chunks
        )

        knowledge_graph = (
            self._build_knowledge_graph(
                chunks
            )
        )

        vector_id_to_chunk_id = {
            index: chunk["chunk_id"]
            for index, chunk
            in enumerate(chunks)
        }

        bm25_id_to_chunk_id = {
            index: chunk["chunk_id"]
            for index, chunk
            in enumerate(chunks)
        }

        return IndexBundle(
            chunks=chunks,
            embeddings=embeddings,
            vector_index=vector_index,
            bm25=bm25,
            tokenized_chunks=tokenized_chunks,
            knowledge_graph=knowledge_graph,
            vector_id_to_chunk_id=(
                vector_id_to_chunk_id
            ),
            bm25_id_to_chunk_id=(
                bm25_id_to_chunk_id
            ),
            embedding_model_name=(
                self.embedding_model_name
            ),
        )


    # ========================================================
    # VALIDATION
    # ========================================================

    @staticmethod
    def _validate_chunks(
        chunks: list[dict[str, Any]],
    ) -> None:
        """
        Validate the canonical chunk structure.
        """

        if not chunks:
            raise ValueError(
                "Cannot build Node 2 indexes from "
                "an empty chunk list."
            )

        required_fields = {
            "chunk_id",
            "document_id",
            "version_id",
            "text",
            "page_start",
            "page_end",
            "heading_path",
        }

        chunk_ids = set()

        for chunk in chunks:

            missing = (
                required_fields
                - chunk.keys()
            )

            if missing:
                raise ValueError(
                    "Chunk "
                    f"'{chunk.get('chunk_id', '<unknown>')}' "
                    f"is missing fields: "
                    f"{sorted(missing)}"
                )

            chunk_id = chunk["chunk_id"]

            if chunk_id in chunk_ids:
                raise ValueError(
                    f"Duplicate chunk ID: {chunk_id}"
                )

            chunk_ids.add(chunk_id)

            if not chunk["text"].strip():
                raise ValueError(
                    f"Chunk '{chunk_id}' has empty text."
                )


    # ========================================================
    # EMBEDDINGS
    # ========================================================

    def _build_embeddings(
        self,
        chunks: list[dict[str, Any]],
    ) -> np.ndarray:
        """
        Generate normalized embeddings for all chunks.
        """

        texts = [
            chunk["text"]
            for chunk in chunks
        ]

        embeddings = (
            self.embedding_model.encode(
                texts,
                show_progress_bar=True,
                convert_to_numpy=True,
            )
        )

        embeddings = np.asarray(
            embeddings,
            dtype=np.float32,
        )

        faiss.normalize_L2(
            embeddings
        )

        return embeddings


    # ========================================================
    # FAISS
    # ========================================================

    @staticmethod
    def _build_vector_index(
        embeddings: np.ndarray,
    ) -> faiss.Index:
        """
        Build a cosine-similarity FAISS index.

        Since embeddings are L2-normalized, inner product
        is equivalent to cosine similarity.
        """

        dimension = embeddings.shape[1]

        vector_index = (
            faiss.IndexFlatIP(dimension)
        )

        vector_index.add(
            embeddings
        )

        return vector_index


    # ========================================================
    # BM25
    # ========================================================

    @staticmethod
    def _tokenize_chunks(
        chunks: list[dict[str, Any]],
    ) -> list[list[str]]:
        """
        Create simple lowercase tokenization for BM25.
        """

        return [
            chunk["text"].lower().split()
            for chunk in chunks
        ]


    # ========================================================
    # KNOWLEDGE GRAPH
    # ========================================================

    def _build_knowledge_graph(
        self,
        chunks: list[dict[str, Any]],
    ) -> nx.MultiDiGraph:
        """
        Build the Node 2 knowledge graph.
        """

        graph = nx.MultiDiGraph()

        self._add_document_and_version_nodes(
            graph,
            chunks,
        )

        self._add_chunk_nodes(
            graph,
            chunks,
        )

        self._add_stakeholder_relationships(
            graph,
            chunks,
        )

        self._add_legal_term_relationships(
            graph,
            chunks,
        )

        self._add_date_relationships(
            graph,
            chunks,
        )

        return graph


    # ========================================================
    # DOCUMENT / VERSION LAYER
    # ========================================================

    @staticmethod
    def _add_document_and_version_nodes(
        graph: nx.MultiDiGraph,
        chunks: list[dict[str, Any]],
    ) -> None:
        """
        Add document and version nodes.

        Structure:

            document
                |
                +--has_version--> version
        """

        documents: dict[
            str,
            str,
        ] = {}

        for chunk in chunks:

            document_id = chunk[
                "document_id"
            ]

            version_id = chunk[
                "version_id"
            ]

            documents[
                document_id
            ] = version_id

        for document_id, version_id in (
            documents.items()
        ):

            graph.add_node(
                document_id,
                node_type="document",
                document_id=document_id,
            )

            version_node_id = (
                f"{document_id}:{version_id}"
            )

            graph.add_node(
                version_node_id,
                node_type="version",
                document_id=document_id,
                version_id=version_id,
            )

            graph.add_edge(
                document_id,
                version_node_id,
                relation="has_version",
            )


    # ========================================================
    # CHUNK LAYER
    # ========================================================

    @staticmethod
    def _add_chunk_nodes(
        graph: nx.MultiDiGraph,
        chunks: list[dict[str, Any]],
    ) -> None:
        """
        Add chunk nodes and connect them to versions.

        Structure:

            version
                |
                +--contains--> chunk
        """

        for chunk in chunks:

            chunk_id = chunk[
                "chunk_id"
            ]

            graph.add_node(
                chunk_id,
                node_type="chunk",
                document_id=chunk[
                    "document_id"
                ],
                version_id=chunk[
                    "version_id"
                ],
                page_start=chunk[
                    "page_start"
                ],
                page_end=chunk[
                    "page_end"
                ],
                heading_path=chunk[
                    "heading_path"
                ],
                text=chunk["text"],
            )

            version_node_id = (
                f"{chunk['document_id']}:"
                f"{chunk['version_id']}"
            )

            graph.add_edge(
                version_node_id,
                chunk_id,
                relation="contains",
            )


    # ========================================================
    # STAKEHOLDERS
    # ========================================================

    def _add_stakeholder_relationships(
        self,
        graph: nx.MultiDiGraph,
        chunks: list[dict[str, Any]],
    ) -> None:
        """
        Add contractual stakeholder relationships.

        This is currently a controlled-role fallback.
        It is intentionally kept isolated so that a future
        entity resolver can replace it without changing the
        graph architecture.
        """

        for chunk in chunks:

            chunk_id = chunk[
                "chunk_id"
            ]

            text = chunk[
                "text"
            ].lower()

            for role in self.stakeholder_roles:

                if role.lower() not in text:
                    continue

                node_id = (
                    "stakeholder:"
                    f"{role.lower()}"
                )

                graph.add_node(
                    node_id,
                    node_type="stakeholder",
                    name=role,
                )

                graph.add_edge(
                    chunk_id,
                    node_id,
                    relation="mentions",
                )


    # ========================================================
    # LEGAL TERMS
    # ========================================================

    def _add_legal_term_relationships(
        self,
        graph: nx.MultiDiGraph,
        chunks: list[dict[str, Any]],
    ) -> None:
        """
        Add legal-term nodes and chunk relationships.
        """

        for chunk in chunks:

            chunk_id = chunk[
                "chunk_id"
            ]

            text = chunk[
                "text"
            ].lower()

            for legal_term in self.legal_terms:

                if legal_term.lower() not in text:
                    continue

                node_id = (
                    "legal_term:"
                    f"{legal_term.lower()}"
                )

                graph.add_node(
                    node_id,
                    node_type="legal_term",
                    name=legal_term,
                )

                graph.add_edge(
                    chunk_id,
                    node_id,
                    relation="mentions",
                )


    # ========================================================
    # DATES
    # ========================================================

    @staticmethod
    def _add_date_relationships(
        graph: nx.MultiDiGraph,
        chunks: list[dict[str, Any]],
    ) -> None:
        """
        Extract date mentions using deterministic patterns.

        Dates are deliberately handled with regex rather than
        an ML model because the supported formats are highly
        structured.
        """

        date_patterns = [

            r"\b(?:January|February|March|April|May|June|July|August|"
            r"September|October|November|December)"
            r"\s+\d{1,2},\s+\d{4}\b",

            r"\b(?:Jan\.|Feb\.|Mar\.|Apr\.|May|Jun\.|Jul\.|Aug\.|"
            r"Sep\.|Sept\.|Oct\.|Nov\.|Dec\.)"
            r"\s+\d{1,2},\s+\d{4}\b",

            r"\b\d{1,2}/\d{1,2}/\d{4}\b",

            r"\b\d{1,2}-\d{1,2}-\d{4}\b",
        ]

        date_pattern = re.compile(
            "|".join(
                f"(?:{pattern})"
                for pattern in date_patterns
            ),
            flags=re.IGNORECASE,
        )

        for chunk in chunks:

            chunk_id = chunk[
                "chunk_id"
            ]

            text = chunk[
                "text"
            ]

            text = text.replace(
                "\xa0",
                " ",
            )

            text = re.sub(
                r"\s+",
                " ",
                text,
            ).strip()

            matches = date_pattern.findall(
                text
            )

            for date_text in matches:

                date_text = " ".join(
                    date_text.split()
                )

                node_id = (
                    "date:"
                    f"{date_text.lower()}"
                )

                graph.add_node(
                    node_id,
                    node_type="date",
                    value=date_text,
                )

                graph.add_edge(
                    chunk_id,
                    node_id,
                    relation="mentions",
                )