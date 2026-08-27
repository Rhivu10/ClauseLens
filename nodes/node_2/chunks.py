"""
ClauseLens — Node 2
Structured chunk preparation.

This module converts Node 1's output into the canonical
chunk representation consumed by the Node 2 indexer.
"""

from typing import Any


REQUIRED_CHUNK_FIELDS = {
    "chunk_id",
    "text",
    "page_start",
    "page_end",
    "heading_path",
}


def prepare_chunks(
    documents: dict[str, dict[str, Any]],
    version_ids: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """
    Convert Node 1 document output into Node 2 structured chunks.

    Parameters
    ----------
    documents:
        Output produced by Node 1's process_documents().

    version_ids:
        Optional mapping from document_id to version_id.

    Returns
    -------
    list[dict[str, Any]]
        Canonical Node 2 chunk records.
    """

    if version_ids is None:
        version_ids = {}

    all_chunks = []

    for document_name, document in documents.items():

        document_id = document.get(
            "document_id",
            document_name,
        )

        chunks = document.get(
            "chunks",
            [],
        )

        if not isinstance(chunks, list):
            raise TypeError(
                f"Chunks for '{document_id}' must be a list."
            )

        version_id = version_ids.get(
            document_id
        )

        for chunk in chunks:

            missing_fields = (
                REQUIRED_CHUNK_FIELDS
                - chunk.keys()
            )

            if missing_fields:
                raise ValueError(
                    f"Chunk is missing required fields: "
                    f"{sorted(missing_fields)}"
                )

            text = chunk["text"]

            if not isinstance(text, str):
                raise TypeError(
                    f"Chunk '{chunk['chunk_id']}' "
                    f"text must be a string."
                )

            if not text.strip():
                raise ValueError(
                    f"Chunk '{chunk['chunk_id']}' "
                    f"contains empty text."
                )

            structured_chunk = {
                "chunk_id": chunk["chunk_id"],
                "document_id": document_id,
                "version_id": version_id,
                "text": text,
                "page_start": chunk["page_start"],
                "page_end": chunk["page_end"],
                "heading_path": chunk["heading_path"],
            }

            all_chunks.append(
                structured_chunk
            )

    _validate_chunk_ids(all_chunks)

    return all_chunks


def _validate_chunk_ids(
    chunks: list[dict[str, Any]],
) -> None:
    """
    Ensure every chunk ID is unique.
    """

    chunk_ids = [
        chunk["chunk_id"]
        for chunk in chunks
    ]

    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError(
            "Duplicate chunk IDs detected."
        )
