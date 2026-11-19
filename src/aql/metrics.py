"""Retrieval metrics that do not rely on an LLM grader."""

from collections.abc import Sequence


def recall_at_k(
    expected_document_ids: set[str], retrieved_document_ids: Sequence[str], k: int
) -> float:
    if not expected_document_ids:
        raise ValueError("A retrieval metric requires at least one relevant document")
    if k < 1:
        raise ValueError("k must be positive")
    return len(expected_document_ids.intersection(retrieved_document_ids[:k])) / len(
        expected_document_ids
    )


def reciprocal_rank_at_k(
    expected_document_ids: set[str], retrieved_document_ids: Sequence[str], k: int
) -> float:
    if not expected_document_ids:
        raise ValueError("A retrieval metric requires at least one relevant document")
    if k < 1:
        raise ValueError("k must be positive")
    for index, document_id in enumerate(retrieved_document_ids[:k], start=1):
        if document_id in expected_document_ids:
            return 1.0 / index
    return 0.0
