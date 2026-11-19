"""Real PostgreSQL vector indexing and exact cosine retrieval using synthetic vectors."""

from pathlib import Path

import pytest
from psycopg import Connection

from aql.corpus import seed_corpus
from aql.embeddings import DIMENSIONS
from aql.retrieval import fuse_rrf, search_lexical, search_semantic
from aql.vector_index import assert_index_complete, index_embeddings

CORPUS = Path(__file__).resolve().parents[2] / "data/synthetic_docs/corpus_v1.json"


def vector(first: float, second: float = 0.0) -> list[float]:
    return [first, second] + [0.0] * (DIMENSIONS - 2)


class FakeEmbedder:
    model_key = "test-vectors-v1"

    def encode(self, texts: list[str]) -> list[list[float]]:
        return [vector(1.0) if "maximum budget" in text else vector(0.0, 1.0) for text in texts]


@pytest.mark.integration
def test_vector_materialization_and_current_only_semantic_search(database: Connection) -> None:
    seed_corpus(database, CORPUS)
    embedder = FakeEmbedder()

    with pytest.raises(ValueError, match="incomplete"):
        assert_index_complete(database, embedder.model_key)

    first = index_embeddings(database, embedder)
    assert first.total == first.inserted == 18
    assert first.existing == 0
    assert_index_complete(database, embedder.model_key)

    second = index_embeddings(database, embedder)
    assert second.inserted == 0 and second.existing == 18
    rows = search_semantic(database, vector(1.0), embedder.model_key, top_k=5)
    assert rows[0].document_slug == "webhook-delivery-policy"
    assert rows[0].document_version == 2
    assert all(
        row.document_version != 1 or row.document_slug != "webhook-delivery-policy" for row in rows
    )
    assert rows[0].score == pytest.approx(1.0)

    lexical = search_lexical(database, "webhook delivery maximum budget five attempts", 5)
    fusion = fuse_rrf(lexical, rows, 5)
    assert fusion[0].document_slug == "webhook-delivery-policy"

    class WrongModel(FakeEmbedder):
        model_key = "other-revision"

    with pytest.raises(ValueError, match="different embedding model"):
        index_embeddings(database, WrongModel())


@pytest.mark.integration
def test_embedding_failure_rolls_back_partial_materialization(database: Connection) -> None:
    seed_corpus(database, CORPUS)

    class FailingEmbedder(FakeEmbedder):
        def encode(self, texts: list[str]) -> list[list[float]]:
            return [vector(1.0)] * (len(texts) - 1) + [[0.0] * DIMENSIONS]

    with pytest.raises(ValueError, match="nonzero"):
        with database.transaction():
            index_embeddings(database, FailingEmbedder())

    embedded = database.execute(
        "SELECT count(*) AS n FROM chunks WHERE embedding IS NOT NULL"
    ).fetchone()
    assert embedded["n"] == 0
    assert index_embeddings(database, FakeEmbedder()).inserted == 18
