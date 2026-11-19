"""Inspectable FTS baseline, exact pgvector search and rank-only hybrid fusion."""

from dataclasses import dataclass

from psycopg import Connection

from aql.embeddings import vector_literal


@dataclass(frozen=True)
class SearchHit:
    chunk_id: str
    document_slug: str
    document_version: int
    title: str
    content: str
    score: float


def search_lexical(connection: Connection, question: str, top_k: int = 5) -> list[SearchHit]:
    if not 1 <= top_k <= 10:
        raise ValueError("top_k must be between 1 and 10")
    rows = connection.execute(
        """
        WITH query AS (SELECT plainto_tsquery('english', %s) AS tsq)
        SELECT
            c.id::text AS chunk_id, d.slug AS document_slug,
            d.version AS document_version, d.title, c.content,
            ts_rank_cd(c.search_vector, query.tsq) AS score
        FROM chunks c
        JOIN documents d ON d.id = c.document_id
        CROSS JOIN query
        WHERE d.is_current AND c.search_vector @@ query.tsq
        ORDER BY score DESC, d.slug, c.ordinal
        LIMIT %s
        """,
        (question, top_k),
    ).fetchall()
    return [SearchHit(**row) for row in rows]


def search_semantic(
    connection: Connection, query_vector: list[float], model_key: str, top_k: int = 5
) -> list[SearchHit]:
    """Exact nearest-neighbor cosine search, filtered to current matching-model chunks.

    No ANN index at this scale: exact search makes comparisons deterministic and avoids
    conflating model relevance with approximate-index recall loss.
    """
    if not 1 <= top_k <= 10:
        raise ValueError("top_k must be between 1 and 10")
    rows = connection.execute(
        """
        WITH query AS (SELECT %s::vector(384) AS embedding)
        SELECT c.id::text AS chunk_id, d.slug AS document_slug,
               d.version AS document_version, d.title, c.content,
               1 - (c.embedding <=> query.embedding) AS score
        FROM chunks c
        JOIN documents d ON d.id = c.document_id
        CROSS JOIN query
        WHERE d.is_current AND c.embedding_model = %s
        ORDER BY c.embedding <=> query.embedding, d.slug, c.ordinal
        LIMIT %s
        """,
        (vector_literal(query_vector), model_key, top_k),
    ).fetchall()
    return [SearchHit(**row) for row in rows]


def fuse_rrf(
    lexical_hits: list[SearchHit], semantic_hits: list[SearchHit], top_k: int = 5
) -> list[SearchHit]:
    """Reciprocal rank fusion at the chunk level; raw FTS/vector scores are incomparable."""
    if not 1 <= top_k <= 10:
        raise ValueError("top_k must be between 1 and 10")
    scores: dict[str, float] = {}
    hits: dict[str, SearchHit] = {}
    for ranking in (lexical_hits, semantic_hits):
        if len({hit.chunk_id for hit in ranking}) != len(ranking):
            raise ValueError("Each retrieval ranking must contain unique chunk IDs")
        for rank, hit in enumerate(ranking, start=1):
            previous = hits.get(hit.chunk_id)
            if previous and (
                previous.document_slug != hit.document_slug
                or previous.document_version != hit.document_version
                or previous.content != hit.content
            ):
                raise ValueError("One chunk ID was returned with conflicting source metadata")
            hits[hit.chunk_id] = hit
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (60 + rank)

    ordered = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))[:top_k]
    return [
        SearchHit(
            chunk_id=chunk_id,
            document_slug=hits[chunk_id].document_slug,
            document_version=hits[chunk_id].document_version,
            title=hits[chunk_id].title,
            content=hits[chunk_id].content,
            score=scores[chunk_id],
        )
        for chunk_id in ordered
    ]
