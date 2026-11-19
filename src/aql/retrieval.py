"""Simple, inspectable FTS baseline. Semantic retrieval is a later checkpoint."""

from dataclasses import dataclass

from psycopg import Connection


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
