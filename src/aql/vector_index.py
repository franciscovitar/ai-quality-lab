"""Transactional pgvector materialization with model identity and completeness checks."""

from dataclasses import dataclass

from psycopg import Connection

from aql.embeddings import Embedder, vector_literal


@dataclass(frozen=True)
class IndexStats:
    total: int
    inserted: int
    existing: int


def index_embeddings(connection: Connection, embedder: Embedder) -> IndexStats:
    """Insert missing vectors, never silently mix or overwrite model revisions.

    Caller owns the transaction. On any encoding, validation or database error,
    the caller must roll it back; the standard CLI uses the connection context.
    """
    rows = connection.execute(
        """SELECT id, content, embedding_model FROM chunks
           ORDER BY document_id, ordinal FOR UPDATE"""
    ).fetchall()
    wrong_models = [row for row in rows if row["embedding_model"] not in (None, embedder.model_key)]
    if wrong_models:
        raise ValueError("A different embedding model already owns these chunks")

    pending = [row for row in rows if row["embedding_model"] is None]
    vectors = embedder.encode([row["content"] for row in pending])
    if len(vectors) != len(pending):
        raise ValueError("The embedder returned a different number of vectors")

    for row, vector in zip(pending, vectors, strict=True):
        updated = connection.execute(
            """UPDATE chunks SET embedding = %s::vector, embedding_model = %s
               WHERE id = %s AND embedding IS NULL AND embedding_model IS NULL""",
            (vector_literal(vector), embedder.model_key, row["id"]),
        )
        if updated.rowcount != 1:
            raise RuntimeError("An embedding changed concurrently; retry the entire transaction")
    return IndexStats(total=len(rows), inserted=len(pending), existing=len(rows) - len(pending))


def assert_index_complete(connection: Connection, model_key: str) -> None:
    """Never measure a partially indexed corpus as if it were retrieval failure."""
    row = connection.execute(
        """SELECT count(*) AS total,
                  count(*) FILTER (WHERE c.embedding_model = %s AND c.embedding IS NOT NULL)
                    AS indexed
           FROM chunks c JOIN documents d ON d.id = c.document_id
           WHERE d.is_current""",
        (model_key,),
    ).fetchone()
    if not row or not row["total"] or row["indexed"] != row["total"]:
        raise ValueError("Current corpus has incomplete or mismatched embeddings")
