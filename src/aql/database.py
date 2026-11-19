"""Database access with predictable timeouts and no secret-bearing errors in HTTP responses."""

import os
from collections.abc import Iterator
from contextlib import contextmanager

import psycopg
from psycopg import Connection
from psycopg.rows import dict_row


class DatabaseNotConfigured(RuntimeError):
    """A required local DATABASE_URL was not set."""


@contextmanager
def open_database(url: str | None = None) -> Iterator[Connection]:
    connection_url = url or os.getenv("DATABASE_URL")
    if not connection_url:
        raise DatabaseNotConfigured("DATABASE_URL is required")

    with psycopg.connect(
        connection_url,
        connect_timeout=3,
        options="-c statement_timeout=5000",
        row_factory=dict_row,
    ) as connection:
        yield connection


def is_ready(connection: Connection) -> bool:
    row = connection.execute(
        """
        SELECT
            to_regclass('public.documents') IS NOT NULL AS documents_exist,
            EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector') AS vector_exists
        """
    ).fetchone()
    return bool(row and row["documents_exist"] and row["vector_exists"])
