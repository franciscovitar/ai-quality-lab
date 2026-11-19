"""Synthetic corpus ingestion: stable identities, verified idempotence, one transaction."""

import hashlib
import json
import uuid
from pathlib import Path
from typing import Any

from psycopg import Connection

from aql.chunking import CHUNKER_VERSION, chunk_paragraphs, normalize_content

NAMESPACE = uuid.UUID("53d1686f-eeb3-493b-8e9f-20e405c4f9ad")


def sha256(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _seed_document(connection: Connection, document: dict[str, Any]) -> int:
    slug = document["slug"]
    version = document["version"]
    content = normalize_content(document["content"])
    checksum = sha256(content)
    identity = uuid.uuid5(NAMESPACE, f"document:{slug}:v{version}")

    connection.execute(
        """
        INSERT INTO documents
            (id, slug, version, title, effective_on, is_current, content, content_sha256)
        VALUES (%s, %s, %s, %s, %s::date, %s, %s, %s)
        ON CONFLICT (slug, version) DO NOTHING
        """,
        (
            identity,
            slug,
            version,
            document["title"],
            document["effective_on"],
            document["is_current"],
            content,
            checksum,
        ),
    )
    existing = connection.execute(
        """SELECT id, title, effective_on::text AS effective_on,
                  is_current, content_sha256 FROM documents
           WHERE slug = %s AND version = %s""",
        (slug, version),
    ).fetchone()
    if not existing or (
        existing["id"] != identity
        or existing["content_sha256"] != checksum
        or existing["title"] != document["title"]
        or existing["effective_on"] != document["effective_on"]
        or existing["is_current"] != document["is_current"]
    ):
        raise ValueError(f"Existing document differs from seed: {slug}:v{version}")

    chunks = chunk_paragraphs(content)
    for chunk in chunks:
        chunk_key = f"chunk:{slug}:v{version}:{CHUNKER_VERSION}:{chunk.ordinal}"
        chunk_id = uuid.uuid5(NAMESPACE, chunk_key)
        connection.execute(
            """
            INSERT INTO chunks
                (id, document_id, ordinal, chunker_version, start_offset, end_offset,
                 content, content_sha256)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (document_id, chunker_version, ordinal) DO NOTHING
            """,
            (
                chunk_id,
                identity,
                chunk.ordinal,
                CHUNKER_VERSION,
                chunk.start_offset,
                chunk.end_offset,
                chunk.content,
                sha256(chunk.content),
            ),
        )
        existing_chunk = connection.execute(
            """SELECT id, content_sha256, start_offset, end_offset
               FROM chunks WHERE document_id = %s AND chunker_version = %s AND ordinal = %s""",
            (identity, CHUNKER_VERSION, chunk.ordinal),
        ).fetchone()
        if not existing_chunk or (
            existing_chunk["id"] != chunk_id
            or existing_chunk["content_sha256"] != sha256(chunk.content)
            or existing_chunk["start_offset"] != chunk.start_offset
            or existing_chunk["end_offset"] != chunk.end_offset
        ):
            raise ValueError(f"Existing chunk differs from seed: {slug}:v{version}:{chunk.ordinal}")
    return len(chunks)


def seed_corpus(connection: Connection, path: Path) -> tuple[int, int]:
    """All inserts are in the caller's transaction; failures roll back every document."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported corpus schema")
    documents = payload["documents"]
    keys = [(item["slug"], item["version"]) for item in documents]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate seed document identity")

    chunk_count = 0
    for document in documents:
        chunk_count += _seed_document(connection, document)
    return len(documents), chunk_count
