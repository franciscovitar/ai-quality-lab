"""Corpus with versioned source material, FTS and future semantic vectors.

Revision ID: 0001_corpus
Revises:
Create Date: 2026-11-19
"""

from alembic import op

revision = "0001_corpus"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute(
        """
        CREATE TABLE documents (
            id uuid PRIMARY KEY,
            slug text NOT NULL,
            version integer NOT NULL CHECK (version > 0),
            title text NOT NULL CHECK (length(title) > 0),
            effective_on date NOT NULL,
            is_current boolean NOT NULL DEFAULT false,
            content text NOT NULL CHECK (length(content) > 0),
            content_sha256 varchar(64) NOT NULL CHECK (content_sha256 ~ '^[0-9a-f]{64}$'),
            created_at timestamptz NOT NULL DEFAULT now(),
            CONSTRAINT uq_documents_slug_version UNIQUE (slug, version)
        )
        """
    )
    op.execute("CREATE UNIQUE INDEX uq_documents_current_slug ON documents (slug) WHERE is_current")
    op.execute(
        """
        CREATE TABLE chunks (
            id uuid PRIMARY KEY,
            document_id uuid NOT NULL REFERENCES documents(id) ON DELETE RESTRICT,
            ordinal integer NOT NULL CHECK (ordinal >= 0),
            chunker_version text NOT NULL CHECK (length(chunker_version) > 0),
            start_offset integer NOT NULL CHECK (start_offset >= 0),
            end_offset integer NOT NULL CHECK (end_offset > start_offset),
            content text NOT NULL CHECK (length(content) > 0),
            content_sha256 varchar(64) NOT NULL CHECK (content_sha256 ~ '^[0-9a-f]{64}$'),
            embedding vector(384),
            embedding_model text,
            search_vector tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
            CONSTRAINT uq_chunks_document_chunker_ordinal
                UNIQUE (document_id, chunker_version, ordinal),
            CONSTRAINT ck_chunks_embedding_model_pair
                CHECK ((embedding IS NULL) = (embedding_model IS NULL))
        )
        """
    )
    op.execute("CREATE INDEX ix_chunks_search ON chunks USING gin (search_vector)")
    op.execute("CREATE INDEX ix_chunks_document ON chunks (document_id)")


def downgrade() -> None:
    op.execute("DROP TABLE chunks")
    op.execute("DROP TABLE documents")
    # Do not remove the vector extension: it could be shared with other schemas.
