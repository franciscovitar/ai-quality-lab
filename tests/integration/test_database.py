import json
from pathlib import Path

import pytest
from psycopg import Connection

from aql.corpus import seed_corpus
from aql.database import is_ready
from aql.retrieval import search_lexical

CORPUS = Path(__file__).resolve().parents[2] / "data/synthetic_docs/corpus_v1.json"


@pytest.mark.integration
def test_migration_and_pgvector_are_present(database: Connection) -> None:
    assert is_ready(database)
    version = database.execute("SELECT version_num FROM alembic_version").fetchone()
    assert version["version_num"] == "0001_corpus"


@pytest.mark.integration
def test_seed_is_idempotent(database: Connection) -> None:
    first = seed_corpus(database, CORPUS)
    second = seed_corpus(database, CORPUS)
    assert first == second
    assert first[0] == 7
    assert first[1] > first[0]
    row = database.execute("SELECT count(*) AS total FROM documents").fetchone()
    assert row["total"] == 7
    row = database.execute("SELECT count(*) AS total FROM chunks").fetchone()
    assert row["total"] == first[1]


@pytest.mark.integration
def test_search_finds_current_policy_not_retired_version(database: Connection) -> None:
    seed_corpus(database, CORPUS)
    hits = search_lexical(database, "webhook delivery maximum budget five attempts", top_k=10)
    policy_hits = [hit for hit in hits if hit.document_slug == "webhook-delivery-policy"]
    assert policy_hits
    assert all(hit.document_version == 2 for hit in policy_hits)
    assert all(hit.chunk_id and hit.content for hit in hits)


@pytest.mark.integration
def test_search_handles_untrusted_user_input_without_sql_injection(database: Connection) -> None:
    seed_corpus(database, CORPUS)
    hits = search_lexical(database, "') OR 1=1 --", top_k=10)
    assert not hits
    assert database.execute("SELECT count(*) AS count FROM documents").fetchone()["count"] == 7


@pytest.mark.integration
def test_failed_seed_rolls_back_every_insert(database: Connection, tmp_path: Path) -> None:
    source = json.loads(CORPUS.read_text(encoding="utf-8"))
    new_first = {
        "slug": "atomicity-sentinel",
        "version": 1,
        "title": "Atomicity Sentinel",
        "effective_on": "2026-11-19",
        "is_current": True,
        "content": "This row must not survive a failed seed.",
    }
    collision = {
        "slug": "atomicity-sentinel",
        "version": 2,
        "title": "Conflicting current version",
        "effective_on": "2026-11-20",
        "is_current": True,
        "content": "This would create two current versions of one document.",
    }
    source["documents"] = [new_first, collision]
    path = tmp_path / "failing-corpus.json"
    path.write_text(json.dumps(source), encoding="utf-8")
    with pytest.raises(Exception, match="uq_documents_current_slug"):
        with database.transaction():
            seed_corpus(database, path)
    assert database.execute("SELECT count(*) AS total FROM documents").fetchone()["total"] == 0
