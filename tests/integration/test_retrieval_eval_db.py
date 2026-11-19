"""The golden retrieval suite runs against PostgreSQL, not stubbed results."""

import json
from pathlib import Path

import pytest
from psycopg import Connection

from aql.corpus import seed_corpus
from aql.evaluation import evaluate_retrieval, find_regressions, load_dataset, report_json
from aql.retrieval import search_lexical

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "data/synthetic_docs/corpus_v1.json"
DATASET = ROOT / "data/evals/retrieval_v1.json"
BASELINE = ROOT / "data/evals/baselines/postgres_fts_v1.json"


@pytest.mark.integration
def test_locked_goldens_run_reproducibly_against_postgres(database: Connection) -> None:
    seed_corpus(database, CORPUS)
    goldens = load_dataset(DATASET, CORPUS)

    def retrieve(query: str, top_k: int):
        return search_lexical(database, query, top_k)

    first = evaluate_retrieval(goldens, retrieve, top_k=5)
    second = evaluate_retrieval(goldens, retrieve, top_k=5)

    assert report_json(first) == report_json(second)
    assert first["summary"]["cases"] == 16
    assert first["summary"]["answerable"] == 12
    assert first["summary"]["unanswerable"] == 4
    # A verified positive and a known strict-AND failure: this is a measured baseline.
    by_case = {row["case_id"]: row for row in first["cases"]}
    assert by_case["R001"]["recall_at_k"] == 1.0
    assert by_case["R015"]["recall_at_k"] == 0.0
    assert all(row["retrieved_chunks"] <= 5 for row in first["cases"])
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    assert find_regressions(baseline, first) == []
