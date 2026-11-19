import json
from pathlib import Path

import pytest

from aql.evaluation import (
    evaluate_retrieval,
    find_regressions,
    load_dataset,
    report_json,
)
from aql.retrieval import SearchHit

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "data/synthetic_docs/corpus_v1.json"
DATASET = ROOT / "data/evals/retrieval_v1.json"
BASELINE = ROOT / "data/evals/baselines/postgres_fts_v1.json"


def hit(slug: str, version: int, chunk_id: str) -> SearchHit:
    return SearchHit(
        chunk_id=chunk_id,
        document_slug=slug,
        document_version=version,
        title="test",
        content="synthetic content",
        score=1.0,
    )


def test_locked_dataset_has_distinct_splits_and_versioned_goldens() -> None:
    loaded = load_dataset(DATASET, CORPUS)
    assert len(loaded.data.cases) == 16
    assert {case.split for case in loaded.data.cases} == {"dev", "holdout"}
    assert {case.expected_behavior for case in loaded.data.cases} == {"answer", "abstain"}
    assert "webhook-delivery-policy@v2" in loaded.current_sources
    assert "webhook-delivery-policy@v1" not in loaded.current_sources


def test_committed_baseline_is_bound_to_current_goldens() -> None:
    golden = load_dataset(DATASET, CORPUS)
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    assert baseline["dataset_sha256"] == golden.dataset_sha256
    assert baseline["corpus_sha256"] == golden.data.corpus_sha256
    assert baseline["summary"]["cases"] == len(golden.data.cases)
    assert {row["case_id"] for row in baseline["cases"]} == {
        case.case_id for case in golden.data.cases
    }


def test_corpus_change_requires_review_even_if_labels_still_exist(tmp_path: Path) -> None:
    altered_corpus = tmp_path / "changed.json"
    altered_corpus.write_bytes(CORPUS.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="corpus changed"):
        load_dataset(DATASET, altered_corpus)


def test_retired_version_is_rejected_as_ground_truth(tmp_path: Path) -> None:
    altered_dataset = json.loads(DATASET.read_text(encoding="utf-8"))
    altered_dataset["cases"][0]["expected_documents"][0]["version"] = 1
    dataset_path = tmp_path / "stale-label.json"
    dataset_path.write_text(json.dumps(altered_dataset), encoding="utf-8")
    with pytest.raises(ValueError, match="not a current source"):
        load_dataset(dataset_path, CORPUS)


@pytest.mark.parametrize(
    "change",
    [
        lambda data: data["cases"][1].update(case_id="R001"),
        lambda data: data["cases"][0].update(expected_documents=[]),
        lambda data: data["cases"][10].update(expected_documents=[{"slug": "x", "version": 1}]),
        lambda data: data["cases"][0].update(tags=["policy", "policy"]),
    ],
)
def test_invalid_goldens_fail_closed(tmp_path: Path, change) -> None:
    altered = json.loads(DATASET.read_text(encoding="utf-8"))
    change(altered)
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(altered), encoding="utf-8")
    with pytest.raises(ValueError):
        load_dataset(path, CORPUS)


def test_retrieval_eval_deduplicates_chunks_without_hiding_misses() -> None:
    golden = load_dataset(DATASET, CORPUS)

    def retrieve(question: str, top_k: int) -> list[SearchHit]:
        assert top_k == 5
        if question == "webhook delivery maximum budget five attempts":
            return [
                hit("webhook-delivery-policy", 2, "chunk-a"),
                hit("webhook-delivery-policy", 2, "chunk-b"),
            ]
        return []

    report = evaluate_retrieval(golden, retrieve)
    first = report["cases"][0]
    assert first["retrieved_chunks"] == 2
    assert first["retrieved_documents"] == ["webhook-delivery-policy@v2"]
    assert first["recall_at_k"] == 1.0
    assert first["mrr_at_k"] == 1.0
    assert report["summary"]["unanswerable_no_hit_rate"] == 1.0
    assert report["summary"]["recall_at_k"] < 1.0
    assert report_json(report) == report_json(evaluate_retrieval(golden, retrieve))


def test_retrieval_eval_rejects_stale_hits_duplicate_ids_and_over_budget() -> None:
    golden = load_dataset(DATASET, CORPUS)
    for bad_hits, message in [
        ([hit("webhook-delivery-policy", 1, "a")], "stale/unknown"),
        ([hit("webhook-delivery-policy", 2, "a")] * 2, "duplicate chunk"),
        ([hit("webhook-delivery-policy", 2, str(i)) for i in range(6)], "budget"),
    ]:

        def retrieve(_q: str, _k: int, results=bad_hits):
            return results

        with pytest.raises(ValueError, match=message):
            evaluate_retrieval(golden, retrieve)


def test_comparator_catches_per_case_regression() -> None:
    golden = load_dataset(DATASET, CORPUS)
    known = {
        "webhook delivery maximum budget five attempts": [hit("webhook-delivery-policy", 2, "a")]
    }
    baseline = evaluate_retrieval(golden, lambda q, _: known.get(q, []))
    candidate = evaluate_retrieval(golden, lambda _q, _k: [])
    assert "R001: recall_at_k" in "\n".join(find_regressions(baseline, candidate))
    assert "R001: mrr_at_k" in "\n".join(find_regressions(baseline, candidate))

    modified = json.loads(report_json(candidate))
    modified["dataset_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="Incomparable"):
        find_regressions(baseline, modified)


def test_comparator_catches_unanswerable_false_positive() -> None:
    golden = load_dataset(DATASET, CORPUS)
    baseline = evaluate_retrieval(golden, lambda _q, _k: [])
    candidate = evaluate_retrieval(
        golden,
        lambda q, _: (
            [hit("security-boundary", 1, "a")]
            if q == "What is the production database root password?"
            else []
        ),
    )
    assert "R011" in "\n".join(find_regressions(baseline, candidate))


def test_comparator_rejects_duplicate_case_or_tampered_labels() -> None:
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    duplicate = json.loads(report_json(baseline))
    duplicate["cases"].append(duplicate["cases"][0])
    with pytest.raises(ValueError, match="case IDs differ or repeat"):
        find_regressions(baseline, duplicate)

    tampered = json.loads(report_json(baseline))
    tampered["cases"][0]["expected_documents"] = []
    with pytest.raises(ValueError, match="expected_documents changed"):
        find_regressions(baseline, tampered)


def test_retrieval_budget_must_be_supported() -> None:
    golden = load_dataset(DATASET, CORPUS)
    with pytest.raises(ValueError, match="top_k"):
        evaluate_retrieval(golden, lambda _q, _k: [], top_k=11)
