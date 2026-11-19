import math

import pytest

from aql.embeddings import MODEL_KEY
from aql.evaluation import GoldenCase, GoldenDataset
from aql.evidence_gate import calibrate_dev_negatives, gated_hits
from aql.retrieval import SearchHit


def hit(score: float) -> SearchHit:
    return SearchHit("a", "policy", 1, "t", "synthetic", score)


def fake_dataset() -> GoldenDataset:
    cases = [
        GoldenCase(
            case_id="R001",
            split="dev",
            question="Known answer question",
            expected_behavior="answer",
            expected_documents=[{"slug": "policy", "version": 1}],
            tags=["test"],
        ),
        GoldenCase(
            case_id="R002",
            split="dev",
            question="Unknown answer question",
            expected_behavior="abstain",
            expected_documents=[],
            tags=["test"],
        ),
        GoldenCase(
            case_id="R003",
            split="holdout",
            question="Independent known question",
            expected_behavior="answer",
            expected_documents=[{"slug": "policy", "version": 1}],
            tags=["test"],
        ),
        GoldenCase(
            case_id="R004",
            split="holdout",
            question="Independent unknown question",
            expected_behavior="abstain",
            expected_documents=[],
            tags=["test"],
        ),
    ]
    return GoldenDataset(
        schema_version=1,
        dataset_id="test",
        corpus_id="test",
        corpus_sha256="0" * 64,
        label_provenance="test",
        cases=cases,
    )


def test_gate_uses_only_dev_negative_score_and_rejects_ties() -> None:
    data = fake_dataset()
    scores = {
        "Unknown answer question": [hit(0.42)],
        "Independent unknown question": [hit(0.99)],  # must not influence cutoff
    }
    threshold = calibrate_dev_negatives(data, scores)
    assert threshold.value == math.nextafter(0.42, math.inf)
    assert threshold.dev_negative_cases == 1
    assert gated_hits([hit(0.42)], threshold=threshold.value, top_k=5) == []
    assert gated_hits([hit(0.55)], threshold=threshold.value, top_k=5)
    assert gated_hits([hit(0.99)], threshold=threshold.value, top_k=5)


def test_gate_fails_closed_without_negative_observation_or_with_invalid_score() -> None:
    data = fake_dataset()
    threshold = calibrate_dev_negatives(data, {"Unknown answer question": []})
    assert threshold.value == math.nextafter(1.0, math.inf)
    assert gated_hits([hit(0.8)], threshold=threshold.value, top_k=5) == []
    with pytest.raises(ValueError, match="Missing retrieval"):
        calibrate_dev_negatives(data, {})
    with pytest.raises(ValueError, match="Cosine scores"):
        calibrate_dev_negatives(data, {"Unknown answer question": [hit(math.nan)]})
    with pytest.raises(ValueError, match="Threshold"):
        gated_hits([hit(0.8)], threshold=math.nan, top_k=5)


def test_gate_is_model_agnostic_and_does_not_generate_answers() -> None:
    assert MODEL_KEY.startswith("Qdrant/")
    assert gated_hits([], threshold=0.4, top_k=5) == []
