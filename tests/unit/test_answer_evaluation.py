"""Composed full-answer fixtures: deterministic coverage, citation and verdict gates."""

import json
import sys
from pathlib import Path

import pytest

from aql.answer_evaluation import evaluate_answers, load_answers, render_answer
from aql.answer_runner import main
from aql.claim_evaluation import (
    ClaimPrediction,
    PredictionFile,
    citation_presence_baseline,
    load_claims,
)

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "data/evals/answer_cases_v1.json"


def locked():
    claims = load_claims(
        ROOT / "data/evals/claim_pairs_v1.json",
        ROOT / "data/evals/evidence_stress_v1.json",
        ROOT / "data/evals/retrieval_stress_v1.json",
        ROOT / "data/synthetic_docs/corpus_v1.json",
    )
    return load_answers(DATASET, claims)


def report_with_authored_claim_labels():
    answers = locked()
    predictions = [
        ClaimPrediction(case_id=case.case_id, label=case.expected_label)
        for case in answers.claims.data.cases
    ]
    return evaluate_answers(answers, predictions, strategy="reference-only")


def test_answer_compositions_are_bound_to_claims_and_complete_question_set() -> None:
    answers = locked()
    assert len(answers.data.answers) == 21
    assert answers.data.claims_dataset_sha256 == answers.claims.dataset_sha256
    assert len(answers.requirements) == 6
    assert {case.question_ref for case in answers.data.answers} == {
        case.case_id for case in answers.claims.evidence.data.cases
    }
    assert "not actual LLM outputs" in answers.data.label_provenance


def test_rendered_answers_expose_every_asserted_claim_and_structured_citation() -> None:
    answers = locked()
    by_id = {case.answer_id: case for case in answers.data.answers}
    assert render_answer(answers, by_id["A001"]).startswith(
        "HTTP 429 and HTTP 503 are retryable webhook responses. [R101:0]"
    )
    assert render_answer(answers, by_id["A009"]).endswith("[R105:0] [R105:1]")
    assert "sources do not provide enough evidence" in render_answer(answers, by_id["A019"])


def test_reference_analysis_exposes_all_expected_structural_failures() -> None:
    report = report_with_authored_claim_labels()
    summary = report["summary"]
    assert summary == {
        "cases": 21,
        "reference_accepted": 12,
        "strategy_accepted": 12,
        "decision_agreement": 21,
        "false_approvals": 0,
        "false_rejections": 0,
        "correct_abstentions": 6,
        "unanswerable_answers": 2,
        "answerable_abstentions": 1,
    }
    cases = {row["answer_id"]: row for row in report["cases"]}
    assert cases["A002"]["all_citations_align_to_labeled_sources"] is True
    assert cases["A002"]["all_claims_supported_by_authored_labels"] is False
    assert cases["A005"]["required_claim_coverage"] is False
    assert cases["A010"]["all_citations_align_to_labeled_sources"] is False
    assert cases["A008"]["reference_accepted"] is False
    assert cases["A015"]["expected_behavior"] == "abstain"
    assert cases["A021"]["required_claim_coverage"] is True
    assert cases["A021"]["all_claims_supported_by_authored_labels"] is False
    assert cases["A003"]["reference_accepted"] is True


def test_naive_citation_only_accepts_both_unsupported_full_answers() -> None:
    answers = locked()
    report = evaluate_answers(
        answers, citation_presence_baseline(answers.claims), strategy="intentionally-unsafe"
    )
    assert report["summary"]["decision_agreement"] == 19
    assert report["summary"]["false_approvals"] == 2
    assert {
        row["answer_id"]
        for row in report["cases"]
        if row["strategy_accepted"] and not row["reference_accepted"]
    } == {"A002", "A021"}


def test_missing_or_duplicate_predictions_and_blank_strategy_are_rejected() -> None:
    answers = locked()
    initial = citation_presence_baseline(answers.claims)
    with pytest.raises(ValueError, match="every atomic claim once"):
        evaluate_answers(answers, initial[1:], strategy="missing")
    with pytest.raises(ValueError, match="every atomic claim once"):
        evaluate_answers(answers, [*initial[:-1], initial[0]], strategy="duplicate")
    with pytest.raises(ValueError, match="named prediction strategy"):
        evaluate_answers(answers, initial, strategy="  ")


@pytest.mark.parametrize(
    "mutate",
    [
        lambda data: data.update(claims_dataset_sha256="0" * 64),
        lambda data: data["answers"].pop(),
        lambda data: data["answers"][1].update(answer_id="A001"),
        lambda data: data["answers"][0]["statements"][0].update(claim_id="C999"),
        lambda data: data["answers"][0]["statements"][0]["citations"][0].update(span_index=99),
        lambda data: data["answers"][0]["statements"][0]["citations"][0].update(case_id="R999"),
        lambda data: data["answers"][0].update(mode="abstain"),
        lambda data: data["answers"][0]["statements"].append(data["answers"][0]["statements"][0]),
        lambda data: data["answers"][0]["statements"][0]["citations"].append(
            data["answers"][0]["statements"][0]["citations"][0]
        ),
        lambda data: data["question_requirements"][0]["required_claim_ids"].append("C999"),
        lambda data: data["question_requirements"].pop(),
        lambda data: data["answers"][0].update(question_ref="R999"),
        lambda data: data["answers"][0].update(unexpected=True),
    ],
)
def test_damaged_answer_fixture_is_rejected(tmp_path: Path, mutate) -> None:
    answers = locked()
    payload = json.loads(DATASET.read_text(encoding="utf-8"))
    mutate(payload)
    changed = tmp_path / "altered.json"
    changed.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        load_answers(changed, answers.claims)


def test_answer_runner_reports_only_authored_analysis_and_optional_predictions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "answer-report.json"
    monkeypatch.setattr(sys, "argv", ["aql.answer_runner", "--output", str(output)])
    main()
    original = output.read_bytes()
    report = json.loads(original)
    assert set(report) == {"schema_version", "reference", "citation_baseline"}
    assert report["citation_baseline"]["summary"]["false_approvals"] == 2
    assert "not LLM outputs" in capsys.readouterr().out
    main()
    assert output.read_bytes() == original

    answers = locked()
    predictions = PredictionFile(
        schema_version=1,
        dataset_sha256=answers.claims.dataset_sha256,
        predictions=citation_presence_baseline(answers.claims),
    )
    source = tmp_path / "predictions.json"
    source.write_text(predictions.model_dump_json(), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        ["aql.answer_runner", "--output", str(output), "--predictions", str(source)],
    )
    main()
    assert json.loads(output.read_bytes())["external"]["summary"]["false_approvals"] == 2
    source.write_text(
        predictions.model_copy(update={"dataset_sha256": "0" * 64}).model_dump_json(),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit, match="2"):
        main()
