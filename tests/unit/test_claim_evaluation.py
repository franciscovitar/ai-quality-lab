"""Gold claim labels, safe provenance and an intentionally unsafe citation baseline."""

import json
import sys
from pathlib import Path

import pytest

from aql.claim_evaluation import (
    ClaimPrediction,
    PredictionFile,
    citation_presence_baseline,
    evaluate_claim_predictions,
    load_claims,
)
from aql.claim_runner import main
from aql.evaluation import report_json

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "data/synthetic_docs/corpus_v1.json"
RETRIEVAL = ROOT / "data/evals/retrieval_stress_v1.json"
EVIDENCE = ROOT / "data/evals/evidence_stress_v1.json"
CLAIMS = ROOT / "data/evals/claim_pairs_v1.json"


def loaded():
    return load_claims(CLAIMS, EVIDENCE, RETRIEVAL, CORPUS)


def test_claims_are_hash_locked_with_both_classes_in_each_split() -> None:
    goldens = loaded()
    assert len(goldens.data.cases) == 24
    assert goldens.data.evidence_dataset_sha256 == goldens.evidence.dataset_sha256
    assert len(goldens.anchors_by_claim["C014"]) == 2
    for split in ("dev", "holdout"):
        cases = [case for case in goldens.data.cases if case.split == split]
        assert len(cases) == 12
        assert sum(case.expected_label == "entailed" for case in cases) == 6
        assert sum(case.expected_label == "not_entailed" for case in cases) == 6

    # Both cases cite the *same authentic paragraph*, but only one claim follows.
    yes, no = goldens.data.cases[:2]
    assert yes.evidence_refs == no.evidence_refs
    assert yes.expected_label == "entailed"
    assert no.expected_label == "not_entailed"
    assert "not blinded or independently adjudicated" in goldens.data.label_provenance


def test_citation_presence_is_a_measurably_unsafe_baseline() -> None:
    goldens = loaded()
    predictions = citation_presence_baseline(goldens)
    report = evaluate_claim_predictions(goldens, predictions, strategy="citation-presence")
    summary = report["summary"]
    assert summary["cases"] == 24
    assert summary["correct"] == 12
    assert summary["accuracy"] == 0.5
    assert summary["entailed_precision"] == 0.5
    assert summary["entailed_recall"] == 1.0
    assert summary["not_entailed_recall"] == 0.0
    assert summary["false_supported_rate"] == 1.0
    assert summary["macro_f1"] == 0.333333
    assert summary["confusion"] == {
        "entailed_as_entailed": 12,
        "entailed_as_not_entailed": 0,
        "not_entailed_as_entailed": 12,
        "not_entailed_as_not_entailed": 0,
    }
    assert all(slice_report["correct"] == 6 for slice_report in report["by_split"].values())
    assert all(
        slice_report["false_supported_rate"] == 1.0
        for slice_report in report["by_negative_error_type"].values()
    )


def test_oracle_fixture_can_score_perfectly_but_is_not_a_model_result() -> None:
    goldens = loaded()
    oracle = [
        ClaimPrediction(case_id=case.case_id, label=case.expected_label)
        for case in goldens.data.cases
    ]
    report = evaluate_claim_predictions(goldens, oracle, strategy="test-oracle")
    assert report["summary"]["correct"] == 24
    assert report["summary"]["macro_f1"] == 1.0
    assert report["summary"]["false_supported_rate"] == 0.0
    assert report_json(report) == report_json(
        evaluate_claim_predictions(goldens, oracle, strategy="test-oracle")
    )


@pytest.mark.parametrize("change", ["missing", "duplicate", "unknown"])
def test_prediction_set_must_cover_exactly_all_cases(change: str) -> None:
    goldens = loaded()
    predictions = citation_presence_baseline(goldens)
    if change == "missing":
        predictions.pop()
    elif change == "duplicate":
        predictions[-1] = predictions[0]
    else:
        predictions[-1] = ClaimPrediction(case_id="C999", label="entailed")
    with pytest.raises(ValueError, match="cover each claim exactly once"):
        evaluate_claim_predictions(goldens, predictions, strategy="invalid")


@pytest.mark.parametrize(
    "alter",
    [
        lambda data: data.update(evidence_dataset_sha256="0" * 64),
        lambda data: data["cases"][0]["evidence_refs"][0].update(case_id="R112"),
        lambda data: data["cases"][0]["evidence_refs"][0].update(span_index=9),
        lambda data: data["cases"][0].update(error_type="missing_detail"),
        lambda data: data["cases"][1].pop("error_type"),
        lambda data: data["cases"][0]["evidence_refs"].append(data["cases"][0]["evidence_refs"][0]),
        lambda data: data["cases"][0].update(case_id="C002"),
        lambda data: data["cases"].pop(),
        lambda data: data["cases"][0].update(extra_field="not allowed"),
    ],
)
def test_invalid_claim_dataset_is_rejected(tmp_path: Path, alter) -> None:
    data = json.loads(CLAIMS.read_text(encoding="utf-8"))
    alter(data)
    changed = tmp_path / "invalid-claims.json"
    changed.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError):
        load_claims(changed, EVIDENCE, RETRIEVAL, CORPUS)


def test_runner_writes_deterministic_baseline_report(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "claim-support-report.json"
    monkeypatch.setattr(sys, "argv", ["aql.claim_runner", "--output", str(output)])
    main()
    first = output.read_bytes()
    main()
    assert output.read_bytes() == first
    assert b'"false_supported_rate": 1.0' in first
    assert "false-supported=12/12" in capsys.readouterr().out


def test_imported_predictions_are_fingerprint_bound(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    goldens = loaded()
    predictions = citation_presence_baseline(goldens)
    prediction_file = tmp_path / "predictions.json"
    output = tmp_path / "evaluation.json"
    payload = PredictionFile(
        schema_version=1, dataset_sha256=goldens.dataset_sha256, predictions=predictions
    )
    prediction_file.write_text(payload.model_dump_json(), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "aql.claim_runner",
            "--predictions",
            str(prediction_file),
            "--output",
            str(output),
        ],
    )
    main()
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["summary"]["false_supported_rate"] == 1.0
    assert report["strategy"] == "externally-supplied-claim-verdicts-v1"

    prediction_file.write_text(
        payload.model_copy(update={"dataset_sha256": "0" * 64}).model_dump_json(),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit, match="2"):
        main()
