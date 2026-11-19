"""NLI dataset and adapter contract without importing or downloading model dependencies."""

import json
import sys
from pathlib import Path

import pytest

from aql.claim_evaluation import load_claims
from aql.nli import (
    EXPECTED_LABELS,
    MODEL_KEY,
    NliObservation,
    PinnedNliCrossEncoder,
    predicted_nli_label,
    premise_for_claim,
    validate_model_labels,
)
from aql.nli_runner import main

ROOT = Path(__file__).resolve().parents[2]


def goldens():
    return load_claims(
        ROOT / "data/evals/claim_pairs_v1.json",
        ROOT / "data/evals/evidence_stress_v1.json",
        ROOT / "data/evals/retrieval_stress_v1.json",
        ROOT / "data/synthetic_docs/corpus_v1.json",
    )


def test_source_premise_contains_exact_citations_but_not_gold_label_or_rationale() -> None:
    claims = goldens()
    for case in claims.data.cases:
        expected = "\n".join(anchor.span.quote for anchor in claims.anchors_by_claim[case.case_id])
        premise = premise_for_claim(claims, case.case_id)
        assert premise == expected
        assert case.rationale not in premise
    assert "\n" in premise_for_claim(claims, "C014")
    with pytest.raises(KeyError):
        premise_for_claim(claims, "C999")


def test_pinned_model_mapping_is_verified_not_guessed() -> None:
    valid = {
        "model_type": "roberta",
        "pad_token_id": 1,
        "id2label": {str(key): value for key, value in EXPECTED_LABELS.items()},
    }
    validate_model_labels(valid)
    for change in (
        {"id2label": {"0": "LABEL_0", "1": "LABEL_1", "2": "LABEL_2"}},
        {"id2label": {"0": "entailment", "1": "contradiction", "2": "neutral"}},
        {"pad_token_id": 0},
        {"model_type": "bert"},
    ):
        with pytest.raises(ValueError, match="configuration is incompatible"):
            validate_model_labels(valid | change)


@pytest.mark.parametrize(
    ("scores", "expected"),
    [
        ([-2.0, 3.0, 1.0], "entailment"),
        ([5.0, 1.0, 2.0], "contradiction"),
        ([-1.0, 0.5, 6.5], "neutral"),
        ([3.0, 3.0, 0.0], "contradiction"),
    ],
)
def test_three_class_prediction_is_deterministic(scores, expected) -> None:
    assert predicted_nli_label(scores) == expected


@pytest.mark.parametrize("scores", [[], [1.0, 2.0], [1.0, 2.0, 3.0, 4.0], [0, 1, float("nan")]])
def test_bad_logits_fail_closed(scores) -> None:
    with pytest.raises(ValueError):
        predicted_nli_label(scores)


def test_only_entailment_counts_as_support() -> None:
    for label in EXPECTED_LABELS.values():
        observation = NliObservation("C001", label, 53, (1.2, 1.3, 0.1))
        expected = "entailed" if label == "entailment" else "not_entailed"
        assert observation.prediction.label == expected
        assert observation.as_dict() == {
            "case_id": "C001",
            "nli_label": label,
            "token_count": 53,
            "logits": [1.2, 1.3, 0.1],
        }


def test_runner_scores_only_supplied_model_output_and_records_provenance(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from aql import nli_runner

    seen = []

    class FakeModel:
        model_key = MODEL_KEY

        def __init__(self, *, allow_download: bool):
            seen.append(allow_download)

        def infer(self, claims):
            # Deliberately predict unsupported for *every* case, without gold access.
            return [
                NliObservation(case.case_id, "neutral", 15, (0.0, 0.0, 1.0))
                for case in claims.data.cases
            ]

    monkeypatch.setattr(nli_runner, "PinnedNliCrossEncoder", FakeModel)
    output = tmp_path / "nli.json"
    monkeypatch.setattr(sys, "argv", ["aql.nli_runner", "--output", str(output)])
    main()
    first_bytes = output.read_bytes()
    report = json.loads(first_bytes)
    assert seen == [False]
    assert report["summary"]["correct"] == 12
    assert report["summary"]["false_supported_rate"] == 0.0
    assert len(report["model_observations"]) == 24
    assert report["model"]["snapshot"] == MODEL_KEY
    assert report["model"]["truncation"].startswith("reject")
    assert "12/24" in capsys.readouterr().out
    main()
    assert output.read_bytes() == first_bytes
    monkeypatch.setattr(
        sys,
        "argv",
        ["aql.nli_runner", "--output", str(output), "--download-model"],
    )
    main()
    assert seen == [False, False, True]


def test_actual_model_requires_explicit_opt_in_and_optional_dependencies() -> None:
    assert MODEL_KEY.endswith("/onnx/model.onnx")
    assert PinnedNliCrossEncoder.__init__.__kwdefaults__ == {
        "allow_download": False,
        "cache_dir": None,
    }
