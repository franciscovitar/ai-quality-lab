"""Source-locked reviewer packet, anti-leakage and agreement edge-case tests."""

import json
import sys
from copy import deepcopy

import pytest

from aql.evaluation import report_json
from aql.human_review import (
    ReviewPacket,
    blank_submission_template,
    build_review_packet,
    compare_submissions,
    label_agreement,
    packet_json,
    packet_sha256,
    validate_submission,
)
from aql.human_review_runner import main


@pytest.fixture(scope="module")
def packet():
    return build_review_packet()


def synthetic_completed_submission(packet, reviewer_id="reviewer_a"):
    """An explicitly fake TEST fixture, never an independent human review."""
    result = blank_submission_template(packet, reviewer_id)
    for case in packet.cases:
        for claim in case.claims:
            row = next(r for r in result["claims"] if r["claim_id"] == claim.claim_id)
            row["label"] = "supported"
            row["inspected_citations"] = [case.sources[0].citation_id]
            row["rationale"] = "Synthetic test fixture only; explicitly not a human judgment."
        response = next(r for r in result["responses"] if r["case_id"] == case.case_id)
        response["label"] = "addresses_question"
        response["rationale"] = "Synthetic test fixture only; this is not a real reviewer."
    return result


def raw(data):
    return report_json(data).encode("utf-8")


def test_packet_includes_source_quotes_offsets_and_all_claims_without_gold(packet):
    assert len(packet.cases) == 12
    assert sum(len(case.claims) for case in packet.cases) == 15
    assert packet.blinding.endswith("SEGMENTATION_IS_AUTHOR_SUPPLIED")
    for case in packet.cases:
        assert case.question
        assert case.generated_text
        assert case.sources
        for claim in case.claims:
            assert case.generated_text[claim.start_char : claim.end_char] == claim.text
    encoded = packet_json(packet)
    assert "author_verdict" not in encoded
    assert "expected_behavior" not in encoded
    assert "reference_answer" not in encoded
    assert "project_author_provisional" not in encoded
    assert "model_logits" not in encoded
    assert "rationale" not in encoded
    assert encoded == packet_json(build_review_packet())
    assert len(packet_sha256(packet)) == 64


def test_blank_templates_are_deliberately_unsubmitable(packet):
    for key in ("reviewer_a", "reviewer_b"):
        blank = blank_submission_template(packet, key)
        assert len(blank["claims"]) == 15
        assert len(blank["responses"]) == 12
        assert all(row["label"] is None for row in blank["claims"])
        with pytest.raises(ValueError):
            validate_submission(packet, raw(blank))


def test_two_synthetic_test_submissions_compare_without_certifying_any_humans(packet):
    a = synthetic_completed_submission(packet)
    b = synthetic_completed_submission(packet, "reviewer_b")
    comparison = compare_submissions(packet, raw(a), raw(b))
    assert comparison["completed_submission_files"] == 2
    assert comparison["independent_human_identities_verified"] == 0
    assert comparison["adjudication_status"] == "not_performed"
    assert comparison["claims_three_class"]["raw_agreement"] == 1.0
    # Perfect agreement with a single class has zero denominator for kappa.
    assert comparison["claims_three_class"]["cohen_kappa"] is None
    assert not comparison["claim_disagreements"]


def test_disagreement_recorded_and_no_auto_adjudication(packet):
    a = synthetic_completed_submission(packet)
    b = synthetic_completed_submission(packet, "reviewer_b")
    b["claims"][0]["label"] = "not_established"
    b["responses"][0]["label"] = "partial"
    report = compare_submissions(packet, raw(a), raw(b))
    assert len(report["claim_disagreements"]) == 1
    assert report["claim_disagreements"][0]["claim_id"] == "R101/C01"
    assert len(report["response_disagreements"]) == 1
    assert report["claims_three_class"]["raw_agreement"] == round(14 / 15, 6)
    assert report["adjudication_status"] == "not_performed"
    assert report["independent_human_identities_verified"] == 0


def test_cohens_kappa_observed_and_expected_agreement():
    labels = ("supported", "contradicted", "not_established")
    perfect = label_agreement(
        ["supported", "contradicted", "not_established"],
        ["supported", "contradicted", "not_established"],
        labels,
    )
    assert perfect["raw_agreement"] == 1.0
    assert perfect["expected_chance_agreement"] == round(1 / 3, 6)
    assert perfect["cohen_kappa"] == 1.0
    chance = label_agreement(
        ["supported", "supported", "contradicted", "contradicted"],
        ["supported", "contradicted", "contradicted", "supported"],
        labels,
    )
    assert chance["raw_agreement"] == 0.5
    assert chance["expected_chance_agreement"] == 0.5
    assert chance["cohen_kappa"] == 0.0


@pytest.mark.parametrize(
    "a,b",
    [([], []), (["supported"], []), (["supported"], ["unknown"])],
)
def test_kappa_rejects_invalid_sequences(a, b):
    with pytest.raises(ValueError):
        label_agreement(a, b, ("supported", "not_supported"))


@pytest.mark.parametrize(
    "mutation,expected",
    [
        ("wrong_packet", "different frozen packet"),
        ("missing_claim", "every claim exactly once"),
        ("duplicate_claim", "every claim exactly once"),
        ("wrong_claim_order", "every claim exactly once"),
        ("missing_response", "every answer exactly once"),
        ("unseen_citation", "not supplied to the generator"),
        ("no_rationale", "rationale"),
        ("extra_field", "Extra inputs"),
    ],
)
def test_incomplete_or_tampered_reviewer_submissions_fail(packet, mutation, expected):
    data = deepcopy(synthetic_completed_submission(packet))
    if mutation == "wrong_packet":
        data["packet_sha256"] = "a" * 64
    elif mutation == "missing_claim":
        data["claims"].pop()
    elif mutation == "duplicate_claim":
        data["claims"][1]["claim_id"] = data["claims"][0]["claim_id"]
    elif mutation == "wrong_claim_order":
        data["claims"][0], data["claims"][1] = data["claims"][1], data["claims"][0]
    elif mutation == "missing_response":
        data["responses"].pop()
    elif mutation == "unseen_citation":
        data["claims"][0]["inspected_citations"] = ["R112:9"]
    elif mutation == "no_rationale":
        data["claims"][0]["rationale"] = ""
    else:
        data["claims"][0]["gold_label"] = "supported"
    with pytest.raises(ValueError, match=expected):
        validate_submission(packet, raw(data))


def test_same_reviewer_identifier_is_not_two_reviewers(packet):
    a = synthetic_completed_submission(packet)
    with pytest.raises(ValueError, match="identifiers must differ"):
        compare_submissions(packet, raw(a), raw(a))


def test_modified_packet_claim_offsets_rejected(packet):
    altered = packet.model_dump(mode="json")
    altered["cases"][0]["claims"][0]["start_char"] += 1
    with pytest.raises(ValueError, match="offsets"):
        ReviewPacket.model_validate(altered)


def test_cli_prepare_and_compare_require_finished_submissions(packet, monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["human_review_runner", "prepare", "--out-dir", str(tmp_path)])
    main()
    generated_packet = json.loads((tmp_path / "packet.json").read_text())
    assert generated_packet["packet_id"] == packet.packet_id
    a = synthetic_completed_submission(packet)
    b = synthetic_completed_submission(packet, "reviewer_b")
    left, right = tmp_path / "left.json", tmp_path / "right.json"
    left.write_bytes(raw(a))
    right.write_bytes(raw(b))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "human_review_runner",
            "compare",
            "--out-dir",
            str(tmp_path),
            "--left",
            str(left),
            "--right",
            str(right),
        ],
    )
    main()
    comparison = json.loads((tmp_path / "reviewer-agreement.json").read_text())
    assert comparison["independent_human_identities_verified"] == 0
