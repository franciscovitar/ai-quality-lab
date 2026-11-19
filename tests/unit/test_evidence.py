"""Deterministic span/source provenance checks; never LLM answer entailment."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from aql.evidence import (
    Citation,
    EvidenceSelection,
    evaluate_span_retrieval,
    grade_evidence_selection,
    load_evidence,
)
from aql.retrieval import SearchHit

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "data/synthetic_docs/corpus_v1.json"
GOLDENS = ROOT / "data/evals/retrieval_stress_v1.json"
EVIDENCE = ROOT / "data/evals/evidence_stress_v1.json"


def loaded():
    return load_evidence(EVIDENCE, GOLDENS, CORPUS)


def search_hit(bundle, chunk_id: str) -> SearchHit:
    chunk = bundle.chunks_by_id[chunk_id]
    return SearchHit(
        chunk_id=chunk.chunk_id,
        document_slug=chunk.document_slug,
        document_version=chunk.document_version,
        title="synthetic test",
        content=chunk.content,
        score=0.75,
    )


def citations(bundle, case_id: str) -> tuple[Citation, ...]:
    return tuple(
        Citation(
            chunk_id=anchor.chunk_id,
            start_offset=anchor.span.start_offset,
            quote=anchor.span.quote,
        )
        for anchor in bundle.anchors_by_case[case_id]
    )


def test_all_labels_are_locked_and_span_offsets_match_current_chunks() -> None:
    bundle = loaded()
    assert len(bundle.data.cases) == 12
    assert len(bundle.chunks_by_id) == 16  # retired policy chunks excluded
    assert sum(len(spans) for spans in bundle.anchors_by_case.values()) == 8
    assert sum(case.expected_behavior == "abstain" for case in bundle.data.cases) == 6
    assert all(bundle.anchors_by_case[f"R{n}"] for n in range(101, 107))
    assert all(not bundle.anchors_by_case[f"R{n}"] for n in range(107, 113))
    assert "not independently adjudicated" in bundle.data.label_provenance


def test_evidence_span_recall_exposes_partial_support_even_with_correct_document() -> None:
    bundle = loaded()
    by_question = {case.question: case.case_id for case in bundle.retrieval.data.cases}

    def retrieve(question: str, _k: int) -> list[SearchHit]:
        case_id = by_question[question]
        anchors = bundle.anchors_by_case[case_id]
        if case_id == "R102":
            anchors = anchors[:1]  # same relevant document but misses another needed fact
        return [
            search_hit(bundle, chunk_id)
            for chunk_id in dict.fromkeys(anchor.chunk_id for anchor in anchors)
        ]

    report = evaluate_span_retrieval(bundle, retrieve, strategy="unit-controlled", top_k=5)
    assert report["summary"]["required_spans"] == 8
    assert report["summary"]["covered_spans"] == 7
    assert report["summary"]["span_recall_at_k"] == 0.875
    assert report["summary"]["full_span_coverage_rate"] == 0.833333
    assert report["summary"]["unanswerable_no_hit_rate"] == 1.0
    row = next(row for row in report["cases"] if row["case_id"] == "R102")
    assert row["span_recall_at_k"] == 0.5
    assert row["full_span_coverage"] is False


def test_authentic_citations_can_be_verified_without_claiming_entailment() -> None:
    bundle = loaded()
    cited = citations(bundle, "R102")
    hits = [
        search_hit(bundle, chunk_id)
        for chunk_id in dict.fromkeys(citation.chunk_id for citation in cited)
    ]
    complete = EvidenceSelection(case_id="R102", decision="answer", citations=cited)
    grade = grade_evidence_selection(bundle, complete, hits)
    assert grade["passed"] is True
    assert grade["covered_spans"] == 2
    assert grade["issues"] == []

    incomplete = EvidenceSelection(case_id="R102", decision="answer", citations=cited[:1])
    grade = grade_evidence_selection(bundle, incomplete, hits)
    assert grade["passed"] is False
    assert grade["issues"] == ["missing_required_evidence"]


def test_a_topic_match_does_not_override_negative_answerability() -> None:
    bundle = loaded()
    policy_anchor = bundle.anchors_by_case["R101"][0]
    hit = search_hit(bundle, policy_anchor.chunk_id)
    unsupported = EvidenceSelection(
        case_id="R107", decision="answer", citations=citations(bundle, "R101")
    )
    grade = grade_evidence_selection(bundle, unsupported, [hit])
    assert grade["passed"] is False
    assert "wrong_answerability_decision" in grade["issues"]
    assert "authentic_but_unlabeled_evidence" in grade["issues"]

    safe = EvidenceSelection(case_id="R107", decision="abstain", citations=())
    assert grade_evidence_selection(bundle, safe, [hit])["passed"] is True
    misformatted = EvidenceSelection(
        case_id="R107", decision="abstain", citations=citations(bundle, "R101")
    )
    assert (
        "citation_on_abstention" in grade_evidence_selection(bundle, misformatted, [hit])["issues"]
    )


@pytest.mark.parametrize(
    "citation_change",
    [
        lambda c: c.model_copy(update={"quote": "Faked quote that is not in source."}),
        lambda c: c.model_copy(update={"start_offset": c.start_offset + 1}),
        lambda c: c.model_copy(update={"chunk_id": "stale-or-fabricated-chunk"}),
    ],
)
def test_fabricated_or_mislocated_citation_is_rejected(citation_change) -> None:
    bundle = loaded()
    original = citations(bundle, "R101")[0]
    selection = EvidenceSelection(
        case_id="R101", decision="answer", citations=(citation_change(original),)
    )
    result = grade_evidence_selection(bundle, selection, [search_hit(bundle, original.chunk_id)])
    assert result["passed"] is False
    assert "invalid_or_unretrieved_citation" in result["issues"]


def test_not_retrieved_citation_is_not_authorized_by_gold_labels() -> None:
    bundle = loaded()
    selection = EvidenceSelection(
        case_id="R101", decision="answer", citations=citations(bundle, "R101")
    )
    result = grade_evidence_selection(bundle, selection, [])
    assert result["passed"] is False
    assert "invalid_or_unretrieved_citation" in result["issues"]


def test_authentic_but_irrelevant_quote_does_not_count_as_support() -> None:
    bundle = loaded()
    required = citations(bundle, "R101")[0]
    chunk = bundle.chunks_by_id[required.chunk_id]
    other_quote = "A webhook delivery has a maximum budget of five attempts per generation."
    assert chunk.content.startswith(other_quote)
    selected = EvidenceSelection(
        case_id="R101",
        decision="answer",
        citations=(
            Citation(
                chunk_id=chunk.chunk_id,
                start_offset=chunk.start_offset,
                quote=other_quote,
            ),
        ),
    )
    grade = grade_evidence_selection(bundle, selected, [search_hit(bundle, chunk.chunk_id)])
    assert grade["passed"] is False
    assert grade["issues"] == ["authentic_but_unlabeled_evidence", "missing_required_evidence"]


def test_retrieval_with_unknown_or_misidentified_chunks_fails_closed() -> None:
    bundle = loaded()
    question = bundle.retrieval.data.cases[0].question
    legit = search_hit(bundle, bundle.anchors_by_case["R101"][0].chunk_id)
    for invalid in [
        replace(legit, chunk_id="made-up"),
        replace(legit, document_version=1),
        replace(legit, content="tampered"),
    ]:
        with pytest.raises(ValueError, match="unknown, stale or tampered"):
            evaluate_span_retrieval(
                bundle,
                lambda q, _k, candidate=invalid: [candidate] if q == question else [],
                strategy="reject-fakes",
            )
    for results in [[legit, legit], [legit] * 6]:
        with pytest.raises(ValueError, match="duplicate chunks|too many"):
            evaluate_span_retrieval(
                bundle,
                lambda q, _k, candidate=results: candidate if q == question else [],
                strategy="reject-duplicates",
                top_k=5,
            )


@pytest.mark.parametrize(
    "alter",
    [
        lambda data: data.update(retrieval_dataset_sha256="0" * 64),
        lambda data: data["cases"][0]["support_spans"][0].update(start_offset=72),
        lambda data: data["cases"][0]["support_spans"][0].update(quote="not in corpus"),
        lambda data: data["cases"][0]["support_spans"][0].update(document_version=1),
        lambda data: data["cases"][0].update(expected_behavior="abstain"),
        lambda data: data["cases"].pop(),
        lambda data: data["cases"][1]["support_spans"].append(data["cases"][1]["support_spans"][0]),
    ],
)
def test_unreviewed_or_inconsistent_evidence_labels_fail_closed(tmp_path: Path, alter) -> None:
    raw = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    alter(raw)
    changed = tmp_path / "unreviewed-evidence.json"
    changed.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError):
        load_evidence(changed, GOLDENS, CORPUS)


def test_corpus_change_invalidates_evidence_and_retrieval_labels(tmp_path: Path) -> None:
    changed = tmp_path / "changed-corpus.json"
    changed.write_bytes(CORPUS.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="corpus changed"):
        load_evidence(EVIDENCE, GOLDENS, changed)
