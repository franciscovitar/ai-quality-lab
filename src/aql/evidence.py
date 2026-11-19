"""Source-anchored evidence labels and deterministic evidence-selection checks.

This module checks *where evidence came from*, not whether arbitrary natural-language
claims are entailed by it. A valid quotation is not proof of answer correctness.
"""

import hashlib
import json
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from aql.chunking import CHUNKER_VERSION, chunk_paragraphs, normalize_content
from aql.corpus import NAMESPACE
from aql.evaluation import LockedDataset, load_dataset
from aql.retrieval import SearchHit


class EvidenceSpan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    document_slug: str = Field(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
    document_version: int = Field(ge=1)
    start_offset: int = Field(ge=0)
    quote: str = Field(min_length=10, max_length=1000)

    @property
    def end_offset(self) -> int:
        return self.start_offset + len(self.quote)

    @property
    def source_key(self) -> str:
        return f"{self.document_slug}@v{self.document_version}"


class EvidenceCase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str = Field(pattern=r"^R[0-9]{3}$")
    expected_behavior: Literal["answer", "abstain"]
    reference_answer: str | None = None
    abstain_reason: (
        Literal["missing_numeric_detail", "missing_identity", "unsafe_request", "missing_secret"]
        | None
    ) = None
    support_spans: tuple[EvidenceSpan, ...]

    @model_validator(mode="after")
    def validate_label(self) -> "EvidenceCase":
        if self.expected_behavior == "answer":
            if not self.reference_answer or not self.reference_answer.strip():
                raise ValueError("Answerable cases require a reference answer for human review")
            if not self.support_spans or self.abstain_reason is not None:
                raise ValueError("Answerable cases require spans and no abstain reason")
        elif self.support_spans or self.reference_answer is not None or not self.abstain_reason:
            raise ValueError("Unanswerable cases must have only an abstain reason")
        if len(set(self.support_spans)) != len(self.support_spans):
            raise ValueError("Duplicate support spans")
        return self


class EvidenceDataset(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1]
    dataset_id: str = Field(min_length=1)
    retrieval_dataset_id: str = Field(min_length=1)
    retrieval_dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    corpus_id: str = Field(min_length=1)
    corpus_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    label_provenance: str = Field(min_length=1)
    cases: tuple[EvidenceCase, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_order(self) -> "EvidenceDataset":
        ids = [case.case_id for case in self.cases]
        if ids != sorted(set(ids)):
            raise ValueError("Evidence case IDs must be unique and sorted")
        return self


@dataclass(frozen=True)
class KnownChunk:
    chunk_id: str
    source_key: str
    document_slug: str
    document_version: int
    start_offset: int
    end_offset: int
    content: str


@dataclass(frozen=True)
class AnchoredSpan:
    span: EvidenceSpan
    chunk_id: str


@dataclass(frozen=True)
class LockedEvidence:
    data: EvidenceDataset
    dataset_sha256: str
    retrieval: LockedDataset
    chunks_by_id: dict[str, KnownChunk]
    anchors_by_case: dict[str, tuple[AnchoredSpan, ...]]


def load_evidence(evidence_path: Path, retrieval_path: Path, corpus_path: Path) -> LockedEvidence:
    """Fail closed on changed datasets, stale sources, incorrect quotes or bad offsets."""
    retrieval = load_dataset(retrieval_path, corpus_path)
    evidence_bytes = evidence_path.read_bytes()
    data = EvidenceDataset.model_validate_json(evidence_bytes)
    if (
        data.retrieval_dataset_id != retrieval.data.dataset_id
        or data.retrieval_dataset_sha256 != retrieval.dataset_sha256
        or data.corpus_id != retrieval.data.corpus_id
        or data.corpus_sha256 != retrieval.data.corpus_sha256
    ):
        raise ValueError("Evidence labels do not match the locked retrieval dataset and corpus")

    goldens = {case.case_id: case for case in retrieval.data.cases}
    if set(goldens) != {case.case_id for case in data.cases}:
        raise ValueError("Evidence labels must cover every retrieval case exactly once")

    corpus = json.loads(corpus_path.read_bytes())
    chunks_by_id: dict[str, KnownChunk] = {}
    chunks_by_source: dict[str, tuple[KnownChunk, ...]] = {}
    content_by_source: dict[str, str] = {}
    for document in corpus["documents"]:
        if not document["is_current"]:
            continue
        slug, version = document["slug"], document["version"]
        source_key = f"{slug}@v{version}"
        content = normalize_content(document["content"])
        content_by_source[source_key] = content
        source_chunks = []
        for paragraph in chunk_paragraphs(content):
            chunk_id = str(
                uuid.uuid5(
                    NAMESPACE, f"chunk:{slug}:v{version}:{CHUNKER_VERSION}:{paragraph.ordinal}"
                )
            )
            chunk = KnownChunk(
                chunk_id,
                source_key,
                slug,
                version,
                paragraph.start_offset,
                paragraph.end_offset,
                paragraph.content,
            )
            if chunk_id in chunks_by_id:
                raise ValueError("Chunk identity collision in the source corpus")
            chunks_by_id[chunk_id] = chunk
            source_chunks.append(chunk)
        chunks_by_source[source_key] = tuple(source_chunks)

    anchors_by_case: dict[str, tuple[AnchoredSpan, ...]] = {}
    for case in data.cases:
        golden = goldens[case.case_id]
        if case.expected_behavior != golden.expected_behavior:
            raise ValueError(f"{case.case_id}: evidence/retrieval behavior disagrees")
        relevant_sources = {label.key for label in golden.expected_documents}
        anchored = []
        for span in case.support_spans:
            if span.source_key not in relevant_sources:
                raise ValueError(f"{case.case_id}: support must come from labeled current sources")
            source = content_by_source.get(span.source_key)
            if source is None or source[span.start_offset : span.end_offset] != span.quote:
                raise ValueError(f"{case.case_id}: evidence quote/offset does not match the source")
            containing = [
                chunk
                for chunk in chunks_by_source[span.source_key]
                if chunk.start_offset <= span.start_offset and span.end_offset <= chunk.end_offset
            ]
            if len(containing) != 1:
                raise ValueError(f"{case.case_id}: evidence must fit exactly one source chunk")
            anchored.append(AnchoredSpan(span, containing[0].chunk_id))
        anchors_by_case[case.case_id] = tuple(anchored)

    return LockedEvidence(
        data,
        hashlib.sha256(evidence_bytes).hexdigest(),
        retrieval,
        chunks_by_id,
        anchors_by_case,
    )


def _checked_hits(
    evidence: LockedEvidence, hits: Sequence[SearchHit], *, top_k: int
) -> dict[str, KnownChunk]:
    if len(hits) > top_k or len({hit.chunk_id for hit in hits}) != len(hits):
        raise ValueError("Retrieval returned too many or duplicate chunks")
    checked = {}
    for hit in hits:
        chunk = evidence.chunks_by_id.get(hit.chunk_id)
        if chunk is None or (
            hit.document_slug != chunk.document_slug
            or hit.document_version != chunk.document_version
            or hit.content != chunk.content
        ):
            raise ValueError("Retrieval returned an unknown, stale or tampered source chunk")
        checked[hit.chunk_id] = chunk
    return checked


def evaluate_span_retrieval(
    evidence: LockedEvidence,
    retrieve: Callable[[str, int], Sequence[SearchHit]],
    *,
    strategy: str,
    top_k: int = 5,
) -> dict:
    """Measure retrieval of *labeled exact source spans*, not answer correctness."""
    if not 1 <= top_k <= 10 or not strategy.strip():
        raise ValueError("A named strategy and top_k between 1 and 10 are required")
    goldens = {case.case_id: case for case in evidence.retrieval.data.cases}
    rows = []
    for case in evidence.data.cases:
        hits = tuple(retrieve(goldens[case.case_id].question, top_k))
        retrieved = _checked_hits(evidence, hits, top_k=top_k)
        anchors = evidence.anchors_by_case[case.case_id]
        covered = [anchor for anchor in anchors if anchor.chunk_id in retrieved]
        answerable = case.expected_behavior == "answer"
        rows.append(
            {
                "case_id": case.case_id,
                "expected_behavior": case.expected_behavior,
                "retrieved_chunks": len(hits),
                "required_spans": len(anchors),
                "covered_spans": len(covered),
                "span_recall_at_k": len(covered) / len(anchors) if answerable else None,
                "full_span_coverage": len(covered) == len(anchors) if answerable else None,
                "no_hit": not hits if not answerable else None,
            }
        )
    positives = [row for row in rows if row["expected_behavior"] == "answer"]
    negatives = [row for row in rows if row["expected_behavior"] == "abstain"]
    return {
        "schema_version": 1,
        "evidence_dataset_id": evidence.data.dataset_id,
        "evidence_dataset_sha256": evidence.dataset_sha256,
        "retrieval_dataset_sha256": evidence.retrieval.dataset_sha256,
        "corpus_sha256": evidence.data.corpus_sha256,
        "strategy": strategy,
        "top_k_chunks": top_k,
        "label_provenance": evidence.data.label_provenance,
        "summary": {
            "cases": len(rows),
            "answerable": len(positives),
            "unanswerable": len(negatives),
            "required_spans": sum(row["required_spans"] for row in positives),
            "covered_spans": sum(row["covered_spans"] for row in positives),
            "span_recall_at_k": round(
                sum(row["covered_spans"] for row in positives)
                / sum(row["required_spans"] for row in positives),
                6,
            ),
            "full_span_coverage_rate": round(
                sum(row["full_span_coverage"] for row in positives) / len(positives), 6
            ),
            "unanswerable_no_hit_rate": round(
                sum(row["no_hit"] for row in negatives) / len(negatives), 6
            ),
        },
        "cases": rows,
    }


class Citation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk_id: str = Field(min_length=1)
    start_offset: int = Field(ge=0)
    quote: str = Field(min_length=1, max_length=2000)

    @property
    def end_offset(self) -> int:
        return self.start_offset + len(self.quote)


class EvidenceSelection(BaseModel):
    """Structured evidence decision. No generated answer text is graded here."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str = Field(pattern=r"^R[0-9]{3}$")
    decision: Literal["answer", "abstain"]
    citations: tuple[Citation, ...]


def grade_evidence_selection(
    evidence: LockedEvidence,
    selection: EvidenceSelection,
    retrieved_hits: Sequence[SearchHit],
) -> dict:
    """Verify cited exact bytes/offsets, retrieval origin and gold-span coverage.

    This only checks a structured decision plus source selection. It cannot judge
    whether a generated claim is entailed by the quote or actually used the quote.
    """
    labels = {case.case_id: case for case in evidence.data.cases}
    if selection.case_id not in labels:
        raise ValueError("Unknown evidence case")
    checked = _checked_hits(evidence, retrieved_hits, top_k=10)
    case = labels[selection.case_id]
    anchors = evidence.anchors_by_case[selection.case_id]
    covered = set()
    issues = []
    for citation in selection.citations:
        chunk = checked.get(citation.chunk_id)
        relative_start = citation.start_offset - chunk.start_offset if chunk else 0
        relative_end = citation.end_offset - chunk.start_offset if chunk else 0
        if (
            chunk is None
            or not chunk.start_offset <= citation.start_offset
            or citation.end_offset > chunk.end_offset
            or chunk.content[relative_start:relative_end] != citation.quote
        ):
            issues.append("invalid_or_unretrieved_citation")
            continue
        matches = [
            n
            for n, anchor in enumerate(anchors)
            if anchor.chunk_id == citation.chunk_id
            and citation.start_offset <= anchor.span.start_offset
            and citation.end_offset >= anchor.span.end_offset
        ]
        if not matches:
            issues.append("authentic_but_unlabeled_evidence")
        covered.update(matches)

    if selection.decision != case.expected_behavior:
        issues.append("wrong_answerability_decision")
    if selection.decision == "abstain" and selection.citations:
        issues.append("citation_on_abstention")
    if selection.decision == "answer" and len(covered) != len(anchors):
        issues.append("missing_required_evidence")

    return {
        "case_id": selection.case_id,
        "decision_correct": selection.decision == case.expected_behavior,
        "required_spans": len(anchors),
        "covered_spans": len(covered),
        "all_citations_labeled_and_authentic": not any(
            issue in {"invalid_or_unretrieved_citation", "authentic_but_unlabeled_evidence"}
            for issue in issues
        ),
        "passed": not issues,
        "issues": sorted(set(issues)),
    }
