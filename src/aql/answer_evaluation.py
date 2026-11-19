"""Compose and score *pre-annotated* synthetic answers, with explicit epistemic limits.

This module does NOT generate answers, split arbitrary prose into claims, or infer
entailment. Its reference truth comes from authored claim labels; an experimental
model may provide independent claim verdicts for comparison after inference.
"""

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from aql.claim_evaluation import ClaimEvidenceRef, ClaimPrediction, LockedClaims

ABSTENTION_TEXT = "The supplied sources do not provide enough evidence to answer this question."


class AnswerStatement(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    claim_id: str = Field(pattern=r"^C[0-9]{3}$")
    citations: tuple[ClaimEvidenceRef, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_citations(self) -> "AnswerStatement":
        if len(set(self.citations)) != len(self.citations):
            raise ValueError("A response sentence must not repeat a citation")
        return self


class AnswerCase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    answer_id: str = Field(pattern=r"^A[0-9]{3}$")
    question_ref: str = Field(pattern=r"^R[0-9]{3}$")
    scenario: str = Field(min_length=12, max_length=120)
    mode: Literal["answer", "abstain"]
    statements: tuple[AnswerStatement, ...]

    @model_validator(mode="after")
    def validate_mode(self) -> "AnswerCase":
        if (self.mode == "answer") != bool(self.statements):
            raise ValueError("Answers require statements; abstentions must contain none")
        claim_ids = [statement.claim_id for statement in self.statements]
        if len(set(claim_ids)) != len(claim_ids):
            raise ValueError("A response must not duplicate a claim ID")
        return self


class QuestionRequirement(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    question_ref: str = Field(pattern=r"^R[0-9]{3}$")
    required_claim_ids: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique(self) -> "QuestionRequirement":
        if tuple(sorted(set(self.required_claim_ids))) != self.required_claim_ids:
            raise ValueError("Reference requirements must be unique and sorted")
        return self


class AnswerDataset(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1]
    dataset_id: str = Field(min_length=1)
    claims_dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    expected_answers: int = Field(ge=1)
    label_provenance: str = Field(min_length=1)
    question_requirements: tuple[QuestionRequirement, ...] = Field(min_length=1)
    answers: tuple[AnswerCase, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_counts(self) -> "AnswerDataset":
        answer_ids = [answer.answer_id for answer in self.answers]
        if len(answer_ids) != self.expected_answers or answer_ids != sorted(set(answer_ids)):
            raise ValueError("Answer fixture IDs/count must be unique, sorted and expected")
        refs = [item.question_ref for item in self.question_requirements]
        if refs != sorted(set(refs)):
            raise ValueError("Question requirements must be unique and sorted")
        return self


@dataclass(frozen=True)
class LockedAnswers:
    data: AnswerDataset
    dataset_sha256: str
    claims: LockedClaims
    requirements: dict[str, frozenset[str]]


def load_answers(path: Path, claims: LockedClaims) -> LockedAnswers:
    """Validate questions, exact citation anchors, required claims and hash chain."""
    content = path.read_bytes()
    data = AnswerDataset.model_validate_json(content)
    if data.claims_dataset_sha256 != claims.dataset_sha256:
        raise ValueError("Answer fixtures refer to a different claim dataset fingerprint")

    claim_by_id = {case.case_id: case for case in claims.data.cases}
    question_by_id = {case.case_id: case for case in claims.evidence.data.cases}
    answerable = {
        case.case_id for case in question_by_id.values() if case.expected_behavior == "answer"
    }
    requirements = {
        item.question_ref: frozenset(item.required_claim_ids) for item in data.question_requirements
    }
    if set(requirements) != answerable:
        raise ValueError("Reference requirements must cover exactly all answerable questions")
    for question_ref, ids in requirements.items():
        for claim_id in ids:
            claim = claim_by_id.get(claim_id)
            if (
                claim is None
                or claim.expected_label != "entailed"
                or {ref.case_id for ref in claim.evidence_refs} != {question_ref}
            ):
                raise ValueError("Required answer claims must be supported by their question")

    if {case.question_ref for case in data.answers} != set(question_by_id):
        raise ValueError("Answer fixtures must exercise every stress question")
    for answer in data.answers:
        if answer.question_ref not in question_by_id:
            raise ValueError(f"{answer.answer_id} points to an unknown question")
        for statement in answer.statements:
            if statement.claim_id not in claim_by_id:
                raise ValueError(f"{answer.answer_id} points to an unknown atomic claim")
            for ref in statement.citations:
                anchors = claims.evidence.anchors_by_case.get(ref.case_id)
                if anchors is None or ref.span_index >= len(anchors):
                    raise ValueError(f"{answer.answer_id} uses an unknown source citation")
    return LockedAnswers(data, hashlib.sha256(content).hexdigest(), claims, requirements)


def render_answer(answers: LockedAnswers, answer: AnswerCase) -> str:
    """Render complete prose from exhaustively annotated, exact atomic sentences."""
    if answer.mode == "abstain":
        return ABSTENTION_TEXT
    by_id = {case.case_id: case for case in answers.claims.data.cases}
    parts = []
    for statement in answer.statements:
        refs = " ".join(f"[{ref.case_id}:{ref.span_index}]" for ref in statement.citations)
        parts.append(f"{by_id[statement.claim_id].claim} {refs}")
    return " ".join(parts)


def evaluate_answers(
    answers: LockedAnswers,
    predictions: Sequence[ClaimPrediction],
    *,
    strategy: str,
) -> dict:
    """Compare complete-answer decisions against the same authored reference labels.

    This is a fixed, exhaustively pre-segmented fixture, NOT autonomous claim
    extraction or an independent judge. Predictions must cover the entire claim set.
    """
    if not strategy.strip():
        raise ValueError("A named prediction strategy is required")
    ids = [item.case_id for item in predictions]
    gold_ids = {item.case_id for item in answers.claims.data.cases}
    if len(ids) != len(set(ids)) or set(ids) != gold_ids:
        raise ValueError("Predictions must cover every atomic claim once")
    predicted = {item.case_id: item.label for item in predictions}
    claims = {case.case_id: case for case in answers.claims.data.cases}
    questions = {case.case_id: case for case in answers.claims.evidence.data.cases}

    rows = []
    for answer in answers.data.answers:
        question = questions[answer.question_ref]
        expected_mode = question.expected_behavior
        response_claim_ids = {statement.claim_id for statement in answer.statements}
        required = answers.requirements.get(answer.question_ref, frozenset())
        coverage = required.issubset(response_claim_ids) if expected_mode == "answer" else None
        aligned = (
            all(
                set(statement.citations) == set(claims[statement.claim_id].evidence_refs)
                for statement in answer.statements
            )
            if answer.mode == "answer"
            else None
        )
        actual_supported = (
            all(
                claims[statement.claim_id].expected_label == "entailed"
                for statement in answer.statements
            )
            if answer.mode == "answer"
            else None
        )
        model_supported = (
            all(predicted[statement.claim_id] == "entailed" for statement in answer.statements)
            if answer.mode == "answer"
            else None
        )
        if expected_mode == "abstain":
            reference_accepted = model_accepted = answer.mode == "abstain"
        elif answer.mode == "abstain":
            reference_accepted = model_accepted = False
        else:
            reference_accepted = bool(coverage and aligned and actual_supported)
            model_accepted = bool(coverage and aligned and model_supported)

        rows.append(
            {
                "answer_id": answer.answer_id,
                "question_ref": answer.question_ref,
                "scenario": answer.scenario,
                "expected_behavior": expected_mode,
                "response_mode": answer.mode,
                "response_text": render_answer(answers, answer),
                "asserted_claim_ids": [item.claim_id for item in answer.statements],
                "required_claim_ids": sorted(required),
                "required_claim_coverage": coverage,
                "all_citations_align_to_labeled_sources": aligned,
                "all_claims_supported_by_authored_labels": actual_supported,
                "all_claims_supported_by_supplied_predictions": model_supported,
                "reference_accepted": reference_accepted,
                "strategy_accepted": model_accepted,
                "decision_matches_reference": reference_accepted == model_accepted,
            }
        )

    def summary(subset: Sequence[dict]) -> dict:
        return {
            "cases": len(subset),
            "reference_accepted": sum(row["reference_accepted"] for row in subset),
            "strategy_accepted": sum(row["strategy_accepted"] for row in subset),
            "decision_agreement": sum(row["decision_matches_reference"] for row in subset),
            "false_approvals": sum(
                row["strategy_accepted"] and not row["reference_accepted"] for row in subset
            ),
            "false_rejections": sum(
                row["reference_accepted"] and not row["strategy_accepted"] for row in subset
            ),
            "correct_abstentions": sum(
                row["expected_behavior"] == "abstain" and row["response_mode"] == "abstain"
                for row in subset
            ),
            "unanswerable_answers": sum(
                row["expected_behavior"] == "abstain" and row["response_mode"] == "answer"
                for row in subset
            ),
            "answerable_abstentions": sum(
                row["expected_behavior"] == "answer" and row["response_mode"] == "abstain"
                for row in subset
            ),
        }

    return {
        "schema_version": 1,
        "answer_dataset_id": answers.data.dataset_id,
        "answer_dataset_sha256": answers.dataset_sha256,
        "claims_dataset_sha256": answers.claims.dataset_sha256,
        "label_provenance": answers.data.label_provenance,
        "strategy": strategy,
        "manual_claim_segmentation": True,
        "not_measured": [
            "actual LLM-generated prose",
            "automatic atomic-claim extraction and completeness",
            "independent human adjudication",
        ],
        "summary": summary(rows),
        "by_expected_behavior": {
            name: summary([row for row in rows if row["expected_behavior"] == name])
            for name in ("answer", "abstain")
        },
        "cases": rows,
    }
