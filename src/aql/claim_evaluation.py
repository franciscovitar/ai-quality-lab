"""Version-locked claim-to-evidence judgments and deterministic verdict scoring.

This scores *provided verdicts against authored reference labels*. It does not infer
entailment, generate answers or call an LLM. Authentic citations alone are insufficient.
"""

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from aql.evidence import AnchoredSpan, LockedEvidence, load_evidence

Verdict = Literal["entailed", "not_entailed"]


class ClaimEvidenceRef(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str = Field(pattern=r"^R[0-9]{3}$")
    span_index: int = Field(ge=0)


class ClaimCase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str = Field(pattern=r"^C[0-9]{3}$")
    split: Literal["dev", "holdout"]
    claim: str = Field(min_length=10, max_length=500)
    expected_label: Verdict
    evidence_refs: tuple[ClaimEvidenceRef, ...] = Field(min_length=1)
    rationale: str = Field(min_length=12, max_length=500)
    error_type: (
        Literal["missing_detail", "contradiction", "scope_shift", "unsupported_extension"] | None
    ) = None

    @model_validator(mode="after")
    def validate_case(self) -> "ClaimCase":
        if self.claim.strip() != self.claim or self.rationale.strip() != self.rationale:
            raise ValueError("Claims and rationales must not have surrounding whitespace")
        if self.expected_label == "entailed" and self.error_type is not None:
            raise ValueError("Entailed claims must not carry a negative error category")
        if self.expected_label == "not_entailed" and self.error_type is None:
            raise ValueError("Unsupported claims require a reason category")
        if len(set(self.evidence_refs)) != len(self.evidence_refs):
            raise ValueError("Evidence refs must not repeat within one claim")
        return self


class ClaimDataset(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1]
    dataset_id: str = Field(min_length=1)
    expected_cases: int = Field(ge=1)
    evidence_dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    label_provenance: str = Field(min_length=1)
    cases: tuple[ClaimCase, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_coverage(self) -> "ClaimDataset":
        if len(self.cases) != self.expected_cases:
            raise ValueError("Claim dataset shrank or grew without explicit review")
        case_ids = [case.case_id for case in self.cases]
        if case_ids != sorted(set(case_ids)):
            raise ValueError("Claim IDs must be unique and sorted")
        for split in ("dev", "holdout"):
            labels = {case.expected_label for case in self.cases if case.split == split}
            if labels != {"entailed", "not_entailed"}:
                raise ValueError(f"{split} must contain both entailment labels")
        return self


@dataclass(frozen=True)
class LockedClaims:
    data: ClaimDataset
    dataset_sha256: str
    evidence: LockedEvidence
    anchors_by_claim: dict[str, tuple[AnchoredSpan, ...]]


def load_claims(
    claims_path: Path, evidence_path: Path, retrieval_path: Path, corpus_path: Path
) -> LockedClaims:
    """Validate entire source → retrieval → evidence → claim annotation chain."""
    evidence = load_evidence(evidence_path, retrieval_path, corpus_path)
    content = claims_path.read_bytes()
    data = ClaimDataset.model_validate_json(content)
    if data.evidence_dataset_sha256 != evidence.dataset_sha256:
        raise ValueError("Claim labels are bound to a different evidence dataset")

    anchors_by_claim = {}
    for case in data.cases:
        resolved = []
        for ref in case.evidence_refs:
            anchors = evidence.anchors_by_case.get(ref.case_id)
            if anchors is None or ref.span_index >= len(anchors):
                raise ValueError(f"{case.case_id}: unknown or unanswerable evidence anchor")
            resolved.append(anchors[ref.span_index])
        anchors_by_claim[case.case_id] = tuple(resolved)

    return LockedClaims(data, hashlib.sha256(content).hexdigest(), evidence, anchors_by_claim)


class ClaimPrediction(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str = Field(pattern=r"^C[0-9]{3}$")
    label: Verdict


class PredictionFile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1]
    dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    predictions: tuple[ClaimPrediction, ...] = Field(min_length=1)


def citation_presence_baseline(claims: LockedClaims) -> list[ClaimPrediction]:
    """Intentionally unsafe baseline: every case points at genuine source quotes."""
    return [ClaimPrediction(case_id=case.case_id, label="entailed") for case in claims.data.cases]


def _summary(rows: Sequence[dict]) -> dict:
    tp = sum(row["actual"] == "entailed" and row["predicted"] == "entailed" for row in rows)
    tn = sum(row["actual"] == "not_entailed" and row["predicted"] == "not_entailed" for row in rows)
    fp = sum(row["actual"] == "not_entailed" and row["predicted"] == "entailed" for row in rows)
    fn = sum(row["actual"] == "entailed" and row["predicted"] == "not_entailed" for row in rows)

    def fraction(numerator: int, denominator: int) -> float | None:
        return round(numerator / denominator, 6) if denominator else None

    def f1(true: int, false_positive: int, false_negative: int) -> float:
        denominator = 2 * true + false_positive + false_negative
        return 2 * true / denominator if denominator else 0.0

    support_recall = fraction(tp, tp + fn)
    non_support_recall = fraction(tn, tn + fp)
    return {
        "cases": len(rows),
        "entailed": tp + fn,
        "not_entailed": tn + fp,
        "correct": tp + tn,
        "accuracy": fraction(tp + tn, len(rows)),
        "balanced_accuracy": (
            round((support_recall + non_support_recall) / 2, 6)
            if support_recall is not None and non_support_recall is not None
            else None
        ),
        "entailed_precision": fraction(tp, tp + fp),
        "entailed_recall": support_recall,
        "not_entailed_recall": non_support_recall,
        "false_supported_rate": fraction(fp, fp + tn),
        "macro_f1": round((f1(tp, fp, fn) + f1(tn, fn, fp)) / 2, 6),
        "confusion": {
            "entailed_as_entailed": tp,
            "entailed_as_not_entailed": fn,
            "not_entailed_as_entailed": fp,
            "not_entailed_as_not_entailed": tn,
        },
    }


def evaluate_claim_predictions(
    claims: LockedClaims,
    predictions: Sequence[ClaimPrediction],
    *,
    strategy: str,
) -> dict:
    """No inferred support: compare independently supplied labels against goldens."""
    if not strategy.strip():
        raise ValueError("A named prediction strategy is required")
    predicted_ids = [item.case_id for item in predictions]
    gold_ids = {case.case_id for case in claims.data.cases}
    if len(set(predicted_ids)) != len(predicted_ids) or set(predicted_ids) != gold_ids:
        raise ValueError("Predictions must cover each claim exactly once, without duplicates")
    by_id = {item.case_id: item.label for item in predictions}

    rows = [
        {
            "case_id": case.case_id,
            "split": case.split,
            "claim": case.claim,
            "actual": case.expected_label,
            "predicted": by_id[case.case_id],
            "correct": by_id[case.case_id] == case.expected_label,
            "error_type": case.error_type,
            "evidence_refs": [ref.model_dump() for ref in case.evidence_refs],
        }
        for case in claims.data.cases
    ]

    return {
        "schema_version": 1,
        "dataset_id": claims.data.dataset_id,
        "dataset_sha256": claims.dataset_sha256,
        "evidence_dataset_sha256": claims.evidence.dataset_sha256,
        "corpus_sha256": claims.evidence.data.corpus_sha256,
        "strategy": strategy,
        "label_provenance": claims.data.label_provenance,
        "summary": _summary(rows),
        "by_split": {
            split: _summary([row for row in rows if row["split"] == split])
            for split in ("dev", "holdout")
        },
        "by_negative_error_type": {
            category: _summary([row for row in rows if row["error_type"] == category])
            for category in sorted({row["error_type"] for row in rows if row["error_type"]})
        },
        "cases": rows,
    }
