"""Audit frozen, real model responses against explicitly authored claim annotations.

This validates annotation *integrity*, not the truth of the author's judgments.
No free-form claim extraction, independent human review or production gate exists.
"""

import hashlib
import re
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from aql.generation_evaluation import (
    GeneratedBatch,
    PromptBatch,
    build_prompt_batch,
    evaluate_generated_batch,
)

FROZEN_PROMPTS_SHA256 = "f5ab781fad24ed92567f128fc499aa48a3cfa1557eaa0ce2b8a1919ce2c3edee"
FROZEN_GENERATED_SHA256 = "1cb8fe3b4ea7bd1455d5f716dcc53c900160f41d3d16a326d72b41e44f184186"
FROZEN_CI_RUN = 44588063650
CITATION_ONLY = re.compile(r"\[R[0-9]{3}:[0-9]+\]\s*")


class ReviewSegment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["claim", "non_claim"]
    text: str = Field(min_length=1)
    verdict: Literal["supported", "contradicted", "not_established"] | None = None
    evidence_refs: tuple[str, ...] = ()
    rationale: str | None = None
    non_claim_reason: Literal["citation_marker", "formatting", "discourse"] | None = None

    @model_validator(mode="after")
    def validate_parts(self) -> "ReviewSegment":
        if self.kind == "claim":
            if (
                self.verdict is None
                or not self.evidence_refs
                or len(set(self.evidence_refs)) != len(self.evidence_refs)
                or not self.rationale
                or len(self.rationale) < 20
                or self.non_claim_reason is not None
            ):
                raise ValueError("Claims require verdict, unique inspected sources and explanation")
        elif self.verdict is not None or self.evidence_refs or self.rationale is not None:
            raise ValueError("Non-claims cannot hide a verdict or source attribution")
        elif self.non_claim_reason == "citation_marker":
            if not CITATION_ONLY.fullmatch(self.text):
                raise ValueError("Citation-marker segments must only contain one marker")
        elif self.non_claim_reason == "formatting":
            if self.text.strip():
                raise ValueError("Formatting segments can contain whitespace only")
        elif self.non_claim_reason == "discourse":
            if self.text.strip() != "The answer is:":
                raise ValueError("Unreviewed discourse text is not allowed")
        else:
            raise ValueError("Every non-claim requires a narrowly constrained reason")
        return self


class CaseReview(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str = Field(pattern=r"^R[0-9]{3}$")
    responsiveness: Literal["addresses_question", "partial", "does_not_answer"]
    response_rationale: str = Field(min_length=20)
    segments: tuple[ReviewSegment, ...] = Field(min_length=1)


class FrozenGeneratedReview(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1]
    review_id: Literal["synthetic-generated-claim-review-v1"]
    review_status: Literal["project_author_provisional"]
    independent_reviewers: Literal[0]
    source_ci_run_id: Literal[44588063650]
    prompts_sha256: Literal["f5ab781fad24ed92567f128fc499aa48a3cfa1557eaa0ce2b8a1919ce2c3edee"]
    generated_sha256: Literal["1cb8fe3b4ea7bd1455d5f716dcc53c900160f41d3d16a326d72b41e44f184186"]
    annotation_method: str = Field(min_length=35)
    cases: tuple[CaseReview, ...] = Field(min_length=1)


def review_frozen_generation(
    prompts_bytes: bytes,
    generated_bytes: bytes,
    review_bytes: bytes,
    expected_prompts: PromptBatch,
) -> dict:
    """Reject altered evidence, omitted answer text or unjustified citation IDs."""
    if hashlib.sha256(prompts_bytes).hexdigest() != FROZEN_PROMPTS_SHA256:
        raise ValueError("Frozen CI prompts changed")
    if hashlib.sha256(generated_bytes).hexdigest() != FROZEN_GENERATED_SHA256:
        raise ValueError("Frozen CI generated outputs changed")

    # Existing verifier checks full source-chain, model, prompt and corpus identity.
    protocol = evaluate_generated_batch(expected_prompts, prompts_bytes, generated_bytes)
    generated = GeneratedBatch.model_validate_json(generated_bytes)
    annotations = FrozenGeneratedReview.model_validate_json(review_bytes)
    prompts = PromptBatch.model_validate_json(prompts_bytes)
    if len(annotations.cases) != len(generated.cases) or [x.case_id for x in annotations.cases] != [
        x.case_id for x in generated.cases
    ]:
        raise ValueError("Every generated answer must have exactly one review in source order")

    rows = []
    all_counts: Counter[str] = Counter()
    for question, output, review, protocol_case in zip(
        prompts.cases, generated.cases, annotations.cases, protocol["cases"], strict=True
    ):
        if "".join(segment.text for segment in review.segments) != output.text:
            raise ValueError(
                f"{review.case_id}: annotation segments do not cover the exact raw output"
            )
        allowed_refs = {source.citation_id for source in question.context}
        claim_segments = [segment for segment in review.segments if segment.kind == "claim"]
        if not claim_segments and not protocol_case["strict_abstention"]:
            raise ValueError(f"{review.case_id}: answered text has no reviewed claims")
        if not claim_segments and review.responsiveness != "does_not_answer":
            raise ValueError(f"{review.case_id}: a refusal is not an answered question")
        if any(set(segment.evidence_refs) - allowed_refs for segment in claim_segments):
            raise ValueError(
                f"{review.case_id}: annotations cite sources not shown to the generator"
            )
        counts = Counter(segment.verdict for segment in claim_segments)
        all_counts.update(counts)
        rows.append(
            {
                "case_id": review.case_id,
                "expected_behavior": question.expected_behavior,
                "responsiveness": review.responsiveness,
                "claim_count": len(claim_segments),
                "verdict_counts": {
                    v: counts[v] for v in ("supported", "contradicted", "not_established")
                },
                "all_claims_author_labeled_supported": bool(claim_segments)
                and counts["supported"] == len(claim_segments),
                "strict_abstention": protocol_case["strict_abstention"],
                "citation_protocol_ok": protocol_case["citation_protocol_ok"],
                "claims": [
                    {
                        "text": segment.text,
                        "verdict": segment.verdict,
                        "evidence_refs": list(segment.evidence_refs),
                        "rationale": segment.rationale,
                    }
                    for segment in claim_segments
                ],
                "response_rationale": review.response_rationale,
            }
        )

    return {
        "schema_version": 1,
        "review_id": annotations.review_id,
        "review_status": annotations.review_status,
        "independent_reviewers": annotations.independent_reviewers,
        "source_ci_run_id": FROZEN_CI_RUN,
        "prompts_sha256": FROZEN_PROMPTS_SHA256,
        "generated_sha256": FROZEN_GENERATED_SHA256,
        "review_sha256": hashlib.sha256(review_bytes).hexdigest(),
        "assessment": "Project-authored labels with full text coverage; not independent factuality",
        "summary": {
            "generated_answers": len(rows),
            "annotated_claims": sum(all_counts.values()),
            "supported": all_counts["supported"],
            "contradicted": all_counts["contradicted"],
            "not_established": all_counts["not_established"],
            "all_claims_author_labeled_supported_cases": sum(
                row["all_claims_author_labeled_supported"] for row in rows
            ),
            "fully_responsive_cases_author_labeled": sum(
                row["responsiveness"] == "addresses_question" for row in rows
            ),
            "strict_abstentions": protocol["summary"]["strict_abstentions"],
            "citation_protocol_ok": protocol["summary"]["citation_protocol_ok"],
        },
        "cases": rows,
    }


def load_and_review(
    *,
    prompts_path: Path,
    generated_path: Path,
    review_path: Path,
    context_path: Path,
    evidence_path: Path,
    retrieval_path: Path,
    corpus_path: Path,
) -> dict:
    from aql.evidence import load_evidence

    evidence = load_evidence(evidence_path, retrieval_path, corpus_path)
    expected = build_prompt_batch(context_path, evidence)
    return review_frozen_generation(
        prompts_path.read_bytes(),
        generated_path.read_bytes(),
        review_path.read_bytes(),
        expected,
    )
