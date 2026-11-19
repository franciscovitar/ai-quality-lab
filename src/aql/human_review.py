"""Prepare label-hidden claim packets and compare two completed reviewer submissions.

Packets reuse project-authored segmentation, so they are NOT a blind held-out set.
No reviewer is fabricated, recruited, contacted or treated as verified by this code.
"""

import hashlib
import math
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from aql.claim_runner import DEFAULT_EVIDENCE, DEFAULT_RETRIEVAL
from aql.eval_runner import DEFAULT_CORPUS
from aql.evaluation import report_json
from aql.generated_claim_review import (
    FROZEN_GENERATED_SHA256,
    FROZEN_PROMPTS_SHA256,
    FrozenGeneratedReview,
    load_and_review,
)
from aql.generation_evaluation import GeneratedBatch, PromptBatch
from aql.generation_runner import DEFAULT_CONTEXTS

PROMPTS = Path("data/evals/frozen_generation_v1/prompts.json")
GENERATED = Path("data/evals/frozen_generation_v1/generated-outputs.json")
ANNOTATIONS = Path("data/evals/generated_claim_review_v1.json")

CLAIM_LABELS = ("supported", "contradicted", "not_established")
RESPONSE_LABELS = ("addresses_question", "partial", "does_not_answer")


class PacketSource(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    citation_id: str
    quote: str


class PacketClaim(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    claim_id: str = Field(pattern=r"^R[0-9]{3}/C[0-9]{2}$")
    text: str = Field(min_length=1)
    start_char: int = Field(ge=0)
    end_char: int = Field(gt=0)


class PacketCase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str = Field(pattern=r"^R[0-9]{3}$")
    question: str = Field(min_length=5)
    generated_text: str = Field(min_length=1)
    sources: tuple[PacketSource, ...] = Field(min_length=1)
    claims: tuple[PacketClaim, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def verify_offsets(self) -> "PacketCase":
        if any(
            claim.start_char >= claim.end_char
            or self.generated_text[claim.start_char : claim.end_char] != claim.text
            for claim in self.claims
        ):
            raise ValueError("Packet claim offsets do not match generated raw text")
        if len({c.claim_id for c in self.claims}) != len(self.claims):
            raise ValueError("Packet claim IDs must be unique")
        return self


class ReviewPacket(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1]
    packet_id: Literal["synthetic-generated-review-packet-v1"]
    generator_prompts_sha256: Literal[
        "f5ab781fad24ed92567f128fc499aa48a3cfa1557eaa0ce2b8a1919ce2c3edee"
    ]
    generated_outputs_sha256: Literal[
        "1cb8fe3b4ea7bd1455d5f716dcc53c900160f41d3d16a326d72b41e44f184186"
    ]
    blinding: Literal["AUTHOR_AND_NLI_VERDICTS_WITHHELD;SEGMENTATION_IS_AUTHOR_SUPPLIED"]
    cases: tuple[PacketCase, ...] = Field(min_length=1)


class ClaimDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    claim_id: str = Field(pattern=r"^R[0-9]{3}/C[0-9]{2}$")
    label: Literal["supported", "contradicted", "not_established"]
    inspected_citations: tuple[str, ...] = Field(min_length=1)
    rationale: str = Field(min_length=20)

    @model_validator(mode="after")
    def unique_sources(self) -> "ClaimDecision":
        if len(set(self.inspected_citations)) != len(self.inspected_citations):
            raise ValueError("Repeated inspected source ID")
        return self


class ResponseDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str = Field(pattern=r"^R[0-9]{3}$")
    label: Literal["addresses_question", "partial", "does_not_answer"]
    rationale: str = Field(min_length=20)


class ReviewerSubmission(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1]
    packet_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reviewer_id: str = Field(pattern=r"^[a-z][a-z0-9_-]{2,30}$")
    claims: tuple[ClaimDecision, ...] = Field(min_length=1)
    responses: tuple[ResponseDecision, ...] = Field(min_length=1)


def packet_json(packet: ReviewPacket) -> str:
    return report_json(packet.model_dump(mode="json"))


def packet_sha256(packet: ReviewPacket) -> str:
    return hashlib.sha256(packet_json(packet).encode("utf-8")).hexdigest()


def build_review_packet() -> ReviewPacket:
    """Rebuild all public evidence, using annotation positions but excluding labels."""
    load_and_review(
        prompts_path=PROMPTS,
        generated_path=GENERATED,
        review_path=ANNOTATIONS,
        context_path=DEFAULT_CONTEXTS,
        evidence_path=DEFAULT_EVIDENCE,
        retrieval_path=DEFAULT_RETRIEVAL,
        corpus_path=DEFAULT_CORPUS,
    )
    prompts = PromptBatch.model_validate_json(PROMPTS.read_bytes())
    outputs = GeneratedBatch.model_validate_json(GENERATED.read_bytes())
    review = FrozenGeneratedReview.model_validate_json(ANNOTATIONS.read_bytes())

    cases = []
    for prompt, generated, author in zip(prompts.cases, outputs.cases, review.cases, strict=True):
        offset = 0
        claims = []
        for segment in author.segments:
            end = offset + len(segment.text)
            if segment.kind == "claim":
                claims.append(
                    PacketClaim(
                        claim_id=f"{prompt.case_id}/C{len(claims) + 1:02d}",
                        text=segment.text,
                        start_char=offset,
                        end_char=end,
                    )
                )
            offset = end
        cases.append(
            PacketCase(
                case_id=prompt.case_id,
                question=prompt.question,
                generated_text=generated.text,
                sources=tuple(
                    PacketSource(citation_id=c.citation_id, quote=c.quote) for c in prompt.context
                ),
                claims=tuple(claims),
            )
        )
    return ReviewPacket(
        schema_version=1,
        packet_id="synthetic-generated-review-packet-v1",
        generator_prompts_sha256=FROZEN_PROMPTS_SHA256,
        generated_outputs_sha256=FROZEN_GENERATED_SHA256,
        blinding="AUTHOR_AND_NLI_VERDICTS_WITHHELD;SEGMENTATION_IS_AUTHOR_SUPPLIED",
        cases=tuple(cases),
    )


def blank_submission_template(packet: ReviewPacket, reviewer_id: str) -> dict:
    """Intentionally invalid until a real reviewer supplies all decisions and notes."""
    if not reviewer_id or reviewer_id not in ("reviewer_a", "reviewer_b"):
        raise ValueError("Use one of two separate reviewer template identifiers")
    return {
        "schema_version": 1,
        "packet_sha256": packet_sha256(packet),
        "reviewer_id": reviewer_id,
        "claims": [
            {"claim_id": c.claim_id, "label": None, "inspected_citations": [], "rationale": ""}
            for case in packet.cases
            for c in case.claims
        ],
        "responses": [
            {"case_id": case.case_id, "label": None, "rationale": ""} for case in packet.cases
        ],
    }


def validate_submission(packet: ReviewPacket, raw: bytes) -> ReviewerSubmission:
    submission = ReviewerSubmission.model_validate_json(raw)
    if submission.packet_sha256 != packet_sha256(packet):
        raise ValueError("Reviewer submission belongs to a different frozen packet")
    ids = [c.claim_id for case in packet.cases for c in case.claims]
    if [claim.claim_id for claim in submission.claims] != ids:
        raise ValueError("Reviewer must label every claim exactly once, in packet order")
    if [response.case_id for response in submission.responses] != [
        case.case_id for case in packet.cases
    ]:
        raise ValueError("Reviewer must assess every answer exactly once")
    allowed_by_case = {
        case.case_id: {source.citation_id for source in case.sources} for case in packet.cases
    }
    for claim in submission.claims:
        if set(claim.inspected_citations) - allowed_by_case[claim.claim_id.split("/")[0]]:
            raise ValueError("Reviewer cited a source not supplied to the generator")
    return submission


def label_agreement(left: list[str], right: list[str], labels: tuple[str, ...]) -> dict:
    """Observed agreement and unweighted Cohen's kappa on matched categorical items."""
    if not left or len(left) != len(right):
        raise ValueError("Agreement requires equal, non-empty label sequences")
    if any(x not in labels for x in left + right):
        raise ValueError("Unknown label used in reviewer agreement")
    n = len(left)
    observed = sum(a == b for a, b in zip(left, right, strict=True)) / n
    counts_left, counts_right = Counter(left), Counter(right)
    chance = sum(counts_left[x] * counts_right[x] for x in labels) / (n * n)
    # When both coders choose a single class for all items, kappa is
    # undefined (division by zero), even if raw agreement is exactly one.
    kappa = None if math.isclose(chance, 1.0) else (observed - chance) / (1 - chance)
    return {
        "items": n,
        "raw_agreement": round(observed, 6),
        "expected_chance_agreement": round(chance, 6),
        "cohen_kappa": None if kappa is None else round(kappa, 6),
        "kappa_note": "undefined for no label variation" if kappa is None else None,
    }


def compare_submissions(packet: ReviewPacket, left_raw: bytes, right_raw: bytes) -> dict:
    """Compare two files; never certify they came from independent humans."""
    left = validate_submission(packet, left_raw)
    right = validate_submission(packet, right_raw)
    if left.reviewer_id == right.reviewer_id:
        raise ValueError("Reviewer identifiers must differ")

    claim_a = [c.label for c in left.claims]
    claim_b = [c.label for c in right.claims]
    response_a = [r.label for r in left.responses]
    response_b = [r.label for r in right.responses]
    disagreements = [
        {
            "claim_id": a.claim_id,
            "labels": {left.reviewer_id: a.label, right.reviewer_id: b.label},
        }
        for a, b in zip(left.claims, right.claims, strict=True)
        if a.label != b.label
    ]
    response_disagreements = [
        {
            "case_id": a.case_id,
            "labels": {left.reviewer_id: a.label, right.reviewer_id: b.label},
        }
        for a, b in zip(left.responses, right.responses, strict=True)
        if a.label != b.label
    ]
    return {
        "schema_version": 1,
        "packet_sha256": packet_sha256(packet),
        "reviewer_ids": [left.reviewer_id, right.reviewer_id],
        "completed_submission_files": 2,
        "independent_human_identities_verified": 0,
        "adjudication_status": "not_performed",
        "claims_three_class": label_agreement(claim_a, claim_b, CLAIM_LABELS),
        "claims_binary_supported": label_agreement(
            ["supported" if c == "supported" else "not_supported" for c in claim_a],
            ["supported" if c == "supported" else "not_supported" for c in claim_b],
            ("supported", "not_supported"),
        ),
        "response_completeness": label_agreement(response_a, response_b, RESPONSE_LABELS),
        "claim_disagreements": disagreements,
        "response_disagreements": response_disagreements,
        "limits": (
            "Two parseable files do not verify two independent people; segmentation was supplied "
            "by the project author; no adjudicated truth or sealed holdout is produced"
        ),
    }
