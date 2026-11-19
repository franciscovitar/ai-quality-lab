"""Measure pinned NLI verdicts on source-locked, manually segmented real model text.

Agreement is to provisional project-author labels, NEVER independently verified truth.
"""

from collections import Counter
from dataclasses import dataclass
from typing import Literal

from aql.generation_evaluation import PromptBatch
from aql.nli import MAX_PAIR_TOKENS, MODEL_KEY, NliObservation, predicted_nli_label

Verdict = Literal["supported", "contradicted", "not_established"]


@dataclass(frozen=True)
class GeneratedNliPair:
    pair_id: str
    case_id: str
    claim_text: str
    premise: str
    evidence_refs: tuple[str, ...]
    author_verdict: Verdict

    def model_input(self) -> tuple[str, str, str]:
        """Give the NLI engine only source and claim, without label or rationale."""
        return self.pair_id, self.premise, self.claim_text.strip()


def generated_nli_pairs(prompt_batch: PromptBatch, reviewed_report: dict) -> list[GeneratedNliPair]:
    """Construct exact, case-specific premises from citations the generator actually saw."""
    if (
        reviewed_report.get("review_status") != "project_author_provisional"
        or reviewed_report.get("independent_reviewers") != 0
    ):
        raise ValueError("Generated NLI comparison requires declared provisional author review")

    reviews = reviewed_report["cases"]
    if len(reviews) != len(prompt_batch.cases) or [r["case_id"] for r in reviews] != [
        p.case_id for p in prompt_batch.cases
    ]:
        raise ValueError("Provisional review questions differ from frozen prompts")

    pairs = []
    for prompt, review in zip(prompt_batch.cases, reviews, strict=True):
        available = {ref.citation_id: ref.quote for ref in prompt.context}
        for index, claim in enumerate(review["claims"], start=1):
            refs = tuple(claim["evidence_refs"])
            if not refs or len(set(refs)) != len(refs) or any(ref not in available for ref in refs):
                raise ValueError(f"{prompt.case_id}: invalid claim-to-source provenance")
            if claim["verdict"] not in ("supported", "contradicted", "not_established"):
                raise ValueError("Unexpected author claim label")
            if not claim["text"].strip():
                raise ValueError("NLI hypotheses cannot be empty")
            pairs.append(
                GeneratedNliPair(
                    pair_id=f"{prompt.case_id}/C{index:02d}",
                    case_id=prompt.case_id,
                    claim_text=claim["text"],
                    premise="\n".join(available[ref] for ref in refs),
                    evidence_refs=refs,
                    author_verdict=claim["verdict"],
                )
            )
    if len(pairs) != reviewed_report["summary"]["annotated_claims"]:
        raise ValueError("Generated-claim inventory differs from source-bound review")
    return pairs


def evaluate_generated_nli(
    *, pairs: list[GeneratedNliPair], observations: list[NliObservation], reviewed_report: dict
) -> dict:
    """Compare observations to provisional labels, retaining every disagreeing pair."""
    if len(observations) != len(pairs) or [x.case_id for x in observations] != [
        pair.pair_id for pair in pairs
    ]:
        raise ValueError("NLI must return exactly one result for each reviewed claim in order")

    outcomes: Counter[str] = Counter()
    three_way: Counter[str] = Counter()
    rows = []
    for pair, observation in zip(pairs, observations, strict=True):
        if (
            observation.nli_label != predicted_nli_label(observation.logits)
            or not 1 <= observation.token_count <= MAX_PAIR_TOKENS
        ):
            raise ValueError("NLI label, logits or token-budget metadata is invalid")
        author_supported = pair.author_verdict == "supported"
        model_supported = observation.nli_label == "entailment"
        category = (
            "both_support"
            if author_supported and model_supported
            else "author_support_model_reject"
            if author_supported
            else "author_reject_model_support"
            if model_supported
            else "both_reject"
        )
        outcomes[category] += 1
        three_way[observation.nli_label] += 1
        rows.append(
            {
                "pair_id": pair.pair_id,
                "case_id": pair.case_id,
                "claim": pair.claim_text,
                "premise": pair.premise,
                "inspected_citations": list(pair.evidence_refs),
                "provisional_author_verdict": pair.author_verdict,
                "nli_label": observation.nli_label,
                "nli_binary": "supported" if model_supported else "not_supported",
                "agreement_category": category,
                "token_count": observation.token_count,
                "logits": [round(logit, 6) for logit in observation.logits],
            }
        )

    # Deliberately unsafe negative control: entailment of each extracted claim
    # does not establish question-answering, valid citations or abstention.
    per_question = []
    for reviewed_case in reviewed_report["cases"]:
        case_claims = [row for row in rows if row["case_id"] == reviewed_case["case_id"]]
        naive_accept = bool(case_claims) and all(
            row["nli_binary"] == "supported" for row in case_claims
        )
        per_question.append(
            {
                "case_id": reviewed_case["case_id"],
                "expected_behavior": reviewed_case["expected_behavior"],
                "provisional_author_responsiveness": reviewed_case["responsiveness"],
                "citation_protocol_ok": reviewed_case["citation_protocol_ok"],
                "naive_all_claims_entailed_accept": naive_accept,
                "false_accept_of_unanswerable": naive_accept
                and reviewed_case["expected_behavior"] == "abstain",
                "claim_count": len(case_claims),
            }
        )

    return {
        "schema_version": 1,
        "source_review_id": reviewed_report["review_id"],
        "source_review_sha256": reviewed_report["review_sha256"],
        "generated_sha256": reviewed_report["generated_sha256"],
        "review_status": "project_author_provisional",
        "independent_reviewers": 0,
        "model_key": MODEL_KEY,
        "inference": "pinned ONNX CPU; source excerpts as premise, raw claim as hypothesis",
        "decision": "entailment only => supported; contradiction/neutral => not supported",
        "reference": "Agreement to project-authored labels only; no independent answer factuality",
        "naive_answer_policy": (
            "NEGATIVE CONTROL: accept if every extracted claim is entailed; "
            "ignores original question, abstention, citations and completeness"
        ),
        "summary": {
            "claim_occurrences": len(rows),
            "agreement_with_provisional_author": outcomes["both_support"] + outcomes["both_reject"],
            "both_support": outcomes["both_support"],
            "both_reject": outcomes["both_reject"],
            "author_support_model_reject": outcomes["author_support_model_reject"],
            "author_reject_model_support": outcomes["author_reject_model_support"],
            "nli_three_way": {
                label: three_way[label] for label in ("contradiction", "entailment", "neutral")
            },
            "naive_answers_accepted": sum(
                row["naive_all_claims_entailed_accept"] for row in per_question
            ),
            "naive_unanswerable_false_accepts": sum(
                row["false_accept_of_unanswerable"] for row in per_question
            ),
        },
        "cases": rows,
        "naive_answer_decisions": per_question,
    }
