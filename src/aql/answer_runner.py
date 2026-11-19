"""Analyze deliberately composed full-answer fixtures (not LLM-generated outputs)."""

import argparse
from pathlib import Path

from aql.answer_evaluation import evaluate_answers, load_answers
from aql.claim_evaluation import (
    ClaimPrediction,
    PredictionFile,
    citation_presence_baseline,
    load_claims,
)
from aql.claim_runner import DEFAULT_CLAIMS, DEFAULT_EVIDENCE, DEFAULT_RETRIEVAL
from aql.eval_runner import DEFAULT_CORPUS
from aql.evaluation import report_json

DEFAULT_ANSWERS = Path("data/evals/answer_cases_v1.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_ANSWERS)
    parser.add_argument("--claims", type=Path, default=DEFAULT_CLAIMS)
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--retrieval", type=Path, default=DEFAULT_RETRIEVAL)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path("test-results/answer-composition-report.json")
    )
    args = parser.parse_args()

    claims = load_claims(args.claims, args.evidence, args.retrieval, args.corpus)
    answers = load_answers(args.dataset, claims)
    reference_predictions = [
        ClaimPrediction(case_id=case.case_id, label=case.expected_label)
        for case in claims.data.cases
    ]
    reference = evaluate_answers(
        answers, reference_predictions, strategy="authored-claim-label-reference-oracle"
    )
    naive = evaluate_answers(
        answers,
        citation_presence_baseline(claims),
        strategy="unsafe-citation-presence-baseline",
    )
    report = {"schema_version": 1, "reference": reference, "citation_baseline": naive}
    if args.predictions is not None:
        external = PredictionFile.model_validate_json(args.predictions.read_bytes())
        if external.dataset_sha256 != claims.dataset_sha256:
            parser.error("Predictions target a different claim dataset SHA-256")
        report["external"] = evaluate_answers(
            answers, external.predictions, strategy="external-claim-predictions"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report_json(report), encoding="utf-8")
    s = naive["summary"]
    print(
        f"Authored answer compositions {answers.data.dataset_id}: {s['cases']} fixtures, "
        f"citation-only false approvals={s['false_approvals']}, "
        f"false rejections={s['false_rejections']} "
        "[not LLM outputs, not independent reference labels]"
    )


if __name__ == "__main__":
    main()
