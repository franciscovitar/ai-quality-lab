"""Evaluate a claim-verdict file, or show why genuine citations are not enough."""

import argparse
from pathlib import Path

from aql.claim_evaluation import (
    PredictionFile,
    citation_presence_baseline,
    evaluate_claim_predictions,
    load_claims,
)
from aql.eval_runner import DEFAULT_CORPUS
from aql.evaluation import report_json

DEFAULT_CLAIMS = Path("data/evals/claim_pairs_v1.json")
DEFAULT_EVIDENCE = Path("data/evals/evidence_stress_v1.json")
DEFAULT_RETRIEVAL = Path("data/evals/retrieval_stress_v1.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_CLAIMS)
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--retrieval", type=Path, default=DEFAULT_RETRIEVAL)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--predictions", type=Path, default=None)
    parser.add_argument(
        "--output", type=Path, default=Path("test-results/claim-support-report.json")
    )
    args = parser.parse_args()

    goldens = load_claims(args.dataset, args.evidence, args.retrieval, args.corpus)
    if args.predictions is not None:
        prediction_file = PredictionFile.model_validate_json(args.predictions.read_bytes())
        if prediction_file.dataset_sha256 != goldens.dataset_sha256:
            parser.error("Predictions target a different claim dataset SHA-256")
        predictions = prediction_file.predictions
        strategy = "externally-supplied-claim-verdicts-v1"
    else:
        predictions = citation_presence_baseline(goldens)
        strategy = "unsafe-citation-presence-baseline-v1"

    report = evaluate_claim_predictions(goldens, predictions, strategy=strategy)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report_json(report), encoding="utf-8")
    summary = report["summary"]
    print(
        f"Claim eval {goldens.data.dataset_id}: {summary['correct']}/{summary['cases']} "
        f"correct, false-supported={summary['confusion']['not_entailed_as_entailed']}/"
        f"{summary['not_entailed']}, macro-F1={summary['macro_f1']} "
        f"[{strategy}; not an LLM judgment]"
    )


if __name__ == "__main__":
    main()
