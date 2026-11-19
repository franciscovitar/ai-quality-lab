"""Audit full text coverage of provisional, author-labeled real generated claims."""

import argparse
from pathlib import Path

from aql.claim_runner import DEFAULT_EVIDENCE, DEFAULT_RETRIEVAL
from aql.eval_runner import DEFAULT_CORPUS
from aql.evaluation import report_json
from aql.generated_claim_review import load_and_review
from aql.generation_runner import DEFAULT_CONTEXTS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prompts", type=Path, default=Path("data/evals/frozen_generation_v1/prompts.json")
    )
    parser.add_argument(
        "--generated",
        type=Path,
        default=Path("data/evals/frozen_generation_v1/generated-outputs.json"),
    )
    parser.add_argument(
        "--annotations", type=Path, default=Path("data/evals/generated_claim_review_v1.json")
    )
    parser.add_argument("--contexts", type=Path, default=DEFAULT_CONTEXTS)
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--retrieval", type=Path, default=DEFAULT_RETRIEVAL)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument(
        "--output", type=Path, default=Path("test-results/generated-claim-review-report.json")
    )
    args = parser.parse_args()

    report = load_and_review(
        prompts_path=args.prompts,
        generated_path=args.generated,
        review_path=args.annotations,
        context_path=args.contexts,
        evidence_path=args.evidence,
        retrieval_path=args.retrieval,
        corpus_path=args.corpus,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report_json(report), encoding="utf-8")
    counts = report["summary"]
    print(
        f"Provisional claim review: {counts['annotated_claims']} author-labeled claims "
        f"across {counts['generated_answers']} frozen answers; "
        f"supported={counts['supported']}, contradicted={counts['contradicted']}, "
        f"not_established={counts['not_established']} "
        "[NOT independent correctness; no release gate]"
    )


if __name__ == "__main__":
    main()
