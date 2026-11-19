"""Run the locked retrieval golden set against an actual migrated PostgreSQL DB."""

import argparse
import json
from pathlib import Path

from aql.database import open_database
from aql.evaluation import evaluate_retrieval, find_regressions, load_dataset, report_json
from aql.retrieval import search_lexical

DEFAULT_CORPUS = Path("data/synthetic_docs/corpus_v1.json")
DEFAULT_DATASET = Path("data/evals/retrieval_v1.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--compare", type=Path, help="Optional prior report, same dataset and retrieval config"
    )
    args = parser.parse_args()

    golden = load_dataset(args.dataset, args.corpus)
    with open_database() as connection:
        report = evaluate_retrieval(
            golden,
            lambda question, k: search_lexical(connection, question, k),
            top_k=args.top_k,
        )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report_json(report), encoding="utf-8")

    summary = report["summary"]
    print(
        f"Retrieval eval {golden.data.dataset_id}: {summary['cases']} cases, "
        f"Recall@{args.top_k}={summary['recall_at_k']}, "
        f"MRR@{args.top_k}={summary['mrr_at_k']}, "
        f"unanswerable_no_hit_rate={summary['unanswerable_no_hit_rate']} "
        "(retrieval proxy; no LLM answers or abstention evaluated)"
    )

    if args.compare:
        regressions = find_regressions(json.loads(args.compare.read_text()), report)
        if regressions:
            parser.exit(1, "Retrieval regression:\n" + "\n".join(regressions) + "\n")
        print("Baseline comparison: no per-case regressions")


if __name__ == "__main__":
    main()
