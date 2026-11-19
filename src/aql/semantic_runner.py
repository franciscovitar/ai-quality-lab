"""Opt-in exact vector and RRF evaluations against the unchanged lexical goldens."""

import argparse
from pathlib import Path

from aql.database import open_database
from aql.embeddings import MODEL_KEY, PinnedFastEmbed
from aql.eval_runner import DEFAULT_CORPUS, DEFAULT_DATASET
from aql.evaluation import evaluate_retrieval, find_regressions, load_dataset, report_json
from aql.retrieval import fuse_rrf, search_lexical, search_semantic
from aql.vector_index import assert_index_complete, index_embeddings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=Path("test-results/semantic"))
    parser.add_argument(
        "--download-model",
        action="store_true",
        help="Explicitly permit downloading the pinned public model (~90 MB)",
    )
    args = parser.parse_args()

    goldens = load_dataset(args.dataset, args.corpus)
    embedder = PinnedFastEmbed(allow_download=args.download_model)
    questions = {case.question for case in goldens.data.cases}
    vectors = dict(
        zip(
            sorted(questions),
            embedder.encode(sorted(questions)),
            strict=True,
        )
    )

    with open_database() as connection:
        indexed = index_embeddings(connection, embedder)
        assert_index_complete(connection, MODEL_KEY)

        def lexical(q: str, k: int):
            return search_lexical(connection, q, k)

        def semantic(q: str, k: int):
            return search_semantic(connection, vectors[q], MODEL_KEY, k)

        def hybrid(q: str, k: int):
            return fuse_rrf(lexical(q, 10), semantic(q, 10), k)

        reports = {
            "lexical": evaluate_retrieval(goldens, lexical, top_k=args.top_k),
            "semantic": evaluate_retrieval(
                goldens, semantic, top_k=args.top_k, strategy=f"exact-cosine:{MODEL_KEY}"
            ),
            "hybrid": evaluate_retrieval(
                goldens, hybrid, top_k=args.top_k, strategy=f"rrf60:{MODEL_KEY}"
            ),
        }

    # Preserve the proven lexical floor: the semantic run must not change FTS.
    from json import loads

    baseline = loads(Path("data/evals/baselines/postgres_fts_v1.json").read_text())
    regressions = find_regressions(baseline, reports["lexical"])
    if regressions:
        parser.exit(1, "Lexical regression: " + "\n".join(regressions))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, report in reports.items():
        (args.output_dir / f"{name}.json").write_text(report_json(report), encoding="utf-8")

    comparison = {
        "dataset_sha256": goldens.dataset_sha256,
        "model_key": MODEL_KEY,
        "top_k_chunks": args.top_k,
        "index_stats": {
            "total": indexed.total,
            "inserted": indexed.inserted,
            "existing": indexed.existing,
        },
        "metrics": {name: report["summary"] for name, report in reports.items()},
        "per_case_recall": {
            row["case_id"]: {
                name: next(
                    item["recall_at_k"]
                    for item in reports[name]["cases"]
                    if item["case_id"] == row["case_id"]
                )
                for name in reports
            }
            for row in reports["lexical"]["cases"]
        },
    }
    (args.output_dir / "comparison.json").write_text(report_json(comparison), encoding="utf-8")
    for name, report in reports.items():
        print(
            f"{name} Recall@{args.top_k}={report['summary']['recall_at_k']} "
            f"MRR@{args.top_k}={report['summary']['mrr_at_k']} "
            f"no-hit={report['summary']['unanswerable_no_hit_rate']}"
        )
    print(f"Exact vector model: {MODEL_KEY} | comparison written to {args.output_dir}")


if __name__ == "__main__":
    main()
