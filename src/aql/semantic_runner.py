"""Opt-in exact vector and RRF evaluations against the unchanged lexical goldens."""

import argparse
import json
from pathlib import Path

from aql.database import open_database
from aql.embeddings import MODEL_KEY, PinnedFastEmbed
from aql.eval_runner import DEFAULT_CORPUS, DEFAULT_DATASET
from aql.evaluation import evaluate_retrieval, find_regressions, load_dataset, report_json
from aql.evidence import evaluate_span_retrieval, load_evidence
from aql.evidence_gate import calibrate_dev_negatives, gated_hits
from aql.retrieval import fuse_rrf, search_lexical, search_semantic
from aql.vector_index import assert_index_complete, index_embeddings

DEFAULT_STRESS_DATASET = Path("data/evals/retrieval_stress_v1.json")
DEFAULT_EVIDENCE_DATASET = Path("data/evals/evidence_stress_v1.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--stress-dataset", type=Path, default=DEFAULT_STRESS_DATASET)
    parser.add_argument("--evidence-dataset", type=Path, default=DEFAULT_EVIDENCE_DATASET)
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
    stress_goldens = load_dataset(args.stress_dataset, args.corpus)
    evidence = load_evidence(args.evidence_dataset, args.stress_dataset, args.corpus)
    overlap = {case.question for case in goldens.data.cases} & {
        case.question for case in stress_goldens.data.cases
    }
    if overlap:
        parser.error("Stress questions must be distinct from the calibration dataset")
    embedder = PinnedFastEmbed(allow_download=args.download_model)
    questions = {case.question for case in (*goldens.data.cases, *stress_goldens.data.cases)}
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

        # Reuse the same top-10 candidate windows for fair, repeatable evaluation.
        raw_lexical = {
            case.question: search_lexical(connection, case.question, 10)
            for case in goldens.data.cases
        }
        raw_semantic = {
            case.question: search_semantic(connection, vectors[case.question], MODEL_KEY, 10)
            for case in goldens.data.cases
        }
        raw_stress_semantic = {
            case.question: search_semantic(connection, vectors[case.question], MODEL_KEY, 10)
            for case in stress_goldens.data.cases
        }
        raw_stress_lexical = {
            case.question: search_lexical(connection, case.question, 10)
            for case in stress_goldens.data.cases
        }
        # Calibrate ONLY against the original dev negatives. Challenge-set labels
        # never participate, including cases marked dev in the stress file.
        threshold = calibrate_dev_negatives(goldens.data, raw_semantic)

        def lexical(q: str, k: int):
            return raw_lexical[q][:k]

        def semantic(q: str, k: int):
            return raw_semantic[q][:k]

        def hybrid(q: str, k: int):
            return fuse_rrf(lexical(q, 10), semantic(q, 10), k)

        def semantic_gated(q: str, k: int):
            return gated_hits(raw_semantic[q], threshold=threshold.value, top_k=k)

        def stress_semantic(q: str, k: int):
            return raw_stress_semantic[q][:k]

        def stress_gated(q: str, k: int):
            return gated_hits(raw_stress_semantic[q], threshold=threshold.value, top_k=k)

        reports = {
            "lexical": evaluate_retrieval(goldens, lexical, top_k=args.top_k),
            "semantic": evaluate_retrieval(
                goldens, semantic, top_k=args.top_k, strategy=f"exact-cosine:{MODEL_KEY}"
            ),
            "hybrid": evaluate_retrieval(
                goldens, hybrid, top_k=args.top_k, strategy=f"rrf60:{MODEL_KEY}"
            ),
            "semantic_gated": evaluate_retrieval(
                goldens,
                semantic_gated,
                top_k=args.top_k,
                strategy=f"dev-negative-top1-gate:{MODEL_KEY}",
            ),
            "stress_semantic": evaluate_retrieval(
                stress_goldens,
                stress_semantic,
                top_k=args.top_k,
                strategy=f"challenge-exact-cosine:{MODEL_KEY}",
            ),
            "stress_gated": evaluate_retrieval(
                stress_goldens,
                stress_gated,
                top_k=args.top_k,
                strategy=f"frozen-original-dev-gate:{MODEL_KEY}",
            ),
        }
        span_reports = {
            "stress_spans_lexical": evaluate_span_retrieval(
                evidence,
                lambda q, k: raw_stress_lexical[q][:k],
                top_k=args.top_k,
                strategy="stress-lexical-exact-span",
            ),
            "stress_spans_semantic": evaluate_span_retrieval(
                evidence,
                stress_semantic,
                top_k=args.top_k,
                strategy=f"stress-cosine-exact-span:{MODEL_KEY}",
            ),
            "stress_spans_gated": evaluate_span_retrieval(
                evidence,
                stress_gated,
                top_k=args.top_k,
                strategy=f"stress-score-gated-exact-span:{MODEL_KEY}",
            ),
        }

    # Preserve the proven lexical floor: the semantic run must not change FTS.
    baseline = json.loads(Path("data/evals/baselines/postgres_fts_v1.json").read_text())
    regressions = find_regressions(baseline, reports["lexical"])
    if regressions:
        parser.exit(1, "Lexical regression: " + "\n".join(regressions))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, report in reports.items():
        (args.output_dir / f"{name}.json").write_text(report_json(report), encoding="utf-8")
    for name, report in span_reports.items():
        (args.output_dir / f"{name}.json").write_text(report_json(report), encoding="utf-8")

    comparison = {
        "dataset_sha256": goldens.dataset_sha256,
        "stress_dataset_sha256": stress_goldens.dataset_sha256,
        "evidence_dataset_sha256": evidence.dataset_sha256,
        "model_key": MODEL_KEY,
        "top_k_chunks": args.top_k,
        "index_stats": {
            "total": indexed.total,
            "inserted": indexed.inserted,
            "existing": indexed.existing,
        },
        "evidence_gate": {
            "threshold": threshold.value,
            "dev_negative_cases": threshold.dev_negative_cases,
            "objective": threshold.objective,
            "calibration_split": "dev",
            "holdout_used_for_calibration": False,
            "stress_labels_used_for_calibration": False,
            "note": "Top-1 cosine score proxy only; no evidence entailment or LLM abstention test",
        },
        "metrics": {name: report["summary"] for name, report in reports.items()},
        "span_metrics": {name: report["summary"] for name, report in span_reports.items()},
        "top1_semantic_scores": {
            case.case_id: {
                "split": case.split,
                "behavior": case.expected_behavior,
                "cosine": raw_semantic[case.question][0].score
                if raw_semantic[case.question]
                else None,
                "accepted": bool(semantic_gated(case.question, args.top_k)),
            }
            for case in goldens.data.cases
        },
        "stress_top1_semantic_scores": {
            case.case_id: {
                "split": case.split,
                "behavior": case.expected_behavior,
                "cosine": raw_stress_semantic[case.question][0].score
                if raw_stress_semantic[case.question]
                else None,
                "accepted": bool(stress_gated(case.question, args.top_k)),
            }
            for case in stress_goldens.data.cases
        },
        "per_case_recall": {
            row["case_id"]: {
                name: next(
                    item["recall_at_k"]
                    for item in reports[name]["cases"]
                    if item["case_id"] == row["case_id"]
                )
                for name in ("lexical", "semantic", "hybrid", "semantic_gated")
            }
            for row in reports["lexical"]["cases"]
        },
        "stress_per_case_recall": {
            row["case_id"]: {
                name: next(
                    item["recall_at_k"]
                    for item in reports[name]["cases"]
                    if item["case_id"] == row["case_id"]
                )
                for name in ("stress_semantic", "stress_gated")
            }
            for row in reports["stress_semantic"]["cases"]
        },
    }
    (args.output_dir / "comparison.json").write_text(report_json(comparison), encoding="utf-8")
    for name, report in reports.items():
        print(
            f"{name} Recall@{args.top_k}={report['summary']['recall_at_k']} "
            f"MRR@{args.top_k}={report['summary']['mrr_at_k']} "
            f"no-hit={report['summary']['unanswerable_no_hit_rate']}"
        )
    for name, report in span_reports.items():
        print(
            f"{name} span-recall@{args.top_k}={report['summary']['span_recall_at_k']} "
            f"full-coverage={report['summary']['full_span_coverage_rate']} "
            f"negative-no-hit={report['summary']['unanswerable_no_hit_rate']}"
        )
    print(f"Exact vector model: {MODEL_KEY} | comparison written to {args.output_dir}")


if __name__ == "__main__":
    main()
