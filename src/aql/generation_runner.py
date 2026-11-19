"""Prepare source-locked prompts and evaluate recorded local-model outputs."""

import argparse
from pathlib import Path

from aql.claim_runner import DEFAULT_EVIDENCE, DEFAULT_RETRIEVAL
from aql.eval_runner import DEFAULT_CORPUS
from aql.evaluation import report_json
from aql.evidence import load_evidence
from aql.generation_evaluation import build_prompt_batch, evaluate_generated_batch

DEFAULT_CONTEXTS = Path("data/evals/generation_contexts_v1.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "evaluate"))
    parser.add_argument("--contexts", type=Path, default=DEFAULT_CONTEXTS)
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--retrieval", type=Path, default=DEFAULT_RETRIEVAL)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument(
        "--prompts", type=Path, default=Path("test-results/generation/prompts.json")
    )
    parser.add_argument(
        "--generated", type=Path, default=Path("test-results/generation/generated-outputs.json")
    )
    parser.add_argument("--report", type=Path, default=Path("test-results/generation/report.json"))
    args = parser.parse_args()

    evidence = load_evidence(args.evidence, args.retrieval, args.corpus)
    prompts = build_prompt_batch(args.contexts, evidence)
    if args.phase == "prepare":
        args.prompts.parent.mkdir(parents=True, exist_ok=True)
        args.prompts.write_text(report_json(prompts.model_dump(mode="json")), encoding="utf-8")
        print(f"Prepared {len(prompts.cases)} fingerprint-bound synthetic generation prompts")
    else:
        report = evaluate_generated_batch(
            prompts,
            args.prompts.read_bytes(),
            args.generated.read_bytes(),
        )
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report_json(report), encoding="utf-8")
        s = report["summary"]
        print(
            f"Local generation: {s['cases']} outputs, "
            f"strict behavior matches={s['behavior_matches_expected']}, "
            f"citation syntax passes={s['citation_protocol_ok']}, "
            f"invalid citation cases={s['invalid_citation_cases']} "
            "[not answer correctness]"
        )


if __name__ == "__main__":
    main()
