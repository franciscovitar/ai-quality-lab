"""Run the pinned local NLI model and score its 3-way verdicts on authored claims."""

import argparse
from pathlib import Path

from aql.answer_evaluation import evaluate_answers, load_answers
from aql.claim_evaluation import evaluate_claim_predictions, load_claims
from aql.claim_runner import DEFAULT_CLAIMS, DEFAULT_EVIDENCE, DEFAULT_RETRIEVAL
from aql.eval_runner import DEFAULT_CORPUS
from aql.evaluation import report_json
from aql.generated_claim_review import load_and_review
from aql.generated_nli import evaluate_generated_nli, generated_nli_pairs
from aql.generation_evaluation import PromptBatch
from aql.generation_runner import DEFAULT_CONTEXTS
from aql.nli import MAX_PAIR_TOKENS, PinnedNliCrossEncoder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_CLAIMS)
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--retrieval", type=Path, default=DEFAULT_RETRIEVAL)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output", type=Path, default=Path("test-results/nli/nli-report.json"))
    parser.add_argument("--download-model", action="store_true")
    parser.add_argument("--answer-dataset", type=Path)
    parser.add_argument("--generated-claims", action="store_true")
    parser.add_argument(
        "--generated-claims-output",
        type=Path,
        default=Path("test-results/nli/generated-claims-nli-report.json"),
    )
    parser.add_argument(
        "--answer-output",
        type=Path,
        default=Path("test-results/nli/answer-composition-model-report.json"),
    )
    args = parser.parse_args()

    goldens = load_claims(args.dataset, args.evidence, args.retrieval, args.corpus)
    model = PinnedNliCrossEncoder(allow_download=args.download_model)
    observations = model.infer(goldens)
    report = evaluate_claim_predictions(
        goldens,
        [observation.prediction for observation in observations],
        strategy="pinned-minilm2-cross-encoder-nli-argmax-v1",
    )
    report["model"] = {
        "snapshot": model.model_key,
        "runtime": "onnxruntime-cpu",
        "input": "exact source quote(s) as premise; claim text as hypothesis",
        "max_pair_tokens": MAX_PAIR_TOKENS,
        "truncation": "reject (never silently truncate)",
        "three_way_to_binary": "entailment = entailed; contradiction or neutral = not_entailed",
        "calibration": "none; argmax raw logits",
    }
    report["model_observations"] = [observation.as_dict() for observation in observations]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report_json(report), encoding="utf-8")

    if args.answer_dataset is not None:
        answers = load_answers(args.answer_dataset, goldens)
        answer_report = evaluate_answers(
            answers,
            [observation.prediction for observation in observations],
            strategy="pinned-nli-on-preannotated-answer-compositions-v1",
        )
        answer_report["model"] = model.model_key
        args.answer_output.parent.mkdir(parents=True, exist_ok=True)
        args.answer_output.write_text(report_json(answer_report), encoding="utf-8")
        s = answer_report["summary"]
        print(
            f"Composed answers {s['decision_agreement']}/{s['cases']} decisions match "
            f"authored reference, false approvals={s['false_approvals']}, "
            f"false rejections={s['false_rejections']} [not real generated answers]"
        )

    if args.generated_claims:
        frozen_prompts = Path("data/evals/frozen_generation_v1/prompts.json")
        reviewed = load_and_review(
            prompts_path=frozen_prompts,
            generated_path=Path("data/evals/frozen_generation_v1/generated-outputs.json"),
            review_path=Path("data/evals/generated_claim_review_v1.json"),
            context_path=DEFAULT_CONTEXTS,
            evidence_path=args.evidence,
            retrieval_path=args.retrieval,
            corpus_path=args.corpus,
        )
        prompt_batch = PromptBatch.model_validate_json(frozen_prompts.read_bytes())
        pairs = generated_nli_pairs(prompt_batch, reviewed)
        generated_observations = model.infer_pairs([pair.model_input() for pair in pairs])
        generated_report = evaluate_generated_nli(
            pairs=pairs, observations=generated_observations, reviewed_report=reviewed
        )
        args.generated_claims_output.parent.mkdir(parents=True, exist_ok=True)
        args.generated_claims_output.write_text(report_json(generated_report), encoding="utf-8")
        g = generated_report["summary"]
        print(
            f"NLI on actual generated claims: {g['agreement_with_provisional_author']}/"
            f"{g['claim_occurrences']} agreement with project author; "
            f"author-rejected/model-supported={g['author_reject_model_support']} "
            "[NOT independent factuality]"
        )

    summary = report["summary"]
    print(
        f"Pinned NLI {goldens.data.dataset_id}: {summary['correct']}/{summary['cases']} "
        f"correct; false-supported={summary['confusion']['not_entailed_as_entailed']}/"
        f"{summary['not_entailed']}; macro-F1={summary['macro_f1']} "
        f"[model={model.model_key}; authored labels, not a safety guarantee]"
    )


if __name__ == "__main__":
    main()
