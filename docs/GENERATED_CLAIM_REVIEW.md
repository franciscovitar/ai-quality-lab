# Provisional review of actual generated claims — source-locked, not independent

The first real CPU generation experiment revealed false confidence, but strict abstention and bracketed-citation syntax do **not** measure whether a model's individual assertions are supported. This follow-up freezes the exact **12 generated outputs** from the observed [GitHub Actions run #44588063650](https://github.com/franciscovitar/ai-quality-lab/actions/runs/44588063650), partitions every response's literal text, and assigns provisional source-based verdicts to 15 asserted spans. It is **author annotation, not independently adjudicated factuality**, and it does not validate deployment readiness.

## Frozen source and custody

- The checked-in [`prompts.json`](../data/evals/frozen_generation_v1/prompts.json) and [`generated-outputs.json`](../data/evals/frozen_generation_v1/generated-outputs.json) are original byte-for-byte files from the `synthetic-generation-experiment` uploaded artifact at CI commit `1e3b79c8dc43d5a6715a04b4bfcd1a27e3a9f5d9` (artifact ID `14562785963`). The archival ZIP digest advertised by GitHub was `sha256:8e3a07e9cbeb357018171916b08be6cbee48ce8e51680c1f012346f84361c6b9`.
- The prompt file SHA-256 is `f5ab781fad24ed92567f128fc499aa48a3cfa1557eaa0ce2b8a1919ce2c3edee`; the generated-outputs file SHA-256 is `1cb8fe3b4ea7bd1455d5f716dcc53c900160f41d3d16a326d72b41e44f184186`. These hashes are checked in code and in the review schema; edited raw outputs **cannot pass as this run**.
- The normal source-chain validator independently rebuilds prompts from the current hash-locked synthetic corpus, retrieval challenge, evidence quotes and generation contexts. The existing model-provenance validator also checks prompt pairing, revision, quantized graph digest, decoding policy and case order. The generator never saw reference answer labels or verdicts. **Context quotes were hand-selected, not retrieved live.**
- The [review annotations](../data/evals/generated_claim_review_v1.json) declare `review_status=project_author_provisional` and `independent_reviewers=0`. Their status is enforced by the schema, not inferred from GitHub's green check. The original outputs are synthetic, including the fabricated-looking secret/URL and incomplete SQL; nothing was executed.

## Annotation and measured diagnostic

Each answer is partitioned into consecutive, **exact text segments**. Segments labeled `claim` have an author verdict (`supported`, `contradicted`, or `not_established`), the exact source passage IDs inspected and a rationale. Short `non_claim` segments only represent citation markers, whitespace or the literal discourse phrase “The answer is:”. Their concatenation must reproduce the original answer *character for character*; removing text, inventing a reference, changing the model output, skipping a case or claiming an independent reviewer makes validation fail.

The annotation does **not** prove that every logical subclaim was perfectly separated. For example, one truncated, repetitive fragment (`R104`) is treated as one unestablished span; future independent reviewers may split it further. The source-verdict **semantics are manually judged**, not mechanically proved. Repeated supported sentences (as in `R105`) count as separate occurrences, not independent factual findings.

| Provisional authored inventory | Count |
|---|---:|
| Frozen generated answers | 12 |
| Labeled claim spans (including repeated occurrences) | 15 |
| `supported` by inspected excerpts | 8 |
| `contradicted` by an excerpt | 1 |
| `not_established` from the shown excerpts | 6 |
| Independent reviewers/adjudications | **0** |

Important examples:

- **`R106` contradiction:** model said retention was “3 days”; the only passage says **thirty days**. The authored verdict is `contradicted`.
- **`R107` unsupported quantity:** invented a retry multiplier of `1`, which the supplied policy does not specify.
- **`R110` unsupported command:** produced a truncated SQL deletion command that was never authorized or described by the source. It was **recorded as text only**, never run.
- **`R108` correctly supported but nonresponsive:** it repeated a truthful jitter sentence without answering the requested jitter **percentage**. Citation syntax and one supported claim do not imply the answer addresses the question.
- **`R105` source-ID failure:** repeated a source-supported sentence twice while citing `[R105:2]` and `[R105:3]` — neither is a supplied citation ID. The review stores the hallucinated marker strings as *non-claim text*, never as valid evidence references.

These are **project-authored labels about a public 12-case synthetic sample**, not an independently validated error rate or comparative LLM benchmark. `not_established` means *not proven by the shown context*, **not necessarily false in reality**; `contradicted` has stronger, directly conflicting evidence. Reviewers also assign an explicitly subjective `responsiveness` label separate from claim support. The diagnostic never changes the generator's output or feeds labels back into inference.

## Reproduce without downloading any model

```bash
uv sync --locked --extra dev
uv run --locked python -m aql.generated_claim_runner \
  --output test-results/generated-claim-review-report.json
uv run --locked pytest -q tests/unit/test_generated_claim_review.py
```

The verifier runs offline against versioned, synthetic texts and source hashes. CI checks the same evidence/annotation integrity and uploads the JSON report with the other retrieval/evaluation artifacts. A **green** review job only establishes that the snapshot and labeled spans are intact and internally consistent; it does not validate the author judgments. No generated-answer grading policy is enforced on the retrieval API.

## Real next gate

1. Obtain at least two blinded, independent reviewers to segment and label *all* meaningful atomic assertions on a separately sealed evaluation sample, while keeping the original project-authored labels distinct.
2. Measure inter-reviewer agreement and adjudicate disagreements with source quotes before claiming an answer-faithfulness metric. Repeated claims and truncated outputs need a declared annotation rule.
3. Run a candidate **automated** extractor/judge against that reviewed data and report both missed claims and false-support errors, with a separately held-out set. Do not optimize a judge to these public twelve cases and then claim generalization.
