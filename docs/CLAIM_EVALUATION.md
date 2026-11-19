# Claim-to-evidence evaluation — labeled pairs, not an automated entailment judge

The earlier [evidence protocol](EVIDENCE_PROTOCOL.md) showed that a quote can be authentic, current and actually retrieved, without proving that an arbitrary *claim* follows from it. This checkpoint introduces an explicit benchmark for that missing decision, plus an **experimental pinned ONNX natural-language inference (NLI) model**. That model has been measured on authored labels, but is **not** a verified factuality or safety gate.

## Locked protocol and labels

[`data/evals/claim_pairs_v1.json`](../data/evals/claim_pairs_v1.json) contains **24 manually authored atomic claims**, each paired with one or more of the eight validated source spans in [`evidence_stress_v1.json`](../data/evals/evidence_stress_v1.json). Its raw SHA-256 binding to that evidence dataset forms a chain back to the retrieval goldens and synthetic corpus. Loader validation also rejects unknown span indices, mismatched versions/quotes through the evidence loader, duplicated claim IDs, repeated refs, malformed verdicts and an unexpected dataset size.

| Split | Entailed | Not entailed | Total |
|---|---:|---:|---:|
| `dev` | 6 | 6 | 12 |
| `holdout` | 6 | 6 | 12 |
| **Total** | **12** | **12** | **24** |

The labels distinguish only **`entailed`** from **`not_entailed`** (which includes explicit contradictions, missing precise details, scope changes and unsupported extensions). They do not force the grader to infer contradiction versus ignorance when either means *do not assert this claim from this quote*. Negative cases have project-authored reasons and a category; positive cases may depend on more than one span.

**Key adversarial pair:** `C001` and `C002` cite **exactly the same authentic passage** about retrying HTTP 429/503 with jitter. The passage supports the claim that 429/503 are retryable. It **does not** support an exact jitter percentage. This is a falsifiable example of why *citation exists* is weaker than *claim follows from citation*.

The cases are **not blind or independently adjudicated**. They were authored in this repository after inspecting the source and earlier retrieval results. The `holdout` split shares source spans and subject matter with `dev`, so it must **not** be sold as an independent test of a learned grader. Independent review and a new sealed corpus are prerequisites for external accuracy claims.

## Metrics and report contract

`evaluate_claim_predictions` receives an **explicit verdict for each claim**. It does not infer a verdict from text, query a model or look up the expected label to generate a prediction. Its deterministic JSON report includes every case and label, a confusion matrix, accuracy, balanced accuracy, positive precision/recall, not-entailed recall, macro-F1, false-supported rate, dev/holdout slices and negative-error-category slices. It refuses missing, duplicated or extra case IDs.

The **false-supported rate** is especially important here: 

\[ \operatorname{FSR} = \frac{\text{not-entailed claims predicted entailed}}{\text{all not-entailed claims}} \]

The built-in baseline `unsafe-citation-presence-baseline-v1` predicts `entailed` for **every** case solely because all cases have valid supporting-source references. It is intentionally unsafe and must be *reported as a negative control*, not advertised as an AI model. With the authored 12/12 balance, it gets **12/24 cases correct (50% accuracy), macro-F1 0.333333 and 12/12 false supports (FSR 1.0)**. Green CI means this negative control executed, **not** that it passed a safety threshold.

**Observed in [GitHub Actions #44578890769](https://github.com/franciscovitar/ai-quality-lab/actions/runs/44578890769), commit `9f8a28e349291b04c0b5f2c70e15eafee838b407`:** all three jobs passed, with **91 tests (83 unit, 8 PostgreSQL integration), zero skipped**, real HTTP Compose and the pinned ONNX experiment. The runner printed `12/24 correct, false-supported=12/12, macro-F1=0.333333`; I checked the uploaded `claim-support-report.json` inside the `retrieval-evaluation` artifact. The report binds to claim dataset SHA-256 `1c6dc7fb386aa6da7f506caf98fc40788bfa52371688de8349c13b1ed42b53c6`, records **6/12 correct on each public split** and includes all 24 individual predictions. This is measured behavior of a deliberately naive constant-label rule, **not** a trained entailment judge.

Run locally without PostgreSQL, embeddings or any paid service:

```bash
uv run --locked pytest -q tests/unit/test_claim_evaluation.py
uv run --locked python -m aql.claim_runner --output test-results/claim-support-report.json
```

You can score predictions from a separate, opt-in model adapter **without changing the goldens**:

```json
{
  "schema_version": 1,
  "dataset_sha256": "<sha256 of claim_pairs_v1.json raw bytes>",
  "predictions": [
    {"case_id": "C001", "label": "entailed"},
    {"case_id": "C002", "label": "not_entailed"}
  ]
}
```

The file must include **all 24 IDs** exactly once, and its SHA must match the current claim dataset. The two rows above illustrate the schema only; they are **not a complete runnable predictions file**. Run with `--predictions path/to/predictions.json`; the evaluator refuses stale fingerprints and invalid labels. This interface also permits other judges to be compared against the same frozen references.

## Actual local NLI experiment — observed, with failures

**Model:** [`cross-encoder/nli-MiniLM2-L6-H768`](https://huggingface.co/cross-encoder/nli-MiniLM2-L6-H768), [Apache-2.0](https://huggingface.co/cross-encoder/nli-MiniLM2-L6-H768/blob/9aa22b250d25a10ac39ab4a6b2bc28213507cb0c/config.json), revision `9aa22b250d25a10ac39ab4a6b2bc28213507cb0c`, **only** `onnx/model.onnx` plus its configuration/tokenizer, executed through ONNX Runtime on CPU. It was originally trained on SNLI/MultiNLI, **not on our operational citations**. The optional download is roughly 329 MB. No hosted inference endpoint, paid API, Torch runtime or label-informed calibration is used.

`src/aql/nli.py` feeds **the exact, validated quote(s) as premise** and **the claim as hypothesis**. It does not include the expected label, error category or project-authored rationale. It checks the pinned model's three-class mapping and token budget before inference; input pairs over 256 tokens are rejected, never silently truncated. The largest raw logit determines `contradiction`/`entailment`/`neutral`. Only `entailment` is mapped to `entailed`; the other two map to `not_entailed`. The archived report retains model identity, per-case logits (not calibrated probabilities), predicted labels and token counts.

**Real GitHub runner evidence:** [Actions #44580522095](https://github.com/franciscovitar/ai-quality-lab/actions/runs/44580522095), code commit `8dd0514ab0cee19430fb538540f8f944758c6a84`. All **four jobs passed**, including the real ONNX experiment, Compose HTTP and **104 tests (96 unit, 8 PostgreSQL integration, 0 skipped)**. The `claim-nli-experiment` artifact contains the actual 24-case report.

| Measured metric, 24 authored pairs | Citation-presence negative control | Pinned ONNX NLI |
|---|---:|---:|
| Correct | 12/24 | **20/24** |
| Accuracy | 50.00% | **83.33%** |
| Macro-F1 | 0.333333 | **0.832168** |
| Unsupported claims falsely approved | 12/12 | **1/12** |
| Supported claims incorrectly rejected | 0/12 | **3/12** |

The NLI confusion matrix was **TP 9, TN 11, FP 1, FN 3**, with **10/12 correct** in each public split. Four mistakes remain:

- `C001` and `C018` are **supported** HTTP 429/503 retry-policy claims; the model classified both `neutral` (false rejections).
- `C010` is a **supported** replay-audit requirement; the model classified it `contradiction` (false rejection).
- `C005` reverses the incident timeline: it says the worker crashed *before* storing a receipt, though the quote says *after*. The model classified it `entailment` — a **false approval of a direct contradiction**. This is the most important unsafe error; the raw entailment logit was 1.850035.

These are **not** regression thresholds: deliberately changing prompts/thresholds on the same tiny public set would overfit it. A green experiment job confirms execution and artifact preservation, **not** that the model met a release-quality acceptance criterion.

Reproduce in a local isolated environment with Python 3.13 (download the public pinned snapshot only when you explicitly allow it):

```bash
uv sync --locked --extra semantic
uv run --locked --extra semantic python -m aql.nli_runner --download-model
# Later runs can omit --download-model to use only the cached snapshot:
uv run --locked --extra semantic python -m aql.nli_runner
```

This needs no PostgreSQL and writes `test-results/nli/nli-report.json`. The normal service, base dependencies and standard Python tests do not download this model.

## Limitations and next gate

- The label is an authored judgment, not a mathematical consequence of offsets. Quote provenance is mechanically checked; **semantic entailment labels still need independent human review**.
- A correct 24-way verdict classifier would **not** prove that a language model generated a factually complete response, used the citation in reasoning, or avoided unsupported extra claims. Those require a separate model-output dataset and claim decomposition.
- Two labels and 24 samples cannot establish calibrated probabilities, real-world false-accept rates or security guarantees. Do not optimize repeatedly on this public challenge and call its `holdout` score a generalization estimate.
- This ONNX judge is pinned and measured, but has **four case-level mistakes**, including a direct contradiction falsely supported. It is **not enabled in the HTTP API** and no answer is generated. Latency benchmarking, independent adjudication, broader stress testing and a separately sealed test set are still missing.
- A later model-output evaluator must also measure decomposition/coverage of generated answers, refusal behavior and end-to-end attribution — not just isolated claim pairs.

Relevant methodology: [Attributable to Identified Sources](https://aclanthology.org/2023.cl-4.2/), [AttributionBench](https://aclanthology.org/2024.findings-acl.886/), and [OpenAI evaluation best practices](https://developers.openai.com/api/docs/guides/evaluation-best-practices). This project has not implemented their full grading protocols.
