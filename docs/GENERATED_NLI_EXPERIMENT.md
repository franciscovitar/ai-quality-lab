# Pinned NLI on actual generated claims — agreement is not factuality

This experiment extends the [frozen real-output review](GENERATED_CLAIM_REVIEW.md). It reuses the project's public, full-revision-pinned `cross-encoder/nli-MiniLM2-L6-H768` ONNX classifier to judge **15 author-segmented claims from 12 actual model-generated answers**, against only the exact source passages shown to the generator. The goal is to measure failures and feasibility, **not** to validate a production answer gate.

## Protocol

1. Validate the byte-frozen prompt/output SHA-256 values, generator graph metadata and complete source-bound [provisional annotations](../data/evals/generated_claim_review_v1.json). The original model outputs cannot be silently changed or replaced.
2. For each claim occurrence, build a unique `Rxxx/Cyy` identifier. NLI premise = **only** the inspected, version-locked source quote(s) actually supplied to the generator. NLI hypothesis = the **exact model claim**. No author verdict, rationale, expected answerability, question label or claimed reference correctness is supplied to the classifier.
3. Use the existing [`cross-encoder/nli-MiniLM2-L6-H768`](https://huggingface.co/cross-encoder/nli-MiniLM2-L6-H768) at fixed revision `9aa22b250d25a10ac39ab4a6b2bc28213507cb0c`, CPU ONNX Runtime, configuration-validated three-class label mapping and argmax. Token pairs above 256 tokens are errors (no silent truncation). **No training or tuning** on these twelve public prompts.
4. Only NLI `entailment` counts as a model-supported claim. `neutral` and `contradiction` reject support; record the *three-way* label, logits, token count, inspected exact premise, original text and source/review digests. Compare predictions to the **project author's provisional** `supported` versus `contradicted`/`not_established` labels, not an independently adjudicated gold standard.
5. Separately compute a deliberately unsafe **negative-control answer policy**: *accept a response whenever every author-extracted claim receives model entailment*. This policy does not check if the response answers the question, whether citations are valid, whether required facts are missing or whether the question should have been refused. It is not used by the API.

The original 24 authored claim-pair experiment and 21 composed-answer fixtures still run with the **same loaded NLI model**. Real-output comparison is an optional flag in the experiment runner; the production FastAPI search remains retrieval-only.

## First actual local CPU result — provisional comparison

On a local pinned ONNX run against the frozen GitHub generator outputs:

| Measure | Observation |
|---|---:|
| Author-reviewed generated claim occurrences | 15 |
| Agreement with *provisional author labels* | **14/15** |
| Model falsely supported an author-rejected claim | 0/7 |
| Model rejected an author-supported claim | **1/8** |
| Model three-way output | 7 entailment, 7 neutral, 1 contradiction |
| Responses naively accepted by all-claims-entailed rule | 4/12 |
| **Unsupported questions naively accepted despite requiring abstention** | **1/6** |
| Independent author-label adjudications | **0** |

### Why the apparently high agreement still fails at answer level

- **`R102/C01` is the sole disagreement.** The author provisionally marks the statement “the incident report describes why two sends led to only one durable effect” as supported by the two incident passages; the NLI model assigns **neutral** (`logits=[-2.226362,-0.556308,2.674664]`). The author decision is not independently adjudicated; model/author agreement is *not* validated model accuracy.
- **`R108/C01` is the unsafe whole-answer false acceptance.** The model correctly sees support for the sentence “Retryable HTTP 429 and 503 responses use exponential backoff with jitter”, yet the original question asks for the **percentage of jitter**, which neither the source nor the reply supplies. The all-claims-entailed policy would accept this **unanswerable** case; the strict abstention protocol expected a refusal. This is why entailment of stated claims is not a sufficient answer-quality test.
- **`R105` illustrates fabricated citations passing a claim-only judge.** Both duplicated claim occurrences are classified as entailed because the underlying sentence is sourced; the actual generator output cites `[R105:2]` and `[R105:3]`, which were never given. An all-claims-entailment policy accepts the response despite its invalid evidence markers.
- **`R106/C01` is a direct contradiction**: the generated answer said **3 days**; the source says **thirty days**. Here NLI returned its *contradiction* class, which is an encouraging diagnostic observation, not sufficient validation of overall reliability.

The negative control accepted exactly `R101`, `R103`, `R105` and `R108`. At least `R108` must abstain; `R105` has invalid citations. Other answers contain explicit unsupported numbers or commands and were rejected by the claim-only rule. None of these labels should be treated as a general hallucination, safety or factuality rate: the sample is tiny, project-authored, public and includes a duplicated claim.

## Reproduction

Python 3.13 and `uv` with the optional, pinned ONNX-only dependency group:

```bash
uv sync --locked --extra dev --extra semantic
uv run --locked --extra semantic python -m aql.nli_runner \
  --answer-dataset data/evals/answer_cases_v1.json \
  --generated-claims --download-model
uv run --locked pytest -q tests/unit/test_generated_nli.py
```

The public-model download is **explicit** with `--download-model`; omit it to require a locally cached model. It is a research-only execution; third-party download/runtime components may produce network or telemetry traffic and must be used only in an appropriate environment. The job archives its complete observed `test-results/nli/generated-claims-nli-report.json` as a short-lived CI artifact; the report is generated by the pinned model and is **not fabricated from test fixtures**.

## Next real gate

The missing critical inputs are **independent reviewers**, a separately sealed set with exhaustive atomic-claim labels, answer-completeness/contradiction annotations and an adjudication record. A future automated claim extractor must be evaluated for **missed assertions** as well as wrong entailment, and a proposed answer gate must combine question sufficiency, citation integrity and cross-claim consistency. Do not train or tune against this public 12-case record and report the same set as a blind evaluation.
