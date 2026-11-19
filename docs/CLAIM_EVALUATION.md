# Claim-to-evidence evaluation — labeled pairs, not an automated entailment judge

The earlier [evidence protocol](EVIDENCE_PROTOCOL.md) showed that a quote can be authentic, current and actually retrieved, without proving that an arbitrary *claim* follows from it. This checkpoint introduces an explicit benchmark for that missing decision. **The benchmark is implemented and scored; no natural-language entailment model is implemented or evaluated yet.**

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

The file must include **all 24 IDs** exactly once, and its SHA must match the current claim dataset. The two rows above illustrate the schema only; they are **not a complete runnable predictions file**. Run with `--predictions path/to/predictions.json`; the evaluator refuses stale fingerprints and invalid labels. This interface permits future pinned NLI or LLM judges to be evaluated using the same frozen references. No grade is claimed until a real candidate supplies all verdicts and is run.

## Limitations and next gate

- The label is an authored judgment, not a mathematical consequence of offsets. Quote provenance is mechanically checked; **semantic entailment labels still need independent human review**.
- A correct 24-way verdict classifier would **not** prove that a language model generated a factually complete response, used the citation in reasoning, or avoided unsupported extra claims. Those require a separate model-output dataset and claim decomposition.
- Two labels and 24 samples cannot establish calibrated probabilities, real-world false-accept rates or security guarantees. Do not optimize repeatedly on this public challenge and call its `holdout` score a generalization estimate.
- A future, opt-in judge needs a fixed model revision, recorded latency/cost if applicable, negative control tests, confusion analysis, and a separately held-out **independently reviewed** benchmark before any generated-answer integration.

Relevant methodology: [Attributable to Identified Sources](https://aclanthology.org/2023.cl-4.2/), [AttributionBench](https://aclanthology.org/2024.findings-acl.886/), and [OpenAI evaluation best practices](https://developers.openai.com/api/docs/guides/evaluation-best-practices). This project has not implemented their full grading protocols.
