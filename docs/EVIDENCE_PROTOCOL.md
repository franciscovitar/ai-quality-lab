# Evidence spans and citation integrity — experimental protocol

Retrieving a relevant **document** is not equivalent to retrieving the **sentence that supports a fact**. Nor is an authentic quote proof that a future model's answer follows from it. These are three separate evaluation questions, and AI Quality Lab now measures the first two without pretending to solve the third.

## Fixed, inspectable inputs

- **Corpus:** `data/synthetic_docs/corpus_v1.json` (current documents only, hash-locked).
- **Questions:** `data/evals/retrieval_stress_v1.json` (12 project-authored questions, 6 answerable and 6 near-miss negatives).
- **Evidence annotations:** [`data/evals/evidence_stress_v1.json`](../data/evals/evidence_stress_v1.json), locked to the **raw SHA-256 of both inputs**. Six answerable cases have eight exact answer-bearing sentence spans; six negative cases have an explicit abstain-reason category and **no** invented support spans.
- Each positive example includes a *reference answer for human review*, one or more current source version IDs, Python string character offsets and exact supporting quotes. Multi-fact questions `R102` and `R105` require **two different spans each**.

The loader rejects missing/extra/duplicated cases, mismatched behavior labels, unknown or retired source versions, forged quotes, off-by-one offsets and spans crossing chunk boundaries. To identify an eligible chunk, it recomputes the UUIDv5 from the corpus, document version, chunker version and paragraph ordinal; this is the same derivation used by ingestion. It also checks the returned hit's chunk content and source metadata before assigning evidence credit. **Neither labels nor reference answers are sent to the retriever.**

All labels were authored inside this project after seeing retrieval results. They **have not been independently reviewed** and neither `dev` nor `holdout` is blind. This is an engineering challenge set and a reproducibility fixture, not a measured production accuracy benchmark.

## What gets measured

| Layer | Measurement | What it establishes | What it cannot establish |
|---|---|---|---|
| Source retrieval | Recall@k / MRR@k | A correct document version appeared in top-k chunks | Whether the supporting sentence was present |
| Evidence retrieval | Exact anchor **span recall@k** / fraction of cases with **all required spans covered** | A returned paragraph contains every project-labeled sentence needed for a reference answer | Whether an arbitrary model answer is entailed or factually complete |
| Negative retrieval | Fraction of negative queries returning zero chunks | How often the retriever/gate emits no candidates | Actual LLM abstention; a model may decline despite hits |
| Structured citation check | Authentic source quote + document-relative offset + chunk ID actually retrieved + labeled-span coverage | No invented/stale/unretrieved citation and no missing labeled support | The meaning or causal origin of an unrestricted natural-language claim |

The structured `EvidenceSelection` contract is **decision + citations**, with **no generated answer text**. `grade_evidence_selection` returns an inspectable issue list such as `wrong_answerability_decision`, `invalid_or_unretrieved_citation`, `authentic_but_unlabeled_evidence`, or `missing_required_evidence`. Rejecting a topically related quote for the wrong claim is intentional: a true sentence from a document can still be irrelevant to the claim requiring evidence.

**Do not report a passing structured citation test as faithfulness or grounding of LLM prose.** Citation integrity validates source identity and coverage of project-authored anchors; claim entailment, paraphrase correctness, attribution causality and completeness remain unsolved here. A malicious instruction embedded in a retrieved document remains *data*, never a command for the grader or a reason to expose secrets.

## Reproduction

Unit-only validation, without database or external model:

```bash
uv run --locked pytest -q tests/unit/test_evidence.py
```

Live retrieval of those source spans is run inside the optional pinned ONNX comparison job against **disposable PostgreSQL 17**:

```bash
uv run --locked --extra semantic python -m aql.semantic_runner --download-model
```

The runner retains separate JSON artifacts named `stress_spans_lexical.json`, `stress_spans_semantic.json` and `stress_spans_gated.json` plus a `comparison.json` summary. Each report includes the evidence dataset hash, strategy, top-k chunk budget, per-case covered/required spans, full-coverage boolean, and negative no-hit proxy. No cloud embedding API, paid LLM or generated answer is involved. Do not run it on a non-disposable database.

### Observed checkpoint, not a production threshold

[GitHub Actions #44574024515](https://github.com/franciscovitar/ai-quality-lab/actions/runs/44574024515), commit `adae4ecc513ed946efd3a9e122e2b4f09f6a55e3`, executed the pinned CPU ONNX model and PostgreSQL 17, **74 tests (66 unit + 8 integration, zero omitted)**, and all three jobs passed. The uploaded comparison has evidence dataset SHA-256 `5a0d31e9bdc60380cad373364af532b10cb9638b1b4f680064fa83e6863d7795`.

| Same 12 challenge questions, top-5 **chunks** | FTS | Exact semantic | Score-gated semantic |
|---|---:|---:|---:|
| Exact labeled spans present | **0/8** | **8/8** | **8/8** |
| Answerable questions with *all* spans present | **0/6** | **6/6** | **6/6** |
| Unanswerable questions returning no chunks | **6/6** | **0/6** | **0/6** |

Both multi-span examples (`R102`, `R105`) had all their annotated passages recovered by the semantic candidate. The FTS zeros are specific to the deliberately paraphrased challenge queries and strict lexical search, **not** a statement that FTS always fails on the 16 original questions. The score gate uses only the original two dev-negative cases to determine its cutoff, and **still accepts all six near-miss negatives**. This reinforces why *present supporting chunks* and *correct decision to answer* must be separately evaluated. These results are on post-hoc authored labels, not independently reviewed judgments.

## What would justify the next step

1. Independent review of reference answers **and** supporting quotes, ideally with disagreement/adjudication records.
2. Explicit claim-level labels and counterexamples now have a first [24-pair authored benchmark](CLAIM_EVALUATION.md), but **independent review and a genuinely new holdout are still needed**.
3. A **real** entailment/abstention grader evaluated against those reviewed labels, with uncertainty reporting and a sealed sample; only then introduce bounded cited-answer generation.

Research informing the separation: [Attributable to Identified Sources (AIS)](https://aclanthology.org/2023.cl-4.2/), [AttributionBench](https://aclanthology.org/2024.findings-acl.886/), [Correctness is not Faithfulness in RAG Attributions](https://arxiv.org/abs/2412.18004), and [OpenAI evaluation best practices](https://developers.openai.com/api/docs/guides/evaluation-best-practices). These are methodology references, not claims that the current implementation matches the papers' full evaluation protocols.
