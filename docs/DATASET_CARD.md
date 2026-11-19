# Retrieval golden dataset card — V1

**ID:** `synthetic-ops-retrieval-v1`  
**File:** `data/evals/retrieval_v1.json`  
**Corpus:** `synthetic-ops-v1`, pinned by SHA-256 of the source JSON bytes  
**Purpose:** measure current-only PostgreSQL lexical retrieval before building semantic/hybrid retrieval.

## Provenance and labels

All **16 queries and labels were authored for this portfolio laboratory** against a fictitious webhook-operations corpus. They are not user logs, production queries, or third-party benchmarks. A label points to a version-qualified relevant document (`slug@vN`) rather than a subjective model response. There has been **no independent human adjudication**, real-user representativeness study, or annotation-agreement measurement.

| Split | Answerable | Unanswerable | Total |
|---|---:|---:|---:|
| `dev` | 6 | 2 | 8 |
| `holdout` | 6 | 2 | 8 |
| **Total** | **12** | **4** | **16** |

Cases cover exact lexical matches, paraphrases, multi-document questions, operations, security, stale-version questions and out-of-domain negatives. `R015` intentionally requires two sources. `R013` asks about a retired policy, which must never be retrieved as a current source. `R016` asks about malicious instructions *as data*, not as instructions for the evaluator to follow.

The `holdout` split is a **declared comparison subset**, but the queries are public and authored by this project. It is *not* an independent blind test set, so repeated tuning against it would invalidate any generalization claim. We report both splits separately, and a stronger external/held-out set is future work.

## Metric semantics

- `top_k_chunks`: retrieval budget in chunks (the API's actual contract), maximum 10. Multiple chunks from the same document are deduplicated before calculating document-level metrics.
- `Recall@k`: relevant *current document versions* seen in the first `k` returned chunks, divided by all labeled relevant documents; measured only for answerable queries.
- `MRR@k`: reciprocal rank of the first relevant unique document, with rank determined by deduplicated document order; measured only for answerable queries.
- `HitRate@k`: fraction of answerable questions retrieving at least one relevant version. This does not imply every required source was found.
- `unanswerable_no_hit_rate`: fraction of unanswerable questions returning **no chunks**. This is a conservative retrieval-only proxy, **not an LLM abstention rate**. A model could abstain correctly despite a retrieved hit, or hallucinate without hits.
- Per-case rows include ranks, expected/retrieved versions and source IDs; aggregate metrics are macro averages, and reports include the `dev`/`holdout` and tag slices. Empty subgroup metrics are `null`, never an invented zero.

**Key limitation:** evaluation labels are document-level, not answer-bearing chunk spans. A hit in an unrelated paragraph of the correct document can score as relevant. A future dataset revision should add reviewed evidence spans and use chunk-level citation metrics; a changed corpus byte or retired expected version makes the current suite fail to load until the labels are explicitly reviewed.

## Reproducibility and regression policy

`python -m aql.eval_runner` requires a migrated, seeded PostgreSQL database. It produces deterministic JSON (no wall-clock fields), including source corpus and dataset hashes, retrieval strategy, top-k, every case and per-slice metrics. A repeated run on the same database snapshot should yield identical bytes. GitHub Actions publishes this synthetic report as a 14-day artifact.

`--compare prior.json` refuses reports with different dataset/corpus fingerprints, strategy or top-k, and fails on **any per-case loss** in Recall/MRR or an extra hit on an unanswerable case. Aggregate gains cannot hide individual regressions. The observed PostgreSQL 17 report from [GitHub Actions #44548351392](https://github.com/franciscovitar/ai-quality-lab/actions/runs/44548351392) is pinned at `data/evals/baselines/postgres_fts_v1.json`; CI compares future same-dataset runs against it. This is a **regression floor for the lexical baseline**, not an independently validated quality target. Improvements may pass without changing the snapshot, while changing data/labels requires deliberate re-baselining.

The pinned initial score is Recall@5 = MRR@5 = **0.833333** over 12 answerable cases (10 successful, two complete misses); all four negative questions had zero retrieved chunks. `R015` (multi-document) and `R016` (paraphrase around malicious instructions) are the two misses. It would be incorrect to report these numbers as model accuracy, evidence grounding or a robust external benchmark.

No LLM was called. The dataset does not measure answer faithfulness, citation correctness, toxicity, model latency, token usage, dollars or production reliability. Those require later, separately instrumented experiments.

Methodology references: [OpenAI evaluation best practices](https://developers.openai.com/api/docs/guides/evaluation-best-practices), [BEIR retrieval metrics](https://github.com/beir-cellar/beir/wiki/Metrics-available).
