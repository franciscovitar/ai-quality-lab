# Semantic and hybrid retrieval experiment

This experiment is **separate** from the unchanged PostgreSQL full-text baseline. It downloads a public ONNX model only with an explicit `--download-model` flag and runs on CPU. No hosted embeddings or paid LLM calls occur.

## Fixed implementation choices

| Item | Pinned value |
|---|---|
| Library | FastEmbed `0.9.1`, in the optional `semantic` dependency group |
| Model name | `sentence-transformers/all-MiniLM-L6-v2` |
| ONNX distribution | `Qdrant/all-MiniLM-L6-v2-onnx` |
| Model snapshot | `323b6364c3138880024d5ff3c353c9aa0d2e45ff` |
| Dimensions | 384; only finite nonzero vectors accepted |
| Execution | CPU ONNX Runtime, two threads, no custom remote Python code |
| Similarity | pgvector cosine, exact nearest neighbor; current versions only |
| Hybrid | Reciprocal Rank Fusion (`k=60`), same chunk IDs and top-k result budget |

The pinned ONNX snapshot is from 2025, before a later tokenizer truncation/padding mismatch reported upstream. A commit-pinned snapshot is loaded via Hugging Face `snapshot_download(revision=...)`, then handed to FastEmbed via `specific_model_path`; FastEmbed is set to `local_files_only=True` so it never silently substitutes the repository's moving `main`. Model-specific metadata (repo+commit+library) is stored with each vector. It is not safe to compare vectors from different model identities.

For just 18 synthetic chunks, **exact cosine search** is preferable to HNSW: approximate search can introduce a *second* source of recall loss independent of embedding quality. No ANN speedup is claimed. Full-text search and vector ranking use different score scales, so hybrid combines **ranks**, not raw scores.

## How to reproduce on a disposable database

After migrating and seeding the isolated PostgreSQL test database (see README), run:

```bash
uv sync --locked --extra dev --extra semantic
# One-time public model download, only if permitted:
uv run --locked --extra semantic python -m aql.semantic_runner --download-model
# Subsequent runs can be offline (cached pinned snapshot):
uv run --locked --extra semantic python -m aql.semantic_runner
```

The runner materializes 384-dimensional vectors in the same PostgreSQL database and verifies that **all current chunks** are indexed before evaluating. It refuses a different model already stored on any chunk and does not overwrite existing vectors. Run it only against the disposable demo/test DB, not against unknown existing data. A failed transaction rolls back all of its updates.

Outputs in `test-results/semantic/` include separate `lexical.json`, `semantic.json`, `hybrid.json`, `semantic_gated.json`, `stress_semantic.json`, `stress_gated.json` and `comparison.json`. All use the same corpus versions, top-k chunk budget and document-level scoring; the **original** and **stress** datasets are scored separately. The lexical report must match the frozen FTS baseline on every original case. Semantic/hybrid scores are **not** compared to the lexical baseline through the same-strategy regression gate: instead, the side-by-side comparison displays gains and losses without suppressing trade-offs.

### Experimental evidence-sufficiency proxy (checkpoint 4)

`semantic_gated` is a **score-only policy experiment**, not a calibrated real-world abstention classifier. The cutoff is the next representable float above the highest top-1 cosine score among unanswerable cases in the **`dev` split only**. Retrieved chunks are withheld when their top-1 score is below that cutoff. The untouched `holdout` cases are evaluated afterward with the frozen threshold. The comparison artifact records the exact threshold, calibration split, per-case cosine scores and acceptance decisions. This design makes the trade-off inspectable without selecting a threshold on held-out answers.

There are only **two dev negative cases**; a policy that rejects both does **not** establish a low real-world false-accept rate. A cosine value alone never proves that a source entails an answer. Future evidence checks need reviewed claim-to-span labels, larger independent samples and model answer behavior. The current API still exposes lexical retrieval, **not** a generated answer or a confident-answer endpoint.

**Observed in [GitHub Actions #44566570399](https://github.com/franciscovitar/ai-quality-lab/actions/runs/44566570399), commit `85fb40bee432464a30d3192ad19ac39fa568ddb5`:** the two dev negatives had top-1 cosine scores `0.259384…` and `0.321321…`; the cutoff was the next float after `0.32132143739823826`. The two untouched holdout negatives scored `0.067599…` and `0.209700…` and were also rejected. All 12 answerable questions (including the two previous FTS misses) were accepted with labeled relevant sources in top-5. The lowest accepted positive scored `0.445688…`, a gap of about `0.1244` above the cutoff. This is a property of *this* small synthetic corpus and model snapshot; the same rule may fail on near-miss questions, changed documents or other domains. The runner artifact retains the exact 16 per-case scores so this statement is auditable.

### Post-hoc near-miss challenge: frozen threshold, new queries

[`retrieval_stress_v1.json`](../data/evals/retrieval_stress_v1.json) adds **12 new project-authored questions** against the **same hash-locked corpus**: six labeled answerable paraphrases and six labeled unanswerable near misses. The negatives ask for missing numeric configuration, a nonexistent approval identity, an unsafe SQL deletion command, an absent incident queue-depth measurement, or a nonexistent production key. Both `dev` and `holdout` each have three positives and three negatives, but neither split is used to retune the threshold: **the gate is still calibrated exclusively from the original dataset's two dev negatives**. The runner embeds the combined questions in one batch, applies that frozen threshold to the stress set and publishes per-case scores and reject/accept decisions. This is intended to expose false acceptance of superficially relevant text.

These questions were written **after** the original scores had been inspected, and their labels were not independently adjudicated. The stress file is a regression challenge set, not a blind test or statistical estimate. The same original lexical baseline remains unchanged. Even a successful reject/accept decision is **not** equivalent to supported answer generation; a later answer-level evaluation needs annotated evidence spans and citation verification.

#### Observed failure: semantic proximity is not answer sufficiency

**Measured in [GitHub Actions #44567666413](https://github.com/franciscovitar/ai-quality-lab/actions/runs/44567666413), commit `5eaea02184a75c9fb07d4893597dcf68f5c699df`.** All three CI jobs passed: **56 tests (48 unit, 8 PostgreSQL integration), zero skipped**, full Compose HTTP smoke, and the real pinned ONNX model job. On the original 16 cases, `semantic_gated` retained the 12 answerable hits and rejected all 4 negatives. On the new 12-case stress set, the identical within-run cutoff **accepted 6/6 answerable questions but also accepted 6/6 unanswerable questions**. Thus stress `Recall@5 = MRR@5 = 1.0` on six positives, while **unanswerable no-hit rate = 0/6**. This is a failed *retrieval acceptance policy*, not a failure of the unit/DB/Compose tests and not a measured hallucination rate (no LLM generated answers).

The cutoff measured in this run was `0.3213213980197907`, from the two original dev negatives only. The stress negative cases had these top-1 cosine scores and all passed the cutoff:

| Case | Missing or disallowed fact | Top-1 cosine | Gate decision |
|---|---|---:|---|
| R107 | Exact exponential retry multiplier | 0.699 | Accepted incorrectly |
| R108 | Backoff jitter percentage | 0.694 | Accepted incorrectly |
| R109 | Named replay approver | 0.515 | Accepted incorrectly |
| R110 | Destructive SQL deletion command | 0.610 | Accepted incorrectly |
| R111 | Queue depth at incident time | 0.541 | Accepted incorrectly |
| R112 | Real production API key | 0.523 | Accepted incorrectly |

The **lowest positive** top-1 cosine on the stress set was `0.484884…` (`R102`), while the **highest negative** was `0.698740…` (`R107`). These ranges overlap. Therefore, **no single threshold on the top-1 cosine can simultaneously accept all six labeled positives and reject all six labeled negatives in this challenge set**, regardless of how the scalar cutoff is selected. Increasing the threshold to reject `R107` would necessarily reject at least `R102`. This is stronger evidence than merely observing that our initial cutoff needs adjustment: score-only relevance cannot represent answerability for these examples.

The two runs gave original-dev thresholds differing by about `0.00000004` despite the pinned model; floating-point cosine values should be compared with a tolerance, not assumed byte-identical across runners. The acceptance failure is far larger than that drift. The exact model ID, dataset SHA-256 hashes, per-case scores, decisions and metrics are preserved in the short-lived `semantic-retrieval-experiment` artifact. **A green workflow here means the experiment executed reliably, not that the safety policy passed its challenge.** The challenge has no production enforcement or API endpoint.

**Next research gate:** annotate source spans that actually entail each expected fact (including explicit absence), independently review a larger hard-negative sample, and evaluate an evidence-to-claim abstention decision separately from embedding similarity. Do not choose a new cosine threshold from these stress labels and present the same set as unbiased proof.

## Interpretation and limits

This is an experiment on **16 original, handcrafted queries**; results do not establish external generalization. In [GitHub Actions #44565483960](https://github.com/franciscovitar/ai-quality-lab/actions/runs/44565483960), commit `9471979b7903dd5cf87c62b7a9343dbae64c58a5`, the pinned model loaded and all three retrieval strategies ran against PostgreSQL 17. **Measured Recall@5/MRR@5**: FTS **0.833333/0.833333**, exact semantic **1.0/1.0**, hybrid **1.0/1.0** (12 answerable cases). Both the multi-document case `R015` and paraphrased security case `R016` improved. Yet the unanswerable zero-hit rate fell from **1.0** (FTS) to **0.0** for both vector candidates (4 negatives). **Do not equate a vector neighbor with a justified answer.** A relevance threshold and answer-level abstention classifier are separate future work.

The runtime does not report arbitrary benchmark latency, tokens, cost or faithfulness. A future benchmark should isolate cold model load, index warmup and query timing, and include more independently labeled cases.

Primary technical references: [model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), [ONNX distribution](https://huggingface.co/Qdrant/all-MiniLM-L6-v2-onnx), [FastEmbed](https://github.com/qdrant/fastembed), [pgvector exact/approximate search](https://github.com/pgvector/pgvector).
