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

Outputs in `test-results/semantic/` include separate `lexical.json`, `semantic.json`, `hybrid.json`, `semantic_gated.json` and `comparison.json`. All use the same source versions, question labels, top-k chunk budget and document-level scoring. The lexical report must match the frozen FTS baseline on every case. Semantic/hybrid scores are **not** compared to the lexical baseline through the same-strategy regression gate: instead, the side-by-side comparison displays gains and losses without suppressing trade-offs.

### Experimental evidence-sufficiency proxy (checkpoint 4)

`semantic_gated` is a **score-only policy experiment**, not a calibrated real-world abstention classifier. The cutoff is the next representable float above the highest top-1 cosine score among unanswerable cases in the **`dev` split only**. Retrieved chunks are withheld when their top-1 score is below that cutoff. The untouched `holdout` cases are evaluated afterward with the frozen threshold. The comparison artifact records the exact threshold, calibration split, per-case cosine scores and acceptance decisions. This design makes the trade-off inspectable without selecting a threshold on held-out answers.

There are only **two dev negative cases**; a policy that rejects both does **not** establish a low real-world false-accept rate. A cosine value alone never proves that a source entails an answer. Future evidence checks need reviewed claim-to-span labels, larger independent samples and model answer behavior. The current API still exposes lexical retrieval, **not** a generated answer or a confident-answer endpoint.

## Interpretation and limits

This is an experiment on **16 original, handcrafted queries**; results do not establish external generalization. In [GitHub Actions #44565483960](https://github.com/franciscovitar/ai-quality-lab/actions/runs/44565483960), commit `9471979b7903dd5cf87c62b7a9343dbae64c58a5`, the pinned model loaded and all three retrieval strategies ran against PostgreSQL 17. **Measured Recall@5/MRR@5**: FTS **0.833333/0.833333**, exact semantic **1.0/1.0**, hybrid **1.0/1.0** (12 answerable cases). Both the multi-document case `R015` and paraphrased security case `R016` improved. Yet the unanswerable zero-hit rate fell from **1.0** (FTS) to **0.0** for both vector candidates (4 negatives). **Do not equate a vector neighbor with a justified answer.** A relevance threshold and answer-level abstention classifier are separate future work.

The runtime does not report arbitrary benchmark latency, tokens, cost or faithfulness. A future benchmark should isolate cold model load, index warmup and query timing, and include more independently labeled cases.

Primary technical references: [model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), [ONNX distribution](https://huggingface.co/Qdrant/all-MiniLM-L6-v2-onnx), [FastEmbed](https://github.com/qdrant/fastembed), [pgvector exact/approximate search](https://github.com/pgvector/pgvector).
