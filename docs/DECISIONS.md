# Architecture decision log

## ADR-001: No agent framework in V1

**Accepted for initial design.** Current functionality is a bounded retrieve-and-measure pipeline. Autonomous tool calling or orchestration frameworks would add complexity and a larger security boundary without serving the first evaluation question.

## ADR-002: PostgreSQL + pgvector

**Accepted.** The relational source of truth stores versions, provenance and experiment results. PostgreSQL FTS is the baseline; the extension reserves a route to semantic/hybrid retrieval. At current scale, a separate vector database is unjustified. Index choice and recall trade-offs will be measured later.

## ADR-003: Separate deterministic CI from real model experiments

**Accepted.** Unit/integration checks run without a paid model. Actual answer quality will require opt-in model runs, transparent per-run metadata, reviewer calibration and budgets. CI green alone is not proof of factual LLM accuracy.

## ADR-004: Synthetic data and strict seed identity

**Accepted.** Only original fictitious operational data is included. Documents/chunks use stable IDs and hashes; conflicting reruns fail rather than silently rewrite historical sources.

## ADR-005: Local-first and minimal exposed services

**Accepted.** Docker binds only the API to loopback; PostgreSQL stays on the internal Compose network. Test DB uses its own project, credentials and ephemeral filesystem. No production security or public hosting claim.

## ADR-006: Version-locked, document-level retrieval goldens

**Accepted for checkpoint 2.** The first comparison uses a small authored synthetic dataset with explicit positive/negative cases, `dev` and `holdout` slices, source version labels, corpus SHA-256 pinning and per-case results. No network or paid model is needed. Metrics use the same chunk top-k budget as the HTTP API, then deduplicate source versions. Regressions are evaluated per case so improving aggregate scores cannot conceal a previously correct query getting worse. `unanswerable_no_hit_rate` is only a retrieval proxy; no answer abstention or grounding is claimed. The public, author-created `holdout` split is not independently blind, and version labels are coarser than evidence spans; both limits are tracked in the dataset card.

