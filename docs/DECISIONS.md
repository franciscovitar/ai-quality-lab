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

## ADR-007: Pinned CPU ONNX embeddings, exact vector scan and rank fusion

**Accepted for experiment.** Use FastEmbed in an optional dependency group, explicitly download a full-SHA-pinned public ONNX snapshot, and index the synthetic corpus in the existing `vector(384)` fields. The implementation refuses mixed model revisions and incomplete indexes, and preserves transaction rollback on bad embeddings. At this scale pgvector exact cosine ordering avoids approximate-index recall loss. RRF operates on ranked chunk IDs and has a declared candidate window; it never mixes incompatible raw FTS and cosine scores. The existing lexical baseline and its regression gate stay untouched. No semantic relevance or hybrid improvement is asserted before running the real pinned model against PostgreSQL.

## ADR-008: Do not ship a score-only evidence gate

**Rejected as an answerability policy, retained as a measured negative experiment.** A top-1 cosine cutoff calibrated on the original two dev negatives rejected all four original unanswerable queries, but [a later 12-case near-miss challenge](https://github.com/franciscovitar/ai-quality-lab/actions/runs/44567666413) accepted all six new unanswerable questions while retaining six positives. Their cosine distributions overlap, so merely moving a global threshold cannot make every challenge decision correct. We keep the original threshold rule and all cases visible, never retune on the stress labels and never treat a green CI experiment as a production safety claim. The HTTP API stays lexical-only; no score-gated answer endpoint exists. Before proposing generated answers, require reviewed supporting spans, negative controls, citation verification and an explicitly evaluated abstention policy. See [the experiment](SEMANTIC_EXPERIMENT.md).

## ADR-009: Separate exact citation integrity from semantic entailment

**Accepted as an intermediate, non-generative checkpoint.** The project-authored evidence pack pins exact quotations, document versions and character offsets to the same immutable corpus as the stress retrieval goldens. Tests require that citation spans come from chunks actually retrieved and that all reference spans for a case are covered. A fabricated, retired, unrelated or un-retrieved quote is rejected even if a document title sounds relevant. The grader checks **structured evidence selection only**; it deliberately does not accept free-form answer text or claim that an authentic citation entails the answer. Independent annotation review, actual claim-to-span entailment, abstention behavior and evaluation against a larger external set are future work. See [the protocol](EVIDENCE_PROTOCOL.md).

**Measured:** [Actions #44574024515](https://github.com/franciscovitar/ai-quality-lab/actions/runs/44574024515) retrieved all eight project-labeled spans with exact semantic search, but the score-gated retriever emitted candidates for every unanswerable challenge. This does not change the decision: structural authenticity and coverage are prerequisites, not an entailment or answerability certificate.

## ADR-010: Bind claim labels to source spans and score supplied verdicts

**Accepted as a benchmark scaffold, not as a claim-entailment model.** Exactly 24 authored atomic claim-to-evidence pairs include same-citation positives and near-miss negatives, with corpus/retrieval/evidence/claim SHA provenance. The evaluator checks that the prediction set is complete, computes class-level confusion and false-supported rate, and can accept an external future judge's verdict file. The built-in citation-presence baseline intentionally declares every claim `entailed`: its expected 12/24 accuracy and 12/12 false-supported negatives demonstrate that authentic citations do not imply support. This deterministic negative control must not be called model accuracy. A genuinely independent annotated holdout and an evaluated entailment model are required before using the harness to justify actual generated answers. See [protocol](CLAIM_EVALUATION.md).

**Measured:** [CI run #44578890769](https://github.com/franciscovitar/ai-quality-lab/actions/runs/44578890769) passed 91 tests (8 real PostgreSQL, zero skips) and archived the negative control with 12/24 correct and 12/12 false supports. Its green result demonstrates reproducible *measurement of failure*, not success at entailment.

