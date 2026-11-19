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

## ADR-011: Experiment with a pinned NLI model; do not use it as a release gate

**Accepted for experiment, rejected for production answer validation.** A public, Apache-2.0 cross-encoder trained on SNLI/MultiNLI is downloaded only with explicit opt-in at a full model commit. ONNX Runtime CPU runs the fixed graph with exact, hash-validated source quotes as premise and authored claims as hypothesis; the model sees neither label nor rationale. The pinned config must declare the expected contradiction/entailment/neutral class mapping, and any pair above 256 tokens is rejected instead of silently truncated. Entailment alone counts as evidence support; we do not optimize a threshold on this public 24-case set. The API does not call the model.

**Measured:** [CI run #44580522095](https://github.com/franciscovitar/ai-quality-lab/actions/runs/44580522095), commit `8dd0514ab0cee19430fb538540f8f944758c6a84`, passed 104 tests with zero skips and all four jobs, including the isolated real-model experiment. The model scored 20/24, macro-F1 0.832168, and 1/12 false supports, versus the naive citation baseline's 12/24 and 12/12 false supports. The one false approval `C005` directly reverses an incident timeline, so green CI is **not** an evidence-safety acceptance. No independently adjudicated or sealed test set, latency SLA, answer extraction or generalization claim exists. See [the full failure analysis](CLAIM_EVALUATION.md#actual-local-nli-experiment--observed-with-failures).

## ADR-012: Evaluate composed response decisions without pretending to extract claims automatically

**Accepted as a diagnostic fixture scaffold; rejected as a production answer gate.** Twenty-one deliberately composed answers pin their source label chain through SHA-256 and explicitly enumerate every sentence's claim and citation. Required claim IDs provide a *manually specified* minimum-fact checklist. A response is accepted only if it addresses an answerable question, contains all required IDs, has exact citation-to-reference alignment and every asserted claim receives `entailed`; an unanswerable question must be declined. This separation makes incomplete yet true responses, irrelevant authentic citations, unsafe answered negatives and unnecessary abstention observable. We record an authored reference oracle and unsafe citation-only baseline distinctly; neither is model-generated evidence. A model experiment can supply a complete set of isolated claim verdicts, but these do not check discourse-level consistency or extract all assertions from unrestricted prose.

**Measured:** [CI run #44582652843](https://github.com/franciscovitar/ai-quality-lab/actions/runs/44582652843), commit `fdcd55fe076f249ad018d785be988bf07cb5043d`, passed 124 tests (8 real PostgreSQL, zero skipped), Compose and both ONNX jobs. Against the authored reference decisions for 21 synthetic responses, the citation-only rule falsely approved 2; the actual NLI model's reused claim verdicts matched **18/21**, with **one false approval and two false rejections**. The false approval `A021` is an internally contradictory timeline accepted because the isolated sentence judge had called both opposing statements supported. This is why **exhaustive claim extraction, independent review and whole-answer consistency tests** precede any use as a safety or deployment gate. See [answer protocol](ANSWER_EVALUATION.md).

## ADR-013: Preserve actual generated failures before building an answer endpoint

**Accepted for isolated research; rejected as deployment readiness.** To close the gap between fully authored answer fixtures and actual free-form outputs, use a small, public and full-SHA-pinned quantized ONNX text generator with locked CPU-only Node inference. The 12 prompt contexts contain only exact, version-locked synthetic-source quotes, never reference decisions or evaluation rationales. Generation uses a declared greedy policy and bounded output tokens; the graph bytes are checked against SHA-256 and every prompt/output/duration is archived. A missing or tampered source hash must fail closed. No paid API, live retrieval-to-generation orchestration or new HTTP route is authorized by this decision.

**Measure narrow behaviors without relabeling them as factuality.** Exact `INSUFFICIENT_EVIDENCE` is the strict refusal protocol. Citation checks validate bracket markers against the passages shown in the prompt, but do not establish entailment, completeness or correct reasoning. In particular, a case may pass marker syntax while the prose never answers the question. Future independent reviewers must exhaustively annotate all atomic generated claims, and evaluate the reviewed set separately from the project-authored public examples. Prompt fixes after observing this run require a new version rather than a rewritten historical baseline.

**Measured:** [CI run #44585749946](https://github.com/franciscovitar/ai-quality-lab/actions/runs/44585749946), commit `45adfdda6c64b2a33cf2d1466f0018c5f1b21462`, finished with five jobs green and 144 tests (8 PostgreSQL integration, zero skips). Its **actual 12 generated outputs** failed the exact abstention protocol **6/6 times** on unanswerable questions; only **3/12** outputs passed structural citation checks, and one response fabricated source markers. Some outputs invented numerical details or SQL not present in the supplied passages. The report and raw answers remain in the time-limited `synthetic-generation-experiment` Actions artifact. Green CI here means the failure evidence is reproducible and preserved, not that the generator is accurate. See [the experiment](GENERATION_EXPERIMENT.md).

## ADR-014: Freeze generated outputs and verify annotation coverage without claiming adjudication

**Accepted as a source-locked author review; not a factuality release gate.** Freeze original JSON bytes from a named, previously successful GitHub generator run, alongside raw prompt SHA-256 and model/source revision checks. Keep every generated character in an explicit ordered claim/non-claim segment. A non-claim cannot silently swallow arbitrary prose; each authored claim must identify source passages the model actually saw and record `supported`, `contradicted` or `not_established`, with rationale. Changing a response, skipping an answer, dropping text or inventing inspected evidence is a test failure. The annotation schema fixes `review_status=project_author_provisional` and `independent_reviewers=0`. These guards prove traceability and basic coverage; they **cannot establish that claims are atomic, that labels are semantically accurate or that a reviewer is independent**.

**Pre-CI local observation:** the original 12 outputs from [Actions #44588063650](https://github.com/franciscovitar/ai-quality-lab/actions/runs/44588063650) yield 15 author-marked claim spans: 8 supported, 1 explicitly contradicted (`R106`, 3 instead of 30 days), and 6 not established. `R108` proves that a claim can be labeled supported yet fail to answer the question. The incomplete `R104` is a declared segmentation limitation. This classification is provisional until independent reviewers adjudicate it. See [the review protocol](GENERATED_CLAIM_REVIEW.md).

