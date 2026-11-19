# AI Quality Lab

**Engineering a reliable retrieval system is more than getting an LLM to answer a prompt.** AI Quality Lab is a portfolio project for measurable, testable RAG systems: a versioned synthetic corpus, reproducible retrieval experiments, evidence-backed answers, and automated quality gates.

### Start here: understand the result in two minutes

**Open the local visual demo at [http://localhost:8001/demo](http://localhost:8001/demo)** after running `docker compose up --build --wait`. Pick one of five frozen, **real model responses** and compare what the model wrote with the exact source passages it received. Start with **R106**: it confidently says *3 days* when the sole provided document says *thirty days*. Other cases show invented citations, an answer that misses the question, unsupported SQL and a supported response.

The browser makes **no live model calls**; the source files and annotations are synthetic, version-locked project artifacts. Citation syntax and abstention checks are automatic; factual-support labels are **provisional judgments by the project author**, with zero independent reviewers. Green CI means the implementation and reproducibility checks passed, **not** that the model is safe.

- **[Two-minute engineering case study](docs/CASE_STUDY.md):** the problem, one concrete failure, trade-offs and observed evidence.
- **[Project direction (demo-first)](docs/PORTFOLIO_FOCUS.md) · [decision issue #2](https://github.com/franciscovitar/ai-quality-lab/issues/2):** what we prioritize next and what is paused.

> **Current state — retrieval, citation integrity, experimental NLI and an isolated real generation run verified in CI.** PostgreSQL/FTS, Docker Compose/HTTP, pinned CPU ONNX embeddings and claim-to-evidence inference have been exercised on GitHub runners. The pre-annotated answer evaluator remains separate from a **12-question, actual small-model generation experiment**. That run is now frozen with **provisional author claim annotations**; the generator failed all six strict abstention cases. **There is no independent factuality validation, deployed RAG answer endpoint or production accuracy claim.** See [the real-generation experiment](docs/GENERATION_EXPERIMENT.md), [provisional claim review](docs/GENERATED_CLAIM_REVIEW.md) and [roadmap](#roadmap).

## Why this exists

Backend correctness and AI answer correctness are different problems. An integration test can prove that SQL and API contracts work, but cannot prove that a generated answer is supported by the right sources. The full lab will separate retrieval evaluation, answer grounding, abstention, operational cost, and reproducibility.

The sample corpus is wholly synthetic. It describes fictional webhook operations to provide meaningful cases involving stale policies, duplicate deliveries, on-call runbooks, and malicious instructions embedded in documents. No customer, university, or private PAS data is used.

## Architecture in one minute

- **Python/FastAPI:** explicit API contracts; retrieval and evaluation will remain inspectable modules.
- **PostgreSQL 17 + pgvector:** relational versions, full-text baseline and a reserved vector column for later hybrid retrieval.
- **Synthetic corpus:** source versions, hashes, paragraph offsets and stable IDs; never update a source silently.
- **Golden retrieval sets:** 16 original cases plus a separately reported 12-case near-miss challenge; current-version labels, positive/negative outcomes, split/tag slices and hash-locked corpus provenance.
- **Evidence annotations:** eight exact source spans for six challenge positives, six explicit abstention labels and a strict structured citation-integrity evaluator. This verifies quote provenance and required-span coverage, **not free-form LLM entailment**.
- **Claim benchmark:** 24 authored claim-to-span pairs (12 supported, 12 not), fingerprint-bound verdict evaluation, confusion matrix, an unsafe citation-presence baseline **and a measured pinned ONNX NLI baseline**. The model makes four errors, including one false approval of a contradiction; it is not a safety gate.
- **Composed-answer benchmark:** 21 deliberately authored response fixtures tied to those atomic claims. Checks exact citation alignment, minimum required facts and abstention, and compares an oracle, an unsafe citation baseline and actual NLI verdicts. These **pre-annotated fixtures** are not graded real model outputs.
- **Real generation experiment:** separately run a revision- and SHA-256-pinned 135M-parameter ONNX model on 12 source-bound prompts, preserve its raw responses and evaluate **only** exact abstention and citation-marker syntax. Hand-selected source passages are not live retrieval. No automatic atomic-claim extraction or generated-answer factuality grade exists.
- **Frozen-output claim review:** preserve the exact 12 real CPU outputs from GitHub, partition every character into labeled claim/non-claim spans and record 15 provisional **project-authored** claim assessments. The verifier checks text/source integrity, **not whether those judgments are true**; there are zero independent reviewers.
- **NLI on real generated claims:** reuse the pinned cross-encoder to classify those 15 source/claim pairs and compare predictions with the **provisional** labels. It correctly rejects the retention-period contradiction but a deliberately unsafe all-claims-entailed answer rule still approves an unanswerable question. No independently validated factuality result follows.
- **Human-review handoff, not human results:** deterministic source-locked, label-hidden packet for 12 real answers/15 author-segmented claims; two **blank** templates, strict submission validation, disagreement reporting and Cohen's kappa with undefined edge cases. **No actual external reviewers have contributed yet.**
- **Semantic experiment:** optional, commit-pinned local ONNX embeddings with pgvector exact cosine and RRF, compared against the same goldens; no paid model API.
- **Testing:** pure unit tests plus PostgreSQL integration tests; CI fails on missing/skipped DB coverage.
- **Future:** independently reviewed real-output claims and abstention, bounded retrieval-linked answers with citations, comparative model-output evaluation and a dashboard.

See [Architecture](docs/ARCHITECTURE.md), [Dataset card](docs/DATASET_CARD.md), [Semantic experiment](docs/SEMANTIC_EXPERIMENT.md), [Evidence protocol](docs/EVIDENCE_PROTOCOL.md), [Claim-evaluation protocol](docs/CLAIM_EVALUATION.md), [Composed-answer protocol](docs/ANSWER_EVALUATION.md), [Actual generation experiment](docs/GENERATION_EXPERIMENT.md), [Frozen-output claim review](docs/GENERATED_CLAIM_REVIEW.md), [NLI on real generated claims](docs/GENERATED_NLI_EXPERIMENT.md), [Human-review protocol](docs/HUMAN_REVIEW_PROTOCOL.md) and [Decisions](docs/DECISIONS.md) for details and limits.

## Quick start — full local demo stack

Prerequisites: Docker Desktop/Engine with Compose and capacity to start PostgreSQL 17.

```bash
docker compose up --build --wait
# Open http://localhost:8001/demo in your browser for the five-case walkthrough.
curl http://localhost:8001/health
curl http://localhost:8001/ready
curl http://localhost:8001/v1/corpus/versions
curl -X POST http://localhost:8001/v1/retrieval/search \
  -H 'Content-Type: application/json' \
  -d '{"question":"What is the retry budget for webhooks?","top_k":5}'
```

`compose.yaml` starts PostgreSQL, applies `0001_corpus`, verifies the synthetic seed, and then starts the API. PostgreSQL is **not** exposed to the host; the API binds only to `127.0.0.1:8001`. All credentials are obvious, disposable placeholders and must never be reused outside an isolated machine.

The CI workflow runs `scripts/compose_smoke.py` against that same full stack, checking migrated readiness, all six current documents, a current-policy search, and **the frozen demo API, HTML, JS and CSS over real HTTP**. Inspect the completed run before considering the new demo smoke verified.

To stop without deleting the named demonstration volume:

```bash
docker compose down
```

If you intentionally need to erase the demo database, inspect the Compose project and its volume first. Do not use `down -v` against unrelated projects.

## Local Python tests

Use Python 3.13 and [uv](https://docs.astral.sh/uv/) for the committed dependency lockfile. Docker Compose is required for database-backed integration tests; the unit suite can run without Docker.

```bash
uv sync --locked --extra dev
uv run --locked pytest -q tests/unit
```

For real PostgreSQL tests (from the checkout root):

```bash
docker compose -f compose.test.yaml up -d --wait
export DATABASE_URL=postgresql://aql_test:aql_test_local_only@127.0.0.1:55435/ai_quality_test
export TEST_DATABASE_URL="$DATABASE_URL"
uv run --locked alembic upgrade head
REQUIRE_INTEGRATION_TESTS=1 uv run --locked pytest -q --junitxml=test-report.xml
uv run --locked python scripts/check_test_report.py test-report.xml --min-integration 8
uv run --locked python -m aql.seed
uv run --locked python -m aql.eval_runner --output test-results/retrieval-report.json
docker compose -f compose.test.yaml down
```

On Windows PowerShell, set variables with `$env:DATABASE_URL = '...'` and `$env:TEST_DATABASE_URL = $env:DATABASE_URL`; then set `$env:REQUIRE_INTEGRATION_TESTS = '1'`. The test Compose project uses a disposable database name `ai_quality_test` and a temporary data mount; its port is loopback-only. Tests refuse other DB names and nonlocal hosts.

Without `TEST_DATABASE_URL`, integration tests skip **locally** and are **not evidence**. In CI, `REQUIRE_INTEGRATION_TESTS=1` plus the JUnit checker makes missing or skipped integration tests fail.

## Version-locked retrieval evaluation

The [golden dataset](data/evals/retrieval_v1.json) specifies 16 original, fictional queries (12 answerable, 4 unanswerable). Labels reference exact, *current* source versions rather than free-form model answers. The loader checks the corpus SHA-256, rejects stale labels, duplicate questions and invalid positive/negative classifications. There is no independent human adjudication or real-world representativeness claim.

After migrating and seeding your **disposable** local test database as above, run:

```bash
uv run --locked python -m aql.eval_runner --top-k 5 \
  --output test-results/retrieval-report.json
# Compare only against a prior report from the same dataset, corpus and config:
uv run --locked python -m aql.eval_runner --top-k 5 \
  --compare data/evals/baselines/postgres_fts_v1.json
```

Reports contain document-level Recall@k, MRR@k and HitRate@k for answerable questions; a separate `unanswerable_no_hit_rate` measures whether *retrieval* returns nothing on unanswerable questions. **It does not measure LLM abstention, correctness, grounding or cost.** Rankings are bounded by `top_k` **chunks** first, then deduplicated by document version for scoring. Each failure remains visible by case and tag; optional baseline comparison fails if *any* previously correct case regresses, even when averages improve. CI stores the JSON as a short-lived artifact.

The declared `holdout` split is public and project-authored, not a blind external test set. On [GitHub Actions #44548351392](https://github.com/franciscovitar/ai-quality-lab/actions/runs/44548351392), the 16 cases yielded **Recall@5 = 83.33%, MRR@5 = 83.33%, and 4/4 unanswerable questions with zero retrieved chunks**. The two failures were a multi-document question and a paraphrase, consistent with strict lexical AND. The real PostgreSQL JSON report is stored as a [versioned baseline](data/evals/baselines/postgres_fts_v1.json), and CI will reject per-case regressions against it. The [dataset card](docs/DATASET_CARD.md) documents all limitations, including document- rather than evidence-span-level labels.

### Optional semantic and hybrid comparison

The pinned ONNX model uses FastEmbed on CPU; it is **not installed or downloaded by the normal API image**. On an isolated, migrated and seeded PostgreSQL database with `DATABASE_URL` set:

```bash
uv sync --locked --extra dev --extra semantic
uv run --locked --extra semantic python -m aql.semantic_runner --download-model
```

The `--download-model` option explicitly fetches the public 90 MB model, fixed to a known snapshot. Without it, the runner requires the pinned model to be cached. The script writes embeddings transactionally, refuses other model revisions, then evaluates FTS, exact cosine and rank-based hybrid on identical questions and generates comparison JSON. The normal CI tests pgvector with controlled synthetic vectors; the **separate real-model job passed** in [Actions #44565483960](https://github.com/franciscovitar/ai-quality-lab/actions/runs/44565483960). The measured trade-off is important:

| 16 synthetic questions | FTS | Exact semantic | RRF hybrid |
|---|---:|---:|---:|
| Recall@5 on 12 answerable cases | 83.33% | **100%** | **100%** |
| MRR@5 on 12 answerable cases | 83.33% | **100%** | **100%** |
| Zero hits on 4 unanswerable cases | **4/4** | **0/4** | **0/4** |

Semantic/hybrid retrieval found both source versions needed by the handcrafted multi-document case and the paraphrased security case. But vector search **always returns neighbors**; it retrieved content on every unanswerable question too. This is *not* a pass on answer safety, abstention or generalization. An exploratory relevance-score gate is evaluated below; see [implementation and reproducibility](docs/SEMANTIC_EXPERIMENT.md).

An **experimental score-only gate** now uses the maximum cosine score of the two unanswerable `dev` questions to choose a rejection cutoff, then applies it unchanged to `holdout`. Its results are reported separately as `semantic_gated`; they are **not** included in the 100% raw semantic retrieval claim. It does not verify entailment, citations or LLM abstention. See the [experiment protocol](docs/SEMANTIC_EXPERIMENT.md).

A second [`retrieval_stress_v1.json`](data/evals/retrieval_stress_v1.json) challenge set introduces 12 deliberately difficult questions (six answerable, six *near-miss* unanswerable). The same frozen, original-dev threshold is applied to those questions **without recalibration**; `stress_semantic.json` and `stress_gated.json` expose per-case acceptance and retrieval outcomes. This post-hoc, project-authored dataset is neither blinded nor independently reviewed, and its scores must be assessed separately from the original 16-question comparison.

The [score-gate CI run #44566570399](https://github.com/franciscovitar/ai-quality-lab/actions/runs/44566570399) verified **54 tests (46 unit, 8 PostgreSQL integration), zero omitted**, and all three jobs passed. The observed dev-derived threshold was **0.3213214374**; gated semantic retrieval retained **12/12 relevant source sets** and returned no results for **4/4 unanswerable cases**. These numbers are preliminary by design: two dev negatives cannot calibrate a reliable real-world acceptance policy. The model comparison artifact records every cosine score and decision.

**The harder questions exposed the limitation.** In [Actions #44567666413](https://github.com/franciscovitar/ai-quality-lab/actions/runs/44567666413), commit `5eaea02184a75c9fb07d4893597dcf68f5c699df`, all three jobs were green (**56 tests: 48 unit + 8 PostgreSQL integration, 0 skipped**, real HTTP Compose, pinned ONNX). Applying the original-dev score cutoff *without tuning on the 12 new questions* retained **6/6 answerable cases but falsely accepted all 6/6 unanswerable near misses**. Even more importantly, the lowest positive top-1 cosine (`0.485`) was **below** the highest negative (`0.699`), so **no single cutoff can classify all 12 correctly**. The artifact preserves each case and score. Green CI means the experiment ran; **it is not an answer-safety pass**. No LLM answer was generated or measured. See [the complete failure analysis](docs/SEMANTIC_EXPERIMENT.md#observed-failure-semantic-proximity-is-not-answer-sufficiency).

### Exact evidence spans and citation integrity

The separate [evidence protocol](docs/EVIDENCE_PROTOCOL.md) adds eight project-authored answer-bearing quotes across the six challenge positives. Each quote is anchored to a source version and exact character offsets, and the loader verifies it against the hash-locked corpus and the stable chunk ID. The experimental runner now reports **exact span coverage** separately from document recall for lexical, semantic and score-gated retrieval. A structured decision/citation grader rejects fake, stale, unretrieved or irrelevant quotes and flags missing required spans. These are **deterministic source-integrity checks**, not an automatic judge of a generated claim's meaning, confidence or faithfulness. The current HTTP API remains lexical-only, with no answer generator.

**Measured in [Actions #44574024515](https://github.com/franciscovitar/ai-quality-lab/actions/runs/44574024515), commit `adae4ecc513ed946efd3a9e122e2b4f09f6a55e3`:** all three jobs passed with **74 tests (66 unit, 8 PostgreSQL integration), 0 skipped** and real HTTP Compose. On the 12-case challenge, FTS retrieved **0/8** labeled supporting spans, while exact semantic and score-gated retrieval found **8/8**, including both required passages for the two multi-fact cases. **But neither semantic strategy returned zero hits on any of the six unanswerable questions (0/6)**; FTS returned no hits on 6/6. This is a small, purpose-built dataset: strong evidence retrieval and failed rejection coexist. The per-case reports and dataset hash are stored in the run's `semantic-retrieval-experiment` artifact. No generated answer, entailment or citation faithfulness was tested.

### Claim-to-evidence evaluation — negative control vs. real experimental NLI

The [claim benchmark](docs/CLAIM_EVALUATION.md) adds **24 authored claim/quote pairs** that deliberately mix supported and unsupported statements using the **same authentic source passages**. Every label is tied by SHA-256 through the evidence dataset to the unchanged source corpus. The CLI scores *supplied* `entailed`/`not_entailed` verdicts, rejects missing or duplicated predictions, and reports false-supported rate and a full confusion matrix. Its initial **citation-only negative control** predicts every claim is supported: it correctly classifies **12/24**, but falsely marks **12/12 unsupported claims as supported**. Those are baseline arithmetic on authored labels, **not model-generated judgments or a verified entailment system**. CI runs and archives the negative-control report separately from the retrieval metrics.

**Observed on [Actions #44578890769](https://github.com/franciscovitar/ai-quality-lab/actions/runs/44578890769), commit `9f8a28e349291b04c0b5f2c70e15eafee838b407`:** **91 tests passed** (83 unit + 8 PostgreSQL integration; 0 omitted), with all three jobs green. The archived claim report confirmed the full 24-case confusion matrix: **12/24 accuracy, 12/12 falsely supported negatives, macro-F1 0.333333**. This is an intentionally unsafe *negative control*, not an LLM faithfulness result. The earlier FTS regression baseline and the full Compose HTTP smoke also passed.

The next experiment adds an **actual NLI model**: [`cross-encoder/nli-MiniLM2-L6-H768`](https://huggingface.co/cross-encoder/nli-MiniLM2-L6-H768) at a **full pinned revision**, run locally with ONNX Runtime on CPU. It receives the exact source quote and claim, never a reference answer or rationale. The model's three NLI labels become a binary evidence verdict; there is no gold-based threshold optimization or automatic access from the HTTP API.

**Measured on [Actions #44580522095](https://github.com/franciscovitar/ai-quality-lab/actions/runs/44580522095), commit `8dd0514ab0cee19430fb538540f8f944758c6a84`:** all **4 jobs passed**, **104 tests (96 unit + 8 PostgreSQL integration, zero skips)**, full real Compose HTTP and both optional ONNX jobs. The actual model got **20/24 correct**, macro-F1 **0.832168**, with **1/12 unsupported claims falsely approved** and **3/12 supported claims rejected**. Of particular concern, `C005` reversed a receipt/crash timeline but was still classified as supported. This is promising as a diagnostic experiment, **not a calibrated safety or generalization result**. The full 24-case predictions, logits and confusion matrix are in the `claim-nli-experiment` artifact; see [failure analysis and reproduction](docs/CLAIM_EVALUATION.md#actual-local-nli-experiment--observed-with-failures).

### Entire-answer compositions — what a claim grader misses

A new [response-level dataset](data/evals/answer_cases_v1.json) composes **21 manually pre-annotated synthetic answers** around the twelve stress questions, including incomplete answers, authentic-but-wrong citations, invented details, needless abstentions, correct refusals and directly contradictory sentences. A strict loader pins the answer dataset to the 24 original claim labels and source evidence; the scorer checks exact reference-claim coverage, per-sentence citation alignment, all-claims-supported and answer-versus-abstain behavior. No text generation or automatic claim extraction is implied.

**Observed on [Actions #44582652843](https://github.com/franciscovitar/ai-quality-lab/actions/runs/44582652843), commit `fdcd55fe076f249ad018d785be988bf07cb5043d`:** all **four jobs green**, **124 tests (116 unit + 8 real PostgreSQL integration, 0 omitted)**, Compose HTTP and both pinned ONNX jobs. Against the author's own fixture labels, the citation-only control **falsely accepted 2 of 21 answers**. Reusing actual model NLI verdicts gave **18/21 matching reference decisions, with 1 false approval and 2 false rejections**. Crucially, `A021` states a crash happened both *before* and *after* saving a receipt, and the NLI-based evaluator **approved that contradictory answer** because its isolated claim judge had mistakenly marked both claims supported. This experiment demonstrates a real limitation rather than a production-quality pass. See the [protocol, full per-case errors and reproduction](docs/ANSWER_EVALUATION.md).

### Actual generated answers — strict abstention and citation syntax

The [first real-generation experiment](docs/GENERATION_EXPERIMENT.md) runs a public, pinned SmolLM2 135M-parameter **quantized ONNX model locally on CPU** using Node 24 and locked `@huggingface/transformers`. It sees 12 questions and **hand-selected exact synthetic-source excerpts**, never expected labels or answers. This off-API exercise stores every raw response, the prompt and source-chain hashes, model graph SHA-256 and inference duration. It does **not** invoke the live retrieval pipeline or automatically extract claims.

**Observed on [GitHub Actions #44585749946](https://github.com/franciscovitar/ai-quality-lab/actions/runs/44585749946), commit `45adfdda6c64b2a33cf2d1466f0018c5f1b21462`:** all **five jobs passed**, including 144 Python tests (136 unit, **8 real PostgreSQL integration, 0 skipped**), HTTP Compose, two earlier ONNX research jobs and this new generator. The archived raw-output report showed **0/6 correct strict abstentions** on unsupported questions, only **3/12 structurally valid citation outcomes**, and **one answer with invented citation markers**. Actual generations also supplied a made-up SQL deletion instruction and unsupported numeric details. Those failures are visible even though CI is green: the job tests reproducibility and honest reporting, **not model acceptance**. The citation result is a syntax check, *not* a measure of answer factuality or completeness.

### Frozen real-answer claims — author review, no independent validation

The [claim review protocol](docs/GENERATED_CLAIM_REVIEW.md) freezes the exact prompt/output bytes from [CI #44588063650](https://github.com/franciscovitar/ai-quality-lab/actions/runs/44588063650) into versioned synthetic JSON. Every character of all 12 outputs is covered by a claim or narrowly permitted non-claim segment. The **15 provisional author-labeled claim spans** include **8 supported, 1 contradicted and 6 not established** by the excerpts inspected. The contradiction is concrete: the generator asserted a **3-day** retention period when the only source says **30 days**. The labels are **not independently reviewed**; one repetitive, incomplete output also illustrates that full text coverage does **not** prove perfect atomic decomposition. Mutating a frozen output, omitting a segment or referencing an unseen source fails integrity tests. CI runs this deterministic review separately from real model inference.

### Real pinned NLI on the model's own generated claims

The [real-output NLI experiment](docs/GENERATED_NLI_EXPERIMENT.md) uses the same already-pinned CPU cross-encoder to examine all 15 manually segmented *real* generated claims against exactly their inspected source quotes. The model sees **no author label, rationale or expected response**, and retains raw logits and three-class verdicts for inspection. **Observed locally and in [Actions #44592082406](https://github.com/franciscovitar/ai-quality-lab/actions/runs/44592082406)**, it agreed with provisional author labels on **14/15**, with one supported claim marked neutral (`R102`). **Crucially**, an unsafe rule that accepts an answer whenever *all* its extracted claims are model-entailed would accept **4/12 responses, including `R108`, which asks an unanswerable jitter-percentage question**. It also accepts `R105`, whose generator output fabricated citation markers. This is measured model behavior versus provisional labels, **not independently established factuality, a calibrated release gate or evidence that the API can safely generate answers**.

### Label-hidden human-review materials — no actual adjudication yet

The [human review protocol](docs/HUMAN_REVIEW_PROTOCOL.md) now generates an **exact-source review packet** and two **deliberately empty** reviewer templates for the 12 frozen real answers and 15 author-segmented claims. Reviewers can separately assess source support and whether each answer addresses the question. Completed submissions must cover every question/claim, use only inspected source IDs and match the packet digest; the comparison reports three-class and binary agreement, Cohen's kappa and unresolved disagreements. It **does not fabricate reviewers or adjudication**: no external humans have reviewed the data, the segmentation remains authored by the project, and a public dataset cannot be retroactively described as a blind holdout. Run `uv run --locked python -m aql.human_review_runner prepare` after installing the dev dependencies; CI archives only the **blank** materials.

## Current endpoints

| Method | Path | Behavior |
|---|---|---|
| `GET` | `/health` | Process liveness, no DB dependency |
| `GET` | `/ready` | 200 only with reachable, migrated PostgreSQL + pgvector |
| `GET` | `/v1/corpus/versions` | Current synthetic documents |
| `POST` | `/v1/retrieval/search` | Parameterized FTS search, top_k 1–10, current versions only |
| `GET` | `/demo` | Static, recruiter-friendly Spanish demo, no database or model required |
| `GET` | `/v1/demo/examples` | Five SHA-bound original model outputs with exact source passages |

This is an intentionally **non-agentic** first checkpoint. A search result is not an AI-generated answer. `score` is PostgreSQL FTS rank, **not** model confidence or grounding quality.

## Roadmap

1. **Foundation:** schema, seed, lexical baseline, reproducible PostgreSQL tests and CI.
2. **Dataset:** version-locked goldens, split/coverage audit, deterministic retrieval report and pinned lexical regression baseline. **Verified with PostgreSQL 17 in CI.**
3. **Semantic and hybrid retrieval:** pinned ONNX adapter, exact pgvector search and rank fusion; **measured with a real CPU model in CI**.
4. **Evidence and generation:** exact evidence spans, pre-annotated composition and an **isolated real CPU generator experiment** now exist. The score gate failed near-miss cases, the experimental NLI judge falsely approved a contradiction, and the tiny generator failed all six strict abstention checks. None is deployable as an answerability classifier.
5. **Answer evals:** pre-annotated compositions, real generation, a provisional author claim review, NLI judgments on actual outputs and a label-hidden reviewer handoff are implemented. **Additional scoring layers and recruitment of independent reviewers are paused**, not represented as complete.
6. **Current portfolio priority:** five-case visual walkthrough, one-command Compose and the concise [case study](docs/CASE_STUDY.md). The [persistent direction](docs/PORTFOLIO_FOCUS.md) takes precedence over an older experimental roadmap. A larger dashboard and deployment are deliberately out of scope.

No production deployment, real user credentials, private data or paid LLM usage is implied by this roadmap. A public deployment would require separate authentication, rate limiting, privacy and security review.

## Engineering evidence

The [checkpoint 3 CI run](https://github.com/franciscovitar/ai-quality-lab/actions/runs/44565483960) on commit `9471979b7903dd5cf87c62b7a9343dbae64c58a5` passed **51 tests (43 unit, 8 PostgreSQL integration; 0 skipped)**, checked the unchanged lexical baseline, passed full Docker Compose/HTTP smoke and ran the pinned ONNX model against PostgreSQL to produce FTS/semantic/hybrid comparison artifacts. These are observed retrieval results, **not** evidence of generated-answer correctness or production reliability.

The [checkpoint 4 score-gate run](https://github.com/franciscovitar/ai-quality-lab/actions/runs/44566570399) on commit `85fb40bee432464a30d3192ad19ac39fa568ddb5` passed **54 tests (46 unit, 8 integration; 0 skipped)**, the same Compose smoke and the real-model retrieval/gate experiment. Only two dev-negative questions were used to determine the threshold, so perfect performance on this tiny synthetic sample is **not** a calibrated generalization or answer-faithfulness claim.

The [near-miss challenge run](https://github.com/franciscovitar/ai-quality-lab/actions/runs/44567666413) on commit `5eaea02184a75c9fb07d4893597dcf68f5c699df` passed **56 tests (48 unit, 8 integration; 0 skipped)** and all three jobs; its **negative finding** is that a score-only gate accepted all six newly authored unanswerable near misses. That failed *experimental policy metric* is reported as such rather than concealed by a green test badge.

The [evidence-span CI run](https://github.com/franciscovitar/ai-quality-lab/actions/runs/44574024515) on commit `adae4ecc513ed946efd3a9e122e2b4f09f6a55e3` passed **74 tests (66 unit, 8 integration; 0 skipped)** and all three jobs. Exact semantic retrieval covered all **8/8** project-labeled supporting spans in six answerable challenges, while the same method still retrieved candidates for **6/6 unanswerable near misses**. This is a span-coverage result, not LLM-answer correctness or generalization.

The [claim-pair benchmark run](https://github.com/franciscovitar/ai-quality-lab/actions/runs/44578890769) on commit `9f8a28e349291b04c0b5f2c70e15eafee838b407` passed **91 tests (83 unit, 8 integration; 0 skipped)** and all three jobs. Its archived **citation-presence negative control** falsely supported all twelve unsupported claims despite referring to authentic sources. No model was used to judge entailment, and no model answer accuracy is claimed.

The [pinned NLI run](https://github.com/franciscovitar/ai-quality-lab/actions/runs/44580522095) on commit `8dd0514ab0cee19430fb538540f8f944758c6a84` passed all **four jobs** and **104 tests (96 unit, 8 PostgreSQL integration, 0 skipped)**. Its independent, real CPU ONNX job classified **20/24 authored claim/quote pairs correctly**, with **1 false approval of 12 unsupported claims** and **3 missed supports**. The full per-case artifact documents `C005`, a false approval of a timeline contradiction. This result evaluates an off-the-shelf NLI classifier against public synthetic labels, not generated answers, real-world factuality or production-grade safety.

The [composed-answer CI run](https://github.com/franciscovitar/ai-quality-lab/actions/runs/44582652843) on commit `fdcd55fe076f249ad018d785be988bf07cb5043d` passed all **four jobs** and **124 tests (116 unit, 8 integration, 0 skipped)**. Against 21 manually segmented answer fixtures, real NLI claim verdicts produced **18/21 agreement** with authored reference decisions. One whole answer containing mutually incompatible crash timelines was **falsely accepted**. This is a composition *diagnostic fixture* result, not an evaluation of actual generated answers.

The [actual-generator CI run](https://github.com/franciscovitar/ai-quality-lab/actions/runs/44585749946) on commit `45adfdda6c64b2a33cf2d1466f0018c5f1b21462` passed **five jobs** and **144 tests (136 unit, 8 PostgreSQL integration, 0 skipped)**. The new CPU job downloaded the SHA-256-verified ONNX graph, generated all 12 source-bound answers and archived prompts, raw responses and protocol report. **0/6 unsupported questions received the required exact abstention; only 3/12 citation-syntax checks passed**. These are real observed model failures and explicitly **not** a generated-answer factuality score.

The [frozen-claim/NLI CI run](https://github.com/franciscovitar/ai-quality-lab/actions/runs/44592082406) on commit `c23c111e65e6792f8591da8d50c94cd0ae395350` passed **all five jobs**, **170 tests (162 unit, 8 PostgreSQL integration, 0 skipped)** and full HTTP Compose. The pinned ONNX model agreed with the project's **provisional** verdicts on **14/15 real generated claim occurrences**. The archived model report also confirms that a deliberately unsafe all-claims-entailed response rule falsely accepted unanswerable case `R108`, while `R105` was accepted despite fabricated citations. These are **diagnostic failures**, not a production-quality pass or independently established error rate.

## References

- [OpenAI evaluation best practices](https://developers.openai.com/api/docs/guides/evaluation-best-practices)
- [Anthropic eval methodology](https://www.anthropic.com/research/demystifying-evals-for-ai-agents)
- [pgvector upstream](https://github.com/pgvector/pgvector)
- [OWASP GenAI LLM Top 10 2026](https://genai.owasp.org/resource/owasp-genai-llm-top-10-2026/)

