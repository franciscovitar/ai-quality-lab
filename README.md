# AI Quality Lab

**Engineering a reliable retrieval system is more than getting an LLM to answer a prompt.** AI Quality Lab is a portfolio project for measurable, testable RAG systems: a versioned synthetic corpus, reproducible retrieval experiments, evidence-backed answers, and automated quality gates.

> **Current state — checkpoint 2 in development.** The PostgreSQL/FTS foundation has real CI and Compose/HTTP evidence. A version-locked golden dataset and deterministic retrieval evaluator are now implemented, with real PostgreSQL verification pending on this change. **There are no LLM-generated answers or semantic embeddings yet.** See [roadmap](#roadmap).

## Why this exists

Backend correctness and AI answer correctness are different problems. An integration test can prove that SQL and API contracts work, but cannot prove that a generated answer is supported by the right sources. The full lab will separate retrieval evaluation, answer grounding, abstention, operational cost, and reproducibility.

The sample corpus is wholly synthetic. It describes fictional webhook operations to provide meaningful cases involving stale policies, duplicate deliveries, on-call runbooks, and malicious instructions embedded in documents. No customer, university, or private PAS data is used.

## Architecture in one minute

- **Python/FastAPI:** explicit API contracts; retrieval and evaluation will remain inspectable modules.
- **PostgreSQL 17 + pgvector:** relational versions, full-text baseline and a reserved vector column for later hybrid retrieval.
- **Synthetic corpus:** source versions, hashes, paragraph offsets and stable IDs; never update a source silently.
- **Golden retrieval set:** 16 synthetic cases with current-version labels, positive/negative outcomes, split/tag slices and hash-locked corpus provenance.
- **Testing:** pure unit tests plus PostgreSQL integration tests; CI fails on missing/skipped DB coverage.
- **Future:** embeddings, RAG answers with citations/abstention, eval harness, comparison dashboard in Next.js.

See [Architecture](docs/ARCHITECTURE.md), [Dataset card](docs/DATASET_CARD.md) and [Decisions](docs/DECISIONS.md) for details and limits.

## Quick start — full local demo stack

Prerequisites: Docker Desktop/Engine with Compose and capacity to start PostgreSQL 17.

```bash
docker compose up --build --wait
curl http://localhost:8001/health
curl http://localhost:8001/ready
curl http://localhost:8001/v1/corpus/versions
curl -X POST http://localhost:8001/v1/retrieval/search \
  -H 'Content-Type: application/json' \
  -d '{"question":"What is the retry budget for webhooks?","top_k":5}'
```

`compose.yaml` starts PostgreSQL, applies `0001_corpus`, verifies the synthetic seed, and then starts the API. PostgreSQL is **not** exposed to the host; the API binds only to `127.0.0.1:8001`. All credentials are obvious, disposable placeholders and must never be reused outside an isolated machine.

The CI workflow also runs `scripts/compose_smoke.py` against that same full stack, checking migrated readiness, all six current documents and a current-policy search over real HTTP. Inspect its run before considering Compose verified.

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
uv run --locked python scripts/check_test_report.py test-report.xml --min-integration 6
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
  --compare path/to/prior-report.json
```

Reports contain document-level Recall@k, MRR@k and HitRate@k for answerable questions; a separate `unanswerable_no_hit_rate` measures whether *retrieval* returns nothing on unanswerable questions. **It does not measure LLM abstention, correctness, grounding or cost.** Rankings are bounded by `top_k` **chunks** first, then deduplicated by document version for scoring. Each failure remains visible by case and tag; optional baseline comparison fails if *any* previously correct case regresses, even when averages improve. CI stores the JSON as a short-lived artifact.

The declared `holdout` split is public and project-authored, not a blind external test set. The [dataset card](docs/DATASET_CARD.md) documents that and other limitations, including document- rather than evidence-span-level labels.

## Current endpoints

| Method | Path | Behavior |
|---|---|---|
| `GET` | `/health` | Process liveness, no DB dependency |
| `GET` | `/ready` | 200 only with reachable, migrated PostgreSQL + pgvector |
| `GET` | `/v1/corpus/versions` | Current synthetic documents |
| `POST` | `/v1/retrieval/search` | Parameterized FTS search, top_k 1–10, current versions only |

This is an intentionally **non-agentic** first checkpoint. A search result is not an AI-generated answer. `score` is PostgreSQL FTS rank, **not** model confidence or grounding quality.

## Roadmap

1. **Foundation:** schema, seed, lexical baseline, reproducible PostgreSQL tests and CI.
2. **Dataset:** version-locked goldens, split/coverage audit, deterministic retrieval report. **Implemented, real PostgreSQL CI result pending.**
3. **Semantic and hybrid retrieval:** fixed embedding model metadata and fair FTS-vs-hybrid comparison.
4. **Generation:** bounded LLM provider adapter, cited answers and explicit abstention.
5. **Answer evals:** deterministic grounding/citation graders, calibrated semantic review, immutable LLM runs and regression reporting.
6. **Dashboard and case study:** compare failures, latency/tokens/cost, dockerized end-to-end demo.

No production deployment, real user credentials, private data or paid LLM usage is implied by this roadmap. A public deployment would require separate authentication, rate limiting, privacy and security review.

## Engineering evidence

The [checkpoint 1 CI run](https://github.com/franciscovitar/ai-quality-lab/actions/runs/44545714409) verified 27 tests (22 unit, 5 PostgreSQL integration; 0 skipped) and Docker Compose with real HTTP: `/health`, `/ready`, six current documents and a current-policy search. That evidence applies to commit `0c2d499b4fa31d52ce34adb7c8edc5571105514f`, **not automatically to later evaluation changes**. The new golden harness must be tested separately in CI before scores are claimed. Fake providers, once added, will test orchestration and contracts, **not real model quality**.

## References

- [OpenAI evaluation best practices](https://developers.openai.com/api/docs/guides/evaluation-best-practices)
- [Anthropic eval methodology](https://www.anthropic.com/research/demystifying-evals-for-ai-agents)
- [pgvector upstream](https://github.com/pgvector/pgvector)
- [OWASP GenAI LLM Top 10 2026](https://genai.owasp.org/resource/owasp-genai-llm-top-10-2026/)

