# Active project direction — demo first (19 November 2026)

**Decision by Enzo:** make AI Quality Lab easy to understand, show and explain to a recruiter before expanding its research infrastructure. This decision is durable in [GitHub issue #2](https://github.com/franciscovitar/ai-quality-lab/issues/2). It supersedes any older roadmap emphasis on more benchmarks, judges, annotation layers or model comparisons.

## What to build next

1. **A two-minute visual demo.** Show a real frozen synthetic question, the small model's exact answer, the exact source passages it saw and the observed problem. Include a supported answer (`R101`), a directly contradictory fact (`R106`, **3 vs 30 days**), a source-supported non-answer (`R108`), fabricated citation markers (`R105`) and an ungrounded SQL suggestion (`R110`). Make automatic checks clearly distinguishable from project-author provisional judgments.
2. **One-command local reproduction.** Prefer the existing Docker Compose API and frozen JSON; no hosted inference, secret credentials, paid APIs or new machine-learning framework. Expose a simple local demo page, test that displayed outputs match frozen source bytes and preserve current CI.
3. **An interviewer-friendly case study.** Explain the problem, a small architecture diagram, how to run the demo, one meaningful failure, the engineering trade-offs and the real limits. A reader should understand the value in about two minutes.

## Explicit pause

Do **not** prioritize more evaluation layers, new scoreboards, new model candidates, human reviewer recruitment or production deployment until this portfolio presentation is complete or Enzo changes direction. Existing experiments stay intact; do not rewrite their historical measurements or claim model factuality is certified.

## Scope and permission

- Work on [`feat/ai-quality-foundation` / PR #1](https://github.com/franciscovitar/ai-quality-lab/pull/1), keeping the pull request in **draft**. Do not merge or change `main` without a separate decision.
- All demo data remains synthetic. The HTTP API is still local-only and no live model is called when a recruiter views a frozen example.
- The original 12-case model generation CI run, author annotations and experimental NLI report are **different evidence levels**. A green CI proves reproducibility and software checks, not correctness of every generated sentence.

## Resume in a new chat

Say **“Continuemos AI Quality Lab desde GitHub issue #2 y `docs/PORTFOLIO_FOCUS.md`”**. Read the **current** issue, PR and branch before making changes. Continue demo/case study work. Do not resume the old research roadmap by default.

**Done when:** a developer can launch the local page with the existing Compose stack, select and understand five representative cases, and read a concise case study with real links to source/CI. No human adjudication or public deployment is needed for this checkpoint.
