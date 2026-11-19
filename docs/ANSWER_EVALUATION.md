# Composed-answer evaluation — pre-annotated synthetic fixtures

This checkpoint moves from **single claim/quote pairs** to **entire response decisions**: did the answer abstain, were its citations the correct ones, did it include the facts needed for the question, and were *all* asserted claims supported? It is an evaluation **scaffold**, not a claim that a production generator is already safe.

## Experimental boundary

[`data/evals/answer_cases_v1.json`](../data/evals/answer_cases_v1.json) contains **21 manually composed answer fixtures** for all 12 stress questions (six answerable and six unanswerable). Each answered fixture is a list of fully annotated atomic claim IDs and exact source-span citation IDs. `render_answer` creates readable prose from the corresponding authored claim text, making it impossible for these particular fixtures to hide an **unannotated** extra sentence. **It does not segment arbitrary model text**, and it has no independent human review.

The answer file binds to the raw SHA-256 of the 24-case [claim dataset](CLAIM_EVALUATION.md), which in turn binds to exact evidence quotes and source versions. Loading fails on missing/extra/duplicated answers, wrong fingerprints, unknown claims or citations, missing answerable-question reference requirements and inconsistent answer/abstention declarations. Wrong-but-authentic citations are deliberately **allowed as evaluation cases**, then flagged as wrong by the scorer.

The six answerable questions have manually defined *minimal claim-ID coverage requirements*. This is deliberately **strict exact-ID coverage**, not semantic recall of paraphrases; a different valid phrasing may be rejected by this experimental harness. The requirements, scenario set and labels were authored together, so they are neither sealed nor externally validated.

## Decision rule and metrics

For an answerable question, a response is accepted only when:

1. It answers rather than abstains.
2. It contains all manually required claim IDs for that question.
3. Every sentence cites exactly its annotated source-span IDs (not just any real document).
4. Every asserted atomic claim is labeled `entailed` by the **selected** verdict source.

An unanswerable question accepts only an abstention. The report preserves the rendered full answer, all asserted/required claim IDs, reference-supported versus prediction-supported flags, per-answer decisions and false approvals/rejections relative to manually authored claim labels.

This scorer can use three **distinct, explicitly named** sources of claim judgments:

- `authored-claim-label-reference-oracle`: the dataset's own labels, useful for testing the report machinery, **not a real model**.
- `unsafe-citation-presence-baseline`: predicts all 24 claims supported, a deliberately unsafe negative control.
- A complete externally supplied file or the **actual pinned ONNX NLI model** run over the same claim pairs. This model sees only each claim and its original exact source quote; it does **not** consider the full answer's discourse structure, source-retrieval trace or gold label.

For the 21 project-authored fixtures, the **reference-label analysis** accepts **12** and rejects **9**. Six cases are correct abstentions; two deliberately answer an unanswerable question; one unnecessarily abstains despite source support. The naive citation-only rule gets **19/21 reference decisions right** and falsely approves **2/21**: `A002` (invented jitter percentage attached to a real citation) and `A021` (internally inconsistent timeline, both statements cite the same real incident). These are **deterministic comparisons against known fixture labels**, not independent factuality results.

### Real NLI verdicts reused over these answers

**Measured in [GitHub Actions #44582652843](https://github.com/franciscovitar/ai-quality-lab/actions/runs/44582652843), code commit `fdcd55fe076f249ad018d785be988bf07cb5043d`:** all four jobs passed, including **124 tests (116 unit, 8 PostgreSQL integration, 0 skipped)**, full Compose HTTP and pinned CPU ONNX retrieval/NLI. The uploaded `claim-nli-experiment` artifact contains both 24 standalone claim predictions and 21 composed-answer decisions. The NLI model still scored 20/24 standalone claims correctly; this checkpoint **reuses those exact verdicts** rather than performing whole-answer inference.

| Response-level benchmark, 21 authored fixtures | Reference-label oracle | Citation-only baseline | Real pinned NLI verdicts |
|---|---:|---:|---:|
| Decisions agreeing with authored reference | 21/21 (by definition) | 19/21 | **18/21** |
| Wrongly approved responses | 0 | 2 | **1** |
| Wrongly rejected responses | 0 | 0 | **2** |

The model wrongly rejects valid `A001` (retry policy) and `A007` (replay-audit requirements). More seriously, it **accepts `A021`**, a complete answer simultaneously saying the worker crashed *before* and *after* receipt storage. The citation alignment and manually required fact coverage both pass; the **false entailment verdict on atomic claim `C005`** causes the response-level false approval. This is a concrete failure of sentence-local entailment as a whole-answer consistency test — **do not deploy it as a correctness gate**. The 21 responses reuse the same publicly authored claim dataset, so this agreement rate is not a generalization estimate.

Other diagnostic cases include:

- `A005`: statements are true and cited but omit required queue/dead-letter checks.
- `A010`: a true security claim links to an **authentic but unrelated** crash citation.
- `A008`: unnecessary abstention on a supported replay-audit question.
- `A015` and `A020`: answering when the correct behavior is abstention.
- `A021`: a complete response contains *both* “crashed after storing” and “crashed before storing.” It reveals how an isolated claim judge can miss a contradiction even when every citation looks authentic.

## Reproduce without PostgreSQL or a model

```bash
uv run --locked --extra dev pytest -q tests/unit/test_answer_evaluation.py
uv run --locked --extra dev python -m aql.answer_runner \
  --output test-results/answer-composition-report.json
```

The default report contains the reference-label oracle **and** the citation-only negative control in separate fields. You may also pass `--predictions path/to/predictions.json`, using the exact complete 24-item, SHA-bound claim prediction contract documented in [Claim evaluation](CLAIM_EVALUATION.md).

To run an actual opt-in **local** ONNX judge on the claim pairs and reuse its 24 verdicts over these composed answers:

```bash
uv sync --locked --extra semantic
uv run --locked --extra semantic python -m aql.nli_runner \
  --download-model --answer-dataset data/evals/answer_cases_v1.json
```

This writes both `test-results/nli/nli-report.json` and `test-results/nli/answer-composition-model-report.json`. Omit `--download-model` on later runs to require the same pinned cached snapshot. No hosted inference or paid API is called.

## What must come next

- **Real answer generation and ingestion:** record a fixed, versioned provider/model and actual outputs with reproducible prompts, provenance, budgets and immutable IDs. The current API is retrieval-only.
- **Independent claim extraction/review:** have a separate reviewer enumerate **every factual assertion** (including implications) in actual answers, and measure extraction omissions. Otherwise false claims can escape all grading.
- **Independent coverage and contradiction review:** a list of exact required claim IDs and a single binary NLI classifier cannot prove semantic completeness or global consistency; a response may be internally contradictory even if sentence-level judgments all say `entailed`.
- **Sealed test set, calibration and risk:** the current fixture responses and label rationales are public, authored together and adapted from a single synthetic domain. None of the percentages estimates real-world deployment performance. A green CI report means the experiment ran, not that answers are safe to ship.

Methodological inspirations (not complete reproductions): [FActScore (EMNLP 2023)](https://aclanthology.org/2023.emnlp-main.741/) evaluates atomic factual support; [RAGAS (EACL 2024)](https://aclanthology.org/2024.eacl-demo.16/) separates faithfulness to retrieved context; [AttributionBench (Findings ACL 2024)](https://aclanthology.org/2024.findings-acl.886/) illustrates the limits of automated attribution judgments.
