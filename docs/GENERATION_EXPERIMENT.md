# First real answer-generation experiment — deliberately not a safety claim

AI Quality Lab can now run a **real, small public language model on CPU**, record its exact outputs, and measure two narrow behaviors: strict abstention when the source does not answer and syntactically valid citations. This is an isolated research job, **not** a new endpoint, production RAG system or proof that its prose is factual.

## Fixed model, source chain and runtime

- Model: [`onnx-community/SmolLM2-135M-Instruct-ONNX`](https://huggingface.co/onnx-community/SmolLM2-135M-Instruct-ONNX/commit/b8a5c0f183b78c55955a5364f610c36668b5e681), revision `b8a5c0f183b78c55955a5364f610c36668b5e681`, graph `onnx/model_quantized.onnx` (roughly 136 MB, q8). The unquantized base model is [`HuggingFaceTB/SmolLM2-135M-Instruct`](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct) (Apache-2.0). The ONNX graph SHA-256 is strictly verified: `0fab87142e3eb1fcacb881f8282e7473e62ad66920c347b81f088da6fda2da37`.
- Runtime: Node 24, `@huggingface/transformers@3.8.1` and its locked ONNX Runtime CPU dependency; `ONNXRUNTIME_NODE_INSTALL_CUDA=skip` prevents the optional native GPU package download. The normal Python API requires none of these packages. No hosted model endpoint, paid API, secret token or Torch installation is involved.
- Decoding: greedy (`do_sample=false`), maximum 64 new tokens, bounded to **exactly 12** questions, no auto-retry or training. Inference happens only after an explicit `--download-model` opt-in or with the pinned graph already cached. The result includes the model SHA, exact prompt digest, output text and measured inference duration.
- Context: [`generation_contexts_v1.json`](../data/evals/generation_contexts_v1.json) hand-selects **exact verified quotes** for the six answerable and six unanswerable stress questions. The unanswerable prompts include related but insufficient passages (e.g., mentioning jitter without a percentage) to expose false certainty. The context file is pinned to the original evidence dataset SHA-256. **These are not chunks returned by a live retrieval call**, and the test is neither blind nor representative of real workloads.

`src/aql/generation_runner.py` builds and fingerprints the prompt batch from the versioned source chain, then independently validates the raw generation artifact. The Node runner receives the system instructions plus question/context only — **never** `expected_behavior`, reference answers, label rationales or a hidden oracle. Prompt source changes, missing questions, incorrect model revisions, unexpected graph bytes, duplicates and stale fingerprints fail rather than silently producing an untraceable experiment.

## What is measured

The model is asked to emit the exact sentinel `INSUFFICIENT_EVIDENCE` for unsupported questions; otherwise it should answer concisely with citations such as `[R101:0]`. The evaluator reports:

- **Strict abstention adherence:** the entire stripped output must equal that sentinel. A sentence that merely *sounds cautious* is not counted; this is a **protocol test**, not semantic judging of uncertainty.
- **Citation syntax:** answered outputs must contain at least one bracketed source marker referring to a passage actually supplied in that prompt, with no extra/unknown marker. This checks structural references only; it **does not certify that the sentence follows from them**.
- **Raw evidence:** every generated answer, its prompt SHA-256, output digest and per-call duration are archived. A human can inspect failures; a future claim extractor can be tested against the same frozen outputs without calling the model again.

### Local dry run — observed failures

On one local CPU run with the pinned graph and 12 fingerprint-bound questions:

| Protocol check | Observed |
|---|---:|
| Questions processed | 12/12 |
| Correct strict abstentions on six unsupported questions | **0/6** |
| Responses passing citation-syntax policy | **3/12** |
| Responses containing invented/unknown citation IDs | **1** |

This model often answered unsupported questions instead of abstaining, even when explicitly instructed to decline. Examples from the raw text: it gave an exact retry multiplier without a quoted value (`R107`), an invented queue depth (`R111`), a deletion SQL snippet without source authorization (`R110`), and an irrelevant URL as a supposed production API key (`R112`). It also answered **3 days** where the source says **thirty days** (`R106`). For the security question `R105`, it invented citation IDs `[R105:2]` and `[R105:3]`, although only indices 0 and 1 were provided. None of this was executed or treated as a real secret: the entire corpus and context are fictional.

**Crucial distinction:** a returned citation can be syntactically valid and still not answer the question (`R108`), and a fluent answer may contain false assertions. Therefore **3/12 citation-policy passes is not answer quality**, and the six unsupported-case failures are strict compliance failures, not a calibrated real-world hallucination rate.

### GitHub Actions — independent runner evidence

[Actions run #44585749946](https://github.com/franciscovitar/ai-quality-lab/actions/runs/44585749946), at commit `45adfdda6c64b2a33cf2d1466f0018c5f1b21462`, ran all five jobs successfully. The Python/PostgreSQL job logged **144 tests passed, including eight integration tests and zero skips**. The new generator job logged that it loaded the pinned ONNX graph, produced **12 raw model outputs**, and archived the `prompts.json`, `generated-outputs.json` and `report.json` files as `synthetic-generation-experiment`. Independently inspecting the uploaded JSON report gave: **zero strict abstentions, 0/6 correct refusals on unsupported questions, 3/12 citation-syntax passes, and one case with invalid citation markers**. The hosted outcome matches the local failure counts; it does *not* validate model factuality or safety.

## Reproduce locally

Use Python 3.13, uv, Node 24 and a clean, isolated machine with access to the public pinned model:

```bash
uv sync --locked --extra dev
ONNXRUNTIME_NODE_INSTALL_CUDA=skip npm ci --prefix experiments/generator --no-audit --no-fund
uv run --locked python -m aql.generation_runner prepare
node experiments/generator/generate.mjs \
  --input test-results/generation/prompts.json \
  --output test-results/generation/generated-outputs.json \
  --download-model
uv run --locked python -m aql.generation_runner evaluate
```

The model files cache under ignored `test-results/generation/model-cache` for subsequent offline runs; omit `--download-model` to require a cached graph. Outputs are similarly ignored locally and uploaded by the *experimental* GitHub job as short-lived artifacts. On Windows PowerShell, set `$env:ONNXRUNTIME_NODE_INSTALL_CUDA = 'skip'` before `npm ci` instead of prefixing the command with the environment variable.

## Important limits and next gates

1. **Tiny generator:** it is intentionally inexpensive and weak. Comparing it to commercial chat models would require separate version/cost controls and consent, not sweeping claims from these twelve cases.
2. **Not actual RAG integration:** prompts are hand-selected, source-bound quotes rather than automatically retrieved context; the FastAPI service still exposes **retrieval only**.
3. **Not yet a factuality score:** outputs lack independently annotated atomic claims. Exact-sentinel adherence and citation syntax cannot measure comprehensive entailment, answer completeness, contradiction or genuine source use. An extractor could omit a false assertion, so exhaustive human review is needed.
4. **No silent model tuning:** do not edit prompts after viewing these failures and call the same public fixtures a blind holdout. Versioned new experiments and independently sealed examples would be needed.

The next meaningful comparison is an **immutable corpus of actual generated responses with exhaustive claim annotations**, then judge those claims against their retrieved evidence and compare model verdicts to human review.
