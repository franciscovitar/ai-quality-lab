/** CPU-only experimental text generation; pinned to one public ONNX graph. */
import { createHash } from 'node:crypto';
import { performance } from 'node:perf_hooks';
import { readFile, mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';

import { env, pipeline } from '@huggingface/transformers';

const MODEL_REPO = 'onnx-community/SmolLM2-135M-Instruct-ONNX';
const MODEL_REVISION = 'b8a5c0f183b78c55955a5364f610c36668b5e681';
const MODEL_GRAPH = 'onnx/model_quantized.onnx';
const MODEL_GRAPH_SHA256 = '0fab87142e3eb1fcacb881f8282e7473e62ad66920c347b81f088da6fda2da37';
const MODEL_KEY = `${MODEL_REPO}@${MODEL_REVISION}/${MODEL_GRAPH}`;
const GENERATOR_VERSION = '@huggingface/transformers@3.8.1';
const MAX_NEW_TOKENS = 64;

const digest = (buffer) => createHash('sha256').update(buffer).digest('hex');

function getArguments(argv) {
  const values = {};
  for (let i = 0; i < argv.length; i += 1) {
    const flag = argv[i];
    if (flag === '--download-model') {
      values.download = true;
    } else if (['--input', '--output', '--cache-dir'].includes(flag)) {
      if (!argv[i + 1] || argv[i + 1].startsWith('--')) {
        throw new Error(`${flag} needs a value`);
      }
      values[flag.slice(2)] = argv[++i];
    } else {
      throw new Error(`Unknown argument: ${flag}`);
    }
  }
  if (!values.input || !values.output) {
    throw new Error('Expected --input prompts.json and --output generated-outputs.json');
  }
  return values;
}

function checkPromptBatch(batch) {
  if (
    batch.schema_version !== 1 ||
    batch.prompt_version !== 'source-is-data-cite-or-abstain-v1' ||
    typeof batch.system_prompt !== 'string' ||
    !Array.isArray(batch.cases) ||
    batch.cases.length !== 12
  ) {
    throw new Error('Only the version-locked 12-case prompt format is supported');
  }
  let previous = '';
  for (const item of batch.cases) {
    if (
      !/^R\d{3}$/.test(item.case_id) ||
      item.case_id <= previous ||
      typeof item.user_prompt !== 'string' ||
      item.user_prompt.length > 4000 ||
      digest(`${batch.system_prompt}\n${item.user_prompt}`) !== item.prompt_sha256
    ) {
      throw new Error(`Invalid or tampered generation prompt ${item.case_id}`);
    }
    previous = item.case_id;
  }
}

function getGeneratedText(result) {
  if (!Array.isArray(result) || result.length !== 1) {
    throw new Error('Text-generation pipeline did not return one result');
  }
  let content = result[0].generated_text;
  if (Array.isArray(content)) {
    const last = content.at(-1);
    if (last?.role !== 'assistant') {
      throw new Error('Chat pipeline failed to return an assistant message');
    }
    content = last.content;
  }
  if (typeof content !== 'string' || !content.trim() || content.length > 6000) {
    throw new Error('Empty or oversized model output');
  }
  return content;
}

async function main() {
  const args = getArguments(process.argv.slice(2));
  const input = await readFile(args.input);
  const prompts = JSON.parse(input.toString('utf8'));
  checkPromptBatch(prompts);

  const cacheDir = path.resolve(args['cache-dir'] ?? 'test-results/generation/model-cache');
  await mkdir(cacheDir, { recursive: true });
  env.cacheDir = cacheDir;
  env.allowRemoteModels = args.download === true;

  const generator = await pipeline('text-generation', MODEL_REPO, {
    revision: MODEL_REVISION,
    dtype: 'q8',
    device: 'cpu',
  });
  const graphPath = path.join(cacheDir, MODEL_REPO, MODEL_REVISION, MODEL_GRAPH);
  const actualHash = digest(await readFile(graphPath));
  if (actualHash !== MODEL_GRAPH_SHA256) {
    throw new Error(`Pinned ONNX graph SHA-256 mismatch: ${actualHash}`);
  }

  const cases = [];
  for (const item of prompts.cases) {
    // The model receives ONLY instructions, question and exact source quotes.
    // Its authored expected behavior and any reference labels are not in this input.
    const messages = [
      { role: 'system', content: prompts.system_prompt },
      { role: 'user', content: item.user_prompt },
    ];
    const start = performance.now();
    const result = await generator(messages, {
      do_sample: false,
      max_new_tokens: MAX_NEW_TOKENS,
      return_full_text: false,
    });
    cases.push({
      case_id: item.case_id,
      prompt_sha256: item.prompt_sha256,
      text: getGeneratedText(result),
      elapsed_ms: Math.round((performance.now() - start) * 1000) / 1000,
    });
  }

  const output = {
    schema_version: 1,
    input_sha256: digest(input),
    model_key: MODEL_KEY,
    model_graph_sha256: actualHash,
    generator_version: GENERATOR_VERSION,
    max_new_tokens: MAX_NEW_TOKENS,
    do_sample: false,
    cases,
  };
  await mkdir(path.dirname(path.resolve(args.output)), { recursive: true });
  await writeFile(args.output, JSON.stringify(output, null, 2) + '\n', 'utf8');
  console.log(`Generated ${cases.length} recorded responses with pinned CPU ONNX ${MODEL_KEY}`);
}

main().catch((error) => {
  console.error(`Generator experiment failed: ${error.message}`);
  process.exitCode = 1;
});
