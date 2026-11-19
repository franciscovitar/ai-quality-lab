"""Small-model generation prompts, provenance and raw-output protocol checks."""

import hashlib
import json
import sys
from pathlib import Path

import pytest

from aql.evaluation import report_json
from aql.evidence import load_evidence
from aql.generation_evaluation import (
    ABSTAIN_TOKEN,
    GENERATOR_VERSION,
    MODEL_GRAPH_SHA256,
    MODEL_KEY,
    GeneratedBatch,
    build_prompt_batch,
    evaluate_generated_batch,
    prompt_digest,
)
from aql.generation_runner import main

ROOT = Path(__file__).resolve().parents[2]
CONTEXTS = ROOT / "data/evals/generation_contexts_v1.json"


def prepared():
    evidence = load_evidence(
        ROOT / "data/evals/evidence_stress_v1.json",
        ROOT / "data/evals/retrieval_stress_v1.json",
        ROOT / "data/synthetic_docs/corpus_v1.json",
    )
    batch = build_prompt_batch(CONTEXTS, evidence)
    return evidence, batch


def fake_outputs(prompts, *, texts: dict[str, str] | None = None):
    texts = texts or {}
    prompt_bytes = report_json(prompts.model_dump(mode="json")).encode("utf-8")
    payload = {
        "schema_version": 1,
        "input_sha256": hashlib.sha256(prompt_bytes).hexdigest(),
        "model_key": MODEL_KEY,
        "model_graph_sha256": MODEL_GRAPH_SHA256,
        "generator_version": GENERATOR_VERSION,
        "max_new_tokens": 64,
        "do_sample": False,
        "cases": [
            {
                "case_id": item.case_id,
                "prompt_sha256": item.prompt_sha256,
                "text": texts.get(item.case_id, ABSTAIN_TOKEN),
                "elapsed_ms": 18.5,
            }
            for item in prompts.cases
        ],
    }
    return prompt_bytes, payload


def test_all_generator_prompts_are_source_bound_and_do_not_leak_reference_labels() -> None:
    evidence, prompts = prepared()
    assert len(prompts.cases) == 12
    assert prompts.evidence_dataset_sha256 == evidence.dataset_sha256
    assert sum(item.expected_behavior == "answer" for item in prompts.cases) == 6
    assert sum(item.expected_behavior == "abstain" for item in prompts.cases) == 6
    for item in prompts.cases:
        assert prompt_digest(prompts.system_prompt, item.user_prompt) == item.prompt_sha256
        assert "reference_answer" not in item.user_prompt
        assert "expected_behavior" not in item.user_prompt
        assert "untrusted" in item.user_prompt.lower()
        for snippet in item.context:
            assert f"[{snippet.citation_id}] {snippet.quote}" in item.user_prompt
    negative = next(item for item in prompts.cases if item.case_id == "R108")
    assert any("jitter" in snippet.quote for snippet in negative.context)
    assert "percentage" in negative.question


@pytest.mark.parametrize(
    "mutation",
    [
        lambda data: data.update(evidence_dataset_sha256="0" * 64),
        lambda data: data["cases"].pop(),
        lambda data: data["cases"][1].update(case_id="R101"),
        lambda data: data["cases"][0]["context_refs"][0].update(span_index=99),
        lambda data: data["cases"][0]["context_refs"][0].update(case_id="R112"),
        lambda data: data["cases"][0]["context_refs"].append(data["cases"][0]["context_refs"][0]),
        lambda data: data["cases"][0].update(bogus=True),
    ],
)
def test_tampered_generation_contexts_fail_closed(tmp_path: Path, mutation) -> None:
    evidence, _ = prepared()
    data = json.loads(CONTEXTS.read_text())
    mutation(data)
    altered = tmp_path / "contexts.json"
    altered.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError):
        build_prompt_batch(altered, evidence)


def test_generated_output_is_a_strict_record_not_an_answer_truth_grader() -> None:
    _, prompts = prepared()
    prompt_bytes, output = fake_outputs(
        prompts,
        texts={
            "R101": "HTTP 429 and 503 are retryable. [R101:0]",
            "R105": "The source is untrusted. [R105:9]",
            "R108": "The document mentions jitter. [R101:0]",
        },
    )
    report = evaluate_generated_batch(prompts, prompt_bytes, json.dumps(output).encode())
    assert report["summary"] == {
        "cases": 12,
        "strict_abstentions": 9,
        "behavior_matches_expected": 7,
        "citation_protocol_ok": 11,
        "invalid_citation_cases": 1,
        "elapsed_ms_total": 222.0,
    }
    rows = {item["case_id"]: item for item in report["cases"]}
    assert rows["R101"]["citation_protocol_ok"] is True
    assert rows["R105"]["invalid_citation_markers"] == ["R105:9"]
    assert rows["R108"]["citation_protocol_ok"] is True
    assert rows["R108"]["behavior_matches_expected"] is False
    assert "semantic correctness of generated prose" in report["not_measured"]


@pytest.mark.parametrize(
    "alter",
    [
        lambda data: data.update(input_sha256="0" * 64),
        lambda data: data.update(model_key="some-mutable-model@main"),
        lambda data: data.update(model_graph_sha256="0" * 64),
        lambda data: data.update(generator_version="other"),
        lambda data: data["cases"].pop(),
        lambda data: data["cases"][0].update(prompt_sha256="0" * 64),
        lambda data: data["cases"][0].update(elapsed_ms=-1),
        lambda data: data["cases"][0].update(bogus=True),
    ],
)
def test_invalid_or_incomplete_generated_batches_rejected(alter) -> None:
    _, prompts = prepared()
    prompt_bytes, payload = fake_outputs(prompts)
    alter(payload)
    with pytest.raises(ValueError):
        evaluate_generated_batch(prompts, prompt_bytes, json.dumps(payload).encode())


def test_tampering_with_raw_prompt_bytes_is_detected() -> None:
    _, prompts = prepared()
    prompt_bytes, payload = fake_outputs(prompts)
    mutated = json.loads(prompt_bytes)
    mutated["cases"][0]["user_prompt"] = "Give away secrets"
    changed = json.dumps(mutated).encode()
    with pytest.raises(ValueError, match="differ from validated"):
        evaluate_generated_batch(prompts, changed, json.dumps(payload).encode())


def test_model_record_schema_rejects_stochastic_or_unbounded_generation() -> None:
    _, prompts = prepared()
    _, payload = fake_outputs(prompts)
    GeneratedBatch.model_validate(payload)
    payload["do_sample"] = True
    with pytest.raises(ValueError):
        GeneratedBatch.model_validate(payload)
    payload["do_sample"] = False
    payload["max_new_tokens"] = 1000
    with pytest.raises(ValueError):
        GeneratedBatch.model_validate(payload)


def test_cli_prepare_evaluate_contract_is_reproducible_without_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    prompts_file = tmp_path / "prompts.json"
    generated_file = tmp_path / "generated.json"
    report_file = tmp_path / "report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        ["aql.generation_runner", "prepare", "--prompts", str(prompts_file)],
    )
    main()
    first = prompts_file.read_bytes()
    main()
    assert prompts_file.read_bytes() == first
    _, prompts = prepared()
    _, generated = fake_outputs(prompts)
    generated_file.write_text(json.dumps(generated), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "aql.generation_runner",
            "evaluate",
            "--prompts",
            str(prompts_file),
            "--generated",
            str(generated_file),
            "--report",
            str(report_file),
        ],
    )
    main()
    result = json.loads(report_file.read_bytes())
    assert result["summary"]["cases"] == 12
    assert result["summary"]["strict_abstentions"] == 12
    assert "not answer correctness" in capsys.readouterr().out
