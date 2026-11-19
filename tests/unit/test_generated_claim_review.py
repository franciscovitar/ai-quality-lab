"""Negative controls for provenance and full-text author-annotation coverage."""

import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from aql.claim_runner import DEFAULT_EVIDENCE, DEFAULT_RETRIEVAL
from aql.eval_runner import DEFAULT_CORPUS
from aql.evidence import load_evidence
from aql.generated_claim_review import review_frozen_generation
from aql.generated_claim_runner import main
from aql.generation_evaluation import build_prompt_batch
from aql.generation_runner import DEFAULT_CONTEXTS

PROMPTS = Path("data/evals/frozen_generation_v1/prompts.json")
GENERATED = Path("data/evals/frozen_generation_v1/generated-outputs.json")
REVIEW = Path("data/evals/generated_claim_review_v1.json")


@pytest.fixture(scope="module")
def source_bound_prompts():
    evidence = load_evidence(DEFAULT_EVIDENCE, DEFAULT_RETRIEVAL, DEFAULT_CORPUS)
    return build_prompt_batch(DEFAULT_CONTEXTS, evidence)


@pytest.fixture()
def frozen_bytes():
    return PROMPTS.read_bytes(), GENERATED.read_bytes(), REVIEW.read_bytes()


def validate(source_bound_prompts, frozen_bytes):
    return review_frozen_generation(*frozen_bytes, source_bound_prompts)


def changed_review(data, modify):
    modified = deepcopy(json.loads(data))
    modify(modified)
    return json.dumps(modified).encode("utf-8")


def test_frozen_real_outputs_have_exact_provisional_claim_inventory(
    source_bound_prompts, frozen_bytes
):
    report = validate(source_bound_prompts, frozen_bytes)
    assert report["review_status"] == "project_author_provisional"
    assert report["independent_reviewers"] == 0
    assert report["source_ci_run_id"] == 44588063650
    assert report["summary"] == {
        "generated_answers": 12,
        "annotated_claims": 15,
        "supported": 8,
        "contradicted": 1,
        "not_established": 6,
        "all_claims_author_labeled_supported_cases": 5,
        "fully_responsive_cases_author_labeled": 7,
        "strict_abstentions": 0,
        "citation_protocol_ok": 3,
    }
    by_id = {case["case_id"]: case for case in report["cases"]}
    assert by_id["R106"]["claims"][0]["verdict"] == "contradicted"
    assert by_id["R110"]["claims"][0]["verdict"] == "not_established"
    assert by_id["R108"]["all_claims_author_labeled_supported"]
    assert by_id["R108"]["responsiveness"] == "does_not_answer"
    assert not by_id["R105"]["citation_protocol_ok"]


@pytest.mark.parametrize("which", [0, 1])
def test_source_bytes_changed_even_to_parseable_json_are_rejected(
    source_bound_prompts, frozen_bytes, which
):
    changed = list(frozen_bytes)
    raw = json.loads(changed[which])
    raw["dataset_id" if which == 0 else "model_key"] = "tampered"
    changed[which] = json.dumps(raw).encode("utf-8")
    with pytest.raises(ValueError, match="Frozen CI"):
        validate(source_bound_prompts, tuple(changed))


def test_discarding_one_claim_substring_cannot_silently_shrink_review(
    source_bound_prompts, frozen_bytes
):
    edited = changed_review(
        frozen_bytes[2],
        lambda r: r["cases"][5]["segments"][0].update(
            text=(
                'The answer to the question "For how many days may synthetic '
                'request metadata be retained?"'
            )
        ),
    )
    with pytest.raises(ValueError, match="exact raw output"):
        validate(source_bound_prompts, (*frozen_bytes[:2], edited))


def test_adding_unreviewed_prose_cannot_be_passed_off_as_formatting(
    source_bound_prompts, frozen_bytes
):
    edited = changed_review(
        frozen_bytes[2],
        lambda r: r["cases"][0]["segments"][0].update(
            kind="non_claim", non_claim_reason="formatting", text="invented fact"
        ),
    )
    with pytest.raises(ValueError, match="Formatting segments"):
        validate(source_bound_prompts, (*frozen_bytes[:2], edited))


def test_cannot_omit_an_answer_entirely(source_bound_prompts, frozen_bytes):
    edited = changed_review(frozen_bytes[2], lambda r: r["cases"].pop())
    with pytest.raises(ValueError, match="Every generated answer"):
        validate(source_bound_prompts, (*frozen_bytes[:2], edited))


def test_cannot_duplicate_an_answer_or_reorder_it(source_bound_prompts, frozen_bytes):
    edited = changed_review(frozen_bytes[2], lambda r: r["cases"][2].update(case_id="R102"))
    with pytest.raises(ValueError, match="Every generated answer"):
        validate(source_bound_prompts, (*frozen_bytes[:2], edited))


def test_claim_sources_must_have_been_given_to_generator(source_bound_prompts, frozen_bytes):
    edited = changed_review(
        frozen_bytes[2],
        lambda r: r["cases"][6]["segments"][0].update(evidence_refs=["R106:0"]),
    )
    with pytest.raises(ValueError, match="not shown to the generator"):
        validate(source_bound_prompts, (*frozen_bytes[:2], edited))


def test_claim_requires_provisional_verdict_and_reason(source_bound_prompts, frozen_bytes):
    edited = changed_review(
        frozen_bytes[2], lambda r: r["cases"][10]["segments"][0].pop("rationale")
    )
    with pytest.raises(ValueError, match="Claims require"):
        validate(source_bound_prompts, (*frozen_bytes[:2], edited))


def test_fabricated_citation_marker_is_retained_but_not_trusted(source_bound_prompts, frozen_bytes):
    edited = changed_review(
        frozen_bytes[2],
        lambda r: r["cases"][4]["segments"][0].update(
            text="[R105:2] fake claim", non_claim_reason="citation_marker"
        ),
    )
    with pytest.raises(ValueError, match="Citation-marker segments"):
        validate(source_bound_prompts, (*frozen_bytes[:2], edited))


def test_author_cannot_claim_independent_validation_in_schema(source_bound_prompts, frozen_bytes):
    edited = changed_review(
        frozen_bytes[2], lambda r: r.update(review_status="independently_verified")
    )
    with pytest.raises(ValueError, match="review_status"):
        validate(source_bound_prompts, (*frozen_bytes[:2], edited))


def test_source_chain_mismatch_is_rejected(source_bound_prompts, frozen_bytes):
    corrupted_source = source_bound_prompts.model_copy(update={"system_prompt": "wrong prompt"})
    with pytest.raises(ValueError, match="validated source-bound prompts"):
        validate(corrupted_source, frozen_bytes)


def test_claim_review_command_preserves_provisional_status(tmp_path, monkeypatch):
    output = tmp_path / "review.json"
    monkeypatch.setattr(sys, "argv", ["review", "--output", str(output)])
    main()
    report = json.loads(output.read_text())
    assert report["summary"]["annotated_claims"] == 15
    assert report["review_status"] == "project_author_provisional"
    assert report["independent_reviewers"] == 0
