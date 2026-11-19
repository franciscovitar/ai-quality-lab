"""Real-output NLI comparison contracts without requiring a model download."""

import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from aql.claim_runner import DEFAULT_EVIDENCE, DEFAULT_RETRIEVAL
from aql.eval_runner import DEFAULT_CORPUS
from aql.generated_claim_review import load_and_review
from aql.generated_nli import evaluate_generated_nli, generated_nli_pairs
from aql.generation_evaluation import PromptBatch
from aql.generation_runner import DEFAULT_CONTEXTS
from aql.nli import MODEL_KEY, NliObservation

PROMPTS = Path("data/evals/frozen_generation_v1/prompts.json")
GENERATED = Path("data/evals/frozen_generation_v1/generated-outputs.json")
ANNOTATIONS = Path("data/evals/generated_claim_review_v1.json")


@pytest.fixture(scope="module")
def review_and_pairs():
    report = load_and_review(
        prompts_path=PROMPTS,
        generated_path=GENERATED,
        review_path=ANNOTATIONS,
        context_path=DEFAULT_CONTEXTS,
        evidence_path=DEFAULT_EVIDENCE,
        retrieval_path=DEFAULT_RETRIEVAL,
        corpus_path=DEFAULT_CORPUS,
    )
    batch = PromptBatch.model_validate_json(PROMPTS.read_bytes())
    return report, generated_nli_pairs(batch, report)


def mock_outputs(pairs, label="entailment"):
    logits_by_label = {
        "entailment": (0.0, 2.0, 0.0),
        "contradiction": (2.0, 0.0, 0.0),
        "neutral": (0.0, 0.0, 2.0),
    }
    return [NliObservation(pair.pair_id, label, 54, logits_by_label[label]) for pair in pairs]


def test_model_receives_raw_claims_and_exact_source_only(review_and_pairs):
    report, pairs = review_and_pairs
    assert len(pairs) == report["summary"]["annotated_claims"] == 15
    assert len({pair.pair_id for pair in pairs}) == 15
    for pair in pairs:
        model_input = pair.model_input()
        assert model_input == (pair.pair_id, pair.premise, pair.claim_text.strip())
        assert "project_author_provisional" not in str(model_input)
        assert pair.author_verdict not in model_input
    retention = next(pair for pair in pairs if pair.case_id == "R106")
    assert retention.author_verdict == "contradicted"
    assert "thirty days" in retention.premise
    assert "3 days" in retention.claim_text
    copied_claims = [pair for pair in pairs if pair.case_id == "R105"]
    assert copied_claims[0].claim_text == copied_claims[1].claim_text
    assert copied_claims[0].pair_id != copied_claims[1].pair_id
    assert copied_claims[0].premise == copied_claims[1].premise


def test_all_supported_baseline_exposes_false_supports(review_and_pairs):
    report, pairs = review_and_pairs
    result = evaluate_generated_nli(
        pairs=pairs, observations=mock_outputs(pairs), reviewed_report=report
    )
    assert result["summary"] == {
        "claim_occurrences": 15,
        "agreement_with_provisional_author": 8,
        "both_support": 8,
        "both_reject": 0,
        "author_support_model_reject": 0,
        "author_reject_model_support": 7,
        "nli_three_way": {"contradiction": 0, "entailment": 15, "neutral": 0},
        "naive_answers_accepted": 12,
        "naive_unanswerable_false_accepts": 6,
    }
    assert result["independent_reviewers"] == 0
    assert "no independent answer factuality" in result["reference"]


def test_all_neutral_baseline_is_not_a_factuality_success(review_and_pairs):
    report, pairs = review_and_pairs
    result = evaluate_generated_nli(
        pairs=pairs, observations=mock_outputs(pairs, "neutral"), reviewed_report=report
    )
    assert result["summary"]["agreement_with_provisional_author"] == 7
    assert result["summary"]["author_support_model_reject"] == 8
    assert result["summary"]["author_reject_model_support"] == 0
    assert result["summary"]["naive_answers_accepted"] == 0


@pytest.mark.parametrize("mutation", ["omit", "duplicate", "reorder", "unknown"])
def test_missing_or_mispaired_nli_observations_fail(review_and_pairs, mutation):
    report, pairs = review_and_pairs
    outputs = mock_outputs(pairs)
    if mutation == "omit":
        outputs.pop()
    elif mutation == "duplicate":
        outputs[1] = outputs[0]
    elif mutation == "reorder":
        outputs[0], outputs[1] = outputs[1], outputs[0]
    else:
        outputs[0] = NliObservation("R999/C01", "neutral", 54, (0.0, 0.0, 2.0))
    with pytest.raises(ValueError, match="exactly one result"):
        evaluate_generated_nli(pairs=pairs, observations=outputs, reviewed_report=report)


@pytest.mark.parametrize("mutation", ["mismatched_label", "overlong", "nan"])
def test_invalid_model_observation_fails(review_and_pairs, mutation):
    report, pairs = review_and_pairs
    outputs = mock_outputs(pairs)
    if mutation == "mismatched_label":
        outputs[0] = NliObservation(pairs[0].pair_id, "neutral", 54, (0.0, 2.0, 0.0))
    elif mutation == "overlong":
        outputs[0] = NliObservation(pairs[0].pair_id, "entailment", 257, (0.0, 2.0, 0.0))
    else:
        outputs[0] = NliObservation(pairs[0].pair_id, "entailment", 54, (0.0, float("nan"), 0.0))
    with pytest.raises(ValueError):
        evaluate_generated_nli(pairs=pairs, observations=outputs, reviewed_report=report)


def test_cannot_promote_a_provisional_review_to_independent(review_and_pairs):
    report, _ = review_and_pairs
    batch = PromptBatch.model_validate_json(PROMPTS.read_bytes())
    forged = deepcopy(report)
    forged["review_status"] = "independently_adjudicated"
    with pytest.raises(ValueError, match="provisional author"):
        generated_nli_pairs(batch, forged)


def test_cannot_refer_to_unseen_evidence_even_if_report_altered(review_and_pairs):
    report, _ = review_and_pairs
    batch = PromptBatch.model_validate_json(PROMPTS.read_bytes())
    forged = deepcopy(report)
    forged["cases"][0]["claims"][0]["evidence_refs"] = ["R112:0"]
    with pytest.raises(ValueError, match="invalid claim-to-source"):
        generated_nli_pairs(batch, forged)


def test_runner_reuses_model_for_original_claims_and_frozen_generated_claims(
    monkeypatch, tmp_path, capsys
):
    from aql import nli_runner

    seen = []

    class FakeModel:
        model_key = MODEL_KEY

        def __init__(self, *, allow_download):
            seen.append(("init", allow_download))

        def infer(self, locked_claims):
            seen.append(("authored_claims", len(locked_claims.data.cases)))
            return [
                NliObservation(case.case_id, "neutral", 40, (0.0, 0.0, 2.0))
                for case in locked_claims.data.cases
            ]

        def infer_pairs(self, triples):
            seen.append(("generated_claims", len(triples)))
            assert len(triples) == 15
            assert all(len(item) == 3 for item in triples)
            assert all("project_author_provisional" not in str(item) for item in triples)
            return [NliObservation(item[0], "neutral", 40, (0.0, 0.0, 2.0)) for item in triples]

    monkeypatch.setattr(nli_runner, "PinnedNliCrossEncoder", FakeModel)
    output = tmp_path / "original.json"
    actual = tmp_path / "generated.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "nli_runner",
            "--output",
            str(output),
            "--generated-claims",
            "--generated-claims-output",
            str(actual),
        ],
    )
    nli_runner.main()
    assert seen == [("init", False), ("authored_claims", 24), ("generated_claims", 15)]
    scored = json.loads(actual.read_bytes())
    assert scored["summary"]["claim_occurrences"] == 15
    assert scored["summary"]["author_support_model_reject"] == 8
    assert scored["summary"]["naive_answers_accepted"] == 0
    assert scored["review_status"] == "project_author_provisional"
    assert "NOT independent factuality" in capsys.readouterr().out
