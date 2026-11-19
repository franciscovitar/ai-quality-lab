"""Challenge-set provenance and separation from the frozen threshold calibration."""

from pathlib import Path

from aql.evaluation import load_dataset

CORPUS = Path("data/synthetic_docs/corpus_v1.json")
ORIGINAL = Path("data/evals/retrieval_v1.json")
STRESS = Path("data/evals/retrieval_stress_v1.json")


def test_challenge_set_is_complete_and_locked_to_the_original_corpus() -> None:
    original = load_dataset(ORIGINAL, CORPUS)
    stress = load_dataset(STRESS, CORPUS)

    assert len(stress.data.cases) == 12
    assert {case.case_id for case in stress.data.cases} == {
        f"R{number}" for number in range(101, 113)
    }
    assert sum(case.expected_behavior == "answer" for case in stress.data.cases) == 6
    assert sum(case.expected_behavior == "abstain" for case in stress.data.cases) == 6
    assert stress.data.corpus_sha256 == original.data.corpus_sha256
    assert stress.data.corpus_id == original.data.corpus_id
    assert stress.dataset_sha256 != original.dataset_sha256
    assert "not blinded" in stress.data.label_provenance


def test_challenge_questions_are_new_and_cover_near_miss_negatives() -> None:
    original = load_dataset(ORIGINAL, CORPUS)
    stress = load_dataset(STRESS, CORPUS)

    assert {case.question for case in original.data.cases}.isdisjoint(
        case.question for case in stress.data.cases
    )
    for split in ("dev", "holdout"):
        cases = [case for case in stress.data.cases if case.split == split]
        assert sum(case.expected_behavior == "answer" for case in cases) == 3
        negatives = [case for case in cases if case.expected_behavior == "abstain"]
        assert len(negatives) == 3
        assert all("near_miss" in case.tags for case in negatives)
