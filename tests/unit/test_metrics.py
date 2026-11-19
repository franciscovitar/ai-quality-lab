import pytest

from aql.metrics import recall_at_k, reciprocal_rank_at_k


@pytest.mark.parametrize(
    ("ranking", "expected_recall", "expected_mrr"),
    [
        (["a", "b", "c"], 1.0, 0.5),
        (["a", "x", "b"], 0.5, 1.0 / 3),
        (["x", "y", "z"], 0.0, 0.0),
        (["b", "b", "a"], 0.5, 1.0),
    ],
)
def test_retrieval_metrics(ranking: list[str], expected_recall: float, expected_mrr: float) -> None:
    relevant = {"b", "c"}
    assert recall_at_k(relevant, ranking, 3) == expected_recall
    assert reciprocal_rank_at_k(relevant, ranking, 3) == expected_mrr


def test_metrics_reject_invalid_ground_truth_or_k() -> None:
    with pytest.raises(ValueError):
        recall_at_k(set(), ["x"], 5)
    with pytest.raises(ValueError):
        reciprocal_rank_at_k({"x"}, ["x"], 0)
