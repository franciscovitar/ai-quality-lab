import pytest

from aql.retrieval import SearchHit, fuse_rrf


def hit(chunk_id: str, *, slug: str = "policy", content: str = "same") -> SearchHit:
    return SearchHit(chunk_id, slug, 1, "title", content, 0.9)


def test_rrf_prioritizes_results_confirmed_by_both_methods() -> None:
    hits = fuse_rrf([hit("a"), hit("b")], [hit("b"), hit("c")], 3)
    assert [entry.chunk_id for entry in hits] == ["b", "a", "c"]
    assert hits[0].score > hits[1].score


def test_rrf_rejects_duplicate_or_inconsistent_chunks() -> None:
    with pytest.raises(ValueError, match="unique chunk IDs"):
        fuse_rrf([hit("a"), hit("a")], [])
    with pytest.raises(ValueError, match="conflicting"):
        fuse_rrf([hit("a")], [hit("a", content="tampered")])


def test_rrf_respects_chunk_budget_even_with_disjoint_lists() -> None:
    hits = fuse_rrf([hit(str(i)) for i in range(10)], [hit(str(i)) for i in range(10, 20)], 5)
    assert len(hits) == 5
    with pytest.raises(ValueError, match="top_k"):
        fuse_rrf([], [], 0)
