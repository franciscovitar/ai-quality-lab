"""A conservative retrieval-score gate, not a claim of answer correctness."""

import math
from collections.abc import Sequence
from dataclasses import dataclass

from aql.evaluation import GoldenDataset
from aql.retrieval import SearchHit


@dataclass(frozen=True)
class CalibratedThreshold:
    value: float
    dev_negative_cases: int
    objective: str = "exclude_all_dev_unanswerable_top1_scores"


def _check_score(hit: SearchHit) -> float:
    score = float(hit.score)
    if not math.isfinite(score) or not -1.000001 <= score <= 1.000001:
        raise ValueError("Cosine scores must be finite and in [-1, 1]")
    return score


def calibrate_dev_negatives(
    dataset: GoldenDataset, hits_by_question: dict[str, Sequence[SearchHit]]
) -> CalibratedThreshold:
    """Select a threshold solely from dev negatives; do not peek at holdout labels.

    The cutoff is the smallest double above the maximum *dev-negative* top-1 cosine.
    It guarantees zero false accepts on the observed dev negatives, but provides no
    statistical or generalization guarantee, particularly with only two cases.
    """
    negatives = [
        case
        for case in dataset.cases
        if case.split == "dev" and case.expected_behavior == "abstain"
    ]
    if not negatives:
        raise ValueError("Dev split has no unanswerable cases to calibrate against")
    scores = []
    for case in negatives:
        if case.question not in hits_by_question:
            raise ValueError(f"Missing retrieval scores for {case.case_id}")
        hits = hits_by_question[case.question]
        if hits:
            scores.append(_check_score(hits[0]))

    # Above the theoretical cosine maximum is a finite, JSON-safe fail-closed cutoff.
    threshold = math.nextafter(max(scores) if scores else 1.0, math.inf)
    return CalibratedThreshold(threshold, len(negatives))


def gated_hits(hits: Sequence[SearchHit], *, threshold: float, top_k: int) -> list[SearchHit]:
    if not 1 <= top_k <= 10:
        raise ValueError("top_k must be 1 through 10")
    if math.isnan(threshold):
        raise ValueError("Threshold cannot be NaN")
    if not hits or _check_score(hits[0]) < threshold:
        return []
    return list(hits[:top_k])
