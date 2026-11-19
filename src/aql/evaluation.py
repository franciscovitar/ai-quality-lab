"""Version-locked retrieval judgments and deterministic, non-LLM evaluation.

The top-k budget is in *chunks*, because that is what the retrieval API returns.
Metrics deduplicate those chunks into version-qualified source documents.
"""

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from aql.metrics import recall_at_k, reciprocal_rank_at_k
from aql.retrieval import SearchHit


class SourceLabel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    slug: str = Field(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
    version: int = Field(ge=1)

    @property
    def key(self) -> str:
        return f"{self.slug}@v{self.version}"


class GoldenCase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str = Field(pattern=r"^R[0-9]{3}$")
    split: Literal["dev", "holdout"]
    question: str = Field(min_length=3, max_length=500)
    expected_behavior: Literal["answer", "abstain"]
    expected_documents: tuple[SourceLabel, ...]
    tags: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_judgment(self) -> "GoldenCase":
        if self.question.strip() != self.question:
            raise ValueError("Question must not have surrounding whitespace")
        if self.expected_behavior == "answer" and not self.expected_documents:
            raise ValueError("Answerable cases need at least one relevant source")
        if self.expected_behavior == "abstain" and self.expected_documents:
            raise ValueError("Unanswerable cases must not have relevant sources")
        if len(set(source.key for source in self.expected_documents)) != len(
            self.expected_documents
        ):
            raise ValueError("Relevant source labels cannot be repeated")
        if len(set(self.tags)) != len(self.tags) or any(
            not tag or not tag.replace("_", "").isalnum() for tag in self.tags
        ):
            raise ValueError("Tags must be distinct, non-empty identifiers")
        return self


class GoldenDataset(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1]
    dataset_id: str = Field(min_length=1)
    corpus_id: str = Field(min_length=1)
    corpus_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    label_provenance: str = Field(min_length=1)
    cases: tuple[GoldenCase, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_coverage(self) -> "GoldenDataset":
        ids = [case.case_id for case in self.cases]
        if ids != sorted(set(ids)):
            raise ValueError("Case IDs must be unique and sorted")
        questions = [" ".join(case.question.lower().split()) for case in self.cases]
        if len(set(questions)) != len(questions):
            raise ValueError("Duplicate questions hide dataset coverage")
        for split in ("dev", "holdout"):
            split_cases = [case for case in self.cases if case.split == split]
            if not {case.expected_behavior for case in split_cases} == {"answer", "abstain"}:
                raise ValueError(f"{split} needs answerable and unanswerable cases")
        return self


@dataclass(frozen=True)
class LockedDataset:
    data: GoldenDataset
    dataset_sha256: str
    current_sources: frozenset[str]


def load_dataset(dataset_path: Path, corpus_path: Path) -> LockedDataset:
    """Verify content hash and labels before using a golden dataset."""
    dataset_bytes = dataset_path.read_bytes()
    data = GoldenDataset.model_validate_json(dataset_bytes)
    corpus_bytes = corpus_path.read_bytes()
    if hashlib.sha256(corpus_bytes).hexdigest() != data.corpus_sha256:
        raise ValueError("The corpus changed; golden labels require a deliberate review")
    corpus = json.loads(corpus_bytes)
    if corpus["corpus_id"] != data.corpus_id:
        raise ValueError("Golden dataset points at the wrong corpus ID")

    current_sources = frozenset(
        SourceLabel(slug=doc["slug"], version=doc["version"]).key
        for doc in corpus["documents"]
        if doc["is_current"]
    )
    for case in data.cases:
        for label in case.expected_documents:
            if label.key not in current_sources:
                raise ValueError(f"{case.case_id}: label is not a current source: {label.key}")

    return LockedDataset(data, hashlib.sha256(dataset_bytes).hexdigest(), current_sources)


def _aggregate(rows: Sequence[dict]) -> dict:
    positives = [row for row in rows if row["expected_behavior"] == "answer"]
    negatives = [row for row in rows if row["expected_behavior"] == "abstain"]

    def average(numbers: list[float | int]) -> float | None:
        return round(sum(numbers) / len(numbers), 6) if numbers else None

    return {
        "cases": len(rows),
        "answerable": len(positives),
        "unanswerable": len(negatives),
        "recall_at_k": average([row["recall_at_k"] for row in positives]),
        "mrr_at_k": average([row["mrr_at_k"] for row in positives]),
        "hit_rate_at_k": average([int(row["recall_at_k"] > 0) for row in positives]),
        "unanswerable_no_hit_rate": average([int(row["no_hit"]) for row in negatives]),
    }


def evaluate_retrieval(
    dataset: LockedDataset,
    retrieve: Callable[[str, int], Sequence[SearchHit]],
    *,
    top_k: int = 5,
) -> dict:
    if not 1 <= top_k <= 10:
        raise ValueError("top_k must be 1 through 10 (chunks)")

    rows: list[dict] = []
    for case in dataset.data.cases:
        hits = list(retrieve(case.question, top_k))
        if len(hits) > top_k:
            raise ValueError(f"{case.case_id}: retriever exceeded the top_k chunk budget")
        chunk_ids = [hit.chunk_id for hit in hits]
        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError(f"{case.case_id}: duplicate chunk IDs")

        # One document with several matching chunks counts as one retrieved source.
        ranked_sources = list(
            dict.fromkeys(f"{hit.document_slug}@v{hit.document_version}" for hit in hits)
        )
        stale_sources = set(ranked_sources) - dataset.current_sources
        if stale_sources:
            raise ValueError(f"{case.case_id}: retrieved stale/unknown sources: {stale_sources}")

        relevant = {label.key for label in case.expected_documents}
        answerable = case.expected_behavior == "answer"
        rows.append(
            {
                "case_id": case.case_id,
                "split": case.split,
                "tags": list(case.tags),
                "expected_behavior": case.expected_behavior,
                "expected_documents": sorted(relevant),
                "retrieved_documents": ranked_sources,
                "retrieved_chunks": len(hits),
                "recall_at_k": recall_at_k(relevant, ranked_sources, top_k) if answerable else None,
                "mrr_at_k": reciprocal_rank_at_k(relevant, ranked_sources, top_k)
                if answerable
                else None,
                "no_hit": not hits if not answerable else None,
            }
        )

    return {
        "schema_version": 1,
        "dataset_id": dataset.data.dataset_id,
        "dataset_sha256": dataset.dataset_sha256,
        "corpus_id": dataset.data.corpus_id,
        "corpus_sha256": dataset.data.corpus_sha256,
        "strategy": "postgres-english-fts-v1",
        "top_k_chunks": top_k,
        "label_provenance": dataset.data.label_provenance,
        "summary": _aggregate(rows),
        "by_split": {
            split: _aggregate([row for row in rows if row["split"] == split])
            for split in ("dev", "holdout")
        },
        "by_tag": {
            tag: _aggregate([row for row in rows if tag in row["tags"]])
            for tag in sorted({tag for row in rows for tag in row["tags"]})
        },
        "cases": rows,
    }


def find_regressions(baseline: dict, candidate: dict) -> list[str]:
    """Per-case regression check; aggregate improvements cannot hide a worse case."""
    keys = ("schema_version", "dataset_sha256", "corpus_sha256", "strategy", "top_k_chunks")
    if any(baseline.get(key) != candidate.get(key) for key in keys):
        raise ValueError("Incomparable reports: dataset, corpus or retrieval budget changed")
    prior = {row["case_id"]: row for row in baseline["cases"]}
    current = {row["case_id"]: row for row in candidate["cases"]}
    if (
        prior.keys() != current.keys()
        or len(prior) != len(baseline["cases"])
        or len(current) != len(candidate["cases"])
    ):
        raise ValueError("Incomparable reports: case IDs differ or repeat")

    regressions = []
    for case_id in sorted(prior):
        before, after = prior[case_id], current[case_id]
        for field in ("expected_behavior", "expected_documents", "split", "tags"):
            if before[field] != after[field]:
                raise ValueError(f"Incomparable reports: {case_id} {field} changed")
        if before["expected_behavior"] == "answer":
            for metric in ("recall_at_k", "mrr_at_k"):
                if after[metric] + 1e-6 < before[metric]:
                    regressions.append(f"{case_id}: {metric} {before[metric]} -> {after[metric]}")
        elif before["no_hit"] and not after["no_hit"]:
            regressions.append(f"{case_id}: unexpected retrieval on unanswerable question")
    return regressions


def report_json(report: dict) -> str:
    """Stable bytes across repeated runs on the same indexed corpus."""
    return json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
