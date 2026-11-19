"""Explicitly versioned local ONNX embeddings; no network access by default."""

import math
from collections.abc import Sequence
from typing import Protocol

DIMENSIONS = 384
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REPO = "Qdrant/all-MiniLM-L6-v2-onnx"
# An inspected 2025 revision predating an upstream tokenizer padding/truncation change.
MODEL_REVISION = "323b6364c3138880024d5ff3c353c9aa0d2e45ff"
MODEL_KEY = f"{MODEL_REPO}@{MODEL_REVISION}/fastembed-0.9.1"


class Embedder(Protocol):
    @property
    def model_key(self) -> str: ...

    def encode(self, texts: Sequence[str]) -> list[list[float]]: ...


def validate_vector(vector: Sequence[float]) -> list[float]:
    if len(vector) != DIMENSIONS:
        raise ValueError(f"Embedding must contain exactly {DIMENSIONS} dimensions")
    values = [float(value) for value in vector]
    if any(not math.isfinite(value) for value in values):
        raise ValueError("Embedding contains a nonfinite value")
    if sum(value * value for value in values) < 1e-12:
        raise ValueError("Embedding must be nonzero for cosine similarity")
    return values


def vector_literal(vector: Sequence[float]) -> str:
    """Validate numeric input before binding it as a pgvector text value."""
    return "[" + ",".join(format(value, ".9g") for value in validate_vector(vector)) + "]"


class PinnedFastEmbed:
    """Loads only a specified Hugging Face snapshot with FastEmbed's ONNX runtime.

    A caller must opt in to model downloads; the normal service and CI do not fetch models.
    """

    def __init__(self, *, allow_download: bool = False, cache_dir: str | None = None) -> None:
        try:
            from fastembed import TextEmbedding
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise RuntimeError(
                "Install the optional semantic dependencies: uv sync --extra semantic"
            ) from exc

        local_dir = snapshot_download(
            repo_id=MODEL_REPO,
            revision=MODEL_REVISION,
            allow_patterns=[
                "model.onnx",
                "tokenizer.json",
                "tokenizer_config.json",
                "config.json",
                "special_tokens_map.json",
            ],
            cache_dir=cache_dir,
            local_files_only=not allow_download,
        )
        self._model = TextEmbedding(
            model_name=MODEL_NAME,
            specific_model_path=local_dir,
            providers=["CPUExecutionProvider"],
            local_files_only=True,
            threads=2,
        )

    @property
    def model_key(self) -> str:
        return MODEL_KEY

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        embeddings = [validate_vector(vector) for vector in self._model.embed(list(texts))]
        if len(embeddings) != len(texts):
            raise ValueError("The embedder returned a different number of vectors")
        return embeddings
