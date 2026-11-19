import math
import sys
from types import ModuleType

import pytest

from aql.embeddings import DIMENSIONS, MODEL_KEY, PinnedFastEmbed, validate_vector, vector_literal


def vector(value: float = 1.0) -> list[float]:
    return [value] + [0.0] * (DIMENSIONS - 1)


def test_vectors_require_384_finite_nonzero_components() -> None:
    assert validate_vector(vector()) == vector()
    assert vector_literal(vector()).startswith("[1,0,0,")
    for bad in ([1.0], [0.0] * DIMENSIONS, [math.nan] + [0.0] * 383):
        with pytest.raises(ValueError):
            validate_vector(bad)


@pytest.mark.parametrize("allow_download", [False, True])
def test_fastembed_pins_snapshot_and_never_uses_implicit_remote_resolution(
    monkeypatch, allow_download: bool
) -> None:
    captured = {}

    def snapshot(**kwargs):
        captured["snapshot"] = kwargs
        return "/checked-pinned-snapshot"

    class FakeModel:
        def __init__(self, **kwargs):
            captured["model"] = kwargs

        def embed(self, texts):
            return [vector() for _ in texts]

    hub = ModuleType("huggingface_hub")
    hub.snapshot_download = snapshot
    fastembed = ModuleType("fastembed")
    fastembed.TextEmbedding = FakeModel
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setitem(sys.modules, "fastembed", fastembed)

    embedder = PinnedFastEmbed(allow_download=allow_download)
    assert embedder.model_key == MODEL_KEY
    assert embedder.encode(["first", "second"]) == [vector(), vector()]
    assert captured["snapshot"]["revision"] in MODEL_KEY
    assert captured["snapshot"]["local_files_only"] is not allow_download
    assert captured["model"]["specific_model_path"] == "/checked-pinned-snapshot"
    assert captured["model"]["local_files_only"] is True
    assert captured["model"]["providers"] == ["CPUExecutionProvider"]


def test_embedder_rejects_missing_or_extra_vectors(monkeypatch) -> None:
    class BrokenModel:
        def __init__(self, **_kwargs):
            pass

        def embed(self, _texts):
            return [vector()]

    hub = ModuleType("huggingface_hub")
    hub.snapshot_download = lambda **_: "/pinned"
    fastembed = ModuleType("fastembed")
    fastembed.TextEmbedding = BrokenModel
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setitem(sys.modules, "fastembed", fastembed)
    with pytest.raises(ValueError, match="different number"):
        PinnedFastEmbed().encode(["first", "second"])
