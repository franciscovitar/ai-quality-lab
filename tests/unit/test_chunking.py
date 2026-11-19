import pytest

from aql.chunking import CHUNKER_VERSION, chunk_paragraphs, normalize_content


def test_normalizes_line_endings() -> None:
    assert normalize_content("\r\nAlpha\r\n\r\nBeta\r\n") == "Alpha\n\nBeta"


def test_deterministic_chunks_preserve_offsets() -> None:
    source = "First paragraph.\n\nSecond paragraph.\n\nFirst paragraph."
    chunks = chunk_paragraphs(source)
    assert CHUNKER_VERSION == "paragraph-v1"
    assert [chunk.ordinal for chunk in chunks] == [0, 1, 2]
    assert [source[chunk.start_offset : chunk.end_offset] for chunk in chunks] == [
        chunk.content for chunk in chunks
    ]
    assert chunks[2].start_offset > chunks[0].start_offset


def test_chunking_rejects_blank_source() -> None:
    with pytest.raises(ValueError, match="empty"):
        chunk_paragraphs(" \r\n ")
