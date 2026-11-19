"""Deterministic paragraph splitting with exact character-offset citations."""

from dataclasses import dataclass

CHUNKER_VERSION = "paragraph-v1"


@dataclass(frozen=True)
class TextChunk:
    ordinal: int
    content: str
    start_offset: int
    end_offset: int


def normalize_content(content: str) -> str:
    return content.replace("\r\n", "\n").replace("\r", "\n").strip()


def chunk_paragraphs(content: str) -> list[TextChunk]:
    normalized = normalize_content(content)
    if not normalized:
        raise ValueError("Cannot chunk empty content")

    chunks: list[TextChunk] = []
    position = 0
    for paragraph in normalized.split("\n\n"):
        text = paragraph.strip()
        if not text:
            continue
        start = normalized.find(text, position)
        if start < 0:
            raise ValueError("Unable to locate a paragraph in the normalized source")
        end = start + len(text)
        chunks.append(TextChunk(len(chunks), text, start, end))
        position = end
    return chunks
