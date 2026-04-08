"""Text chunking for multi-message responses."""

from __future__ import annotations


def split_chunks(text: str, max_chars: int = 4000) -> list[str]:
    """Split *text* into chunks of at most *max_chars*, breaking at newlines."""
    if len(text) <= max_chars:
        return [text]
    chunks: list[str] = []
    while text:
        if len(text) <= max_chars:
            chunks.append(text)
            break
        cut = text.rfind("\n", 0, max_chars)
        if cut <= 0:
            cut = max_chars
        chunks.append(text[:cut])
        text = text[cut:].lstrip("\n")
    return chunks
