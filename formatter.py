"""Text chunking and formatting helpers for multi-message responses."""

from __future__ import annotations


def html_escape(text: str) -> str:
    """Escape HTML special characters for Telegram HTML parse mode."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


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
