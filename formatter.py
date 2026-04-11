"""Text chunking, formatting, and shared file helpers."""

from __future__ import annotations

import logging
import os
import tempfile
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .protocols import Messenger

log = logging.getLogger(__name__)


def is_safe_path(path: str, base_dir: Optional[str]) -> bool:
    """Return True only if *path* resolves within *base_dir*."""
    if not base_dir:
        return False
    real_base = os.path.realpath(base_dir)
    real_path = os.path.realpath(path)
    return real_path.startswith(real_base + os.sep) or real_path == real_base


async def send_temp_document(
    messenger: "Messenger",
    chat_id: str,
    thread_id: Optional[str],
    content: str,
    filename: str,
    caption: str = "",
) -> None:
    """Write *content* to a temp file, send as document, then clean up."""
    suffix = os.path.splitext(filename)[1] or ".txt"
    with tempfile.NamedTemporaryFile(mode="w", suffix=suffix, prefix="oc-tg-", delete=False) as f:
        f.write(content)
        tmp_path = f.name
    try:
        await messenger.send_document(chat_id, tmp_path, thread_id, caption=caption or filename)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def html_escape(text: str) -> str:
    """Escape HTML special characters for Telegram HTML parse mode."""
    return (text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;"))


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
