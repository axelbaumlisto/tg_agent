"""Throttled Telegram message editor with typing indicator."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Optional

import aiohttp

from . import config

log = logging.getLogger(__name__)

TG_API = "https://api.telegram.org"


async def tg_request(
    session: aiohttp.ClientSession,
    method: str,
    **params: Any,
) -> dict:
    url = f"{TG_API}/bot{config.BOT_TOKEN}/{method}"
    async with session.post(url, json={k: v for k, v in params.items() if v is not None}) as resp:
        data = await resp.json(content_type=None)
        if not data.get("ok"):
            log.warning("TG %s failed: %s", method, data)
        return data


class MessageEditor:
    """Accumulates text deltas, sends throttled edits to one Telegram message."""

    def __init__(
        self,
        http: aiohttp.ClientSession,
        chat_id: int,
        thread_id: Optional[int] = None,
    ):
        self.http = http
        self.chat_id = chat_id
        self.thread_id = thread_id
        self.msg_id: Optional[int] = None
        self.accumulated_text: str = ""
        self._last_edit_at: float = 0.0
        self._pending_edit: Optional[asyncio.Task] = None
        self._last_sent_text: str = ""

    async def send_placeholder(self) -> int:
        """Send initial placeholder message, return message_id."""
        data = await tg_request(
            self.http,
            "sendMessage",
            chat_id=self.chat_id,
            message_thread_id=self.thread_id,
            text="\u2699\ufe0f",
        )
        self.msg_id = data["result"]["message_id"]
        self.accumulated_text = ""
        self._last_sent_text = ""
        return self.msg_id

    async def append_delta(self, delta: str) -> None:
        """Buffer text delta; schedule throttled edit."""
        self.accumulated_text += delta
        now = time.monotonic()
        elapsed = now - self._last_edit_at
        if elapsed >= config.EDIT_INTERVAL_SECONDS:
            await self._do_edit()
        elif self._pending_edit is None or self._pending_edit.done():
            delay = config.EDIT_INTERVAL_SECONDS - elapsed
            self._pending_edit = asyncio.create_task(self._delayed_edit(delay))

    async def _delayed_edit(self, delay: float) -> None:
        await asyncio.sleep(delay)
        await self._do_edit()

    async def _do_edit(self) -> None:
        if not self.msg_id or not self.accumulated_text:
            return
        text = self.accumulated_text
        if text == self._last_sent_text:
            return
        self._last_sent_text = text
        self._last_edit_at = time.monotonic()
        await tg_request(
            self.http,
            "editMessageText",
            chat_id=self.chat_id,
            message_id=self.msg_id,
            text=text[:4096],
            parse_mode=None,
        )

    async def finalize(self, text: str) -> None:
        """Final edit — set the complete response text."""
        if self._pending_edit and not self._pending_edit.done():
            self._pending_edit.cancel()
        if not self.msg_id:
            return
        self.accumulated_text = text
        result = await tg_request(
            self.http,
            "editMessageText",
            chat_id=self.chat_id,
            message_id=self.msg_id,
            text=text[:4096],
            parse_mode="Markdown",
        )
        if not result.get("ok"):
            await tg_request(
                self.http,
                "editMessageText",
                chat_id=self.chat_id,
                message_id=self.msg_id,
                text=text[:4096],
                parse_mode=None,
            )

    async def set_error(self, text: str) -> None:
        if not self.msg_id:
            return
        await tg_request(
            self.http,
            "editMessageText",
            chat_id=self.chat_id,
            message_id=self.msg_id,
            text=f"\u274c {text}"[:4096],
        )

    async def send_typing(self) -> None:
        await tg_request(
            self.http,
            "sendChatAction",
            chat_id=self.chat_id,
            message_thread_id=self.thread_id,
            action="typing",
        )


async def send_message(
    http: aiohttp.ClientSession,
    chat_id: int,
    thread_id: Optional[int],
    text: str,
    parse_mode: Optional[str] = "Markdown",
    reply_markup: Optional[dict] = None,
) -> dict:
    return await tg_request(
        http,
        "sendMessage",
        chat_id=chat_id,
        message_thread_id=thread_id,
        text=text[:4096],
        parse_mode=parse_mode,
        reply_markup=reply_markup,
    )
