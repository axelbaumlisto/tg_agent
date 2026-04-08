"""Telegram messenger — implements the Messenger protocol via the Bot API."""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from pathlib import Path
from typing import Any, AsyncIterator, Optional

import aiohttp

from .. import config
from ..protocols import Attachment, IncomingMessage, PermissionInfo

log = logging.getLogger(__name__)

TG_API = "https://api.telegram.org"

_MIME_EXT = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/gif": ".gif",
    "image/webp": ".webp",
    "audio/ogg": ".ogg",
    "audio/mpeg": ".mp3",
    "application/pdf": ".pdf",
}


class TelegramMessenger:
    """``Messenger`` implementation backed by the Telegram Bot API."""

    def __init__(self, bot_token: str):
        self._token = bot_token
        self._http: Optional[aiohttp.ClientSession] = None
        self._offset: int = 0
        self._pending_callbacks: dict[str, str] = {}

    # -- protocol properties -------------------------------------------------

    @property
    def name(self) -> str:
        return "telegram"

    @property
    def max_message_length(self) -> int:
        return 4096

    # -- HTTP helpers --------------------------------------------------------

    async def _ensure_http(self) -> aiohttp.ClientSession:
        if self._http is None or self._http.closed:
            self._http = aiohttp.ClientSession()
        return self._http

    async def _tg(self, method: str, **params: Any) -> dict:
        http = await self._ensure_http()
        url = f"{TG_API}/bot{self._token}/{method}"
        filtered = {k: v for k, v in params.items() if v is not None}
        async with http.post(url, json=filtered) as resp:
            data = await resp.json(content_type=None)
            if not data.get("ok"):
                log.warning("TG %s failed: %s", method, data)
            return data

    # -- Messenger protocol --------------------------------------------------

    async def send_message(
        self,
        recipient: str,
        text: str,
        thread_id: Optional[str] = None,
        *,
        parse_mode: Optional[str] = None,
        reply_markup: Any = None,
    ) -> str:
        data = await self._tg(
            "sendMessage",
            chat_id=int(recipient),
            message_thread_id=int(thread_id) if thread_id else None,
            text=text[:4096],
            parse_mode=parse_mode,
            reply_markup=reply_markup,
        )
        msg_id = data.get("result", {}).get("message_id")
        if msg_id is None:
            log.error("sendMessage returned no message_id: %s", data)
            return ""
        return str(msg_id)

    async def edit_message(
        self,
        recipient: str,
        message_id: str,
        text: str,
        *,
        parse_mode: Optional[str] = None,
    ) -> bool:
        if not message_id:
            return False
        data = await self._tg(
            "editMessageText",
            chat_id=int(recipient),
            message_id=int(message_id),
            text=text[:4096],
            parse_mode=parse_mode,
        )
        return data.get("ok", False)

    async def send_typing(self, recipient: str, thread_id: Optional[str] = None) -> None:
        await self._tg(
            "sendChatAction",
            chat_id=int(recipient),
            message_thread_id=int(thread_id) if thread_id else None,
            action="typing",
        )

    async def download_attachment(self, raw_attachment: dict) -> Attachment:
        file_id = raw_attachment["file_id"]
        mime = raw_attachment.get("mime", "application/octet-stream")
        filename = raw_attachment.get("filename", "file")

        data = await self._tg("getFile", file_id=file_id)
        file_path = data.get("result", {}).get("file_path", "")
        if not file_path:
            raise RuntimeError(f"getFile returned no file_path for {file_id}")

        ext = Path(filename).suffix or _MIME_EXT.get(mime, "")
        local_name = f"ab-tg-{uuid.uuid4().hex[:12]}{ext}"
        local = Path("/tmp") / local_name

        http = await self._ensure_http()
        download_url = f"{TG_API}/file/bot{self._token}/{file_path}"
        async with http.get(download_url) as resp:
            local.write_bytes(await resp.read())

        log.debug("downloaded %s → %s", filename, local)
        return Attachment(mime=mime, filename=filename, local_path=local)

    @staticmethod
    def cleanup_attachment(att: Attachment) -> None:
        """Remove a previously downloaded attachment from disk."""
        try:
            os.unlink(att.local_path)
        except OSError:
            pass

    # -- listen (long-polling) -----------------------------------------------

    async def listen(self) -> AsyncIterator[IncomingMessage]:
        while True:
            updates = await self._get_updates()
            for upd in updates:
                self._offset = upd["update_id"] + 1

                if "callback_query" in upd:
                    cq = upd["callback_query"]
                    msg = cq.get("message", {})
                    chat_id = str(msg.get("chat", {}).get("id", ""))
                    perm_msg_id = str(msg.get("message_id", ""))
                    self._pending_callbacks[perm_msg_id] = cq["id"]

                    yield IncomingMessage(
                        sender_id=chat_id,
                        thread_id=str(msg["message_thread_id"]) if msg.get("message_thread_id") else None,
                        text="",
                        callback_data=cq.get("data", ""),
                        raw=cq,
                    )

                elif "message" in upd:
                    msg = upd["message"]
                    chat_id = str(msg["chat"]["id"])
                    thread_id = str(msg["message_thread_id"]) if msg.get("message_thread_id") else None
                    text = msg.get("text") or msg.get("caption") or ""

                    attachments: list[Attachment] = []
                    for ref in self._extract_attachment_refs(msg):
                        try:
                            attachments.append(await self.download_attachment(ref))
                        except Exception as exc:
                            log.warning("attachment download failed: %s", exc)

                    yield IncomingMessage(
                        sender_id=chat_id,
                        thread_id=thread_id,
                        text=text,
                        attachments=attachments,
                        raw=msg,
                        is_command=text.startswith("/"),
                    )

    async def _get_updates(self, timeout: int = 30) -> list[dict]:
        try:
            http = await self._ensure_http()
            url = f"{TG_API}/bot{self._token}/getUpdates"
            params = {
                "offset": self._offset,
                "timeout": timeout,
                "allowed_updates": '["message","callback_query"]',
            }
            async with http.get(url, params=params,
                                timeout=aiohttp.ClientTimeout(total=timeout + 10)) as resp:
                data = await resp.json(content_type=None)
                return data.get("result", [])
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            log.warning("getUpdates error: %s", exc)
            await asyncio.sleep(1)
            return []

    @staticmethod
    def _extract_attachment_refs(msg: dict) -> list[dict]:
        refs: list[dict] = []
        if msg.get("photo"):
            photo = msg["photo"][-1]
            refs.append({"file_id": photo["file_id"], "mime": "image/jpeg", "filename": "photo.jpg"})
        if msg.get("document"):
            doc = msg["document"]
            refs.append({
                "file_id": doc["file_id"],
                "mime": doc.get("mime_type", "application/octet-stream"),
                "filename": doc.get("file_name", "file"),
            })
        if msg.get("voice"):
            refs.append({"file_id": msg["voice"]["file_id"], "mime": "audio/ogg", "filename": "voice.ogg"})
        return refs

    # -- permissions ---------------------------------------------------------

    async def send_permission_request(
        self,
        recipient: str,
        thread_id: Optional[str],
        perm: PermissionInfo,
    ) -> str:
        text = f"\U0001f510 *Permission required*\n`{perm.title}`"
        if perm.pattern:
            text += f"\n`{perm.pattern}`"
        keyboard = {
            "inline_keyboard": [[
                {"text": "\u2705 Once", "callback_data": f"perm:once:{perm.session_id}:{perm.id}"},
                {"text": "\u2705 Always", "callback_data": f"perm:always:{perm.session_id}:{perm.id}"},
                {"text": "\u274c Deny", "callback_data": f"perm:reject:{perm.session_id}:{perm.id}"},
            ]]
        }
        return await self.send_message(recipient, text, thread_id, parse_mode="Markdown", reply_markup=keyboard)

    async def resolve_permission_ui(self, recipient: str, message_id: str, result: str) -> None:
        labels = {
            "once": "\u2705 Approved (once)",
            "always": "\u2705 Approved (always)",
            "reject": "\u274c Denied",
        }
        label = labels.get(result, result)
        await self.edit_message(recipient, message_id, f"\U0001f510 {label}")

        cq_id = self._pending_callbacks.pop(message_id, None)
        if cq_id:
            await self._tg("answerCallbackQuery", callback_query_id=cq_id)

    # -- Telegraph (long content) --------------------------------------------

    async def create_telegraph_page(self, title: str, text: str) -> str:
        from ._telegraph import create_page
        http = await self._ensure_http()
        return await create_page(http, title, text)

    # -- cleanup -------------------------------------------------------------

    async def close(self) -> None:
        if self._http and not self._http.closed:
            await self._http.close()
            self._http = None
