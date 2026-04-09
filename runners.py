"""SessionRunner — per-session state machine using protocol abstractions.

Drives one agent session ↔ one messenger chat/thread pair.
All interactions go through Messenger and AgentBackend protocols —
no Telegram or OpenCode imports.
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import time
from typing import Callable, Coroutine, Any, Optional

from . import config
from .formatter import html_escape, split_chunks
from .protocols import (
    AgentBackend,
    AgentEvent,
    MessagePart,
    Messenger,
    ModelRef,
    PermissionRequest,
    QuestionRequest,
    ReasoningDelta,
    SessionError,
    SessionIdle,
    StatusUpdate,
    TextDelta,
    ToolEnd,
    ToolStart,
)

log = logging.getLogger(__name__)

STATES = ("warmup", "idle", "generating", "sleeping", "reconnecting", "error")


LongContentHandler = Callable[[str, str], Coroutine[Any, Any, str]]


class SessionRunner:
    """Drives one agent session ↔ one messenger conversation."""

    def __init__(
        self,
        session_id: str,
        chat_id: str,
        thread_id: Optional[str],
        agent: AgentBackend,
        messenger: Messenger,
        *,
        directory: Optional[str] = None,
        auto_approve: bool = False,
        long_content_handler: Optional[LongContentHandler] = None,
    ):
        self.session_id = session_id
        self.chat_id = chat_id
        self.thread_id = thread_id
        self.agent = agent
        self.messenger = messenger
        self.directory = directory
        self.auto_approve = auto_approve
        self._long_content = long_content_handler

        self.state: str = "warmup"
        self.last_active: float = time.monotonic()
        self.sse_task: Optional[asyncio.Task] = None
        self.typing_task: Optional[asyncio.Task] = None

        self._msg_id: Optional[str] = None
        self._accumulated_text: str = ""
        self._last_edit_at: float = 0.0
        self._pending_edit: Optional[asyncio.Task] = None
        self._last_sent_text: str = ""
        self._stream_parse_mode: Optional[str] = None

        self._reasoning_text: str = ""
        self._in_reasoning: bool = False
        self._reasoning_start: float = 0.0

        self._response_text: str = ""
        self._tool_lines: dict[str, str] = {}
        self._tool_order: list[str] = []

        self.pending_permissions: dict[str, str] = {}
        self.pending_questions: dict[str, QuestionRequest] = {}

        self._last_parts: list[MessagePart] = []
        self._last_model: Optional[ModelRef] = None
        self._retry_callback: Optional[Callable[["SessionRunner", str], Coroutine[Any, Any, None]]] = None
        self._success_callback: Optional[Callable[[str, str], None]] = None

    # -- public API ---------------------------------------------------------

    async def wake_and_prompt(
        self,
        parts: list[MessagePart],
        model: Optional[ModelRef] = None,
    ) -> None:
        self.last_active = time.monotonic()

        reconnecting = self.state == "sleeping"

        if self.sse_task is None or self.sse_task.done():
            self.sse_task = asyncio.create_task(self._sse_loop())

        self.state = "generating"
        self._accumulated_text = ""
        self._last_sent_text = ""
        self._last_edit_at = 0.0
        self._stream_parse_mode = None
        self._reasoning_text = ""
        self._in_reasoning = False
        self._reasoning_start = 0.0
        self._response_text = ""
        self._tool_lines.clear()
        self._tool_order.clear()

        placeholder = "\U0001f504 Reconnecting\u2026" if reconnecting else "\u2699\ufe0f"
        self._msg_id = await self.messenger.send_message(self.chat_id, placeholder, self.thread_id)
        self._start_typing()

        preview = "; ".join(p.text[:60] if hasattr(p, "text") else str(p)[:60] for p in parts)
        log.info("prompt → session %s: %s", self.session_id, preview)

        self._last_parts = parts
        self._last_model = model

        try:
            await self.agent.send_prompt(self.session_id, parts, model, directory=self.directory)
        except Exception as exc:
            log.error("send_prompt failed: %s", exc)
            if self._msg_id:
                await self.messenger.edit_message(self.chat_id, self._msg_id, f"\u274c {exc}")
            self.state = "error"

    async def close_sse(self) -> None:
        self._stop_typing()
        if self.sse_task and not self.sse_task.done():
            self.sse_task.cancel()
            try:
                await self.sse_task
            except (asyncio.CancelledError, Exception):
                pass
        self.sse_task = None
        self.state = "sleeping"

    async def reconnect(self) -> None:
        self.state = "reconnecting"
        if self.sse_task and not self.sse_task.done():
            self.sse_task.cancel()
        self.sse_task = asyncio.create_task(self._sse_loop())

    async def shutdown(self) -> None:
        self._stop_typing()
        if self.sse_task and not self.sse_task.done():
            self.sse_task.cancel()
            try:
                await self.sse_task
            except (asyncio.CancelledError, Exception):
                pass

    # -- SSE loop -----------------------------------------------------------

    async def _sse_loop(self) -> None:
        try:
            async for event in self.agent.subscribe_events(self.session_id, directory=self.directory):
                self.last_active = time.monotonic()
                await self.handle_event(event)
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            log.error("SSE loop crashed: %s", exc)
            self.state = "error"

    # -- event dispatch -----------------------------------------------------

    _REASONING_TAIL = 600

    _EVENT_HANDLERS: dict[type, str] = {
        ReasoningDelta: "_on_reasoning",
        TextDelta: "_on_text",
        ToolStart: "_on_tool_start",
        ToolEnd: "_on_tool_end",
        PermissionRequest: "_on_permission",
        QuestionRequest: "_on_question",
        SessionIdle: "_on_idle",
        SessionError: "_on_error",
        StatusUpdate: "_on_status",
    }

    async def handle_event(self, event: AgentEvent) -> None:
        handler_name = self._EVENT_HANDLERS.get(type(event))
        if handler_name:
            await getattr(self, handler_name)(event)

    async def _on_reasoning(self, event: ReasoningDelta) -> None:
        if not self._in_reasoning:
            self._reasoning_start = time.monotonic()
        self._in_reasoning = True
        self._reasoning_text += event.text
        self._rebuild_composite()
        await self._throttled_edit()

    async def _on_text(self, event: TextDelta) -> None:
        if self._in_reasoning:
            self._in_reasoning = False
        self._response_text += event.text
        self._rebuild_composite()
        await self._throttled_edit()

    async def _on_tool_start(self, event: ToolStart) -> None:
        cid = event.call_id or f"_anon_{len(self._tool_order)}"
        self._tool_lines[cid] = f"\U0001f527 <b>{html_escape(event.name)}</b>\u2026"
        if cid not in self._tool_order:
            self._tool_order.append(cid)
        self._rebuild_composite()
        await self._throttled_edit()

    async def _on_tool_end(self, event: ToolEnd) -> None:
        cid = event.call_id
        name_esc = html_escape(event.name)
        if event.state == "completed":
            label = f"\u2705 <b>{name_esc}</b>"
            if event.title:
                label += f" \u2014 {html_escape(event.title[:80])}"
        else:
            label = f"\u274c <b>{name_esc}</b>"
            if event.error:
                label += f" \u2014 {html_escape(event.error[:80])}"
        self._tool_lines[cid] = label
        self._rebuild_composite()
        await self._throttled_edit()
        if event.state == "completed" and event.name in ("write", "save"):
            await self._try_deliver_file(event.title)

    def _rebuild_composite(self) -> None:
        """Build a single message from reasoning + tools + streaming response.

        Layout (all phases coexist, scrolling window of last 5 tools):
            💭 <reasoning tail>
            🔧 tool_a…
            ✅ tool_b — title
            🔧 tool_c…
            ─────────
            <response text so far>

        On finalize, the entire message is replaced with the final response.
        """
        parts: list[str] = []

        if self._in_reasoning and self._reasoning_text:
            tail = self._reasoning_text[-self._REASONING_TAIL:]
            if len(self._reasoning_text) > self._REASONING_TAIL:
                tail = "\u2026" + tail
            parts.append(f"\U0001f4ad <i>{html_escape(tail)}</i>")

        last_tools = self._tool_order[-5:]
        for cid in last_tools:
            line = self._tool_lines.get(cid, "")
            if line:
                parts.append(line)

        if self._response_text:
            if parts:
                parts.append("\u2500" * 12)
            parts.append(self._response_text)

        self._accumulated_text = "\n".join(parts) if parts else "\u2699\ufe0f"
        has_html = self._in_reasoning or self._tool_lines
        self._stream_parse_mode = "HTML" if has_html and not self._response_text else None

    async def _on_permission(self, event: PermissionRequest) -> None:
        if event.info.id in self.pending_permissions:
            log.debug("duplicate permission %s, skipping", event.info.id)
            return
        log.info("permission request %s: %s (%s)", event.info.id, event.info.title, event.info.pattern)
        if self.auto_approve:
            log.info("auto-approving permission %s", event.info.id)
            await self.agent.respond_permission(
                self.session_id, event.info.id, "always",
                directory=self.directory,
            )
            return
        mid = await self.messenger.send_permission_request(
            self.chat_id, self.thread_id, event.info,
        )
        self.pending_permissions[event.info.id] = mid

    async def _on_question(self, event: QuestionRequest) -> None:
        log.info("question request %s: %d questions", event.request_id, len(event.questions))
        if self.auto_approve and event.questions:
            answers = []
            for q in event.questions:
                opts = q.get("options", [])
                default = next((o for o in opts if o.get("default")), opts[0] if opts else None)
                answers.append({"id": q.get("id", ""), "value": default.get("value", "") if default else ""})
            try:
                await self.agent.reply_question(event.request_id, answers, directory=self.directory)
                return
            except Exception as exc:
                log.warning("auto-reply question failed: %s", exc)
        lines = ["\u2753 The model has a question:"]
        for q in event.questions:
            prompt_text = q.get("prompt", q.get("text", ""))
            lines.append(f"\n{prompt_text}")
            for i, opt in enumerate(q.get("options", []), 1):
                label = opt.get("label", opt.get("value", f"Option {i}"))
                lines.append(f"  {i}. {label}")
        lines.append(f"\nReply with: `q:{event.request_id}:<answer>`")
        await self.messenger.send_message(
            self.chat_id, "\n".join(lines), self.thread_id, parse_mode="Markdown",
        )
        self.pending_questions[event.request_id] = event

    async def _on_idle(self, event: SessionIdle) -> None:
        if self.state != "idle":
            log.info("SessionIdle for %s (accumulated %d chars)", self.session_id, len(self._accumulated_text))
            self.pending_permissions.clear()
            if self._success_callback and self._last_model:
                self._success_callback(
                    self._last_model.provider_id, self._last_model.model_id,
                )
            await self._finalize_response()

    async def _on_error(self, event: SessionError) -> None:
        if config.is_provider_error(event.error) and self._retry_callback:
            log.warning("provider error detected: %s — triggering fallback", event.error[:120])
            if self._msg_id:
                await self.messenger.edit_message(
                    self.chat_id, self._msg_id,
                    f"\u26a0\ufe0f Provider error, switching\u2026",
                )
            await self._retry_callback(self, event.error)
            return
        self._stop_typing()
        if self._msg_id:
            max_len = self.messenger.max_message_length
            await self.messenger.edit_message(
                self.chat_id, self._msg_id,
                f"\u274c {event.error}"[:max_len],
            )
        self.state = "error"

    async def _on_status(self, event: StatusUpdate) -> None:
        if event.status == "idle" and self.state != "idle":
            await self._finalize_response()
        elif event.status == "busy":
            self.state = "generating"
        elif event.status == "retry":
            log.info("session %s: %s", self.session_id, event.message)

    # -- streaming edits (throttled) ----------------------------------------

    async def _throttled_edit(self) -> None:
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
        if not self._msg_id or not self._accumulated_text:
            return
        text = self._accumulated_text
        if text == self._last_sent_text:
            return
        self._last_sent_text = text
        self._last_edit_at = time.monotonic()
        max_len = self.messenger.max_message_length
        await self.messenger.edit_message(
            self.chat_id, self._msg_id, text[:max_len],
            parse_mode=self._stream_parse_mode,
        )
        await self.messenger.send_typing(self.chat_id, self.thread_id)

    # -- finalize (session idle) --------------------------------------------

    async def _finalize_response(self) -> None:
        self._stop_typing()
        self._in_reasoning = False
        if self._pending_edit and not self._pending_edit.done():
            self._pending_edit.cancel()

        text = self._response_text or self._accumulated_text
        if not text.strip():
            if self._msg_id:
                await self.messenger.edit_message(self.chat_id, self._msg_id, "(empty response)")
            self._reset_editor()
            self.state = "idle"
            return

        max_len = self.messenger.max_message_length
        chunks = split_chunks(text, max_len)

        if len(chunks) <= config.MAX_MESSAGE_CHUNKS:
            await self._send_chunks(chunks)
        else:
            await self._send_long_response(text, chunks)

        self._reset_editor()
        self.state = "idle"

    async def _send_long_response(self, text: str, chunks: list[str]) -> None:
        """Handle responses too long for inline messages — try document, then Telegraph."""
        sent_doc = False
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".md", prefix="response-", delete=False,
            ) as f:
                f.write(text)
                tmp_path = f.name
            await self.messenger.send_document(
                self.chat_id, tmp_path, self.thread_id,
                caption="Full response",
            )
            sent_doc = True
        except Exception as exc:
            log.warning("document delivery for long response failed: %s", exc)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        if sent_doc:
            if self._msg_id:
                preview = text[:300] + "\u2026" if len(text) > 300 else text
                ok = await self.messenger.edit_message(
                    self.chat_id, self._msg_id, preview, parse_mode="Markdown",
                )
                if not ok:
                    await self.messenger.edit_message(self.chat_id, self._msg_id, preview)
            return

        if self._long_content:
            first_line = text.split("\n", 1)[0][:120] or "Response"
            try:
                url = await self._long_content(first_line, text)
                if self._msg_id:
                    await self.messenger.edit_message(
                        self.chat_id, self._msg_id,
                        f"\U0001f4c4 [Full response]({url})",
                        parse_mode="Markdown",
                    )
                return
            except Exception as exc:
                log.warning("long content handler failed, falling back: %s", exc)

        await self._send_chunks(chunks[:config.MAX_MESSAGE_CHUNKS + 2])

    async def _send_chunks(self, chunks: list[str]) -> None:
        if self._msg_id and chunks:
            ok = await self.messenger.edit_message(
                self.chat_id, self._msg_id, chunks[0], parse_mode="Markdown",
            )
            if not ok:
                await self.messenger.edit_message(self.chat_id, self._msg_id, chunks[0])
        for chunk in chunks[1:]:
            await self.messenger.send_message(self.chat_id, chunk, self.thread_id)

    def _reset_editor(self) -> None:
        self._accumulated_text = ""
        self._response_text = ""
        self._tool_lines.clear()
        self._tool_order.clear()
        self._msg_id = None

    # -- file delivery -------------------------------------------------------

    _DELIVERABLE_EXTENSIONS = {
        ".pdf", ".zip", ".tar", ".gz", ".tgz", ".bz2",
        ".py", ".rs", ".js", ".ts", ".sh", ".md", ".txt",
        ".csv", ".json", ".yaml", ".yml", ".toml",
    }

    async def _try_deliver_file(self, title: str) -> None:
        """If the tool title looks like a file path with a deliverable extension, send it."""
        if not title:
            return
        path = title.strip()
        if not os.path.isabs(path):
            base = self.directory or config.OC_DIRECTORY
            path = os.path.join(base, path)
        if not os.path.isfile(path):
            return
        ext = os.path.splitext(path)[1].lower()
        if ext not in self._DELIVERABLE_EXTENSIONS:
            return
        size = os.path.getsize(path)
        if size > 50 * 1024 * 1024:
            return
        try:
            await self.messenger.send_document(
                self.chat_id, path, self.thread_id,
                caption=os.path.basename(path),
            )
        except Exception as exc:
            log.warning("file delivery failed for %s: %s", path, exc)

    # -- typing indicator ---------------------------------------------------

    def _start_typing(self) -> None:
        self._stop_typing()
        self.typing_task = asyncio.create_task(self._typing_loop())

    def _stop_typing(self) -> None:
        if self.typing_task and not self.typing_task.done():
            self.typing_task.cancel()
        self.typing_task = None

    async def _typing_loop(self) -> None:
        try:
            while True:
                await self.messenger.send_typing(self.chat_id, self.thread_id)
                await asyncio.sleep(config.TYPING_INTERVAL_SECONDS)
        except asyncio.CancelledError:
            pass
