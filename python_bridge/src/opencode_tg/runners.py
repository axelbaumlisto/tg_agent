"""SessionRunner — per-session state machine using protocol abstractions.

Drives one agent session <-> one messenger chat/thread pair.
All interactions go through Messenger and AgentBackend protocols —
no Telegram or OpenCode imports.

Architecture:
  SessionRunner  — orchestrator: SSE loop, event dispatch, permissions/questions
  CompositeUI    — builds and edits the single composite message (reasoning+tools+response)
  FileDelivery   — delivers files from tool output to the chat
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Callable, Coroutine, Any, Optional

from . import config
from .formatter import html_escape, is_safe_path, send_temp_document, split_chunks
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

LongContentHandler = Callable[[str, str], Coroutine[Any, Any, str]]

_TOOL_WINDOW = 5
_TITLE_TRUNCATE = 80
_PREVIEW_TRUNCATE = 300
_MAX_DELIVERABLE_BYTES = 50 * 1024 * 1024

_DELIVERABLE_EXTENSIONS = {
    ".pdf", ".zip", ".tar", ".gz", ".tgz", ".bz2",
    ".py", ".rs", ".js", ".ts", ".sh", ".md", ".txt",
    ".csv", ".json", ".yaml", ".yml", ".toml",
}


# ---------------------------------------------------------------------------
# CompositeUI — single-message UX (reasoning + tools + response)
# ---------------------------------------------------------------------------

class CompositeUI:
    """Builds and throttle-edits a single composite message in the chat."""

    _REASONING_TAIL = 600

    def __init__(self, messenger: Messenger, chat_id: str, thread_id: Optional[str]) -> None:
        self.messenger = messenger
        self.chat_id = chat_id
        self.thread_id = thread_id

        self.msg_id: Optional[str] = None
        self._accumulated_text: str = ""
        self._last_edit_at: float = 0.0
        self._pending_edit: Optional[asyncio.Task] = None
        self._last_sent_text: str = ""
        self._stream_parse_mode: Optional[str] = None

        self.reasoning_text: str = ""
        self.in_reasoning: bool = False
        self.reasoning_start: float = 0.0

        self.response_text: str = ""
        self.tool_lines: dict[str, str] = {}
        self.tool_order: list[str] = []

        self.typing_task: Optional[asyncio.Task] = None

    def reset(self) -> None:
        """Clear all UI state for a new prompt cycle."""
        self._accumulated_text = ""
        self._last_sent_text = ""
        self._last_edit_at = 0.0
        self._stream_parse_mode = None
        self.reasoning_text = ""
        self.in_reasoning = False
        self.reasoning_start = 0.0
        self.response_text = ""
        self.tool_lines.clear()
        self.tool_order.clear()
        if self._pending_edit and not self._pending_edit.done():
            self._pending_edit.cancel()
            self._pending_edit = None

    def reset_editor(self) -> None:
        """Clear accumulated text and message reference after finalize."""
        self._accumulated_text = ""
        self.response_text = ""
        self.tool_lines.clear()
        self.tool_order.clear()
        self.msg_id = None

    # -- composite build ----------------------------------------------------

    def rebuild(self) -> None:
        """Recompute the composite message from reasoning + tools + response."""
        parts: list[str] = []
        has_html = bool(self.in_reasoning or self.tool_lines)

        if self.in_reasoning and self.reasoning_text:
            tail = self.reasoning_text[-self._REASONING_TAIL:]
            if len(self.reasoning_text) > self._REASONING_TAIL:
                tail = "\u2026" + tail
            parts.append(f"\U0001f4ad <i>{html_escape(tail)}</i>")

        last_tools = self.tool_order[-_TOOL_WINDOW:]
        for cid in last_tools:
            line = self.tool_lines.get(cid, "")
            if line:
                parts.append(line)

        if self.response_text:
            if parts:
                parts.append("\u2500" * 12)
            if has_html:
                parts.append(html_escape(self.response_text))
            else:
                parts.append(self.response_text)

        self._accumulated_text = "\n".join(parts) if parts else "\u2699\ufe0f"
        self._stream_parse_mode = "HTML" if has_html else None

    # -- throttled edits ----------------------------------------------------

    async def throttled_edit(self) -> None:
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
        if not self.msg_id or not self._accumulated_text:
            return
        text = self._accumulated_text
        if text == self._last_sent_text:
            return
        self._last_sent_text = text
        self._last_edit_at = time.monotonic()
        max_len = self.messenger.max_message_length
        await self.messenger.edit_message(
            self.chat_id, self.msg_id, text[:max_len],
            parse_mode=self._stream_parse_mode,
        )
        await self.messenger.send_typing(self.chat_id, self.thread_id)

    # -- finalize -----------------------------------------------------------

    async def finalize(self, long_content: Optional[LongContentHandler] = None) -> None:
        """Replace composite message with the final response text."""
        self.stop_typing()
        self.in_reasoning = False
        if self._pending_edit and not self._pending_edit.done():
            self._pending_edit.cancel()

        text = self.response_text or self._accumulated_text
        if not text.strip():
            if self.msg_id:
                await self.messenger.edit_message(self.chat_id, self.msg_id, "(empty response)")
            self.reset_editor()
            return

        max_len = self.messenger.max_message_length
        chunks = split_chunks(text, max_len)

        if len(chunks) <= config.MAX_MESSAGE_CHUNKS:
            await self._send_chunks(chunks)
        else:
            await self._send_long_response(text, chunks, long_content)

        self.reset_editor()

    async def _send_long_response(
        self, text: str, chunks: list[str], long_content: Optional[LongContentHandler],
    ) -> None:
        sent_doc = False
        try:
            await send_temp_document(
                self.messenger, self.chat_id, self.thread_id,
                text, "response.md", caption="Full response",
            )
            sent_doc = True
        except Exception as exc:
            log.warning("document delivery for long response failed: %s", exc)

        if sent_doc:
            if self.msg_id:
                preview = text[:_PREVIEW_TRUNCATE] + "\u2026" if len(text) > _PREVIEW_TRUNCATE else text
                await self.messenger.edit_message(self.chat_id, self.msg_id, preview)
            return

        if long_content:
            first_line = text.split("\n", 1)[0][:120] or "Response"
            try:
                url = await long_content(first_line, text)
                if self.msg_id:
                    await self.messenger.edit_message(
                        self.chat_id, self.msg_id,
                        f"\U0001f4c4 [Full response]({url})",
                        parse_mode="Markdown",
                    )
                return
            except Exception as exc:
                log.warning("long content handler failed, falling back: %s", exc)

        await self._send_chunks(chunks[:config.MAX_MESSAGE_CHUNKS + 2])

    async def _send_chunks(self, chunks: list[str]) -> None:
        if self.msg_id and chunks:
            ok = await self.messenger.edit_message(
                self.chat_id, self.msg_id, chunks[0], parse_mode="Markdown",
            )
            if not ok:
                await self.messenger.edit_message(self.chat_id, self.msg_id, chunks[0])
        for chunk in chunks[1:]:
            await self.messenger.send_message(self.chat_id, chunk, self.thread_id)

    # -- typing indicator ---------------------------------------------------

    def start_typing(self) -> None:
        self.stop_typing()
        self.typing_task = asyncio.create_task(self._typing_loop())

    def stop_typing(self) -> None:
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


# ---------------------------------------------------------------------------
# FileDelivery — push deliverable files to the chat
# ---------------------------------------------------------------------------

class FileDelivery:
    """Deliver files written by tools to the messenger chat."""

    def __init__(self, messenger: Messenger, chat_id: str, thread_id: Optional[str], directory: Optional[str]) -> None:
        self.messenger = messenger
        self.chat_id = chat_id
        self.thread_id = thread_id
        self.directory = directory

    async def try_deliver(self, title: str) -> None:
        """If *title* looks like a deliverable file path, send it as a document."""
        if not title:
            return
        path = title.strip()
        if not os.path.isabs(path):
            base = self.directory or config.OC_DIRECTORY
            path = os.path.join(base, path)
        if not is_safe_path(path, self.directory or config.OC_DIRECTORY):
            return
        if not os.path.isfile(path):
            return
        ext = os.path.splitext(path)[1].lower()
        if ext not in _DELIVERABLE_EXTENSIONS:
            return
        size = os.path.getsize(path)
        if size > _MAX_DELIVERABLE_BYTES:
            return
        try:
            await self.messenger.send_document(
                self.chat_id, path, self.thread_id,
                caption=os.path.basename(path),
            )
        except Exception as exc:
            log.warning("file delivery failed for %s: %s", path, exc)


# ---------------------------------------------------------------------------
# SessionRunner — orchestrator
# ---------------------------------------------------------------------------

class SessionRunner:
    """Drives one agent session <-> one messenger conversation."""

    RetryCallback = Callable[["SessionRunner", str], Coroutine[Any, Any, None]]
    SuccessCallback = Callable[[str, str], None]

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
        on_provider_error: Optional["SessionRunner.RetryCallback"] = None,
        on_provider_success: Optional["SessionRunner.SuccessCallback"] = None,
    ):
        self.session_id = session_id
        self.chat_id = chat_id
        self.thread_id = thread_id
        self.agent = agent
        self.messenger = messenger
        self.directory = directory
        self.auto_approve = auto_approve
        self._long_content = long_content_handler
        self._on_provider_error = on_provider_error
        self._on_provider_success = on_provider_success

        self.state: str = "warmup"
        self.last_active: float = time.monotonic()
        self.sse_task: Optional[asyncio.Task] = None

        self._ui = CompositeUI(messenger, chat_id, thread_id)
        self._delivery = FileDelivery(messenger, chat_id, thread_id, directory)

        self.pending_permissions: dict[str, str] = {}
        self.pending_questions: dict[str, QuestionRequest] = {}

        self._last_parts: list[MessagePart] = []
        self._last_model: Optional[ModelRef] = None

    # -- compatibility shims (used by tests and manager) --------------------

    @property
    def _msg_id(self) -> Optional[str]:
        return self._ui.msg_id

    @_msg_id.setter
    def _msg_id(self, value: Optional[str]) -> None:
        self._ui.msg_id = value

    @property
    def _accumulated_text(self) -> str:
        return self._ui._accumulated_text

    @_accumulated_text.setter
    def _accumulated_text(self, value: str) -> None:
        self._ui._accumulated_text = value

    @property
    def _response_text(self) -> str:
        return self._ui.response_text

    @_response_text.setter
    def _response_text(self, value: str) -> None:
        self._ui.response_text = value

    @property
    def _tool_lines(self) -> dict[str, str]:
        return self._ui.tool_lines

    @property
    def _tool_order(self) -> list[str]:
        return self._ui.tool_order

    @property
    def typing_task(self) -> Optional[asyncio.Task]:
        return self._ui.typing_task

    # -- public properties --------------------------------------------------

    @property
    def last_model(self) -> Optional[ModelRef]:
        return self._last_model

    @property
    def current_msg_id(self) -> Optional[str]:
        return self._ui.msg_id

    @property
    def in_reasoning(self) -> bool:
        return self._ui.in_reasoning

    @property
    def reasoning_start(self) -> float:
        return self._ui.reasoning_start

    async def retry_with_model(self, model: ModelRef, status_text: str) -> None:
        """Re-send the last prompt with a different model, updating the UI."""
        if self._ui.msg_id:
            await self.messenger.edit_message(self.chat_id, self._ui.msg_id, status_text)
        await self.wake_and_prompt(self._last_parts, model)

    async def show_error(self, text: str) -> None:
        """Display an error in the current message and set state to error."""
        if self._ui.msg_id:
            await self.messenger.edit_message(self.chat_id, self._ui.msg_id, text)
        self.state = "error"

    def stop_typing(self) -> None:
        self._ui.stop_typing()

    async def _try_deliver_file(self, title: str) -> None:
        """Delegate to FileDelivery (compat shim)."""
        await self._delivery.try_deliver(title)

    async def _finalize_response(self) -> None:
        """Delegate to CompositeUI.finalize (compat shim)."""
        await self._ui.finalize(self._long_content)
        self.state = "idle"

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
        self._ui.reset()
        self.pending_permissions.clear()
        self.pending_questions.clear()

        placeholder = "\U0001f504 Reconnecting\u2026" if reconnecting else "\u2699\ufe0f"
        self._ui.msg_id = await self.messenger.send_message(self.chat_id, placeholder, self.thread_id)
        self._ui.start_typing()

        preview = "; ".join(p.text[:60] if hasattr(p, "text") else str(p)[:60] for p in parts)
        log.info("prompt -> session %s: %s", self.session_id, preview)

        self._last_parts = parts
        self._last_model = model

        try:
            await self.agent.send_prompt(self.session_id, parts, model, directory=self.directory)
        except Exception as exc:
            log.error("send_prompt failed: %s", exc)
            self._ui.stop_typing()
            await self.close_sse()
            if self._ui.msg_id:
                await self.messenger.edit_message(self.chat_id, self._ui.msg_id, f"\u274c {exc}")
            self.state = "error"

    async def close_sse(self) -> None:
        self._ui.stop_typing()
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
            try:
                await self.sse_task
            except (asyncio.CancelledError, Exception):
                pass
        self.sse_task = asyncio.create_task(self._sse_loop())

    async def shutdown(self) -> None:
        self._ui.stop_typing()
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
        if not self._ui.in_reasoning:
            self._ui.reasoning_start = time.monotonic()
        self._ui.in_reasoning = True
        self._ui.reasoning_text += event.text
        self._ui.rebuild()
        await self._ui.throttled_edit()

    async def _on_text(self, event: TextDelta) -> None:
        if self._ui.in_reasoning:
            self._ui.in_reasoning = False
        self._ui.response_text += event.text
        self._ui.rebuild()
        await self._ui.throttled_edit()

    async def _on_tool_start(self, event: ToolStart) -> None:
        cid = event.call_id or f"_anon_{len(self._ui.tool_order)}"
        self._ui.tool_lines[cid] = f"\U0001f527 <b>{html_escape(event.name)}</b>\u2026"
        if cid not in self._ui.tool_order:
            self._ui.tool_order.append(cid)
        self._ui.rebuild()
        await self._ui.throttled_edit()

    async def _on_tool_end(self, event: ToolEnd) -> None:
        cid = event.call_id or ""
        if cid not in self._ui.tool_lines:
            for key in reversed(self._ui.tool_order):
                if key.startswith("_anon_") and self._ui.tool_lines.get(key, "").endswith("\u2026"):
                    cid = key
                    break
        name_esc = html_escape(event.name)
        if event.state == "completed":
            label = f"\u2705 <b>{name_esc}</b>"
            if event.title:
                label += f" \u2014 {html_escape(event.title[:_TITLE_TRUNCATE])}"
        else:
            label = f"\u274c <b>{name_esc}</b>"
            if event.error:
                label += f" \u2014 {html_escape(event.error[:_TITLE_TRUNCATE])}"
        self._ui.tool_lines[cid] = label
        self._ui.rebuild()
        await self._ui.throttled_edit()
        if event.state == "completed" and event.name in ("write", "save"):
            await self._delivery.try_deliver(event.title)

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
            log.info("SessionIdle for %s (accumulated %d chars)", self.session_id, len(self._ui._accumulated_text))
            self.pending_permissions.clear()
            if self._on_provider_success and self._last_model:
                self._on_provider_success(
                    self._last_model.provider_id, self._last_model.model_id,
                )
            await self._ui.finalize(self._long_content)
            self.state = "idle"

    async def _on_error(self, event: SessionError) -> None:
        if config.is_provider_error(event.error) and self._on_provider_error:
            log.warning("provider error detected: %s -- triggering fallback", event.error[:120])
            await self._on_provider_error(self, event.error)
            return
        self._ui.stop_typing()
        if self._ui.msg_id:
            max_len = self.messenger.max_message_length
            await self.messenger.edit_message(
                self.chat_id, self._ui.msg_id,
                f"\u274c {event.error}"[:max_len],
            )
        self.state = "error"

    async def _on_status(self, event: StatusUpdate) -> None:
        if event.status == "idle" and self.state != "idle":
            await self._ui.finalize(self._long_content)
            self.state = "idle"
        elif event.status == "busy":
            self.state = "generating"
        elif event.status == "retry":
            log.info("session %s: %s", self.session_id, event.message)
