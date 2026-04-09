"""Session manager — maps messenger chat/thread pairs to SessionRunners.

Works with any Messenger + AgentBackend via protocol abstractions.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Callable, Coroutine, Any, Optional

from . import config
from .protocols import AgentBackend, ChatKey, MessagePart, Messenger, ModelRef
from .runners import SessionRunner, LongContentHandler
from .sessions import SessionStore

_PROVIDER_FAIL_TTL = 300.0

log = logging.getLogger(__name__)


class SessionManager:
    """Manages the lifecycle of all active SessionRunners."""

    def __init__(
        self,
        store: SessionStore,
        agent: AgentBackend,
        messenger: Messenger,
        *,
        long_content_handler: Optional[LongContentHandler] = None,
    ):
        self._store = store
        self._agent = agent
        self._messenger = messenger
        self._long_content = long_content_handler
        self._runners: dict[str, SessionRunner] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._failed_providers: dict[str, float] = {}

    # -- key helper ---------------------------------------------------------

    @staticmethod
    def _key(chat_id: str, thread_id: Optional[str]) -> str:
        return str(ChatKey(chat_id, thread_id))

    def _get_lock(self, key: str) -> asyncio.Lock:
        lock = self._locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[key] = lock
        return lock

    # -- message dispatch ---------------------------------------------------

    def _get_directory(self, chat_id: str, thread_id: Optional[str]) -> Optional[str]:
        return self._store.get_directory(chat_id, thread_id)

    def _get_auto_approve(self, chat_id: str, thread_id: Optional[str]) -> bool:
        return self._store.get_auto_approve(chat_id, thread_id)

    async def handle_message(
        self,
        chat_id: str,
        thread_id: Optional[str],
        parts: list[MessagePart],
        model: Optional[ModelRef] = None,
    ) -> None:
        key = self._key(chat_id, thread_id)
        directory = self._get_directory(chat_id, thread_id)
        auto_approve = self._get_auto_approve(chat_id, thread_id)
        async with self._get_lock(key):
            session_id = self._store.get(chat_id, thread_id)

            if not session_id:
                title = f"{self._messenger.name}:{chat_id}"
                if thread_id:
                    title += f"/{thread_id}"
                session_id = await self._agent.create_session(title, directory=directory)
                self._store.set(chat_id, thread_id, session_id)
                log.info("new session %s for key %s", session_id, key)

            runner = self._runners.get(key)
            if runner is None:
                runner = SessionRunner(
                    session_id, chat_id, thread_id,
                    self._agent, self._messenger,
                    directory=directory,
                    auto_approve=auto_approve,
                    long_content_handler=self._long_content,
                )
                runner._retry_callback = self._handle_provider_fallback
                runner._success_callback = self.clear_provider_failure
                self._runners[key] = runner
                log.info("created runner for key %s (dir=%s)", key, directory or "(default)")
            else:
                runner.directory = directory
                runner.auto_approve = auto_approve

            effective_model = model or self._store.get_model(chat_id, thread_id)
            await runner.wake_and_prompt(parts, effective_model)

    # -- session control ----------------------------------------------------

    async def reset_session(self, chat_id: str, thread_id: Optional[str]) -> None:
        key = self._key(chat_id, thread_id)
        directory = self._get_directory(chat_id, thread_id)
        async with self._get_lock(key):
            runner = self._runners.pop(key, None)
            if runner:
                await runner.shutdown()
            old_sid = self._store.get(chat_id, thread_id)
            if old_sid:
                try:
                    await self._agent.delete_session(old_sid, directory=directory)
                except Exception as exc:
                    log.warning("delete session failed: %s", exc)
            self._store.clear_session(chat_id, thread_id)
            log.info("reset session for key %s", key)

    def get_session_id(self, chat_id: str, thread_id: Optional[str]) -> Optional[str]:
        return self._store.get(chat_id, thread_id)

    def set_model(self, chat_id: str, thread_id: Optional[str], provider_id: str, model_id: str) -> None:
        self._store.set_model(chat_id, thread_id, provider_id, model_id)

    def get_model(self, chat_id: str, thread_id: Optional[str]) -> Optional[ModelRef]:
        return self._store.get_model(chat_id, thread_id)

    def set_directory(self, chat_id: str, thread_id: Optional[str], directory: str) -> None:
        self._store.set_directory(chat_id, thread_id, directory)
        key = self._key(chat_id, thread_id)
        runner = self._runners.get(key)
        if runner:
            runner.directory = directory

    def get_directory(self, chat_id: str, thread_id: Optional[str]) -> Optional[str]:
        return self._store.get_directory(chat_id, thread_id)

    def set_auto_approve(self, chat_id: str, thread_id: Optional[str], value: bool) -> None:
        self._store.set_auto_approve(chat_id, thread_id, value)
        key = self._key(chat_id, thread_id)
        runner = self._runners.get(key)
        if runner:
            runner.auto_approve = value

    def get_auto_approve(self, chat_id: str, thread_id: Optional[str]) -> bool:
        return self._store.get_auto_approve(chat_id, thread_id)

    async def abort_session(self, chat_id: str, thread_id: Optional[str]) -> None:
        """Abort the running session generation."""
        key = self._key(chat_id, thread_id)
        runner = self._runners.get(key)
        session_id = self._store.get(chat_id, thread_id)
        if session_id:
            directory = self._get_directory(chat_id, thread_id)
            await self._agent.abort_session(session_id, directory=directory)
        if runner:
            runner.state = "idle"
            runner._stop_typing()

    def find_session_for_perm(self, perm_id: str) -> Optional[str]:
        """Look up session_id by perm_id across all runners."""
        for runner in self._runners.values():
            if perm_id in runner.pending_permissions:
                return runner.session_id
        return None

    def find_runner_for_question(self, request_id: str) -> Optional[SessionRunner]:
        """Look up the runner that has a pending question with *request_id*."""
        for runner in self._runners.values():
            if request_id in runner.pending_questions:
                return runner
        return None

    def find_directory_for_session(self, session_id: str) -> Optional[str]:
        """Return the working directory associated with *session_id*."""
        ck = self._store.find_by_session_id(session_id)
        if ck:
            return self.get_directory(ck.chat_id, ck.thread_id)
        return None

    # -- provider fallback ---------------------------------------------------

    async def _handle_provider_fallback(self, runner: SessionRunner, error: str) -> None:
        current = runner._last_model
        current_key = f"{current.provider_id}/{current.model_id}" if current else ""
        self._failed_providers[current_key] = time.monotonic()
        log.warning("provider %s failed: %s", current_key or "(default)", error[:120])

        fallback = self._pick_fallback(current_key, self._failed_providers)
        if not fallback:
            log.error("no fallback provider available (tried: %s)", list(self._failed_providers))
            if runner._msg_id:
                await self._messenger.edit_message(
                    runner.chat_id, runner._msg_id,
                    f"\u274c All providers exhausted. Last error: {error[:200]}",
                )
            runner.state = "error"
            return

        fb_model = ModelRef(provider_id=fallback[0], model_id=fallback[1])
        log.info("falling back to %s/%s", fallback[0], fallback[1])
        if runner._msg_id:
            await self._messenger.edit_message(
                runner.chat_id, runner._msg_id,
                f"\u26a0\ufe0f Switching to {fallback[0]}/{fallback[1]}\u2026",
            )
        await runner.wake_and_prompt(runner._last_parts, fb_model)

    @staticmethod
    def _pick_fallback(failed_key: str, failed_providers: dict[str, float]) -> Optional[tuple[str, str]]:
        now = time.monotonic()
        for p, m in config.FALLBACK_MODELS:
            key = f"{p}/{m}"
            if key == failed_key:
                continue
            fail_time = failed_providers.get(key)
            if fail_time is not None and now - fail_time < _PROVIDER_FAIL_TTL:
                continue
            return (p, m)
        return None

    def clear_provider_failure(self, provider_id: str, model_id: str) -> None:
        key = f"{provider_id}/{model_id}"
        self._failed_providers.pop(key, None)

    # -- startup / reconnect ------------------------------------------------

    async def startup_reconnect(self) -> None:
        entries = self._store.all_entries()
        log.info("startup: checking %d stored sessions", len(entries))
        for key, session_id in entries.items():
            chat_id, thread_id = self._parse_key(key)
            directory = self._get_directory(chat_id, thread_id)
            info = await self._agent.get_session(session_id, directory=directory)
            if info is None:
                log.info("startup: session %s gone (404), removing", session_id)
                self._store.delete(chat_id, thread_id)
                continue
            auto_approve = self._get_auto_approve(chat_id, thread_id)
            runner = SessionRunner(
                session_id, chat_id, thread_id,
                self._agent, self._messenger,
                directory=directory,
                auto_approve=auto_approve,
                long_content_handler=self._long_content,
            )
            runner.state = "sleeping"
            self._runners[key] = runner
            log.info("startup: restored runner for %s (session %s, dir=%s)", key, session_id, directory or "(default)")

    @staticmethod
    def _parse_key(key: str) -> tuple[str, Optional[str]]:
        ck = ChatKey.parse(key)
        return ck.chat_id, ck.thread_id

    # -- idle sweep ---------------------------------------------------------

    async def idle_sweep(self) -> None:
        now = time.monotonic()
        for key, runner in list(self._runners.items()):
            if runner.state in ("idle", "generating"):
                elapsed = now - runner.last_active
                if elapsed > config.IDLE_TIMEOUT_SECONDS:
                    log.info("idle sweep: closing SSE for %s (idle %.0fs)", key, elapsed)
                    await runner.close_sse()

    async def check_dead_sse(self) -> None:
        for key, runner in list(self._runners.items()):
            if runner.state in ("idle", "generating"):
                if runner.sse_task and runner.sse_task.done():
                    log.info("dead SSE detected for %s, reconnecting", key)
                    await runner.reconnect()

    async def check_stalled_generating(self) -> None:
        """Force-reconnect runners stuck in ``generating`` too long.

        Also detects reasoning-only stalls — continuous reasoning events
        without any text or tool output.
        """
        now = time.monotonic()
        for key, runner in list(self._runners.items()):
            if runner.state == "generating":
                stall = now - runner.last_active
                if stall > config.GENERATING_STALL_SECONDS:
                    log.warning(
                        "generating stall for %s (%.0fs), reconnecting SSE",
                        key, stall,
                    )
                    await runner.reconnect()
                elif (
                    runner._in_reasoning
                    and runner._reasoning_start > 0
                    and now - runner._reasoning_start > config.REASONING_STALL_SECONDS
                ):
                    log.warning(
                        "reasoning stall for %s (%.0fs), reconnecting SSE",
                        key, now - runner._reasoning_start,
                    )
                    await runner.reconnect()

    # -- shutdown -----------------------------------------------------------

    async def shutdown_all(self) -> None:
        for runner in self._runners.values():
            await runner.shutdown()
        self._runners.clear()
