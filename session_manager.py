"""Session manager — maps messenger chat/thread pairs to SessionRunners.

Works with any Messenger + AgentBackend via protocol abstractions.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Callable, Coroutine, Any, Optional

from . import config
from .protocols import AgentBackend, MessagePart, Messenger, ModelRef
from .runners import SessionRunner, LongContentHandler
from .sessions import SessionStore

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

    # -- key helper ---------------------------------------------------------

    @staticmethod
    def _key(chat_id: str, thread_id: Optional[str]) -> str:
        return f"{chat_id}:{thread_id}" if thread_id else chat_id

    def _get_lock(self, key: str) -> asyncio.Lock:
        lock = self._locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[key] = lock
        return lock

    # -- message dispatch ---------------------------------------------------

    async def handle_message(
        self,
        chat_id: str,
        thread_id: Optional[str],
        parts: list[MessagePart],
        model: Optional[ModelRef] = None,
    ) -> None:
        key = self._key(chat_id, thread_id)
        async with self._get_lock(key):
            session_id = self._store.get(chat_id, thread_id)

            if not session_id:
                title = f"{self._messenger.name}:{chat_id}"
                if thread_id:
                    title += f"/{thread_id}"
                session_id = await self._agent.create_session(title)
                self._store.set(chat_id, thread_id, session_id)
                log.info("new session %s for key %s", session_id, key)

            runner = self._runners.get(key)
            if runner is None:
                runner = SessionRunner(
                    session_id, chat_id, thread_id,
                    self._agent, self._messenger,
                    long_content_handler=self._long_content,
                )
                self._runners[key] = runner
                log.info("created runner for key %s", key)

            effective_model = model or self._store.get_model(chat_id, thread_id)
            await runner.wake_and_prompt(parts, effective_model)

    # -- session control ----------------------------------------------------

    async def reset_session(self, chat_id: str, thread_id: Optional[str]) -> None:
        key = self._key(chat_id, thread_id)
        async with self._get_lock(key):
            runner = self._runners.pop(key, None)
            if runner:
                await runner.shutdown()
            old_sid = self._store.get(chat_id, thread_id)
            if old_sid:
                try:
                    await self._agent.delete_session(old_sid)
                except Exception as exc:
                    log.warning("delete session failed: %s", exc)
            self._store.delete(chat_id, thread_id)
            self._store.delete_model(chat_id, thread_id)
            log.info("reset session for key %s", key)

    def get_session_id(self, chat_id: str, thread_id: Optional[str]) -> Optional[str]:
        return self._store.get(chat_id, thread_id)

    def set_model(self, chat_id: str, thread_id: Optional[str], provider_id: str, model_id: str) -> None:
        self._store.set_model(chat_id, thread_id, provider_id, model_id)

    def get_model(self, chat_id: str, thread_id: Optional[str]) -> Optional[ModelRef]:
        return self._store.get_model(chat_id, thread_id)

    # -- startup / reconnect ------------------------------------------------

    async def startup_reconnect(self) -> None:
        entries = self._store.all_entries()
        log.info("startup: checking %d stored sessions", len(entries))
        for key, session_id in entries.items():
            info = await self._agent.get_session(session_id)
            if info is None:
                log.info("startup: session %s gone (404), removing", session_id)
                chat_id, thread_id = self._parse_key(key)
                self._store.delete(chat_id, thread_id)
                continue
            chat_id, thread_id = self._parse_key(key)
            runner = SessionRunner(
                session_id, chat_id, thread_id,
                self._agent, self._messenger,
                long_content_handler=self._long_content,
            )
            runner.state = "sleeping"
            self._runners[key] = runner
            log.info("startup: restored runner for %s (session %s)", key, session_id)

    @staticmethod
    def _parse_key(key: str) -> tuple[str, Optional[str]]:
        if ":" in key:
            chat_id, thread_id = key.split(":", 1)
            return chat_id, thread_id if thread_id != "None" else None
        return key, None

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

    # -- shutdown -----------------------------------------------------------

    async def shutdown_all(self) -> None:
        for runner in self._runners.values():
            await runner.shutdown()
        self._runners.clear()
