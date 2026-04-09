"""Tests for SSE stall detection, reconnect state reset, session poll, and watchdog."""
from __future__ import annotations

import asyncio
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from opencode_tg.oc_client import OcClient, SseStallError
from opencode_tg.agents.opencode import OpenCodeBackend
from opencode_tg.protocols import ReasoningDelta, TextDelta
from opencode_tg.session_manager import SessionManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FakeContent:
    """Simulates ``resp.content.iter_any()`` with pre-loaded byte chunks."""

    def __init__(self, chunks: list[bytes]):
        self._chunks = chunks

    async def iter_any(self):
        for c in self._chunks:
            yield c


class FakeResponse:
    def __init__(self, status: int, content: FakeContent):
        self.status = status
        self.content = content
        self.request_info = MagicMock()
        self.history = ()

    async def text(self):
        return "error"

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


# ---------------------------------------------------------------------------
# test_sse_stall: keepalives arrive but no real events → SseStallError
# ---------------------------------------------------------------------------

class TestSseStall(unittest.IsolatedAsyncioTestCase):

    async def test_raises_on_keepalives_only(self):
        chunks = [
            b"data: {\"type\": \"session.status\", \"properties\": {\"sessionID\": \"s1\"}}\n",
            b":\n",
            b":\n",
            b":\n",
        ]
        fake_resp = FakeResponse(200, FakeContent(chunks))

        client = OcClient.__new__(OcClient)
        client.base_url = "http://fake"
        client.directory = "/fake"
        client._session = MagicMock()

        sess_mock = MagicMock()
        sess_mock.get.return_value = fake_resp

        with patch.object(client, "_get_session", new_callable=AsyncMock, return_value=sess_mock), \
             patch("opencode_tg.oc_client.config") as cfg:
            cfg.STALL_TIMEOUT_SECONDS = 0.0

            events: list[dict] = []
            with self.assertRaises(SseStallError):
                async for ev in client._sse_stream("s1"):
                    events.append(ev)

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["type"], "session.status")

    async def test_no_stall_when_real_events_arrive(self):
        chunks = [
            b"data: {\"type\": \"a\", \"properties\": {\"sessionID\": \"s1\"}}\n",
            b":\n",
            b"data: {\"type\": \"b\", \"properties\": {\"sessionID\": \"s1\"}}\n",
        ]
        fake_resp = FakeResponse(200, FakeContent(chunks))

        client = OcClient.__new__(OcClient)
        client.base_url = "http://fake"
        client.directory = "/fake"

        sess_mock = MagicMock()
        sess_mock.get.return_value = fake_resp

        with patch.object(client, "_get_session", new_callable=AsyncMock, return_value=sess_mock), \
             patch("opencode_tg.oc_client.config") as cfg:
            cfg.STALL_TIMEOUT_SECONDS = 999.0

            events = [ev async for ev in client._sse_stream("s1")]

        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]["type"], "a")
        self.assertEqual(events[1]["type"], "b")


# ---------------------------------------------------------------------------
# test_reconnect_resets_state: __sse_reconnected resets current_part
# ---------------------------------------------------------------------------

class TestReconnectResetsState(unittest.IsolatedAsyncioTestCase):

    async def test_current_part_resets_on_reconnect_event(self):
        raw_events = [
            {"type": "message.part.updated", "properties": {"part": {"type": "reasoning"}}},
            {"type": "message.part.delta", "properties": {"delta": "thinking"}},
            {"type": "__sse_reconnected", "properties": {"sessionID": "s1"}},
            {"type": "message.part.delta", "properties": {"delta": "answer"}},
        ]

        async def fake_subscribe(session_id, **kwargs):
            for ev in raw_events:
                yield ev

        backend = OpenCodeBackend.__new__(OpenCodeBackend)
        backend._oc = MagicMock()
        backend._oc.subscribe_events = fake_subscribe

        results = []
        async for ev in backend.subscribe_events("s1"):
            results.append(ev)

        self.assertEqual(len(results), 2)
        self.assertIsInstance(results[0], ReasoningDelta)
        self.assertEqual(results[0].text, "thinking")
        self.assertIsInstance(results[1], TextDelta)
        self.assertEqual(results[1].text, "answer")


# ---------------------------------------------------------------------------
# test_reconnect_polls_session: synthetic session.idle emitted
# ---------------------------------------------------------------------------

class TestReconnectPollsSession(unittest.IsolatedAsyncioTestCase):

    async def test_emits_synthetic_idle_when_session_is_idle(self):
        call_count = 0

        async def fake_sse_stream(session_id, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                yield {"type": "message.part.delta", "properties": {"delta": "hi", "sessionID": session_id}}
                raise SseStallError("stall")
            else:
                return

        client = OcClient.__new__(OcClient)
        client.base_url = "http://fake"
        client.directory = "/fake"
        client._session = None

        async def fake_get_session(sid, **kwargs):
            return {"id": sid, "status": "idle"}

        with patch.object(client, "_sse_stream", fake_sse_stream), \
             patch.object(client, "get_session", fake_get_session), \
             patch("opencode_tg.oc_client.config") as cfg:
            cfg.RECONNECT_BACKOFF_MAX = 0.01

            collected = []
            async for ev in client.subscribe_events("s1"):
                collected.append(ev)
                if len(collected) >= 4:
                    break

        types = [e["type"] for e in collected]
        self.assertIn("message.part.delta", types)
        self.assertIn("__sse_reconnected", types)
        self.assertIn("session.idle", types)


# ---------------------------------------------------------------------------
# test_stalled_generating_watchdog: reconnects runner stuck generating
# ---------------------------------------------------------------------------

class TestStalledGeneratingWatchdog(unittest.IsolatedAsyncioTestCase):

    async def test_reconnects_stalled_runner(self):
        runner = MagicMock()
        runner.state = "generating"
        runner.last_active = time.monotonic() - 600
        runner._in_reasoning = False
        runner._reasoning_start = 0.0
        runner.reconnect = AsyncMock()

        manager = SessionManager.__new__(SessionManager)
        manager._runners = {"chat:1": runner}

        with patch("opencode_tg.session_manager.config") as cfg:
            cfg.GENERATING_STALL_SECONDS = 300
            cfg.REASONING_STALL_SECONDS = 120
            await manager.check_stalled_generating()

        runner.reconnect.assert_awaited_once()

    async def test_skips_recent_runner(self):
        runner = MagicMock()
        runner.state = "generating"
        runner.last_active = time.monotonic()
        runner._in_reasoning = False
        runner._reasoning_start = 0.0
        runner.reconnect = AsyncMock()

        manager = SessionManager.__new__(SessionManager)
        manager._runners = {"chat:1": runner}

        with patch("opencode_tg.session_manager.config") as cfg:
            cfg.GENERATING_STALL_SECONDS = 300
            cfg.REASONING_STALL_SECONDS = 120
            await manager.check_stalled_generating()

        runner.reconnect.assert_not_awaited()

    async def test_skips_idle_runner(self):
        runner = MagicMock()
        runner.state = "idle"
        runner.last_active = time.monotonic() - 600
        runner.reconnect = AsyncMock()

        manager = SessionManager.__new__(SessionManager)
        manager._runners = {"chat:1": runner}

        with patch("opencode_tg.session_manager.config") as cfg:
            cfg.GENERATING_STALL_SECONDS = 300
            await manager.check_stalled_generating()

        runner.reconnect.assert_not_awaited()
