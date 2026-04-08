"""Tests for protocol abstractions and OpenCodeBackend event conversion."""

from __future__ import annotations

import asyncio
import json
import unittest

from aiohttp import web
from aiohttp.test_utils import AioHTTPTestCase

from opencode_tg.agents.opencode import OpenCodeBackend
from opencode_tg.protocols import (
    Attachment,
    IncomingMessage,
    MessagePart,
    ModelRef,
    PermissionRequest,
    SessionError,
    SessionIdle,
    StatusUpdate,
    TextDelta,
    ToolEnd,
    ToolStart,
)


class TestMessagePart(unittest.TestCase):
    def test_to_dict_strips_none(self):
        p = MessagePart(type="text", text="hello")
        d = p.to_dict()
        self.assertEqual(d, {"type": "text", "text": "hello"})
        self.assertNotIn("mime", d)

    def test_file_part(self):
        p = MessagePart(type="file", mime="image/png", filename="x.png", url="file:///tmp/x.png")
        d = p.to_dict()
        self.assertEqual(d["type"], "file")
        self.assertEqual(d["url"], "file:///tmp/x.png")


class TestModelRef(unittest.TestCase):
    def test_to_dict(self):
        m = ModelRef(provider_id="openai", model_id="gpt-4o")
        self.assertEqual(m.to_dict(), {"providerID": "openai", "modelID": "gpt-4o"})


class TestEventConversion(unittest.TestCase):
    """Test OpenCodeBackend._convert for all event types."""

    def _convert(self, raw, session_id="s1"):
        return OpenCodeBackend._convert(raw, session_id)

    def test_text_delta(self):
        raw = {"type": "message.part.delta", "properties": {"field": "text", "delta": "hello"}}
        events = self._convert(raw)
        self.assertEqual(len(events), 1)
        self.assertIsInstance(events[0], TextDelta)
        self.assertEqual(events[0].text, "hello")

    def test_text_delta_empty(self):
        raw = {"type": "message.part.delta", "properties": {"field": "text", "delta": ""}}
        self.assertEqual(len(self._convert(raw)), 0)

    def test_tool_start(self):
        raw = {
            "type": "message.part.updated",
            "properties": {
                "type": "tool-invocation", "state": "running",
                "toolName": "bash", "callID": "c1",
            },
        }
        events = self._convert(raw)
        self.assertEqual(len(events), 1)
        self.assertIsInstance(events[0], ToolStart)
        self.assertEqual(events[0].name, "bash")
        self.assertEqual(events[0].call_id, "c1")

    def test_tool_completed(self):
        raw = {
            "type": "message.part.updated",
            "properties": {
                "type": "tool-invocation", "state": "completed",
                "toolName": "bash", "callID": "c1", "title": "done",
            },
        }
        events = self._convert(raw)
        self.assertIsInstance(events[0], ToolEnd)
        self.assertEqual(events[0].state, "completed")
        self.assertEqual(events[0].title, "done")

    def test_tool_error(self):
        raw = {
            "type": "message.part.updated",
            "properties": {
                "type": "tool-invocation", "state": "error",
                "toolName": "bash", "callID": "c1", "error": "fail",
            },
        }
        events = self._convert(raw)
        self.assertIsInstance(events[0], ToolEnd)
        self.assertEqual(events[0].state, "error")
        self.assertEqual(events[0].error, "fail")

    def test_non_tool_part_ignored(self):
        raw = {
            "type": "message.part.updated",
            "properties": {"type": "text", "state": "running"},
        }
        self.assertEqual(len(self._convert(raw)), 0)

    def test_session_idle(self):
        raw = {"type": "session.idle", "properties": {"sessionID": "s1"}}
        events = self._convert(raw)
        self.assertEqual(len(events), 1)
        self.assertIsInstance(events[0], SessionIdle)

    def test_session_status_idle(self):
        raw = {"type": "session.status", "properties": {"status": {"type": "idle"}}}
        events = self._convert(raw)
        self.assertIsInstance(events[0], SessionIdle)

    def test_session_status_busy(self):
        raw = {"type": "session.status", "properties": {"status": {"type": "busy"}}}
        events = self._convert(raw)
        self.assertIsInstance(events[0], StatusUpdate)
        self.assertEqual(events[0].status, "busy")

    def test_session_status_retry(self):
        raw = {
            "type": "session.status",
            "properties": {"status": {"type": "retry", "attempt": 2, "message": "rate limit"}},
        }
        events = self._convert(raw)
        self.assertIsInstance(events[0], StatusUpdate)
        self.assertEqual(events[0].status, "retry")
        self.assertIn("retry #2", events[0].message)

    def test_permission_updated(self):
        raw = {
            "type": "permission.updated",
            "properties": {"id": "p1", "title": "Run bash", "pattern": "*.sh"},
        }
        events = self._convert(raw, session_id="s1")
        self.assertIsInstance(events[0], PermissionRequest)
        self.assertEqual(events[0].info.id, "p1")
        self.assertEqual(events[0].info.session_id, "s1")
        self.assertEqual(events[0].info.title, "Run bash")

    def test_session_error(self):
        raw = {"type": "session.error", "properties": {"error": "boom"}}
        events = self._convert(raw)
        self.assertIsInstance(events[0], SessionError)
        self.assertEqual(events[0].error, "boom")

    def test_unknown_type_ignored(self):
        raw = {"type": "server.heartbeat", "properties": {}}
        self.assertEqual(len(self._convert(raw)), 0)


# ---------------------------------------------------------------------------
# OpenCodeBackend integration test against mock server
# ---------------------------------------------------------------------------

def _build_app() -> web.Application:
    app = web.Application()

    async def create_session(request: web.Request) -> web.Response:
        body = await request.json()
        return web.json_response({"id": "ses_1"})

    async def get_session(request: web.Request) -> web.Response:
        return web.json_response({"id": request.match_info["sid"]})

    async def delete_session(request: web.Request) -> web.Response:
        return web.Response(status=200)

    async def prompt_async(request: web.Request) -> web.Response:
        await request.json()
        return web.Response(status=204)

    async def respond_permission(request: web.Request) -> web.Response:
        await request.json()
        return web.json_response(True)

    async def list_providers(request: web.Request) -> web.Response:
        return web.json_response([
            {"id": "openai", "models": {"gpt-4o": {"name": "GPT-4o"}}},
            {"id": "anthropic", "models": [{"id": "claude-sonnet-4-20250514", "name": "Claude 4 Sonnet"}]},
        ])

    async def sse_events(request: web.Request) -> web.StreamResponse:
        resp = web.StreamResponse(
            status=200,
            headers={"Content-Type": "text/event-stream"},
        )
        await resp.prepare(request)
        events = [
            {"type": "message.part.delta", "properties": {"sessionID": "ses_1", "field": "text", "delta": "hi"}},
            {"type": "message.part.updated", "properties": {
                "sessionID": "ses_1", "type": "tool-invocation",
                "toolName": "bash", "callID": "c1", "state": "running",
            }},
            {"type": "message.part.updated", "properties": {
                "sessionID": "ses_1", "type": "tool-invocation",
                "toolName": "bash", "callID": "c1", "state": "completed", "title": "done",
            }},
            {"type": "session.idle", "properties": {"sessionID": "ses_1"}},
        ]
        for ev in events:
            await resp.write(f"data: {json.dumps(ev)}\n\n".encode())
        return resp

    app.router.add_post("/session", create_session)
    app.router.add_get("/session/{sid}", get_session)
    app.router.add_delete("/session/{sid}", delete_session)
    app.router.add_post("/session/{sid}/prompt_async", prompt_async)
    app.router.add_post("/session/{sid}/permissions/{pid}", respond_permission)
    app.router.add_get("/provider", list_providers)
    app.router.add_get("/event", sse_events)
    return app


class TestOpenCodeBackend(AioHTTPTestCase):
    async def get_application(self) -> web.Application:
        return _build_app()

    def _make_backend(self) -> OpenCodeBackend:
        base = f"http://127.0.0.1:{self.server.port}"
        backend = OpenCodeBackend(base_url=base, directory="/test")
        backend._oc._session = self.client.session
        return backend

    async def test_create_session(self):
        b = self._make_backend()
        sid = await b.create_session("test")
        self.assertEqual(sid, "ses_1")

    async def test_send_prompt(self):
        b = self._make_backend()
        await b.send_prompt("ses_1", [MessagePart(type="text", text="hi")])

    async def test_send_prompt_with_model(self):
        b = self._make_backend()
        await b.send_prompt(
            "ses_1",
            [MessagePart(type="text", text="hi")],
            model=ModelRef(provider_id="openai", model_id="gpt-4o"),
        )

    async def test_list_models(self):
        b = self._make_backend()
        models = await b.list_models()
        self.assertTrue(len(models) >= 2)
        ids = [(m.provider_id, m.model_id) for m in models]
        self.assertIn(("openai", "gpt-4o"), ids)

    async def test_subscribe_events_typed(self):
        b = self._make_backend()
        events = []
        async for ev in b.subscribe_events("ses_1"):
            events.append(ev)
            if isinstance(ev, SessionIdle):
                break
        types = [type(e).__name__ for e in events]
        self.assertIn("TextDelta", types)
        self.assertIn("ToolStart", types)
        self.assertIn("ToolEnd", types)
        self.assertIn("SessionIdle", types)

    async def test_respond_permission(self):
        b = self._make_backend()
        await b.respond_permission("ses_1", "p1", "once")


if __name__ == "__main__":
    unittest.main()
