"""Unit tests for oc_client — mock HTTP server via aiohttp.test_utils."""

from __future__ import annotations

import asyncio
import json
import unittest

from aiohttp import web
from aiohttp.test_utils import AioHTTPTestCase

from opencode_tg.oc_client import OcClient, OcClientError, parse_sse_line


# ---------------------------------------------------------------------------
# SSE line parser
# ---------------------------------------------------------------------------


class TestParseSseLine(unittest.TestCase):
    def test_valid_json(self):
        line = 'data: {"type":"session.idle","properties":{"sessionID":"s1"}}'
        result = parse_sse_line(line)
        self.assertEqual(result["type"], "session.idle")

    def test_not_data_prefix(self):
        self.assertIsNone(parse_sse_line("event: message"))

    def test_invalid_json(self):
        self.assertIsNone(parse_sse_line("data: {broken"))

    def test_empty_line(self):
        self.assertIsNone(parse_sse_line(""))


# ---------------------------------------------------------------------------
# Mock OpenCode server
# ---------------------------------------------------------------------------


def _build_app() -> web.Application:
    app = web.Application()

    async def create_session(request: web.Request) -> web.Response:
        body = await request.json()
        directory = request.headers.get("x-opencode-directory", "")
        return web.json_response({"id": "ses_mock_1", "directory": directory})

    async def get_session(request: web.Request) -> web.Response:
        sid = request.match_info["sid"]
        if sid == "ses_missing":
            return web.Response(status=404)
        return web.json_response({"id": sid, "title": "mock"})

    async def delete_session(request: web.Request) -> web.Response:
        sid = request.match_info["sid"]
        if sid == "ses_gone":
            return web.Response(status=404)
        return web.Response(status=200)

    async def prompt_async(request: web.Request) -> web.Response:
        await request.json()
        return web.Response(status=204)

    async def respond_permission(request: web.Request) -> web.Response:
        await request.json()
        return web.json_response(True)

    async def list_providers(request: web.Request) -> web.Response:
        return web.json_response([
            {"id": "openai", "models": [{"id": "gpt-4o"}]},
            {"id": "anthropic", "models": [{"id": "claude-sonnet-4-20250514"}]},
        ])

    async def sse_events(request: web.Request) -> web.StreamResponse:
        resp = web.StreamResponse(
            status=200,
            headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"},
        )
        await resp.prepare(request)
        events = [
            {"type": "server.connected", "properties": {}},
            {
                "type": "message.part.delta",
                "properties": {"sessionID": "ses_mock_1", "field": "text", "delta": "Hello"},
            },
            {
                "type": "message.part.updated",
                "properties": {
                    "sessionID": "ses_mock_1",
                    "type": "tool-invocation",
                    "toolName": "bash",
                    "state": "running",
                },
            },
            {
                "type": "session.idle",
                "properties": {"sessionID": "ses_mock_1"},
            },
        ]
        for ev in events:
            line = f"data: {json.dumps(ev)}\n\n"
            await resp.write(line.encode())
        return resp

    app.router.add_post("/session", create_session)
    app.router.add_get("/session/{sid}", get_session)
    app.router.add_delete("/session/{sid}", delete_session)
    app.router.add_post("/session/{sid}/prompt_async", prompt_async)
    app.router.add_post("/session/{sid}/permissions/{pid}", respond_permission)
    app.router.add_get("/provider", list_providers)
    app.router.add_get("/event", sse_events)
    return app


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


class TestOcClient(AioHTTPTestCase):
    async def get_application(self) -> web.Application:
        return _build_app()

    def _make_client(self) -> OcClient:
        base = f"http://127.0.0.1:{self.server.port}"
        client = OcClient(base_url=base, directory="/test/dir")
        # Share the test server's connector so requests hit our mock
        client._session = self.client.session
        return client

    # -- create_session -----------------------------------------------------

    async def test_create_session(self):
        oc = self._make_client()
        sid = await oc.create_session("My Session")
        self.assertEqual(sid, "ses_mock_1")

    # -- get_session --------------------------------------------------------

    async def test_get_session_found(self):
        oc = self._make_client()
        info = await oc.get_session("ses_abc")
        self.assertIsNotNone(info)
        self.assertEqual(info["id"], "ses_abc")

    async def test_get_session_not_found(self):
        oc = self._make_client()
        info = await oc.get_session("ses_missing")
        self.assertIsNone(info)

    # -- delete_session -----------------------------------------------------

    async def test_delete_session_ok(self):
        oc = self._make_client()
        await oc.delete_session("ses_abc")

    async def test_delete_session_not_found_ok(self):
        oc = self._make_client()
        await oc.delete_session("ses_gone")

    # -- prompt_async -------------------------------------------------------

    async def test_prompt_async(self):
        oc = self._make_client()
        await oc.prompt_async(
            "ses_mock_1",
            [{"type": "text", "text": "Hello"}],
        )

    async def test_prompt_async_with_model(self):
        oc = self._make_client()
        await oc.prompt_async(
            "ses_mock_1",
            [{"type": "text", "text": "Hello"}],
            model={"providerID": "openai", "modelID": "gpt-4o"},
        )

    # -- respond_permission -------------------------------------------------

    async def test_respond_permission(self):
        oc = self._make_client()
        await oc.respond_permission("ses_mock_1", "perm_1", "once")

    # -- list_providers -----------------------------------------------------

    async def test_list_providers(self):
        oc = self._make_client()
        providers = await oc.list_providers()
        self.assertEqual(len(providers), 2)
        self.assertEqual(providers[0]["id"], "openai")

    # -- subscribe_events (SSE) ---------------------------------------------

    async def test_subscribe_events(self):
        oc = self._make_client()
        collected: list[dict] = []
        async for event in oc.subscribe_events("ses_mock_1"):
            collected.append(event)
            if event.get("type") == "session.idle":
                break
        types = [e["type"] for e in collected]
        self.assertIn("server.connected", types)
        self.assertIn("message.part.delta", types)
        self.assertIn("message.part.updated", types)
        self.assertIn("session.idle", types)

    async def test_subscribe_events_filters_other_sessions(self):
        """Events for other sessions are skipped."""
        oc = self._make_client()
        collected: list[dict] = []
        async for event in oc.subscribe_events("ses_OTHER"):
            collected.append(event)
            if event.get("type") == "session.idle":
                break
            if len(collected) > 20:
                break
        session_events = [
            e for e in collected
            if e.get("properties", {}).get("sessionID", "") == "ses_mock_1"
        ]
        self.assertEqual(len(session_events), 0)


if __name__ == "__main__":
    unittest.main()
