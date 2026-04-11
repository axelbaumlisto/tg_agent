"""Unit tests for oc_client — mock HTTP server via aiohttp.test_utils."""

from __future__ import annotations

import unittest

from aiohttp import web
from aiohttp.test_utils import AioHTTPTestCase

from opencode_tg.oc_client import OcClient, parse_sse_line
from .mock_oc_server import build_mock_app


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


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


class TestOcClient(AioHTTPTestCase):
    async def get_application(self) -> web.Application:
        return build_mock_app()

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
        # succeeds without raising — the mock doesn't remove server-side state
        await oc.delete_session("ses_abc")

    async def test_delete_session_not_found_ok(self):
        oc = self._make_client()
        await oc.delete_session("ses_gone")

    # -- prompt_async -------------------------------------------------------

    async def test_prompt_async(self):
        oc = self._make_client()
        result = await oc.prompt_async(
            "ses_mock_1",
            [{"type": "text", "text": "Hello"}],
        )
        self.assertTrue(result is None or isinstance(result, dict))

    async def test_prompt_async_with_model(self):
        oc = self._make_client()
        result = await oc.prompt_async(
            "ses_mock_1",
            [{"type": "text", "text": "Hello"}],
            model={"providerID": "openai", "modelID": "gpt-4o"},
        )
        self.assertTrue(result is None or isinstance(result, dict))

    # -- respond_permission -------------------------------------------------

    async def test_respond_permission(self):
        oc = self._make_client()
        result = await oc.respond_permission("ses_mock_1", "perm_1", "once")
        self.assertTrue(result is None or isinstance(result, dict))

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


class TestOcClientExtraEndpoints(AioHTTPTestCase):
    """Cover previously untested OcClient methods."""

    async def get_application(self) -> web.Application:
        return build_mock_app()

    def _make_client(self) -> OcClient:
        base = f"http://127.0.0.1:{self.server.port}"
        oc = OcClient(base_url=base)
        oc._session = self.client.session
        return oc

    async def test_abort_session(self):
        oc = self._make_client()
        await oc.abort_session("ses_mock_1")

    async def test_list_sessions(self):
        oc = self._make_client()
        sessions = await oc.list_sessions()
        self.assertIsInstance(sessions, list)
        self.assertTrue(len(sessions) >= 1)

    async def test_session_messages(self):
        oc = self._make_client()
        msgs = await oc.session_messages("ses_mock_1")
        self.assertIsInstance(msgs, list)

    async def test_session_diff(self):
        oc = self._make_client()
        diff = await oc.session_diff("ses_mock_1")
        self.assertIsInstance(diff, str)

    async def test_fork_session(self):
        oc = self._make_client()
        new_id = await oc.fork_session("ses_mock_1")
        self.assertTrue(len(new_id) > 0)

    async def test_file_status(self):
        oc = self._make_client()
        files = await oc.file_status()
        self.assertIsInstance(files, list)

    async def test_find_text(self):
        oc = self._make_client()
        results = await oc.find_text("match")
        self.assertIsInstance(results, list)

    async def test_find_files(self):
        oc = self._make_client()
        results = await oc.find_files("main")
        self.assertIsInstance(results, list)


if __name__ == "__main__":
    unittest.main()
