"""Tests for per-chat directory, auto-approve, file delivery, and new commands.

Covers:
  - SessionStore.{get,set,delete}_directory
  - SessionStore.{get,set}_auto_approve
  - OcClient._headers directory override
  - SessionRunner auto-approve (permission auto-granted)
  - SessionRunner._try_deliver_file
  - /project, /approve, /diff, /git command handlers
  - SessionManager directory/auto_approve threading
"""
from __future__ import annotations

import asyncio
import json
import os
import pathlib
import tempfile
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from opencode_tg.sessions import SessionStore
from opencode_tg.oc_client import OcClient
from opencode_tg.runners import SessionRunner
from opencode_tg.session_manager import SessionManager
from opencode_tg.protocols import (
    MessagePart, ModelRef, PermissionInfo, PermissionRequest,
    SessionIdle, TextDelta, ToolEnd, ToolStart,
)
from opencode_tg import commands


# ---------------------------------------------------------------------------
# SessionStore: directory + auto_approve
# ---------------------------------------------------------------------------

class TestSessionStoreDirectory(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self._tmp.write(b"{}")
        self._tmp.close()
        self.store = SessionStore(pathlib.Path(self._tmp.name))

    def tearDown(self):
        os.unlink(self._tmp.name)

    def test_get_directory_default_none(self):
        self.assertIsNone(self.store.get_directory("123", None))

    def test_set_and_get_directory(self):
        self.store.set_directory("123", None, "/home/user/project")
        self.assertEqual(self.store.get_directory("123", None), "/home/user/project")

    def test_set_directory_with_thread(self):
        self.store.set_directory("123", "456", "/home/user/frontend")
        self.assertEqual(self.store.get_directory("123", "456"), "/home/user/frontend")
        self.assertIsNone(self.store.get_directory("123", None))

    def test_delete_directory(self):
        self.store.set_directory("123", None, "/tmp/test")
        self.store.delete_directory("123", None)
        self.assertIsNone(self.store.get_directory("123", None))

    def test_directory_persists_with_session(self):
        self.store.set("123", None, "ses_1")
        self.store.set_directory("123", None, "/projects/alpha")
        self.assertEqual(self.store.get("123", None), "ses_1")
        self.assertEqual(self.store.get_directory("123", None), "/projects/alpha")

    def test_directory_survives_reload(self):
        self.store.set_directory("123", None, "/srv/app")
        store2 = SessionStore(pathlib.Path(self._tmp.name))
        self.assertEqual(store2.get_directory("123", None), "/srv/app")


class TestSessionStoreClearSession(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self._tmp.write(b"{}")
        self._tmp.close()
        self.store = SessionStore(pathlib.Path(self._tmp.name))

    def tearDown(self):
        os.unlink(self._tmp.name)

    def test_clear_preserves_directory(self):
        self.store.set("123", None, "ses_1")
        self.store.set_model("123", None, "openai", "gpt-4o")
        self.store.set_directory("123", None, "/projects/alpha")
        self.store.set_auto_approve("123", None, True)

        self.store.clear_session("123", None)

        self.assertIsNone(self.store.get("123", None))
        self.assertIsNone(self.store.get_model("123", None))
        self.assertEqual(self.store.get_directory("123", None), "/projects/alpha")
        self.assertTrue(self.store.get_auto_approve("123", None))

    def test_clear_no_entry_is_noop(self):
        self.store.clear_session("999", None)

    def test_clear_removes_empty_entry(self):
        self.store.set("123", None, "ses_1")
        self.store.clear_session("123", None)
        self.assertIsNone(self.store.get("123", None))


class TestSessionStoreAutoApprove(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self._tmp.write(b"{}")
        self._tmp.close()
        self.store = SessionStore(pathlib.Path(self._tmp.name))

    def tearDown(self):
        os.unlink(self._tmp.name)

    def test_default_is_false(self):
        self.assertFalse(self.store.get_auto_approve("123", None))

    def test_set_true(self):
        self.store.set_auto_approve("123", None, True)
        self.assertTrue(self.store.get_auto_approve("123", None))

    def test_set_false(self):
        self.store.set_auto_approve("123", None, True)
        self.store.set_auto_approve("123", None, False)
        self.assertFalse(self.store.get_auto_approve("123", None))

    def test_per_thread_isolation(self):
        self.store.set_auto_approve("123", "t1", True)
        self.assertFalse(self.store.get_auto_approve("123", None))
        self.assertTrue(self.store.get_auto_approve("123", "t1"))

    def test_survives_reload(self):
        self.store.set_auto_approve("123", None, True)
        store2 = SessionStore(pathlib.Path(self._tmp.name))
        self.assertTrue(store2.get_auto_approve("123", None))


# ---------------------------------------------------------------------------
# OcClient: _headers directory override
# ---------------------------------------------------------------------------

class TestOcClientHeaders(unittest.TestCase):
    def test_headers_default(self):
        client = OcClient.__new__(OcClient)
        client.directory = "/default/dir"
        h = client._headers()
        self.assertEqual(h["x-opencode-directory"], "/default/dir")

    def test_headers_override(self):
        client = OcClient.__new__(OcClient)
        client.directory = "/default/dir"
        h = client._headers(directory="/custom/dir")
        self.assertEqual(h["x-opencode-directory"], "/custom/dir")

    def test_headers_none_falls_back(self):
        client = OcClient.__new__(OcClient)
        client.directory = "/default/dir"
        h = client._headers(directory=None)
        self.assertEqual(h["x-opencode-directory"], "/default/dir")


# ---------------------------------------------------------------------------
# SessionRunner: auto-approve
# ---------------------------------------------------------------------------

class TestRunnerAutoApprove(unittest.IsolatedAsyncioTestCase):
    async def test_auto_approve_grants_permission(self):
        agent = AsyncMock()
        messenger = AsyncMock()
        messenger.max_message_length = 4096

        runner = SessionRunner(
            "ses_1", "chat_1", None,
            agent, messenger,
            auto_approve=True,
        )
        runner.state = "generating"

        perm = PermissionRequest(info=PermissionInfo(
            id="perm_1", session_id="ses_1", title="Run bash", pattern="echo hi",
        ))
        await runner._handle_event(perm)

        agent.respond_permission.assert_awaited_once_with(
            "ses_1", "perm_1", "always", directory=None,
        )
        self.assertEqual(len(runner.pending_permissions), 0)

    async def test_manual_approve_shows_keyboard(self):
        agent = AsyncMock()
        messenger = AsyncMock()
        messenger.max_message_length = 4096
        messenger.send_permission_request = AsyncMock(return_value="msg_42")

        runner = SessionRunner(
            "ses_1", "chat_1", None,
            agent, messenger,
            auto_approve=False,
        )
        runner.state = "generating"

        perm = PermissionRequest(info=PermissionInfo(
            id="perm_2", session_id="ses_1", title="Write file",
        ))
        await runner._handle_event(perm)

        agent.respond_permission.assert_not_awaited()
        messenger.send_permission_request.assert_awaited_once()
        self.assertIn("perm_2", runner.pending_permissions)


# ---------------------------------------------------------------------------
# SessionRunner: file delivery
# ---------------------------------------------------------------------------

class TestRunnerFileDelivery(unittest.IsolatedAsyncioTestCase):
    async def test_delivers_existing_file(self):
        messenger = AsyncMock()
        messenger.max_message_length = 4096
        messenger.send_document = AsyncMock(return_value="msg_99")

        runner = SessionRunner(
            "ses_1", "chat_1", "thread_1",
            AsyncMock(), messenger,
            directory="/tmp",
        )

        with tempfile.NamedTemporaryFile(suffix=".py", dir="/tmp", delete=False) as f:
            f.write(b"print('hello')\n")
            path = f.name

        try:
            await runner._try_deliver_file(path)
            messenger.send_document.assert_awaited_once()
            call_args = messenger.send_document.call_args
            self.assertEqual(call_args[0][0], "chat_1")
            self.assertEqual(call_args[0][1], path)
        finally:
            os.unlink(path)

    async def test_skips_nonexistent_file(self):
        messenger = AsyncMock()
        messenger.send_document = AsyncMock()
        runner = SessionRunner(
            "ses_1", "chat_1", None,
            AsyncMock(), messenger,
        )
        await runner._try_deliver_file("/tmp/does_not_exist_12345.py")
        messenger.send_document.assert_not_awaited()

    async def test_skips_non_deliverable_extension(self):
        messenger = AsyncMock()
        messenger.send_document = AsyncMock()
        runner = SessionRunner(
            "ses_1", "chat_1", None,
            AsyncMock(), messenger,
        )
        with tempfile.NamedTemporaryFile(suffix=".xyz", dir="/tmp", delete=False) as f:
            f.write(b"data")
            path = f.name

        try:
            await runner._try_deliver_file(path)
            messenger.send_document.assert_not_awaited()
        finally:
            os.unlink(path)

    async def test_resolves_relative_path_with_directory(self):
        messenger = AsyncMock()
        messenger.max_message_length = 4096
        messenger.send_document = AsyncMock(return_value="msg_1")

        runner = SessionRunner(
            "ses_1", "chat_1", None,
            AsyncMock(), messenger,
            directory="/tmp",
        )

        with tempfile.NamedTemporaryFile(suffix=".txt", dir="/tmp", delete=False) as f:
            f.write(b"content")
            fname = os.path.basename(f.name)
            full_path = f.name

        try:
            await runner._try_deliver_file(fname)
            messenger.send_document.assert_awaited_once()
        finally:
            os.unlink(full_path)

    async def test_tool_end_triggers_delivery(self):
        agent = AsyncMock()
        messenger = AsyncMock()
        messenger.max_message_length = 4096
        messenger.edit_message = AsyncMock(return_value=True)
        messenger.send_document = AsyncMock(return_value="msg_1")

        runner = SessionRunner(
            "ses_1", "chat_1", None,
            agent, messenger,
            directory="/tmp",
        )
        runner.state = "generating"
        runner.tool_msgs["c1"] = "tool_msg_1"

        with tempfile.NamedTemporaryFile(suffix=".py", dir="/tmp", delete=False) as f:
            f.write(b"print(1)\n")
            path = f.name

        try:
            event = ToolEnd(name="write", call_id="c1", state="completed", title=path)
            await runner._handle_event(event)
            messenger.send_document.assert_awaited_once()
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Commands: /project
# ---------------------------------------------------------------------------

class TestProjectCommand(unittest.IsolatedAsyncioTestCase):
    def _make_manager(self):
        manager = MagicMock()
        manager.get_directory = MagicMock(return_value=None)
        manager.set_directory = MagicMock()
        manager.reset_session = AsyncMock()
        return manager

    async def test_project_no_args_shows_current(self):
        messenger = AsyncMock()
        manager = self._make_manager()
        manager.get_directory.return_value = "/home/user/project"
        agent = AsyncMock()

        result = await commands.handle_command(
            "/project", "123", None, messenger, manager, agent,
        )
        self.assertTrue(result)
        messenger.send_message.assert_awaited_once()
        msg_text = messenger.send_message.call_args[0][1]
        self.assertIn("/home/user/project", msg_text)

    async def test_project_valid_dir(self):
        messenger = AsyncMock()
        manager = self._make_manager()
        agent = AsyncMock()

        result = await commands.handle_command(
            "/project /tmp", "123", None, messenger, manager, agent,
        )
        self.assertTrue(result)
        manager.set_directory.assert_called_once_with("123", None, "/tmp")
        manager.reset_session.assert_awaited_once()

    async def test_project_invalid_dir(self):
        messenger = AsyncMock()
        manager = self._make_manager()
        agent = AsyncMock()

        result = await commands.handle_command(
            "/project /nonexistent/dir/12345", "123", None, messenger, manager, agent,
        )
        self.assertTrue(result)
        manager.set_directory.assert_not_called()
        msg_text = messenger.send_message.call_args[0][1]
        self.assertIn("not found", msg_text.lower())


# ---------------------------------------------------------------------------
# Commands: /approve
# ---------------------------------------------------------------------------

class TestApproveCommand(unittest.IsolatedAsyncioTestCase):
    def _make_manager(self):
        manager = MagicMock()
        manager.set_auto_approve = MagicMock()
        manager.get_auto_approve = MagicMock(return_value=False)
        return manager

    async def test_approve_on(self):
        messenger = AsyncMock()
        manager = self._make_manager()
        agent = AsyncMock()

        result = await commands.handle_command(
            "/approve on", "123", None, messenger, manager, agent,
        )
        self.assertTrue(result)
        manager.set_auto_approve.assert_called_once_with("123", None, True)

    async def test_approve_off(self):
        messenger = AsyncMock()
        manager = self._make_manager()
        agent = AsyncMock()

        result = await commands.handle_command(
            "/approve off", "123", None, messenger, manager, agent,
        )
        self.assertTrue(result)
        manager.set_auto_approve.assert_called_once_with("123", None, False)

    async def test_approve_no_args_shows_status(self):
        messenger = AsyncMock()
        manager = self._make_manager()
        manager.get_auto_approve.return_value = True
        agent = AsyncMock()

        result = await commands.handle_command(
            "/approve", "123", None, messenger, manager, agent,
        )
        self.assertTrue(result)
        msg_text = messenger.send_message.call_args[0][1]
        self.assertIn("ON", msg_text)


# ---------------------------------------------------------------------------
# Commands: /git
# ---------------------------------------------------------------------------

class TestGitCommand(unittest.IsolatedAsyncioTestCase):
    async def test_git_no_args_shows_usage(self):
        messenger = AsyncMock()
        manager = MagicMock()
        agent = AsyncMock()

        result = await commands.handle_command(
            "/git", "123", None, messenger, manager, agent,
        )
        self.assertTrue(result)
        msg_text = messenger.send_message.call_args[0][1]
        self.assertIn("Usage", msg_text)

    async def test_git_status_sends_prompt(self):
        messenger = AsyncMock()
        manager = MagicMock()
        manager.get_model = MagicMock(return_value=None)
        manager.handle_message = AsyncMock()
        agent = AsyncMock()

        result = await commands.handle_command(
            "/git status", "123", "t1", messenger, manager, agent,
        )
        self.assertTrue(result)
        manager.handle_message.assert_awaited_once()
        parts = manager.handle_message.call_args[0][2]
        self.assertIn("git status", parts[0].text)


# ---------------------------------------------------------------------------
# Commands: /diff
# ---------------------------------------------------------------------------

class TestDiffCommand(unittest.IsolatedAsyncioTestCase):
    async def test_diff_no_session(self):
        messenger = AsyncMock()
        manager = MagicMock()
        manager.get_session_id = MagicMock(return_value=None)
        agent = AsyncMock()

        result = await commands.handle_command(
            "/diff", "123", None, messenger, manager, agent,
        )
        self.assertTrue(result)
        msg_text = messenger.send_message.call_args[0][1]
        self.assertIn("No active session", msg_text)


# ---------------------------------------------------------------------------
# SessionManager: directory + auto_approve wiring
# ---------------------------------------------------------------------------

class TestSessionManagerDirectoryWiring(unittest.IsolatedAsyncioTestCase):
    def _make_manager(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        tmp.write(b"{}")
        tmp.close()
        self._tmp_path = tmp.name

        store = SessionStore(pathlib.Path(tmp.name))
        agent = AsyncMock()
        agent.create_session = AsyncMock(return_value="ses_new")
        agent.get_session = AsyncMock(return_value={"id": "ses_new", "status": "idle"})
        messenger = AsyncMock()
        messenger.max_message_length = 4096
        messenger.name = "test"
        messenger.send_message = AsyncMock(return_value="msg_1")
        messenger.edit_message = AsyncMock(return_value=True)
        messenger.send_typing = AsyncMock()

        mgr = SessionManager(store, agent, messenger)
        return mgr, store, agent, messenger

    def tearDown(self):
        if hasattr(self, "_tmp_path"):
            os.unlink(self._tmp_path)

    async def test_set_and_get_directory(self):
        mgr, store, _, _ = self._make_manager()
        mgr.set_directory("123", None, "/projects/alpha")
        self.assertEqual(mgr.get_directory("123", None), "/projects/alpha")

    async def test_set_and_get_auto_approve(self):
        mgr, _, _, _ = self._make_manager()
        self.assertFalse(mgr.get_auto_approve("123", None))
        mgr.set_auto_approve("123", None, True)
        self.assertTrue(mgr.get_auto_approve("123", None))

    async def test_directory_passed_to_create_session(self):
        mgr, store, agent, messenger = self._make_manager()
        store.set_directory("123", None, "/custom/project")

        parts = [MessagePart(type="text", text="hello")]

        async def fake_subscribe(sid, **kwargs):
            yield SessionIdle()
        agent.subscribe_events = fake_subscribe

        await mgr.handle_message("123", None, parts)

        agent.create_session.assert_awaited_once()
        call_kwargs = agent.create_session.call_args
        self.assertEqual(call_kwargs.kwargs.get("directory"), "/custom/project")

    async def test_auto_approve_propagated_to_runner(self):
        mgr, store, agent, messenger = self._make_manager()
        store.set_auto_approve("123", None, True)
        store.set("123", None, "ses_existing")

        async def fake_subscribe(sid, **kwargs):
            yield SessionIdle()
        agent.subscribe_events = fake_subscribe

        parts = [MessagePart(type="text", text="test")]
        await mgr.handle_message("123", None, parts)

        runner = mgr._runners.get("123")
        self.assertIsNotNone(runner)
        self.assertTrue(runner.auto_approve)

    async def test_directory_updated_on_existing_runner(self):
        mgr, store, agent, messenger = self._make_manager()
        store.set("123", None, "ses_1")

        async def fake_subscribe(sid, **kwargs):
            yield SessionIdle()
        agent.subscribe_events = fake_subscribe

        parts = [MessagePart(type="text", text="first")]
        await mgr.handle_message("123", None, parts)

        runner = mgr._runners.get("123")
        self.assertIsNone(runner.directory)

        store.set_directory("123", None, "/new/dir")
        await mgr.handle_message("123", None, parts)

        self.assertEqual(runner.directory, "/new/dir")


# ---------------------------------------------------------------------------
# Commands: /id shows directory + auto_approve
# ---------------------------------------------------------------------------

class TestIdCommand(unittest.IsolatedAsyncioTestCase):
    async def test_id_shows_directory_and_approve(self):
        messenger = AsyncMock()
        manager = MagicMock()
        manager.get_session_id = MagicMock(return_value="ses_1")
        manager.get_directory = MagicMock(return_value="/custom/path")
        manager.get_auto_approve = MagicMock(return_value=True)
        agent = AsyncMock()

        result = await commands.handle_command(
            "/id", "123", "t1", messenger, manager, agent,
        )
        self.assertTrue(result)
        msg_text = messenger.send_message.call_args[0][1]
        self.assertIn("/custom/path", msg_text)
        self.assertIn("True", msg_text)


if __name__ == "__main__":
    unittest.main()
