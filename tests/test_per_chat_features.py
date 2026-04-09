"""Tests for per-chat directory, auto-approve, file delivery, and new commands.

Covers:
  - OcClient._headers directory override
  - SessionRunner auto-approve (permission auto-granted)
  - SessionRunner._try_deliver_file
  - SessionRunner QuestionRequest handling
  - /project, /approve, /diff, /git, /stop, /undo, /redo command handlers
  - /files, /cat, /grep, /find, /history, /sessions, /todo, /agent, /tools
  - SessionManager directory/auto_approve threading
"""
from __future__ import annotations

import os
import pathlib
import tempfile
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from opencode_tg.sessions import SessionStore
from opencode_tg.oc_client import OcClient
from opencode_tg.runners import SessionRunner
from opencode_tg.session_manager import SessionManager
from opencode_tg.protocols import (
    MessagePart, PermissionInfo, PermissionRequest,
    QuestionRequest, SessionIdle, ToolEnd,
)
from opencode_tg import commands


# ---------------------------------------------------------------------------
# OcClient: _headers directory override
# ---------------------------------------------------------------------------

class TestOcClientHeaders(unittest.TestCase):
    def _make_client(self, directory: str = "/default/dir") -> OcClient:
        return OcClient(base_url="http://localhost:0", directory=directory)

    def test_headers_default(self):
        client = self._make_client()
        h = client._headers()
        self.assertEqual(h["x-opencode-directory"], "/default/dir")

    def test_headers_override(self):
        client = self._make_client()
        h = client._headers(directory="/custom/dir")
        self.assertEqual(h["x-opencode-directory"], "/custom/dir")

    def test_headers_none_falls_back(self):
        client = self._make_client()
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
        await runner.handle_event(perm)

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
        await runner.handle_event(perm)

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
            await runner.handle_event(event)
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


# ---------------------------------------------------------------------------
# Commands: /stop
# ---------------------------------------------------------------------------

class TestStopCommand(unittest.IsolatedAsyncioTestCase):
    async def test_stop_no_session(self):
        messenger = AsyncMock()
        manager = MagicMock()
        manager.get_session_id = MagicMock(return_value=None)
        manager.get_directory = MagicMock(return_value=None)
        agent = AsyncMock()

        result = await commands.handle_command("/stop", "123", None, messenger, manager, agent)
        self.assertTrue(result)
        msg = messenger.send_message.call_args[0][1]
        self.assertIn("No active session", msg)

    async def test_stop_aborts_session(self):
        messenger = AsyncMock()
        manager = MagicMock()
        manager.get_session_id = MagicMock(return_value="ses_1")
        manager.get_directory = MagicMock(return_value="/tmp")
        agent = AsyncMock()

        result = await commands.handle_command("/stop", "123", None, messenger, manager, agent)
        self.assertTrue(result)
        agent.abort_session.assert_awaited_once_with("ses_1", directory="/tmp")
        msg = messenger.send_message.call_args[0][1]
        self.assertIn("stopped", msg.lower())

    async def test_stop_failure(self):
        messenger = AsyncMock()
        manager = MagicMock()
        manager.get_session_id = MagicMock(return_value="ses_1")
        manager.get_directory = MagicMock(return_value=None)
        agent = AsyncMock()
        agent.abort_session = AsyncMock(side_effect=Exception("timeout"))

        result = await commands.handle_command("/stop", "123", None, messenger, manager, agent)
        self.assertTrue(result)
        msg = messenger.send_message.call_args[0][1]
        self.assertIn("failed", msg.lower())


# ---------------------------------------------------------------------------
# Commands: /undo (revert API)
# ---------------------------------------------------------------------------

class TestUndoCommand(unittest.IsolatedAsyncioTestCase):
    async def test_undo_no_session(self):
        messenger = AsyncMock()
        manager = MagicMock()
        manager.get_session_id = MagicMock(return_value=None)
        manager.get_directory = MagicMock(return_value=None)
        agent = AsyncMock()

        result = await commands.handle_command("/undo", "123", None, messenger, manager, agent)
        self.assertTrue(result)
        msg = messenger.send_message.call_args[0][1]
        self.assertIn("No active session", msg)

    async def test_undo_calls_revert(self):
        messenger = AsyncMock()
        manager = MagicMock()
        manager.get_session_id = MagicMock(return_value="ses_1")
        manager.get_directory = MagicMock(return_value="/proj")
        agent = AsyncMock()

        result = await commands.handle_command("/undo", "123", None, messenger, manager, agent)
        self.assertTrue(result)
        agent.revert_session.assert_awaited_once_with("ses_1", directory="/proj")

    async def test_undo_failure(self):
        messenger = AsyncMock()
        manager = MagicMock()
        manager.get_session_id = MagicMock(return_value="ses_1")
        manager.get_directory = MagicMock(return_value=None)
        agent = AsyncMock()
        agent.revert_session = AsyncMock(side_effect=Exception("no snapshots"))

        result = await commands.handle_command("/undo", "123", None, messenger, manager, agent)
        self.assertTrue(result)
        msg = messenger.send_message.call_args[0][1]
        self.assertIn("failed", msg.lower())


# ---------------------------------------------------------------------------
# Commands: /redo (unrevert API)
# ---------------------------------------------------------------------------

class TestRedoCommand(unittest.IsolatedAsyncioTestCase):
    async def test_redo_calls_unrevert(self):
        messenger = AsyncMock()
        manager = MagicMock()
        manager.get_session_id = MagicMock(return_value="ses_1")
        manager.get_directory = MagicMock(return_value=None)
        agent = AsyncMock()

        result = await commands.handle_command("/redo", "123", None, messenger, manager, agent)
        self.assertTrue(result)
        agent.unrevert_session.assert_awaited_once_with("ses_1", directory=None)


# ---------------------------------------------------------------------------
# Commands: /diff (API-based)
# ---------------------------------------------------------------------------

class TestDiffApiCommand(unittest.IsolatedAsyncioTestCase):
    async def test_diff_returns_content(self):
        messenger = AsyncMock()
        messenger.max_message_length = 4096
        manager = MagicMock()
        manager.get_session_id = MagicMock(return_value="ses_1")
        manager.get_directory = MagicMock(return_value="/proj")
        agent = AsyncMock()
        agent.session_diff = AsyncMock(return_value="+new line\n-old line")

        result = await commands.handle_command("/diff", "123", None, messenger, manager, agent)
        self.assertTrue(result)
        msg = messenger.send_message.call_args[0][1]
        self.assertIn("+new line", msg)

    async def test_diff_empty(self):
        messenger = AsyncMock()
        messenger.max_message_length = 4096
        manager = MagicMock()
        manager.get_session_id = MagicMock(return_value="ses_1")
        manager.get_directory = MagicMock(return_value=None)
        agent = AsyncMock()
        agent.session_diff = AsyncMock(return_value="")

        result = await commands.handle_command("/diff", "123", None, messenger, manager, agent)
        self.assertTrue(result)
        msg = messenger.send_message.call_args[0][1]
        self.assertIn("No pending changes", msg)


# ---------------------------------------------------------------------------
# Commands: /files, /cat, /grep, /find
# ---------------------------------------------------------------------------

class TestFileCommands(unittest.IsolatedAsyncioTestCase):
    async def test_files_list(self):
        messenger = AsyncMock()
        manager = MagicMock()
        manager.get_directory = MagicMock(return_value=None)
        agent = AsyncMock()
        agent.file_status = AsyncMock(return_value=[{"path": "src/main.py", "status": "modified"}, {"path": "README.md", "status": "added"}])

        result = await commands.handle_command("/files", "123", None, messenger, manager, agent)
        self.assertTrue(result)
        msg = messenger.send_message.call_args[0][1]
        self.assertIn("src/main.py", msg)

    async def test_files_status(self):
        messenger = AsyncMock()
        manager = MagicMock()
        manager.get_directory = MagicMock(return_value=None)
        agent = AsyncMock()
        agent.file_status = AsyncMock(return_value=[{"path": "foo.py", "status": "M"}])

        result = await commands.handle_command("/files status", "123", None, messenger, manager, agent)
        self.assertTrue(result)
        msg = messenger.send_message.call_args[0][1]
        self.assertIn("foo.py", msg)

    async def test_cat_no_args(self):
        messenger = AsyncMock()
        manager = MagicMock()
        manager.get_directory = MagicMock(return_value=None)
        agent = AsyncMock()

        result = await commands.handle_command("/cat", "123", None, messenger, manager, agent)
        self.assertTrue(result)
        msg = messenger.send_message.call_args[0][1]
        self.assertIn("Usage", msg)

    async def test_cat_reads_file(self):
        messenger = AsyncMock()
        messenger.max_message_length = 4096
        manager = MagicMock()
        manager.get_directory = MagicMock(return_value="/tmp")
        agent = AsyncMock()

        with tempfile.NamedTemporaryFile(suffix=".py", dir="/tmp", delete=False, mode="w") as f:
            f.write("print('hello')")
            fname = os.path.basename(f.name)
            path = f.name

        try:
            result = await commands.handle_command(f"/cat {fname}", "123", None, messenger, manager, agent)
            self.assertTrue(result)
            msg = messenger.send_message.call_args[0][1]
            self.assertIn("print('hello')", msg)
        finally:
            os.unlink(path)

    async def test_grep_no_args(self):
        messenger = AsyncMock()
        manager = MagicMock()
        manager.get_directory = MagicMock(return_value=None)
        agent = AsyncMock()

        result = await commands.handle_command("/grep", "123", None, messenger, manager, agent)
        self.assertTrue(result)
        msg = messenger.send_message.call_args[0][1]
        self.assertIn("Usage", msg)

    async def test_grep_sends_prompt(self):
        messenger = AsyncMock()
        manager = MagicMock()
        manager.get_directory = MagicMock(return_value=None)
        manager.get_model = MagicMock(return_value=None)
        manager.handle_message = AsyncMock()
        agent = AsyncMock()

        result = await commands.handle_command("/grep hello", "123", None, messenger, manager, agent)
        self.assertTrue(result)
        manager.handle_message.assert_awaited_once()
        parts = manager.handle_message.call_args[0][2]
        self.assertIn("hello", parts[0].text.lower())

    async def test_find_sends_prompt(self):
        messenger = AsyncMock()
        manager = MagicMock()
        manager.get_directory = MagicMock(return_value=None)
        manager.get_model = MagicMock(return_value=None)
        manager.handle_message = AsyncMock()
        agent = AsyncMock()

        result = await commands.handle_command("/find *.py", "123", None, messenger, manager, agent)
        self.assertTrue(result)
        manager.handle_message.assert_awaited_once()


# ---------------------------------------------------------------------------
# Commands: /history, /sessions, /todo
# ---------------------------------------------------------------------------

class TestHistoryCommands(unittest.IsolatedAsyncioTestCase):
    async def test_history_shows_messages(self):
        messenger = AsyncMock()
        manager = MagicMock()
        manager.get_session_id = MagicMock(return_value="ses_1")
        manager.get_directory = MagicMock(return_value=None)
        agent = AsyncMock()
        agent.session_messages = AsyncMock(return_value=[
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello there"},
        ])

        result = await commands.handle_command("/history", "123", None, messenger, manager, agent)
        self.assertTrue(result)
        msg = messenger.send_message.call_args[0][1]
        self.assertIn("[user]", msg)
        self.assertIn("hi", msg)

    async def test_sessions_list(self):
        messenger = AsyncMock()
        manager = MagicMock()
        manager.get_directory = MagicMock(return_value=None)
        agent = AsyncMock()
        agent.list_sessions = AsyncMock(return_value=[
            {"id": "abc123", "title": "Test", "status": "idle"},
        ])

        result = await commands.handle_command("/sessions", "123", None, messenger, manager, agent)
        self.assertTrue(result)
        msg = messenger.send_message.call_args[0][1]
        self.assertIn("abc123", msg)

    async def test_todo_shows_items(self):
        messenger = AsyncMock()
        manager = MagicMock()
        manager.get_session_id = MagicMock(return_value="ses_1")
        manager.get_directory = MagicMock(return_value=None)
        agent = AsyncMock()
        agent.session_todo = AsyncMock(return_value=[
            {"content": "Implement X", "status": "pending"},
            {"content": "Test Y", "status": "completed"},
        ])

        result = await commands.handle_command("/todo", "123", None, messenger, manager, agent)
        self.assertTrue(result)
        msg = messenger.send_message.call_args[0][1]
        self.assertIn("Implement X", msg)


# ---------------------------------------------------------------------------
# Commands: /agent, /tools
# ---------------------------------------------------------------------------

class TestAgentToolsCommands(unittest.IsolatedAsyncioTestCase):
    async def test_tools_lists_available(self):
        messenger = AsyncMock()
        manager = MagicMock()
        manager.get_directory = MagicMock(return_value=None)
        agent = AsyncMock()
        agent.tool_ids = AsyncMock(return_value=["bash", "write", "read", "grep"])

        result = await commands.handle_command("/tools", "123", None, messenger, manager, agent)
        self.assertTrue(result)
        msg = messenger.send_message.call_args[0][1]
        self.assertIn("bash", msg)
        self.assertIn("write", msg)

    async def test_tools_fallback_when_api_empty(self):
        messenger = AsyncMock()
        manager = MagicMock()
        manager.get_directory = MagicMock(return_value=None)
        agent = AsyncMock()
        agent.tool_ids = AsyncMock(return_value=[])

        result = await commands.handle_command("/tools", "123", None, messenger, manager, agent)
        self.assertTrue(result)
        msg = messenger.send_message.call_args[0][1]
        self.assertIn("bash", msg)

    async def test_agent_lists_agents(self):
        messenger = AsyncMock()
        manager = MagicMock()
        manager.get_directory = MagicMock(return_value=None)
        agent = AsyncMock()
        agent.app_agents = AsyncMock(return_value=[
            {"name": "coder", "description": "Code generation"},
        ])

        result = await commands.handle_command("/agent", "123", None, messenger, manager, agent)
        self.assertTrue(result)
        msg = messenger.send_message.call_args[0][1]
        self.assertIn("coder", msg)


# ---------------------------------------------------------------------------
# SessionRunner: QuestionRequest handling
# ---------------------------------------------------------------------------

class TestRunnerQuestionHandling(unittest.IsolatedAsyncioTestCase):
    async def test_auto_approve_answers_question(self):
        agent = AsyncMock()
        messenger = AsyncMock()
        messenger.max_message_length = 4096

        runner = SessionRunner(
            "ses_1", "chat_1", None,
            agent, messenger,
            auto_approve=True,
        )
        runner.state = "generating"

        event = QuestionRequest(
            request_id="q_1",
            session_id="ses_1",
            questions=[{
                "id": "q1",
                "prompt": "Which framework?",
                "options": [
                    {"value": "react", "label": "React", "default": True},
                    {"value": "vue", "label": "Vue"},
                ],
            }],
        )
        await runner.handle_event(event)
        agent.reply_question.assert_awaited_once()
        call_args = agent.reply_question.call_args
        self.assertEqual(call_args[0][0], "q_1")

    async def test_manual_shows_question_to_user(self):
        agent = AsyncMock()
        messenger = AsyncMock()
        messenger.max_message_length = 4096
        messenger.send_message = AsyncMock(return_value="msg_42")

        runner = SessionRunner(
            "ses_1", "chat_1", None,
            agent, messenger,
            auto_approve=False,
        )
        runner.state = "generating"

        event = QuestionRequest(
            request_id="q_2",
            session_id="ses_1",
            questions=[{
                "id": "q1",
                "prompt": "Proceed?",
                "options": [{"value": "yes"}, {"value": "no"}],
            }],
        )
        await runner.handle_event(event)
        agent.reply_question.assert_not_awaited()
        msg = messenger.send_message.call_args[0][1]
        self.assertIn("Proceed?", msg)


# ---------------------------------------------------------------------------
# ToolEnd with output
# ---------------------------------------------------------------------------

class TestToolEndOutput(unittest.IsolatedAsyncioTestCase):
    async def test_tool_end_shows_output(self):
        agent = AsyncMock()
        messenger = AsyncMock()
        messenger.max_message_length = 4096
        messenger.edit_message = AsyncMock(return_value=True)

        runner = SessionRunner(
            "ses_1", "chat_1", None,
            agent, messenger,
        )
        runner.state = "generating"
        runner.tool_msgs["c1"] = "tool_msg_1"

        event = ToolEnd(
            name="bash", call_id="c1", state="completed",
            title="ls -la", output="total 42\ndrwxr-xr-x ...",
        )
        await runner.handle_event(event)

        msg = messenger.edit_message.call_args[0][2]
        self.assertIn("bash", msg)
        self.assertIn("total 42", msg)


if __name__ == "__main__":
    unittest.main()
