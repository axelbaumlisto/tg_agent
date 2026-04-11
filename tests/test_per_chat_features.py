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
    QuestionRequest, ReasoningDelta, SessionIdle,
    TextDelta, ToolEnd, ToolStart,
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
        runner._msg_id = "main_msg"

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

    async def test_git_status_direct_api(self):
        messenger = AsyncMock()
        manager = MagicMock()
        manager.get_directory = MagicMock(return_value="/proj")
        agent = AsyncMock()
        agent.vcs_get = AsyncMock(return_value={"branch": "main", "dirty": True})

        result = await commands.handle_command(
            "/git status", "123", "t1", messenger, manager, agent,
        )
        self.assertTrue(result)
        agent.vcs_get.assert_awaited_once()
        sent_text = messenger.send_message.call_args[0][1]
        self.assertIn("main", sent_text)


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

    async def test_grep_direct_api(self):
        messenger = AsyncMock()
        manager = MagicMock()
        manager.get_directory = MagicMock(return_value="/proj")
        agent = AsyncMock()
        agent.find_text = AsyncMock(return_value=[
            {"file": "main.py", "line": 10, "text": "hello world"},
        ])

        result = await commands.handle_command("/grep hello", "123", None, messenger, manager, agent)
        self.assertTrue(result)
        agent.find_text.assert_awaited_once_with("hello", directory="/proj")
        sent_text = messenger.send_message.call_args[0][1]
        self.assertIn("main.py", sent_text)

    async def test_grep_no_results(self):
        messenger = AsyncMock()
        manager = MagicMock()
        manager.get_directory = MagicMock(return_value=None)
        agent = AsyncMock()
        agent.find_text = AsyncMock(return_value=[])

        result = await commands.handle_command("/grep xyz", "123", None, messenger, manager, agent)
        self.assertTrue(result)
        sent_text = messenger.send_message.call_args[0][1]
        self.assertIn("No matches", sent_text)

    async def test_find_direct_api(self):
        messenger = AsyncMock()
        manager = MagicMock()
        manager.get_directory = MagicMock(return_value="/proj")
        agent = AsyncMock()
        agent.find_files = AsyncMock(return_value=[
            {"file": "setup.py"}, {"file": "main.py"},
        ])

        result = await commands.handle_command("/find *.py", "123", None, messenger, manager, agent)
        self.assertTrue(result)
        agent.find_files.assert_awaited_once_with("*.py", directory="/proj")
        sent_text = messenger.send_message.call_args[0][1]
        self.assertIn("setup.py", sent_text)


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
    async def test_tool_end_shows_status_in_composite(self):
        agent = AsyncMock()
        messenger = AsyncMock()
        messenger.max_message_length = 4096
        messenger.edit_message = AsyncMock(return_value=True)

        runner = SessionRunner(
            "ses_1", "chat_1", None,
            agent, messenger,
        )
        runner.state = "generating"
        runner._msg_id = "main_msg"

        start = ToolStart(name="bash", call_id="c1")
        await runner.handle_event(start)

        composite = runner._accumulated_text
        self.assertIn("bash", composite)
        self.assertIn("\U0001f527", composite)

        end = ToolEnd(
            name="bash", call_id="c1", state="completed",
            title="ls -la", output="total 42\ndrwxr-xr-x ...",
        )
        await runner.handle_event(end)

        composite = runner._accumulated_text
        self.assertIn("bash", composite)
        self.assertIn("\u2705", composite)
        self.assertIn("ls -la", composite)


# ---------------------------------------------------------------------------
# /summarize command
# ---------------------------------------------------------------------------

class TestSummarizeCommand(unittest.IsolatedAsyncioTestCase):
    async def test_summarize_returns_text(self):
        messenger = AsyncMock()
        messenger.max_message_length = 4096
        manager = MagicMock()
        manager.get_session_id = MagicMock(return_value="ses_1")
        manager.get_directory = MagicMock(return_value="/proj")
        agent = AsyncMock()
        agent.summarize_session = AsyncMock(return_value="The session discussed X and Y.")

        result = await commands.handle_command("/summarize", "123", None, messenger, manager, agent)
        self.assertTrue(result)
        agent.summarize_session.assert_awaited_once_with("ses_1", directory="/proj")
        all_texts = [str(c) for c in messenger.send_message.call_args_list]
        combined = " ".join(all_texts)
        self.assertIn("X and Y", combined)

    async def test_summarize_no_session(self):
        messenger = AsyncMock()
        manager = MagicMock()
        manager.get_session_id = MagicMock(return_value=None)
        agent = AsyncMock()

        result = await commands.handle_command("/summarize", "123", None, messenger, manager, agent)
        self.assertTrue(result)
        sent_text = messenger.send_message.call_args[0][1]
        self.assertIn("No active session", sent_text)


# ---------------------------------------------------------------------------
# Document fallback for long /cat and /diff
# ---------------------------------------------------------------------------

class TestDocumentFallback(unittest.IsolatedAsyncioTestCase):
    async def test_cat_long_file_sends_document(self):
        messenger = AsyncMock()
        messenger.max_message_length = 4096
        manager = MagicMock()
        agent = AsyncMock()

        tmpdir = tempfile.mkdtemp()
        manager.get_directory = MagicMock(return_value=tmpdir)
        fpath = os.path.join(tmpdir, "big.txt")
        with open(fpath, "w") as f:
            f.write("x" * 5000)

        try:
            result = await commands.handle_command(f"/cat {fpath}", "123", None, messenger, manager, agent)
            self.assertTrue(result)
            messenger.send_document.assert_awaited_once()
        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)

    async def test_diff_long_sends_document(self):
        messenger = AsyncMock()
        messenger.max_message_length = 4096
        manager = MagicMock()
        manager.get_session_id = MagicMock(return_value="ses_1")
        manager.get_directory = MagicMock(return_value=None)
        agent = AsyncMock()
        agent.session_diff = AsyncMock(return_value="+" * 5000)

        result = await commands.handle_command("/diff", "123", None, messenger, manager, agent)
        self.assertTrue(result)
        messenger.send_document.assert_awaited_once()

    async def test_cat_short_file_sends_inline(self):
        messenger = AsyncMock()
        messenger.max_message_length = 4096
        manager = MagicMock()
        agent = AsyncMock()

        tmpdir = tempfile.mkdtemp()
        manager.get_directory = MagicMock(return_value=tmpdir)
        fpath = os.path.join(tmpdir, "small.txt")
        with open(fpath, "w") as f:
            f.write("short content")

        try:
            result = await commands.handle_command(f"/cat {fpath}", "123", None, messenger, manager, agent)
            self.assertTrue(result)
            messenger.send_document.assert_not_awaited()
            sent_text = messenger.send_message.call_args[0][1]
            self.assertIn("short content", sent_text)
        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Audio extraction in TelegramMessenger
# ---------------------------------------------------------------------------

class TestAudioExtraction(unittest.TestCase):
    def test_audio_message_extracted(self):
        from opencode_tg.messengers.telegram import TelegramMessenger
        msg = {
            "audio": {
                "file_id": "aud_123",
                "mime_type": "audio/mpeg",
                "file_name": "song.mp3",
                "duration": 180,
            }
        }
        refs = TelegramMessenger._extract_attachment_refs(msg)
        self.assertEqual(len(refs), 1)
        self.assertEqual(refs[0]["file_id"], "aud_123")
        self.assertEqual(refs[0]["mime"], "audio/mpeg")
        self.assertEqual(refs[0]["filename"], "song.mp3")

    def test_audio_defaults(self):
        from opencode_tg.messengers.telegram import TelegramMessenger
        msg = {"audio": {"file_id": "aud_456"}}
        refs = TelegramMessenger._extract_attachment_refs(msg)
        self.assertEqual(len(refs), 1)
        self.assertEqual(refs[0]["mime"], "audio/mpeg")
        self.assertEqual(refs[0]["filename"], "audio.mp3")

    def test_voice_and_audio_separate(self):
        from opencode_tg.messengers.telegram import TelegramMessenger
        msg = {
            "voice": {"file_id": "v1"},
            "audio": {"file_id": "a1", "mime_type": "audio/mp4", "file_name": "clip.m4a"},
        }
        refs = TelegramMessenger._extract_attachment_refs(msg)
        self.assertEqual(len(refs), 2)
        file_ids = {r["file_id"] for r in refs}
        self.assertEqual(file_ids, {"v1", "a1"})


# ---------------------------------------------------------------------------
# SessionManager.find_runner_for_question
# ---------------------------------------------------------------------------

class TestFindRunnerForQuestion(unittest.IsolatedAsyncioTestCase):
    def _make_manager(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        tmp.write(b"{}")
        tmp.close()
        self._tmp_path = tmp.name

        store = SessionStore(pathlib.Path(tmp.name))
        agent = AsyncMock()
        messenger = AsyncMock()
        messenger.max_message_length = 4096
        messenger.name = "test"
        from opencode_tg.session_manager import SessionManager
        mgr = SessionManager(store, agent, messenger)
        return mgr

    def tearDown(self):
        if hasattr(self, "_tmp_path"):
            os.unlink(self._tmp_path)

    async def test_finds_runner_with_pending_question(self):
        from opencode_tg.protocols import QuestionRequest
        mgr = self._make_manager()
        runner = MagicMock()
        runner.pending_questions = {"req_42": QuestionRequest("req_42", "ses_1", [])}
        mgr._runners["chat_1"] = runner

        found = mgr.find_runner_for_question("req_42")
        self.assertIs(found, runner)

    async def test_returns_none_when_not_found(self):
        mgr = self._make_manager()
        runner = MagicMock()
        runner.pending_questions = {}
        mgr._runners["chat_1"] = runner

        self.assertIsNone(mgr.find_runner_for_question("req_nope"))


# ---------------------------------------------------------------------------
# /find no results
# ---------------------------------------------------------------------------

class TestFindNoResults(unittest.IsolatedAsyncioTestCase):
    async def test_find_no_results(self):
        messenger = AsyncMock()
        manager = MagicMock()
        manager.get_directory = MagicMock(return_value=None)
        agent = AsyncMock()
        agent.find_files = AsyncMock(return_value=[])

        result = await commands.handle_command("/find *.xyz", "123", None, messenger, manager, agent)
        self.assertTrue(result)
        sent_text = messenger.send_message.call_args[0][1]
        self.assertIn("No files", sent_text)


# ---------------------------------------------------------------------------
# /summarize error handling
# ---------------------------------------------------------------------------

class TestSummarizeError(unittest.IsolatedAsyncioTestCase):
    async def test_summarize_api_error(self):
        messenger = AsyncMock()
        messenger.max_message_length = 4096
        manager = MagicMock()
        manager.get_session_id = MagicMock(return_value="ses_1")
        manager.get_directory = MagicMock(return_value=None)
        agent = AsyncMock()
        agent.summarize_session = AsyncMock(side_effect=RuntimeError("API down"))

        result = await commands.handle_command("/summarize", "123", None, messenger, manager, agent)
        self.assertTrue(result)
        all_texts = " ".join(str(c) for c in messenger.send_message.call_args_list)
        self.assertIn("Summarize failed", all_texts)


# ---------------------------------------------------------------------------
# Single-message composite: reasoning → tools → response
# ---------------------------------------------------------------------------

class TestCompositeMessage(unittest.IsolatedAsyncioTestCase):
    def _make_runner(self):
        agent = AsyncMock()
        messenger = AsyncMock()
        messenger.max_message_length = 4096
        messenger.edit_message = AsyncMock(return_value=True)
        runner = SessionRunner("ses_1", "chat_1", None, agent, messenger)
        runner.state = "generating"
        runner._msg_id = "main_msg"
        return runner, messenger

    async def test_reasoning_tools_response_single_message(self):
        runner, messenger = self._make_runner()

        await runner.handle_event(ReasoningDelta(text="Let me think..."))
        self.assertIn("\U0001f4ad", runner._accumulated_text)
        self.assertIn("think", runner._accumulated_text)

        await runner.handle_event(ToolStart(name="bash", call_id="c1"))
        self.assertIn("\U0001f527", runner._accumulated_text)
        self.assertIn("bash", runner._accumulated_text)

        await runner.handle_event(ToolEnd(name="bash", call_id="c1", state="completed", title="ls"))
        self.assertIn("\u2705", runner._accumulated_text)

        await runner.handle_event(TextDelta(text="Here is the result"))
        self.assertIn("Here is the result", runner._accumulated_text)
        self.assertNotIn("\U0001f4ad", runner._accumulated_text)

        messenger.send_message.assert_not_called()

    async def test_no_separate_messages_for_tools(self):
        runner, messenger = self._make_runner()

        await runner.handle_event(ToolStart(name="read", call_id="t1"))
        await runner.handle_event(ToolStart(name="write", call_id="t2"))

        messenger.send_message.assert_not_called()
        self.assertIn("read", runner._accumulated_text)
        self.assertIn("write", runner._accumulated_text)

    async def test_finalize_uses_response_text(self):
        runner, messenger = self._make_runner()

        await runner.handle_event(ReasoningDelta(text="thinking..."))
        await runner.handle_event(ToolStart(name="bash", call_id="c1"))
        await runner.handle_event(ToolEnd(name="bash", call_id="c1", state="completed", title="ls"))
        runner._response_text = "Final answer here"

        await runner._finalize_response()

        edit_text = messenger.edit_message.call_args[0][2]
        self.assertIn("Final answer", edit_text)
        self.assertNotIn("\U0001f527", edit_text)
        self.assertEqual(runner.state, "idle")


# ---------------------------------------------------------------------------
# Runner long response sends document
# ---------------------------------------------------------------------------

class TestRunnerLongResponseDocument(unittest.IsolatedAsyncioTestCase):
    async def test_long_response_sent_as_document(self):
        agent = AsyncMock()
        messenger = AsyncMock()
        messenger.max_message_length = 4096
        messenger.edit_message = AsyncMock(return_value=True)
        messenger.send_document = AsyncMock(return_value="doc_msg_1")

        runner = SessionRunner(
            "ses_1", "chat_1", None,
            agent, messenger,
        )
        runner.state = "generating"
        runner._msg_id = "msg_1"
        runner._accumulated_text = "x" * 20000

        await runner._finalize_response()

        messenger.send_document.assert_awaited_once()
        self.assertEqual(runner.state, "idle")

    async def test_long_response_falls_back_to_telegraph(self):
        agent = AsyncMock()
        messenger = AsyncMock()
        messenger.max_message_length = 4096
        messenger.edit_message = AsyncMock(return_value=True)
        messenger.send_document = AsyncMock(side_effect=RuntimeError("doc fail"))

        telegraph_called = False
        async def fake_telegraph(title, text):
            nonlocal telegraph_called
            telegraph_called = True
            return "https://telegra.ph/test"

        runner = SessionRunner(
            "ses_1", "chat_1", None,
            agent, messenger,
            long_content_handler=fake_telegraph,
        )
        runner.state = "generating"
        runner._msg_id = "msg_1"
        runner._accumulated_text = "y" * 20000

        await runner._finalize_response()

        self.assertTrue(telegraph_called)
        self.assertEqual(runner.state, "idle")


# ---------------------------------------------------------------------------
# File upload to project directory (bot.py integration)
# ---------------------------------------------------------------------------

class TestFileUploadToProjectDir(unittest.IsolatedAsyncioTestCase):
    async def test_attachments_copied_to_project_dir(self):
        from opencode_tg.bot import _handle_message
        from opencode_tg.protocols import Attachment

        with tempfile.TemporaryDirectory() as project_dir:
            src = tempfile.NamedTemporaryFile(delete=False, suffix=".py")
            src.write(b"print('hello')")
            src.close()

            msg = MagicMock()
            msg.text = "check this"
            msg.sender_id = "123"
            msg.thread_id = None
            msg.attachments = [Attachment(mime="text/x-python", filename="test.py", local_path=src.name)]

            manager = MagicMock()
            manager.get_directory = MagicMock(return_value=project_dir)
            manager.get_model = MagicMock(return_value=None)
            manager.handle_message = AsyncMock()

            messenger = AsyncMock()
            messenger.cleanup_attachment = MagicMock()

            agent = AsyncMock()

            await _handle_message(msg, manager, messenger, agent)

            manager.handle_message.assert_awaited_once()
            parts = manager.handle_message.call_args[0][2]
            file_part = [p for p in parts if p.type == "file"][0]
            self.assertIn(project_dir, file_part.url)
            self.assertTrue(os.path.isfile(os.path.join(project_dir, "test.py")))

            os.unlink(src.name)


# ---------------------------------------------------------------------------
# Path safety for /cat
# ---------------------------------------------------------------------------

class TestCatPathSafety(unittest.IsolatedAsyncioTestCase):
    async def test_cat_rejects_path_traversal(self):
        messenger = AsyncMock()
        messenger.max_message_length = 4096
        manager = MagicMock()
        manager.get_directory = MagicMock(return_value="/home/test/project")
        agent = AsyncMock()

        result = await commands.handle_command("/cat /etc/passwd", "123", None, messenger, manager, agent)
        self.assertTrue(result)
        sent = messenger.send_message.call_args[0][1]
        self.assertIn("Access denied", sent)

    async def test_cat_allows_file_within_project(self):
        messenger = AsyncMock()
        messenger.max_message_length = 4096
        manager = MagicMock()
        agent = AsyncMock()

        tmpdir = tempfile.mkdtemp()
        manager.get_directory = MagicMock(return_value=tmpdir)
        fpath = os.path.join(tmpdir, "ok.txt")
        with open(fpath, "w") as f:
            f.write("allowed")

        try:
            result = await commands.handle_command(f"/cat {fpath}", "123", None, messenger, manager, agent)
            self.assertTrue(result)
            sent = messenger.send_message.call_args[0][1]
            self.assertIn("allowed", sent)
        finally:
            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# /fork and /reject commands
# ---------------------------------------------------------------------------

class TestForkCommand(unittest.IsolatedAsyncioTestCase):
    async def test_fork_creates_new_session(self):
        messenger = AsyncMock()
        messenger.max_message_length = 4096
        manager = MagicMock()
        manager.get_session_id = MagicMock(return_value="ses_1")
        manager.get_directory = MagicMock(return_value=None)
        manager.bind_forked_session = MagicMock()
        agent = AsyncMock()
        agent.fork_session = AsyncMock(return_value="ses_forked_123")

        result = await commands.handle_command("/fork", "123", None, messenger, manager, agent)
        self.assertTrue(result)
        agent.fork_session.assert_awaited_once_with("ses_1", directory=None)
        manager.bind_forked_session.assert_called_once_with("123", None, "ses_forked_123")
        sent = messenger.send_message.call_args[0][1]
        self.assertIn("ses_forked_1", sent)

    async def test_fork_empty_id_rejected(self):
        messenger = AsyncMock()
        messenger.max_message_length = 4096
        manager = MagicMock()
        manager.get_session_id = MagicMock(return_value="ses_1")
        manager.get_directory = MagicMock(return_value=None)
        agent = AsyncMock()
        agent.fork_session = AsyncMock(return_value="")

        result = await commands.handle_command("/fork", "123", None, messenger, manager, agent)
        self.assertTrue(result)
        sent = messenger.send_message.call_args[0][1]
        self.assertIn("empty", sent.lower())

    async def test_fork_no_session(self):
        messenger = AsyncMock()
        messenger.max_message_length = 4096
        manager = MagicMock()
        manager.get_session_id = MagicMock(return_value=None)
        manager.get_directory = MagicMock(return_value=None)
        agent = AsyncMock()

        result = await commands.handle_command("/fork", "123", None, messenger, manager, agent)
        self.assertTrue(result)
        agent.fork_session.assert_not_awaited()


class TestRejectCommand(unittest.IsolatedAsyncioTestCase):
    async def test_reject_sends_request(self):
        messenger = AsyncMock()
        messenger.max_message_length = 4096
        manager = MagicMock()
        manager.get_directory = MagicMock(return_value=None)
        agent = AsyncMock()
        agent.reject_question = AsyncMock()

        result = await commands.handle_command("/reject req_abc", "123", None, messenger, manager, agent)
        self.assertTrue(result)
        agent.reject_question.assert_awaited_once_with("req_abc", directory=None)

    async def test_reject_no_args(self):
        messenger = AsyncMock()
        messenger.max_message_length = 4096
        manager = MagicMock()
        manager.get_directory = MagicMock(return_value=None)
        agent = AsyncMock()

        result = await commands.handle_command("/reject", "123", None, messenger, manager, agent)
        self.assertTrue(result)
        agent.reject_question.assert_not_awaited()


# ---------------------------------------------------------------------------
# Multi-question reply
# ---------------------------------------------------------------------------

class TestMultiQuestionReply(unittest.IsolatedAsyncioTestCase):
    async def test_multi_answer_splits_by_comma(self):
        from opencode_tg.bot import _try_question_reply
        from opencode_tg.protocols import QuestionRequest

        q_event = QuestionRequest(
            request_id="rq1",
            session_id="ses_1",
            questions=[{"id": "a"}, {"id": "b"}],
        )

        runner = MagicMock()
        runner.pending_questions = {"rq1": q_event}
        runner.directory = None

        manager = MagicMock()
        manager.find_runner_for_question = MagicMock(return_value=runner)

        agent = AsyncMock()
        messenger = AsyncMock()

        msg = MagicMock()
        msg.text = "q:rq1:yes, no"
        msg.sender_id = "123"
        msg.thread_id = None

        consumed = await _try_question_reply(msg, manager, agent, messenger)
        self.assertTrue(consumed)
        call_args = agent.reply_question.call_args
        answers = call_args[0][1]
        self.assertEqual(len(answers), 2)
        self.assertEqual(answers[0]["value"], "yes")
        self.assertEqual(answers[1]["value"], "no")


# ---------------------------------------------------------------------------
# ChatKey parse edge case
# ---------------------------------------------------------------------------

class TestChatKeyParseEdge(unittest.TestCase):
    def test_empty_thread_id_is_none(self):
        from opencode_tg.protocols import ChatKey
        ck = ChatKey.parse("123:")
        self.assertEqual(ck.chat_id, "123")
        self.assertIsNone(ck.thread_id)

    def test_none_string_thread_id_is_none(self):
        from opencode_tg.protocols import ChatKey
        ck = ChatKey.parse("123:None")
        self.assertEqual(ck.chat_id, "123")
        self.assertIsNone(ck.thread_id)


# ---------------------------------------------------------------------------
# html_escape includes quotes
# ---------------------------------------------------------------------------

class TestHtmlEscapeQuotes(unittest.TestCase):
    def test_quotes_escaped(self):
        from opencode_tg.formatter import html_escape
        self.assertIn("&quot;", html_escape('say "hello"'))


# ---------------------------------------------------------------------------
# _safe_task error handler
# ---------------------------------------------------------------------------

class TestSafeTask(unittest.IsolatedAsyncioTestCase):
    async def test_safe_task_logs_exception(self):
        from opencode_tg.bot import _safe_task

        async def _boom():
            raise ValueError("test boom")

        with self.assertLogs("opencode_tg.bot", level="ERROR") as cm:
            task = _safe_task(_boom())
            await task
        self.assertTrue(any("test boom" in msg for msg in cm.output))


# ---------------------------------------------------------------------------
# config .env quoting
# ---------------------------------------------------------------------------

class TestConfigEnvQuoting(unittest.TestCase):
    def test_strips_quotes_from_env_values(self):
        from opencode_tg.config import _load_env
        import pathlib
        with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
            f.write('KEY1="quoted_val"\n')
            f.write("KEY2='single_quoted'\n")
            f.write("KEY3=plain\n")
            tmp = f.name
        try:
            result = _load_env(pathlib.Path(tmp))
            self.assertEqual(result["KEY1"], "quoted_val")
            self.assertEqual(result["KEY2"], "single_quoted")
            self.assertEqual(result["KEY3"], "plain")
        finally:
            os.unlink(tmp)


# ---------------------------------------------------------------------------
# Telegraph error handling
# ---------------------------------------------------------------------------

class TestTelegraphErrorHandling(unittest.IsolatedAsyncioTestCase):
    async def test_create_page_raises_on_bad_result(self):
        from opencode_tg.messengers._telegraph import create_page, get_or_create_account
        import aiohttp
        from unittest.mock import patch

        async with aiohttp.ClientSession() as session:
            with patch("opencode_tg.messengers._telegraph.get_or_create_account", return_value="tok"):
                with patch("opencode_tg.messengers._telegraph._telegraph_call", return_value={"ok": False, "error": "bad"}):
                    with self.assertRaises(RuntimeError):
                        await create_page(session, "test", "content")


# ---------------------------------------------------------------------------
# create_session guards bad response
# ---------------------------------------------------------------------------

class TestCreateSessionGuard(unittest.IsolatedAsyncioTestCase):
    async def test_create_session_bad_response_raises(self):
        from opencode_tg.oc_client import OcClient, OcClientError
        from unittest.mock import patch, AsyncMock as AM

        oc = OcClient.__new__(OcClient)
        oc._base = "http://localhost"
        oc._http = None

        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value={"status": "ok"})

        with patch.object(oc, "_request", return_value=mock_resp):
            with self.assertRaises(OcClientError):
                await oc.create_session("test")


# ---------------------------------------------------------------------------
# _is_safe_path with no base
# ---------------------------------------------------------------------------

class TestSafePathNoBase(unittest.TestCase):
    def test_denies_when_no_base(self):
        from opencode_tg.commands import _is_safe_path
        self.assertFalse(_is_safe_path("/etc/passwd", None))


# ---------------------------------------------------------------------------
# Attachment filename sanitization
# ---------------------------------------------------------------------------

class TestAttachmentSanitization(unittest.IsolatedAsyncioTestCase):
    async def test_traversal_filename_sanitized(self):
        from opencode_tg.bot import _handle_message
        from opencode_tg.protocols import Attachment

        with tempfile.TemporaryDirectory() as project_dir:
            src = tempfile.NamedTemporaryFile(delete=False, suffix=".py")
            src.write(b"print('pwn')")
            src.close()

            msg = MagicMock()
            msg.text = "check"
            msg.sender_id = "123"
            msg.thread_id = None
            msg.attachments = [Attachment(mime="text/x-python", filename="../../evil.py", local_path=src.name)]

            manager = MagicMock()
            manager.get_directory = MagicMock(return_value=project_dir)
            manager.get_model = MagicMock(return_value=None)
            manager.handle_message = AsyncMock()

            messenger = AsyncMock()
            messenger.cleanup_attachment = MagicMock()
            agent = AsyncMock()

            await _handle_message(msg, manager, messenger, agent)

            parts = manager.handle_message.call_args[0][2]
            file_part = [p for p in parts if p.type == "file"][0]
            self.assertIn(project_dir, file_part.url)
            self.assertTrue(os.path.isfile(os.path.join(project_dir, "evil.py")))
            self.assertFalse(os.path.exists(os.path.join(project_dir, "..", "..", "evil.py")))

            os.unlink(src.name)


# ---------------------------------------------------------------------------
# Question reply failure keeps question
# ---------------------------------------------------------------------------

class TestQuestionReplyRetry(unittest.IsolatedAsyncioTestCase):
    async def test_failed_reply_keeps_question(self):
        from opencode_tg.bot import _try_question_reply
        from opencode_tg.protocols import QuestionRequest

        q_event = QuestionRequest(
            request_id="rq1",
            session_id="ses_1",
            questions=[{"id": "a"}],
        )

        runner = MagicMock()
        runner.pending_questions = {"rq1": q_event}
        runner.directory = None

        manager = MagicMock()
        manager.find_runner_for_question = MagicMock(return_value=runner)

        agent = AsyncMock()
        agent.reply_question = AsyncMock(side_effect=RuntimeError("fail"))
        messenger = AsyncMock()

        msg = MagicMock()
        msg.text = "q:rq1:answer"
        msg.sender_id = "123"
        msg.thread_id = None

        await _try_question_reply(msg, manager, agent, messenger)
        self.assertIn("rq1", runner.pending_questions)


# ---------------------------------------------------------------------------
# config safe int
# ---------------------------------------------------------------------------

class TestConfigSafeInt(unittest.TestCase):
    def test_bad_value_returns_default(self):
        from opencode_tg.config import _int
        self.assertEqual(_int("NONEXISTENT_KEY_12345"), 0)


if __name__ == "__main__":
    unittest.main()
