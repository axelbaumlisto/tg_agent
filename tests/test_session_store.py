"""Pytest-style tests for SessionStore using conftest fixtures."""
from __future__ import annotations

import pathlib

from opencode_tg.sessions import SessionStore


class TestSessionStoreDirectory:
    def test_get_directory_default_none(self, tmp_store: SessionStore):
        assert tmp_store.get_directory("123", None) is None

    def test_set_and_get_directory(self, tmp_store: SessionStore):
        tmp_store.set_directory("123", None, "/home/user/project")
        assert tmp_store.get_directory("123", None) == "/home/user/project"

    def test_set_directory_with_thread(self, tmp_store: SessionStore):
        tmp_store.set_directory("123", "456", "/home/user/frontend")
        assert tmp_store.get_directory("123", "456") == "/home/user/frontend"
        assert tmp_store.get_directory("123", None) is None

    def test_delete_directory(self, tmp_store: SessionStore):
        tmp_store.set_directory("123", None, "/tmp/test")
        tmp_store.delete_directory("123", None)
        assert tmp_store.get_directory("123", None) is None

    def test_directory_persists_with_session(self, tmp_store: SessionStore):
        tmp_store.set("123", None, "ses_1")
        tmp_store.set_directory("123", None, "/projects/alpha")
        assert tmp_store.get("123", None) == "ses_1"
        assert tmp_store.get_directory("123", None) == "/projects/alpha"

    def test_directory_survives_reload(self, tmp_store: SessionStore, tmp_path: pathlib.Path):
        tmp_store.set_directory("123", None, "/srv/app")
        store2 = SessionStore(tmp_path / "sessions.json")
        assert store2.get_directory("123", None) == "/srv/app"


class TestSessionStoreClearSession:
    def test_clear_preserves_directory(self, tmp_store: SessionStore):
        tmp_store.set("123", None, "ses_1")
        tmp_store.set_model("123", None, "openai", "gpt-4o")
        tmp_store.set_directory("123", None, "/projects/alpha")
        tmp_store.set_auto_approve("123", None, True)

        tmp_store.clear_session("123", None)

        assert tmp_store.get("123", None) is None
        assert tmp_store.get_model("123", None) is None
        assert tmp_store.get_directory("123", None) == "/projects/alpha"
        assert tmp_store.get_auto_approve("123", None) is True

    def test_clear_no_entry_is_noop(self, tmp_store: SessionStore):
        tmp_store.clear_session("999", None)

    def test_clear_removes_empty_entry(self, tmp_store: SessionStore):
        tmp_store.set("123", None, "ses_1")
        tmp_store.clear_session("123", None)
        assert tmp_store.get("123", None) is None


class TestSessionStoreAutoApprove:
    def test_default_is_false(self, tmp_store: SessionStore):
        assert tmp_store.get_auto_approve("123", None) is False

    def test_set_true(self, tmp_store: SessionStore):
        tmp_store.set_auto_approve("123", None, True)
        assert tmp_store.get_auto_approve("123", None) is True

    def test_set_false(self, tmp_store: SessionStore):
        tmp_store.set_auto_approve("123", None, True)
        tmp_store.set_auto_approve("123", None, False)
        assert tmp_store.get_auto_approve("123", None) is False

    def test_per_thread_isolation(self, tmp_store: SessionStore):
        tmp_store.set_auto_approve("123", "t1", True)
        assert tmp_store.get_auto_approve("123", None) is False
        assert tmp_store.get_auto_approve("123", "t1") is True

    def test_survives_reload(self, tmp_store: SessionStore, tmp_path: pathlib.Path):
        tmp_store.set_auto_approve("123", None, True)
        store2 = SessionStore(tmp_path / "sessions.json")
        assert store2.get_auto_approve("123", None) is True
