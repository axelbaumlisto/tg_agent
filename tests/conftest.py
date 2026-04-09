"""Shared fixtures for opencode_tg tests."""

from __future__ import annotations

import pathlib
import tempfile
from unittest.mock import AsyncMock

import pytest

from opencode_tg.protocols import AgentBackend
from opencode_tg.sessions import SessionStore


@pytest.fixture
def tmp_store(tmp_path: pathlib.Path) -> SessionStore:
    """A fresh SessionStore backed by a temporary JSON file."""
    path = tmp_path / "sessions.json"
    path.write_text("{}")
    return SessionStore(path)


@pytest.fixture
def mock_messenger() -> AsyncMock:
    """An AsyncMock messenger with standard properties set."""
    m = AsyncMock()
    m.max_message_length = 4096
    m.name = "test"
    m.send_message = AsyncMock(return_value="msg_1")
    m.edit_message = AsyncMock(return_value=True)
    m.send_typing = AsyncMock()
    return m


@pytest.fixture
def mock_agent() -> AsyncMock:
    """An AsyncMock agent backend with common defaults."""
    a = AsyncMock(spec=AgentBackend)
    a.create_session = AsyncMock(return_value="ses_test")
    a.get_session = AsyncMock(return_value={"id": "ses_test", "status": "idle"})
    return a
