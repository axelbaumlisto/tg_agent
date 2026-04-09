"""Tests for bot module functions."""
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from opencode_tg.bot import _is_allowed, _handle_callback


class TestIsAllowed(unittest.TestCase):
    @patch("opencode_tg.bot.config")
    def test_allows_all_when_empty_set(self, mock_config):
        mock_config.ALLOWED_CHAT_IDS = set()
        self.assertTrue(_is_allowed("12345"))

    @patch("opencode_tg.bot.config")
    def test_allows_listed_chat(self, mock_config):
        mock_config.ALLOWED_CHAT_IDS = {"123", "456"}
        self.assertTrue(_is_allowed("123"))

    @patch("opencode_tg.bot.config")
    def test_blocks_unlisted_chat(self, mock_config):
        mock_config.ALLOWED_CHAT_IDS = {"123", "456"}
        self.assertFalse(_is_allowed("789"))


class TestHandleCallback(unittest.IsolatedAsyncioTestCase):
    async def test_permission_callback_flow(self):
        msg = MagicMock()
        msg.callback_data = "p:a:perm_1"
        msg.sender_id = "chat_1"
        msg.raw = {"message": {"message_id": 42}}

        agent = AsyncMock()
        messenger = AsyncMock()
        manager = MagicMock()
        manager.find_session_for_perm = MagicMock(return_value="ses_1")
        manager.find_directory_for_session = MagicMock(return_value="/proj")

        await _handle_callback(msg, agent, messenger, manager)

        agent.respond_permission.assert_awaited_once_with(
            "ses_1", "perm_1", "always", directory="/proj",
        )
        messenger.resolve_permission_ui.assert_awaited_once()

    async def test_invalid_callback_data_ignored(self):
        msg = MagicMock()
        msg.callback_data = "invalid_data"

        agent = AsyncMock()
        messenger = AsyncMock()
        manager = MagicMock()

        await _handle_callback(msg, agent, messenger, manager)

        agent.respond_permission.assert_not_awaited()

    async def test_unknown_perm_id_logs_warning(self):
        msg = MagicMock()
        msg.callback_data = "p:o:perm_unknown"
        msg.sender_id = "chat_1"
        msg.raw = {}

        agent = AsyncMock()
        messenger = AsyncMock()
        manager = MagicMock()
        manager.find_session_for_perm = MagicMock(return_value=None)

        await _handle_callback(msg, agent, messenger, manager)
        agent.respond_permission.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
