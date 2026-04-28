"""Tests for CompositeUI and FileDelivery — isolated from SessionRunner."""

from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

from opencode_tg.runners import CompositeUI, FileDelivery


class TestCompositeUIBuild(unittest.TestCase):
    def _make_ui(self) -> CompositeUI:
        messenger = AsyncMock()
        messenger.max_message_length = 4096
        return CompositeUI(messenger, "chat_1", None)

    def test_empty_produces_gear(self):
        ui = self._make_ui()
        ui.rebuild()
        self.assertEqual(ui._accumulated_text, "\u2699\ufe0f")

    def test_reasoning_shows_thought_bubble(self):
        ui = self._make_ui()
        ui.in_reasoning = True
        ui.reasoning_text = "thinking hard"
        ui.rebuild()
        self.assertIn("\U0001f4ad", ui._accumulated_text)
        self.assertIn("thinking hard", ui._accumulated_text)

    def test_tool_lines_in_window(self):
        ui = self._make_ui()
        for i in range(8):
            cid = f"t{i}"
            ui.tool_lines[cid] = f"tool {i}"
            ui.tool_order.append(cid)
        ui.rebuild()
        self.assertNotIn("tool 0", ui._accumulated_text)
        self.assertIn("tool 7", ui._accumulated_text)

    def test_response_text_appended(self):
        ui = self._make_ui()
        ui.response_text = "Answer: 42"
        ui.rebuild()
        self.assertIn("Answer: 42", ui._accumulated_text)

    def test_separator_between_tools_and_response(self):
        ui = self._make_ui()
        ui.tool_lines["c1"] = "tool line"
        ui.tool_order.append("c1")
        ui.response_text = "response"
        ui.rebuild()
        self.assertIn("\u2500", ui._accumulated_text)

    def test_reset_clears_state(self):
        ui = self._make_ui()
        ui.in_reasoning = True
        ui.reasoning_text = "stuff"
        ui.response_text = "answer"
        ui.tool_lines["c1"] = "t"
        ui.tool_order.append("c1")
        ui.reset()
        self.assertFalse(ui.in_reasoning)
        self.assertEqual(ui.reasoning_text, "")
        self.assertEqual(ui.response_text, "")
        self.assertEqual(len(ui.tool_lines), 0)
        self.assertEqual(len(ui.tool_order), 0)

    def test_html_parse_mode_with_tools(self):
        ui = self._make_ui()
        ui.tool_lines["c1"] = "line"
        ui.tool_order.append("c1")
        ui.rebuild()
        self.assertEqual(ui._stream_parse_mode, "HTML")

    def test_plain_parse_mode_text_only(self):
        ui = self._make_ui()
        ui.response_text = "plain text"
        ui.rebuild()
        self.assertIsNone(ui._stream_parse_mode)


class TestCompositeUIFinalize(unittest.IsolatedAsyncioTestCase):
    async def test_finalize_edits_final_text(self):
        messenger = AsyncMock()
        messenger.max_message_length = 4096
        messenger.edit_message = AsyncMock(return_value=True)
        ui = CompositeUI(messenger, "chat_1", None)
        ui.msg_id = "msg_1"
        ui.response_text = "Final answer"
        await ui.finalize()
        messenger.edit_message.assert_awaited()
        text = messenger.edit_message.call_args[0][2]
        self.assertIn("Final answer", text)
        self.assertIsNone(ui.msg_id)

    async def test_finalize_empty_response(self):
        messenger = AsyncMock()
        messenger.max_message_length = 4096
        messenger.edit_message = AsyncMock(return_value=True)
        ui = CompositeUI(messenger, "chat_1", None)
        ui.msg_id = "msg_1"
        await ui.finalize()
        messenger.edit_message.assert_awaited_with("chat_1", "msg_1", "(empty response)")


class TestFileDeliveryIsolated(unittest.IsolatedAsyncioTestCase):
    async def test_delivers_existing_file(self):
        messenger = AsyncMock()
        messenger.send_document = AsyncMock(return_value="doc")
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "test.py")
            with open(path, "w") as f:
                f.write("print('hi')")
            delivery = FileDelivery(messenger, "chat_1", None, d)
            await delivery.try_deliver(path)
            messenger.send_document.assert_awaited_once()

    async def test_skips_unsupported_extension(self):
        messenger = AsyncMock()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "data.bin")
            with open(path, "w") as f:
                f.write("binary")
            delivery = FileDelivery(messenger, "chat_1", None, d)
            await delivery.try_deliver(path)
            messenger.send_document.assert_not_awaited()

    async def test_skips_outside_directory(self):
        messenger = AsyncMock()
        delivery = FileDelivery(messenger, "chat_1", None, "/tmp/safe")
        await delivery.try_deliver("/etc/passwd")
        messenger.send_document.assert_not_awaited()

    async def test_skips_empty_title(self):
        messenger = AsyncMock()
        delivery = FileDelivery(messenger, "chat_1", None, "/tmp")
        await delivery.try_deliver("")
        messenger.send_document.assert_not_awaited()

    async def test_skips_no_directory(self):
        messenger = AsyncMock()
        with patch("opencode_tg.runners.config") as cfg:
            cfg.OC_DIRECTORY = ""
            delivery = FileDelivery(messenger, "chat_1", None, None)
            await delivery.try_deliver("/tmp/test.py")
            messenger.send_document.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
