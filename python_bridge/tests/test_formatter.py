"""Tests for formatter utilities."""
import os
import tempfile
import unittest

from unittest.mock import AsyncMock

from opencode_tg.formatter import html_escape, is_safe_path, send_temp_document, split_chunks


class TestSplitChunks(unittest.TestCase):
    def test_short_text_returns_single_chunk(self):
        result = split_chunks("hello", 100)
        self.assertEqual(result, ["hello"])

    def test_exact_max_returns_single_chunk(self):
        text = "a" * 100
        result = split_chunks(text, 100)
        self.assertEqual(result, [text])

    def test_splits_at_newline(self):
        text = "line1\nline2\nline3"
        result = split_chunks(text, 12)
        self.assertEqual(result[0], "line1\nline2")
        self.assertEqual(result[1], "line3")

    def test_splits_at_max_when_no_newline(self):
        text = "a" * 200
        result = split_chunks(text, 100)
        self.assertEqual(len(result), 2)
        self.assertEqual(len(result[0]), 100)
        self.assertEqual(len(result[1]), 100)

    def test_empty_text(self):
        result = split_chunks("", 100)
        self.assertEqual(result, [""])

    def test_unicode_text(self):
        text = "\u0410" * 50 + "\n" + "\u0411" * 50
        result = split_chunks(text, 60)
        self.assertEqual(len(result), 2)

    def test_many_chunks(self):
        text = "\n".join(f"line{i}" for i in range(50))
        result = split_chunks(text, 30)
        self.assertGreater(len(result), 1)
        for chunk in result:
            self.assertLessEqual(len(chunk), 30)


class TestHtmlEscape(unittest.TestCase):
    def test_escapes_all_specials(self):
        self.assertEqual(html_escape('<a "b">&'), '&lt;a &quot;b&quot;&gt;&amp;')

    def test_plain_text_unchanged(self):
        self.assertEqual(html_escape("hello world"), "hello world")


class TestIsSafePath(unittest.TestCase):
    def test_inside_base(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertTrue(is_safe_path(os.path.join(d, "sub", "file.txt"), d))

    def test_outside_base(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertFalse(is_safe_path("/etc/passwd", d))

    def test_traversal_denied(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertFalse(is_safe_path(os.path.join(d, "..", "etc", "passwd"), d))

    def test_none_base_denied(self):
        self.assertFalse(is_safe_path("/tmp/file", None))

    def test_empty_string_base_denied(self):
        self.assertFalse(is_safe_path("/tmp/file", ""))

    def test_base_equals_path(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertTrue(is_safe_path(d, d))


class TestSendTempDocument(unittest.IsolatedAsyncioTestCase):
    async def test_sends_and_cleans_up(self):
        messenger = AsyncMock()
        messenger.send_document = AsyncMock(return_value="doc_1")
        await send_temp_document(messenger, "chat1", None, "content", "test.txt")
        messenger.send_document.assert_awaited_once()
        call_args = messenger.send_document.call_args
        sent_path = call_args[0][1]
        self.assertFalse(os.path.exists(sent_path))

    async def test_cleanup_on_failure(self):
        messenger = AsyncMock()
        messenger.send_document = AsyncMock(side_effect=RuntimeError("fail"))
        with self.assertRaises(RuntimeError):
            await send_temp_document(messenger, "chat1", None, "content", "test.txt")


if __name__ == "__main__":
    unittest.main()
