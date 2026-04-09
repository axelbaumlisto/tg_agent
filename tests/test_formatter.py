"""Tests for formatter.split_chunks."""
import unittest

from opencode_tg.formatter import split_chunks


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


if __name__ == "__main__":
    unittest.main()
