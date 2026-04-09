"""Tests for ChatKey dataclass and OcClient normalization helpers."""
import unittest

from opencode_tg.protocols import ChatKey
from opencode_tg.oc_client import OcClient


class TestChatKey(unittest.TestCase):
    def test_str_without_thread(self):
        self.assertEqual(str(ChatKey("123")), "123")

    def test_str_with_thread(self):
        self.assertEqual(str(ChatKey("123", "456")), "123:456")

    def test_parse_simple(self):
        ck = ChatKey.parse("123")
        self.assertEqual(ck.chat_id, "123")
        self.assertIsNone(ck.thread_id)

    def test_parse_with_thread(self):
        ck = ChatKey.parse("123:456")
        self.assertEqual(ck.chat_id, "123")
        self.assertEqual(ck.thread_id, "456")

    def test_parse_none_thread(self):
        ck = ChatKey.parse("123:None")
        self.assertEqual(ck.chat_id, "123")
        self.assertIsNone(ck.thread_id)

    def test_frozen(self):
        ck = ChatKey("123", "456")
        with self.assertRaises(AttributeError):
            ck.chat_id = "789"

    def test_equality(self):
        self.assertEqual(ChatKey("a", "b"), ChatKey("a", "b"))
        self.assertNotEqual(ChatKey("a", "b"), ChatKey("a", "c"))

    def test_hashable(self):
        s = {ChatKey("1"), ChatKey("1"), ChatKey("2")}
        self.assertEqual(len(s), 2)

    def test_roundtrip(self):
        ck = ChatKey("abc", "def")
        self.assertEqual(ChatKey.parse(str(ck)), ck)

    def test_roundtrip_no_thread(self):
        ck = ChatKey("abc")
        self.assertEqual(ChatKey.parse(str(ck)), ck)


class TestAsListHelper(unittest.TestCase):
    def test_none_returns_empty(self):
        self.assertEqual(OcClient._as_list(None), [])

    def test_list_passthrough(self):
        self.assertEqual(OcClient._as_list([1, 2, 3]), [1, 2, 3])

    def test_dict_with_matching_key(self):
        self.assertEqual(OcClient._as_list({"items": [1]}, "items"), [1])

    def test_dict_tries_keys_in_order(self):
        data = {"second": [2]}
        self.assertEqual(OcClient._as_list(data, "first", "second"), [2])

    def test_dict_with_nested_dict_values(self):
        data = {"all": {"a": {"id": "a"}, "b": {"id": "b"}}}
        result = OcClient._as_list(data, "all")
        self.assertEqual(len(result), 2)

    def test_dict_no_matching_key(self):
        self.assertEqual(OcClient._as_list({"x": 1}, "y", "z"), [])

    def test_other_type_returns_empty(self):
        self.assertEqual(OcClient._as_list(42), [])


class TestAsStrHelper(unittest.TestCase):
    def test_none_returns_empty(self):
        self.assertEqual(OcClient._as_str(None), "")

    def test_str_passthrough(self):
        self.assertEqual(OcClient._as_str("hello"), "hello")

    def test_dict_with_matching_key(self):
        self.assertEqual(OcClient._as_str({"content": "hi"}, "content"), "hi")

    def test_dict_tries_keys_in_order(self):
        self.assertEqual(OcClient._as_str({"b": "val"}, "a", "b"), "val")

    def test_dict_no_key_returns_json(self):
        result = OcClient._as_str({"x": 1}, "y")
        self.assertIn('"x"', result)

    def test_other_type_str(self):
        self.assertEqual(OcClient._as_str(42), "42")


if __name__ == "__main__":
    unittest.main()
