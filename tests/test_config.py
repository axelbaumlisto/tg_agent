"""Tests for config module helpers."""
import unittest

from opencode_tg.config import is_provider_error


class TestIsProviderError(unittest.TestCase):
    def test_insufficient_balance(self):
        self.assertTrue(is_provider_error("Error: insufficient balance"))

    def test_rate_limit(self):
        self.assertTrue(is_provider_error("rate_limit exceeded"))

    def test_invalid_api_key(self):
        self.assertTrue(is_provider_error("Invalid API Key"))

    def test_billing_error(self):
        self.assertTrue(is_provider_error("billing issue"))

    def test_quota_exceeded(self):
        self.assertTrue(is_provider_error("Quota Exceeded for this model"))

    def test_auth_error(self):
        self.assertTrue(is_provider_error("authentication_error: invalid token"))

    def test_normal_error_not_provider(self):
        self.assertFalse(is_provider_error("Connection refused"))

    def test_timeout_not_provider(self):
        self.assertFalse(is_provider_error("Request timed out"))

    def test_empty_string(self):
        self.assertFalse(is_provider_error(""))

    def test_case_insensitive(self):
        self.assertTrue(is_provider_error("INSUFFICIENT_QUOTA for model X"))


if __name__ == "__main__":
    unittest.main()
