"""Tests for watchdog_loop."""
import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from opencode_tg.watchdog import watchdog_loop

_real_sleep = asyncio.sleep


class TestWatchdogLoop(unittest.IsolatedAsyncioTestCase):
    async def test_calls_sweep_methods(self):
        manager = AsyncMock()
        manager.idle_sweep = AsyncMock()
        manager.check_dead_sse = AsyncMock()
        manager.check_stalled_generating = AsyncMock()

        call_count = 0

        async def fast_sleep(delay):
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                raise asyncio.CancelledError
            await _real_sleep(0)

        with patch("opencode_tg.watchdog.asyncio.sleep", side_effect=fast_sleep):
            with self.assertRaises(asyncio.CancelledError):
                await watchdog_loop(manager)

        manager.idle_sweep.assert_awaited()
        manager.check_dead_sse.assert_awaited()
        manager.check_stalled_generating.assert_awaited()

    async def test_survives_exception(self):
        manager = AsyncMock()
        manager.idle_sweep = AsyncMock(side_effect=RuntimeError("oops"))
        manager.check_dead_sse = AsyncMock()
        manager.check_stalled_generating = AsyncMock()

        sleep_count = 0

        async def fast_sleep(delay):
            nonlocal sleep_count
            sleep_count += 1
            if sleep_count >= 3:
                raise asyncio.CancelledError
            await _real_sleep(0)

        with patch("opencode_tg.watchdog.asyncio.sleep", side_effect=fast_sleep):
            with self.assertRaises(asyncio.CancelledError):
                await watchdog_loop(manager)

        self.assertGreaterEqual(manager.idle_sweep.await_count, 2)


if __name__ == "__main__":
    unittest.main()
