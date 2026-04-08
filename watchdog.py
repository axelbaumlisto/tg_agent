"""Periodic health check — idle sweep + dead SSE reconnect."""

from __future__ import annotations

import asyncio
import logging

from . import config
from .session_manager import SessionManager

log = logging.getLogger(__name__)


async def watchdog_loop(manager: SessionManager) -> None:
    """Run forever: periodically sweep idle sessions and reconnect dead SSE."""
    while True:
        await asyncio.sleep(config.WATCHDOG_INTERVAL_SECONDS)
        try:
            await manager.idle_sweep()
            await manager.check_dead_sse()
        except Exception as exc:
            log.error("watchdog error: %s", exc)
