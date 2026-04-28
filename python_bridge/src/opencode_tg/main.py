"""CLI entrypoint for the Telegram bot."""
from __future__ import annotations

import asyncio
import sys


def main() -> None:
    from .bot import main as bot_main  # noqa: delayed import
    try:
        asyncio.run(bot_main())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
