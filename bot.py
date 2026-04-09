"""Main entrypoint — wires messenger + agent backend + session manager."""

from __future__ import annotations

import asyncio
import logging
import sys

from . import config
from .agents.opencode import OpenCodeBackend
from .commands import handle_command, try_model_selection
from .messengers.telegram import TelegramMessenger
from .protocols import MessagePart
from .session_manager import SessionManager
from .sessions import SessionStore
from .watchdog import watchdog_loop

log = logging.getLogger(__name__)


def _is_allowed(chat_id: str) -> bool:
    if not config.ALLOWED_CHAT_IDS:
        return True
    return chat_id in config.ALLOWED_CHAT_IDS


async def _handle_callback(msg, agent, messenger, manager):
    _RESPONSE_MAP = {"o": "once", "a": "always", "d": "reject"}
    parts = msg.callback_data.split(":", 2)
    if len(parts) != 3 or parts[0] != "p":
        return
    _, short_resp, perm_id = parts
    response = _RESPONSE_MAP.get(short_resp, short_resp)
    session_id = manager.find_session_for_perm(perm_id)
    if not session_id:
        log.warning("no session found for perm %s", perm_id)
        return
    chat_thread = manager._store.find_by_session_id(session_id)
    perm_dir = None
    if chat_thread:
        perm_dir = manager.get_directory(chat_thread[0], chat_thread[1])
    log.info("permission callback: %s %s (session %s)", response, perm_id, session_id)
    try:
        await agent.respond_permission(session_id, perm_id, response, directory=perm_dir)
    except Exception as exc:
        log.warning("permission response failed: %s", exc)
    perm_msg_id = str(msg.raw.get("message", {}).get("message_id", ""))
    await messenger.resolve_permission_ui(msg.sender_id, perm_msg_id, response)


async def _handle_command(msg, messenger, manager, agent):
    consumed = await handle_command(
        msg.text, msg.sender_id, msg.thread_id, messenger, manager, agent,
    )
    if not consumed:
        await messenger.send_message(
            msg.sender_id,
            "Unknown command. Available:\n"
            "/reset /new /model /models /id\n"
            "/project /approve /diff /undo /git",
            msg.thread_id,
        )


async def _handle_message(msg, manager, messenger):
    if msg.text and await try_model_selection(
        msg.text, msg.sender_id, msg.thread_id, messenger, manager,
    ):
        return

    parts: list[MessagePart] = []
    if msg.text:
        parts.append(MessagePart(type="text", text=msg.text))
    for att in msg.attachments:
        parts.append(MessagePart(
            type="file", mime=att.mime,
            filename=att.filename, url=f"file://{att.local_path}",
        ))
    if not parts:
        return

    try:
        model = manager.get_model(msg.sender_id, msg.thread_id)
        await manager.handle_message(msg.sender_id, msg.thread_id, parts, model)
    except Exception as exc:
        log.error("handle_message error: %s", exc, exc_info=True)
        await messenger.send_message(
            msg.sender_id, f"\u274c Error: {exc}", msg.thread_id,
        )
    finally:
        for att in msg.attachments:
            messenger.cleanup_attachment(att)


async def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not config.BOT_TOKEN:
        log.error("TELEGRAM_BOT_TOKEN not set — check .env")
        sys.exit(1)

    agent = OpenCodeBackend(base_url=config.OC_BASE_URL, directory=config.OC_DIRECTORY)
    messenger = TelegramMessenger(bot_token=config.BOT_TOKEN)
    store = SessionStore()

    long_handler = getattr(messenger, "create_telegraph_page", None)
    manager = SessionManager(
        store, agent, messenger,
        long_content_handler=long_handler,
    )
    await manager.startup_reconnect()

    watchdog = asyncio.create_task(watchdog_loop(manager))

    log.info(
        "bot started (messenger=%s, agent=opencode, oc=%s)",
        messenger.name, config.OC_BASE_URL,
    )

    try:
        async for msg in messenger.listen():
            if not _is_allowed(msg.sender_id):
                log.warning("blocked message from unauthorized chat %s", msg.sender_id)
                continue
            if msg.callback_data:
                asyncio.create_task(_handle_callback(msg, agent, messenger, manager))
            elif msg.is_command:
                asyncio.create_task(_handle_command(msg, messenger, manager, agent))
            else:
                asyncio.create_task(_handle_message(msg, manager, messenger))
    except (KeyboardInterrupt, asyncio.CancelledError):
        log.info("shutting down...")
    finally:
        watchdog.cancel()
        await manager.shutdown_all()
        await agent.close()
        await messenger.close()


if __name__ == "__main__":
    asyncio.run(main())
