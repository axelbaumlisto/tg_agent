"""Main entrypoint — wires messenger + agent backend + session manager."""

from __future__ import annotations

import asyncio
import logging
import os
import sys

from . import config
from .agents.opencode import OpenCodeBackend
from .commands import handle_command, try_model_selection, ALL_COMMANDS
from .messengers.telegram import TelegramMessenger
from .protocols import MessagePart
from .session_manager import SessionManager
from .sessions import SessionStore
from .watchdog import watchdog_loop

log = logging.getLogger(__name__)

BOT_COMMANDS: list[tuple[str, str]] = [
    ("reset", "Reset / start a new session"),
    ("stop", "Stop current generation"),
    ("undo", "Revert last change"),
    ("redo", "Restore reverted change"),
    ("model", "Set model: /model provider/model"),
    ("models", "List available models"),
    ("project", "Set project directory"),
    ("approve", "Toggle auto-approve: on/off"),
    ("diff", "Show session diff"),
    ("git", "Run git command"),
    ("files", "List modified files"),
    ("cat", "Show file contents"),
    ("grep", "Search code for pattern"),
    ("find", "Find files by pattern"),
    ("history", "Show recent messages"),
    ("sessions", "List sessions"),
    ("todo", "Show session TODOs"),
    ("summarize", "Summarize current session"),
    ("agent", "List or use agents"),
    ("tools", "List available tools"),
    ("id", "Show chat/session info"),
]


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
    perm_dir = manager.find_directory_for_session(session_id)
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
            f"Unknown command. Available:\n{ALL_COMMANDS}",
            msg.thread_id,
        )


import re
import shutil

_QUESTION_RE = re.compile(r"^q:([^:]+):(.+)$", re.DOTALL)


async def _try_question_reply(msg, manager, agent, messenger) -> bool:
    """Handle ``q:<request_id>:<answer>`` replies. Returns True if consumed."""
    if not msg.text:
        return False
    m = _QUESTION_RE.match(msg.text.strip())
    if not m:
        return False
    request_id, answer = m.group(1), m.group(2).strip()
    runner = manager.find_runner_for_question(request_id)
    if not runner:
        await messenger.send_message(msg.sender_id, "\u274c Question not found or expired.", msg.thread_id)
        return True
    event = runner.pending_questions.pop(request_id)
    answers = []
    for q in event.questions:
        answers.append({"id": q.get("id", ""), "value": answer})
    directory = runner.directory
    try:
        await agent.reply_question(request_id, answers, directory=directory)
        await messenger.send_message(msg.sender_id, f"\u2705 Answer sent.", msg.thread_id)
    except Exception as exc:
        await messenger.send_message(msg.sender_id, f"\u274c Reply failed: {exc}", msg.thread_id)
    return True


async def _handle_message(msg, manager, messenger, agent):
    if msg.text and await _try_question_reply(msg, manager, agent, messenger):
        return

    if msg.text and await try_model_selection(
        msg.text, msg.sender_id, msg.thread_id, messenger, manager,
    ):
        return

    project_dir = manager.get_directory(msg.sender_id, msg.thread_id)

    parts: list[MessagePart] = []
    copied_paths: list[str] = []
    if msg.text:
        parts.append(MessagePart(type="text", text=msg.text))
    for att in msg.attachments:
        local = str(att.local_path)
        if project_dir and os.path.isdir(project_dir):
            dest = os.path.join(project_dir, att.filename)
            try:
                shutil.copy2(local, dest)
                copied_paths.append(dest)
                local = dest
            except OSError as exc:
                log.warning("copy to project dir failed: %s", exc)
        parts.append(MessagePart(
            type="file", mime=att.mime,
            filename=att.filename, url=f"file://{local}",
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

    await messenger.set_commands(BOT_COMMANDS)

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
                asyncio.create_task(_handle_message(msg, manager, messenger, agent))
    except (KeyboardInterrupt, asyncio.CancelledError):
        log.info("shutting down...")
    finally:
        watchdog.cancel()
        await manager.shutdown_all()
        await agent.close()
        await messenger.close()


if __name__ == "__main__":
    asyncio.run(main())
