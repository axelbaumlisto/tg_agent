"""Bot commands: /reset /new /model /models /id — protocol-generic."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Optional, TYPE_CHECKING

from .protocols import AgentBackend, Messenger, ModelInfo, ModelRef

if TYPE_CHECKING:
    from .session_manager import SessionManager

log = logging.getLogger(__name__)

_model_menus: dict[str, list[ModelInfo]] = {}


def _chat_key(chat_id: str, thread_id: Optional[str]) -> str:
    return f"{chat_id}:{thread_id}" if thread_id else chat_id


async def handle_command(
    text: str,
    chat_id: str,
    thread_id: Optional[str],
    messenger: Messenger,
    manager: SessionManager,
    agent: AgentBackend,
) -> bool:
    """Handle a slash command. Returns True if consumed."""
    parts = text.strip().split(None, 1)
    cmd = parts[0].lower().split("@")[0]
    args = parts[1].strip() if len(parts) > 1 else ""

    if cmd in ("/reset", "/new"):
        await manager.reset_session(chat_id, thread_id)
        await messenger.send_message(chat_id, "\U0001f504 Session reset.", thread_id)
        return True

    if cmd == "/model":
        if "/" in args:
            provider_id, model_id = args.split("/", 1)
            manager.set_model(chat_id, thread_id, provider_id.strip(), model_id.strip())
            await messenger.send_message(
                chat_id, f"\u2705 Model: `{args}`", thread_id, parse_mode="Markdown",
            )
        else:
            await messenger.send_message(
                chat_id, "Usage: `/model providerID/modelID`\nor use /models and reply with a number.",
                thread_id, parse_mode="Markdown",
            )
        return True

    if cmd == "/models":
        try:
            models = await agent.list_models()
            current = manager.get_model(chat_id, thread_id)
            key = _chat_key(chat_id, thread_id)
            _model_menus[key] = models

            grouped: dict[str, list[tuple[int, ModelInfo]]] = defaultdict(list)
            for idx, m in enumerate(models, 1):
                grouped[m.provider_id].append((idx, m))

            lines = ["Models (+ = current):\n"]
            for provider in sorted(grouped):
                lines.append(provider)
                for idx, m in grouped[provider]:
                    marker = " +" if (current and current.provider_id == m.provider_id
                                      and current.model_id == m.model_id) else ""
                    lines.append(f"  {idx}. {m.name or m.model_id}{marker}")
                lines.append("")
            lines.append("Reply with a number to switch.")

            await messenger.send_message(
                chat_id, "\n".join(lines), thread_id,
            )
        except Exception as exc:
            await messenger.send_message(chat_id, f"\u274c Error: {exc}", thread_id)
        return True

    if cmd == "/id":
        key = f"{chat_id}:{thread_id}" if thread_id else chat_id
        sid = manager.get_session_id(chat_id, thread_id)
        await messenger.send_message(
            chat_id,
            f"chat: `{chat_id}`\nthread: `{thread_id}`\nkey: `{key}`\nsession: `{sid}`",
            thread_id,
            parse_mode="Markdown",
        )
        return True

    return False


async def try_model_selection(
    text: str,
    chat_id: str,
    thread_id: Optional[str],
    messenger: Messenger,
    manager: SessionManager,
) -> bool:
    """If *text* is a bare integer matching a recent /models menu, switch and return True."""
    stripped = text.strip()
    if not stripped.isdigit():
        return False

    key = _chat_key(chat_id, thread_id)
    menu = _model_menus.get(key)
    if not menu:
        return False

    idx = int(stripped)
    if idx < 1 or idx > len(menu):
        return False

    model = menu[idx - 1]
    manager.set_model(chat_id, thread_id, model.provider_id, model.model_id)
    _model_menus.pop(key, None)
    await messenger.send_message(
        chat_id,
        f"\u2705 Model: `{model.provider_id}/{model.model_id}`",
        thread_id,
        parse_mode="Markdown",
    )
    return True
