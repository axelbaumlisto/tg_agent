"""Bot commands: /reset /new /model /models /id /project /approve /diff /undo /git — protocol-generic."""

from __future__ import annotations

import glob
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Optional, TYPE_CHECKING

from . import config
from .protocols import AgentBackend, Messenger, MessagePart, ModelInfo, ModelRef

if TYPE_CHECKING:
    from .session_manager import SessionManager

log = logging.getLogger(__name__)

_model_menus: dict[str, list[ModelInfo]] = {}

_OPENCODE_STORAGE = Path.home() / ".local" / "share" / "opencode"
_DIFF_DIR = _OPENCODE_STORAGE / "storage" / "session_diff"
_SNAPSHOT_DIR = _OPENCODE_STORAGE / "snapshot"


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
        cur_dir = manager.get_directory(chat_id, thread_id) or config.OC_DIRECTORY
        approve = manager.get_auto_approve(chat_id, thread_id)
        await messenger.send_message(
            chat_id,
            f"chat: `{chat_id}`\nthread: `{thread_id}`\nkey: `{key}`\n"
            f"session: `{sid}`\ndir: `{cur_dir}`\napprove: `{approve}`",
            thread_id,
            parse_mode="Markdown",
        )
        return True

    if cmd == "/project":
        return await _handle_project(args, chat_id, thread_id, messenger, manager)

    if cmd == "/approve":
        return await _handle_approve(args, chat_id, thread_id, messenger, manager)

    if cmd == "/diff":
        return await _handle_diff(chat_id, thread_id, messenger, manager)

    if cmd == "/undo":
        return await _handle_undo(args, chat_id, thread_id, messenger, manager, agent)

    if cmd == "/git":
        return await _handle_git(args, chat_id, thread_id, messenger, manager, agent)

    return False


# ---------------------------------------------------------------------------
# /project
# ---------------------------------------------------------------------------

async def _handle_project(
    args: str,
    chat_id: str,
    thread_id: Optional[str],
    messenger: Messenger,
    manager: SessionManager,
) -> bool:
    if not args:
        cur = manager.get_directory(chat_id, thread_id) or config.OC_DIRECTORY
        await messenger.send_message(
            chat_id,
            f"\U0001f4c2 Current directory:\n`{cur}`\n\nUsage: `/project /path/to/dir`",
            thread_id,
            parse_mode="Markdown",
        )
        return True

    target = os.path.expanduser(args)
    if not os.path.isabs(target):
        target = os.path.abspath(target)

    if not os.path.isdir(target):
        await messenger.send_message(
            chat_id,
            f"\u274c Directory not found: `{target}`",
            thread_id,
            parse_mode="Markdown",
        )
        return True

    manager.set_directory(chat_id, thread_id, target)
    await manager.reset_session(chat_id, thread_id)
    await messenger.send_message(
        chat_id,
        f"\u2705 Project set to `{target}`\nSession reset for new context.",
        thread_id,
        parse_mode="Markdown",
    )
    return True


# ---------------------------------------------------------------------------
# /approve
# ---------------------------------------------------------------------------

async def _handle_approve(
    args: str,
    chat_id: str,
    thread_id: Optional[str],
    messenger: Messenger,
    manager: SessionManager,
) -> bool:
    low = args.lower().strip()
    if low in ("on", "all", "yes", "true", "1"):
        manager.set_auto_approve(chat_id, thread_id, True)
        await messenger.send_message(
            chat_id,
            "\u26a0\ufe0f Auto-approve ON — all tool permissions will be granted automatically.",
            thread_id,
        )
    elif low in ("off", "no", "false", "0"):
        manager.set_auto_approve(chat_id, thread_id, False)
        await messenger.send_message(
            chat_id,
            "\U0001f510 Auto-approve OFF — permissions require manual approval.",
            thread_id,
        )
    else:
        current = manager.get_auto_approve(chat_id, thread_id)
        state = "ON" if current else "OFF"
        await messenger.send_message(
            chat_id,
            f"\U0001f510 Auto-approve: {state}\n\nUsage: `/approve on` or `/approve off`",
            thread_id,
            parse_mode="Markdown",
        )
    return True


# ---------------------------------------------------------------------------
# /diff
# ---------------------------------------------------------------------------

async def _handle_diff(
    chat_id: str,
    thread_id: Optional[str],
    messenger: Messenger,
    manager: SessionManager,
) -> bool:
    session_id = manager.get_session_id(chat_id, thread_id)
    if not session_id:
        await messenger.send_message(chat_id, "No active session.", thread_id)
        return True

    diff_file = _DIFF_DIR / f"{session_id}.diff"
    if not diff_file.exists():
        pattern = str(_DIFF_DIR / f"{session_id}*")
        candidates = sorted(glob.glob(pattern))
        if candidates:
            diff_file = Path(candidates[-1])
        else:
            await messenger.send_message(
                chat_id, "\U0001f4ad No pending changes for this session.", thread_id,
            )
            return True

    try:
        content = diff_file.read_text(errors="replace")
    except OSError as exc:
        await messenger.send_message(chat_id, f"\u274c Error reading diff: {exc}", thread_id)
        return True

    if not content.strip():
        await messenger.send_message(
            chat_id, "\U0001f4ad No pending changes for this session.", thread_id,
        )
        return True

    max_len = messenger.max_message_length - 20
    if len(content) > max_len:
        content = content[:max_len] + "\n\u2026(truncated)"
    await messenger.send_message(
        chat_id, f"```diff\n{content}\n```", thread_id, parse_mode="Markdown",
    )
    return True


# ---------------------------------------------------------------------------
# /undo
# ---------------------------------------------------------------------------

async def _handle_undo(
    args: str,
    chat_id: str,
    thread_id: Optional[str],
    messenger: Messenger,
    manager: SessionManager,
    agent: AgentBackend,
) -> bool:
    session_id = manager.get_session_id(chat_id, thread_id)
    if not session_id:
        await messenger.send_message(chat_id, "No active session.", thread_id)
        return True

    snapshot_dir = _SNAPSHOT_DIR / session_id
    if not snapshot_dir.exists():
        await messenger.send_message(
            chat_id, "\U0001f4ad No snapshots available for undo.", thread_id,
        )
        return True

    snapshots = sorted(snapshot_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    if not snapshots:
        await messenger.send_message(
            chat_id, "\U0001f4ad No snapshots available for undo.", thread_id,
        )
        return True

    prompt_text = f"Revert the last changes. Use the most recent snapshot to undo."
    directory = manager.get_directory(chat_id, thread_id)
    model = manager.get_model(chat_id, thread_id)
    parts = [MessagePart(type="text", text=prompt_text)]
    await manager.handle_message(chat_id, thread_id, parts, model)
    return True


# ---------------------------------------------------------------------------
# /git
# ---------------------------------------------------------------------------

async def _handle_git(
    args: str,
    chat_id: str,
    thread_id: Optional[str],
    messenger: Messenger,
    manager: SessionManager,
    agent: AgentBackend,
) -> bool:
    if not args:
        await messenger.send_message(
            chat_id,
            "Usage: `/git status`, `/git diff`, `/git log`, `/git commit -m \"msg\"`, `/git push`",
            thread_id,
            parse_mode="Markdown",
        )
        return True

    prompt_text = f"Run this git command and show the output: git {args}"
    parts = [MessagePart(type="text", text=prompt_text)]
    model = manager.get_model(chat_id, thread_id)
    await manager.handle_message(chat_id, thread_id, parts, model)
    return True


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
