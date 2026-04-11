"""Bot commands — protocol-generic.

Commands:
  Session:  /reset /new /stop /undo /redo
  Model:    /model /models
  Project:  /project /approve /id
  Code:     /diff /git /files /cat /grep /find
  History:  /history /sessions /todo
  Agent:    /agent /tools
"""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional, TYPE_CHECKING

from . import config
from .formatter import html_escape, is_safe_path, send_temp_document
from .protocols import AgentBackend, ChatKey, Messenger, MessagePart, ModelInfo, ModelRef

if TYPE_CHECKING:
    from .session_manager import SessionManager

log = logging.getLogger(__name__)


_model_menus: dict[str, list[ModelInfo]] = {}
_MODEL_MENU_MAX = 50

ALL_COMMANDS = (
    "/reset /new /stop /fork /reject /undo /redo\n"
    "/model /models /id /project /approve\n"
    "/diff /git /files /cat /grep /find\n"
    "/history /sessions /todo /summarize\n"
    "/agent /tools"
)


@dataclass
class CommandContext:
    """All values a command handler may need."""
    args: str
    chat_id: str
    thread_id: Optional[str]
    messenger: Messenger
    manager: "SessionManager"
    agent: AgentBackend
    directory: Optional[str]


CommandHandler = Callable[[CommandContext], Awaitable[bool]]

_REGISTRY: dict[str, CommandHandler] = {}


def command(*names: str):
    """Register a handler for one or more slash command names."""
    def decorator(fn: CommandHandler) -> CommandHandler:
        for n in names:
            _REGISTRY[n] = fn
        return fn
    return decorator


async def _require_session(ctx: CommandContext) -> Optional[str]:
    """Return session_id if active, else send 'No active session' and return None."""
    sid = ctx.manager.get_session_id(ctx.chat_id, ctx.thread_id)
    if not sid:
        await ctx.messenger.send_message(ctx.chat_id, "No active session.", ctx.thread_id)
    return sid


_DOC_THRESHOLD = 3500


async def _send_as_document(ctx: CommandContext, content: str, filename: str, caption: str = "") -> None:
    """Write *content* to a temp file and send as a Telegram document."""
    await send_temp_document(ctx.messenger, ctx.chat_id, ctx.thread_id, content, filename, caption)


def _is_safe_path(path: str, base_dir: Optional[str]) -> bool:
    """Check that *path* is within *base_dir* or config.OC_DIRECTORY."""
    return is_safe_path(path, base_dir or config.OC_DIRECTORY)


async def _delegate_prompt(ctx: CommandContext, prompt_text: str) -> bool:
    """Send a prompt to the agent on behalf of the user (for /git, /grep, /find)."""
    parts = [MessagePart(type="text", text=prompt_text)]
    model = ctx.manager.get_model(ctx.chat_id, ctx.thread_id)
    await ctx.manager.handle_message(ctx.chat_id, ctx.thread_id, parts, model)
    return True


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
    directory = manager.get_directory(chat_id, thread_id)

    handler = _REGISTRY.get(cmd)
    if handler:
        ctx = CommandContext(args, chat_id, thread_id, messenger, manager, agent, directory)
        return await handler(ctx)
    return False


# ---------------------------------------------------------------------------
# Registered command handlers
# ---------------------------------------------------------------------------


@command("/reset", "/new")
async def _handle_reset(ctx: CommandContext) -> bool:
    await ctx.manager.reset_session(ctx.chat_id, ctx.thread_id)
    await ctx.messenger.send_message(ctx.chat_id, "\U0001f504 Session reset.", ctx.thread_id)
    return True


@command("/stop")
async def _handle_stop(ctx: CommandContext) -> bool:
    session_id = await _require_session(ctx)
    if not session_id:
        return True
    try:
        await ctx.agent.abort_session(session_id, directory=ctx.directory)
        await ctx.messenger.send_message(ctx.chat_id, "\u23f9 Generation stopped.", ctx.thread_id)
    except Exception as exc:
        await ctx.messenger.send_message(ctx.chat_id, f"\u274c Stop failed: {exc}", ctx.thread_id)
    return True


@command("/fork")
async def _handle_fork(ctx: CommandContext) -> bool:
    session_id = await _require_session(ctx)
    if not session_id:
        return True
    try:
        new_id = await ctx.agent.fork_session(session_id, directory=ctx.directory)
        if not new_id:
            await ctx.messenger.send_message(ctx.chat_id, "\u274c Fork returned empty session.", ctx.thread_id)
            return True
        ctx.manager.bind_forked_session(ctx.chat_id, ctx.thread_id, new_id)
        await ctx.messenger.send_message(
            ctx.chat_id,
            f"\U0001f500 Forked to new session <code>{html_escape(new_id[:12])}</code>",
            ctx.thread_id,
            parse_mode="HTML",
        )
    except Exception as exc:
        await ctx.messenger.send_message(ctx.chat_id, f"\u274c Fork failed: {exc}", ctx.thread_id)
    return True


@command("/reject")
async def _handle_reject(ctx: CommandContext) -> bool:
    if not ctx.args:
        await ctx.messenger.send_message(
            ctx.chat_id, "Usage: `/reject <request_id>`", ctx.thread_id, parse_mode="Markdown",
        )
        return True
    request_id = ctx.args.strip()
    try:
        await ctx.agent.reject_question(request_id, directory=ctx.directory)
        await ctx.messenger.send_message(ctx.chat_id, "\u2716 Question rejected.", ctx.thread_id)
    except Exception as exc:
        await ctx.messenger.send_message(ctx.chat_id, f"\u274c Reject failed: {exc}", ctx.thread_id)
    return True


@command("/model")
async def _handle_model(ctx: CommandContext) -> bool:
    if "/" in ctx.args:
        provider_id, model_id = ctx.args.split("/", 1)
        ctx.manager.set_model(ctx.chat_id, ctx.thread_id, provider_id.strip(), model_id.strip())
        await ctx.messenger.send_message(
            ctx.chat_id, f"\u2705 Model: `{ctx.args}`", ctx.thread_id, parse_mode="Markdown",
        )
    else:
        await ctx.messenger.send_message(
            ctx.chat_id, "Usage: `/model providerID/modelID`\nor use /models and reply with a number.",
            ctx.thread_id, parse_mode="Markdown",
        )
    return True


@command("/models")
async def _handle_models(ctx: CommandContext) -> bool:
    try:
        models = await ctx.agent.list_models()
        current = ctx.manager.get_model(ctx.chat_id, ctx.thread_id)
        key = str(ChatKey(ctx.chat_id, ctx.thread_id))
        if len(_model_menus) >= _MODEL_MENU_MAX:
            oldest = next(iter(_model_menus))
            del _model_menus[oldest]
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

        await ctx.messenger.send_message(ctx.chat_id, "\n".join(lines), ctx.thread_id)
    except Exception as exc:
        await ctx.messenger.send_message(ctx.chat_id, f"\u274c Error: {exc}", ctx.thread_id)
    return True


@command("/id")
async def _handle_id(ctx: CommandContext) -> bool:
    key = str(ChatKey(ctx.chat_id, ctx.thread_id))
    sid = ctx.manager.get_session_id(ctx.chat_id, ctx.thread_id)
    cur_dir = ctx.directory or config.OC_DIRECTORY
    approve = ctx.manager.get_auto_approve(ctx.chat_id, ctx.thread_id)
    await ctx.messenger.send_message(
        ctx.chat_id,
        f"chat: `{ctx.chat_id}`\nthread: `{ctx.thread_id}`\nkey: `{key}`\n"
        f"session: `{sid}`\ndir: `{cur_dir}`\napprove: `{approve}`",
        ctx.thread_id,
        parse_mode="Markdown",
    )
    return True


@command("/project")
async def _handle_project(ctx: CommandContext) -> bool:
    if not ctx.args:
        cur = ctx.manager.get_directory(ctx.chat_id, ctx.thread_id) or config.OC_DIRECTORY
        await ctx.messenger.send_message(
            ctx.chat_id,
            f"\U0001f4c2 Current directory:\n`{cur}`\n\nUsage: `/project /path/to/dir`",
            ctx.thread_id,
            parse_mode="Markdown",
        )
        return True

    target = os.path.expanduser(ctx.args)
    if not os.path.isabs(target):
        target = os.path.abspath(target)

    if not os.path.isdir(target):
        await ctx.messenger.send_message(
            ctx.chat_id,
            f"\u274c Directory not found: `{target}`",
            ctx.thread_id,
            parse_mode="Markdown",
        )
        return True

    ctx.manager.set_directory(ctx.chat_id, ctx.thread_id, target)
    await ctx.manager.reset_session(ctx.chat_id, ctx.thread_id)
    await ctx.messenger.send_message(
        ctx.chat_id,
        f"\u2705 Project set to `{target}`\nSession reset for new context.",
        ctx.thread_id,
        parse_mode="Markdown",
    )
    return True


@command("/approve")
async def _handle_approve(ctx: CommandContext) -> bool:
    low = ctx.args.lower().strip()
    if low in ("on", "all", "yes", "true", "1"):
        ctx.manager.set_auto_approve(ctx.chat_id, ctx.thread_id, True)
        await ctx.messenger.send_message(
            ctx.chat_id,
            "\u26a0\ufe0f Auto-approve ON — all tool permissions will be granted automatically.",
            ctx.thread_id,
        )
    elif low in ("off", "no", "false", "0"):
        ctx.manager.set_auto_approve(ctx.chat_id, ctx.thread_id, False)
        await ctx.messenger.send_message(
            ctx.chat_id,
            "\U0001f510 Auto-approve OFF — permissions require manual approval.",
            ctx.thread_id,
        )
    else:
        current = ctx.manager.get_auto_approve(ctx.chat_id, ctx.thread_id)
        state = "ON" if current else "OFF"
        await ctx.messenger.send_message(
            ctx.chat_id,
            f"\U0001f510 Auto-approve: {state}\n\nUsage: `/approve on` or `/approve off`",
            ctx.thread_id,
            parse_mode="Markdown",
        )
    return True


@command("/diff")
async def _handle_diff(ctx: CommandContext) -> bool:
    session_id = await _require_session(ctx)
    if not session_id:
        return True
    try:
        content = await ctx.agent.session_diff(session_id, directory=ctx.directory)
    except Exception as exc:
        await ctx.messenger.send_message(ctx.chat_id, f"\u274c Error: {exc}", ctx.thread_id)
        return True
    if not content or not content.strip():
        await ctx.messenger.send_message(
            ctx.chat_id, "\U0001f4ad No pending changes for this session.", ctx.thread_id,
        )
        return True
    if len(content) > _DOC_THRESHOLD:
        await _send_as_document(ctx, content, "changes.diff", caption="Session diff")
    else:
        await ctx.messenger.send_message(
            ctx.chat_id, f"```diff\n{content}\n```", ctx.thread_id, parse_mode="Markdown",
        )
    return True


@command("/undo")
async def _handle_undo(ctx: CommandContext) -> bool:
    session_id = await _require_session(ctx)
    if not session_id:
        return True
    try:
        await ctx.agent.revert_session(session_id, directory=ctx.directory)
        await ctx.messenger.send_message(ctx.chat_id, "\u21a9\ufe0f Changes reverted.", ctx.thread_id)
    except Exception as exc:
        await ctx.messenger.send_message(ctx.chat_id, f"\u274c Revert failed: {exc}", ctx.thread_id)
    return True


@command("/redo")
async def _handle_redo(ctx: CommandContext) -> bool:
    session_id = await _require_session(ctx)
    if not session_id:
        return True
    try:
        await ctx.agent.unrevert_session(session_id, directory=ctx.directory)
        await ctx.messenger.send_message(ctx.chat_id, "\u21aa\ufe0f Changes restored.", ctx.thread_id)
    except Exception as exc:
        await ctx.messenger.send_message(ctx.chat_id, f"\u274c Redo failed: {exc}", ctx.thread_id)
    return True


@command("/git")
async def _handle_git(ctx: CommandContext) -> bool:
    if not ctx.args:
        await ctx.messenger.send_message(
            ctx.chat_id,
            "Usage: `/git status`, `/git diff`, `/git log`, `/git commit -m \"msg\"`, `/git push`",
            ctx.thread_id,
            parse_mode="Markdown",
        )
        return True
    if ctx.args.strip().lower() == "status":
        try:
            vcs = await ctx.agent.vcs_get(directory=ctx.directory)
            branch = vcs.get("branch", vcs.get("head", "?"))
            dirty = vcs.get("dirty", False)
            ahead = vcs.get("ahead", 0)
            behind = vcs.get("behind", 0)
            lines = [f"\U0001f4e6 Branch: `{branch}`"]
            if dirty:
                lines.append("  Modified (dirty)")
            if ahead:
                lines.append(f"  Ahead: {ahead}")
            if behind:
                lines.append(f"  Behind: {behind}")
            changes = vcs.get("changes", vcs.get("files", []))
            if isinstance(changes, list):
                for f in changes[:20]:
                    if isinstance(f, dict):
                        path = f.get("path", f.get("file", "?"))
                        status = f.get("status", "?")
                        lines.append(f"  {status:10s} {path}")
                    else:
                        lines.append(f"  {f}")
                if len(changes) > 20:
                    lines.append(f"  \u2026and {len(changes) - 20} more")
            await ctx.messenger.send_message(
                ctx.chat_id, "\n".join(lines), ctx.thread_id, parse_mode="Markdown",
            )
            return True
        except Exception as exc:
            log.debug("vcs_get failed, falling back to prompt: %s", exc)
    return await _delegate_prompt(ctx, f"Run this git command and show the output: git {ctx.args}")


@command("/files")
async def _handle_files(ctx: CommandContext) -> bool:
    try:
        files = await ctx.agent.file_status(directory=ctx.directory)
        if not files:
            await ctx.messenger.send_message(ctx.chat_id, "No modified files.", ctx.thread_id)
            return True
        if ctx.args == "status":
            lines = ["\U0001f4c4 Modified files:"]
        else:
            lines = [f"\U0001f4c4 Project files ({len(files)}):"]
        for f in files[:50]:
            status = f.get("status", "?")
            path = f.get("path", f.get("file", "?"))
            added = f.get("added", 0)
            removed = f.get("removed", 0)
            stat_str = f"+{added}/-{removed}" if added or removed else ""
            lines.append(f"  {status:10s} {path} {stat_str}")
        if len(files) > 50:
            lines.append(f"  \u2026and {len(files) - 50} more")
        await ctx.messenger.send_message(ctx.chat_id, "\n".join(lines), ctx.thread_id)
    except Exception as exc:
        await ctx.messenger.send_message(ctx.chat_id, f"\u274c Error: {exc}", ctx.thread_id)
    return True


@command("/cat")
async def _handle_cat(ctx: CommandContext) -> bool:
    if not ctx.args:
        await ctx.messenger.send_message(
            ctx.chat_id, "Usage: `/cat path/to/file`", ctx.thread_id, parse_mode="Markdown",
        )
        return True
    path = ctx.args.strip()
    if not os.path.isabs(path):
        base = ctx.directory or config.OC_DIRECTORY or "."
        path = os.path.join(base, path)
    if not _is_safe_path(path, ctx.directory):
        await ctx.messenger.send_message(
            ctx.chat_id, "\u274c Access denied: path outside project directory.",
            ctx.thread_id,
        )
        return True
    try:
        if os.path.isfile(path):
            with open(path, "r", errors="replace") as f:
                content = f.read()
        else:
            await ctx.messenger.send_message(
                ctx.chat_id, f"\u274c File not found: `{ctx.args.strip()}`",
                ctx.thread_id, parse_mode="Markdown",
            )
            return True
        if not content:
            await ctx.messenger.send_message(ctx.chat_id, "(empty file)", ctx.thread_id)
            return True
        if len(content) > _DOC_THRESHOLD:
            await _send_as_document(ctx, content, os.path.basename(path))
        else:
            escaped = html_escape(content)
            await ctx.messenger.send_message(
                ctx.chat_id, f"<pre>{escaped}</pre>", ctx.thread_id, parse_mode="HTML",
            )
    except Exception as exc:
        await ctx.messenger.send_message(ctx.chat_id, f"\u274c Error: {exc}", ctx.thread_id)
    return True


@command("/grep")
async def _handle_grep(ctx: CommandContext) -> bool:
    if not ctx.args:
        await ctx.messenger.send_message(
            ctx.chat_id, "Usage: `/grep pattern`", ctx.thread_id, parse_mode="Markdown",
        )
        return True
    try:
        results = await ctx.agent.find_text(ctx.args.strip(), directory=ctx.directory)
        if not results:
            await ctx.messenger.send_message(ctx.chat_id, "No matches found.", ctx.thread_id)
            return True
        lines = [f"\U0001f50d Results for <code>{html_escape(ctx.args.strip())}</code>:"]
        for r in results[:30]:
            path = html_escape(r.get("file", r.get("path", "?")))
            line_no = r.get("line", "")
            text = html_escape(r.get("text", r.get("content", ""))[:120])
            entry = f"  {path}"
            if line_no:
                entry += f":{line_no}"
            if text:
                entry += f" — {text}"
            lines.append(entry)
        if len(results) > 30:
            lines.append(f"  \u2026and {len(results) - 30} more")
        content = "\n".join(lines)
        if len(content) > _DOC_THRESHOLD:
            await _send_as_document(ctx, content, "grep-results.txt")
        else:
            await ctx.messenger.send_message(ctx.chat_id, content, ctx.thread_id, parse_mode="HTML")
    except Exception as exc:
        await ctx.messenger.send_message(ctx.chat_id, f"\u274c Grep failed: {exc}", ctx.thread_id)
    return True


@command("/find")
async def _handle_find(ctx: CommandContext) -> bool:
    if not ctx.args:
        await ctx.messenger.send_message(
            ctx.chat_id, "Usage: `/find pattern`", ctx.thread_id, parse_mode="Markdown",
        )
        return True
    try:
        results = await ctx.agent.find_files(ctx.args.strip(), directory=ctx.directory)
        if not results:
            await ctx.messenger.send_message(ctx.chat_id, "No files found.", ctx.thread_id)
            return True
        lines = [f"\U0001f4c2 Files matching <code>{html_escape(ctx.args.strip())}</code>:"]
        for r in results[:50]:
            path = html_escape(r.get("file", r.get("path", str(r))))
            lines.append(f"  {path}")
        if len(results) > 50:
            lines.append(f"  \u2026and {len(results) - 50} more")
        content = "\n".join(lines)
        if len(content) > _DOC_THRESHOLD:
            await _send_as_document(ctx, content, "find-results.txt")
        else:
            await ctx.messenger.send_message(ctx.chat_id, content, ctx.thread_id, parse_mode="HTML")
    except Exception as exc:
        await ctx.messenger.send_message(ctx.chat_id, f"\u274c Find failed: {exc}", ctx.thread_id)
    return True


@command("/history")
async def _handle_history(ctx: CommandContext) -> bool:
    session_id = await _require_session(ctx)
    if not session_id:
        return True
    try:
        messages = await ctx.agent.session_messages(session_id, limit=10, directory=ctx.directory)
        if not messages:
            await ctx.messenger.send_message(ctx.chat_id, "No messages in session.", ctx.thread_id)
            return True
        lines = ["\U0001f4dc Recent messages:"]
        for m in messages[-10:]:
            info = m.get("info", m)
            role = info.get("role", "?")
            parts = m.get("parts", [])
            text_parts = [
                p.get("text", "") for p in parts
                if isinstance(p, dict) and p.get("type") == "text" and p.get("text")
            ]
            content = " ".join(text_parts) if text_parts else m.get("content", m.get("text", ""))
            if isinstance(content, list):
                content = " ".join(
                    p.get("text", "") for p in content if isinstance(p, dict) and p.get("text")
                )
            preview = str(content)[:100]
            lines.append(f"  [{role}] {preview}")
        await ctx.messenger.send_message(ctx.chat_id, "\n".join(lines), ctx.thread_id)
    except Exception as exc:
        await ctx.messenger.send_message(ctx.chat_id, f"\u274c Error: {exc}", ctx.thread_id)
    return True


@command("/sessions")
async def _handle_sessions(ctx: CommandContext) -> bool:
    try:
        sessions = await ctx.agent.list_sessions(limit=10, directory=ctx.directory)
        if not sessions:
            await ctx.messenger.send_message(ctx.chat_id, "No sessions found.", ctx.thread_id)
            return True
        lines = ["\U0001f4cb Sessions:"]
        for s in sessions:
            sid = s.get("id", "?")
            title = s.get("title", "")
            status = s.get("status", "?")
            lines.append(f"  `{sid[:8]}` {title} ({status})")
        await ctx.messenger.send_message(ctx.chat_id, "\n".join(lines), ctx.thread_id, parse_mode="Markdown")
    except Exception as exc:
        await ctx.messenger.send_message(ctx.chat_id, f"\u274c Error: {exc}", ctx.thread_id)
    return True


@command("/todo")
async def _handle_todo(ctx: CommandContext) -> bool:
    session_id = await _require_session(ctx)
    if not session_id:
        return True
    try:
        todos = await ctx.agent.session_todo(session_id, directory=ctx.directory)
        if not todos:
            await ctx.messenger.send_message(ctx.chat_id, "No TODOs in session.", ctx.thread_id)
            return True
        lines = ["\u2611\ufe0f TODOs:"]
        for t in todos:
            status = t.get("status", "?")
            content = t.get("content", t.get("text", "?"))
            marker = "\u2705" if status in ("completed", "done") else "\u2b1c"
            lines.append(f"  {marker} {content}")
        await ctx.messenger.send_message(ctx.chat_id, "\n".join(lines), ctx.thread_id)
    except Exception as exc:
        await ctx.messenger.send_message(ctx.chat_id, f"\u274c Error: {exc}", ctx.thread_id)
    return True


@command("/summarize")
async def _handle_summarize(ctx: CommandContext) -> bool:
    session_id = await _require_session(ctx)
    if not session_id:
        return True
    try:
        await ctx.messenger.send_message(ctx.chat_id, "\u23f3 Summarizing session\u2026", ctx.thread_id)
        summary = await ctx.agent.summarize_session(session_id, directory=ctx.directory)
        if not summary or not summary.strip():
            await ctx.messenger.send_message(ctx.chat_id, "No summary available.", ctx.thread_id)
            return True
        max_len = ctx.messenger.max_message_length - 20
        if len(summary) > max_len:
            summary = summary[:max_len] + "\n\u2026(truncated)"
        await ctx.messenger.send_message(ctx.chat_id, summary, ctx.thread_id)
    except Exception as exc:
        await ctx.messenger.send_message(ctx.chat_id, f"\u274c Summarize failed: {exc}", ctx.thread_id)
    return True


@command("/agent")
async def _handle_agent(ctx: CommandContext) -> bool:
    if not ctx.args:
        try:
            agents = await ctx.agent.app_agents(directory=ctx.directory)
            if not agents:
                await ctx.messenger.send_message(ctx.chat_id, "No agents available.", ctx.thread_id)
                return True
            lines = ["\U0001f916 Available agents:"]
            for i, a in enumerate(agents, 1):
                name = a.get("name", a.get("id", f"Agent {i}"))
                desc = a.get("description", "")
                lines.append(f"  {i}. {name}" + (f" — {desc[:60]}" if desc else ""))
            await ctx.messenger.send_message(ctx.chat_id, "\n".join(lines), ctx.thread_id)
        except Exception as exc:
            await ctx.messenger.send_message(ctx.chat_id, f"\u274c Error: {exc}", ctx.thread_id)
        return True
    return await _delegate_prompt(
        ctx, f"[system:agent={ctx.args}] Use the agent named '{ctx.args}' for the next task.",
    )


@command("/tools")
async def _handle_tools(ctx: CommandContext) -> bool:
    try:
        tools = await ctx.agent.tool_ids(directory=ctx.directory)
        if tools:
            lines = [f"\U0001f527 Available tools ({len(tools)}):"]
            for t in tools:
                lines.append(f"  \u2022 {t}")
            await ctx.messenger.send_message(ctx.chat_id, "\n".join(lines), ctx.thread_id)
        else:
            known_tools = [
                "bash", "write", "read", "glob", "grep",
                "fetch", "patch", "todo_read", "todo_write",
            ]
            lines = ["\U0001f527 Standard tools:"]
            for t in known_tools:
                lines.append(f"  \u2022 {t}")
            await ctx.messenger.send_message(ctx.chat_id, "\n".join(lines), ctx.thread_id)
    except Exception as exc:
        await ctx.messenger.send_message(ctx.chat_id, f"\u274c Error: {exc}", ctx.thread_id)
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

    key = str(ChatKey(chat_id, thread_id))
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
