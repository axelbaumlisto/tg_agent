"""Permission request UI — inline keyboard + callback handler."""

from __future__ import annotations

import logging
from typing import Optional

import aiohttp

from . import config
from .editor import tg_request, send_message
from .oc_client import OcClient

log = logging.getLogger(__name__)


async def send_permission_keyboard(
    http: aiohttp.ClientSession,
    chat_id: int,
    thread_id: Optional[int],
    perm_id: str,
    session_id: str,
    title: str,
    pattern: str = "",
) -> int:
    """Send an inline-keyboard message for a permission request. Returns msg_id."""
    text = f"\U0001f510 *Permission required*\n`{title}`"
    if pattern:
        text += f"\n`{pattern}`"

    keyboard = {
        "inline_keyboard": [[
            {"text": "\u2705 Once", "callback_data": f"perm:once:{session_id}:{perm_id}"},
            {"text": "\u2705 Always", "callback_data": f"perm:always:{session_id}:{perm_id}"},
            {"text": "\u274c Deny", "callback_data": f"perm:reject:{session_id}:{perm_id}"},
        ]]
    }
    data = await send_message(http, chat_id, thread_id, text, reply_markup=keyboard)
    return data.get("result", {}).get("message_id", 0)


async def handle_callback(
    http: aiohttp.ClientSession,
    callback_query: dict,
    oc_client: OcClient,
) -> None:
    """Process a permission callback button press."""
    data_str = callback_query.get("data", "")
    parts = data_str.split(":", 3)
    if len(parts) != 4 or parts[0] != "perm":
        return

    _, response, session_id, perm_id = parts

    try:
        await oc_client.respond_permission(session_id, perm_id, response)
    except Exception as exc:
        log.warning("permission response failed: %s", exc)

    labels = {"once": "\u2705 Approved (once)", "always": "\u2705 Approved (always)", "reject": "\u274c Denied"}
    label = labels.get(response, response)

    msg = callback_query.get("message", {})
    if msg.get("message_id"):
        await tg_request(
            http,
            "editMessageText",
            chat_id=msg["chat"]["id"],
            message_id=msg["message_id"],
            text=f"\U0001f510 {label}",
        )

    await tg_request(http, "answerCallbackQuery", callback_query_id=callback_query["id"])
