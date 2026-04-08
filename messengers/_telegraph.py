"""Telegraph page creation for long responses (Telegram-specific)."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import aiohttp

from .. import config

log = logging.getLogger(__name__)

TELEGRAPH_API = "https://api.telegra.ph"


async def _telegraph_call(
    session: aiohttp.ClientSession,
    method: str,
    **params: Any,
) -> dict:
    url = f"{TELEGRAPH_API}/{method}"
    async with session.post(url, json=params) as resp:
        data = await resp.json(content_type=None)
        if not data.get("ok"):
            log.warning("Telegraph %s failed: %s", method, data)
        return data


async def get_or_create_account(session: aiohttp.ClientSession) -> str:
    try:
        token = config.TELEGRAPH_TOKEN_FILE.read_text().strip()
        if token:
            return token
    except OSError:
        pass

    data = await _telegraph_call(
        session,
        "createAccount",
        short_name="AgentBridge-Bot",
        author_name="Agent Bridge Bot",
    )
    token = data["result"]["access_token"]
    config.TELEGRAPH_TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    config.TELEGRAPH_TOKEN_FILE.write_text(token)
    log.info("created Telegraph account, token saved")
    return token


def _md_to_nodes(text: str) -> list:
    nodes: list = []
    parts = re.split(r"(```[\s\S]*?```)", text)
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if part.startswith("```") and part.endswith("```"):
            code = part[3:]
            if "\n" in code:
                code = code[code.index("\n") + 1:]
            code = code.rstrip("`").strip()
            nodes.append({"tag": "pre", "children": [{"tag": "code", "children": [code]}]})
        else:
            for para in part.split("\n\n"):
                para = para.strip()
                if para:
                    nodes.append({"tag": "p", "children": [para]})
    return nodes or [{"tag": "p", "children": [text[:200] or "(empty)"]}]


async def create_page(
    session: aiohttp.ClientSession,
    title: str,
    markdown_text: str,
) -> str:
    token = await get_or_create_account(session)
    nodes = _md_to_nodes(markdown_text)
    data = await _telegraph_call(
        session,
        "createPage",
        access_token=token,
        title=title[:256] or "Response",
        content=nodes,
        author_name="Agent Bridge Bot",
    )
    url: str = data["result"]["url"]
    log.info("created Telegraph page: %s", url)
    return url
