"""Download Telegram media (photo / document / voice) to /tmp."""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Optional

import aiohttp

from . import config

log = logging.getLogger(__name__)

TG_API = "https://api.telegram.org"


async def download_media(
    http: aiohttp.ClientSession,
    file_id: str,
    mime: str,
    filename: str,
) -> dict:
    """Download a Telegram file and return an OpenCode file-part dict.

    Returns ``{"type": "file", "mime": …, "filename": …, "url": "file:///tmp/…"}``.
    """
    url = f"{TG_API}/bot{config.BOT_TOKEN}/getFile"
    async with http.get(url, params={"file_id": file_id}) as resp:
        data = await resp.json(content_type=None)

    file_path = data.get("result", {}).get("file_path", "")
    if not file_path:
        raise RuntimeError(f"getFile returned no file_path for {file_id}")

    ext = Path(filename).suffix or _ext_from_mime(mime)
    local_name = f"oc-tg-{uuid.uuid4().hex[:12]}{ext}"
    local_path = Path("/tmp") / local_name

    download_url = f"{TG_API}/file/bot{config.BOT_TOKEN}/{file_path}"
    async with http.get(download_url) as resp:
        local_path.write_bytes(await resp.read())

    log.debug("downloaded %s → %s", filename, local_path)
    return {
        "type": "file",
        "mime": mime,
        "filename": filename,
        "url": f"file://{local_path}",
    }


def _ext_from_mime(mime: str) -> str:
    mapping = {
        "image/jpeg": ".jpg",
        "image/png": ".png",
        "image/gif": ".gif",
        "image/webp": ".webp",
        "audio/ogg": ".ogg",
        "audio/mpeg": ".mp3",
        "application/pdf": ".pdf",
    }
    return mapping.get(mime, "")
