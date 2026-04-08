"""Async HTTP wrapper for the OpenCode server API.

Covers session lifecycle, prompt submission, permission responses,
provider listing, and SSE event streaming with automatic reconnect.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, AsyncIterator, Optional

import aiohttp

from . import config

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SSE event parser
# ---------------------------------------------------------------------------

def parse_sse_line(line: str) -> Optional[dict]:
    """Parse a single ``data: …`` SSE line into a dict, or *None*."""
    if not line.startswith("data: "):
        return None
    payload = line[6:]
    try:
        return json.loads(payload)
    except (json.JSONDecodeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class OcClientError(Exception):
    """Non-retryable error from the OpenCode server."""

    def __init__(self, status: int, body: str):
        self.status = status
        self.body = body
        super().__init__(f"OpenCode HTTP {status}: {body}")


class OcClient:
    """Async wrapper around the OpenCode HTTP API.

    A single long-lived ``aiohttp.ClientSession`` is created lazily and
    re-used for all requests.  Call :meth:`close` (or use as async
    context-manager) to release it.
    """

    def __init__(
        self,
        base_url: str = config.OC_BASE_URL,
        directory: str = config.OC_DIRECTORY,
    ):
        self.base_url = base_url.rstrip("/")
        self.directory = directory
        self._session: Optional[aiohttp.ClientSession] = None

    # -- lifecycle ----------------------------------------------------------

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=600, connect=5)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def __aenter__(self) -> OcClient:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    # -- helpers ------------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        return {"x-opencode-directory": self.directory}

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: Any = None,
        expect_status: set[int] | None = None,
    ) -> aiohttp.ClientResponse:
        session = await self._get_session()
        url = f"{self.base_url}{path}"
        kwargs: dict[str, Any] = {"headers": self._headers()}
        if json_body is not None:
            kwargs["json"] = json_body

        resp = await session.request(method, url, **kwargs)

        ok_statuses = expect_status or {200, 201, 204}
        if resp.status not in ok_statuses and not (200 <= resp.status < 300):
            body = await resp.text()
            raise OcClientError(resp.status, body)
        return resp

    # -- session management -------------------------------------------------

    async def create_session(self, title: str = "") -> str:
        """Create a new OpenCode session. Returns the session ID."""
        body: dict[str, Any] = {}
        if title:
            body["title"] = title
        resp = await self._request("POST", "/session", json_body=body)
        data = await resp.json(content_type=None)
        session_id: str = data["id"]
        log.info("created OC session %s (title=%r)", session_id, title)
        return session_id

    async def get_session(self, session_id: str) -> Optional[dict]:
        """Return session info dict, or *None* if 404."""
        session = await self._get_session()
        url = f"{self.base_url}/session/{session_id}"
        async with session.get(url, headers=self._headers()) as resp:
            if resp.status == 404:
                return None
            if resp.status >= 400:
                body = await resp.text()
                raise OcClientError(resp.status, body)
            return await resp.json(content_type=None)

    async def delete_session(self, session_id: str) -> None:
        """Delete a session. Idempotent (404 is not an error)."""
        session = await self._get_session()
        url = f"{self.base_url}/session/{session_id}"
        async with session.delete(url, headers=self._headers()) as resp:
            if resp.status not in (200, 204, 404):
                body = await resp.text()
                raise OcClientError(resp.status, body)
        log.info("deleted OC session %s", session_id)

    # -- messaging ----------------------------------------------------------

    async def prompt_async(
        self,
        session_id: str,
        parts: list[dict],
        model: Optional[dict] = None,
    ) -> None:
        """Fire-and-forget prompt (POST prompt_async, expects 204)."""
        body: dict[str, Any] = {"parts": parts}
        if model:
            body["model"] = model
        resp = await self._request(
            "POST",
            f"/session/{session_id}/prompt_async",
            json_body=body,
            expect_status={204, 200},
        )
        resp.release()
        log.debug("prompt_async sent to session %s", session_id)

    # -- permissions --------------------------------------------------------

    async def respond_permission(
        self,
        session_id: str,
        perm_id: str,
        response: str,
    ) -> None:
        """Approve / deny a permission request.

        *response* must be ``"once"``, ``"always"``, or ``"reject"``.
        """
        resp = await self._request(
            "POST",
            f"/session/{session_id}/permissions/{perm_id}",
            json_body={"response": response},
        )
        resp.release()
        log.debug("permission %s → %s (session %s)", perm_id, response, session_id)

    # -- providers ----------------------------------------------------------

    async def list_providers(self) -> list[dict]:
        """GET /provider → list of provider dicts."""
        resp = await self._request("GET", "/provider")
        data = await resp.json(content_type=None)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            all_providers = data.get("all", data.get("providers", data.get("items", {})))
            if isinstance(all_providers, dict):
                return list(all_providers.values())
            if isinstance(all_providers, list):
                return all_providers
        return []

    async def list_connected_providers(self) -> tuple[set[str], list[dict]]:
        """GET /provider → (connected provider IDs, all provider dicts)."""
        resp = await self._request("GET", "/provider")
        data = await resp.json(content_type=None)
        connected: set[str] = set()
        all_providers: list[dict] = []
        if isinstance(data, dict):
            raw_conn = data.get("connected", [])
            if isinstance(raw_conn, list):
                connected = {str(c) for c in raw_conn}
            raw_all = data.get("all", [])
            if isinstance(raw_all, list):
                all_providers = raw_all
            elif isinstance(raw_all, dict):
                all_providers = list(raw_all.values())
        elif isinstance(data, list):
            all_providers = data
        return connected, all_providers

    # -- SSE ----------------------------------------------------------------

    async def subscribe_events(
        self,
        session_id: str,
    ) -> AsyncIterator[dict]:
        """Yield parsed SSE event dicts for *session_id*.

        Reconnects automatically with exponential back-off on disconnect.
        Filter events by ``sessionID`` in ``properties`` where applicable.

        The generator runs until cancelled (``async for`` break or
        ``aclose()``).
        """
        backoff = 0.5
        max_backoff = config.RECONNECT_BACKOFF_MAX

        while True:
            try:
                async for event in self._sse_stream(session_id):
                    backoff = 0.5  # reset on successful event
                    yield event
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                log.warning("SSE stream error (%s), reconnecting in %.1fs", exc, backoff)
            except StopAsyncIteration:
                log.info("SSE stream closed for session %s, reconnecting in %.1fs", session_id, backoff)

            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, max_backoff)

    async def _sse_stream(self, session_id: str) -> AsyncIterator[dict]:
        """Single SSE connection; yields events; raises on disconnect."""
        session = await self._get_session()
        url = f"{self.base_url}/event"
        headers = {**self._headers(), "Accept": "text/event-stream"}

        timeout = aiohttp.ClientTimeout(total=None, sock_read=300)
        async with session.get(url, headers=headers, timeout=timeout) as resp:
            if resp.status != 200:
                body = await resp.text()
                raise aiohttp.ClientResponseError(
                    resp.request_info,
                    resp.history,
                    status=resp.status,
                    message=f"SSE connect failed: {body}",
                )
            log.info("SSE connected for session %s", session_id)

            buf = ""
            async for chunk in resp.content.iter_any():
                buf += chunk.decode("utf-8", errors="replace")
                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    line = line.rstrip("\r")
                    if not line:
                        continue
                    event = parse_sse_line(line)
                    if event is None:
                        continue
                    props = event.get("properties", {})
                    evt_session = props.get("sessionID", "")
                    if evt_session and evt_session != session_id:
                        continue
                    yield event
