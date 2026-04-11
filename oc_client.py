"""Async HTTP wrapper for the OpenCode server API.

Covers session lifecycle, prompt submission, permission responses,
provider listing, and SSE event streaming with automatic reconnect.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, AsyncIterator, Optional
from urllib.parse import quote as _url_quote

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
# Exceptions
# ---------------------------------------------------------------------------


class OcClientError(Exception):
    """Non-retryable error from the OpenCode server."""

    def __init__(self, status: int, body: str):
        self.status = status
        self.body = body
        super().__init__(f"OpenCode HTTP {status}: {body}")


class SseStallError(Exception):
    """No useful SSE event for too long despite a live connection."""


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

    def _headers(self, directory: Optional[str] = None) -> dict[str, str]:
        return {"x-opencode-directory": directory or self.directory}

    @staticmethod
    async def _safe_json(resp: aiohttp.ClientResponse) -> Any:
        """Parse JSON from response, returning None for empty/HTML bodies."""
        text = await resp.text()
        stripped = text.strip()
        if not stripped:
            return None
        if stripped.startswith("<!doctype") or stripped.startswith("<html"):
            return None
        try:
            return json.loads(stripped)
        except (json.JSONDecodeError, ValueError):
            return None

    @staticmethod
    def _as_list(data: Any, *keys: str) -> list:
        """Normalize an API response into a list, trying *keys* on dicts."""
        if data is None:
            return []
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for k in keys:
                if k in data:
                    val = data[k]
                    if isinstance(val, list):
                        return val
                    if isinstance(val, dict):
                        return list(val.values())
        return []

    @staticmethod
    def _as_str(data: Any, *keys: str) -> str:
        """Normalize an API response into a string, trying *keys* on dicts."""
        if data is None:
            return ""
        if isinstance(data, str):
            return data
        if isinstance(data, dict):
            for k in keys:
                if k in data:
                    return str(data[k])
            return json.dumps(data, indent=2)
        return str(data)

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: Any = None,
        expect_status: set[int] | None = None,
        directory: Optional[str] = None,
    ) -> aiohttp.ClientResponse:
        session = await self._get_session()
        url = f"{self.base_url}{path}"
        kwargs: dict[str, Any] = {"headers": self._headers(directory)}
        if json_body is not None:
            kwargs["json"] = json_body

        resp = await session.request(method, url, **kwargs)

        ok_statuses = expect_status or {200, 201, 204}
        if resp.status not in ok_statuses and not (200 <= resp.status < 300):
            body = await resp.text()
            raise OcClientError(resp.status, body)
        return resp

    # -- session management -------------------------------------------------

    async def create_session(
        self, title: str = "", *, directory: Optional[str] = None,
    ) -> str:
        """Create a new OpenCode session. Returns the session ID."""
        body: dict[str, Any] = {}
        if title:
            body["title"] = title
        resp = await self._request("POST", "/session", json_body=body, directory=directory)
        data = await resp.json(content_type=None)
        if not isinstance(data, dict) or "id" not in data:
            raise OcClientError(0, f"create_session: unexpected response: {data!r}")
        session_id: str = data["id"]
        log.info("created OC session %s (title=%r)", session_id, title)
        return session_id

    async def get_session(
        self, session_id: str, *, directory: Optional[str] = None,
    ) -> Optional[dict]:
        """Return session info dict, or *None* if 404."""
        session = await self._get_session()
        url = f"{self.base_url}/session/{session_id}"
        async with session.get(url, headers=self._headers(directory)) as resp:
            if resp.status == 404:
                return None
            if resp.status >= 400:
                body = await resp.text()
                raise OcClientError(resp.status, body)
            return await resp.json(content_type=None)

    async def delete_session(
        self, session_id: str, *, directory: Optional[str] = None,
    ) -> None:
        """Delete a session. Idempotent (404 is not an error)."""
        session = await self._get_session()
        url = f"{self.base_url}/session/{session_id}"
        async with session.delete(url, headers=self._headers(directory)) as resp:
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
        *,
        directory: Optional[str] = None,
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
            directory=directory,
        )
        resp.release()
        log.debug("prompt_async sent to session %s", session_id)

    # -- permissions --------------------------------------------------------

    async def respond_permission(
        self,
        session_id: str,
        perm_id: str,
        response: str,
        *,
        directory: Optional[str] = None,
    ) -> None:
        """Approve / deny a permission request.

        *response* must be ``"once"``, ``"always"``, or ``"reject"``.
        """
        resp = await self._request(
            "POST",
            f"/session/{session_id}/permissions/{perm_id}",
            json_body={"response": response},
            directory=directory,
        )
        resp.release()
        log.debug("permission %s → %s (session %s)", perm_id, response, session_id)

    # -- abort / revert / diff -----------------------------------------------

    async def abort_session(
        self, session_id: str, *, directory: Optional[str] = None,
    ) -> None:
        """Abort a running session (stop generation)."""
        resp = await self._request(
            "POST", f"/session/{session_id}/abort",
            expect_status={200, 204}, directory=directory,
        )
        resp.release()
        log.info("aborted session %s", session_id)

    async def revert_session(
        self, session_id: str, *, message_id: Optional[str] = None,
        directory: Optional[str] = None,
    ) -> None:
        """Revert the last (or specific) message changes atomically.

        If *message_id* is not given, fetches messages and uses the last
        assistant message ID (the API requires a valid ``msg_*`` ID).
        """
        if not message_id:
            msgs = await self.session_messages(session_id, directory=directory)
            for m in reversed(msgs):
                info = m.get("info", m)
                if info.get("role") == "assistant":
                    message_id = info.get("id", "")
                    break
            if not message_id:
                raise OcClientError(400, "No assistant message to revert")

        body = {"messageID": message_id}
        resp = await self._request(
            "POST", f"/session/{session_id}/revert",
            json_body=body, expect_status={200, 204},
            directory=directory,
        )
        resp.release()
        log.info("reverted session %s (msg=%s)", session_id, message_id)

    async def unrevert_session(
        self, session_id: str, *, directory: Optional[str] = None,
    ) -> None:
        """Restore previously reverted messages."""
        resp = await self._request(
            "POST", f"/session/{session_id}/unrevert",
            expect_status={200, 204}, directory=directory,
        )
        resp.release()

    async def session_diff(
        self, session_id: str, *, message_id: Optional[str] = None,
        directory: Optional[str] = None,
    ) -> str:
        """Get the diff for a session (or specific message). Returns diff text."""
        params = f"?messageID={_url_quote(message_id, safe='')}" if message_id else ""
        resp = await self._request(
            "GET", f"/session/{session_id}/diff{params}",
            directory=directory,
        )
        data = await self._safe_json(resp)
        if isinstance(data, list):
            parts = []
            for item in data:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    parts.append(self._as_str(item, "diff", "content"))
            return "\n".join(parts)
        return self._as_str(data, "diff", "content")

    async def session_messages(
        self, session_id: str, *, limit: int = 20,
        directory: Optional[str] = None,
    ) -> list[dict]:
        """Get messages for a session (GET /session/{id}/message)."""
        resp = await self._request(
            "GET", f"/session/{session_id}/message",
            directory=directory,
        )
        data = await self._safe_json(resp)
        return self._as_list(data, "messages", "items")

    async def list_sessions(
        self, *, limit: int = 10, directory: Optional[str] = None,
    ) -> list[dict]:
        """List recent sessions."""
        resp = await self._request(
            "GET", f"/session?limit={limit}",
            directory=directory,
        )
        data = await self._safe_json(resp)
        return self._as_list(data, "sessions", "items")

    async def fork_session(
        self, session_id: str, *, message_id: Optional[str] = None,
        directory: Optional[str] = None,
    ) -> str:
        """Fork a session at a specific message. Returns new session_id."""
        body: dict[str, Any] = {}
        if message_id:
            body["messageID"] = message_id
        resp = await self._request(
            "POST", f"/session/{session_id}/fork",
            json_body=body or None, directory=directory,
        )
        data = await self._safe_json(resp) or {}
        return data.get("id", data.get("sessionID", ""))

    async def summarize_session(
        self, session_id: str, *, directory: Optional[str] = None,
    ) -> str:
        """Summarize a session. Returns summary text."""
        resp = await self._request(
            "POST", f"/session/{session_id}/summarize",
            json_body={"auto": True}, directory=directory,
        )
        data = await self._safe_json(resp)
        return self._as_str(data, "summary")

    async def session_todo(
        self, session_id: str, *, directory: Optional[str] = None,
    ) -> list[dict]:
        """Get todo list for a session."""
        resp = await self._request(
            "GET", f"/session/{session_id}/todo",
            directory=directory,
        )
        data = await self._safe_json(resp)
        return self._as_list(data, "todos", "items")

    # -- questions -----------------------------------------------------------

    async def reply_question(
        self, request_id: str, answers: list[dict],
        *, directory: Optional[str] = None,
    ) -> None:
        """Reply to a question from the model."""
        resp = await self._request(
            "POST", f"/question/{request_id}/reply",
            json_body={"answers": answers},
            directory=directory,
        )
        resp.release()

    async def reject_question(
        self, request_id: str, *, directory: Optional[str] = None,
    ) -> None:
        """Reject a question from the model."""
        resp = await self._request(
            "POST", f"/question/{request_id}/reject",
            expect_status={200, 204}, directory=directory,
        )
        resp.release()

    # -- file / find ---------------------------------------------------------

    async def file_list(
        self, *, directory: Optional[str] = None,
    ) -> list[dict]:
        """List files in the project."""
        resp = await self._request("GET", "/file/list", directory=directory)
        data = await self._safe_json(resp)
        return self._as_list(data, "files", "items")

    async def file_read(
        self, path: str, *, directory: Optional[str] = None,
    ) -> str:
        """Read a file's contents."""
        encoded = _url_quote(path, safe="")
        resp = await self._request(
            "GET", f"/file/read?path={encoded}",
            directory=directory,
        )
        data = await self._safe_json(resp)
        return self._as_str(data, "content", "text")

    async def file_status(
        self, *, directory: Optional[str] = None,
    ) -> list[dict]:
        """Get file modification status (like git status)."""
        resp = await self._request("GET", "/file/status", directory=directory)
        data = await self._safe_json(resp)
        return self._as_list(data, "files", "items")

    async def find_text(
        self, pattern: str, *, directory: Optional[str] = None,
    ) -> list[dict]:
        """Search for text in the project (ripgrep)."""
        encoded = _url_quote(pattern, safe="")
        resp = await self._request(
            "GET", f"/find/text?pattern={encoded}",
            directory=directory,
        )
        data = await self._safe_json(resp)
        return self._as_list(data, "results", "matches", "items")

    async def find_files(
        self, pattern: str, *, directory: Optional[str] = None,
    ) -> list[dict]:
        """Find files by name pattern."""
        encoded = _url_quote(pattern, safe="")
        resp = await self._request(
            "GET", f"/find/files?pattern={encoded}",
            directory=directory,
        )
        data = await self._safe_json(resp)
        return self._as_list(data, "files", "items")

    async def find_symbols(
        self, query: str, *, directory: Optional[str] = None,
    ) -> list[dict]:
        """Find symbols (LSP)."""
        encoded = _url_quote(query, safe="")
        resp = await self._request(
            "GET", f"/find/symbols?query={encoded}",
            directory=directory,
        )
        data = await self._safe_json(resp)
        return self._as_list(data, "symbols", "items")

    # -- VCS / worktree -------------------------------------------------------

    async def vcs_get(
        self, *, directory: Optional[str] = None,
    ) -> dict:
        """Get VCS (git) status: branch, changes, etc."""
        resp = await self._request("GET", "/vcs", directory=directory)
        return await self._safe_json(resp) or {}

    async def worktree_list(
        self, *, directory: Optional[str] = None,
    ) -> list[dict]:
        """List git worktrees."""
        resp = await self._request("GET", "/worktree", directory=directory)
        data = await self._safe_json(resp)
        return self._as_list(data, "worktrees", "items")

    async def worktree_create(
        self, *, branch: Optional[str] = None,
        directory: Optional[str] = None,
    ) -> dict:
        """Create a git worktree."""
        body: dict[str, Any] = {}
        if branch:
            body["branch"] = branch
        resp = await self._request(
            "POST", "/worktree", json_body=body, directory=directory,
        )
        return await self._safe_json(resp) or {}

    async def worktree_remove(
        self, worktree_id: str, *, directory: Optional[str] = None,
    ) -> None:
        """Remove a git worktree."""
        resp = await self._request(
            "DELETE", f"/worktree/{worktree_id}",
            expect_status={200, 204}, directory=directory,
        )
        resp.release()

    # -- tools / agents -------------------------------------------------------

    async def tool_ids(
        self, *, directory: Optional[str] = None,
    ) -> list[str]:
        """List available tool IDs."""
        resp = await self._request("GET", "/tool/ids", directory=directory)
        data = await self._safe_json(resp)
        return self._as_list(data, "ids", "tools")

    async def app_agents(
        self, *, directory: Optional[str] = None,
    ) -> list[dict]:
        """List available agents."""
        resp = await self._request("GET", "/app/agents", directory=directory)
        data = await self._safe_json(resp)
        return self._as_list(data, "agents", "items")

    # -- providers ----------------------------------------------------------

    async def list_providers(self) -> list[dict]:
        """GET /provider → list of provider dicts."""
        resp = await self._request("GET", "/provider")
        data = await resp.json(content_type=None)
        return self._as_list(data, "all", "providers", "items")

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
        *,
        directory: Optional[str] = None,
    ) -> AsyncIterator[dict]:
        """Yield parsed SSE event dicts for *session_id*.

        Reconnects automatically with exponential back-off on disconnect.
        After each reconnect, polls ``GET /session/{id}`` and emits a
        synthetic ``session.idle`` if the session finished while disconnected.
        Also emits ``__sse_reconnected`` so upstream layers can reset state.
        """
        backoff = 0.5
        max_backoff = config.RECONNECT_BACKOFF_MAX
        first_connect = True

        while True:
            try:
                async for event in self._sse_stream(session_id, directory=directory):
                    backoff = 0.5
                    yield event
            except (aiohttp.ClientError, asyncio.TimeoutError, SseStallError) as exc:
                log.warning("SSE stream error (%s), reconnecting in %.1fs", exc, backoff)
            except GeneratorExit:
                return

            if not first_connect:
                yield {"type": "__sse_reconnected", "properties": {"sessionID": session_id}}

                try:
                    info = await self.get_session(session_id, directory=directory)
                    status = (info or {}).get("status")
                    if status in (None, "idle"):
                        log.info("session %s is idle after reconnect, emitting synthetic idle", session_id)
                        yield {"type": "session.idle", "properties": {"sessionID": session_id}}
                except Exception:
                    log.debug("failed to poll session %s after reconnect", session_id, exc_info=True)

            first_connect = False
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, max_backoff)

    async def _sse_stream(
        self, session_id: str, *, directory: Optional[str] = None,
    ) -> AsyncIterator[dict]:
        """Single SSE connection; yields events; raises on disconnect."""
        session = await self._get_session()
        url = f"{self.base_url}/event"
        headers = {**self._headers(directory), "Accept": "text/event-stream"}
        stall_timeout = config.STALL_TIMEOUT_SECONDS

        timeout = aiohttp.ClientTimeout(total=None, sock_read=120)
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
            last_useful = time.monotonic()
            async for chunk in resp.content.iter_any():
                buf += chunk.decode("utf-8", errors="replace")
                while "\n" in buf:
                    line, buf = buf.split("\n", 1)
                    line = line.rstrip("\r")
                    if not line:
                        continue
                    if line.startswith(":"):
                        log.debug("SSE keepalive for %s", session_id)
                        if time.monotonic() - last_useful > stall_timeout:
                            raise SseStallError(
                                f"no useful event for {stall_timeout}s (session {session_id})"
                            )
                        continue
                    event = parse_sse_line(line)
                    if event is None:
                        log.debug("SSE unparsed line: %.120s", line)
                        continue
                    props = event.get("properties", {})
                    evt_session = props.get("sessionID", "")
                    if evt_session != session_id:
                        continue
                    last_useful = time.monotonic()
                    yield event
