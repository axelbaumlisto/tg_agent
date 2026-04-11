"""Shared mock OpenCode HTTP server for unit/integration tests."""

from __future__ import annotations

import json

from aiohttp import web


def build_mock_app() -> web.Application:
    """Build an aiohttp app that mimics the OpenCode server API."""
    app = web.Application()

    async def create_session(request: web.Request) -> web.Response:
        body = await request.json()
        directory = request.headers.get("x-opencode-directory", "")
        return web.json_response({"id": "ses_mock_1", "directory": directory})

    async def get_session(request: web.Request) -> web.Response:
        sid = request.match_info["sid"]
        if sid == "ses_missing":
            return web.Response(status=404)
        return web.json_response({"id": sid, "title": "mock"})

    async def delete_session(request: web.Request) -> web.Response:
        sid = request.match_info["sid"]
        if sid == "ses_gone":
            return web.Response(status=404)
        return web.Response(status=200)

    async def prompt_async(request: web.Request) -> web.Response:
        await request.json()
        return web.Response(status=204)

    async def respond_permission(request: web.Request) -> web.Response:
        await request.json()
        return web.json_response(True)

    async def list_providers(request: web.Request) -> web.Response:
        return web.json_response([
            {"id": "openai", "models": [{"id": "gpt-4o"}]},
            {"id": "anthropic", "models": [{"id": "claude-sonnet-4-20250514"}]},
        ])

    async def sse_events(request: web.Request) -> web.StreamResponse:
        resp = web.StreamResponse(
            status=200,
            headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"},
        )
        await resp.prepare(request)
        events = [
            {"type": "server.connected", "properties": {"sessionID": "ses_mock_1"}},
            {
                "type": "message.part.delta",
                "properties": {"sessionID": "ses_mock_1", "field": "text", "delta": "Hello"},
            },
            {
                "type": "message.part.updated",
                "properties": {
                    "sessionID": "ses_mock_1",
                    "type": "tool-invocation",
                    "toolName": "bash",
                    "callID": "c1",
                    "state": "running",
                },
            },
            {
                "type": "message.part.updated",
                "properties": {
                    "sessionID": "ses_mock_1",
                    "type": "tool-invocation",
                    "toolName": "bash",
                    "callID": "c1",
                    "state": "completed",
                    "title": "done",
                },
            },
            {"type": "session.idle", "properties": {"sessionID": "ses_mock_1"}},
        ]
        for ev in events:
            line = f"data: {json.dumps(ev)}\n\n"
            await resp.write(line.encode())
        return resp

    async def abort_session(request: web.Request) -> web.Response:
        return web.Response(status=200)

    async def session_messages(request: web.Request) -> web.Response:
        return web.json_response([{"role": "user", "content": "hi"}])

    async def session_diff(request: web.Request) -> web.Response:
        return web.json_response({"diff": "--- a\n+++ b"})

    async def list_sessions(request: web.Request) -> web.Response:
        return web.json_response([{"id": "ses_mock_1", "title": "mock"}])

    async def fork_session(request: web.Request) -> web.Response:
        try:
            await request.json()
        except Exception:
            pass
        return web.json_response({"id": "ses_fork_1"})

    async def file_status(request: web.Request) -> web.Response:
        return web.json_response([{"path": "main.py", "status": "modified"}])

    async def find_text(request: web.Request) -> web.Response:
        return web.json_response([{"file": "main.py", "line": 1, "text": "match"}])

    async def find_files(request: web.Request) -> web.Response:
        return web.json_response([{"file": "main.py"}])

    app.router.add_post("/session", create_session)
    app.router.add_get("/session/{sid}", get_session)
    app.router.add_delete("/session/{sid}", delete_session)
    app.router.add_post("/session/{sid}/prompt_async", prompt_async)
    app.router.add_post("/session/{sid}/permissions/{pid}", respond_permission)
    app.router.add_post("/session/{sid}/abort", abort_session)
    app.router.add_get("/session/{sid}/message", session_messages)
    app.router.add_get("/session/{sid}/diff", session_diff)
    app.router.add_get("/session", list_sessions)
    app.router.add_post("/session/{sid}/fork", fork_session)
    app.router.add_get("/file/status", file_status)
    app.router.add_get("/find/text", find_text)
    app.router.add_get("/find/files", find_files)
    app.router.add_get("/provider", list_providers)
    app.router.add_get("/event", sse_events)
    return app
