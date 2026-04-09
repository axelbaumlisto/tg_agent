"""OpenCode agent backend — translates OcClient into the AgentBackend protocol."""

from __future__ import annotations

import logging
from typing import AsyncIterator, Optional

from ..oc_client import OcClient
from ..protocols import (
    AgentBackend,
    AgentEvent,
    MessagePart,
    ModelInfo,
    ModelRef,
    PermissionInfo,
    PermissionRequest,
    ReasoningDelta,
    SessionError,
    SessionIdle,
    StatusUpdate,
    TextDelta,
    ToolEnd,
    ToolStart,
)

log = logging.getLogger(__name__)


class OpenCodeBackend:
    """``AgentBackend`` implementation backed by an OpenCode HTTP server."""

    def __init__(self, base_url: str, directory: str):
        self._oc = OcClient(base_url=base_url, directory=directory)

    # -- session lifecycle ---------------------------------------------------

    async def create_session(self, title: str, *, directory: Optional[str] = None) -> str:
        return await self._oc.create_session(title, directory=directory)

    async def delete_session(self, session_id: str, *, directory: Optional[str] = None) -> None:
        await self._oc.delete_session(session_id, directory=directory)

    async def get_session(self, session_id: str, *, directory: Optional[str] = None) -> Optional[dict]:
        return await self._oc.get_session(session_id, directory=directory)

    # -- prompting -----------------------------------------------------------

    async def send_prompt(
        self,
        session_id: str,
        parts: list[MessagePart],
        model: Optional[ModelRef] = None,
        *,
        directory: Optional[str] = None,
    ) -> None:
        raw_parts = [p.to_dict() for p in parts]
        raw_model = model.to_dict() if model else None
        await self._oc.prompt_async(session_id, raw_parts, raw_model, directory=directory)

    # -- permissions ---------------------------------------------------------

    async def respond_permission(
        self, session_id: str, perm_id: str, response: str,
        *, directory: Optional[str] = None,
    ) -> None:
        await self._oc.respond_permission(session_id, perm_id, response, directory=directory)

    # -- models --------------------------------------------------------------

    async def list_models(self) -> list[ModelInfo]:
        connected, providers = await self._oc.list_connected_providers()
        result: list[ModelInfo] = []
        for p in providers:
            pid = p.get("id", "?")
            if connected and pid not in connected:
                continue
            result.extend(self._extract_models(pid, p))
        return result

    @staticmethod
    def _extract_models(pid: str, provider: dict) -> list[ModelInfo]:
        result: list[ModelInfo] = []
        models = provider.get("models", {})
        if isinstance(models, dict):
            for mid, info in models.items():
                name = info.get("name", mid) if isinstance(info, dict) else mid
                result.append(ModelInfo(provider_id=pid, model_id=mid, name=name))
        elif isinstance(models, list):
            for m in models:
                result.append(ModelInfo(
                    provider_id=pid,
                    model_id=m.get("id", "?"),
                    name=m.get("name", ""),
                ))
        return result

    # -- event streaming -----------------------------------------------------

    async def subscribe_events(
        self, session_id: str, *, directory: Optional[str] = None,
    ) -> AsyncIterator[AgentEvent]:
        """Yield typed ``AgentEvent`` objects from the OpenCode SSE stream.

        Tracks the current message-part type (``reasoning`` vs ``text``)
        via ``message.part.updated`` events so that ``message.part.delta``
        events — which always arrive with ``field="text"`` — are correctly
        routed to ``ReasoningDelta`` or ``TextDelta``.
        """
        current_part = "text"
        async for raw in self._oc.subscribe_events(session_id, directory=directory):
            etype = raw.get("type", "")

            if etype == "__sse_reconnected":
                log.info("SSE reconnected for %s, resetting current_part", session_id)
                current_part = "text"
                continue

            if etype == "message.part.updated":
                part = raw.get("properties", {}).get("part", {})
                ptype = part.get("type", "")
                if ptype in ("reasoning", "text"):
                    current_part = ptype

            for ev in self._convert(raw, session_id, current_part):
                yield ev

    @staticmethod
    def _convert(
        raw: dict,
        session_id: str,
        current_part: str = "text",
    ) -> list[AgentEvent]:
        etype = raw.get("type", "")
        props = raw.get("properties", {})
        out: list[AgentEvent] = []

        if etype == "message.part.delta":
            delta = props.get("delta", "")
            if delta:
                if current_part == "reasoning":
                    out.append(ReasoningDelta(text=delta))
                else:
                    out.append(TextDelta(text=delta))

        elif etype == "message.part.updated":
            part = props.get("part", props)
            ptype = part.get("type", "")
            if ptype in ("tool-invocation", "tool"):
                tool_state = part.get("state", {})
                status = tool_state.get("status", "") if isinstance(tool_state, dict) else str(tool_state)
                name = part.get("toolName") or part.get("tool") or "unknown"
                call_id = part.get("callID", part.get("id", ""))
                if status == "running":
                    out.append(ToolStart(name=name, call_id=call_id))
                elif status in ("completed", "result"):
                    title = part.get("title", "")
                    if not title and isinstance(tool_state, dict):
                        title = tool_state.get("title", "")
                    out.append(ToolEnd(
                        name=name, call_id=call_id, state="completed",
                        title=title,
                    ))
                elif status == "error":
                    error = part.get("error", "")
                    if not error and isinstance(tool_state, dict):
                        error = tool_state.get("error", "")
                    out.append(ToolEnd(
                        name=name, call_id=call_id, state="error",
                        error=error,
                    ))

        elif etype == "session.idle":
            out.append(SessionIdle())

        elif etype == "session.status":
            status = props.get("status", props)
            stype = status.get("type", "") if isinstance(status, dict) else str(status)
            if stype == "idle":
                out.append(SessionIdle())
            elif stype == "busy":
                out.append(StatusUpdate(status="busy"))
            elif stype == "retry":
                msg = ""
                if isinstance(status, dict):
                    attempt = status.get("attempt", "?")
                    msg = f"retry #{attempt}: {status.get('message', '')}"
                out.append(StatusUpdate(status="retry", message=msg))

        elif etype in ("permission.updated", "permission.asked"):
            perm_id = props.get("id", "")
            if perm_id:
                patterns = props.get("patterns", [])
                pattern_str = ", ".join(patterns) if patterns else props.get("pattern", "")
                out.append(PermissionRequest(info=PermissionInfo(
                    id=perm_id,
                    session_id=session_id,
                    title=props.get("title", props.get("permission", "")),
                    pattern=pattern_str,
                )))

        elif etype == "session.error":
            out.append(SessionError(error=props.get("error", str(props))))

        return out

    # -- cleanup -------------------------------------------------------------

    async def close(self) -> None:
        await self._oc.close()
