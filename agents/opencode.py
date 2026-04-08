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

    async def create_session(self, title: str) -> str:
        return await self._oc.create_session(title)

    async def delete_session(self, session_id: str) -> None:
        await self._oc.delete_session(session_id)

    async def get_session(self, session_id: str) -> Optional[dict]:
        return await self._oc.get_session(session_id)

    # -- prompting -----------------------------------------------------------

    async def send_prompt(
        self,
        session_id: str,
        parts: list[MessagePart],
        model: Optional[ModelRef] = None,
    ) -> None:
        raw_parts = [p.to_dict() for p in parts]
        raw_model = model.to_dict() if model else None
        await self._oc.prompt_async(session_id, raw_parts, raw_model)

    # -- permissions ---------------------------------------------------------

    async def respond_permission(
        self, session_id: str, perm_id: str, response: str,
    ) -> None:
        await self._oc.respond_permission(session_id, perm_id, response)

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

    async def subscribe_events(self, session_id: str) -> AsyncIterator[AgentEvent]:
        """Yield typed ``AgentEvent`` objects from the OpenCode SSE stream."""
        async for raw in self._oc.subscribe_events(session_id):
            for ev in self._convert(raw, session_id):
                yield ev

    @staticmethod
    def _convert(raw: dict, session_id: str) -> list[AgentEvent]:
        etype = raw.get("type", "")
        props = raw.get("properties", {})
        out: list[AgentEvent] = []

        if etype == "message.part.delta":
            delta = props.get("delta", "")
            field = props.get("field", "")
            if delta and field in ("text", ""):
                out.append(TextDelta(text=delta))
            elif delta and field == "reasoning":
                out.append(ReasoningDelta(text=delta))

        elif etype == "message.part.updated":
            if props.get("type") != "tool-invocation":
                return out
            state = props.get("state", "")
            name = props.get("toolName", "unknown")
            call_id = props.get("callID", props.get("id", ""))
            if state == "running":
                out.append(ToolStart(name=name, call_id=call_id))
            elif state in ("completed", "result"):
                out.append(ToolEnd(
                    name=name, call_id=call_id, state="completed",
                    title=props.get("title", ""),
                ))
            elif state == "error":
                out.append(ToolEnd(
                    name=name, call_id=call_id, state="error",
                    error=props.get("error", ""),
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

        elif etype == "permission.updated":
            perm_id = props.get("id", "")
            if perm_id:
                out.append(PermissionRequest(info=PermissionInfo(
                    id=perm_id,
                    session_id=session_id,
                    title=props.get("title", props.get("permission", "")),
                    pattern=props.get("pattern", ""),
                )))

        elif etype == "session.error":
            out.append(SessionError(error=props.get("error", str(props))))

        return out

    # -- cleanup -------------------------------------------------------------

    async def close(self) -> None:
        await self._oc.close()
