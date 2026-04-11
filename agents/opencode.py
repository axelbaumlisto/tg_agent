"""OpenCode agent backend — translates OcClient into the AgentBackend protocol."""

from __future__ import annotations

import json
import logging
from typing import AsyncIterator, Callable, Optional

from ..oc_client import OcClient
from ..protocols import (
    AgentEvent,
    MessagePart,
    ModelInfo,
    ModelRef,
    PermissionInfo,
    PermissionRequest,
    QuestionRequest,
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

    # -- abort / revert / diff -----------------------------------------------

    async def abort_session(self, session_id: str, *, directory: Optional[str] = None) -> None:
        await self._oc.abort_session(session_id, directory=directory)

    async def revert_session(self, session_id: str, *, message_id: Optional[str] = None, directory: Optional[str] = None) -> None:
        await self._oc.revert_session(session_id, message_id=message_id, directory=directory)

    async def unrevert_session(self, session_id: str, *, directory: Optional[str] = None) -> None:
        await self._oc.unrevert_session(session_id, directory=directory)

    async def session_diff(self, session_id: str, *, message_id: Optional[str] = None, directory: Optional[str] = None) -> str:
        return await self._oc.session_diff(session_id, message_id=message_id, directory=directory)

    async def session_messages(self, session_id: str, *, limit: int = 20, directory: Optional[str] = None) -> list[dict]:
        return await self._oc.session_messages(session_id, limit=limit, directory=directory)

    async def list_sessions(self, *, limit: int = 10, directory: Optional[str] = None) -> list[dict]:
        return await self._oc.list_sessions(limit=limit, directory=directory)

    async def fork_session(self, session_id: str, *, message_id: Optional[str] = None, directory: Optional[str] = None) -> str:
        return await self._oc.fork_session(session_id, message_id=message_id, directory=directory)

    async def session_todo(self, session_id: str, *, directory: Optional[str] = None) -> list[dict]:
        return await self._oc.session_todo(session_id, directory=directory)

    async def summarize_session(self, session_id: str, *, directory: Optional[str] = None) -> str:
        return await self._oc.summarize_session(session_id, directory=directory)

    # -- questions -----------------------------------------------------------

    async def reply_question(self, request_id: str, answers: list[dict], *, directory: Optional[str] = None) -> None:
        await self._oc.reply_question(request_id, answers, directory=directory)

    async def reject_question(self, request_id: str, *, directory: Optional[str] = None) -> None:
        await self._oc.reject_question(request_id, directory=directory)

    # -- file / find ---------------------------------------------------------

    async def file_list(self, *, directory: Optional[str] = None) -> list[dict]:
        return await self._oc.file_list(directory=directory)

    async def file_read(self, path: str, *, directory: Optional[str] = None) -> str:
        return await self._oc.file_read(path, directory=directory)

    async def file_status(self, *, directory: Optional[str] = None) -> list[dict]:
        return await self._oc.file_status(directory=directory)

    async def find_text(self, pattern: str, *, directory: Optional[str] = None) -> list[dict]:
        return await self._oc.find_text(pattern, directory=directory)

    async def find_files(self, pattern: str, *, directory: Optional[str] = None) -> list[dict]:
        return await self._oc.find_files(pattern, directory=directory)

    # -- VCS -----------------------------------------------------------------

    async def vcs_get(self, *, directory: Optional[str] = None) -> dict:
        return await self._oc.vcs_get(directory=directory)

    # -- tools / agents -------------------------------------------------------

    async def tool_ids(self, *, directory: Optional[str] = None) -> list[str]:
        return await self._oc.tool_ids(directory=directory)

    async def app_agents(self, *, directory: Optional[str] = None) -> list[dict]:
        return await self._oc.app_agents(directory=directory)

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

            for ev in self.convert_event(raw, session_id, current_part):
                yield ev

    @staticmethod
    def convert_event(
        raw: dict,
        session_id: str,
        current_part: str = "text",
    ) -> list[AgentEvent]:
        etype = raw.get("type", "")
        handler = _EVENT_CONVERTERS.get(etype)
        if handler:
            return handler(raw, session_id, current_part)
        return []

    # -- cleanup -------------------------------------------------------------

    async def close(self) -> None:
        await self._oc.close()


# -- convert_event dispatch handlers ----------------------------------------

def _conv_part_delta(raw: dict, session_id: str, current_part: str) -> list[AgentEvent]:
    delta = raw.get("properties", {}).get("delta", "")
    if not delta:
        return []
    cls = ReasoningDelta if current_part == "reasoning" else TextDelta
    return [cls(text=delta)]


def _conv_part_updated(raw: dict, session_id: str, current_part: str) -> list[AgentEvent]:
    props = raw.get("properties", {})
    part = props.get("part", props)
    ptype = part.get("type", "")
    if ptype not in ("tool-invocation", "tool"):
        return []

    tool_state = part.get("state", {})
    status = tool_state.get("status", "") if isinstance(tool_state, dict) else str(tool_state)
    name = part.get("toolName") or part.get("tool") or "unknown"
    call_id = part.get("callID", part.get("id", ""))

    if status == "running":
        return [ToolStart(name=name, call_id=call_id)]

    if status in ("completed", "result"):
        title = part.get("title", "")
        if not title and isinstance(tool_state, dict):
            title = tool_state.get("title", "")
        output = ""
        if isinstance(tool_state, dict):
            output = tool_state.get("output", tool_state.get("content", ""))
        if not output:
            output = part.get("output", part.get("content", ""))
        if isinstance(output, (dict, list)):
            output = json.dumps(output, ensure_ascii=False)[:500]
        return [ToolEnd(
            name=name, call_id=call_id, state="completed",
            title=title, output=str(output)[:1000] if output else "",
        )]

    if status == "error":
        error = part.get("error", "")
        if not error and isinstance(tool_state, dict):
            error = tool_state.get("error", "")
        return [ToolEnd(name=name, call_id=call_id, state="error", error=error)]

    return []


def _conv_session_idle(raw: dict, session_id: str, current_part: str) -> list[AgentEvent]:
    return [SessionIdle()]


def _conv_session_status(raw: dict, session_id: str, current_part: str) -> list[AgentEvent]:
    props = raw.get("properties", {})
    status = props.get("status", props)
    stype = status.get("type", "") if isinstance(status, dict) else str(status)
    if stype == "idle":
        return [SessionIdle()]
    if stype == "busy":
        return [StatusUpdate(status="busy")]
    if stype == "retry":
        msg = ""
        if isinstance(status, dict):
            attempt = status.get("attempt", "?")
            msg = f"retry #{attempt}: {status.get('message', '')}"
        return [StatusUpdate(status="retry", message=msg)]
    return []


def _conv_permission(raw: dict, session_id: str, current_part: str) -> list[AgentEvent]:
    props = raw.get("properties", {})
    perm_id = props.get("id", "")
    if not perm_id:
        return []
    patterns = props.get("patterns", [])
    pattern_str = ", ".join(patterns) if patterns else props.get("pattern", "")
    return [PermissionRequest(info=PermissionInfo(
        id=perm_id,
        session_id=session_id,
        title=props.get("title", props.get("permission", "")),
        pattern=pattern_str,
    ))]


def _conv_session_error(raw: dict, session_id: str, current_part: str) -> list[AgentEvent]:
    props = raw.get("properties", {})
    return [SessionError(error=props.get("error", str(props)))]


def _conv_question(raw: dict, session_id: str, current_part: str) -> list[AgentEvent]:
    props = raw.get("properties", {})
    req_id = props.get("requestID", props.get("id", ""))
    if not req_id:
        return []
    return [QuestionRequest(
        request_id=req_id,
        session_id=session_id,
        questions=props.get("questions", []),
    )]


_EventConverter = Callable[[dict, str, str], list[AgentEvent]]

_EVENT_CONVERTERS: dict[str, _EventConverter] = {
    "message.part.delta": _conv_part_delta,
    "message.part.updated": _conv_part_updated,
    "session.idle": _conv_session_idle,
    "session.status": _conv_session_status,
    "permission.updated": _conv_permission,
    "permission.asked": _conv_permission,
    "session.error": _conv_session_error,
    "question.asked": _conv_question,
    "question.updated": _conv_question,
}
