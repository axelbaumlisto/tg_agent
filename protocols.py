"""Abstract protocols and shared data types for the messenger-agent bridge.

Two extension points:
  - ``Messenger``   — any chat platform (Telegram, Discord, Slack, …)
  - ``AgentBackend`` — any coding agent  (OpenCode, Claude Code, Cursor, …)

Implement one protocol to plug in your own client.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Optional, Protocol, Union, runtime_checkable


# ---------------------------------------------------------------------------
# Shared value types
# ---------------------------------------------------------------------------

@dataclass
class Attachment:
    """A downloaded media file."""
    mime: str
    filename: str
    local_path: Path


@dataclass
class MessagePart:
    """One part of a prompt sent to the agent."""
    type: str  # "text" | "file"
    text: Optional[str] = None
    mime: Optional[str] = None
    filename: Optional[str] = None
    url: Optional[str] = None

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class ModelRef:
    """Provider + model identifier."""
    provider_id: str
    model_id: str

    def to_dict(self) -> dict:
        return {"providerID": self.provider_id, "modelID": self.model_id}


@dataclass
class ModelInfo:
    """One model entry returned by list_models."""
    provider_id: str
    model_id: str
    name: str = ""


@dataclass
class IncomingMessage:
    """A message received from any messenger."""
    sender_id: str
    thread_id: Optional[str]
    text: str
    attachments: list[Attachment] = field(default_factory=list)
    raw: dict = field(default_factory=dict)
    is_command: bool = False
    callback_data: Optional[str] = None


@dataclass
class PermissionInfo:
    """A permission request from the agent."""
    id: str
    session_id: str
    title: str
    pattern: str = ""


# ---------------------------------------------------------------------------
# Agent events (tagged union via dataclass variants)
# ---------------------------------------------------------------------------

@dataclass
class TextDelta:
    """Incremental text from the agent response."""
    text: str


@dataclass
class ReasoningDelta:
    """Incremental reasoning/thinking text from the agent."""
    text: str


@dataclass
class ToolStart:
    """A tool call has started."""
    name: str
    call_id: str


@dataclass
class ToolEnd:
    """A tool call finished."""
    name: str
    call_id: str
    state: str  # "completed" | "error"
    title: str = ""
    error: str = ""


@dataclass
class PermissionRequest:
    """The agent needs user approval."""
    info: PermissionInfo


@dataclass
class SessionIdle:
    """The agent finished generating."""
    pass


@dataclass
class SessionError:
    """The agent hit an error."""
    error: str


@dataclass
class StatusUpdate:
    """Informational status change (busy, retry, …)."""
    status: str
    message: str = ""


AgentEvent = Union[TextDelta, ReasoningDelta, ToolStart, ToolEnd, PermissionRequest, SessionIdle, SessionError, StatusUpdate]


# ---------------------------------------------------------------------------
# Messenger protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class Messenger(Protocol):
    """Any chat platform that can send/receive messages."""

    @property
    def name(self) -> str: ...

    @property
    def max_message_length(self) -> int: ...

    async def send_message(
        self,
        recipient: str,
        text: str,
        thread_id: Optional[str] = None,
        *,
        parse_mode: Optional[str] = None,
        reply_markup: Any = None,
    ) -> str:
        """Send a message. Return the platform message_id (as str)."""
        ...

    async def edit_message(
        self,
        recipient: str,
        message_id: str,
        text: str,
        *,
        parse_mode: Optional[str] = None,
    ) -> bool:
        """Edit an existing message. Return True on success."""
        ...

    async def send_typing(self, recipient: str, thread_id: Optional[str] = None) -> None:
        """Show a typing/processing indicator."""
        ...

    async def download_attachment(self, raw_attachment: dict) -> Attachment:
        """Download a platform attachment to a local file."""
        ...

    async def listen(self) -> AsyncIterator[IncomingMessage]:
        """Yield incoming messages. Platform-specific transport."""
        ...

    async def send_permission_request(
        self,
        recipient: str,
        thread_id: Optional[str],
        perm: PermissionInfo,
    ) -> str:
        """Show a permission prompt. Return the msg_id for later update."""
        ...

    async def resolve_permission_ui(self, recipient: str, message_id: str, result: str) -> None:
        """Update the permission UI after user response."""
        ...

    async def close(self) -> None:
        """Release resources."""
        ...


# ---------------------------------------------------------------------------
# Agent backend protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class AgentBackend(Protocol):
    """Any coding agent that can run sessions, accept prompts, and stream events."""

    async def create_session(self, title: str) -> str:
        """Create a new agent session. Return session_id."""
        ...

    async def delete_session(self, session_id: str) -> None: ...

    async def get_session(self, session_id: str) -> Optional[dict]: ...

    async def send_prompt(
        self,
        session_id: str,
        parts: list[MessagePart],
        model: Optional[ModelRef] = None,
    ) -> None:
        """Fire-and-forget prompt submission."""
        ...

    async def subscribe_events(self, session_id: str) -> AsyncIterator[AgentEvent]:
        """Stream typed events from the agent for this session."""
        ...

    async def respond_permission(self, session_id: str, perm_id: str, response: str) -> None: ...

    async def list_models(self) -> list[ModelInfo]: ...

    async def close(self) -> None: ...
