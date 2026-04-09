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

@dataclass(frozen=True)
class ChatKey:
    """Unique identifier for a chat/thread pair."""
    chat_id: str
    thread_id: Optional[str] = None

    def __str__(self) -> str:
        return f"{self.chat_id}:{self.thread_id}" if self.thread_id else self.chat_id

    @classmethod
    def parse(cls, key: str) -> "ChatKey":
        if ":" in key:
            chat_id, tid = key.split(":", 1)
            return cls(chat_id, tid if tid != "None" else None)
        return cls(key)


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
    output: str = ""


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
class QuestionRequest:
    """The agent asks the user a clarifying question."""
    request_id: str
    session_id: str
    questions: list[dict] = field(default_factory=list)


@dataclass
class StatusUpdate:
    """Informational status change (busy, retry, …)."""
    status: str
    message: str = ""


AgentEvent = Union[TextDelta, ReasoningDelta, ToolStart, ToolEnd, PermissionRequest, QuestionRequest, SessionIdle, SessionError, StatusUpdate]


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

    async def send_document(
        self,
        recipient: str,
        file_path: str,
        thread_id: Optional[str] = None,
        *,
        caption: Optional[str] = None,
    ) -> str:
        """Send a file/document to the chat. Return msg_id."""
        ...

    async def close(self) -> None:
        """Release resources."""
        ...


# ---------------------------------------------------------------------------
# Agent backend protocol
# ---------------------------------------------------------------------------

class SessionLifecycle(Protocol):
    """Create, delete, abort, and query sessions."""

    async def create_session(self, title: str, *, directory: Optional[str] = None) -> str: ...

    async def delete_session(self, session_id: str, *, directory: Optional[str] = None) -> None: ...

    async def get_session(self, session_id: str, *, directory: Optional[str] = None) -> Optional[dict]: ...

    async def abort_session(self, session_id: str, *, directory: Optional[str] = None) -> None: ...

    async def list_sessions(self, *, limit: int = 10, directory: Optional[str] = None) -> list[dict]: ...

    async def fork_session(self, session_id: str, *, message_id: Optional[str] = None, directory: Optional[str] = None) -> str: ...


class PromptBackend(Protocol):
    """Send prompts and stream agent events."""

    async def send_prompt(
        self,
        session_id: str,
        parts: list[MessagePart],
        model: Optional[ModelRef] = None,
        *,
        directory: Optional[str] = None,
    ) -> None: ...

    async def subscribe_events(
        self, session_id: str, *, directory: Optional[str] = None,
    ) -> AsyncIterator[AgentEvent]: ...

    async def respond_permission(
        self, session_id: str, perm_id: str, response: str,
        *, directory: Optional[str] = None,
    ) -> None: ...

    async def reply_question(self, request_id: str, answers: list[dict], *, directory: Optional[str] = None) -> None: ...

    async def reject_question(self, request_id: str, *, directory: Optional[str] = None) -> None: ...


class SessionHistory(Protocol):
    """Query session messages, diffs, and todos."""

    async def session_messages(self, session_id: str, *, limit: int = 20, directory: Optional[str] = None) -> list[dict]: ...

    async def session_diff(self, session_id: str, *, message_id: Optional[str] = None, directory: Optional[str] = None) -> str: ...

    async def session_todo(self, session_id: str, *, directory: Optional[str] = None) -> list[dict]: ...

    async def summarize_session(self, session_id: str, *, directory: Optional[str] = None) -> str: ...

    async def revert_session(self, session_id: str, *, message_id: Optional[str] = None, directory: Optional[str] = None) -> None: ...

    async def unrevert_session(self, session_id: str, *, directory: Optional[str] = None) -> None: ...


class FileBackend(Protocol):
    """File and code search operations."""

    async def file_list(self, *, directory: Optional[str] = None) -> list[dict]: ...

    async def file_read(self, path: str, *, directory: Optional[str] = None) -> str: ...

    async def file_status(self, *, directory: Optional[str] = None) -> list[dict]: ...

    async def find_text(self, pattern: str, *, directory: Optional[str] = None) -> list[dict]: ...

    async def find_files(self, pattern: str, *, directory: Optional[str] = None) -> list[dict]: ...

    async def vcs_get(self, *, directory: Optional[str] = None) -> dict: ...


@runtime_checkable
class AgentBackend(SessionLifecycle, PromptBackend, SessionHistory, FileBackend, Protocol):
    """Composed protocol: any coding agent that implements all sub-protocols."""

    async def tool_ids(self, *, directory: Optional[str] = None) -> list[str]: ...

    async def app_agents(self, *, directory: Optional[str] = None) -> list[dict]: ...

    async def list_models(self) -> list[ModelInfo]: ...

    async def close(self) -> None: ...
