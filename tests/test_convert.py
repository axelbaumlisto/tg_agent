"""Test that OpenCodeBackend._convert correctly routes reasoning vs text deltas."""
from opencode_tg.agents.opencode import OpenCodeBackend
from opencode_tg.protocols import ReasoningDelta, TextDelta, SessionIdle


SID = "ses_test"


def _delta_event(text: str) -> dict:
    return {"type": "message.part.delta", "properties": {"delta": text, "field": "text"}}


def _part_updated(part_type: str, text: str = "") -> dict:
    return {
        "type": "message.part.updated",
        "properties": {
            "sessionID": SID,
            "part": {"type": part_type, "text": text, "id": "prt_1", "sessionID": SID, "messageID": "msg_1"},
        },
    }


def test_delta_defaults_to_text():
    events = OpenCodeBackend._convert(_delta_event("hello"), SID, current_part="text")
    assert len(events) == 1
    assert isinstance(events[0], TextDelta)
    assert events[0].text == "hello"


def test_delta_as_reasoning():
    events = OpenCodeBackend._convert(_delta_event("thinking..."), SID, current_part="reasoning")
    assert len(events) == 1
    assert isinstance(events[0], ReasoningDelta)
    assert events[0].text == "thinking..."


def test_empty_delta_ignored():
    events = OpenCodeBackend._convert(_delta_event(""), SID, current_part="text")
    assert events == []


def test_part_updated_does_not_emit_text_events():
    events = OpenCodeBackend._convert(_part_updated("reasoning"), SID)
    assert all(not isinstance(e, (TextDelta, ReasoningDelta)) for e in events)


def test_session_idle():
    raw = {"type": "session.idle", "properties": {}}
    events = OpenCodeBackend._convert(raw, SID)
    assert len(events) == 1
    assert isinstance(events[0], SessionIdle)
