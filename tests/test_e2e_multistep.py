"""Parallel multi-step E2E: two 10-step conversations in bot-chat topics.

Scenario A (RU, topic 10542): programmable keyboards
Scenario B (EN, topic 10545): programmable mice

Each scenario starts with /reset to ensure clean context.
Both run concurrently via asyncio.gather.

Requires:
  - Bot running: python3 -m opencode_tg.bot
  - OpenCode server on :14096
  - Telethon session: research_session (user 6196099449)

Run:
  python3 -m pytest opencode_tg/tests/test_e2e_multistep.py -v -s --timeout=1200
"""
from __future__ import annotations

import asyncio
import logging
import time
import unittest
from dataclasses import dataclass, field
from typing import Callable

from telethon import TelegramClient
from telethon.tl.types import Message

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_ID = 38309428
API_HASH = "1f9a006d55531cfd387246cd0fff83f8"
SESSION = "/home/spex/.zeroclaw/workspace/skills/telegram-reader/.session/research_session"
BOT = "zGsR_bot"
TOPIC_RU = 10542
TOPIC_EN = 10545

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

Validator = Callable[[str, list[Message]], None]


@dataclass
class Step:
    prompt: str
    checks: list[Validator]
    timeout: int = 120


@dataclass
class StepResult:
    index: int
    prompt: str
    passed: bool
    errors: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        tag = "PASS" if self.passed else "FAIL"
        detail = f" ({'; '.join(self.errors)})" if self.errors else ""
        return f"[{self.index}] {tag}{detail} — {self.prompt[:50]}"


# ---------------------------------------------------------------------------
# Validators (factory functions → Validator)
# ---------------------------------------------------------------------------

def responded() -> Validator:
    """Bot sent at least one non-empty reply."""
    def _check(text: str, msgs: list[Message]) -> None:
        assert len(text.strip()) > 0, "empty response"
    return _check


def min_len(n: int) -> Validator:
    def _check(text: str, _msgs: list[Message]) -> None:
        assert len(text) >= n, f"too short ({len(text)}<{n})"
    return _check


def has_any(*substrings: str) -> Validator:
    """Response contains at least one of *substrings* (case-insensitive)."""
    def _check(text: str, _msgs: list[Message]) -> None:
        low = text.lower()
        assert any(s.lower() in low for s in substrings), (
            f"none of {substrings!r} found"
        )
    return _check


def has_tool() -> Validator:
    """Response includes tool-call indicators."""
    def _check(text: str, _msgs: list[Message]) -> None:
        assert "\U0001f527" in text or "\u2705" in text, "no tool indicators"
    return _check


def has_code() -> Validator:
    """Response looks like it contains a code block or code-like content."""
    def _check(text: str, _msgs: list[Message]) -> None:
        markers = ("```", "def ", "fn ", "import ", "use ", "print(", "println!", "#include")
        low = text.lower()
        assert any(m.lower() in low for m in markers), "no code found"
    return _check


# ---------------------------------------------------------------------------
# Topic messaging
# ---------------------------------------------------------------------------

def _in_topic(msg: Message, topic_id: int) -> bool:
    rt = getattr(msg, "reply_to", None)
    if not rt:
        return False
    top = getattr(rt, "reply_to_top_id", None) or getattr(rt, "reply_to_msg_id", None)
    return top == topic_id


async def _auto_approve(client: TelegramClient, msg: Message) -> None:
    """Click '✅ Always' or '✅ Once' inline button if present."""
    markup = getattr(msg, "reply_markup", None)
    if not markup or not hasattr(markup, "rows"):
        return
    for row in markup.rows:
        for btn in row.buttons:
            label = getattr(btn, "text", "") or ""
            if "always" in label.lower() or "Always" in label:
                try:
                    await msg.click(text=label)
                    log.info("auto-approved permission: %s (msg %d)", label, msg.id)
                except Exception as exc:
                    log.warning("auto-approve click failed: %s", exc)
                return


async def send_and_wait(
    client: TelegramClient,
    topic_id: int,
    text: str,
    *,
    timeout: int = 120,
    stable_for: float = 15.0,
) -> list[Message]:
    """Send *text* in a bot-chat topic and wait for a stable response.

    "Stable" = the combined text of all bot replies (including edits)
    hasn't changed for *stable_for* seconds AND contains real content
    (not just the ⚙️ placeholder or pure reasoning).

    Auto-approves permission requests (inline keyboard) so tool calls proceed.
    """
    sent = await client.send_message(BOT, text, reply_to=topic_id)
    sent_id = sent.id
    deadline = time.monotonic() + timeout
    last_snapshot = ""
    last_change = time.monotonic()
    seen_ids: set[int] = set()
    approved_ids: set[int] = set()

    while time.monotonic() < deadline:
        await asyncio.sleep(2.5)
        msgs = await client.get_messages(BOT, limit=30)
        topic_msgs = [
            m for m in msgs
            if not m.out and m.id > sent_id and _in_topic(m, topic_id)
        ]
        seen_ids.update(m.id for m in topic_msgs)

        for m in topic_msgs:
            if m.id not in approved_ids and getattr(m, "reply_markup", None):
                await _auto_approve(client, m)
                approved_ids.add(m.id)

        snapshot = "|".join(
            f"{m.id}:{m.text or ''}" for m in sorted(topic_msgs, key=lambda m: m.id)
        )
        if snapshot != last_snapshot:
            last_snapshot = snapshot
            last_change = time.monotonic()
            continue

        content = " ".join((m.text or "") for m in topic_msgs).strip()
        is_placeholder = content in ("", "\u2699\ufe0f")

        has_tool_indicator = "\U0001f527" in content or "\u2705" in content
        is_reasoning_only = (
            content.startswith("\U0001f4ad") and not has_tool_indicator
        )

        if not is_placeholder and not is_reasoning_only and time.monotonic() - last_change >= stable_for:
            break

    if seen_ids:
        refreshed = await client.get_messages(BOT, ids=list(seen_ids))
        return sorted(refreshed, key=lambda m: m.id)
    return []


# ---------------------------------------------------------------------------
# Scenario runner (shared logic — DRY)
# ---------------------------------------------------------------------------

def _is_stuck(text: str) -> bool:
    """Response is stuck: only placeholder or unfinished reasoning."""
    stripped = text.strip()
    if stripped in ("", "\u2699\ufe0f"):
        return True
    if stripped.startswith("\U0001f4ad") and "\U0001f527" not in stripped and "\u2705" not in stripped:
        return True
    return False


async def _reset_topic(client: TelegramClient, topic_id: int, label: str) -> None:
    reset_msgs = await send_and_wait(client, topic_id, "/reset", timeout=20, stable_for=5)
    reset_text = " ".join(m.text or "" for m in reset_msgs).lower()
    if "reset" not in reset_text:
        log.warning("[%s] /reset may have failed: %s", label, reset_text[:100])
    else:
        log.info("[%s] context reset OK", label)


async def run_scenario(
    client: TelegramClient,
    topic_id: int,
    steps: list[Step],
    label: str,
) -> list[StepResult]:
    """Reset context, then execute each step sequentially.

    If a step gets stuck (only placeholder/reasoning after timeout),
    the session is reset before continuing so subsequent steps
    aren't poisoned by a dead session.
    """
    await _reset_topic(client, topic_id, label)

    results: list[StepResult] = []
    session_dirty = False

    for i, step in enumerate(steps, 1):
        if session_dirty:
            log.info("[%s] resetting after stuck step", label)
            await _reset_topic(client, topic_id, label)
            session_dirty = False

        log.info("[%s] step %d/%d: %s", label, i, len(steps), step.prompt[:60])
        msgs = await send_and_wait(client, topic_id, step.prompt, timeout=step.timeout)
        combined = " ".join(m.text or "" for m in msgs)

        log.info("[%s] step %d got %d msgs, %d chars: %.200s",
                 label, i, len(msgs), len(combined), combined[:200])

        if _is_stuck(combined):
            log.warning("[%s] step %d STUCK — will reset before next step", label, i)
            session_dirty = True

        errors: list[str] = []
        for check in step.checks:
            try:
                check(combined, msgs)
            except AssertionError as exc:
                errors.append(str(exc))

        result = StepResult(index=i, prompt=step.prompt[:60], passed=not errors, errors=errors)
        results.append(result)
        log.info("[%s] %s", label, result)

    return results


# ---------------------------------------------------------------------------
# Step definitions
# ---------------------------------------------------------------------------

KEYBOARD_STEPS: list[Step] = [
    Step(
        "Найди мне пожалуйста программируемые клавиатуры",
        [responded(), min_len(100)],
        timeout=180,
    ),
    Step(
        "Какие из них программируются?",
        [responded(), min_len(50)],
    ),
    Step(
        "Напиши Python скрипт который эмулирует работу с программируемой "
        "клавиатурой используя библиотеку pynput — переназначение клавиш и макросы. "
        "Просто напиши готовый скрипт без уточняющих вопросов.",
        [responded(), has_code(), min_len(100)],
        timeout=180,
    ),
    Step(
        "Сохрани скрипт в /tmp/keyboard_macro.py и запусти с флагом --help чтобы протестировать",
        [responded(), has_tool()],
        timeout=300,
    ),
    Step(
        "Какие недостатки ты видишь в этом скрипте?",
        [responded(), min_len(100)],
        timeout=180,
    ),
    Step(
        "Сделай документацию по этому скрипту в формате markdown",
        [responded(), min_len(200)],
        timeout=240,
    ),
    Step(
        "Сохрани скрипт и документацию в /tmp/keyboard_project/, "
        "создай zip архив /tmp/keyboard_project.zip",
        [responded(), has_tool()],
        timeout=300,
    ),
    Step(
        "Перепиши Python скрипт клавиатуры на Rust используя enigo крейт",
        [responded(), has_code(), min_len(100)],
        timeout=240,
    ),
    Step(
        "Сгенерируй PDF из документации через pandoc — "
        "сохрани в /tmp/keyboard_project/docs.pdf",
        [responded(), has_tool()],
        timeout=300,
    ),
    Step(
        "Создай второй архив /tmp/keyboard_project_v2.zip который включает "
        "Rust версию и PDF",
        [responded(), has_tool()],
        timeout=300,
    ),
]

MOUSE_STEPS: list[Step] = [
    Step(
        "Find me programmable mice please",
        [responded(), min_len(100)],
        timeout=180,
    ),
    Step(
        "Which of them are programmable?",
        [responded(), min_len(50)],
    ),
    Step(
        "Write a Python script that uses the pynput library to remap mouse buttons "
        "and add macros for a Razer DeathAdder mouse. Include button remapping and "
        "a scroll-click macro. Just write the complete script, don't ask clarifying questions.",
        [responded(), has_code(), min_len(100)],
        timeout=180,
    ),
    Step(
        "Save this script to /tmp/mouse_macro.py and run it with --help flag to test it",
        [responded(), has_tool()],
        timeout=300,
    ),
    Step(
        "What disadvantages do you see in this script?",
        [responded(), min_len(100)],
        timeout=180,
    ),
    Step(
        "Create documentation for this script in markdown format",
        [responded(), min_len(200)],
        timeout=240,
    ),
    Step(
        "Save the script and documentation to /tmp/mouse_project/, "
        "create a zip archive /tmp/mouse_project.zip",
        [responded(), has_tool()],
        timeout=300,
    ),
    Step(
        "Rewrite the Python mouse script in Rust using the enigo crate",
        [responded(), has_code(), min_len(100)],
        timeout=240,
    ),
    Step(
        "Generate a PDF from the documentation using pandoc — "
        "save it to /tmp/mouse_project/docs.pdf",
        [responded(), has_tool()],
        timeout=300,
    ),
    Step(
        "Create a second archive /tmp/mouse_project_v2.zip that includes "
        "the Rust version and the PDF",
        [responded(), has_tool()],
        timeout=300,
    ),
]


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------

class TestMultiStepE2E(unittest.IsolatedAsyncioTestCase):
    client: TelegramClient

    async def asyncSetUp(self) -> None:
        self.client = TelegramClient(SESSION, API_ID, API_HASH)
        await self.client.start()

    async def asyncTearDown(self) -> None:
        await self.client.disconnect()

    async def test_parallel_scenarios(self) -> None:
        """Run RU-keyboard and EN-mouse scenarios in parallel, both starting fresh."""
        ru_results, en_results = await asyncio.gather(
            run_scenario(self.client, TOPIC_RU, KEYBOARD_STEPS, "RU-keyboard"),
            run_scenario(self.client, TOPIC_EN, MOUSE_STEPS, "EN-mouse"),
        )

        log.info("=" * 60)
        log.info("RESULTS — RU-keyboard")
        for r in ru_results:
            log.info("  %s", r)
        log.info("RESULTS — EN-mouse")
        for r in en_results:
            log.info("  %s", r)
        log.info("=" * 60)

        all_results = ru_results + en_results
        failed = [r for r in all_results if not r.passed]
        self.assertEqual(
            len(failed), 0,
            "Failed steps:\n" + "\n".join(f"  {r}" for r in failed),
        )


if __name__ == "__main__":
    unittest.main()
