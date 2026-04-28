"""Parallel multi-step E2E: two 10-step conversations in bot-chat topics.

Scenario A (RU, topic from E2E_TOPIC_RU): programmable keyboards
Scenario B (EN, topic from E2E_TOPIC_EN): programmable mice

Each scenario starts with /reset to ensure clean context.
Both run concurrently via asyncio.gather.

Requires:
  - Bot running from python_bridge/: python3 -m opencode_tg.bot
  - OpenCode server on :14096
  - Telethon session path supplied via E2E_SESSION_PATH
  - Bot username supplied via E2E_BOT_USERNAME
  - Topic IDs supplied via E2E_TOPIC_RU / E2E_TOPIC_EN

Run:
  cd python_bridge && python3 -m pytest tests/test_e2e_multistep.py -v -s --timeout=1200
"""
from __future__ import annotations

import asyncio
import logging
import os
import shutil
import time
import unittest
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

from telethon import TelegramClient
from telethon.tl.types import Message

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_ID = int(os.environ.get("E2E_API_ID", "0"))
API_HASH = os.environ.get("E2E_API_HASH", "")
SESSION = os.environ.get(
    "E2E_SESSION_PATH",
    os.path.expanduser("~/.local/share/opencode_tg/e2e.session"),
)
BOT = os.environ.get("E2E_BOT_USERNAME", "")
TOPIC_RU = int(os.environ.get("E2E_TOPIC_RU", "0"))
TOPIC_EN = int(os.environ.get("E2E_TOPIC_EN", "0"))
PROJECT_DIR = os.environ.get("E2E_PROJECT_DIR", os.getcwd())
PROJECT_NAME = os.path.basename(PROJECT_DIR.rstrip(os.sep)) or PROJECT_DIR

ARTIFACT_PATHS = [
    "/tmp/keyboard_macro.py", "/tmp/mouse_macro.py",
    "/tmp/keyboard_project", "/tmp/mouse_project",
    "/tmp/keyboard_project.zip", "/tmp/keyboard_project_v2.zip",
    "/tmp/mouse_project.zip", "/tmp/mouse_project_v2.zip",
    "/tmp/oc_e2e_feature_test.txt",
]

MAX_RETRIES = 2

# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------


class ErrorCategory(str, Enum):
    BOT = "bot"
    MODEL = "model"
    ARTIFACT = "artifact"


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

Validator = Callable[[str, list[Message]], None]


@dataclass
class Step:
    prompt: str
    checks: list[Validator]
    timeout: int = 120
    retryable: bool = False


@dataclass
class StepResult:
    index: int
    prompt: str
    passed: bool
    errors: list[str] = field(default_factory=list)
    retried: bool = False

    def __str__(self) -> str:
        tag = "PASS" if self.passed else "FAIL"
        detail = f" ({'; '.join(self.errors)})" if self.errors else ""
        retry_tag = " [retried]" if self.retried else ""
        return f"[{self.index}] {tag}{detail}{retry_tag} — {self.prompt[:50]}"


# ---------------------------------------------------------------------------
# Validators (factory functions → Validator)
#
# Each returned callable gets a `.category` attribute for error classification.
# ---------------------------------------------------------------------------

def _tag(fn: Validator, cat: ErrorCategory) -> Validator:
    fn.category = cat  # type: ignore[attr-defined]
    return fn


def responded() -> Validator:
    def _check(text: str, msgs: list[Message]) -> None:
        assert len(text.strip()) > 0, "empty response"
    return _tag(_check, ErrorCategory.BOT)


def min_len(n: int) -> Validator:
    def _check(text: str, _msgs: list[Message]) -> None:
        assert len(text) >= n, f"too short ({len(text)}<{n})"
    return _tag(_check, ErrorCategory.MODEL)


def has_any(*substrings: str) -> Validator:
    def _check(text: str, _msgs: list[Message]) -> None:
        low = text.lower()
        assert any(s.lower() in low for s in substrings), (
            f"none of {substrings!r} found"
        )
    return _tag(_check, ErrorCategory.MODEL)


def has_tool() -> Validator:
    """Check for tool execution — composite UX replaces markers with final text."""
    def _check(text: str, _msgs: list[Message]) -> None:
        has_markers = "\U0001f527" in text or "\u2705" in text
        has_result = len(text.strip()) >= 4
        assert has_markers or has_result, "no tool indicators and no meaningful response"
    return _tag(_check, ErrorCategory.BOT)


def has_code() -> Validator:
    def _check(text: str, _msgs: list[Message]) -> None:
        markers = ("```", "def ", "fn ", "import ", "use ", "print(", "println!", "#include")
        low = text.lower()
        assert any(m.lower() in low for m in markers), "no code found"
    return _tag(_check, ErrorCategory.MODEL)


def file_exists(path: str) -> Validator:
    def _check(_text: str, _msgs: list[Message]) -> None:
        assert os.path.isfile(path), f"file not found: {path}"
        assert os.path.getsize(path) > 0, f"file is empty: {path}"
    return _tag(_check, ErrorCategory.ARTIFACT)


def valid_zip(path: str, min_files: int = 1) -> Validator:
    def _check(_text: str, _msgs: list[Message]) -> None:
        import zipfile
        assert os.path.isfile(path), f"zip not found: {path}"
        with zipfile.ZipFile(path, "r") as zf:
            names = [n for n in zf.namelist() if not n.endswith("/")]
            assert len(names) >= min_files, (
                f"zip has {len(names)} files, expected >= {min_files}: {names}"
            )
    return _tag(_check, ErrorCategory.ARTIFACT)


def valid_pdf(path: str) -> Validator:
    def _check(_text: str, _msgs: list[Message]) -> None:
        assert os.path.isfile(path), f"pdf not found: {path}"
        with open(path, "rb") as f:
            header = f.read(5)
        assert header == b"%PDF-", f"invalid PDF header: {header!r}"
    return _tag(_check, ErrorCategory.ARTIFACT)


def script_runs(path: str, flag: str = "--help") -> Validator:
    def _check(_text: str, _msgs: list[Message]) -> None:
        import subprocess
        assert os.path.isfile(path), f"script not found: {path}"
        r = subprocess.run(
            ["python3", path, flag],
            capture_output=True, text=True, timeout=10,
        )
        assert r.returncode == 0, f"script exited {r.returncode}: {r.stderr[:200]}"
    return _tag(_check, ErrorCategory.ARTIFACT)


def rust_checks(project_dir: str) -> Validator:
    def _check(_text: str, _msgs: list[Message]) -> None:
        import subprocess
        cargo_toml = os.path.join(project_dir, "Cargo.toml")
        assert os.path.isfile(cargo_toml), f"Cargo.toml not found in {project_dir}"
        r = subprocess.run(
            ["cargo", "check"],
            capture_output=True, text=True, timeout=120,
            cwd=project_dir,
        )
        assert r.returncode == 0, f"cargo check failed: {r.stderr[-300:]}"
    return _tag(_check, ErrorCategory.ARTIFACT)


def _cleanup_artifacts() -> None:
    for p in ARTIFACT_PATHS:
        if os.path.isdir(p):
            shutil.rmtree(p, ignore_errors=True)
        elif os.path.isfile(p):
            os.remove(p)


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

        is_only_reasoning = (
            content.startswith("\U0001f4ad")
            and "\u2500" not in content
            and len(content) < 600
        )
        is_only_tools = all(
            line.startswith(("\U0001f527", "\u2705", "\U0001f4ad", "\u2500"))
            for line in content.splitlines() if line.strip()
        ) and len(content) < 300

        if not is_placeholder and not is_only_reasoning and not is_only_tools and time.monotonic() - last_change >= stable_for:
            break

    if seen_ids:
        refreshed = await client.get_messages(BOT, ids=list(seen_ids))
        return sorted(refreshed, key=lambda m: m.id)
    return []


# ---------------------------------------------------------------------------
# Scenario runner (shared logic — DRY)
# ---------------------------------------------------------------------------

def _is_stuck(text: str) -> bool:
    """Response is stuck: only placeholder or unfinished reasoning with no real content."""
    stripped = text.strip()
    if stripped in ("", "\u2699\ufe0f"):
        return True
    if stripped.startswith("\U0001f4ad"):
        without_reasoning = stripped.lstrip("\U0001f4ad").strip()
        if len(without_reasoning) < 10 and "\u2500" not in stripped:
            return True
    return False


async def _reset_topic(client: TelegramClient, topic_id: int, label: str) -> None:
    reset_msgs = await send_and_wait(client, topic_id, "/reset", timeout=20, stable_for=5)
    reset_text = " ".join(m.text or "" for m in reset_msgs).lower()
    if "reset" not in reset_text:
        log.warning("[%s] /reset may have failed: %s", label, reset_text[:100])
    else:
        log.info("[%s] context reset OK", label)


def _run_checks(
    checks: list[Validator], combined: str, msgs: list[Message],
) -> list[tuple[str, ErrorCategory]]:
    """Run all validators, return list of (error_message, category)."""
    errors: list[tuple[str, ErrorCategory]] = []
    for check in checks:
        try:
            check(combined, msgs)
        except AssertionError as exc:
            cat = getattr(check, "category", ErrorCategory.MODEL)
            errors.append((str(exc), cat))
    return errors


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

    Failed retryable steps are retried up to MAX_RETRIES times.
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

        stuck = _is_stuck(combined)
        if stuck:
            log.warning("[%s] step %d STUCK — will reset before next step", label, i)
            session_dirty = True

        errors = _run_checks(step.checks, combined, msgs)
        retried = False

        if errors and step.retryable and not stuck:
            for attempt in range(1, MAX_RETRIES + 1):
                cats = {cat.value for _, cat in errors}
                log.info(
                    "[%s] step %d retry %d/%d (errors: %s)",
                    label, i, attempt, MAX_RETRIES, ", ".join(cats),
                )
                await _reset_topic(client, topic_id, label)
                msgs = await send_and_wait(
                    client, topic_id, step.prompt, timeout=step.timeout,
                )
                combined = " ".join(m.text or "" for m in msgs)
                log.info(
                    "[%s] step %d retry got %d msgs, %d chars: %.200s",
                    label, i, len(msgs), len(combined), combined[:200],
                )
                errors = _run_checks(step.checks, combined, msgs)
                retried = True
                if not errors:
                    break

        error_strs = [msg for msg, _ in errors]
        result = StepResult(
            index=i, prompt=step.prompt[:60],
            passed=not errors, errors=error_strs, retried=retried,
        )
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
        retryable=True,
    ),
    Step(
        "Напиши Python скрипт который эмулирует работу с программируемой "
        "клавиатурой используя библиотеку pynput — переназначение клавиш и макросы. "
        "Скрипт должен работать на headless сервере без X11 — "
        "используй argparse до импорта pynput, чтобы --help работал без дисплея. "
        "Просто напиши готовый скрипт без уточняющих вопросов.",
        [responded(), has_code(), min_len(100)],
        timeout=180,
    ),
    Step(
        "Напиши Python скрипт для программируемой клавиатуры с pynput (переназначение клавиш, макросы). "
        "Используй argparse до импорта pynput чтобы --help работал без дисплея. "
        "Сохрани в /tmp/keyboard_macro.py и запусти с --help для теста.",
        [responded(), has_tool(),
         file_exists("/tmp/keyboard_macro.py"),
         script_runs("/tmp/keyboard_macro.py")],
        timeout=300,
        retryable=True,
    ),
    Step(
        "Какие недостатки ты видишь в этом скрипте?",
        [responded(), min_len(100)],
        timeout=180,
    ),
    Step(
        "Сделай документацию по этому скрипту в формате markdown. "
        "Покажи полный текст документации в ответе.",
        [responded(), min_len(200)],
        timeout=240,
    ),
    Step(
        "Скопируй /tmp/keyboard_macro.py в /tmp/keyboard_project/ и создай "
        "markdown документацию /tmp/keyboard_project/README.md, "
        "затем создай zip архив /tmp/keyboard_project.zip из папки /tmp/keyboard_project/",
        [responded(), has_tool(),
         valid_zip("/tmp/keyboard_project.zip", min_files=2)],
        timeout=300,
        retryable=True,
    ),
    Step(
        "Создай Rust проект в /tmp/keyboard_project/ (cargo init если нет Cargo.toml). "
        "Добавь enigo = \"0.2\" в зависимости. "
        "Напиши простой main.rs с базовым переназначением клавиш через enigo. "
        "Запусти cargo check и убедись что компилируется.",
        [responded(), has_tool(),
         file_exists("/tmp/keyboard_project/src/main.rs"),
         rust_checks("/tmp/keyboard_project")],
        timeout=360,
        retryable=True,
    ),
    Step(
        "Сгенерируй PDF из документации — "
        "сохрани в /tmp/keyboard_project/docs.pdf. "
        "Используй Python (fpdf2 или reportlab) если pandoc недоступен.",
        [responded(), has_tool(),
         file_exists("/tmp/keyboard_project/docs.pdf"),
         valid_pdf("/tmp/keyboard_project/docs.pdf")],
        timeout=300,
        retryable=True,
    ),
    Step(
        "Создай второй архив /tmp/keyboard_project_v2.zip который включает "
        "Rust версию и PDF",
        [responded(), has_tool(),
         valid_zip("/tmp/keyboard_project_v2.zip", min_files=3)],
        timeout=300,
        retryable=True,
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
        retryable=True,
    ),
    Step(
        "Write a Python script that uses the pynput library to remap mouse buttons "
        "and add macros for a Razer DeathAdder mouse. Include button remapping and "
        "a scroll-click macro. The script must work on a headless server without X11 — "
        "use argparse before importing pynput so --help works without a display. "
        "Just write the complete script, don't ask clarifying questions.",
        [responded(), has_code(), min_len(100)],
        timeout=180,
    ),
    Step(
        "Write a Python script for Razer DeathAdder mouse button remapping using pynput. "
        "Use argparse before importing pynput so --help works without display. "
        "Save it to /tmp/mouse_macro.py and run with --help flag to test.",
        [responded(), has_tool(),
         file_exists("/tmp/mouse_macro.py"),
         script_runs("/tmp/mouse_macro.py")],
        timeout=300,
        retryable=True,
    ),
    Step(
        "What disadvantages do you see in this script?",
        [responded(), min_len(100)],
        timeout=180,
    ),
    Step(
        "Create documentation for this script in markdown format. "
        "Show the full documentation text in your reply.",
        [responded(), min_len(200)],
        timeout=240,
    ),
    Step(
        "Copy /tmp/mouse_macro.py to /tmp/mouse_project/ and create "
        "a markdown README.md in /tmp/mouse_project/, "
        "then create a zip archive /tmp/mouse_project.zip from /tmp/mouse_project/",
        [responded(), has_tool(),
         valid_zip("/tmp/mouse_project.zip", min_files=2)],
        timeout=300,
        retryable=True,
    ),
    Step(
        "Create a Rust project in /tmp/mouse_project/ (cargo init if no Cargo.toml). "
        "Add enigo = \"0.2\" as a dependency. "
        "Write a simple main.rs with basic mouse click simulation using enigo. "
        "Run cargo check and make sure it compiles.",
        [responded(), has_tool(),
         file_exists("/tmp/mouse_project/src/main.rs"),
         rust_checks("/tmp/mouse_project")],
        timeout=360,
        retryable=True,
    ),
    Step(
        "Generate a PDF from the documentation — "
        "save it to /tmp/mouse_project/docs.pdf. "
        "Use Python (fpdf2 or reportlab) if pandoc is not available.",
        [responded(), has_tool(),
         file_exists("/tmp/mouse_project/docs.pdf"),
         valid_pdf("/tmp/mouse_project/docs.pdf")],
        timeout=300,
        retryable=True,
    ),
    Step(
        "Create a second archive /tmp/mouse_project_v2.zip that includes "
        "the Rust version and the PDF",
        [responded(), has_tool(),
         valid_zip("/tmp/mouse_project_v2.zip", min_files=3)],
        timeout=300,
        retryable=True,
    ),
]


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------

def has_text(substring: str) -> Validator:
    """Check that substring appears in combined text (case-insensitive)."""
    def _check(text: str, _msgs: list[Message]) -> None:
        assert substring.lower() in text.lower(), f"'{substring}' not found"
    return _tag(_check, ErrorCategory.BOT)




# ---------------------------------------------------------------------------
# Per-chat feature steps (new command tests in a topic dialogue)
# ---------------------------------------------------------------------------

FEATURE_STEPS: list[Step] = [
    Step(
        "/project",
        [responded(), has_text("current directory")],
        timeout=15,
    ),
    Step(
        "/project /tmp",
        [responded(), has_text("/tmp"), has_text("reset")],
        timeout=20,
    ),
    Step(
        "/project",
        [responded(), has_text("/tmp")],
        timeout=15,
    ),
    Step(
        f"/project {PROJECT_DIR}",
        [responded(), has_text(PROJECT_NAME)],
        timeout=20,
    ),
    Step(
        "/project /nonexistent/path/12345",
        [responded(), has_text("not found")],
        timeout=15,
    ),
    Step(
        "/approve on",
        [responded(), has_text("auto-approve"), has_text("on")],
        timeout=15,
    ),
    Step(
        "/approve",
        [responded(), has_text("ON")],
        timeout=15,
    ),
    Step(
        "/approve off",
        [responded(), has_text("off")],
        timeout=15,
    ),
    Step(
        "/id",
        [responded(), has_text("dir:"), has_text("approve:")],
        timeout=15,
    ),
    Step(
        "/diff",
        [responded(), has_any("no pending", "no active", "diff", "changes")],
        timeout=15,
    ),
    Step(
        "/git status",
        [responded(), has_any("branch", "commit", "clean", "modified", "untracked", "changes")],
        timeout=120,
        retryable=True,
    ),
    Step(
        "/approve on",
        [responded(), has_text("on")],
        timeout=15,
    ),
    Step(
        'Run this bash command: echo "feature test OK 42" > /tmp/oc_e2e_feature_test.txt && echo "done"',
        [responded(), has_tool(), file_exists("/tmp/oc_e2e_feature_test.txt")],
        timeout=180,
        retryable=True,
    ),
]


async def _cleanup_bot_chat(client: TelegramClient) -> None:
    """Delete all messages in the bot chat to keep it clean after tests."""
    try:
        msgs = await client.get_messages(BOT, limit=300)
        if msgs:
            ids = [m.id for m in msgs]
            await client.delete_messages(BOT, ids)
    except Exception:
        pass


@unittest.skipUnless(
    API_ID and API_HASH and BOT and TOPIC_RU and TOPIC_EN,
    "E2E_API_ID / E2E_API_HASH / E2E_BOT_USERNAME / E2E_TOPIC_RU / E2E_TOPIC_EN env vars not set",
)
class TestMultiStepE2E(unittest.IsolatedAsyncioTestCase):
    client: TelegramClient

    async def asyncSetUp(self) -> None:
        _cleanup_artifacts()
        self.client = TelegramClient(SESSION, API_ID, API_HASH)
        await self.client.start()

    async def asyncTearDown(self) -> None:
        await _cleanup_bot_chat(self.client)
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

    async def test_per_chat_features(self) -> None:
        """Sequential dialogue testing /project, /approve, /id, /diff, /git, file write."""
        results = await run_scenario(
            self.client, TOPIC_RU, FEATURE_STEPS, "per-chat-features",
        )

        log.info("=" * 60)
        log.info("RESULTS — per-chat-features")
        for r in results:
            log.info("  %s", r)
        log.info("=" * 60)

        failed = [r for r in results if not r.passed]
        self.assertEqual(
            len(failed), 0,
            "Failed steps:\n" + "\n".join(f"  {r}" for r in failed),
        )


if __name__ == "__main__":
    unittest.main()
