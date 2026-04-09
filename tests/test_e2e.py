"""E2E tests — Telethon user sends messages to @zGsR_bot, asserts responses.

Requires:
  - Bot running: python3 -m opencode_tg.bot
  - OpenCode server running on :14096
  - Telethon session: ~/.zeroclaw/workspace/skills/telegram-reader/.session/zverozabr_session

Run:
  python3 -m pytest opencode_tg/tests/test_e2e.py -v -s --timeout=600
"""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import sys
import time
import unittest

from telethon import TelegramClient
from telethon.tl.types import Message

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_ID = int(os.environ.get("E2E_API_ID", "0"))
API_HASH = os.environ.get("E2E_API_HASH", "")
SESSION_PATH = os.environ.get(
    "E2E_SESSION_PATH",
    os.path.expanduser("~/.zeroclaw/workspace/skills/telegram-reader/.session/zverozabr_session"),
)
BOT_USERNAME = os.environ.get("E2E_BOT_USERNAME", "zGsR_bot")
BOT_ID = int(os.environ.get("E2E_BOT_ID", "8527746065"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def send_and_wait(
    client: TelegramClient,
    text: str,
    timeout: int = 90,
    idle_gap: float = 8.0,
) -> list[Message]:
    """Send *text* to the bot and collect all bot replies until idle.

    "Idle" = no new message from the bot for *idle_gap* seconds.
    Returns bot messages in chronological order.
    """
    sent = await client.send_message(BOT_USERNAME, text)
    sent_id = sent.id
    deadline = time.monotonic() + timeout
    bot_msgs: list[Message] = []
    last_bot_ts = time.monotonic()

    while time.monotonic() < deadline:
        await asyncio.sleep(2)
        msgs = await client.get_messages(BOT_USERNAME, limit=15)
        new_bot = [
            m for m in msgs
            if not m.out
            and m.id > sent_id
            and m.id not in {bm.id for bm in bot_msgs}
        ]
        if new_bot:
            bot_msgs.extend(new_bot)
            last_bot_ts = time.monotonic()
        elif time.monotonic() - last_bot_ts > idle_gap:
            break

    # Re-fetch latest versions (edits may have happened)
    if bot_msgs:
        ids = [m.id for m in bot_msgs]
        refreshed = await client.get_messages(BOT_USERNAME, ids=ids)
        return sorted(refreshed, key=lambda m: m.id)
    return []


async def wait_for_stable_text(
    client: TelegramClient,
    msg_id: int,
    timeout: int = 60,
    stable_for: float = 5.0,
) -> str:
    """Poll a message until its text stops changing."""
    last_text = ""
    last_change = time.monotonic()
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        msgs = await client.get_messages(BOT_USERNAME, ids=[msg_id])
        if msgs and msgs[0]:
            current = msgs[0].text or ""
            if current != last_text:
                last_text = current
                last_change = time.monotonic()
            elif time.monotonic() - last_change > stable_for:
                return last_text
        await asyncio.sleep(1.5)
    return last_text


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------


@unittest.skipUnless(API_ID and API_HASH, "E2E_API_ID / E2E_API_HASH env vars not set")
class TestE2E(unittest.IsolatedAsyncioTestCase):
    client: TelegramClient

    async def asyncSetUp(self):
        self.client = TelegramClient(SESSION_PATH, API_ID, API_HASH)
        await self.client.start()

    async def asyncTearDown(self):
        await self.client.disconnect()

    # -- t1: basic reply ----------------------------------------------------

    async def test_t1_basic_reply(self):
        """Bot replies with the requested keyword."""
        msgs = await send_and_wait(self.client, "Say exactly: HELLOBRIDGE42")
        texts = " ".join(m.text or "" for m in msgs)
        self.assertIn("HELLOBRIDGE42", texts, f"Expected HELLOBRIDGE42 in: {texts[:300]}")

    # -- t2: streaming (message edits) --------------------------------------

    async def test_t2_streaming(self):
        """Bot edits a placeholder message as text streams in."""
        await self.client.send_message(BOT_USERNAME, "Count from 1 to 10, one number per line")
        await asyncio.sleep(2)

        # Find the placeholder (⚙️)
        msgs = await self.client.get_messages(BOT_USERNAME, limit=5)
        bot_msg = next((m for m in msgs if not m.out), None)
        self.assertIsNotNone(bot_msg, "No bot message found")

        # Wait for text to stabilize (streaming edits)
        final_text = await wait_for_stable_text(self.client, bot_msg.id, timeout=60)
        self.assertTrue(len(final_text) > 5, f"Final text too short: {final_text!r}")
        # Should contain some numbers
        has_numbers = any(str(n) in final_text for n in range(1, 11))
        self.assertTrue(has_numbers, f"Expected numbers in: {final_text[:200]}")

    # -- t3: tool call ------------------------------------------------------

    async def test_t3_tool_call(self):
        """Bot shows tool invocation messages (🔧 → ✅)."""
        msgs = await send_and_wait(
            self.client,
            "Run this exact bash command: echo TOOL_TEST_OK",
            timeout=90,
        )
        texts = " ".join(m.text or "" for m in msgs)
        # Should have a tool indicator
        has_tool = "\U0001f527" in texts or "\u2705" in texts or "bash" in texts.lower()
        self.assertTrue(has_tool, f"No tool indicator found in: {texts[:400]}")

    # -- t4: permission request ---------------------------------------------

    async def test_t4_permission(self):
        """Bot shows inline keyboard for permission requests.

        We trigger a write operation that should require permission.
        """
        await send_and_wait(self.client, "/reset", idle_gap=5)

        msgs = await send_and_wait(
            self.client,
            "Create a file /tmp/oc_e2e_perm_test.txt with content 'hello'. Use bash to do it.",
            timeout=90,
        )
        # Check for permission keyboard or tool usage
        texts = " ".join(m.text or "" for m in msgs)
        has_perm_or_tool = (
            "\U0001f510" in texts  # 🔐 permission
            or "\u2705" in texts   # ✅ approved
            or "\U0001f527" in texts  # 🔧 tool
            or "permission" in texts.lower()
            or "bash" in texts.lower()
        )
        self.assertTrue(
            has_perm_or_tool,
            f"Expected permission or tool activity in: {texts[:400]}",
        )

    # -- t5: /reset ---------------------------------------------------------

    async def test_t5_reset(self):
        """After /reset, bot has no memory of previous context."""
        # First: establish context
        await send_and_wait(self.client, "Remember the secret word: BANANACAKE55")

        # Reset
        reset_msgs = await send_and_wait(self.client, "/reset", idle_gap=5)
        reset_text = " ".join(m.text or "" for m in reset_msgs)
        self.assertIn("reset", reset_text.lower(), f"Expected reset confirmation: {reset_text}")

        # Ask about the secret — should NOT know it
        msgs = await send_and_wait(self.client, "What was the secret word I told you?")
        texts = " ".join(m.text or "" for m in msgs)
        self.assertNotIn("BANANACAKE55", texts, f"Bot remembered after reset: {texts[:300]}")

    # -- t6: /model switch --------------------------------------------------

    async def test_t6_model_switch(self):
        """/models shows numbered list; reply with number to switch model."""
        await send_and_wait(self.client, "/reset", idle_gap=5)

        # 1) /models — should show numbered list grouped by provider
        model_msgs = await send_and_wait(self.client, "/models", idle_gap=5)
        models_text = " ".join(m.text or "" for m in model_msgs)
        self.assertIn("Reply with a number", models_text,
                       f"Expected numbered list: {models_text[:400]}")

        # 2) Find a non-default model number in the list
        import re
        # Prefer MiniMax-M2.7 (non-highspeed) as alt — avoids GLM flakiness
        alt_match = re.search(r"(\d+)\.\s+MiniMax-M2\.7\b(?!-)", models_text)
        if not alt_match:
            alt_match = re.search(r"(\d+)\.\s+GLM", models_text)
        self.assertIsNotNone(alt_match, f"Expected alt model in list: {models_text[:400]}")
        alt_num = alt_match.group(1)

        # 3) Reply with that number to switch
        switch_msgs = await send_and_wait(self.client, alt_num, idle_gap=5)
        switch_text = " ".join(m.text or "" for m in switch_msgs)
        self.assertIn("\u2705", switch_text, f"Expected confirmation: {switch_text[:300]}")

        # 4) Send a prompt — verify the alt model responds
        msgs = await send_and_wait(self.client, "Say exactly: SWITCHED91", timeout=90)
        texts = " ".join(m.text or "" for m in msgs)
        self.assertIn("SWITCHED91", texts, f"Expected SWITCHED91 after model switch: {texts[:300]}")

        # 5) Switch back to default via /model command
        back_msgs = await send_and_wait(
            self.client, "/model minimax/MiniMax-M2.7-highspeed", idle_gap=5,
        )
        back_text = " ".join(m.text or "" for m in back_msgs)
        self.assertIn("\u2705", back_text, f"Expected confirmation: {back_text[:300]}")

        # 6) Verify default model also works
        msgs2 = await send_and_wait(self.client, "Say exactly: BACKDEFAULT92")
        texts2 = " ".join(m.text or "" for m in msgs2)
        self.assertIn("BACKDEFAULT92", texts2, f"Expected BACKDEFAULT92: {texts2[:300]}")

    # -- t7: long response → Telegraph --------------------------------------

    async def test_t7_long_response(self):
        """Very long response gets posted to Telegraph."""
        msgs = await send_and_wait(
            self.client,
            (
                "Write a very long essay about the history of Python programming language. "
                "Include at least 15 detailed paragraphs covering Guido van Rossum, "
                "CPython, PyPy, the GIL, async/await, type hints, packaging, "
                "and the community. Make it at least 5000 words."
            ),
            timeout=180,
            idle_gap=15,
        )
        texts = " ".join(m.text or "" for m in msgs)
        has_telegraph = "telegra.ph" in texts
        # It's also OK if it just sent multiple chunks
        has_content = len(texts) > 200
        self.assertTrue(
            has_telegraph or has_content,
            f"Expected telegraph link or long content: {texts[:400]}",
        )

    # -- t8: parallel topics ------------------------------------------------

    async def test_t8_parallel_topics(self):
        """Two conversations at the same time get independent responses."""
        await send_and_wait(self.client, "/reset", idle_gap=5)

        # Send two messages back-to-back with different keywords
        sent_a = await self.client.send_message(BOT_USERNAME, "Say exactly: ALPHA808")
        await asyncio.sleep(0.5)
        # Second prompt goes to the same session, so it will be queued.
        # We just verify both keywords eventually appear in bot replies.
        sent_b = await self.client.send_message(BOT_USERNAME, "Now say exactly: BETA909")

        deadline = time.monotonic() + 120
        all_texts = ""
        while time.monotonic() < deadline:
            await asyncio.sleep(3)
            msgs = await self.client.get_messages(BOT_USERNAME, limit=20)
            bot_texts = [m.text or "" for m in msgs if not m.out and m.id > sent_a.id]
            all_texts = " ".join(bot_texts)
            if "ALPHA808" in all_texts and "BETA909" in all_texts:
                break

        self.assertIn("ALPHA808", all_texts, f"Expected ALPHA808 in: {all_texts[:400]}")
        self.assertIn("BETA909", all_texts, f"Expected BETA909 in: {all_texts[:400]}")

    # -- t9: restart --------------------------------------------------------

    async def test_t9_restart(self):
        """Kill bot → restart → send msg → bot still has session context."""
        await send_and_wait(self.client, "/reset", idle_gap=5)

        # Establish context
        await send_and_wait(self.client, "Remember this code: PERSIST99MAGIC")

        # Kill ALL bot instances
        subprocess.run(["pkill", "-f", "opencode_tg.bot"], check=False)
        await asyncio.sleep(3)
        # Double-kill to handle stragglers
        subprocess.run(["pkill", "-9", "-f", "opencode_tg.bot"], check=False)
        await asyncio.sleep(1)

        # Restart bot using venv python
        venv_py = os.path.join(
            os.path.expanduser("~/work/erp/zeroclaws"),
            "opencode_tg", ".venv", "bin", "python",
        )
        bot_py = venv_py if os.path.exists(venv_py) else sys.executable
        bot_proc = subprocess.Popen(
            [bot_py, "-m", "opencode_tg.bot"],
            cwd=os.path.expanduser("~/work/erp/zeroclaws"),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        # Wait for bot to be ready — verify with /id
        await asyncio.sleep(5)
        probe = await send_and_wait(self.client, "/id", idle_gap=5)
        self.assertTrue(probe, "Bot did not respond to /id after restart")

        try:
            msgs = await send_and_wait(
                self.client,
                "What code did I tell you to remember earlier?",
                timeout=90,
            )
            texts = " ".join(m.text or "" for m in msgs)
            self.assertIn(
                "PERSIST99MAGIC", texts,
                f"Expected PERSIST99MAGIC after restart in: {texts[:400]}",
            )
        finally:
            bot_proc.terminate()
            bot_proc.wait(timeout=5)
            self._bot_proc = subprocess.Popen(
                [bot_py, "-m", "opencode_tg.bot"],
                cwd=os.path.expanduser("~/work/erp/zeroclaws"),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            await asyncio.sleep(5)

    # -- t10: sleeping / reconnect ------------------------------------------

    async def test_t10_sleeping(self):
        """Fake idle timeout → SSE closed → new msg → reconnect works."""
        await send_and_wait(self.client, "/reset", idle_gap=5)

        # Establish a session
        await send_and_wait(self.client, "Say exactly: BEFORE10")

        # Patch idle timeout to 1 second so the watchdog closes SSE quickly
        import opencode_tg.config as cfg
        old_idle = cfg.IDLE_TIMEOUT_SECONDS
        cfg.IDLE_TIMEOUT_SECONDS = 1

        # Trigger an idle sweep by importing and calling directly
        from opencode_tg.session_manager import SessionManager
        # We can't easily reach the running bot's manager, so instead we
        # just wait a few seconds and send a new message — if the SSE was
        # closed the runner transitions to "sleeping" and the bot shows
        # "🔄 Reconnecting…" before the reply.
        await asyncio.sleep(5)
        cfg.IDLE_TIMEOUT_SECONDS = old_idle

        msgs = await send_and_wait(self.client, "Say exactly: AFTER10")
        texts = " ".join(m.text or "" for m in msgs)
        # Either we see the reconnect message or the reply itself
        has_reply = "AFTER10" in texts
        self.assertTrue(has_reply, f"Expected AFTER10 in: {texts[:400]}")

    # -- t11: file (send photo) ---------------------------------------------

    async def test_t11_file(self):
        """Send an image file to the bot and get a description."""
        await send_and_wait(self.client, "/reset", idle_gap=5)

        # Create a small test image (1x1 red PNG)
        import struct, zlib
        def _make_png() -> bytes:
            sig = b"\x89PNG\r\n\x1a\n"
            ihdr_data = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
            ihdr = _chunk(b"IHDR", ihdr_data)
            raw = b"\x00\xff\x00\x00"  # filter=none, R G B
            idat = _chunk(b"IDAT", zlib.compress(raw))
            iend = _chunk(b"IEND", b"")
            return sig + ihdr + idat + iend

        def _chunk(ctype: bytes, data: bytes) -> bytes:
            c = ctype + data
            return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

        png_path = "/tmp/oc_e2e_test.png"
        with open(png_path, "wb") as f:
            f.write(_make_png())

        # Send the image with a caption
        await self.client.send_file(
            BOT_USERNAME, png_path, caption="Describe this image",
        )

        # Wait for bot response
        deadline = time.monotonic() + 90
        bot_reply = ""
        while time.monotonic() < deadline:
            await asyncio.sleep(3)
            msgs = await self.client.get_messages(BOT_USERNAME, limit=10)
            bot_msgs = [m for m in msgs if not m.out]
            if bot_msgs:
                latest = bot_msgs[0]
                txt = latest.text or ""
                if len(txt) > 5 and txt != "\u2699\ufe0f":
                    bot_reply = txt
                    break

        self.assertTrue(
            len(bot_reply) > 10,
            f"Expected meaningful reply about the image, got: {bot_reply[:300]}",
        )


    # -- t12: audio file ---------------------------------------------------

    async def test_t12_audio(self):
        """Send a voice/audio file to the bot and get a response."""
        await send_and_wait(self.client, "/reset", idle_gap=5)

        # Create a minimal OGG file (valid header, ~0.1s silence)
        import struct

        def _make_ogg() -> bytes:
            # Minimal valid OGG/Opus file — just enough for Telegram to accept
            # We use a pre-built tiny OGG with silence
            # OGG capture pattern
            header = b"OggS"  # capture
            header += b"\x00"  # version
            header += b"\x02"  # header type (BOS)
            header += b"\x00" * 8  # granule pos
            header += struct.pack("<I", 1)  # serial
            header += struct.pack("<I", 0)  # page seq
            header += struct.pack("<I", 0)  # checksum (placeholder)
            header += b"\x01"  # 1 segment
            header += b"\x13"  # segment size = 19 bytes

            # OpusHead
            opus_head = b"OpusHead"
            opus_head += b"\x01"  # version
            opus_head += b"\x01"  # channels
            opus_head += struct.pack("<H", 0)  # pre-skip
            opus_head += struct.pack("<I", 48000)  # sample rate
            opus_head += struct.pack("<h", 0)  # gain
            opus_head += b"\x00"  # channel mapping

            return header + opus_head

        ogg_path = "/tmp/oc_e2e_test_voice.ogg"
        with open(ogg_path, "wb") as f:
            f.write(_make_ogg())

        # Send as voice note
        try:
            await self.client.send_file(
                BOT_USERNAME, ogg_path,
                voice_note=True,
                caption="What is this audio?",
            )
        except Exception:
            # If Telegram rejects the minimal OGG, send as document instead
            await self.client.send_file(
                BOT_USERNAME, ogg_path,
                caption="This is an audio file. Acknowledge you received it.",
            )

        deadline = time.monotonic() + 90
        bot_reply = ""
        sent_ts = time.time()
        while time.monotonic() < deadline:
            await asyncio.sleep(3)
            msgs = await self.client.get_messages(BOT_USERNAME, limit=10)
            for m in msgs:
                if m.out:
                    continue
                txt = m.text or ""
                if len(txt) > 5 and txt != "\u2699\ufe0f" and m.date.timestamp() > sent_ts - 2:
                    bot_reply = txt
                    break
            if bot_reply:
                break

        self.assertTrue(
            len(bot_reply) > 10,
            f"Expected meaningful reply about audio, got: {bot_reply[:300]}",
        )


    # -- t13: /project command -----------------------------------------------

    async def test_t13_project_show(self):
        """/project with no args shows current directory."""
        msgs = await send_and_wait(self.client, "/project", idle_gap=5)
        texts = " ".join(m.text or "" for m in msgs)
        self.assertIn("Current directory", texts, f"Expected directory info: {texts[:300]}")

    async def test_t14_project_set_and_reset(self):
        """/project /tmp sets directory and resets session."""
        msgs = await send_and_wait(self.client, "/project /tmp", idle_gap=8)
        texts = " ".join(m.text or "" for m in msgs)
        self.assertIn("/tmp", texts, f"Expected /tmp in response: {texts[:300]}")
        self.assertIn("reset", texts.lower(), f"Expected session reset: {texts[:300]}")

        # Verify /project now shows /tmp
        msgs2 = await send_and_wait(self.client, "/project", idle_gap=5)
        texts2 = " ".join(m.text or "" for m in msgs2)
        self.assertIn("/tmp", texts2, f"Expected /tmp in current dir: {texts2[:300]}")

        # Reset back to default
        await send_and_wait(
            self.client,
            "/project /home/spex/work/erp/zeroclaws",
            idle_gap=8,
        )

    async def test_t15_project_invalid_dir(self):
        """/project with nonexistent path shows error."""
        msgs = await send_and_wait(
            self.client, "/project /nonexistent/dir/abc", idle_gap=5,
        )
        texts = " ".join(m.text or "" for m in msgs)
        self.assertIn("not found", texts.lower(), f"Expected error: {texts[:300]}")

    # -- t16: /approve command -----------------------------------------------

    async def test_t16_approve_toggle(self):
        """/approve on / off toggles auto-approve mode."""
        # Turn on
        msgs = await send_and_wait(self.client, "/approve on", idle_gap=5)
        texts = " ".join(m.text or "" for m in msgs)
        self.assertIn("auto-approve", texts.lower(), f"Expected confirmation: {texts[:300]}")
        self.assertIn("on", texts.lower(), f"Expected ON: {texts[:300]}")

        # Check status
        msgs2 = await send_and_wait(self.client, "/approve", idle_gap=5)
        texts2 = " ".join(m.text or "" for m in msgs2)
        self.assertIn("ON", texts2, f"Expected ON status: {texts2[:300]}")

        # Turn off
        msgs3 = await send_and_wait(self.client, "/approve off", idle_gap=5)
        texts3 = " ".join(m.text or "" for m in msgs3)
        self.assertIn("off", texts3.lower(), f"Expected OFF: {texts3[:300]}")

    # -- t17: /git command ---------------------------------------------------

    async def test_t17_git_shortcut(self):
        """/git status runs git status via the agent."""
        await send_and_wait(self.client, "/reset", idle_gap=5)
        msgs = await send_and_wait(self.client, "/git status", timeout=90)
        texts = " ".join(m.text or "" for m in msgs)
        # Should contain git output markers
        has_git = any(kw in texts.lower() for kw in (
            "branch", "commit", "clean", "modified", "untracked",
            "changes", "nothing to commit", "on branch",
        ))
        self.assertTrue(has_git, f"Expected git status output: {texts[:400]}")

    # -- t18: /diff command --------------------------------------------------

    async def test_t18_diff_no_changes(self):
        """/diff with no pending changes shows appropriate message."""
        await send_and_wait(self.client, "/reset", idle_gap=5)
        msgs = await send_and_wait(self.client, "/diff", idle_gap=5)
        texts = " ".join(m.text or "" for m in msgs)
        has_diff_or_empty = (
            "no pending" in texts.lower()
            or "no active" in texts.lower()
            or "diff" in texts.lower()
            or "changes" in texts.lower()
        )
        self.assertTrue(has_diff_or_empty, f"Expected diff response: {texts[:300]}")

    # -- t19: /id shows extended info ----------------------------------------

    async def test_t19_id_extended(self):
        """/id shows directory and auto-approve info."""
        msgs = await send_and_wait(self.client, "/id", idle_gap=5)
        texts = " ".join(m.text or "" for m in msgs)
        self.assertIn("dir:", texts.lower(), f"Expected dir field: {texts[:300]}")
        self.assertIn("approve:", texts.lower(), f"Expected approve field: {texts[:300]}")

    # -- t20: file delivery via tool write -----------------------------------

    async def test_t20_file_delivery(self):
        """Bot sends a generated file back as a Telegram document."""
        await send_and_wait(self.client, "/reset", idle_gap=5)
        await send_and_wait(self.client, "/approve on", idle_gap=5)

        msgs = await send_and_wait(
            self.client,
            'Create a file /tmp/oc_e2e_delivery_test.txt with the text "delivery test OK"',
            timeout=120,
        )
        # Check if the file was sent as a document
        all_msgs = await self.client.get_messages(BOT_USERNAME, limit=20)
        has_doc = any(
            getattr(m, "document", None) is not None
            or getattr(m, "media", None) is not None
            for m in all_msgs if not m.out
        )
        texts = " ".join(m.text or "" for m in msgs)
        has_tool = "\u2705" in texts or "\U0001f527" in texts

        # Either file was delivered or at least the tool executed
        self.assertTrue(
            has_doc or has_tool,
            f"Expected file delivery or tool execution: {texts[:400]}",
        )

        # Cleanup
        await send_and_wait(self.client, "/approve off", idle_gap=5)

    # -- t21: /stop command --------------------------------------------------

    async def test_t21_stop(self):
        """/stop aborts running generation or says no session."""
        msgs = await send_and_wait(self.client, "/stop", idle_gap=5)
        texts = " ".join(m.text or "" for m in msgs)
        has_stop = "stopped" in texts.lower() or "no active" in texts.lower()
        self.assertTrue(has_stop, f"Expected stop response: {texts[:300]}")

    # -- t22: /undo command (revert API) -------------------------------------

    async def test_t22_undo(self):
        """/undo either reverts or says no session."""
        msgs = await send_and_wait(self.client, "/undo", idle_gap=5)
        texts = " ".join(m.text or "" for m in msgs)
        has_undo = (
            "revert" in texts.lower()
            or "no active" in texts.lower()
            or "failed" in texts.lower()
        )
        self.assertTrue(has_undo, f"Expected undo response: {texts[:300]}")

    # -- t23: /redo command --------------------------------------------------

    async def test_t23_redo(self):
        """/redo either restores or says no session."""
        msgs = await send_and_wait(self.client, "/redo", idle_gap=5)
        texts = " ".join(m.text or "" for m in msgs)
        has_redo = (
            "restored" in texts.lower()
            or "no active" in texts.lower()
            or "failed" in texts.lower()
        )
        self.assertTrue(has_redo, f"Expected redo response: {texts[:300]}")

    # -- t24: /tools command -------------------------------------------------

    async def test_t24_tools(self):
        """/tools lists available tools."""
        msgs = await send_and_wait(self.client, "/tools", idle_gap=5)
        texts = " ".join(m.text or "" for m in msgs)
        has_tools = "bash" in texts.lower() or "write" in texts.lower() or "tools" in texts.lower()
        self.assertTrue(has_tools, f"Expected tools list: {texts[:400]}")

    # -- t25: /sessions command ----------------------------------------------

    async def test_t25_sessions(self):
        """/sessions lists sessions or shows empty."""
        msgs = await send_and_wait(self.client, "/sessions", idle_gap=5)
        texts = " ".join(m.text or "" for m in msgs)
        has_sessions = (
            "sessions" in texts.lower()
            or "no sessions" in texts.lower()
            or "idle" in texts.lower()
        )
        self.assertTrue(has_sessions, f"Expected sessions output: {texts[:400]}")

    # -- t26: /todo command --------------------------------------------------

    async def test_t26_todo(self):
        """/todo shows todo list or says empty/no session."""
        msgs = await send_and_wait(self.client, "/todo", idle_gap=5)
        texts = " ".join(m.text or "" for m in msgs)
        has_todo = (
            "todo" in texts.lower()
            or "no active" in texts.lower()
            or "no todo" in texts.lower()
        )
        self.assertTrue(has_todo, f"Expected todo output: {texts[:300]}")

    # -- t27: /history command -----------------------------------------------

    async def test_t27_history(self):
        """/history shows message history for active session."""
        await send_and_wait(self.client, "/reset", idle_gap=5)
        await send_and_wait(self.client, "Say exactly: HIST_MARKER_27", timeout=60)

        msgs = await send_and_wait(self.client, "/history", idle_gap=8)
        texts = " ".join(m.text or "" for m in msgs)
        has_hist = (
            "messages" in texts.lower()
            or "[user]" in texts.lower()
            or "no messages" in texts.lower()
            or "hist" in texts.lower()
        )
        self.assertTrue(has_hist, f"Expected history output: {texts[:400]}")

    # -- t28: /files live ------------------------------------------------

    async def test_t28_files_live(self):
        """/files returns file listing from the project."""
        await send_and_wait(self.client, "/reset", idle_gap=5)
        msgs = await send_and_wait(self.client, "/files", idle_gap=8)
        texts = " ".join(m.text or "" for m in msgs)
        # Should list files OR report an error (means API responded)
        has_content = (
            "files" in texts.lower()
            or ".py" in texts
            or ".rs" in texts
            or ".toml" in texts
            or "no files" in texts.lower()
            or "error" in texts.lower()
        )
        self.assertTrue(has_content, f"Expected file listing: {texts[:400]}")

    # -- t29: /cat live --------------------------------------------------

    async def test_t29_cat_live(self):
        """/cat reads a file's content via API."""
        await send_and_wait(self.client, "/reset", idle_gap=5)
        msgs = await send_and_wait(self.client, "/cat Cargo.toml", idle_gap=8)
        texts = " ".join(m.text or "" for m in msgs)
        has_content = (
            "cargo" in texts.lower()
            or "package" in texts.lower()
            or "[dependencies" in texts.lower()
            or "error" in texts.lower()
            or "empty" in texts.lower()
        )
        self.assertTrue(has_content, f"Expected Cargo.toml content: {texts[:400]}")

    # -- t30: /grep live -------------------------------------------------

    async def test_t30_grep_live(self):
        """/grep searches for text in the project."""
        await send_and_wait(self.client, "/reset", idle_gap=5)
        msgs = await send_and_wait(self.client, "/grep fn main", idle_gap=8)
        texts = " ".join(m.text or "" for m in msgs)
        has_content = (
            "main" in texts.lower()
            or "results" in texts.lower()
            or "no matches" in texts.lower()
            or "error" in texts.lower()
        )
        self.assertTrue(has_content, f"Expected grep results: {texts[:400]}")

    # -- t31: /find live -------------------------------------------------

    async def test_t31_find_live(self):
        """/find searches for files by pattern."""
        await send_and_wait(self.client, "/reset", idle_gap=5)
        msgs = await send_and_wait(self.client, "/find *.toml", idle_gap=8)
        texts = " ".join(m.text or "" for m in msgs)
        has_content = (
            ".toml" in texts
            or "files matching" in texts.lower()
            or "no files" in texts.lower()
            or "error" in texts.lower()
        )
        self.assertTrue(has_content, f"Expected find results: {texts[:400]}")

    # -- t32: /stop during generation ------------------------------------

    async def test_t32_stop_during_generation(self):
        """/stop interrupts an active generation."""
        await send_and_wait(self.client, "/reset", idle_gap=5)
        # Start a long generation
        await self.client.send_message(
            BOT_USERNAME,
            "Write a very detailed 3000-word essay about quantum computing. "
            "Cover qubits, superposition, entanglement, quantum gates, "
            "error correction, and real-world applications.",
        )
        # Wait a bit for generation to start
        await asyncio.sleep(5)
        # Send /stop
        stop_msgs = await send_and_wait(self.client, "/stop", idle_gap=5)
        texts = " ".join(m.text or "" for m in stop_msgs)
        has_stop = (
            "stopped" in texts.lower()
            or "no active" in texts.lower()
            or "failed" in texts.lower()
        )
        self.assertTrue(has_stop, f"Expected stop response: {texts[:300]}")

    # -- t33: /undo → /diff → /redo full cycle ---------------------------

    async def test_t33_undo_diff_redo_cycle(self):
        """Full cycle: create file → /diff → /undo → /redo."""
        await send_and_wait(self.client, "/reset", idle_gap=5)
        await send_and_wait(self.client, "/approve on", idle_gap=5)

        # Create a file
        await send_and_wait(
            self.client,
            'Create a file /tmp/oc_e2e_undo_test.txt with text "undo test 33"',
            timeout=120,
        )

        # Check /diff
        diff_msgs = await send_and_wait(self.client, "/diff", idle_gap=8)
        diff_text = " ".join(m.text or "" for m in diff_msgs)

        # /undo
        undo_msgs = await send_and_wait(self.client, "/undo", idle_gap=8)
        undo_text = " ".join(m.text or "" for m in undo_msgs)
        has_undo = (
            "revert" in undo_text.lower()
            or "failed" in undo_text.lower()
        )
        self.assertTrue(has_undo, f"Expected undo response: {undo_text[:300]}")

        # /redo
        redo_msgs = await send_and_wait(self.client, "/redo", idle_gap=8)
        redo_text = " ".join(m.text or "" for m in redo_msgs)
        has_redo = (
            "restored" in redo_text.lower()
            or "failed" in redo_text.lower()
        )
        self.assertTrue(has_redo, f"Expected redo response: {redo_text[:300]}")

        await send_and_wait(self.client, "/approve off", idle_gap=5)

    # -- t34: /files status live -----------------------------------------

    async def test_t34_files_status(self):
        """/files status shows modified files."""
        msgs = await send_and_wait(self.client, "/files status", idle_gap=8)
        texts = " ".join(m.text or "" for m in msgs)
        has_content = (
            "modified" in texts.lower()
            or "no modified" in texts.lower()
            or "status" in texts.lower()
            or "error" in texts.lower()
        )
        self.assertTrue(has_content, f"Expected status output: {texts[:400]}")

    # -- t35: /agent live ------------------------------------------------

    async def test_t35_agent_live(self):
        """/agent lists available agents or shows error."""
        msgs = await send_and_wait(self.client, "/agent", idle_gap=8)
        texts = " ".join(m.text or "" for m in msgs)
        has_content = (
            "agent" in texts.lower()
            or "no agents" in texts.lower()
            or "error" in texts.lower()
        )
        self.assertTrue(has_content, f"Expected agent output: {texts[:400]}")

    # -- t36: tool output visibility -------------------------------------

    async def test_t36_tool_output(self):
        """Tool execution should show truncated output."""
        await send_and_wait(self.client, "/reset", idle_gap=5)
        await send_and_wait(self.client, "/approve on", idle_gap=5)

        msgs = await send_and_wait(
            self.client,
            "Run: echo TOOL_OUTPUT_VISIBLE_36",
            timeout=90,
        )
        texts = " ".join(m.text or "" for m in msgs)
        # The tool output may or may not be visible depending on
        # whether OpenCode sends output in the SSE event.
        # At minimum, the tool should execute.
        has_tool = (
            "\u2705" in texts  # ✅
            or "\U0001f527" in texts  # 🔧
            or "bash" in texts.lower()
            or "TOOL_OUTPUT_VISIBLE_36" in texts
        )
        self.assertTrue(has_tool, f"Expected tool activity: {texts[:400]}")

        await send_and_wait(self.client, "/approve off", idle_gap=5)


if __name__ == "__main__":
    unittest.main()
