"""Verify that subscribe_events correctly splits reasoning/text deltas."""
import asyncio
from opencode_tg.agents.opencode import OpenCodeBackend
from opencode_tg.protocols import ReasoningDelta, TextDelta, SessionIdle


async def main():
    backend = OpenCodeBackend(
        base_url="http://127.0.0.1:14096",
        directory="/home/spex/work/erp/zeroclaws",
    )
    sid = await backend.create_session("verify-reasoning-split")
    print(f"session: {sid}")

    reasoning_chars = 0
    text_chars = 0

    async def listen():
        nonlocal reasoning_chars, text_chars
        async for ev in backend.subscribe_events(sid):
            if isinstance(ev, ReasoningDelta):
                reasoning_chars += len(ev.text)
                print(f"  REASONING +{len(ev.text):>4}  total={reasoning_chars}")
            elif isinstance(ev, TextDelta):
                text_chars += len(ev.text)
                print(f"  TEXT      +{len(ev.text):>4}  total={text_chars}")
            elif isinstance(ev, SessionIdle):
                print("  === IDLE ===")
                return

    task = asyncio.create_task(listen())
    await asyncio.sleep(0.5)
    from opencode_tg.protocols import MessagePart
    await backend.send_prompt(sid, [
        MessagePart(type="text", text="Think step by step: what is 17*23?"),
    ])
    await task
    await backend.close()

    print(f"\nResult: reasoning={reasoning_chars} chars, text={text_chars} chars")
    assert reasoning_chars > 0, "Expected reasoning deltas but got none"
    assert text_chars > 0, "Expected text deltas but got none"
    print("PASS")


asyncio.run(main())
