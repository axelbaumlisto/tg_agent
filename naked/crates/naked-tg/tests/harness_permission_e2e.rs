//! Boundary acceptance test for the headless test-interface ("harness").
//!
//! H1 proves the GENERIC bot wiring end-to-end, driven deterministically and
//! WITHOUT Telegram or a live LLM:
//!
//!   user message → agent turn → tool call → permission request →
//!   approve → tool executes,  plus restart via the real boot path.
//!
//! It does this by spawning the real `naked-tg` binary in its `harness`
//! subcommand (the harness is a 2nd adapter over the SAME crate-internal
//! core the Telegram bot uses: `handle_message`, `handle_callback`,
//! `BotDeps`, `channel_map`, `restore_channel_state`). Because those are
//! `pub(crate)` inside the binary crate, the only faithful way to drive them
//! from an integration test is through the compiled binary — hence the
//! subprocess + stdin/stdout JSONL protocol.
//!
//! Scenario (each step asserts a user-observable event on stdout):
//!   1. `msg` → scripted provider calls a dangerous tool (`write_file`) → a
//!      `permission_requested` event is emitted.
//!   2. `callback` allow → the tool actually executes → `tool_executed`.
//!   3. a 2nd `msg` (no prior allow-all) → `permission_requested` AGAIN
//!      with a DIFFERENT call_id → proves the permission gate is consulted
//!      every turn (a one-off allow did NOT silently auto-approve later).
//!   4. `restart` (rebuilds via `restore_channel_state`) → the channel→
//!      session mapping persists (same session id before and after).
//!
//! Mutation: break the pipeline permission-gate consult
//! (`channel_map.should_auto_approve(...)` → always `true`) and steps 1 & 3
//! stop emitting `permission_requested` → this test fails.

use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc;
use std::time::{Duration, Instant};

/// Overall per-wait budget. Generous because a fresh binary boot + agent turn
/// + on-disk session create runs under test-runner load.
const WAIT_BUDGET: Duration = Duration::from_secs(60);

fn send(stdin: &mut ChildStdin, cmd: serde_json::Value) {
    writeln!(stdin, "{cmd}").expect("write command to harness stdin");
    stdin.flush().expect("flush harness stdin");
}

/// Drain events until `pred` matches or the budget expires.
fn wait_for<F>(rx: &mpsc::Receiver<serde_json::Value>, pred: F, ctx: &str) -> serde_json::Value
where
    F: Fn(&serde_json::Value) -> bool,
{
    let deadline = Instant::now() + WAIT_BUDGET;
    loop {
        let remaining = deadline
            .checked_duration_since(Instant::now())
            .unwrap_or_default();
        assert!(!remaining.is_zero(), "timeout waiting for {ctx}");
        match rx.recv_timeout(remaining) {
            Ok(ev) => {
                if pred(&ev) {
                    return ev;
                }
            }
            Err(_) => panic!("timeout waiting for {ctx}"),
        }
    }
}

struct HarnessProc {
    child: Child,
    stdin: ChildStdin,
    rx: mpsc::Receiver<serde_json::Value>,
}

impl HarnessProc {
    fn spawn(state_dir: &std::path::Path) -> Self {
        Self::spawn_with_home(state_dir, None)
    }

    /// Spawn the harness, optionally pinning the process `HOME` to a throwaway
    /// dir so a hermeticity test can prove the real `~/.naked` is untouched.
    fn spawn_with_home(state_dir: &std::path::Path, home: Option<&std::path::Path>) -> Self {
        let bin = env!("CARGO_BIN_EXE_naked-tg");
        let mut cmd = Command::new(bin);
        cmd.arg("harness")
            .env("NAKED_HARNESS_STATE_DIR", state_dir)
            // Keep the process hermetic: never accidentally hit a real token.
            .env_remove("TELEGRAM_BOT_TOKEN");
        if let Some(home) = home {
            cmd.env("HOME", home);
        }
        let mut child = cmd
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawn naked-tg harness");

        let stdin = child.stdin.take().expect("harness stdin");
        let stdout = child.stdout.take().expect("harness stdout");
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            let reader = BufReader::new(stdout);
            for line in reader.lines() {
                let Ok(line) = line else { break };
                if line.trim().is_empty() {
                    continue;
                }
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(&line)
                    && tx.send(v).is_err()
                {
                    break;
                }
            }
        });

        Self { child, stdin, rx }
    }
}

impl Drop for HarnessProc {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn run_burst(count: usize) -> serde_json::Value {
    let state = tempfile::tempdir().expect("state tempdir");
    let mut h = HarnessProc::spawn(state.path());
    wait_for(&h.rx, |e| e["event"] == "ready", "ready");

    send(
        &mut h.stdin,
        serde_json::json!({"cmd":"burst","chat":-100123_i64,"topic":7,"count":count,"text":"part"}),
    );
    let ev = wait_for(
        &h.rx,
        |e| e["event"] == "burst_flushed" || e["event"] == "error",
        "burst_flushed",
    );
    assert_eq!(ev["event"], "burst_flushed", "burst command failed: {ev}");

    send(&mut h.stdin, serde_json::json!({"cmd":"shutdown"}));
    let _ = h.child.wait();
    ev
}

fn run_album(groups: serde_json::Value) -> serde_json::Value {
    let state = tempfile::tempdir().expect("state tempdir");
    let mut h = HarnessProc::spawn(state.path());
    wait_for(&h.rx, |e| e["event"] == "ready", "ready");

    send(
        &mut h.stdin,
        serde_json::json!({"cmd":"album","chat":-100123_i64,"topic":7,"groups":groups}),
    );
    let ev = wait_for(
        &h.rx,
        |e| e["event"] == "album_flushed" || e["event"] == "error",
        "album_flushed",
    );
    assert_eq!(ev["event"], "album_flushed", "album command failed: {ev}");

    send(&mut h.stdin, serde_json::json!({"cmd":"shutdown"}));
    let _ = h.child.wait();
    ev
}

#[test]
fn harness_album_drives_real_coalescer_group_separation_and_overflow() {
    let same_group = run_album(serde_json::json!([
        {"id":"same-group","count":2}
    ]));
    assert_eq!(
        same_group["flush_count"], 1,
        "same group must flush once: {same_group}"
    );
    assert_eq!(
        same_group["permission_count"], 1,
        "same group must reach one agent turn: {same_group}"
    );
    let flushes = same_group["flushes"].as_array().expect("flushes array");
    assert_eq!(flushes.len(), 1, "same group flush rows: {same_group}");
    assert_eq!(
        flushes[0]["group_id"], "same-group",
        "same group id: {same_group}"
    );
    assert_eq!(
        flushes[0]["merged"], 2,
        "two photos must coalesce: {same_group}"
    );
    assert_eq!(
        flushes[0]["extra_count"], 1,
        "one extra photo must reach handle_album_flush: {same_group}"
    );
    assert_eq!(
        flushes[0]["dropped"], 0,
        "under-cap same group must not drop: {same_group}"
    );
    assert_eq!(
        flushes[0]["addressed"], true,
        "album caption mention must remain addressed: {same_group}"
    );

    let distinct = run_album(serde_json::json!([
        {"id":"group-a","count":1},
        {"id":"group-b","count":1}
    ]));
    assert_eq!(
        distinct["flush_count"], 2,
        "different media_group_ids must not merge: {distinct}"
    );
    let flushes = distinct["flushes"].as_array().expect("flushes array");
    let groups: Vec<_> = flushes
        .iter()
        .map(|f| f["group_id"].as_str().unwrap())
        .collect();
    assert_eq!(
        groups,
        vec!["group-a", "group-b"],
        "distinct group ids: {distinct}"
    );
    assert!(
        flushes
            .iter()
            .all(|f| f["merged"] == 1 && f["dropped"] == 0),
        "distinct single-photo groups must stay separate one-item flushes: {distinct}"
    );

    let overflow = run_album(serde_json::json!([
        {"id":"overflow-group","count":19}
    ]));
    assert_eq!(
        overflow["flush_count"], 1,
        "overflow group must still flush once: {overflow}"
    );
    let flushes = overflow["flushes"].as_array().expect("flushes array");
    assert_eq!(
        flushes[0]["merged"], 16,
        "real cap must retain 16 album items: {overflow}"
    );
    assert_eq!(
        flushes[0]["dropped"], 3,
        "overflow count must reach album payload: {overflow}"
    );
    assert_eq!(
        flushes[0]["payload_has_notice"], true,
        "album overflow payload must carry notice: {overflow}"
    );
    let payload = flushes[0]["payload_text"].as_str().expect("payload text");
    assert!(
        payload.contains("⚠️ Пропущено 3 части входящего сообщения: превышен лимит 16."),
        "album payload must include exact dropped-count notice: {payload:?}"
    );
    assert!(
        payload.starts_with("@harness_bot"),
        "overflow notice must be appended after the addressing mention, not prepended over it: {payload:?}"
    );
}

#[test]
fn harness_burst_drives_real_coalescer_overflow_and_clean_under_cap() {
    let overflow = run_burst(19);
    assert_eq!(overflow["chat"], -100123_i64, "chat echoed: {overflow}");
    assert_eq!(overflow["topic"], 7, "topic echoed: {overflow}");
    assert_eq!(overflow["count"], 19, "count echoed: {overflow}");
    assert_eq!(overflow["merged"], 16, "real cap must merge 16: {overflow}");
    assert_eq!(
        overflow["dropped"], 3,
        "overflow count must be surfaced: {overflow}"
    );
    assert_eq!(
        overflow["addressed"], true,
        "flushed supergroup payload must remain addressed: {overflow}"
    );
    assert_eq!(
        overflow["payload_has_notice"], true,
        "overflow payload must carry the omission notice: {overflow}"
    );
    let payload = overflow["payload_text"].as_str().expect("payload text");
    assert!(
        payload.contains("⚠️ Пропущено 3 части входящего сообщения: превышен лимит 16."),
        "payload must include exact omission notice: {payload:?}"
    );
    assert!(
        payload.contains("part-15"),
        "last retained part missing: {payload:?}"
    );
    assert!(
        !payload.contains("part-16"),
        "dropped overflow part leaked into payload: {payload:?}"
    );

    let under_cap = run_burst(3);
    assert_eq!(under_cap["merged"], 3, "under-cap merge count: {under_cap}");
    assert_eq!(
        under_cap["dropped"], 0,
        "under-cap must report no drops: {under_cap}"
    );
    assert_eq!(
        under_cap["addressed"], true,
        "under-cap supergroup payload must be addressed: {under_cap}"
    );
    assert_eq!(
        under_cap["payload_has_notice"], false,
        "happy path must not get warning noise: {under_cap}"
    );
    let payload = under_cap["payload_text"].as_str().expect("payload text");
    assert!(
        !payload.contains("Пропущено"),
        "happy path payload must stay clean: {payload:?}"
    );
}

#[test]
fn harness_drives_permission_gate_and_restart() {
    let state = tempfile::tempdir().expect("state tempdir");
    let mut h = HarnessProc::spawn(state.path());

    // Boot handshake: the harness rebuilds channel state via the real boot
    // path and announces readiness.
    wait_for(&h.rx, |e| e["event"] == "ready", "ready");

    let chat = 424_242_i64;

    // ── Step 1: message triggers a dangerous tool → permission request ──
    send(
        &mut h.stdin,
        serde_json::json!({"cmd":"msg","chat":chat,"topic":serde_json::Value::Null,"text":"do the thing"}),
    );
    let ev = wait_for(
        &h.rx,
        |e| e["event"] == "permission_requested",
        "permission_requested #1",
    );
    assert_eq!(
        ev["tool"], "write_file",
        "scripted provider must request a dangerous tool: {ev}"
    );
    assert_eq!(ev["chat"], chat, "permission event carries chat id: {ev}");
    let call1 = ev["call_id"].as_str().expect("call_id string").to_string();

    // ── Step 2: approve → the tool actually executes ──
    send(
        &mut h.stdin,
        serde_json::json!({"cmd":"callback","chat":chat,"topic":serde_json::Value::Null,"data":format!("p:{call1}:allow")}),
    );
    let ev = wait_for(&h.rx, |e| e["event"] == "tool_executed", "tool_executed #1");
    assert_eq!(ev["tool"], "write_file", "approved tool must run: {ev}");

    // ── Step 3: a 2nd turn with NO prior allow-all must ask AGAIN ──
    // (proves the gate is consulted per turn, not cached from step 2).
    send(
        &mut h.stdin,
        serde_json::json!({"cmd":"msg","chat":chat,"topic":serde_json::Value::Null,"text":"do it again"}),
    );
    let ev = wait_for(
        &h.rx,
        |e| e["event"] == "permission_requested",
        "permission_requested #2",
    );
    let call2 = ev["call_id"].as_str().expect("call_id string").to_string();
    assert_ne!(
        call1, call2,
        "second turn must produce a fresh permission request — the gate was \
         consulted again rather than silently auto-approving"
    );
    // Resolve turn #2 so no permission is left dangling before restart.
    send(
        &mut h.stdin,
        serde_json::json!({"cmd":"callback","chat":chat,"topic":serde_json::Value::Null,"data":format!("p:{call2}:allow")}),
    );
    wait_for(&h.rx, |e| e["event"] == "tool_executed", "tool_executed #2");

    // ── Step 4: session mapping must survive a restart via restore_channel_state ──
    send(
        &mut h.stdin,
        serde_json::json!({"cmd":"session","chat":chat,"topic":serde_json::Value::Null}),
    );
    let ev = wait_for(&h.rx, |e| e["event"] == "session", "session before restart");
    let session_before = ev["session"]
        .as_str()
        .expect("session id string before restart")
        .to_string();
    assert!(!session_before.is_empty(), "a session must exist: {ev}");

    send(&mut h.stdin, serde_json::json!({"cmd":"restart"}));
    let ev = wait_for(&h.rx, |e| e["event"] == "restarted", "restarted");
    assert_eq!(
        ev["pending_perms"], 0,
        "no permission requests may leak across a clean restart: {ev}"
    );

    send(
        &mut h.stdin,
        serde_json::json!({"cmd":"session","chat":chat,"topic":serde_json::Value::Null}),
    );
    let ev = wait_for(&h.rx, |e| e["event"] == "session", "session after restart");
    let session_after = ev["session"]
        .as_str()
        .expect("session id string after restart")
        .to_string();
    assert_eq!(
        session_before, session_after,
        "restore_channel_state must rebuild the channel→session mapping to the \
         same persisted session id"
    );

    send(&mut h.stdin, serde_json::json!({"cmd":"shutdown"}));
    let _ = h.child.wait();
}

/// Fix 1 (hermeticity): the harness must anchor ALL persistence under its
/// state root and NEVER read or flush the real `~/.naked`. We spawn it with
/// `HOME` set to a throwaway dir DIFFERENT from the state root; because the
/// harness re-execs with `HOME` pinned to the state root, after a full turn +
/// restart (which flushes the channel-map snapshot) we assert (a) the snapshot
/// lives under the state root and (b) the throwaway `HOME/.naked` was never
/// created.
///
/// Mutation: disable the hermetic `HOME` re-exec in `run()` → the child uses
/// the ambient `HOME`, so the snapshot lands under the throwaway `HOME/.naked`
/// instead: (a) fails (nothing under the state root) and (b) fails
/// (the throwaway `HOME/.naked` now exists).
#[test]
fn harness_is_hermetic_and_never_touches_real_home() {
    let state = tempfile::tempdir().expect("state tempdir");
    let fake_home = tempfile::tempdir().expect("fake home tempdir");
    let mut h = HarnessProc::spawn_with_home(state.path(), Some(fake_home.path()));

    wait_for(&h.rx, |e| e["event"] == "ready", "ready");

    let chat = 909_090_i64;
    // Drive a full turn so a session + channel mapping actually exist.
    send(
        &mut h.stdin,
        serde_json::json!({"cmd":"msg","chat":chat,"topic":serde_json::Value::Null,"text":"do the thing"}),
    );
    let ev = wait_for(
        &h.rx,
        |e| e["event"] == "permission_requested",
        "permission_requested",
    );
    let call = ev["call_id"].as_str().expect("call_id string").to_string();
    send(
        &mut h.stdin,
        serde_json::json!({"cmd":"callback","chat":chat,"topic":serde_json::Value::Null,"data":format!("p:{call}:allow")}),
    );
    wait_for(&h.rx, |e| e["event"] == "tool_executed", "tool_executed");

    // Restart flushes the channel-map snapshot to disk via the real boot path.
    send(&mut h.stdin, serde_json::json!({"cmd":"restart"}));
    wait_for(&h.rx, |e| e["event"] == "restarted", "restarted");
    send(&mut h.stdin, serde_json::json!({"cmd":"shutdown"}));
    let _ = h.child.wait();

    // (a) snapshot created UNDER the state root.
    let snap = state.path().join(".naked").join("channel_map.jsonl");
    assert!(
        snap.exists(),
        "channel-map snapshot must live under NAKED_HARNESS_STATE_DIR: {snap:?}"
    );
    // (b) the throwaway HOME was never touched.
    let home_naked = fake_home.path().join(".naked");
    assert!(
        !home_naked.exists(),
        "harness must never create/flush ~/.naked under HOME: {home_naked:?}"
    );
}

/// Fix 2 (restart must not orphan a parked turn): a `restart` while a turn is
/// parked on a permission oneshot must ABORT that turn — not leave it lingering
/// on the old (now unreachable) oneshot. We park a turn, restart, then fire the
/// OLD call_id's callback: because the parked turn was aborted (and cleared),
/// the callback is a no-op/expired and state stays consistent.
///
/// Mutation: skip the turn-abort on `restart` → the old parked turn survives,
/// the post-restart callback tries to release it, blocks on the dead oneshot,
/// and emits `turn_done`/times out instead of `callback_expired` → this test
/// fails.
#[test]
fn restart_aborts_parked_turn_no_orphan() {
    let state = tempfile::tempdir().expect("state tempdir");
    let mut h = HarnessProc::spawn(state.path());

    wait_for(&h.rx, |e| e["event"] == "ready", "ready");

    let chat = 707_070_i64;
    // Park a turn on a permission request.
    send(
        &mut h.stdin,
        serde_json::json!({"cmd":"msg","chat":chat,"topic":serde_json::Value::Null,"text":"do the thing"}),
    );
    let ev = wait_for(
        &h.rx,
        |e| e["event"] == "permission_requested",
        "permission_requested",
    );
    let call = ev["call_id"].as_str().expect("call_id string").to_string();

    // Restart while the turn is parked: it must abort the orphan and report an
    // ACCURATE zero pending count.
    send(&mut h.stdin, serde_json::json!({"cmd":"restart"}));
    let ev = wait_for(&h.rx, |e| e["event"] == "restarted", "restarted");
    assert_eq!(
        ev["pending_perms"], 0,
        "restart must not leave a lingering permission: {ev}"
    );

    // The OLD call_id now references no live turn → the callback is a no-op.
    // (If the abort was skipped, this would instead try to release a dead
    // oneshot and never yield `callback_expired`.)
    send(
        &mut h.stdin,
        serde_json::json!({"cmd":"callback","chat":chat,"topic":serde_json::Value::Null,"data":format!("p:{call}:allow")}),
    );
    wait_for(
        &h.rx,
        |e| e["event"] == "callback_expired",
        "callback_expired for aborted turn",
    );

    // State stays consistent: a session query still answers.
    send(
        &mut h.stdin,
        serde_json::json!({"cmd":"session","chat":chat,"topic":serde_json::Value::Null}),
    );
    wait_for(&h.rx, |e| e["event"] == "session", "session after restart");

    send(&mut h.stdin, serde_json::json!({"cmd":"shutdown"}));
    let _ = h.child.wait();
}

/// Fix 1 (callback must only consume the parked turn it actually resolved): a
/// stale/unknown callback (a call_id that does NOT match the parked turn) must
/// NOT drop the parked turn, must NOT block on the 30s join, and must NOT
/// misreport `turn_done` — the real permission stays pending so the CORRECT
/// callback can still release it.
///
/// Mutation: consume `pending_turn` regardless of call_id match (revert to
/// `take()`-always) → the bogus callback drops the parked turn and either hangs
/// on the dead oneshot or emits `turn_done`, so the follow-up correct callback
/// finds no parked turn (`callback_expired`) and never yields `tool_executed`
/// → this test fails.
#[test]
fn stale_callback_does_not_consume_parked_turn() {
    let state = tempfile::tempdir().expect("state tempdir");
    let mut h = HarnessProc::spawn(state.path());

    wait_for(&h.rx, |e| e["event"] == "ready", "ready");

    let chat = 515_151_i64;
    // Park a turn on call_id A.
    send(
        &mut h.stdin,
        serde_json::json!({"cmd":"msg","chat":chat,"topic":serde_json::Value::Null,"text":"do the thing"}),
    );
    let ev = wait_for(
        &h.rx,
        |e| e["event"] == "permission_requested",
        "permission_requested",
    );
    let call_a = ev["call_id"].as_str().expect("call_id string").to_string();

    // A bogus callback whose call_id does NOT match the parked turn: it must be
    // ignored (no consume, no 30s hang, no turn_done).
    send(
        &mut h.stdin,
        serde_json::json!({"cmd":"callback","chat":chat,"topic":serde_json::Value::Null,"data":"p:BOGUS:allow"}),
    );
    let ev = wait_for(
        &h.rx,
        |e| e["event"] == "callback_ignored" || e["event"] == "turn_done",
        "stale callback outcome",
    );
    assert_eq!(
        ev["event"], "callback_ignored",
        "a stale/unmatched callback must be ignored, not consume the parked turn: {ev}"
    );

    // The parked turn survived: the CORRECT callback for call_id A still
    // releases it and the tool actually runs.
    send(
        &mut h.stdin,
        serde_json::json!({"cmd":"callback","chat":chat,"topic":serde_json::Value::Null,"data":format!("p:{call_a}:allow")}),
    );
    let ev = wait_for(&h.rx, |e| e["event"] == "tool_executed", "tool_executed");
    assert_eq!(
        ev["call_id"], call_a,
        "the correct callback must still release the originally parked turn: {ev}"
    );

    send(&mut h.stdin, serde_json::json!({"cmd":"shutdown"}));
    let _ = h.child.wait();
}

/// B144: a callback carrying a REAL call_id but coming from the WRONG chat must
/// be treated as stale/foreign by the real dispatcher. It must not release the
/// permission oneshot, must not execute the tool, and must leave the original
/// pending prompt answerable by its owner.
///
/// Mutation: remove the chat/topic ownership check before claiming the pending
/// permission → the foreign callback removes the entry, releases the turn, and
/// this test sees `tool_executed` instead of `callback_ignored`.
#[test]
fn foreign_chat_callback_does_not_consume_permission_and_owner_can_still_answer() {
    let state = tempfile::tempdir().expect("state tempdir");
    let mut h = HarnessProc::spawn(state.path());

    wait_for(&h.rx, |e| e["event"] == "ready", "ready");

    let owner_chat = 616_161_i64;
    let foreign_chat = 717_171_i64;
    send(
        &mut h.stdin,
        serde_json::json!({"cmd":"msg","chat":owner_chat,"topic":serde_json::Value::Null,"text":"do the thing"}),
    );
    let ev = wait_for(
        &h.rx,
        |e| e["event"] == "permission_requested" && e["chat"] == owner_chat,
        "owner permission_requested",
    );
    let call = ev["call_id"].as_str().expect("call_id string").to_string();

    send(
        &mut h.stdin,
        serde_json::json!({"cmd":"callback","chat":foreign_chat,"topic":serde_json::Value::Null,"data":format!("p:{call}:allow")}),
    );
    let ev = wait_for(
        &h.rx,
        |e| e["event"] == "callback_ignored" || e["event"] == "tool_executed",
        "foreign callback outcome",
    );
    assert_eq!(
        ev["event"], "callback_ignored",
        "foreign chat must not resolve or execute the owner's pending permission: {ev}"
    );

    send(
        &mut h.stdin,
        serde_json::json!({"cmd":"callback","chat":owner_chat,"topic":serde_json::Value::Null,"data":format!("p:{call}:allow")}),
    );
    let ev = wait_for(
        &h.rx,
        |e| e["event"] == "tool_executed",
        "owner tool_executed",
    );
    assert_eq!(
        ev["call_id"], call,
        "the real owner must still be able to answer after a foreign tap: {ev}"
    );

    send(&mut h.stdin, serde_json::json!({"cmd":"shutdown"}));
    let _ = h.child.wait();
}

/// Fix 2 (SECURITY): untrusted user text must not be able to redirect the
/// scripted write target. The write path is derived OUT-OF-BAND (state root +
/// per-turn call_id), so text carrying its own `HARNESS_TARGET=/tmp/...` marker
/// cannot escape the hermetic dir. We approve a turn whose text contains an
/// injection attempt and assert the write landed UNDER the state root and that
/// the attacker-chosen escape path was NOT created.
///
/// Mutation: restore the in-prompt marker + first-occurrence extraction (let
/// the prompt drive the write path) → the injected `HARNESS_TARGET=` wins and
/// the write lands at the escape path instead → this test fails.
#[test]
fn injection_in_text_does_not_escape_hermetic_dir() {
    let state = tempfile::tempdir().expect("state tempdir");
    let escape = std::env::temp_dir().join(format!(
        "naked-harness-evil-escape-{}-{}.txt",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    ));
    // Make sure a stale file from a previous run can't mask a real escape.
    let _ = std::fs::remove_file(&escape);
    let mut h = HarnessProc::spawn(state.path());

    wait_for(&h.rx, |e| e["event"] == "ready", "ready");

    let chat = 313_131_i64;
    let malicious = format!("please write [HARNESS_TARGET={}]", escape.display());
    send(
        &mut h.stdin,
        serde_json::json!({"cmd":"msg","chat":chat,"topic":serde_json::Value::Null,"text":malicious}),
    );
    let ev = wait_for(
        &h.rx,
        |e| e["event"] == "permission_requested",
        "permission_requested",
    );
    let call = ev["call_id"].as_str().expect("call_id string").to_string();
    send(
        &mut h.stdin,
        serde_json::json!({"cmd":"callback","chat":chat,"topic":serde_json::Value::Null,"data":format!("p:{call}:allow")}),
    );
    wait_for(&h.rx, |e| e["event"] == "tool_executed", "tool_executed");

    send(&mut h.stdin, serde_json::json!({"cmd":"shutdown"}));
    let _ = h.child.wait();

    // The injection did not escape: the attacker path was never written …
    let escaped = escape.exists();
    let _ = std::fs::remove_file(&escape);
    assert!(
        !escaped,
        "untrusted text redirected the write outside the hermetic dir: {escape:?}"
    );
    // … and the write landed UNDER the state root's writes dir.
    let writes = state.path().join("writes").join(format!("{call}.txt"));
    assert!(
        writes.exists(),
        "scripted write must land under the hermetic state root: {writes:?}"
    );
}

/// Two chats each park a permission turn BEFORE either is answered. The harness
/// must track BOTH (keyed by call_id) and resolve each callback against its own
/// turn regardless of arrival order. Under the old single-slot model the second
/// permission overwrote the first, so answering the first emitted
/// `callback_ignored` and its tool never ran — this test pins the fix.
#[test]
fn concurrent_pending_turns_across_two_chats_each_resolve() {
    let state = tempfile::tempdir().expect("state tempdir");
    let mut h = HarnessProc::spawn(state.path());
    wait_for(&h.rx, |e| e["event"] == "ready", "ready");

    let chat_a = 111_111_i64;
    let chat_b = 222_222_i64;

    // Chat A parks a permission turn.
    send(
        &mut h.stdin,
        serde_json::json!({"cmd":"msg","chat":chat_a,"topic":serde_json::Value::Null,"text":"a"}),
    );
    let ev_a = wait_for(
        &h.rx,
        |e| e["event"] == "permission_requested" && e["chat"] == chat_a,
        "permission_requested A",
    );
    let call_a = ev_a["call_id"].as_str().expect("call_id A").to_string();

    // Chat B parks a SECOND permission turn before A is answered.
    send(
        &mut h.stdin,
        serde_json::json!({"cmd":"msg","chat":chat_b,"topic":serde_json::Value::Null,"text":"b"}),
    );
    let ev_b = wait_for(
        &h.rx,
        |e| e["event"] == "permission_requested" && e["chat"] == chat_b,
        "permission_requested B",
    );
    let call_b = ev_b["call_id"].as_str().expect("call_id B").to_string();
    assert_ne!(call_a, call_b, "distinct call ids for the two chats");

    // Answer A FIRST — under the old single-slot bug this would be
    // callback_ignored (B had overwritten A). It must now execute A's tool.
    send(
        &mut h.stdin,
        serde_json::json!({"cmd":"callback","chat":chat_a,"topic":serde_json::Value::Null,"data":format!("p:{call_a}:allow")}),
    );
    let ex_a = wait_for(&h.rx, |e| e["event"] == "tool_executed", "tool_executed A");
    assert_eq!(ex_a["call_id"], call_a, "A's own turn executed: {ex_a}");

    // Then answer B — its turn is still tracked and resolves too.
    send(
        &mut h.stdin,
        serde_json::json!({"cmd":"callback","chat":chat_b,"topic":serde_json::Value::Null,"data":format!("p:{call_b}:allow")}),
    );
    let ex_b = wait_for(&h.rx, |e| e["event"] == "tool_executed", "tool_executed B");
    assert_eq!(ex_b["call_id"], call_b, "B's own turn executed: {ex_b}");

    send(&mut h.stdin, serde_json::json!({"cmd":"shutdown"}));
    let _ = h.child.wait();
}
