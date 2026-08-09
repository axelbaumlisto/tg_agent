//! Headless test-interface ("harness") — a second adapter over the SAME core
//! the Telegram bot uses.
//!
//! Ports & adapters: instead of Telegram long-polling, the harness drives the
//! real crate-internal seams — [`crate::message_handler::handle_message`] (via
//! [`BotDeps::handle`]), [`crate::callbacks::handle_callback`], the
//! [`ChannelSessionMap`], and the real boot path
//! [`crate::wiring::channel_state::restore_channel_state`] — from a scriptable
//! stdin/stdout JSONL protocol. This lets a test exercise the full lifecycle
//! (message → turn → tool call → permission request → approve/deny → tool
//! executes, plus restart) deterministically and WITHOUT Telegram or a live
//! LLM.
//!
//! The harness is a thin adapter: it only wires transport + provider + protocol
//! to the real handlers. It contains NO business logic copied from the bot.
//!
//! Two seams are pluggable:
//!   * **Transport** — a local HTTP capture server ([`start_capture_server`])
//!     records the bot's outbound Telegram API calls and returns canned OK
//!     JSON, exactly like the `mock_bot` / `Bot::set_api_url` pattern used by
//!     the existing streaming tests. Permission requests are DERIVED from the
//!     captured permission card (the `sendMessage` carrying a `p:<id>:allow`
//!     inline keyboard).
//!   * **Provider** — H1 uses a deterministic [`ScriptedProvider`] that emits a
//!     dangerous tool call on each fresh user turn, so a permission request
//!     fires without a live model. Its write target is derived OUT-OF-BAND
//!     (from the state root, never from the untrusted prompt), so caller text
//!     can never redirect the write. H2 will select a real provider when
//!     `NAKED_CONFIG` points at a live config; the selection seam is
//!     [`build_provider`].

use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use serde_json::{Value, json};
use teloxide::prelude::*;
use teloxide::types::{CallbackQuery, Message};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{RwLock, mpsc};
use tokio_stream::Stream;

use naked_core::AgentCore;
use naked_core::config::{Config, TelegramConfig};
use naked_core::error::Result as CoreResult;
use naked_core::provider::{ChatRequest, Provider};
use naked_core::types::{ModelInfo, StreamChunk};

use crate::message_handler::BotDeps;
use crate::shared::{ChannelSessionMap, PendingPermissions};

/// Deterministic provider used by H1. On a fresh user turn it emits a single
/// dangerous `write_file` tool call so the permission gate is deterministically
/// triggered; on the follow-up call (after the tool result is fed back) it
/// closes the turn — this is how the real agent loop terminates after one tool
/// round.
///
/// SECURITY: the write target is derived OUT-OF-BAND from [`writes_dir`] (the
/// state root) plus the per-turn call_id, and is stored on the provider at
/// construction. It is NEVER read from the (untrusted) prompt, so no caller
/// text can redirect the write outside the hermetic dir. The target is an
/// absolute path outside the workspace, so `write_file` reports it as
/// `Permission::Dangerous`.
struct ScriptedProvider {
    writes_dir: PathBuf,
    call_seq: AtomicU64,
    /// Per-construction nonce so call_ids are unique ACROSS restarts. Each
    /// `boot_core` builds a fresh provider; a plain counter would reset to 0
    /// and reuse `harness-call-1`, colliding with the write file left by a
    /// pre-restart approved turn. Nanos-at-construction makes every boot's ids
    /// disjoint.
    boot_nonce: u128,
}

impl ScriptedProvider {
    fn new(writes_dir: PathBuf) -> Self {
        let boot_nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        Self {
            writes_dir,
            call_seq: AtomicU64::new(0),
            boot_nonce,
        }
    }

    fn script(&self, request: &ChatRequest) -> Vec<StreamChunk> {
        // Only the LAST message matters: a tool_result being fed back into the
        // loop closes the turn, while a fresh user turn re-triggers a tool call
        // (rather than a prior tool_use anywhere in history short-circuiting the
        // second turn).
        if last_is_tool_result(request) {
            return vec![
                StreamChunk::Text("harness turn complete".into()),
                StreamChunk::Done,
            ];
        }
        let n = self.call_seq.fetch_add(1, Ordering::Relaxed) + 1;
        // Nonce prefix keeps ids unique across restarts (see `boot_nonce`).
        let id = format!("harness-call-{}-{n}", self.boot_nonce);
        // OUT-OF-BAND target: `<state-root>/writes/<call_id>.txt`, independent of
        // any prompt content. The harness re-derives the SAME path from the
        // observed call_id (see `handle_msg`), so it knows which file to poll.
        let path = self.writes_dir.join(format!("{id}.txt"));
        vec![
            StreamChunk::ToolUse {
                id,
                name: "write_file".into(),
                input: json!({
                    "file_path": path.to_string_lossy(),
                    "contents": "written by harness",
                }),
            },
            StreamChunk::Done,
        ]
    }
}

/// True when the last message fed to the provider is a tool_result (the loop
/// feeding a completed tool call back in) rather than a fresh user turn.
fn last_is_tool_result(request: &ChatRequest) -> bool {
    request
        .messages
        .last()
        .and_then(|m| m.get("content"))
        .and_then(|c| c.as_array())
        .map(|blocks| {
            blocks
                .iter()
                .any(|b| b.get("type") == Some(&json!("tool_result")))
        })
        .unwrap_or(false)
}

#[async_trait]
impl Provider for ScriptedProvider {
    fn name(&self) -> &str {
        "harness"
    }
    fn models(&self) -> Vec<ModelInfo> {
        vec![ModelInfo {
            provider: "harness".into(),
            model_id: "harness-model".into(),
            display_name: "Harness".into(),
        }]
    }
    async fn stream_chat(
        &self,
        request: ChatRequest,
    ) -> CoreResult<Pin<Box<dyn Stream<Item = StreamChunk> + Send>>> {
        Ok(Box::pin(tokio_stream::iter(self.script(&request))))
    }
}

/// Provider selection seam. H1 always uses the deterministic scripted provider
/// (with its write target pinned to the hermetic state root); H2 will branch
/// here on `NAKED_CONFIG` to boot the real provider. Kept as a single switch so
/// the live wiring lands in exactly one place.
fn build_provider(base: &Path) -> Box<dyn Provider> {
    // H2 hook: `if std::env::var("NAKED_CONFIG").is_ok() { real provider }`.
    Box::new(ScriptedProvider::new(writes_dir(base)))
}

/// The hermetic directory the scripted provider writes into, derived solely
/// from the state root. Single source of truth shared by [`build_provider`] and
/// the harness poll path (DRY) so both agree on where a tool write lands.
fn writes_dir(base: &Path) -> PathBuf {
    base.join("writes")
}

/// Extract the `<id>` from a permission callback `p:<id>:<action>`. Returns
/// `None` for any other callback shape.
fn callback_call_id(data: &str) -> Option<String> {
    let rest = data.strip_prefix("p:")?;
    let (id, _action) = rest.rsplit_once(':')?;
    (!id.is_empty()).then(|| id.to_string())
}

// ── Transport: local HTTP capture server ────────────────────────────────────

/// One captured outbound Telegram API call.
#[derive(Debug)]
struct CapturedCall {
    method: String,
    body: Value,
}

/// Stand up a local capture server. Returns its base URL (which the `Bot` is
/// pointed at via `set_api_url`, and which raw calls use as `base_url`) and a
/// receiver streaming every captured outbound call.
async fn start_capture_server() -> std::io::Result<(String, mpsc::UnboundedReceiver<CapturedCall>)>
{
    let listener = TcpListener::bind("127.0.0.1:0").await?;
    let addr = listener.local_addr()?;
    let base = format!("http://{addr}");
    let (tx, rx) = mpsc::unbounded_channel();
    tokio::spawn(async move {
        while let Ok((stream, _)) = listener.accept().await {
            let tx = tx.clone();
            tokio::spawn(async move {
                let _ = serve_connection(stream, tx).await;
            });
        }
    });
    Ok((base, rx))
}

async fn serve_connection(
    mut stream: TcpStream,
    tx: mpsc::UnboundedSender<CapturedCall>,
) -> std::io::Result<()> {
    let mut buf = Vec::new();
    let mut tmp = [0u8; 4096];
    let header_end = loop {
        let n = stream.read(&mut tmp).await?;
        if n == 0 {
            return Ok(());
        }
        buf.extend_from_slice(&tmp[..n]);
        if let Some(pos) = find_subslice(&buf, b"\r\n\r\n") {
            break pos + 4;
        }
        if buf.len() > 1 << 20 {
            return Ok(());
        }
    };
    let headers = String::from_utf8_lossy(&buf[..header_end]).to_string();
    let content_len = parse_content_length(&headers);
    let path = parse_request_path(&headers);
    while buf.len() < header_end + content_len {
        let n = stream.read(&mut tmp).await?;
        if n == 0 {
            break;
        }
        buf.extend_from_slice(&tmp[..n]);
    }
    let body_end = (header_end + content_len).min(buf.len());
    let body: Value = serde_json::from_slice(&buf[header_end..body_end]).unwrap_or(Value::Null);
    let method = path.rsplit('/').next().unwrap_or("").to_string();
    let result = canned_result(&method);
    let _ = tx.send(CapturedCall { method, body });

    let payload = format!("{{\"ok\":true,\"result\":{result}}}");
    let response = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        payload.len(),
        payload
    );
    stream.write_all(response.as_bytes()).await?;
    stream.flush().await?;
    Ok(())
}

/// Canned Telegram result body for a method. Methods that return a `Message`
/// (send*/edit*) get a minimal valid Message so teloxide can deserialise the
/// response; everything else (answerCallbackQuery, deleteMessage,
/// sendChatAction, …) returns `true`.
fn canned_result(method: &str) -> String {
    let m = method.to_ascii_lowercase();
    let returns_message = m.contains("editmessage")
        || m == "sendmessage"
        || m == "sendphoto"
        || m == "senddocument"
        || m == "copymessage"
        || m == "forwardmessage";
    if returns_message {
        static MSG_ID: AtomicU64 = AtomicU64::new(1000);
        let id = MSG_ID.fetch_add(1, Ordering::Relaxed);
        format!(
            "{{\"message_id\":{id},\"date\":0,\"chat\":{{\"id\":1,\"type\":\"private\",\"first_name\":\"harness\"}},\"text\":\"ok\"}}"
        )
    } else {
        "true".to_string()
    }
}

fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

fn parse_content_length(headers: &str) -> usize {
    for line in headers.lines() {
        if let Some(v) = line
            .split_once(':')
            .filter(|(k, _)| k.trim().eq_ignore_ascii_case("content-length"))
            .map(|(_, v)| v.trim())
        {
            return v.parse().unwrap_or(0);
        }
    }
    0
}

fn parse_request_path(headers: &str) -> String {
    headers
        .lines()
        .next()
        .and_then(|line| line.split_whitespace().nth(1))
        .unwrap_or("")
        .to_string()
}

/// Detect the permission card in a captured call: a `sendMessage` whose inline
/// keyboard carries a `p:<id>:allow` callback. Returns `(call_id, tool_name)`.
fn permission_card(call: &CapturedCall) -> Option<(String, String)> {
    if !call.method.eq_ignore_ascii_case("sendmessage") {
        return None;
    }
    let call_id = extract_perm_call_id(&call.body)?;
    let tool = extract_tool_name(&call.body).unwrap_or_default();
    Some((call_id, tool))
}

/// Pull `<id>` out of a `p:<id>:allow` callback_data value anywhere in the body.
/// Serialising and splitting on `"` isolates each JSON string token, so the
/// callback_data value stands alone regardless of how teloxide nests the markup.
fn extract_perm_call_id(body: &Value) -> Option<String> {
    let s = body.to_string();
    for token in s.split('"') {
        if let Some(rest) = token.strip_prefix("p:")
            && let Some(id) = rest.strip_suffix(":allow")
            && !id.is_empty()
            && !id.contains(':')
        {
            return Some(id.to_string());
        }
    }
    None
}

/// The permission card text is `🔐 <b>{tool}</b> [level](preview)…`.
fn extract_tool_name(body: &Value) -> Option<String> {
    let text = body.get("text")?.as_str()?;
    let start = text.find("<b>")? + 3;
    let end = text[start..].find("</b>")? + start;
    Some(text[start..end].to_string())
}

// ── State + boot ────────────────────────────────────────────────────────────

/// Everything that survives across commands. The AgentCore, channel map and
/// config are rebuilt on `restart`; the transport (bot + capture server) and
/// the fixed identity are created once.
struct Harness {
    config: Config,
    agent: Arc<AgentCore>,
    channel_map: Arc<ChannelSessionMap>,
    pending_perms: PendingPermissions,
    album_buffer: crate::album::InboundCoalescer,
    task_tracker: Arc<tokio::sync::Semaphore>,
    bot: Bot,
    bot_token: Arc<String>,
    bot_identity: Arc<naked_tg::bot_identity::BotIdentity>,
    http_client: Arc<reqwest::Client>,
    base_url: Arc<String>,
    captured_rx: mpsc::UnboundedReceiver<CapturedCall>,
    base_dir: PathBuf,
    msg_seq: i32,
    /// Turns that emitted a permission card and are parked waiting for a
    /// callback to release them, keyed by call_id. A map (not a single slot)
    /// so multiple chats/topics can each have a pending permission concurrently
    /// — a callback resolves ITS own turn regardless of arrival order,
    /// faithfully modelling the real bot's multi-chat pending-permission state.
    pending_turns: std::collections::HashMap<String, PendingTurn>,
}

struct PendingTurn {
    join: tokio::task::JoinHandle<()>,
    tool: String,
    call_id: String,
    target: PathBuf,
}

/// Guard env var: set on the re-exec'd child so [`run`] redirects `HOME`
/// exactly once (see [`reexec_hermetic`]).
const HERMETIC_GUARD: &str = "NAKED_HARNESS_HERMETIC";

/// Resolve the hermetic state root for this harness process. Prefers
/// `NAKED_HARNESS_STATE_DIR`, accepts `NAKED_STATE_ROOT` as the shorter manual
/// harness alias, otherwise a UNIQUE per-process tempdir so a bare
/// `naked-tg harness` can never read or write the real `~/.naked`. `run`
/// installs this as `HOME` (via a one-time re-exec) so EVERY persistence path
/// — sessions store, channel-map snapshot + restore, and naked-core telemetry
/// / memory (all anchored at `HOME/.naked`) — lands under it.
fn base_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("NAKED_HARNESS_STATE_DIR") {
        return PathBuf::from(dir);
    }
    if let Ok(dir) = std::env::var("NAKED_STATE_ROOT") {
        return PathBuf::from(dir);
    }
    let pid = std::process::id();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    std::env::temp_dir().join(format!("naked-harness-{pid}-{nanos}"))
}

/// Re-exec this binary once with `HOME` (and `NAKED_HARNESS_STATE_DIR`) pinned
/// to the hermetic root, so ALL naked-core persistence — channel-map snapshot,
/// model-health / observation telemetry, and the memory store, every one of
/// which resolves `HOME/.naked` internally — lands under the root and NEVER
/// under the real `~/.naked`. In-process `std::env::set_var` needs `unsafe`
/// (forbidden workspace-wide), so we redirect via a child process `Command`
/// env, which is safe. The `HERMETIC_GUARD` marker makes this run at most once.
/// stdin/stdout/stderr are inherited, so the JSONL protocol is transparent.
fn reexec_hermetic() -> ! {
    let root = base_dir();
    let _ = std::fs::create_dir_all(&root);
    let exe = std::env::current_exe().expect("harness current_exe");
    let status = std::process::Command::new(exe)
        .arg("harness")
        .env(HERMETIC_GUARD, "1")
        .env("HOME", &root)
        .env("NAKED_HARNESS_STATE_DIR", &root)
        .status()
        .expect("re-exec hermetic harness");
    std::process::exit(status.code().unwrap_or(1));
}

fn build_config(base: &Path) -> Config {
    Config {
        workspace: base.join("workspace"),
        session_dir: base.join("sessions"),
        default_provider: "harness".into(),
        default_model: "harness-model".into(),
        telegram: TelegramConfig {
            coalesce_text_ms: 50,
            ..Default::default()
        },
        ..Default::default()
    }
}

impl Harness {
    async fn boot(base_dir: PathBuf) -> std::io::Result<Self> {
        std::fs::create_dir_all(base_dir.join("workspace"))?;
        std::fs::create_dir_all(base_dir.join("sessions"))?;

        let (base, captured_rx) = start_capture_server().await?;
        let bot_token = "0:HARNESS".to_string();
        let bot = Bot::new(&bot_token)
            .set_api_url(reqwest::Url::parse(&base).expect("valid capture url"));

        let config = build_config(&base_dir);
        let (agent, channel_map) = boot_core(&config, &base_dir).await;

        Ok(Self {
            config,
            agent,
            channel_map,
            pending_perms: Arc::new(RwLock::new(std::collections::HashMap::new())),
            album_buffer: crate::album::InboundCoalescer::default(),
            task_tracker: Arc::new(tokio::sync::Semaphore::new(50)),
            bot,
            bot_token: Arc::new(bot_token),
            bot_identity: Arc::new(naked_tg::bot_identity::BotIdentity {
                id: 0,
                username: "harness_bot".to_string(),
            }),
            http_client: Arc::new(reqwest::Client::new()),
            base_url: Arc::new(base),
            captured_rx,
            base_dir,
            msg_seq: 0,
            pending_turns: std::collections::HashMap::new(),
        })
    }

    /// Rebuild the AgentCore + channel map via the REAL boot path — the same
    /// `restore_sessions` + `restore_channel_state` sequence `wiring::build`
    /// runs at startup. Returns how many channel→session links were restored.
    async fn restart(&mut self) -> usize {
        // A real process restart kills everything in flight. If a turn is
        // parked on a permission oneshot, ABORT it before rebuilding —
        // otherwise the old task keeps the OLD oneshot while the new (empty)
        // pending map can never release it, leaving an orphan task lingering
        // until timeout while state falsely reports `pending_perms: 0`.
        for (_, pt) in self.pending_turns.drain() {
            pt.join.abort();
        }
        let _ = self.channel_map.flush().await;
        let (agent, channel_map) = boot_core(&self.config, &self.base_dir).await;
        self.agent = agent;
        self.channel_map = channel_map;
        // Fresh permission map, exactly as a real process boot would have.
        self.pending_perms = Arc::new(RwLock::new(std::collections::HashMap::new()));
        // Count of channel→session links that were rebuilt from persisted
        // session metadata (the authoritative restore source).
        self.agent.channel_session_mappings().await.len()
    }

    /// Assemble a fresh `BotDeps` from shared components + current config. Built
    /// per command so `config.telegram.allowed_chat_ids` growth (below) is
    /// always reflected without touching AgentCore state.
    fn deps(&self) -> BotDeps {
        BotDeps {
            bot: self.bot.clone(),
            agent: self.agent.clone(),
            channel_map: self.channel_map.clone(),
            config: self.config.clone(),
            pending_perms: self.pending_perms.clone(),
            http_client: self.http_client.clone(),
            base_url: self.base_url.clone(),
            rate_limiter: naked_tg::rate_limit::RateLimiter::new(),
            attribution_flag: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            bot_token: self.bot_token.clone(),
            bot_identity: self.bot_identity.clone(),
            tg_attach_queue: naked_tg::tg_attach::new_queue(),
            research_scheduler: None,
            per_chat_locks: Arc::new(crate::per_chat_locks::PerChatLocks::new()),
        }
    }

    /// Ensure this chat passes the `is_allowed` gate (empty allow-list denies
    /// all). Harness commands define which chats exist, so we admit each on
    /// first sight — the permission GATE under test is the per-tool one, not
    /// the chat allow-list.
    fn admit_chat(&mut self, chat: i64) {
        if !self.config.telegram.allowed_chat_ids.contains(&chat) {
            self.config.telegram.allowed_chat_ids.push(chat);
        }
    }
}

async fn boot_core(config: &Config, base: &Path) -> (Arc<AgentCore>, Arc<ChannelSessionMap>) {
    let agent = Arc::new(AgentCore::new(config.clone(), build_provider(base)));
    agent.init_self_ref();
    // Same sequence as `wiring::build`: hydrate persisted sessions, then
    // rebuild the channel→session map (+ yolo/allow-list) from that metadata.
    // `HOME` is pinned to the hermetic root by [`run`] before we get here, so
    // the channel-map snapshot + restore path resolve under `<root>/.naked`
    // and never touch the developer's live `~/.naked` state.
    let _ = agent.restore_sessions().await;
    let channel_map = crate::wiring::channel_state::restore_channel_state(&agent).await;
    (agent, channel_map)
}

// ── Protocol loop ───────────────────────────────────────────────────────────

/// Entry point for `naked-tg harness`.
pub(crate) async fn run() {
    // Full hermeticity: before building anything, re-exec once with `HOME`
    // pinned to the state root so no persistence path can reach the real
    // `~/.naked` (see [`reexec_hermetic`]).
    if std::env::var_os(HERMETIC_GUARD).is_none() {
        reexec_hermetic();
    }

    let mut harness = match Harness::boot(base_dir()).await {
        Ok(h) => h,
        Err(e) => {
            eprintln!("harness boot failed: {e}");
            return;
        }
    };

    let mut out = tokio::io::stdout();
    emit(&mut out, json!({"event": "ready"})).await;

    let mut lines = BufReader::new(tokio::io::stdin()).lines();
    while let Ok(Some(line)) = lines.next_line().await {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let cmd: Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(e) => {
                emit(
                    &mut out,
                    json!({"event":"error","message":format!("bad json: {e}")}),
                )
                .await;
                continue;
            }
        };
        match cmd.get("cmd").and_then(Value::as_str) {
            Some("msg") => handle_msg(&mut harness, &cmd, &mut out).await,
            Some("burst") => handle_burst(&mut harness, &cmd, &mut out).await,
            Some("album") => handle_album(&mut harness, &cmd, &mut out).await,
            Some("callback") => handle_callback_cmd(&mut harness, &cmd, &mut out).await,
            Some("session") => handle_session(&mut harness, &cmd, &mut out).await,
            Some("restart") => {
                let restored = harness.restart().await;
                let pending = harness.pending_perms.read().await.len();
                emit(
                    &mut out,
                    json!({"event":"restarted","restored":restored,"pending_perms":pending}),
                )
                .await;
            }
            Some("shutdown") => break,
            other => {
                emit(
                    &mut out,
                    json!({"event":"error","message":format!("unknown cmd: {other:?}")}),
                )
                .await;
            }
        }
    }
}

async fn emit(out: &mut tokio::io::Stdout, value: Value) {
    let line = format!("{value}\n");
    let _ = out.write_all(line.as_bytes()).await;
    let _ = out.flush().await;
}

fn chat_topic(cmd: &Value) -> (i64, Option<i32>) {
    let chat = cmd.get("chat").and_then(Value::as_i64).unwrap_or(0);
    let topic = cmd
        .get("topic")
        .and_then(Value::as_i64)
        .map(|t| t as i32)
        .filter(|t| *t != 0);
    (chat, topic)
}

#[derive(Debug)]
struct BurstObservation {
    merged: usize,
    dropped: usize,
    addressed: bool,
    payload_text: String,
}

#[derive(Debug)]
struct AlbumGroupSpec {
    id: String,
    count: usize,
}

#[derive(Debug)]
struct AlbumObservation {
    group_id: String,
    merged: usize,
    dropped: usize,
    addressed: bool,
    payload_text: String,
    extra_count: usize,
}

async fn handle_burst(harness: &mut Harness, cmd: &Value, out: &mut tokio::io::Stdout) {
    let (chat, topic) = chat_topic(cmd);
    harness.admit_chat(chat);
    let count = cmd.get("count").and_then(Value::as_u64).unwrap_or(0) as usize;
    if !(1..=256).contains(&count) {
        emit(
            out,
            json!({"event":"error","message":"burst count must be in 1..=256"}),
        )
        .await;
        return;
    }
    let text = cmd.get("text").and_then(Value::as_str).unwrap_or("part");
    if text.trim().is_empty() {
        emit(
            out,
            json!({"event":"error","message":"burst text must be non-empty"}),
        )
        .await;
        return;
    }

    let deps = harness.deps();
    let debounce = Duration::from_millis(deps.config.telegram.coalesce_text_ms);
    let (obs_tx, mut obs_rx) = mpsc::unbounded_channel::<BurstObservation>();

    for idx in 0..count {
        harness.msg_seq += 1;
        let part_text = if count == 1 {
            text.to_string()
        } else {
            format!("{text}-{idx}")
        };
        let msg = build_message(
            harness.msg_seq,
            chat,
            topic,
            &part_text,
            &harness.bot_identity,
        );
        let Some(key) = crate::runtime::text_burst_key_if_eligible(&deps, &msg).await else {
            emit(
                out,
                json!({"event":"error","message":"burst message was not eligible for text coalescing","chat":chat,"topic":topic,"index":idx}),
            )
            .await;
            return;
        };

        let deps_for_flush = deps.clone();
        let task_tracker = harness.task_tracker.clone();
        let bot_identity = harness.bot_identity.clone();
        let obs_tx = obs_tx.clone();
        match harness
            .album_buffer
            .submit_text(msg, key, debounce, move |msgs| {
                crate::runtime::handle_text_burst_flush(
                    msgs,
                    deps_for_flush,
                    task_tracker,
                    move |primary, merged| {
                        let payload_text = primary
                            .text()
                            .or_else(|| primary.caption())
                            .unwrap_or_default()
                            .to_string();
                        let dropped = overflow_notice_count(&payload_text).unwrap_or(0);
                        let addressed =
                            naked_tg::bot_identity::is_addressed_to_bot(primary, &bot_identity);
                        let _ = obs_tx.send(BurstObservation {
                            merged,
                            dropped,
                            addressed,
                            payload_text,
                        });
                    },
                )
            })
            .await
        {
            crate::album::Decision::Buffered => {}
            crate::album::Decision::Solo(_) => {
                emit(
                    out,
                    json!({"event":"error","message":"burst unexpectedly bypassed text coalescing","chat":chat,"topic":topic,"index":idx}),
                )
                .await;
                return;
            }
        }
    }
    drop(obs_tx);

    let deadline = tokio::time::Instant::now() + Duration::from_secs(30);
    let mut observation: Option<BurstObservation> = None;
    let mut permission: Option<(String, String)> = None;
    loop {
        if let (Some(obs), Some((call_id, tool))) = (observation.as_ref(), permission.as_ref()) {
            emit(
                out,
                json!({
                    "event":"burst_flushed",
                    "chat":chat,
                    "topic":topic,
                    "count":count,
                    "merged":obs.merged,
                    "dropped":obs.dropped,
                    "addressed":obs.addressed,
                    "payload_has_notice":obs.dropped > 0,
                    "payload_text":obs.payload_text.clone(),
                    "call_id":call_id,
                    "tool":tool,
                }),
            )
            .await;
            return;
        }

        tokio::select! {
            obs = obs_rx.recv(), if observation.is_none() => {
                match obs {
                    Some(obs) => observation = Some(obs),
                    None => {
                        emit(
                            out,
                            json!({"event":"error","message":"burst flush observation channel closed","chat":chat,"topic":topic}),
                        )
                        .await;
                        return;
                    }
                }
            }
            captured = harness.captured_rx.recv(), if permission.is_none() => {
                match captured {
                    Some(call) => {
                        if let Some(card) = permission_card(&call) {
                            permission = Some(card);
                        }
                    }
                    None => {
                        emit(
                            out,
                            json!({"event":"error","message":"capture channel closed while waiting for burst card","chat":chat,"topic":topic}),
                        )
                        .await;
                        return;
                    }
                }
            }
            _ = tokio::time::sleep_until(deadline) => {
                emit(
                    out,
                    json!({"event":"error","message":"timeout waiting for burst flush/card","chat":chat,"topic":topic}),
                )
                .await;
                return;
            }
        }
    }
}

async fn handle_album(harness: &mut Harness, cmd: &Value, out: &mut tokio::io::Stdout) {
    let (chat, topic) = chat_topic(cmd);
    harness.admit_chat(chat);
    let groups = match parse_album_groups(cmd) {
        Ok(groups) => groups,
        Err(message) => {
            emit(out, json!({"event":"error","message":message})).await;
            return;
        }
    };
    let total_count: usize = groups.iter().map(|g| g.count).sum();
    let expected_flushes = groups.len();
    let deps = harness.deps();
    let (obs_tx, mut obs_rx) = mpsc::unbounded_channel::<AlbumObservation>();

    for group in &groups {
        for idx in 0..group.count {
            harness.msg_seq += 1;
            let caption = (idx == 0).then(|| format!("album {}", group.id));
            let msg = build_photo_message(
                harness.msg_seq,
                chat,
                topic,
                &group.id,
                caption.as_deref(),
                &harness.bot_identity,
            );
            let deps_for_flush = deps.clone();
            let task_tracker = harness.task_tracker.clone();
            let bot_identity = harness.bot_identity.clone();
            let obs_tx = obs_tx.clone();
            match harness
                .album_buffer
                .submit_album(msg, move |msgs| {
                    crate::runtime::handle_album_flush(
                        msgs,
                        deps_for_flush,
                        task_tracker,
                        move |primary, extras, merged| {
                            let group_id = primary
                                .media_group_id()
                                .map(|id| id.0.to_string())
                                .unwrap_or_default();
                            let payload_text = primary
                                .caption()
                                .or_else(|| primary.text())
                                .unwrap_or_default()
                                .to_string();
                            let dropped = overflow_notice_count(&payload_text).unwrap_or(0);
                            let addressed =
                                naked_tg::bot_identity::is_addressed_to_bot(primary, &bot_identity);
                            let _ = obs_tx.send(AlbumObservation {
                                group_id,
                                merged,
                                dropped,
                                addressed,
                                payload_text,
                                extra_count: extras.len(),
                            });
                        },
                    )
                })
                .await
            {
                crate::album::Decision::Buffered => {}
                crate::album::Decision::Solo(_) => {
                    emit(
                        out,
                        json!({"event":"error","message":"album message unexpectedly bypassed coalescing","chat":chat,"topic":topic,"group_id":group.id,"index":idx}),
                    )
                    .await;
                    return;
                }
            }
        }
    }
    drop(obs_tx);

    let deadline = tokio::time::Instant::now() + Duration::from_secs(30);
    let mut observations: Vec<AlbumObservation> = Vec::new();
    let mut permissions: Vec<(String, String)> = Vec::new();
    loop {
        if observations.len() == expected_flushes && !permissions.is_empty() {
            observations.sort_by(|a, b| a.group_id.cmp(&b.group_id));
            let flushes: Vec<Value> = observations
                .iter()
                .map(|obs| {
                    json!({
                        "group_id":obs.group_id.clone(),
                        "merged":obs.merged,
                        "dropped":obs.dropped,
                        "addressed":obs.addressed,
                        "payload_has_notice":obs.dropped > 0,
                        "payload_text":obs.payload_text.clone(),
                        "extra_count":obs.extra_count,
                    })
                })
                .collect();
            let permission_tools: Vec<Value> = permissions
                .iter()
                .map(|(call_id, tool)| json!({"call_id":call_id,"tool":tool}))
                .collect();
            emit(
                out,
                json!({
                    "event":"album_flushed",
                    "chat":chat,
                    "topic":topic,
                    "total_count":total_count,
                    "flush_count":observations.len(),
                    "permission_count":permissions.len(),
                    "flushes":flushes,
                    "permissions":permission_tools,
                }),
            )
            .await;
            return;
        }

        tokio::select! {
            obs = obs_rx.recv(), if observations.len() < expected_flushes => {
                match obs {
                    Some(obs) => observations.push(obs),
                    None => {
                        emit(
                            out,
                            json!({"event":"error","message":"album flush observation channel closed","chat":chat,"topic":topic,"observed":observations.len(),"expected":expected_flushes}),
                        )
                        .await;
                        return;
                    }
                }
            }
            captured = harness.captured_rx.recv() => {
                match captured {
                    Some(call) => {
                        if let Some(card) = permission_card(&call) {
                            permissions.push(card);
                        }
                    }
                    None => {
                        emit(
                            out,
                            json!({"event":"error","message":"capture channel closed while waiting for album card","chat":chat,"topic":topic}),
                        )
                        .await;
                        return;
                    }
                }
            }
            _ = tokio::time::sleep_until(deadline) => {
                emit(
                    out,
                    json!({"event":"error","message":"timeout waiting for album flush/card","chat":chat,"topic":topic,"observed":observations.len(),"expected":expected_flushes,"permissions":permissions.len()}),
                )
                .await;
                return;
            }
        }
    }
}

fn parse_album_groups(cmd: &Value) -> Result<Vec<AlbumGroupSpec>, String> {
    let groups = cmd
        .get("groups")
        .and_then(Value::as_array)
        .ok_or_else(|| "album groups must be a non-empty array".to_string())?;
    if groups.is_empty() {
        return Err("album groups must be a non-empty array".to_string());
    }
    let mut out = Vec::with_capacity(groups.len());
    let mut total = 0usize;
    for (idx, group) in groups.iter().enumerate() {
        let id = group
            .get("id")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .ok_or_else(|| format!("album group {idx} id must be non-empty"))?
            .to_string();
        let count = group.get("count").and_then(Value::as_u64).unwrap_or(0) as usize;
        if !(1..=256).contains(&count) {
            return Err(format!("album group {idx} count must be in 1..=256"));
        }
        total = total.saturating_add(count);
        out.push(AlbumGroupSpec { id, count });
    }
    if total > 512 {
        return Err("album total count must be <=512".to_string());
    }
    Ok(out)
}

fn overflow_notice_count(payload: &str) -> Option<usize> {
    let marker = "⚠️ Пропущено ";
    let rest = payload.get(payload.find(marker)? + marker.len()..)?;
    let digits = rest.split_whitespace().next()?;
    digits.parse().ok()
}

async fn handle_msg(harness: &mut Harness, cmd: &Value, out: &mut tokio::io::Stdout) {
    let (chat, topic) = chat_topic(cmd);
    harness.admit_chat(chat);
    let user_text = cmd.get("text").and_then(Value::as_str).unwrap_or("");
    harness.msg_seq += 1;
    // The write target is derived OUT-OF-BAND by the provider (from the state
    // root + the per-turn call_id), so the untrusted user text is passed
    // VERBATIM and can NEVER influence the write path (Fix 2 / security).
    let msg = build_message(
        harness.msg_seq,
        chat,
        topic,
        user_text,
        &harness.bot_identity,
    );

    let deps = harness.deps();
    let mut join = tokio::spawn(async move {
        if let Err(e) = deps.handle(msg, Vec::new()).await {
            eprintln!("handle_message error: {e}");
        }
    });

    // Wait for either the permission card to be captured (turn parks awaiting a
    // callback) or the turn to finish on its own (no gated tool).
    let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(30);
    loop {
        tokio::select! {
            _ = &mut join => {
                emit(out, json!({"event":"turn_done","chat":chat,"topic":topic})).await;
                return;
            }
            captured = harness.captured_rx.recv() => {
                match captured {
                    Some(call) => {
                        if let Some((call_id, tool)) = permission_card(&call) {
                            // Re-derive the same OUT-OF-BAND write path the
                            // provider used for this call_id, so we know which
                            // file to poll for tool execution.
                            let target =
                                writes_dir(&harness.base_dir).join(format!("{call_id}.txt"));
                            // Defensive: ensure no stale file makes a later
                            // deny look executed. Unique nonce ids already
                            // prevent cross-restart collisions; this guarantees
                            // `target.exists()` strictly means THIS turn wrote.
                            let _ = std::fs::remove_file(&target);
                            emit(out, json!({
                                "event":"permission_requested",
                                "call_id":call_id,
                                "tool":tool,
                                "chat":chat,
                                "topic":topic,
                            })).await;
                            harness.pending_turns.insert(
                                call_id.clone(),
                                PendingTurn {
                                    join,
                                    tool,
                                    call_id,
                                    target,
                                },
                            );
                            return;
                        }
                        // Non-card outbound (typing, placeholder, live edit): ignore.
                    }
                    None => return,
                }
            }
            _ = tokio::time::sleep_until(deadline) => {
                // Abort the spawned turn so it cannot keep running untracked
                // (else it may later push to `pending_perms` with no parked
                // turn to resolve it, leaking state).
                join.abort();
                emit(out, json!({"event":"error","message":"timeout waiting for turn/card"})).await;
                return;
            }
        }
    }
}

async fn handle_callback_cmd(harness: &mut Harness, cmd: &Value, out: &mut tokio::io::Stdout) {
    let (chat, topic) = chat_topic(cmd);
    harness.admit_chat(chat);
    let data = cmd.get("data").and_then(Value::as_str).unwrap_or("");
    let q = build_callback_query(harness.msg_seq, chat, topic, data);

    let deps = harness.deps();
    if let Err(e) = crate::callbacks::handle_callback(deps, q, harness.pending_perms.clone()).await
    {
        emit(
            out,
            json!({"event":"error","message":format!("handle_callback: {e}")}),
        )
        .await;
    }

    // Only CONSUME the parked turn when THIS callback actually resolved the
    // permission it is parked on (Fix 1): its call_id must match the parked
    // turn's call_id AND the real handler must have removed that call_id from
    // `pending_perms` (i.e. resolved it — allow/deny/yolo). A stale/unknown
    // callback (wrong or unmatched call_id) leaves the parked turn AND its real
    // pending permission intact, so the correct callback can still match it.
    // Resolve THIS callback's own parked turn (by its call_id), independent of
    // any other chat's pending turns. Resolved only when the call_id is a
    // parked turn AND the real handler removed it from `pending_perms`.
    let cb_id = callback_call_id(data);
    let resolved = match &cb_id {
        Some(cb) => {
            harness.pending_turns.contains_key(cb)
                && !harness.pending_perms.read().await.contains_key(cb)
        }
        None => false,
    };

    if !resolved {
        // Either no turns are parked (e.g. all aborted by a `restart`) or this
        // callback did not match/resolve any parked turn. Emit a no-op event
        // WITHOUT dropping any parked turn or blocking on the 30s join.
        let event = if harness.pending_turns.is_empty() {
            "callback_expired"
        } else {
            "callback_ignored"
        };
        emit(out, json!({"event":event,"chat":chat,"topic":topic})).await;
        return;
    }

    // Releasing the oneshot lets the parked turn run the tool to completion.
    let pt = harness
        .pending_turns
        .remove(&cb_id.expect("resolved implies Some(call_id)"))
        .expect("resolved implies a parked turn");
    let _ = tokio::time::timeout(std::time::Duration::from_secs(30), pt.join).await;
    if pt.target.exists() {
        emit(
            out,
            json!({"event":"tool_executed","tool":pt.tool,"call_id":pt.call_id}),
        )
        .await;
    } else {
        // Denied / timed out: the tool never ran, so its file is absent.
        emit(
            out,
            json!({"event":"turn_done","chat":chat,"topic":topic,"tool_executed":false}),
        )
        .await;
    }
}

async fn handle_session(harness: &mut Harness, cmd: &Value, out: &mut tokio::io::Stdout) {
    let (chat, topic) = chat_topic(cmd);
    let session = harness.channel_map.get(chat, topic).await;
    emit(
        out,
        json!({"event":"session","chat":chat,"topic":topic,"session":session}),
    )
    .await;
}

// ── teloxide value construction ─────────────────────────────────────────────

/// Build a teloxide `Message`. A `null` topic → private chat (always addressed).
/// A non-null topic → supergroup forum message with a leading `@bot` mention
/// entity so the group-addressing gate admits it, plus `message_thread_id` so
/// the (chat, topic) key propagates through `ChatCtx`.
fn build_message(
    seq: i32,
    chat: i64,
    topic: Option<i32>,
    text: &str,
    identity: &naked_tg::bot_identity::BotIdentity,
) -> Message {
    let mut v = match topic {
        None => json!({
            "message_id": seq,
            "date": 0,
            "chat": {"id": chat, "type": "private", "first_name": "harness"},
            "from": {"id": 1, "is_bot": false, "first_name": "harness"},
            "text": text,
        }),
        Some(tid) => {
            let mention = format!("@{}", identity.username);
            let full = format!("{mention} {text}");
            let mention_len = mention.chars().count() as i64;
            json!({
                "message_id": seq,
                "date": 0,
                "chat": {"id": chat, "type": "supergroup", "title": "harness"},
                "from": {"id": 1, "is_bot": false, "first_name": "harness"},
                "message_thread_id": tid,
                "is_topic_message": true,
                "text": full,
                "entities": [{"type": "mention", "offset": 0, "length": mention_len}],
            })
        }
    };
    // `date` must be a valid unix ts for teloxide; 0 is accepted by its serde.
    v["date"] = json!(1_700_000_000);
    serde_json::from_value(v).expect("valid harness Message json")
}

fn build_photo_message(
    seq: i32,
    chat: i64,
    topic: Option<i32>,
    media_group_id: &str,
    caption: Option<&str>,
    identity: &naked_tg::bot_identity::BotIdentity,
) -> Message {
    let mut v = json!({
        "message_id": seq,
        "date": 1_700_000_000,
        "chat": match topic {
            Some(_) => json!({"id": chat, "type": "supergroup", "title": "harness"}),
            None => json!({"id": chat, "type": "private", "first_name": "harness"}),
        },
        "from": {"id": 1, "is_bot": false, "first_name": "harness"},
        "media_group_id": media_group_id,
        "photo": [
            {"file_id": format!("photo-{seq}"), "file_unique_id": format!("unique-{seq}"), "width": 1, "height": 1, "file_size": 10}
        ],
    });
    if let Some(tid) = topic {
        v["message_thread_id"] = json!(tid);
        v["is_topic_message"] = json!(true);
        let mention = format!("@{}", identity.username);
        let full_caption = match caption {
            Some(c) if !c.trim().is_empty() => format!("{mention} {c}"),
            _ => mention.clone(),
        };
        v["caption"] = json!(full_caption);
        v["caption_entities"] = json!([
            {"type": "mention", "offset": 0, "length": mention.chars().count()}
        ]);
    } else if let Some(caption) = caption.filter(|c| !c.trim().is_empty()) {
        v["caption"] = json!(caption);
    }
    serde_json::from_value(v).expect("valid harness photo Message json")
}

fn build_callback_query(seq: i32, chat: i64, topic: Option<i32>, data: &str) -> CallbackQuery {
    let mut message = json!({
        "message_id": seq + 100_000,
        "date": 1_700_000_000,
        "chat": {"id": chat, "type": if topic.is_some() {"supergroup"} else {"private"}, "first_name": "harness", "title": "harness"},
        "text": "permission card",
    });
    if let Some(tid) = topic {
        message["message_thread_id"] = json!(tid);
        message["is_topic_message"] = json!(true);
    }
    let v = json!({
        "id": format!("cb-{seq}"),
        "from": {"id": 1, "is_bot": false, "first_name": "harness"},
        "chat_instance": "harness-instance",
        "data": data,
        "message": message,
    });
    serde_json::from_value(v).expect("valid harness CallbackQuery json")
}
