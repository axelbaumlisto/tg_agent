use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde_json::{Value, json};
use tokio::fs;
use tokio::sync::{Mutex, RwLock};

/// Number of seconds in one day.
const SECS_PER_DAY: i64 = 24 * 3600;

/// YOLO auto-expires after this many seconds (30 days).
const YOLO_TTL_SECS: i64 = 30 * SECS_PER_DAY;

/// Pre-30d-migration cutoff (72h). Legacy snapshot rows written before the
/// per-row `ttl_secs` marker existed are admitted on load ONLY if they are
/// still inside this original 72h window. This prevents the 72h→30d change
/// from retroactively reviving grants that already expired under the old
/// policy (silent auto-approve without fresh user action).
///
/// `pub` so cross-crate restore paths (naked-tg binary's `restore_yolo`) apply
/// the same legacy cutoff for `SessionConfig` grants that predate the per-grant
/// `yolo_ttl_secs` marker — one legacy constant, no duplicate literal.
pub const YOLO_LEGACY_TTL_SECS: i64 = 72 * 3600;

/// Number of DISTINCT explicit enables in a chat after which yolo escalates
/// from a temporary per-topic 30d grant to a permanent chat-wide grant.
/// Single source of truth for the escalation threshold — no magic number.
pub const YOLO_PERMANENT_AFTER: u32 = 2;

/// Sentinel `remaining` value reported for a permanent (never-expiring) grant.
/// Callers render this as "навсегда" instead of a huge number (see
/// [`format_yolo_window`]). Single source of truth for the sentinel.
pub const YOLO_PERMANENT_SENTINEL: i64 = i64::MAX;

/// Per-chat escalation state. Intentionally minimal: the running count of
/// distinct explicit enables, whether the chat has crossed
/// [`YOLO_PERMANENT_AFTER`] into a permanent grant, and the dedup guard.
///
/// `last_call` holds the last permission `call_id` that incremented the count.
/// It lives INSIDE this record (not a separate map) so the dedup decision and
/// the count mutation happen under the single `yolo_chat` lock — concurrent
/// duplicate callback deliveries cannot under/double-count. Ephemeral, not
/// persisted; dedup only matters within a card's lifetime.
#[derive(Clone)]
struct ChatYolo {
    count: u32,
    permanent: bool,
    last_call: Option<String>,
}

/// Tier reached by an explicit yolo enable, used to render the toast.
pub enum YoloTier {
    /// Temporary per-topic grant; `remaining_days` is the freshly granted window.
    Temporary { remaining_days: i64 },
    /// Permanent chat-wide grant (never expires until `/yolo off`).
    Permanent,
}

/// Result of an explicit yolo enable: the tier reached plus the running count
/// of distinct enables for the chat. Both enable paths (callback + command)
/// build the user-facing toast from this via [`YoloEscalation::toast`] (DRY).
pub struct YoloEscalation {
    pub tier: YoloTier,
    pub count: u32,
}

impl YoloEscalation {
    /// User-facing toast, identical across the callback and command paths.
    /// `approved` is the number of pending permission requests auto-approved.
    pub fn toast(&self, approved: usize) -> String {
        match self.tier {
            YoloTier::Temporary { remaining_days } => format!(
                "⚡ YOLO ON ({remaining_days} дней) — {approved} approved. \
                 Включи ещё раз → навсегда для этого чата."
            ),
            YoloTier::Permanent => {
                format!("⚡ YOLO ON (навсегда для чата) — {approved} approved.")
            }
        }
    }
}

/// Render a remaining-days readout for display sites: the permanent sentinel
/// becomes "навсегда", any other value becomes "{n}d". Single source of truth
/// for the sentinel→label mapping shared by the restore-log display site.
pub fn format_yolo_window(remaining_days: i64) -> String {
    if remaining_days == YOLO_PERMANENT_SENTINEL {
        "навсегда".to_string()
    } else {
        format!("{remaining_days}d")
    }
}

/// Load-time admission rule: a persisted grant is admitted iff it is still
/// inside its TTL window measured from `enabled_at`. Single source of truth
/// shared by snapshot load (`open`) and `enable_yolo_at` so the legacy-vs-new
/// distinction is enforced in exactly one place.
fn yolo_within_ttl(enabled_at: i64, ttl_secs: i64) -> bool {
    now_secs() - enabled_at < ttl_secs
}

/// Composite key: (chat_id, thread_id).
/// thread_id=0 means "no topic" (general / DM without topics).
type ChannelKey = (i64, i64);

/// A single yolo grant: when it was enabled and the TTL window it was granted
/// under. The per-grant `ttl_secs` is honored end-to-end (admission, active
/// check, remaining time, re-persist) so a grant is never silently promoted to
/// a longer window than the user authorized. New grants get [`YOLO_TTL_SECS`]
/// (30d); legacy grants loaded without a marker keep [`YOLO_LEGACY_TTL_SECS`]
/// (72h).
#[derive(Clone, Copy)]
struct YoloGrant {
    enabled_at: i64,
    ttl_secs: i64,
    /// Whether this grant came from an EXPLICIT user enable (`/yolo` or the
    /// inline ⚡ button) rather than an automated flow (research dispatch).
    /// Only explicit grants seed the per-chat escalation count on restart
    /// ([`seed_missing_chat_counts`]) — an automated grant must never make a
    /// later first `/yolo` count as the 2nd enable (wrongly permanent).
    explicit: bool,
}

fn make_key(chat_id: i64, thread_id: Option<i32>) -> ChannelKey {
    (chat_id, thread_id.unwrap_or(0) as i64)
}

pub fn format_tg_channel_id(chat_id: i64, thread_id: Option<i32>) -> String {
    format!("tg:{}:{}", chat_id, thread_id.unwrap_or(0))
}

fn is_safe_tool(tool_name: &str) -> bool {
    matches!(
        tool_name,
        "read_file" | "web_search" | "glob_search" | "grep_search" | "agent_status"
    )
}

fn parse_tg_channel_id(s: &str) -> Option<ChannelKey> {
    let parts: Vec<&str> = s.splitn(3, ':').collect();
    if parts.len() == 3 && parts[0] == "tg" {
        let chat_id = parts[1].parse::<i64>().ok()?;
        let thread_id = parts[2].parse::<i64>().ok()?;
        Some((chat_id, thread_id))
    } else {
        None
    }
}

/// Maps Telegram (chat_id, thread_id) pairs to agent session IDs.
/// Each topic gets its own session — topics act as separate dialogs.
///
/// Permission model (evaluated top-to-bottom, first match wins):
///  1. **yolo** — auto-approve everything. Either a TEMPORARY per-topic grant
///     (30d TTL, scoped to one topic) OR a PERMANENT chat-wide grant
///     (never expires until `/yolo off`; makes `is_yolo` true for EVERY topic
///     in the chat regardless of per-topic grants).
///  2. **allow_list** — per-topic set of tool names: auto-approve only listed tools
///  3. Otherwise ask the user
pub struct ChannelSessionMap {
    map: RwLock<HashMap<ChannelKey, String>>,
    session_locks: RwLock<HashMap<ChannelKey, std::sync::Arc<tokio::sync::Mutex<()>>>>,
    /// Per-topic yolo grants, each carrying its own enabled-at + TTL window.
    yolo: RwLock<HashMap<ChannelKey, YoloGrant>>,
    /// Per-chat escalation state (distinct-enable count + permanent flag +
    /// dedup guard). Keyed by chat_id only — permanent applies chat-wide across
    /// all topics. The dedup `call_id` lives inside `ChatYolo` so dedup and
    /// count mutation share this single lock (atomic against concurrent taps).
    yolo_chat: RwLock<HashMap<i64, ChatYolo>>,
    allow_list: RwLock<HashMap<ChannelKey, HashSet<String>>>,
    /// Durable backing file (JSONL snapshot). `None` means the map is
    /// purely in-memory (used by unit tests). Populated by [`Self::open`]
    /// and rewritten atomically by [`Self::flush`]; `naked-tg`'s `main`
    /// opens the snapshot at startup and re-flushes every 30s.
    snapshot_path: RwLock<Option<PathBuf>>,
    /// Serializes [`Self::flush`] end-to-end. Snapshot construction + write +
    /// rename all run under this lock so two concurrent flushes (e.g. `/yolo
    /// off` and a permanent-escalation persist) cannot interleave into the
    /// same temp file nor rename a stale buffer over a newer one — the last
    /// flush to acquire the lock is the well-defined last writer.
    flush_lock: Mutex<()>,
}

fn now_secs() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
}

impl ChannelSessionMap {
    pub fn new() -> Self {
        Self {
            map: RwLock::new(HashMap::new()),
            session_locks: RwLock::new(HashMap::new()),
            yolo: RwLock::new(HashMap::new()),
            yolo_chat: RwLock::new(HashMap::new()),
            allow_list: RwLock::new(HashMap::new()),
            snapshot_path: RwLock::new(None),
            flush_lock: Mutex::new(()),
        }
    }

    // ── Durable snapshot (A1) ──────────────────────────────────────────
    //
    // Structure: a single JSONL file under `<naked_root>/channel_map.jsonl`.
    // Each line is one of:
    //   {"type":"map",    "chat_id":..., "thread_id":..., "session_id":"..."}
    //   {"type":"yolo",   "chat_id":..., "thread_id":..., "enabled_at":..., "ttl_secs":...}
    //                    (ttl_secs absent = legacy 72h grant)
    //   {"type":"allow",  "chat_id":..., "thread_id":..., "tool":"..."}
    //
    // Simplicity > perf: the whole snapshot rewrites on every `flush`,
    // atomically via temp-file + rename. For a bot whose channel-map
    // changes O(1) times per minute this is a rounding error.

    /// Open a (possibly existing) snapshot at `<dir>/channel_map.jsonl`.
    /// Returns a map pre-loaded with any persisted rows. Creates the
    /// parent directory on demand. Missing file = empty map, not an error.
    pub async fn open(dir: &Path) -> Result<Self> {
        fs::create_dir_all(dir)
            .await
            .with_context(|| format!("channel_map: create {}", dir.display()))?;
        let path = dir.join("channel_map.jsonl");
        let this = Self::new();
        *this.snapshot_path.write().await = Some(path.clone());
        if path.exists() {
            let raw = fs::read_to_string(&path)
                .await
                .with_context(|| format!("channel_map: read {}", path.display()))?;
            let mut map = this.map.write().await;
            let mut yolo = this.yolo.write().await;
            let mut yolo_chat = this.yolo_chat.write().await;
            let mut allow = this.allow_list.write().await;
            for line in raw.lines() {
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }
                let v: Value = match serde_json::from_str(line) {
                    Ok(v) => v,
                    Err(e) => {
                        tracing::warn!("channel_map: skipping malformed snapshot line: {e}");
                        continue;
                    }
                };
                let chat_id = v.get("chat_id").and_then(Value::as_i64);
                let ty = v.get("type").and_then(Value::as_str);
                // Per-chat escalation rows carry no thread_id.
                if ty == Some("yolo_chat") {
                    if let Some(chat_id) = chat_id {
                        let count = v.get("count").and_then(Value::as_u64).unwrap_or(0) as u32;
                        let permanent =
                            v.get("permanent").and_then(Value::as_bool).unwrap_or(false);
                        yolo_chat.insert(
                            chat_id,
                            ChatYolo {
                                count,
                                permanent,
                                last_call: None,
                            },
                        );
                    }
                    continue;
                }
                let thread_id = v.get("thread_id").and_then(Value::as_i64);
                let (Some(chat_id), Some(thread_id), Some(ty)) = (chat_id, thread_id, ty) else {
                    continue;
                };
                let key: ChannelKey = (chat_id, thread_id);
                match ty {
                    "map" => {
                        if let Some(session_id) = v.get("session_id").and_then(Value::as_str) {
                            map.insert(key, session_id.to_string());
                        }
                    }
                    "yolo" => {
                        if let Some(enabled_at) = v.get("enabled_at").and_then(Value::as_i64) {
                            // Per-row TTL marker governs load admission: rows
                            // written after the 30d migration carry `ttl_secs`
                            // (=30d); legacy rows without it keep the original
                            // 72h cutoff so already-expired grants aren't revived.
                            let ttl = v
                                .get("ttl_secs")
                                .and_then(Value::as_i64)
                                .unwrap_or(YOLO_LEGACY_TTL_SECS);
                            // Missing `explicit` marker = pre-upgrade legacy
                            // row. Before this feature BOTH user `/yolo` AND
                            // automated research used `enable_yolo`, so such a
                            // row is AMBIGUOUS (could be automated). Fail safe:
                            // default explicit=FALSE so an ambiguous legacy
                            // grant is NOT counted toward permanent escalation
                            // ([`seed_missing_chat_counts`]). Defaulting to true
                            // could seed count=1 and let the user's first
                            // post-upgrade `/yolo` jump the chat to a PERMANENT
                            // chat-wide auto-approve-all off ambiguous data —
                            // and auto-approve is security-sensitive. Rows
                            // written AFTER the upgrade always carry an accurate
                            // marker, so this default only affects legacy data.
                            let explicit =
                                v.get("explicit").and_then(Value::as_bool).unwrap_or(false);
                            if yolo_within_ttl(enabled_at, ttl) {
                                yolo.insert(
                                    key,
                                    YoloGrant {
                                        enabled_at,
                                        ttl_secs: ttl,
                                        explicit,
                                    },
                                );
                            }
                        }
                    }
                    "allow" => {
                        if let Some(tool) = v.get("tool").and_then(Value::as_str) {
                            allow.entry(key).or_default().insert(tool.to_string());
                        }
                    }
                    _ => {}
                }
            }
        }
        Ok(this)
    }

    /// Rewrite the snapshot with the current in-memory state atomically
    /// (temp-file + rename). No-op if the map wasn't opened with a
    /// `snapshot_path`.
    pub async fn flush(&self) -> Result<()> {
        // Hold the flush lock across the ENTIRE snapshot build + write + rename
        // so concurrent flushes serialize: no interleaved temp writes and no
        // stale-buffer rename landing after a newer one. The snapshot is built
        // *inside* the lock, so each flush persists the latest in-memory state.
        let _guard = self.flush_lock.lock().await;
        let path = match self.snapshot_path.read().await.clone() {
            Some(p) => p,
            None => return Ok(()), // in-memory map, nothing to do
        };
        // ── Point-in-time consistent cut (B87) ────────────────────────
        // Acquire ALL four state read-guards BEFORE serializing anything.
        // Lock order (canonical, matches `open` and `seed_missing_chat_counts`):
        //   map → yolo → yolo_chat → allow_list.
        // Previously each map was read and DROPPED in turn; a writer
        // interleaving between the `yolo` and `yolo_chat` reads (e.g. a
        // `/yolo` escalation mutating both, or `/yolo off` clearing both)
        // could tear the snapshot into a combination that never existed in
        // memory — e.g. a per-topic grant row paired with a permanent
        // escalation row from a later state — and that impossible state
        // would be revived on restart. Holding all guards until the buffer
        // is fully built makes the snapshot an atomic cut; guards are
        // released before file I/O so writers are not blocked on disk.
        let map = self.map.read().await;
        let yolo = self.yolo.read().await;
        let yolo_chat = self.yolo_chat.read().await;
        let allow = self.allow_list.read().await;
        let mut buf = String::new();
        for ((chat_id, thread_id), session_id) in map.iter() {
            let line = json!({
                "type": "map",
                "chat_id": chat_id,
                "thread_id": thread_id,
                "session_id": session_id,
            });
            buf.push_str(&line.to_string());
            buf.push('\n');
        }
        for ((chat_id, thread_id), grant) in yolo.iter() {
            let line = json!({
                "type": "yolo",
                "chat_id": chat_id,
                "thread_id": thread_id,
                "enabled_at": grant.enabled_at,
                "ttl_secs": grant.ttl_secs,
                "explicit": grant.explicit,
            });
            buf.push_str(&line.to_string());
            buf.push('\n');
        }
        for (chat_id, state) in yolo_chat.iter() {
            let line = json!({
                "type": "yolo_chat",
                "chat_id": chat_id,
                "count": state.count,
                "permanent": state.permanent,
            });
            buf.push_str(&line.to_string());
            buf.push('\n');
        }
        for ((chat_id, thread_id), tools) in allow.iter() {
            for tool in tools {
                let line = json!({
                    "type": "allow",
                    "chat_id": chat_id,
                    "thread_id": thread_id,
                    "tool": tool,
                });
                buf.push_str(&line.to_string());
                buf.push('\n');
            }
        }
        drop(allow);
        drop(yolo_chat);
        drop(yolo);
        drop(map);
        // Unique per-process temp path (defense-in-depth against an external
        // concurrent process sharing the dir; in-process flushes are already
        // serialized by `flush_lock`). Clean up the temp on write failure so a
        // partial file never lingers.
        let tmp = path.with_extension(format!("jsonl.tmp.{}", std::process::id()));
        if let Err(e) = fs::write(&tmp, buf).await {
            let _ = fs::remove_file(&tmp).await;
            return Err(e).with_context(|| format!("channel_map: write {}", tmp.display()));
        }
        fs::rename(&tmp, &path).await.with_context(|| {
            format!(
                "channel_map: rename {} -> {}",
                tmp.display(),
                path.display()
            )
        })?;
        Ok(())
    }

    pub async fn get(&self, chat_id: i64, thread_id: Option<i32>) -> Option<String> {
        self.map
            .read()
            .await
            .get(&make_key(chat_id, thread_id))
            .cloned()
    }

    pub async fn get_or_insert_with<F, Fut>(
        &self,
        chat_id: i64,
        thread_id: Option<i32>,
        create: F,
    ) -> String
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = String>,
    {
        let key = make_key(chat_id, thread_id);
        if let Some(existing) = self.map.read().await.get(&key).cloned() {
            return existing;
        }

        let lock = self.session_lock_for(key).await;
        let _guard = lock.lock().await;
        if let Some(existing) = self.map.read().await.get(&key).cloned() {
            return existing;
        }

        let session_id = create().await;
        self.map.write().await.insert(key, session_id.clone());
        session_id
    }

    async fn session_lock_for(&self, key: ChannelKey) -> std::sync::Arc<tokio::sync::Mutex<()>> {
        if let Some(existing) = self.session_locks.read().await.get(&key).cloned() {
            return existing;
        }
        let mut locks = self.session_locks.write().await;
        locks
            .entry(key)
            .or_insert_with(|| std::sync::Arc::new(tokio::sync::Mutex::new(())))
            .clone()
    }

    /// All (chat_id, thread_id, session_id) entries.
    pub async fn all_entries(&self) -> Vec<(i64, i64, String)> {
        self.map
            .read()
            .await
            .iter()
            .map(|((cid, tid), sid)| (*cid, *tid, sid.clone()))
            .collect()
    }

    pub async fn set(&self, chat_id: i64, thread_id: Option<i32>, session_id: String) {
        self.map
            .write()
            .await
            .insert(make_key(chat_id, thread_id), session_id);
    }

    /// All DISTINCT session IDs mapped to any topic of `chat_id`.
    ///
    /// Used by the `/yolo off` revocation path to clear the persisted
    /// `SessionConfig` yolo grant for EVERY topic of the chat — not just the
    /// current one. Otherwise a sibling topic's `yolo_enabled_at` would revive
    /// a temporary grant on restart (revocation bypass).
    pub async fn sessions_for_chat(&self, chat_id: i64) -> Vec<String> {
        let map = self.map.read().await;
        let mut seen = HashSet::new();
        let mut out = Vec::new();
        for ((cid, _tid), sid) in map.iter() {
            if *cid == chat_id && seen.insert(sid.clone()) {
                out.push(sid.clone());
            }
        }
        out
    }

    #[allow(dead_code)]
    pub async fn remove(&self, chat_id: i64, thread_id: Option<i32>) {
        self.map.write().await.remove(&make_key(chat_id, thread_id));
    }

    /// Rebuild mappings from persisted `channel_id` strings (format: "tg:{chat_id}:{thread_id}").
    pub async fn restore_from(&self, mappings: &[(String, String)]) -> usize {
        let mut map = self.map.write().await;
        let mut count = 0;
        for (channel_id, session_id) in mappings {
            if let Some(key) = parse_tg_channel_id(channel_id) {
                map.insert(key, session_id.clone());
                count += 1;
            }
        }
        count
    }

    // ── yolo (per-topic auto-approve-all, 30d TTL) ──────────────────────

    /// Whether the chat has a permanent (never-expiring) chat-wide grant.
    async fn is_permanent(&self, chat_id: i64) -> bool {
        self.yolo_chat
            .read()
            .await
            .get(&chat_id)
            .map(|c| c.permanent)
            .unwrap_or(false)
    }

    /// Check if yolo is active and not expired. A permanent chat-wide grant
    /// makes this true for EVERY topic in the chat; otherwise falls back to the
    /// per-topic temporary grant. Auto-removes expired temporary entries.
    pub async fn is_yolo(&self, chat_id: i64, thread_id: Option<i32>) -> bool {
        if self.is_permanent(chat_id).await {
            return true;
        }
        let key = make_key(chat_id, thread_id);
        let yolo = self.yolo.read().await;
        if let Some(&grant) = yolo.get(&key) {
            if yolo_within_ttl(grant.enabled_at, grant.ttl_secs) {
                return true;
            }
            drop(yolo);
            self.yolo.write().await.remove(&key);
        }
        false
    }

    /// Explicit yolo enable. Refreshes the per-topic temporary 30d grant AND
    /// advances the per-chat escalation: a distinct action increments `count`
    /// (saturating) and, at [`YOLO_PERMANENT_AFTER`], flips the chat permanent.
    ///
    /// `dedup` is the permission `call_id` for the callback path (so tapping the
    /// same card twice counts once) or `None` for the command path (each
    /// `/yolo` is one distinct action). Returns the tier + running count so
    /// callers render the toast via [`YoloEscalation::toast`].
    pub async fn enable_yolo(
        &self,
        chat_id: i64,
        thread_id: Option<i32>,
        dedup: Option<&str>,
    ) -> YoloEscalation {
        // Per-topic temporary grant, tagged EXPLICIT so it seeds the
        // escalation count on restart (Ring 1 behavior, unchanged otherwise).
        self.set_topic_grant(chat_id, thread_id, true).await;

        // Dedup decision AND count mutation happen under ONE lock so concurrent
        // duplicate callback deliveries cannot race between the two: tapping the
        // same permission card twice (same call_id) counts once; distinct
        // actions (other cards / commands) count independently.
        let (count, permanent) = {
            let mut chat = self.yolo_chat.write().await;
            let entry = chat.entry(chat_id).or_insert(ChatYolo {
                count: 0,
                permanent: false,
                last_call: None,
            });
            let should_count = match dedup {
                Some(call_id) => {
                    if entry.last_call.as_deref() == Some(call_id) {
                        false
                    } else {
                        entry.last_call = Some(call_id.to_string());
                        true
                    }
                }
                None => {
                    entry.last_call = None;
                    true
                }
            };
            if should_count {
                entry.count = entry.count.saturating_add(1);
                if entry.count >= YOLO_PERMANENT_AFTER {
                    entry.permanent = true;
                }
            }
            (entry.count, entry.permanent)
        };

        let tier = if permanent {
            YoloTier::Permanent
        } else {
            YoloTier::Temporary {
                remaining_days: self.yolo_remaining_days(chat_id, thread_id).await,
            }
        };
        YoloEscalation { tier, count }
    }

    /// Set/refresh the per-topic temporary 30d grant WITHOUT touching the
    /// per-chat escalation. Used by automated flows (e.g. research dispatch)
    /// that need auto-approve for a run but must not count as an explicit user
    /// escalation toward permanent. Explicit enables go through [`enable_yolo`].
    pub async fn grant_temporary_yolo(&self, chat_id: i64, thread_id: Option<i32>) {
        self.set_topic_grant(chat_id, thread_id, false).await;
    }

    /// Insert/refresh a per-topic 30d grant, tagging it EXPLICIT (user enable)
    /// or automated. Single source of truth for building a fresh [`YoloGrant`]
    /// so the two callers ([`enable_yolo`], [`grant_temporary_yolo`]) can't
    /// drift on the TTL or the `explicit` marker (DRY).
    async fn set_topic_grant(&self, chat_id: i64, thread_id: Option<i32>, explicit: bool) {
        self.yolo.write().await.insert(
            make_key(chat_id, thread_id),
            YoloGrant {
                enabled_at: now_secs(),
                ttl_secs: YOLO_TTL_SECS,
                explicit,
            },
        );
    }

    /// Seed escalation counts for chats that have a surviving EXPLICIT yolo
    /// grant but no `yolo_chat` row. Such a chat either predates escalation
    /// tracking (pre-Ring2 snapshot) OR was restored purely from a
    /// `SessionConfig` grant — which revives the in-memory grant via
    /// [`enable_yolo_at`] but creates NO `yolo_chat` row. Seeding count=1
    /// ensures the user's NEXT explicit enable is correctly counted as the 2nd
    /// → permanent instead of restarting at a fresh temporary grant.
    ///
    /// Automated grants (research dispatch, `explicit == false`) are skipped:
    /// they never represent a user enable, so they must not seed a count that
    /// would make the user's first `/yolo` jump to permanent.
    ///
    /// Single seeding point (DRY): call ONCE after ALL restore sources
    /// (snapshot `open` + `SessionConfig` restore) have populated the in-memory
    /// `yolo` map. Idempotent — chats that already carry a row are untouched,
    /// and chats with neither a grant nor a row get no entry.
    pub async fn seed_missing_chat_counts(&self) {
        let yolo = self.yolo.read().await;
        let mut yolo_chat = self.yolo_chat.write().await;
        for (&(cid, _tid), grant) in yolo.iter() {
            // Only EXPLICIT grants represent a user enable worth counting. A
            // chat whose only grants are automated (research) must NOT be
            // seeded, or the user's first `/yolo` would wrongly count as the
            // 2nd enable and jump straight to permanent.
            if grant.explicit {
                yolo_chat.entry(cid).or_insert(ChatYolo {
                    count: 1,
                    permanent: false,
                    last_call: None,
                });
            }
        }
    }

    /// Clear ALL yolo state for a chat (called by `/yolo off`): remove the
    /// per-chat escalation row (count, permanent flag AND the `last_call` dedup
    /// guard all live in `ChatYolo`, so dropping the row clears them together)
    /// and remove every per-topic temporary grant belonging to the chat.
    pub async fn clear_yolo_chat(&self, chat_id: i64) {
        self.yolo_chat.write().await.remove(&chat_id);
        self.yolo
            .write()
            .await
            .retain(|(cid, _tid), _| *cid != chat_id);
    }

    /// Enable yolo with a specific timestamp and TTL window (for restoring from
    /// disk). Admits iff the grant is still inside `ttl_secs` from `enabled_at`
    /// — callers pass the per-grant TTL (30d marker) or the legacy 72h cutoff
    /// for pre-marker grants, so already-expired grants are never revived.
    ///
    /// Contract: the restored grant is tagged `explicit: true`. On this path
    /// the persisted `yolo_enabled_at` only ever originates from an explicit
    /// user enable (`/yolo` or the inline ⚡ button) — automated research
    /// grants use [`Self::grant_temporary_yolo`] (`explicit: false`, never
    /// persisted into a tg-channel `SessionConfig`) and must NEVER escalate
    /// toward permanent. The `explicit` marker is what lets
    /// [`Self::seed_missing_chat_counts`] seed the per-chat escalation count from a
    /// restored grant, so the user's next explicit enable correctly counts as
    /// the 2nd → permanent.
    pub async fn enable_yolo_at(
        &self,
        chat_id: i64,
        thread_id: Option<i32>,
        enabled_at: i64,
        ttl_secs: i64,
    ) {
        if yolo_within_ttl(enabled_at, ttl_secs) {
            self.yolo.write().await.insert(
                make_key(chat_id, thread_id),
                YoloGrant {
                    enabled_at,
                    ttl_secs,
                    // Restore-path grants come from a persisted user enable
                    // (SessionConfig `yolo_enabled_at`), so they are explicit.
                    explicit: true,
                },
            );
        }
    }

    /// Seconds remaining until yolo expires, or 0 if not active. A permanent
    /// chat-wide grant reports [`YOLO_PERMANENT_SENTINEL`] (never expires).
    pub async fn yolo_remaining_secs(&self, chat_id: i64, thread_id: Option<i32>) -> i64 {
        if self.is_permanent(chat_id).await {
            return YOLO_PERMANENT_SENTINEL;
        }
        let key = make_key(chat_id, thread_id);
        if let Some(&grant) = self.yolo.read().await.get(&key) {
            let elapsed = now_secs() - grant.enabled_at;
            if elapsed < grant.ttl_secs {
                return grant.ttl_secs - elapsed;
            }
        }
        0
    }

    /// Whole days remaining until yolo expires: 0 when yolo is
    /// inactive/expired, [`YOLO_PERMANENT_SENTINEL`] when the chat holds a
    /// permanent (never-expiring) chat-wide grant — callers render the
    /// sentinel as "навсегда" via [`format_yolo_window`], never as a number.
    ///
    /// An active temporary window rounds UP so any remaining time reads as
    /// >=1d (the display never shows "0d" while auto-approve is still on).
    pub async fn yolo_remaining_days(&self, chat_id: i64, thread_id: Option<i32>) -> i64 {
        let secs = self.yolo_remaining_secs(chat_id, thread_id).await;
        if secs == YOLO_PERMANENT_SENTINEL {
            YOLO_PERMANENT_SENTINEL
        } else if secs <= 0 {
            0
        } else {
            (secs + SECS_PER_DAY - 1) / SECS_PER_DAY
        }
    }

    /// Disable yolo for a topic (called on /new). Only clears the per-topic
    /// temporary grant — the per-chat escalation count/permanent state is left
    /// intact so a permanent chat survives `/new`.
    pub async fn disable_yolo(&self, chat_id: i64, thread_id: Option<i32>) {
        self.yolo
            .write()
            .await
            .remove(&make_key(chat_id, thread_id));
    }

    /// Test-only view of a chat's escalation state: `(count, permanent)`,
    /// or `(0, false)` when the chat has no entry.
    #[cfg(test)]
    pub(crate) async fn chat_yolo_state(&self, chat_id: i64) -> (u32, bool) {
        self.yolo_chat
            .read()
            .await
            .get(&chat_id)
            .map(|c| (c.count, c.permanent))
            .unwrap_or((0, false))
    }

    // ── allow-list (per-topic tool whitelist) ────────────────────────────

    /// Returns true if `tool_name` should be auto-approved for this topic.
    /// Tools that are always safe to auto-approve (read-only, no side effects).
    /// Check if a tool should be auto-approved for this chat.
    ///
    /// SOLID/KISS: Permission level is the single source of truth.
    /// ReadOnly tools are auto-approved by the core policy (never reach here).
    /// WorkspaceWrite/Dangerous tools reach here and need yolo or allow_list.
    pub async fn should_auto_approve(
        &self,
        chat_id: i64,
        thread_id: Option<i32>,
        _tool_name: &str,
    ) -> bool {
        if is_safe_tool(_tool_name) {
            return true;
        }
        if self.is_yolo(chat_id, thread_id).await {
            return true;
        }
        let key = make_key(chat_id, thread_id);
        if let Some(tools) = self.allow_list.read().await.get(&key) {
            return tools.contains(_tool_name);
        }
        false
    }

    pub async fn allow_add(&self, chat_id: i64, thread_id: Option<i32>, tool_name: &str) -> bool {
        let key = make_key(chat_id, thread_id);
        self.allow_list
            .write()
            .await
            .entry(key)
            .or_default()
            .insert(tool_name.to_string())
    }

    pub async fn allow_remove(
        &self,
        chat_id: i64,
        thread_id: Option<i32>,
        tool_name: &str,
    ) -> bool {
        let key = make_key(chat_id, thread_id);
        let mut map = self.allow_list.write().await;
        if let Some(tools) = map.get_mut(&key) {
            let removed = tools.remove(tool_name);
            if tools.is_empty() {
                map.remove(&key);
            }
            removed
        } else {
            false
        }
    }

    pub async fn allow_get(&self, chat_id: i64, thread_id: Option<i32>) -> Vec<String> {
        let key = make_key(chat_id, thread_id);
        self.allow_list
            .read()
            .await
            .get(&key)
            .map(|s| {
                let mut v: Vec<String> = s.iter().cloned().collect();
                v.sort();
                v
            })
            .unwrap_or_default()
    }
}

impl Default for ChannelSessionMap {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn get_set_remove() {
        let map = ChannelSessionMap::new();
        assert!(map.get(123, None).await.is_none());

        map.set(123, None, "session-1".into()).await;
        assert_eq!(map.get(123, None).await.as_deref(), Some("session-1"));

        map.remove(123, None).await;
        assert!(map.get(123, None).await.is_none());
    }

    #[tokio::test]
    async fn separate_chats() {
        let map = ChannelSessionMap::new();
        map.set(1, None, "s1".into()).await;
        map.set(2, None, "s2".into()).await;

        assert_eq!(map.get(1, None).await.as_deref(), Some("s1"));
        assert_eq!(map.get(2, None).await.as_deref(), Some("s2"));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn get_or_insert_with_is_atomic_for_one_key() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicUsize, Ordering};
        use tokio::sync::Barrier;

        let map = Arc::new(ChannelSessionMap::new());
        let creates = Arc::new(AtomicUsize::new(0));
        let start = Arc::new(Barrier::new(16));
        let mut tasks = Vec::new();

        for _ in 0..16 {
            let map = map.clone();
            let creates = creates.clone();
            let start = start.clone();
            tasks.push(tokio::spawn(async move {
                start.wait().await;
                map.get_or_insert_with(9, Some(3), || async {
                    let n = creates.fetch_add(1, Ordering::SeqCst);
                    tokio::time::sleep(std::time::Duration::from_millis(5)).await;
                    format!("session-{n}")
                })
                .await
            }));
        }

        let mut ids = Vec::new();
        for task in tasks {
            ids.push(task.await.expect("task panicked"));
        }
        assert!(ids.iter().all(|id| id == "session-0"));
        assert_eq!(creates.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn topics_are_separate_sessions() {
        let map = ChannelSessionMap::new();
        let chat = 100;
        map.set(chat, None, "general".into()).await;
        map.set(chat, Some(5), "topic-5".into()).await;
        map.set(chat, Some(9), "topic-9".into()).await;

        assert_eq!(map.get(chat, None).await.as_deref(), Some("general"));
        assert_eq!(map.get(chat, Some(5)).await.as_deref(), Some("topic-5"));
        assert_eq!(map.get(chat, Some(9)).await.as_deref(), Some("topic-9"));
    }

    #[tokio::test]
    async fn yolo_per_topic() {
        let map = ChannelSessionMap::new();
        assert!(!map.is_yolo(1, None).await);
        assert!(!map.is_yolo(1, Some(5)).await);

        // Single explicit enable: temporary, topic-scoped.
        map.enable_yolo(1, Some(5), Some("card-1")).await;
        assert!(!map.is_yolo(1, None).await);
        assert!(map.is_yolo(1, Some(5)).await);

        // Re-tapping the SAME card is idempotent (dedup): still count=1,
        // temporary — not escalated to permanent.
        map.enable_yolo(1, Some(5), Some("card-1")).await;
        assert!(map.is_yolo(1, Some(5)).await);
        assert_eq!(map.chat_yolo_state(1).await, (1, false));

        // disable clears the temporary topic grant.
        map.disable_yolo(1, Some(5)).await;
        assert!(!map.is_yolo(1, Some(5)).await);
    }

    #[tokio::test]
    async fn allow_list_basics() {
        let map = ChannelSessionMap::new();
        // Without yolo or allow_list, only read-only safe tools auto-approve.
        assert!(map.should_auto_approve(1, Some(2), "read_file").await);
        assert!(!map.should_auto_approve(1, Some(2), "bash").await);
        assert!(!map.should_auto_approve(1, Some(2), "write_file").await);

        map.allow_add(1, Some(2), "bash").await;

        assert!(map.should_auto_approve(1, Some(2), "bash").await);
        assert!(!map.should_auto_approve(1, Some(2), "write_file").await);
        assert!(!map.should_auto_approve(1, Some(3), "bash").await);

        let list = map.allow_get(1, Some(2)).await;
        assert_eq!(list, vec!["bash"]);

        map.allow_remove(1, Some(2), "bash").await;
        assert!(!map.should_auto_approve(1, Some(2), "bash").await);
    }

    #[tokio::test]
    async fn yolo_overrides_allow_list() {
        let map = ChannelSessionMap::new();
        assert!(!map.should_auto_approve(1, Some(5), "anything").await);

        map.enable_yolo(1, Some(5), None).await;
        assert!(map.should_auto_approve(1, Some(5), "anything").await);
        assert!(map.should_auto_approve(1, Some(5), "bash").await);
    }

    #[tokio::test]
    async fn should_auto_approve_checks_both() {
        let map = ChannelSessionMap::new();
        assert!(map.should_auto_approve(1, None, "web_search").await);
        assert!(!map.should_auto_approve(1, None, "bash").await);

        map.allow_add(1, None, "bash").await;
        assert!(map.should_auto_approve(1, None, "bash").await);
        assert!(!map.should_auto_approve(1, None, "write_file").await);

        map.enable_yolo(1, None, None).await;
        assert!(map.should_auto_approve(1, None, "write_file").await);
    }

    #[tokio::test]
    async fn yolo_expires_after_ttl() {
        let map = ChannelSessionMap::new();
        // Set yolo with a timestamp 31 days in the past (expired under 30d window)
        let expired_ts = now_secs() - (31 * SECS_PER_DAY);
        map.enable_yolo_at(1, Some(5), expired_ts, YOLO_TTL_SECS)
            .await;
        assert!(!map.is_yolo(1, Some(5)).await);

        // Set yolo with a timestamp 1 hour in the past (still valid)
        let recent_ts = now_secs() - 3600;
        map.enable_yolo_at(1, Some(5), recent_ts, YOLO_TTL_SECS)
            .await;
        assert!(map.is_yolo(1, Some(5)).await);
        assert!(map.yolo_remaining_secs(1, Some(5)).await > 0);
    }

    #[tokio::test]
    async fn yolo_active_within_30d_window_and_expires_after() {
        let map = ChannelSessionMap::new();
        // Enabled 29 days ago — still inside the 30-day window.
        map.enable_yolo_at(1, Some(5), now_secs() - (29 * SECS_PER_DAY), YOLO_TTL_SECS)
            .await;
        assert!(map.is_yolo(1, Some(5)).await);

        // Enabled 31 days ago — just past the 30-day window.
        map.enable_yolo_at(2, Some(7), now_secs() - (31 * SECS_PER_DAY), YOLO_TTL_SECS)
            .await;
        assert!(!map.is_yolo(2, Some(7)).await);
    }

    #[tokio::test]
    async fn yolo_remaining_is_about_30d_after_enable() {
        let map = ChannelSessionMap::new();
        map.enable_yolo(1, None, None).await;
        let remaining = map.yolo_remaining_secs(1, None).await;
        // Right after enable, ~30 days remain (allow a small execution delta).
        assert!(remaining > 30 * SECS_PER_DAY - 5);
        assert!(remaining <= 30 * SECS_PER_DAY);
        // Days readout used by the display sites: ~30d right after enable
        // (30 with no elapsed tick, 29 if a whole second has ticked).
        let days = map.yolo_remaining_days(1, None).await;
        assert!(days == 29 || days == 30, "unexpected days readout: {days}");
    }

    #[tokio::test]
    async fn yolo_remaining_days_rounds_up_in_final_day_but_zero_when_inactive() {
        let map = ChannelSessionMap::new();
        // Enabled 29 days + 12h ago: ~12h remain inside the 30-day window.
        // Flooring would read 0d while auto-approve is still on; must read >=1d.
        let almost_expired = now_secs() - (29 * SECS_PER_DAY + 12 * 3600);
        map.enable_yolo_at(1, Some(5), almost_expired, YOLO_TTL_SECS)
            .await;
        assert!(map.is_yolo(1, Some(5)).await);
        assert_eq!(map.yolo_remaining_days(1, Some(5)).await, 1);

        // Expired / inactive topics report 0 days.
        map.enable_yolo_at(2, Some(7), now_secs() - (31 * SECS_PER_DAY), YOLO_TTL_SECS)
            .await;
        assert!(!map.is_yolo(2, Some(7)).await);
        assert_eq!(map.yolo_remaining_days(2, Some(7)).await, 0);
        assert_eq!(map.yolo_remaining_days(99, None).await, 0);
    }

    #[tokio::test]
    async fn legacy_yolo_snapshot_row_loads_under_30d_window() {
        // A legacy {"type":"yolo",...,"enabled_at":...} row (NO ttl_secs marker)
        // must keep the original 72h cutoff on load. A 20-day-old legacy row was
        // already expired under the old 72h policy and must NOT be revived by the
        // 72h→30d change; only rows still inside the original 72h window survive.
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("channel_map.jsonl");
        let recent_at = now_secs() - 3600; // 1h ago: inside 72h
        let expired_at = now_secs() - (20 * SECS_PER_DAY); // expired under 72h
        let contents = format!(
            "{}\n{}\n",
            json!({"type": "yolo", "chat_id": 10, "thread_id": 0, "enabled_at": recent_at}),
            json!({"type": "yolo", "chat_id": 11, "thread_id": 0, "enabled_at": expired_at}),
        );
        std::fs::write(&path, contents).expect("write snapshot");

        let map = ChannelSessionMap::open(dir.path()).await.expect("open");
        assert!(
            map.is_yolo(10, None).await,
            "legacy row inside 72h should be active"
        );
        assert!(
            !map.is_yolo(11, None).await,
            "legacy 20-day-old row was expired under 72h and must not be revived"
        );
    }

    #[tokio::test]
    async fn legacy_yolo_row_without_ttl_marker_uses_72h_cutoff() {
        // Legacy rows lack `ttl_secs` and must be judged against the 72h cutoff.
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("channel_map.jsonl");
        let five_days_ago = now_secs() - (5 * SECS_PER_DAY);
        let two_hours_ago = now_secs() - (2 * 3600);
        let contents = format!(
            "{}\n{}\n",
            json!({"type": "yolo", "chat_id": 20, "thread_id": 0, "enabled_at": five_days_ago}),
            json!({"type": "yolo", "chat_id": 21, "thread_id": 0, "enabled_at": two_hours_ago}),
        );
        std::fs::write(&path, contents).expect("write snapshot");

        let map = ChannelSessionMap::open(dir.path()).await.expect("open");
        assert!(
            !map.is_yolo(20, None).await,
            "5-day-old legacy row is past 72h and must be dropped"
        );
        assert!(
            map.is_yolo(21, None).await,
            "2-hour-old legacy row is inside 72h and must be active"
        );
    }

    #[tokio::test]
    async fn new_yolo_row_with_ttl_marker_uses_30d() {
        // A freshly enabled grant round-trips through flush/open and stays active
        // for the full 30-day window (the ttl_secs marker overrides the legacy
        // 72h cutoff).
        let dir = tempfile::tempdir().expect("tempdir");
        let map = ChannelSessionMap::open(dir.path()).await.expect("open");
        map.enable_yolo(30, None, None).await;
        map.flush().await.expect("flush");
        let reopened = ChannelSessionMap::open(dir.path()).await.expect("reopen");
        assert!(
            reopened.is_yolo(30, None).await,
            "freshly written row should reload as active"
        );

        // Directly written marked rows: 20 days old is active, 31 days old dropped.
        let path = dir.path().join("channel_map.jsonl");
        let twenty_days = now_secs() - (20 * SECS_PER_DAY);
        let thirtyone_days = now_secs() - (31 * SECS_PER_DAY);
        let contents = format!(
            "{}\n{}\n",
            json!({"type": "yolo", "chat_id": 31, "thread_id": 0, "enabled_at": twenty_days, "ttl_secs": YOLO_TTL_SECS}),
            json!({"type": "yolo", "chat_id": 32, "thread_id": 0, "enabled_at": thirtyone_days, "ttl_secs": YOLO_TTL_SECS}),
        );
        std::fs::write(&path, contents).expect("write snapshot");
        let map2 = ChannelSessionMap::open(dir.path()).await.expect("open2");
        assert!(
            map2.is_yolo(31, None).await,
            "20-day-old marked row should be active under 30d"
        );
        assert!(
            !map2.is_yolo(32, None).await,
            "31-day-old marked row should be dropped under 30d"
        );
    }

    #[tokio::test]
    async fn enable_yolo_at_honors_explicit_ttl_legacy_vs_new() {
        // Mirrors the SessionConfig restore path: legacy grants (no persisted
        // TTL) are admitted with the 72h cutoff; new grants carry the 30d TTL.
        let map = ChannelSessionMap::new();

        // Legacy grant 5 days old, judged under 72h -> already expired, dropped.
        map.enable_yolo_at(
            1,
            None,
            now_secs() - (5 * SECS_PER_DAY),
            YOLO_LEGACY_TTL_SECS,
        )
        .await;
        assert!(
            !map.is_yolo(1, None).await,
            "5-day-old legacy grant is past 72h and must not be revived"
        );

        // Legacy grant 2h old, inside 72h -> restored.
        map.enable_yolo_at(2, None, now_secs() - (2 * 3600), YOLO_LEGACY_TTL_SECS)
            .await;
        assert!(
            map.is_yolo(2, None).await,
            "2-hour-old legacy grant is inside 72h and must be restored"
        );

        // New grant 20 days old with the 30d TTL -> restored.
        map.enable_yolo_at(3, None, now_secs() - (20 * SECS_PER_DAY), YOLO_TTL_SECS)
            .await;
        assert!(
            map.is_yolo(3, None).await,
            "20-day-old grant under 30d TTL must be restored"
        );

        // New grant 31 days old with the 30d TTL -> dropped.
        map.enable_yolo_at(4, None, now_secs() - (31 * SECS_PER_DAY), YOLO_TTL_SECS)
            .await;
        assert!(
            !map.is_yolo(4, None).await,
            "31-day-old grant is past the 30d TTL and must be dropped"
        );
    }

    #[tokio::test]
    async fn legacy_grant_keeps_72h_ttl_end_to_end() {
        // Core regression: a legacy grant (no ttl_secs marker) enabled 71h ago
        // is admitted under the 72h cutoff, but its per-grant TTL must stay 72h
        // end-to-end. It must NOT be silently promoted to the 30d window nor
        // re-persisted as a 30d grant. (Mutation target: flush/is_yolo using the
        // global YOLO_TTL_SECS instead of grant.ttl_secs would fail this test.)
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("channel_map.jsonl");
        let enabled_71h_ago = now_secs() - (71 * 3600);
        std::fs::write(
            &path,
            format!(
                "{}\n",
                json!({"type": "yolo", "chat_id": 40, "thread_id": 0, "enabled_at": enabled_71h_ago}),
            ),
        )
        .expect("write snapshot");

        let map = ChannelSessionMap::open(dir.path()).await.expect("open");
        assert!(map.is_yolo(40, None).await, "71h-old legacy row within 72h");
        let remaining = map.yolo_remaining_secs(40, None).await;
        // ~1h remains (72h - 71h), NOT ~29 days. Allow a small execution delta.
        assert!(
            remaining > 0 && remaining <= 3600,
            "legacy grant must report ~1h remaining, not the 30d window: {remaining}"
        );

        // Re-persist and reload: the row must still carry ttl_secs=72h, so a
        // reload 73h after the original enable drops it (never promoted to 30d).
        map.flush().await.expect("flush");
        let raw = std::fs::read_to_string(&path).expect("read snapshot");
        let yolo_line = raw
            .lines()
            .find(|l| l.contains("\"type\":\"yolo\""))
            .expect("yolo row present after flush");
        let v: Value = serde_json::from_str(yolo_line).expect("parse yolo row");
        assert_eq!(
            v.get("ttl_secs").and_then(Value::as_i64),
            Some(YOLO_LEGACY_TTL_SECS),
            "flushed legacy grant must keep ttl_secs=72h, not be promoted to 30d"
        );

        // Rewrite the same grant as if 73h have now elapsed; on reload it is
        // past its own 72h TTL and must be dropped.
        let enabled_73h_ago = now_secs() - (73 * 3600);
        std::fs::write(
            &path,
            format!(
                "{}\n",
                json!({"type": "yolo", "chat_id": 40, "thread_id": 0, "enabled_at": enabled_73h_ago, "ttl_secs": YOLO_LEGACY_TTL_SECS}),
            ),
        )
        .expect("rewrite snapshot");
        let reopened = ChannelSessionMap::open(dir.path()).await.expect("reopen");
        assert!(
            !reopened.is_yolo(40, None).await,
            "legacy grant past its own 72h TTL must expire, never extended to 30d"
        );
    }

    #[tokio::test]
    async fn new_grant_has_30d_ttl() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("channel_map.jsonl");
        let map = ChannelSessionMap::open(dir.path()).await.expect("open");

        map.enable_yolo(50, None, None).await;
        let remaining = map.yolo_remaining_secs(50, None).await;
        assert!(
            remaining > 30 * SECS_PER_DAY - 5 && remaining <= 30 * SECS_PER_DAY,
            "new grant must report ~30d remaining: {remaining}"
        );

        // flush persists ttl_secs=30d for the new grant.
        map.flush().await.expect("flush");
        let raw = std::fs::read_to_string(&path).expect("read snapshot");
        let yolo_line = raw
            .lines()
            .find(|l| l.contains("\"type\":\"yolo\""))
            .expect("yolo row present");
        let v: Value = serde_json::from_str(yolo_line).expect("parse yolo row");
        assert_eq!(
            v.get("ttl_secs").and_then(Value::as_i64),
            Some(YOLO_TTL_SECS),
            "new grant must persist ttl_secs=30d"
        );

        // A marked grant 20 days old reloads active; 31 days old is dropped.
        let twenty_days = now_secs() - (20 * SECS_PER_DAY);
        let thirtyone_days = now_secs() - (31 * SECS_PER_DAY);
        std::fs::write(
            &path,
            format!(
                "{}\n{}\n",
                json!({"type": "yolo", "chat_id": 51, "thread_id": 0, "enabled_at": twenty_days, "ttl_secs": YOLO_TTL_SECS}),
                json!({"type": "yolo", "chat_id": 52, "thread_id": 0, "enabled_at": thirtyone_days, "ttl_secs": YOLO_TTL_SECS}),
            ),
        )
        .expect("rewrite snapshot");
        let reopened = ChannelSessionMap::open(dir.path()).await.expect("reopen");
        assert!(
            reopened.is_yolo(51, None).await,
            "20-day-old 30d grant must be active"
        );
        assert!(
            !reopened.is_yolo(52, None).await,
            "31-day-old 30d grant must be dropped"
        );
    }

    // ── Ring 2: per-chat permanent escalation ──────────────────────────

    #[tokio::test]
    async fn first_enable_is_temporary_count_one_topic_scoped() {
        let map = ChannelSessionMap::new();
        let esc = map.enable_yolo(1, Some(5), None).await;
        assert!(matches!(esc.tier, YoloTier::Temporary { .. }));
        assert_eq!(esc.count, 1);
        assert_eq!(map.chat_yolo_state(1).await, (1, false));
        // Temporary grant is topic-scoped: active in the enabled topic only.
        assert!(map.is_yolo(1, Some(5)).await);
        assert!(!map.is_yolo(1, Some(6)).await);
        assert!(!map.is_yolo(1, None).await);
    }

    #[tokio::test]
    async fn second_distinct_enable_becomes_permanent_chat_wide() {
        let map = ChannelSessionMap::new();
        // 1st enable in topic 5.
        let first = map.enable_yolo(1, Some(5), None).await;
        assert!(matches!(first.tier, YoloTier::Temporary { .. }));
        // 2nd DISTINCT enable (different topic) crosses the threshold.
        let second = map.enable_yolo(1, Some(6), None).await;
        assert!(matches!(second.tier, YoloTier::Permanent));
        assert_eq!(second.count, 2);
        assert_eq!(map.chat_yolo_state(1).await, (2, true));
        // Permanent makes is_yolo TRUE for a DIFFERENT topic that was never
        // explicitly enabled, and remaining reports the sentinel.
        assert!(map.is_yolo(1, Some(999)).await);
        assert!(map.is_yolo(1, None).await);
        assert_eq!(
            map.yolo_remaining_secs(1, Some(999)).await,
            YOLO_PERMANENT_SENTINEL
        );
        assert_eq!(
            map.yolo_remaining_days(1, Some(999)).await,
            YOLO_PERMANENT_SENTINEL
        );
        // A different chat is unaffected.
        assert!(!map.is_yolo(2, Some(999)).await);
    }

    #[tokio::test]
    async fn same_call_id_counts_once_distinct_reach_permanent() {
        let map = ChannelSessionMap::new();
        // Tapping the SAME permission card twice (same call_id) counts once.
        map.enable_yolo(1, Some(5), Some("call-a")).await;
        let again = map.enable_yolo(1, Some(5), Some("call-a")).await;
        assert_eq!(again.count, 1);
        assert_eq!(map.chat_yolo_state(1).await, (1, false));
        assert!(!map.is_yolo(1, Some(999)).await);
        // A DISTINCT card (different call_id) is the 2nd action → permanent.
        let distinct = map.enable_yolo(1, Some(5), Some("call-b")).await;
        assert_eq!(distinct.count, 2);
        assert!(matches!(distinct.tier, YoloTier::Permanent));
        assert!(map.is_yolo(1, Some(999)).await);
    }

    #[tokio::test]
    async fn dedup_state_lives_in_chat_row_and_is_dropped_with_it() {
        // The dedup guard is folded into the per-chat `ChatYolo` row (shares the
        // yolo_chat lock with the count), so clearing the chat drops it too:
        // after `/yolo off` a repeat of the SAME call_id starts a fresh count.
        let map = ChannelSessionMap::new();
        let first = map.enable_yolo(1, Some(5), Some("call-a")).await;
        assert_eq!(first.count, 1);
        // Same call_id again → deduped, still 1 (guard lives in the row).
        let dup = map.enable_yolo(1, Some(5), Some("call-a")).await;
        assert_eq!(dup.count, 1);
        // Clearing the chat removes the whole row incl. the dedup guard.
        map.clear_yolo_chat(1).await;
        assert_eq!(map.chat_yolo_state(1).await, (0, false));
        // The previously-deduped call_id is no longer remembered → counts fresh.
        let after = map.enable_yolo(1, Some(5), Some("call-a")).await;
        assert_eq!(after.count, 1);
    }

    #[tokio::test]
    async fn card_then_command_reach_permanent() {
        let map = ChannelSessionMap::new();
        map.enable_yolo(1, Some(5), Some("call-a")).await; // card
        let cmd = map.enable_yolo(1, None, None).await; // /yolo command
        assert_eq!(cmd.count, 2);
        assert!(matches!(cmd.tier, YoloTier::Permanent));
    }

    #[tokio::test]
    async fn permanent_survives_disable_yolo() {
        let map = ChannelSessionMap::new();
        map.enable_yolo(1, Some(5), None).await;
        map.enable_yolo(1, Some(6), None).await;
        assert_eq!(map.chat_yolo_state(1).await, (2, true));
        // `/new` disables the topic temporary grant but must NOT touch the
        // per-chat permanent state.
        map.disable_yolo(1, Some(5)).await;
        assert_eq!(map.chat_yolo_state(1).await, (2, true));
        assert!(map.is_yolo(1, Some(5)).await, "permanent survives /new");
        assert!(map.is_yolo(1, Some(6)).await);
    }

    #[tokio::test]
    async fn permanent_survives_flush_open_roundtrip() {
        let dir = tempfile::tempdir().expect("tempdir");
        let map = ChannelSessionMap::open(dir.path()).await.expect("open");
        map.enable_yolo(1, Some(5), None).await;
        map.enable_yolo(1, Some(6), None).await;
        assert!(map.is_yolo(1, Some(999)).await);
        map.flush().await.expect("flush");
        let reopened = ChannelSessionMap::open(dir.path()).await.expect("reopen");
        assert_eq!(reopened.chat_yolo_state(1).await, (2, true));
        assert!(
            reopened.is_yolo(1, Some(999)).await,
            "permanent must survive restart (channel_map snapshot authoritative)"
        );
    }

    #[tokio::test]
    async fn clear_yolo_chat_resets_everything() {
        let map = ChannelSessionMap::new();
        map.enable_yolo(1, Some(5), None).await;
        map.enable_yolo(1, Some(6), None).await;
        assert!(map.is_yolo(1, Some(999)).await);
        map.clear_yolo_chat(1).await;
        assert_eq!(map.chat_yolo_state(1).await, (0, false));
        assert!(!map.is_yolo(1, Some(999)).await);
        assert!(!map.is_yolo(1, Some(5)).await);
        assert!(!map.is_yolo(1, Some(6)).await);
        // A fresh enable after clear starts the count over at 1 (temporary).
        let esc = map.enable_yolo(1, Some(5), None).await;
        assert_eq!(esc.count, 1);
        assert!(matches!(esc.tier, YoloTier::Temporary { .. }));
    }

    #[tokio::test]
    async fn legacy_ambiguous_grant_without_explicit_marker_not_seeded() {
        // Fix 1 (SAFETY): a pre-upgrade legacy yolo row lacks the `explicit`
        // marker and is AMBIGUOUS — before this feature BOTH user `/yolo` and
        // automated research used enable_yolo, so it could be automated.
        // Defaulting it to explicit=true would seed count=1 and let the user's
        // first post-upgrade `/yolo` escalate the chat to PERMANENT
        // auto-approve-all off ambiguous data. It must fail safe: the missing
        // marker defaults to explicit=false → NOT seeded. (Mutation target:
        // defaulting the missing marker back to true seeds count=1 and fails
        // this test.)
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("channel_map.jsonl");
        let recent = now_secs() - 3600;
        std::fs::write(
            &path,
            format!(
                "{}\n",
                json!({"type": "yolo", "chat_id": 7, "thread_id": 0, "enabled_at": recent, "ttl_secs": YOLO_TTL_SECS}),
            ),
        )
        .expect("write snapshot");

        let map = ChannelSessionMap::open(dir.path()).await.expect("open");
        // The grant itself still admits (auto-approve for the topic stays
        // active); only the escalation SEED is withheld.
        assert!(
            map.is_yolo(7, None).await,
            "legacy grant still auto-approves"
        );
        map.seed_missing_chat_counts().await;
        assert_eq!(
            map.chat_yolo_state(7).await,
            (0, false),
            "ambiguous legacy grant (no explicit marker) must NOT seed the count"
        );
        // The user's next `/yolo` is the 1st EXPLICIT enable → temporary and
        // topic-scoped, NOT the 2nd → permanent.
        let esc = map.enable_yolo(7, None, None).await;
        assert_eq!(esc.count, 1);
        assert!(matches!(esc.tier, YoloTier::Temporary { .. }));
        assert!(
            !map.is_yolo(7, Some(123)).await,
            "first explicit enable is topic-scoped, not permanent chat-wide"
        );
    }

    #[tokio::test]
    async fn session_config_restored_grant_seeds_count_one_then_escalates() {
        // The SessionConfig restore path revives a grant via enable_yolo_at
        // (mirrors restore_yolo) but creates NO yolo_chat row. Without the
        // post-restore seed pass such a chat would start count=0 and the next
        // /yolo would be a fresh temporary grant instead of escalating.
        let map = ChannelSessionMap::new();
        map.enable_yolo_at(8, None, now_secs() - 3600, YOLO_TTL_SECS)
            .await;
        assert!(map.is_yolo(8, None).await);
        assert_eq!(
            map.chat_yolo_state(8).await,
            (0, false),
            "SessionConfig restore alone creates no yolo_chat row"
        );

        // The single post-restore seed pass covers this source too.
        map.seed_missing_chat_counts().await;
        assert_eq!(
            map.chat_yolo_state(8).await,
            (1, false),
            "seed pass makes count=1 for the restored-from-session-config chat"
        );

        // Next explicit enable is now the 2nd → permanent.
        let esc = map.enable_yolo(8, None, None).await;
        assert_eq!(esc.count, 2);
        assert!(matches!(esc.tier, YoloTier::Permanent));
        assert!(map.is_yolo(8, Some(999)).await);
    }

    #[tokio::test]
    async fn chat_with_neither_grant_nor_row_gets_no_entry() {
        let dir = tempfile::tempdir().expect("tempdir");
        let map = ChannelSessionMap::open(dir.path()).await.expect("open");
        assert_eq!(map.chat_yolo_state(42).await, (0, false));
    }

    #[tokio::test]
    async fn yolo_chat_persistence_roundtrip() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("channel_map.jsonl");
        let map = ChannelSessionMap::open(dir.path()).await.expect("open");
        // Drive to permanent, then flush and inspect the persisted row.
        map.enable_yolo(1, Some(5), None).await;
        map.enable_yolo(1, Some(6), None).await;
        map.flush().await.expect("flush");
        let raw = std::fs::read_to_string(&path).expect("read snapshot");
        let row = raw
            .lines()
            .find(|l| l.contains("\"type\":\"yolo_chat\""))
            .expect("yolo_chat row present after flush");
        let v: Value = serde_json::from_str(row).expect("parse yolo_chat row");
        assert_eq!(v.get("chat_id").and_then(Value::as_i64), Some(1));
        assert_eq!(v.get("count").and_then(Value::as_u64), Some(2));
        assert_eq!(v.get("permanent").and_then(Value::as_bool), Some(true));
        // Reload preserves the escalation state.
        let reopened = ChannelSessionMap::open(dir.path()).await.expect("reopen");
        assert_eq!(reopened.chat_yolo_state(1).await, (2, true));
    }

    #[tokio::test]
    async fn grant_temporary_yolo_does_not_escalate() {
        // Automated flows (research) refresh the temporary grant without
        // counting toward permanent escalation.
        let map = ChannelSessionMap::new();
        map.grant_temporary_yolo(1, Some(5)).await;
        map.grant_temporary_yolo(1, Some(5)).await;
        assert_eq!(map.chat_yolo_state(1).await, (0, false));
        assert!(map.is_yolo(1, Some(5)).await);
        assert!(
            !map.is_yolo(1, Some(6)).await,
            "no permanent, topic-scoped only"
        );
    }

    #[tokio::test]
    async fn only_explicit_grants_seed_chat_count() {
        // Fix 1: an automated (research) grant must NOT seed the escalation
        // count on restart; only an explicit user enable seeds count=1.
        let map = ChannelSessionMap::new();
        // Chat 1: automated grant only (explicit=false).
        map.grant_temporary_yolo(1, Some(5)).await;
        // Chat 2: explicit grant via the restore path (explicit=true).
        map.enable_yolo_at(2, None, now_secs() - 3600, YOLO_TTL_SECS)
            .await;
        map.seed_missing_chat_counts().await;

        assert_eq!(
            map.chat_yolo_state(1).await,
            (0, false),
            "automated-only grant must not seed a chat count"
        );
        assert_eq!(
            map.chat_yolo_state(2).await,
            (1, false),
            "explicit grant seeds count=1"
        );

        // Chat 1's first real `/yolo` is the 1st explicit enable → temporary,
        // NOT wrongly promoted to permanent.
        let esc = map.enable_yolo(1, Some(5), None).await;
        assert_eq!(esc.count, 1);
        assert!(matches!(esc.tier, YoloTier::Temporary { .. }));
        assert!(
            !map.is_yolo(1, Some(6)).await,
            "still topic-scoped, not permanent"
        );
    }

    #[tokio::test]
    async fn explicit_marker_persists_across_flush_reload() {
        // Fix 1: the `explicit` marker must round-trip through the snapshot so
        // an automated grant is still skipped by the post-restore seed pass
        // after a restart.
        let dir = tempfile::tempdir().expect("tempdir");
        let map = ChannelSessionMap::open(dir.path()).await.expect("open");
        map.grant_temporary_yolo(3, None).await; // automated (explicit=false)
        map.enable_yolo(4, None, None).await; // explicit user enable
        map.flush().await.expect("flush");

        let reopened = ChannelSessionMap::open(dir.path()).await.expect("reopen");
        reopened.seed_missing_chat_counts().await;
        assert_eq!(
            reopened.chat_yolo_state(3).await,
            (0, false),
            "reloaded automated grant kept explicit=false and was not seeded"
        );
        assert_eq!(
            reopened.chat_yolo_state(4).await,
            (1, false),
            "reloaded explicit grant seeds count=1"
        );
    }

    #[tokio::test]
    async fn sessions_for_chat_enumerates_all_topics_deduped() {
        // Two topics of the SAME chat map to two distinct sessions; `/yolo off`
        // must reach BOTH to clear their persisted SessionConfig grants.
        let map = ChannelSessionMap::new();
        let chat = 100;
        map.set(chat, Some(1), "sid1".into()).await;
        map.set(chat, Some(2), "sid2".into()).await;
        // A different chat's session must not leak in.
        map.set(200, Some(9), "other".into()).await;
        // Enable yolo in both topics of `chat`.
        map.enable_yolo(chat, Some(1), None).await;
        map.enable_yolo(chat, Some(2), None).await;

        let mut sids = map.sessions_for_chat(chat).await;
        sids.sort();
        assert_eq!(sids, vec!["sid1".to_string(), "sid2".to_string()]);
        assert!(
            !map.sessions_for_chat(chat)
                .await
                .contains(&"other".to_string())
        );

        // Dedup: one session mapped under two keys appears exactly once.
        let dchat = 300;
        map.set(dchat, None, "dup".into()).await;
        map.set(dchat, Some(7), "dup".into()).await;
        assert_eq!(map.sessions_for_chat(dchat).await, vec!["dup".to_string()]);

        // A chat with no mappings enumerates nothing.
        assert!(map.sessions_for_chat(999).await.is_empty());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn concurrent_flushes_produce_consistent_snapshot() {
        // Two flush() calls can run concurrently in production: `/yolo off`'s
        // flush and a permanent-escalation persist. Without serialization they
        // (a) write through the SAME temp path (corrupt/interleaved file) and
        // (b) can rename a stale buffer over a newer one. `flush_lock` makes
        // snapshot-build + write + rename atomic w.r.t. other flushes.
        use std::sync::Arc;
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("channel_map.jsonl");
        let map = Arc::new(ChannelSessionMap::open(dir.path()).await.expect("open"));

        // Populate real yolo / yolo_chat / map / allow rows so the snapshot is
        // non-trivial (a wider write window for any race to manifest).
        map.enable_yolo(1, Some(5), None).await;
        map.enable_yolo(1, Some(6), None).await; // 2nd distinct enable -> permanent
        map.enable_yolo(2, Some(7), None).await;
        for i in 0..16 {
            map.set(3, Some(8 + i), format!("sid{i}")).await;
            map.allow_add(3, Some(8 + i), "exec").await;
        }
        assert!(map.is_permanent(1).await, "chat 1 escalated to permanent");

        // Fire many rounds of two genuinely-parallel flushes. In the correct
        // build the lock serializes them so both always succeed, no temp is
        // left behind, and the snapshot stays valid JSONL — deterministic, so
        // the test never flakes green. If `flush_lock` is removed, the two
        // flushes share the per-process temp path: one can rename it away
        // before the other's rename, surfacing a rename error (flush returns
        // Err), and/or leave a corrupt/leftover temp — which this test catches.
        for _ in 0..50 {
            let a = {
                let map = Arc::clone(&map);
                tokio::spawn(async move { map.flush().await })
            };
            let b = {
                let map = Arc::clone(&map);
                tokio::spawn(async move { map.flush().await })
            };
            a.await.expect("join a").expect("flush a succeeds");
            b.await.expect("join b").expect("flush b succeeds");

            // Snapshot must always parse cleanly (no interleaved/truncated line)
            // and reopen without error.
            let raw = std::fs::read_to_string(&path).expect("read snapshot");
            for line in raw.lines().filter(|l| !l.trim().is_empty()) {
                serde_json::from_str::<Value>(line).expect("every snapshot line is valid JSON");
            }
            ChannelSessionMap::open(dir.path()).await.expect("reopen");
        }

        // Defense-in-depth: the old FIXED shared temp path is never used, and
        // no temp file lingers after successful flushes.
        assert!(
            !path.with_extension("jsonl.tmp").exists(),
            "flush must not use the fixed shared temp path"
        );
        let leftovers: Vec<String> = std::fs::read_dir(dir.path())
            .expect("read dir")
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| n.contains("jsonl.tmp"))
            .collect();
        assert!(
            leftovers.is_empty(),
            "no leftover temp files: {leftovers:?}"
        );
    }

    // ── B87: flush snapshot is an atomic cut (no multi-lock TOCTOU) ────

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn flush_holds_yolo_and_yolo_chat_guards_simultaneously() {
        // B87 lock-hold proof (deterministic): while flush is parked acquiring
        // `yolo_chat`, it must STILL hold the `map` and `yolo` read guards —
        // the snapshot is then an atomic point-in-time cut and NO writer can
        // interleave between the per-map reads. (Mutation target: the old
        // read-then-drop-then-lock flush released `yolo` before taking
        // `yolo_chat`, so the try_write probes below always succeed and the
        // wait loop times out — this test fails deterministically.)
        use std::sync::Arc;
        use std::time::{Duration, Instant};

        let dir = tempfile::tempdir().expect("tempdir");
        let cmap = Arc::new(ChannelSessionMap::open(dir.path()).await.expect("open"));
        cmap.enable_yolo(1, Some(5), None).await;

        // Park flush right before its `yolo_chat` read (the last-but-one
        // guard in the canonical order map → yolo → yolo_chat → allow_list).
        let gate = cmap.yolo_chat.write().await;
        let flush_task = {
            let cmap = Arc::clone(&cmap);
            tokio::spawn(async move { cmap.flush().await })
        };

        // While the gate is held flush cannot pass `yolo_chat`, so once BOTH
        // probes fail the earlier guards are provably held simultaneously
        // with the parked `yolo_chat` acquisition.
        let deadline = Instant::now() + Duration::from_secs(10);
        let mut proven = false;
        while Instant::now() < deadline {
            let map_held = cmap.map.try_write().is_err();
            let yolo_held = cmap.yolo.try_write().is_err();
            if map_held && yolo_held {
                proven = true;
                break;
            }
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
        assert!(
            proven,
            "flush parked on yolo_chat must still hold the map+yolo read guards \
             (atomic snapshot cut); a drop-then-relock flush leaves an interleave \
             window for a writer to tear the snapshot"
        );

        drop(gate);
        flush_task.await.expect("join").expect("flush succeeds");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn flush_snapshot_never_tears_yolo_off_interleave() {
        // B87 semantic proof: a `/yolo off`-style writer interleaving between
        // flush's `yolo` and `yolo_chat` reads must never produce a persisted
        // state that did not exist in memory — chat-1 grant rows WITHOUT the
        // matching escalation row would revive revoked grants on restart
        // (revocation bypass). With the atomic cut the writer either lands
        // fully before or fully after the snapshot.
        use std::sync::Arc;
        use std::time::Duration;

        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("channel_map.jsonl");
        let cmap = Arc::new(ChannelSessionMap::open(dir.path()).await.expect("open"));
        cmap.enable_yolo(1, Some(5), None).await;
        cmap.enable_yolo(1, Some(6), None).await; // 2nd distinct enable → permanent
        assert_eq!(cmap.chat_yolo_state(1).await, (2, true));

        // Park flush right before its `yolo_chat` read.
        let mut gate = cmap.yolo_chat.write().await;
        let flush_task = {
            let cmap = Arc::clone(&cmap);
            tokio::spawn(async move { cmap.flush().await })
        };
        // Let flush start and park on the gate.
        tokio::time::sleep(Duration::from_millis(250)).await;

        // Attempt the interleaved `/yolo off` (clear both maps for chat 1)
        // exactly between flush's `yolo` and `yolo_chat` reads. Under the
        // atomic flush the `yolo` guard is still held while flush is parked,
        // so the writer CANNOT interleave (try_write fails); under the old
        // drop-then-relock flush the guard is free and the tear happens.
        let interleaved = match cmap.yolo.try_write() {
            Ok(mut yolo) => {
                yolo.retain(|(cid, _tid), _| *cid != 1);
                gate.remove(&1);
                true
            }
            Err(_) => false,
        };
        drop(gate);
        flush_task.await.expect("join").expect("flush succeeds");

        let count_chat1_rows = || {
            let raw = std::fs::read_to_string(&path).expect("read snapshot");
            let mut yolo_rows = 0usize;
            let mut chat_rows = 0usize;
            for line in raw.lines().filter(|l| !l.trim().is_empty()) {
                let v: Value = serde_json::from_str(line).expect("valid JSONL");
                if v.get("chat_id").and_then(Value::as_i64) != Some(1) {
                    continue;
                }
                match v.get("type").and_then(Value::as_str) {
                    Some("yolo") => yolo_rows += 1,
                    Some("yolo_chat") => chat_rows += 1,
                    _ => {}
                }
            }
            (yolo_rows, chat_rows)
        };

        let (yolo_rows, chat_rows) = count_chat1_rows();
        assert!(
            (yolo_rows > 0) == (chat_rows > 0),
            "torn snapshot: {yolo_rows} yolo grant row(s) with {chat_rows} yolo_chat \
             row(s) for chat 1 — a combination that never existed in memory \
             (revoked grants would be revived on restart)"
        );

        if !interleaved {
            // Atomic flush: the cut is the complete pre-clear state…
            assert_eq!(
                (yolo_rows, chat_rows),
                (2, 1),
                "atomic cut must capture the full pre-clear state"
            );
            // …and completing the clear afterwards persists the other atomic
            // state (neither grant nor escalation rows).
            cmap.clear_yolo_chat(1).await;
            cmap.flush().await.expect("re-flush");
            assert_eq!(
                count_chat1_rows(),
                (0, 0),
                "post-clear flush must persist the fully-cleared atomic state"
            );
        }
    }
}
