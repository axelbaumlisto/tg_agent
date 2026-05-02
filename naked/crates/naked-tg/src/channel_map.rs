use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde_json::{Value, json};
use tokio::fs;
use tokio::sync::RwLock;

/// YOLO auto-expires after this many seconds (72 hours).
const YOLO_TTL_SECS: i64 = 72 * 3600;

/// Composite key: (chat_id, thread_id).
/// thread_id=0 means "no topic" (general / DM without topics).
type ChannelKey = (i64, i64);

fn make_key(chat_id: i64, thread_id: Option<i32>) -> ChannelKey {
    (chat_id, thread_id.unwrap_or(0) as i64)
}

pub fn format_tg_channel_id(chat_id: i64, thread_id: Option<i32>) -> String {
    format!("tg:{}:{}", chat_id, thread_id.unwrap_or(0))
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
///  1. **yolo** — per-topic flag: auto-approve everything
///  2. **allow_list** — per-topic set of tool names: auto-approve only listed tools
///  3. Otherwise ask the user
pub struct ChannelSessionMap {
    map: RwLock<HashMap<ChannelKey, String>>,
    /// Value = unix timestamp (secs) when yolo was enabled.
    yolo: RwLock<HashMap<ChannelKey, i64>>,
    allow_list: RwLock<HashMap<ChannelKey, HashSet<String>>>,
    /// Durable backing file (JSONL snapshot). `None` means the map is
    /// purely in-memory (used by unit tests). Populated by [`Self::open`]
    /// and rewritten atomically by [`Self::flush`]; `naked-tg`'s `main`
    /// opens the snapshot at startup and re-flushes every 30s.
    snapshot_path: RwLock<Option<PathBuf>>,
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
            yolo: RwLock::new(HashMap::new()),
            allow_list: RwLock::new(HashMap::new()),
            snapshot_path: RwLock::new(None),
        }
    }

    // ── Durable snapshot (A1) ──────────────────────────────────────────
    //
    // Structure: a single JSONL file under `<naked_root>/channel_map.jsonl`.
    // Each line is one of:
    //   {"type":"map",    "chat_id":..., "thread_id":..., "session_id":"..."}
    //   {"type":"yolo",   "chat_id":..., "thread_id":..., "enabled_at":...}
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
                let thread_id = v.get("thread_id").and_then(Value::as_i64);
                let ty = v.get("type").and_then(Value::as_str);
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
                        if let Some(enabled_at) = v.get("enabled_at").and_then(Value::as_i64)
                            && now_secs() - enabled_at < YOLO_TTL_SECS
                        {
                            yolo.insert(key, enabled_at);
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
        let path = match self.snapshot_path.read().await.clone() {
            Some(p) => p,
            None => return Ok(()), // in-memory map, nothing to do
        };
        let mut buf = String::new();
        let map = self.map.read().await;
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
        drop(map);
        let yolo = self.yolo.read().await;
        for ((chat_id, thread_id), enabled_at) in yolo.iter() {
            let line = json!({
                "type": "yolo",
                "chat_id": chat_id,
                "thread_id": thread_id,
                "enabled_at": enabled_at,
            });
            buf.push_str(&line.to_string());
            buf.push('\n');
        }
        drop(yolo);
        let allow = self.allow_list.read().await;
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
        let tmp = path.with_extension("jsonl.tmp");
        fs::write(&tmp, buf)
            .await
            .with_context(|| format!("channel_map: write {}", tmp.display()))?;
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

    // ── yolo (per-topic auto-approve-all, 72h TTL) ──────────────────────

    /// Check if yolo is active and not expired. Auto-removes expired entries.
    pub async fn is_yolo(&self, chat_id: i64, thread_id: Option<i32>) -> bool {
        let key = make_key(chat_id, thread_id);
        let yolo = self.yolo.read().await;
        if let Some(&enabled_at) = yolo.get(&key) {
            if now_secs() - enabled_at < YOLO_TTL_SECS {
                return true;
            }
            drop(yolo);
            self.yolo.write().await.remove(&key);
        }
        false
    }

    /// Enable yolo with current timestamp. Returns `true` if newly enabled.
    pub async fn enable_yolo(&self, chat_id: i64, thread_id: Option<i32>) -> bool {
        let key = make_key(chat_id, thread_id);
        let was_absent = !self.yolo.read().await.contains_key(&key);
        self.yolo.write().await.insert(key, now_secs());
        was_absent
    }

    /// Enable yolo with a specific timestamp (for restoring from disk).
    pub async fn enable_yolo_at(&self, chat_id: i64, thread_id: Option<i32>, enabled_at: i64) {
        if now_secs() - enabled_at < YOLO_TTL_SECS {
            self.yolo
                .write()
                .await
                .insert(make_key(chat_id, thread_id), enabled_at);
        }
    }

    /// Seconds remaining until yolo expires, or 0 if not active.
    pub async fn yolo_remaining_secs(&self, chat_id: i64, thread_id: Option<i32>) -> i64 {
        let key = make_key(chat_id, thread_id);
        if let Some(&enabled_at) = self.yolo.read().await.get(&key) {
            let elapsed = now_secs() - enabled_at;
            if elapsed < YOLO_TTL_SECS {
                return YOLO_TTL_SECS - elapsed;
            }
        }
        0
    }

    /// Disable yolo for a topic (called on /new).
    pub async fn disable_yolo(&self, chat_id: i64, thread_id: Option<i32>) {
        self.yolo
            .write()
            .await
            .remove(&make_key(chat_id, thread_id));
    }

    // ── allow-list (per-topic tool whitelist) ────────────────────────────

    /// Returns true if `tool_name` should be auto-approved for this topic.
    /// Tools that are always safe to auto-approve (read-only, no side effects).
    const SAFE_TOOLS: &'static [&'static str] = &[
        "read_file",
        "web_search",
        "glob_search",
        "grep_search",
        "agent_status",
    ];

    pub async fn should_auto_approve(
        &self,
        chat_id: i64,
        thread_id: Option<i32>,
        tool_name: &str,
    ) -> bool {
        // A4: Read-only tools always auto-approve.
        if Self::SAFE_TOOLS.contains(&tool_name) {
            return true;
        }
        if self.is_yolo(chat_id, thread_id).await {
            return true;
        }
        let key = make_key(chat_id, thread_id);
        if let Some(tools) = self.allow_list.read().await.get(&key) {
            return tools.contains(tool_name);
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

        map.enable_yolo(1, Some(5)).await;
        assert!(!map.is_yolo(1, None).await);
        assert!(map.is_yolo(1, Some(5)).await);

        // enable again is idempotent
        map.enable_yolo(1, Some(5)).await;
        assert!(map.is_yolo(1, Some(5)).await);

        // disable
        map.disable_yolo(1, Some(5)).await;
        assert!(!map.is_yolo(1, Some(5)).await);
    }

    #[tokio::test]
    async fn allow_list_basics() {
        let map = ChannelSessionMap::new();
        // bash is NOT safe — needs explicit allow or yolo
        assert!(!map.should_auto_approve(1, Some(2), "bash").await);
        // A4: read_file IS safe — always auto-approved
        assert!(map.should_auto_approve(1, Some(2), "read_file").await);
        assert!(map.should_auto_approve(1, Some(2), "web_search").await);
        assert!(map.should_auto_approve(1, Some(2), "grep_search").await);

        map.allow_add(1, Some(2), "bash").await;

        assert!(map.should_auto_approve(1, Some(2), "bash").await);
        assert!(!map.should_auto_approve(1, Some(2), "write_file").await);
        assert!(!map.should_auto_approve(1, Some(3), "bash").await);

        let list = map.allow_get(1, Some(2)).await;
        assert_eq!(list, vec!["bash"]);

        map.allow_remove(1, Some(2), "bash").await;
        assert!(!map.should_auto_approve(1, Some(2), "bash").await);
        // read_file still auto-approved (safe tool)
        assert!(map.should_auto_approve(1, Some(2), "read_file").await);
    }

    #[tokio::test]
    async fn yolo_overrides_allow_list() {
        let map = ChannelSessionMap::new();
        assert!(!map.should_auto_approve(1, Some(5), "anything").await);

        map.enable_yolo(1, Some(5)).await;
        assert!(map.should_auto_approve(1, Some(5), "anything").await);
        assert!(map.should_auto_approve(1, Some(5), "bash").await);
    }

    #[tokio::test]
    async fn should_auto_approve_checks_both() {
        let map = ChannelSessionMap::new();
        assert!(!map.should_auto_approve(1, None, "bash").await);

        map.allow_add(1, None, "bash").await;
        assert!(map.should_auto_approve(1, None, "bash").await);
        assert!(!map.should_auto_approve(1, None, "write_file").await);

        map.enable_yolo(1, None).await;
        assert!(map.should_auto_approve(1, None, "write_file").await);
    }

    #[tokio::test]
    async fn yolo_expires_after_ttl() {
        let map = ChannelSessionMap::new();
        // Set yolo with a timestamp 73 hours in the past (expired)
        let expired_ts = now_secs() - (73 * 3600);
        map.enable_yolo_at(1, Some(5), expired_ts).await;
        assert!(!map.is_yolo(1, Some(5)).await);

        // Set yolo with a timestamp 1 hour in the past (still valid)
        let recent_ts = now_secs() - 3600;
        map.enable_yolo_at(1, Some(5), recent_ts).await;
        assert!(map.is_yolo(1, Some(5)).await);
        assert!(map.yolo_remaining_secs(1, Some(5)).await > 0);
    }
}
