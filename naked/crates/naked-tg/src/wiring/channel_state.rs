use std::sync::Arc;

use naked_core::AgentCore;
use naked_tg::channel_map::{ChannelSessionMap, YOLO_LEGACY_TTL_SECS, format_yolo_window};

pub(crate) async fn restore_channel_state(agent: &Arc<AgentCore>) -> Arc<ChannelSessionMap> {
    let naked_dir = naked_home_dir();
    let channel_map = open_channel_map(&naked_dir).await;
    restore_session_mappings(agent, &channel_map).await;
    // Single seeding point (DRY): after BOTH the snapshot open and the
    // SessionConfig restore have populated the in-memory yolo map, seed a
    // count=1 escalation row for any chat that has a grant but no `yolo_chat`
    // row. This covers legacy snapshots AND session-config-only revivals in one
    // pass, so their next explicit `/yolo` correctly escalates to permanent.
    channel_map.seed_missing_chat_counts().await;
    flush_initial_snapshot(&channel_map).await;
    spawn_periodic_flush(channel_map.clone());
    channel_map
}

fn naked_home_dir() -> std::path::PathBuf {
    // Reconstruct the naked home dir (same formula as bootstrap.rs).
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    std::path::PathBuf::from(&home).join(".naked")
}

async fn open_channel_map(naked_dir: &std::path::Path) -> Arc<ChannelSessionMap> {
    // Open the durable channel-map snapshot. On any failure we fall back to
    // an in-memory map so the bot still starts; restoration via session-meta
    // below remains the authoritative recovery path.
    match ChannelSessionMap::open(naked_dir).await {
        Ok(m) => Arc::new(m),
        Err(e) => {
            tracing::warn!("channel_map snapshot open failed ({e:#}); using in-memory only");
            Arc::new(ChannelSessionMap::new())
        }
    }
}

async fn restore_session_mappings(agent: &Arc<AgentCore>, channel_map: &Arc<ChannelSessionMap>) {
    // Rebuild channel→session mapping from persisted metadata. This is the
    // authoritative path — the JSONL snapshot above is a cache that gets
    // refreshed below once the in-memory state is fully populated.
    let mappings = agent.channel_session_mappings().await;
    let restored_links = channel_map.restore_from(&mappings).await;
    if restored_links > 0 {
        tracing::info!("Restored {restored_links} channel→session link(s)");
    }

    // Restore yolo + allow_list from persisted session configs.
    for (channel_id, session_id) in &mappings {
        restore_session_config(agent, channel_map, channel_id, session_id).await;
    }
}

async fn restore_session_config(
    agent: &Arc<AgentCore>,
    channel_map: &Arc<ChannelSessionMap>,
    channel_id: &str,
    session_id: &str,
) {
    let sc = agent.load_session_config_pub(session_id);
    let parts: Vec<&str> = channel_id.splitn(3, ':').collect();
    if parts.len() == 3
        && parts[0] == "tg"
        && let (Ok(cid), Ok(raw_tid)) = (parts[1].parse::<i64>(), parts[2].parse::<i64>())
    {
        let tid = if raw_tid == 0 {
            None
        } else {
            Some(raw_tid as i32)
        };
        restore_yolo(
            channel_map,
            cid,
            tid,
            session_id,
            sc.yolo_enabled_at,
            sc.yolo_ttl_secs,
        )
        .await;
        if let Some(tools) = &sc.allow_list {
            restore_allow_list(channel_map, cid, tid, session_id, tools).await;
        }
    }
}

async fn restore_yolo(
    channel_map: &Arc<ChannelSessionMap>,
    cid: i64,
    tid: Option<i32>,
    session_id: &str,
    enabled_at: Option<i64>,
    ttl_secs: Option<i64>,
) {
    if let Some(enabled_at) = enabled_at {
        // Legacy grants (no persisted TTL) are judged against the original 72h
        // cutoff so the 72h→30d change never revives a grant that already
        // expired under the old policy; new grants carry their 30d window.
        let ttl = ttl_secs.unwrap_or(YOLO_LEGACY_TTL_SECS);
        channel_map.enable_yolo_at(cid, tid, enabled_at, ttl).await;
        if channel_map.is_yolo(cid, tid).await {
            // Permanent chat-wide grants report a sentinel; render "навсегда"
            // instead of a huge number at this display site.
            let remaining = format_yolo_window(channel_map.yolo_remaining_days(cid, tid).await);
            tracing::info!(
                cid,
                ?tid,
                %remaining,
                "restored yolo for session {} ({remaining} left)",
                &session_id[..8] // REGISTRY-WAIVE: B48 — session ID is ASCII hex
            );
        } else {
            tracing::info!(cid, ?tid, "yolo expired for session {}", &session_id[..8]); // REGISTRY-WAIVE: B48 — session ID is ASCII hex
        }
    }
}

async fn restore_allow_list(
    channel_map: &Arc<ChannelSessionMap>,
    cid: i64,
    tid: Option<i32>,
    session_id: &str,
    tools: &[String],
) {
    for tool in tools {
        channel_map.allow_add(cid, tid, tool).await;
    }
    if !tools.is_empty() {
        tracing::info!(
            cid,
            ?tid,
            n = tools.len(),
            "restored allow-list for session {}",
            &session_id[..8] // REGISTRY-WAIVE: B48 — session ID is ASCII hex
        );
    }
}

async fn flush_initial_snapshot(channel_map: &Arc<ChannelSessionMap>) {
    // Refresh the durable snapshot to reflect everything we just restored.
    if let Err(e) = channel_map.flush().await {
        tracing::warn!("channel_map: initial flush failed: {e:#}");
    }
}

fn spawn_periodic_flush(channel_map: Arc<ChannelSessionMap>) {
    // Periodic snapshot writer: cheap atomic temp+rename every 30s.
    tokio::spawn(async move {
        let mut tick = tokio::time::interval(std::time::Duration::from_secs(30));
        tick.tick().await; // skip the immediate first tick
        loop {
            tick.tick().await;
            if let Err(e) = channel_map.flush().await {
                tracing::warn!("channel_map: periodic flush failed: {e:#}");
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::{naked_home_dir, restore_yolo};
    use naked_core::config::YOLO_TTL_SECS;
    use naked_tg::channel_map::ChannelSessionMap;
    use std::sync::Arc;

    #[test]
    fn naked_home_dir_uses_home_env() {
        let dir = naked_home_dir();
        assert!(dir.ends_with(".naked"));
    }

    fn now_secs() -> i64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0)
    }

    /// The SessionConfig revival path must apply the same legacy-vs-new
    /// distinction as the snapshot loader: a legacy grant (no persisted TTL)
    /// already expired under the old 72h policy must NOT be revived for 30d,
    /// while a new grant carrying its 30d TTL is restored within that window.
    #[tokio::test]
    async fn restore_yolo_legacy_grant_not_revived_new_grant_honored() {
        const DAY: i64 = 24 * 3600;
        let map = Arc::new(ChannelSessionMap::new());
        let sid = "0123456789abcdef";

        // Legacy grant (ttl None) 5 days old -> expired under 72h, dropped.
        restore_yolo(&map, 1, None, sid, Some(now_secs() - 5 * DAY), None).await;
        assert!(
            !map.is_yolo(1, None).await,
            "5-day-old legacy SessionConfig grant must not be revived under 72h"
        );

        // Legacy grant (ttl None) 2h old -> inside 72h, restored.
        restore_yolo(&map, 2, None, sid, Some(now_secs() - 2 * 3600), None).await;
        assert!(
            map.is_yolo(2, None).await,
            "2-hour-old legacy grant is inside 72h and must be restored"
        );

        // New grant (ttl 30d) 20 days old -> restored.
        restore_yolo(
            &map,
            3,
            None,
            sid,
            Some(now_secs() - 20 * DAY),
            Some(YOLO_TTL_SECS),
        )
        .await;
        assert!(
            map.is_yolo(3, None).await,
            "20-day-old grant under its 30d TTL must be restored"
        );

        // New grant (ttl 30d) 31 days old -> dropped.
        restore_yolo(
            &map,
            4,
            None,
            sid,
            Some(now_secs() - 31 * DAY),
            Some(YOLO_TTL_SECS),
        )
        .await;
        assert!(
            !map.is_yolo(4, None).await,
            "31-day-old grant is past its 30d TTL and must be dropped"
        );
    }
}
