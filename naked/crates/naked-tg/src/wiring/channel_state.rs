use std::sync::Arc;

use naked_core::AgentCore;
use naked_tg::channel_map::ChannelSessionMap;

pub(crate) async fn restore_channel_state(agent: &Arc<AgentCore>) -> Arc<ChannelSessionMap> {
    let naked_dir = naked_home_dir();
    let channel_map = open_channel_map(&naked_dir).await;
    restore_session_mappings(agent, &channel_map).await;
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
        restore_yolo(channel_map, cid, tid, session_id, sc.yolo_enabled_at).await;
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
) {
    if let Some(enabled_at) = enabled_at {
        channel_map.enable_yolo_at(cid, tid, enabled_at).await;
        if channel_map.is_yolo(cid, tid).await {
            let remaining_h = channel_map.yolo_remaining_secs(cid, tid).await / 3600;
            tracing::info!(
                cid,
                ?tid,
                remaining_h,
                "restored yolo for session {} ({remaining_h}h left)",
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
    use super::naked_home_dir;

    #[test]
    fn naked_home_dir_uses_home_env() {
        let dir = naked_home_dir();
        assert!(dir.ends_with(".naked"));
    }
}
