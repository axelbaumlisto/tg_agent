//! `naked vacuum-sessions` subcommand — migrate and GC session artifacts.

use anyhow::Result;
use naked_core::AgentCore;
use naked_core::config::Config;

/// Migrate existing JSONL sessions that still carry inline base64 image
/// payloads. For each session, load → re-save (which runs through
/// `extern_image_blocks` and writes images to `<session>/artifacts/`), then
/// sweep orphan artifacts. Reports byte savings per session and a grand total.
///
/// Why a separate command instead of "migrate on first save": existing
/// long-lived sessions may never get a save event again (channel-only chats
/// closed by the user), and we want a single deterministic operator action to
/// reclaim disk space after upgrading.
pub(crate) async fn vacuum_sessions_cmd(extra_args: &[String]) -> Result<()> {
    // Optional `--max-age-days N` flag runs an age-based sweep across every
    // session's artifacts directory *in addition to* reachability GC. Default
    // N=0 means "age sweep disabled; reachability-only".
    let max_age_days: u64 = extra_args
        .iter()
        .position(|a| a == "--max-age-days")
        .and_then(|i| extra_args.get(i + 1))
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(0);
    let max_age_secs = max_age_days.saturating_mul(24 * 3600);

    let config = Config::load()?;
    let provider = naked_core::build_provider_from_config(&config)?;
    let agent = AgentCore::new(config.clone(), provider);
    let restored = agent.restore_sessions().await.unwrap_or_default();
    eprintln!("Found {} session(s).", restored.len());
    if max_age_days > 0 {
        eprintln!("Age-based sweep enabled: delete img_* older than {max_age_days} day(s).");
    }

    let store = agent.store();

    let mut total_saved: i64 = 0;
    let mut migrated = 0usize;
    let mut aged_out = 0usize;
    for sid in &restored {
        let before = std::fs::metadata(store.session_root(sid).join("session.jsonl"))
            .map(|m| m.len() as i64)
            .unwrap_or(0);

        let session = match store.load(sid).await? {
            Some(s) => s,
            None => continue,
        };
        store.save(&session).await?;

        let removed = store.gc_orphan_image_artifacts(sid).await.unwrap_or(0);
        let aged = if max_age_secs > 0 {
            store
                .gc_old_image_artifacts(sid, max_age_secs)
                .await
                .unwrap_or(0)
        } else {
            0
        };
        aged_out += aged;

        let after = std::fs::metadata(store.session_root(sid).join("session.jsonl"))
            .map(|m| m.len() as i64)
            .unwrap_or(0);
        let saved = before - after;
        total_saved += saved;
        if saved != 0 || removed > 0 || aged > 0 {
            migrated += 1;
            eprintln!(
                "  • {}…  saved {saved:+} bytes, gc'd {removed} orphan + {aged} aged artifact(s)",
                &sid[..sid.len().min(8)]
            );
        }
    }
    eprintln!(
        "Done. {migrated} session(s) had changes; jsonl delta {total_saved:+} bytes; \
         {aged_out} aged artifact(s) removed."
    );
    Ok(())
}
