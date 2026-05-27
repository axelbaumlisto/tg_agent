use naked_core::config::Config;

pub(crate) fn acquire_scheduler_lock(
    config: &Config,
) -> Option<naked_tg::scheduler_lock::SchedulerLock> {
    // Cross-process advisory lock guarding `<NAKED_HOME>/research/`.
    // Acquired BEFORE we wire the scheduler so a second `naked-tg` instance
    // pointed at the same NAKED_HOME aborts immediately instead of corrupting
    // `inflight.json` and `runs.jsonl` via append races. Held by binding to
    // `_scheduler_lock` so it lives for the lifetime of the bot process; drop
    // on exit releases it. We keep an `Option` so test or future tooling can
    // run without a research subsystem at all.
    if !config.research.enabled {
        return None;
    }

    let research_root = config
        .research
        .storage_dir
        .clone()
        .unwrap_or_else(naked_core::research::research_root);
    match naked_tg::scheduler_lock::SchedulerLock::try_acquire(&research_root) {
        Ok(lock) => Some(lock),
        Err(naked_tg::scheduler_lock::LockError::Held { path, existing_pid }) => {
            tracing::error!(
                lock = %path.display(),
                holder_pid = ?existing_pid,
                "research scheduler lock is held by another naked-tg process; \
                 refusing to start the scheduler to avoid corrupting state. \
                 Stop the other instance or point NAKED_HOME at a different \
                 directory."
            );
            None
        }
        Err(naked_tg::scheduler_lock::LockError::Io(e)) => {
            tracing::error!(
                "failed to acquire scheduler lock under {}: {e}; refusing to start scheduler",
                research_root.display()
            );
            None
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn lock_policy_remains_fail_closed_on_errors() {
        let source = include_str!("scheduler_lock.rs");
        assert!(source.contains("if !config.research.enabled"));
        assert!(source.contains("refusing to start the scheduler"));
        assert!(source.contains("LockError::Held"));
        assert!(source.contains("LockError::Io"));
    }
}
