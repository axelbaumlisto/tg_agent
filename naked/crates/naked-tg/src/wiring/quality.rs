use std::sync::Arc;

use naked_core::AgentCore;

pub(crate) async fn install_quality_managers(agent: &Arc<AgentCore>) {
    // PLAN_QUALITY_v1 wiring (T2/T5/T6): install pluggable managers.
    // Each is opt-in via config / disk presence; missing = silent off-path
    // (zero overhead).
    install_lsp(agent);
    install_lifecycle_hooks(agent).await;
    install_permissions(agent);
    tracing::info!("PLAN_QUALITY_v1 wiring installed: lsp + hooks + permissions");
}

fn install_lsp(agent: &Arc<AgentCore>) {
    // T2 LSP manager. ENABLED by default (post-edit compiler feedback is the
    // biggest quality multiplier in PLAN_QUALITY_v1). Lazy: LSP servers spawn
    // on first edit per language. Operators who want to disable can set
    // NAKED_LSP_DISABLED=1.
    let lsp_cfg = naked_core::lsp::LspConfig::default();
    tracing::info!(
        lsp_enabled = lsp_cfg.enabled,
        lsp_warn_included = lsp_cfg.include_warnings,
        lsp_max_diagnostics = lsp_cfg.max_diagnostics_per_file,
        "PLAN_QUALITY_v1 LSP manager configured"
    );
    let lsp = Arc::new(naked_core::lsp::LspManager::new(lsp_cfg));
    agent.set_lsp(lsp);
}

async fn install_lifecycle_hooks(agent: &Arc<AgentCore>) {
    // T6 lifecycle hooks: load ~/.naked/hooks.json if present.
    // Empty file / missing path = no hooks installed (silent).
    let hooks = Arc::new(naked_core::lifecycle_hooks::LifecycleHookRunner::new());
    hooks.load_default().await;
    agent.set_lifecycle_hooks(hooks);
}

fn install_permissions(agent: &Arc<AgentCore>) {
    // T5 permission ruleset: load ~/.naked/permissions.json if present.
    // Empty file / missing path = empty ruleset = every tool falls through to
    // the existing UI prompt (Ask).
    let ruleset = naked_core::permissions::Store::load();
    let permissions = Arc::new(tokio::sync::RwLock::new(ruleset));
    agent.set_permissions(permissions);
}

#[cfg(test)]
mod tests {
    #[test]
    fn quality_wiring_mentions_all_three_managers() {
        let source = include_str!("quality.rs");
        assert!(source.contains("install_lsp"));
        assert!(source.contains("install_lifecycle_hooks"));
        assert!(source.contains("install_permissions"));
    }
}
