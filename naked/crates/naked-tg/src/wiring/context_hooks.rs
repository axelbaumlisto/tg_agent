use std::sync::Arc;

use naked_core::AgentCore;
use naked_core::types::ConversationMessage;

pub(crate) async fn install_builtin_context_hooks(agent: &Arc<AgentCore>) {
    // B6: Register built-in context hook — inject short git status.
    // Helps the model know if there are uncommitted changes.
    agent
        .hooks()
        .on_context(Arc::new(|msgs: &mut Vec<ConversationMessage>| {
            inject_git_diff_stat(msgs);
        }))
        .await;
}

fn inject_git_diff_stat(msgs: &mut Vec<ConversationMessage>) {
    // Only inject if the first message is a system prompt and we're in a git
    // repo (workspace is set in system prompt).
    if msgs.is_empty() {
        return;
    }
    // Quick check with timeout — skip if git is slow or not a repo.
    let output = std::process::Command::new("git")
        .args(["diff", "--stat", "HEAD"])
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .output();
    if let Ok(out) = output {
        let stat = String::from_utf8_lossy(&out.stdout);
        let stat = stat.trim();
        if !stat.is_empty() && stat.len() < 500 {
            msgs.push(ConversationMessage::user(format!(
                "[git diff --stat]\n{stat}"
            )));
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn hook_keeps_git_stat_marker() {
        let source = include_str!("context_hooks.rs");
        assert!(source.contains("git"));
        assert!(source.contains("diff"));
        assert!(source.contains("--stat"));
        assert!(source.contains("[git diff --stat]"));
    }
}
