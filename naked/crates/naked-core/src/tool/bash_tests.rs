use super::*;
use crate::tool::Tool;

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

#[test]
fn classify_read_only_commands() {
    assert_eq!(classify_bash("ls -la"), BashRisk::ReadOnly);
    assert_eq!(classify_bash("cat file.txt"), BashRisk::ReadOnly);
    assert_eq!(classify_bash("git status"), BashRisk::ReadOnly);
    assert_eq!(classify_bash("git diff HEAD"), BashRisk::ReadOnly);
    assert_eq!(classify_bash("rg pattern src/"), BashRisk::ReadOnly);
    assert_eq!(classify_bash("pwd"), BashRisk::ReadOnly);
    assert_eq!(classify_bash("cargo test --lib"), BashRisk::ReadOnly);
    assert_eq!(classify_bash("head -20 file.rs"), BashRisk::ReadOnly);
    assert_eq!(classify_bash("find . -name '*.rs'"), BashRisk::ReadOnly);
}

#[test]
fn classify_piped_read_only() {
    assert_eq!(classify_bash("cat file.txt | grep foo"), BashRisk::ReadOnly);
    assert_eq!(classify_bash("ls -la | wc -l"), BashRisk::ReadOnly);
}

#[test]
fn classify_write_commands() {
    assert_eq!(classify_bash("echo x > file.txt"), BashRisk::Write);
    assert_eq!(classify_bash("git commit -m 'msg'"), BashRisk::Write);
    assert_eq!(classify_bash("cp a.txt b.txt"), BashRisk::Write);
    assert_eq!(classify_bash("mkdir -p new_dir"), BashRisk::Write);
    assert_eq!(classify_bash("npm install"), BashRisk::Write);
}

#[test]
fn classify_destructive_commands() {
    assert_eq!(classify_bash("rm -rf /"), BashRisk::Destructive);
    assert_eq!(classify_bash("rm -rf /*"), BashRisk::Destructive);
    assert_eq!(classify_bash("mkfs.ext4 /dev/sda"), BashRisk::Destructive);
    assert_eq!(
        classify_bash("dd if=/dev/zero of=/dev/sda"),
        BashRisk::Destructive
    );
}

#[test]
fn classify_bash_permission_mapping() {
    let tool = BashTool::new(30);
    assert_eq!(
        tool.effective_permission(&serde_json::json!({"command": "ls"}), Path::new("/")),
        Permission::ReadOnly
    );
    assert_eq!(
        tool.effective_permission(&serde_json::json!({"command": "rm -rf /"}), Path::new("/")),
        Permission::Dangerous
    );
    assert_eq!(
        tool.effective_permission(
            &serde_json::json!({"command": "npm install"}),
            Path::new("/")
        ),
        Permission::WorkspaceWrite
    );
}

#[test]
fn spec_has_correct_name() {
    let tool = BashTool::new(30);
    assert_eq!(tool.spec().name, "bash");
    assert_eq!(tool.spec().permission, Permission::Dangerous);
}

#[tokio::test]
async fn execute_echo() {
    let tool = BashTool::new(10);
    let result = tool
        .execute(
            serde_json::json!({"command": "echo hello"}),
            Path::new("/tmp"),
        )
        .await;
    assert!(!result.is_error);
    assert_eq!(result.output.trim(), "hello");
}

#[tokio::test]
async fn execute_failing_command() {
    let tool = BashTool::new(10);
    let result = tool
        .execute(serde_json::json!({"command": "false"}), Path::new("/tmp"))
        .await;
    assert!(result.is_error);
}

#[tokio::test]
async fn execute_captures_stderr() {
    let tool = BashTool::new(10);
    let result = tool
        .execute(
            serde_json::json!({"command": "echo err >&2"}),
            Path::new("/tmp"),
        )
        .await;
    assert!(result.output.contains("err"));
}

#[tokio::test]
async fn execute_timeout() {
    let tool = BashTool::new(1);
    let result = tool
        .execute(
            serde_json::json!({"command": "sleep 10", "timeout": 1}),
            Path::new("/tmp"),
        )
        .await;
    assert!(result.is_error);
    assert!(result.output.contains("timed out"));
}

#[tokio::test]
async fn execute_invalid_input() {
    let tool = BashTool::new(10);
    let result = tool
        .execute(serde_json::json!({"wrong_key": "ls"}), Path::new("/tmp"))
        .await;
    assert!(result.is_error);
    assert!(result.output.contains("Invalid input"));
}

#[tokio::test]
async fn execute_respects_cwd() {
    let dir = tempfile::tempdir().unwrap();
    let tool = BashTool::new(10);
    let result = tool
        .execute(serde_json::json!({"command": "pwd"}), dir.path())
        .await;
    assert!(!result.is_error);
    assert!(result.output.trim().contains(dir.path().to_str().unwrap()));
}

#[tokio::test]
async fn persistent_bash_flag_off_is_per_call_legacy() {
    let manager = Arc::new(super::super::persistent_bash::PersistentBashManager::new());
    let tool = BashTool::new(10);
    let result = tool
        .execute(
            serde_json::json!({"command": "export NAKED_FLAG_OFF=leak"}),
            Path::new("/tmp"),
        )
        .await;
    assert!(!result.is_error);
    assert_eq!(manager.session_count().await, 0);

    let result = tool
        .execute(
            serde_json::json!({"command": "printf '%s' \"${NAKED_FLAG_OFF:-unset}\""}),
            Path::new("/tmp"),
        )
        .await;
    assert!(!result.is_error);
    assert_eq!(result.output, "unset");
}

struct MockOps {
    calls: AtomicUsize,
}

#[async_trait::async_trait]
impl super::super::ops::ToolOps for MockOps {
    async fn read_file(&self, _path: &Path) -> std::io::Result<String> {
        Ok(String::new())
    }

    async fn write_file(&self, _path: &Path, _contents: &str) -> std::io::Result<()> {
        Ok(())
    }

    async fn file_exists(&self, _path: &Path) -> bool {
        true
    }

    async fn exec(
        &self,
        command: &str,
        _cwd: &Path,
        _timeout_secs: u64,
    ) -> std::io::Result<super::super::ops::ExecResult> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        assert_eq!(command, "echo remote");
        Ok(super::super::ops::ExecResult {
            stdout: b"remote\n".to_vec(),
            stderr: Vec::new(),
            exit_code: 0,
        })
    }

    fn label(&self) -> &str {
        "mock"
    }
}

#[tokio::test]
async fn persistent_bash_remote_ops_path_unchanged() {
    let ops = Arc::new(MockOps {
        calls: AtomicUsize::new(0),
    });
    let manager = Arc::new(super::super::persistent_bash::PersistentBashManager::new());
    let tool = BashTool::with_ops(10, ops.clone()).with_persistent(manager.clone(), "s".into());
    let result = tool
        .execute(
            serde_json::json!({"command": "echo remote"}),
            Path::new("/tmp"),
        )
        .await;

    assert!(!result.is_error);
    assert_eq!(result.output.trim(), "remote");
    assert_eq!(ops.calls.load(Ordering::Relaxed), 1);
    assert_eq!(manager.session_count().await, 0);
}

#[test]
fn blocks_systemctl_restart() {
    assert!(is_self_destructive(
        "systemctl --user restart naked-tg.service"
    ));
    assert!(is_self_destructive("systemctl restart naked-tg"));
    assert!(is_self_destructive(
        "cargo build && systemctl --user restart naked-tg.service"
    ));
}

#[test]
fn blocks_kill_commands() {
    assert!(is_self_destructive("kill -9 12345"));
    assert!(is_self_destructive("pkill naked-tg"));
    assert!(is_self_destructive("killall naked"));
}

#[test]
fn allows_normal_commands() {
    assert!(!is_self_destructive("ls -la"));
    assert!(!is_self_destructive("cargo build --release"));
    assert!(!is_self_destructive("cargo test"));
    assert!(!is_self_destructive("systemctl status naked-tg"));
    assert!(!is_self_destructive(
        "cat /etc/systemd/system/naked-tg.service"
    ));
}

// ── B85: git history-destructive guard ──────────────────────────────────────

#[test]
fn b85_git_append_only_ops_allowed() {
    // add / commit / push (no --force) ADD to history → never blocked.
    assert_eq!(git_history_destructive_op("git add -A"), None);
    assert_eq!(git_history_destructive_op("git add ."), None);
    assert_eq!(
        git_history_destructive_op("git commit -m 'fix: thing'"),
        None
    );
    assert_eq!(git_history_destructive_op("git commit -am 'wip'"), None);
    assert_eq!(git_history_destructive_op("git push"), None);
    assert_eq!(git_history_destructive_op("git push origin main"), None);
    assert_eq!(
        git_history_destructive_op("git push -u origin feature"),
        None
    );
    // read-only git stays allowed.
    assert_eq!(git_history_destructive_op("git log --oneline"), None);
    assert_eq!(git_history_destructive_op("git status"), None);
    assert_eq!(git_history_destructive_op("git diff HEAD~1"), None);
    // pipelines of allowed ops.
    assert_eq!(
        git_history_destructive_op("git add -A && git commit -m x && git push"),
        None
    );
}

#[test]
fn b85_git_history_destructive_ops_blocked() {
    assert_eq!(
        git_history_destructive_op("git reset --hard HEAD~3"),
        Some("reset")
    );
    assert_eq!(
        git_history_destructive_op("git reset HEAD file"),
        Some("reset")
    );
    assert_eq!(
        git_history_destructive_op("git checkout -- ."),
        Some("checkout")
    );
    assert_eq!(
        git_history_destructive_op("git checkout main"),
        Some("checkout")
    );
    assert_eq!(
        git_history_destructive_op("git switch other-branch"),
        Some("switch")
    );
    assert_eq!(
        git_history_destructive_op("git restore src/x.rs"),
        Some("restore")
    );
    assert_eq!(git_history_destructive_op("git clean -fd"), Some("clean"));
    assert_eq!(
        git_history_destructive_op("git rebase -i HEAD~5"),
        Some("rebase")
    );
    assert_eq!(
        git_history_destructive_op("git revert abc123"),
        Some("revert")
    );
    // force-push variants.
    assert_eq!(
        git_history_destructive_op("git push --force"),
        Some("push --force")
    );
    assert_eq!(
        git_history_destructive_op("git push --force-with-lease origin main"),
        Some("push --force")
    );
    assert_eq!(
        git_history_destructive_op("git push -f origin main"),
        Some("push --force")
    );
    // hidden inside a pipeline after an allowed op.
    assert_eq!(
        git_history_destructive_op("git add -A && git reset --hard"),
        Some("reset")
    );
    // with -C path global flag before the subcommand.
    assert_eq!(
        git_history_destructive_op("git -C /repo checkout main"),
        Some("checkout")
    );
}

#[test]
fn b85_no_false_positives() {
    // Non-git commands that merely CONTAIN the words must not be blocked.
    assert_eq!(git_history_destructive_op("cargo test reset_logic"), None);
    assert_eq!(git_history_destructive_op("ls checkout/"), None);
    assert_eq!(git_history_destructive_op("cat restore.md"), None);
    assert_eq!(git_history_destructive_op("./switch.sh"), None);
    assert_eq!(
        git_history_destructive_op("echo 'git reset is dangerous'"),
        None
    );
    // `git push` plain must NOT match the -f force heuristic via an unrelated -f.
    assert_eq!(
        git_history_destructive_op("grep -f patterns.txt file"),
        None
    );
    assert_eq!(
        git_history_destructive_op("git push && grep -f patterns.txt file"),
        None
    );
}

#[tokio::test]
async fn b85_execute_blocks_git_reset_end_to_end() {
    use serde_json::json;
    let tool = BashTool::new(10);
    let res = tool
        .execute(
            json!({"command": "git reset --hard HEAD~2"}),
            std::path::Path::new("/tmp"),
        )
        .await;
    assert!(res.is_error, "git reset --hard must be blocked: {res:?}");
    assert!(res.output.contains("discards or rewrites"));
}
