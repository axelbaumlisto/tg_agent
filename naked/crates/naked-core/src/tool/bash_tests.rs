use super::*;
use crate::tool::Tool;

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
