use std::path::Path;
use std::time::Duration;

use async_trait::async_trait;
use serde::Deserialize;

use crate::types::{Permission, ToolResult, ToolSpec};

use super::Tool;

// ── Bash command classification ────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BashRisk {
    ReadOnly,
    Write,
    Destructive,
}

const READ_ONLY_COMMANDS: &[&str] = &[
    "ls",
    "cat",
    "head",
    "tail",
    "less",
    "more",
    "wc",
    "file",
    "stat",
    "du",
    "df",
    "pwd",
    "echo",
    "printf",
    "whoami",
    "hostname",
    "uname",
    "date",
    "env",
    "printenv",
    "which",
    "type",
    "whereis",
    "id",
    "groups",
    "find",
    "locate",
    "tree",
    "realpath",
    "readlink",
    "basename",
    "dirname",
    "rg",
    "grep",
    "egrep",
    "fgrep",
    "ag",
    "ack",
    "diff",
    "cmp",
    "md5sum",
    "sha256sum",
    "sha1sum",
    "xxd",
    "git log",
    "git status",
    "git diff",
    "git show",
    "git branch",
    "git tag",
    "git remote",
    "git stash list",
    "git rev-parse",
    "git describe",
    "git shortlog",
    "git blame",
    "git ls-files",
    "git ls-tree",
    "git cat-file",
    "cargo check",
    "cargo clippy",
    "cargo test",
    "cargo doc",
    "cargo metadata",
    "cargo tree",
    "cargo fmt -- --check",
    "rustc --version",
    "rustup show",
    "rustup toolchain list",
    "python --version",
    "python3 --version",
    "pip list",
    "pip show",
    "pip freeze",
    "node --version",
    "npm list",
    "npm outdated",
    "npm info",
    "jq",
    "yq",
    "bat",
    "hexdump",
    "docker ps",
    "docker images",
    "docker inspect",
    "docker logs",
    "ps",
    "top",
    "htop",
    "free",
    "uptime",
    "lsof",
    "ss",
    "netstat",
    "curl -s",
    "curl -I",
    "wget -q -O -",
];

const DESTRUCTIVE_PATTERNS: &[&str] = &[
    "rm -rf /",
    "rm -rf /*",
    "mkfs",
    "dd if=",
    "> /dev/sd",
    ":(){ :|:& };:",
    "chmod -R 777 /",
    "chown -R",
    "shutdown",
    "reboot",
    "init 0",
    "kill -9 1",
    "pkill -9 init",
    "format c:",
];

pub fn classify_bash(command: &str) -> BashRisk {
    let trimmed = command.trim();
    let lower = trimmed.to_ascii_lowercase();

    for pattern in DESTRUCTIVE_PATTERNS {
        if lower.contains(pattern) {
            return BashRisk::Destructive;
        }
    }

    if lower.contains('>') || lower.contains(">>") || lower.contains("tee ") {
        return BashRisk::Write;
    }

    let segments: Vec<&str> = trimmed.split(&['|', ';', '&'][..]).collect();
    let all_safe = segments.iter().all(|seg| {
        let seg = seg.trim();
        READ_ONLY_COMMANDS
            .iter()
            .any(|ro| seg == *ro || seg.starts_with(&format!("{ro} ")))
    });

    if all_safe {
        BashRisk::ReadOnly
    } else {
        BashRisk::Write
    }
}

pub struct BashTool {
    timeout: Duration,
}

impl BashTool {
    pub fn new(timeout_secs: u64) -> Self {
        Self {
            timeout: Duration::from_secs(timeout_secs),
        }
    }
}

#[derive(Deserialize)]
struct BashInput {
    command: String,
    #[serde(default = "default_timeout")]
    timeout: Option<u64>,
}

fn default_timeout() -> Option<u64> {
    None
}

#[async_trait]
impl Tool for BashTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "bash".into(),
            description: "Execute a bash command. Use for running shell commands, scripts, git operations, and system tasks.".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The bash command to execute"
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Optional timeout in seconds (default: tool_timeout_secs from config)"
                    }
                },
                "required": ["command"]
            }),
            permission: Permission::Dangerous,
        }
    }

    fn effective_permission(&self, input: &serde_json::Value, _cwd: &Path) -> Permission {
        let cmd = input.get("command").and_then(|v| v.as_str()).unwrap_or("");
        match classify_bash(cmd) {
            BashRisk::ReadOnly => Permission::ReadOnly,
            BashRisk::Write => Permission::WorkspaceWrite,
            BashRisk::Destructive => Permission::Dangerous,
        }
    }

    async fn execute(&self, input: serde_json::Value, cwd: &Path) -> ToolResult {
        let input: BashInput = match serde_json::from_value(input) {
            Ok(v) => v,
            Err(e) => {
                return ToolResult {
                    output: format!("Invalid input: {e}"),
                    is_error: true,
                };
            }
        };

        let timeout = input
            .timeout
            .map(Duration::from_secs)
            .unwrap_or(self.timeout);

        let result = tokio::time::timeout(timeout, async {
            tokio::process::Command::new("bash")
                .arg("-c")
                .arg(&input.command)
                .current_dir(cwd)
                .output()
                .await
        })
        .await;

        match result {
            Ok(Ok(output)) => {
                const MAX_STREAM: usize = 16_384;
                let stdout = truncate_output(&String::from_utf8_lossy(&output.stdout), MAX_STREAM);
                let stderr = truncate_output(&String::from_utf8_lossy(&output.stderr), MAX_STREAM);
                let combined = if stderr.is_empty() {
                    stdout
                } else if stdout.is_empty() {
                    stderr
                } else {
                    format!("{stdout}\n--- stderr ---\n{stderr}")
                };
                ToolResult {
                    output: combined,
                    is_error: !output.status.success(),
                }
            }
            Ok(Err(e)) => ToolResult {
                output: format!("Failed to execute: {e}"),
                is_error: true,
            },
            Err(_) => ToolResult {
                output: format!("Command timed out after {}s", timeout.as_secs()),
                is_error: true,
            },
        }
    }
}

fn truncate_output(s: &str, max_bytes: usize) -> String {
    if s.len() <= max_bytes {
        return s.to_string();
    }
    let mut end = max_bytes;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    let cut = s.len() - end;
    format!(
        "{}\n\n[output truncated — exceeded {max_bytes} bytes, {cut} bytes cut]",
        &s[..end]
    )
}

#[cfg(test)]
mod tests {
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
}
