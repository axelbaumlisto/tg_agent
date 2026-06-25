use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use serde::Deserialize;

use crate::types::{Permission, ToolResult, ToolSpec};

use super::Tool;

pub(crate) const MAX_COLLECTED: usize = 512 * 1024; // 512KB max

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

/// Block commands that would restart/kill the bot's own process.
fn is_self_destructive(cmd: &str) -> bool {
    let lower = cmd.to_lowercase();
    let patterns = [
        "systemctl restart naked",
        "systemctl stop naked",
        "systemctl kill naked",
        "systemctl --user restart naked",
        "systemctl --user stop naked",
        "systemctl --user kill naked",
        "kill -9",
        "kill -KILL",
        "pkill naked",
        "killall naked",
    ];
    patterns.iter().any(|p| lower.contains(&p.to_lowercase()))
}

/// B85: Block git operations that DISCARD or REWRITE existing work in a shared
/// repo. Operator policy (2026-06-22): a coding agent in someone else's repo may
/// `git add` / `commit` / `push` (these only ADD to history, nothing is lost),
/// but MUST NOT `reset` / `checkout` / `switch` / `restore` / `clean` /
/// `push --force` / `rebase` (these throw away or move other users' work).
///
/// Returns the offending subcommand name when the command contains a blocked
/// git operation anywhere in a `;`/`|`/`&&` pipeline. Matching is on the FIRST
/// two tokens after a `git` token so `cargo test reset` or a file literally
/// named "checkout" is not a false positive.
pub(crate) fn git_history_destructive_op(cmd: &str) -> Option<&'static str> {
    // Subcommands that can lose/move existing work. `clean` removes untracked
    // files; `restore` discards working-tree changes; `reset`/`checkout`/`switch`
    // move HEAD or overwrite files; `rebase` rewrites history.
    const BLOCKED_GIT_SUBCMDS: &[&str] = &[
        "reset", "checkout", "switch", "restore", "clean", "rebase", "revert",
    ];
    for segment in cmd.split(&['|', ';', '&', '\n'][..]) {
        let tokens: Vec<&str> = segment.split_whitespace().collect();
        let mut i = 0;
        while i < tokens.len() {
            // Find a `git` token (allows env prefixes like `GIT_DIR=… git reset`).
            if tokens[i] == "git" {
                // Skip global flags like `-C path`, `--no-pager` to reach the subcommand.
                let mut j = i + 1;
                while j < tokens.len()
                    && (tokens[j].starts_with('-')
                        || (j >= 1 && (tokens[j - 1] == "-C" || tokens[j - 1] == "--git-dir")))
                {
                    j += 1;
                }
                if let Some(sub) = tokens.get(j) {
                    let sub_lower = sub.to_ascii_lowercase();
                    if let Some(found) = BLOCKED_GIT_SUBCMDS.iter().find(|b| **b == sub_lower) {
                        return Some(found);
                    }
                    // Force-push: `git push --force` / `-f` / `--force-with-lease`
                    // rewrites the remote ref. Plain `git push` is allowed
                    // (append-only). Scope force detection to this same shell
                    // segment so `git push && grep -f patterns.txt file` stays allowed.
                    if sub_lower == "push"
                        && tokens[j + 1..].iter().any(|token| {
                            let token = token.to_ascii_lowercase();
                            matches!(token.as_str(), "-f" | "--force" | "--force-with-lease")
                        })
                    {
                        return Some("push --force");
                    }
                }
            }
            i += 1;
        }
    }
    None
}

pub(crate) fn git_history_block_reason(op: &str) -> String {
    format!(
        "Blocked: `git {op}` discards or rewrites existing work in a shared repo. You may \
         `git add`/`commit`/`push` (append-only), but not reset/checkout/switch/restore/clean/\
         rebase/revert/force-push. If the operator explicitly wants this, they must run it in \
         their own terminal."
    )
}

pub struct BashTool {
    timeout: Duration,
    /// B7: Optional remote ops. When set, commands run via SSH.
    ops: Option<std::sync::Arc<dyn super::ops::ToolOps>>,
    /// Optional local persistent bash session. Never used for remote ops.
    persistent: Option<(Arc<super::persistent_bash::PersistentBashManager>, String)>,
}

impl BashTool {
    pub fn new(timeout_secs: u64) -> Self {
        Self {
            timeout: Duration::from_secs(timeout_secs),
            ops: None,
            persistent: None,
        }
    }

    /// Create with remote ops (SSH execution).
    pub fn with_ops(timeout_secs: u64, ops: std::sync::Arc<dyn super::ops::ToolOps>) -> Self {
        Self {
            timeout: Duration::from_secs(timeout_secs),
            ops: Some(ops),
            persistent: None,
        }
    }

    pub fn with_persistent(
        mut self,
        manager: Arc<super::persistent_bash::PersistentBashManager>,
        session_id: String,
    ) -> Self {
        self.persistent = Some((manager, session_id));
        self
    }

    /// Parse + validate input. Shared by execute() and execute_with_progress().
    fn parse_input(&self, input: serde_json::Value) -> Result<(BashInput, Duration), ToolResult> {
        let input: BashInput = super::parse_tool_input(input)?;
        if is_self_destructive(&input.command) {
            return Err(ToolResult::err(
                "Blocked: this command would restart/kill the bot process. \
                 Use the operator's terminal instead.",
            ));
        }
        if let Some(op) = git_history_destructive_op(&input.command) {
            // execute-time backstop; intentionally NOT counted — production loop
            // attempts are counted at L1 policy.
            return Err(ToolResult::err(git_history_block_reason(op)));
        }
        let timeout = input
            .timeout
            .map(Duration::from_secs)
            .unwrap_or(self.timeout);
        Ok((input, timeout))
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
        let (input, timeout) = match self.parse_input(input) {
            Ok(v) => v,
            Err(e) => return e,
        };

        // B7: Dispatch to remote ops or local execution.
        enum ExecOutcome {
            Ok {
                stdout: Vec<u8>,
                stderr: Vec<u8>,
                success: bool,
            },
            ExecErr(String),
            Timeout,
        }

        let outcome = if let Some(ref ops) = self.ops {
            match ops.exec(&input.command, cwd, timeout.as_secs()).await {
                Ok(r) => ExecOutcome::Ok {
                    stdout: r.stdout,
                    stderr: r.stderr,
                    success: r.exit_code == 0,
                },
                Err(e) if e.kind() == std::io::ErrorKind::TimedOut => ExecOutcome::Timeout,
                Err(e) => ExecOutcome::ExecErr(e.to_string()),
            }
        } else if let Some((manager, session_id)) = &self.persistent {
            match manager.get_or_create(session_id, timeout).await {
                Ok(shell) => match shell.run(&input.command, cwd, timeout).await {
                    Ok(output) => ExecOutcome::Ok {
                        stdout: output.stdout,
                        stderr: output.stderr,
                        success: output.exit_code == 0,
                    },
                    Err(super::persistent_bash::PersistentBashError::Timeout) => {
                        ExecOutcome::Timeout
                    }
                    Err(e) => ExecOutcome::ExecErr(e.to_string()),
                },
                Err(super::persistent_bash::PersistentBashError::Timeout) => ExecOutcome::Timeout,
                Err(e) => ExecOutcome::ExecErr(e.to_string()),
            }
        } else {
            super::persistent_bash::PersistentBashOutcome::Disabled.bump();
            match tokio::time::timeout(timeout, async {
                tokio::process::Command::new("bash")
                    .arg("-c")
                    .arg(&input.command)
                    .current_dir(cwd)
                    .output()
                    .await
            })
            .await
            {
                Ok(Ok(output)) => ExecOutcome::Ok {
                    stdout: output.stdout,
                    stderr: output.stderr,
                    success: output.status.success(),
                },
                Ok(Err(e)) => ExecOutcome::ExecErr(e.to_string()),
                Err(_) => ExecOutcome::Timeout,
            }
        };

        match outcome {
            ExecOutcome::Ok {
                stdout,
                stderr,
                success,
            } => format_bash_output(&stdout, &stderr, success, &timeout),
            ExecOutcome::ExecErr(e) => ToolResult::err(format!("Failed to execute: {e}")),
            ExecOutcome::Timeout => {
                ToolResult::err(format!("Command timed out after {}s", timeout.as_secs()))
            }
        }
    }

    async fn execute_with_progress(
        &self,
        input: serde_json::Value,
        cwd: &Path,
        progress: tokio::sync::mpsc::Sender<crate::types::AgentEvent>,
    ) -> ToolResult {
        // For remote ops and persistent local shells, fall back to non-streaming execute.
        if self.ops.is_some() || self.persistent.is_some() {
            return self.execute(input, cwd).await;
        }

        let (input, timeout_dur) = match self.parse_input(input) {
            Ok(v) => v,
            Err(e) => return e,
        };

        use tokio::io::AsyncBufReadExt;
        use tokio::process::Command;

        let mut child = match Command::new("bash")
            .arg("-c")
            .arg(&input.command)
            .current_dir(cwd)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
        {
            Ok(c) => c,
            Err(e) => {
                return ToolResult::err(format!("Failed to spawn bash: {e}"));
            }
        };

        let stdout = child.stdout.take().expect("stdout piped");
        let stderr = child.stderr.take().expect("stderr piped");

        // Merge stdout + stderr into a single stream, keep last N lines as tail.
        let (line_tx, mut line_rx) = tokio::sync::mpsc::channel::<String>(256);

        // Reader tasks:
        let tx1 = line_tx.clone();
        let stdout_task = tokio::spawn(async move {
            let mut reader = tokio::io::BufReader::new(stdout).lines();
            while let Ok(Some(line)) = reader.next_line().await {
                if tx1.send(line).await.is_err() {
                    break;
                }
            }
        });
        let tx2 = line_tx;
        let stderr_task = tokio::spawn(async move {
            let mut reader = tokio::io::BufReader::new(stderr).lines();
            while let Ok(Some(line)) = reader.next_line().await {
                if tx2.send(line).await.is_err() {
                    break;
                }
            }
        });

        // Collector: accumulate all output + send ToolOutput every 2s.
        const TAIL_LINES: usize = 5;
        const PROGRESS_INTERVAL: Duration = Duration::from_secs(2);
        let mut all_lines: Vec<String> = Vec::new();
        let mut total_bytes: usize = 0;
        let mut last_progress = std::time::Instant::now();
        let call_id = String::new(); // We don't have call_id here, use empty.

        let deadline = tokio::time::Instant::now() + timeout_dur;

        loop {
            tokio::select! {
                line = line_rx.recv() => {
                    match line {
                        Some(l) => {
                            total_bytes += l.len() + 1;
                            if total_bytes < MAX_COLLECTED {
                                all_lines.push(l);
                            }
                            // Send progress every PROGRESS_INTERVAL:
                            if last_progress.elapsed() >= PROGRESS_INTERVAL {
                                let tail: String = all_lines
                                    .iter()
                                    .rev()
                                    .take(TAIL_LINES)
                                    .collect::<Vec<_>>()
                                    .into_iter()
                                    .rev()
                                    .cloned()
                                    .collect::<Vec<_>>()
                                    .join("\n");
                                let _ = progress
                                    .send(crate::types::AgentEvent::ToolOutput {
                                        call_id: call_id.clone(),
                                        chunk: tail,
                                    })
                                    .await;
                                last_progress = std::time::Instant::now();
                            }
                        }
                        None => break, // Both readers done.
                    }
                }
                _ = tokio::time::sleep_until(deadline) => {
                    let _ = child.kill().await;
                    stdout_task.abort();
                    stderr_task.abort();
                    return ToolResult::err(format!("Command timed out after {}s", timeout_dur.as_secs()));
                }
            }
        }

        // Wait for child to exit.
        let status = child.wait().await;
        stdout_task.abort();
        stderr_task.abort();

        let success = status.map(|s| s.success()).unwrap_or(false);
        let raw = all_lines.join("\n");
        const MAX_STREAM: usize = 16_384;
        let output = truncate_output(&raw, MAX_STREAM);

        let total_lines = all_lines.len();
        let mut combined = output;
        if total_bytes > MAX_STREAM || total_lines > 500 {
            combined.push_str(&format!(
                "\n[{total_lines} lines, {total_bytes} bytes total\
                 {}]",
                if total_bytes > MAX_STREAM {
                    ", truncated"
                } else {
                    ""
                }
            ));
        }

        if success {
            ToolResult::ok(combined)
        } else {
            ToolResult::err(combined)
        }
    }
}

fn truncate_output(s: &str, max_bytes: usize) -> String {
    if s.len() <= max_bytes {
        return s.to_string();
    }
    let truncated = crate::util::head_truncate(s, max_bytes);
    let cut = s.len() - truncated.len();
    format!("{truncated}\n\n[output truncated — exceeded {max_bytes} bytes, {cut} bytes cut]",)
}

#[cfg(test)]
#[path = "bash_tests.rs"]
mod tests;

#[tokio::test]
async fn execute_large_output_saves_to_file() {
    let tool = BashTool::new(10);
    // Generate output larger than MAX_STREAM (16KB)
    let input = serde_json::json!({
        "command": "seq 1 2000 | while read n; do echo \"line_$n padding_data_to_make_it_bigger_0123456789\"; done"
    });
    let result = tool.execute(input, std::path::Path::new("/tmp")).await;
    assert!(!result.is_error, "command should succeed");
    // Output should mention truncation and temp file
    assert!(
        result.output.contains("[truncated:"),
        "should contain truncation note, got: {}",
        &result.output[result.output.len().saturating_sub(200)..]
    );
    assert!(
        result.output.contains("/tmp/naked_bash_"),
        "should contain temp file path, got: {}",
        &result.output[result.output.len().saturating_sub(200)..]
    );
    // Temp file should exist
    let path_start = result.output.find("/tmp/naked_bash_").unwrap();
    let path_end = result.output[path_start..].find(']').unwrap() + path_start;
    let path = &result.output[path_start..path_end];
    assert!(
        std::path::Path::new(path).exists(),
        "temp file should exist: {path}"
    );
    // Clean up
    let _ = std::fs::remove_file(path);
}

/// Format bash execution output: combine stdout/stderr, truncate, save overflow.
fn format_bash_output(
    out_bytes: &[u8],
    err_bytes: &[u8],
    success: bool,
    timeout: &std::time::Duration,
) -> ToolResult {
    const MAX_STREAM: usize = 16_384;
    let raw_stdout = String::from_utf8_lossy(out_bytes);
    let raw_stderr = String::from_utf8_lossy(err_bytes);
    let total_bytes = raw_stdout.len() + raw_stderr.len();
    let total_lines = raw_stdout.lines().count() + raw_stderr.lines().count();
    let is_truncated = raw_stdout.len() > MAX_STREAM || raw_stderr.len() > MAX_STREAM;

    let stdout = truncate_output(&raw_stdout, MAX_STREAM);
    let stderr = truncate_output(&raw_stderr, MAX_STREAM);
    let mut combined = if stderr.is_empty() {
        stdout
    } else if stdout.is_empty() {
        stderr
    } else {
        format!("{stdout}\n--- stderr ---\n{stderr}")
    };

    if is_truncated {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        combined.hash(&mut hasher);
        let hash = format!("{:x}", hasher.finish());
        let hash = &hash[..8]; // REGISTRY-WAIVE: B48 — hex hash is ASCII
        let path = format!("/tmp/naked_bash_{hash}.log");
        let full = if raw_stderr.is_empty() {
            raw_stdout.to_string()
        } else {
            format!("{raw_stdout}\n--- stderr ---\n{raw_stderr}")
        };
        let _ = std::fs::write(&path, &full);
        combined.push_str(&format!(
            "\n\n[truncated: {total_lines} lines, {total_bytes} bytes total. \
             Full output: {path}]"
        ));
    }

    let _ = timeout; // used by caller for error message
    if success {
        ToolResult::ok(combined)
    } else {
        ToolResult::err(combined)
    }
}
