//! B7: Pluggable operations for tools.
//!
//! Tools can work with local filesystem (default) or remote hosts (SSH).
//! Each tool that does I/O accepts an `Arc<dyn ToolOps>` to abstract
//! the underlying transport.

use async_trait::async_trait;
use std::path::Path;

/// Result of executing a command.
pub struct ExecResult {
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
    pub exit_code: i32,
}

/// Pluggable operations for file access and command execution.
#[async_trait]
pub trait ToolOps: Send + Sync {
    /// Read file contents.
    async fn read_file(&self, path: &Path) -> std::io::Result<String>;

    /// Write contents to a file (creating parent dirs).
    async fn write_file(&self, path: &Path, contents: &str) -> std::io::Result<()>;

    /// Check if a file exists and is readable.
    async fn file_exists(&self, path: &Path) -> bool;

    /// Execute a command in a given working directory.
    async fn exec(
        &self,
        command: &str,
        cwd: &Path,
        timeout_secs: u64,
    ) -> std::io::Result<ExecResult>;

    /// Display name for logging (e.g. "local", "ssh:nova-1").
    fn label(&self) -> &str;
}

/// Default: local filesystem operations.
pub struct LocalOps;

#[async_trait]
impl ToolOps for LocalOps {
    async fn read_file(&self, path: &Path) -> std::io::Result<String> {
        tokio::fs::read_to_string(path).await
    }

    async fn write_file(&self, path: &Path, contents: &str) -> std::io::Result<()> {
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }
        tokio::fs::write(path, contents).await
    }

    async fn file_exists(&self, path: &Path) -> bool {
        tokio::fs::metadata(path).await.is_ok()
    }

    async fn exec(
        &self,
        command: &str,
        cwd: &Path,
        timeout_secs: u64,
    ) -> std::io::Result<ExecResult> {
        let output = tokio::time::timeout(
            std::time::Duration::from_secs(timeout_secs),
            tokio::process::Command::new("bash")
                .arg("-c")
                .arg(command)
                .current_dir(cwd)
                .output(),
        )
        .await
        .map_err(|_| std::io::Error::new(std::io::ErrorKind::TimedOut, "command timed out"))?;

        let output = output?;
        Ok(ExecResult {
            stdout: output.stdout,
            stderr: output.stderr,
            exit_code: output.status.code().unwrap_or(-1),
        })
    }

    fn label(&self) -> &str {
        "local"
    }
}

/// SSH-based remote operations.
pub struct SshOps {
    /// SSH destination (e.g. "root@144.31.151.82" or SSH config alias "nova-1")
    pub host: String,
    /// SSH key path (optional, uses default if None)
    pub key: Option<String>,
}

#[async_trait]
impl ToolOps for SshOps {
    async fn read_file(&self, path: &Path) -> std::io::Result<String> {
        let result = self
            .exec(&format!("cat {}", shell_escape(path)), Path::new("/"), 30)
            .await?;
        if result.exit_code != 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                String::from_utf8_lossy(&result.stderr).to_string(),
            ));
        }
        Ok(String::from_utf8_lossy(&result.stdout).to_string())
    }

    async fn write_file(&self, path: &Path, contents: &str) -> std::io::Result<()> {
        // Create parent dirs + write via heredoc
        let parent = path
            .parent()
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_default();
        let escaped_contents = contents.replace('\'', "'\\''");
        let cmd = format!(
            "mkdir -p {} && cat > {} << 'NAKED_EOF'\n{}\nNAKED_EOF",
            shell_escape(Path::new(&parent)),
            shell_escape(path),
            escaped_contents,
        );
        let result = self.exec(&cmd, Path::new("/"), 30).await?;
        if result.exit_code != 0 {
            return Err(std::io::Error::other(
                String::from_utf8_lossy(&result.stderr).to_string(),
            ));
        }
        Ok(())
    }

    async fn file_exists(&self, path: &Path) -> bool {
        self.exec(
            &format!("test -f {}", shell_escape(path)),
            Path::new("/"),
            10,
        )
        .await
        .map(|r| r.exit_code == 0)
        .unwrap_or(false)
    }

    async fn exec(
        &self,
        command: &str,
        cwd: &Path,
        timeout_secs: u64,
    ) -> std::io::Result<ExecResult> {
        let remote_cmd = format!("cd {} && {}", shell_escape(cwd), command);

        let mut ssh_cmd = tokio::process::Command::new("ssh");
        ssh_cmd
            .arg("-o")
            .arg("ConnectTimeout=10")
            .arg("-o")
            .arg("BatchMode=yes")
            .arg("-o")
            .arg("StrictHostKeyChecking=no");

        if let Some(ref key) = self.key {
            ssh_cmd.arg("-i").arg(key);
        }

        ssh_cmd.arg(&self.host).arg(&remote_cmd);

        let output = tokio::time::timeout(
            std::time::Duration::from_secs(timeout_secs),
            ssh_cmd.output(),
        )
        .await
        .map_err(|_| {
            std::io::Error::new(
                std::io::ErrorKind::TimedOut,
                format!("SSH command timed out after {timeout_secs}s"),
            )
        })?;

        let output = output?;
        Ok(ExecResult {
            stdout: output.stdout,
            stderr: output.stderr,
            exit_code: output.status.code().unwrap_or(-1),
        })
    }

    fn label(&self) -> &str {
        &self.host
    }
}

fn shell_escape(path: &Path) -> String {
    let s = path.to_string_lossy();
    format!("'{}'", s.replace('\'', "'\\''"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn local_ops_read_write() {
        let dir = tempfile::tempdir().unwrap();
        let ops = LocalOps;
        let path = dir.path().join("test.txt");

        ops.write_file(&path, "hello world").await.unwrap();
        assert!(ops.file_exists(&path).await);

        let content = ops.read_file(&path).await.unwrap();
        assert_eq!(content, "hello world");
    }

    #[tokio::test]
    async fn local_ops_exec() {
        let ops = LocalOps;
        let result = ops.exec("echo hello", Path::new("/tmp"), 5).await.unwrap();
        assert_eq!(result.exit_code, 0);
        assert_eq!(String::from_utf8_lossy(&result.stdout).trim(), "hello");
    }

    #[tokio::test]
    async fn local_ops_exec_timeout() {
        let ops = LocalOps;
        let result = ops.exec("sleep 10", Path::new("/tmp"), 1).await;
        assert!(result.is_err());
    }

    #[test]
    fn shell_escape_basic() {
        assert_eq!(shell_escape(Path::new("/tmp/file.txt")), "'/tmp/file.txt'");
        assert_eq!(
            shell_escape(Path::new("/tmp/it's a file")),
            "'/tmp/it'\\''s a file'"
        );
    }
}
