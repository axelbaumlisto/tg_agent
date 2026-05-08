//! `diagnostics` tool — workspace environment info.

use crate::types::{Permission, ToolResult, ToolSpec};
use std::path::Path;

pub struct DiagnosticsTool;

#[async_trait::async_trait]
impl crate::tool::Tool for DiagnosticsTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "diagnostics".into(),
            description: "Collect workspace diagnostics: OS, git, toolchain versions, disk space."
                .into(),
            parameters: serde_json::json!({"type": "object", "properties": {}}),
            permission: Permission::ReadOnly,
        }
    }

    async fn execute(&self, _input: serde_json::Value, cwd: &Path) -> ToolResult {
        let mut info = Vec::new();

        info.push(format!("workspace: {}", cwd.display()));
        info.push(format!(
            "os: {} {}",
            std::env::consts::OS,
            std::env::consts::ARCH
        ));

        for (cmd, args) in [
            ("git", vec!["--version"]),
            ("rustc", vec!["--version"]),
            ("cargo", vec!["--version"]),
            ("python3", vec!["--version"]),
            ("node", vec!["--version"]),
        ] {
            let ver = tokio::process::Command::new(cmd)
                .args(&args)
                .output()
                .await
                .ok()
                .filter(|o| o.status.success())
                .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
                .unwrap_or_else(|| "not found".into());
            info.push(format!("{cmd}: {ver}"));
        }

        // Git status:
        if let Ok(out) = tokio::process::Command::new("git")
            .args(["status", "--short"])
            .current_dir(cwd)
            .output()
            .await
        {
            let changes = String::from_utf8_lossy(&out.stdout).lines().count();
            info.push(format!("git changes: {changes} files"));
        }

        // Disk:
        if let Ok(out) = tokio::process::Command::new("df")
            .args(["-h", "."])
            .current_dir(cwd)
            .output()
            .await
            && let Some(line) = String::from_utf8_lossy(&out.stdout).lines().nth(1)
        {
            info.push(format!("disk: {}", line.trim()));
        }

        ToolResult::ok(info.join("\n"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool::Tool;

    #[tokio::test]
    async fn diagnostics_runs() {
        let tool = DiagnosticsTool;
        let r = tool.execute(serde_json::json!({}), Path::new(".")).await;
        assert!(!r.is_error);
        assert!(r.output.contains("os:"));
        assert!(r.output.contains("workspace:"));
    }
}
