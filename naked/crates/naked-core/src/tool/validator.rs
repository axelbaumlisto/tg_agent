//! Post-edit validation — run language-specific checks after file edits.
//!
//! The [`PostEditValidator`] trait is frontend-agnostic. Implementations
//! detect the language from the file extension and run a quick check
//! (cargo check, python -m py_compile, npx tsc, etc.). Results are
//! appended to the tool output so the model sees errors immediately.

use std::path::Path;

/// Result of a post-edit validation run.
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Human-readable diagnostic lines (compiler errors, warnings).
    pub diagnostics: String,
    /// Number of errors found.
    pub error_count: usize,
    /// Number of warnings found.
    pub warning_count: usize,
}

impl ValidationResult {
    /// Format as a block to append to tool output.
    pub fn to_tool_suffix(&self) -> String {
        if self.error_count == 0 && self.warning_count == 0 {
            return String::new();
        }
        let header = if self.error_count > 0 {
            format!(
                "\n\n⚠️ Post-edit check: {} error(s), {} warning(s)",
                self.error_count, self.warning_count
            )
        } else {
            format!("\n\n💡 Post-edit check: {} warning(s)", self.warning_count)
        };
        let diag = if self.diagnostics.len() > 2000 {
            let mut end = 2000;
            while end > 0 && !self.diagnostics.is_char_boundary(end) {
                end -= 1;
            }
            format!("{}…\n[truncated]", &self.diagnostics[..end])
        } else {
            self.diagnostics.clone()
        };
        format!("{header}\n```\n{diag}\n```")
    }
}

/// Detect which validator to run based on file extension and workspace.
pub fn detect_check_command(file: &Path, cwd: &Path) -> Option<CheckCommand> {
    let ext = file.extension()?.to_str()?;
    match ext {
        "rs" => {
            // Only run cargo check if Cargo.toml exists in cwd or ancestors
            let mut dir = cwd.to_path_buf();
            loop {
                if dir.join("Cargo.toml").exists() {
                    return Some(CheckCommand {
                        program: "cargo".into(),
                        args: vec!["check".into(), "--message-format=short".into()],
                        cwd: dir,
                        timeout_secs: 60,
                    });
                }
                if !dir.pop() {
                    break;
                }
            }
            None
        }
        "py" => Some(CheckCommand {
            program: "python3".into(),
            args: vec![
                "-m".into(),
                "py_compile".into(),
                file.to_string_lossy().into(),
            ],
            cwd: cwd.to_path_buf(),
            timeout_secs: 10,
        }),
        "ts" | "tsx" => {
            if cwd.join("tsconfig.json").exists() {
                Some(CheckCommand {
                    program: "npx".into(),
                    args: vec!["tsc".into(), "--noEmit".into()],
                    cwd: cwd.to_path_buf(),
                    timeout_secs: 30,
                })
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Command to run for validation.
#[derive(Debug, Clone)]
pub struct CheckCommand {
    pub program: String,
    pub args: Vec<String>,
    pub cwd: std::path::PathBuf,
    pub timeout_secs: u64,
}

/// Run a post-edit validation check. Returns None if no validator
/// applies or the check passes cleanly.
pub async fn run_post_edit_check(file: &Path, cwd: &Path) -> Option<ValidationResult> {
    let cmd = detect_check_command(file, cwd)?;

    let output = match tokio::time::timeout(
        std::time::Duration::from_secs(cmd.timeout_secs),
        tokio::process::Command::new(&cmd.program)
            .args(&cmd.args)
            .current_dir(&cmd.cwd)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .output(),
    )
    .await
    {
        Ok(Ok(out)) => out,
        Ok(Err(_)) | Err(_) => return None, // command not found or timeout
    };

    if output.status.success() {
        return None; // clean build
    }

    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let combined = if stderr.is_empty() {
        stdout.to_string()
    } else {
        format!("{stderr}\n{stdout}")
    };

    let error_count = combined.matches("error").count().min(99);
    let warning_count = combined.matches("warning").count().min(99);

    Some(ValidationResult {
        diagnostics: combined,
        error_count: error_count.max(if output.status.success() { 0 } else { 1 }),
        warning_count,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn detect_rust_with_cargo_toml() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("Cargo.toml"), "[package]").unwrap();
        let file = tmp.path().join("src/main.rs");
        let cmd = detect_check_command(&file, tmp.path());
        assert!(cmd.is_some());
        let cmd = cmd.unwrap();
        assert_eq!(cmd.program, "cargo");
        assert!(cmd.args.contains(&"check".to_string()));
    }

    #[test]
    fn detect_rust_without_cargo_toml() {
        let tmp = tempfile::tempdir().unwrap();
        let file = tmp.path().join("foo.rs");
        assert!(detect_check_command(&file, tmp.path()).is_none());
    }

    #[test]
    fn detect_python() {
        let file = PathBuf::from("/tmp/test.py");
        let cmd = detect_check_command(&file, Path::new("/tmp"));
        assert!(cmd.is_some());
        assert_eq!(cmd.unwrap().program, "python3");
    }

    #[test]
    fn detect_typescript_with_tsconfig() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("tsconfig.json"), "{}").unwrap();
        let file = tmp.path().join("app.tsx");
        let cmd = detect_check_command(&file, tmp.path());
        assert!(cmd.is_some());
        assert_eq!(cmd.unwrap().program, "npx");
    }

    #[test]
    fn detect_typescript_without_tsconfig() {
        let tmp = tempfile::tempdir().unwrap();
        let file = tmp.path().join("app.ts");
        assert!(detect_check_command(&file, tmp.path()).is_none());
    }

    #[test]
    fn detect_unknown_extension() {
        let file = PathBuf::from("/tmp/readme.md");
        assert!(detect_check_command(&file, Path::new("/tmp")).is_none());
    }

    #[test]
    fn validation_result_empty_on_clean() {
        let r = ValidationResult {
            diagnostics: String::new(),
            error_count: 0,
            warning_count: 0,
        };
        assert!(r.to_tool_suffix().is_empty());
    }

    #[test]
    fn validation_result_formats_errors() {
        let r = ValidationResult {
            diagnostics: "error[E0308]: mismatched types".into(),
            error_count: 1,
            warning_count: 0,
        };
        let s = r.to_tool_suffix();
        assert!(s.contains("⚠️"));
        assert!(s.contains("1 error"));
        assert!(s.contains("E0308"));
    }

    #[test]
    fn validation_result_truncates_long_output() {
        let r = ValidationResult {
            diagnostics: "x".repeat(3000),
            error_count: 1,
            warning_count: 0,
        };
        let s = r.to_tool_suffix();
        assert!(s.contains("[truncated]"));
        assert!(s.len() < 2500);
    }

    #[tokio::test]
    async fn run_check_on_valid_python() {
        let tmp = tempfile::tempdir().unwrap();
        let py = tmp.path().join("ok.py");
        std::fs::write(&py, "x = 1\n").unwrap();
        let result = run_post_edit_check(&py, tmp.path()).await;
        assert!(result.is_none(), "valid python should pass");
    }

    #[tokio::test]
    async fn run_check_on_invalid_python() {
        let tmp = tempfile::tempdir().unwrap();
        let py = tmp.path().join("bad.py");
        std::fs::write(&py, "def f(\n").unwrap();
        let result = run_post_edit_check(&py, tmp.path()).await;
        assert!(result.is_some(), "invalid python should fail");
        let r = result.unwrap();
        assert!(r.error_count > 0);
    }

    #[tokio::test]
    async fn run_check_on_nonexistent_validator() {
        let tmp = tempfile::tempdir().unwrap();
        let md = tmp.path().join("readme.md");
        std::fs::write(&md, "# hello").unwrap();
        let result = run_post_edit_check(&md, tmp.path()).await;
        assert!(result.is_none(), "no validator for .md");
    }
}
