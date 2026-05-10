//! Typed diagnostic + render-for-model.

use std::path::Path;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Severity {
    /// LSP severity 1.
    Error = 1,
    /// LSP severity 2.
    Warning = 2,
    /// LSP severities 3-4 collapsed.
    Info = 3,
}

impl Severity {
    #[must_use]
    pub fn from_lsp_code(code: u8) -> Self {
        match code {
            1 => Self::Error,
            2 => Self::Warning,
            _ => Self::Info,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Diagnostic {
    pub severity: Severity,
    pub line: u32,
    pub col: u32,
    pub message: String,
    pub source: Option<String>,
}

impl Diagnostic {
    pub fn one_line(&self, path_for_label: &Path) -> String {
        let label = match self.severity {
            Severity::Error => "ERROR",
            Severity::Warning => "WARN",
            Severity::Info => "INFO",
        };
        let src = self
            .source
            .as_deref()
            .map(|s| format!("[{s}] "))
            .unwrap_or_default();
        let msg = self.message.replace('\n', " ");
        // Truncate very long messages.
        let trimmed = if msg.chars().count() > 240 {
            let mut t: String = msg.chars().take(237).collect();
            t.push('…');
            t
        } else {
            msg
        };
        format!(
            "{label} {}:{}:{}: {src}{trimmed}",
            path_for_label.display(),
            self.line + 1,
            self.col + 1,
        )
    }
}

/// Build the synthetic message body that gets injected into the
/// next request. `Vec::is_empty()` callers should NOT call this.
#[must_use]
pub fn render_for_model(path: &Path, diags: &[Diagnostic]) -> String {
    if diags.is_empty() {
        return String::new();
    }
    let mut out = String::with_capacity(diags.len() * 80 + 64);
    out.push_str("[lsp] post-edit diagnostics:\n");
    for d in diags {
        out.push_str(&d.one_line(path));
        out.push('\n');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn d(sev: Severity, line: u32, col: u32, msg: &str) -> Diagnostic {
        Diagnostic {
            severity: sev,
            line,
            col,
            message: msg.into(),
            source: None,
        }
    }

    #[test]
    fn render_groups_by_severity() {
        let path = PathBuf::from("src/foo.rs");
        let diags = vec![
            d(Severity::Error, 41, 4, "type mismatch"),
            d(Severity::Warning, 12, 0, "unused import"),
        ];
        let rendered = render_for_model(&path, &diags);
        assert!(rendered.contains("[lsp] post-edit diagnostics"));
        assert!(rendered.contains("ERROR src/foo.rs:42:5"));
        assert!(rendered.contains("WARN src/foo.rs:13:1"));
    }

    #[test]
    fn one_line_includes_source_label() {
        let mut diag = d(Severity::Error, 0, 0, "X");
        diag.source = Some("rustc".into());
        let line = diag.one_line(Path::new("foo.rs"));
        assert!(line.contains("[rustc]"), "line: {line}");
    }

    #[test]
    fn message_truncated_at_240() {
        let huge = "x".repeat(500);
        let diag = d(Severity::Error, 0, 0, &huge);
        let line = diag.one_line(Path::new("a.rs"));
        // 'ERROR a.rs:1:1: ' prefix + 237 chars + '…' = ~256 bytes.
        // We just verify the truncation happened.
        assert!(line.ends_with('…'), "line: {line}");
    }

    #[test]
    fn empty_diagnostics_returns_empty_string() {
        assert_eq!(render_for_model(Path::new("foo"), &[]), "");
    }

    #[test]
    fn severity_from_lsp_code() {
        assert_eq!(Severity::from_lsp_code(1), Severity::Error);
        assert_eq!(Severity::from_lsp_code(2), Severity::Warning);
        assert_eq!(Severity::from_lsp_code(3), Severity::Info);
        assert_eq!(Severity::from_lsp_code(99), Severity::Info);
    }
}
