//! Unified diff generation for tool output.
//!
//! After write_file / edit_file, show the model exactly what changed
//! via a unified diff. Concise, standard format the model understands.

/// Generate a unified diff between old and new content.
/// Returns empty string if contents are identical.
pub fn unified_diff(path: &str, old: &str, new: &str) -> String {
    if old == new {
        return String::new();
    }

    let diff = similar::TextDiff::from_lines(old, new);
    let mut output = String::new();

    // Count changes:
    let mut adds = 0usize;
    let mut dels = 0usize;
    for change in diff.iter_all_changes() {
        match change.tag() {
            similar::ChangeTag::Insert => adds += 1,
            similar::ChangeTag::Delete => dels += 1,
            _ => {}
        }
    }

    if adds == 0 && dels == 0 {
        return String::new();
    }

    output.push_str(&format!("--- a/{path}\n+++ b/{path}\n"));

    for hunk in diff.unified_diff().context_radius(3).iter_hunks() {
        output.push_str(&format!("{hunk}"));
    }

    // Summary line:
    output.push_str(&format!("\n({adds} insertions, {dels} deletions)"));
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_returns_empty() {
        assert!(unified_diff("f.rs", "hello\n", "hello\n").is_empty());
    }

    #[test]
    fn simple_add() {
        let d = unified_diff("f.rs", "a\n", "a\nb\n");
        assert!(d.contains("+b"));
        assert!(d.contains("1 insertions"));
    }

    #[test]
    fn simple_delete() {
        let d = unified_diff("f.rs", "a\nb\n", "a\n");
        assert!(d.contains("-b"));
        assert!(d.contains("1 deletions"));
    }

    #[test]
    fn modification() {
        let d = unified_diff("f.rs", "old line\n", "new line\n");
        assert!(d.contains("-old line"));
        assert!(d.contains("+new line"));
    }

    #[test]
    fn has_path_header() {
        let d = unified_diff("src/main.rs", "a\n", "b\n");
        assert!(d.contains("--- a/src/main.rs"));
        assert!(d.contains("+++ b/src/main.rs"));
    }

    #[test]
    fn new_file_all_additions() {
        let d = unified_diff("new.rs", "", "fn main() {}\n");
        assert!(d.contains("+fn main()"));
    }
}
