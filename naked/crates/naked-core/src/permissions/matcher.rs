//! Glob matching for permission patterns.
//!
//! Supports:
//!   * `*`  — match any sequence of chars except `/`
//!   * `**` — match any sequence including `/`
//!   * `?`  — match exactly one char
//!   * `~/` and `$HOME/` — expand to `$HOME` at match time
//!   * `*` is the explicit match-anything wildcard; an empty pattern matches nothing
//!   * Path-like patterns are canonicalised before matching. Ruleset evaluation uses
//!     a tri-state matcher so canonicalisation failures force `Ask` instead of being
//!     collapsed into a non-match that a later `Allow` can override.

use std::{
    io::ErrorKind,
    path::{Path, PathBuf},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MatchResult {
    Yes,
    No,
    Indeterminate,
}

#[must_use]
pub fn matches(pattern: &str, target: &str) -> bool {
    matches!(match_rule(pattern, target), MatchResult::Yes)
}

#[must_use]
pub(crate) fn match_rule(pattern: &str, target: &str) -> MatchResult {
    if pattern.is_empty() {
        return MatchResult::No;
    }
    if pattern == "*" {
        return MatchResult::Yes;
    }

    let pattern = expand_home(pattern);
    let target = expand_home(target);
    if is_path_like_pattern(&pattern) {
        let pattern = match canonicalize_path_pattern(&pattern) {
            Ok(pattern) => pattern,
            Err(()) => return MatchResult::Indeterminate,
        };
        let target = match canonicalize_nearest(&target) {
            Ok(target) => target,
            Err(()) => return MatchResult::Indeterminate,
        };
        return if match_glob(&pattern, &target) {
            MatchResult::Yes
        } else {
            MatchResult::No
        };
    }

    if match_glob(&pattern, &target) {
        MatchResult::Yes
    } else {
        MatchResult::No
    }
}

fn is_path_like_pattern(pattern: &str) -> bool {
    pattern.contains('/') || pattern.starts_with('~') || pattern.starts_with("$HOME")
}

fn expand_home(s: &str) -> String {
    let home = std::env::var("HOME").unwrap_or_default();
    if home.is_empty() {
        return s.to_string();
    }
    let mut out = s.replace("$HOME/", &format!("{home}/"));
    if out.starts_with("~/") {
        out = format!("{home}/{}", &out[2..]);
    }
    out
}

fn canonicalize_path_pattern(pattern: &str) -> Result<String, ()> {
    let first_glob = pattern.find(['*', '?']);
    match first_glob {
        Some(idx) => {
            let (prefix, suffix) = pattern.split_at(idx);
            let prefix_for_canon = if prefix.is_empty() { "." } else { prefix };
            let mut out = canonicalize_nearest(prefix_for_canon)?;
            if !suffix.is_empty() && (prefix.is_empty() || prefix.ends_with('/')) {
                out.push('/');
            }
            out.push_str(suffix);
            Ok(out)
        }
        None => canonicalize_nearest(pattern),
    }
}

fn canonicalize_nearest(path: &str) -> Result<String, ()> {
    if path.is_empty() {
        return Err(());
    }

    let mut probe = absolute_path(path)?;
    let mut suffix = Vec::new();
    loop {
        match std::fs::symlink_metadata(&probe) {
            Ok(_) => {
                let base = std::fs::canonicalize(&probe).map_err(|_| ())?;
                let joined = join_normalized_suffix(base, suffix);
                return Ok(joined.to_string_lossy().into_owned());
            }
            Err(e) if e.kind() == ErrorKind::NotFound => {}
            Err(_) => return Err(()),
        }
        let name = probe.file_name().ok_or(())?.to_os_string();
        suffix.insert(0, name);
        if !probe.pop() {
            return Err(());
        }
    }
}

fn absolute_path(path: &str) -> Result<PathBuf, ()> {
    let path = Path::new(path);
    if path.is_absolute() {
        Ok(path.to_path_buf())
    } else {
        Ok(std::env::current_dir().map_err(|_| ())?.join(path))
    }
}

fn join_normalized_suffix(mut base: PathBuf, suffix: Vec<std::ffi::OsString>) -> PathBuf {
    for part in suffix {
        if part == "." {
            continue;
        }
        if part == ".." {
            base.pop();
        } else {
            base.push(part);
        }
    }
    base
}

/// Recursive glob matcher. Handles `*`, `**`, `?`. Linear in
/// `pattern.len() + target.len()` for typical inputs.
fn match_glob(pattern: &str, target: &str) -> bool {
    let p: Vec<char> = pattern.chars().collect();
    let t: Vec<char> = target.chars().collect();
    glob_inner(&p, 0, &t, 0)
}

fn glob_inner(p: &[char], pi: usize, t: &[char], ti: usize) -> bool {
    if pi >= p.len() {
        return ti >= t.len();
    }
    match p[pi] {
        '*' => {
            // `**` matches across `/`; single `*` does not.
            let double = pi + 1 < p.len() && p[pi + 1] == '*';
            let next_pi = if double { pi + 2 } else { pi + 1 };
            // Try matching 0..=remaining chars greedily.
            for skip in 0..=(t.len() - ti) {
                if !double {
                    // Single `*` cannot consume `/`.
                    if t[ti..ti + skip].contains(&'/') {
                        break;
                    }
                }
                if glob_inner(p, next_pi, t, ti + skip) {
                    return true;
                }
            }
            false
        }
        '?' => {
            if ti < t.len() && glob_inner(p, pi + 1, t, ti + 1) {
                return true;
            }
            false
        }
        c => {
            if ti < t.len() && t[ti] == c {
                glob_inner(p, pi + 1, t, ti + 1)
            } else {
                false
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn star_matches_anything() {
        assert!(matches("*", "anything"));
        assert!(!matches("", "anything"));
    }

    #[test]
    fn single_star_does_not_cross_slash() {
        assert!(matches("src/*.rs", "src/foo.rs"));
        assert!(!matches("src/*.rs", "src/sub/foo.rs"));
    }

    #[test]
    fn double_star_crosses_slash() {
        assert!(matches("src/**", "src/foo.rs"));
        assert!(matches("src/**", "src/sub/dir/foo.rs"));
        assert!(matches("src/**/*.rs", "src/sub/foo.rs"));
    }

    #[test]
    fn question_mark_matches_one_char() {
        assert!(matches("a?c", "abc"));
        assert!(!matches("a?c", "abbc"));
        assert!(!matches("a?c", "ac"));
    }

    #[test]
    fn dotfile_matches() {
        assert!(matches(".env*", ".env"));
        assert!(matches(".env*", ".env.local"));
        assert!(!matches(".env*", "config.env"));
    }

    #[test]
    fn canonical_path_pattern_allows_create_under_existing_parent() {
        let dir = tempfile::TempDir::new().unwrap();
        let project = dir.path().join("project");
        std::fs::create_dir(&project).unwrap();
        let pattern = format!("{}/**", project.display());
        let target = project.join("new.txt");
        assert!(matches(&pattern, &target.display().to_string()));
    }

    #[test]
    fn canonical_path_pattern_rejects_dot_dot_escape() {
        let dir = tempfile::TempDir::new().unwrap();
        let project = dir.path().join("project");
        let ssh = dir.path().join(".ssh");
        std::fs::create_dir(&project).unwrap();
        std::fs::create_dir(&ssh).unwrap();
        let pattern = format!("{}/**", project.display());
        let target = project.join("..").join(".ssh").join("authorized_keys");
        assert!(!matches(&pattern, &target.display().to_string()));
    }

    #[cfg(unix)]
    #[test]
    fn canonical_path_pattern_rejects_symlink_escape() {
        use std::os::unix::fs::symlink;

        let dir = tempfile::TempDir::new().unwrap();
        let project = dir.path().join("project");
        let outside = dir.path().join("outside");
        std::fs::create_dir(&project).unwrap();
        std::fs::create_dir(&outside).unwrap();
        symlink(&outside, project.join("link")).unwrap();
        let pattern = format!("{}/**", project.display());
        let target = project.join("link").join("created.txt");
        assert!(!matches(&pattern, &target.display().to_string()));
    }

    #[test]
    fn tilde_expands_home_smoke() {
        // We can't safely mutate $HOME in this workspace (deny
        // unsafe-code); just verify the matcher resolves with
        // whatever the real $HOME is. If $HOME is unset the
        // expansion is a no-op and the test still proves the
        // expand-then-glob pipeline works.
        let home = std::env::var("HOME").unwrap_or_default();
        if !home.is_empty() {
            let target = format!("{home}/foo");
            assert!(matches("~/foo", &target));
            assert!(matches("$HOME/foo", &target));
        }
    }
}
