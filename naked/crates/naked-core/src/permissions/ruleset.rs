//! Permission ruleset core data + evaluator.

use serde::{Deserialize, Serialize};

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Action {
    Allow,
    Deny,
    /// Default — surface UI prompt to the user.
    #[default]
    Ask,
}

/// User reply to an Ask-prompted UI question. Drives whether the
/// answer becomes a persistent rule (`Always`) or one-off (`Once`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Reply {
    Once,
    Always,
    Reject,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Rule {
    /// Tool name — matches via [`matcher::matches`].
    /// `"*"` matches every tool; empty strings match nothing so partial
    /// or malformed persisted rules fail closed to `Ask`.
    pub permission: String,
    /// Path / target pattern — matches via [`matcher::matches`].
    pub pattern: String,
    pub action: Action,
}

impl Rule {
    pub fn new(permission: impl Into<String>, pattern: impl Into<String>, action: Action) -> Self {
        Self {
            permission: permission.into(),
            pattern: pattern.into(),
            action,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Ruleset {
    pub rules: Vec<Rule>,
}

impl Ruleset {
    /// Precedence-based evaluator for security decisions. `Deny` wins over
    /// every other matching rule regardless of order, `Ask` wins over `Allow`,
    /// and no match returns `Action::Ask`. Canonicalisation failures in a
    /// relevant rule are also `Ask`: an indeterminate protective rule must not
    /// be collapsed into a non-match that a later broad `Allow` can erase.
    #[must_use]
    pub fn evaluate(&self, tool: &str, target: &str) -> Action {
        let mut saw_allow = false;
        let mut saw_ask = false;
        let mut saw_indeterminate = false;
        for rule in &self.rules {
            match super::matcher::match_rule(&rule.permission, tool) {
                super::matcher::MatchResult::Yes => {}
                super::matcher::MatchResult::No => continue,
                super::matcher::MatchResult::Indeterminate => {
                    saw_indeterminate = true;
                    continue;
                }
            }

            match super::matcher::match_rule(&rule.pattern, target) {
                super::matcher::MatchResult::Yes => match rule.action {
                    Action::Deny => return Action::Deny,
                    Action::Ask => saw_ask = true,
                    Action::Allow => saw_allow = true,
                },
                super::matcher::MatchResult::No => {}
                super::matcher::MatchResult::Indeterminate => saw_indeterminate = true,
            }
        }
        if saw_ask || saw_indeterminate {
            Action::Ask
        } else if saw_allow {
            Action::Allow
        } else {
            Action::Ask
        }
    }

    pub fn push(&mut self, rule: Rule) {
        self.rules.push(rule);
    }

    pub fn remove_matching(&mut self, permission: &str, pattern: &str) -> usize {
        let before = self.rules.len();
        self.rules
            .retain(|r| !(r.permission == permission && r.pattern == pattern));
        before - self.rules.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_ruleset_asks() {
        let r = Ruleset::default();
        assert_eq!(r.evaluate("read", "src/foo.rs"), Action::Ask);
    }

    #[test]
    fn deny_precedence_wins_regardless_of_order() {
        let mut r = Ruleset::default();
        r.push(Rule::new("*", ".env*", Action::Deny));
        r.push(Rule::new("read", "*", Action::Allow));
        let got = r.evaluate("read", ".env");
        println!("deny-then-allow => {got:?}");
        assert_eq!(got, Action::Deny);
    }

    #[test]
    fn ask_does_not_erase_deny_and_can_override_allow() {
        let mut r = Ruleset::default();
        r.push(Rule::new("read", "src/**", Action::Deny));
        r.push(Rule::new("read", "src/secrets.rs", Action::Ask));
        assert_eq!(r.evaluate("read", "src/secrets.rs"), Action::Deny);

        let mut r = Ruleset::default();
        r.push(Rule::new("read", "src/**", Action::Allow));
        r.push(Rule::new("read", "src/secrets.rs", Action::Ask));
        assert_eq!(r.evaluate("read", "src/secrets.rs"), Action::Ask);
        assert_eq!(r.evaluate("read", "src/foo.rs"), Action::Allow);
    }

    #[test]
    fn empty_rule_patterns_match_nothing() {
        let mut r = Ruleset::default();
        r.push(Rule::new("", "", Action::Allow));
        let got = r.evaluate("bash", "rm -rf /");
        println!("empty-rule => {got:?}");
        assert_eq!(got, Action::Ask);
    }

    #[test]
    fn canonical_path_escape_falls_back_to_ask() {
        let dir = tempfile::TempDir::new().unwrap();
        let project = dir.path().join("project");
        let ssh = dir.path().join(".ssh");
        std::fs::create_dir(&project).unwrap();
        std::fs::create_dir(&ssh).unwrap();
        let mut r = Ruleset::default();
        r.push(Rule::new(
            "write",
            format!("{}/**", project.display()),
            Action::Allow,
        ));
        let target = project.join("..").join(".ssh").join("authorized_keys");
        let got = r.evaluate("write", &target.display().to_string());
        println!("canonical-dotdot-escape => {got:?}");
        assert_eq!(got, Action::Ask);
    }

    #[cfg(unix)]
    #[test]
    fn canonicalization_failure_for_deny_forces_ask_not_later_allow() {
        use std::os::unix::fs::symlink;

        let dir = tempfile::TempDir::new().unwrap();
        let loop_path = dir.path().join("loop");
        symlink(&loop_path, &loop_path).unwrap();

        let mut r = Ruleset::default();
        r.push(Rule::new(
            "write",
            format!("{}/**", loop_path.display()),
            Action::Deny,
        ));
        r.push(Rule::new("write", "*", Action::Allow));

        let target = loop_path.join("secret");
        let got = r.evaluate("write", &target.display().to_string());
        println!("symlink-loop deny + broad allow => {got:?}");
        assert_eq!(got, Action::Ask);
    }

    #[cfg(unix)]
    #[test]
    fn dangling_symlink_parent_failure_forces_ask_not_later_allow() {
        use std::os::unix::fs::symlink;

        let dir = tempfile::TempDir::new().unwrap();
        let dead_parent = dir.path().join("dead-parent");
        symlink(dir.path().join("missing-target"), &dead_parent).unwrap();

        let mut r = Ruleset::default();
        r.push(Rule::new(
            "write",
            format!("{}/**", dead_parent.display()),
            Action::Deny,
        ));
        r.push(Rule::new("write", "*", Action::Allow));

        let target = dead_parent.join("secret");
        let got = r.evaluate("write", &target.display().to_string());
        println!("dangling-symlink-parent deny + broad allow => {got:?}");
        assert_eq!(got, Action::Ask);
    }

    #[test]
    fn unknown_tool_returns_ask() {
        let mut r = Ruleset::default();
        r.push(Rule::new("read", "*", Action::Allow));
        assert_eq!(r.evaluate("write", "src/foo.rs"), Action::Ask);
    }

    #[test]
    fn star_permission_matches_any_tool() {
        let mut r = Ruleset::default();
        r.push(Rule::new("*", ".env*", Action::Deny));
        assert_eq!(r.evaluate("read", ".env"), Action::Deny);
        assert_eq!(r.evaluate("write", ".env.local"), Action::Deny);
        assert_eq!(r.evaluate("read", "config.toml"), Action::Ask);
    }

    #[test]
    fn remove_matching_drops_rules() {
        let mut r = Ruleset::default();
        r.push(Rule::new("read", "src/**", Action::Allow));
        r.push(Rule::new("read", "src/**", Action::Deny)); // dup pattern
        r.push(Rule::new("write", "src/**", Action::Allow));
        let n = r.remove_matching("read", "src/**");
        assert_eq!(n, 2);
        assert_eq!(r.rules.len(), 1);
    }
}
