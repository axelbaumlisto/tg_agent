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
    /// Empty string `""` or `"*"` matches every tool.
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
    /// Last-match-wins evaluator. Returns `Action::Ask` when no rule
    /// matches — that's the default behaviour for any (tool, target)
    /// pair the user hasn't expressed an opinion on.
    #[must_use]
    pub fn evaluate(&self, tool: &str, target: &str) -> Action {
        let mut decision = Action::Ask;
        for rule in &self.rules {
            if super::matcher::matches(&rule.permission, tool)
                && super::matcher::matches(&rule.pattern, target)
            {
                decision = rule.action;
            }
        }
        decision
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
    fn last_match_wins() {
        let mut r = Ruleset::default();
        r.push(Rule::new("read", "src/**", Action::Allow));
        r.push(Rule::new("read", "src/secrets.rs", Action::Deny));
        assert_eq!(r.evaluate("read", "src/secrets.rs"), Action::Deny);
        assert_eq!(r.evaluate("read", "src/foo.rs"), Action::Allow);
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
