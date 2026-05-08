//! Adaptive reasoning-effort tier selection per turn.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReasoningEffort {
    Low,
    Medium,
    High,
    Max,
}

impl ReasoningEffort {
    pub fn label(self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::Max => "max",
        }
    }
}

/// Pick effort from message content.  Sub-agents always get Low.
pub fn select(is_subagent: bool, message: &str) -> ReasoningEffort {
    if is_subagent {
        return ReasoningEffort::Low;
    }
    let lo = message.to_ascii_lowercase();
    if lo.contains("debug") || lo.contains("error") || lo.contains("bug") {
        return ReasoningEffort::Max;
    }
    if lo.contains("search") || lo.contains("lookup") || lo.contains("find") {
        return ReasoningEffort::Low;
    }
    if lo.contains("plan") || lo.contains("architect") || lo.contains("design") {
        return ReasoningEffort::High;
    }
    ReasoningEffort::Medium
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn subagent_low() {
        assert_eq!(select(true, "debug error"), ReasoningEffort::Low);
    }
    #[test]
    fn debug_max() {
        assert_eq!(select(false, "debug crash"), ReasoningEffort::Max);
    }
    #[test]
    fn search_low() {
        assert_eq!(select(false, "find the file"), ReasoningEffort::Low);
    }
    #[test]
    fn default_medium() {
        assert_eq!(select(false, "implement X"), ReasoningEffort::Medium);
    }
}
