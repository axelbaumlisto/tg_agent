//! Context capacity monitoring — warns before hitting context limits.
//!
//! `CapacityGuard` checks token usage and returns a `Pressure` level.
//! The agent loop emits `AgentEvent::CapacityWarning` which frontends
//! (TG, CLI, API) render appropriately.

/// Context pressure level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pressure {
    /// Below 70% — no action needed.
    Normal,
    /// 70-85% — warn user, context getting full.
    Warning,
    /// 85-95% — auto-compact recommended.
    Critical,
    /// Above 95% — emergency, must compact now.
    Emergency,
}

impl Pressure {
    /// Human-readable label for UI.
    pub fn label(&self) -> &'static str {
        match self {
            Pressure::Normal => "normal",
            Pressure::Warning => "⚠️ context filling up",
            Pressure::Critical => "🟠 context nearly full",
            Pressure::Emergency => "🔴 context overflow — compacting",
        }
    }

    pub fn is_actionable(&self) -> bool {
        !matches!(self, Pressure::Normal)
    }
}

/// Check context pressure based on token usage.
pub fn check_pressure(estimated_tokens: u64, context_window: u64) -> Pressure {
    if context_window == 0 {
        return Pressure::Normal;
    }
    let pct = (estimated_tokens * 100) / context_window;
    match pct {
        0..=69 => Pressure::Normal,
        70..=84 => Pressure::Warning,
        85..=94 => Pressure::Critical,
        _ => Pressure::Emergency,
    }
}

/// Format a capacity status line for UI display.
pub fn format_capacity(estimated_tokens: u64, context_window: u64) -> String {
    if context_window == 0 {
        return String::new();
    }
    let pct = (estimated_tokens * 100) / context_window;
    let est_k = estimated_tokens / 1000;
    let win_k = context_window / 1000;
    format!("{est_k}K/{win_k}K ({pct}%)")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normal_below_70() {
        assert_eq!(check_pressure(50_000, 128_000), Pressure::Normal);
        assert_eq!(check_pressure(0, 128_000), Pressure::Normal);
        assert_eq!(check_pressure(89_000, 128_000), Pressure::Normal); // 69.5%
    }

    #[test]
    fn warning_70_to_84() {
        assert_eq!(check_pressure(90_000, 128_000), Pressure::Warning); // 70.3%
        assert_eq!(check_pressure(107_000, 128_000), Pressure::Warning); // 83.6%
    }

    #[test]
    fn critical_85_to_94() {
        assert_eq!(check_pressure(110_000, 128_000), Pressure::Critical); // 85.9%
        assert_eq!(check_pressure(120_000, 128_000), Pressure::Critical); // 93.8%
    }

    #[test]
    fn emergency_above_95() {
        assert_eq!(check_pressure(122_000, 128_000), Pressure::Emergency); // 95.3%
        assert_eq!(check_pressure(128_000, 128_000), Pressure::Emergency);
        assert_eq!(check_pressure(200_000, 128_000), Pressure::Emergency);
    }

    #[test]
    fn zero_window_is_normal() {
        assert_eq!(check_pressure(100_000, 0), Pressure::Normal);
    }

    #[test]
    fn format_capacity_display() {
        assert_eq!(format_capacity(90_000, 128_000), "90K/128K (70%)");
        assert_eq!(format_capacity(50_000, 128_000), "50K/128K (39%)");
    }

    #[test]
    fn pressure_labels() {
        assert!(!Pressure::Normal.is_actionable());
        assert!(Pressure::Warning.is_actionable());
        assert!(Pressure::Critical.is_actionable());
        assert!(Pressure::Emergency.is_actionable());
    }
}
