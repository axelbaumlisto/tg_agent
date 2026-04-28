use std::collections::HashMap;

use crate::types::TurnUsage;

#[derive(Debug, Clone, Default)]
pub struct UsageTracker {
    pub total_input_tokens: u64,
    pub total_output_tokens: u64,
    pub total_cache_read_tokens: u64,
    pub total_cache_write_tokens: u64,
    pub turn_count: usize,
    pub tool_call_count: usize,
    pub tool_frequency: HashMap<String, usize>,
}

impl UsageTracker {
    pub fn record_turn(&mut self, usage: &TurnUsage) {
        self.total_input_tokens += usage.input_tokens;
        self.total_output_tokens += usage.output_tokens;
        self.total_cache_read_tokens += usage.cache_read_tokens;
        self.total_cache_write_tokens += usage.cache_write_tokens;
        self.turn_count += 1;
    }

    pub fn record_tool_call(&mut self, tool_name: &str) {
        self.tool_call_count += 1;
        *self
            .tool_frequency
            .entry(tool_name.to_string())
            .or_default() += 1;
    }

    pub fn estimated_cost_usd(&self, pricing: &ModelPricing) -> f64 {
        let input_cost = self.total_input_tokens as f64 * pricing.input_per_million / 1_000_000.0;
        let output_cost =
            self.total_output_tokens as f64 * pricing.output_per_million / 1_000_000.0;
        let cache_read_cost =
            self.total_cache_read_tokens as f64 * pricing.cache_read_per_million / 1_000_000.0;
        let cache_write_cost =
            self.total_cache_write_tokens as f64 * pricing.cache_write_per_million / 1_000_000.0;
        input_cost + output_cost + cache_read_cost + cache_write_cost
    }

    pub fn cumulative_usage(&self) -> TurnUsage {
        TurnUsage {
            input_tokens: self.total_input_tokens,
            output_tokens: self.total_output_tokens,
            cache_read_tokens: self.total_cache_read_tokens,
            cache_write_tokens: self.total_cache_write_tokens,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ModelPricing {
    pub input_per_million: f64,
    pub output_per_million: f64,
    pub cache_read_per_million: f64,
    pub cache_write_per_million: f64,
}

impl ModelPricing {
    pub fn sonnet() -> Self {
        Self {
            input_per_million: 3.0,
            output_per_million: 15.0,
            cache_read_per_million: 0.30,
            cache_write_per_million: 3.75,
        }
    }
}

pub fn pricing_for_model(model: &str) -> ModelPricing {
    let lower = model.to_lowercase();
    if lower.contains("haiku") {
        ModelPricing {
            input_per_million: 0.25,
            output_per_million: 1.25,
            cache_read_per_million: 0.03,
            cache_write_per_million: 0.30,
        }
    } else if lower.contains("opus") {
        ModelPricing {
            input_per_million: 15.0,
            output_per_million: 75.0,
            cache_read_per_million: 1.50,
            cache_write_per_million: 18.75,
        }
    } else {
        ModelPricing::sonnet()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_turn_accumulates() {
        let mut tracker = UsageTracker::default();
        tracker.record_turn(&TurnUsage {
            input_tokens: 100,
            output_tokens: 50,
            cache_read_tokens: 10,
            cache_write_tokens: 5,
        });
        tracker.record_turn(&TurnUsage {
            input_tokens: 200,
            output_tokens: 100,
            cache_read_tokens: 0,
            cache_write_tokens: 0,
        });
        assert_eq!(tracker.total_input_tokens, 300);
        assert_eq!(tracker.total_output_tokens, 150);
        assert_eq!(tracker.total_cache_read_tokens, 10);
        assert_eq!(tracker.total_cache_write_tokens, 5);
        assert_eq!(tracker.turn_count, 2);
    }

    #[test]
    fn record_tool_call_tracks_frequency() {
        let mut tracker = UsageTracker::default();
        tracker.record_tool_call("bash");
        tracker.record_tool_call("bash");
        tracker.record_tool_call("read_file");
        assert_eq!(tracker.tool_call_count, 3);
        assert_eq!(tracker.tool_frequency["bash"], 2);
        assert_eq!(tracker.tool_frequency["read_file"], 1);
    }

    #[test]
    fn estimated_cost_sonnet() {
        let tracker = UsageTracker {
            total_input_tokens: 1_000_000,
            total_output_tokens: 1_000_000,
            ..Default::default()
        };
        let cost = tracker.estimated_cost_usd(&ModelPricing::sonnet());
        let expected = 3.0 + 15.0; // $3/M input + $15/M output
        assert!((cost - expected).abs() < 0.001);
    }

    #[test]
    fn cumulative_usage_snapshot() {
        let mut tracker = UsageTracker::default();
        tracker.record_turn(&TurnUsage {
            input_tokens: 10,
            output_tokens: 20,
            cache_read_tokens: 3,
            cache_write_tokens: 4,
        });
        let cu = tracker.cumulative_usage();
        assert_eq!(cu.input_tokens, 10);
        assert_eq!(cu.output_tokens, 20);
        assert_eq!(cu.cache_read_tokens, 3);
        assert_eq!(cu.cache_write_tokens, 4);
    }

    #[test]
    fn pricing_for_model_haiku() {
        let p = pricing_for_model("claude-3-haiku-20240307");
        assert!((p.input_per_million - 0.25).abs() < 0.001);
    }

    #[test]
    fn pricing_for_model_opus() {
        let p = pricing_for_model("claude-3-opus");
        assert!((p.input_per_million - 15.0).abs() < 0.001);
    }

    #[test]
    fn pricing_for_model_defaults_to_sonnet() {
        let p = pricing_for_model("claude-sonnet-4-20250514");
        assert!((p.input_per_million - 3.0).abs() < 0.001);
    }
}
