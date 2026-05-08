use serde::{Deserialize, Serialize};

// -- Usage -------------------------------------------------------------------

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct TurnUsage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    #[serde(default)]
    pub cache_read_tokens: u64,
    #[serde(default)]
    pub cache_write_tokens: u64,
}

impl TurnUsage {
    pub fn total_tokens(&self) -> u64 {
        self.input_tokens + self.output_tokens + self.cache_read_tokens + self.cache_write_tokens
    }

    /// Estimate USD cost based on model name. Returns `(input_cost, output_cost, total)`.
    pub fn estimate_cost(&self, model: &str) -> (f64, f64, f64) {
        let (inp_per_m, out_per_m) = model_pricing(model);
        let input_cost =
            (self.input_tokens as f64 + self.cache_read_tokens as f64 * 0.1) * inp_per_m / 1e6;
        let output_cost = self.output_tokens as f64 * out_per_m / 1e6;
        (input_cost, output_cost, input_cost + output_cost)
    }
}

fn model_pricing(model: &str) -> (f64, f64) {
    let m = model.to_ascii_lowercase();
    if m.contains("opus") {
        (15.0, 75.0)
    } else if m.contains("haiku") {
        (0.25, 1.25)
    } else if m.contains("sonnet") || m.contains("claude-3-5") || m.contains("claude-3.5") {
        (3.0, 15.0)
    } else if m.contains("gpt-4o-mini") {
        (0.15, 0.60)
    } else if m.contains("gpt-4o") || m.contains("gpt-4-turbo") {
        (5.0, 15.0)
    } else if m.contains("gpt-4") {
        (30.0, 60.0)
    } else if m.contains("gpt-3.5") {
        (0.50, 1.50)
    } else if m.contains("deepseek") {
        (0.14, 0.28)
    } else if m.contains("llama")
        || m.contains("mixtral")
        || m.contains("minimax")
        || m.contains("m2p7")
    {
        (0.20, 0.20)
    } else if m.contains("glm") || m.contains("chatglm") {
        (0.10, 0.10)
    } else if m.contains("moonshot") || m.contains("kimi") {
        (0.30, 0.30)
    } else {
        (1.0, 3.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn turn_usage_default_is_zero() {
        let u = TurnUsage::default();
        assert_eq!(u.input_tokens, 0);
        assert_eq!(u.output_tokens, 0);
    }

    #[test]
    fn turn_usage_add() {
        let mut a = TurnUsage {
            input_tokens: 100,
            output_tokens: 50,
            cache_read_tokens: 0,
            cache_write_tokens: 0,
        };
        let b = TurnUsage {
            input_tokens: 200,
            output_tokens: 150,
            cache_read_tokens: 0,
            cache_write_tokens: 0,
        };
        a.input_tokens += b.input_tokens;
        a.output_tokens += b.output_tokens;
        assert_eq!(a.input_tokens, 300);
        assert_eq!(a.output_tokens, 200);
    }
}
