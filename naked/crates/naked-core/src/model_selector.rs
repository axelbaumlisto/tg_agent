//! Auto model selection — picks the right model for each turn.
//!
//! `select_model()` analyzes the user message, context size, and provider
//! health to choose the optimal (provider, model) pair. This runs in
//! naked-core and is frontend-agnostic.
//!
//! Design: simple rules, not ML. Fast, deterministic, testable.

/// Hint about the kind of task the user is asking for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskHint {
    /// Quick question, greeting, simple lookup.
    Quick,
    /// Code writing, debugging, refactoring.
    Code,
    /// Research — needs large context, many tool calls.
    Research,
    /// Analysis, planning, complex reasoning.
    Deep,
    /// Translation, summarization, simple transform.
    Transform,
}

/// Result of auto model selection.
#[derive(Debug, Clone)]
pub struct ModelChoice {
    pub provider: String,
    pub model: String,
    pub reason: String,
}

/// Classify user message into a task hint.
pub fn classify_task(message: &str) -> TaskHint {
    let lower = message.to_ascii_lowercase();

    // Research keywords:
    if lower.contains("research")
        || lower.contains("исследу")
        || lower.contains("найди")
        || lower.contains("поищи")
        || lower.contains("собери информацию")
    {
        return TaskHint::Research;
    }

    // Code keywords:
    if lower.contains("debug")
        || lower.contains("error")
        || lower.contains("fix")
        || lower.contains("refactor")
        || lower.contains("implement")
        || lower.contains("напиши код")
        || lower.contains("баг")
        || lower.contains("ошибк")
        || lower.contains("cargo")
        || lower.contains("компил")
    {
        return TaskHint::Code;
    }

    // Translation/transform:
    if lower.contains("перевед")
        || lower.contains("переведи")
        || lower.contains("translat")
        || lower.contains("summariz")
        || lower.contains("кратко")
    {
        return TaskHint::Transform;
    }

    // Deep analysis:
    if lower.contains("план")
        || lower.contains("анализ")
        || lower.contains("architect")
        || lower.contains("design")
        || lower.contains("стратег")
        || lower.len() > 500
    {
        return TaskHint::Deep;
    }

    // Short messages = quick:
    if lower.len() < 100 {
        return TaskHint::Quick;
    }

    TaskHint::Deep
}

/// Select model based on task, context, and available providers.
///
/// `available_providers` is a list of (provider, model) pairs that are
/// currently healthy and configured. The selector picks from this list.
pub fn select_model(
    task: TaskHint,
    context_tokens: u64,
    available: &[(String, String)],
) -> Option<ModelChoice> {
    if available.is_empty() {
        return None;
    }

    // Priority lists by task type. First match in available wins.
    let preferences: &[(&str, &str)] = match task {
        TaskHint::Quick => &[
            ("groq", "llama-3.3-70b-versatile"),
            ("qwen", "qwen-turbo"),
            ("groq", "llama-3.1-8b-instant"),
            ("deepseek", "deepseek-v4-flash"),
        ],
        TaskHint::Code => &[
            ("kimi-code", "kimi-for-coding"),
            ("fireworks", "accounts/fireworks/models/deepseek-v4-pro"),
            ("deepseek", "deepseek-v4-pro"),
            ("qwen", "qwen3-coder-plus"),
        ],
        TaskHint::Research => &[
            ("kimi-code", "kimi-for-coding"),
            ("qwen", "qwen3.6-plus"),
            ("deepseek", "deepseek-v4-pro"),
        ],
        TaskHint::Deep => &[
            ("qwen", "qwen3.6-plus"),
            ("deepseek", "deepseek-v4-pro"),
            ("fireworks", "accounts/fireworks/models/deepseek-v4-pro"),
        ],
        TaskHint::Transform => &[
            ("groq", "llama-3.3-70b-versatile"),
            ("qwen", "qwen-turbo"),
            ("moonshot", "moonshot-v1-32k"),
        ],
    };

    // If context is large (>50K), prefer models with 128K+ windows:
    let large_context = context_tokens > 50_000;
    if large_context {
        // kimi-code has 1M, qwen3.6-plus has 1M, deepseek-v4-pro has 128K
        let large_prefs: &[(&str, &str)] = &[
            ("kimi-code", "kimi-for-coding"),
            ("qwen", "qwen3.6-plus"),
            ("deepseek", "deepseek-v4-pro"),
        ];
        for (prov, model) in large_prefs {
            if available.iter().any(|(p, m)| p == prov && m == model) {
                return Some(ModelChoice {
                    provider: prov.to_string(),
                    model: model.to_string(),
                    reason: format!("large context ({context_tokens} tokens) → {prov}/{model}"),
                });
            }
        }
    }

    // Normal selection: first match in preference list:
    for (prov, model) in preferences {
        if available.iter().any(|(p, m)| p == prov && m == model) {
            return Some(ModelChoice {
                provider: prov.to_string(),
                model: model.to_string(),
                reason: format!("{task:?} → {prov}/{model}"),
            });
        }
    }

    // Fallback: first available model:
    let (p, m) = &available[0];
    Some(ModelChoice {
        provider: p.clone(),
        model: m.clone(),
        reason: format!("fallback → {p}/{m}"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_quick() {
        assert_eq!(classify_task("привет"), TaskHint::Quick);
        assert_eq!(classify_task("hi"), TaskHint::Quick);
        assert_eq!(classify_task("что нового?"), TaskHint::Quick);
    }

    #[test]
    fn classify_code() {
        assert_eq!(classify_task("debug this error"), TaskHint::Code);
        assert_eq!(classify_task("fix the compilation error"), TaskHint::Code);
        assert_eq!(classify_task("напиши код на rust"), TaskHint::Code);
        assert_eq!(classify_task("cargo test fails"), TaskHint::Code);
    }

    #[test]
    fn classify_research() {
        assert_eq!(
            classify_task("research apartments in da nang"),
            TaskHint::Research
        );
        assert_eq!(classify_task("поищи информацию"), TaskHint::Research);
    }

    #[test]
    fn classify_transform() {
        assert_eq!(classify_task("переведи на русский"), TaskHint::Transform);
        assert_eq!(classify_task("translate to english"), TaskHint::Transform);
        assert_eq!(classify_task("кратко перескажи"), TaskHint::Transform);
    }

    #[test]
    fn classify_deep() {
        assert_eq!(classify_task("разработай план архитектуры"), TaskHint::Deep);
        assert_eq!(classify_task("design a system for..."), TaskHint::Deep);
    }

    #[test]
    fn select_quick_prefers_groq() {
        let available = vec![
            ("qwen".into(), "qwen3.6-plus".into()),
            ("groq".into(), "llama-3.3-70b-versatile".into()),
        ];
        let choice = select_model(TaskHint::Quick, 5000, &available).unwrap();
        assert_eq!(choice.provider, "groq");
    }

    #[test]
    fn select_code_prefers_kimi() {
        let available = vec![
            ("qwen".into(), "qwen3.6-plus".into()),
            ("kimi-code".into(), "kimi-for-coding".into()),
            ("groq".into(), "llama-3.3-70b-versatile".into()),
        ];
        let choice = select_model(TaskHint::Code, 5000, &available).unwrap();
        assert_eq!(choice.provider, "kimi-code");
    }

    #[test]
    fn select_large_context_overrides() {
        let available = vec![
            ("groq".into(), "llama-3.3-70b-versatile".into()),
            ("qwen".into(), "qwen3.6-plus".into()),
        ];
        // Large context → prefers qwen3.6-plus (1M) even for Quick task:
        let choice = select_model(TaskHint::Quick, 60_000, &available).unwrap();
        assert_eq!(choice.provider, "qwen");
        assert!(choice.reason.contains("large context"));
    }

    #[test]
    fn select_fallback_when_no_preferred() {
        let available = vec![("custom".into(), "custom-model".into())];
        let choice = select_model(TaskHint::Code, 5000, &available).unwrap();
        assert_eq!(choice.provider, "custom");
        assert!(choice.reason.contains("fallback"));
    }

    #[test]
    fn select_empty_returns_none() {
        assert!(select_model(TaskHint::Quick, 0, &[]).is_none());
    }
}
