//! In-flight model-switch logic for Telegram sessions.
//!
//! When a user selects a new model while the agent is mid-generation,
//! the bridge can abort the current turn and re-dispatch with the new
//! model instead of silently applying it to the *next* turn only.
//!
//! Design: pure state-machine logic, no I/O. The caller (main.rs)
//! provides the abort/re-dispatch actions.

use std::sync::Arc;
use tokio::sync::Mutex;

/// Per-chat state tracking a pending model switch.
#[derive(Debug, Clone)]
pub struct PendingSwitch {
    /// The new provider (e.g. "moonshot").
    pub provider: Option<String>,
    /// The new model (e.g. "moonshot-v1-8k").
    pub model: String,
    /// Continuation text to prepend to the re-dispatched turn.
    pub continuation: Option<String>,
}

/// Shared state for model-switch coordination between the callback
/// handler (which sets the pending switch) and the stream loop
/// (which checks and executes it).
#[derive(Debug, Default)]
pub struct ModelSwitchState {
    /// If set, the stream loop should abort and re-dispatch.
    pending: Option<PendingSwitch>,
}

impl ModelSwitchState {
    pub fn new() -> Self {
        Self::default()
    }

    /// Request a model switch. Returns `true` if a switch was staged.
    pub fn request(&mut self, switch: PendingSwitch) -> bool {
        self.pending = Some(switch);
        true
    }

    /// Take the pending switch (consuming it). Returns `None` if no
    /// switch was requested.
    pub fn take(&mut self) -> Option<PendingSwitch> {
        self.pending.take()
    }

    /// Check if a switch is pending without consuming it.
    pub fn is_pending(&self) -> bool {
        self.pending.is_some()
    }

    /// Clear any pending switch (e.g. when the turn ends naturally).
    pub fn clear(&mut self) {
        self.pending = None;
    }
}

/// Thread-safe handle for model-switch state shared between the
/// callback handler and the stream loop.
pub type SharedModelSwitch = Arc<Mutex<ModelSwitchState>>;

/// Create a new shared model-switch state.
pub fn new_shared() -> SharedModelSwitch {
    Arc::new(Mutex::new(ModelSwitchState::new()))
}

/// Decide whether to abort for a model switch.
///
/// Call this at safe yield points in the stream loop (between events,
/// after a tool completes). Returns `Some(switch)` if the loop should
/// abort and re-dispatch.
pub async fn check_and_take(state: &SharedModelSwitch) -> Option<PendingSwitch> {
    state.lock().await.take()
}

/// Build the continuation prompt for a re-dispatched turn.
///
/// Format: "Continue the previous request using {provider}/{model}.
/// Resume from where you left off."
pub fn build_continuation(switch: &PendingSwitch) -> String {
    let model_label = match &switch.provider {
        Some(p) => format!("{p}/{}", switch.model),
        None => switch.model.clone(),
    };
    if let Some(custom) = &switch.continuation {
        return custom.clone();
    }
    format!(
        "Continue the previous Telegram request using the newly selected model ({model_label}). \
         Resume from the last unfinished step instead of restarting from scratch."
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_state_has_no_pending() {
        let state = ModelSwitchState::new();
        assert!(!state.is_pending());
    }

    #[test]
    fn request_sets_pending() {
        let mut state = ModelSwitchState::new();
        let ok = state.request(PendingSwitch {
            provider: Some("moonshot".into()),
            model: "moonshot-v1-8k".into(),
            continuation: None,
        });
        assert!(ok);
        assert!(state.is_pending());
    }

    #[test]
    fn take_consumes_pending() {
        let mut state = ModelSwitchState::new();
        state.request(PendingSwitch {
            provider: Some("qwen".into()),
            model: "qwen3.6-plus".into(),
            continuation: None,
        });
        let switch = state.take();
        assert!(switch.is_some());
        assert!(!state.is_pending());
        assert_eq!(switch.unwrap().model, "qwen3.6-plus");
    }

    #[test]
    fn take_without_pending_returns_none() {
        let mut state = ModelSwitchState::new();
        assert!(state.take().is_none());
    }

    #[test]
    fn clear_removes_pending() {
        let mut state = ModelSwitchState::new();
        state.request(PendingSwitch {
            provider: None,
            model: "x".into(),
            continuation: None,
        });
        state.clear();
        assert!(!state.is_pending());
    }

    #[test]
    fn second_request_replaces_first() {
        let mut state = ModelSwitchState::new();
        state.request(PendingSwitch {
            provider: None,
            model: "old".into(),
            continuation: None,
        });
        state.request(PendingSwitch {
            provider: None,
            model: "new".into(),
            continuation: None,
        });
        let switch = state.take().unwrap();
        assert_eq!(switch.model, "new");
    }

    #[test]
    fn build_continuation_default() {
        let switch = PendingSwitch {
            provider: Some("moonshot".into()),
            model: "moonshot-v1-8k".into(),
            continuation: None,
        };
        let text = build_continuation(&switch);
        assert!(text.contains("moonshot/moonshot-v1-8k"));
        assert!(text.contains("Continue"));
    }

    #[test]
    fn build_continuation_custom() {
        let switch = PendingSwitch {
            provider: None,
            model: "x".into(),
            continuation: Some("my custom text".into()),
        };
        assert_eq!(build_continuation(&switch), "my custom text");
    }

    #[test]
    fn build_continuation_no_provider() {
        let switch = PendingSwitch {
            provider: None,
            model: "llama-3.3".into(),
            continuation: None,
        };
        let text = build_continuation(&switch);
        assert!(text.contains("llama-3.3"));
        assert!(!text.contains('/'));
    }

    #[tokio::test]
    async fn shared_check_and_take() {
        let shared = new_shared();
        assert!(check_and_take(&shared).await.is_none());

        shared.lock().await.request(PendingSwitch {
            provider: Some("qwen".into()),
            model: "qwen3.6-plus".into(),
            continuation: None,
        });

        let switch = check_and_take(&shared).await;
        assert!(switch.is_some());
        assert_eq!(switch.unwrap().model, "qwen3.6-plus");
        // Consumed
        assert!(check_and_take(&shared).await.is_none());
    }
}
