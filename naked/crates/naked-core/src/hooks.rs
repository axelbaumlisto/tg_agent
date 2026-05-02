//! B6: Lightweight hook system for agent lifecycle events.
//!
//! Instead of a full event bus (27 event types), two focused hooks
//! cover 90% of Pi's extension use cases:
//!
//! - **ContextHook**: modify messages before each LLM call
//!   (inject git diff, test results, dynamic context)
//! - **ToolCallHook**: intercept tool calls before execution
//!   (modify args, block calls, add flags)

use std::sync::Arc;
use tokio::sync::RwLock;

use crate::types::ConversationMessage;

/// Called before each LLM call. Can modify the message list in-place.
/// Use to inject dynamic context, remove stale messages, etc.
///
/// Receives: mutable reference to the messages that will be sent to the LLM.
/// The system prompt is messages[0] (role=system).
pub type ContextHookFn = dyn Fn(&mut Vec<ConversationMessage>) + Send + Sync;

/// Called before each tool execution. Can modify args or block the call.
///
/// Receives: tool name, mutable input args.
/// Returns: `true` to allow, `false` to block (tool returns "blocked by hook").
pub type ToolCallHookFn = dyn Fn(&str, &mut serde_json::Value) -> bool + Send + Sync;

/// Registry of hooks. Thread-safe, append-only during normal operation.
pub struct HookRegistry {
    context_hooks: RwLock<Vec<Arc<ContextHookFn>>>,
    tool_call_hooks: RwLock<Vec<Arc<ToolCallHookFn>>>,
}

impl HookRegistry {
    pub fn new() -> Self {
        Self {
            context_hooks: RwLock::new(Vec::new()),
            tool_call_hooks: RwLock::new(Vec::new()),
        }
    }

    /// Register a context hook (called before each LLM call).
    pub async fn on_context(&self, hook: Arc<ContextHookFn>) {
        self.context_hooks.write().await.push(hook);
    }

    /// Register a tool call hook (called before each tool execution).
    pub async fn on_tool_call(&self, hook: Arc<ToolCallHookFn>) {
        self.tool_call_hooks.write().await.push(hook);
    }

    /// Run all context hooks on the message list.
    pub async fn run_context_hooks(&self, messages: &mut Vec<ConversationMessage>) {
        let hooks = self.context_hooks.read().await;
        for hook in hooks.iter() {
            hook(messages);
        }
    }

    /// Run all tool call hooks. Returns false if any hook blocks the call.
    pub async fn run_tool_call_hooks(
        &self,
        tool_name: &str,
        input: &mut serde_json::Value,
    ) -> bool {
        let hooks = self.tool_call_hooks.read().await;
        for hook in hooks.iter() {
            if !hook(tool_name, input) {
                return false;
            }
        }
        true
    }

    /// Number of registered hooks (for /status).
    pub async fn count(&self) -> (usize, usize) {
        (
            self.context_hooks.read().await.len(),
            self.tool_call_hooks.read().await.len(),
        )
    }
}

impl Default for HookRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn context_hook_modifies_messages() {
        let registry = HookRegistry::new();

        // Hook that appends a marker to the last message
        registry
            .on_context(Arc::new(|msgs: &mut Vec<ConversationMessage>| {
                msgs.push(ConversationMessage::user("[hook-injected]"));
            }))
            .await;

        let mut messages = vec![
            ConversationMessage::system("sys"),
            ConversationMessage::user("hello"),
        ];

        registry.run_context_hooks(&mut messages).await;
        assert_eq!(messages.len(), 3);
        assert_eq!(messages[2].text_content(), "[hook-injected]");
    }

    #[tokio::test]
    async fn tool_call_hook_blocks() {
        let registry = HookRegistry::new();

        // Block any rm -rf command
        registry
            .on_tool_call(Arc::new(|name: &str, input: &mut serde_json::Value| {
                if name == "bash" {
                    if let Some(cmd) = input.get("command").and_then(|v| v.as_str()) {
                        if cmd.contains("rm -rf") {
                            return false;
                        }
                    }
                }
                true
            }))
            .await;

        // Safe command — allowed
        let mut input = serde_json::json!({"command": "ls -la"});
        assert!(registry.run_tool_call_hooks("bash", &mut input).await);

        // Dangerous command — blocked
        let mut input = serde_json::json!({"command": "rm -rf /"});
        assert!(!registry.run_tool_call_hooks("bash", &mut input).await);

        // Non-bash — always allowed
        let mut input = serde_json::json!({"file_path": "test.txt"});
        assert!(registry.run_tool_call_hooks("read_file", &mut input).await);
    }

    #[tokio::test]
    async fn tool_call_hook_modifies_args() {
        let registry = HookRegistry::new();

        // Hook that adds --color=never to all bash commands
        registry
            .on_tool_call(Arc::new(|name: &str, input: &mut serde_json::Value| {
                if name == "bash" {
                    if let Some(cmd) = input.get("command").and_then(|v| v.as_str()) {
                        if !cmd.contains("--color") {
                            input["command"] =
                                serde_json::Value::String(format!("{cmd} --color=never"));
                        }
                    }
                }
                true
            }))
            .await;

        let mut input = serde_json::json!({"command": "ls -la"});
        registry.run_tool_call_hooks("bash", &mut input).await;
        assert_eq!(input["command"], "ls -la --color=never");
    }

    #[tokio::test]
    async fn multiple_hooks_chain() {
        let registry = HookRegistry::new();

        registry
            .on_context(Arc::new(|msgs: &mut Vec<ConversationMessage>| {
                msgs.push(ConversationMessage::user("[hook-1]"));
            }))
            .await;

        registry
            .on_context(Arc::new(|msgs: &mut Vec<ConversationMessage>| {
                msgs.push(ConversationMessage::user("[hook-2]"));
            }))
            .await;

        let mut messages = vec![];
        registry.run_context_hooks(&mut messages).await;
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].text_content(), "[hook-1]");
        assert_eq!(messages[1].text_content(), "[hook-2]");
    }

    #[tokio::test]
    async fn count_hooks() {
        let registry = HookRegistry::new();
        assert_eq!(registry.count().await, (0, 0));

        registry.on_context(Arc::new(|_| {})).await;
        registry.on_tool_call(Arc::new(|_, _| true)).await;
        registry.on_tool_call(Arc::new(|_, _| true)).await;

        assert_eq!(registry.count().await, (1, 2));
    }
}
