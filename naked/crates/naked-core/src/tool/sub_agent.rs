use std::path::Path;
use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use crate::agent_registry::{AgentRegistry, AgentStatus};
use crate::history::ConversationHistory;
use crate::loop_::{AgentLoop, LoopConfig};
use crate::provider::Provider;
use crate::tool::bash::BashTool;
use crate::tool::file_ops::{EditFileTool, ReadFileTool, WriteFileTool};
use crate::tool::registry::ToolRegistry;
use crate::tool::search::{GlobSearchTool, GrepSearchTool};
use crate::tool::web_search::WebSearchTool;
use crate::types::{
    AgentEvent, ContentBlock, Permission, Role, SubAgentEvent, ToolResult, ToolSpec,
};

use super::Tool;

const SUB_AGENT_MAX_ITERATIONS: usize = 30;

pub struct SubAgentTool {
    provider: Arc<dyn Provider>,
    model: String,
    tool_timeout_secs: u64,
    exa_keys: Vec<String>,
    registry: AgentRegistry,
}

impl SubAgentTool {
    pub fn new(
        provider: Arc<dyn Provider>,
        model: String,
        tool_timeout_secs: u64,
        exa_keys: Vec<String>,
    ) -> Self {
        Self {
            provider,
            model,
            tool_timeout_secs,
            exa_keys,
            registry: AgentRegistry::new(),
        }
    }

    pub fn with_registry(mut self, registry: AgentRegistry) -> Self {
        self.registry = registry;
        self
    }

    pub fn registry(&self) -> &AgentRegistry {
        &self.registry
    }

    fn build_tools(&self, mode: &str) -> ToolRegistry {
        let tools: Vec<Box<dyn Tool>> = match mode {
            "explore" => vec![
                Box::new(ReadFileTool),
                Box::new(GlobSearchTool),
                Box::new(GrepSearchTool),
                Box::new(BashTool::new(self.tool_timeout_secs)),
                Box::new(WebSearchTool::from_legacy_exa(self.exa_keys.clone())),
            ],
            _ => vec![
                Box::new(BashTool::new(self.tool_timeout_secs)),
                Box::new(ReadFileTool),
                Box::new(WriteFileTool),
                Box::new(EditFileTool),
                Box::new(GlobSearchTool),
                Box::new(GrepSearchTool),
                Box::new(WebSearchTool::from_legacy_exa(self.exa_keys.clone())),
            ],
        };
        ToolRegistry::new(tools)
    }
}

#[async_trait]
impl Tool for SubAgentTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "sub_agent".into(),
            description: "Delegate a task to a child agent that can use tools independently. \
                          Use for exploration, analysis, or isolated subtasks. \
                          The child agent runs with its own context and returns a text result."
                .into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Task description for the sub-agent"
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["explore", "general"],
                        "description": "explore = read-only tools only (safe); general = full tool set (default: explore)"
                    }
                },
                "required": ["prompt"]
            }),
            permission: Permission::ReadOnly,
        }
    }

    fn effective_permission(&self, input: &serde_json::Value, _cwd: &Path) -> Permission {
        let mode = input
            .get("mode")
            .and_then(|v| v.as_str())
            .unwrap_or("explore");
        match mode {
            "general" => Permission::WorkspaceWrite,
            _ => Permission::ReadOnly,
        }
    }

    async fn execute(&self, input: serde_json::Value, cwd: &Path) -> ToolResult {
        let (drop_tx, _) = mpsc::channel(1);
        self.execute_with_progress(input, cwd, drop_tx).await
    }

    async fn execute_with_progress(
        &self,
        input: serde_json::Value,
        cwd: &Path,
        progress: mpsc::Sender<AgentEvent>,
    ) -> ToolResult {
        let prompt = match input.get("prompt").and_then(|v| v.as_str()) {
            Some(p) if !p.trim().is_empty() => p.trim().to_string(),
            _ => {
                return ToolResult {
                    output: "Error: 'prompt' field is required and must be non-empty".into(),
                    is_error: true,
                };
            }
        };

        let mode = input
            .get("mode")
            .and_then(|v| v.as_str())
            .unwrap_or("explore");

        let agent_id = format!(
            "sa-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis()
                % 100_000
        );

        let preview = if prompt.len() > 80 {
            format!("{}…", &prompt[..prompt.floor_char_boundary(80)])
        } else {
            prompt.clone()
        };

        let cancel = CancellationToken::new();
        self.registry
            .register(&agent_id, &preview, mode, cancel.clone())
            .await;

        let _ = progress
            .send(AgentEvent::SubAgentProgress {
                agent_id: agent_id.clone(),
                event: SubAgentEvent::Started {
                    prompt_preview: preview,
                },
            })
            .await;

        let tools = self.build_tools(mode);
        let config = LoopConfig {
            max_iterations: SUB_AGENT_MAX_ITERATIONS,
            cwd: cwd.to_path_buf(),
            model: self.model.clone(),
            max_tokens: 8192,
            temperature: Some(0.0),
            reasoning: None,
            provider: String::new(),
            health: None,
        };

        let system = format!(
            "You are a focused sub-agent. Complete the task and report your findings concisely.\n\
             Working directory: {}\n\
             Mode: {mode}",
            cwd.display()
        );

        let mut history = ConversationHistory::new(system);
        history.push_user(&prompt);

        let provider_box: Box<dyn Provider> = Box::new(ArcProvider(self.provider.clone()));
        let agent = AgentLoop::new(provider_box, tools, config);

        let (tx, mut rx) = mpsc::channel(256);

        // Forward child events to parent in real-time (not after completion).
        let fwd_progress = progress.clone();
        let fwd_agent_id = agent_id.clone();
        let fwd_registry = self.registry.clone();
        let (text_tx, mut text_rx) = mpsc::channel::<String>(256);
        let fwd_handle = tokio::spawn(async move {
            while let Some(event) = rx.recv().await {
                match &event {
                    AgentEvent::ToolStart { name, input, .. } => {
                        fwd_registry.update_tool(&fwd_agent_id, name).await;
                        let inp_preview = serde_json::to_string(input).unwrap_or_default();
                        let inp_short = if inp_preview.len() > 120 {
                            format!("{}…", &inp_preview[..inp_preview.floor_char_boundary(120)])
                        } else {
                            inp_preview
                        };
                        let _ = fwd_progress
                            .send(AgentEvent::SubAgentProgress {
                                agent_id: fwd_agent_id.clone(),
                                event: SubAgentEvent::ToolUse {
                                    name: name.clone(),
                                    input_preview: inp_short,
                                },
                            })
                            .await;
                    }
                    AgentEvent::ToolEnd { name, state, .. } => {
                        let _ = fwd_progress
                            .send(AgentEvent::SubAgentProgress {
                                agent_id: fwd_agent_id.clone(),
                                event: SubAgentEvent::ToolDone {
                                    name: name.clone(),
                                    state: state.clone(),
                                },
                            })
                            .await;
                    }
                    AgentEvent::TextDelta(t) => {
                        let _ = text_tx.send(t.clone()).await;
                        let _ = fwd_progress
                            .send(AgentEvent::SubAgentProgress {
                                agent_id: fwd_agent_id.clone(),
                                event: SubAgentEvent::TextDelta(t.clone()),
                            })
                            .await;
                    }
                    AgentEvent::ThinkingDelta(_) | AgentEvent::Heartbeat => {
                        let _ = fwd_progress.send(AgentEvent::Heartbeat).await;
                    }
                    _ => {}
                }
            }
        });

        let result = agent.run(&mut history, tx, cancel, None, None).await;

        // Wait for the forwarder to drain remaining events.
        drop(fwd_handle);
        let mut text_parts = Vec::new();
        while let Ok(t) = text_rx.try_recv() {
            text_parts.push(t);
        }

        match result {
            Ok(usage) => {
                self.registry
                    .finish(&agent_id, AgentStatus::Completed, usage.total_tokens())
                    .await;
                let _ = progress
                    .send(AgentEvent::SubAgentProgress {
                        agent_id: agent_id.clone(),
                        event: SubAgentEvent::Finished {
                            tokens: usage.total_tokens(),
                        },
                    })
                    .await;

                let mut output = if text_parts.is_empty() {
                    history
                        .messages()
                        .iter()
                        .rev()
                        .find(|m| m.role == Role::Assistant)
                        .map(|m| {
                            m.blocks
                                .iter()
                                .filter_map(|b| {
                                    if let ContentBlock::Text { text } = b {
                                        Some(text.as_str())
                                    } else {
                                        None
                                    }
                                })
                                .collect::<Vec<_>>()
                                .join("\n")
                        })
                        .unwrap_or_else(|| "(sub-agent produced no text output)".into())
                } else {
                    text_parts.join("")
                };

                output.push_str(&format!(
                    "\n\n[sub-agent: {} tokens, mode={mode}]",
                    usage.total_tokens(),
                ));

                ToolResult {
                    output,
                    is_error: false,
                }
            }
            Err(e) => {
                self.registry
                    .finish(&agent_id, AgentStatus::Failed(e.to_string()), 0)
                    .await;
                let _ = progress
                    .send(AgentEvent::SubAgentProgress {
                        agent_id: agent_id.clone(),
                        event: SubAgentEvent::Error(e.to_string()),
                    })
                    .await;
                ToolResult {
                    output: format!("Sub-agent error: {e}"),
                    is_error: true,
                }
            }
        }
    }
}

struct ArcProvider(Arc<dyn Provider>);

#[async_trait]
impl Provider for ArcProvider {
    fn name(&self) -> &str {
        self.0.name()
    }
    fn models(&self) -> Vec<crate::types::ModelInfo> {
        self.0.models()
    }
    async fn stream_chat(
        &self,
        request: crate::provider::ChatRequest,
    ) -> crate::error::Result<
        std::pin::Pin<Box<dyn tokio_stream::Stream<Item = crate::types::StreamChunk> + Send>>,
    > {
        self.0.stream_chat(request).await
    }
}
