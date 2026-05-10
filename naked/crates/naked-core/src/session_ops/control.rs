//! Session control — abort, compact_session, compaction pipeline.

use crate::AgentCore;
use crate::config::EffectiveSessionConfig;
use crate::error::{AgentError, Result};
use crate::memory;
use crate::provider::{self, Provider};
use crate::session::SessionState;
use crate::types::{self, AgentEvent};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

impl AgentCore {
    /// Structured compaction prompt (initial or iterative update).
    pub(crate) fn compaction_prompt(previous_summary: Option<&str>) -> String {
        if let Some(prev) = previous_summary {
            format!(
                "<previous-summary>\n{prev}\n</previous-summary>\n\n\
                 The messages above are NEW conversation since the last summary. \
                 Update the existing summary with new information.\n\
                 RULES: PRESERVE existing info. ADD new progress/decisions. \
                 Move In Progress → Done when completed. UPDATE Next Steps.\n\n\
                 {}",
                Self::COMPACTION_FORMAT
            )
        } else {
            format!(
                "Summarize the conversation above into a structured checkpoint.\n\n{}",
                Self::COMPACTION_FORMAT
            )
        }
    }

    pub(crate) const COMPACTION_FORMAT: &'static str = "\
Use this EXACT format:\n\n\
## Goal\n\
[What is the user trying to accomplish?]\n\n\
## Constraints & Preferences\n\
- [Any constraints or preferences mentioned]\n\n\
## Progress\n\
### Done\n\
- [x] [Completed tasks]\n\n\
### In Progress\n\
- [ ] [Current work]\n\n\
### Blocked\n\
- [Issues if any]\n\n\
## Key Decisions\n\
- **[Decision]**: [Rationale]\n\n\
## Next Steps\n\
1. [What should happen next]\n\n\
## Critical Context\n\
- [File paths, function names, error messages needed to continue]\n\n\
Keep each section concise. Preserve exact paths and identifiers.";

    /// Call the current model to summarize conversation for compaction.
    async fn llm_summarize(
        provider: &dyn Provider,
        model: &str,
        conversation_text: &str,
        previous_summary: Option<&str>,
    ) -> Result<String> {
        use tokio_stream::StreamExt;

        const MAX_COMPACTION_INPUT: usize = 16_000;
        // UTF-8 safe truncation — conversation_text routinely contains
        // multi-byte text (Russian, Vietnamese, emoji), and a raw byte
        // slice panics inside a codepoint. Allocating a new String here is
        // cheap relative to the model call that follows.
        let truncated_owned;
        let input: &str = if conversation_text.chars().count() > MAX_COMPACTION_INPUT {
            truncated_owned = conversation_text
                .chars()
                .take(MAX_COMPACTION_INPUT)
                .collect::<String>();
            truncated_owned.as_str()
        } else {
            conversation_text
        };

        tracing::info!(
            input_chars = input.len(),
            "LLM compaction: sending to model"
        );

        let system = "You are a context compaction assistant. Create a structured summary \
            that another LLM will use to continue the work. Be concise, preserve exact file paths, \
            function names, and error messages. Respond in the same language the user used.";

        let user_prompt = format!(
            "<conversation>\n{input}\n</conversation>\n\n\
             {}",
            Self::compaction_prompt(previous_summary)
        );

        let request = provider::ChatRequest {
            model: model.to_string(),
            system: system.to_string(),
            messages: vec![serde_json::json!({
                "role": "user",
                "content": user_prompt,
            })],
            tools: vec![],
            max_tokens: 2048,
            temperature: Some(0.0),
            reasoning: None,
        };

        let mut stream = provider.stream_chat(request).await.map_err(|e| {
            AgentError::ProviderTyped(crate::provider::error::ProviderError::Other {
                status: 0,
                body: format!("compaction LLM: {e}"),
            })
        })?;

        let mut text = String::new();
        while let Some(chunk) = stream.next().await {
            match chunk {
                types::StreamChunk::Text(t) => text.push_str(&t),
                types::StreamChunk::Done => break,
                types::StreamChunk::Error(e) => {
                    return Err(AgentError::ProviderTyped(
                        crate::provider::error::ProviderError::Other {
                            status: 0,
                            body: format!("compaction stream: {e}"),
                        },
                    ));
                }
                _ => {}
            }
        }

        if text.trim().is_empty() {
            return Err(AgentError::ProviderTyped(
                crate::provider::error::ProviderError::Other {
                    status: 0,
                    body: "compaction LLM returned empty".into(),
                },
            ));
        }

        Ok(text)
    }

    /// Phase 2: run LLM compaction, re-acquire lock, apply result, prepare
    /// history + loop config. Drops lock before returning.
    /// `pub(crate)` so `dispatch_turn` in the `turn` submodule can call it.
    pub(crate) async fn compact_and_prepare(
        &self,
        session_id: &str,
        setup: &crate::turn::TurnSetup,
        effective: &EffectiveSessionConfig,
        tx: &mpsc::Sender<AgentEvent>,
    ) -> Result<crate::turn::TurnSpawnData> {
        let llm_summary = self
            .run_llm_compaction(&setup.compaction_input, &setup.provider_name, &setup.model)
            .await;

        let mut sessions = self.ss.sessions.write().await;
        let session = sessions
            .get_mut(session_id)
            .ok_or_else(|| AgentError::SessionNotFound(session_id.to_string()))?;
        self.apply_compaction_and_gc(
            session,
            session_id,
            &setup.compaction_input,
            llm_summary.as_deref(),
            tx,
        )
        .await;

        let session_root = self.ss.store.session_root(session_id);
        let sender = self.session_sender(session_id).await;
        let (history, original_system_prompt) = crate::turn::prepare_history(
            session,
            &session_root,
            effective,
            sender.as_deref(),
            &self.config().memory,
        )
        .await;

        let artifacts = self.ss.store.artifacts_dir(session_id);
        if let Err(e) = tokio::fs::create_dir_all(&artifacts).await {
            tracing::warn!("could not create artifacts dir: {e}");
        }
        let cwd = if session.workspace.as_os_str().is_empty() || !session.workspace.exists() {
            artifacts
        } else {
            session.workspace.clone()
        };

        let (eff_max_tokens, eff_temperature) =
            crate::turn::resolve_generation_params(&self.config(), &setup.provider_name, effective);
        let mut loop_config = crate::turn::build_loop_config(
            effective.max_iterations,
            cwd.clone(),
            setup.model.clone(),
            eff_max_tokens,
            eff_temperature,
            effective.reasoning.clone(),
            setup.provider_name.clone(),
            self.provider_svc.health(),
            self.token_tracker.clone(),
            &self.config().session_dir,
            session_id,
        );
        // PLAN_QUALITY_v1 wiring: copy installed managers from
        // AgentCore (Some(...) when the bot has called set_lsp /
        // set_lifecycle_hooks / set_permissions; None otherwise).
        if let Ok(g) = self.lsp.read() {
            loop_config.lsp = g.clone();
        }
        if let Ok(g) = self.lifecycle_hooks.read() {
            loop_config.lifecycle_hooks = g.clone();
        }
        if let Ok(g) = self.permissions.read() {
            loop_config.permissions = g.clone();
        }
        let session_workspace = session.workspace.clone();

        let cancel = CancellationToken::new();
        self.ss
            .cancels
            .write()
            .await
            .insert(session_id.to_string(), cancel.clone());
        drop(sessions);

        Ok(crate::turn::TurnSpawnData {
            history,
            original_system_prompt,
            session_workspace,
            loop_config,
        })
    }

    /// Trigger history compaction for a session. Returns (before, after) message counts.
    /// No-op if compaction not needed.
    pub async fn compact_session(&self, session_id: &str) -> Option<(usize, usize)> {
        if let Some(session) = self.ss.sessions.write().await.get_mut(session_id) {
            session.history.auto_compact()
        } else {
            None
        }
    }

    pub async fn abort(&self, session_id: &str) {
        if let Some(cancel) = self.ss.cancels.read().await.get(session_id) {
            cancel.cancel();
        }
        if let Some(session) = self.ss.sessions.write().await.get_mut(session_id) {
            session.state = SessionState::Idle;
        }
    }

    /// Run LLM-based compaction if needed, returning the summary.
    async fn run_llm_compaction(
        &self,
        ci: &crate::turn::CompactionInput,
        provider_name: &str,
        model: &str,
    ) -> Option<String> {
        let text_for_llm = ci.compact_text.as_deref()?;
        tracing::info!(ci.before_msgs, "attempting LLM-based compaction");
        let provider_arc = self.provider_for(provider_name).await;

        if self.config().memory.daily_enabled && self.config().memory.pre_compaction_flush {
            memory::digest::pre_compaction_flush(
                &*provider_arc,
                model,
                &ci.workspace,
                &memory::types::MemoryScope::Project,
                text_for_llm,
            )
            .await;
        }

        match Self::llm_summarize(
            &*provider_arc,
            model,
            text_for_llm,
            ci.previous_summary.as_deref(),
        )
        .await
        {
            Ok(mut summary) => {
                crate::turn::append_file_tags(&mut summary, &ci.read_files, &ci.modified_files);
                tracing::info!("LLM compaction succeeded");
                Some(summary)
            }
            Err(e) => {
                tracing::warn!("LLM compaction failed, falling back to deterministic: {e}");
                None
            }
        }
    }

    /// Apply compaction to session history and GC orphan image artifacts.
    async fn apply_compaction_and_gc(
        &self,
        session: &mut crate::session::Session,
        session_id: &str,
        ci: &crate::turn::CompactionInput,
        llm_summary: Option<&str>,
        tx: &mpsc::Sender<AgentEvent>,
    ) {
        let compacted = if ci.needs_compact {
            crate::turn::apply_compaction(session, llm_summary, ci.before_msgs)
        } else {
            None
        };

        if let Some((before, after)) = compacted {
            let summary_hint = llm_summary.and_then(crate::turn::extract_summary_hint);
            let files_count = ci.read_files.len() + ci.modified_files.len();
            tracing::info!("context compacted: {before} msgs -> {after} msgs");
            let _ = tx
                .send(AgentEvent::ContextCompacted {
                    before_msgs: before,
                    after_msgs: after,
                    summary_hint,
                    files_count,
                })
                .await;
            match self.ss.store.gc_orphan_image_artifacts(session_id).await {
                Ok(n) if n > 0 => {
                    tracing::info!(removed = n, session = session_id, "post-compaction GC");
                }
                Ok(_) => {}
                Err(e) => {
                    tracing::warn!(session = session_id, "post-compaction GC failed: {e}");
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::AgentCore;

    #[test]
    fn compaction_prompt_fresh_has_format_instructions() {
        let prompt = AgentCore::compaction_prompt(None);
        assert!(prompt.contains("## Goal"), "missing Goal section");
        assert!(prompt.contains("## Progress"), "missing Progress section");
        assert!(
            prompt.contains("Summarize"),
            "missing Summarize instruction"
        );
        assert!(!prompt.contains("<previous-summary>"));
    }

    #[test]
    fn compaction_prompt_with_previous_includes_it() {
        let prompt = AgentCore::compaction_prompt(Some("old summary here"));
        assert!(prompt.contains("<previous-summary>"));
        assert!(prompt.contains("old summary here"));
        assert!(prompt.contains("Update the existing summary"));
        assert!(prompt.contains("## Goal"));
    }

    #[test]
    fn compaction_format_has_all_sections() {
        let fmt = AgentCore::COMPACTION_FORMAT;
        for section in [
            "## Goal",
            "## Progress",
            "### Done",
            "### In Progress",
            "## Next Steps",
        ] {
            assert!(fmt.contains(section), "missing {section}");
        }
    }
}
