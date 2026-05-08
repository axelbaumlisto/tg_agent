//! Cycle-restart logic extracted from `AgentLoop::run`.

use crate::error::Result;
use crate::history::ConversationHistory;
use crate::types::AgentEvent;
use tokio::sync::mpsc;

impl super::AgentLoop {
    pub(super) async fn advance_cycle_if_needed(
        &self,
        history: &mut ConversationHistory,
        tx: &mpsc::Sender<AgentEvent>,
    ) -> Result<bool> {
        let cycle_cfg = match self.config.cycle_config.as_ref() {
            Some(c) => c,
            None => return Ok(false),
        };
        let est = history.estimated_tokens() as u64;
        if !crate::session::cycle::should_advance_cycle(est, cycle_cfg) {
            return Ok(false);
        }
        let session_id = self.config.session_id.as_deref().unwrap_or("unknown");
        let cycle_num = history.cycle_count();
        let checkpoint = crate::session::cycle::build_checkpoint(
            cycle_num,
            history.messages(),
            est,
            None, // TODO: pass working_set when available
            cycle_cfg,
        );
        if let Ok(archive_path) =
            self.config
                .cycle_archiver
                .archive(session_id, cycle_num, history.messages())
        {
            let restart_prompt =
                crate::session::cycle::build_restart_prompt(&checkpoint, history.system_prompt());
            let archived_count = history.message_count();
            history.clear_for_cycle_restart(&restart_prompt);
            self.config.observer.on_cycle_restart(
                cycle_num as u64,
                archived_count,
                &archive_path.display().to_string(),
            );
            let _ = tx
                .send(AgentEvent::CycleRestarted {
                    cycle_number: cycle_num,
                    archived_messages: archived_count,
                    archive_path: archive_path.display().to_string(),
                })
                .await;
            return Ok(true);
        }
        Ok(false)
    }
}
