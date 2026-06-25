//! Live Run registry for Telegram streams.
//!
//! CYCLE 2 Step 2: this is a **kind-agnostic** in-memory UI/control index.
//! It is not a research ledger, not a scheduler, and not a durable resume
//! mechanism. Research may map `Inflight.attempt_id` into `run_id`, but the
//! registry itself routes Runs by `run_id` regardless of [`RunKind`].

use std::collections::{HashMap, HashSet};
use std::sync::{Mutex, MutexGuard};
use std::time::{Duration, Instant};

use naked_core::types::SteerMessage;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

pub type RunId = String;
pub type SessionId = String;
pub type SourceRef = String;

pub const STEP2_SINGLE_RUN_CAP: usize = 1;
pub const MULTI_RUN_THREAD_CAP: usize = 3;
pub const MIRROR_SINK_CAP: usize = 2;
pub const ACTIVE_RUN_SILENCE_WARN_THRESHOLD: Duration = Duration::from_secs(120);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SilenceState {
    pub silence_age: Duration,
    pub alert: bool,
}

pub fn run_silence_state(
    last_progress: Instant,
    now: Instant,
    threshold: Duration,
) -> SilenceState {
    let silence_age = now.saturating_duration_since(last_progress);
    SilenceState {
        silence_age,
        alert: silence_age > threshold,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ChatThreadKey {
    pub chat_id: i64,
    pub thread_id: Option<i32>,
}

impl ChatThreadKey {
    pub const fn new(chat_id: i64, thread_id: Option<i32>) -> Self {
        Self { chat_id, thread_id }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MessageKey {
    pub chat_id: i64,
    pub message_id: i32,
}

impl MessageKey {
    pub const fn new(chat_id: i64, message_id: i32) -> Self {
        Self {
            chat_id,
            message_id,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunOrigin {
    pub chat_id: i64,
    pub thread_id: Option<i32>,
}

impl RunOrigin {
    pub const fn new(chat_id: i64, thread_id: Option<i32>) -> Self {
        Self { chat_id, thread_id }
    }

    pub const fn key(&self) -> ChatThreadKey {
        ChatThreadKey::new(self.chat_id, self.thread_id)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RunKind {
    ChatTurn,
    Research {
        spec_id: String,
    },
    /// Reserved for a future agent kind. CYCLE 2 builds the substrate only;
    /// it does not implement sub-agent/code-agent behaviour.
    SubAgent {
        label: String,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunStatus {
    Streaming,
    AwaitingTool,
    Idle,
    Done,
    Failed,
    Aborted,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct RenderedRunState {
    pub live_html: String,
    pub final_html: Option<String>,
    pub status_line: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunSink {
    pub chat_id: i64,
    pub thread_id: Option<i32>,
    pub message_id: i32,
}

impl RunSink {
    pub const fn new(chat_id: i64, thread_id: Option<i32>, message_id: i32) -> Self {
        Self {
            chat_id,
            thread_id,
            message_id,
        }
    }

    pub const fn message_key(&self) -> MessageKey {
        MessageKey::new(self.chat_id, self.message_id)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MovedRunNotice {
    pub run_id: RunId,
    pub new_origin: RunOrigin,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MoveRunPlan {
    pub run_id: RunId,
    pub session_id: SessionId,
    pub old_origin: RunOrigin,
    pub new_origin: RunOrigin,
    pub old_bubble: Option<RunSink>,
    pub summary: RunSummary,
}

#[derive(Debug, Clone)]
pub struct RunHandle {
    pub run_id: RunId,
    pub session_id: SessionId,
    pub origin: RunOrigin,
    pub kind: RunKind,
    /// Generic source reference. Internally indexed as `by_spec_id` for the
    /// current plan, but callers should treat this as `by_source_ref`.
    pub source_ref: Option<SourceRef>,
    pub status: RunStatus,
    pub steer: mpsc::Sender<SteerMessage>,
    pub abort: CancellationToken,
    pub bubble_message_id: Option<i32>,
    pub rendered: RenderedRunState,
    last_progress: Instant,
    silence_warned: bool,
    mirror_sinks: Vec<RunSink>,
    bound_messages: HashSet<MessageKey>,
}

impl RunHandle {
    pub fn new(input: RegisterRunInput, run_id: RunId) -> Self {
        Self {
            run_id,
            session_id: input.session_id,
            origin: input.origin,
            kind: input.kind,
            source_ref: input.source_ref,
            status: RunStatus::Streaming,
            steer: input.steer,
            abort: input.abort,
            bubble_message_id: None,
            rendered: RenderedRunState::default(),
            last_progress: Instant::now(),
            silence_warned: false,
            mirror_sinks: Vec::new(),
            bound_messages: HashSet::new(),
        }
    }

    fn summary(&self) -> RunSummary {
        RunSummary {
            run_id: self.run_id.clone(),
            session_id: self.session_id.clone(),
            origin: self.origin.clone(),
            kind: self.kind.clone(),
            source_ref: self.source_ref.clone(),
            status: self.status,
            bubble_message_id: self.bubble_message_id,
            rendered: self.rendered.clone(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct RunControl {
    pub run_id: RunId,
    pub session_id: SessionId,
    pub steer: mpsc::Sender<SteerMessage>,
    pub abort: CancellationToken,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunSummary {
    pub run_id: RunId,
    pub session_id: SessionId,
    pub origin: RunOrigin,
    pub kind: RunKind,
    pub source_ref: Option<SourceRef>,
    pub status: RunStatus,
    pub bubble_message_id: Option<i32>,
    pub rendered: RenderedRunState,
}

#[derive(Debug, Clone)]
pub struct RegisterRunInput {
    /// Optional caller-provided id. Research maps `Inflight.attempt_id` here;
    /// other agent kinds leave it empty and let the registry generate one.
    pub requested_run_id: Option<RunId>,
    pub session_id: SessionId,
    pub origin: RunOrigin,
    pub kind: RunKind,
    pub source_ref: Option<SourceRef>,
    pub steer: mpsc::Sender<SteerMessage>,
    pub abort: CancellationToken,
}

#[derive(Debug, Clone)]
pub struct RegisterRunOptions {
    pub max_runs_per_thread: usize,
}

impl RegisterRunOptions {
    pub const fn step2_single_run() -> Self {
        Self {
            max_runs_per_thread: STEP2_SINGLE_RUN_CAP,
        }
    }

    pub const fn cap_three() -> Self {
        Self {
            max_runs_per_thread: MULTI_RUN_THREAD_CAP,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RegisterRunError {
    DuplicateRunId {
        run_id: RunId,
    },
    DuplicateSessionId {
        session_id: SessionId,
        existing: RunId,
    },
    DuplicateSourceRef {
        source_ref: SourceRef,
        existing: RunId,
    },
    ThreadCapacityExceeded {
        key: ChatThreadKey,
        active: usize,
        cap: usize,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BindMessageError {
    RunNotFound { run_id: RunId },
    MessageAlreadyBound { key: MessageKey, existing: RunId },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MirrorSinkError {
    RunNotFound {
        run_id: RunId,
    },
    SinkCapacityExceeded {
        run_id: RunId,
        active: usize,
        cap: usize,
    },
    MessageAlreadyBound {
        key: MessageKey,
        existing: RunId,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MoveRunError {
    RunNotFound {
        run_id: RunId,
    },
    ThreadCapacityExceeded {
        key: ChatThreadKey,
        active: usize,
        cap: usize,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResolveForSteer {
    NotFound,
    Unique(RunSummary),
    Ambiguous(Vec<RunSummary>),
}

#[derive(Debug, Default)]
struct RunRegistryState {
    by_run: HashMap<RunId, RunHandle>,
    by_chat_thread: HashMap<ChatThreadKey, Vec<RunId>>,
    by_spec_id: HashMap<SourceRef, RunId>,
    by_message: HashMap<MessageKey, RunId>,
    by_session: HashMap<SessionId, RunId>,
    moved_messages: HashMap<MessageKey, MovedRunNotice>,
    next_generated: u64,
}

#[derive(Debug, Default)]
pub struct RunRegistry {
    state: Mutex<RunRegistryState>,
}

impl RunRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register_run(
        &self,
        input: RegisterRunInput,
        options: RegisterRunOptions,
    ) -> Result<RunSummary, RegisterRunError> {
        let mut state = self.lock_state();
        let run_id = match input.requested_run_id.clone() {
            Some(id) => id,
            None => state.generate_run_id(),
        };
        if state.by_run.contains_key(&run_id) {
            return Err(RegisterRunError::DuplicateRunId { run_id });
        }
        if let Some(existing) = state.by_session.get(&input.session_id) {
            return Err(RegisterRunError::DuplicateSessionId {
                session_id: input.session_id,
                existing: existing.clone(),
            });
        }
        if let Some(source_ref) = input.source_ref.as_ref()
            && let Some(existing) = state.by_spec_id.get(source_ref)
        {
            return Err(RegisterRunError::DuplicateSourceRef {
                source_ref: source_ref.clone(),
                existing: existing.clone(),
            });
        }
        let thread_key = input.origin.key();
        let active = state
            .by_chat_thread
            .get(&thread_key)
            .map_or(0, std::vec::Vec::len);
        let cap = options.max_runs_per_thread;
        if active >= cap {
            return Err(RegisterRunError::ThreadCapacityExceeded {
                key: thread_key,
                active,
                cap,
            });
        }

        let handle = RunHandle::new(input, run_id.clone());
        let summary = handle.summary();
        state
            .by_chat_thread
            .entry(thread_key)
            .or_default()
            .push(run_id.clone());
        if let Some(source_ref) = handle.source_ref.clone() {
            state.by_spec_id.insert(source_ref, run_id.clone());
        }
        state
            .by_session
            .insert(handle.session_id.clone(), run_id.clone());
        state.by_run.insert(run_id, handle);
        Ok(summary)
    }

    pub fn bind_message(
        &self,
        run_id: &str,
        key: MessageKey,
    ) -> Result<RunSummary, BindMessageError> {
        let mut state = self.lock_state();
        if let Some(existing) = state.by_message.get(&key)
            && existing != run_id
        {
            return Err(BindMessageError::MessageAlreadyBound {
                key,
                existing: existing.clone(),
            });
        }
        let handle = state
            .by_run
            .get_mut(run_id)
            .ok_or_else(|| BindMessageError::RunNotFound {
                run_id: run_id.to_string(),
            })?;
        handle.bubble_message_id = Some(key.message_id);
        handle.bound_messages.insert(key);
        let summary = handle.summary();
        state.by_message.insert(key, run_id.to_string());
        Ok(summary)
    }

    pub fn move_run(
        &self,
        run_id: &str,
        new_origin: RunOrigin,
        options: RegisterRunOptions,
    ) -> Result<MoveRunPlan, MoveRunError> {
        let mut state = self.lock_state();
        let (old_origin, old_bubble, session_id) = {
            let handle = state
                .by_run
                .get(run_id)
                .ok_or_else(|| MoveRunError::RunNotFound {
                    run_id: run_id.to_string(),
                })?;
            (
                handle.origin.clone(),
                handle.bubble_message_id.map(|message_id| {
                    RunSink::new(handle.origin.chat_id, handle.origin.thread_id, message_id)
                }),
                handle.session_id.clone(),
            )
        };
        let old_key = old_origin.key();
        let new_key = new_origin.key();
        if old_key != new_key {
            let active = state
                .by_chat_thread
                .get(&new_key)
                .map_or(0, std::vec::Vec::len);
            let cap = options.max_runs_per_thread;
            if active >= cap {
                return Err(MoveRunError::ThreadCapacityExceeded {
                    key: new_key,
                    active,
                    cap,
                });
            }
            remove_run_id_from_thread(&mut state.by_chat_thread, old_key, run_id);
            state
                .by_chat_thread
                .entry(new_key)
                .or_default()
                .push(run_id.to_string());
        }
        if let Some(old_bubble) = old_bubble.as_ref() {
            state.moved_messages.insert(
                old_bubble.message_key(),
                MovedRunNotice {
                    run_id: run_id.to_string(),
                    new_origin: new_origin.clone(),
                },
            );
        }
        let handle = state
            .by_run
            .get_mut(run_id)
            .expect("run exists after pre-check");
        handle.origin = new_origin.clone();
        let summary = handle.summary();
        Ok(MoveRunPlan {
            run_id: run_id.to_string(),
            session_id,
            old_origin,
            new_origin,
            old_bubble,
            summary,
        })
    }

    pub fn remove_run(&self, run_id: &str) -> Option<RunSummary> {
        let mut state = self.lock_state();
        let handle = state.by_run.remove(run_id)?;
        remove_run_id_from_thread(&mut state.by_chat_thread, handle.origin.key(), run_id);
        if let Some(source_ref) = handle.source_ref.as_ref()
            && state
                .by_spec_id
                .get(source_ref)
                .is_some_and(|id| id == run_id)
        {
            state.by_spec_id.remove(source_ref);
        }
        if state
            .by_session
            .get(&handle.session_id)
            .is_some_and(|id| id == run_id)
        {
            state.by_session.remove(&handle.session_id);
        }
        for key in &handle.bound_messages {
            if state.by_message.get(key).is_some_and(|id| id == run_id) {
                state.by_message.remove(key);
            }
            if state
                .moved_messages
                .get(key)
                .is_some_and(|notice| notice.run_id == run_id)
            {
                state.moved_messages.remove(key);
            }
        }
        Some(handle.summary())
    }

    pub fn resolve_for_steer(&self, key: ChatThreadKey) -> ResolveForSteer {
        let state = self.lock_state();
        let summaries = state.summaries_for_thread(key);
        match summaries.len() {
            0 => ResolveForSteer::NotFound,
            1 => ResolveForSteer::Unique(summaries[0].clone()),
            _ => ResolveForSteer::Ambiguous(summaries),
        }
    }

    pub fn list_for_thread(&self, key: ChatThreadKey) -> Vec<RunSummary> {
        self.lock_state().summaries_for_thread(key)
    }

    /// Returns true while a run with this generic source reference is registered.
    ///
    /// `by_spec_id` is removed in `remove_run` on completion/abort/failure, so
    /// membership here means the run is currently active.
    pub fn has_active_run_for_source_ref(&self, source_ref: &str) -> bool {
        self.lock_state().by_spec_id.contains_key(source_ref)
    }

    pub fn get_run(&self, run_id: &str) -> Option<RunSummary> {
        self.lock_state().by_run.get(run_id).map(RunHandle::summary)
    }

    pub fn control_for_run(&self, run_id: &str) -> Option<RunControl> {
        self.lock_state().by_run.get(run_id).map(|h| RunControl {
            run_id: h.run_id.clone(),
            session_id: h.session_id.clone(),
            steer: h.steer.clone(),
            abort: h.abort.clone(),
        })
    }

    pub fn resolve_message(&self, key: MessageKey) -> Option<RunSummary> {
        let state = self.lock_state();
        let run_id = state.by_message.get(&key)?;
        state.by_run.get(run_id).map(RunHandle::summary)
    }

    pub fn moved_notice_for_message(
        &self,
        key: MessageKey,
        expected_run_id: Option<&str>,
    ) -> Option<MovedRunNotice> {
        let state = self.lock_state();
        let notice = state.moved_messages.get(&key)?;
        if expected_run_id.is_none_or(|run_id| run_id == notice.run_id) {
            Some(notice.clone())
        } else {
            None
        }
    }

    pub fn add_mirror_sink(
        &self,
        run_id: &str,
        sink: RunSink,
    ) -> Result<RunSummary, MirrorSinkError> {
        let mut state = self.lock_state();
        let key = sink.message_key();
        if let Some(existing) = state.by_message.get(&key)
            && existing != run_id
        {
            return Err(MirrorSinkError::MessageAlreadyBound {
                key,
                existing: existing.clone(),
            });
        }
        let handle = state
            .by_run
            .get_mut(run_id)
            .ok_or_else(|| MirrorSinkError::RunNotFound {
                run_id: run_id.to_string(),
            })?;
        if handle.mirror_sinks.len() >= MIRROR_SINK_CAP
            && !handle.mirror_sinks.iter().any(|existing| existing == &sink)
        {
            return Err(MirrorSinkError::SinkCapacityExceeded {
                run_id: run_id.to_string(),
                active: handle.mirror_sinks.len(),
                cap: MIRROR_SINK_CAP,
            });
        }
        if !handle.mirror_sinks.iter().any(|existing| existing == &sink) {
            handle.mirror_sinks.push(sink);
        }
        handle.bound_messages.insert(key);
        let summary = handle.summary();
        state.by_message.insert(key, run_id.to_string());
        Ok(summary)
    }

    pub fn remove_mirror_sink(&self, run_id: &str, key: MessageKey) -> Option<RunSummary> {
        let mut state = self.lock_state();
        let summary = {
            let handle = state.by_run.get_mut(run_id)?;
            handle.mirror_sinks.retain(|sink| sink.message_key() != key);
            handle.bound_messages.remove(&key);
            handle.summary()
        };
        if state.by_message.get(&key).is_some_and(|id| id == run_id) {
            state.by_message.remove(&key);
        }
        if state
            .moved_messages
            .get(&key)
            .is_some_and(|notice| notice.run_id == run_id)
        {
            state.moved_messages.remove(&key);
        }
        Some(summary)
    }

    pub fn primary_sink(&self, run_id: &str) -> Option<RunSink> {
        self.lock_state().by_run.get(run_id).and_then(|handle| {
            handle.bubble_message_id.map(|message_id| {
                RunSink::new(handle.origin.chat_id, handle.origin.thread_id, message_id)
            })
        })
    }

    pub fn mirror_sinks(&self, run_id: &str) -> Vec<RunSink> {
        self.lock_state()
            .by_run
            .get(run_id)
            .map(|h| h.mirror_sinks.clone())
            .unwrap_or_default()
    }

    pub fn bound_message_keys(&self, run_id: &str) -> Vec<MessageKey> {
        self.lock_state()
            .by_run
            .get(run_id)
            .map(|h| h.bound_messages.iter().copied().collect())
            .unwrap_or_default()
    }

    pub fn update_rendered_state(
        &self,
        run_id: &str,
        rendered: RenderedRunState,
    ) -> Option<RunSummary> {
        let mut state = self.lock_state();
        let handle = state.by_run.get_mut(run_id)?;
        handle.rendered = rendered;
        Some(handle.summary())
    }

    pub fn update_status(&self, run_id: &str, status: RunStatus) -> Option<RunSummary> {
        let mut state = self.lock_state();
        let handle = state.by_run.get_mut(run_id)?;
        handle.status = status;
        Some(handle.summary())
    }

    pub fn mark_run_progress(&self, run_id: &str) -> Option<RunSummary> {
        self.mark_run_progress_at(run_id, Instant::now())
    }

    pub fn mark_run_progress_at(&self, run_id: &str, now: Instant) -> Option<RunSummary> {
        let mut state = self.lock_state();
        let handle = state.by_run.get_mut(run_id)?;
        handle.last_progress = now;
        handle.silence_warned = false;
        Some(handle.summary())
    }

    pub fn active_run_max_silent_seconds(&self) -> u64 {
        self.active_run_max_silent_seconds_at(Instant::now())
    }

    pub fn active_run_max_silent_seconds_at(&self, now: Instant) -> u64 {
        let state = self.lock_state();
        state
            .by_run
            .values()
            .map(|handle| run_silence_state(handle.last_progress, now, Duration::ZERO).silence_age)
            .max()
            .unwrap_or_default()
            .as_secs()
    }

    pub fn warn_silent_runs(&self, threshold: Duration) {
        self.warn_silent_runs_at(Instant::now(), threshold);
    }

    pub fn warn_silent_runs_at(&self, now: Instant, threshold: Duration) {
        let warnings = {
            let mut state = self.lock_state();
            let mut warnings = Vec::new();
            for handle in state.by_run.values_mut() {
                let silence = run_silence_state(handle.last_progress, now, threshold);
                if silence.alert && !handle.silence_warned {
                    handle.silence_warned = true;
                    warnings.push((
                        handle.run_id.clone(),
                        handle.session_id.clone(),
                        handle.kind.clone(),
                        silence.silence_age.as_secs(),
                    ));
                }
            }
            warnings
        };
        for (run_id, session_id, kind, silent_seconds) in warnings {
            tracing::warn!(
                run_id = %run_id,
                session_id = %session_id,
                ?kind,
                silent_seconds,
                threshold_seconds = threshold.as_secs(),
                "active run has emitted no non-heartbeat progress past threshold"
            );
        }
    }

    #[cfg(test)]
    fn contains_run(&self, run_id: &str) -> bool {
        self.lock_state().by_run.contains_key(run_id)
    }

    #[cfg(test)]
    fn contains_session(&self, session_id: &str) -> bool {
        self.lock_state().by_session.contains_key(session_id)
    }

    #[cfg(test)]
    async fn remove_run_after_test_seam(
        &self,
        run_id: &str,
        parked_at_seam: std::sync::Arc<tokio::sync::Notify>,
        release_cleanup: std::sync::Arc<tokio::sync::Notify>,
    ) -> Option<RunSummary> {
        {
            let state = self.lock_state();
            assert!(
                state.by_run.contains_key(run_id),
                "test seam must discover target before pausing"
            );
        }
        parked_at_seam.notify_one();
        release_cleanup.notified().await;
        self.remove_run(run_id)
    }

    #[cfg(test)]
    fn source_ref_count(&self) -> usize {
        self.lock_state().by_spec_id.len()
    }

    fn lock_state(&self) -> MutexGuard<'_, RunRegistryState> {
        self.state
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
    }
}

impl RunRegistryState {
    fn generate_run_id(&mut self) -> RunId {
        self.next_generated = self.next_generated.saturating_add(1);
        format!("r{:016x}", self.next_generated)
    }

    fn summaries_for_thread(&self, key: ChatThreadKey) -> Vec<RunSummary> {
        self.by_chat_thread
            .get(&key)
            .into_iter()
            .flat_map(|ids| ids.iter())
            .filter_map(|id| self.by_run.get(id).map(RunHandle::summary))
            .collect()
    }
}

fn remove_run_id_from_thread(
    by_chat_thread: &mut HashMap<ChatThreadKey, Vec<RunId>>,
    key: ChatThreadKey,
    run_id: &str,
) {
    if let Some(ids) = by_chat_thread.get_mut(&key) {
        ids.retain(|id| id != run_id);
        if ids.is_empty() {
            by_chat_thread.remove(&key);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use tokio::sync::Notify;

    fn input(
        run_id: &str,
        session_id: &str,
        chat_id: i64,
        source_ref: Option<&str>,
    ) -> RegisterRunInput {
        let (steer, _rx) = mpsc::channel(4);
        RegisterRunInput {
            requested_run_id: Some(run_id.to_string()),
            session_id: session_id.to_string(),
            origin: RunOrigin::new(chat_id, Some(10)),
            kind: match source_ref {
                Some(spec_id) => RunKind::Research {
                    spec_id: spec_id.to_string(),
                },
                None => RunKind::ChatTurn,
            },
            source_ref: source_ref.map(str::to_string),
            steer,
            abort: CancellationToken::new(),
        }
    }

    fn register(
        registry: &RunRegistry,
        run_id: &str,
        session_id: &str,
        chat_id: i64,
        source_ref: Option<&str>,
        cap: usize,
    ) -> RunSummary {
        registry
            .register_run(
                input(run_id, session_id, chat_id, source_ref),
                RegisterRunOptions {
                    max_runs_per_thread: cap,
                },
            )
            .expect("register run")
    }

    #[test]
    fn run_silence_state_thresholds() {
        let last_progress = Instant::now();
        let threshold = Duration::from_secs(10);
        assert_eq!(
            run_silence_state(
                last_progress,
                last_progress + Duration::from_secs(9),
                threshold
            ),
            SilenceState {
                silence_age: Duration::from_secs(9),
                alert: false,
            }
        );
        assert_eq!(
            run_silence_state(
                last_progress,
                last_progress + Duration::from_secs(10),
                threshold
            ),
            SilenceState {
                silence_age: Duration::from_secs(10),
                alert: false,
            }
        );
        assert_eq!(
            run_silence_state(
                last_progress,
                last_progress + Duration::from_secs(11),
                threshold
            ),
            SilenceState {
                silence_age: Duration::from_secs(11),
                alert: true,
            }
        );
    }

    #[test]
    fn run_registry_tracks_active_run_silence_age() {
        let registry = RunRegistry::new();
        register(&registry, "run-a", "sid-a", 100, None, 3);
        let base = Instant::now();
        registry.mark_run_progress_at("run-a", base).unwrap();
        assert_eq!(
            registry.active_run_max_silent_seconds_at(base + Duration::from_secs(37)),
            37
        );
        registry
            .mark_run_progress_at("run-a", base + Duration::from_secs(40))
            .unwrap();
        assert_eq!(
            registry.active_run_max_silent_seconds_at(base + Duration::from_secs(41)),
            1
        );
    }

    #[test]
    fn run_registry_silence_warn_debounces_until_progress() {
        let registry = RunRegistry::new();
        register(&registry, "run-a", "sid-a", 100, None, 3);
        let base = Instant::now();
        registry.mark_run_progress_at("run-a", base).unwrap();
        registry.warn_silent_runs_at(base + Duration::from_secs(11), Duration::from_secs(10));
        assert!(registry.lock_state().by_run["run-a"].silence_warned);
        registry.warn_silent_runs_at(base + Duration::from_secs(12), Duration::from_secs(10));
        assert!(registry.lock_state().by_run["run-a"].silence_warned);
        registry
            .mark_run_progress_at("run-a", base + Duration::from_secs(13))
            .unwrap();
        assert!(!registry.lock_state().by_run["run-a"].silence_warned);
    }

    #[test]
    fn run_registry_register_run_inserts_all_indexes() {
        let registry = RunRegistry::new();
        let summary = register(&registry, "attempt-1", "sid-1", 100, Some("spec-1"), 1);
        assert_eq!(summary.run_id, "attempt-1");
        assert_eq!(
            registry
                .list_for_thread(ChatThreadKey::new(100, Some(10)))
                .len(),
            1
        );
        assert!(matches!(
            registry.resolve_for_steer(ChatThreadKey::new(100, Some(10))),
            ResolveForSteer::Unique(_)
        ));
        assert_eq!(registry.source_ref_count(), 1);
    }

    #[test]
    fn has_active_run_for_source_ref_tracks_registration_lifetime() {
        let registry = RunRegistry::new();
        assert!(!registry.has_active_run_for_source_ref("spec-x"));

        register(&registry, "run-x", "sid-x", 100, Some("spec-x"), 1);
        assert!(registry.has_active_run_for_source_ref("spec-x"));
        assert!(!registry.has_active_run_for_source_ref("spec-other"));

        registry.remove_run("run-x").expect("remove registered run");
        assert!(!registry.has_active_run_for_source_ref("spec-x"));
    }

    #[test]
    fn run_registry_register_rejects_duplicate_run_id() {
        let registry = RunRegistry::new();
        register(&registry, "run-a", "sid-a", 100, None, 3);
        let err = registry
            .register_run(
                input("run-a", "sid-b", 101, None),
                RegisterRunOptions::cap_three(),
            )
            .unwrap_err();
        assert_eq!(
            err,
            RegisterRunError::DuplicateRunId {
                run_id: "run-a".to_string()
            }
        );
    }

    #[test]
    fn run_registry_register_rejects_duplicate_session_id() {
        let registry = RunRegistry::new();
        register(&registry, "run-a", "sid-a", 100, None, 3);
        let err = registry
            .register_run(
                input("run-b", "sid-a", 101, None),
                RegisterRunOptions::cap_three(),
            )
            .unwrap_err();
        assert_eq!(
            err,
            RegisterRunError::DuplicateSessionId {
                session_id: "sid-a".to_string(),
                existing: "run-a".to_string(),
            }
        );
    }

    #[test]
    fn feature_flag_disabled_preserves_single_run() {
        let registry = RunRegistry::new();
        register(&registry, "run-a", "sid-a", 100, None, STEP2_SINGLE_RUN_CAP);
        let err = registry
            .register_run(
                input("run-b", "sid-b", 100, None),
                RegisterRunOptions::step2_single_run(),
            )
            .unwrap_err();
        assert!(matches!(
            err,
            RegisterRunError::ThreadCapacityExceeded {
                active: 1,
                cap: 1,
                ..
            }
        ));
    }

    #[test]
    fn feature_flag_enabled_allows_three_rejects_fourth() {
        let registry = RunRegistry::new();
        register(&registry, "run-a", "sid-a", 100, None, MULTI_RUN_THREAD_CAP);
        register(&registry, "run-b", "sid-b", 100, None, MULTI_RUN_THREAD_CAP);
        register(&registry, "run-c", "sid-c", 100, None, MULTI_RUN_THREAD_CAP);
        let err = registry
            .register_run(
                input("run-d", "sid-d", 100, None),
                RegisterRunOptions::cap_three(),
            )
            .unwrap_err();
        assert!(matches!(
            err,
            RegisterRunError::ThreadCapacityExceeded {
                active: 3,
                cap: 3,
                ..
            }
        ));
    }

    #[test]
    fn terminal_cleanup_removes_only_target_run_indexes_and_acks() {
        let registry = RunRegistry::new();
        register(&registry, "run-a", "sid-a", 100, Some("spec-a"), 3);
        register(&registry, "run-b", "sid-b", 100, Some("spec-b"), 3);
        registry
            .bind_message("run-a", MessageKey::new(100, 55))
            .unwrap();
        registry
            .bind_message("run-b", MessageKey::new(100, 56))
            .unwrap();
        registry.remove_run("run-a").unwrap();
        assert!(registry.get_run("run-a").is_none());
        assert!(registry.get_run("run-b").is_some());
        assert!(registry.resolve_message(MessageKey::new(100, 55)).is_none());
        assert_eq!(
            registry
                .resolve_message(MessageKey::new(100, 56))
                .unwrap()
                .run_id,
            "run-b"
        );
        assert!(registry.contains_session("sid-b"));
    }

    #[test]
    fn abort_sendnow_steer_target_only_selected_run() {
        let registry = RunRegistry::new();
        let (tx_a, mut rx_a) = mpsc::channel(4);
        let (tx_b, mut rx_b) = mpsc::channel(4);
        registry
            .register_run(
                RegisterRunInput {
                    steer: tx_a,
                    ..input("run-a", "sid-a", 100, None)
                },
                RegisterRunOptions::cap_three(),
            )
            .unwrap();
        registry
            .register_run(
                RegisterRunInput {
                    steer: tx_b,
                    ..input("run-b", "sid-b", 100, None)
                },
                RegisterRunOptions::cap_three(),
            )
            .unwrap();
        let control = registry.control_for_run("run-b").unwrap();
        control
            .steer
            .try_send(SteerMessage {
                msg_id: 1,
                text: "only b".into(),
                is_edit: false,
            })
            .unwrap();
        assert!(rx_a.try_recv().is_err(), "run A must be untouched");
        assert_eq!(rx_b.try_recv().unwrap().text, "only b");
    }

    #[test]
    fn mirror_adds_sink_not_run() {
        let registry = RunRegistry::new();
        register(&registry, "run-a", "sid-a", 100, None, 3);
        let before_runs = registry
            .list_for_thread(ChatThreadKey::new(100, Some(10)))
            .len();
        let before_session = registry.contains_session("sid-a");
        registry
            .add_mirror_sink("run-a", RunSink::new(200, Some(20), 77))
            .expect("mirror sink added");
        assert_eq!(before_runs, 1);
        assert_eq!(
            registry
                .list_for_thread(ChatThreadKey::new(100, Some(10)))
                .len(),
            before_runs,
            "mirror must not create a new RunHandle"
        );
        assert_eq!(before_session, registry.contains_session("sid-a"));
        assert_eq!(
            registry.mirror_sinks("run-a"),
            vec![RunSink::new(200, Some(20), 77)]
        );
        assert_eq!(
            registry
                .resolve_message(MessageKey::new(200, 77))
                .unwrap()
                .run_id,
            "run-a"
        );
    }

    #[test]
    fn mirror_sink_cap_rejects_friendly() {
        let registry = RunRegistry::new();
        register(&registry, "run-a", "sid-a", 100, None, 3);
        registry
            .add_mirror_sink("run-a", RunSink::new(201, None, 1))
            .unwrap();
        registry
            .add_mirror_sink("run-a", RunSink::new(202, None, 2))
            .unwrap();
        let err = registry
            .add_mirror_sink("run-a", RunSink::new(203, None, 3))
            .unwrap_err();
        assert_eq!(
            err,
            MirrorSinkError::SinkCapacityExceeded {
                run_id: "run-a".to_string(),
                active: 2,
                cap: MIRROR_SINK_CAP,
            }
        );
        assert_eq!(registry.mirror_sinks("run-a").len(), MIRROR_SINK_CAP);
    }

    #[test]
    fn run_registry_register_step2_rejects_second_run_same_thread() {
        let registry = RunRegistry::new();
        register(&registry, "run-a", "sid-a", 100, None, STEP2_SINGLE_RUN_CAP);
        let err = registry
            .register_run(
                input("run-b", "sid-b", 100, None),
                RegisterRunOptions::step2_single_run(),
            )
            .unwrap_err();
        assert_eq!(
            err,
            RegisterRunError::ThreadCapacityExceeded {
                key: ChatThreadKey::new(100, Some(10)),
                active: 1,
                cap: 1,
            }
        );
    }

    #[test]
    fn run_registry_bind_message_adds_callback_index() {
        let registry = RunRegistry::new();
        register(&registry, "run-a", "sid-a", 100, None, 1);
        let summary = registry
            .bind_message("run-a", MessageKey::new(100, 55))
            .expect("bind message");
        assert_eq!(summary.bubble_message_id, Some(55));
        let resolved = registry
            .resolve_message(MessageKey::new(100, 55))
            .expect("message resolves");
        assert_eq!(resolved.run_id, "run-a");
    }

    #[test]
    fn run_registry_move_run_atomic_vec_move_and_cap() {
        move_run_a_to_b_atomic_origin_and_cap();
    }

    #[test]
    fn move_run_a_to_b_atomic_origin_and_cap() {
        let registry = RunRegistry::new();
        register(&registry, "run-a", "sid-a", 100, Some("spec-a"), 3);
        registry
            .bind_message("run-a", MessageKey::new(100, 55))
            .expect("old A bubble bound");
        registry
            .add_mirror_sink("run-a", RunSink::new(150, Some(10), 155))
            .expect("mirror preserved by move");
        register(&registry, "run-b", "sid-b", 200, None, 1);
        let err = registry
            .move_run(
                "run-a",
                RunOrigin::new(200, Some(10)),
                RegisterRunOptions::step2_single_run(),
            )
            .unwrap_err();
        assert_eq!(
            err,
            MoveRunError::ThreadCapacityExceeded {
                key: ChatThreadKey::new(200, Some(10)),
                active: 1,
                cap: 1,
            }
        );
        assert_eq!(
            registry
                .list_for_thread(ChatThreadKey::new(100, Some(10)))
                .len(),
            1,
            "failed cap-checked move keeps run at old origin"
        );
        assert_eq!(
            registry
                .list_for_thread(ChatThreadKey::new(200, Some(10)))
                .len(),
            1
        );
        assert_eq!(
            registry
                .resolve_message(MessageKey::new(100, 55))
                .expect("old message still resolves pre-move")
                .run_id,
            "run-a"
        );

        let moved = registry
            .move_run(
                "run-a",
                RunOrigin::new(300, Some(10)),
                RegisterRunOptions::cap_three(),
            )
            .expect("move succeeds");
        assert_eq!(moved.summary.origin.chat_id, 300);
        assert_eq!(moved.old_origin, RunOrigin::new(100, Some(10)));
        assert_eq!(moved.old_bubble, Some(RunSink::new(100, Some(10), 55)));
        assert!(
            registry
                .list_for_thread(ChatThreadKey::new(100, Some(10)))
                .is_empty(),
            "old key must be emptied/deleted"
        );
        assert_eq!(
            registry
                .list_for_thread(ChatThreadKey::new(300, Some(10)))
                .len(),
            1
        );
        let summary = registry.get_run("run-a").expect("run remains live");
        assert_eq!(summary.session_id, "sid-a");
        assert_eq!(summary.source_ref.as_deref(), Some("spec-a"));
        assert_eq!(
            registry
                .resolve_message(MessageKey::new(100, 55))
                .expect("old frozen bubble remains mapped for moved-toast callbacks")
                .run_id,
            "run-a"
        );
        assert_eq!(
            registry.moved_notice_for_message(MessageKey::new(100, 55), Some("run-a")),
            Some(MovedRunNotice {
                run_id: "run-a".to_string(),
                new_origin: RunOrigin::new(300, Some(10)),
            })
        );
        assert_eq!(
            registry.mirror_sinks("run-a"),
            vec![RunSink::new(150, Some(10), 155)]
        );
    }

    #[test]
    fn move_preserves_by_session_single_run() {
        let registry = RunRegistry::new();
        register(&registry, "run-a", "sid-a", 100, None, 3);
        assert!(registry.contains_session("sid-a"));
        registry
            .move_run(
                "run-a",
                RunOrigin::new(300, Some(10)),
                RegisterRunOptions::cap_three(),
            )
            .expect("move succeeds");
        assert!(registry.contains_session("sid-a"));
        let err = registry
            .register_run(
                input("run-dup", "sid-a", 300, None),
                RegisterRunOptions::cap_three(),
            )
            .unwrap_err();
        assert_eq!(
            err,
            RegisterRunError::DuplicateSessionId {
                session_id: "sid-a".to_string(),
                existing: "run-a".to_string(),
            }
        );
    }

    #[test]
    fn run_registry_remove_run_clears_all_indexes_transactionally() {
        let registry = RunRegistry::new();
        register(&registry, "run-a", "sid-a", 100, Some("spec-a"), 1);
        registry
            .bind_message("run-a", MessageKey::new(100, 55))
            .expect("bind message");
        let removed = registry.remove_run("run-a").expect("removed");
        assert_eq!(removed.run_id, "run-a");
        assert!(!registry.contains_run("run-a"));
        assert!(
            registry
                .list_for_thread(ChatThreadKey::new(100, Some(10)))
                .is_empty()
        );
        assert!(registry.resolve_message(MessageKey::new(100, 55)).is_none());
        assert_eq!(registry.source_ref_count(), 0);
    }

    #[test]
    fn bare_text_ambiguous_prompts_not_pick() {
        let registry = RunRegistry::new();
        register(&registry, "run-a", "sid-a", 100, None, 3);
        register(&registry, "run-b", "sid-b", 100, None, 3);
        assert!(matches!(
            registry.resolve_for_steer(ChatThreadKey::new(100, Some(10))),
            ResolveForSteer::Ambiguous(runs) if runs.len() == 2
        ));
    }

    #[test]
    fn edited_message_ambiguous_does_not_pick() {
        let registry = RunRegistry::new();
        register(&registry, "run-a", "sid-a", 100, None, 3);
        register(&registry, "run-b", "sid-b", 100, None, 3);
        assert!(matches!(
            registry.resolve_for_steer(ChatThreadKey::new(100, Some(10))),
            ResolveForSteer::Ambiguous(_)
        ));
    }

    #[test]
    fn run_registry_resolve_for_steer_unique_ambiguous_not_found() {
        let registry = RunRegistry::new();
        assert_eq!(
            registry.resolve_for_steer(ChatThreadKey::new(100, Some(10))),
            ResolveForSteer::NotFound
        );
        register(&registry, "run-a", "sid-a", 100, None, 3);
        assert!(matches!(
            registry.resolve_for_steer(ChatThreadKey::new(100, Some(10))),
            ResolveForSteer::Unique(summary) if summary.run_id == "run-a"
        ));
        register(&registry, "run-b", "sid-b", 100, None, 3);
        assert!(matches!(
            registry.resolve_for_steer(ChatThreadKey::new(100, Some(10))),
            ResolveForSteer::Ambiguous(summaries) if summaries.len() == 2
        ));
    }

    #[test]
    fn run_registry_list_for_thread_returns_cloned_summaries() {
        let registry = RunRegistry::new();
        register(&registry, "run-a", "sid-a", 100, None, 3);
        let mut summaries = registry.list_for_thread(ChatThreadKey::new(100, Some(10)));
        summaries.clear();
        assert_eq!(
            registry
                .list_for_thread(ChatThreadKey::new(100, Some(10)))
                .len(),
            1
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn b61_cleanup_register_interleaving_preserves_b_and_removes_a() {
        let registry = Arc::new(RunRegistry::new());
        register(&registry, "run-a", "sid-a", 100, Some("spec-a"), 3);
        registry
            .bind_message("run-a", MessageKey::new(100, 55))
            .expect("bind message");

        let parked_at_seam = Arc::new(Notify::new());
        let release_cleanup = Arc::new(Notify::new());

        let cleanup_task = {
            let registry = registry.clone();
            let parked_at_seam = parked_at_seam.clone();
            let release_cleanup = release_cleanup.clone();
            tokio::spawn(async move {
                registry
                    .remove_run_after_test_seam("run-a", parked_at_seam, release_cleanup)
                    .await
                    .expect("A removed")
            })
        };

        parked_at_seam.notified().await;

        let register_task = {
            let registry = registry.clone();
            tokio::spawn(
                async move { register(&registry, "run-b", "sid-b", 100, Some("spec-b"), 3) },
            )
        };
        let registered = register_task.await.expect("register task panicked");
        assert_eq!(registered.run_id, "run-b");

        release_cleanup.notify_one();
        let removed = cleanup_task.await.expect("cleanup task panicked");
        assert_eq!(removed.run_id, "run-a");

        assert!(
            !registry.contains_run("run-a"),
            "A must be absent from by_run"
        );
        assert!(registry.contains_run("run-b"), "B must remain in by_run");
        let thread_runs = registry.list_for_thread(ChatThreadKey::new(100, Some(10)));
        assert_eq!(thread_runs.len(), 1, "thread index must contain only B");
        assert_eq!(thread_runs[0].run_id, "run-b");
        assert!(
            !registry.contains_session("sid-a"),
            "A must be absent from by_session"
        );
        assert!(
            registry.contains_session("sid-b"),
            "B must remain in by_session"
        );
        assert!(registry.resolve_message(MessageKey::new(100, 55)).is_none());
        assert_eq!(
            registry.source_ref_count(),
            1,
            "only spec-b remains indexed"
        );
    }
}
