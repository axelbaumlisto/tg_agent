//! Ratatui frontend entrypoint and terminal lifecycle guard.

pub(crate) mod app;
pub(crate) mod view;

use std::future;
use std::io::{self, Write};
use std::path::Path;
use std::sync::Once;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use crossterm::cursor::{Hide, Show};
use crossterm::event::{Event, EventStream, KeyEvent, KeyEventKind};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use naked_core::AgentCore;
use naked_core::config::Config;
use naked_core::types::{AgentEvent, AgentHandle, PermissionResponse};
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use tokio::sync::mpsc;
use tokio::time::{self, MissedTickBehavior};
use tokio_stream::StreamExt;
use tokio_util::sync::CancellationToken;

use crate::event_map;
use crate::tui::app::{App, KeyOutcome, RunState};

// Process-global by design: `naked tui` is a CLI entrypoint with one TUI per
// process. Sharing this flag lets Drop, SIGTERM, and panic hooks race safely and
// still restore raw/alternate-screen terminal state at most once.
static TERMINAL_RESTORE_RAN: AtomicBool = AtomicBool::new(false);
static PANIC_HOOK_INSTALLED: Once = Once::new();

type PanicHook = Box<dyn Fn(&std::panic::PanicHookInfo<'_>) + Sync + Send + 'static>;

pub(crate) struct TerminalGuard;

struct ActiveTurn {
    events: mpsc::Receiver<AgentEvent>,
    permissions: mpsc::Sender<PermissionResponse>,
    abort: CancellationToken,
}

struct TermSignal {
    #[cfg(unix)]
    inner: tokio::signal::unix::Signal,
}

impl TermSignal {
    fn new() -> anyhow::Result<Self> {
        #[cfg(unix)]
        {
            Ok(Self {
                inner: tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())?,
            })
        }
        #[cfg(not(unix))]
        {
            Ok(Self {})
        }
    }

    async fn recv(&mut self) {
        #[cfg(unix)]
        {
            let _ = self.inner.recv().await;
        }
        #[cfg(not(unix))]
        future::pending::<()>().await;
    }
}

impl TerminalGuard {
    pub(crate) fn new() -> anyhow::Result<Self> {
        enable_raw_mode()?;
        let mut stdout = io::stdout();
        if let Err(err) = execute!(stdout, EnterAlternateScreen, Hide) {
            restore_terminal_best_effort();
            return Err(err.into());
        }

        Ok(Self)
    }

    pub(crate) fn restore_best_effort(&self) {
        restore_terminal_best_effort();
    }
}

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        self.restore_best_effort();
    }
}

pub(crate) async fn tui_main(_args: &[String]) -> anyhow::Result<()> {
    let (config, config_source) = Config::load_with_source_bytes()?;
    maybe_warn_shared_state(
        std::env::var_os("NAKED_CONFIG").is_some(),
        config_source.as_ref().map(|(path, _)| path.as_path()),
    );

    let provider = naked_core::build_provider_from_config(&config)?;
    let agent = AgentCore::new(config.clone(), provider);
    agent.init_mcp().await;
    let restored = agent.restore_sessions().await.unwrap_or_default();
    let workspace = config.workspace.clone();
    let session_id = agent.create_session(&workspace).await;
    let model_tag = format!("{}/{}", config.default_provider, config.default_model);

    install_terminal_panic_hook_once();
    let _guard = TerminalGuard::new()?;
    let backend = CrosstermBackend::new(io::stdout());
    let mut terminal = Terminal::new(backend)?;

    let mut app = App::new(model_tag, 5_000);
    app.push_line(format!("session: {session_id}"));
    app.push_line(format!("workspace: {}", workspace.display()));
    if !restored.is_empty() {
        app.push_line(format!("restored {} session(s)", restored.len()));
    }

    let mut input_events = EventStream::new();
    let mut render_tick = time::interval(Duration::from_millis(33));
    render_tick.set_missed_tick_behavior(MissedTickBehavior::Skip);
    let mut active_turn: Option<ActiveTurn> = None;
    let mut dirty = true;

    let mut sigterm = TermSignal::new()?;

    loop {
        if dirty {
            terminal.draw(|frame| crate::tui::view::render(frame, &app))?;
            dirty = false;
        }

        tokio::select! {
            maybe_input = input_events.next() => {
                match maybe_input {
                    Some(Ok(Event::Key(key))) if is_key_press(key) => {
                        match app.on_key(key) {
                            KeyOutcome::SendPrompt(text) => {
                                match agent.send_prompt(&session_id, &text).await {
                                    Ok(handle) => active_turn = Some(ActiveTurn::from(handle)),
                                    Err(err) => {
                                        app.push_line(format!("error: {err}"));
                                        app.run_state = RunState::Idle;
                                        active_turn = None;
                                    }
                                }
                            }
                            KeyOutcome::AbortTurn => {
                                if let Some(active) = &active_turn {
                                    active.abort.cancel();
                                }
                            }
                            KeyOutcome::PermissionResponse { call_id, allowed } => {
                                if let Some(active) = &active_turn {
                                    let response = PermissionResponse { call_id, allowed };
                                    if let Err(err) = active.permissions.send(response).await {
                                        app.push_line(format!("permission response failed: {err}"));
                                    }
                                }
                            }
                            KeyOutcome::Quit | KeyOutcome::HardQuit => return Ok(()),
                            KeyOutcome::None => {}
                        }
                        dirty = true;
                    }
                    Some(Ok(_)) => {}
                    Some(Err(err)) => {
                        app.push_line(format!("input error: {err}"));
                        terminal.draw(|frame| crate::tui::view::render(frame, &app))?;
                        return Ok(());
                    }
                    None => return Ok(()),
                }
            }
            _ = render_tick.tick() => {
                terminal.draw(|frame| crate::tui::view::render(frame, &app))?;
                dirty = false;
            }
            maybe_event = recv_active_event(&mut active_turn) => {
                if disarm_on_closed(&maybe_event) {
                    app.apply(crate::event_map::TranscriptUpdate::TurnDone);
                    active_turn = None;
                } else if let Some(event) = maybe_event {
                    app.apply(event_map::map_event(event));
                    if should_clear_active(app.run_state, false) {
                        active_turn = None;
                    }
                }
                dirty = true;
            }
            _ = sigterm.recv() => {
                restore_terminal_best_effort();
                return Ok(());
            }
        }
    }
}

impl From<AgentHandle> for ActiveTurn {
    fn from(handle: AgentHandle) -> Self {
        let AgentHandle {
            events,
            permissions,
            steer: _,
            abort,
        } = handle;
        Self {
            events,
            permissions,
            abort,
        }
    }
}

async fn recv_active_event(turn: &mut Option<ActiveTurn>) -> Option<AgentEvent> {
    if !has_active_arm(turn) {
        return future::pending().await;
    }

    match turn.as_mut() {
        Some(active) => active.events.recv().await,
        None => future::pending().await,
    }
}

fn should_run_restore(flag: &AtomicBool) -> bool {
    flag.compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
        .is_ok()
}

fn restore_once_with(flag: &AtomicBool, restore: impl FnOnce()) -> bool {
    if !should_run_restore(flag) {
        return false;
    }

    restore();
    true
}

pub(crate) fn restore_terminal_best_effort() {
    restore_once_with(&TERMINAL_RESTORE_RAN, || {
        let _ = disable_raw_mode();
        let mut stdout = io::stdout();
        let _ = execute!(stdout, LeaveAlternateScreen, Show);
        let _ = stdout.flush();
    });
}

fn install_terminal_panic_hook_once() {
    PANIC_HOOK_INSTALLED.call_once(|| {
        let prev = std::panic::take_hook();
        std::panic::set_hook(compose_panic_restore(restore_terminal_best_effort, prev));
    });
}

fn compose_panic_restore(restore: impl Fn() + Sync + Send + 'static, prev: PanicHook) -> PanicHook {
    Box::new(move |info| {
        restore();
        prev(info);
    })
}

fn is_key_press(key: KeyEvent) -> bool {
    matches!(key.kind, KeyEventKind::Press)
}

fn disarm_on_closed(recv_result: &Option<AgentEvent>) -> bool {
    recv_result.is_none()
}

fn should_clear_active(app_run_state: RunState, channel_closed: bool) -> bool {
    channel_closed || app_run_state == RunState::Idle
}

fn has_active_arm(turn: &Option<ActiveTurn>) -> bool {
    turn.is_some()
}

fn maybe_warn_shared_state(naked_config_set: bool, config_source: Option<&Path>) {
    if should_warn_shared_state(naked_config_set, config_source) {
        eprintln!(
            "⚠ naked tui shares ~/.naked session/research state with a running daemon — use a separate workspace to isolate."
        );
    }
}

fn should_warn_shared_state(naked_config_set: bool, config_source: Option<&Path>) -> bool {
    naked_config_set
        || config_source
            .and_then(Path::file_name)
            .and_then(|name| name.to_str())
            == Some("naked.json")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;
    use std::sync::atomic::AtomicUsize;

    #[test]
    fn abort_returns_to_idle_only_on_idle_event_or_channel_close() {
        assert!(!should_clear_active(RunState::Aborting, false));
        assert!(should_clear_active(RunState::Idle, false));
        assert!(should_clear_active(RunState::Aborting, true));
    }

    #[test]
    fn closed_event_channel_disarms_receiver_arm() {
        assert!(disarm_on_closed(&None));
        assert!(!disarm_on_closed(&Some(AgentEvent::Heartbeat)));
    }

    #[test]
    fn idle_state_has_no_active_receiver_arm() {
        let turn: Option<ActiveTurn> = None;

        assert!(!has_active_arm(&turn));
    }

    #[test]
    fn terminal_restore_guard_is_idempotent() {
        let restored = AtomicBool::new(false);

        assert!(should_run_restore(&restored));
        assert!(!should_run_restore(&restored));
        assert!(restored.load(Ordering::Acquire));
    }

    #[test]
    fn terminal_restore_runs_on_drop() {
        struct FakeGuard<'a> {
            restored: &'a AtomicBool,
            calls: &'a AtomicUsize,
        }

        impl Drop for FakeGuard<'_> {
            fn drop(&mut self) {
                restore_once_with(self.restored, || {
                    self.calls.fetch_add(1, Ordering::AcqRel);
                });
            }
        }

        let restored = AtomicBool::new(false);
        let calls = AtomicUsize::new(0);
        {
            let _guard = FakeGuard {
                restored: &restored,
                calls: &calls,
            };
        }

        assert_eq!(calls.load(Ordering::Acquire), 1);
        assert!(restored.load(Ordering::Acquire));
        assert!(!restore_once_with(&restored, || {
            calls.fetch_add(1, Ordering::AcqRel);
        }));
        assert_eq!(calls.load(Ordering::Acquire), 1);
    }

    #[test]
    fn panic_hook_invokes_restore_before_chaining() {
        static PANIC_HOOK_TEST_LOCK: Mutex<()> = Mutex::new(());

        let _guard = PANIC_HOOK_TEST_LOCK.lock().expect("panic hook test lock");
        let original = std::panic::take_hook();
        let order = std::sync::Arc::new(Mutex::new(Vec::new()));
        let restore_order = std::sync::Arc::clone(&order);
        let prev_order = std::sync::Arc::clone(&order);
        std::panic::set_hook(compose_panic_restore(
            move || restore_order.lock().expect("restore order").push("restore"),
            Box::new(move |_| prev_order.lock().expect("prev order").push("prev")),
        ));

        let result = std::panic::catch_unwind(|| panic!("exercise panic hook order"));
        std::panic::set_hook(original);

        assert!(result.is_err());
        assert_eq!(
            *order.lock().expect("order after panic"),
            vec!["restore", "prev"]
        );
    }
}
