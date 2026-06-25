use std::collections::HashMap;
use std::path::Path;
use std::process::Stdio;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use tokio::io::{AsyncBufRead, AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::Mutex;
use uuid::Uuid;

use super::bash::MAX_COLLECTED;

const PERSISTENT_BASH_STDOUT_TRUNCATED_MARKER: &[u8] =
    b"\n[persistent bash stdout truncated: collection cap exceeded; continuing to scan for sentinel]\n";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PersistentBashOutcome {
    Ok,
    Timeout,
    Killed,
    Restart,
    Error,
    Disabled,
    Busy,
}

impl PersistentBashOutcome {
    pub fn bump(self) {
        let counter = match self {
            Self::Ok => &crate::types::PERSISTENT_BASH_OK_COUNT,
            Self::Timeout => &crate::types::PERSISTENT_BASH_TIMEOUT_COUNT,
            Self::Killed => &crate::types::PERSISTENT_BASH_KILLED_COUNT,
            Self::Restart => &crate::types::PERSISTENT_BASH_RESTART_COUNT,
            Self::Error => &crate::types::PERSISTENT_BASH_ERROR_COUNT,
            Self::Disabled => &crate::types::PERSISTENT_BASH_DISABLED_COUNT,
            Self::Busy => &crate::types::PERSISTENT_BASH_BUSY_COUNT,
        };
        counter.fetch_add(1, Ordering::Relaxed);
    }
}

#[derive(Debug)]
pub struct PersistentBashOutput {
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
    pub exit_code: i32,
}

#[derive(Debug)]
pub enum PersistentBashError {
    Timeout,
    Busy { max_sessions: usize },
    Io(std::io::Error),
    Protocol(String),
}

impl std::fmt::Display for PersistentBashError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Timeout => write!(f, "command timed out"),
            Self::Busy { max_sessions } => write!(
                f,
                "persistent bash: all {max_sessions} sessions busy, try again"
            ),
            Self::Io(e) => write!(f, "{e}"),
            Self::Protocol(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for PersistentBashError {}

impl From<std::io::Error> for PersistentBashError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

const DEFAULT_MAX_PERSISTENT_BASH_SESSIONS: usize = 8;
const DEFAULT_PERSISTENT_BASH_IDLE_TTL: Duration = Duration::from_secs(10 * 60);

pub struct PersistentBashManager {
    sessions: Mutex<HashMap<String, ManagedShellSession>>,
    max_sessions: usize,
    idle_ttl: Duration,
}

struct ManagedShellSession {
    session: Arc<ShellSession>,
    last_used: Instant,
}

impl Default for PersistentBashManager {
    fn default() -> Self {
        Self {
            sessions: Mutex::new(HashMap::new()),
            max_sessions: DEFAULT_MAX_PERSISTENT_BASH_SESSIONS.max(1),
            idle_ttl: DEFAULT_PERSISTENT_BASH_IDLE_TTL,
        }
    }
}

impl PersistentBashManager {
    pub fn new() -> Self {
        Self::default()
    }

    #[cfg(test)]
    fn with_limits(max_sessions: usize, idle_ttl: Duration) -> Self {
        Self {
            sessions: Mutex::new(HashMap::new()),
            max_sessions: max_sessions.max(1),
            idle_ttl,
        }
    }

    pub async fn get_or_create(
        &self,
        session_id: &str,
        wait_timeout: Duration,
    ) -> Result<Arc<ShellSession>, PersistentBashError> {
        let deadline = Instant::now() + wait_timeout;
        loop {
            let now = Instant::now();
            let evicted = {
                let mut sessions = self.sessions.lock().await;
                if let Some(managed) = sessions.get_mut(session_id) {
                    managed.last_used = now;
                    return Ok(managed.session.clone());
                }

                if let Some(evicted) = evict_one_idle_expired(&mut sessions, now, self.idle_ttl) {
                    Some(evicted)
                } else if sessions.len() >= self.max_sessions {
                    evict_one_lru_idle(&mut sessions)
                } else {
                    let session = Arc::new(ShellSession::new());
                    sessions.insert(
                        session_id.to_string(),
                        ManagedShellSession {
                            session: session.clone(),
                            last_used: now,
                        },
                    );
                    return Ok(session);
                }
            };

            if let Some(session) = evicted {
                session.close().await;
            } else if now >= deadline {
                PersistentBashOutcome::Busy.bump();
                return Err(PersistentBashError::Busy {
                    max_sessions: self.max_sessions,
                });
            } else {
                let remaining = deadline.saturating_duration_since(now);
                tokio::time::sleep(remaining.min(Duration::from_millis(10))).await;
            }
        }
    }

    pub async fn close_session(&self, session_id: &str) {
        let session = self
            .sessions
            .lock()
            .await
            .remove(session_id)
            .map(|managed| managed.session);
        if let Some(session) = session {
            session.close().await;
        }
    }

    pub async fn session_count(&self) -> usize {
        self.sessions.lock().await.len()
    }
}

fn evict_one_idle_expired(
    sessions: &mut HashMap<String, ManagedShellSession>,
    now: Instant,
    idle_ttl: Duration,
) -> Option<Arc<ShellSession>> {
    let victim = sessions
        .iter()
        .filter(|(_, managed)| {
            now.duration_since(managed.last_used) >= idle_ttl && managed.session.is_idle()
        })
        .min_by_key(|(_, managed)| managed.last_used)
        .map(|(session_id, _)| session_id.clone())?;
    sessions.remove(&victim).map(|managed| managed.session)
}

fn evict_one_lru_idle(
    sessions: &mut HashMap<String, ManagedShellSession>,
) -> Option<Arc<ShellSession>> {
    let victim = sessions
        .iter()
        .filter(|(_, managed)| managed.session.is_idle())
        .min_by_key(|(_, managed)| managed.last_used)
        .map(|(session_id, _)| session_id.clone())?;
    sessions.remove(&victim).map(|managed| managed.session)
}

impl Drop for PersistentBashManager {
    fn drop(&mut self) {
        if let Ok(mut sessions) = self.sessions.try_lock() {
            for (_, managed) in sessions.drain() {
                // Best-effort kill on drop only; explicit close_session/eviction reaps.
                managed.session.kill_now();
            }
        }
    }
}

pub struct ShellSession {
    state: Mutex<ShellState>,
}

struct ShellState {
    child: Option<Child>,
    stdin: Option<tokio::process::ChildStdin>,
    stdout: Option<BufReader<tokio::process::ChildStdout>>,
    pgid: Option<u32>,
}

impl ShellSession {
    fn new() -> Self {
        Self {
            state: Mutex::new(ShellState::empty()),
        }
    }

    pub async fn run(
        &self,
        command: &str,
        cwd: &Path,
        timeout: Duration,
    ) -> Result<PersistentBashOutput, PersistentBashError> {
        let mut state = self.state.lock().await;
        if !state.is_alive() {
            state.start(cwd).await?;
            PersistentBashOutcome::Restart.bump();
        }

        match tokio::time::timeout(timeout, state.run_framed(command, new_sentinel_nonce())).await {
            Ok(Ok(output)) => {
                PersistentBashOutcome::Ok.bump();
                Ok(output)
            }
            Ok(Err(e)) => {
                state.kill_and_reap().await;
                PersistentBashOutcome::Error.bump();
                Err(e)
            }
            Err(_) => {
                state.kill_and_reap().await;
                PersistentBashOutcome::Timeout.bump();
                Err(PersistentBashError::Timeout)
            }
        }
    }

    pub async fn close(&self) {
        let mut state = self.state.lock().await;
        state.kill_and_reap().await;
        PersistentBashOutcome::Killed.bump();
    }

    fn kill_now(&self) {
        if let Ok(mut state) = self.state.try_lock() {
            state.kill_now();
            PersistentBashOutcome::Killed.bump();
        }
    }

    fn is_idle(&self) -> bool {
        self.state.try_lock().is_ok()
    }

    #[cfg(test)]
    async fn process_id(&self) -> Option<u32> {
        let state = self.state.lock().await;
        state.child.as_ref().and_then(Child::id)
    }
}

impl ShellState {
    fn empty() -> Self {
        Self {
            child: None,
            stdin: None,
            stdout: None,
            pgid: None,
        }
    }

    fn is_alive(&mut self) -> bool {
        match self.child.as_mut() {
            Some(child) => matches!(child.try_wait(), Ok(None)),
            None => false,
        }
    }

    async fn start(&mut self, cwd: &Path) -> Result<(), PersistentBashError> {
        self.kill_and_reap().await;

        let mut command = Command::new("bash");
        command
            .arg("--noprofile")
            .arg("--norc")
            .arg("-s")
            .current_dir(cwd)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .env("PS1", "")
            .env("PS2", "");

        #[cfg(unix)]
        command.process_group(0);

        let mut child = command.spawn()?;
        let pgid = child.id();
        self.stdin = child.stdin.take();
        self.stdout = child.stdout.take().map(BufReader::new);
        self.pgid = pgid;
        self.child = Some(child);
        Ok(())
    }

    async fn run_framed(
        &mut self,
        command: &str,
        nonce: String,
    ) -> Result<PersistentBashOutput, PersistentBashError> {
        let stdout_sentinel = format!("__NAKED_DONE_{nonce}_");
        let stdin = self.stdin.as_mut().ok_or_else(|| {
            PersistentBashError::Protocol("persistent bash stdin is closed".to_string())
        })?;

        stdin
            .write_all(format!("{{\n{command}\n\n}} 2>&1\n").as_bytes())
            .await?;
        stdin
            .write_all(
                format!(
                    "__naked_status=$?; printf '{stdout_sentinel}%s__\\n' \"$__naked_status\"\n"
                )
                .as_bytes(),
            )
            .await?;
        stdin.flush().await?;

        let stdout = self.stdout.as_mut().ok_or_else(|| {
            PersistentBashError::Protocol("persistent bash stdout is closed".to_string())
        })?;

        let mut out = Vec::new();
        let exit_code = read_until_stdout_sentinel(stdout, &stdout_sentinel, &mut out).await?;

        Ok(PersistentBashOutput {
            stdout: out,
            stderr: Vec::new(),
            exit_code,
        })
    }

    async fn kill_and_reap(&mut self) {
        if let Some(stdin) = self.stdin.as_mut() {
            let _ = stdin.write_all(b"exit\n").await;
            let _ = stdin.flush().await;
        }
        if let Some(mut child) = self.child.take() {
            match tokio::time::timeout(Duration::from_millis(250), child.wait()).await {
                Ok(_) => {}
                Err(_) => {
                    if let Some(pgid) = self.pgid {
                        kill_process_group(pgid);
                    }
                    let _ = child.start_kill();
                    let _ = child.wait().await;
                }
            }
        }
        self.stdin = None;
        self.stdout = None;
        self.pgid = None;
    }

    fn kill_now(&mut self) {
        if let Some(pgid) = self.pgid {
            kill_process_group(pgid);
        }
        if let Some(child) = self.child.as_mut() {
            let _ = child.start_kill();
        }
    }
}

async fn read_until_stdout_sentinel<R: AsyncBufRead + Unpin>(
    reader: &mut R,
    sentinel_prefix: &str,
    output: &mut Vec<u8>,
) -> Result<i32, PersistentBashError> {
    let mut line = Vec::new();
    let mut marker_appended = false;
    loop {
        line.clear();
        let n = reader.read_until(b'\n', &mut line).await?;
        if n == 0 {
            return Err(PersistentBashError::Protocol(
                "persistent bash stdout closed before sentinel".to_string(),
            ));
        }
        let line_text = String::from_utf8_lossy(&line);
        let trimmed_line = line_text.trim_end_matches(['\r', '\n']);
        if let Some(exit_code) = parse_stdout_sentinel_line(trimmed_line, sentinel_prefix)? {
            return Ok(exit_code);
        }

        if output.len() + line.len() <= MAX_COLLECTED {
            output.extend_from_slice(&line);
        } else if !marker_appended {
            output.extend_from_slice(PERSISTENT_BASH_STDOUT_TRUNCATED_MARKER);
            marker_appended = true;
        }
    }
}

fn parse_stdout_sentinel_line(
    trimmed_line: &str,
    sentinel_prefix: &str,
) -> Result<Option<i32>, PersistentBashError> {
    let Some(code_text) = trimmed_line
        .strip_prefix(sentinel_prefix)
        .and_then(|rest| rest.strip_suffix("__"))
    else {
        return Ok(None);
    };

    code_text.parse::<i32>().map(Some).map_err(|e| {
        PersistentBashError::Protocol(format!(
            "invalid persistent bash sentinel exit code `{code_text}`: {e}"
        ))
    })
}

fn new_sentinel_nonce() -> String {
    Uuid::new_v4().simple().to_string()
}

#[cfg(unix)]
fn kill_process_group(pgid: u32) {
    let _ = std::process::Command::new("kill")
        .arg("-KILL")
        .arg("--")
        .arg(format!("-{pgid}"))
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();
}

#[cfg(not(unix))]
fn kill_process_group(_pgid: u32) {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    use tokio::sync::Barrier;

    #[tokio::test]
    async fn persistent_bash_preserves_export_within_session() {
        let manager = PersistentBashManager::new();
        let session = manager
            .get_or_create("s1", Duration::from_secs(5))
            .await
            .unwrap();
        session
            .run(
                "export NAKED_PERSIST_TEST=one",
                Path::new("/tmp"),
                Duration::from_secs(5),
            )
            .await
            .unwrap();
        let out = session
            .run(
                "echo $NAKED_PERSIST_TEST",
                Path::new("/tmp"),
                Duration::from_secs(5),
            )
            .await
            .unwrap();
        assert_eq!(String::from_utf8_lossy(&out.stdout).trim(), "one");
        manager.close_session("s1").await;
    }

    #[tokio::test]
    async fn persistent_bash_preserves_cwd_within_session() {
        let dir = tempfile::tempdir().unwrap();
        let d = dir.path().join("d");
        let manager = PersistentBashManager::new();
        let session = manager
            .get_or_create("s1", Duration::from_secs(5))
            .await
            .unwrap();
        session
            .run("mkdir d && cd d", dir.path(), Duration::from_secs(5))
            .await
            .unwrap();
        let out = session
            .run("pwd", dir.path(), Duration::from_secs(5))
            .await
            .unwrap();
        assert_eq!(
            PathBuf::from(String::from_utf8_lossy(&out.stdout).trim()),
            d
        );
        manager.close_session("s1").await;
    }

    #[tokio::test]
    async fn persistent_bash_isolates_sessions() {
        let d1 = tempfile::tempdir().unwrap();
        let d2 = tempfile::tempdir().unwrap();
        let manager = PersistentBashManager::new();
        let s1 = manager
            .get_or_create("s1", Duration::from_secs(5))
            .await
            .unwrap();
        let s2 = manager
            .get_or_create("s2", Duration::from_secs(5))
            .await
            .unwrap();
        s1.run("export X=s1; cd /", d1.path(), Duration::from_secs(5))
            .await
            .unwrap();
        s2.run("export X=s2", d2.path(), Duration::from_secs(5))
            .await
            .unwrap();
        let o1 = s1
            .run("echo $X; pwd", d1.path(), Duration::from_secs(5))
            .await
            .unwrap();
        let o2 = s2
            .run("echo $X; pwd", d2.path(), Duration::from_secs(5))
            .await
            .unwrap();
        assert!(String::from_utf8_lossy(&o1.stdout).starts_with("s1\n/"));
        assert!(String::from_utf8_lossy(&o2.stdout).starts_with("s2\n"));
        assert!(String::from_utf8_lossy(&o2.stdout).contains(d2.path().to_str().unwrap()));
        manager.close_session("s1").await;
        manager.close_session("s2").await;
    }

    #[tokio::test]
    async fn persistent_bash_serializes_concurrent_commands() {
        let manager = Arc::new(PersistentBashManager::new());
        let session = manager
            .get_or_create("s", Duration::from_secs(5))
            .await
            .unwrap();
        let barrier = Arc::new(Barrier::new(3));

        let s1 = session.clone();
        let b1 = barrier.clone();
        let t1 = tokio::spawn(async move {
            b1.wait().await;
            s1.run(
                "printf 'A1\\n'; sleep 1; printf 'A2\\n'",
                Path::new("/tmp"),
                Duration::from_secs(5),
            )
            .await
            .unwrap()
        });

        let s2 = session.clone();
        let b2 = barrier.clone();
        let t2 = tokio::spawn(async move {
            b2.wait().await;
            s2.run(
                "printf 'B1\\n'; sleep 1; printf 'B2\\n'",
                Path::new("/tmp"),
                Duration::from_secs(5),
            )
            .await
            .unwrap()
        });

        barrier.wait().await;
        let (a, b) = tokio::join!(t1, t2);
        let a = String::from_utf8_lossy(&a.unwrap().stdout).to_string();
        let b = String::from_utf8_lossy(&b.unwrap().stdout).to_string();
        assert_eq!(a, "A1\nA2\n", "first command output was {a:?}");
        assert_eq!(b, "B1\nB2\n", "second command output was {b:?}");
        manager.close_session("s").await;
    }

    #[tokio::test]
    async fn persistent_bash_timeout_kills_and_session_recovers() {
        let manager = PersistentBashManager::new();
        let session = manager
            .get_or_create("s", Duration::from_secs(5))
            .await
            .unwrap();
        let err = session
            .run("sleep 5", Path::new("/tmp"), Duration::from_secs(1))
            .await
            .unwrap_err();
        assert!(matches!(err, PersistentBashError::Timeout));
        let out = session
            .run("echo alive", Path::new("/tmp"), Duration::from_secs(5))
            .await
            .unwrap();
        assert_eq!(String::from_utf8_lossy(&out.stdout).trim(), "alive");
        manager.close_session("s").await;
    }

    #[tokio::test]
    async fn persistent_bash_command_printing_sentinel_like_output_does_not_break_framing() {
        let manager = PersistentBashManager::new();
        let session = manager
            .get_or_create("s", Duration::from_secs(5))
            .await
            .unwrap();
        let out = session
            .run(
                "printf '__NAKED_DONE_00000000000000000000000000000000_0__\\n'; sleep 1; printf 'after\\n'",
                Path::new("/tmp"),
                Duration::from_secs(5),
            )
            .await
            .unwrap();
        assert_eq!(
            String::from_utf8_lossy(&out.stdout),
            "__NAKED_DONE_00000000000000000000000000000000_0__\nafter\n"
        );

        let next = session
            .run(
                "echo still-serialized",
                Path::new("/tmp"),
                Duration::from_secs(5),
            )
            .await
            .unwrap();
        assert_eq!(
            String::from_utf8_lossy(&next.stdout).trim(),
            "still-serialized"
        );
        manager.close_session("s").await;
    }

    #[tokio::test]
    async fn persistent_bash_get_or_create_bounded_when_all_busy() {
        let manager = Arc::new(PersistentBashManager::with_limits(
            1,
            Duration::from_secs(60),
        ));
        let session = manager
            .get_or_create("s1", Duration::from_secs(5))
            .await
            .unwrap();
        let barrier = Arc::new(Barrier::new(2));

        let held_session = session.clone();
        let held_barrier = barrier.clone();
        let holder = tokio::spawn(async move {
            held_barrier.wait().await;
            held_session
                .run("sleep 5", Path::new("/tmp"), Duration::from_secs(10))
                .await
        });
        barrier.wait().await;
        tokio::time::sleep(Duration::from_millis(50)).await;

        let started = Instant::now();
        let result = tokio::time::timeout(
            Duration::from_secs(1),
            manager.get_or_create("s2", Duration::from_millis(75)),
        )
        .await
        .expect("get_or_create hung beyond fail-fast timeout");

        assert!(started.elapsed() < Duration::from_millis(500));
        assert!(matches!(
            result,
            Err(PersistentBashError::Busy { max_sessions: 1 })
        ));

        holder.abort();
        let _ = holder.await;
        manager.close_session("s1").await;
    }

    #[tokio::test]
    async fn persistent_bash_sentinel_requires_exact_full_line() {
        let sentinel_prefix = "__NAKED_DONE_seedednonce_";
        let mut reader = BufReader::new(
            b"echo prefix __NAKED_DONE_seedednonce_0__ suffix\n__NAKED_DONE_seedednonce_0__\n"
                .as_slice(),
        );
        let mut output = Vec::new();

        let exit_code = read_until_stdout_sentinel(&mut reader, sentinel_prefix, &mut output)
            .await
            .unwrap();

        assert_eq!(exit_code, 0);
        assert_eq!(
            String::from_utf8_lossy(&output),
            "echo prefix __NAKED_DONE_seedednonce_0__ suffix\n"
        );
    }

    #[tokio::test]
    async fn persistent_bash_stdout_capped_but_sentinel_still_parsed() {
        let sentinel_prefix = "__NAKED_DONE_seedednonce_";
        let line = b"body line padding padding padding padding padding padding\n";
        let line_count = (MAX_COLLECTED / line.len()) + 64;
        let mut input = Vec::with_capacity((line_count * line.len()) + 64);
        for _ in 0..line_count {
            input.extend_from_slice(line);
        }
        input.extend_from_slice(b"__NAKED_DONE_seedednonce_17__\n");
        let mut reader = BufReader::new(input.as_slice());
        let mut output = Vec::new();

        let exit_code = read_until_stdout_sentinel(&mut reader, sentinel_prefix, &mut output)
            .await
            .unwrap();

        assert_eq!(exit_code, 17);
        assert!(
            output.len() <= MAX_COLLECTED + PERSISTENT_BASH_STDOUT_TRUNCATED_MARKER.len(),
            "output should stay bounded, got {} bytes",
            output.len()
        );
        assert!(
            output
                .windows(PERSISTENT_BASH_STDOUT_TRUNCATED_MARKER.len())
                .any(|window| window == PERSISTENT_BASH_STDOUT_TRUNCATED_MARKER),
            "truncation marker missing from stdout body"
        );
    }

    #[tokio::test]
    async fn persistent_bash_manager_evicts_lru_over_cap() {
        let manager = PersistentBashManager::with_limits(2, Duration::from_secs(60));
        let s1 = manager
            .get_or_create("s1", Duration::from_secs(5))
            .await
            .unwrap();
        s1.run("echo one", Path::new("/tmp"), Duration::from_secs(5))
            .await
            .unwrap();
        let s1_pid = s1.process_id().await;
        let s2 = manager
            .get_or_create("s2", Duration::from_secs(5))
            .await
            .unwrap();
        s2.run("echo two", Path::new("/tmp"), Duration::from_secs(5))
            .await
            .unwrap();

        let s3 = manager
            .get_or_create("s3", Duration::from_secs(5))
            .await
            .unwrap();
        s3.run("echo three", Path::new("/tmp"), Duration::from_secs(5))
            .await
            .unwrap();

        assert!(manager.session_count().await <= 2);
        #[cfg(target_os = "linux")]
        if let Some(pid) = s1_pid {
            tokio::time::sleep(Duration::from_millis(100)).await;
            assert!(!PathBuf::from(format!("/proc/{pid}")).exists());
        }

        let out = s2
            .run("echo two-again", Path::new("/tmp"), Duration::from_secs(5))
            .await
            .unwrap();
        assert_eq!(String::from_utf8_lossy(&out.stdout).trim(), "two-again");
        manager.close_session("s2").await;
        manager.close_session("s3").await;
    }

    #[cfg(target_os = "linux")]
    #[tokio::test]
    async fn persistent_bash_no_zombie_after_close() {
        let manager = PersistentBashManager::new();
        let session = manager
            .get_or_create("s", Duration::from_secs(5))
            .await
            .unwrap();
        let out = session
            .run("echo $$", Path::new("/tmp"), Duration::from_secs(5))
            .await
            .unwrap();
        let pid = String::from_utf8_lossy(&out.stdout)
            .trim()
            .parse::<u32>()
            .unwrap();
        assert!(PathBuf::from(format!("/proc/{pid}")).exists());
        manager.close_session("s").await;
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert!(!PathBuf::from(format!("/proc/{pid}")).exists());
    }
}
