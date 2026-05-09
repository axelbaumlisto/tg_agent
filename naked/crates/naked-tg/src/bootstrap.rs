//! Process bootstrap: PID-lock, log directory, tracing init, metrics.
//!
//! `run()` is the entry point called from `#[tokio::main] main()`.
//! It initialises the process-level infrastructure, builds the DI graph
//! via [`crate::wiring::build`], then hands off to the event loop in
//! [`crate::runtime::run_event_loop`].

pub(crate) async fn run() {
    naked_tg::guarded::install_panic_hook();

    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    let naked_dir = std::path::PathBuf::from(&home).join(".naked");
    let log_dir = naked_dir.join("logs");
    std::fs::create_dir_all(&log_dir).ok();

    // ── pid-lock: kill ALL stale instances before starting ─────────────
    let my_pid = std::process::id();
    let pid_path = naked_dir.join("naked-tg.pid");

    // 1) Kill process from pid file
    if let Ok(old) = std::fs::read_to_string(&pid_path)
        && let Ok(pid) = old.trim().parse::<u32>()
        && pid != my_pid
        && std::path::Path::new(&format!("/proc/{pid}")).exists()
    {
        eprintln!("Killing stale naked-tg pid={pid}");
        let _ = std::process::Command::new("kill")
            .args(["-9", &pid.to_string()])
            .status();
    }
    // 2) Kill any other naked-tg binaries (exact process name match)
    if let Ok(output) = std::process::Command::new("pgrep")
        .args(["-x", "naked-tg"])
        .output()
    {
        for line in String::from_utf8_lossy(&output.stdout).lines() {
            if let Ok(pid) = line.trim().parse::<u32>()
                && pid != my_pid
            {
                eprintln!("Killing extra naked-tg pid={pid}");
                let _ = std::process::Command::new("kill")
                    .args(["-9", &pid.to_string()])
                    .status();
            }
        }
    }

    std::fs::write(&pid_path, my_pid.to_string()).ok();

    // 14-day daily rotation: one file per day, auto-delete anything older
    // than two weeks. Previous setting kept only 3 days which made it
    // painful to investigate incidents that got reported late.
    let file_appender = tracing_appender::rolling::RollingFileAppender::builder()
        .rotation(tracing_appender::rolling::Rotation::DAILY)
        .filename_prefix("naked-tg")
        .filename_suffix("log")
        .max_log_files(14)
        .build(&log_dir)
        .expect("failed to init log appender");
    let (non_blocking_file, _guard) = tracing_appender::non_blocking(file_appender);

    let env_filter = tracing_subscriber::EnvFilter::from_default_env()
        .add_directive("naked=info".parse().expect("static directive"));

    use tracing_subscriber::fmt::writer::MakeWriterExt;
    let combined = std::io::stderr.and(non_blocking_file);

    tracing_subscriber::fmt()
        .with_env_filter(env_filter)
        .with_writer(combined)
        .with_ansi(false)
        .init();

    // Spawn Prometheus /metrics endpoint if NAKED_METRICS_ADDR is set.
    // No-op otherwise; production only opts in explicitly.
    crate::metrics::serve_prometheus_if_enabled();

    // Build the DI graph, then hand off to the event loop.
    let wired = crate::wiring::build().await;
    crate::runtime::run_event_loop(wired).await;
}
