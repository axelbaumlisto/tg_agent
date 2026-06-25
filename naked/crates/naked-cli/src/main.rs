#![cfg_attr(not(test), warn(clippy::unwrap_used))]
mod commands;
mod event_map;
mod research;
mod tui;

use std::{
    fs::{File, OpenOptions},
    io::{self, Write},
    path::PathBuf,
};

use anyhow::Result;

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let filter = tracing_subscriber::EnvFilter::from_default_env()
        .add_directive("naked=info".parse().expect("static directive"));

    // In full-screen TUI mode, tracing must NOT write to stdout/stderr — it would
    // interleave with the ratatui alt-screen render and corrupt the frame. Route
    // logs to ~/.naked/naked-tui.log instead. The line-mode REPL (default) keeps
    // logging to stdout unchanged.
    if is_tui_invocation(&args) {
        tracing_subscriber::fmt()
            .with_env_filter(filter)
            .with_writer(make_tui_log_writer)
            .init();
    } else {
        tracing_subscriber::fmt().with_env_filter(filter).init();
    }

    commands::run(args).await
}

fn is_tui_invocation(args: &[String]) -> bool {
    args.get(1).is_some_and(|arg| arg == "tui")
}

fn make_tui_log_writer() -> TuiLogWriter {
    open_tui_log_file()
        .map(TuiLogWriter::File)
        .unwrap_or_else(|_| TuiLogWriter::Stderr(io::stderr()))
}

fn open_tui_log_file() -> io::Result<File> {
    let home = std::env::var_os("HOME")
        .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "HOME is not set"))?;
    let log_dir = PathBuf::from(home).join(".naked");
    std::fs::create_dir_all(&log_dir)?;
    OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_dir.join("naked-tui.log"))
}

enum TuiLogWriter {
    File(File),
    Stderr(io::Stderr),
}

impl Write for TuiLogWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        match self {
            Self::File(file) => file.write(buf),
            Self::Stderr(stderr) => stderr.write(buf),
        }
    }

    fn flush(&mut self) -> io::Result<()> {
        match self {
            Self::File(file) => file.flush(),
            Self::Stderr(stderr) => stderr.flush(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::is_tui_invocation;

    #[test]
    fn is_tui_invocation_detects_tui_subcommand() {
        assert!(is_tui_invocation(&strings(&["naked", "tui"])));
        assert!(is_tui_invocation(&strings(&["naked", "tui", "x"])));
        assert!(!is_tui_invocation(&strings(&["naked"])));
        assert!(!is_tui_invocation(&strings(&["naked", "chat"])));
        assert!(!is_tui_invocation(&strings(&["naked", "--help"])));
        assert!(!is_tui_invocation(&[]));
    }

    fn strings(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).to_owned()).collect()
    }
}
