//! Abstracts the FS side-effect of archiving cycle messages (DIP).

use crate::types::ConversationMessage;
use std::path::PathBuf;

pub trait CycleArchiver: Send + Sync {
    /// Write messages for the completed cycle. Returns the archive path on success.
    fn archive(
        &self,
        session_id: &str,
        cycle_num: u32,
        messages: &[ConversationMessage],
    ) -> Result<PathBuf, std::io::Error>;
}

pub struct FsCycleArchiver {
    pub data_dir: PathBuf,
}

impl CycleArchiver for FsCycleArchiver {
    fn archive(
        &self,
        session_id: &str,
        cycle_num: u32,
        messages: &[ConversationMessage],
    ) -> Result<PathBuf, std::io::Error> {
        crate::session::cycle::write_archive(&self.data_dir, session_id, cycle_num, messages)
    }
}

pub struct NoopCycleArchiver;

impl CycleArchiver for NoopCycleArchiver {
    fn archive(
        &self,
        _session_id: &str,
        _cycle_num: u32,
        _messages: &[ConversationMessage],
    ) -> Result<PathBuf, std::io::Error> {
        Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "cycle archiving disabled (no data_dir configured)",
        ))
    }
}
