//! Testable helpers factored out of `media.rs`.
//!
//! `media.rs` is a long, Telegram-HTTP-heavy module that lives under
//! the binary target (`mod media;` in `main.rs`). The pure predicates
//! below are exported via `lib.rs` so integration tests can pin their
//! contracts without pulling in the rest of the media pipeline.

use std::sync::atomic::{AtomicBool, Ordering};

use naked_core::config::AudioProviderCfg;

// ─────────────────────────────────────────────────────────────────────────────
// DownloadErrorClass — classification of `anyhow::Error`s returned by the
// Telegram download path. Mirrors the private `is_transient_download_error`
// inside `media.rs` but exposes a typed view so callers (and the E2 unit test)
// can branch on the decision instead of re-deriving it from a string.
// ─────────────────────────────────────────────────────────────────────────────

/// Typed classification of a download-path error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DownloadErrorKind {
    /// Safe to retry — transient transport or 5xx/429 upstream failure.
    Transient,
    /// Do not retry — the file_id is bad, expired, or access revoked.
    Terminal,
}

/// Public wrapper around `is_transient_download_error`. Owns a single
/// `DownloadErrorKind` so consumers don't re-implement the heuristic.
#[derive(Debug, Clone, Copy)]
pub struct DownloadErrorClass(DownloadErrorKind);

impl DownloadErrorClass {
    /// Classify an `anyhow::Error` by walking its chain and matching on
    /// known substrings. Symmetric with `media::is_transient_download_error`.
    pub fn from_anyhow(err: &anyhow::Error) -> Self {
        let chain = err.chain().map(|e| e.to_string()).collect::<Vec<_>>();
        let combined = chain.join(" | ").to_lowercase();

        let transport_hit = combined.contains("timed out")
            || combined.contains("timeout")
            || combined.contains("connection reset")
            || combined.contains("connection closed")
            || combined.contains("broken pipe")
            || combined.contains("dns")
            || combined.contains("tls")
            || combined.contains("handshake")
            || combined.contains("unexpected eof")
            || combined.contains("empty body")
            || combined.contains("body read failed")
            || combined.contains("zero bytes")
            || combined.contains("0 bytes");

        if transport_hit {
            return Self(DownloadErrorKind::Transient);
        }

        for code in ["500", "502", "503", "504", "429"] {
            let needle_colon = format!("status: {code}");
            let needle_eq = format!("status={code}");
            if combined.contains(&needle_colon) || combined.contains(&needle_eq) {
                return Self(DownloadErrorKind::Transient);
            }
        }

        Self(DownloadErrorKind::Terminal)
    }

    pub fn kind(self) -> DownloadErrorKind {
        self.0
    }

    pub fn is_transient(self) -> bool {
        matches!(self.0, DownloadErrorKind::Transient)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// AudioConfigDiagnostics — per-session warn-once gate for missing audio
// transcription config. Replaces the silent user-facing placeholder that used
// to show up verbatim for every voice message when `tg_media.audio = None`.
// ─────────────────────────────────────────────────────────────────────────────

/// What the media pipeline should do with a voice message when the audio
/// provider is not configured.
#[derive(Debug, Clone)]
pub enum VoiceHandling {
    /// Transcribe normally using the configured provider.
    Transcribe,
    /// First voice of this session — emit the `warn` line once to operator
    /// logs, then drop the media silently from the conversation.
    FirstWarnThenDrop { warn: String },
    /// Subsequent voices within the same session — drop silently, no log
    /// churn, no user-facing placeholder.
    QuietDrop,
}

/// Per-session / per-process diagnostic that makes sure operators hear
/// about a missing audio config exactly once, not once per voice message.
pub struct AudioConfigDiagnostics {
    warned: AtomicBool,
}

impl Default for AudioConfigDiagnostics {
    fn default() -> Self {
        Self::new()
    }
}

impl AudioConfigDiagnostics {
    pub fn new() -> Self {
        Self {
            warned: AtomicBool::new(false),
        }
    }

    /// Reset the "already warned" flag. Useful when a new
    /// `AudioConfigDiagnostics` is created per session at session boot.
    pub fn reset(&self) {
        self.warned.store(false, Ordering::Relaxed);
    }

    /// Classify an incoming voice message against the audio config. If
    /// `audio` is `Some(_)`, returns `Transcribe`. Otherwise the first
    /// call returns `FirstWarnThenDrop`; subsequent calls return
    /// `QuietDrop`.
    pub fn handle_voice(&self, audio: Option<&AudioProviderCfg>) -> VoiceHandling {
        if audio.is_some() {
            return VoiceHandling::Transcribe;
        }
        let was_warned = self.warned.swap(true, Ordering::Relaxed);
        if was_warned {
            VoiceHandling::QuietDrop
        } else {
            VoiceHandling::FirstWarnThenDrop {
                warn: "tg_media.audio is not configured — voice messages are being dropped. \
                       Set tg_media.audio = { api_url, model, api_key } in naked.json."
                    .to_string(),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transient_classifier_catches_empty_body() {
        let err = anyhow::anyhow!("file download body read failed: empty body");
        assert!(DownloadErrorClass::from_anyhow(&err).is_transient());
    }

    #[test]
    fn transient_classifier_catches_zero_bytes() {
        let err = anyhow::anyhow!("download read 0 bytes from CDN");
        assert!(DownloadErrorClass::from_anyhow(&err).is_transient());
    }

    #[test]
    fn terminal_for_4xx_non_429() {
        let err = anyhow::anyhow!("getFile returned status: 404 Not Found");
        assert!(!DownloadErrorClass::from_anyhow(&err).is_transient());
    }

    #[test]
    fn audio_diagnostics_warns_once() {
        let d = AudioConfigDiagnostics::new();
        match d.handle_voice(None) {
            VoiceHandling::FirstWarnThenDrop { .. } => {}
            other => panic!("expected FirstWarnThenDrop, got {other:?}"),
        }
        for _ in 0..4 {
            match d.handle_voice(None) {
                VoiceHandling::QuietDrop => {}
                other => panic!("expected QuietDrop, got {other:?}"),
            }
        }
    }
}
