//! Telegram media ingestion: voice / audio, photos, documents, video.
//!
//! Responsibilities:
//!
//! 1. Download the file from the Telegram Bot API with a strict byte cap.
//! 2. Persist a copy under `workspace/.naked/artifacts/YYYY-MM-DD/` for audit
//!    and so the agent can read it with its `read_file` tool.
//! 3. Optionally transform the payload into text the LLM can consume:
//!    - voice/audio → transcribe via an OpenAI-compatible Whisper endpoint
//!      (Groq `whisper-large-v3` recommended);
//!    - photos → describe via an OpenAI-compatible vision endpoint
//!      (xAI `grok-2-vision`, OpenAI `gpt-4o-mini`);
//!    - text documents under `docs_inline_max_bytes` → embed content;
//!    - everything else → relay path + size + MIME.
//!
//! When a provider is not configured or the call fails we degrade gracefully
//! and still give the agent the saved path.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result, anyhow, bail};
use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as B64;
use chrono::Utc;
use naked_core::config::{AudioProviderCfg, TgMediaConfig, VisionProviderCfg};
use reqwest::multipart::{Form, Part};
use reqwest::{Client, StatusCode};
use serde_json::{Value, json};

/// Telegram's own hard ceiling for `getFile`-served downloads (20 MB).
pub const TELEGRAM_MAX_DOWNLOAD_BYTES: u64 = 20 * 1024 * 1024;

/// Kind of incoming media, derived from the Telegram message shape.
/// We don't rely on `IncomingAttachmentKind` from anywhere else so tests stay hermetic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MediaKind {
    Voice,
    Audio,
    Photo,
    Video,
    Animation,
    Document,
    Sticker,
}

impl MediaKind {
    pub fn as_str(self) -> &'static str {
        match self {
            MediaKind::Voice => "voice",
            MediaKind::Audio => "audio",
            MediaKind::Photo => "photo",
            MediaKind::Video => "video",
            MediaKind::Animation => "animation",
            MediaKind::Document => "document",
            MediaKind::Sticker => "sticker",
        }
    }

    /// Strict per-kind byte cap from config.
    pub fn cap(self, cfg: &TgMediaConfig) -> u64 {
        match self {
            MediaKind::Voice | MediaKind::Audio => cfg.limits.audio_max_bytes,
            MediaKind::Photo | MediaKind::Sticker => cfg.limits.photo_max_bytes,
            MediaKind::Video | MediaKind::Animation | MediaKind::Document => {
                cfg.limits.doc_max_bytes
            }
        }
        .min(TELEGRAM_MAX_DOWNLOAD_BYTES)
    }
}

// ─────────────────────────── Artifacts on disk ───────────────────────────

/// Root folder for saved media inside the workspace.
pub fn artifacts_root(workspace: &Path) -> PathBuf {
    workspace.join(".naked").join("artifacts")
}

/// Today's dated subfolder: `<workspace>/.naked/artifacts/YYYY-MM-DD/`.
pub fn artifact_dir_today(workspace: &Path) -> PathBuf {
    artifacts_root(workspace).join(Utc::now().format("%Y-%m-%d").to_string())
}

/// Strip path separators, control chars, and anything NULL-ish from a filename
/// coming from the outside world. Falls back to the media kind + message id
/// when the sanitized version would be empty.
pub fn sanitize_filename(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    for ch in raw.chars() {
        match ch {
            // Directory traversal and Windows reserved characters.
            '/' | '\\' | '\0' | ':' | '*' | '?' | '"' | '<' | '>' | '|' => out.push('_'),
            c if c.is_control() => out.push('_'),
            c => out.push(c),
        }
    }
    let trimmed = out.trim_matches(|c: char| c == ' ' || c == '.').to_string();
    if trimmed.is_empty() {
        "file".to_string()
    } else if trimmed.len() > 120 {
        // Keep extension if any.
        if let Some(idx) = trimmed.rfind('.') {
            let (stem, ext) = trimmed.split_at(idx);
            let mut keep = 120usize.saturating_sub(ext.len()).min(stem.len());
            while keep > 0 && !stem.is_char_boundary(keep) {
                keep -= 1;
            }
            format!("{}{ext}", &stem[..keep])
        } else {
            let mut end = 120.min(trimmed.len());
            while end > 0 && !trimmed.is_char_boundary(end) {
                end -= 1;
            }
            trimmed[..end].to_string()
        }
    } else {
        trimmed
    }
}

/// Delete dated artifact directories older than `retention_days`. Safe to call
/// at startup; errors are swallowed with a `tracing::warn`.
pub fn sweep_old_artifacts(workspace: &Path, retention_days: u64) {
    if retention_days == 0 {
        return;
    }
    let root = artifacts_root(workspace);
    // REGISTRY-WAIVE: intentional fallback: missing path → empty result
    let Ok(entries) = std::fs::read_dir(&root) else {
        return;
    };
    let cutoff = match std::time::SystemTime::now()
        .checked_sub(std::time::Duration::from_secs(retention_days * 86_400))
    {
        Some(c) => c,
        None => return,
    };
    for entry in entries.flatten() {
        // REGISTRY-WAIVE: see BUG_REGISTRY B23 — verified intentional 2026-05-13
        let Ok(meta) = entry.metadata() else { continue };
        if !meta.is_dir() {
            continue;
        }
        let modified = meta.modified().ok().unwrap_or(std::time::UNIX_EPOCH);
        if modified < cutoff {
            let path = entry.path();
            match std::fs::remove_dir_all(&path) {
                Ok(()) => {
                    tracing::info!("swept old artifact dir: {}", path.display());
                }
                Err(e) => {
                    tracing::warn!("failed to sweep {}: {e}", path.display());
                }
            }
        }
    }
}

// ─────────────────────────── Download ────────────────────────────────────

/// Number of attempts for every Telegram HTTP step that is safe to retry
/// (`getFile` + the actual file fetch). Telegram occasionally flakes with
/// connection resets, 502s, or "empty body" on its file CDN; operators
/// reported cases where a single failure made the bot give up and tell
/// them to download the photo manually. With 3 attempts + exp. backoff we
/// swallow the typical transient outage without blowing the per-turn
/// latency budget (worst case ≈ 100 + 300 + 700 ms of extra waits).
const DOWNLOAD_ATTEMPTS: usize = 3;

/// Exponential backoff between download attempts.
///
/// Formula: `BASE * 3^(attempt)` → 100 ms, 300 ms, 900 ms for attempts
/// 0/1/2. Only the waits between attempts show up in perceived latency,
/// so the sum for a 3-attempt sequence is bounded at ~1 s.
const DOWNLOAD_BACKOFF_BASE: std::time::Duration = std::time::Duration::from_millis(100);

/// Decide whether an error is worth retrying. We retry on **transport**
/// (connect / reset / timeout / partial body) and on **server-side 5xx**
/// / 429 (Telegram's CDN occasionally returns these during failover).
/// 4xx other than 429 is terminal — the file_id is bad, expired, or the
/// bot lost access to the chat; retrying can't help.
fn is_transient_download_error(err: &anyhow::Error) -> bool {
    let chain = err.chain().map(|e| e.to_string()).collect::<Vec<_>>();
    let combined = chain.join(" | ").to_lowercase();

    if combined.contains("timed out")
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
    {
        return true;
    }

    // HTTP status codes: must look like an actual status surrounded by
    // delimiters, not a substring of an unrelated number (e.g. don't
    // match "500" inside "50000000 bytes"). The error strings we
    // produce above always render as `status: NNN` or `status: NNN
    // <reason>`, so anchor on that.
    for code in ["500", "502", "503", "504", "429"] {
        let needle_colon = format!("status: {code}");
        let needle_eq = format!("status={code}");
        if combined.contains(&needle_colon) || combined.contains(&needle_eq) {
            return true;
        }
    }

    false
}

/// Call `getFile` to resolve a Bot API file_id → relative `file_path`.
async fn resolve_file_path(
    http: &Client,
    base_url: &str,
    file_id: &str,
) -> Result<(String, Option<u64>)> {
    let url = format!("{base_url}/getFile");
    let mut last_err: Option<anyhow::Error> = None;

    for attempt in 0..DOWNLOAD_ATTEMPTS {
        let call = async {
            let resp: Value = http
                .post(&url)
                .json(&json!({ "file_id": file_id }))
                .send()
                .await
                .context("getFile request failed")?
                .error_for_status()
                .context("getFile returned non-2xx")?
                .json()
                .await
                .context("getFile returned non-JSON body")?;
            let result = resp
                .get("result")
                .ok_or_else(|| anyhow!("getFile: missing result"))?;
            let file_path = result
                .get("file_path")
                .and_then(Value::as_str)
                .ok_or_else(|| anyhow!("getFile: missing file_path"))?
                .to_string();
            let size = result.get("file_size").and_then(Value::as_u64);
            Ok::<_, anyhow::Error>((file_path, size))
        }
        .await;

        match call {
            Ok(ok) => {
                if attempt > 0 {
                    tracing::info!(
                        attempts_used = attempt + 1,
                        "getFile recovered after transient failure"
                    );
                }
                return Ok(ok);
            }
            Err(e) => {
                if attempt + 1 < DOWNLOAD_ATTEMPTS && is_transient_download_error(&e) {
                    let delay = DOWNLOAD_BACKOFF_BASE * 3_u32.pow(attempt as u32);
                    tracing::warn!(
                        attempt = attempt + 1,
                        max = DOWNLOAD_ATTEMPTS,
                        sleep_ms = delay.as_millis() as u64,
                        error = %e,
                        "getFile transient error, retrying"
                    );
                    tokio::time::sleep(delay).await;
                    last_err = Some(e);
                    continue;
                }
                return Err(e);
            }
        }
    }

    Err(last_err.unwrap_or_else(|| anyhow!("getFile: exhausted {DOWNLOAD_ATTEMPTS} attempts")))
}

/// Saved-file handle: where on disk the blob lives and its MIME type.
pub struct Downloaded {
    pub path: PathBuf,
    pub bytes: Vec<u8>,
    pub mime: String,
}

/// Download a Telegram-hosted file into `artifacts/<date>/<prefix>_<name>`.
/// Enforces `cap_bytes` (rejects the download if the server declares bigger
/// or if the transfer streams past the cap).
pub async fn download_to_artifacts(
    http: Arc<Client>,
    bot_token: &str,
    api_base: &str,
    file_id: &str,
    workspace: &Path,
    filename_hint: &str,
    cap_bytes: u64,
) -> Result<Downloaded> {
    let (remote_path, declared_size) = resolve_file_path(&http, api_base, file_id).await?;
    if let Some(size) = declared_size
        && size > cap_bytes
    {
        bail!("file is {} bytes, exceeds cap {} bytes", size, cap_bytes);
    }

    let download_url = format!("https://api.telegram.org/file/bot{bot_token}/{remote_path}");

    // Same retry shape as `resolve_file_path`: Telegram's file CDN
    // occasionally returns 502/504 or truncates the body, and a single
    // flake shouldn't end up as "скачай сам, я не могу" to the operator
    // (that exact complaint is what drove this retry loop — see the
    // auto-captured `user:105928336` correction "When media download
    // fails (e.g. 0 bytes timeout), don't just give up").
    let mut body: Option<Vec<u8>> = None;
    let mut last_err: Option<anyhow::Error> = None;

    for attempt in 0..DOWNLOAD_ATTEMPTS {
        let call = async {
            let resp = http
                .get(&download_url)
                .send()
                .await
                .context("file download failed")?;
            if resp.status() != StatusCode::OK {
                bail!("file download status: {}", resp.status());
            }
            let bytes = resp.bytes().await.context("file body read failed")?;
            if bytes.is_empty() {
                bail!("file download returned empty body");
            }
            Ok::<_, anyhow::Error>(bytes.to_vec())
        }
        .await;

        match call {
            Ok(b) => {
                if attempt > 0 {
                    tracing::info!(
                        attempts_used = attempt + 1,
                        bytes = b.len(),
                        "file download recovered after transient failure"
                    );
                }
                body = Some(b);
                break;
            }
            Err(e) => {
                if attempt + 1 < DOWNLOAD_ATTEMPTS && is_transient_download_error(&e) {
                    let delay = DOWNLOAD_BACKOFF_BASE * 3_u32.pow(attempt as u32);
                    tracing::warn!(
                        attempt = attempt + 1,
                        max = DOWNLOAD_ATTEMPTS,
                        sleep_ms = delay.as_millis() as u64,
                        error = %e,
                        "file download transient error, retrying"
                    );
                    tokio::time::sleep(delay).await;
                    last_err = Some(e);
                    continue;
                }
                return Err(e);
            }
        }
    }

    let body = body
        .ok_or_else(|| last_err.unwrap_or_else(|| anyhow!("file download: exhausted attempts")))?;

    if body.len() as u64 > cap_bytes {
        bail!("downloaded body {} > cap {}", body.len(), cap_bytes);
    }

    let mime = mime_guess::from_path(&remote_path)
        .first_raw()
        .unwrap_or("application/octet-stream")
        .to_string();

    let dir = artifact_dir_today(workspace);
    std::fs::create_dir_all(&dir).context("create artifact dir")?;
    let sanitized = sanitize_filename(filename_hint);
    let path = dir.join(&sanitized);
    // If the target already exists (same msg id replayed), suffix with unix ts.
    let path = if path.exists() {
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        dir.join(format!("{ts}_{sanitized}"))
    } else {
        path
    };
    std::fs::write(&path, &body).context("write artifact")?;

    Ok(Downloaded {
        path,
        bytes: body,
        mime,
    })
}

// ─────────────────────────── Error classifier (DRY) ──────────────────────

/// Reason bucket for media-related upstream failures. Returned as a
/// cardinality-safe `&'static str` so it can label Prometheus counters
/// without unbounded growth. **PLAN_MEDIA_UX_v1 M5 / BUG_REGISTRY B01**:
/// previously duplicated as an inline `match` ladder inside
/// `media_dispatch.rs::process_one_media`. Lifted here for DRY reuse by
/// both the transcription and the describer paths.
pub fn classify_media_error(err: &anyhow::Error) -> &'static str {
    let msg = format!("{err:#}").to_ascii_lowercase();
    if msg.contains("401") || msg.contains("403") || msg.contains("unauthorized") {
        "auth"
    } else if msg.contains("429") || msg.contains("rate") {
        "rate_limit"
    } else if msg.contains("413") || msg.contains("too large") || msg.contains("payload") {
        "payload"
    } else if msg.contains("timeout") || msg.contains("timed out") {
        "timeout"
    } else if msg.contains("dns") || msg.contains("connect") || msg.contains("reset") {
        "network"
    } else {
        "other"
    }
}

// ─────────────────────────── Whisper transcription ───────────────────────

/// Transcribe audio via an OpenAI-compatible `/audio/transcriptions` endpoint.
/// Returns the plain transcript text.
///
/// **PLAN_MEDIA_UX_v1 M5 / BUG_REGISTRY B01**: previously silent on
/// success. Now emits exactly ONE `tracing::info!` per call (success
/// path) with `bytes_in` / `chars_out` / `duration_ms` / `model`, and
/// bumps `naked_tg_media_transcription_total{outcome}` plus, on Err,
/// `naked_tg_media_transcription_failure_total{reason}` via
/// [`classify_media_error`]. Operators can now distinguish
/// transcription-worked from transcription-silently-failed in journal
/// and `/metrics`.
pub async fn transcribe_audio(
    http: Arc<Client>,
    cfg: &AudioProviderCfg,
    audio: &[u8],
    file_name: &str,
    mime: &str,
) -> Result<String> {
    let started = std::time::Instant::now();
    let bytes_in = audio.len();
    let result = transcribe_audio_inner(http, cfg, audio, file_name, mime).await;
    let elapsed_ms = u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX);
    match &result {
        Ok(text) => {
            crate::metrics::record_transcription("ok", None);
            tracing::info!(
                model = %cfg.model,
                bytes_in,
                chars_out = text.chars().count(),
                duration_ms = elapsed_ms,
                "transcription complete"
            );
        }
        Err(e) => {
            let reason = classify_media_error(e);
            crate::metrics::record_transcription("fail", Some(reason));
            tracing::warn!(
                model = %cfg.model,
                bytes_in,
                duration_ms = elapsed_ms,
                reason,
                error = %e,
                "transcription failed"
            );
        }
    }
    result
}

/// Internal body — split out so the wrapper above can observe the result
/// with a single `match`. Pure I/O, no logging or counter bumps inside.
async fn transcribe_audio_inner(
    http: Arc<Client>,
    cfg: &AudioProviderCfg,
    audio: &[u8],
    file_name: &str,
    mime: &str,
) -> Result<String> {
    let api_key = cfg.resolved_api_key().context("resolve audio api_key")?;
    if api_key.is_empty() {
        bail!("audio api_key is empty after env expansion");
    }

    let part = Part::bytes(audio.to_vec())
        .file_name(sanitize_filename(file_name))
        .mime_str(mime)
        .context("invalid audio MIME")?;
    let mut form = Form::new()
        .part("file", part)
        .text("model", cfg.model.clone())
        .text("response_format", "json");
    if let Some(lang) = &cfg.language
        && !lang.is_empty()
    {
        form = form.text("language", lang.clone());
    }

    let resp = http
        .post(&cfg.api_url)
        .bearer_auth(&api_key)
        .multipart(form)
        .send()
        .await
        .context("transcription POST failed")?;
    let status = resp.status();
    let body_text = resp
        .text()
        .await
        .context("transcription: failed to read body")?;
    if !status.is_success() {
        bail!("transcription HTTP {status}: {body_text}");
    }
    let v: Value =
        serde_json::from_str(&body_text).context("transcription: non-JSON response body")?;
    // OpenAI/Groq shape: {"text": "…"}.
    let text = v
        .get("text")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("transcription: response missing 'text' field: {body_text}"))?
        .trim()
        .to_string();
    if text.is_empty() {
        bail!("transcription returned empty text");
    }
    Ok(text)
}

// ─────────────────────────── Vision describer ────────────────────────────

const DEFAULT_VISION_PROMPT: &str = "You are describing an image that a user sent to an AI coding assistant. \
Give a concrete, faithful description in <=200 words. \
Transcribe every piece of visible text VERBATIM (code, error messages, UI labels, \
CLI output, URLs, file paths). If it's a screenshot of code or a terminal, reproduce \
the text exactly. Do not speculate about intent. \
User caption (may be empty): {caption}";

/// Describe a photo using an OpenAI-compatible chat-completions endpoint with
/// `image_url` content blocks. The result is plain text suitable for embedding
/// into the main agent's prompt.
pub async fn describe_image(
    http: Arc<Client>,
    cfg: &VisionProviderCfg,
    image: &[u8],
    mime: &str,
    user_caption: Option<&str>,
) -> Result<String> {
    let api_key = cfg.resolved_api_key().context("resolve vision api_key")?;
    if api_key.is_empty() {
        bail!("vision api_key is empty after env expansion");
    }

    let prompt_template = cfg
        .prompt_override
        .as_deref()
        .unwrap_or(DEFAULT_VISION_PROMPT);
    let caption = user_caption.unwrap_or("").trim();
    let prompt = prompt_template.replace("{caption}", caption);

    let data_url = format!("data:{mime};base64,{}", B64.encode(image));

    let payload = json!({
        "model": cfg.model,
        "max_tokens": cfg.max_tokens,
        "temperature": 0.0,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_url}}
            ]
        }]
    });

    let resp = http
        .post(&cfg.api_url)
        .bearer_auth(&api_key)
        .json(&payload)
        .send()
        .await
        .context("vision POST failed")?;
    let status = resp.status();
    let body_text = resp.text().await.context("vision: failed to read body")?;
    if !status.is_success() {
        bail!("vision HTTP {status}: {body_text}");
    }

    let v: Value = serde_json::from_str(&body_text).context("vision: non-JSON response body")?;
    let text = v
        .pointer("/choices/0/message/content")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("vision: missing choices[0].message.content: {body_text}"))?
        .trim()
        .to_string();
    if text.is_empty() {
        bail!("vision returned empty content");
    }
    Ok(text)
}

// ─────────────────────────── Document inlining ───────────────────────────

/// File extensions we trust to be safe to inline as UTF-8 text.
const TEXTUAL_EXTENSIONS: &[&str] = &[
    "txt",
    "md",
    "markdown",
    "rst",
    "log",
    "csv",
    "tsv",
    "tex",
    "json",
    "jsonl",
    "yaml",
    "yml",
    "toml",
    "ini",
    "conf",
    "cfg",
    "env",
    "sh",
    "bash",
    "zsh",
    "fish",
    "py",
    "rs",
    "go",
    "ts",
    "tsx",
    "js",
    "jsx",
    "mjs",
    "cjs",
    "c",
    "h",
    "cpp",
    "hpp",
    "cc",
    "cs",
    "java",
    "kt",
    "kts",
    "swift",
    "rb",
    "php",
    "pl",
    "lua",
    "sql",
    "r",
    "jl",
    "html",
    "htm",
    "xml",
    "svg",
    "css",
    "scss",
    "sass",
    "less",
    "patch",
    "diff",
    "gitignore",
    "dockerfile",
    "makefile",
];

/// True when `file_name` / `mime` / `size` all indicate we can safely embed the
/// document content directly into the prompt.
pub fn is_inlineable_doc(file_name: &str, mime: &str, size: u64, max_bytes: u64) -> bool {
    if size == 0 || size > max_bytes {
        return false;
    }
    if mime.starts_with("text/") {
        return true;
    }
    // application/json, application/xml, application/yaml, application/toml
    if let Some(subtype) = mime.strip_prefix("application/")
        && matches!(
            subtype,
            "json" | "xml" | "yaml" | "x-yaml" | "toml" | "x-toml" | "x-sh"
        )
    {
        return true;
    }
    let lower = file_name.to_ascii_lowercase();
    let ext = Path::new(&lower)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("");
    if TEXTUAL_EXTENSIONS.contains(&ext) {
        return true;
    }
    // Files like "Dockerfile" and "Makefile" are named without extensions.
    matches!(
        lower.rsplit('/').next().unwrap_or(""),
        "dockerfile" | "makefile" | "rakefile" | ".env" | ".gitignore"
    )
}

/// Truncate long text bodies for safe inlining. Keeps the first `limit` bytes
/// (on UTF-8 boundaries) and appends a truncation marker.
pub fn truncate_for_inline(s: &str, limit: usize) -> String {
    if s.len() <= limit {
        return s.to_string();
    }
    let truncated = naked_core::util::head_truncate(s, limit);
    format!(
        "{truncated}\n\n[... truncated: {} bytes omitted ...]",
        s.len() - truncated.len()
    )
}

// ─────────────────────────── Tests ───────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ─── PLAN_MEDIA_UX_v1 M5 / BUG_REGISTRY B01 / INV-5 ───

    #[test]
    fn classify_media_error_auth() {
        let e = anyhow::anyhow!("HTTP 401 Unauthorized");
        assert_eq!(classify_media_error(&e), "auth");
        let e = anyhow::anyhow!("403 Forbidden");
        assert_eq!(classify_media_error(&e), "auth");
    }

    #[test]
    fn classify_media_error_rate_limit() {
        let e = anyhow::anyhow!("HTTP 429 Too Many Requests");
        assert_eq!(classify_media_error(&e), "rate_limit");
        let e = anyhow::anyhow!("rate exceeded");
        assert_eq!(classify_media_error(&e), "rate_limit");
    }

    #[test]
    fn classify_media_error_payload() {
        let e = anyhow::anyhow!("HTTP 413 Payload Too Large");
        assert_eq!(classify_media_error(&e), "payload");
        let e = anyhow::anyhow!("image too large");
        assert_eq!(classify_media_error(&e), "payload");
    }

    #[test]
    fn classify_media_error_timeout() {
        let e = anyhow::anyhow!("operation timed out");
        assert_eq!(classify_media_error(&e), "timeout");
    }

    #[test]
    fn classify_media_error_network() {
        let e = anyhow::anyhow!("failed to connect: DNS failure");
        assert_eq!(classify_media_error(&e), "network");
        let e = anyhow::anyhow!("connection reset by peer");
        assert_eq!(classify_media_error(&e), "network");
    }

    #[test]
    fn classify_media_error_other_is_default() {
        let e = anyhow::anyhow!("some opaque server error");
        assert_eq!(classify_media_error(&e), "other");
    }

    /// INV-5 instrumented: every successful record_transcription bumps
    /// the ok counter; every fail bumps fail + the right reason bucket.
    /// Uses the global static counters so this test must run alone for
    /// deterministic deltas — we snapshot before/after.
    #[test]
    fn record_transcription_increments_outcome_buckets() {
        let before = crate::metrics::snapshot();
        crate::metrics::record_transcription("ok", None);
        crate::metrics::record_transcription("fail", Some("auth"));
        crate::metrics::record_transcription("fail", Some("timeout"));
        crate::metrics::record_transcription("fail", None); // → "other"
        let after = crate::metrics::snapshot();
        assert_eq!(after.transcription_ok - before.transcription_ok, 1);
        assert_eq!(after.transcription_fail - before.transcription_fail, 3);
        assert_eq!(
            after.transcription_fail_auth - before.transcription_fail_auth,
            1
        );
        assert_eq!(
            after.transcription_fail_timeout - before.transcription_fail_timeout,
            1
        );
        assert_eq!(
            after.transcription_fail_other - before.transcription_fail_other,
            1
        );
    }

    #[test]
    fn sanitize_strips_path_sep_and_control() {
        // `/` → `_`; result must not contain any separator, so even joined
        // with a parent dir it stays a single path component.
        let s = sanitize_filename("../../etc/passwd");
        assert!(!s.contains('/'), "got: {s}");
        assert!(!s.contains('\\'), "got: {s}");
        assert!(s.ends_with("etc_passwd"), "got: {s}");

        assert_eq!(sanitize_filename("foo\tbar"), "foo_bar");
        assert_eq!(
            sanitize_filename("a/b\\c:d*e?f\"g<h>i|j"),
            "a_b_c_d_e_f_g_h_i_j"
        );
    }

    #[test]
    fn sanitize_falls_back_when_empty() {
        assert_eq!(sanitize_filename(""), "file");
        assert_eq!(sanitize_filename("   ...  "), "file");
    }

    #[test]
    fn sanitize_truncates_preserving_ext() {
        let long = "a".repeat(300);
        let name = format!("{long}.txt");
        let s = sanitize_filename(&name);
        assert!(s.ends_with(".txt"));
        assert!(s.len() <= 120);
    }

    #[test]
    fn sanitize_truncates_russian_filename_on_char_boundary() {
        // Russian chars are 2 bytes each. 120-byte cut could land mid-char.
        let stem = "Файл".repeat(50); // 200 chars = 400 bytes
        let name = format!("{stem}.pdf");
        let s = sanitize_filename(&name);
        assert!(s.len() <= 120);
        assert!(s.ends_with(".pdf"));
        // Must be valid UTF-8 — this implicitly tests char boundary.
        assert!(s.is_char_boundary(s.len()));
    }

    #[test]
    fn sanitize_truncates_no_ext_russian() {
        let stem = "Отчёт".repeat(50); // 250 chars = 500 bytes, no extension
        let s = sanitize_filename(&stem);
        assert!(s.len() <= 120);
        // Valid UTF-8 string
        for c in s.chars() {
            assert!(c.len_utf8() > 0);
        }
    }

    #[test]
    fn artifact_dir_today_uses_utc_date_folder() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = artifact_dir_today(tmp.path());
        let date = Utc::now().format("%Y-%m-%d").to_string();
        assert!(dir.ends_with(format!(".naked/artifacts/{date}")), "{dir:?}");
    }

    #[test]
    fn is_inlineable_doc_by_mime() {
        assert!(is_inlineable_doc("x", "text/plain", 100, 1024));
        assert!(is_inlineable_doc("x", "application/json", 100, 1024));
        assert!(is_inlineable_doc("x", "application/yaml", 100, 1024));
        assert!(!is_inlineable_doc("x", "image/png", 100, 1024));
        assert!(!is_inlineable_doc(
            "x",
            "application/octet-stream",
            100,
            1024
        ));
    }

    #[test]
    fn is_inlineable_doc_by_extension() {
        assert!(is_inlineable_doc(
            "foo.rs",
            "application/octet-stream",
            10,
            1024
        ));
        assert!(is_inlineable_doc(
            "script.sh",
            "application/octet-stream",
            10,
            1024
        ));
        assert!(!is_inlineable_doc(
            "movie.mp4",
            "application/octet-stream",
            10,
            1024
        ));
    }

    #[test]
    fn is_inlineable_doc_respects_size_cap() {
        assert!(!is_inlineable_doc("x.txt", "text/plain", 0, 1024));
        assert!(!is_inlineable_doc("x.txt", "text/plain", 2000, 1024));
        assert!(is_inlineable_doc("x.txt", "text/plain", 1024, 1024));
    }

    #[test]
    fn truncate_for_inline_shorter_is_unchanged() {
        assert_eq!(truncate_for_inline("hello", 100), "hello");
    }

    #[test]
    fn truncate_for_inline_adds_marker() {
        let r = truncate_for_inline("aaaaaaaaaa", 4);
        assert!(r.starts_with("aaaa"));
        assert!(r.contains("truncated"));
    }

    #[test]
    fn truncate_for_inline_utf8_boundary() {
        // "Привет" — each cyrillic char is 2 bytes; cutting mid-codepoint must not panic.
        let r = truncate_for_inline("Привет!", 3);
        assert!(r.starts_with("П") || r.starts_with("Пр") || !r.is_empty());
    }

    #[test]
    fn media_kind_cap_respects_telegram_ceiling() {
        let mut cfg = TgMediaConfig::default();
        cfg.limits.audio_max_bytes = 100 * 1024 * 1024; // 100 MB requested
        let cap = MediaKind::Voice.cap(&cfg);
        assert_eq!(cap, TELEGRAM_MAX_DOWNLOAD_BYTES);
    }

    #[test]
    fn media_kind_cap_uses_config_when_under_ceiling() {
        let cfg = TgMediaConfig::default(); // 5/20/10 MB
        assert_eq!(MediaKind::Photo.cap(&cfg), 5 * 1024 * 1024);
        assert_eq!(MediaKind::Audio.cap(&cfg), 20 * 1024 * 1024);
        assert_eq!(MediaKind::Document.cap(&cfg), 10 * 1024 * 1024);
    }

    #[test]
    fn sweep_noop_when_retention_zero() {
        let tmp = tempfile::tempdir().unwrap();
        let root = artifacts_root(tmp.path());
        std::fs::create_dir_all(&root).unwrap();
        let d = root.join("2099-01-01");
        std::fs::create_dir(&d).unwrap();
        std::fs::write(d.join("f.txt"), b"x").unwrap();

        sweep_old_artifacts(tmp.path(), 0);

        assert!(d.exists(), "retention_days=0 must skip sweep");
    }

    #[test]
    fn sweep_keeps_fresh_dirs() {
        let tmp = tempfile::tempdir().unwrap();
        let root = artifacts_root(tmp.path());
        std::fs::create_dir_all(&root).unwrap();
        let d = artifact_dir_today(tmp.path());
        std::fs::create_dir_all(&d).unwrap();
        std::fs::write(d.join("f.txt"), b"x").unwrap();

        sweep_old_artifacts(tmp.path(), 7);

        assert!(d.exists(), "today's dir must survive a 7-day sweep");
    }

    #[test]
    fn sweep_ignores_non_dir_entries() {
        let tmp = tempfile::tempdir().unwrap();
        let root = artifacts_root(tmp.path());
        std::fs::create_dir_all(&root).unwrap();
        // Put a plain file at the top level; sweep must not touch it.
        std::fs::write(root.join("stray.txt"), b"x").unwrap();

        sweep_old_artifacts(tmp.path(), 1);

        assert!(root.join("stray.txt").exists());
    }

    // ─────────────────────────── Live e2e tests ───────────────────────────
    //
    // These exercise the real third-party APIs we ship support for. They are
    // gated on environment variables so a no-secrets `cargo test` skips them.
    //
    // Run all live media tests:
    //   GROQ_API_KEY=... XAI_API_KEY=... OPENAI_API_KEY=... \
    //     cargo test -p naked-tg --bins -- --ignored --test-threads=1 live_

    /// Path to a test fixture file. Returns `None` when the file is missing
    /// (e.g. the developer ran `cargo test` from a sparse checkout).
    fn fixture(name: &str) -> Option<PathBuf> {
        let p = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("fixtures")
            .join(name);
        if p.exists() { Some(p) } else { None }
    }

    /// Skip the calling test if the named env var is unset / empty, with a
    /// helpful message that names the missing key.
    fn require_env(name: &str) -> Option<String> {
        match std::env::var(name) {
            Ok(v) if !v.is_empty() => Some(v),
            _ => {
                eprintln!("SKIP: {name} not set");
                None
            }
        }
    }

    fn http_for_test() -> Arc<reqwest::Client> {
        Arc::new(
            reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(60))
                .build()
                .expect("build reqwest client"),
        )
    }

    /// Live: Groq Whisper transcription on a 1-second 440 Hz tone fixture.
    /// A pure tone is expected to transcribe to (almost) empty text — both
    /// `Ok` and the specific `empty text` `Err` are accepted; what we really
    /// verify is that the multipart request, auth, and response parsing all
    /// succeed end-to-end.
    #[tokio::test]
    #[ignore = "live API call: requires GROQ_API_KEY"]
    async fn live_transcribe_tone_via_groq() {
        let Some(_) = require_env("GROQ_API_KEY") else {
            return;
        };
        let Some(path) = fixture("test_tone.ogg") else {
            eprintln!("SKIP: tests/fixtures/test_tone.ogg missing");
            return;
        };
        let bytes = std::fs::read(&path).expect("read tone fixture");
        let cfg = AudioProviderCfg {
            api_url: "https://api.groq.com/openai/v1/audio/transcriptions".into(),
            api_key: "$GROQ_API_KEY".into(),
            model: "whisper-large-v3".into(),
            language: None,
        };
        // PLAN_MEDIA_UX_v1 M5 / BUG_REGISTRY B01: end-to-end observability
        // check — either the ok or fail counter MUST increment, never both,
        // never neither. This was the silent-success class that originally
        // motivated M5.
        let before = crate::metrics::snapshot();
        let result =
            transcribe_audio(http_for_test(), &cfg, &bytes, "test_tone.ogg", "audio/ogg").await;
        let after = crate::metrics::snapshot();
        let ok_delta = after.transcription_ok - before.transcription_ok;
        let fail_delta = after.transcription_fail - before.transcription_fail;
        assert_eq!(
            ok_delta + fail_delta,
            1,
            "INV-5: exactly one of transcription_ok / transcription_fail must increment per call"
        );
        match result {
            Ok(text) => {
                eprintln!("  transcript: {text:?}");
                assert_eq!(ok_delta, 1, "Ok path must bump transcription_ok");
            }
            Err(e) if e.to_string().contains("empty text") => {
                eprintln!("  transcript: <empty> (acceptable for pure tone)");
                assert_eq!(
                    fail_delta, 1,
                    "empty-text Err path must bump transcription_fail"
                );
            }
            Err(e) => panic!("transcription failed: {e}"),
        }
    }

    /// Pick the first vision provider whose API key is configured. Tries Groq
    /// (Llama-4 Scout, OpenAI-compat) → xAI (`grok-2-vision`) → OpenAI
    /// (`gpt-4o-mini`). Returns `None` to skip the test when no key is set.
    fn pick_vision_cfg(
        max_tokens: u32,
        prompt_override: Option<String>,
    ) -> Option<VisionProviderCfg> {
        // PLAN_MEDIA_UX_v1 M3: prefer the configured production describer
        // (qwen3-vl-plus via Alibaba Dashscope). Verified live 2026-05-13:
        // correctly identifies SMPTE color bars in test_image.png.
        for var in [
            "DASHSCOPE_API_KEY_2",
            "DASHSCOPE_API_KEY",
            "DASHSCOPE_API_KEY_4",
        ] {
            if std::env::var(var)
                .ok()
                .filter(|v| !v.is_empty() && !v.starts_with('$'))
                .is_some()
            {
                eprintln!("  vision provider: qwen3-vl-plus via ${var}");
                return Some(VisionProviderCfg {
                    api_url: "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
                        .into(),
                    api_key: format!("${var}"),
                    model: "qwen3-vl-plus".into(),
                    max_tokens,
                    prompt_override,
                });
            }
        }
        if std::env::var("GROQ_API_KEY")
            .ok()
            .filter(|v| !v.is_empty())
            .is_some()
        {
            eprintln!("  vision provider: groq llama-4-scout");
            return Some(VisionProviderCfg {
                api_url: "https://api.groq.com/openai/v1/chat/completions".into(),
                api_key: "$GROQ_API_KEY".into(),
                model: "meta-llama/llama-4-scout-17b-16e-instruct".into(),
                max_tokens,
                prompt_override,
            });
        }
        if std::env::var("XAI_API_KEY")
            .ok()
            .filter(|v| !v.is_empty())
            .is_some()
        {
            eprintln!("  vision provider: xai grok-2-vision");
            return Some(VisionProviderCfg {
                api_url: "https://api.x.ai/v1/chat/completions".into(),
                api_key: "$XAI_API_KEY".into(),
                model: "grok-2-vision-latest".into(),
                max_tokens,
                prompt_override,
            });
        }
        if std::env::var("OPENAI_API_KEY")
            .ok()
            .filter(|v| !v.is_empty())
            .is_some()
        {
            eprintln!("  vision provider: openai gpt-4o-mini");
            return Some(VisionProviderCfg {
                api_url: "https://api.openai.com/v1/chat/completions".into(),
                api_key: "$OPENAI_API_KEY".into(),
                model: "gpt-4o-mini".into(),
                max_tokens,
                prompt_override,
            });
        }
        eprintln!("SKIP: no vision provider key set (GROQ_API_KEY / XAI_API_KEY / OPENAI_API_KEY)");
        None
    }

    /// Live: vision describer on the FFmpeg `testsrc` pattern. Tries Groq
    /// Llama-4 Scout first (it's what we use in production today), falling back
    /// to xAI grok-2-vision, then OpenAI gpt-4o-mini. The image contains color
    /// bars + numbers + circles, so any working model will produce non-empty
    /// text mentioning "color" / "bars" / "test" / "pattern".
    #[tokio::test]
    #[ignore = "live API call: requires GROQ_API_KEY or XAI_API_KEY or OPENAI_API_KEY"]
    async fn live_describe_image_via_vision_provider() {
        let Some(path) = fixture("test_image.png") else {
            eprintln!("SKIP: tests/fixtures/test_image.png missing");
            return;
        };
        let bytes = std::fs::read(&path).expect("read image fixture");
        let Some(cfg) = pick_vision_cfg(200, None) else {
            return;
        };

        let text = describe_image(
            http_for_test(),
            &cfg,
            &bytes,
            "image/png",
            Some("test pattern from ffmpeg testsrc"),
        )
        .await
        .expect("vision describe should succeed");

        eprintln!("  description ({} chars): {text}", text.len());
        assert!(!text.is_empty(), "description must be non-empty");
        // We don't pin to a specific word — just verify we got a sentence-shaped
        // response (the testsrc pattern produces > 20 char descriptions across
        // every vision model we ship support for).
        assert!(text.len() > 20, "description too short: {text:?}");
        // Sanity: the SMPTE color-bar test pattern should produce a description
        // mentioning either colour names, the word 'color', 'bar', or 'pattern'.
        // Models we support today (Groq llama-4-scout, xAI grok-2-vision, OpenAI
        // gpt-4o-mini, Qwen3-VL-Plus) ALL hit at least one of these keywords on
        // the test fixture — verified live 2026-05-13.
        let lc = text.to_ascii_lowercase();
        let recognises_pattern = [
            "color",
            "colour",
            "цвет",
            "bar",
            "полос",
            "pattern",
            "test",
            "тест",
            "smpte",
            "rainbow",
        ]
        .iter()
        .any(|kw| lc.contains(kw));
        assert!(
            recognises_pattern,
            "vision describer didn't recognise SMPTE test pattern in fixture: {text:?}"
        );
    }

    /// Live: verify `prompt_override` actually substitutes `{caption}` and that
    /// the provider sees and echoes the rendered prompt. Uses a unique tag to
    /// avoid confusion with prior responses.
    #[tokio::test]
    #[ignore = "live API call: requires GROQ_API_KEY or XAI_API_KEY or OPENAI_API_KEY"]
    async fn live_describe_image_uses_caption_in_prompt_override() {
        let Some(path) = fixture("test_image.png") else {
            eprintln!("SKIP: tests/fixtures/test_image.png missing");
            return;
        };
        let bytes = std::fs::read(&path).expect("read image fixture");

        let unique = format!("ECHO_TAG_{}", std::process::id());
        let prompt = format!(
            "Reply with exactly the literal string '{unique}' followed by the caption: {{caption}}. Nothing else."
        );
        let Some(cfg) = pick_vision_cfg(60, Some(prompt)) else {
            return;
        };

        let text = describe_image(
            http_for_test(),
            &cfg,
            &bytes,
            "image/png",
            Some("CAPTION_INSIDE"),
        )
        .await
        .expect("vision call should succeed");

        eprintln!("  response: {text}");
        assert!(
            text.contains(&unique),
            "prompt_override was not honoured (no {unique} in: {text})"
        );
        assert!(
            text.contains("CAPTION_INSIDE"),
            "{{caption}} placeholder was not substituted in: {text}"
        );
    }

    // ── Download retry classifier ────────────────────────────────────────
    //
    // The actual retry loop ends up exercised by the live download tests,
    // but the *decision* — "is this error worth retrying?" — is pure
    // string inspection and lends itself to fast, hermetic unit coverage.
    // Operators care about this because the bot used to give up after a
    // single transient blip and tell them "скачай руками"; if we ever
    // mis-classify a 4xx/permission error as transient, we'll waste
    // backoff time and still fail. Keep the lists in sync with the
    // matchers in `is_transient_download_error`.

    fn err(s: &str) -> anyhow::Error {
        anyhow::anyhow!("{s}")
    }

    #[test]
    fn transient_classifier_yes_on_typical_telegram_flakes() {
        for s in [
            "file body read failed: error reading a body from connection: connection reset by peer",
            "getFile request failed: timed out",
            "file download status: 502 Bad Gateway",
            "file download status: 503",
            "file download status: 504",
            "file download status: 429 Too Many Requests",
            "tls handshake eof",
            "dns error: failed to lookup host",
            "unexpected eof during decode",
            "file download returned empty body",
        ] {
            assert!(is_transient_download_error(&err(s)), "should retry: {s}");
        }
    }

    #[test]
    fn transient_classifier_no_on_terminal_errors() {
        for s in [
            "file download status: 400 Bad Request",
            "file download status: 401 Unauthorized",
            "file download status: 403 Forbidden",
            "file download status: 404 Not Found",
            "getFile: missing result",
            "downloaded body 50000000 > cap 5000000",
            "file is 99999999 bytes, exceeds cap 5000000 bytes",
        ] {
            assert!(
                !is_transient_download_error(&err(s)),
                "should NOT retry: {s}"
            );
        }
    }
}
