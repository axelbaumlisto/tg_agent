//! Image artifact management: externalization, GC, rotation.

use std::path::Path;

use crate::error::{AgentError, Result};
use crate::types::{ContentBlock, ConversationMessage, IMAGE_REF_SENTINEL_PREFIX};
use base64::Engine;

const IMAGE_REF_PREFIX: &str = IMAGE_REF_SENTINEL_PREFIX;

/// Session JSONL rotation threshold.
const ROTATE_AFTER_BYTES: u64 = 256 * 1024;
/// Maximum number of rotated JSONL files to keep.
const MAX_ROTATED_FILES: usize = 3;

fn ext_for_mime(mime: &str) -> &'static str {
    match mime.to_ascii_lowercase().as_str() {
        "image/png" => "png",
        "image/jpeg" | "image/jpg" => "jpg",
        "image/webp" => "webp",
        "image/gif" => "gif",
        "image/heic" | "image/heif" => "heic",
        _ => "bin",
    }
}

pub(crate) async fn extern_image_blocks(
    artifacts_dir: &Path,
    msg: &ConversationMessage,
) -> Result<ConversationMessage> {
    let needs_extern = msg
        .blocks
        .iter()
        .any(|b| matches!(b, ContentBlock::Image { .. }));
    if !needs_extern {
        return Ok(msg.clone());
    }

    // Async fs from the start: callers are async and a session can carry
    // multi-MB images. Even short blocking writes hurt the runtime under
    // concurrent saves (multiple chats appending in parallel).
    tokio::fs::create_dir_all(artifacts_dir)
        .await
        .map_err(|e| AgentError::Session(format!("create artifacts dir: {e}")))?;

    let mut out = msg.clone();
    for block in &mut out.blocks {
        if let ContentBlock::Image {
            mime,
            data_base64,
            detail,
        } = block
        {
            let bytes = base64::engine::general_purpose::STANDARD
                .decode(data_base64.as_bytes())
                .map_err(|e| AgentError::Session(format!("decode image base64: {e}")))?;
            // blake3 over (mime || 0x00 || bytes) — content-addressable so the
            // same image attached twice (or the same image across sessions
            // restored from backup) produces the same filename. Switched away
            // from `DefaultHasher` because that's a stdlib implementation
            // detail (SipHash today, may change), and a long-lived on-disk
            // identifier deserves a stable cryptographic digest.
            let mut hasher = blake3::Hasher::new();
            hasher.update(mime.as_bytes());
            hasher.update(&[0u8]);
            hasher.update(&bytes);
            let digest = hasher.finalize();
            let ext = ext_for_mime(mime);
            // 16 hex chars (64 bits) is plenty for collision resistance per
            // session — one session would need ~2^32 distinct images before
            // a single collision is likely. Keeps filenames human-typable.
            let fname = format!("img_{}.{ext}", &digest.to_hex().as_str()[..16]);
            let abs_path = artifacts_dir.join(&fname);
            // tokio::fs::try_exists avoids racing with another task that's
            // writing the same artifact; if it returns Err just attempt the
            // write — the file system is the final source of truth.
            let exists = tokio::fs::try_exists(&abs_path).await.unwrap_or(false);
            if !exists {
                tokio::fs::write(&abs_path, &bytes)
                    .await
                    .map_err(|e| AgentError::Session(format!("write artifact: {e}")))?;
            }
            // Relative to artifacts_dir; `intern_image_blocks` rejoins it.
            let mut payload = serde_json::json!({
                "mime": mime,
                "path": fname,
                "bytes": bytes.len(),
            });
            if let Some(d) = detail {
                payload["detail"] = serde_json::Value::String(d.as_str().to_string());
            }
            *block = ContentBlock::Text {
                text: format!("{IMAGE_REF_PREFIX}{payload}"),
            };
        }
    }
    Ok(out)
}

/// Inverse of `extern_image_blocks`: rehydrate sentinel Text blocks back into
/// `ContentBlock::Image` by reading bytes from the artifacts directory and
/// re-encoding to base64. Sentinel blocks whose artifact file is missing or
/// whose JSON is malformed are kept as-is (degraded but visible) rather than
/// dropped — losing user attachments silently is worse than showing a marker.
pub(crate) async fn intern_image_blocks(
    artifacts_dir: &Path,
    mut msg: ConversationMessage,
) -> ConversationMessage {
    for block in &mut msg.blocks {
        if let ContentBlock::Text { text } = block
            && let Some(rest) = text.strip_prefix(IMAGE_REF_PREFIX)
            && let Ok(payload) = serde_json::from_str::<serde_json::Value>(rest)
            && let (Some(mime), Some(rel_path)) = (
                payload.get("mime").and_then(|v| v.as_str()),
                payload.get("path").and_then(|v| v.as_str()),
            )
        {
            let abs = artifacts_dir.join(rel_path);
            match tokio::fs::read(&abs).await {
                Ok(bytes) => {
                    let b64 = base64::engine::general_purpose::STANDARD.encode(&bytes);
                    let detail =
                        payload
                            .get("detail")
                            .and_then(|v| v.as_str())
                            .and_then(|s| match s {
                                "low" => Some(crate::types::ImageDetail::Low),
                                "high" => Some(crate::types::ImageDetail::High),
                                "auto" => Some(crate::types::ImageDetail::Auto),
                                _ => None,
                            });
                    *block = ContentBlock::Image {
                        mime: mime.to_string(),
                        data_base64: b64,
                        detail,
                    };
                }
                Err(_) => {
                    // Artifact missing — keep the sentinel so the user can see
                    // that an image was here and locate the broken reference.
                }
            }
        }
    }
    msg
}

/// Walk `session.jsonl` to collect every artifact filename mentioned by a
/// sentinel marker, then delete files in `artifacts_dir` that aren't in that
/// set. Conservative on parse errors: a malformed line aborts the walk
/// (returning Ok(0)) so we never delete artifacts based on partial knowledge.
pub(crate) async fn gc_orphan_image_artifacts_impl(
    session_jsonl: &Path,
    artifacts_dir: &Path,
) -> Result<usize> {
    if !session_jsonl.exists() || !artifacts_dir.is_dir() {
        return Ok(0);
    }
    let content = tokio::fs::read_to_string(session_jsonl).await?;
    let mut referenced: std::collections::HashSet<String> = std::collections::HashSet::new();
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let v: serde_json::Value = match serde_json::from_str(trimmed) {
            Ok(v) => v,
            Err(_) => return Ok(0),
        };
        // Walk the JSON tree looking for any string value starting with the
        // sentinel marker. The marker carries `{"path": "img_..."}` JSON
        // tail — extracting the path is a substring scan rather than a
        // structural walk because we don't know which content-block schema
        // emitted the marker (current is Text, but future may differ).
        collect_referenced_artifacts(&v, &mut referenced);
    }

    let mut entries = tokio::fs::read_dir(artifacts_dir).await?;
    let mut removed = 0usize;
    while let Some(entry) = entries.next_entry().await? {
        let name = entry.file_name();
        let name_s = name.to_string_lossy().to_string();
        if !name_s.starts_with("img_") {
            continue;
        }
        if !referenced.contains(&name_s) {
            // try_exists is not strictly required — if the file disappeared
            // between read_dir and remove, ignore the NotFound.
            match tokio::fs::remove_file(entry.path()).await {
                Ok(_) => removed += 1,
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                Err(e) => return Err(e.into()),
            }
        }
    }
    Ok(removed)
}

/// Delete every `img_*` file in `artifacts_dir` whose mtime is older than
/// `max_age_secs` seconds relative to now. Files younger than the cutoff
/// are left alone. Missing directory → Ok(0); we never create it.
///
/// Note: this is a *reachability-free* sweep. The caller is responsible
/// for ensuring the sessions whose artifacts get culled have been
/// compacted or archived first, otherwise history references may become
/// stale (but will still render as `[image missing]` placeholders —
/// never a crash).
pub(crate) async fn gc_old_image_artifacts_impl(
    artifacts_dir: &Path,
    max_age_secs: u64,
) -> Result<usize> {
    if !artifacts_dir.is_dir() {
        return Ok(0);
    }
    let now = std::time::SystemTime::now();
    let cutoff = std::time::Duration::from_secs(max_age_secs);
    let mut entries = tokio::fs::read_dir(artifacts_dir).await?;
    let mut removed = 0usize;
    while let Some(entry) = entries.next_entry().await? {
        let name = entry.file_name();
        let name_s = name.to_string_lossy();
        if !name_s.starts_with("img_") {
            continue;
        }
        let meta = match entry.metadata().await {
            Ok(m) => m,
            Err(_) => continue,
        };
        let mtime = meta.modified().unwrap_or(now);
        let age = now.duration_since(mtime).unwrap_or_default();
        if age > cutoff {
            match tokio::fs::remove_file(entry.path()).await {
                Ok(_) => removed += 1,
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                Err(e) => return Err(e.into()),
            }
        }
    }
    Ok(removed)
}

fn collect_referenced_artifacts(
    v: &serde_json::Value,
    out: &mut std::collections::HashSet<String>,
) {
    match v {
        serde_json::Value::String(s) => {
            if let Some(rest) = s.strip_prefix(IMAGE_REF_PREFIX)
                && let Ok(payload) = serde_json::from_str::<serde_json::Value>(rest)
                && let Some(path) = payload.get("path").and_then(|p| p.as_str())
            {
                out.insert(path.to_string());
            }
        }
        serde_json::Value::Array(arr) => {
            for x in arr {
                collect_referenced_artifacts(x, out);
            }
        }
        serde_json::Value::Object(map) => {
            for (_, x) in map {
                collect_referenced_artifacts(x, out);
            }
        }
        _ => {}
    }
}

pub(crate) async fn rotate_if_needed(path: &Path) -> Result<()> {
    if !path.exists() {
        return Ok(());
    }
    let meta = tokio::fs::metadata(path).await?;
    if meta.len() < ROTATE_AFTER_BYTES {
        return Ok(());
    }

    for i in (1..MAX_ROTATED_FILES).rev() {
        let from = path.with_extension(format!("jsonl.{i}"));
        let to = path.with_extension(format!("jsonl.{}", i + 1));
        if from.exists() {
            let _ = tokio::fs::rename(&from, &to).await;
        }
    }

    let rotated = path.with_extension("jsonl.1");
    let _ = tokio::fs::rename(path, &rotated).await;

    Ok(())
}
